"""Vendor files outside the interfaces tree: package targets and application vendor entries.

Every download here is served by a local HTTP stub; nothing reaches the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from osprey.apps import manifest as app_manifest
from osprey.apps.loader import load
from osprey.cli.vendor_cmd import vendor as vendor_group
from osprey.interfaces import vendor
from osprey.interfaces._serving import run_app_server
from osprey.interfaces.common_middleware import NoCacheStaticMiddleware
from osprey.interfaces.design_system.panels.validator import validate_panel

LIB = b"/* lib 1.0 */ var lib = 1;\n"
UPLOT = b"/* uplot */ var uPlot = function () { return '#123456'; };\n"
LIB_SHA = hashlib.sha256(LIB).hexdigest()
UPLOT_SHA = hashlib.sha256(UPLOT).hexdigest()
#: Where the stub serves the url entry's file; the URL carries the version the
#: entry's file name repeats.
LIB_PATH = "/files/lib/1.0.0/lib.js"
LIB_FILE = "lib-1.0.0.js"
UPLOT_FILE = "uplot-1.6.30.min.js"


class _Stub:
    """A local CDN: serves fixed bytes by path and counts the requests."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {
            "/files/lib.js": LIB,
            LIB_PATH: LIB,
            "/npm/uplot@1.6.30/dist/uPlot.iife.min.js": UPLOT,
        }
        self.hits: list[str] = []
        self.base = ""

    def app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/{path:path}")
        async def serve(path: str) -> Response:
            key = "/" + path
            self.hits.append(key)
            if key not in self.files:
                return Response(status_code=404)
            return Response(content=self.files[key], media_type="application/javascript")

        return app


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Stub]:
    server = _Stub()
    with run_app_server(server.app()) as base:
        server.base = base
        monkeypatch.setattr(
            app_manifest, "PACKAGE_URL_TEMPLATE", base + "/npm/{package}@{version}/{file}"
        )
        yield server


def _write_app(root: Path, name: str, vendor_entries: list[str]) -> None:
    app_dir = root / "apps" / name
    app_dir.mkdir(parents=True)
    lines = [
        f"name: {name}",
        "containers:",
        "  backend:",
        "    runtime: osprey",
        "vendor:",
        *vendor_entries,
    ]
    (app_dir / "app.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _deployment(root: Path, stub: _Stub, *, into: str = "panel/vendor/" + LIB_FILE) -> list:
    """A deployment holding one application with a url entry and a package entry.

    The url entry is written as the manifest requires (``https://``) and pointed at
    the stub after loading; the package entry reaches the stub through the template.
    """
    _write_app(
        root,
        "knob",
        [
            "  - url: https://cdn.example.org" + LIB_PATH,
            f"    into: {into}",
            f"    sha256: {LIB_SHA}",
            "  - package: uplot",
            "    version: 1.6.30",
            "    file: dist/uPlot.iife.min.js",
            "    into: panel/vendor/" + UPLOT_FILE,
            f"    sha256: {UPLOT_SHA}",
        ],
    )
    manifests = load(root)
    url_entry = manifests[0].vendor[0]
    manifests[0].vendor[0] = url_entry.model_copy(update={"url": stub.base + LIB_PATH})
    return manifests


# --- application vendor entries ------------------------------------------------------


def test_fetches_url_and_package_entries_into_the_app_dir(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub)

    fetched = vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    vendor_dir = tmp_path / "apps" / "knob" / "panel" / "vendor"
    assert (vendor_dir / LIB_FILE).read_bytes() == LIB
    assert (vendor_dir / UPLOT_FILE).read_bytes() == UPLOT
    assert sorted(Path(p).name for p in fetched) == [LIB_FILE, UPLOT_FILE]
    assert "/npm/uplot@1.6.30/dist/uPlot.iife.min.js" in stub.hits


def test_a_verified_copy_is_reused_without_a_request(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub)
    vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)
    stub.hits.clear()

    fetched = vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    assert fetched == []
    assert stub.hits == []


def test_offline_uses_the_verified_copy(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub)
    vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)
    stub.hits.clear()

    assert vendor.fetch_app_vendor(tmp_path, manifests, offline=True, quiet=True) == []
    assert stub.hits == []


def test_offline_without_a_copy_fails_naming_the_app_and_package(
    tmp_path: Path, stub: _Stub
) -> None:
    manifests = _deployment(tmp_path, stub)
    (tmp_path / "apps" / "knob" / "panel" / "vendor").mkdir(parents=True)
    (tmp_path / "apps" / "knob" / "panel" / "vendor" / LIB_FILE).write_bytes(LIB)

    with pytest.raises(vendor.AppVendorError) as exc:
        vendor.fetch_app_vendor(tmp_path, manifests, offline=True, quiet=True)

    assert exc.value.app == "knob"
    assert exc.value.source == "uplot@1.6.30/dist/uPlot.iife.min.js"
    assert "knob" in str(exc.value) and "uplot@1.6.30" in str(exc.value)
    assert "offline" in str(exc.value)
    assert stub.hits == []


def test_offline_with_a_tampered_copy_fails(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub)
    vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)
    (tmp_path / "apps" / "knob" / "panel" / "vendor" / UPLOT_FILE).write_bytes(b"tampered")

    with pytest.raises(vendor.AppVendorError, match="uplot@1.6.30"):
        vendor.fetch_app_vendor(tmp_path, manifests, offline=True, quiet=True)


def test_a_failed_fetch_fails_naming_the_app_and_leaves_no_file(
    tmp_path: Path, stub: _Stub
) -> None:
    manifests = _deployment(tmp_path, stub)
    del stub.files["/npm/uplot@1.6.30/dist/uPlot.iife.min.js"]

    with pytest.raises(vendor.AppVendorError) as exc:
        vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    assert exc.value.app == "knob"
    assert "uplot@1.6.30" in str(exc.value)
    assert not (tmp_path / "apps" / "knob" / "panel" / "vendor" / UPLOT_FILE).exists()


def test_a_failed_fetch_keeps_using_a_verified_copy(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub)
    vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)
    stub.files.clear()

    assert vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True) == []


def test_a_hash_mismatch_fails_and_leaves_no_file(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub)
    stub.files[LIB_PATH] = b"something else"

    with pytest.raises(vendor.AppVendorError, match="SHA256 mismatch"):
        vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    assert not (tmp_path / "apps" / "knob" / "panel" / "vendor" / LIB_FILE).exists()


@pytest.mark.parametrize("into", ["../escape.js", "panel/../../escape.js", "/tmp/escape.js"])
def test_a_target_outside_the_app_dir_is_refused_before_any_download(
    tmp_path: Path, stub: _Stub, into: str
) -> None:
    manifests = _deployment(tmp_path, stub, into=into)

    with pytest.raises(vendor.AppVendorError, match="outside apps/knob/"):
        vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    assert stub.hits == []
    assert not (tmp_path / "apps" / "escape.js").exists()


def test_a_symlink_out_of_the_app_dir_is_refused(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "apps" / "knob" / "panel").symlink_to(outside)

    with pytest.raises(vendor.AppVendorError, match="outside apps/knob/"):
        vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    assert list(outside.iterdir()) == []


def test_offline_defaults_to_the_deployment_mode(
    tmp_path: Path, stub: _Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = _deployment(tmp_path, stub)
    monkeypatch.setenv("OSPREY_OFFLINE", "1")

    with pytest.raises(vendor.AppVendorError, match="offline"):
        vendor.fetch_app_vendor(tmp_path, manifests, quiet=True)

    assert stub.hits == []


def test_an_app_without_vendor_entries_needs_no_network(tmp_path: Path) -> None:
    _write_app(tmp_path, "plain", ["  []"])

    assert vendor.fetch_app_vendor(tmp_path, load(tmp_path), offline=False) == []


def test_the_panel_validator_skips_an_app_panels_vendor_dir(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub)
    vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)
    panel_dir = tmp_path / "apps" / "knob" / "panel"
    (panel_dir / "manifest.json").write_text(
        '{"id": "knob", "label": "Knob", "entry": "index.html"}', encoding="utf-8"
    )
    (panel_dir / "index.html").write_text(
        '<!doctype html><script src="/design-system/js/theme-boot.js"></script>'
        '<link rel="stylesheet" href="/design-system/css/tokens.css">'
        '<link rel="stylesheet" href="/static/fonts/fonts.css">\n',
        encoding="utf-8",
    )

    findings = validate_panel(panel_dir)

    assert b"#123456" in (panel_dir / "vendor" / UPLOT_FILE).read_bytes()
    assert not findings


# --- a vendored file name carries its version -----------------------------------------


def test_a_package_entry_whose_file_name_lacks_its_version_is_refused(
    tmp_path: Path, stub: _Stub
) -> None:
    _write_app(
        tmp_path,
        "knob",
        [
            "  - package: uplot",
            "    version: 1.6.30",
            "    file: dist/uPlot.iife.min.js",
            "    into: panel/vendor/uplot.min.js",
            f"    sha256: {UPLOT_SHA}",
        ],
    )

    with pytest.raises(vendor.AppVendorError) as exc:
        vendor.fetch_app_vendor(tmp_path, load(tmp_path), offline=False, quiet=True)

    assert exc.value.app == "knob"
    assert exc.value.source == "uplot@1.6.30/dist/uPlot.iife.min.js"
    assert "uplot.min.js" in str(exc.value) and "1.6.30" in str(exc.value)
    assert stub.hits == []


@pytest.mark.parametrize(
    ("url_path", "into"),
    [
        (LIB_PATH, "panel/vendor/lib.js"),
        (LIB_PATH, "panel/vendor/lib-2.0.0.js"),
        ("/files/lib.js", "panel/vendor/lib-1.0.0.js"),
    ],
    ids=["unversioned-name", "other-version", "unversioned-url"],
)
def test_a_url_entry_whose_file_name_lacks_the_urls_version_is_refused(
    tmp_path: Path, stub: _Stub, url_path: str, into: str
) -> None:
    manifests = _deployment(tmp_path, stub, into=into)
    manifests[0].vendor[0] = manifests[0].vendor[0].model_copy(update={"url": stub.base + url_path})

    with pytest.raises(vendor.AppVendorError, match="carry the version") as exc:
        vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    assert exc.value.app == "knob"
    assert stub.hits == []


@pytest.mark.parametrize(
    ("version", "file_name"),
    [
        ("1.6.3", "uplot-11.6.3.min.js"),
        ("1.6.3", "uplot-1.6.30.min.js"),
        ("1.6.3", "uplot-1.6.3.1.min.js"),
    ],
    ids=["longer-major", "longer-patch", "extra-part"],
)
def test_a_package_version_counts_only_as_a_whole_token(
    tmp_path: Path, stub: _Stub, version: str, file_name: str
) -> None:
    _write_app(
        tmp_path,
        "knob",
        [
            "  - package: uplot",
            f"    version: {version}",
            "    file: dist/uPlot.iife.min.js",
            f"    into: panel/vendor/{file_name}",
            f"    sha256: {UPLOT_SHA}",
        ],
    )

    with pytest.raises(vendor.AppVendorError, match="carry the version"):
        vendor.fetch_app_vendor(tmp_path, load(tmp_path), offline=False, quiet=True)

    assert stub.hits == []


def test_a_url_version_counts_only_as_a_whole_token(tmp_path: Path, stub: _Stub) -> None:
    manifests = _deployment(tmp_path, stub, into="panel/vendor/lib-1.0.0.1.js")

    with pytest.raises(vendor.AppVendorError, match="carry the version"):
        vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    assert stub.hits == []


def test_a_destination_that_cannot_be_created_fails_naming_the_app(
    tmp_path: Path, stub: _Stub
) -> None:
    manifests = _deployment(tmp_path, stub)
    (tmp_path / "apps" / "knob" / "panel").write_text("not a directory", encoding="utf-8")

    with pytest.raises(vendor.AppVendorError) as exc:
        vendor.fetch_app_vendor(tmp_path, manifests, offline=False, quiet=True)

    assert exc.value.app == "knob"
    assert exc.value.source == stub.base + LIB_PATH


# --- the app vendor mount is cached; proxied panel traffic is left alone ------------


#: A vendored file as a panel backend serves it, under the app panel vendor mount.
VENDORED = "/vendor/uplot-1.6.30.min.js"


@pytest.fixture
def panel_client() -> TestClient:
    """A panel backend's routes, and the same files as the terminal relays them."""
    app = FastAPI()
    app.add_middleware(NoCacheStaticMiddleware)

    @app.api_route(VENDORED, methods=["GET", "HEAD", "POST"])
    async def vendored() -> Response:
        return Response(content=UPLOT, media_type="application/javascript")

    @app.get("/vendor/no-store-1.0.0.js")
    async def vendored_no_store() -> Response:
        return Response(
            content=UPLOT,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/panel/knob" + VENDORED)
    async def proxied() -> Response:
        return Response(content=UPLOT, media_type="application/javascript")

    @app.get("/panel/knob/~c/abc123" + VENDORED)
    async def cap() -> Response:
        return Response(content=UPLOT, media_type="application/javascript")

    return TestClient(app)


IMMUTABLE = "public, max-age=31536000, immutable"


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_an_app_vendor_file_is_cached_as_immutable(panel_client: TestClient, method: str) -> None:
    resp = panel_client.request(method, VENDORED)
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == IMMUTABLE


def test_a_missing_app_vendor_file_is_not_cached(panel_client: TestClient) -> None:
    resp = panel_client.get("/vendor/missing-1.0.0.js")
    assert resp.status_code == 404
    assert resp.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"


def test_a_vendor_answer_to_another_method_is_not_immutable(panel_client: TestClient) -> None:
    resp = panel_client.post(VENDORED)
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] != IMMUTABLE


def test_proxied_panel_traffic_keeps_its_upstream_headers(panel_client: TestClient) -> None:
    resp = panel_client.get("/panel/knob" + VENDORED)
    assert resp.status_code == 200
    assert "Cache-Control" not in resp.headers


def test_a_cap_path_is_not_cached(panel_client: TestClient) -> None:
    resp = panel_client.get("/panel/knob/~c/abc123" + VENDORED)
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") != IMMUTABLE


def test_an_upstream_no_store_is_never_overridden(panel_client: TestClient) -> None:
    resp = panel_client.get("/vendor/no-store-1.0.0.js")
    assert resp.status_code == 200
    assert "no-store" in resp.headers["Cache-Control"]
    assert "immutable" not in resp.headers["Cache-Control"]


# --- package-based targets of OSPREY's own vendor manifest -----------------------------


def _own_manifest(base: str) -> dict:
    return {
        "assets": [
            {
                "name": "lib",
                "filename": "lib.js",
                "url": base + "/files/lib.js",
                "sha256": LIB_SHA,
                "targets": [
                    "web_terminal/static/vendor/",
                    {"dir": "services/knob/panel/vendor/", "base": "package"},
                ],
            }
        ]
    }


def test_a_package_target_lands_under_the_package_root(tmp_path: Path, stub: _Stub) -> None:
    interfaces_root, package_root = tmp_path / "interfaces", tmp_path / "pkg"
    with patch.object(vendor, "_load_manifest", return_value=_own_manifest(stub.base)):
        vendor.fetch_all(base_dir=interfaces_root, package_dir=package_root, quiet=True)
        ok, problems = vendor.verify_all(base_dir=interfaces_root, package_dir=package_root)

    assert (package_root / "services/knob/panel/vendor/lib.js").read_bytes() == LIB
    assert (interfaces_root / "web_terminal/static/vendor/lib.js").read_bytes() == LIB
    assert problems == [] and len(ok) == 2


def test_a_target_filter_fetches_only_that_target(tmp_path: Path, stub: _Stub) -> None:
    interfaces_root, package_root = tmp_path / "interfaces", tmp_path / "pkg"
    with patch.object(vendor, "_load_manifest", return_value=_own_manifest(stub.base)):
        downloaded = vendor.fetch_all(
            base_dir=interfaces_root,
            package_dir=package_root,
            quiet=True,
            targets=["/services/knob/panel/vendor"],
        )

    assert downloaded == [str(package_root / "services/knob/panel/vendor/lib.js")]
    assert not (interfaces_root / "web_terminal").exists()


def test_an_unknown_target_is_refused_before_any_download(tmp_path: Path, stub: _Stub) -> None:
    with (
        patch.object(vendor, "_load_manifest", return_value=_own_manifest(stub.base)),
        pytest.raises(ValueError, match="unknown vendor target"),
    ):
        vendor.fetch_all(base_dir=tmp_path, package_dir=tmp_path, targets=["nope/"])

    assert stub.hits == []


def test_an_unknown_target_base_is_refused(tmp_path: Path) -> None:
    manifest = {
        "assets": [{**_own_manifest("x")["assets"][0], "targets": [{"dir": "a", "base": "x"}]}]
    }
    with (
        patch.object(vendor, "_load_manifest", return_value=manifest),
        pytest.raises(ValueError, match="unknown base"),
    ):
        vendor.fetch_all(base_dir=tmp_path, package_dir=tmp_path)


def test_the_cli_passes_the_target_filter() -> None:
    with patch.object(vendor, "fetch_all", return_value=[]) as fetch_all:
        result = CliRunner().invoke(vendor_group, ["fetch", "-q", "--target", "a/", "-t", "b/"])

    assert result.exit_code == 0, result.output
    assert fetch_all.call_args.kwargs["targets"] == ("a/", "b/")


def test_the_cli_reports_an_unknown_target() -> None:
    with patch.object(vendor, "fetch_all", side_effect=ValueError("unknown vendor target(s): x")):
        result = CliRunner().invoke(vendor_group, ["fetch", "-q", "--target", "x"])

    assert result.exit_code == 1
    assert "unknown vendor target" in result.output
