"""Vendor files outside the interfaces tree: package targets and panel vendor dirs.

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
from osprey.cli.vendor_cmd import vendor as vendor_group
from osprey.interfaces import vendor
from osprey.interfaces._serving import run_app_server
from osprey.interfaces.common_middleware import NoCacheStaticMiddleware

LIB = b"/* lib 1.0 */ var lib = 1;\n"
UPLOT = b"/* uplot */ var uPlot = function () { return '#123456'; };\n"
LIB_SHA = hashlib.sha256(LIB).hexdigest()
UPLOT_SHA = hashlib.sha256(UPLOT).hexdigest()


class _Stub:
    """A local CDN: serves fixed bytes by path and counts the requests."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {
            "/files/lib.js": LIB,
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
