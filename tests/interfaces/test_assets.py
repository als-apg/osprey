"""Tests for the first-party asset set: fingerprints, caching, gzip, the module graph."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from urllib.parse import urljoin

import pytest
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.testclient import TestClient

from osprey.interfaces import _assets
from osprey.interfaces._app_setup import AssetStaticFiles
from osprey.interfaces._assets import (
    CACHE_IMMUTABLE,
    CACHE_REVALIDATE,
    AssetSet,
    etag_matches,
    split_version,
)
from osprey.interfaces.common_middleware import NoCacheStaticMiddleware

#: Long enough to pass the size floor, repetitive enough to compress well.
BIG_MODULE = "export const filler = 'abcdefghij';\n" * 100


@pytest.fixture(autouse=True)
def _no_rescan_throttle(monkeypatch):
    """Every ``version()`` call walks the set, so a test sees its own edits."""
    monkeypatch.setattr(_assets, "_RESCAN_INTERVAL_S", 0.0)


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    root = tmp_path / "static"
    (root / "js" / "lib").mkdir(parents=True)
    (root / "js" / "app.js").write_text(
        "import { a } from './lib/a.js';\n"
        "import '/design-system/js/shared.js';\n"
        "export { b } from './lib/b.js';\n"
        "const later = () => import('./lib/lazy.js');\n"
        "import './lib/missing.js';\n"
    )
    (root / "js" / "lib" / "a.js").write_text("import './b.js';\nexport const a = 1;\n")
    (root / "js" / "lib" / "b.js").write_text("export const b = 2;\n")
    (root / "js" / "lib" / "lazy.js").write_text("export const lazy = 3;\n")
    (root / "js" / "big.js").write_text(BIG_MODULE)
    (root / "small.css").write_text("a{}")
    (root / "image.png").write_bytes(bytes(range(256)) * 16)
    return root


@pytest.fixture
def design_dir(tmp_path: Path) -> Path:
    root = tmp_path / "design"
    (root / "js").mkdir(parents=True)
    (root / "js" / "shared.js").write_text("export const shared = 0;\n")
    return root


@pytest.fixture
def assets(static_dir: Path, design_dir: Path) -> AssetSet:
    return AssetSet({"/design-system": design_dir, "/static": static_dir})


@pytest.fixture
def client(assets: AssetSet, static_dir: Path, design_dir: Path) -> TestClient:
    app = Starlette(
        routes=[
            Mount(
                "/design-system",
                app=AssetStaticFiles(directory=str(design_dir), assets=assets),
            ),
            Mount("/static", app=AssetStaticFiles(directory=str(static_dir), assets=assets)),
        ]
    )
    app.add_middleware(NoCacheStaticMiddleware)
    return TestClient(app)


class TestSplitVersion:
    def test_a_fingerprinted_path_gives_both_parts(self):
        assert split_version("_v/0123456789abcdef/js/app.js") == ("0123456789abcdef", "js/app.js")

    @pytest.mark.parametrize(
        "path",
        [
            "js/app.js",
            "_v/short/js/app.js",
            "_v/0123456789ABCDEF/js/app.js",
            "js/_v/0123456789abcdef/x",
        ],
    )
    def test_anything_else_is_a_plain_path(self, path):
        assert split_version(path) == (None, path)


class TestEtagMatches:
    def test_matches_any_of_the_listed_tags(self):
        assert etag_matches('"a", "b"', '"b"')
        assert etag_matches('W/"a"', '"a"')
        assert etag_matches("*", '"a"')

    def test_no_header_or_another_tag_does_not_match(self):
        assert not etag_matches(None, '"a"')
        assert not etag_matches('"c"', '"a"', '"b"')


class TestFingerprint:
    def test_is_stable_while_nothing_changes(self, assets):
        assert assets.version() == assets.version()
        assert len(assets.version()) == 16

    def test_changes_when_any_file_changes(self, assets, design_dir):
        before = assets.version()
        (design_dir / "js" / "shared.js").write_text("export const shared = 1;\n")
        assert assets.version() != before

    def test_changes_when_a_file_is_added(self, assets, static_dir):
        before = assets.version()
        (static_dir / "js" / "new.js").write_text("export {};\n")
        assert assets.version() != before

    def test_url_carries_the_fingerprint_under_its_mount(self, assets):
        fingerprint = assets.version()
        assert assets.url("/static/js/app.js") == f"/static/_v/{fingerprint}/js/app.js"
        assert assets.url("/static/js/app.js?x=1") == f"/static/_v/{fingerprint}/js/app.js?x=1"
        assert assets.root("/design-system") == f"/design-system/_v/{fingerprint}/"

    def test_url_leaves_other_paths_alone(self, assets):
        assert assets.url("/api/panels") == "/api/panels"
        assert assets.url("https://cdn.example/x.js") == "https://cdn.example/x.js"
        already = assets.url("/static/js/app.js")
        assert assets.url(already) == already


class TestModuleGraph:
    def test_lists_everything_statically_reachable_once(self, assets):
        assert assets.module_graph(["/static/js/app.js"]) == (
            "/static/js/app.js",
            "/static/js/lib/a.js",
            "/design-system/js/shared.js",
            "/static/js/lib/b.js",
        )

    def test_leaves_out_dynamic_imports_and_missing_files(self, assets):
        graph = assets.module_graph(["/static/js/app.js"])
        assert "/static/js/lib/lazy.js" not in graph
        assert "/static/js/lib/missing.js" not in graph

    def test_follows_an_edit(self, assets, static_dir):
        assets.module_graph(["/static/js/lib/b.js"])
        (static_dir / "js" / "lib" / "b.js").write_text(
            "import './lazy.js';\nexport const b = 2;\n"
        )
        assert assets.module_graph(["/static/js/lib/b.js"]) == (
            "/static/js/lib/b.js",
            "/static/js/lib/lazy.js",
        )


class TestServing:
    def test_a_plain_url_is_revalidated(self, client):
        resp = client.get("/static/js/app.js")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == CACHE_REVALIDATE

    def test_the_current_fingerprint_is_immutable(self, client, assets):
        resp = client.get(assets.url("/static/js/app.js"))
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == CACHE_IMMUTABLE
        assert resp.text == client.get("/static/js/app.js").text

    def test_a_stale_fingerprint_serves_the_file_but_is_not_kept(self, client):
        """A page from before a deploy still loads, and its files are not
        stored for a year under a name that no longer describes them."""
        resp = client.get("/static/_v/0000000000000000/js/app.js")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == CACHE_REVALIDATE

    def test_a_missing_file_is_a_404_under_either_name(self, client, assets):
        assert client.get("/static/js/nope.js").status_code == 404
        assert client.get(assets.url("/static/js/nope.js")).status_code == 404

    def test_the_etag_is_the_content_and_answers_304(self, client, static_dir):
        first = client.get("/static/js/app.js")
        etag = first.headers["etag"]

        again = client.get("/static/js/app.js", headers={"if-none-match": etag})
        assert again.status_code == 304
        assert again.headers["cache-control"] == CACHE_REVALIDATE

        (static_dir / "js" / "app.js").write_text("export const changed = true;\n")
        changed = client.get("/static/js/app.js", headers={"if-none-match": etag})
        assert changed.status_code == 200
        assert changed.headers["etag"] != etag

    def test_a_large_text_file_is_gzipped_for_a_client_that_accepts_it(self, client):
        resp = client.get("/static/js/big.js", headers={"accept-encoding": "gzip"})
        assert resp.headers["content-encoding"] == "gzip"
        assert "accept-encoding" in resp.headers["vary"].lower()
        assert int(resp.headers["content-length"]) < len(BIG_MODULE) // 2
        assert resp.text == BIG_MODULE

    def test_the_gzipped_body_is_the_file(self, assets, static_dir):
        path = static_dir / "js" / "big.js"
        entry = assets.compressed(path, path.stat())
        assert entry.compressed is not None
        assert gzip.decompress(entry.compressed).decode() == BIG_MODULE
        assert entry.compressed_etag != entry.etag

    def test_identity_is_served_to_a_client_that_does_not_accept_gzip(self, client):
        resp = client.get("/static/js/big.js", headers={"accept-encoding": "identity"})
        assert "content-encoding" not in resp.headers
        assert resp.text == BIG_MODULE

    @pytest.mark.parametrize("path", ["/static/small.css", "/static/image.png"])
    def test_small_and_binary_files_are_not_gzipped(self, client, path):
        resp = client.get(path, headers={"accept-encoding": "gzip"})
        assert resp.status_code == 200
        assert "content-encoding" not in resp.headers

    def test_a_byte_range_addresses_the_file_itself(self, client):
        resp = client.get(
            "/static/js/big.js", headers={"accept-encoding": "gzip", "range": "bytes=0-9"}
        )
        assert resp.status_code == 206
        assert "content-encoding" not in resp.headers
        assert resp.text == BIG_MODULE[:10]

    def test_either_etag_revalidates(self, client):
        gzipped = client.get("/static/js/big.js", headers={"accept-encoding": "gzip"})
        plain = client.get("/static/js/big.js", headers={"accept-encoding": "identity"})
        assert gzipped.headers["etag"] != plain.headers["etag"]
        for etag in (gzipped.headers["etag"], plain.headers["etag"]):
            resp = client.get("/static/js/big.js", headers={"if-none-match": etag})
            assert resp.status_code == 304


class TestStyleBundles:
    """Sheets of the set, served as one: same order, same rules, one request."""

    @pytest.fixture
    def sheets(self, static_dir: Path, design_dir: Path) -> tuple[str, ...]:
        (design_dir / "css").mkdir()
        (design_dir / "css" / "tokens.css").write_text(":root { --ink: #111; }\n")
        (static_dir / "fonts").mkdir()
        (static_dir / "fonts" / "body.woff2").write_bytes(b"\x00font")
        (static_dir / "fonts" / "fonts.css").write_text(
            '@charset "utf-8";\n'
            "@font-face { font-family: Body; src: url('./body.woff2') format('woff2'); }\n"
            ".mark { background: url(data:image/png;base64,AAAA); }\n"
            ".far { background: url(https://example.org/far.png); }\n"
        )
        (static_dir / "css").mkdir()
        (static_dir / "css" / "shell.css").write_text(
            ".shell { background: url(../image.png); color: var(--ink); }\n" * 60
        )
        return ("/design-system/css/tokens.css", "/static/fonts/fonts.css", "/static/css/shell.css")

    @pytest.fixture
    def bundled(self, assets: AssetSet, sheets: tuple[str, ...]) -> AssetSet:
        assets.define_bundle("shell", "/static", sheets)
        return assets

    def test_the_address_is_fingerprinted_under_the_mount(self, bundled):
        assert bundled.bundle_url("shell") == f"/static/_v/{bundled.version()}/_bundle/shell.css"

    def test_holds_every_sheet_once_in_the_order_given(self, bundled, sheets):
        body = bundled.bundle("shell").body.decode()
        marks = [body.index(f"/* {sheet} */") for sheet in sheets]
        assert marks == sorted(marks)
        assert body.index("--ink: #111") < body.index("font-family: Body")
        assert body.index("font-family: Body") < body.index(".shell {")
        assert body.count("font-family: Body") == 1

    def test_a_relative_reference_still_reaches_its_file(self, bundled, client):
        url = bundled.bundle_url("shell")
        body = bundled.bundle("shell").body.decode()
        references = re.findall(r'url\("([^":]+)"\)', body)
        assert len(set(references)) == 2  # the font and the image
        reached = {urljoin(url, reference) for reference in references}
        version = bundled.version()
        assert reached == {
            f"/static/_v/{version}/fonts/body.woff2",
            f"/static/_v/{version}/image.png",
        }
        for address in reached:
            answer = client.get(address)
            assert answer.status_code == 200
            assert answer.headers["cache-control"] == CACHE_IMMUTABLE

    def test_references_that_are_not_relative_are_left_alone(self, bundled):
        body = bundled.bundle("shell").body.decode()
        assert "url(data:image/png;base64,AAAA)" in body
        assert "url(https://example.org/far.png)" in body

    def test_a_charset_rule_is_dropped(self, bundled):
        assert "@charset" not in bundled.bundle("shell").body.decode()

    @pytest.mark.parametrize("rule", ["@import url('other.css');", "@namespace svg url(x);"])
    def test_a_sheet_that_must_lead_its_own_file_is_not_bundled(self, bundled, static_dir, rule):
        (static_dir / "css" / "shell.css").write_text(f"{rule}\n.shell {{}}\n")
        assert bundled.bundle("shell") is None
        assert bundled.bundle_url("shell") is None

    def test_such_a_rule_inside_a_comment_does_not_count(self, bundled, static_dir):
        (static_dir / "css" / "shell.css").write_text("/* @import is not used */\n.shell {}\n")
        assert bundled.bundle_url("shell") is not None

    def test_a_missing_sheet_or_an_unknown_name_gives_no_bundle(self, bundled, static_dir):
        assert bundled.bundle_url("other") is None
        (static_dir / "css" / "shell.css").unlink()
        assert bundled.bundle_url("shell") is None

    def test_follows_an_edit_to_any_sheet(self, bundled, design_dir):
        before = bundled.bundle_url("shell")
        (design_dir / "css" / "tokens.css").write_text(":root { --ink: #222; }\n")
        assert bundled.bundle_url("shell") != before
        assert "--ink: #222" in bundled.bundle("shell").body.decode()

    def test_is_served_immutable_compressed_and_revalidated(self, bundled, client):
        url = bundled.bundle_url("shell")
        plain = client.get(url, headers={"Accept-Encoding": "identity"})
        assert plain.status_code == 200
        assert plain.headers["content-type"] == "text/css; charset=utf-8"
        assert plain.headers["cache-control"] == CACHE_IMMUTABLE
        assert "content-encoding" not in plain.headers
        assert plain.content == bundled.bundle("shell").body

        packed = client.get(url, headers={"Accept-Encoding": "gzip"})
        assert packed.headers["content-encoding"] == "gzip"
        assert packed.content == plain.content  # the client decodes it
        assert int(packed.headers["content-length"]) < len(plain.content)

        for etag in (plain.headers["etag"], packed.headers["etag"]):
            again = client.get(url, headers={"If-None-Match": etag})
            assert again.status_code == 304
            assert again.content == b""

    def test_a_stale_fingerprint_serves_the_bundle_but_is_not_kept(self, bundled, client):
        assert bundled.bundle_url("shell") is not None
        stale = client.get("/static/_v/0000000000000000/_bundle/shell.css")
        assert stale.status_code == 200
        assert stale.headers["cache-control"] == CACHE_REVALIDATE

    def test_an_unknown_bundle_is_a_404(self, bundled, client):
        version = bundled.version()
        assert client.get(f"/static/_v/{version}/_bundle/other.css").status_code == 404
        assert client.get(f"/static/_v/{version}/_bundle/shell.js").status_code == 404
