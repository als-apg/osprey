"""The artifact gallery's content sandbox (M10, default-deny point 1).

Every content route answers with ``Content-Security-Policy: sandbox
allow-scripts allow-modals allow-popups allow-downloads``, so agent-authored pages run in an opaque
origin; the shell is the one exempt route. Sandboxed pages hold no cookie and
every ``/u/<user>/`` asset is behind the proxy's ``auth_request``, so the
pages the gallery builds must be self-contained: in offline mode they load
nothing from the server, and the vendor libraries come from the directory
``osprey vendor fetch`` fills (a fixture tree here).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts import inline_assets
from osprey.interfaces.artifacts.app import create_app
from osprey.interfaces.artifacts.content_sandbox import not_content
from osprey.interfaces.opaque_origin import OPAQUE_CSP, is_inert

SANDBOX = "sandbox allow-scripts allow-modals allow-popups allow-downloads"

# Fixture vendor files: (manifest name, target dir relative to the vendor root, filename, text).
_FIXTURE_VENDOR = (
    ("marked", "artifacts/static/js/vendor/", "marked.min.js", "window.marked={/*FIX_MARKED*/};"),
    ("highlight.js", "artifacts/static/js/vendor/", "highlight.min.js", "/*FIX_HLJS*/"),
    ("KaTeX JS", "artifacts/static/js/vendor/", "katex.min.js", "/*FIX_KATEX_JS*/"),
    ("DOMPurify", "web_terminal/static/vendor/", "purify.min.js", "/*FIX_PURIFY*/"),
    (
        "highlight.js atom-one-dark theme",
        "artifacts/static/css/vendor/",
        "atom-one-dark.min.css",
        ".hljs{/*FIX_DARK*/}",
    ),
    (
        "highlight.js atom-one-light theme",
        "artifacts/static/css/vendor/",
        "atom-one-light.min.css",
        ".hljs{/*FIX_LIGHT*/}",
    ),
    (
        "KaTeX CSS",
        "artifacts/static/css/vendor/",
        "katex.min.css",
        "@font-face{font-family:KaTeX_AMS;font-style:normal;font-weight:400;"
        'src:url(fonts/KaTeX_AMS-Regular.woff2) format("woff2"),'
        'url(fonts/KaTeX_AMS-Regular.woff) format("woff"),'
        'url(fonts/KaTeX_AMS-Regular.ttf) format("truetype")}'
        ".katex{/*FIX_KATEX_CSS*/}",
    ),
    (
        "Plotly.js",
        "artifacts/static/js/vendor/",
        "plotly-3.3.1.min.js",
        "/**\n* plotly.js v3.3.1\n*/window.Plotly={/*FIX_PLOTLY*/};",
    ),
)
_FONT_BYTES = b"wOF2-fixture-font"


@pytest.fixture
def vendor_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "vendor-root"
    for _name, target, filename, text in _FIXTURE_VENDOR:
        path = root / target / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    fonts = root / "artifacts/static/css/vendor/fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    (fonts / "KaTeX_AMS-Regular.woff2").write_bytes(_FONT_BYTES)
    (fonts / "KaTeX_AMS-Regular.woff").write_bytes(b"woff")
    monkeypatch.setattr(inline_assets, "VENDOR_ROOT", root)
    return root


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(create_app(workspace_root=tmp_path / "ws"))


def _store(client: TestClient):
    return client.app.state.artifact_store


def _save(client, content: bytes, filename: str, artifact_type: str, mime: str):
    return _store(client).save_file(
        file_content=content,
        filename=filename,
        artifact_type=artifact_type,
        title=filename,
        description="",
        mime_type=mime,
        tool_source="test",
    )


def _notebook_bytes() -> bytes:
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell("print('NB_TOKEN')")]
    return nbformat.writes(nb).encode()


def _csp_policies(resp) -> list[str]:
    return resp.headers.get_list("content-security-policy")


# ---------------------------------------------------------------------------
# Reference scan: what a page would load from the server
# ---------------------------------------------------------------------------

_ATTR_RE = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_IMPORT_RE = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(?\s*)["']([^"']+)["']""")
_CSS_URL_RE = re.compile(r"""url\(\s*["']?([^)"']+)""")
_DATA_JS_RE = re.compile(r"data:text/javascript;base64,([A-Za-z0-9+/=]+)")
_DATA_CSS_RE = re.compile(r"data:text/css;base64,([A-Za-z0-9+/=]+)")
_MODULE_RE = re.compile(r"""<script type=["']module["']>(.*?)</script>""", re.DOTALL)
_STYLE_RE = re.compile(r"<style>(.*?)</style>", re.DOTALL)


def _decoded(pattern: re.Pattern[str], text: str) -> list[str]:
    return [base64.b64decode(m).decode("utf-8") for m in pattern.findall(text)]


def _server_references(page: str) -> list[str]:
    """Every subresource the page (and its data: modules/stylesheets) would fetch
    that is not a data: URL — root-relative, relative or absolute."""
    refs: list[str] = []
    refs += [v for v in _ATTR_RE.findall(page) if not v.startswith("data:")]
    css_blocks = _STYLE_RE.findall(page) + _decoded(_DATA_CSS_RE, page)
    for css in css_blocks:
        refs += [u for u in _CSS_URL_RE.findall(css) if not u.startswith("data:")]
    modules = _MODULE_RE.findall(page)
    seen: set[str] = set()
    while modules:
        source = modules.pop()
        if source in seen:
            continue
        seen.add(source)
        refs += [s for s in _IMPORT_RE.findall(source) if not s.startswith("data:")]
        modules += _decoded(_DATA_JS_RE, source)
    return refs


def _credentialed(refs: list[str]) -> list[str]:
    return [r for r in refs if not re.match(r"https?://", r) and not r.startswith("#")]


# ---------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------


class TestEveryContentRouteIsSandboxed:
    @pytest.mark.parametrize(
        ("content", "filename", "artifact_type", "mime"),
        [
            (b"<html><head></head><body>hi</body></html>", "a.html", "html", "text/html"),
            (b"<div class='plotly-graph-div'></div>", "p.html", "plot_html", "text/html"),
            (b"<table></table>", "t.html", "table_html", "text/html"),
            (b"<div></div>", "d.html", "dashboard_html", "text/html"),
            (b"\x89PNG\r\n\x1a\n", "i.png", "image", "image/png"),
            (b"plain", "x.txt", "text", "text/plain"),
            (b"# md", "m.md", "markdown", "text/markdown"),
            (b'{"a": 1}', "j.json", "json", "application/json"),
            (b"<svg xmlns='http://www.w3.org/2000/svg'/>", "s.svg", "image", "image/svg+xml"),
        ],
    )
    def test_files_route(self, client, content, filename, artifact_type, mime):
        entry = _save(client, content, filename, artifact_type, mime)
        resp = client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        # The opaque origin's inert list (PNG, plain text, JSON) is served with
        # nosniff and no sandbox, so a PDF or a picture renders; every other
        # type is sandboxed.
        if is_inert(mime):
            assert _csp_policies(resp) == []
            assert resp.headers["x-content-type-options"] == "nosniff"
        else:
            assert SANDBOX in _csp_policies(resp)

    def test_files_route_error_statuses(self, client):
        resp = client.get("/files/nope/x.html")
        assert resp.status_code == 404
        assert SANDBOX in _csp_policies(resp)

    def test_rendered_markdown(self, client):
        entry = _save(client, b"# Title", "m.md", "markdown", "text/markdown")
        resp = client.get(f"/api/markdown/{entry.id}/rendered")
        assert resp.status_code == 200
        assert SANDBOX in _csp_policies(resp)
        assert SANDBOX in _csp_policies(client.get("/api/markdown/nope/rendered"))

    def test_rendered_notebook(self, client):
        entry = _save(client, _notebook_bytes(), "n.ipynb", "notebook", "application/x-ipynb+json")
        resp = client.get(f"/api/notebooks/{entry.id}/rendered")
        assert resp.status_code == 200
        assert "NB_TOKEN" in resp.text
        assert SANDBOX in _csp_policies(resp)
        assert SANDBOX in _csp_policies(client.get("/api/notebooks/nope/rendered"))

    def test_unmatched_path(self, client):
        assert SANDBOX in _csp_policies(client.get("/no/such/route"))

    def test_constant_is_the_contract(self):
        assert OPAQUE_CSP == SANDBOX

    def test_shell_is_the_exempt_route(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert SANDBOX not in _csp_policies(resp)


class TestDefaultDeny:
    def test_a_later_route_without_opting_out_is_sandboxed(self, client):
        @client.app.get("/later-content")
        async def later():
            return PlainTextResponse("<b>agent</b>", media_type="text/html")

        resp = client.get("/later-content")
        assert resp.status_code == 200
        assert SANDBOX in _csp_policies(resp)

    def test_only_an_explicit_opt_out_is_exempt(self, client):
        @client.app.get("/later-chrome")
        @not_content
        async def chrome():
            return PlainTextResponse("ok")

        assert SANDBOX not in _csp_policies(client.get("/later-chrome"))

    def test_sandbox_is_its_own_policy_beside_an_upstream_one(self, client):
        upstream = "sandbox allow-scripts allow-same-origin"

        @client.app.get("/with-own-csp")
        async def own():
            return PlainTextResponse("x", headers={"Content-Security-Policy": upstream})

        policies = _csp_policies(client.get("/with-own-csp"))
        assert upstream in policies
        assert SANDBOX in policies  # a separate header: the stricter one still applies


# ---------------------------------------------------------------------------
# Self-contained pages
# ---------------------------------------------------------------------------


class TestOfflinePagesAreSelfContained:
    @pytest.mark.usefixtures("vendor_root")
    def test_rendered_markdown_references_nothing_credentialed(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        entry = _save(client, b"# T\n\n$x^2$", "m.md", "markdown", "text/markdown")
        page = client.get(f"/api/markdown/{entry.id}/rendered").text

        assert _credentialed(_server_references(page)) == []
        # Offline: nothing external either.
        assert _server_references(page) == []
        for token in ("FIX_MARKED", "FIX_HLJS", "FIX_KATEX_JS", "FIX_PURIFY", "FIX_KATEX_CSS"):
            assert token in page, token
        # hljs themes ride as data: stylesheets the theme manager can swap.
        styles = "".join(_decoded(_DATA_CSS_RE, page))
        assert "FIX_DARK" in styles and "FIX_LIGHT" in styles
        # KaTeX fonts as data: URIs, woff2 only.
        assert "data:font/woff2;base64," + base64.b64encode(_FONT_BYTES).decode() in page
        assert "KaTeX_AMS-Regular.woff)" not in page and ".ttf" not in page
        # The renderer module (and its imports) inlined, with the sanitising sink.
        modules = "".join(_decoded(_DATA_JS_RE, page))
        assert "export function renderMarkdownInto" in modules
        assert "DOMPurify.sanitize" in modules

    @pytest.mark.usefixtures("vendor_root")
    def test_rendered_markdown_has_no_innerhtml_sink_outside_md_render(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        entry = _save(client, b"# T", "m.md", "markdown", "text/markdown")
        page = client.get(f"/api/markdown/{entry.id}/rendered").text
        page_modules = _MODULE_RE.findall(page)
        assert any("renderMarkdownInto(" in m for m in page_modules)
        assert not any("innerHTML" in m or "renderMathInMarkdown" in m for m in page_modules)

    @pytest.mark.usefixtures("vendor_root")
    def test_agent_markdown_is_never_rewritten(self, client, monkeypatch):
        """The inliner runs on the template only: agent text that looks like a
        first-party module import passes through as data, untouched."""
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        hostile = b"<script type=\"module\">import x from '/static/js/nope.js';</script>"
        entry = _save(client, hostile, "m.md", "markdown", "text/markdown")
        resp = client.get(f"/api/markdown/{entry.id}/rendered")
        assert resp.status_code == 200
        assert "'/static/js/nope.js'" in resp.text  # inside the inert JSON source

    @pytest.mark.usefixtures("vendor_root")
    @pytest.mark.parametrize("artifact_type", ["plot_html", "html", "table_html", "dashboard_html"])
    def test_served_html_references_nothing_credentialed(self, client, monkeypatch, artifact_type):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        entry = _save(
            client,
            b'<html><head></head><body><div class="plotly-graph-div"></div></body></html>',
            "p.html",
            artifact_type,
            "text/html",
        )
        page = client.get(f"/files/{entry.id}/{entry.filename}").text
        assert _server_references(page) == []
        # The design system rides inline: theme-boot, tokens and the follower module.
        assert "initTheme({ role: 'follower' })" in page
        assert "--bg-primary" in page
        expected_plotly = 1 if artifact_type == "plot_html" else 0
        assert page.count("FIX_PLOTLY") == expected_plotly

    def test_plot_page_in_cdn_mode_keeps_cdn_plotly_and_inlines_first_party(
        self, client, monkeypatch
    ):
        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        entry = _save(
            client,
            b'<html><head></head><body><div class="plotly-graph-div"></div></body></html>',
            "p.html",
            "plot_html",
            "text/html",
        )
        page = client.get(f"/files/{entry.id}/{entry.filename}").text
        refs = _server_references(page)
        assert _credentialed(refs) == []
        assert refs == [inline_assets.cdn_url("Plotly.js")]

    def test_rendered_markdown_in_cdn_mode_keeps_cdn_urls(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        entry = _save(client, b"# T", "m.md", "markdown", "text/markdown")
        page = client.get(f"/api/markdown/{entry.id}/rendered").text
        refs = _server_references(page)
        assert _credentialed(refs) == []
        for name in ("marked", "highlight.js", "KaTeX JS", "KaTeX CSS", "DOMPurify"):
            assert inline_assets.cdn_url(name) in refs, name
        assert "export function renderMarkdownInto" in "".join(_decoded(_DATA_JS_RE, page))


class TestPlotlyOneInlineCopy:
    def _page(self, client, html: str, artifact_type: str = "html") -> str:
        entry = _save(client, html.encode(), "p.html", artifact_type, "text/html")
        return client.get(f"/files/{entry.id}/{entry.filename}").text

    @pytest.mark.usefixtures("vendor_root")
    def test_offline_any_version_any_tag_count_is_one_copy(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        page = self._page(
            client,
            "<html><head>"
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"'
            ' integrity="sha384-abc" crossorigin="anonymous"></script>'
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            '<script src="/static/js/vendor/plotly-3.3.1.min.js"></script>'
            "</head><body></body></html>",
            "plot_html",
        )
        assert page.count("FIX_PLOTLY") == 1
        assert "cdn.plot.ly" not in page
        assert "integrity=" not in page
        assert _server_references(page) == []

    @pytest.mark.usefixtures("vendor_root")
    def test_a_page_with_its_own_inline_bundle_gets_no_second(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        page = self._page(
            client,
            "<html><head><script>/**\n* plotly.js v2.35.2\n*/window.Plotly={};</script>"
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
            "</head><body></body></html>",
            "plot_html",
        )
        assert "FIX_PLOTLY" not in page
        assert "cdn.plot.ly" not in page

    @pytest.mark.usefixtures("vendor_root")
    def test_cdn_mode_inlines_a_local_static_tag_and_keeps_cdn_tags(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        local = self._page(
            client,
            '<html><head><script src="/static/js/vendor/plotly-3.3.1.min.js"></script>'
            "</head><body></body></html>",
            "plot_html",
        )
        assert local.count("FIX_PLOTLY") == 1
        assert _credentialed(_server_references(local)) == []

        cdn = self._page(
            client,
            '<html><head><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
            "</head><body></body></html>",
            "plot_html",
        )
        assert "https://cdn.plot.ly/plotly-2.35.2.min.js" in cdn
        assert "FIX_PLOTLY" not in cdn


class TestGalleryShell:
    @pytest.mark.usefixtures("vendor_root")
    def test_offline_shell_inlines_dompurify(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        assert "FIX_PURIFY" in client.get("/").text

    def test_cdn_shell_loads_dompurify_from_cdn(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        assert (
            f'<script src="{inline_assets.cdn_url("DOMPurify")}"></script>' in client.get("/").text
        )


class TestInliner:
    def test_module_data_url_resolves_relative_and_root_imports(self):
        url = inline_assets.module_data_url("/design-system/js/theme-manager.js")
        source = base64.b64decode(url.split(",", 1)[1]).decode()
        specs = _IMPORT_RE.findall(source)
        assert specs and all(s.startswith("data:text/javascript;base64,") for s in specs)

    def test_unknown_first_party_module_is_an_error(self):
        with pytest.raises(FileNotFoundError):
            inline_assets.module_data_url("/design-system/js/no-such.js")
        with pytest.raises(FileNotFoundError):
            inline_assets.module_data_url("/design-system/../../vendor.py")

    def test_inline_script_cannot_close_early(self):
        block = inline_assets.inline_script("var s = '</script><img onerror=x>';")
        assert block.count("</script>") == 1 and block.endswith("</script>")
