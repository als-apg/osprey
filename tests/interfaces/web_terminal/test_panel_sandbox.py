"""The terminal's panel sandbox: every agent-reachable document in an opaque origin.

M10's second default-deny point (``web_terminal/panel_sandbox.py``). Pinned
here over the whole app:

* the policy is the terminal's **own** header — an upstream
  ``sandbox allow-scripts allow-same-origin`` sits beside it and cannot loosen
  it — on every status, for every response type off the closed non-document
  allowlist, ``nosniff`` with it;
* only a panel's declared entry pages escape it, matched after percent-decoding
  and dot-segment removal (``%2e``, ``%2f``, ``%5c``, double encoding);
* ``/panel-static`` is never allowlisted, its agent ``.svg`` included, and its
  HTML is served self-contained (the reference panel references no server
  path);
* :data:`AGENT_CONTENT_ROUTES` matches the companion servers' route tables (the
  drift test) and every row it names lands in a sandbox through the terminal.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

import osprey.interfaces.design_system.panels as design_system_panels
from osprey.interfaces.opaque_origin import OPAQUE_CSP, document_headers
from osprey.interfaces.web_terminal.agent_content_routes import (
    AGENT_CONTENT_ROUTES,
    SCANNED_PACKAGES,
    RouteRef,
)
from osprey.interfaces.web_terminal.panel_discovery import DiscoveredPanel
from osprey.interfaces.web_terminal.panel_sandbox import (
    FRAMEWORK_ENTRY_PAGES,
    NON_DOCUMENT_TYPES,
    builtin_entry_pages,
    is_non_document_type,
    matches_entry_page,
    needs_panel_sandbox,
    path_readings,
    validate_entry_template,
)
from osprey.interfaces.web_terminal.routes.proxy import _PANEL_STATE_MAP
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID, SIDECAR_PANELS, SidecarPanel
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

from ._proxy_fakes import _FakeStreamResponse, panel_app

POLICY = "sandbox allow-scripts allow-modals allow-popups allow-downloads"

#: Document types: each gets the sandbox and nosniff (``None`` = no type at all).
DOCUMENT_TYPES = [
    "text/html; charset=utf-8",
    "image/svg+xml",
    "application/xml",
    "text/xml",
    "text/markdown",
    "application/octet-stream",
    "text/event-stream",
    None,
]

#: Non-document types: relayed with no sandbox of the terminal's.
PLAIN_TYPES = [
    # The opaque origin's inert list (``opaque_origin.INERT_TYPES``).
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/json",
    "application/problem+json",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "text/css",
    "text/javascript",
    "application/javascript",
    "font/woff2",
    "font/ttf",
]

UPSTREAM = "http://127.0.0.1:9999"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def app_and_client(workspace_dir):
    """The whole app: a runtime-registered panel, a config panel, framework URLs set."""
    custom = [
        # A runtime registration: no configDefined, so it declares nothing.
        {"id": "agent-page", "label": "AGENT", "url": UPSTREAM},
        # A config panel: its configured path is its entry document.
        {"id": "cfg", "label": "CFG", "url": UPSTREAM, "path": "/app/", "configDefined": True},
    ]
    for app, client in panel_app(workspace_dir, custom):
        for attr in _PANEL_STATE_MAP.values():
            setattr(app.state, attr, UPSTREAM)
        yield app, client


def _upstream(app, *, content_type, status=200, body="<html><body>x</body></html>", headers=()):
    """Answer every proxied request with one response; returns the request mock."""

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        reply = {} if content_type is None else {"content-type": content_type}
        reply.update(dict(extra_headers))
        return httpx.Response(status_code=status, content=body.encode(), headers=reply)

    extra_headers = list(headers)
    mock = AsyncMock(side_effect=fake_request)
    app.state.proxy_client.request = mock
    return mock


def _policies(response) -> list[str]:
    return response.headers.get_list("content-security-policy")


def _sandboxed(response) -> bool:
    return POLICY in _policies(response)


# ---------------------------------------------------------------------------
# The policy module on its own
# ---------------------------------------------------------------------------


class TestPolicyRules:
    def test_the_directive_is_the_opaque_origins(self):
        assert OPAQUE_CSP == POLICY

    @pytest.mark.parametrize("content_type", PLAIN_TYPES)
    def test_non_document_types_are_allowlisted(self, content_type):
        assert is_non_document_type(content_type)
        assert not needs_panel_sandbox("/anything", content_type, ())

    @pytest.mark.parametrize("content_type", DOCUMENT_TYPES)
    def test_document_types_are_sandboxed(self, content_type):
        assert not is_non_document_type(content_type)
        assert needs_panel_sandbox("/anything", content_type, ())

    @pytest.mark.parametrize(
        "content_type",
        [
            "application/json;x=,text/html",
            "application/json, text/html",
            "image/png,text/html",
            'application/json; charset="a,b"',
            "application/json\x0b",
            "image/png/x",
            "",
        ],
    )
    def test_a_malformed_or_listed_type_is_a_document(self, content_type):
        assert not is_non_document_type(content_type)
        assert needs_panel_sandbox("/anything", content_type, ())

    def test_the_allowlist_is_closed_to_documents(self):
        for content_type in NON_DOCUMENT_TYPES:
            assert not content_type.startswith(("text/html", "image/svg", "text/plain"))
            assert "xml" not in content_type

    @pytest.mark.parametrize(
        ("path", "templates", "expected"),
        [
            ("", ("/",), True),
            ("/", ("/",), True),
            ("index.html", ("/",), False),
            ("lab", ("/lab",), True),
            ("lab/", ("/lab",), True),
            ("labx", ("/lab",), False),
            ("lab/tree/a.ipynb", ("/lab/*",), True),
            ("s/screen-1", ("/s/*",), True),
            ("drafts/x", ("/s/*",), False),
            # Dot segments, literal and encoded, are resolved before matching.
            ("lab/../files/x.html", ("/lab/*",), False),
            ("lab/%2e%2e/files/x.html", ("/lab/*",), False),
            ("lab/%2E%2E/files/x.html", ("/lab/*",), False),
            # Encoded slash and backslash.
            ("lab%2f..%2ffiles%2fx.html", ("/lab/*",), False),
            ("lab%5c..%5cfiles%5cx.html", ("/lab/*",), False),
            ("lab\\..\\files\\x.html", ("/lab/*",), False),
            # Double encoding decodes to a fixed point.
            ("lab/%252e%252e/files/x.html", ("/lab/*",), False),
            ("lab/%25252e%25252e/files/x.html", ("/lab/*",), False),
            # A control character matches nothing.
            ("lab\x00", ("/lab",), False),
            # Encoded but harmless still matches.
            ("lab/tree/a%20b.ipynb", ("/lab/*",), True),
            # No templates: nothing is an entry page.
            ("", (), False),
        ],
    )
    def test_entry_page_matching(self, path, templates, expected):
        assert matches_entry_page(path, templates) is expected

    def test_every_reading_must_match(self):
        readings = path_readings("lab/%2e%2e/files/x.html")
        assert readings == ("/lab/%2e%2e/files/x.html", "/files/x.html")

    @pytest.mark.parametrize(
        "template", ["lab", "/lab/*/x", "/lab*", "/panel-static/x", "/a/../b", "/%2e", "/a\\b"]
    )
    def test_invalid_templates_are_refused(self, template):
        assert validate_entry_template(template) is not None
        assert not matches_entry_page(template.lstrip("/"), (template,))

    def test_sidecar_panel_declares_no_entry_page_by_default(self):
        assert SidecarPanel("x", "a:b").entry_pages == ()

    def test_every_declared_template_is_valid(self):
        declared = [*FRAMEWORK_ENTRY_PAGES.values()]
        declared += [panel.entry_pages for panel in SIDECAR_PANELS.values()]
        for templates in declared:
            assert templates, "a declared panel names at least one entry page"
            for template in templates:
                assert validate_entry_template(template) is None, template

    def test_every_framework_panel_names_its_entry_documents(self):
        framework_ids = {d.panel_id for d in FRAMEWORK_WEB_SERVERS.values()}
        assert set(FRAMEWORK_ENTRY_PAGES) == framework_ids

    def test_jupyterlabs_application_is_its_entry_and_its_content_is_not(self):
        pages = builtin_entry_pages("jupyter")
        for path in ("lab", "lab/tree/x.ipynb", "tree", "doc/tree/x.ipynb", "notebooks/x.ipynb"):
            assert matches_entry_page(path, pages), path
        for path in (
            "files/x.html",
            "nbconvert/html/x.ipynb",
            "view/x.html",
            "",
            "api/status",
            "lab/extensions/ext/static/index.html",
            "lab/api/settings",
            "static/lab/index.html",
        ):
            assert not matches_entry_page(path, pages), path

    def test_osprey_names_no_canopy_page(self):
        # The framework allowlist never names a CANOPY page. On the integration
        # line CANOPY is a registered sidecar panel, and its entry pages come
        # only from its own registry entry, which never lets a draft through.
        assert CANOPY_PANEL_ID not in FRAMEWORK_ENTRY_PAGES
        canopy_pages = SIDECAR_PANELS[CANOPY_PANEL_ID].entry_pages
        assert not any(t.startswith("/drafts") for t in canopy_pages)
        assert not matches_entry_page("drafts/x.html", canopy_pages)


# ---------------------------------------------------------------------------
# Through the proxy
# ---------------------------------------------------------------------------


class TestProxiedPanelSandbox:
    def test_an_upstream_policy_cannot_remove_the_terminals(self, app_and_client):
        app, client = app_and_client
        upstream_policy = "sandbox allow-scripts allow-same-origin"
        _upstream(
            app,
            content_type="text/html",
            headers=[("content-security-policy", upstream_policy)],
        )

        resp = client.get("/panel/agent-page/report.html")

        policies = _policies(resp)
        assert POLICY in policies, "the terminal's policy is its own header"
        assert upstream_policy in policies, "the upstream's is relayed beside it, not merged"
        assert len(policies) == 2
        assert resp.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize("content_type", DOCUMENT_TYPES)
    def test_every_document_type_is_sandboxed(self, app_and_client, content_type):
        app, client = app_and_client
        _upstream(app, content_type=content_type)

        resp = client.get("/panel/agent-page/thing")

        assert _sandboxed(resp)
        assert resp.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize("content_type", PLAIN_TYPES)
    def test_non_document_types_pass_untouched(self, app_and_client, content_type):
        app, client = app_and_client
        _upstream(app, content_type=content_type, body="{}")

        resp = client.get("/panel/agent-page/thing")

        # The terminal's own rule adds nothing here. ``agent-page`` is a runtime
        # registration, so the opaque origin's headers still apply: an inert
        # type gets ``nosniff`` alone, anything else (a stylesheet, a script)
        # its sandbox.
        expected = document_headers(content_type)
        assert _sandboxed(resp) == ("Content-Security-Policy" in expected)
        assert resp.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize("status", [200, 201, 304, 400, 403, 404, 500, 503])
    def test_every_status(self, app_and_client, status):
        app, client = app_and_client
        _upstream(app, content_type="text/html", status=status)

        resp = client.get("/panel/agent-page/x")

        assert resp.status_code == status
        assert _sandboxed(resp)

    def test_a_relayed_redirect_is_sandboxed(self, app_and_client):
        app, client = app_and_client
        _upstream(
            app,
            content_type="text/html",
            status=302,
            headers=[("location", f"{UPSTREAM}/elsewhere")],
        )

        resp = client.get("/panel/agent-page/x", follow_redirects=False)

        assert resp.status_code == 302
        assert _sandboxed(resp)

    def test_the_proxys_own_refusals_are_sandboxed(self, app_and_client):
        _app, client = app_and_client
        resp = client.get("/panel/no-such-panel/x")

        assert resp.status_code == 404
        assert _sandboxed(resp)
        assert resp.headers["x-content-type-options"] == "nosniff"

    def test_an_upstream_failure_is_sandboxed(self, app_and_client):
        app, client = app_and_client
        app.state.proxy_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))

        resp = client.get("/panel/agent-page/x")

        assert resp.status_code == 502
        assert _sandboxed(resp)

    def test_a_streamed_response_is_sandboxed(self, app_and_client):
        app, client = app_and_client
        app.state.proxy_client.send = AsyncMock(return_value=_FakeStreamResponse())

        resp = client.get("/panel/agent-page/stream", headers={"accept": "text/event-stream"})

        assert _sandboxed(resp)

    def test_a_type_list_that_ends_in_html_is_sandboxed(self, app_and_client):
        app, client = app_and_client
        _upstream(app, content_type="application/json;x=,text/html")

        resp = client.get("/panel/agent-page/x")

        assert _sandboxed(resp)

    def test_an_upstream_nosniff_is_not_doubled(self, app_and_client):
        app, client = app_and_client
        _upstream(app, content_type="text/html", headers=[("x-content-type-options", "nosniff")])

        resp = client.get("/panel/agent-page/x")

        assert resp.headers.get_list("x-content-type-options") == ["nosniff"]

    def test_a_runtime_registration_has_no_entry_page(self, app_and_client):
        app, client = app_and_client
        _upstream(app, content_type="text/html")

        assert _sandboxed(client.get("/panel/agent-page"))
        assert _sandboxed(client.get("/panel/agent-page/"))

    def test_a_config_panels_declared_path_is_its_entry_page(self, app_and_client):
        app, client = app_and_client
        _upstream(app, content_type="text/html")

        assert not _sandboxed(client.get("/panel/cfg/app/"))
        assert _sandboxed(client.get("/panel/cfg/other.html"))
        assert _sandboxed(client.get("/panel/cfg/app/../other.html"))

    def test_an_allowlisted_page_is_not_sandboxed(self, app_and_client):
        app, client = app_and_client
        _upstream(app, content_type="text/html")

        for url in ("/panel/jupyter/lab", "/panel/jupyter/lab/tree/x.ipynb", "/panel/ariel/"):
            resp = client.get(url)
            assert resp.status_code == 200, url
            assert not _sandboxed(resp), url
            assert "x-content-type-options" not in resp.headers, url
        assert not _sandboxed(client.get("/panel/ariel"))

    @pytest.mark.parametrize(
        "url",
        [
            "/panel/jupyter/files/report.html",
            "/panel/jupyter/view/report.html",
            "/panel/jupyter/lab/%2e%2e/files/report.html",
            "/panel/jupyter/lab%2f..%2ffiles%2freport.html",
            "/panel/jupyter/lab%5c..%5cfiles%5creport.html",
            "/panel/jupyter/lab/%252e%252e/files/report.html",
            "/panel/ariel/api/attachments/1",
            "/panel/ariel/index.html",
            "/panel/artifacts/files/a/b.html",
        ],
    )
    def test_a_spelling_that_leaves_the_entry_page_is_sandboxed(self, app_and_client, url):
        app, client = app_and_client
        _upstream(app, content_type="text/html")

        resp = client.get(url)

        assert _sandboxed(resp), url

    def test_the_hubs_own_assets_keep_their_types(self, app_and_client):
        _app, client = app_and_client
        css = client.get("/panel/jupyter/design-system/css/tokens.css")
        js = client.get("/panel/jupyter/terminal-static/js/api.js")

        assert css.status_code == 200 and not _sandboxed(css)
        assert js.status_code == 200 and not _sandboxed(js)
        # A miss on those routes is a text answer, sandboxed like any other.
        assert _sandboxed(client.get("/panel/jupyter/design-system/css/missing.css"))


# ---------------------------------------------------------------------------
# /panel-static
# ---------------------------------------------------------------------------


@pytest.fixture
def discovered(app_and_client, tmp_path):
    """A discovered agent bundle and the reference panel, served from ``/panel-static``."""
    app, client = app_and_client
    bundle = tmp_path / "agent-bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text("<html><body>agent</body></html>", encoding="utf-8")
    (bundle / "logo.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', encoding="utf-8"
    )
    (bundle / "data.json").write_text("{}", encoding="utf-8")
    reference = Path(design_system_panels.__file__).parent / "reference"
    app.state.discovered_panel_dirs = {
        "agent": DiscoveredPanel("agent", "Agent", "index.html", bundle),
        "reference": DiscoveredPanel("reference", "Reference", "index.html", reference),
    }
    return app, client


#: A server path in a page: an ``src``/``href`` attribute, a CSS ``url()`` or an
#: import specifier that starts at the root.
_SERVER_PATH_RE = re.compile(
    r"""(?:\b(?:src|href)\s*=\s*["']|url\(\s*["']?|\bfrom\s*["']|\bimport\s*["'(])/(?!/)"""
)


class TestPanelStatic:
    @pytest.mark.parametrize(
        ("urls", "status"),
        [
            pytest.param(
                ("/panel-static/agent/", "/panel-static/agent/index.html"),
                200,
                id="the-entry-page-is-never-allowlisted",
            ),
            pytest.param(
                ("/panel-static/agent/nope.html", "/panel-static/nobody/"),
                404,
                id="a-miss",
            ),
        ],
    )
    def test_the_entry_page_and_a_miss_are_sandboxed(self, discovered, urls, status):
        _app, client = discovered
        for url in urls:
            resp = client.get(url)
            assert resp.status_code == status, url
            assert _sandboxed(resp), url

    def test_an_agent_svg_is_sandboxed_and_not_sniffed(self, discovered):
        _app, client = discovered
        resp = client.get("/panel-static/agent/logo.svg")

        assert resp.headers["content-type"].startswith("image/svg+xml")
        assert _sandboxed(resp)
        assert resp.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize(
        "url",
        [
            "/panel-static/agent/%2e%2e/agent/logo.svg",
            "/panel-static/agent/.%2flogo.svg",
            "/panel-static/agent/%5clogo.svg",
            "/panel-static/agent/sub/..%2flogo.svg",
        ],
    )
    def test_encoded_spellings_are_sandboxed_whatever_they_resolve_to(self, discovered, url):
        _app, client = discovered
        resp = client.get(url)

        assert _sandboxed(resp), url

    def test_json_is_left_alone(self, discovered):
        _app, client = discovered
        resp = client.get("/panel-static/agent/data.json")

        assert resp.status_code == 200
        assert not _sandboxed(resp)

    def test_the_reference_panel_is_served_self_contained(self, discovered):
        _app, client = discovered
        resp = client.get("/panel-static/reference/")

        assert resp.status_code == 200
        assert _sandboxed(resp)
        page = resp.text
        assert _SERVER_PATH_RE.search(page) is None, _SERVER_PATH_RE.search(page)
        # The shared theme arrived inline rather than being dropped.
        assert "--bg-primary" in page
        assert "data:text/javascript" in page
        assert "initTheme({ role: 'follower' })" in page

    def test_the_reference_panel_source_still_passes_the_panel_contract(self):
        from osprey.interfaces.design_system.panels.validator import validate_panel

        reference = Path(design_system_panels.__file__).parent / "reference"
        assert validate_panel(reference) == []

    def test_a_page_naming_a_missing_design_system_asset_is_served_as_written(
        self, discovered, tmp_path
    ):
        app, client = discovered
        bundle = tmp_path / "broken"
        bundle.mkdir()
        page = '<html><head><script src="/design-system/js/nope.js"></script></head></html>'
        (bundle / "index.html").write_text(page, encoding="utf-8")
        app.state.discovered_panel_dirs["broken"] = DiscoveredPanel(
            "broken", "Broken", "index.html", bundle
        )

        resp = client.get("/panel-static/broken/")

        assert resp.status_code == 200
        assert resp.text == page
        assert _sandboxed(resp)


# ---------------------------------------------------------------------------
# AGENT_CONTENT_ROUTES: the drift test and the probe rows
# ---------------------------------------------------------------------------

#: Response constructors whose bytes are not JSON (unless a literal JSON type
#: is named, see :func:`_builds_non_json`).
_NON_JSON_NAMES = frozenset(
    {
        "EventSourceResponse",
        "FileResponse",
        "HTMLResponse",
        "PlainTextResponse",
        "StreamingResponse",
        "RedirectResponse",
        "TemplateResponse",
        "Response",
        "attachment_response",
    }
)


def _names(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
    return found


def _builds_non_json(func: ast.AST, decorator: ast.Call) -> bool:
    """Whether a route function builds a response whose bytes are not JSON.

    A bare ``Response(...)`` naming ``media_type="application/json"`` is JSON.
    A route whose decorator names a non-JSON ``response_class`` (an event
    stream, a page) builds one whatever its body returns.
    """
    for keyword in decorator.keywords:
        if keyword.arg == "response_class" and _names(keyword.value) & _NON_JSON_NAMES:
            return True
    used = _names(func) & _NON_JSON_NAMES
    if used - {"Response"}:
        return True
    if "Response" not in used:
        return False
    for call in ast.walk(func):
        if isinstance(call, ast.Call) and _names(call.func) == {"Response"}:
            media = [k for k in call.keywords if k.arg == "media_type"]
            if not media or not (
                isinstance(media[0].value, ast.Constant)
                and media[0].value.value == "application/json"
            ):
                return True
    return False


def _is_get_route(decorator: ast.expr) -> str | None:
    """The path literal of a ``.get(...)`` / GET ``.api_route(...)`` decorator."""
    if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
        return None
    if decorator.func.attr not in ("get", "api_route") or not decorator.args:
        return None
    first = decorator.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    if decorator.func.attr == "api_route":
        methods = [k for k in decorator.keywords if k.arg == "methods"]
        if methods and "GET" not in ast.unparse(methods[0].value):
            return None
    return first.value


def _scan_non_json_get_routes() -> set[RouteRef]:
    """Every GET route in the companion servers whose endpoint builds non-JSON bytes."""
    found: set[RouteRef] = set()
    for package in SCANNED_PACKAGES:
        root = Path(importlib.import_module(package).__file__).parent
        for source in sorted(root.rglob("*.py")):
            relative = source.relative_to(root).with_suffix("")
            module = ".".join((package, *relative.parts)).removesuffix(".__init__")
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    template = _is_get_route(decorator)
                    if template is not None and _builds_non_json(node, decorator):
                        found.add(RouteRef(module, template))
    return found


def _framework_entry_routes() -> set[RouteRef]:
    """The entry pages of each companion server, as routes of its app module."""
    routes: set[RouteRef] = set()
    for definition in FRAMEWORK_WEB_SERVERS.values():
        module = definition.factory_path.split(":", 1)[0]
        for template in FRAMEWORK_ENTRY_PAGES[definition.panel_id]:
            routes.add(RouteRef(module, template))
    return routes


def _in_scanned_packages(ref: RouteRef) -> bool:
    return any(ref.module == p or ref.module.startswith(p + ".") for p in SCANNED_PACKAGES)


class TestAgentContentRoutes:
    def test_every_non_json_get_route_is_enumerated_or_an_entry_page(self):
        scanned = _scan_non_json_get_routes()
        listed = {ref for row in AGENT_CONTENT_ROUTES for ref in row.routes}
        classified = {
            ref for ref in listed if _in_scanned_packages(ref)
        } | _framework_entry_routes()

        unclassified = scanned - classified
        assert not unclassified, (
            "a companion-server GET route returns non-JSON bytes but is neither in "
            f"AGENT_CONTENT_ROUTES nor an entry page: {sorted(unclassified, key=str)}"
        )
        stale = classified - scanned
        assert not stale, f"enumerated routes no longer exist: {sorted(stale, key=str)}"

    def test_one_row_per_surface_kind(self):
        kinds = [row.kind for row in AGENT_CONTENT_ROUTES]
        assert len(kinds) == len(set(kinds))
        for row in AGENT_CONTENT_ROUTES:
            assert row.routes and row.sample_path.startswith("/"), row.kind
            assert "panel_sandbox" in row.sandboxed_by, row.kind

    def test_the_channel_finder_artifact_route_is_enumerated(self):
        refs = {ref for row in AGENT_CONTENT_ROUTES for ref in row.routes}
        assert (
            RouteRef("osprey.interfaces.channel_finder.pending_review_api", "/artifacts/{filename}")
            in refs
        )

    @pytest.mark.parametrize("row", AGENT_CONTENT_ROUTES, ids=lambda row: row.kind)
    def test_every_sample_path_lands_in_the_sandbox(self, discovered, row):
        """The table's sample of each kind, as a document, answers with the policy."""
        app, client = discovered
        _upstream(app, content_type="text/html")
        url = row.sample_path.format(
            artifact_id="a1",
            filename="report.html",
            attachment_id="1",
            draft_id="d1",
            panel_id="agent" if row.kind == "discovered-panel" else "agent-page",
            path="report.html",
        )

        resp = client.get(url)

        assert _sandboxed(resp), (row.kind, url, resp.status_code)
