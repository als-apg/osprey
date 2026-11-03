"""Each protection of an agent-written page held to the code by its own check.

The route table (``agent_content_routes.py``) names the surfaces; each
protection below maps to one class, so a protection cannot be claimed without a
check, and a check that goes red means the code no longer does what the table
says:

* ``artifact_sandbox`` -- the artifact app, built whole, answers every
  enumerated content route with the sandbox policy, and exempts only its shell;
* ``panel_sandbox`` -- the companion servers' non-JSON GET routes are exactly
  the enumeration (plus entry pages), and the terminal app, built whole, lands
  every enumerated surface kind in its own sandbox;
* ``navigation_guard`` -- the terminal app answers a hostile panel URL 422 on
  both routes that take one, the agent's ``open`` op and runtime registration,
  and a person's focus report takes no URL at all (server half); the vitest
  suite pins ``buildEmbedSrc`` refusing it (client half);
* ``markdown_sanitiser`` -- every ``marked.parse`` / ``renderMathInMarkdown``
  call in the panels' JavaScript and the artifact app's page templates sits in
  a helper whose output only reaches the DOM through DOMPurify (static scan),
  and the vitest suites drive each helper with hostile markdown.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import osprey.interfaces
from osprey.interfaces.artifacts.app import create_app as create_artifact_app
from osprey.interfaces.artifacts.content_sandbox import (
    ContentSandboxMiddleware,
    is_not_content,
)
from osprey.interfaces.opaque_origin import OPAQUE_CSP
from osprey.interfaces.panel_urls import navigation_url_refusal, normalise_navigation_url
from osprey.interfaces.web_terminal.agent_content_routes import AGENT_CONTENT_ROUTES
from osprey.interfaces.web_terminal.panel_discovery import DiscoveredPanel
from osprey.interfaces.web_terminal.routes.proxy import _PANEL_STATE_MAP
from tests.interfaces.web_terminal._proxy_fakes import panel_app
from tests.interfaces.web_terminal.test_panel_sandbox import (
    _framework_entry_routes,
    _in_scanned_packages,
    _scan_non_json_get_routes,
)

REPO = Path(__file__).resolve().parents[2]
INTERFACES = Path(osprey.interfaces.__file__).parent
ARTIFACTS_MODULE = "osprey.interfaces.artifacts.app"
UPSTREAM = "http://127.0.0.1:9999"

#: Each protection and the class below that holds it to the code.
PROTECTION_CHECKS = {
    "artifact_sandbox": "TestArtifactSandbox",
    "panel_sandbox": "TestPanelSandbox",
    "navigation_guard": "TestNavigationGuard",
    "markdown_sanitiser": "TestMarkdownSanitiser",
}


def _csp(response) -> list[str]:
    return response.headers.get_list("content-security-policy")


class TestTheSetIsHeldToTheCode:
    def test_every_sandbox_the_route_table_names_has_a_check(self):
        named = {by for row in AGENT_CONTENT_ROUTES for by in row.sandboxed_by}
        assert {"artifact_sandbox", "panel_sandbox"} <= named
        assert {by for by in named if by.endswith("_sandbox")} - {
            "route_sandbox",
            "upstream_sandbox",
        } <= set(PROTECTION_CHECKS)

    def test_every_check_exists_and_tests_something(self):
        for protection, name in PROTECTION_CHECKS.items():
            check = globals().get(name)
            assert isinstance(check, type), (protection, name)
            assert any(attr.startswith("test_") for attr in vars(check)), (protection, name)


# ---------------------------------------------------------------------------
# artifact_sandbox
# ---------------------------------------------------------------------------


@pytest.fixture
def artifact_client(tmp_path) -> TestClient:
    return TestClient(create_artifact_app(workspace_root=tmp_path / "ws"))


def _artifact_refs() -> list[str]:
    """The artifact app's enumerated routes behind the artifact sandbox."""
    return sorted(
        {
            ref.template
            for row in AGENT_CONTENT_ROUTES
            if "artifact_sandbox" in row.sandboxed_by
            for ref in row.routes
            if ref.module == ARTIFACTS_MODULE
        }
    )


def _get_routes(app) -> dict[str, APIRoute]:
    return {
        route.path: route
        for route in app.routes
        if isinstance(route, APIRoute) and "GET" in route.methods
    }


class TestArtifactSandbox:
    def test_the_middleware_is_installed(self, artifact_client):
        assert any(m.cls is ContentSandboxMiddleware for m in artifact_client.app.user_middleware)

    def test_only_the_shell_opts_out(self, artifact_client):
        exempt = {
            path
            for path, route in _get_routes(artifact_client.app).items()
            if is_not_content(route.endpoint)
        }
        assert exempt == {"/"}

    @pytest.mark.parametrize("template", _artifact_refs())
    def test_every_enumerated_route_exists_and_is_not_exempt(self, artifact_client, template):
        routes = _get_routes(artifact_client.app)

        assert template in routes, f"enumerated artifact route {template} no longer exists"
        assert not is_not_content(routes[template].endpoint)

    @pytest.mark.parametrize(
        "template", [t for t in _artifact_refs() if t != "/api/events"]
    )  # an event stream never ends; the two tests above cover it
    def test_every_enumerated_route_answers_with_the_sandbox(self, artifact_client, template):
        store = artifact_client.app.state.artifact_store
        page = store.save_file(
            file_content=b"<html><body><script>fetch('/api/artifacts')</script></body></html>",
            filename="report.html",
            artifact_type="html",
            title="report",
            description="",
            mime_type="text/html",
            tool_source="test",
        )
        markdown = store.save_file(
            file_content=b"# Title\n\n<img src=x onerror=alert(1)>",
            filename="notes.md",
            artifact_type="markdown",
            title="notes",
            description="",
            mime_type="text/markdown",
            tool_source="test",
        )
        artifact_id = markdown.id if "markdown" in template else page.id
        url = template.format(artifact_id=artifact_id, filename=page.filename)

        resp = artifact_client.get(url)

        assert OPAQUE_CSP in _csp(resp), (url, resp.status_code)

    def test_the_policy_is_the_contract(self):
        assert OPAQUE_CSP == "sandbox allow-scripts allow-modals allow-popups allow-downloads"


# ---------------------------------------------------------------------------
# panel_sandbox (and the terminal app shared with the navigation guard)
# ---------------------------------------------------------------------------


@pytest.fixture
def terminal(tmp_path):
    """The whole terminal app: a runtime-registered panel and a discovered bundle."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    bundle = tmp_path / "agent-bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text("<html><body>agent</body></html>", encoding="utf-8")
    custom = [{"id": "agent-page", "label": "AGENT", "url": UPSTREAM}]
    for app, client in panel_app(workspace, custom):
        for attr in _PANEL_STATE_MAP.values():
            setattr(app.state, attr, UPSTREAM)
        app.state.discovered_panel_dirs = {
            "agent": DiscoveredPanel("agent", "Agent", "index.html", bundle),
        }

        async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
            return httpx.Response(
                status_code=200,
                content=b"<html><body>agent</body></html>",
                headers={"content-type": "text/html"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
        yield app, client


class TestPanelSandbox:
    def test_the_enumeration_matches_the_companion_servers(self):
        scanned = _scan_non_json_get_routes()
        listed = {ref for row in AGENT_CONTENT_ROUTES for ref in row.routes}
        classified = {
            ref for ref in listed if _in_scanned_packages(ref)
        } | _framework_entry_routes()

        assert scanned - classified == set(), "unenumerated non-JSON GET routes"
        assert classified - scanned == set(), "enumerated routes that no longer exist"

    @pytest.mark.parametrize("row", AGENT_CONTENT_ROUTES, ids=lambda row: row.kind)
    def test_every_surface_kind_lands_in_the_terminals_sandbox(self, terminal, row):
        assert "panel_sandbox" in row.sandboxed_by
        _app, client = terminal
        url = row.sample_path.format(
            artifact_id="a1",
            filename="report.html",
            attachment_id="1",
            draft_id="d1",
            panel_id="agent" if row.kind == "discovered-panel" else "agent-page",
            path="report.html",
        )

        resp = client.get(url)

        assert OPAQUE_CSP in _csp(resp), (row.kind, url, resp.status_code)


# ---------------------------------------------------------------------------
# navigation_guard
# ---------------------------------------------------------------------------

_HOSTILE = [
    "javascript:alert(1)",
    " \tJaVa\nScRiPt:alert(1)",
    "data:text/html,x",
    "//evil.example/x",
]
NAV_GUARD_VITEST = REPO / "tests/interfaces/web_terminal/static/panel-nav-guard.test.mjs"
IFRAME_SYNC = INTERFACES / "web_terminal/static/js/panel-iframe-sync.js"


def _refusal(url: str) -> str:
    """The guard's own refusal text, so a 422 from anything else does not pass."""
    refusal = navigation_url_refusal(normalise_navigation_url(url))
    assert refusal is not None, url
    return refusal


class TestNavigationGuard:
    @pytest.mark.parametrize("url", _HOSTILE)
    def test_panel_open_refuses_a_hostile_url(self, terminal, url):
        app, client = terminal

        resp = client.post(
            "/api/panel-op/open", json={"request_id": "g1", "member": "artifacts", "url": url}
        )

        assert resp.status_code == 422, resp.text
        assert resp.json()["reason"] == "invalid_url"
        assert resp.json()["detail"] == _refusal(url)
        assert getattr(app.state, "active_panel", None) != "artifacts"

    @pytest.mark.parametrize("url", _HOSTILE)
    def test_runtime_registration_refuses_a_hostile_url(self, terminal, url):
        app, client = terminal
        # Registration open, so the refusal is the guard's and not the switch's.
        app.state.allow_runtime_panels = True
        app.state.runtime_panel_allowlist = None

        resp = client.post("/api/panels/register", json={"id": "evil", "label": "E", "url": url})

        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == _refusal(url)
        assert all(panel.get("id") != "evil" for panel in app.state.custom_panels)

    @pytest.mark.parametrize(
        "url",
        ["/api/panels", "/panel/ariel/", "/panel/artifacts/../ariel/", "http://evil.example/x"],
    )
    def test_panel_open_refuses_a_url_that_leaves_the_panel(self, terminal, url):
        """The guard's second step: a url stays under the member's own mount
        (or on its own server), so another path on this origin, another
        panel's mount, an encoded way out of the mount and a foreign host are
        refused with the mount named."""
        app, client = terminal

        resp = client.post(
            "/api/panel-op/open", json={"request_id": "g3", "member": "artifacts", "url": url}
        )

        assert resp.status_code == 422, resp.text
        assert resp.json()["reason"] == "invalid_url"
        assert "/panel/artifacts" in resp.json()["detail"]
        assert getattr(app.state, "active_panel", None) != "artifacts"

    def test_a_path_passes_the_guard(self, terminal):
        """A path is admitted: the open gets past the guard to the dock relay.

        This terminal has no dock client, so what refuses the op is the relay
        (``no_dock_client``, 409), never the guard's 422.
        """
        _, client = terminal

        resp = client.post(
            "/api/panel-op/open",
            json={"request_id": "g2", "member": "artifacts", "url": "/panel/artifacts/"},
        )

        assert resp.status_code == 409, resp.text
        assert resp.json()["reason"] == "no_dock_client"

    def test_the_client_half_is_pinned_by_vitest(self):
        source = IFRAME_SYNC.read_text(encoding="utf-8")
        assert re.search(r"export function buildEmbedSrc\s*\(", source)
        assert "REFUSED_EMBED_SRC = 'about:blank'" in source

        suite = NAV_GUARD_VITEST.read_text(encoding="utf-8")
        assert "panel-iframe-sync.js" in suite
        assert "buildEmbedSrc" in suite and "REFUSED_EMBED_SRC" in suite
        assert "javascript:" in suite
        assert re.search(r"export function isUrlWithinMount\s*\(", source)
        assert "isUrlWithinMount" in suite


# ---------------------------------------------------------------------------
# markdown_sanitiser
# ---------------------------------------------------------------------------

#: What turns markdown into HTML: the two spellings the contract names, plus a
#: read of the global ``marked`` under another name (``globalThis.marked``).
_SINK_RE = re.compile(
    r"\bmarked\s*\.\s*parse\s*\(|\brenderMathInMarkdown\s*\(|\b(?:globalThis|window)\s*\)?\s*\.\s*marked\b"
)
_DEFINITION_RE = re.compile(r"\bfunction\s+renderMathInMarkdown\s*\($")
_TOP_FUNCTION_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE)

#: ``(file below interfaces/, top-level function)`` pairs allowed to hold a
#: sink, each because its output only reaches the DOM through DOMPurify.
SANITISING_HELPERS = {
    # Holds marked.parse; its one caller is renderMathInMarkdown, which
    # sanitises the whole string after the KaTeX swap.
    ("artifacts/static/js/md-render.js", "renderUnsanitized"),
    ("artifacts/static/js/md-render.js", "renderMarkdownInto"),
    # Builds the chat parser; its one caller is renderMarkdownInto.
    ("web_terminal/static/js/chat-markdown.js", "chatMarkedParse"),
    ("okf_panel/static/js/app.js", "renderMarkdownInto"),
}

#: The one sanitised assignment each sink's output flows into.
SANITISED_SINKS = {
    ("artifacts/static/js/md-render.js", "renderMarkdownInto"): (
        "DOMPurify.sanitize(renderMathInMarkdown(text))"
    ),
    # sanitizeReply is DOMPurify.sanitize with the reply's forbidden tags,
    # attributes and foreign-media hook.
    ("web_terminal/static/js/chat-markdown.js", "renderMarkdownInto"): (
        "sanitizeReply(DOMPurify, parse(withMath ? extractMath(source, found) : source))"
    ),
    ("okf_panel/static/js/app.js", "renderMarkdownInto"): (
        "purify.sanitize(String(parser.parse(text)))"
    ),
}

#: ``(helper file, helper, the only function allowed to call it)``: a helper
#: whose own return value is unsanitised HTML.
UNSANITISED_HELPERS = [
    ("artifacts/static/js/md-render.js", "renderUnsanitized", "renderMathInMarkdown"),
    ("web_terminal/static/js/chat-markdown.js", "chatMarkedParse", "renderMarkdownInto"),
]

VITEST_SUITES = {
    "artifacts/static/js/md-render.js": "tests/interfaces/artifacts/content-sandbox.test.mjs",
    "web_terminal/static/js/chat-markdown.js": "tests/interfaces/web_terminal/chat-render.test.mjs",
    "okf_panel/static/js/app.js": "tests/interfaces/okf_panel/markdown-sanitize.test.mjs",
}


def _strip_js_comments(source: str) -> str:
    """Blank block and line comments, keeping offsets (and so line numbers)."""

    def blank(text: str) -> str:
        return re.sub(r"[^\n]", " ", text)

    source = re.sub(r"/\*.*?\*/", lambda m: blank(m.group(0)), source, flags=re.DOTALL)
    # A `//` after `:` or a quote is a URL or string text, not a comment.
    return re.sub(r"(?m)(^|[^:\\'\"])(//[^\n]*)", lambda m: m.group(1) + blank(m.group(2)), source)


def _enclosing_function(source: str, offset: int) -> str | None:
    """The top-level (column 0) function whose body holds *offset*, if any."""
    for match in _TOP_FUNCTION_RE.finditer(source):
        if match.start() > offset:
            break
        end = source.find("\n}", match.end())
        if end == -1 or offset < end + 2:
            return match.group(1)
    return None


def _function_body(source: str, name: str) -> str:
    """Text of top-level function *name*, header to its column-0 closing brace."""
    match = re.search(
        rf"^(?:export\s+)?(?:async\s+)?function\s+{re.escape(name)}\s*\(", source, re.MULTILINE
    )
    assert match, name
    end = source.find("\n}", match.end())
    return source[match.start() : end + 2]


def _sinks(source: str) -> list[tuple[int, str | None, str]]:
    """``(line, enclosing function, matched text)`` for each markdown sink."""
    code = _strip_js_comments(source)
    found = []
    for match in _SINK_RE.finditer(code):
        if _DEFINITION_RE.search(code[: match.end()]):
            continue  # ``function renderMathInMarkdown(`` declares, not calls
        line = code.count("\n", 0, match.start()) + 1
        found.append((line, _enclosing_function(code, match.start()), match.group(0)))
    return found


def _panel_scripts() -> list[Path]:
    """Every first-party script under ``interfaces/**/static``, vendor excluded."""
    return sorted(
        path
        for path in INTERFACES.rglob("*")
        if path.suffix in (".js", ".mjs")
        and "static" in path.relative_to(INTERFACES).parts
        and "vendor" not in path.relative_to(INTERFACES).parts
        and "node_modules" not in path.parts
    )


def _page_templates() -> dict[str, str]:
    """The artifact app's module-level page-template strings, by name."""
    source = (INTERFACES / "artifacts" / "app.py").read_text(encoding="utf-8")
    templates = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and "<" in node.value.value:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        templates[target.id] = node.value.value
    return templates


class TestMarkdownSanitiser:
    def test_the_scan_sees_the_panels(self):
        relative = {str(p.relative_to(INTERFACES)) for p in _panel_scripts()}
        assert set(VITEST_SUITES) <= relative
        assert not any("vendor" in r for r in relative)

    def test_every_sink_sits_in_a_sanitising_helper(self):
        stray = []
        for path in _panel_scripts():
            rel = str(path.relative_to(INTERFACES))
            for line, function, text in _sinks(path.read_text(encoding="utf-8")):
                if (rel, function) not in SANITISING_HELPERS:
                    stray.append(f"{rel}:{line} {text!r} in {function or '<module>'}")
        assert not stray, "markdown rendered outside a sanitising helper:\n" + "\n".join(stray)

    def test_every_helper_still_holds_a_sink(self):
        """A stale allowlist entry would let a new sink hide behind its name."""
        seen = set()
        for path in _panel_scripts():
            rel = str(path.relative_to(INTERFACES))
            seen |= {(rel, fn) for _line, fn, _text in _sinks(path.read_text(encoding="utf-8"))}
        assert SANITISING_HELPERS <= seen, SANITISING_HELPERS - seen

    @pytest.mark.parametrize(("rel", "function"), sorted(SANITISED_SINKS), ids=lambda v: str(v))
    def test_each_sink_assigns_only_sanitised_html(self, rel, function):
        body = _function_body((INTERFACES / rel).read_text(encoding="utf-8"), function)
        expected = SANITISED_SINKS[(rel, function)]

        assert expected in body, (rel, function)
        for assignment in re.findall(r"\.innerHTML\s*=\s*([^;]+);", body):
            assert assignment.strip() == "clean" or "sanitize(" in assignment, assignment

    @pytest.mark.parametrize(("rel", "helper", "caller"), UNSANITISED_HELPERS, ids=lambda v: v)
    def test_unsanitised_helpers_have_one_sanitising_caller(self, rel, helper, caller):
        callers = set()
        call = re.compile(rf"\b{re.escape(helper)}\s*\(")
        for path in _panel_scripts():
            code = _strip_js_comments(path.read_text(encoding="utf-8"))
            for match in call.finditer(code):
                if re.search(r"\bfunction\s+$", code[: match.start()]):
                    continue
                callers.add(
                    (str(path.relative_to(INTERFACES)), _enclosing_function(code, match.start()))
                )
        # md-render's own helper calls itself from nowhere else; renderUnsanitized
        # sinks are already confined by the scan above, chatMarkedParse is not a sink.
        assert callers == {(rel, caller)}, callers

    def test_the_artifact_page_templates_render_through_the_helper(self):
        templates = _page_templates()
        assert "_MARKDOWN_PAGE_TEMPLATE" in templates
        for name, text in templates.items():
            assert not _sinks(text), f"{name} renders markdown without the sanitising helper"
        page = templates["_MARKDOWN_PAGE_TEMPLATE"]
        assert "renderMarkdownInto" in page and "md-render.js" in page

    def test_the_scan_catches_a_raw_sink(self):
        raw = (
            "export function show(el, text) {\n"
            "  el.innerHTML = marked.parse(text); // hostile\n"
            "}\n"
            "function also(t) {\n"
            "  return renderMathInMarkdown(t);\n"
            "}\n"
            "const m = globalThis.marked;\n"
            "/* marked.parse(text) in a comment is not a call */\n"
        )
        assert [(line, fn) for line, fn, _ in _sinks(raw)] == [(2, "show"), (5, "also"), (7, None)]

    @pytest.mark.parametrize(("helper_file", "suite"), sorted(VITEST_SUITES.items()))
    def test_each_helper_is_driven_by_vitest_with_hostile_markdown(self, helper_file, suite):
        text = (REPO / suite).read_text(encoding="utf-8")

        assert helper_file.rsplit("/", 1)[1] in text, (suite, helper_file)
        assert "renderMarkdownInto" in text
        assert "onerror" in text and "<script>" in text
