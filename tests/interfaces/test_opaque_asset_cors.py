"""C-ASSETS: first-party assets an opaque document imports are CORS-readable.

An agent-authored document runs at an opaque origin (CSP ``sandbox`` without
``allow-same-origin``), so its module, stylesheet and font requests arrive with
``Origin: null`` and without the session cookie. Every C-ASSETS path therefore
answers with ``Access-Control-Allow-Origin: *`` (never credentials) and is
exempt from the auth gate — and no agent-writable path gets either.

The "no cookie is needed" tests carry ``no_auth_seam``: the suite-wide seam
would otherwise authenticate every TestClient request, and a pass would prove
nothing about the exemption. A control test in each section shows the gate is
live on the same app.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Route
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.common_middleware import (
    ASSET_CORS_HEADERS,
    FIRST_PARTY_PANEL_STATIC,
    PANEL_ASSET_SUBPATHS,
    is_exempt_path,
    is_panel_asset_path,
)
from osprey.interfaces.web_terminal.routes import proxy

from .web_terminal._proxy_fakes import panel_app

ACAO = "access-control-allow-origin"
ACAC = "access-control-allow-credentials"
NULL_ORIGIN = {"Origin": "null"}


def _assert_asset_cors(resp: httpx.Response) -> None:
    assert resp.status_code == 200, resp.text[:200]
    assert resp.headers.get(ACAO) == ASSET_CORS_HEADERS["Access-Control-Allow-Origin"] == "*"
    assert ACAC not in resp.headers
    assert "set-cookie" not in resp.headers


def _assert_no_cors(resp: httpx.Response) -> None:
    assert ACAO not in resp.headers
    assert ACAC not in resp.headers


# ---------------------------------------------------------------------------
# The exemption predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/panel/my-dash/design-system/js/theme-manager.js",
        "/panel/artifacts/design-system/css/tokens.css",
        "/panel/jupyter/static/fonts/fonts.css",
        "/panel/artifacts/static/js/md-render.js",
        "/panel/artifacts/static/js/vendor/plotly-3.3.1.min.js",
        "/panel/artifacts/static/css/vendor/katex.min.css",
        "/panel/artifacts/static/css/vendor/fonts/KaTeX_Main-Regular.woff2",
    ],
)
def test_panel_asset_paths_are_recognised(path):
    """The hub-local asset routes under ``/panel/<id>/`` are C-ASSETS."""
    assert is_panel_asset_path(path)


@pytest.mark.parametrize(
    "path",
    [
        # Agent-writable bytes, whoever serves them.
        "/panel-static/agent-panel/index.html",
        "/panel-static/agent-panel/static/js/app.js",
        "/panel/jupyter/files/notebook-output.html",
        "/panel/artifacts/files/abc123/report.html",
        "/panel/artifacts/api/markdown/abc/rendered",
        # A runtime or discovered panel's own static tree is its own bytes.
        "/panel/my-dash/static/js/app.js",
        "/panel/my-dash/static/css/app.css",
        "/panel/channel-finder/static/js/app.js",
        # The bare sub-path is relayed by the catch-all, never answered locally.
        "/panel/my-dash/design-system",
        "/panel/artifacts/static/js",
        # Climbing out of an asset path.
        "/panel/my-dash/design-system/../api/config",
        "/panel/../design-system/x.js",
        # An id outside the mount-segment class.
        "/panel/a'b/design-system/js/theme-manager.js",
        "/panel/.hidden/design-system/js/theme-manager.js",
        "/panel//design-system/js/theme-manager.js",
        "/u/alice/panel/my-dash/design-system/x.js",
    ],
)
def test_other_paths_are_not_panel_assets(path):
    """Nothing agent-writable or catch-all-relayed is a panel asset path."""
    assert not is_panel_asset_path(path)


@pytest.mark.parametrize(
    "path", ["/panel-static/p/index.html", "/files/a/b.html", "/panel/jupyter/files/x"]
)
def test_agent_writable_paths_stay_gated(path):
    """The general exemption does not reach agent-writable paths either."""
    assert not is_exempt_path(path)


# ---------------------------------------------------------------------------
# Every interface app's static mounts
# ---------------------------------------------------------------------------


@pytest.fixture
def interface_client(tmp_path):
    """A bare interface app with an /api route and a module in its /static tree."""
    static = tmp_path / "static"
    (static / "js").mkdir(parents=True)
    (static / "js" / "module.js").write_text("export const x = 1;\n")
    app = FastAPI()

    @app.get("/api/secret")
    async def secret():
        return {"ok": True}

    configure_interface_app(app, static_dir=static)
    with TestClient(app) as client:
        yield client


@pytest.mark.no_auth_seam
@pytest.mark.parametrize(
    "path",
    [
        "/design-system/js/theme-manager.js",
        "/design-system/css/tokens.css",
        "/static/fonts/fonts.css",
        "/static/js/module.js",
    ],
)
def test_a_module_request_from_an_opaque_origin_needs_no_cookie(interface_client, path):
    """``Origin: null``, no cookie: the mount answers with ``ACAO: *`` only."""
    resp = interface_client.get(path, headers=NULL_ORIGIN)
    _assert_asset_cors(resp)


@pytest.mark.no_auth_seam
def test_the_interface_gate_is_live_without_the_seam(interface_client):
    """Control: the same app refuses an unauthenticated non-asset request."""
    resp = interface_client.get("/api/secret", headers=NULL_ORIGIN)
    assert resp.status_code == 401
    _assert_no_cors(resp)


def test_an_api_route_gets_no_cors_header(interface_client):
    """Only the mounts are cross-origin readable; an authenticated /api is not."""
    resp = interface_client.get("/api/secret")
    assert resp.status_code == 200
    _assert_no_cors(resp)


# ---------------------------------------------------------------------------
# The hub: panel-scoped asset routes
# ---------------------------------------------------------------------------


@pytest.fixture
def hub(tmp_path):
    """The web terminal with a URL-backed custom panel and a discovered bundle."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    custom = [{"id": "my-dash", "label": "DASH", "url": "http://localhost:9"}]
    for app, client in panel_app(workspace, custom):
        bundle = tmp_path / "bundle"
        (bundle / "static" / "js").mkdir(parents=True)
        (bundle / "index.html").write_text("<script type=module src=static/js/app.js></script>")
        (bundle / "static" / "js" / "app.js").write_text("export {};\n")
        app.state.discovered_panel_dirs = {
            "agent-panel": SimpleNamespace(directory=str(bundle), entry="index.html")
        }
        setattr(app.state, proxy._PANEL_STATE_MAP["artifacts"], "http://127.0.0.1:9")
        setattr(app.state, proxy._PANEL_STATE_MAP["jupyter"], "http://127.0.0.1:9")
        yield app, client


def _backend_must_not_be_called(app):
    app.state.proxy_client.request = AsyncMock(
        side_effect=AssertionError("asset request reached the panel backend")
    )


def _backend_answers_with_cors(app):
    """A backend that tries to grant CORS on agent bytes; the proxy must drop it."""
    app.state.proxy_client.request = AsyncMock(
        return_value=httpx.Response(
            200,
            headers={
                "content-type": "text/html",
                "access-control-allow-origin": "*",
                "access-control-allow-credentials": "true",
            },
            content=b"<p>agent bytes</p>",
        )
    )


_HUB_ASSETS = [
    "/panel/my-dash/design-system/js/theme-manager.js",
    "/panel/my-dash/design-system/css/tokens.css",
    "/panel/my-dash/static/fonts/fonts.css",
    "/panel/artifacts/design-system/js/theme-manager.js",
    "/panel/artifacts/static/js/md-render.js",
    "/panel/artifacts/static/js/gallery.js",
    "/panel/artifacts/static/css/gallery.css",
    "/design-system/js/theme-manager.js",
    "/static/fonts/fonts.css",
]


@pytest.mark.no_auth_seam
@pytest.mark.parametrize("path", _HUB_ASSETS)
def test_a_hub_module_request_from_an_opaque_origin_needs_no_cookie(hub, path):
    """Each hub C-ASSETS path answers ``Origin: null`` with ``ACAO: *``, locally."""
    app, client = hub
    _backend_must_not_be_called(app)
    _assert_asset_cors(client.get(path, headers=NULL_ORIGIN))


@pytest.mark.no_auth_seam
def test_head_is_exempt_too(hub):
    app, client = hub
    _backend_must_not_be_called(app)
    resp = client.head("/panel/artifacts/static/js/md-render.js", headers=NULL_ORIGIN)
    assert resp.status_code == 200
    assert resp.headers.get(ACAO) == "*"


@pytest.mark.no_auth_seam
@pytest.mark.parametrize(
    "path",
    [
        "/panel-static/agent-panel/index.html",
        "/panel-static/agent-panel/static/js/app.js",
        "/panel/my-dash/static/js/app.js",
        "/panel/jupyter/files/out.html",
        "/panel/artifacts/files/abc/report.html",
        "/api/panels",
    ],
)
def test_agent_writable_hub_paths_need_a_credential(hub, path):
    """Control and negative: agent-writable paths are still refused without one."""
    app, client = hub
    _backend_must_not_be_called(app)
    resp = client.get(path, headers=NULL_ORIGIN)
    assert resp.status_code == 401
    _assert_no_cors(resp)


@pytest.mark.no_auth_seam
@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def test_only_get_and_head_are_exempt_on_a_panel_asset_path(hub, method):
    """Any other method falls through to the catch-all, so it stays gated."""
    app, client = hub
    _backend_must_not_be_called(app)
    resp = client.request(method, "/panel/my-dash/design-system/js/theme-manager.js")
    assert resp.status_code == 401


@pytest.mark.no_auth_seam
def test_a_websocket_on_a_panel_asset_path_stays_gated(hub):
    """The exemption is HTTP-only: a socket there would reach the panel's sidecar."""
    _app, client = hub
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/panel/my-dash/design-system/socket") as ws:
            ws.receive_text()


def test_agent_writable_hub_paths_never_carry_the_header(hub):
    """Authenticated and served, agent bytes still get no CORS header."""
    app, client = hub
    _backend_answers_with_cors(app)

    for path in (
        "/panel-static/agent-panel/index.html",
        "/panel-static/agent-panel/static/js/app.js",
        "/panel/my-dash/static/js/app.js",
        "/panel/jupyter/files/out.html",
        "/panel/artifacts/files/abc/report.html",
    ):
        resp = client.get(path, headers=NULL_ORIGIN)
        assert resp.status_code == 200, path
        _assert_no_cors(resp)


def test_a_first_party_module_is_rewritten_into_the_panel_namespace(hub):
    """Root-absolute imports resolve back to C-ASSETS routes, as the catch-all did."""
    app, client = hub
    _backend_must_not_be_called(app)
    resp = client.get("/panel/artifacts/static/js/gallery.js")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/javascript")
    assert '"/panel/artifacts/design-system/js/theme-manager.js"' in resp.text
    assert '"/design-system/js/theme-manager.js"' not in resp.text


def test_a_first_party_static_path_cannot_leave_its_tree(hub):
    app, client = hub
    _backend_must_not_be_called(app)
    resp = client.get("/panel/artifacts/static/js/%2E%2E/index.html")
    assert resp.status_code == 404
    resp = client.get("/panel/artifacts/static/js/no-such-module.js")
    assert resp.status_code == 404


def test_a_design_system_request_with_an_unsafe_panel_id_is_refused(hub):
    """The id is spliced into the rewritten body, so a quote never gets there."""
    app, client = hub
    _backend_must_not_be_called(app)
    resp = client.get("/panel/a'b/design-system/js/theme-manager.js")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Route table: every C-ASSETS route is local and declared above the catch-all
# ---------------------------------------------------------------------------


def _route_paths() -> list[str]:
    return [route.path for route in proxy.router.routes if isinstance(route, Route)]


def test_every_panel_asset_route_sits_above_the_catch_all():
    paths = _route_paths()
    catch_all = paths.index("/panel/{panel_id}/{path:path}")
    expected = [f"/panel/{{panel_id}}/{sub}/{{asset_path:path}}" for sub in PANEL_ASSET_SUBPATHS]
    expected += [
        f"/panel/{panel_id}/{sub}/{{asset_path:path}}"
        for panel_id, subs in FIRST_PARTY_PANEL_STATIC.items()
        for sub in subs
    ]
    for route_path in expected:
        assert route_path in paths, route_path
        assert paths.index(route_path) < catch_all, route_path


def test_first_party_static_trees_belong_to_framework_panels_only():
    """A runtime or discovered panel can never be named a first-party tree."""
    for panel_id in FIRST_PARTY_PANEL_STATIC:
        assert panel_id in proxy._PANEL_STATE_MAP
        assert panel_id in proxy._FIRST_PARTY_STATIC_ROOTS
        assert proxy._FIRST_PARTY_STATIC_ROOTS[panel_id].is_dir()
