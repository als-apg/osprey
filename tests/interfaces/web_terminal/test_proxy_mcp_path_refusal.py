"""A sidecar's MCP transport is not reachable through the panel proxy.

A sidecar may serve an MCP transport beside its pages and declare where
(``SidecarPanel.mcp_path``). The panel proxy carries pages, so for such a
sidecar it refuses that path for every caller — and refuses every spelling
that this hop and the upstream could read differently, so no request passes
the check here and lands on the transport there:

* 404 for a ``.`` or ``..`` segment below ``/panel/<id>/``, and for a percent
  escape of ``.``, ``/`` or ``\\`` that survives decoding or arrives in the raw
  path;
* 403 when the upstream path, with ``;params`` dropped and repeated slashes
  collapsed, is the declared path or lies below it.

Both are decided before any upstream call, on the HTTP leg and the WebSocket
leg, and neither applies to a panel that declares no transport.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces.web_terminal.routes import proxy
from osprey.profiles import web_panels
from osprey.profiles.web_panels import SidecarPanel
from osprey.registry.web import panel_url_state_attr

from ._proxy_fakes import _FakeConnect, _patch_connect, panel_app

BACKEND_URL = "http://127.0.0.1:9500"

#: Declares a transport at ``/mcp``.
HOP_PANEL = "with-mcp"
#: A sidecar that declares none; the refusals must leave it alone.
PLAIN_PANEL = "without-mcp"

#: Path below ``/panel/<id>/`` as the client sends it → the status it is refused with.
REFUSED = [
    ("mcp", 403),
    ("mcp/", 403),
    ("/mcp", 403),
    ("mcp;x", 403),
    ("mcp/x", 403),
    ("%6dcp", 403),
    ("%2e%2e/mcp", 404),
    ("%252e%252e/mcp", 404),
    ("m%2fcp", 404),
]
REFUSED_IDS = [
    "/mcp",
    "/mcp/",
    "//mcp",
    "/mcp;x",
    "/mcp/x",
    "%6dcp",
    "%2e%2e/mcp",
    "%252e%252e/mcp",
    "m%2fcp",
]


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    """The whole app with two running sidecars, one declaring an MCP transport."""
    for panel_id, mcp_path in ((HOP_PANEL, "/mcp"), (PLAIN_PANEL, None)):
        monkeypatch.setitem(
            web_panels.SIDECAR_PANELS,
            panel_id,
            SidecarPanel("a test sidecar", "module:Class", mcp_path=mcp_path),
        )
        monkeypatch.setitem(proxy._PANEL_STATE_MAP, panel_id, panel_url_state_attr(panel_id))
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    for app, client in panel_app(workspace, []):
        for panel_id in (HOP_PANEL, PLAIN_PANEL):
            setattr(app.state, panel_url_state_attr(panel_id), BACKEND_URL)
        yield app, client


def _watch_upstream(app) -> tuple[AsyncMock, AsyncMock]:
    """Replace both ways the HTTP leg reaches an upstream; return the two mocks."""

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    request = AsyncMock(side_effect=fake_request)
    send = AsyncMock(side_effect=AssertionError("no streamed upstream request expected"))
    app.state.proxy_client.request = request
    app.state.proxy_client.send = send
    return request, send


class TestDeclaration:
    def test_a_sidecar_declares_no_transport_by_default(self):
        assert SidecarPanel("n", "m:C").mcp_path is None


class _Scope:
    """What the refusal reads of a request: its raw path."""

    def __init__(self, raw_path: bytes = b""):
        self.scope = {"raw_path": raw_path}


class TestTheRule:
    """The rule itself, for the spellings an HTTP client resolves before sending."""

    @pytest.fixture(autouse=True)
    def _sidecars(self, monkeypatch):
        monkeypatch.setitem(
            web_panels.SIDECAR_PANELS, HOP_PANEL, SidecarPanel("n", "m:C", mcp_path="/mcp")
        )

    @staticmethod
    def _refusal(path: str, *, raw_path: bytes = b"", backend: str = BACKEND_URL):
        return proxy._mcp_path_refusal(_Scope(raw_path), HOP_PANEL, path, f"{backend}/{path}")

    @pytest.mark.parametrize(
        "path",
        [
            # a dot segment
            ".",
            "..",
            "./page",
            "a/./b",
            "a/../mcp",
            "a/..",
            "../mcp",
            # an escape left in the decoded path
            "a%2eb",
            "a%2Fb",
            "a%5Cb",
        ],
    )
    def test_a_path_that_could_resolve_elsewhere_is_not_found(self, path):
        assert self._refusal(path) == 404

    def test_an_escaped_separator_in_the_raw_path_is_not_found(self):
        assert self._refusal("m/cp", raw_path=b"/panel/with-mcp/m%2Fcp") == 404

    @pytest.mark.parametrize("path", ["mcp", "mcp/", "/mcp", "///mcp//x", "mcp;a=b", ";x/mcp;y/z"])
    def test_the_transport_and_everything_below_it_is_forbidden(self, path):
        assert self._refusal(path) == 403

    @pytest.mark.parametrize("path", ["", "mcpx", "api/mcp", "a.b/c..d", "page;mcp", "50%25"])
    def test_any_other_path_passes(self, path):
        assert self._refusal(path) is None

    def test_a_request_with_no_raw_path_is_judged_on_the_decoded_one(self):
        scope = _Scope()
        del scope.scope["raw_path"]

        assert proxy._mcp_path_refusal(scope, HOP_PANEL, "mcp", f"{BACKEND_URL}/mcp") == 403
        assert proxy._mcp_path_refusal(scope, HOP_PANEL, "page", f"{BACKEND_URL}/page") is None

    def test_a_panel_outside_the_sidecar_registry_is_never_refused(self):
        assert proxy._mcp_path_refusal(_Scope(), "my-dash", "mcp", f"{BACKEND_URL}/mcp") is None


class TestHttpLeg:
    @pytest.mark.parametrize(("path", "status"), REFUSED, ids=REFUSED_IDS)
    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_every_spelling_of_the_transport_is_refused_with_no_upstream_request(
        self, app_and_client, path, status, method
    ):
        app, client = app_and_client
        request, send = _watch_upstream(app)

        resp = client.request(method, f"/panel/{HOP_PANEL}/{path}")

        assert resp.status_code == status
        assert request.await_count == 0
        assert send.await_count == 0

    @pytest.mark.parametrize("path", ["page%5c..%5cmcp", "a%2Fb", "a%252fb", "a%255cb", "a%252Eb"])
    def test_an_escaped_dot_or_separator_is_not_found(self, app_and_client, path):
        app, client = app_and_client
        request, send = _watch_upstream(app)

        resp = client.get(f"/panel/{HOP_PANEL}/{path}")

        assert resp.status_code == 404
        assert request.await_count == 0
        assert send.await_count == 0

    def test_the_query_does_not_hide_the_transport(self, app_and_client):
        app, client = app_and_client
        request, _send = _watch_upstream(app)

        assert client.get(f"/panel/{HOP_PANEL}/mcp?x=/other").status_code == 403
        assert request.await_count == 0

    @pytest.mark.parametrize("path", ["", "index.html", "api/mcp", "mcpx", "static/mcp.js"])
    def test_every_other_path_is_forwarded(self, app_and_client, path):
        app, client = app_and_client
        request, _send = _watch_upstream(app)

        assert client.get(f"/panel/{HOP_PANEL}/{path}").status_code == 200
        assert request.await_args.kwargs["url"] == f"{BACKEND_URL}/{path}"

    @pytest.mark.parametrize("path", ["mcp", "mcp/x", "m%2fcp", "%252e%252e/mcp"])
    def test_a_sidecar_that_declares_no_transport_is_left_alone(self, app_and_client, path):
        app, client = app_and_client
        request, _send = _watch_upstream(app)

        assert client.get(f"/panel/{PLAIN_PANEL}/{path}").status_code == 200
        assert request.await_count == 1


class TestWebSocketLeg:
    @pytest.mark.parametrize(("path", "status"), REFUSED, ids=REFUSED_IDS)
    def test_every_spelling_of_the_transport_is_refused_with_no_upstream_connect(
        self, app_and_client, path, status
    ):
        _app, client = app_and_client
        fake = _FakeConnect()

        with _patch_connect(fake):
            with pytest.raises(WebSocketDisconnect) as closed:
                with client.websocket_connect(f"/panel/{HOP_PANEL}/{path}"):
                    pass  # pragma: no cover - the handshake is refused

        assert closed.value.code == 4000 + status
        assert fake.target is None

    def test_another_path_is_relayed(self, app_and_client):
        _app, client = app_and_client
        fake = _FakeConnect()

        with _patch_connect(fake):
            with client.websocket_connect(f"/panel/{HOP_PANEL}/ws/stream"):
                pass

        assert fake.target == "ws://127.0.0.1:9500/ws/stream"

    def test_a_sidecar_that_declares_no_transport_is_left_alone(self, app_and_client):
        _app, client = app_and_client
        fake = _FakeConnect()

        with _patch_connect(fake):
            with client.websocket_connect(f"/panel/{PLAIN_PANEL}/mcp"):
                pass

        assert fake.target == "ws://127.0.0.1:9500/mcp"
