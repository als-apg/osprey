"""Tests for the panel reverse-proxy X-Forwarded-Prefix header."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import websockets
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from osprey.dispatch import DISPATCHER_MCP_PATH
from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey.interfaces.web_auth import (
    PANEL_TIER_ROUTES,
    WebCredentials,
    reset_web_credentials,
)
from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app
from osprey.interfaces.web_terminal.routes import proxy
from osprey.interfaces.web_terminal.routes.proxy import _EVENTS_PANEL_ID, _PANEL_STATE_MAP
from osprey.utils.identity import TERMINAL_USER_ENV
from osprey.utils.owner_header import OWNER_HEADER
from tests.conftest import dispatcher_route_registry


def _make_client(workspace_dir, custom_panels):
    """Create a TestClient with custom panels configured."""
    enabled = set(UNIVERSAL_PANELS)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled, custom_panels, None),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield app, c


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def app_and_client(workspace_dir):
    """App + client with a custom panel (my-dash → http://localhost:9000)."""
    custom = [
        {"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"},
    ]
    yield from _make_client(workspace_dir, custom)


class TestProxyForwardedPrefix:
    def test_x_forwarded_prefix_set(self, app_and_client):
        """Proxy sets X-Forwarded-Prefix header when forwarding to a panel."""
        app, client = app_and_client

        captured_headers = {}

        # Mock the proxy_client's .request() method (used for non-SSE requests).
        async def fake_request(*, method, url, headers, content):
            captured_headers.update(headers)
            return httpx.Response(
                status_code=200,
                json={"ok": True},
                headers={"content-type": "application/json"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/api/status")
        assert resp.status_code == 200
        assert captured_headers.get("x-forwarded-prefix") == "/panel/my-dash"

    def test_nonexistent_panel_returns_404(self, app_and_client):
        """Request to an unknown panel ID returns 404."""
        _app, client = app_and_client
        resp = client.get("/panel/nonexistent/anything")
        assert resp.status_code == 404

    def test_vendor_js_skips_rewriting(self, app_and_client):
        """Vendor JS files are passed through without path rewriting."""
        app, client = app_and_client

        js_body = 'var x = "/static/js/foo.js";'

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text=js_body,
                headers={"content-type": "application/javascript"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/static/js/vendor/plotly-3.3.1.min.js")
        assert resp.status_code == 200
        # Vendor path — body must NOT be rewritten
        assert resp.text == js_body

    def test_non_vendor_js_is_rewritten(self, app_and_client):
        """Non-vendor JS files still get path rewriting."""
        app, client = app_and_client

        js_body = 'var x = "/static/js/foo.js";'

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text=js_body,
                headers={"content-type": "application/javascript"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/static/js/gallery.js")
        assert resp.status_code == 200
        # Non-vendor path — body MUST be rewritten
        assert "/panel/my-dash/static/js/foo.js" in resp.text

    def test_dashboard_prefix_is_rewritten(self, app_and_client):
        """Root-absolute /dashboard/* paths get prefixed so iframe-embedded
        dashboards (e.g. the event dispatcher) reach their own origin through
        the panel proxy rather than escaping to the web-terminal root."""
        app, client = app_and_client

        html_body = (
            "<script>"
            'fetch("/dashboard/triggers");'
            "fetch('/dashboard/runs');"
            'new EventSource("/dashboard/stream/abc");'
            "</script>"
        )

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text=html_body,
                headers={"content-type": "text/html"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/dashboard")
        assert resp.status_code == 200
        assert '"/panel/my-dash/dashboard/triggers"' in resp.text
        assert "'/panel/my-dash/dashboard/runs'" in resp.text
        assert '"/panel/my-dash/dashboard/stream/abc"' in resp.text


class TestEventsPanelTokenInjection:
    """The EVENTS panel proxy injects the dispatcher bearer token server-side."""

    @pytest.fixture
    def app_and_client_events(self, workspace_dir):
        # The legit EVENTS panel is config-defined; the loader stamps configDefined.
        custom = [
            {
                "id": "events",
                "label": "EVENTS",
                "url": "http://localhost:8020",
                "configDefined": True,
            },
            {"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"},
        ]
        yield from _make_client(workspace_dir, custom)

    @pytest.fixture
    def app_and_client_squatted_events(self, workspace_dir):
        # A runtime-registered "events" entry carries no configDefined marker —
        # this is what an id-squat via POST /api/panels/register looks like.
        custom = [
            {"id": "events", "label": "EVENTS", "url": "http://attacker.lan:3000"},
        ]
        yield from _make_client(workspace_dir, custom)

    def test_events_panel_injects_bearer(self, app_and_client_events, monkeypatch):
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "sekret")

        captured = {}

        async def fake_request(*, method, url, headers, content):
            captured.update(headers)
            return httpx.Response(
                200, json={"ok": True}, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/events/dashboard/state")
        assert resp.status_code == 200
        assert captured.get("authorization") == "Bearer sekret"

    def test_non_events_panel_not_injected(self, app_and_client_events, monkeypatch):
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "sekret")

        captured = {}

        async def fake_request(*, method, url, headers, content):
            captured.update(headers)
            return httpx.Response(
                200, json={"ok": True}, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/api/status")
        assert resp.status_code == 200
        assert "authorization" not in {k.lower() for k in captured}

    def test_events_panel_no_token_no_header(self, app_and_client_events, monkeypatch):
        app, client = app_and_client_events
        monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)

        captured = {}

        async def fake_request(*, method, url, headers, content):
            captured.update(headers)
            return httpx.Response(
                200, json={"ok": True}, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/events/dashboard/state")
        assert resp.status_code == 200
        assert "authorization" not in {k.lower() for k in captured}

    def test_squatted_events_panel_not_injected(self, app_and_client_squatted_events, monkeypatch):
        """An 'events' entry lacking the configDefined marker gets no token.

        Defense-in-depth for the id-squat leak: even if a non-config-defined
        entry reaches the proxy under the id 'events', the dispatcher token must
        not follow it to the (attacker-controlled) origin.
        """
        app, client = app_and_client_squatted_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "sekret")

        captured = {}

        async def fake_request(*, method, url, headers, content):
            captured.update(headers)
            return httpx.Response(
                200, json={"ok": True}, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/events/dashboard/state")
        assert resp.status_code == 200
        assert "authorization" not in {k.lower() for k in captured}


class TestProxyCacheControlDefault:
    """The proxy-wide caching default (_DEFAULT_NO_CACHE): a proxied response
    whose upstream set no Cache-Control gets no-cache stamped (unversioned
    panel assets must never survive a redeploy in a browser cache), while an
    upstream's own explicit caching decision passes through untouched."""

    def test_headerless_upstream_gets_no_cache_default(self, app_and_client):
        app, client = app_and_client

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text="body { color: red; }",
                headers={"content-type": "text/css"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/panel.css")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "no-cache, no-store, must-revalidate"

    def test_explicit_upstream_cache_header_is_preserved(self, app_and_client):
        app, client = app_and_client
        immutable = "public, max-age=31536000, immutable"

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text="var x = 1;",
                headers={"content-type": "application/javascript", "cache-control": immutable},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/static/js/vendor/plotly-3.3.1.min.js")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == immutable


#: One framework panel, read from the registry-derived map rather than named, so
#: a panel that is renamed or added does not quietly stop being covered here.
LAUNCHED_PANEL_ID, LAUNCHED_STATE_ATTR = sorted(_PANEL_STATE_MAP.items())[0]

#: Where that panel's backend listens, and the header its launcher published.
LAUNCHED_BACKEND_URL = "http://127.0.0.1:9500"
LAUNCH_TOKEN = "panel-launch-token"
LAUNCH_HEADERS = {"Authorization": f"Bearer {LAUNCH_TOKEN}"}


def _lower(headers):
    return {k.lower(): v for k, v in headers.items()}


class _FakeUpstreamSocket:
    """A websocket upstream that stays open until the relay task is cancelled."""

    def __init__(self):
        self.sent: list[object] = []

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()  # pragma: no cover - cancelled at teardown
        raise AssertionError("unreachable")


class _FakeConnect:
    """Stands in for ``websockets.connect``, recording the handshake arguments."""

    def __init__(self):
        self.target = None
        self.kwargs = None

    def __call__(self, target, **kwargs):
        self.target = target
        self.kwargs = kwargs
        return self

    async def __aenter__(self):
        return _FakeUpstreamSocket()

    async def __aexit__(self, *exc_info):
        return False


class TestPanelLaunchCredentialInjection:
    """A launched panel's own credential rides both legs, and only where earned.

    The launcher publishes the credential on ``app.state.panel_auth_headers``
    and the proxy injects it, so the browser never holds it. The gate is the one
    the operator secret already uses — declared by OSPREY *and* addressed at
    loopback — so a runtime registration squatting the id, or a declared panel
    pointing off-box, is handed nothing.
    """

    @pytest.fixture
    def app_and_client_launched(self, workspace_dir):
        custom = [
            # Declared by config but off-box: the credential must not leave the machine.
            {
                "id": "offbox",
                "label": "OFFBOX",
                "url": "http://panel.facility.lan:9500",
                "configDefined": True,
            },
            # Loopback but registered at runtime — the shape an agent can create.
            {"id": "registered", "label": "REGISTERED", "url": "http://127.0.0.1:9501"},
        ]
        for app, client in _make_client(workspace_dir, custom):
            # A framework panel's URL is written to app.state by its launcher.
            setattr(app.state, LAUNCHED_STATE_ATTR, LAUNCHED_BACKEND_URL)
            # Published under all three ids on purpose: the gate, not the map, is
            # what must keep the credential away from the two unearned panels.
            app.state.panel_auth_headers = {
                LAUNCHED_PANEL_ID: dict(LAUNCH_HEADERS),
                "offbox": dict(LAUNCH_HEADERS),
                "registered": dict(LAUNCH_HEADERS),
            }
            yield app, client

    @staticmethod
    def _capture_request(app):
        captured: dict[str, str] = {}

        async def fake_request(*, method, url, headers, content, follow_redirects=True):
            captured.update(headers)
            return httpx.Response(
                200, json={"ok": True}, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
        return captured

    @staticmethod
    def _connect(client, path):
        fake = _FakeConnect()
        with patch("websockets.connect", fake):
            with client.websocket_connect(path):
                pass
        return fake

    def test_http_leg_injects_for_a_declared_loopback_panel(self, app_and_client_launched):
        app, client = app_and_client_launched
        captured = self._capture_request(app)

        resp = client.get(f"/panel/{LAUNCHED_PANEL_ID}/api/status")

        assert resp.status_code == 200
        assert _lower(captured)["authorization"] == f"Bearer {LAUNCH_TOKEN}"

    def test_ws_leg_injects_for_a_declared_loopback_panel(self, app_and_client_launched):
        _app, client = app_and_client_launched

        fake = self._connect(client, f"/panel/{LAUNCHED_PANEL_ID}/api/kernels/k1/channels")

        assert fake.kwargs["additional_headers"]["Authorization"] == f"Bearer {LAUNCH_TOKEN}"

    @pytest.mark.parametrize("panel_id", ["offbox", "registered"])
    def test_http_leg_withholds_from_an_unearned_panel(self, app_and_client_launched, panel_id):
        app, client = app_and_client_launched
        captured = self._capture_request(app)

        resp = client.get(f"/panel/{panel_id}/api/status")

        assert resp.status_code == 200
        assert "authorization" not in _lower(captured)
        assert LAUNCH_TOKEN not in " ".join(captured.values())

    @pytest.mark.parametrize("panel_id", ["offbox", "registered"])
    def test_ws_leg_withholds_from_an_unearned_panel(self, app_and_client_launched, panel_id):
        _app, client = app_and_client_launched

        fake = self._connect(client, f"/panel/{panel_id}/ws/stream")

        sent = fake.kwargs["additional_headers"] or {}
        assert LAUNCH_TOKEN not in " ".join(sent.values())

    def test_ws_upstream_carries_the_browser_query(self, app_and_client_launched):
        """A backend that keys a socket off a query parameter gets to see it."""
        _app, client = app_and_client_launched

        fake = self._connect(
            client, f"/panel/{LAUNCHED_PANEL_ID}/api/kernels/k1/channels?session_id=s1"
        )

        assert fake.target == "ws://127.0.0.1:9500/api/kernels/k1/channels?session_id=s1"

    def test_ws_upstream_without_a_query_is_unchanged(self, app_and_client_launched):
        _app, client = app_and_client_launched

        fake = self._connect(client, f"/panel/{LAUNCHED_PANEL_ID}/api/kernels/k1/channels")

        assert fake.target == "ws://127.0.0.1:9500/api/kernels/k1/channels"


@contextlib.contextmanager
def _echo_upstream(subprotocols: list[str] | None = None) -> Iterator[int]:
    """A real websocket echo server on loopback; yields its port.

    It runs on its own loop in a thread so the proxy under test reaches it over
    a real socket, and the subprotocol negotiation is the library's, not a
    stub's. With *subprotocols* the server picks the first it offers that the
    client also offered; without, it negotiates nothing.
    """
    ready = threading.Event()
    state: dict[str, object] = {}

    async def _serve() -> None:
        stop = asyncio.Event()
        state["loop"] = asyncio.get_running_loop()
        state["stop"] = stop

        async def handler(connection):
            async for frame in connection:
                await connection.send(frame)

        async with websockets.serve(handler, "127.0.0.1", 0, subprotocols=subprotocols) as server:
            state["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await stop.wait()

    thread = threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True)
    thread.start()
    assert ready.wait(10), "the echo upstream did not start"
    try:
        yield int(state["port"])  # type: ignore[call-overload]
    finally:
        loop = state["loop"]
        stop = state["stop"]
        loop.call_soon_threadsafe(stop.set)  # type: ignore[attr-defined]
        thread.join(10)


class TestPanelSubprotocolNegotiation:
    """The browser's subprotocol offer reaches the upstream, and its pick comes back.

    A browser that offers a subprotocol and is accepted without one treats the
    handshake as failed. So the proxy relays the offer, lets the upstream choose,
    and accepts the browser with that choice — which means the upstream connects
    first. What that ordering must not change: a failed upstream handshake still
    ends in an accepted-then-closed browser socket, as before.
    """

    PANEL = "echo"
    OFFER = "v1.example.protocol"

    @staticmethod
    def _client(workspace_dir, port):
        custom = [{"id": "echo", "label": "ECHO", "url": f"http://127.0.0.1:{port}"}]
        return _make_client(workspace_dir, custom)

    def test_accepts_the_subprotocol_the_upstream_selects(self, workspace_dir):
        with _echo_upstream(subprotocols=[self.OFFER]) as port:
            for _app, client in self._client(workspace_dir, port):
                with client.websocket_connect(
                    f"/panel/{self.PANEL}/ws", subprotocols=[self.OFFER]
                ) as session:
                    session.send_text("ping")

                    assert session.accepted_subprotocol == self.OFFER
                    assert session.receive_text() == "ping"

    def test_offering_none_is_accepted_with_none(self, workspace_dir):
        with _echo_upstream() as port:
            for _app, client in self._client(workspace_dir, port):
                with client.websocket_connect(f"/panel/{self.PANEL}/ws") as session:
                    session.send_text("ping")

                    assert session.accepted_subprotocol is None
                    assert session.receive_text() == "ping"

    def test_upstream_selecting_none_is_accepted_with_none(self, workspace_dir):
        """An offer the upstream does not take up is answered the way it answered."""
        with _echo_upstream() as port:
            for _app, client in self._client(workspace_dir, port):
                with client.websocket_connect(
                    f"/panel/{self.PANEL}/ws", subprotocols=[self.OFFER]
                ) as session:
                    session.send_text("ping")

                    assert session.accepted_subprotocol is None
                    assert session.receive_text() == "ping"

    def test_binary_frames_relay_unchanged_both_ways(self, workspace_dir):
        """A negotiated binary protocol rides bytes frames; the echo proves both legs."""
        payload = bytes(range(256))
        with _echo_upstream(subprotocols=[self.OFFER]) as port:
            for _app, client in self._client(workspace_dir, port):
                with client.websocket_connect(
                    f"/panel/{self.PANEL}/ws", subprotocols=[self.OFFER]
                ) as session:
                    session.send_bytes(payload)

                    assert session.receive_bytes() == payload

    def test_failed_upstream_handshake_still_closes_the_accepted_socket(self, workspace_dir):
        """The upstream requires a subprotocol the browser did not offer and
        refuses the handshake; the browser is accepted and closed normally."""
        with _echo_upstream(subprotocols=[self.OFFER]) as port:
            for _app, client in self._client(workspace_dir, port):
                with client.websocket_connect(f"/panel/{self.PANEL}/ws") as session:
                    with pytest.raises(WebSocketDisconnect) as closed:
                        session.receive_text()

                assert closed.value.code == 1000


# ---------------------------------------------------------------------------
# The dispatcher MCP hop
# ---------------------------------------------------------------------------

#: The dispatcher's base URL as the EVENTS panel declares it. Loopback, so the
#: hop earns the bearer and the owner mint.
DISPATCHER_BACKEND_URL = "http://localhost:8020"

#: The bearer the container holds and the browser never does.
DISPATCHER_TOKEN = "dispatcher-bearer-value"

#: What every MCP client advertises. The substring puts this hop on the
#: proxy's streaming branch whatever the answer turns out to be.
MCP_ACCEPT = "application/json, text/event-stream"

#: One JSON-RPC call and one answer, as bytes. The answer carries a
#: root-absolute path inside quotes -- the exact shape ``_rewrite_content``
#: rewrites -- so asserting byte identity is an assertion about the rewrite and
#: not only about the transport.
MCP_CALL_BODY = b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
MCP_ANSWER_BODY = (
    b'{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"manual_fire",'
    b'"inputSchema":{"$ref":"/api/schema"}}]}}'
)

#: The transport's session handle, which both ends must keep seeing.
MCP_SESSION_ID = "1f5a9c7e-0b44-4a1d-9cc2-0e5b6d2f7a31"

#: The panel token an agent's own child process holds, and the operator secret
#: it does not.
HOP_PANEL_TOKEN = "panel-token-value"
HOP_OPERATOR_SECRET = "operator-secret-value"


class _FakeStreamResponse:
    """A streamed upstream response, as ``client.send(stream=True)`` returns one."""

    def __init__(self, *, status_code=200, headers=None, chunks=(b"data: hello\n\n",)):
        self.status_code = status_code
        default = {"content-type": "text/event-stream"}
        self.headers = httpx.Headers(default if headers is None else headers)
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def _capture_stream(app, upstream):
    """Stub the streaming branch's ``build_request``/``send``; return its kwargs."""
    captured: dict[str, object] = {}
    real_build = app.state.proxy_client.build_request

    def spy_build(**kwargs):
        captured.update(kwargs)
        return real_build(**kwargs)

    app.state.proxy_client.build_request = spy_build
    app.state.proxy_client.send = AsyncMock(return_value=upstream)
    return captured


def _refuse_rewrite(*_args, **_kwargs):
    raise AssertionError("the dispatcher MCP hop must not rewrite the body")


class TestDispatcherMcpHop:
    """``/panel/events/mcp`` carries a JSON-RPC conversation, not a page.

    The proxy is the one process in the container holding the dispatcher
    bearer, so this hop is the only door an agent in a session has to the MCP
    transport. What travels over it is a protocol: the bytes are the message,
    the upstream's own content-type says whether the answer is a single JSON
    reply or a stream, and the session handle is how the two ends stay in one
    conversation. Every row here pins one of those against the handling the
    proxy applies to a web page.
    """

    @pytest.fixture
    def app_and_client_events(self, workspace_dir):
        custom = [
            {
                "id": "events",
                "label": "EVENTS",
                "url": DISPATCHER_BACKEND_URL,
                "configDefined": True,
                # The one panel-config knob that would otherwise reach a JSON
                # answer on this path. Declared here so every byte-identity row
                # below asserts a refusal rather than an empty default.
                "rewriteJsonPaths": ["mcp"],
            },
        ]
        yield from _make_client(workspace_dir, custom)

    def _post(self, client, headers=None):
        sent = {"accept": MCP_ACCEPT, "content-type": "application/json"}
        sent.update(headers or {})
        return client.post("/panel/events/mcp", headers=sent, content=MCP_CALL_BODY)

    def test_the_hop_reaches_the_transport_path(self, app_and_client_events, monkeypatch):
        """The dispatcher's MCP endpoint, spelled from the constant both ends read."""
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
        captured = _capture_stream(
            app,
            _FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(MCP_ANSWER_BODY,)
            ),
        )

        resp = self._post(client)

        assert resp.status_code == 200
        assert captured["url"] == f"{DISPATCHER_BACKEND_URL}{DISPATCHER_MCP_PATH}"
        assert captured["method"] == "POST"
        assert captured["content"] == MCP_CALL_BODY

    @pytest.mark.parametrize("upstream_type", ["application/json", "text/event-stream"])
    def test_the_answer_is_relayed_unchanged(
        self, app_and_client_events, monkeypatch, upstream_type
    ):
        """A transport answers one call as JSON and the next as a stream.

        Which one it chose is the client's to read, so the content-type comes
        back as the upstream wrote it -- and the body byte for byte, because a
        JSON-RPC envelope is parsed, not rendered.
        """
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
        _capture_stream(
            app,
            _FakeStreamResponse(headers={"content-type": upstream_type}, chunks=(MCP_ANSWER_BODY,)),
        )

        resp = self._post(client)

        assert resp.status_code == 200
        assert resp.headers["content-type"] == upstream_type
        assert resp.content == MCP_ANSWER_BODY

    def test_the_session_id_travels_in_both_directions(self, app_and_client_events, monkeypatch):
        """The handle the transport hands out is the one the next call presents."""
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
        captured = _capture_stream(
            app,
            _FakeStreamResponse(
                headers={"content-type": "application/json", "mcp-session-id": MCP_SESSION_ID},
                chunks=(MCP_ANSWER_BODY,),
            ),
        )

        resp = self._post(client, headers={"mcp-session-id": MCP_SESSION_ID})

        assert resp.status_code == 200
        assert _lower(captured["headers"])["mcp-session-id"] == MCP_SESSION_ID
        assert resp.headers["mcp-session-id"] == MCP_SESSION_ID

    def test_a_notification_is_relayed_as_202_with_no_body(
        self, app_and_client_events, monkeypatch
    ):
        """A JSON-RPC notification is answered with a status and nothing else."""
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
        _capture_stream(
            app, _FakeStreamResponse(status_code=202, headers={"content-length": "0"}, chunks=())
        )

        resp = self._post(client)

        assert resp.status_code == 202
        assert resp.content == b""

    def test_the_streaming_branch_never_rewrites(self, app_and_client_events, monkeypatch):
        """A rewritten envelope is a corrupted one, so the pass is never made."""
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
        monkeypatch.setattr(proxy, "_rewrite_content", _refuse_rewrite)
        _capture_stream(
            app,
            _FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(MCP_ANSWER_BODY,)
            ),
        )

        resp = self._post(client)

        assert resp.status_code == 200
        assert resp.content == MCP_ANSWER_BODY

    def test_the_standard_branch_never_rewrites(self, app_and_client_events, monkeypatch):
        """Nor on the branch a client that advertised no stream would take.

        The panel declares ``rewriteJsonPaths`` covering this very path, so the
        refusal here is the hop's and not the configuration's.
        """
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
        monkeypatch.setattr(proxy, "_rewrite_content", _refuse_rewrite)

        async def fake_request(*, method, url, headers, content, **_kwargs):
            return httpx.Response(
                200, content=MCP_ANSWER_BODY, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.post(
            "/panel/events/mcp",
            headers={"accept": "application/json", "content-type": "application/json"},
            content=MCP_CALL_BODY,
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        assert resp.content == MCP_ANSWER_BODY

    def test_both_ends_of_the_hop_follow_the_constant(self, app_and_client_events, monkeypatch):
        """Move the transport's path and the whole hop moves with it.

        The trailing slash is the live version of this: a transport mounted at
        ``/mcp/`` is reached at ``/mcp/``, and the test that recognises the hop
        has to agree with the URL it forwards to, or the byte relay is granted
        to one request while another gets it. Driven through the branch a
        client advertising no stream takes, where the panel's own
        ``rewriteJsonPaths`` is waiting for anything this hop fails to claim.
        """
        app, client = app_and_client_events
        monkeypatch.setattr(proxy, "DISPATCHER_MCP_PATH", "/mcp/")
        monkeypatch.setattr(proxy, "_rewrite_content", _refuse_rewrite)
        captured: dict[str, object] = {}

        async def fake_request(*, method, url, headers, content, **_kwargs):
            captured["url"] = url
            return httpx.Response(
                200, content=MCP_ANSWER_BODY, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.post(
            "/panel/events/mcp/",
            headers={"accept": "application/json", "content-type": "application/json"},
            content=MCP_CALL_BODY,
        )

        assert resp.status_code == 200
        assert captured["url"] == f"{DISPATCHER_BACKEND_URL}/mcp/"
        assert resp.content == MCP_ANSWER_BODY

    def test_the_hop_carries_the_bearer_and_the_minted_owner(
        self, app_and_client_events, monkeypatch
    ):
        """What the browser sent stops here; what this process vouches for goes on.

        The bearer is the container's, and the owner is the account this
        container runs as -- the name whose chip gates whatever the fired job
        writes. Neither is anything the caller supplied.
        """
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        captured = _capture_stream(
            app,
            _FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(MCP_ANSWER_BODY,)
            ),
        )

        resp = self._post(
            client,
            headers={
                "authorization": "Bearer forged-by-the-caller",
                OWNER_HEADER: "bob",
                "cookie": "osprey_session=live-session-id",
            },
        )

        assert resp.status_code == 200
        forwarded = _lower(captured["headers"])
        owner_names = [name for name in captured["headers"] if name.lower() == OWNER_HEADER.lower()]
        assert forwarded["authorization"] == f"Bearer {DISPATCHER_TOKEN}"
        assert owner_names == [OWNER_HEADER]
        assert forwarded[OWNER_HEADER.lower()] == "alice"
        assert "cookie" not in forwarded

    def test_no_bearer_when_the_container_holds_none(self, app_and_client_events, monkeypatch):
        """A deployment without the token sends no credential rather than a blank one."""
        app, client = app_and_client_events
        monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
        captured = _capture_stream(
            app,
            _FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(MCP_ANSWER_BODY,)
            ),
        )

        resp = self._post(client)

        assert resp.status_code == 200
        assert "authorization" not in _lower(captured["headers"])

    @pytest.fixture
    def untrusted_events_client(self, workspace_dir, request):
        """An EVENTS panel in a shape that earns no credential.

        The parameter is the backend URL and whether the config loader declared
        the panel; each row below drops one half of what the gate requires.
        """
        url, config_defined = request.param
        panel = {"id": "events", "label": "EVENTS", "url": url, "rewriteJsonPaths": ["mcp"]}
        if config_defined:
            panel["configDefined"] = True
        yield from _make_client(workspace_dir, [panel])

    @pytest.mark.parametrize(
        "untrusted_events_client",
        [
            pytest.param(("http://events.example.com:8020", True), id="declared-but-off-box"),
            pytest.param((DISPATCHER_BACKEND_URL, False), id="loopback-but-registered"),
        ],
        indirect=True,
    )
    def test_an_untrusted_events_backend_is_told_nothing(
        self, untrusted_events_client, monkeypatch
    ):
        """Neither the bearer nor the minted owner follows the panel id anywhere.

        The bearer is the deployment's own shared secret and the owner names
        the account whose chip gates the fired job, so a backend earns both
        only by being declared in config *and* addressed on loopback. A
        declared panel pointing off-box would carry the secret off the machine;
        a loopback panel registered at runtime is a listener the agent's own
        sandbox can stand up. Each shape here drops one half, and is handed
        neither -- the hop still forwards, with nothing of this container's on
        it.
        """
        app, client = untrusted_events_client
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        captured = _capture_stream(
            app,
            _FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(MCP_ANSWER_BODY,)
            ),
        )

        resp = self._post(client)

        assert resp.status_code == 200
        forwarded = _lower(captured["headers"])
        assert "authorization" not in forwarded
        assert OWNER_HEADER.lower() not in forwarded


class TestDispatcherMcpHopTier:
    """The credential that opens the hop, behind the terminal's own gate.

    The agent's child process holds the panel token and nothing stronger, which
    is what makes this route panel-tier; a caller holding nothing at all is
    refused before the proxy runs.
    """

    @pytest.fixture
    def gated_client(self):
        """The proxy router behind the real gate, with a known panel token.

        Every row carries ``no_auth_seam``: the suite-wide seam stamps the
        operator secret onto any client aimed at a gated app, which would admit
        these requests on a credential the agent does not have.
        """
        reset_web_credentials()
        app = FastAPI()
        app.include_router(proxy.router)
        app.state.custom_panels = [
            {
                "id": "events",
                "label": "EVENTS",
                "url": DISPATCHER_BACKEND_URL,
                "configDefined": True,
            },
        ]
        app.state.proxy_client = httpx.AsyncClient()
        app.state.web_credentials = WebCredentials(
            operator_secret=HOP_OPERATOR_SECRET, panel_token=HOP_PANEL_TOKEN
        )
        app.add_middleware(WebAuthMiddleware, cookie_name="osprey_terminal_session_8080")
        with TestClient(app, client=("127.0.0.1", 54321)) as client:
            yield app, client
        asyncio.run(app.state.proxy_client.aclose())
        reset_web_credentials()

    @pytest.mark.no_auth_seam
    def test_the_panel_token_alone_reaches_the_hop(self, gated_client):
        """The credential an agent's own child process holds opens this door."""
        app, client = gated_client
        _capture_stream(
            app,
            _FakeStreamResponse(
                headers={"content-type": "application/json"}, chunks=(MCP_ANSWER_BODY,)
            ),
        )

        resp = client.post(
            "/panel/events/mcp",
            headers={
                "accept": MCP_ACCEPT,
                "content-type": "application/json",
                "authorization": f"Bearer {HOP_PANEL_TOKEN}",
            },
            content=MCP_CALL_BODY,
        )

        assert resp.status_code == 200
        assert resp.content == MCP_ANSWER_BODY

    @pytest.mark.no_auth_seam
    def test_without_a_credential_the_hop_is_refused(self, gated_client):
        """The route is panel-tier, not open: the proxy is never reached."""
        app, client = gated_client
        _capture_stream(app, _FakeStreamResponse())

        resp = client.post(
            "/panel/events/mcp",
            headers={"accept": MCP_ACCEPT, "content-type": "application/json"},
            content=MCP_CALL_BODY,
        )

        assert resp.status_code == 401
        app.state.proxy_client.send.assert_not_awaited()


class TestTheHopPathIsSpelledOnce:
    """One path, in every place the wire spells it.

    Two of the spellings read ``DISPATCHER_MCP_PATH`` directly: this proxy's
    forward target, and the dispatcher's own compose environment. Two are
    literals -- the panel-tier route the terminal's gate admits, and the entry
    URL an agent's MCP client dials -- because importing the constant where
    they live would pull a heavy module into a light one. These rows are what
    holds the literals to the constant: a spelling that drifted from it leaves
    the hop refused at the gate or dialed at an address nothing serves, while
    every transport row above stays green.
    """

    def test_the_gate_admits_the_route_the_hop_serves(self):
        """The panel-tier table names this hop, at the path and on the methods it takes."""
        route = f"/panel/{_EVENTS_PANEL_ID}{DISPATCHER_MCP_PATH}"
        hop_rows = {
            (method, path)
            for method, path in PANEL_TIER_ROUTES
            if path.startswith(f"/panel/{_EVENTS_PANEL_ID}")
        }

        assert hop_rows == {("GET", route), ("POST", route), ("DELETE", route)}

    def test_the_agents_entry_url_ends_at_the_hop(self):
        """The URL an agent's client is handed addresses this hop and not a neighbour."""
        from osprey.registry.mcp import EVENT_DISPATCHER_PROXY_URL

        assert EVENT_DISPATCHER_PROXY_URL.endswith(
            f"/panel/{_EVENTS_PANEL_ID}{DISPATCHER_MCP_PATH}"
        )


@pytest.mark.xdist_group("dispatch_mcp_transport_auth")
async def test_the_transport_answers_the_hop_path_without_a_redirect(monkeypatch):
    """The dispatcher serves ``POST /mcp`` itself, rather than pointing at ``/mcp/``.

    A redirect here would be relayed to the caller as ``/panel/events/mcp/``,
    one character off the path the terminal's gate grants an agent's panel
    token -- so the follow-up would be refused and the conversation would end on
    its first call. The transport's mount path and the gate's route table are
    two spellings of one wire, and this is where they are held together.
    """
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", DISPATCHER_TOKEN)
    monkeypatch.setenv("TRIGGERS_YML", "/nonexistent/test-triggers.yml")
    monkeypatch.setenv("DISPATCH_TARGET", "http://worker.invalid:9000")

    with dispatcher_route_registry() as build:
        app = build().http_app(path=DISPATCHER_MCP_PATH, stateless_http=True, json_response=True)

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://dispatcher.invalid"
            ) as client:
                resp = await client.post(
                    DISPATCHER_MCP_PATH,
                    headers={
                        "accept": MCP_ACCEPT,
                        "content-type": "application/json",
                        "authorization": f"Bearer {DISPATCHER_TOKEN}",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "panel-hop-test", "version": "1.0"},
                        },
                    },
                    follow_redirects=False,
                )

    assert resp.status_code == 200
