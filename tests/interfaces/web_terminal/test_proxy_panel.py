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
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app
from osprey.interfaces.web_terminal.routes.proxy import _PANEL_STATE_MAP


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
