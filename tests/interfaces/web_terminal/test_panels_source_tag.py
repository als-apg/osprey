"""Agent-attribution (``source: "agent"``) on panel event frames.

The panel routes accept an optional
``source: "agent"`` field and pass it through into their SSE broadcast
frames; browser-originated POSTs (which never send ``source``) broadcast
without it — the key is *omitted*, not null.  The MCP-side
``notify_panel_*`` helpers stamp ``source: "agent"`` on their POST payloads.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router
from osprey.mcp_server.http import (
    notify_panel_close,
    notify_panel_focus,
    notify_panel_register,
    notify_panel_visibility,
)

_MODULE = "osprey.mcp_server.http"

# Resolve the register-route URL validation to a routable LAN address so the
# SSRF check passes without real DNS (same pattern as test_web_custom_panels).
_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"


# ---- Route-side: source passthrough into broadcast frames ---- #


def _make_client() -> TestClient:
    """Minimal app exposing the panels router with a stub broadcaster."""
    app = FastAPI()
    app.include_router(router)
    app.state.broadcaster = MagicMock()
    app.state.enabled_panels = {"ariel"}
    app.state.custom_panels = []
    app.state.allow_runtime_panels = True
    return TestClient(app)


def _broadcast_frame(client: TestClient) -> dict:
    broadcaster = client.app.state.broadcaster
    broadcaster.broadcast.assert_called_once()
    return broadcaster.broadcast.call_args[0][0]


class TestPanelFocusSource:
    def test_agent_source_broadcast(self):
        client = _make_client()
        resp = client.post("/api/panel-focus", json={"panel": "ariel", "source": "agent"})
        assert resp.status_code == 200
        frame = _broadcast_frame(client)
        assert frame["type"] == "panel_focus"
        assert frame["source"] == "agent"

    def test_no_broadcast_when_source_omitted(self):
        """A source-less focus is a human report: mirrored, never broadcast."""
        client = _make_client()
        resp = client.post("/api/panel-focus", json={"panel": "ariel"})
        assert resp.status_code == 200
        client.app.state.broadcaster.broadcast.assert_not_called()


class TestPanelVisibilitySource:
    def test_agent_source_broadcast(self):
        client = _make_client()
        resp = client.post(
            "/api/panel-visibility",
            json={"panel": "ariel", "visible": True, "source": "agent"},
        )
        assert resp.status_code == 200
        frame = _broadcast_frame(client)
        assert frame["type"] == "panel_visibility"
        assert frame["source"] == "agent"

    def test_no_source_key_when_omitted(self):
        client = _make_client()
        resp = client.post("/api/panel-visibility", json={"panel": "ariel", "visible": False})
        assert resp.status_code == 200
        frame = _broadcast_frame(client)
        assert frame == {"type": "panel_visibility", "panel": "ariel", "visible": False}
        assert "source" not in frame


class TestPanelCloseSource:
    """``/api/panel-close`` — the on-screen half, separate from visibility."""

    def test_agent_source_broadcast(self):
        client = _make_client()
        resp = client.post("/api/panel-close", json={"panel": "ariel", "source": "agent"})
        assert resp.status_code == 200
        frame = _broadcast_frame(client)
        assert frame == {"type": "panel_close", "panel": "ariel", "source": "agent"}

    def test_human_close_broadcast_without_source(self):
        client = _make_client()
        resp = client.post("/api/panel-close", json={"panel": "ariel"})
        assert resp.status_code == 200
        frame = _broadcast_frame(client)
        assert frame == {"type": "panel_close", "panel": "ariel"}
        assert "source" not in frame

    def test_unknown_panel_is_rejected(self):
        client = _make_client()
        resp = client.post("/api/panel-close", json={"panel": "nope", "source": "agent"})
        assert resp.status_code == 422
        client.app.state.broadcaster.broadcast.assert_not_called()

    def test_closing_never_broadcasts_a_visibility_frame(self):
        """Rail membership is untouched — that is the whole point of the verb.

        A ``panel_visibility`` frame here would remove the rail entry on every
        client, which is what the old ``hide_panel`` did and what the operator
        loses the panel to.
        """
        client = _make_client()
        client.post("/api/panel-close", json={"panel": "ariel", "source": "agent"})
        frame = _broadcast_frame(client)
        assert frame["type"] != "panel_visibility"
        assert "visible" not in frame


class TestPanelRegisterSource:
    _BODY = {"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"}

    def test_agent_source_broadcast(self):
        client = _make_client()
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client.post("/api/panels/register", json={**self._BODY, "source": "agent"})
        assert resp.status_code == 200
        frame = _broadcast_frame(client)
        assert frame["type"] == "panel_register"
        assert frame["source"] == "agent"

    def test_no_source_key_when_omitted(self):
        client = _make_client()
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client.post("/api/panels/register", json=self._BODY)
        assert resp.status_code == 200
        frame = _broadcast_frame(client)
        # Full-frame pin (like the focus/visibility siblings): the URL is the
        # browser-facing rewrite, and no ``source`` key may sneak in.
        assert frame == {
            "type": "panel_register",
            "id": "grafana",
            "label": "GRAFANA",
            "url": "/panel/grafana",
            "healthEndpoint": None,
            "path": "/",
        }


# ---- MCP-side: notify_panel_* helpers stamp source: "agent" ---- #


class _CaptureHandler(BaseHTTPRequestHandler):
    """Records (path, body) for each POST (same seam as the notify tests)."""

    captured: list[tuple[str, dict]] = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        type(self).captured.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):  # silence request logging
        pass


@pytest.fixture
def capture_server():
    _CaptureHandler.captured = []
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _patched_url(capture_server: HTTPServer):
    port = capture_server.server_address[1]
    return patch(f"{_MODULE}.web_terminal_url", return_value=f"http://127.0.0.1:{port}")


class TestNotifyHelpersSendAgentSource:
    def test_notify_panel_focus(self, capture_server):
        with _patched_url(capture_server):
            notify_panel_focus("ariel")
        path, body = _CaptureHandler.captured[0]
        assert path == "/api/panel-focus"
        assert body == {"panel": "ariel", "source": "agent"}

    def test_notify_panel_focus_with_url(self, capture_server):
        with _patched_url(capture_server):
            notify_panel_focus("ariel", url="/some/path")
        _, body = _CaptureHandler.captured[0]
        assert body == {"panel": "ariel", "url": "/some/path", "source": "agent"}

    def test_notify_panel_close(self, capture_server):
        """Closing posts to its own route — never to panel-visibility."""
        with _patched_url(capture_server):
            notify_panel_close("ariel")
        path, body = _CaptureHandler.captured[0]
        assert path == "/api/panel-close"
        assert body == {"panel": "ariel", "source": "agent"}

    def test_notify_panel_visibility(self, capture_server):
        with _patched_url(capture_server):
            notify_panel_visibility("ariel", True)
        path, body = _CaptureHandler.captured[0]
        assert path == "/api/panel-visibility"
        assert body == {"panel": "ariel", "visible": True, "source": "agent"}

    def test_notify_panel_register(self, capture_server):
        with _patched_url(capture_server):
            result = notify_panel_register("grafana", "GRAFANA", "http://grafana.lan:3000")
        assert result["ok"] is True
        path, body = _CaptureHandler.captured[0]
        assert path == "/api/panels/register"
        assert body["source"] == "agent"
        # Existing fields are untouched alongside the new tag.
        assert body["id"] == "grafana"
        assert body["label"] == "GRAFANA"
        assert body["url"] == "http://grafana.lan:3000"
