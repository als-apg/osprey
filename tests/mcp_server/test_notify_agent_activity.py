"""Tests for :func:`osprey.mcp_server.http.notify_agent_activity`.

The helper is a fire-and-forget notification: it must never raise, must
honor its (short) timeout, and must produce a body matching the
``/api/agent-activity`` route contract:
``{"tool": ..., "target": {"kind": ..., "panel"?: ..., "detail"?: ...}}``.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from osprey.mcp_server.http import notify_agent_activity

_MODULE = "osprey.mcp_server.http"


def _free_port() -> int:
    """Reserve a localhost port and release it (nothing will be listening)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestStoppedServer:
    def test_no_exception_when_server_down(self):
        port = _free_port()
        with patch(f"{_MODULE}.web_terminal_url", return_value=f"http://127.0.0.1:{port}"):
            notify_agent_activity("channel_read", "channel", detail="SR:BPM1:X")

    def test_no_exception_when_url_resolution_fails(self):
        with patch(f"{_MODULE}.web_terminal_url", side_effect=RuntimeError("config broken")):
            notify_agent_activity("channel_read", "channel")


class _CaptureHandler(BaseHTTPRequestHandler):
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


class TestPayloadShape:
    def test_full_payload(self, capture_server):
        port = capture_server.server_address[1]
        with patch(f"{_MODULE}.web_terminal_url", return_value=f"http://127.0.0.1:{port}"):
            notify_agent_activity("channel_write", "channel", panel="controls", detail="SR:HC1:SP")

        assert len(_CaptureHandler.captured) == 1
        path, body = _CaptureHandler.captured[0]
        assert path == "/api/agent-activity"
        assert body == {
            "tool": "channel_write",
            "target": {"kind": "channel", "panel": "controls", "detail": "SR:HC1:SP"},
        }

    def test_overlong_detail_truncated_to_route_bound(self, capture_server):
        # A bulk channel write can join arbitrarily many channel names into
        # detail; the helper must clamp to the route's 1024-char bound so the
        # emit is never silently rejected with a 422.
        long_detail = ", ".join(f"SR{i:02d}:HCM{i}:SP" for i in range(200))
        assert len(long_detail) > 1024
        port = capture_server.server_address[1]
        with patch(f"{_MODULE}.web_terminal_url", return_value=f"http://127.0.0.1:{port}"):
            notify_agent_activity("channel_write", "channel", detail=long_detail)

        assert len(_CaptureHandler.captured) == 1
        _, body = _CaptureHandler.captured[0]
        sent = body["target"]["detail"]
        assert len(sent) == 1024
        assert sent.endswith("…")
        assert sent[:1023] == long_detail[:1023]

    def test_none_fields_omitted(self, capture_server):
        port = capture_server.server_address[1]
        with patch(f"{_MODULE}.web_terminal_url", return_value=f"http://127.0.0.1:{port}"):
            notify_agent_activity("archiver_read", "data")

        assert len(_CaptureHandler.captured) == 1
        _, body = _CaptureHandler.captured[0]
        assert body == {"tool": "archiver_read", "target": {"kind": "data"}}
        assert "panel" not in body["target"]
        assert "detail" not in body["target"]


class TestTimeout:
    def test_hanging_socket_returns_quickly_without_raising(self):
        # Socket that accepts connections but never responds.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            with patch(f"{_MODULE}.web_terminal_url", return_value=f"http://127.0.0.1:{port}"):
                start = time.monotonic()
                notify_agent_activity("channel_read", "channel", detail="SR:BPM1:X")
                elapsed = time.monotonic() - start
            # Helper timeout is 1s; well under 3s proves it is honored.
            assert elapsed < 2.5
        finally:
            listener.close()
