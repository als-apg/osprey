"""Tests for the panels-context SessionStart hook.

The hook injects two independent context parts:

1. a web-surface line derived from ``OSPREY_WEB_UX`` (``simple`` | ``expert``),
   set per session by the web terminal's launchers — the simple line carries
   the "show_panel('artifacts') when you produce an artifact" instruction that
   backs the chat-only simple onboarding;
2. the panel inventory fetched from ``GET /api/panels``.

Either part may be present alone: the surface line must survive a down web
terminal (it is env-only), and inventory-only output is the pre-existing
behavior for sessions with no surface marker. No parts → silent exit 0.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

HOOK = "osprey_panels_context.py"

PANELS_PAYLOAD = {
    "enabled": ["artifacts", "ariel"],
    "labels": {"artifacts": "WORKSPACE", "ariel": "ARIEL"},
    "visible": ["artifacts"],
    "active": "artifacts",
    "custom": [],
}


@pytest.fixture
def panels_server():
    """Stub web terminal serving GET /api/panels on an ephemeral port."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            body = json.dumps(PANELS_PAYLOAD).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence test output
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _closed_port() -> int:
    """A port with nothing listening (bound momentarily, then released)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _context(result):
    assert result is not None, "expected additionalContext output"
    out = result["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart"
    return out["additionalContext"]


def test_simple_surface_with_inventory(hook_runner, monkeypatch, panels_server):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(panels_server))
    monkeypatch.setenv("OSPREY_WEB_UX", "simple")

    context = _context(hook_runner(HOOK, "", {}))
    # Surface line first, with the artifact-reveal instruction …
    assert "SIMPLE surface" in context
    assert 'show_panel("artifacts")' in context
    # … then the ordinary inventory.
    assert "WORKSPACE (id=artifacts, shown)" in context
    assert "ARIEL (id=ariel, hidden)" in context


def test_expert_surface_with_inventory(hook_runner, monkeypatch, panels_server):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(panels_server))
    monkeypatch.setenv("OSPREY_WEB_UX", "expert")

    context = _context(hook_runner(HOOK, "", {}))
    assert "EXPERT surface" in context
    # The artifact-reveal onboarding line is simple-only.
    assert 'show_panel("artifacts")' not in context
    assert "WORKSPACE (id=artifacts, shown)" in context


def test_simple_surface_survives_down_web_terminal(hook_runner, monkeypatch):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(_closed_port()))
    monkeypatch.setenv("OSPREY_WEB_UX", "simple")

    context = _context(hook_runner(HOOK, "", {}))
    assert "SIMPLE surface" in context
    assert "id=artifacts" not in context  # no inventory available


def test_no_marker_and_down_web_terminal_is_silent(hook_runner, monkeypatch):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(_closed_port()))
    monkeypatch.delenv("OSPREY_WEB_UX", raising=False)

    assert hook_runner(HOOK, "", {}) is None


def test_no_marker_keeps_inventory_only_behavior(hook_runner, monkeypatch, panels_server):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(panels_server))
    monkeypatch.delenv("OSPREY_WEB_UX", raising=False)

    context = _context(hook_runner(HOOK, "", {}))
    assert "surface" not in context
    assert "WORKSPACE (id=artifacts, shown)" in context


def test_unknown_marker_is_ignored(hook_runner, monkeypatch):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(_closed_port()))
    monkeypatch.setenv("OSPREY_WEB_UX", "bogus")

    assert hook_runner(HOOK, "", {}) is None
