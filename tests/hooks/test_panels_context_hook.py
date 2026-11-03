"""Tests for the panels-context SessionStart hook.

The hook injects two independent context parts:

1. a web-surface line derived from ``OSPREY_WEB_UX`` (``chat`` | ``terminal``),
   set per session by the web terminal's launchers — the chat line carries
   the "open_panel('artifacts') when you produce an artifact" instruction;
2. the panel inventory fetched from ``GET /api/panels``;
3. the open-tiles line built from that payload's ``layout`` — every tile by
   the one panel it shows, the terminal's included — omitted when no dock
   client has reported.

Either part may be present alone: the surface line must survive a down web
terminal (it is env-only), and inventory-only output is the pre-existing
behavior for sessions with no surface marker. No parts → silent exit 0.
"""

import json
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
    """Stub web terminal serving GET /api/panels on an ephemeral port.

    Yields the server; a test may set ``server.payload`` to serve another body.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # http.server API
            body = json.dumps(getattr(self.server, "payload", PANELS_PAYLOAD)).encode()
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
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _context(result):
    assert result is not None, "expected additionalContext output"
    out = result["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart"
    return out["additionalContext"]


def test_chat_surface_with_inventory(hook_runner, monkeypatch, panels_server):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(panels_server.server_address[1]))
    monkeypatch.setenv("OSPREY_WEB_UX", "chat")

    context = _context(hook_runner(HOOK, "", {}))
    # Surface line first, with the artifact-reveal instruction …
    assert "operator chat" in context
    assert 'open_panel("artifacts")' in context
    # … then the ordinary inventory.
    assert "WORKSPACE (id=artifacts, on rail)" in context
    assert "ARIEL (id=ariel, off rail)" in context


def test_terminal_surface_with_inventory(hook_runner, monkeypatch, panels_server):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(panels_server.server_address[1]))
    monkeypatch.setenv("OSPREY_WEB_UX", "terminal")

    context = _context(hook_runner(HOOK, "", {}))
    assert "terminal on the Agent tile" in context
    # The artifact-reveal line is the chat's alone.
    assert 'open_panel("artifacts")' not in context
    assert "WORKSPACE (id=artifacts, on rail)" in context


def test_chat_surface_survives_down_web_terminal(reserve_port, hook_runner, monkeypatch):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(reserve_port()))
    monkeypatch.setenv("OSPREY_WEB_UX", "chat")

    context = _context(hook_runner(HOOK, "", {}))
    assert "operator chat" in context
    assert "id=artifacts" not in context  # no inventory available


def test_no_marker_and_down_web_terminal_is_silent(reserve_port, hook_runner, monkeypatch):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(reserve_port()))
    monkeypatch.delenv("OSPREY_WEB_UX", raising=False)

    assert hook_runner(HOOK, "", {}) is None


def test_no_marker_keeps_inventory_only_behavior(hook_runner, monkeypatch, panels_server):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(panels_server.server_address[1]))
    monkeypatch.delenv("OSPREY_WEB_UX", raising=False)

    context = _context(hook_runner(HOOK, "", {}))
    assert "surface" not in context
    assert "WORKSPACE (id=artifacts, on rail)" in context


def test_unknown_marker_is_ignored(reserve_port, hook_runner, monkeypatch):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(reserve_port()))
    monkeypatch.setenv("OSPREY_WEB_UX", "bogus")

    assert hook_runner(HOOK, "", {}) is None


def test_tile_line_from_reported_layout(hook_runner, monkeypatch, panels_server):
    """A real hook process turns the reported dock layout into one tile line."""
    panels_server.payload = dict(
        PANELS_PAYLOAD,
        layout={
            "split": "row",
            "items": [
                {"tile": "t1", "tabs": ["terminal"], "showing": "terminal", "size": 0.5},
                {
                    "tile": "t2",
                    "tabs": ["artifacts"],
                    "showing": "artifacts",
                    "size": 0.5,
                },
            ],
        },
        rev=2,
        layout_dock=True,
    )
    monkeypatch.setenv("OSPREY_WEB_PORT", str(panels_server.server_address[1]))
    monkeypatch.delenv("OSPREY_WEB_UX", raising=False)

    context = _context(hook_runner(HOOK, "", {}))
    assert "Open tiles: [terminal], [artifacts]; active artifacts." in context


def test_no_tile_line_before_any_dock_report(hook_runner, monkeypatch, panels_server):
    panels_server.payload = dict(PANELS_PAYLOAD, layout=None, rev=0, layout_dock=None)
    monkeypatch.setenv("OSPREY_WEB_PORT", str(panels_server.server_address[1]))
    monkeypatch.delenv("OSPREY_WEB_UX", raising=False)

    context = _context(hook_runner(HOOK, "", {}))
    assert "Open tiles" not in context
