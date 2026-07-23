"""Tests for the agent-activity broadcast route (`routes/agent_activity.py`).

``POST /api/agent-activity`` carries a fixed interface contract shared with the
frontend highlighter::

    request:   {"tool": str, "target": {"kind": "panel"|"channel"|"run"|"artifact",
                                        "panel"?: str, "detail"?: str}}
    broadcast: {"type": "agent_activity", "tool": ..., "target": {...}, "ts": ...}

The server adds ``type`` and ``ts`` and fans the frame out through the
``FileEventBroadcaster`` on ``app.state.broadcaster``.  Malformed bodies and
unknown target kinds must 422 without broadcasting anything.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.agent_activity import router


def _make_client() -> TestClient:
    """A minimal app exposing the agent-activity router with a stub broadcaster."""
    app = FastAPI()
    app.include_router(router)
    app.state.broadcaster = MagicMock()
    return TestClient(app)


# ---- Success path: broadcast frame shape per target kind ----


@pytest.mark.parametrize(
    ("body", "expected_target"),
    [
        (
            {"tool": "switch_panel", "target": {"kind": "panel", "panel": "ariel"}},
            {"kind": "panel", "panel": "ariel"},
        ),
        (
            {
                "tool": "read_channel",
                "target": {"kind": "channel", "panel": "channels", "detail": "SR01C:BPM1:X"},
            },
            {"kind": "channel", "panel": "channels", "detail": "SR01C:BPM1:X"},
        ),
        (
            {"tool": "run_scan", "target": {"kind": "run", "detail": "orm-42"}},
            {"kind": "run", "detail": "orm-42"},
        ),
        (
            {"tool": "create_artifact", "target": {"kind": "artifact"}},
            {"kind": "artifact"},
        ),
    ],
)
def test_valid_post_broadcasts_frame_and_returns_ok(body, expected_target):
    """Each valid kind returns {"ok": true} and broadcasts the exact frame shape."""
    client = _make_client()
    resp = client.post("/api/agent-activity", json=body)

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    broadcaster = client.app.state.broadcaster
    broadcaster.broadcast.assert_called_once()
    frame = broadcaster.broadcast.call_args[0][0]
    assert set(frame) == {"type", "tool", "target", "ts"}
    assert frame["type"] == "agent_activity"
    assert frame["tool"] == body["tool"]
    assert frame["target"] == expected_target
    assert isinstance(frame["ts"], float)


def test_optional_target_fields_omitted_when_absent():
    """Absent panel/detail must not appear in the broadcast target (not null)."""
    client = _make_client()
    resp = client.post(
        "/api/agent-activity", json={"tool": "switch_panel", "target": {"kind": "panel"}}
    )
    assert resp.status_code == 200
    frame = client.app.state.broadcaster.broadcast.call_args[0][0]
    assert frame["target"] == {"kind": "panel"}


# ---- Validation failures: 422 and NO broadcast ----


@pytest.mark.parametrize(
    "body",
    [
        # Unknown target kind.
        {"tool": "switch_panel", "target": {"kind": "widget"}},
        # Missing tool.
        {"target": {"kind": "panel", "panel": "ariel"}},
        # Missing target.
        {"tool": "switch_panel"},
        # Target missing kind.
        {"tool": "switch_panel", "target": {"panel": "ariel"}},
        # Target is not an object.
        {"tool": "switch_panel", "target": "panel"},
        # tool is not a string.
        {"tool": 42, "target": {"kind": "panel"}},
        # Over-long strings (bounds: tool/panel 256, detail 1024).
        {"tool": "x" * 257, "target": {"kind": "panel"}},
        {"tool": "switch_panel", "target": {"kind": "panel", "panel": "p" * 257}},
        {"tool": "write_channel", "target": {"kind": "channel", "detail": "d" * 1025}},
    ],
)
def test_malformed_body_422_and_no_broadcast(body):
    """Malformed bodies and unknown kinds are rejected with 422; nothing is broadcast."""
    client = _make_client()
    resp = client.post("/api/agent-activity", json=body)
    assert resp.status_code == 422
    client.app.state.broadcaster.broadcast.assert_not_called()


def test_empty_body_422_and_no_broadcast():
    """A completely empty JSON body is a 422, not a crash or a broadcast."""
    client = _make_client()
    resp = client.post("/api/agent-activity", json={})
    assert resp.status_code == 422
    client.app.state.broadcaster.broadcast.assert_not_called()


# ---- Registration on the composite router ----


def test_route_registered_on_composite_router():
    """The composite web-terminal router exposes POST /api/agent-activity.

    Starlette 1.x ``include_router`` does not flatten, so registration is
    asserted through the OpenAPI schema of an app mounting the composite
    router, never through ``router.routes``.
    """
    from osprey.interfaces.web_terminal.routes import router as composite_router

    app = FastAPI()
    app.include_router(composite_router)
    paths = app.openapi()["paths"]
    assert "/api/agent-activity" in paths
    assert "post" in paths["/api/agent-activity"]
