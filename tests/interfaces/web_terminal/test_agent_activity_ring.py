"""Tests for the agent-activity routes (`routes/agent_activity.py`) and the history ring.

``POST /api/agent-activity`` carries a fixed interface contract shared with the
frontend highlighter::

    request:   {"tool": str, "target": {"kind": "panel"|"channel"|"run"|"artifact"
                                                |"config"|"ui",
                                        "panel"?: str, "detail"?: str}}
    broadcast: {"type": "agent_activity", "tool": ..., "target": {...}, "ts": ...}

The server adds ``type`` and ``ts`` and fans the frame out through the
``FileEventBroadcaster`` on ``app.state.broadcaster``. The SSE stream only
reaches browsers that are already connected, so every accepted event is also
recorded in a bounded deque on ``app.state.agent_activity_ring``, which a
browser that opens or reloads mid-session reads back.

Four concerns are covered, in the sections below:

1. the POST frame shape per target kind, and 422 with nothing broadcast or
   recorded for every malformed body;
2. app startup creates the bounded ring, and an event is recorded before it
   is broadcast;
3. ``GET /api/agent-activity/recent`` reads the ring back newest-first, with a
   clamped ``limit``;
4. the panel routes mirror agent-origin commands into the same ring under
   synthetic tool names, one row per action, and never mirror human gestures.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX, router
from osprey.interfaces.web_terminal.routes.panels import router as panels_router

from .conftest import bare_route_app


def _make_client() -> TestClient:
    """A minimal app exposing the activity router over a ring and a stub broadcaster."""
    return TestClient(bare_route_app(router))


def _post(client: TestClient, tool: str = "open_panel", **target) -> None:
    """POST one activity event, asserting it was accepted."""
    body = {"tool": tool, "target": target or {"kind": "panel"}}
    assert client.post("/api/agent-activity", json=body).status_code == 200


# ---- POST: broadcast frame shape per target kind ----


@pytest.mark.parametrize(
    ("body", "expected_target"),
    [
        (
            {"tool": "open_panel", "target": {"kind": "panel", "panel": "ariel"}},
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
            {"tool": "run_plan", "target": {"kind": "run", "detail": "orm-42"}},
            {"kind": "run", "detail": "orm-42"},
        ),
        (
            {"tool": "create_artifact", "target": {"kind": "artifact"}},
            {"kind": "artifact"},
        ),
        ({"tool": "emit", "target": {"kind": "config"}}, {"kind": "config"}),
        ({"tool": "emit", "target": {"kind": "ui"}}, {"kind": "ui"}),
    ],
)
def test_valid_post_broadcasts_frame_and_returns_ok(body, expected_target):
    """Each valid kind returns {"ok": true} and broadcasts the exact frame shape.

    Optional target fields are omitted when absent, never sent as null.
    """
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


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="empty"),
        pytest.param({"tool": "open_panel", "target": {"kind": "widget"}}, id="unknown-kind"),
        pytest.param({"target": {"kind": "panel", "panel": "ariel"}}, id="missing-tool"),
        pytest.param({"tool": "open_panel"}, id="missing-target"),
        pytest.param({"tool": "open_panel", "target": {"panel": "ariel"}}, id="missing-kind"),
        pytest.param({"tool": "open_panel", "target": "panel"}, id="target-not-object"),
        pytest.param({"tool": 42, "target": {"kind": "panel"}}, id="tool-not-string"),
        # Over-long strings (bounds: tool/panel 256, detail 1024).
        pytest.param({"tool": "x" * 257, "target": {"kind": "panel"}}, id="long-tool"),
        pytest.param(
            {"tool": "open_panel", "target": {"kind": "panel", "panel": "p" * 257}},
            id="long-panel",
        ),
        pytest.param(
            {"tool": "write_channel", "target": {"kind": "channel", "detail": "d" * 1025}},
            id="long-detail",
        ),
    ],
)
def test_malformed_body_422_and_no_broadcast(body):
    """Malformed bodies and unknown kinds are rejected with 422; nothing is broadcast
    and nothing enters the history."""
    client = _make_client()
    resp = client.post("/api/agent-activity", json=body)
    assert resp.status_code == 422
    client.app.state.broadcaster.broadcast.assert_not_called()
    assert len(client.app.state.agent_activity_ring) == 0


@pytest.mark.parametrize("kind", ["widget", "Config", "UI", "", "settings"])
def test_unknown_kinds_still_rejected(kind):
    """The kind Literal stays closed and case-sensitive: never a free-form string."""
    client = _make_client()
    resp = client.post("/api/agent-activity", json={"tool": "emit", "target": {"kind": kind}})

    assert resp.status_code == 422
    assert len(client.app.state.agent_activity_ring) == 0


@pytest.mark.parametrize(
    ("path", "method"),
    [("/api/agent-activity", "post"), ("/api/agent-activity/recent", "get")],
)
def test_routes_registered_on_composite_router(path, method):
    """The composite web-terminal router exposes both activity routes.

    Starlette 1.x ``include_router`` does not flatten, so registration is
    asserted through the OpenAPI schema of an app mounting the composite
    router, never through ``router.routes``.
    """
    from osprey.interfaces.web_terminal.routes import router as composite_router

    app = FastAPI()
    app.include_router(composite_router)
    paths = app.openapi()["paths"]
    assert path in paths
    assert method in paths[path]


# ---- The ring: built at startup, written before the broadcast ----


def test_app_lifespan_builds_a_bounded_ring(tmp_path):
    """The ring is created with its ``app.state`` peers when the app starts up."""
    from osprey.interfaces.web_terminal.app import create_app

    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(tmp_path)},
    ):
        app = create_app(shell_command="echo")
        with TestClient(app):
            ring = app.state.agent_activity_ring

    assert isinstance(ring, deque)
    assert ring.maxlen == ACTIVITY_RING_MAX
    assert len(ring) == 0


def test_append_happens_before_broadcast():
    """A broadcast never fires for an event the ring has not yet recorded.

    A late browser reads the ring; ordering the append first means there is no
    window in which a connected client has seen an event the ring is missing.
    """
    client = _make_client()
    seen_at_broadcast: list[int] = []
    client.app.state.broadcaster.broadcast.side_effect = lambda _frame: seen_at_broadcast.append(
        len(client.app.state.agent_activity_ring)
    )

    _post(client)

    assert seen_at_broadcast == [1]


# ---- GET /api/agent-activity/recent reads the ring back ----


def _get_recent(client: TestClient, **params) -> list[dict]:
    """GET the recent-activity history, asserting a 200, and return its events."""
    resp = client.get("/api/agent-activity/recent", params=params)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"events"}
    return body["events"]


def test_recent_returns_newest_first():
    """The popover wants the latest action at the top, so the ring is reversed."""
    client = _make_client()
    for name in ("first", "second", "third"):
        _post(client, tool=name)

    assert [event["tool"] for event in _get_recent(client)] == ["third", "second", "first"]


def test_recent_returns_the_broadcast_frame_verbatim():
    """Consumers reuse their SSE handler, so the frame must survive the round trip."""
    client = _make_client()
    _post(client, tool="read_channel", kind="channel", detail="SR01C:BPM1:X")

    frame = client.app.state.broadcaster.broadcast.call_args[0][0]
    assert _get_recent(client) == [
        {
            "type": "agent_activity",
            "tool": "read_channel",
            "target": {"kind": "channel", "detail": "SR01C:BPM1:X"},
            "ts": frame["ts"],
        }
    ]


def test_recent_limit_takes_the_newest_events():
    """``limit`` trims the tail of the history, never the head."""
    client = _make_client()
    for index in range(5):
        _post(client, tool=f"tool-{index}")

    assert [e["tool"] for e in _get_recent(client, limit=2)] == ["tool-4", "tool-3"]


def test_recent_defaults_to_the_whole_ring():
    """Omitting ``limit`` returns everything the ring holds."""
    client = _make_client()
    for index in range(ACTIVITY_RING_MAX):
        _post(client, tool=f"tool-{index}")

    assert len(_get_recent(client)) == ACTIVITY_RING_MAX


def test_recent_limit_above_the_ring_max_is_clamped_not_rejected():
    """An over-large ``limit`` yields everything available rather than a 422."""
    client = _make_client()
    for index in range(3):
        _post(client, tool=f"tool-{index}")

    assert len(_get_recent(client, limit=ACTIVITY_RING_MAX * 10)) == 3


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_recent_non_positive_limit_returns_nothing(limit):
    """Zero and negative limits clamp to zero — never to "the whole ring"."""
    client = _make_client()
    _post(client)

    assert _get_recent(client, limit=limit) == []


def test_recent_non_integer_limit_is_rejected():
    """``limit`` stays typed: a junk value is a 422, not a silent default."""
    client = _make_client()
    assert client.get("/api/agent-activity/recent", params={"limit": "lots"}).status_code == 422


# ---- Panel routes mirror agent-origin commands into the ring ----
#
# Panel commands never pass through POST /api/agent-activity — they have their
# own SSE frames — so ``routes/panels.py`` appends an equivalent row directly.
# The synthetic tool name is what the frontend words the entry from, so each
# one is pinned here.

# Resolve the register route's SSRF check to a routable LAN address without
# real DNS.
_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"
#: Panel already in the launcher rail, so focusing it adds no membership.
_MEMBER_PANEL = "ariel"
#: Enabled panel deliberately left OUT of the rail, so focusing it takes the
#: membership-add path that broadcasts a visibility frame before the focus one.
_NON_MEMBER_PANEL = "artifacts"


def _make_panel_client() -> TestClient:
    """An app exposing the panel routes *and* the activity routes over one ring.

    Both routers share ``app.state``, so a mirrored panel row can be read back
    through ``GET /api/agent-activity/recent`` — the round trip a browser makes.
    """
    return TestClient(
        bare_route_app(
            panels_router,
            router,
            enabled_panels={_MEMBER_PANEL, _NON_MEMBER_PANEL},
            custom_panels=[],
            visible_panels=[_MEMBER_PANEL],
            allow_runtime_panels=True,
        )
    )


def _panel_post(client: TestClient, path: str, body: dict) -> None:
    """POST a panel command, asserting it was accepted."""
    assert client.post(path, json=body).status_code == 200


def _only_row(client: TestClient) -> dict:
    """The ring's single row, asserting there is exactly one."""
    ring = client.app.state.agent_activity_ring
    assert len(ring) == 1, [event["tool"] for event in ring]
    return ring[0]


@pytest.mark.parametrize(
    ("path", "body", "tool", "panel"),
    [
        pytest.param(
            "/api/panel-focus",
            {"panel": _MEMBER_PANEL, "source": "agent"},
            "open_panel",
            _MEMBER_PANEL,
            id="focus",
        ),
        pytest.param(
            "/api/panel-visibility",
            {"panel": _NON_MEMBER_PANEL, "visible": True, "source": "agent"},
            "add_panel_to_rail",
            _NON_MEMBER_PANEL,
            id="rail-add",
        ),
        pytest.param(
            "/api/panel-visibility",
            {"panel": _MEMBER_PANEL, "visible": False, "source": "agent"},
            "remove_panel_from_rail",
            _MEMBER_PANEL,
            id="rail-remove",
        ),
        pytest.param(
            "/api/panel-arrange",
            {
                "tiles": [_MEMBER_PANEL, _NON_MEMBER_PANEL],
                "focus": _NON_MEMBER_PANEL,
                "source": "agent",
            },
            "arrange_workspace",
            _NON_MEMBER_PANEL,
            id="arrange",
        ),
        pytest.param(
            "/api/panels/register",
            {
                "id": "grafana",
                "label": "GRAFANA",
                "url": "http://grafana.lan:3000",
                "source": "agent",
            },
            "register_panel",
            "grafana",
            id="register",
        ),
    ],
)
def test_agent_panel_command_mirrors_one_row(path, body, tool, panel):
    """Each agent panel verb is one row, shaped exactly like a broadcast activity frame."""
    client = _make_panel_client()
    with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
        _panel_post(client, path, body)

    row = _only_row(client)
    assert row == {
        "type": "agent_activity",
        "tool": tool,
        "target": {"kind": "panel", "panel": panel},
        "ts": row["ts"],
    }
    assert isinstance(row["ts"], float)


def test_membership_adding_switch_mirrors_only_the_focus():
    """Two frames, one action: the visibility frame that precedes the focus is
    part of the same switch, so history records the switch once."""
    client = _make_panel_client()
    _panel_post(client, "/api/panel-focus", {"panel": _NON_MEMBER_PANEL, "source": "agent"})

    # Both frames really did go out — the mirroring is what is deduped, not the
    # broadcast, so this test would pass vacuously if the path had changed.
    kinds = [call[0][0]["type"] for call in client.app.state.broadcaster.broadcast.call_args_list]
    assert kinds == ["panel_visibility", "panel_focus"]

    row = _only_row(client)
    assert row["tool"] == "open_panel"
    assert row["target"] == {"kind": "panel", "panel": _NON_MEMBER_PANEL}


def test_agent_arrange_without_focus_targets_the_first_tile():
    """No requested focus: the row names the tile the server records as active."""
    client = _make_panel_client()
    _panel_post(
        client,
        "/api/panel-arrange",
        {"tiles": [_NON_MEMBER_PANEL, _MEMBER_PANEL], "source": "agent"},
    )

    assert _only_row(client)["target"] == {"kind": "panel", "panel": _NON_MEMBER_PANEL}
    assert client.app.state.active_panel == _NON_MEMBER_PANEL


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/panel-focus", {"panel": _MEMBER_PANEL}),
        ("/api/panel-focus", {"panel": _NON_MEMBER_PANEL}),  # membership-add path
        ("/api/panel-visibility", {"panel": _MEMBER_PANEL, "visible": False}),
        ("/api/panel-visibility", {"panel": _NON_MEMBER_PANEL, "visible": True}),
        ("/api/panel-arrange", {"tiles": [_MEMBER_PANEL]}),
        (
            "/api/panels/register",
            {"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"},
        ),
    ],
)
def test_human_origin_commands_are_never_mirrored(path, body):
    """The history is the *agent's* activity: an operator's own gestures — which
    carry no ``source`` — leave it empty."""
    client = _make_panel_client()
    with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
        _panel_post(client, path, body)

    assert list(client.app.state.agent_activity_ring) == []


def test_mirrored_rows_read_back_through_the_recent_endpoint():
    """The mirrored rows are consumable by the SSE handler, newest first."""
    client = _make_panel_client()
    _panel_post(client, "/api/panel-focus", {"panel": _MEMBER_PANEL, "source": "agent"})
    _panel_post(
        client,
        "/api/panel-visibility",
        {"panel": _MEMBER_PANEL, "visible": False, "source": "agent"},
    )

    events = _get_recent(client)
    assert [event["tool"] for event in events] == ["remove_panel_from_rail", "open_panel"]
    assert all(set(event) == {"type", "tool", "target", "ts"} for event in events)
    assert all(event["type"] == "agent_activity" for event in events)
