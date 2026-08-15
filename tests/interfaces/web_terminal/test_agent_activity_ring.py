"""Tests for the agent-activity history ring and the activity-kind schema.

The SSE stream (``GET /api/files/events``) only reaches browsers that are
already connected, so ``POST /api/agent-activity`` also records every accepted
event in a bounded deque on ``app.state.agent_activity_ring``.  A browser that
opens or reloads mid-session reads that ring to catch up.

Five concerns are covered, in the sections below:

1. app startup creates the ring alongside its ``app.state`` peers;
2. the POST handler appends each accepted event — and only accepted events —
   to the ring before broadcasting, bounded at ``ACTIVITY_RING_MAX``;
3. the target-kind schema accepts ``config`` and ``ui`` alongside the original
   four kinds, and still rejects everything else;
4. ``GET /api/agent-activity/recent`` reads the ring back newest-first, with a
   clamped ``limit``;
5. the panel routes mirror agent-origin commands into the same ring under
   synthetic tool names, one row per action, and never mirror human gestures.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX, router


def _make_client(*, with_ring: bool = True) -> TestClient:
    """A minimal app exposing the router, a stub broadcaster and (optionally) a ring."""
    app = FastAPI()
    app.include_router(router)
    app.state.broadcaster = MagicMock()
    if with_ring:
        app.state.agent_activity_ring = deque(maxlen=ACTIVITY_RING_MAX)
    return TestClient(app)


def _post(client: TestClient, tool: str = "open_panel", **target) -> None:
    """POST one activity event, asserting it was accepted."""
    body = {"tool": tool, "target": target or {"kind": "panel"}}
    assert client.post("/api/agent-activity", json=body).status_code == 200


# ---- App startup wires the ring onto app.state ----


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


# ---- Ring append semantics ----


def test_accepted_event_is_appended_to_the_ring():
    """The ring entry is the same frame the broadcaster receives."""
    client = _make_client()
    _post(client, tool="read_channel", kind="channel", detail="SR01C:BPM1:X")

    ring = client.app.state.agent_activity_ring
    assert len(ring) == 1
    frame = client.app.state.broadcaster.broadcast.call_args[0][0]
    assert ring[0] == frame
    assert ring[0] == {
        "type": "agent_activity",
        "tool": "read_channel",
        "target": {"kind": "channel", "detail": "SR01C:BPM1:X"},
        "ts": frame["ts"],
    }


def test_ring_keeps_events_in_arrival_order():
    """Oldest first — the ring reads as a timeline, not a stack."""
    client = _make_client()
    for name in ("first", "second", "third"):
        _post(client, tool=name)

    ring = client.app.state.agent_activity_ring
    assert [event["tool"] for event in ring] == ["first", "second", "third"]


def test_ring_is_bounded_and_drops_the_oldest():
    """Past ``ACTIVITY_RING_MAX`` events the ring holds only the newest ones."""
    client = _make_client()
    overflow = 5
    for index in range(ACTIVITY_RING_MAX + overflow):
        _post(client, tool=f"tool-{index}")

    ring = client.app.state.agent_activity_ring
    assert len(ring) == ACTIVITY_RING_MAX
    assert ring[0]["tool"] == f"tool-{overflow}"
    assert ring[-1]["tool"] == f"tool-{ACTIVITY_RING_MAX + overflow - 1}"


def test_rejected_event_is_not_appended():
    """A 422 leaves the ring untouched, just as it broadcasts nothing."""
    client = _make_client()
    resp = client.post("/api/agent-activity", json={"tool": "x", "target": {"kind": "widget"}})

    assert resp.status_code == 422
    assert len(client.app.state.agent_activity_ring) == 0
    client.app.state.broadcaster.broadcast.assert_not_called()


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


def test_missing_ring_does_not_break_the_route():
    """Apps mounting the router standalone simply get no history."""
    client = _make_client(with_ring=False)
    _post(client)

    client.app.state.broadcaster.broadcast.assert_called_once()
    assert not hasattr(client.app.state, "agent_activity_ring")


# ---- Target-kind schema ----


@pytest.mark.parametrize("kind", ["panel", "channel", "run", "artifact", "config", "ui"])
def test_supported_kinds_are_accepted(kind):
    """The original four kinds plus ``config`` and ``ui`` all round-trip."""
    client = _make_client()
    _post(client, tool="emit", kind=kind)

    frame = client.app.state.broadcaster.broadcast.call_args[0][0]
    assert frame["target"]["kind"] == kind
    assert client.app.state.agent_activity_ring[0]["target"]["kind"] == kind


@pytest.mark.parametrize("kind", ["widget", "Config", "UI", "", "settings"])
def test_unknown_kinds_still_rejected(kind):
    """Widening the Literal must not turn it into a free-form string."""
    client = _make_client()
    resp = client.post("/api/agent-activity", json={"tool": "emit", "target": {"kind": kind}})

    assert resp.status_code == 422
    assert len(client.app.state.agent_activity_ring) == 0


# ---- GET /api/agent-activity/recent reads the ring back ----


def _get_recent(client: TestClient, **params) -> list[dict]:
    """GET the recent-activity history, asserting a 200, and return its events."""
    resp = client.get("/api/agent-activity/recent", params=params)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"events"}
    return body["events"]


def test_recent_is_empty_before_any_activity():
    """A fresh ring reads back as an empty list, not a 404 or a null."""
    assert _get_recent(_make_client()) == []


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


def test_recent_without_a_ring_reports_no_history():
    """Standalone mounts have no ring; the read degrades to an empty history."""
    client = _make_client(with_ring=False)
    _post(client)

    assert _get_recent(client) == []


def test_recent_route_registered_on_composite_router():
    """The composite web-terminal router exposes GET /api/agent-activity/recent.

    Starlette 1.x ``include_router`` does not flatten, so registration is
    asserted through the OpenAPI schema, never through ``router.routes``.
    """
    from osprey.interfaces.web_terminal.routes import router as composite_router

    app = FastAPI()
    app.include_router(composite_router)
    paths = app.openapi()["paths"]
    assert "/api/agent-activity/recent" in paths
    assert "get" in paths["/api/agent-activity/recent"]


# ---- Panel routes mirror agent-origin commands into the ring ----
#
# Panel commands never pass through POST /api/agent-activity — they have their
# own SSE frames — so ``routes/panels.py`` appends an equivalent row directly.
# The synthetic tool name is what the frontend words the entry from, so each
# one is pinned here.

# Resolve the register route's SSRF check to a routable LAN address without
# real DNS (the pattern test_panels_source_tag.py uses).
_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"

#: Panel already in the launcher rail, so focusing it adds no membership.
_MEMBER_PANEL = "ariel"
#: Enabled panel deliberately left OUT of the rail, so focusing it takes the
#: membership-add path that broadcasts a visibility frame before the focus one.
_NON_MEMBER_PANEL = "artifacts"


def _make_panel_client(*, with_ring: bool = True) -> TestClient:
    """An app exposing the panel routes *and* the activity routes over one ring.

    Both routers share ``app.state``, so a mirrored panel row can be read back
    through ``GET /api/agent-activity/recent`` — the round trip a browser makes.
    """
    from osprey.interfaces.web_terminal.routes.panels import router as panels_router

    app = FastAPI()
    app.include_router(panels_router)
    app.include_router(router)
    app.state.broadcaster = MagicMock()
    app.state.enabled_panels = {_MEMBER_PANEL, _NON_MEMBER_PANEL}
    app.state.custom_panels = []
    app.state.visible_panels = [_MEMBER_PANEL]
    app.state.allow_runtime_panels = True
    if with_ring:
        app.state.agent_activity_ring = deque(maxlen=ACTIVITY_RING_MAX)
    return TestClient(app)


def _panel_post(client: TestClient, path: str, body: dict) -> None:
    """POST a panel command, asserting it was accepted."""
    assert client.post(path, json=body).status_code == 200


def _only_row(client: TestClient) -> dict:
    """The ring's single row, asserting there is exactly one."""
    ring = client.app.state.agent_activity_ring
    assert len(ring) == 1, [event["tool"] for event in ring]
    return ring[0]


def test_agent_focus_mirrors_an_open_panel_row():
    """A ``open_panel`` row, shaped exactly like a broadcast activity frame."""
    client = _make_panel_client()
    _panel_post(client, "/api/panel-focus", {"panel": _MEMBER_PANEL, "source": "agent"})

    row = _only_row(client)
    assert row == {
        "type": "agent_activity",
        "tool": "open_panel",
        "target": {"kind": "panel", "panel": _MEMBER_PANEL},
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


def test_agent_rail_add_mirrors_an_add_panel_to_rail_row():
    """Putting a panel on the rail is an ``add_panel_to_rail`` row."""
    client = _make_panel_client()
    _panel_post(
        client,
        "/api/panel-visibility",
        {"panel": _NON_MEMBER_PANEL, "visible": True, "source": "agent"},
    )

    row = _only_row(client)
    assert row["tool"] == "add_panel_to_rail"
    assert row["target"] == {"kind": "panel", "panel": _NON_MEMBER_PANEL}


def test_agent_rail_remove_mirrors_a_remove_panel_from_rail_row():
    """A hide is always mirrored — it broadcasts no focus frame to defer to."""
    client = _make_panel_client()
    _panel_post(
        client,
        "/api/panel-visibility",
        {"panel": _MEMBER_PANEL, "visible": False, "source": "agent"},
    )

    kinds = [call[0][0]["type"] for call in client.app.state.broadcaster.broadcast.call_args_list]
    assert kinds == ["panel_visibility"]

    row = _only_row(client)
    assert row["tool"] == "remove_panel_from_rail"
    assert row["target"] == {"kind": "panel", "panel": _MEMBER_PANEL}


def test_agent_arrange_mirrors_one_row_targeting_the_focus():
    """A whole-workspace arrangement is one row, named by where focus lands."""
    client = _make_panel_client()
    _panel_post(
        client,
        "/api/panel-arrange",
        {
            "tiles": [_MEMBER_PANEL, _NON_MEMBER_PANEL],
            "focus": _NON_MEMBER_PANEL,
            "source": "agent",
        },
    )

    row = _only_row(client)
    assert row["tool"] == "arrange_workspace"
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


def test_agent_register_mirrors_a_register_panel_row():
    """A runtime registration is a ``register_panel`` row keyed by the new id."""
    client = _make_panel_client()
    with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
        _panel_post(
            client,
            "/api/panels/register",
            {
                "id": "grafana",
                "label": "GRAFANA",
                "url": "http://grafana.lan:3000",
                "source": "agent",
            },
        )

    row = _only_row(client)
    assert row["tool"] == "register_panel"
    assert row["target"] == {"kind": "panel", "panel": "grafana"}


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


def test_panel_routes_mounted_without_a_ring_do_not_break():
    """Standalone mounts carry no ring; the command still applies and broadcasts."""
    client = _make_panel_client(with_ring=False)
    _panel_post(client, "/api/panel-focus", {"panel": _MEMBER_PANEL, "source": "agent"})

    client.app.state.broadcaster.broadcast.assert_called_once()
    assert not hasattr(client.app.state, "agent_activity_ring")


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
