"""Tests for the session_log MCP tool.

Validates filtering by agent, tool, errors_only, event_type, last_n,
agent_id, since/before, list_agents, and agent_id-based attribution
that attributes tool_calls to subagents.

Fixtures create synthetic event lists in the shape the record read converts
the session to, and stand them in for ``read_session_events``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.mcp_server.conftest import (
    assert_raises_error,
    get_tool_fn,
)


def _get_session_log():
    from osprey.mcp_server.workspace.tools.session_log import session_log

    return get_tool_fn(session_log)


def _ts(minute: int) -> str:
    """Generate an ISO timestamp at a fixed date with the given minute."""
    return datetime(2026, 2, 19, 12, minute, 0, tzinfo=UTC).isoformat()


def _zulu(hour: int) -> str:
    """Generate a stamp in the transcript's own layout: UTC, milliseconds, ``Z`` suffix."""
    return f"2026-02-19T{hour:02d}:00:00.000Z"


def _patch_transcript_reader(events: list[dict], record: str = "complete", reason: str = ""):
    """Return a context manager standing *events* in for the session's record read."""
    from osprey.mcp_server.workspace.tools.session_log import SessionEvents

    return patch(
        "osprey.mcp_server.workspace.tools.session_log.read_session_events",
        return_value=SessionEvents(events, record, reason, "sess-1"),
    )


async def _call_session_log(events: list[dict], tmp_path, **kwargs) -> dict:
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.deployed_render_dir",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(events),
    ):
        return json.loads(await fn(**kwargs))


@pytest.fixture
def zulu_events():
    """Three tool calls stamped 18:00Z, 19:00Z and 20:00Z in the transcript's layout."""
    return [
        {
            "type": "tool_call",
            "timestamp": _zulu(hour),
            "tool": f"tool_{hour}",
            "full_tool_name": f"mcp__osprey_workspace__tool_{hour}",
            "server": "osprey_workspace",
            "is_error": False,
            "session_id": None,
            "arguments": {},
            "result_summary": "ok",
        }
        for hour in (18, 19, 20)
    ]


@pytest.fixture
def basic_events():
    """Known events for basic testing.

    Timeline (minute):
      0  — agent_start (data-visualizer, agent-1)
      1  — tool_call (create_static_plot, is_error=false, agent_id=agent-1)
      2  — tool_call (create_dashboard, is_error=true, agent_id=agent-1)
      3  — agent_stop (data-visualizer, agent-1)
      4  — tool_call (channel_read, from main agent — no agent_id)
    """
    return [
        {
            "type": "agent_start",
            "timestamp": _ts(0),
            "agent_id": "agent-1",
            "agent_type": "data-visualizer",
        },
        {
            "type": "tool_call",
            "timestamp": _ts(1),
            "tool": "create_static_plot",
            "full_tool_name": "mcp__osprey_workspace__create_static_plot",
            "server": "osprey_workspace",
            "is_error": False,
            "session_id": None,
            "agent_id": "agent-1",
            "arguments": {"title": "Test"},
            "result_summary": "ok",
        },
        {
            "type": "tool_call",
            "timestamp": _ts(2),
            "tool": "create_dashboard",
            "full_tool_name": "mcp__osprey_workspace__create_dashboard",
            "server": "osprey_workspace",
            "is_error": True,
            "session_id": None,
            "agent_id": "agent-1",
            "arguments": {"title": "Dash"},
            "result_summary": '{"error": true, "error_message": "render failed"}',
        },
        {
            "type": "agent_stop",
            "timestamp": _ts(3),
            "agent_id": "agent-1",
            "agent_type": "data-visualizer",
        },
        {
            "type": "tool_call",
            "timestamp": _ts(4),
            "tool": "channel_read",
            "full_tool_name": "mcp__controls__channel_read",
            "server": "controls",
            "is_error": False,
            "session_id": None,
            "agent_id": None,
            "arguments": {"channels": ["SR:CURRENT"]},
            "result_summary": "500.0",
        },
    ]


@pytest.fixture
def concurrent_agent_events():
    """Two concurrent data-visualizer agents with distinct agent_ids.

    Timeline (minute):
      0  — agent_start (data-visualizer, agent-A)
      1  — agent_start (data-visualizer, agent-B)
      2  — tool_call (create_static_plot, agent_id=agent-A)
      3  — tool_call (create_dashboard, agent_id=agent-B, is_error=true)
      4  — tool_call (create_interactive_plot, agent_id=agent-A)
      5  — agent_stop (data-visualizer, agent-A)
      6  — agent_stop (data-visualizer, agent-B)
      7  — tool_call (channel_read, no agent_id — main agent)
    """
    return [
        {
            "type": "agent_start",
            "timestamp": _ts(0),
            "agent_id": "agent-A",
            "agent_type": "data-visualizer",
        },
        {
            "type": "agent_start",
            "timestamp": _ts(1),
            "agent_id": "agent-B",
            "agent_type": "data-visualizer",
        },
        {
            "type": "tool_call",
            "timestamp": _ts(2),
            "tool": "create_static_plot",
            "full_tool_name": "mcp__osprey_workspace__create_static_plot",
            "server": "osprey_workspace",
            "is_error": False,
            "session_id": None,
            "agent_id": "agent-A",
            "arguments": {"title": "Plot A"},
            "result_summary": "ok",
        },
        {
            "type": "tool_call",
            "timestamp": _ts(3),
            "tool": "create_dashboard",
            "full_tool_name": "mcp__osprey_workspace__create_dashboard",
            "server": "osprey_workspace",
            "is_error": True,
            "session_id": None,
            "agent_id": "agent-B",
            "arguments": {"title": "Dash B"},
            "result_summary": '{"error": true, "error_message": "render failed"}',
        },
        {
            "type": "tool_call",
            "timestamp": _ts(4),
            "tool": "create_interactive_plot",
            "full_tool_name": "mcp__osprey_workspace__create_interactive_plot",
            "server": "osprey_workspace",
            "is_error": False,
            "session_id": None,
            "agent_id": "agent-A",
            "arguments": {"title": "Interactive A"},
            "result_summary": "ok",
        },
        {
            "type": "agent_stop",
            "timestamp": _ts(5),
            "agent_id": "agent-A",
            "agent_type": "data-visualizer",
            "agent_transcript_path": "/t/agent-a.jsonl",
        },
        {
            "type": "agent_stop",
            "timestamp": _ts(6),
            "agent_id": "agent-B",
            "agent_type": "data-visualizer",
            "agent_transcript_path": "/t/agent-b.jsonl",
        },
        {
            "type": "tool_call",
            "timestamp": _ts(7),
            "tool": "channel_read",
            "full_tool_name": "mcp__controls__channel_read",
            "server": "controls",
            "is_error": False,
            "session_id": None,
            "agent_id": None,
            "arguments": {"channels": ["SR:CURRENT"]},
            "result_summary": "500.0",
        },
    ]


@pytest.mark.asyncio
async def test_no_filters(basic_events, tmp_path):
    """Returns all events when no filters are applied."""
    result = await _call_session_log(basic_events, tmp_path)
    assert result["total_events"] == 5
    assert result["showing"] == 5


@pytest.mark.asyncio
async def test_agent_filter(basic_events, tmp_path):
    """Agent filter returns lifecycle events + tool_calls with matching agent_id."""
    result = await _call_session_log(basic_events, tmp_path, agent="data-visualizer")
    # Should include: agent_start, create_static_plot, create_dashboard, agent_stop
    assert result["total_events"] == 4
    types = [e["type"] for e in result["events"]]
    assert "agent_start" in types
    assert "agent_stop" in types
    assert types.count("tool_call") == 2


@pytest.mark.asyncio
async def test_agent_excludes_outside_calls(basic_events, tmp_path):
    """Tool calls from main agent are excluded by agent filter."""
    result = await _call_session_log(basic_events, tmp_path, agent="data-visualizer")
    tool_names = [e.get("tool") for e in result["events"] if e["type"] == "tool_call"]
    assert "channel_read" not in tool_names


@pytest.mark.asyncio
async def test_tool_filter(basic_events, tmp_path):
    """Tool filter matches by substring on tool name."""
    result = await _call_session_log(basic_events, tmp_path, tool="dashboard")
    assert result["total_events"] == 1
    assert result["events"][0]["tool"] == "create_dashboard"


@pytest.mark.asyncio
async def test_errors_only(basic_events, tmp_path):
    """errors_only returns only tool_call events with is_error=true."""
    result = await _call_session_log(basic_events, tmp_path, errors_only=True)
    assert result["total_events"] == 1
    assert result["events"][0]["tool"] == "create_dashboard"
    assert result["events"][0]["is_error"] is True


@pytest.mark.asyncio
async def test_event_type_filter(basic_events, tmp_path):
    """event_type filter returns only events of that type."""
    result = await _call_session_log(basic_events, tmp_path, event_type="agent_start")
    assert result["total_events"] == 1
    assert result["events"][0]["type"] == "agent_start"


@pytest.mark.asyncio
async def test_last_n(basic_events, tmp_path):
    """last_n returns only the last N events."""
    result = await _call_session_log(basic_events, tmp_path, last_n=2)
    assert result["showing"] == 2
    assert result["total_events"] == 5
    # Should be the last 2 events (agent_stop at min 3, channel_read at min 4)
    assert result["events"][0]["type"] in ("agent_stop", "tool_call")


@pytest.mark.asyncio
async def test_last_n_clamped_to_200(basic_events, tmp_path):
    """Values > 200 for last_n are clamped to 200."""
    result = await _call_session_log(basic_events, tmp_path, last_n=999)
    assert result["filters_applied"]["last_n"] == 200


@pytest.mark.asyncio
async def test_combined_filters(basic_events, tmp_path):
    """Agent + errors_only filters work together."""
    result = await _call_session_log(
        basic_events, tmp_path, agent="data-visualizer", errors_only=True
    )
    assert result["total_events"] == 1
    assert result["events"][0]["tool"] == "create_dashboard"


@pytest.mark.asyncio
async def test_empty_events(tmp_path):
    """Returns empty events list when TranscriptReader returns nothing."""
    result = await _call_session_log([], tmp_path)
    assert result["events"] == []
    assert result["total_events"] == 0


# ---------------------------------------------------------------------------
# Concurrent agents + agent_id-based attribution tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_id_filter(concurrent_agent_events, tmp_path):
    """agent_id filter returns only events for a specific agent instance."""
    result = await _call_session_log(concurrent_agent_events, tmp_path, agent_id="agent-A")
    # agent-A: agent_start + 2 tool_calls + agent_stop = 4
    tool_calls = [e for e in result["events"] if e["type"] == "tool_call"]
    assert len(tool_calls) == 2
    tool_names = {e["tool"] for e in tool_calls}
    assert tool_names == {"create_static_plot", "create_interactive_plot"}
    # No agent-B events
    agent_ids = {e.get("agent_id") for e in result["events"] if e.get("agent_id")}
    assert agent_ids == {"agent-A"}


@pytest.mark.asyncio
async def test_since_filter(concurrent_agent_events, tmp_path):
    """since excludes events before the timestamp."""
    result = await _call_session_log(concurrent_agent_events, tmp_path, since=_ts(5))
    # min 5, 6, 7 → 3 events
    assert result["total_events"] == 3
    timestamps = [e["timestamp"] for e in result["events"]]
    assert all(ts >= _ts(5) for ts in timestamps)


@pytest.mark.asyncio
async def test_before_filter(concurrent_agent_events, tmp_path):
    """before excludes events after the timestamp."""
    result = await _call_session_log(concurrent_agent_events, tmp_path, before=_ts(1))
    # min 0, 1 → 2 events
    assert result["total_events"] == 2
    timestamps = [e["timestamp"] for e in result["events"]]
    assert all(ts <= _ts(1) for ts in timestamps)


@pytest.mark.asyncio
async def test_since_before_combined(concurrent_agent_events, tmp_path):
    """since + before creates a time-range filter."""
    result = await _call_session_log(concurrent_agent_events, tmp_path, since=_ts(2), before=_ts(4))
    # min 2, 3, 4 → 3 events
    assert result["total_events"] == 3
    timestamps = [e["timestamp"] for e in result["events"]]
    assert all(_ts(2) <= ts <= _ts(4) for ts in timestamps)


@pytest.mark.asyncio
async def test_list_agents_mode(concurrent_agent_events, tmp_path):
    """list_agents returns compact agent summary with tool/error counts."""
    result = await _call_session_log(concurrent_agent_events, tmp_path, list_agents=True)
    assert result["total_agents"] == 2
    agents = {a["agent_id"]: a for a in result["agents"]}

    assert agents["agent-A"]["agent_type"] == "data-visualizer"
    assert agents["agent-A"]["tool_count"] == 2
    assert agents["agent-A"]["error_count"] == 0

    assert agents["agent-B"]["agent_type"] == "data-visualizer"
    assert agents["agent-B"]["tool_count"] == 1
    assert agents["agent-B"]["error_count"] == 1

    assert result["filters_applied"]["list_agents"] is True


@pytest.mark.asyncio
async def test_agent_id_correlation(concurrent_agent_events, tmp_path):
    """Two concurrent same-type agents are correctly disambiguated via agent_id."""
    # agent-B should get only create_dashboard (its agent_id)
    result = await _call_session_log(concurrent_agent_events, tmp_path, agent_id="agent-B")

    tool_calls = [e for e in result["events"] if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "create_dashboard"


@pytest.mark.asyncio
async def test_time_window_fallback(tmp_path):
    """Old data without agent_id on tool_calls falls back to time-window matching."""
    # Events without agent_id on tool_calls (simulating pre-TranscriptReader data)
    events = [
        {
            "type": "agent_start",
            "timestamp": _ts(0),
            "agent_id": "agent-old",
            "agent_type": "data-visualizer",
        },
        {
            "type": "tool_call",
            "timestamp": _ts(1),
            "tool": "create_static_plot",
            "full_tool_name": "mcp__osprey_workspace__create_static_plot",
            "server": "osprey_workspace",
            "is_error": False,
            "session_id": None,
            "arguments": {"title": "Old"},
            "result_summary": "ok",
        },
        {
            "type": "agent_stop",
            "timestamp": _ts(2),
            "agent_id": "agent-old",
            "agent_type": "data-visualizer",
        },
    ]

    result = await _call_session_log(events, tmp_path, agent="data-visualizer")

    # Should still match the tool_call via time-window fallback
    assert result["total_events"] == 3
    tool_calls = [e for e in result["events"] if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "create_static_plot"


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["since", "before"])
async def test_invalid_bound_format(tmp_path, bound):
    """A bad time bound returns a validation error naming it, not a crash."""
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await _call_session_log([], tmp_path, **{bound: "not-a-timestamp"})

    envelope = _exc_ctx["envelope"]
    assert bound in envelope["error_message"]


@pytest.mark.asyncio
async def test_offset_since_admits_events_at_or_after_the_instant(zulu_events, tmp_path):
    """A bound with a UTC offset is compared with the stamps as an instant."""
    result = await _call_session_log(zulu_events, tmp_path, since="2026-02-19T12:00:00-07:00")
    assert [e["timestamp"] for e in result["events"]] == [_zulu(19), _zulu(20)]
    assert result["filters_applied"]["since"] == "2026-02-19T12:00:00-07:00"


@pytest.mark.asyncio
async def test_offset_before_keeps_events_at_or_before_the_instant(zulu_events, tmp_path):
    """An upper bound with a UTC offset keeps every event up to and including that instant."""
    result = await _call_session_log(zulu_events, tmp_path, before="2026-02-19T12:00:00-07:00")
    assert [e["timestamp"] for e in result["events"]] == [_zulu(18), _zulu(19)]


@pytest.mark.asyncio
async def test_z_bound_admits_the_event_stamped_at_that_instant(zulu_events, tmp_path):
    """A ``Z`` bound admits the event whose millisecond stamp is the same instant."""
    result = await _call_session_log(zulu_events, tmp_path, since="2026-02-19T19:00:00Z")
    assert [e["timestamp"] for e in result["events"]] == [_zulu(19), _zulu(20)]


@pytest.mark.asyncio
async def test_naive_bound_is_read_as_utc(zulu_events, tmp_path):
    """A bound without an offset is read in the transcript's zone, UTC."""
    since = await _call_session_log(zulu_events, tmp_path, since="2026-02-19T19:00:00")
    assert [e["timestamp"] for e in since["events"]] == [_zulu(19), _zulu(20)]
    before = await _call_session_log(zulu_events, tmp_path, before="2026-02-19T19:00:00")
    assert [e["timestamp"] for e in before["events"]] == [_zulu(18), _zulu(19)]


@pytest.mark.asyncio
async def test_unparseable_stamp_is_outside_a_filtered_window(tmp_path):
    """An event whose stamp cannot be placed in time is kept out of any window."""
    events = [
        {"type": "tool_call", "timestamp": _zulu(19), "tool": "placed", "is_error": False},
        {"type": "tool_call", "timestamp": "not-a-time", "tool": "garbled", "is_error": False},
        {"type": "tool_call", "tool": "unstamped", "is_error": False},
    ]
    filtered = await _call_session_log(events, tmp_path, since="2026-02-19T00:00:00Z")
    assert [e["tool"] for e in filtered["events"]] == ["placed"]

    unbounded = await _call_session_log(events, tmp_path)
    assert unbounded["total_events"] == 3


@pytest.mark.asyncio
async def test_offset_bound_applies_in_list_agents_mode(tmp_path):
    """list_agents sees only agents whose events lie inside the offset bound."""
    events = [
        {
            "type": "agent_start",
            "timestamp": _zulu(18),
            "agent_id": "agent-A",
            "agent_type": "data-visualizer",
        },
        {
            "type": "agent_stop",
            "timestamp": _zulu(18),
            "agent_id": "agent-A",
            "agent_type": "data-visualizer",
        },
        {
            "type": "agent_start",
            "timestamp": _zulu(20),
            "agent_id": "agent-B",
            "agent_type": "data-visualizer",
        },
        {
            "type": "agent_stop",
            "timestamp": _zulu(20),
            "agent_id": "agent-B",
            "agent_type": "data-visualizer",
        },
    ]
    result = await _call_session_log(
        events, tmp_path, list_agents=True, since="2026-02-19T12:00:00-07:00"
    )
    assert result["total_agents"] == 1
    assert result["agents"][0]["agent_id"] == "agent-B"
    assert result["filters_applied"]["since"] == "2026-02-19T12:00:00-07:00"


@pytest.mark.asyncio
async def test_session_is_resolved_from_the_agents_render_dir(tmp_path, monkeypatch):
    """The current transcript is looked up in the RENDER — the directory the agent runs in.

    Not the repo root, and not the session-isolated agent-data root, which
    moves to ``<root>/sessions/<id>`` with OSPREY_SESSION_ID set.
    """
    render_dir = tmp_path / "repo" / "build"
    render_dir.mkdir(parents=True)
    monkeypatch.setenv("OSPREY_SESSION_ID", "sess-should-not-matter")

    seen: list = []

    def _recording(project_dir):
        from osprey.mcp_server.workspace.tools.session_log import SessionEvents

        seen.append(Path(project_dir))
        return SessionEvents()

    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.deployed_render_dir",
            return_value=render_dir,
        ),
        patch("osprey.mcp_server.workspace.tools.session_log.read_session_events", _recording),
    ):
        json.loads(await fn())

    assert seen == [render_dir]
    assert "agent_data" not in str(seen[0])


@pytest.mark.asyncio
async def test_an_unavailable_record_is_an_error_not_an_empty_log(tmp_path):
    """With the record down the tool says so; it never answers an empty, complete log."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.deployed_render_dir",
            return_value=tmp_path,
        ),
        _patch_transcript_reader([], record="unavailable", reason="connection refused"),
    ):
        with assert_raises_error(error_type="record_unavailable") as _exc_ctx:
            await fn()
    assert "connection refused" in _exc_ctx["envelope"]["error_message"]


@pytest.mark.asyncio
async def test_an_incomplete_read_is_marked_incomplete(basic_events, tmp_path):
    result = await _call_session_log(basic_events, tmp_path)
    assert result["record"] == "complete"
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.deployed_render_dir",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events, record="incomplete", reason="wait mark not passed"),
    ):
        partial = json.loads(await fn())
    assert partial["record"] == "incomplete"
    assert partial["record_reason"] == "wait mark not passed"
    assert partial["total_events"] == result["total_events"]
