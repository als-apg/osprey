"""Tests for the session_log MCP tool.

Validates filtering by agent, tool, errors_only, event_type, last_n,
agent_id, since/before, list_agents, and agent_id-based attribution
that attributes tool_calls to subagents.

Fixtures create synthetic event lists that match the output schema of
TranscriptReader, then mock read_current_session() to return them.
"""

import json
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from tests.mcp_server.conftest import get_tool_fn


def _get_session_log():
    from osprey.mcp_server.workspace.tools.session_log import session_log

    return get_tool_fn(session_log)


def _ts(minute: int) -> str:
    """Generate an ISO timestamp at a fixed date with the given minute."""
    return datetime(2026, 2, 19, 12, minute, 0, tzinfo=UTC).isoformat()


def _patch_transcript_reader(events: list[dict]):
    """Return a context manager that patches TranscriptReader.read_current_session."""
    return patch(
        "osprey.mcp_server.workspace.tools.session_log.TranscriptReader",
        **{"return_value.read_current_session.return_value": events},
    )


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
            "full_tool_name": "mcp__workspace__create_static_plot",
            "server": "workspace",
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
            "full_tool_name": "mcp__workspace__create_dashboard",
            "server": "workspace",
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
            "full_tool_name": "mcp__workspace__create_static_plot",
            "server": "workspace",
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
            "full_tool_name": "mcp__workspace__create_dashboard",
            "server": "workspace",
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
            "full_tool_name": "mcp__workspace__create_interactive_plot",
            "server": "workspace",
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
@pytest.mark.unit
async def test_no_filters(basic_events, tmp_path):
    """Returns all events when no filters are applied."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn())
    assert result["total_events"] == 5
    assert result["showing"] == 5


@pytest.mark.asyncio
@pytest.mark.unit
async def test_agent_filter(basic_events, tmp_path):
    """Agent filter returns lifecycle events + tool_calls with matching agent_id."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn(agent="data-visualizer"))
    # Should include: agent_start, create_static_plot, create_dashboard, agent_stop
    assert result["total_events"] == 4
    types = [e["type"] for e in result["events"]]
    assert "agent_start" in types
    assert "agent_stop" in types
    assert types.count("tool_call") == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_agent_excludes_outside_calls(basic_events, tmp_path):
    """Tool calls from main agent are excluded by agent filter."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn(agent="data-visualizer"))
    tool_names = [e.get("tool") for e in result["events"] if e["type"] == "tool_call"]
    assert "channel_read" not in tool_names


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_filter(basic_events, tmp_path):
    """Tool filter matches by substring on tool name."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn(tool="dashboard"))
    assert result["total_events"] == 1
    assert result["events"][0]["tool"] == "create_dashboard"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_errors_only(basic_events, tmp_path):
    """errors_only returns only tool_call events with is_error=true."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn(errors_only=True))
    assert result["total_events"] == 1
    assert result["events"][0]["tool"] == "create_dashboard"
    assert result["events"][0]["is_error"] is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_event_type_filter(basic_events, tmp_path):
    """event_type filter returns only events of that type."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn(event_type="agent_start"))
    assert result["total_events"] == 1
    assert result["events"][0]["type"] == "agent_start"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_last_n(basic_events, tmp_path):
    """last_n returns only the last N events."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn(last_n=2))
    assert result["showing"] == 2
    assert result["total_events"] == 5
    # Should be the last 2 events (agent_stop at min 3, channel_read at min 4)
    assert result["events"][0]["type"] in ("agent_stop", "tool_call")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_last_n_clamped_to_200(basic_events, tmp_path):
    """Values > 200 for last_n are clamped to 200."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn(last_n=999))
    assert result["filters_applied"]["last_n"] == 200


@pytest.mark.asyncio
@pytest.mark.unit
async def test_combined_filters(basic_events, tmp_path):
    """Agent + errors_only filters work together."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(basic_events),
    ):
        result = json.loads(await fn(agent="data-visualizer", errors_only=True))
    assert result["total_events"] == 1
    assert result["events"][0]["tool"] == "create_dashboard"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_empty_events(tmp_path):
    """Returns empty events list when TranscriptReader returns nothing."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader([]),
    ):
        result = json.loads(await fn())
    assert result["events"] == []
    assert result["total_events"] == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_no_transcript(tmp_path):
    """Returns empty events list when no transcript found."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader([]),
    ):
        result = json.loads(await fn())
    assert result["events"] == []
    assert result["total_events"] == 0


# ---------------------------------------------------------------------------
# Concurrent agents + agent_id-based attribution tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_agent_id_filter(concurrent_agent_events, tmp_path):
    """agent_id filter returns only events for a specific agent instance."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(concurrent_agent_events),
    ):
        result = json.loads(await fn(agent_id="agent-A"))
    # agent-A: agent_start + 2 tool_calls + agent_stop = 4
    tool_calls = [e for e in result["events"] if e["type"] == "tool_call"]
    assert len(tool_calls) == 2
    tool_names = {e["tool"] for e in tool_calls}
    assert tool_names == {"create_static_plot", "create_interactive_plot"}
    # No agent-B events
    agent_ids = {e.get("agent_id") for e in result["events"] if e.get("agent_id")}
    assert agent_ids == {"agent-A"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_since_filter(concurrent_agent_events, tmp_path):
    """since excludes events before the timestamp."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(concurrent_agent_events),
    ):
        result = json.loads(await fn(since=_ts(5)))
    # min 5, 6, 7 → 3 events
    assert result["total_events"] == 3
    timestamps = [e["timestamp"] for e in result["events"]]
    assert all(ts >= _ts(5) for ts in timestamps)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_before_filter(concurrent_agent_events, tmp_path):
    """before excludes events after the timestamp."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(concurrent_agent_events),
    ):
        result = json.loads(await fn(before=_ts(1)))
    # min 0, 1 → 2 events
    assert result["total_events"] == 2
    timestamps = [e["timestamp"] for e in result["events"]]
    assert all(ts <= _ts(1) for ts in timestamps)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_since_before_combined(concurrent_agent_events, tmp_path):
    """since + before creates a time-range filter."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(concurrent_agent_events),
    ):
        result = json.loads(await fn(since=_ts(2), before=_ts(4)))
    # min 2, 3, 4 → 3 events
    assert result["total_events"] == 3
    timestamps = [e["timestamp"] for e in result["events"]]
    assert all(_ts(2) <= ts <= _ts(4) for ts in timestamps)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_agents_mode(concurrent_agent_events, tmp_path):
    """list_agents returns compact agent summary with tool/error counts."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(concurrent_agent_events),
    ):
        result = json.loads(await fn(list_agents=True))
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
@pytest.mark.unit
async def test_agent_id_correlation(concurrent_agent_events, tmp_path):
    """Two concurrent same-type agents are correctly disambiguated via agent_id."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(concurrent_agent_events),
    ):
        # agent-B should get only create_dashboard (its agent_id)
        result = json.loads(await fn(agent_id="agent-B"))

    tool_calls = [e for e in result["events"] if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "create_dashboard"


@pytest.mark.asyncio
@pytest.mark.unit
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
            "full_tool_name": "mcp__workspace__create_static_plot",
            "server": "workspace",
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

    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader(events),
    ):
        result = json.loads(await fn(agent="data-visualizer"))

    # Should still match the tool_call via time-window fallback
    assert result["total_events"] == 3
    tool_calls = [e for e in result["events"] if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "create_static_plot"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_since_invalid_format(tmp_path):
    """Bad timestamp returns validation error, not crash."""
    fn = _get_session_log()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.session_log.resolve_workspace_root",
            return_value=tmp_path,
        ),
        _patch_transcript_reader([]),
    ):
        result = await fn(since="not-a-timestamp")

    # make_error returns a dict, check it has error fields
    if isinstance(result, str):
        result = json.loads(result)
    assert result["error"] is True
    assert result["error_type"] == "validation_error"
    assert "since" in result["error_message"]
