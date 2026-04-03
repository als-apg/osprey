"""MCP tool: session_log.

On-demand diagnostic tool that reads Claude Code native transcripts to show
what tool calls subagents made, what errors occurred, and what
intermediate decisions were taken.
"""

import json
import logging
from datetime import datetime

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.mcp_server.workspace.transcript_reader import TranscriptReader
from osprey.utils.workspace import resolve_workspace_root

logger = logging.getLogger("osprey.mcp_server.tools.session_log")

MAX_LAST_N = 200


def _build_agent_windows(events: list[dict], agent_type: str) -> list[tuple[str, str, str]]:
    """Build (agent_id, start_ts, stop_ts) windows for a given agent_type.

    If an agent_start has no corresponding agent_stop, the window extends
    to the end of time (empty stop_ts).
    """
    # Collect starts and stops keyed by agent_id
    starts: dict[str, str] = {}
    stops: dict[str, str] = {}
    for ev in events:
        if ev.get("agent_type") != agent_type:
            continue
        aid = ev.get("agent_id", "")
        if ev["type"] == "agent_start":
            starts[aid] = ev.get("timestamp", "")
        elif ev["type"] == "agent_stop":
            stops[aid] = ev.get("timestamp", "")

    windows = []
    for aid, start_ts in starts.items():
        stop_ts = stops.get(aid, "")
        windows.append((aid, start_ts, stop_ts))
    return windows


def _event_in_windows(event: dict, windows: list[tuple[str, str, str]]) -> bool:
    """Check if a tool_call event's timestamp falls within any agent window."""
    ts = event.get("timestamp", "")
    if not ts:
        return False
    for _aid, start_ts, stop_ts in windows:
        if start_ts and ts >= start_ts:
            if not stop_ts or ts <= stop_ts:
                return True
    return False


def _parse_iso_timestamp(value: str) -> datetime | None:
    """Parse an ISO 8601 timestamp string, returning None on failure."""
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _build_agent_summary(events: list[dict]) -> list[dict]:
    """Build a summary list of agents with tool/error counts."""
    # Collect agent info from lifecycle events
    agents: dict[str, dict] = {}
    for ev in events:
        etype = ev.get("type", "")
        aid = ev.get("agent_id", "")
        if not aid:
            continue
        if etype == "agent_start":
            if aid not in agents:
                agents[aid] = {
                    "agent_id": aid,
                    "agent_type": ev.get("agent_type", ""),
                    "start_time": ev.get("timestamp"),
                    "stop_time": None,
                    "agent_transcript_path": None,
                    "tool_count": 0,
                    "error_count": 0,
                }
        elif etype == "agent_stop":
            if aid not in agents:
                agents[aid] = {
                    "agent_id": aid,
                    "agent_type": ev.get("agent_type", ""),
                    "start_time": None,
                    "stop_time": ev.get("timestamp"),
                    "agent_transcript_path": ev.get("agent_transcript_path"),
                    "tool_count": 0,
                    "error_count": 0,
                }
            else:
                agents[aid]["stop_time"] = ev.get("timestamp")
                agents[aid]["agent_transcript_path"] = ev.get("agent_transcript_path")

    # Count tools per agent using agent_id set by TranscriptReader.
    # Fallback: time-window matching for older data without agent_id on tool events.
    agent_windows: dict[str, tuple[str, str]] = {}
    for aid, info in agents.items():
        if info.get("start_time"):
            agent_windows[aid] = (info["start_time"], info.get("stop_time") or "")

    for ev in events:
        if ev.get("type") != "tool_call":
            continue
        matched_aid = None

        # Primary: direct agent_id on the event
        ev_aid = ev.get("agent_id")
        if ev_aid and ev_aid in agents:
            matched_aid = ev_aid
        else:
            # Fallback: time-window matching
            ts = ev.get("timestamp", "")
            if ts:
                for aid, (start_ts, stop_ts) in agent_windows.items():
                    if start_ts and ts >= start_ts:
                        if not stop_ts or ts <= stop_ts:
                            matched_aid = aid
                            break

        if matched_aid and matched_aid in agents:
            agents[matched_aid]["tool_count"] += 1
            if ev.get("is_error"):
                agents[matched_aid]["error_count"] += 1

    # Return sorted by start_time
    result = list(agents.values())
    # Remove internal field
    for a in result:
        a.pop("agent_transcript_path", None)
    result.sort(key=lambda a: a.get("start_time") or "")
    return result


@mcp.tool()
async def session_log(
    agent: str | None = None,
    agent_id: str | None = None,
    tool: str | None = None,
    last_n: int = 50,
    errors_only: bool = False,
    event_type: str | None = None,
    since: str | None = None,
    before: str | None = None,
    list_agents: bool = False,
) -> str:
    """Query the session log for tool calls and subagent lifecycle events.

    Reads Claude Code native transcripts — no separate audit hook needed.
    Use this to debug subagent behavior — see what tools were called,
    what errors occurred, and what decisions were made.

    Args:
        agent: Filter by agent_type (e.g. "data-visualizer"). Returns tool
            calls attributed to that agent type plus its lifecycle events.
        agent_id: Filter by specific agent instance ID (e.g. "agent-abc").
            More precise than agent — disambiguates concurrent agents of the same type.
        tool: Filter by tool name substring (e.g. "create_dashboard").
        last_n: Return the last N events (default 50, max 200).
        errors_only: Only return tool_call events with is_error=true.
        event_type: Filter by event type: "tool_call", "agent_start", or "agent_stop".
        since: ISO timestamp lower bound — exclude events before this time.
        before: ISO timestamp upper bound — exclude events after this time.
        list_agents: Return agent summary instead of events. Shows each agent with
            tool_count and error_count. Ignores tool/errors_only/event_type/last_n.

    Examples:
        session_log(list_agents=True)                    — agent overview
        session_log(agent_id="agent-abc")                — specific agent's calls
        session_log(errors_only=True, last_n=10)         — recent errors
        session_log(since="2026-02-19T12:00:00+00:00")   — events after noon

    Returns:
        JSON with events list and metadata, or agent summary when list_agents=True.
    """
    try:
        workspace_root = resolve_workspace_root()
    except Exception as e:
        return make_error(
            "internal_error",
            f"Could not resolve workspace root: {e}",
        )

    # Read events from Claude Code native transcripts
    project_dir = workspace_root.parent  # workspace_root = project_dir/_agent_data
    reader = TranscriptReader(project_dir)
    all_events = reader.read_current_session()  # already sorted by timestamp

    # Validate since/before format
    if since and _parse_iso_timestamp(since) is None:
        return make_error(
            "validation_error",
            f"Invalid 'since' timestamp: {since!r}. Expected ISO 8601 format.",
        )
    if before and _parse_iso_timestamp(before) is None:
        return make_error(
            "validation_error",
            f"Invalid 'before' timestamp: {before!r}. Expected ISO 8601 format.",
        )

    # Apply since/before early to reduce working set
    if since or before:
        time_filtered: list[dict] = []
        for ev in all_events:
            ts = ev.get("timestamp", "")
            if not ts:
                continue
            if since and ts < since:
                continue
            if before and ts > before:
                continue
            time_filtered.append(ev)
        all_events = time_filtered

    # list_agents mode — return summary
    if list_agents:
        agents = _build_agent_summary(all_events)
        filters_applied: dict = {"list_agents": True}
        if since:
            filters_applied["since"] = since
        if before:
            filters_applied["before"] = before
        return json.dumps(
            {
                "agents": agents,
                "total_agents": len(agents),
                "filters_applied": filters_applied,
            },
            indent=2,
        )

    # Clamp last_n
    last_n = min(max(last_n, 1), MAX_LAST_N)

    # Resolve target agent type for filtering
    target_agent_type = agent

    if agent_id and not target_agent_type:
        # Infer agent_type from lifecycle events
        for ev in all_events:
            if ev.get("agent_id") == agent_id and ev.get("agent_type"):
                target_agent_type = ev["agent_type"]
                break

    # Build set of agent_ids matching the target_agent_type (for agent filter)
    target_agent_ids: set[str] | None = None
    if target_agent_type and not agent_id:
        target_agent_ids = {
            ev.get("agent_id", "")
            for ev in all_events
            if ev.get("type") in ("agent_start", "agent_stop")
            and ev.get("agent_type") == target_agent_type
            and ev.get("agent_id")
        }

    # Build fallback time windows for agent_type filtering when tool_call
    # events don't carry agent_id (e.g., older data)
    agent_windows = None
    if target_agent_type or agent_id:
        has_agent_id_on_tools = any(
            ev.get("agent_id") for ev in all_events if ev.get("type") == "tool_call"
        )
        if not has_agent_id_on_tools:
            if target_agent_type:
                agent_windows = _build_agent_windows(all_events, target_agent_type)
            if agent_id and agent_windows:
                agent_windows = [(aid, s, e) for aid, s, e in agent_windows if aid == agent_id]

    # Apply filters
    filtered: list[dict] = []
    for ev in all_events:
        # event_type filter
        if event_type and ev.get("type") != event_type:
            continue

        # errors_only filter (only applies to tool_call)
        if errors_only and (ev.get("type") != "tool_call" or not ev.get("is_error")):
            continue

        # tool name substring filter
        if tool:
            ev_tool = ev.get("tool", "") or ev.get("full_tool_name", "")
            if tool.lower() not in ev_tool.lower():
                continue

        # agent/agent_id filter
        if target_agent_type or agent_id:
            ev_type = ev.get("type", "")
            if ev_type in ("agent_start", "agent_stop"):
                if target_agent_type and ev.get("agent_type") != target_agent_type:
                    continue
                if agent_id and ev.get("agent_id") != agent_id:
                    continue
            elif ev_type == "tool_call":
                matched = False
                ev_aid = ev.get("agent_id")
                if ev_aid:
                    # Direct agent_id match (TranscriptReader sets this)
                    if agent_id:
                        matched = ev_aid == agent_id
                    elif target_agent_ids:
                        matched = ev_aid in target_agent_ids
                elif agent_windows is not None:
                    # Fallback to time-window matching for older data
                    matched = _event_in_windows(ev, agent_windows)
                if not matched:
                    continue
            else:
                continue

        filtered.append(ev)

    total = len(filtered)
    showing = filtered[-last_n:]

    filters_applied = {}
    if agent:
        filters_applied["agent"] = agent
    if agent_id:
        filters_applied["agent_id"] = agent_id
    if tool:
        filters_applied["tool"] = tool
    if errors_only:
        filters_applied["errors_only"] = True
    if event_type:
        filters_applied["event_type"] = event_type
    if since:
        filters_applied["since"] = since
    if before:
        filters_applied["before"] = before
    filters_applied["last_n"] = last_n

    return json.dumps(
        {
            "events": showing,
            "total_events": total,
            "showing": len(showing),
            "filters_applied": filters_applied,
        },
        indent=2,
    )
