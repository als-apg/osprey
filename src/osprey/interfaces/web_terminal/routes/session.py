"""Session diagnostics routes — agent hierarchy, logs, summaries, and chat."""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, Request

from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions(request: Request):
    """Return Claude Code sessions registered for this project."""
    discovery = SessionDiscovery(request.app.state.project_cwd)
    sessions = discovery.list_sessions()
    return {"sessions": [asdict(s) for s in sessions]}


@router.get("/api/session-server")
async def session_server_config(request: Request):
    """Return the session diagnostics page URL for iframe embedding."""
    base = str(request.base_url).rstrip("/")
    return {"url": f"{base}/static/session.html"}


def _read_session_events(request: Request, reader) -> list[dict]:
    """Read session events, scoped to a session_id if provided."""
    session_id = request.query_params.get("session_id")
    if session_id:
        return reader.read_session_by_id(session_id)
    return reader.read_current_session()


@router.get("/api/session-agents")
async def session_agents(request: Request):
    """Return agent hierarchy and tool call breakdown for the current session."""
    try:
        from osprey.mcp_server.workspace.tools.session_log import _build_agent_summary
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        events = _read_session_events(request, reader)
        agents = _build_agent_summary(events)

        # Group tool_call events by agent_id
        tool_calls_by_agent: dict[str, list[dict]] = {}
        for ev in events:
            if ev.get("type") != "tool_call":
                continue
            aid = ev.get("agent_id") or "main"
            tool_calls_by_agent.setdefault(aid, []).append(ev)

        return {
            "agents": agents,
            "tool_calls_by_agent": tool_calls_by_agent,
            "total_events": len(events),
        }
    except Exception:
        logger.debug("session-agents: no transcript available", exc_info=True)
        return {"agents": [], "tool_calls_by_agent": {}, "total_events": 0}


@router.get("/api/session-agent-timeline")
async def session_agent_timeline(request: Request):
    """Return the full internal timeline of a subagent."""
    agent_id = request.query_params.get("agent_id", "")
    if not agent_id:
        return {"timeline": [], "error": "agent_id required"}
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        session_id = request.query_params.get("session_id")
        timeline = reader.read_agent_timeline(agent_id, session_id=session_id)
        return {"agent_id": agent_id, "timeline": timeline, "count": len(timeline)}
    except Exception:
        logger.debug("session-agent-timeline: failed", exc_info=True)
        return {"agent_id": agent_id, "timeline": [], "count": 0}


@router.get("/api/session-log")
async def session_log(request: Request):
    """Return filtered tool call events from the current session."""
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        events = _read_session_events(request, reader)

        # Parse query filters
        agent_filter = request.query_params.get("agent")
        agent_id_filter = request.query_params.get("agent_id")
        tool_filter = request.query_params.get("tool")
        errors_only = request.query_params.get("errors_only", "").lower() == "true"
        since = request.query_params.get("since")
        before = request.query_params.get("before")
        last_n = min(int(request.query_params.get("last_n", "100")), 500)

        # Filter to tool_call events
        filtered = [e for e in events if e.get("type") == "tool_call"]

        if agent_filter:
            filtered = [e for e in filtered if e.get("agent_type") == agent_filter]
        if agent_id_filter:
            filtered = [e for e in filtered if (e.get("agent_id") or "main") == agent_id_filter]
        if tool_filter:
            filtered = [
                e
                for e in filtered
                if tool_filter.lower() in (e.get("tool_name") or e.get("tool") or "").lower()
            ]
        if errors_only:
            filtered = [e for e in filtered if e.get("is_error")]
        if since:
            filtered = [e for e in filtered if e.get("timestamp", "") >= since]
        if before:
            filtered = [e for e in filtered if e.get("timestamp", "") < before]

        total = len(filtered)
        filtered = filtered[-last_n:]

        return {"events": filtered, "total_events": total, "showing": len(filtered)}
    except Exception:
        logger.debug("session-log: no transcript available", exc_info=True)
        return {"events": [], "total_events": 0, "showing": 0}


@router.get("/api/session-summary")
async def session_summary(request: Request):
    """Return artifact inventory with channel extraction for the current session."""
    try:
        from osprey.stores.artifact_store import ArtifactStore
        from osprey.mcp_server.workspace.tools.session_summary import _extract_channels

        store = ArtifactStore(request.app.state.workspace_dir)
        session_id = request.query_params.get("session_id")
        entries = store.list_entries(
            session_filter=session_id if session_id else None,
        )

        total_bytes = 0
        categories: dict[str, int] = {}
        tools_used: set[str] = set()
        artifact_types: dict[str, int] = {}
        entry_dicts = []

        for entry in entries:
            total_bytes += entry.size_bytes
            cat = entry.category or "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1
            if entry.tool_source:
                tools_used.add(entry.tool_source)
            artifact_types[entry.artifact_type] = artifact_types.get(entry.artifact_type, 0) + 1
            d = entry.to_dict()
            d["channels"] = _extract_channels(entry)
            entry_dicts.append(d)

        return {
            "entries": entry_dicts,
            "totals": {
                "entry_count": len(entries),
                "total_bytes": total_bytes,
                "categories": categories,
                "tools_used": sorted(tools_used),
                "artifact_types": artifact_types,
            },
        }
    except Exception:
        logger.debug("session-summary: no artifacts available", exc_info=True)
        return {
            "entries": [],
            "totals": {
                "entry_count": 0,
                "total_bytes": 0,
                "categories": {},
                "tools_used": [],
                "artifact_types": {},
            },
        }


@router.get("/api/session-chat")
async def session_chat(request: Request):
    """Return conversation turns from the current session transcript."""
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        session_id = request.query_params.get("session_id")
        if session_id:
            turns = reader.read_chat_history_by_id(session_id)
        else:
            turns = reader.read_current_chat_history()
        return {"turns": turns, "count": len(turns)}
    except Exception:
        logger.debug("session-chat: no transcript available", exc_info=True)
        return {"turns": [], "count": 0}
