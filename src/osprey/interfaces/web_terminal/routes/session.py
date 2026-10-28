"""Session diagnostics routes — agent hierarchy, logs, summaries, and chat.

Every session fact these routes answer is read from the agent record with this
terminal's login, which sees only its own person's sessions, and with a drain
naming the session, so a step the agent just took is in the answer. Each
answer carries a ``record`` field — ``complete``, ``incomplete`` or
``unavailable`` (with ``record_reason``) — and never shows an unread session
as an empty complete one. Only the lookup of which transcript is current
stays local.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from osprey.interfaces.web_terminal import transcript_map
from osprey.mcp_server import session as session_record

logger = logging.getLogger(__name__)

router = APIRouter()


def record_client():
    """The record client this process reads with, configured from its environment.

    Resolved per request and reached through the module, so a test replaces it
    with ``monkeypatch.setattr(session_routes, "record_client", ...)``.
    """
    return session_record.record_client()


def _identity(request: Request) -> str:
    """This terminal's identity: its configured user, else the acting identity."""
    from osprey_connectors.identity import acting_identity

    terminal_user = getattr(request.app.state, "terminal_user", "")
    if isinstance(terminal_user, str) and terminal_user.strip():
        return terminal_user.strip()
    return acting_identity()


def _unavailable(reason: str) -> dict[str, str]:
    return session_record.record_fields(session_record.RECORD_UNAVAILABLE, reason)


def _session_info(row: dict[str, Any]) -> dict[str, Any]:
    """A ``list_sessions`` row as the session picker's entry."""
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        prompt = ""
    return {
        "session_id": row["session_id"],
        "first_message": prompt[:200],
        "last_modified": row.get("last_ts"),
        "message_count": row.get("entries", 0),
    }


@router.get("/api/sessions")
async def list_sessions(request: Request):
    """Return this terminal's sessions, newest first, from the agent record."""
    identity = _identity(request)
    client = record_client()
    rows, answer = await asyncio.to_thread(session_record.own_sessions, identity, client=client)
    if not answer.available:
        return {"sessions": [], **_unavailable(answer.unavailable or "")}
    record = session_record.RECORD_COMPLETE if answer.complete else session_record.RECORD_INCOMPLETE
    return {
        "sessions": [_session_info(row) for row in rows],
        **session_record.record_fields(record, answer.reason),
    }


def _current_session(request: Request) -> str | None:
    """The session a route reads: the ``session_id`` key mapped to the transcript
    it names now, else this terminal's current transcript."""
    session_id = request.query_params.get("session_id")
    if session_id:
        return transcript_map.get(request.app, session_id)
    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

    path = TranscriptReader(request.app.state.project_cwd).find_current_transcript()
    return path.stem if path is not None else None


async def _read_session(request: Request, *, agent_id: str | None = None):
    """The session's timeline read (or its subagent's), or ``None`` when no session is known.

    An id that is not a record token names no session, so it reads as a
    missing one rather than as a record outage.
    """
    from osprey.agent_record.contract import is_token

    session_id = _current_session(request)
    if not session_id or not is_token(session_id):
        return None
    if agent_id is not None and not is_token(agent_id):
        return None
    client = record_client()
    return await asyncio.to_thread(
        session_record.read_timeline, session_id, agent_id=agent_id, client=client
    )


async def _read_session_events(request: Request) -> tuple[list[dict], dict[str, str]]:
    """The session's events and the ``record`` fields of the read behind them."""
    read = await _read_session(request)
    if read is None:
        return [], session_record.record_fields(session_record.RECORD_COMPLETE, "")
    events = session_record.session_events(read.nodes, read.session_id)
    return events, session_record.record_fields(read.record, read.reason)


@router.get("/api/session-agents")
async def session_agents(request: Request):
    """Return agent hierarchy and tool call breakdown for the current session."""
    try:
        from osprey.mcp_server.workspace.tools.session_log import _build_agent_summary

        events, record = await _read_session_events(request)
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
            **record,
        }
    except Exception as exc:
        logger.debug("session-agents: read failed", exc_info=True)
        return {
            "agents": [],
            "tool_calls_by_agent": {},
            "total_events": 0,
            **_unavailable(str(exc)),
        }


@router.get("/api/session-agent-timeline")
async def session_agent_timeline(request: Request):
    """Return the full internal timeline of a subagent."""
    agent_id = request.query_params.get("agent_id", "")
    if not agent_id:
        return {"timeline": [], "error": "agent_id required"}
    try:
        read = await _read_session(request, agent_id=agent_id)
        if read is None:
            return {
                "agent_id": agent_id,
                "timeline": [],
                "count": 0,
                "record": session_record.RECORD_COMPLETE,
            }
        timeline = session_record.agent_steps(read.nodes)
        return {
            "agent_id": agent_id,
            "timeline": timeline,
            "count": len(timeline),
            **session_record.record_fields(read.record, read.reason),
        }
    except Exception as exc:
        logger.debug("session-agent-timeline: failed", exc_info=True)
        return {"agent_id": agent_id, "timeline": [], "count": 0, **_unavailable(str(exc))}


@router.get("/api/session-log")
async def session_log(request: Request):
    """Return filtered tool call events from the current session."""
    try:
        events, record = await _read_session_events(request)

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
        if since or before:
            from osprey.mcp_server.workspace.tools.session_log import _parse_iso_timestamp

            since_at = _parse_iso_timestamp(since) if since else None
            before_at = _parse_iso_timestamp(before) if before else None

            def inside(event: dict) -> bool:
                at = _parse_iso_timestamp(event.get("timestamp", ""))
                if at is None:
                    return False
                if since_at is not None and at < since_at:
                    return False
                return before_at is None or at < before_at

            filtered = [e for e in filtered if inside(e)]

        total = len(filtered)
        filtered = filtered[-last_n:]

        return {"events": filtered, "total_events": total, "showing": len(filtered), **record}
    except Exception as exc:
        logger.debug("session-log: read failed", exc_info=True)
        return {"events": [], "total_events": 0, "showing": 0, **_unavailable(str(exc))}


@router.get("/api/session-summary")
async def session_summary(request: Request):
    """Return artifact inventory with channel extraction for the current session."""
    try:
        from osprey.mcp_server.workspace.tools.session_summary import _extract_channels
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(request.app.state.workspace_dir)
        session_id = request.query_params.get("session_id")
        entries = store.list_entries(
            session_filter=session_id if session_id else None,
            # Session material, not a shipped example.
            exclude_examples=True,
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


#: Values of the ``full`` flag that lift the per-message cap. The browser sends
#: ``1``; the other spellings are accepted so an operator calling the endpoint
#: by hand is not handed a silently shortened conversation.
_FULL_HISTORY_VALUES = frozenset({"1", "true", "yes", "on"})

#: How the CLI opens the user message it injects when a background agent reports.
_TASK_NOTIFICATION_MARK = "<task-notification>"


def _notification_field(text: str, tag: str) -> str:
    """The contents of ``<tag>…</tag>`` in a task notification, or an empty string."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _agent_done_event(text: str) -> dict[str, Any]:
    """The ``agent_done`` frame the live chat drew for this background agent's report."""
    status = _notification_field(text, "status")
    return {
        "type": "system",
        "subtype": "agent_done",
        "tool_use_id": _notification_field(text, "tool-use-id"),
        "status": "stopped" if status == "killed" else status,
        "summary": _notification_field(text, "summary"),
    }


def _wants_full_messages(request: Request) -> bool:
    """Whether the caller asked for whole messages via the ``full`` flag."""
    return request.query_params.get("full", "").strip().lower() in _FULL_HISTORY_VALUES


def _wants_events(request: Request) -> bool:
    """Whether the caller asked for agent turns as replayable chat events."""
    return request.query_params.get("events", "").strip().lower() in _FULL_HISTORY_VALUES


def _ts(value: Any) -> datetime | None:
    """A transcript timestamp as a datetime, or ``None`` when it is unreadable."""
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


#: The note the approval hook gives the model when a person approved a call and the call
#: then ran, or ran and failed. A copy of ``APPROVED_CALL_NOTE`` in the hook at
#: ``templates/claude_code/claude/hooks/osprey_approval.py``; a test pins the two together.
_APPROVED_CALL_NOTE = "The operator approved this call at the approval prompt before it ran."

#: The hook events that run once a call has executed. The approval note arrives on one.
_OUTCOME_HOOK_EVENTS = frozenset({"PostToolUse", "PostToolUseFailure"})

#: How the result of a declined call opens: the Simple view's answer
#: (``OperatorSession.resolve_permission``), and the CLI's rejection at its own prompt.
_DECLINED_RESULTS = (
    "The operator declined this tool call.",
    "The user doesn't want to proceed with this tool use.",
)


def _approved_call_ids(lines: list[str]) -> set[str]:
    """The ``tool_use_id`` of every call a person approved and the harness then ran.

    The approval hook answers the outcome event with the note, and the CLI records
    that answer as an ``attachment`` entry naming the call. Lines without the note's
    text are skipped before they are parsed.
    """
    approved: set[str] = set()
    for line in lines:
        if _APPROVED_CALL_NOTE not in line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        attachment = entry.get("attachment") if isinstance(entry, dict) else None
        if (
            not isinstance(attachment, dict)
            or attachment.get("hookEvent") not in _OUTCOME_HOOK_EVENTS
        ):
            continue
        tool_use_id = attachment.get("toolUseID")
        # The note sits in the hook's stdout and in the context the CLI made of it.
        carries_note = any(_APPROVED_CALL_NOTE in str(value) for value in attachment.values())
        if carries_note and isinstance(tool_use_id, str) and tool_use_id:
            approved.add(tool_use_id)
    return approved


def _replayed_decision(
    tool_use_id: str, text: str, is_error: bool, approved: set[str]
) -> str | None:
    """The decision a replayed call's row shows, or ``None`` when it shows none.

    ``allowed`` for a call the approval note names: a person allowed it, and it ran.
    The transcript cannot tell "Allowed for this session" from a one-off allow, so a
    session allow replays as ``allowed`` too. ``denied`` for an error result that is an
    operator's decline, from either view. A call a safety hook refused never reached the
    operator; its result opens with ``PreToolUse:<tool> hook error:``, so it shows no pill.
    """
    if tool_use_id in approved:
        return "allowed"
    if is_error and text.startswith(_DECLINED_RESULTS):
        return "denied"
    return None


def _operator_text(blocks: list[Any]) -> str:
    """The operator's own words in a user entry: its text blocks, less the CLI's markers.

    The CLI writes its bookkeeping as user text — the marker of an interrupted or
    rejected call, a slash command's echo and its captured output. Each is dropped
    block by block, so a marker beside a tool result or beside the operator's words
    costs neither, and the marker alone never makes a bubble.
    """
    from osprey.mcp_server.workspace.transcript_reader import (
        COMMAND_NAME_PREFIX,
        INTERRUPT_MARKER,
        LOCAL_COMMAND_PREFIXES,
    )

    markers = (INTERRUPT_MARKER, COMMAND_NAME_PREFIX, *LOCAL_COMMAND_PREFIXES)
    return "\n".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and not block["text"].lstrip().startswith(markers)
    ).strip()


def _transcript_nodes(path: Path) -> list[dict[str, Any]]:
    """The timeline nodes the agent record makes of the transcript at *path*.

    The record's own transcript parser and timeline builder, applied to one
    local file, so a transcript can be replayed exactly as the record would
    serve it.
    """
    from dataclasses import replace
    from datetime import UTC, timedelta

    from osprey.agent_record.ingest.transcripts import parse_line
    from osprey.agent_record.reader import RecordItem, build_timeline

    relpath = f"local/{path.name}"
    arrived = datetime.now(UTC)
    items: list[RecordItem] = []
    for index, line in enumerate(path.read_bytes().splitlines()):
        for entry in parse_line("local", relpath, line):
            # Arrival in file order keeps entries stamped alike in file order.
            arrival = arrived + timedelta(microseconds=index)
            items.append(RecordItem(replace(entry, inserted_at=arrival), None))
    items.sort(key=lambda i: (i.entry.ts, i.entry.inserted_at))
    return [node.to_json() for node in build_timeline(items)]


def _node_ts(node: Any) -> str:
    ts = node.get("ts") if isinstance(node, dict) else None
    return ts if isinstance(ts, str) else ""


def _node_body(node: Any) -> dict[str, Any]:
    body = node.get("body") if isinstance(node, dict) else None
    return body if isinstance(body, dict) else {}


def _chat_turns(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A session's main stream as the Simple chat's turns, agent turns carrying their events.

    A user turn is ``{role, content, timestamp}``. Everything the agent did up
    to the next user prompt is one assistant turn whose ``events`` are the
    frames a live stream would have carried — ``text``, ``thinking``,
    ``tool_use``, ``tool_result`` — closed by a ``result`` whose duration is
    read off the timestamps. ``content`` holds the turn's joined text, so a
    client that ignores ``events`` still replays the words. The results of one
    transcript line's calls follow that line, as they did live.

    A ``tool_result`` event carries its call's ``decision`` when the session
    shows one; see :func:`_replayed_decision`. It carries ``refusal`` exactly as
    the live stream does (:func:`~osprey.interfaces.web_terminal.tool_preview.tool_result_refusal`).
    """
    from osprey.interfaces.web_terminal.operator_session import format_tool_name
    from osprey.interfaces.web_terminal.routes.chat import clip_input
    from osprey.interfaces.web_terminal.tool_preview import (
        tool_result_preview,
        tool_result_refusal,
        tool_result_text,
    )

    main = [n for n in nodes if isinstance(n, dict) and not n.get("agent_id")]
    approved = _approved_call_ids(
        [
            raw
            for n in main
            if n.get("kind") == "debug" and isinstance(raw := _node_body(n).get("raw"), str)
        ]
    )

    turns: list[dict[str, Any]] = []
    agent: dict[str, Any] | None = None
    started: datetime | None = None
    last: datetime | None = None

    def close_agent() -> None:
        nonlocal agent
        if agent is None:
            return
        duration = (last - started).total_seconds() * 1000 if started and last else None
        agent["events"].append({"type": "result", "is_error": False, "duration_ms": duration})
        agent["content"] = "\n\n".join(e["content"] for e in agent["events"] if e["type"] == "text")
        turns.append(agent)
        agent = None

    def result_event(result: Mapping[str, Any], tool_use_id: str) -> dict[str, Any]:
        body = _node_body(result)
        text = tool_result_text(body.get("content"))
        is_error = bool(body.get("is_error"))
        # Read before the preview cut, as the live stream does.
        refusal, shown = tool_result_refusal(text, is_error)
        preview, truncated = tool_result_preview(shown)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "is_error": is_error,
            "preview": preview,
            "truncated": truncated,
            "refusal": refusal,
            "decision": _replayed_decision(tool_use_id, text, is_error, approved),
        }

    def open_agent(stamp_text: str) -> dict[str, Any]:
        nonlocal agent
        if agent is None:
            agent = {"role": "assistant", "content": "", "events": [], "timestamp": stamp_text}
        return agent

    for group in session_record.lines(main):
        prompts = [n for n in group if n.get("kind") == "prompt"]
        if prompts:
            bodies = [_node_body(n) for n in prompts]
            if any(b.get("is_meta") for b in bodies):
                continue
            blocks = [{"type": "text", "text": session_record.prompt_text(b)} for b in bodies]
            text = _operator_text(blocks)
            stamp_text = _node_ts(group[0])
            if text and not text.startswith(_TASK_NOTIFICATION_MARK):
                close_agent()
                turns.append({"role": "user", "content": text, "timestamp": stamp_text})
                started = _ts(stamp_text)
                continue
            if text.startswith(_TASK_NOTIFICATION_MARK):
                open_agent(stamp_text)["events"].append(_agent_done_event(text))
                if (stamp := _ts(stamp_text)) is not None:
                    last = stamp
                continue
        pending: list[dict[str, Any]] = []
        for node in group:
            kind = node.get("kind")
            body = _node_body(node)
            if kind not in ("answer", "thinking", "tool_call", "tool_result", "subagent"):
                continue
            if kind == "answer" and not str(body.get("text") or "").strip():
                continue
            current = open_agent(_node_ts(node))
            if (stamp := _ts(_node_ts(node))) is not None:
                last = stamp
            if kind == "answer":
                current["events"].append({"type": "text", "content": body["text"]})
            elif kind == "thinking":
                current["events"].append({"type": "thinking", "content": body.get("text", "")})
            elif kind == "tool_call":
                name = body.get("name", "")
                tool_use_id = body.get("id", "") or node.get("tool_use_id") or ""
                current["events"].append(
                    {
                        "type": "tool_use",
                        "tool_name": format_tool_name(name),
                        "tool_name_raw": name,
                        "tool_use_id": tool_use_id,
                        "input": clip_input(body.get("input", {})),
                    }
                )
                result = node.get("result")
                if isinstance(result, dict):
                    pending.append(result_event(result, tool_use_id))
                    if (stamp := _ts(_node_ts(result))) is not None:
                        last = max(last, stamp) if last else stamp
            else:
                tool_use_id = node.get("tool_use_id") or body.get("tool_use_id", "")
                current["events"].append(result_event(node, tool_use_id))
        if pending:
            open_agent("")["events"].extend(pending)
    close_agent()
    return turns


@router.get("/api/session-chat")
async def session_chat(request: Request):
    """Return conversation turns of a session, read from the agent record.

    ``session_id`` names a session key, and a key stops naming its transcript
    the moment a ``/clear`` replaces the conversation, so the key is mapped
    through :mod:`osprey.interfaces.web_terminal.transcript_map` first: a caller
    that knows only its key still reads the conversation that key is on.

    ``events=1`` answers the Simple chat's turns with their replayable events
    (:func:`_chat_turns`). ``full=1`` leaves each message whole, which is what
    replaying a conversation onto another surface needs. Without either, each
    message is cut at the diagnostics view's cap.
    """
    from osprey.mcp_server.workspace.transcript_reader import MAX_CHAT_MESSAGE_LENGTH

    try:
        read = await _read_session(request)
        if read is None:
            return {"turns": [], "count": 0, "record": session_record.RECORD_COMPLETE}
        if request.query_params.get("session_id") and _wants_events(request):
            turns = _chat_turns(read.nodes)
        elif request.query_params.get("session_id") and _wants_full_messages(request):
            turns = session_record.chat_history(read.nodes, max_chars=None)
        else:
            turns = session_record.chat_history(read.nodes, max_chars=MAX_CHAT_MESSAGE_LENGTH)
        return {
            "turns": turns,
            "count": len(turns),
            **session_record.record_fields(read.record, read.reason),
        }
    except Exception as exc:
        logger.debug("session-chat: read failed", exc_info=True)
        return {"turns": [], "count": 0, **_unavailable(str(exc))}
