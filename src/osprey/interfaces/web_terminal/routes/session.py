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
            session_filter=session_id or None,
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

#: How the result of a declined call opens: the chat's answer
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
    operator's decline, from either surface. A call a safety hook refused never reached the
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
        COMMAND_ECHO_PREFIXES,
        INTERRUPT_MARKER,
        LOCAL_COMMAND_PREFIXES,
    )

    markers = (INTERRUPT_MARKER, *COMMAND_ECHO_PREFIXES, *LOCAL_COMMAND_PREFIXES)
    return "\n".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and not block["text"].lstrip().startswith(markers)
    ).strip()


#: A slash command's echo, as the CLI writes it for a skill or a command file.
_COMMAND_NAME_RE = re.compile(r"<command-name>/?([^<\s]+)</command-name>")
_COMMAND_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)

#: How the CLI opens the instructions it injects when a skill is loaded.
_SKILL_BODY_MARK = "Base directory for this skill"


def _operator_command(blocks: list[Any]) -> tuple[str, str] | None:
    """The command and arguments an operator ran, from the CLI's echo of it.

    The CLI echoes a skill or a command file the operator ran as a user entry
    opening with ``<command-message>``, naming the command in ``<command-name>``
    and its arguments in ``<command-args>``. A built-in's echo opens with
    ``<command-name>`` and is not read here: it stays bookkeeping.

    Returns:
        ``(name, args)``, or ``None`` for an entry that is no such echo. ``args``
        is the echo's text as the CLI wrote it, whitespace and all, so the
        attached-files block the composer appends still reads as one.
    """
    from osprey.mcp_server.workspace.transcript_reader import COMMAND_MESSAGE_PREFIX

    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else None
        if not isinstance(text, str) or not text.lstrip().startswith(COMMAND_MESSAGE_PREFIX):
            continue
        name = _COMMAND_NAME_RE.search(text)
        if name is None:
            return None
        args = _COMMAND_ARGS_RE.search(text)
        return name.group(1), (args.group(1) if args else "")
    return None


#: The stop reasons of a message the agent goes on from: it waits on its tool calls,
#: or the API paused a long turn for the CLI to resume. Any other stop ends the response.
_CONTINUING_STOPS = frozenset({"tool_use", "pause_turn"})


def _stopped_by_operator(blocks: list[Any]) -> bool:
    """Whether a user entry carries the marker the CLI writes when the operator stops a turn."""
    from osprey.mcp_server.workspace.transcript_reader import INTERRUPT_MARKER

    return any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].lstrip().startswith(INTERRUPT_MARKER)
        for block in blocks
    )


def _api_error_event(error: Any) -> dict[str, Any] | None:
    """The ``error`` frame the live chat drew for an answer the model API failed.

    *error* is the kind of failure the CLI wrote on the entry (``authentication_failed``,
    ``rate_limit``, …), which the record keeps on the answer as ``api_error``. The SDK
    reads the same field, and the live stream draws it as this frame
    (``operator_session._event_to_wire``); a test pins the two together.
    Returns ``None`` for an answer that did not fail.
    """
    if not error:
        return None
    return {
        "type": "error",
        "message": f"API error: {error}",
        "error_type": "AssistantMessageError",
    }


#: The tools that start a subagent; the chat draws each call as that agent's chip.
_SUBAGENT_TOOLS = frozenset({"Agent", "Task"})


def _transcript_nodes(path: Path) -> list[dict[str, Any]]:
    """The timeline nodes the agent record makes of the session transcript at *path*.

    The record's own transcript parser and timeline builder, applied to one local
    session: its file and every subagent file beside it
    (``<session>/subagents/agent-<id>.jsonl``), at the paths the record reads them
    under, so a transcript can be replayed exactly as the record would serve it.
    """
    from dataclasses import replace
    from datetime import UTC, timedelta

    from osprey.agent_record.ingest.transcripts import parse_line
    from osprey.agent_record.reader import RecordItem, build_timeline

    files = [(f"local/{path.name}", path)]
    subagents = path.parent / path.stem / "subagents"
    files += [
        (f"local/{path.stem}/subagents/{sub.name}", sub)
        for sub in sorted(subagents.glob("*.jsonl"))
    ]
    arrived = datetime.now(UTC)
    items: list[RecordItem] = []
    index = 0
    for relpath, file in files:
        for line in file.read_bytes().splitlines():
            for entry in parse_line("local", relpath, line):
                # Arrival in reading order keeps entries stamped alike in file order.
                arrival = arrived + timedelta(microseconds=index)
                items.append(RecordItem(replace(entry, inserted_at=arrival), None))
            index += 1
    items.sort(key=lambda i: (i.entry.ts, i.entry.inserted_at))
    return [node.to_json() for node in build_timeline(items)]


def _node_ts(node: Any) -> str:
    ts = node.get("ts") if isinstance(node, dict) else None
    return ts if isinstance(ts, str) else ""


def _node_body(node: Any) -> dict[str, Any]:
    body = node.get("body") if isinstance(node, dict) else None
    return body if isinstance(body, dict) else {}


def _approved_in(nodes: list[dict[str, Any]]) -> set[str]:
    """The calls a person approved, from the hook attachments among *nodes*."""
    return _approved_call_ids(
        [
            raw
            for n in nodes
            if n.get("kind") == "debug" and isinstance(raw := _node_body(n).get("raw"), str)
        ]
    )


def _tool_use_event(node: Mapping[str, Any]) -> dict[str, Any]:
    """The ``tool_use`` frame of a ``tool_call`` node."""
    from osprey.interfaces.web_terminal.operator_session import format_tool_name
    from osprey.interfaces.web_terminal.routes.chat import clip_input

    body = _node_body(node)
    name = body.get("name", "")
    return {
        "type": "tool_use",
        "tool_name": format_tool_name(name),
        "tool_name_raw": name,
        "tool_use_id": body.get("id", "") or node.get("tool_use_id") or "",
        "input": clip_input(body.get("input", {})),
    }


def _tool_result_event(result: Mapping[str, Any], tool_use_id: str, approved: set[str]) -> dict:
    """The ``tool_result`` frame of a result node, read as the live stream reads it."""
    from osprey.interfaces.web_terminal.tool_preview import (
        tool_result_preview,
        tool_result_refusal,
        tool_result_text,
    )

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


def _subagent_events(timeline: list[dict[str, Any]], parent: str) -> list[dict[str, Any]]:
    """A subagent's text, calls and results as the frames the live chat drew in its chip.

    *timeline* is the subagent's own stream as the record nests it under the call
    that started it. Each frame carries ``parent_tool_use_id``, the way the live
    stream marks a subagent's own records, so the replay files it under that
    agent's chip. Its reasoning, and the prompt it was started with, are left
    out, as the chip never shows them.
    """
    approved = _approved_in(timeline)
    events: list[dict[str, Any]] = []
    for node in timeline:
        kind = node.get("kind")
        body = _node_body(node)
        if kind == "answer" and str(body.get("text") or "").strip():
            events.append({"type": "text", "content": body["text"]})
        elif kind == "tool_call":
            call = _tool_use_event(node)
            events.append(call)
            result = node.get("result")
            if isinstance(result, dict):
                events.append(_tool_result_event(result, call["tool_use_id"], approved))
        elif kind in ("tool_result", "subagent") and not node.get("orphan"):
            tool_use_id = node.get("tool_use_id") or body.get("tool_use_id", "")
            events.append(_tool_result_event(node, tool_use_id, approved))
    return [{**event, "parent_tool_use_id": parent} for event in events]


def _subagent_timelines(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each subagent call's own stream, by the ``tool_use_id`` of the call that started it.

    The record nests a subagent's stream under its call once the call's result
    names the agent. A subagent still at work when the session is read has no
    such result yet, so the record serves its stream as a group of its own; each
    such group is the stream of the earliest subagent call before it that has no
    result and no stream yet, the call that is waiting on it.
    """
    timelines: dict[str, list[dict[str, Any]]] = {}
    waiting: list[str] = []
    for node in nodes:
        sub = node.get("subagent")
        if node.get("orphan"):
            timeline = sub.get("timeline") if isinstance(sub, dict) else None
            if waiting and isinstance(timeline, list):
                timelines[waiting.pop(0)] = timeline
            continue
        if node.get("agent_id") or node.get("kind") != "tool_call":
            continue
        call = _tool_use_event(node)
        if call["tool_name_raw"] not in _SUBAGENT_TOOLS:
            continue
        if isinstance(sub, dict) and isinstance(sub.get("timeline"), list):
            timelines[call["tool_use_id"]] = sub["timeline"]
        elif not isinstance(node.get("result"), dict):
            waiting.append(call["tool_use_id"])
    return timelines


def _chat_turns(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A session's main stream as the chat's turns, agent turns carrying their events.

    A user turn is ``{role, content, timestamp}``. Everything the agent did up
    to the next user prompt is one assistant turn whose ``events`` are the
    frames a live stream would have carried — ``text``, ``thinking``,
    ``tool_use``, ``tool_result`` — closed by a ``result`` whose duration is
    read off the timestamps. ``content`` holds the turn's joined text, so a
    client that ignores ``events`` still replays the words. The results of one
    transcript line's calls follow that line, as they did live.

    The session's last turn gets its ``result`` only when the record shows that
    it ended: its last answer stopped (``end_turn``, or any stop other than one
    for tool calls) with no background agent of the turn still to report, or
    the operator stopped the turn. A turn that waits on a call, reads a result,
    waits on a background agent or answers one's report is still running, as a
    reload mid-turn finds it, so it is left open for the live stream to
    continue. An answer that records no stop reason gives no sign either way and
    is read as the end.

    An answer the model API failed (its ``api_error``) replays as the ``error``
    frame the live stream drew for it, ahead of its text. A turn whose last
    answer failed closes with ``is_error`` set.

    A ``thinking`` event carries ``duration_ms``: the time from the entry before
    it to the entry that holds the reasoning, which is how long the agent spent
    on it. It is left out when either timestamp is missing or unreadable.

    A ``tool_result`` event carries its call's ``decision`` when the session
    shows one; see :func:`_replayed_decision`. It carries ``refusal`` exactly as
    the live stream does (:func:`~osprey.interfaces.web_terminal.tool_preview.tool_result_refusal`).

    A subagent's work follows the call that started it, read from the stream the
    record holds for it (see :func:`_subagent_timelines`), so its chip replays
    with the activity it showed live. Those frames stay out of ``content``.

    A skill or a command file the operator ran by its slash command replays as
    the operator's turn, ``/name args`` as typed rather than the CLI's echo of
    it. A skill then opens the agent's turn with the row the live stream drew
    for it (:func:`~osprey.interfaces.web_terminal.slash_commands.operator_skill_frames`).
    """
    from osprey.interfaces.web_terminal.operator_session import _ASYNC_LAUNCH_MARK

    main = [n for n in nodes if isinstance(n, dict) and not n.get("agent_id")]
    approved = _approved_in(main)
    subagents = _subagent_timelines([n for n in nodes if isinstance(n, dict)])

    turns: list[dict[str, Any]] = []
    agent: dict[str, Any] | None = None
    started: datetime | None = None
    last: datetime | None = None
    #: The timestamp of the conversation entry before the one being read.
    previous: datetime | None = None
    #: Whether the record shows the current agent turn over, as its latest entry reads.
    ended = True
    #: Whether the current agent turn's latest answer is one the model API failed.
    failed = False
    #: The current turn's subagent calls, and those the CLI sent to the
    #: background that have not reported yet: the live turn stays open for them.
    agent_calls: set[str] = set()
    background: set[str] = set()

    def close_agent(*, at_end: bool = False) -> None:
        nonlocal agent, failed
        if agent is None:
            return
        # A turn the operator's next prompt follows is over. The session's last
        # turn is over only when its own entries say so.
        if not at_end or (ended and not background):
            duration = (last - started).total_seconds() * 1000 if started and last else None
            agent["events"].append({"type": "result", "is_error": failed, "duration_ms": duration})
        agent["content"] = "\n\n".join(
            e["content"]
            for e in agent["events"]
            if e["type"] == "text" and not e.get("parent_tool_use_id")
        )
        turns.append(agent)
        agent = None
        failed = False
        agent_calls.clear()
        background.clear()

    def open_agent(stamp_text: str) -> dict[str, Any]:
        nonlocal agent
        if agent is None:
            agent = {"role": "assistant", "content": "", "events": [], "timestamp": stamp_text}
        return agent

    def add_result(current: dict[str, Any], event: dict[str, Any]) -> None:
        current["events"].append(event)
        if event["tool_use_id"] in agent_calls and _ASYNC_LAUNCH_MARK in event["preview"]:
            # The CLI sent the subagent to the background: the turn waits on its report.
            background.add(event["tool_use_id"])

    #: The skill an operator's slash command named, until the CLI's entry that
    #: loads it shows it was a skill (a command file loads no such entry).
    ran: tuple[str, str] | None = None
    skills_run = 0
    for group in session_record.lines(main):
        kinds = {n.get("kind") for n in group}
        if kinds <= {"debug"}:
            continue  # the CLI's own records: nothing of the conversation
        stamp_text = _node_ts(group[0])
        prompts = [n for n in group if n.get("kind") == "prompt"]
        bodies = [_node_body(n) for n in prompts]
        blocks = [{"type": "text", "text": session_record.prompt_text(b)} for b in bodies]
        if any(b.get("is_meta") for b in bodies):
            # The instructions a skill injects are not shown; the skill's row is.
            if ran is not None and session_record.prompt_text(bodies[0]).startswith(
                _SKILL_BODY_MARK
            ):
                from osprey.interfaces.web_terminal.slash_commands import operator_skill_frames

                skills_run += 1
                open_agent(stamp_text)["events"].extend(
                    operator_skill_frames(*ran, f"operator-skill-{skills_run}")
                )
                # The agent has the skill's instructions and answers them next.
                ended = False
            ran = None
            continue
        ran = None
        stamp = _ts(stamp_text)
        since, previous = previous, stamp if stamp is not None else previous

        if prompts:
            command = _operator_command(blocks)
            if command is not None:
                from osprey.interfaces.web_terminal.slash_commands import strip_attachment_block

                # What the operator typed, as the live chat showed it.
                name, args = command
                close_agent()
                turns.append(
                    {
                        "role": "user",
                        "content": f"/{name} {args.strip()}".rstrip(),
                        "timestamp": stamp_text,
                    }
                )
                started = stamp
                # A skill's row, as the live one, leaves out the files attached to it.
                ran = (name, strip_attachment_block(args).strip())
                continue
            text = _operator_text(blocks)
            # A background agent's report is injected as a user message; it
            # continues the agent's turn rather than starting an operator one.
            if text and not text.startswith(_TASK_NOTIFICATION_MARK):
                close_agent()
                turns.append({"role": "user", "content": text, "timestamp": stamp_text})
                started = stamp
                continue
            if text.startswith(_TASK_NOTIFICATION_MARK):
                current = open_agent(stamp_text)
                if stamp is not None:
                    last = stamp
                # The agent answers the report next: the turn goes on.
                ended = False
                done = _agent_done_event(text)
                background.discard(done["tool_use_id"])
                current["events"].append(done)
                continue
            if _stopped_by_operator(blocks):
                # The operator stopped the turn, and its background agents with it.
                ended = True
                background.clear()
        conversation = [
            n
            for n in group
            if n.get("kind") in ("answer", "thinking", "tool_call", "tool_result", "subagent")
            and not (n.get("kind") == "answer" and not str(_node_body(n).get("text") or "").strip())
        ]
        if not conversation:
            continue
        current = open_agent(stamp_text)
        if stamp is not None:
            last = stamp
        answers = [n for n in conversation if n.get("kind") == "answer"]
        if answers or any(n.get("kind") in ("thinking", "tool_call") for n in conversation):
            stop = next(
                (b["stop_reason"] for n in answers if (b := _node_body(n)).get("stop_reason")),
                None,
            )
            calls = any(n.get("kind") == "tool_call" for n in conversation)
            # The agent reads its calls' results next; else its answer's stop decides.
            ended = not calls and stop not in _CONTINUING_STOPS
            error = _api_error_event(next((_node_body(n).get("api_error") for n in answers), None))
            failed = error is not None
            if error is not None:
                current["events"].append(error)
        else:
            # The agent reads the results next.
            ended = False
        pending: list[dict[str, Any]] = []
        for node in conversation:
            kind = node.get("kind")
            body = _node_body(node)
            if kind == "answer":
                current["events"].append({"type": "text", "content": body["text"]})
            elif kind == "thinking":
                event: dict[str, Any] = {"type": "thinking", "content": body.get("text", "")}
                if since is not None and stamp is not None:
                    event["duration_ms"] = max(0, round((stamp - since).total_seconds() * 1000))
                current["events"].append(event)
            elif kind == "tool_call":
                call = _tool_use_event(node)
                current["events"].append(call)
                tool_use_id = call["tool_use_id"]
                if call["tool_name_raw"] in _SUBAGENT_TOOLS:
                    agent_calls.add(tool_use_id)
                    if tool_use_id in subagents:
                        current["events"].extend(
                            _subagent_events(subagents[tool_use_id], tool_use_id)
                        )
                result = node.get("result")
                if isinstance(result, dict):
                    pending.append(_tool_result_event(result, tool_use_id, approved))
                    if (at := _ts(_node_ts(result))) is not None:
                        last = max(last, at) if last else at
                        # The entry that carried the result is the one before the next.
                        previous = max(previous, at) if previous else at
            else:
                tool_use_id = node.get("tool_use_id") or body.get("tool_use_id", "")
                add_result(current, _tool_result_event(node, tool_use_id, approved))
        for event in pending:
            add_result(current, event)
    close_agent(at_end=True)
    return turns


@router.get("/api/session-chat")
async def session_chat(request: Request):
    """Return conversation turns of a session, read from the agent record.

    ``session_id`` names a session key, and a key stops naming its transcript
    the moment a ``/clear`` replaces the conversation, so the key is mapped
    through :mod:`osprey.interfaces.web_terminal.transcript_map` first: a caller
    that knows only its key still reads the conversation that key is on.

    ``events=1`` answers the chat's turns with their replayable events
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
