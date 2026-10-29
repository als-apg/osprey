"""Session diagnostics routes — agent hierarchy, logs, summaries, and chat."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from osprey.interfaces.web_terminal import transcript_map
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

if TYPE_CHECKING:
    from pathlib import Path

    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions(request: Request):
    """Return Claude Code sessions registered for this project."""
    discovery = SessionDiscovery(request.app.state.project_cwd)
    sessions = discovery.list_sessions()
    return {"sessions": [asdict(s) for s in sessions]}


def _read_session_events(request: Request, reader: TranscriptReader) -> list[dict]:
    """Read session events, scoped to a session_id if provided.

    A session key is mapped to the transcript it points at now, as
    ``session_chat`` does.
    """
    session_id = request.query_params.get("session_id")
    if session_id:
        return reader.read_session_by_id(transcript_map.get(request.app, session_id))
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
        if session_id:
            session_id = transcript_map.get(request.app, session_id)
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


def _entry_text(blocks: list[Any]) -> str:
    """The text of an entry's first text block, or ``""``."""
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else None
        if isinstance(text, str):
            return text
    return ""


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


def _api_error_event(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The ``error`` frame the live chat drew for an assistant entry the model API failed.

    The CLI writes such an entry with ``isApiErrorMessage`` and the kind of
    failure in ``error`` (``authentication_failed``, ``rate_limit``, …). The SDK
    reads the same field, and the live stream draws it as this frame
    (``operator_session._event_to_wire``); a test pins the two together.
    Returns ``None`` for any other entry.
    """
    error = entry.get("error")
    if not error:
        return None
    return {
        "type": "error",
        "message": f"API error: {error}",
        "error_type": "AssistantMessageError",
    }


#: The tools that start a subagent; the chat draws each call as that agent's chip.
_SUBAGENT_TOOLS = frozenset({"Agent", "Task"})


def _block_events(
    blocks: list[Any], *, assistant: bool, approved: set[str]
) -> list[dict[str, Any]]:
    """The chat frames one transcript entry's content blocks stand for.

    Text counts only from the agent (``assistant``): a user entry's text is the
    operator's, or the prompt a subagent was started with, and neither is the
    agent speaking.
    """
    from osprey.interfaces.web_terminal.operator_session import format_tool_name
    from osprey.interfaces.web_terminal.routes.chat import clip_input
    from osprey.interfaces.web_terminal.tool_preview import (
        tool_result_preview,
        tool_result_refusal,
        tool_result_text,
    )

    events: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and assistant and block.get("text", "").strip():
            events.append({"type": "text", "content": block["text"]})
        elif btype == "thinking":
            events.append({"type": "thinking", "content": block.get("thinking", "")})
        elif btype == "tool_use":
            name = block.get("name", "")
            events.append(
                {
                    "type": "tool_use",
                    "tool_name": format_tool_name(name),
                    "tool_name_raw": name,
                    "tool_use_id": block.get("id", ""),
                    "input": clip_input(block.get("input", {})),
                }
            )
        elif btype == "tool_result":
            result_text = tool_result_text(block.get("content"))
            # Read before the preview cut, as the live stream does.
            refusal, shown = tool_result_refusal(result_text, bool(block.get("is_error")))
            preview, truncated = tool_result_preview(shown)
            events.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id", ""),
                    "is_error": bool(block.get("is_error")),
                    "preview": preview,
                    "truncated": truncated,
                    "refusal": refusal,
                    "decision": _replayed_decision(
                        block.get("tool_use_id", ""),
                        result_text,
                        bool(block.get("is_error")),
                        approved,
                    ),
                }
            )
    return events


def _subagent_transcripts(path: Path) -> dict[str, Path]:
    """Each subagent transcript of the session at ``path``, by the call that started it.

    The CLI writes a subagent's work to ``<session>/subagents/agent-<id>.jsonl``,
    beside a ``.meta.json`` naming the ``toolUseId`` of the Agent call that
    launched it. A meta file that cannot be read, or whose transcript is
    missing, names nothing.
    """
    found: dict[str, Path] = {}
    for meta in (path.parent / path.stem / "subagents").glob("*.meta.json"):
        try:
            tool_use_id = json.loads(meta.read_text()).get("toolUseId")
        except (OSError, ValueError, AttributeError):
            continue
        transcript = meta.with_name(meta.name.removesuffix(".meta.json") + ".jsonl")
        if isinstance(tool_use_id, str) and tool_use_id and transcript.is_file():
            found[tool_use_id] = transcript
    return found


def _subagent_events(path: Path, parent: str) -> list[dict[str, Any]]:
    """A subagent's text, calls and results as the frames the live chat drew in its chip.

    Each frame carries ``parent_tool_use_id``, the way the live stream marks a
    subagent's own records, so the replay files it under that agent's chip. Its
    reasoning is left out, as the chip never shows it.
    """
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    approved = _approved_call_ids(lines)
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        kind = entry.get("type")
        if kind not in ("user", "assistant") or entry.get("isMeta"):
            continue
        blocks = entry.get("message", {}).get("content", [])
        if not isinstance(blocks, list):
            continue  # the prompt the agent was started with
        events.extend(
            {**event, "parent_tool_use_id": parent}
            for event in _block_events(blocks, assistant=kind == "assistant", approved=approved)
            if event["type"] != "thinking"
        )
    return events


def _chat_timeline(path: Path) -> list[dict[str, Any]]:
    """Read a transcript as the Simple chat's turns, agent turns carrying their events.

    A user turn is ``{role, content, timestamp}``. Everything the agent did up
    to the next user prompt is one assistant turn whose ``events`` are the
    frames a live stream would have carried — ``text``, ``thinking``,
    ``tool_use``, ``tool_result`` — closed by a ``result`` whose duration is
    read off the transcript's timestamps. ``content`` holds the turn's joined
    text, so a client that ignores ``events`` still replays the words.

    The transcript's last turn gets its ``result`` only when the transcript
    shows that it ended: its last answer stopped (``end_turn``, or any stop
    other than one for tool calls) with no background agent of the turn still
    to report, or the operator stopped the turn. A turn that waits on a call,
    reads a result, waits on a background agent or answers one's report is
    still running, as a reload mid-turn finds it, so it is left open for the
    live stream to continue. An answer that records no stop reason gives no
    sign either way and is read as the end.

    An answer the model API failed (CLI ``isApiErrorMessage``) replays as the
    ``error`` frame the live stream drew for it, ahead of its text. A turn whose
    last answer failed closes with ``is_error`` set.

    A ``thinking`` event carries ``duration_ms``: the time from the entry before
    it to the entry that holds the reasoning, which is how long the agent spent
    on it. It is left out when either timestamp is missing or unreadable.

    A ``tool_result`` event carries its call's ``decision`` when the transcript
    shows one; see :func:`_replayed_decision`. It carries ``refusal`` exactly as
    the live stream does (:func:`~osprey.interfaces.web_terminal.tool_preview.tool_result_refusal`).

    A subagent's work follows the call that started it, read from its own
    transcript (see :func:`_subagent_events`), so its chip replays with the
    activity it showed live. Those frames stay out of ``content``.

    A skill or a command file the operator ran by its slash command replays as
    the operator's turn, ``/name args`` as typed rather than the CLI's echo of
    it. A skill then opens the agent's turn with the row the live stream drew
    for it (:func:`~osprey.interfaces.web_terminal.slash_commands.operator_skill_frames`).
    """
    from osprey.interfaces.web_terminal.operator_session import _ASYNC_LAUNCH_MARK

    turns: list[dict[str, Any]] = []
    agent: dict[str, Any] | None = None
    started: datetime | None = None
    last: datetime | None = None
    #: The timestamp of the conversation entry before the one being read.
    previous: datetime | None = None

    #: Whether the transcript shows the current agent turn over, as its latest entry reads.
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
        # A turn the operator's next prompt follows is over. The transcript's
        # last turn is over only when its own entries say so.
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

    #: The skill an operator's slash command named, until the CLI's entry that
    #: loads it shows it was a skill (a command file loads no such entry).
    ran: tuple[str, str] | None = None
    skills_run = 0
    lines = path.read_text().splitlines()
    approved = _approved_call_ids(lines)
    subagents = _subagent_transcripts(path)
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        kind = entry.get("type")
        if kind not in ("user", "assistant") or entry.get("isSidechain"):
            continue
        blocks = entry.get("message", {}).get("content", [])
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        if entry.get("isMeta"):
            # The instructions a skill injects are not shown; the skill's row is.
            if ran is not None and _entry_text(blocks).startswith(_SKILL_BODY_MARK):
                from osprey.interfaces.web_terminal.slash_commands import operator_skill_frames

                skills_run += 1
                if agent is None:
                    agent = {
                        "role": "assistant",
                        "content": "",
                        "events": [],
                        "timestamp": entry.get("timestamp", ""),
                    }
                agent["events"].extend(operator_skill_frames(*ran, f"operator-skill-{skills_run}"))
                # The agent has the skill's instructions and answers them next.
                ended = False
            ran = None
            continue
        ran = None
        stamp = _ts(entry.get("timestamp", ""))
        since, previous = previous, stamp if stamp is not None else previous

        text = ""
        stopped = False
        if kind == "user":
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
                        "timestamp": entry.get("timestamp", ""),
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
                turns.append(
                    {"role": "user", "content": text, "timestamp": entry.get("timestamp", "")}
                )
                started = stamp
                continue
            # The operator stopped the turn, and its background agents with it:
            # whatever came before the marker, the turn is over.
            stopped = _stopped_by_operator(blocks)
            if stopped:
                ended = True
                background.clear()
            if not text and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks
            ):
                continue  # the CLI's own bookkeeping: nothing of the conversation

        if agent is None:
            agent = {
                "role": "assistant",
                "content": "",
                "events": [],
                "timestamp": entry.get("timestamp", ""),
            }
        if stamp is not None:
            last = stamp
        if kind == "user" and text.startswith(_TASK_NOTIFICATION_MARK):
            # The agent answers the report next: the turn goes on.
            ended = False
            done = _agent_done_event(text)
            background.discard(done["tool_use_id"])
            agent["events"].append(done)
            continue
        if kind == "assistant":
            ended = entry.get("message", {}).get("stop_reason") not in _CONTINUING_STOPS
            error = _api_error_event(entry)
            failed = error is not None
            if error is not None:
                agent["events"].append(error)
        else:
            # The agent reads its calls' results next, unless the operator stopped it.
            ended = stopped
        for event in _block_events(blocks, assistant=kind == "assistant", approved=approved):
            if event["type"] == "thinking" and since is not None and stamp is not None:
                event["duration_ms"] = max(0, round((stamp - since).total_seconds() * 1000))
            agent["events"].append(event)
            if event["type"] == "tool_use" and event["tool_name_raw"] in _SUBAGENT_TOOLS:
                agent_calls.add(event["tool_use_id"])
                transcript = subagents.get(event["tool_use_id"])
                if transcript is not None:
                    agent["events"].extend(_subagent_events(transcript, event["tool_use_id"]))
            elif (
                event["type"] == "tool_result"
                and event["tool_use_id"] in agent_calls
                and _ASYNC_LAUNCH_MARK in event["preview"]
            ):
                # The CLI sent the subagent to the background: the turn waits on its report.
                background.add(event["tool_use_id"])
    close_agent(at_end=True)
    return turns


@router.get("/api/session-chat")
async def session_chat(request: Request):
    """Return conversation turns from a session's transcript.

    ``session_id`` names a session key, and a key stops naming its transcript
    the moment a ``/clear`` replaces the conversation, so the key is mapped
    through :mod:`osprey.interfaces.web_terminal.transcript_map` first: a caller
    that knows only its key still reads the conversation that key is on.

    ``full=1`` leaves each message whole, which is what replaying a conversation
    onto another surface needs. Without it the reader's own per-message cap
    stands, which is what the diagnostics view has always read.
    """
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        session_id = request.query_params.get("session_id")
        if session_id:
            transcript_id = transcript_map.get(request.app, session_id)
            if _wants_events(request):
                path = reader.find_transcript_by_id(transcript_id)
                turns = _chat_timeline(path) if path else []
            elif _wants_full_messages(request):
                turns = reader.read_chat_history_by_id(transcript_id, max_chars=None)
            else:
                turns = reader.read_chat_history_by_id(transcript_id)
        else:
            turns = reader.read_current_chat_history()
        return {"turns": turns, "count": len(turns)}
    except Exception:
        logger.debug("session-chat: no transcript available", exc_info=True)
        return {"turns": [], "count": 0}
