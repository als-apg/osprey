"""The current session: its metadata, and its facts read from the agent record.

Every fact about a session — its tool calls, subagents, conversation and start
— is read from the agent record through
:class:`~osprey.agent_record.client.RecordClient`, with a drain naming the
session so a step the agent just took is in the answer. The login decides the
scope: a terminal or host session reads its own person's sessions; inside a
dispatch worker (``OSPREY_DISPATCH_RUN=1``) the read is narrowed to the run's
session. Only the lookup of which transcript is current stays local.

The converters here turn timeline nodes (the ``timeline`` read's
``TimelineNode.to_json`` shape) into the event and turn shapes the session
readers answer:

* :func:`session_events` — ``tool_call`` events for MCP tool calls and an
  ``agent_start``/``agent_stop`` pair around every subagent's own events;
* :func:`chat_history` — conversation turns ``{role, content, timestamp}``;
* :func:`agent_steps` — a subagent's steps (``prompt``, ``reasoning``,
  ``thinking``, ``tool_call``, ``tool_result``).

A read answers a :class:`SessionRead` whose ``record`` is ``complete``,
``incomplete`` or ``unavailable``; a caller never shows an unavailable or
incomplete read as a complete empty one.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.mcp_server.workspace.transcript_reader import MAX_CHAT_MESSAGE_LENGTH
from osprey_connectors.identity import acting_identity

logger = logging.getLogger("osprey.mcp_server.session")

#: The ``record`` field of a read: the record answered whole, answered only
#: part, or did not answer at all.
RECORD_COMPLETE = "complete"
RECORD_INCOMPLETE = "incomplete"
RECORD_UNAVAILABLE = "unavailable"

#: Most timeline pages one read follows before it reports the rest missing.
MAX_TIMELINE_PAGES = 100

#: Set to ``1`` in every process a dispatch worker runs a request in.
DISPATCH_RUN_ENV = "OSPREY_DISPATCH_RUN"

_SUBAGENT_TOOLS = ("Agent", "Task")

#: A subagent step's tool-result text is cut to this many characters.
MAX_STEP_RESULT_LENGTH = 2000


def record_client():
    """The record client this process reads with, configured from its environment.

    Resolved per call and reached through the module, so a test replaces it
    with ``monkeypatch.setattr(session, "record_client", ...)``. Never raises:
    an unconfigured client answers every read unavailable.
    """
    from osprey.agent_record.client import RecordClient

    return RecordClient.from_env(os.environ)


def in_dispatch_worker() -> bool:
    """Whether this process serves a dispatch run, whose login must narrow its reads."""
    return os.environ.get(DISPATCH_RUN_ENV) == "1"


@dataclass(frozen=True)
class SessionRead:
    """What one session read answered.

    ``nodes`` are timeline nodes in timeline order; ``record`` is
    :data:`RECORD_COMPLETE`, :data:`RECORD_INCOMPLETE` or
    :data:`RECORD_UNAVAILABLE`, and ``reason`` says why it is not complete.
    """

    session_id: str | None
    nodes: list[dict[str, Any]] = field(default_factory=list)
    record: str = RECORD_COMPLETE
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.record != RECORD_UNAVAILABLE


def record_fields(record: str, reason: str) -> dict[str, str]:
    """A read's state as a response carries it: ``record``, and ``record_reason``
    when the read was not :data:`RECORD_COMPLETE`."""
    if record == RECORD_COMPLETE:
        return {"record": record}
    return {"record": record, "record_reason": reason}


def read_timeline(
    session_id: str,
    *,
    agent_id: str | None = None,
    client: Any = None,
) -> SessionRead:
    """Every node of *session_id*'s timeline (or of its subagent *agent_id*'s).

    The first page waits on a drain of the session, so what the session wrote
    before the call is in the answer; the pages after it follow the cursor of
    the page before, at most :data:`MAX_TIMELINE_PAGES`. Inside a dispatch
    worker every page is narrowed to the session. Never raises.
    """
    from osprey.agent_record.contract import Drain

    client = client if client is not None else record_client()
    narrow = session_id if in_dispatch_worker() else None
    nodes: list[dict[str, Any]] = []
    after: str | None = None
    missing: list[str] = []
    for page in range(MAX_TIMELINE_PAGES):
        wait = Drain(session_id=session_id) if page == 0 else None
        try:
            if agent_id is None:
                answer = client.timeline(session_id, after=after, wait=wait, narrow_session=narrow)
            else:
                answer = client.agent_timeline(
                    session_id, agent_id, after=after, wait=wait, narrow_session=narrow
                )
        except Exception as exc:  # a client never raises; a stand-in might
            logger.warning("session read failed: %s", exc)
            answer = None
        if answer is None or not answer.available:
            reason = (
                answer.unavailable if answer is not None else "the record read failed"
            ) or "the record is unavailable"
            if page == 0:
                return SessionRead(session_id, [], RECORD_UNAVAILABLE, reason)
            return SessionRead(session_id, nodes, RECORD_INCOMPLETE, reason)
        if not answer.complete:
            missing.append(answer.reason or "the record answered only part of the session")
        if isinstance(answer.data, list):
            nodes.extend(node for node in answer.data if isinstance(node, dict))
        after = answer.next
        if not after:
            break
    else:
        missing.append(f"the session has more than {MAX_TIMELINE_PAGES} timeline pages")
    if missing:
        return SessionRead(session_id, nodes, RECORD_INCOMPLETE, "; ".join(missing))
    return SessionRead(session_id, nodes)


def own_sessions(identity: str | None = None, *, client: Any = None) -> tuple[list[dict], Any]:
    """The sessions the login lists that *identity* took part in, newest first.

    Returns the rows (``list_sessions`` rows) and the answer they came from,
    so the caller can render an unavailable or incomplete list as such. A
    row belongs to *identity* when it names it among its identities or its
    persons; ``None`` keeps every row the login sees.
    """
    from osprey.agent_record.contract import Drain

    client = client if client is not None else record_client()
    answer = client.list_sessions(wait=Drain())
    rows = answer.data if answer.available and isinstance(answer.data, list) else []
    rows = [r for r in rows if isinstance(r, dict) and r.get("session_id")]
    if identity:
        rows = [
            r
            for r in rows
            if identity in (r.get("identities") or []) or identity in (r.get("persons") or [])
        ]
    return rows, answer


def current_session_id(project_dir: Path | str | None = None) -> str | None:
    """The id of the session this process serves, or ``None`` when none is known.

    The harness's conversation id first (what the agent's MCP servers and a
    dispatch run carry), then the newest transcript of *project_dir* — the
    one local lookup a session reader keeps.
    """
    from osprey.audit.call import harness_session_id

    sid = harness_session_id()
    if sid:
        return sid
    if project_dir is None:
        return None
    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

    try:
        path = TranscriptReader(project_dir).find_current_transcript()
    except OSError:
        return None
    return path.stem if path is not None else None


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> Mapping[str, Any]:
    """*value* when it is a mapping, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def _body(item: Any) -> Mapping[str, Any]:
    return _mapping(_mapping(item).get("body"))


def _ts(item: Any) -> str:
    ts = item.get("ts") if isinstance(item, Mapping) else None
    return ts if isinstance(ts, str) else ""


def line_of(node: Mapping[str, Any]) -> str:
    """The transcript line a node came from: its key less the content block index.

    The blocks of one transcript line share it; a node with no key is a line
    of its own.
    """
    key = node.get("key")
    if not isinstance(key, str) or not key:
        return f"node:{id(node)}"
    return key.rsplit(":", 1)[0]


def lines(nodes: Iterable[Mapping[str, Any]]) -> Iterator[list[Mapping[str, Any]]]:
    """*nodes* grouped into runs of consecutive nodes from one transcript line."""
    group: list[Mapping[str, Any]] = []
    current = None
    for node in nodes:
        line = line_of(node)
        if group and line != current:
            yield group
            group = []
        current = line
        group.append(node)
    if group:
        yield group


def result_text(content: Any) -> str:
    """A tool result's content as one string."""
    from osprey.mcp_server.workspace.transcript_reader import _extract_text

    if isinstance(content, list):
        return _extract_text(content)
    if isinstance(content, dict):
        return json.dumps(content)
    if content is None:
        return ""
    return str(content)


def prompt_text(body: Mapping[str, Any]) -> str:
    text = body.get("text")
    return text if isinstance(text, str) else ""


def is_bookkeeping_prompt(body: Mapping[str, Any]) -> bool:
    """Whether a prompt is the TUI's own accounting rather than the operator's words."""
    from osprey.mcp_server.workspace.transcript_reader import (
        _BOOKKEEPING_PREFIXES,
        INTERRUPT_MARKER,
    )

    if body.get("is_meta"):
        return True
    text = prompt_text(body).strip()
    return bool(text) and text.startswith((INTERRUPT_MARKER, *_BOOKKEEPING_PREFIXES))


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def session_events(
    nodes: Sequence[Mapping[str, Any]], session_id: str | None, *, agent_id: str | None = None
) -> list[dict[str, Any]]:
    """The tool-call and subagent events of *nodes*, descending into each subagent.

    Every MCP tool call is a ``tool_call`` event; every Agent (or Task) call is
    an ``agent_start``, the subagent's own events, then an ``agent_stop``. A
    subagent whose call is not in the record is bracketed the same way.
    """
    from osprey.mcp_server.workspace.transcript_reader import (
        _is_error_response,
        _split_mcp_tool_name,
        _truncate,
    )

    events: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        body = _body(node)
        result = _mapping(node.get("result"))
        result_body = _body(result)
        subagent = _mapping(node.get("subagent"))
        children = subagent.get("timeline") or []
        if node.get("kind") == "tool_call":
            name = str(body.get("name", ""))
            match = _split_mcp_tool_name(name)
            if match:
                short_name, server, full_name = match
                text = result_text(result_body.get("content", ""))
                is_error = bool(result_body.get("is_error")) or _is_error_response(text)
                events.append(
                    {
                        "type": "tool_call",
                        "timestamp": _ts(node),
                        "tool": short_name,
                        "tool_name": short_name,
                        "full_tool_name": full_name,
                        "server": server,
                        "server_name": server,
                        "is_error": is_error,
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "arguments": body.get("input", {}),
                        "result_summary": _truncate(text, is_error),
                    }
                )
            elif name in _SUBAGENT_TOOLS:
                tool_input = _mapping(body.get("input"))
                agent_type = tool_input.get("subagent_type", tool_input.get("description", ""))
                started = (
                    subagent.get("agent_id")
                    or result_body.get("agent_id")
                    or str(body.get("id", ""))
                )
                inner = session_events(children, session_id, agent_id=started)
                events.append(
                    {
                        "type": "agent_start",
                        "timestamp": _ts(node),
                        "agent_id": started,
                        "agent_type": agent_type,
                    }
                )
                events.extend(inner)
                events.append(
                    {
                        "type": "agent_stop",
                        "timestamp": _ts(result)
                        or (inner[-1]["timestamp"] if inner else "")
                        or _ts(node),
                        "agent_id": started,
                        "agent_type": agent_type,
                    }
                )
                continue
        if children:
            sub_id = subagent.get("agent_id") or None
            inner = session_events(children, session_id, agent_id=sub_id)
            if node.get("orphan") and sub_id:
                first = _ts(children[0]) if isinstance(children[0], Mapping) else ""
                last = _ts(children[-1]) if isinstance(children[-1], Mapping) else first
                events.append(
                    {
                        "type": "agent_start",
                        "timestamp": first,
                        "agent_id": sub_id,
                        "agent_type": "",
                    }
                )
                events.extend(inner)
                events.append(
                    {"type": "agent_stop", "timestamp": last, "agent_id": sub_id, "agent_type": ""}
                )
            else:
                events.extend(inner)
    return events


def chat_history(
    nodes: Sequence[Mapping[str, Any]], *, max_chars: int | None = MAX_CHAT_MESSAGE_LENGTH
) -> list[dict[str, Any]]:
    """The conversation of the main stream as turns ``{role, content, timestamp}``.

    One turn per transcript line that carries words: the operator's prompts
    (less the TUI's bookkeeping) and the agent's answers. Tool calls, results,
    thinking and subagent work are left out. *max_chars* cuts each turn's
    text (by default at the diagnostics view's cap); ``None`` leaves it whole.
    """
    turns: list[dict[str, Any]] = []
    for group in lines(nodes):
        texts: list[str] = []
        role = ""
        for node in group:
            kind = node.get("kind")
            body = _body(node)
            if node.get("agent_id"):
                continue
            if kind == "prompt":
                if is_bookkeeping_prompt(body):
                    texts = []
                    role = ""
                    break
                role = "user"
                texts.append(prompt_text(body))
            elif kind == "answer":
                role = "assistant"
                text = body.get("text")
                if isinstance(text, str):
                    texts.append(text)
        text = "\n".join(t for t in texts if t).strip()
        if not role or not text:
            continue
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + "..."
        turns.append({"role": role, "content": text, "timestamp": _ts(group[0])})
    return turns


def agent_steps(nodes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A subagent's stream as its steps, in the order it took them.

    Its first prompt is ``prompt``; its words are ``reasoning``, its thinking
    ``thinking``; a call is ``tool_call`` and its result ``tool_result``
    (text cut to :data:`MAX_STEP_RESULT_LENGTH`), the results of one line's
    calls following that line.
    """
    steps: list[dict[str, Any]] = []
    prompted = False
    for group in lines(nodes):
        results: list[dict[str, Any]] = []
        for node in group:
            kind = node.get("kind")
            body = _body(node)
            ts = _ts(node)
            if kind == "prompt":
                text = prompt_text(body).strip()
                if not prompted and text:
                    steps.append({"kind": "prompt", "timestamp": ts, "text": text})
                prompted = True
            elif kind == "answer":
                text = str(body.get("text") or "").strip()
                if text:
                    steps.append({"kind": "reasoning", "timestamp": ts, "text": text})
            elif kind == "thinking":
                text = str(body.get("text") or "").strip()
                if text:
                    steps.append({"kind": "thinking", "timestamp": ts, "text": text})
            elif kind == "tool_call":
                steps.append(
                    {
                        "kind": "tool_call",
                        "timestamp": ts,
                        "tool": body.get("name", ""),
                        "arguments": body.get("input", {}),
                    }
                )
                result = node.get("result")
                if isinstance(result, Mapping):
                    results.append(_result_step(result, str(body.get("id", ""))))
            elif kind in ("tool_result", "subagent"):
                results.append(_result_step(node, node.get("tool_use_id") or ""))
        steps.extend(results)
    return steps


def _result_step(result: Mapping[str, Any], tool_use_id: str) -> dict[str, Any]:
    body = _body(result)
    return {
        "kind": "tool_result",
        "timestamp": _ts(result),
        "tool_use_id": result.get("tool_use_id") or tool_use_id,
        "text": result_text(body.get("content", ""))[:MAX_STEP_RESULT_LENGTH],
        "is_error": bool(body.get("is_error")),
    }


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------


def _session_start(session_id: str) -> str | None:
    """The first time the record holds for *session_id*, or ``None``."""
    from osprey.agent_record.contract import Drain

    narrow = session_id if in_dispatch_worker() else None
    answer = record_client().list_sessions(wait=Drain(session_id=session_id), narrow_session=narrow)
    if not answer.available or not isinstance(answer.data, list):
        return None
    for row in answer.data:
        if isinstance(row, Mapping) and row.get("session_id") == session_id:
            first = row.get("first_ts")
            return first if isinstance(first, str) else None
    return None


def gather_session_metadata(created_via: str) -> dict:
    """Collect session metadata for logbook entries.

    Returns a dict with 8 fields: ``session_id``, ``transcript_path``,
    ``session_start_time``, ``git_branch``, ``git_commit_short``,
    ``operator``, ``model_name``, ``created_via``. All fall back gracefully to
    ``None`` except ``operator``, which floors at
    :data:`~osprey_connectors.identity.UNKNOWN_IDENTITY` — an entry always names
    somebody, even if only honestly.

    The session is the one this process serves (:func:`current_session_id`);
    its start is the first time the agent record holds for it. The transcript
    path is the local current-transcript lookup.

    Args:
        created_via: Caller identifier (e.g. ``"ariel-mcp"``, ``"gallery-compose"``).
    """
    import subprocess

    from osprey_connectors.workspace import load_osprey_config, resolve_project_root

    meta: dict = {"created_via": created_via}

    session_id: str | None = None
    transcript_path: str | None = None
    session_start_time: str | None = None
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        project_dir = resolve_project_root(load_osprey_config())
        current = TranscriptReader(project_dir).find_current_transcript()
        if current is not None:
            transcript_path = str(current)
        session_id = current_session_id() or (current.stem if current is not None else None)
    except Exception as exc:
        logger.warning("Session metadata: transcript lookup failed (non-fatal): %s", exc)

    if session_id is None:
        session_id = os.environ.get("OSPREY_SESSION_ID")

    if session_id:
        try:
            session_start_time = _session_start(session_id)
        except Exception as exc:
            logger.warning("Session metadata: record read failed (non-fatal): %s", exc)

    meta["session_id"] = session_id
    meta["transcript_path"] = transcript_path
    meta["session_start_time"] = session_start_time

    # --- Git fields ---
    git_branch: str | None = None
    git_commit_short: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            git_branch = result.stdout.strip() or None
    except Exception as exc:
        logger.warning("Session metadata: git branch failed (non-fatal): %s", exc)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            git_commit_short = result.stdout.strip() or None
    except Exception as exc:
        logger.warning("Session metadata: git commit failed (non-fatal): %s", exc)

    meta["git_branch"] = git_branch
    meta["git_commit_short"] = git_commit_short

    # --- Operator ---
    # The same ladder the audit ledger climbs, so an entry and the records of
    # the session that wrote it name the same person. Never raises, so there is
    # nothing to guard.
    meta["operator"] = acting_identity()

    # --- Model name ---
    model_name: str | None = None
    try:
        project_dir_for_settings = resolve_project_root(load_osprey_config())
        settings_path = project_dir_for_settings / ".claude" / "settings.json"
        if settings_path.exists():
            with open(settings_path) as fh:
                settings = json.load(fh)
            model_name = settings.get("model")
    except Exception as exc:
        logger.warning("Session metadata: settings.json read failed (non-fatal): %s", exc)
    if model_name is None:
        model_name = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("CLAUDE_MODEL")
    meta["model_name"] = model_name

    return meta
