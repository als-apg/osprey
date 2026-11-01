"""The session history a watchdog judges: the record's timeline nodes, as a bounded view.

Two steps, kept apart so the nodes are read once per prompt and every watchdog
builds its own view at its own budget:

* :func:`read_session_nodes` reads the Claude record session id's timeline
  through :func:`osprey.mcp_server.session.read_timeline`, within one deadline.
  Each record call waits at most the time left until that deadline, so a slow
  page ends the read rather than the watchdog's turn; what the read missed is
  reported (``drain_timed_out``, ``record_available``), never shown as a
  complete history.
* :func:`build_session_view_from_nodes` turns the nodes into compact text —
  prompts, answers, tool calls and tool results — appends the prompt that
  triggered the watchdog (the record may not have it yet), and keeps the newest
  entries that fit the token budget, reporting how many were cut.

Deadlines are :func:`time.monotonic` values. Nothing here imports from
``osprey.interfaces``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MAX_ENTRY_CHARS",
    "SessionNodes",
    "SessionView",
    "as_session_nodes",
    "build_session_view_from_nodes",
    "read_session_nodes",
    "rough_tokens",
]

#: One entry's text is cut to this many characters, so a single large tool
#: result cannot crowd every other entry out of the view.
MAX_ENTRY_CHARS = 4000

#: The prefix :class:`~osprey.agent_record.client.RecordClient` gives an
#: answer its call timed out on.
_TIMEOUT_PREFIX = "agent-record did not answer within"

_DEADLINE_REASON = "the history read reached its deadline"


@dataclass(frozen=True)
class SessionNodes:
    """What one history read answered.

    ``nodes`` are timeline nodes in timeline order (possibly only part of the
    session). ``drain_timed_out`` is true when the deadline ended the read
    before the record answered whole; ``record_available`` is false when the
    record did not answer at all (no record configured, unreachable, no
    session id, or the deadline passed before the first page). ``reason`` says
    why the read is not complete, empty when it is.
    """

    nodes: list[dict[str, Any]] = field(default_factory=list)
    drain_timed_out: bool = False
    record_available: bool = True
    reason: str = ""


def as_session_nodes(nodes: SessionNodes | Sequence[Mapping[str, Any]]) -> SessionNodes:
    """*nodes* as one read's answer; bare nodes read as a complete history."""
    if isinstance(nodes, SessionNodes):
        return nodes
    entries: list[Any] = list(nodes)
    return SessionNodes(nodes=entries)


@dataclass(frozen=True)
class SessionView:
    """A bounded view of a session: ``text`` holds ``entries_kept`` entries
    (the triggering prompt included), oldest first; ``entries_cut`` older
    entries did not fit; ``tokens_estimate`` is the text's size by the
    counter the view was built with."""

    text: str
    entries_kept: int
    entries_cut: int
    tokens_estimate: int


def rough_tokens(text: str) -> int:
    """A rough token count of *text*: one token per four characters, rounded up."""
    return (len(text) + 3) // 4


class _DeadlineClient:
    """A record client whose every call waits at most the time left to *deadline*.

    Each call builds a fresh client with ``timeout`` = deadline − now; a call
    asked for after the deadline is not made and answers unavailable.
    """

    def __init__(
        self,
        make_client: Callable[[float], Any],
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        self._make = make_client
        self._deadline = deadline
        self._clock = clock
        self.timed_out = False
        self.unconfigured = ""

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        from osprey.agent_record.client import ClientAnswer

        remaining = self._deadline - self._clock()
        if remaining <= 0:
            self.timed_out = True
            return ClientAnswer.failed(_DEADLINE_REASON)
        client = self._make(remaining)
        unconfigured = getattr(client, "_unconfigured", "")
        if unconfigured:
            self.unconfigured = unconfigured
        answer = getattr(client, name)(*args, **kwargs)
        failed = answer is not None and not answer.available and not self.unconfigured
        if failed and (
            (answer.unavailable or "").startswith(_TIMEOUT_PREFIX)
            or self._clock() >= self._deadline
        ):
            self.timed_out = True
        return answer

    def timeline(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("timeline", *args, **kwargs)

    def agent_timeline(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("agent_timeline", *args, **kwargs)


def _env_client(timeout: float) -> Any:
    from osprey.agent_record.client import RecordClient

    return RecordClient.from_env(os.environ, timeout=timeout)


def read_session_nodes(
    record_session_id: str | None,
    *,
    deadline: float,
    make_client: Callable[[float], Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> SessionNodes:
    """Every timeline node of *record_session_id* the record answers before *deadline*.

    *deadline* is a :func:`time.monotonic` value. *make_client* builds the
    record client for one call from its timeout; it defaults to
    ``RecordClient.from_env(os.environ, timeout=...)``. Never raises.
    """
    from osprey.mcp_server.session import RECORD_COMPLETE, RECORD_UNAVAILABLE, read_timeline

    if not record_session_id:
        return SessionNodes(record_available=False, reason="no record session id")
    client = _DeadlineClient(make_client or _env_client, deadline, clock)
    try:
        read = read_timeline(record_session_id, client=client)
    except Exception as exc:  # read_timeline never raises; stay safe if it does
        return SessionNodes(
            record_available=False, reason=f"the history read failed: {type(exc).__name__}"
        )
    timed_out = client.timed_out and read.record != RECORD_COMPLETE
    return SessionNodes(
        nodes=list(read.nodes),
        drain_timed_out=timed_out,
        record_available=read.record != RECORD_UNAVAILABLE,
        reason=read.reason,
    )


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_ENTRY_CHARS:
        return text[:MAX_ENTRY_CHARS] + " ...[truncated]"
    return text


def _arguments(value: Any) -> str:
    if value is None or value == {}:
        return ""
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _result_entry(result: Mapping[str, Any], prefix: str) -> str:
    from osprey.mcp_server.session import _body, result_text

    body = _body(result)
    label = "TOOL ERROR" if body.get("is_error") else "TOOL RESULT"
    return _clip(f"{prefix}{label}: {result_text(body.get('content', ''))}")


def _entries(nodes: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """``(kind, text)`` per prompt, answer, tool call and tool result, in timeline order."""
    from osprey.mcp_server.session import _body, is_bookkeeping_prompt, prompt_text

    entries: list[tuple[str, str]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        kind = node.get("kind")
        body = _body(node)
        prefix = "[subagent] " if node.get("agent_id") else ""
        if kind == "prompt":
            text = prompt_text(body).strip()
            if text and not is_bookkeeping_prompt(body):
                entries.append(("prompt", _clip(f"{prefix}USER: {text}")))
        elif kind == "answer":
            text = str(body.get("text") or "").strip()
            if text:
                entries.append(("answer", _clip(f"{prefix}ASSISTANT: {text}")))
        elif kind == "tool_call":
            name = str(body.get("name") or "?")
            entries.append(
                ("tool_call", _clip(f"{prefix}TOOL CALL {name}({_arguments(body.get('input'))})"))
            )
            result = node.get("result")
            if isinstance(result, Mapping):
                entries.append(("tool_result", _result_entry(result, prefix)))
        elif kind in ("tool_result", "subagent"):
            entries.append(("tool_result", _result_entry(node, prefix)))
    return entries


def _cut_line(count: int) -> str:
    return f"[{count} earlier entries cut]"


def build_session_view_from_nodes(
    nodes: Sequence[Mapping[str, Any]],
    prompt: str,
    *,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> SessionView:
    """The session as compact text, newest entries kept until *max_tokens*.

    The triggering *prompt* is always the last entry (added unless the
    record's newest prompt already is it) and is kept whatever its size; older
    entries are kept newest first while they fit, and a leading line says how
    many were cut. With no nodes the view is the prompt alone.
    """
    entries = [text for _, text in _entries(nodes)]
    prompt_entry = _clip(f"USER: {prompt.strip()}")
    if entries and entries[-1] == prompt_entry:
        entries.pop()

    def cost(text: str) -> int:
        return count_tokens(text + "\n")

    used = cost(prompt_entry)
    history_cost = sum(cost(text) for text in entries)
    kept: list[str] = []
    if used + history_cost <= max_tokens:
        kept = entries
    else:
        used += cost(_cut_line(len(entries)))
        for text in reversed(entries):
            size = cost(text)
            if used + size > max_tokens:
                break
            used += size
            kept.append(text)
        kept.reverse()
    cut = len(entries) - len(kept)
    lines = ([_cut_line(cut)] if cut else []) + kept + [prompt_entry]
    text = "\n".join(lines)
    return SessionView(
        text=text,
        entries_kept=len(kept) + 1,
        entries_cut=cut,
        tokens_estimate=count_tokens(text),
    )
