"""How the page assembles store rows into what it shows. Pure functions, no I/O.

* The session list is grouped by the terminal key of each session's earliest
  event, so a ``/clear`` followed by a resume reads as one transcript under its
  first terminal.
* A tool call is recorded when it ends, so it is placed at ``started_at``
  (its timestamp less ``duration_ms``).
* A queueserver preflight row is matched to the nearest preceding queue-submit
  tool call of the same user within :data:`PREFLIGHT_MATCH_SECONDS`. The
  queueserver runs agent-authored session plans, so in a person's own view a row
  with no such submit is never shown; under ``all`` and reader views every row
  is shown as stored, the unmatched ones labelled.
* An approval with a ``ToolUseId`` belongs to that tool call; one without is
  assigned to the transcript whose span on the same terminal contains it.
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from osprey.bluesky_tool_names import QUEUE_ADD, QUEUE_START

#: How long after a queue-submit tool call starts a preflight row may still be
#: that submit's.
PREFLIGHT_MATCH_SECONDS = 60

#: The tool calls that put a plan in front of a queueserver.
QUEUE_SUBMIT_TOOLS = frozenset({QUEUE_ADD, QUEUE_START})

_SUBJECT = re.compile(r'"subject"\s*:\s*"((?:[^"\\]|\\.)*)"')

Row = Mapping[str, Any]


def parse_ts(value: Any) -> datetime.datetime:
    """A store ``DateTime64`` as the page receives it, as an aware UTC datetime.

    The store sends nanoseconds; Python keeps microseconds, which is finer than
    any ordering the page makes.
    """
    text = str(value or "").strip()
    if not text:
        return datetime.datetime.min.replace(tzinfo=datetime.UTC)
    head, _, fraction = text.partition(".")
    parsed = datetime.datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
    micros = int((fraction + "000000")[:6]) if fraction else 0
    return parsed.replace(microsecond=micros, tzinfo=datetime.UTC)


@dataclass
class SessionGroup:
    """One transcript of the session list: every session of one terminal."""

    user: str
    terminal: str
    sessions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def first(self) -> dict[str, Any]:
        """The earliest session, whose id names the transcript."""
        return min(self.sessions, key=lambda row: parse_ts(row.get("first_ts")))

    @property
    def last_ts(self) -> str:
        return max((str(row.get("last_ts", "")) for row in self.sessions), default="")


def group_sessions(rows: Iterable[Row]) -> list[SessionGroup]:
    """Session-list rows grouped by ``(User, terminal key)``, newest activity first.

    A session with no terminal key is its own group.
    """
    groups: dict[tuple[str, str], SessionGroup] = {}
    for row in rows:
        user = str(row.get("User", ""))
        terminal = str(row.get("terminal_session") or "")
        key = (user, terminal or "session:" + str(row.get("SessionId", "")))
        group = groups.setdefault(key, SessionGroup(user=user, terminal=terminal))
        group.sessions.append(dict(row))
    for group in groups.values():
        group.sessions.sort(key=lambda row: parse_ts(row.get("first_ts")))
    return sorted(groups.values(), key=lambda g: parse_ts(g.last_ts), reverse=True)


def subject_of(body: Any) -> str:
    """The tool a ledger record names (its ``subject``), or ``""``.

    The page reads records cut to a preview length, so a body that no longer
    parses as JSON is searched for the field instead.
    """
    text = str(body or "")
    try:
        record = json.loads(text)
    except ValueError:
        match = _SUBJECT.search(text)
        return json.loads(f'"{match.group(1)}"') if match else ""
    subject = record.get("subject") if isinstance(record, dict) else None
    return subject if isinstance(subject, str) else ""


def is_queue_submit(subject: str) -> bool:
    """Whether *subject* is a queue-submit tool, however its server prefixes it."""
    return subject.rsplit("__", 1)[-1] in QUEUE_SUBMIT_TOOLS


@dataclass(frozen=True)
class PreflightItem:
    """One preflight row and the submit it was matched to (``None`` if none)."""

    row: dict[str, Any]
    submit: dict[str, Any] | None

    @property
    def matched(self) -> bool:
        return self.submit is not None


def match_preflight(preflight_rows: Iterable[Row], tool_rows: Iterable[Row]) -> list[PreflightItem]:
    """Each preflight row with the nearest preceding queue-submit within the window."""
    submits = sorted(
        (dict(row) for row in tool_rows if is_queue_submit(subject_of(row.get("body")))),
        key=lambda row: parse_ts(row.get("started_at") or row.get("Timestamp")),
    )
    window = datetime.timedelta(seconds=PREFLIGHT_MATCH_SECONDS)
    items = []
    for row in preflight_rows:
        at = parse_ts(row.get("Timestamp"))
        best = None
        for submit in submits:
            started = parse_ts(submit.get("started_at") or submit.get("Timestamp"))
            if started <= at <= started + window:
                best = submit
        items.append(PreflightItem(row=dict(row), submit=best))
    return items


def visible_preflight(items: Sequence[PreflightItem], *, own: bool) -> list[PreflightItem]:
    """What a view shows: matched rows only in an own view, every row otherwise."""
    return [item for item in items if item.matched] if own else list(items)


def assign_approvals(
    approvals: Iterable[Row],
    tool_rows: Iterable[Row],
    *,
    terminal: str,
    first_ts: Any,
    last_ts: Any,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Approvals by the tool call they answered, and the ones joined by time.

    Returns:
        ``(by_tool_use_id, loose)``: an approval whose ``ToolUseId`` is one of
        this transcript's tool calls, keyed by it; and an approval without one
        whose terminal is this transcript's and whose time is inside its span.
        Every other approval belongs to another transcript.
    """
    tool_ids = {str(row.get("ToolUseId")) for row in tool_rows if row.get("ToolUseId")}
    start, end = parse_ts(first_ts), parse_ts(last_ts)
    by_tool: dict[str, list[dict[str, Any]]] = {}
    loose: list[dict[str, Any]] = []
    for row in approvals:
        tool_use_id = str(row.get("ToolUseId") or "")
        if tool_use_id:
            if tool_use_id in tool_ids:
                by_tool.setdefault(tool_use_id, []).append(dict(row))
            continue
        if terminal and str(row.get("TerminalSession") or "") == terminal:
            if start <= parse_ts(row.get("Timestamp")) <= end:
                loose.append(dict(row))
    return by_tool, loose


@dataclass(frozen=True)
class Entry:
    """One line of a transcript, in time order."""

    kind: str
    at: datetime.datetime
    row: dict[str, Any]
    approvals: tuple[dict[str, Any], ...] = ()
    matched: bool = True


def timeline(
    conversation: Iterable[Row],
    tool_rows: Iterable[Row],
    preflight: Iterable[PreflightItem] = (),
    approvals_by_tool: Mapping[str, Sequence[Row]] | None = None,
    loose_approvals: Iterable[Row] = (),
) -> list[Entry]:
    """Every line of one transcript, ordered by when it happened."""
    approvals_by_tool = approvals_by_tool or {}
    entries = [
        Entry(kind="event", at=parse_ts(row.get("Timestamp")), row=dict(row))
        for row in conversation
    ]
    entries += [
        Entry(
            kind="tool_call",
            at=parse_ts(row.get("started_at") or row.get("Timestamp")),
            row=dict(row),
            approvals=tuple(
                dict(a) for a in approvals_by_tool.get(str(row.get("ToolUseId") or ""), ())
            ),
        )
        for row in tool_rows
    ]
    entries += [
        Entry(
            kind="preflight",
            at=parse_ts(item.row.get("Timestamp")),
            row=item.row,
            matched=item.matched,
        )
        for item in preflight
    ]
    entries += [
        Entry(kind="approval", at=parse_ts(row.get("Timestamp")), row=dict(row))
        for row in loose_approvals
    ]
    order = {"event": 0, "tool_call": 1, "approval": 2, "preflight": 3}
    return sorted(entries, key=lambda e: (e.at, order[e.kind]))
