"""The one reader: every read-API method over any :class:`RecordBackend`.

A backend only narrows rows by a coarse :class:`ScopeFilter`; everything a
caller may see is decided here, in Python, the same way for every backend:

**Scope.** Each entry's person is computed at read time through
:class:`osprey.agent_record.attribution.Attribution`, over the run accept
entries the read can see. ``Own(person)`` keeps the entries attributed to that
person; ``All`` keeps everything; ``Reader(name, kinds, days_back)`` keeps its
kinds, arrived within ``days_back`` days. ``Source(identity)`` keeps what the
identity ingested itself (its run entries included), the run entries of every
``run_id`` its own entries name, and the session timelines of those runs (the
session named by the run's accept entry, under the accept entry's identity).
``Source.narrow`` restricts that to one session or to an allow-listed set of
run ids.

**Entries.** Rows are deduplicated on ``(key, chunk.index)`` and grouped by
key; each group is reassembled (:func:`osprey.agent_record.contract.reassemble`).
A chunk set with a missing chunk is returned flagged ``incomplete`` with no
body, and the answer is then not ``complete``; ``full_entry`` answers
``unavailable`` for it.

**Limits.** A page holds at most :data:`PAGE_ROWS` entries, a session list at
most :data:`SESSION_ROWS` sessions and a bridge history at most
:data:`BRIDGE_TURNS` turns (``more`` says another page exists). A read scans at
most :data:`SCAN_ROWS` rows per select; a scan over the whole record keeps the
newest rows. Every answer (pages, session rows, run facts) is counted by one
accountant: at most :data:`MAX_BYTES` of bodies and text, and at most
:data:`READ_SECONDS` spent loading and assembling, checked while rows are
loaded as well as while the answer is cut. Hitting any of those makes the
answer not ``complete``.

**Wait marks.** A ``wait`` mark is polled for at most
:data:`WAIT_SECONDS`; if ingest has not passed it by then the answer is
returned, with the entries that exist, and ``complete: false``. A drain mark
is resolved by the :data:`Drainer` the service gives the reader: what the
caller's scope had pending in the spool, ``.otlp/`` and (with a session) that
session's transcript files when the read arrived. A transcript mark is checked
against the committed offsets of the scope's transcript identities.
``run``, ``run_timeline`` and ``bridge_history`` also drain each run's session
at read time, within the same bound, under the identity that accepted it, and
then read the run entries again. A run with no terminal entry yet is never
complete, and a run that is asked for or named by a bridge turn but has
neither an accept nor a terminal entry makes the answer incomplete. Time spent
waiting is not counted against :data:`READ_SECONDS`: the two budgets are
separate. A reader given no drainer has nothing to drain: a drain passes at
once. A backend failure at any point answers ``unavailable``: a partial answer
is never marked complete.

**Timeline.** One ordered stream: thinking stays inline in time order; each
``tool_call`` is joined by ``tool_use_id`` to its ``decision`` and to its
result (``tool_result``, or the ``subagent`` entry of an Agent/Task call); a
subagent's own entries are nested under the call that started it. A subagent
whose call is not (yet) in the record is shown as its own group at its first
entry's time. ``num_tool_calls`` and ``tool_count`` of a run are counted from
its session's timeline, never stored.

**Timeline cursor.** ``timeline`` and ``agent_timeline`` page by their
top-level nodes in ``(ts, inserted_at, key, chunk_index)`` order, through an
opaque ``after`` cursor (:class:`TimelineCursor`). The first page fixes the
snapshot ``as_of``: the newest ``inserted_at`` among the session's entries the
scope sees. Every later page reads only entries arrived by ``as_of`` and starts
strictly after the cursor's node, so no page repeats or skips a node of the
snapshot. ``next`` is the cursor of the following page while more of the
snapshot remains; ``newer`` says entries arrived after ``as_of``, which a
re-read from the start shows. Backends stamp ``inserted_at`` in commit order,
so an entry committed after a page was read always falls outside its snapshot.
A cursor that does not decode is a :class:`ValueError`.

**Preflight.** A queueserver ``preflight`` audit entry is matched to the
nearest preceding queue-submit ``tool_call`` of the same person that started at
most :data:`PREFLIGHT_MATCH_SECONDS` earlier. Under ``Own`` an unmatched
preflight entry is never shown; every other scope shows each one as stored.
"""

from __future__ import annotations

import base64
import binascii
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from osprey.agent_record.attribution import Attribution, is_run_accept
from osprey.agent_record.contract import (
    CHUNK_PART_FIELD,
    KIND_ANSWER,
    KIND_AUDIT,
    KIND_BRIDGE_TURN,
    KIND_DECISION,
    KIND_PROMPT,
    KIND_RUN,
    KIND_SUBAGENT,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    ORDER_NEWEST_FIRST,
    ROUTE_KINDS,
    ROUTE_SPOOL,
    RUN_ACCEPT,
    RUN_KEY_PREFIX,
    RUN_TERMINAL,
    SURFACE_FIELD,
    UNSPLIT,
    All,
    Entry,
    Narrow,
    Own,
    Query,
    Reader,
    RecordBackend,
    RecordBackendError,
    Scope,
    ScopeFilter,
    Source,
    WaitMark,
    accept_session,
    arrival_order,
    check_kind,
    is_token,
    reassemble,
    run_id_of,
    run_key,
    serialize_body,
)
from osprey.agent_record.identities import (
    PREFLIGHT_SURFACE,
    spool_source_identity,
    transcript_roots,
)
from osprey.bluesky_tool_names import QUEUE_ADD, QUEUE_START

__all__ = [
    "BRIDGE_TURNS",
    "MAX_BYTES",
    "PAGE_ROWS",
    "PREFLIGHT_MATCH_SECONDS",
    "QUEUE_SUBMIT_TOOLS",
    "READ_SECONDS",
    "SCAN_ROWS",
    "SESSION_ROWS",
    "SOURCE_FIELD",
    "WAIT_SECONDS",
    "TimelineCursor",
    "DrainProgress",
    "Drainer",
    "Limits",
    "PreflightMatch",
    "ReadAnswer",
    "RecordItem",
    "RecordReader",
    "TimelineNode",
    "build_timeline",
    "is_queue_submit",
    "match_preflight",
    "visible_preflight",
]

#: Entries one page holds.
PAGE_ROWS = 500
#: Sessions one session list holds.
SESSION_ROWS = 200
#: Turns one bridge history holds: the bridges' own history window.
BRIDGE_TURNS = 100
#: Rows one read scans from the backend before it stops and answers incomplete.
SCAN_ROWS = 100_000
#: Body and text bytes one answer carries before it stops and answers incomplete.
MAX_BYTES = 16 * 1024 * 1024
#: Seconds one read spends loading and assembling (waiting excluded) before it
#: answers incomplete.
READ_SECONDS = 10.0
#: The longest a read waits for ingest to pass a wait mark.
WAIT_SECONDS = 10.0
#: How often a waiting read polls its marks.
WAIT_POLL_SECONDS = 0.25

#: The body field a ``run`` terminal entry states its outcome in.
RUN_STATUS_FIELD = "status"
#: The status a run with no terminal entry yet is reported with.
RUN_RUNNING = "running"
#: The ``bridge_turn`` body fields the reader reads.
BRIDGE_CONVERSATION_FIELD = "conversation"
BRIDGE_QUESTION_FIELD = "question"
#: The body field naming which writer produced an entry (``entries(source=...)``).
SOURCE_FIELD = "source"
#: How long after a queue-submit tool call starts a preflight entry may still be
#: that submit's.
PREFLIGHT_MATCH_SECONDS = 60
#: The tool calls that put a plan in front of a queueserver.
QUEUE_SUBMIT_TOOLS = frozenset({QUEUE_ADD, QUEUE_START})

#: The kinds an entry can carry a ``run_id`` in: the spool's.
_RUN_NAMING_KINDS = ROUTE_KINDS[ROUTE_SPOOL]
#: Main-stream kinds after which a run's final answer starts.
_TURN_BREAK_KINDS = frozenset({KIND_PROMPT, KIND_TOOL_CALL, KIND_TOOL_RESULT, KIND_SUBAGENT})


@dataclass(frozen=True)
class Limits:
    """The row, byte and time limits of one reader."""

    rows: int = PAGE_ROWS
    sessions: int = SESSION_ROWS
    scan_rows: int = SCAN_ROWS
    bytes: int = MAX_BYTES
    seconds: float = READ_SECONDS


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True)
class RecordItem:
    """One entry as a reader returns it: reassembled, with its person.

    ``entry`` is the key's chunk 0 carrying the whole body; when the chunk set
    is ``incomplete`` its body is empty and :attr:`body` is ``None``.
    """

    entry: Entry
    person: str | None
    incomplete: bool = False

    @property
    def key(self) -> str:
        return self.entry.key

    @property
    def kind(self) -> str:
        return self.entry.kind

    @property
    def ts(self) -> datetime:
        return self.entry.ts

    @property
    def body(self) -> Mapping[str, Any] | None:
        return None if self.incomplete else self.entry.body

    def to_json(self) -> dict[str, Any]:
        e = self.entry
        return {
            "key": e.key,
            "kind": e.kind,
            "ts": _iso(e.ts),
            "inserted_at": _iso(e.inserted_at),
            "identity": e.identity,
            "person": self.person,
            "session_id": e.session_id,
            "terminal_session": e.terminal_session,
            "run_id": e.run_id,
            "agent_id": e.agent_id,
            "parent_key": e.parent_key,
            "tool_use_id": e.tool_use_id,
            "body": None if self.incomplete else dict(e.body),
            "incomplete": self.incomplete,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (RecordItem, TimelineNode)):
        return value.to_json()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class ReadAnswer:
    """What every read-API method answers.

    ``available`` is false when the backend failed; ``complete`` is then false
    too and ``data`` holds nothing. Otherwise ``complete`` says nothing the
    answer depends on is still missing (wait marks passed, no incomplete chunk
    set, no scan/byte/time limit hit), and ``more`` says another page exists.
    ``reason`` names why an answer is incomplete or unavailable. A timeline
    page carries ``next``, the cursor of the following page (``None`` on the
    last), and ``newer``, true when entries arrived after its snapshot.
    """

    available: bool
    complete: bool
    data: Any = field(default=None, hash=False)
    more: bool = False
    reason: str = ""
    next: str | None = None
    newer: bool = False

    @classmethod
    def unavailable(cls, reason: str) -> ReadAnswer:
        return cls(available=False, complete=False, data=None, more=False, reason=reason)

    def to_json(self) -> dict[str, Any]:
        if not self.available:
            return {"status": "unavailable", "complete": False, "reason": self.reason}
        return {
            "status": "ok",
            "complete": self.complete,
            "more": self.more,
            "next": self.next,
            "newer": self.newer,
            "reason": self.reason,
            "data": _jsonable(self.data),
        }


#: A node's place in timeline order: ``(ts, inserted_at, key, chunk_index)``.
Position = tuple[datetime, datetime, str, int]

#: The ``inserted_at`` an entry without one sorts and snapshots as.
_NEVER = datetime.min.replace(tzinfo=UTC)


def _arrived(entry: Entry) -> datetime:
    return entry.inserted_at.astimezone(UTC) if entry.inserted_at is not None else _NEVER


def _position(item: RecordItem) -> Position:
    e = item.entry
    return (e.ts.astimezone(UTC), _arrived(e), e.key, e.chunk.index)


@dataclass(frozen=True)
class TimelineCursor:
    """Where a timeline page starts: the snapshot ``as_of`` and the last node read.

    ``as_of`` is ``None`` when the snapshot held no entry; ``position`` is
    ``None`` for a cursor before the first node.
    """

    as_of: datetime | None
    position: Position | None = None

    def encode(self) -> str:
        position = self.position
        data = {
            "as_of": _iso(self.as_of),
            "at": None
            if position is None
            else [position[0].isoformat(), position[1].isoformat(), position[2], position[3]],
        }
        raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, text: str) -> TimelineCursor:
        """The cursor *text* encodes.

        Raises:
            ValueError: When *text* is not a cursor this reader issued.
        """
        try:
            if not isinstance(text, str) or not text:
                raise ValueError("empty")
            raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict) or set(data) != {"as_of", "at"}:
                raise ValueError("fields")
            as_of = None if data["as_of"] is None else _cursor_time(data["as_of"])
            at = data["at"]
            position: Position | None = None
            if at is not None:
                if not isinstance(at, list) or len(at) != 4:
                    raise ValueError("position")
                ts, arrived, key, index = at
                if not isinstance(key, str) or type(index) is not int:
                    raise ValueError("position")
                position = (_cursor_time(ts), _cursor_time(arrived), key, index)
        except (ValueError, TypeError, binascii.Error, UnicodeDecodeError):
            raise ValueError(f"after {text!r} is not a timeline cursor") from None
        return cls(as_of, position)


def _cursor_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("time")
    when = datetime.fromisoformat(value)
    if when.tzinfo is None:
        raise ValueError("time")
    return when.astimezone(UTC)


# ---------------------------------------------------------------------------
# Preflight matching
# ---------------------------------------------------------------------------


def is_queue_submit(name: str) -> bool:
    """Whether tool *name* is a queue submit, however its server prefixes it."""
    return isinstance(name, str) and name.rsplit("__", 1)[-1] in QUEUE_SUBMIT_TOOLS


def is_preflight(item: RecordItem) -> bool:
    """Whether *item* is a queueserver ``preflight`` audit entry."""
    body = item.entry.body
    return item.kind == KIND_AUDIT and body.get(SURFACE_FIELD) == PREFLIGHT_SURFACE


def _tool_name(item: RecordItem) -> str:
    name = item.entry.body.get("name")
    return name if isinstance(name, str) else ""


def _started_at(item: RecordItem) -> datetime:
    """When a tool call started: its time, less its ``duration_ms`` when it states one."""
    duration = item.entry.body.get("duration_ms")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        return item.ts - timedelta(milliseconds=duration)
    return item.ts


@dataclass(frozen=True)
class PreflightMatch:
    """One preflight entry and the submit it was matched to (``None`` if none)."""

    item: RecordItem
    submit: RecordItem | None

    @property
    def matched(self) -> bool:
        return self.submit is not None


def match_preflight(
    preflights: Iterable[RecordItem], tool_calls: Iterable[RecordItem]
) -> list[PreflightMatch]:
    """Each preflight entry with the nearest preceding queue submit of the same
    person that started within :data:`PREFLIGHT_MATCH_SECONDS` before it."""
    submits = sorted(
        (c for c in tool_calls if c.kind == KIND_TOOL_CALL and is_queue_submit(_tool_name(c))),
        key=_started_at,
    )
    window = timedelta(seconds=PREFLIGHT_MATCH_SECONDS)
    matches = []
    for item in preflights:
        best = None
        for submit in submits:
            started = _started_at(submit)
            if submit.person == item.person and started <= item.ts <= started + window:
                best = submit
        matches.append(PreflightMatch(item, best))
    return matches


def visible_preflight(matches: Sequence[PreflightMatch], *, own: bool) -> list[PreflightMatch]:
    """What a view shows: matched entries only in an own view, every one otherwise."""
    return [m for m in matches if m.matched] if own else list(matches)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


@dataclass
class TimelineNode:
    """One entry of a timeline, with what joins to it.

    A ``tool_call`` carries its ``decision`` and ``result``; when the result is
    a subagent, ``agent_id`` names it and ``children`` holds its own timeline.
    A node with no ``item`` is a subagent whose call is not in the record.
    """

    item: RecordItem | None
    decision: RecordItem | None = None
    result: RecordItem | None = None
    agent_id: str = ""
    children: list[TimelineNode] = field(default_factory=list)

    @property
    def ts(self) -> datetime:
        if self.item is not None:
            return self.item.ts
        return self.children[0].ts

    @property
    def position(self) -> Position:
        """This node's place in timeline order; a group without a call is at its first child."""
        if self.item is not None:
            return _position(self.item)
        return self.children[0].position

    def walk(self) -> Iterable[RecordItem]:
        """Every item of this node and its subtree."""
        for item in (self.item, self.decision, self.result):
            if item is not None:
                yield item
        for child in self.children:
            yield from child.walk()

    def to_json(self) -> dict[str, Any]:
        if self.item is not None:
            out = self.item.to_json()
        else:
            out = {"kind": KIND_SUBAGENT, "agent_id": self.agent_id, "orphan": True}
        out["decision"] = self.decision.to_json() if self.decision else None
        out["result"] = self.result.to_json() if self.result else None
        out["subagent"] = (
            {"agent_id": self.agent_id, "timeline": [c.to_json() for c in self.children]}
            if self.agent_id
            else None
        )
        return out


def build_timeline(items: Sequence[RecordItem], *, agent_id: str = "") -> list[TimelineNode]:
    """The timeline of the stream *agent_id* (``""``: the main session) of *items*.

    *items* are in time order. Results and decisions joined to a call leave the
    stream; every subagent stream reached from a call is nested under it, and
    in the main stream a subagent stream reached from no call becomes its own
    group at its first entry's time.
    """
    calls = {i.entry.tool_use_id for i in items if i.kind == KIND_TOOL_CALL and i.entry.tool_use_id}
    results: dict[str, RecordItem] = {}
    decisions: dict[str, RecordItem] = {}
    for i in items:
        tool_use_id = i.entry.tool_use_id
        if tool_use_id not in calls:
            continue
        if i.kind in (KIND_TOOL_RESULT, KIND_SUBAGENT):
            results.setdefault(tool_use_id, i)
        elif i.kind == KIND_DECISION:
            decisions.setdefault(tool_use_id, i)
    attached = {i.key for i in (*results.values(), *decisions.values())}
    streams: dict[str, list[RecordItem]] = {}
    for i in items:
        if i.key in attached:
            continue
        stream = "" if i.kind == KIND_SUBAGENT else i.entry.agent_id
        streams.setdefault(stream, []).append(i)
    used: set[str] = set()

    def build(stream: str) -> list[TimelineNode]:
        used.add(stream)
        nodes = []
        for i in streams.get(stream, []):
            node = TimelineNode(i)
            if i.kind == KIND_TOOL_CALL and i.entry.tool_use_id:
                node.decision = decisions.get(i.entry.tool_use_id)
                node.result = results.get(i.entry.tool_use_id)
                result = node.result
                if result is not None and result.kind == KIND_SUBAGENT:
                    sub = result.entry.agent_id
                    if sub and sub not in used:
                        node.agent_id = sub
                        node.children = build(sub)
            nodes.append(node)
        return nodes

    top = build(agent_id)
    if not agent_id:
        for stream in list(streams):
            if stream not in used:
                children = build(stream)
                if children:
                    top.append(TimelineNode(None, agent_id=stream, children=children))
    top.sort(key=lambda node: node.position)
    return top


def _items_of(nodes: Iterable[TimelineNode]) -> Iterable[RecordItem]:
    for node in nodes:
        yield from node.walk()


def _final_answer(nodes: Sequence[TimelineNode]) -> tuple[str | None, bool]:
    """The main stream's last answer text, and whether every part of it is whole."""
    main = [n.item for n in nodes if n.item is not None and not n.item.entry.agent_id]
    tail: list[RecordItem] = []
    for item in main:
        if item.kind in _TURN_BREAK_KINDS:
            tail = []
        elif item.kind == KIND_ANSWER:
            tail.append(item)
    if not tail:
        return None, True
    whole = all(not i.incomplete for i in tail)
    texts = [i.body.get("text") for i in tail if i.body is not None]
    return "\n".join(t for t in texts if isinstance(t, str)), whole


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


def _measure(value: Any) -> tuple[int, bool]:
    """The bytes *value* adds to an answer, and whether it holds an incomplete entry.

    An entry counts its serialized body, a string its UTF-8 length; mappings,
    sequences and timeline nodes count what they hold. Nothing else counts.
    """
    if isinstance(value, RecordItem):
        if value.incomplete:
            return 0, True
        return len(serialize_body(value.entry.body).encode("utf-8")), False
    if isinstance(value, str):
        return len(value.encode("utf-8")), False
    if isinstance(value, TimelineNode):
        parts: Iterable[Any] = list(value.walk())
    elif isinstance(value, Mapping):
        parts = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = value
    else:
        return 0, False
    size, incomplete = 0, False
    for part in parts:
        n, flagged = _measure(part)
        size += n
        incomplete = incomplete or flagged
    return size, incomplete


def _leading_fields(part: Any) -> dict[str, Any]:
    """The top-level body fields that lie wholly inside chunk 0's slice.

    A chunk set with a missing chunk has no body, but its chunk 0 still holds
    the start of the serialized body; attribution reads ``surface`` from it.
    A value is kept only when the slice shows where it ends.
    """
    if not isinstance(part, str) or not part.startswith("{"):
        return {}
    decoder = json.JSONDecoder()
    fields: dict[str, Any] = {}
    pos = 1
    try:
        while True:
            name, pos = decoder.raw_decode(part, pos)
            if not isinstance(name, str) or part[pos : pos + 1] != ":":
                break
            value, pos = decoder.raw_decode(part, pos + 1)
            end = part[pos : pos + 1]
            if end not in (",", "}"):
                break
            fields[name] = value
            if end == "}":
                break
            pos += 1
    except ValueError:
        pass
    return fields


class _Ctx:
    """The state of one read: its scope, budgets and why it is incomplete.

    The wait budget starts with the read; the read budget is pushed back by
    every second spent waiting (:meth:`RecordReader._wait`), so waiting never
    uses up the time the read has to load and assemble.
    """

    def __init__(self, reader: RecordReader, scope: Scope) -> None:
        self.scope = scope
        self.reasons: list[str] = []
        start = reader._clock()
        self.wait_deadline = start + reader.wait_seconds
        self.read_deadline = start + reader.limits.seconds
        self.size = 0
        self.view: _SourceView | None = None
        self.next: str | None = None
        self.newer = False

    def incomplete(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)


@dataclass(frozen=True)
class _SourceView:
    """What a :class:`Source` scope sees, resolved once per read."""

    identity: str
    narrow: Narrow | None
    runs: frozenset[str]
    run_sessions: Mapping[str, str]
    sessions: frozenset[tuple[str, str]]

    def visible(self, entry: Entry, mapped_identity: str) -> bool:
        run_id = run_id_of(entry)
        in_sessions = (
            bool(entry.session_id) and (mapped_identity, entry.session_id) in self.sessions
        )
        run_ok = entry.kind == KIND_RUN and run_id in self.runs
        own = entry.identity == self.identity
        if self.narrow is None:
            return own or run_ok or in_sessions
        if self.narrow.session_id is not None:
            session = self.narrow.session_id
            own_here = own and (
                entry.session_id == session or self.run_sessions.get(run_id) == session
            )
            return own_here or run_ok or in_sessions
        return (own and run_id in self.runs) or run_ok or in_sessions


class DrainProgress(Protocol):
    """One drain the service resolved: what was pending when it was asked for."""

    def passed(self) -> bool:
        """Whether ingest has committed past it; raises ``RecordBackendError`` on failure."""
        ...


class Drainer(Protocol):
    """Resolve a drain for *scope*: snapshot what its identities (plus *also*)
    have pending and, with *session_id*, that session's transcript files, then
    wake ingest. Only the service, which sees the spool and transcript roots,
    can; the reader just polls the answer."""

    def __call__(
        self, scope: Scope, session_id: str | None, also: frozenset[str]
    ) -> DrainProgress: ...


#: The run facts that can be large; dropped from a run that does not fit.
_LARGE_RUN_FACTS = ("accept", "terminal", "question", "answer")


class RecordReader:
    """The read API over one backend.

    Every method takes the caller's :data:`Scope` (the service resolves it from
    the credential) and an optional ``wait`` mark, and returns a
    :class:`ReadAnswer`. Arguments that are not valid raise ``ValueError``;
    backend failures never raise, they answer ``unavailable``.
    """

    def __init__(
        self,
        backend: RecordBackend,
        config: Mapping[str, Any] | None = None,
        *,
        limits: Limits | None = None,
        wait_seconds: float = WAIT_SECONDS,
        poll_seconds: float = WAIT_POLL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
        drain: Drainer | None = None,
    ) -> None:
        self.backend = backend
        self.drain = drain
        self.config: dict[str, Any] = dict(config or {})
        self.limits = limits or Limits()
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self._clock = clock
        self._sleep = sleep
        self._now = now or (lambda: datetime.now(UTC))
        self._mapped: dict[str, str] = {}

    # -- plumbing -----------------------------------------------------------

    def _map(self, identity: str) -> str:
        known = self._mapped.get(identity)
        if known is None:
            try:
                known = spool_source_identity(self.config, identity)
            except ValueError:
                known = identity
            self._mapped[identity] = known
        return known

    def _answer(
        self,
        scope: Scope,
        wait: WaitMark | None,
        build: Callable[[_Ctx], tuple[Any, bool]],
    ) -> ReadAnswer:
        if not isinstance(scope, (Own, All, Reader, Source)):
            raise ValueError(f"{scope!r} is not a record scope")
        ctx = _Ctx(self, scope)
        try:
            if wait is not None and not self._wait(ctx, [self._check(scope, wait)]):
                ctx.incomplete(f"wait mark not passed within {self.wait_seconds:g} s")
            data, more = build(ctx)
        except RecordBackendError as exc:
            return ReadAnswer.unavailable(f"the record backend is unavailable: {exc}")
        return ReadAnswer(
            available=True,
            complete=not ctx.reasons,
            data=data,
            more=more,
            reason="; ".join(ctx.reasons),
            next=ctx.next,
            newer=ctx.newer,
        )

    def _transcript_identities(self, scope: Scope) -> list[str]:
        """Whose transcript copy a caller's transcript mark is checked against."""
        if isinstance(scope, Source):
            return [scope.identity]
        if isinstance(scope, Own):
            return [scope.person]
        try:
            return sorted(set(transcript_roots(self.config).values()))
        except (ValueError, KeyError, TypeError):
            return []

    def _drained(
        self, scope: Scope, session_id: str | None, also: frozenset[str] = frozenset()
    ) -> Callable[[], bool]:
        """The check of one drain; a reader without a drainer has nothing to drain."""
        if self.drain is None:
            return lambda: True
        return self.drain(scope, session_id, also).passed

    def _check(self, scope: Scope, mark: WaitMark) -> Callable[[], bool]:
        """The check of the caller's *mark*: a drain, or a transcript mark."""
        if mark.drain is not None:
            return self._drained(scope, mark.drain.session_id)
        identities = self._transcript_identities(scope)
        return lambda: any(
            mark.passed(self.backend, transcript_identity=identity) for identity in identities
        )

    def _wait(self, ctx: _Ctx, checks: Sequence[Callable[[], bool]]) -> bool:
        """Poll *checks* until all have passed or the read's wait bound is spent.

        The read budget is pushed back by the time spent here.
        """
        started = self._clock()
        pending = list(checks)
        try:
            while True:
                pending = [check for check in pending if not check()]
                if not pending:
                    return True
                remaining = ctx.wait_deadline - self._clock()
                if remaining <= 0:
                    return False
                self._sleep(min(self.poll_seconds, remaining))
        finally:
            ctx.read_deadline += self._clock() - started

    def _out_of_time(self, ctx: _Ctx) -> bool:
        if self._clock() > ctx.read_deadline:
            ctx.incomplete(f"the read reached its time limit of {self.limits.seconds:g} s")
            return True
        return False

    def _admit(self, ctx: _Ctx, value: Any) -> bool:
        """Count *value* toward the answer; ``False`` (and incomplete) once a limit is hit."""
        if self._out_of_time(ctx):
            return False
        size, incomplete = _measure(value)
        if incomplete:
            ctx.incomplete("an entry's chunk set is incomplete")
        if ctx.size + size > self.limits.bytes:
            ctx.incomplete(f"the answer reached its byte limit of {self.limits.bytes}")
            return False
        ctx.size += size
        return True

    def _merge(self, ctx: _Ctx, scope_filter: ScopeFilter) -> ScopeFilter | None:
        """*scope_filter* narrowed by a :class:`Reader` scope; ``None`` when it admits nothing."""
        scope = ctx.scope
        if not isinstance(scope, Reader):
            return scope_filter
        kinds = scope.kinds & scope_filter.kinds if scope_filter.kinds else scope.kinds
        if not kinds:
            return None
        arrived = scope_filter.arrived_since
        if scope.days_back is not None:
            floor = self._now() - timedelta(days=scope.days_back)
            arrived = floor if arrived is None else max(arrived, floor)
        return replace(scope_filter, kinds=frozenset(kinds), arrived_since=arrived)

    def _select(
        self, ctx: _Ctx, scope_filter: ScopeFilter, query: Query, *, newest: bool = False
    ) -> list[Entry]:
        """At most :attr:`Limits.scan_rows` rows in *query*'s order; with *newest*
        the cut keeps the newest rows."""
        cap = self.limits.scan_rows
        limit = cap + 1 if query.limit is None else min(query.limit, cap + 1)
        flip = newest and query.order != ORDER_NEWEST_FIRST
        if flip:
            query = replace(query, order=ORDER_NEWEST_FIRST)
        rows = self.backend.select(scope_filter, replace(query, limit=limit))
        if len(rows) > cap:
            ctx.incomplete(f"the read reached its scan limit of {cap} rows")
            rows = rows[:cap]
        if flip:
            rows = rows[::-1]
        return rows

    def _attribution(self, entries: Sequence[Entry]) -> Attribution:
        """Attribution over *entries* and the run accepts their sessions and runs need."""
        sessions = {e.session_id for e in entries if e.session_id}
        runs = {r for r in (run_id_of(e) for e in entries) if r}
        accepts: list[Entry] = [e for e in entries if e.kind == KIND_RUN]
        for scope_filter in (
            ScopeFilter(kinds=frozenset({KIND_RUN}), session_ids=frozenset(sessions))
            if sessions
            else None,
            ScopeFilter(kinds=frozenset({KIND_RUN}), run_ids=frozenset(runs)) if runs else None,
        ):
            if scope_filter is not None:
                accepts.extend(
                    self.backend.select(scope_filter, Query(key_prefix=f"{RUN_KEY_PREFIX}:"))
                )
        unique = {e.unique_key: e for e in accepts}
        return Attribution.build(self.config, unique.values())

    def _load(
        self, ctx: _Ctx, scope_filter: ScopeFilter, query: Query, *, newest: bool = False
    ) -> list[RecordItem]:
        """Selected rows deduplicated, reassembled and attributed, before scope.

        Stops (incomplete) with what it has once the read is out of time.
        """
        if self._out_of_time(ctx):
            return []
        rows = self._select(ctx, scope_filter, query, newest=newest)
        groups: dict[str, list[Entry]] = {}
        seen: set[tuple[str, int]] = set()
        for row in rows:
            if row.unique_key in seen:
                continue
            seen.add(row.unique_key)
            groups.setdefault(row.key, []).append(row)
        # (entry as returned, entry attribution reads, incomplete)
        entries: list[tuple[Entry, Entry, bool]] = []
        for chunks in groups.values():
            if self._out_of_time(ctx):
                break
            whole = reassemble(chunks)
            if whole.body is None:
                first = whole.first
                lead = (
                    _leading_fields(first.body.get(CHUNK_PART_FIELD))
                    if first.chunk.index == 0
                    else {}
                )
                entries.append((replace(first, body={}), replace(first, body=lead), True))
            else:
                entry = replace(whole.first, body=whole.body, chunk=UNSPLIT)
                entries.append((entry, entry, False))
        attribution = self._attribution([e for e, _, _ in entries])
        return [
            RecordItem(entry, attribution.person(seen_as), incomplete)
            for entry, seen_as, incomplete in entries
        ]

    def _source_view(self, ctx: _Ctx, scope: Source) -> _SourceView:
        if ctx.view is not None:
            return ctx.view
        own = self._select(
            ctx,
            ScopeFilter(identities=frozenset({scope.identity}), kinds=_RUN_NAMING_KINDS),
            Query(),
            newest=True,
        )
        named = {r for r in (run_id_of(e) for e in own) if r}
        run_sessions: dict[str, str] = {}
        session_identity: dict[str, str] = {}
        if named:
            accepts = self.backend.select(
                ScopeFilter(kinds=frozenset({KIND_RUN}), run_ids=frozenset(named)),
                Query(key_prefix=f"{RUN_KEY_PREFIX}:"),
            )
            ordered = sorted((a for a in accepts if is_run_accept(a)), key=arrival_order)
            for accept in ordered:
                run_id = run_id_of(accept)
                session = accept_session(accept)
                if run_id in named and session and run_id not in run_sessions:
                    run_sessions[run_id] = session
                    session_identity[run_id] = self._map(accept.identity)
        runs = set(named)
        narrow = scope.narrow
        if narrow is not None and narrow.session_id is not None:
            runs = {r for r in runs if run_sessions.get(r) == narrow.session_id}
        elif narrow is not None:
            runs &= narrow.run_ids
        sessions = frozenset(
            (session_identity[r], run_sessions[r]) for r in runs if r in run_sessions
        )
        ctx.view = _SourceView(scope.identity, narrow, frozenset(runs), run_sessions, sessions)
        return ctx.view

    def _in_scope(self, ctx: _Ctx, item: RecordItem) -> bool:
        scope = ctx.scope
        if isinstance(scope, All):
            return True
        if isinstance(scope, Own):
            return item.person == scope.person
        if isinstance(scope, Reader):
            if item.kind not in scope.kinds:
                return False
            if scope.days_back is None:
                return True
            arrived = item.entry.inserted_at or item.ts
            return arrived >= self._now() - timedelta(days=scope.days_back)
        assert isinstance(scope, Source)
        view = self._source_view(ctx, scope)
        return view.visible(item.entry, self._map(item.entry.identity))

    def _collect(
        self,
        ctx: _Ctx,
        scope_filter: ScopeFilter,
        query: Query | None = None,
        *,
        newest: bool = False,
    ) -> list[RecordItem]:
        """The entries the caller's scope sees under *scope_filter* and *query*."""
        merged = self._merge(ctx, scope_filter)
        if merged is None:
            return []
        loaded = self._load(ctx, merged, query or Query(), newest=newest)
        items = [i for i in loaded if self._in_scope(ctx, i)]
        if isinstance(ctx.scope, Own):
            items = self._hide_unmatched_preflight(ctx, items)
        return items

    def _hide_unmatched_preflight(self, ctx: _Ctx, items: list[RecordItem]) -> list[RecordItem]:
        preflights = [i for i in items if is_preflight(i)]
        if not preflights:
            return items
        window = timedelta(seconds=PREFLIGHT_MATCH_SECONDS)
        since = min(i.ts for i in preflights) - window
        until = max(i.ts for i in preflights) + timedelta(microseconds=1)
        calls = self._load(
            ctx,
            ScopeFilter(kinds=frozenset({KIND_TOOL_CALL})),
            Query(since=since, until=until),
            newest=True,
        )
        shown = {
            m.item.key for m in visible_preflight(match_preflight(preflights, calls), own=True)
        }
        return [i for i in items if not is_preflight(i) or i.key in shown]

    @staticmethod
    def _rows(limit: int | None, cap: int) -> int:
        """The rows a page holds: *limit*, at most *cap* (*cap* when unset)."""
        return cap if limit is None else max(0, min(int(limit), cap))

    def _page(self, ctx: _Ctx, items: Sequence[Any], rows: int) -> tuple[list[Any], bool]:
        """The first *rows* of *items* (``more`` when there are others), cut
        further by the byte and time limits (incomplete)."""
        kept: list[Any] = []
        for value in items[:rows]:
            if not self._admit(ctx, value):
                return kept, False
            kept.append(value)
        return kept, len(items) > rows

    def _timeline_page(
        self,
        ctx: _Ctx,
        items: Sequence[RecordItem],
        after: TimelineCursor | None,
        limit: int | None,
        agent_id: str = "",
    ) -> tuple[list[TimelineNode], bool]:
        """One page of the timeline of *items* under the snapshot cursor *after*.

        Sets the page's ``next`` and ``newer`` on *ctx*.
        """
        if after is None:
            as_of = max((_arrived(i.entry) for i in items), default=None)
        else:
            as_of = after.as_of
        snapshot = [i for i in items if as_of is not None and _arrived(i.entry) <= as_of]
        ctx.newer = len(snapshot) < len(items)
        nodes = build_timeline(snapshot, agent_id=agent_id)
        start = after.position if after is not None else None
        if start is not None:
            nodes = [n for n in nodes if n.position > start]
        rows = self._rows(limit, self.limits.rows)
        kept, _ = self._page(ctx, nodes, rows)
        if len(kept) == len(nodes) or (not kept and rows > 0):
            return kept, False
        last = kept[-1].position if kept else start
        ctx.next = TimelineCursor(as_of, last).encode()
        return kept, True

    # -- runs ---------------------------------------------------------------

    def _runs(
        self,
        ctx: _Ctx,
        run_ids: Iterable[str],
        turns: Sequence[RecordItem] | None = None,
    ) -> dict[str, dict[str, Any] | None]:
        """Each run's accept, terminal, session timeline and derived counts, or
        ``None`` for a run the caller's scope sees no entry of.

        A fixed number of selects serves any number of runs: one for their run
        entries (a second after a drain), one for their
        sessions and one for their bridge turns (skipped when *turns*, already
        in scope, are given). Each run's session is drained under the identity
        that accepted it, every drain in one wait; runs with no accept entry
        yet drain the caller's own scope.
        """
        ids = sorted(set(run_ids))
        for run_id in ids:
            if not is_token(run_id):
                raise ValueError(f"run id {run_id!r} is not a token")
        if not ids:
            return {}
        keys = frozenset(run_key(r, phase) for r in ids for phase in (RUN_ACCEPT, RUN_TERMINAL))
        run_filter = ScopeFilter(kinds=frozenset({KIND_RUN}))
        by_key = {i.key: i for i in self._collect(ctx, run_filter, Query(keys=keys))}
        drains = self._run_drains(ctx, ids, by_key)
        if drains:
            first = [(run_ids_, check) for run_ids_, check in drains if not check()]
            if first and not self._wait(ctx, [check for _, check in first]):
                for run_ids_, _ in first:
                    for run_id in run_ids_:
                        ctx.incomplete(
                            f"run {run_id}'s ingest did not drain within {self.wait_seconds:g} s"
                        )
            # What the drain committed is read, not what was there before it.
            # Ingest may commit between the read above and the drain's first
            # check, so a drain that has already passed is read again too.
            by_key = {i.key: i for i in self._collect(ctx, run_filter, Query(keys=keys))}
        found: dict[str, tuple[RecordItem | None, RecordItem | None]] = {}
        for run_id in ids:
            accept = by_key.get(run_key(run_id, RUN_ACCEPT))
            terminal = by_key.get(run_key(run_id, RUN_TERMINAL))
            if accept is None and terminal is None:
                ctx.incomplete(f"run {run_id} has no accept or terminal entry yet")
                continue
            found[run_id] = (accept, terminal)
            if terminal is None:
                ctx.incomplete(f"run {run_id} has no terminal entry yet")
            elif terminal.incomplete:
                ctx.incomplete(f"run {run_id}'s terminal entry is incomplete")
        sessions = {
            run_id: accept_session(accept.entry)
            for run_id, (accept, _) in found.items()
            if accept is not None and accept_session(accept.entry)
        }
        by_session: dict[str, list[RecordItem]] = {}
        if sessions:
            for item in self._collect(ctx, ScopeFilter(session_ids=frozenset(sessions.values()))):
                by_session.setdefault(item.entry.session_id, []).append(item)
        if turns is None and found:
            turns = self._collect(
                ctx,
                ScopeFilter(kinds=frozenset({KIND_BRIDGE_TURN}), run_ids=frozenset(found)),
            )
        questions: dict[str, Any] = {}
        for turn in sorted(turns or (), key=lambda t: (t.ts, t.key)):
            run_id = turn.entry.run_id
            if run_id in found and run_id not in questions and turn.body is not None:
                questions[run_id] = turn.body.get(BRIDGE_QUESTION_FIELD)
        facts: dict[str, dict[str, Any] | None] = dict.fromkeys(ids)
        timelines: dict[str, list[TimelineNode]] = {}
        for run_id, (accept, terminal) in found.items():
            session = sessions.get(run_id, "")
            if session and session not in timelines:
                timelines[session] = build_timeline(by_session.get(session, []))
            nodes = timelines.get(session, [])
            answer, whole = _final_answer(nodes)
            if not whole:
                ctx.incomplete(f"run {run_id}'s answer is incomplete")
            tool_calls = sum(1 for i in _items_of(nodes) if i.kind == KIND_TOOL_CALL)
            status = RUN_RUNNING
            if terminal is not None and terminal.body is not None:
                status = terminal.body.get(RUN_STATUS_FIELD, RUN_RUNNING)
            facts[run_id] = {
                "run_id": run_id,
                "session_id": session or None,
                "requester": accept.person if accept is not None else None,
                "status": status,
                "accept": accept,
                "terminal": terminal,
                "question": questions.get(run_id),
                "answer": answer,
                "num_tool_calls": tool_calls,
                "tool_count": tool_calls,
                "timeline": nodes,
            }
        return facts

    def _run_drains(
        self, ctx: _Ctx, ids: Sequence[str], by_key: Mapping[str, RecordItem]
    ) -> list[tuple[list[str], Callable[[], bool]]]:
        """One drain per run session (and one for the runs with no accept yet),
        each with the runs it serves; none without a drainer."""
        if self.drain is None:
            return []
        by_session: dict[tuple[str | None, str], list[str]] = {}
        for run_id in ids:
            accept = by_key.get(run_key(run_id, RUN_ACCEPT))
            session = accept_session(accept.entry) if accept is not None else ""
            if accept is not None and session:
                by_session.setdefault((session, accept.entry.identity), []).append(run_id)
            else:
                by_session.setdefault((None, ""), []).append(run_id)
        drains = []
        for (drain_session, identity), run_ids in sorted(
            by_session.items(), key=lambda kv: str(kv[0])
        ):
            also = frozenset({identity}) if identity else frozenset()
            drains.append((run_ids, self._drained(ctx.scope, drain_session, also)))
        return drains

    def _run_summary(self, ctx: _Ctx, facts: dict[str, Any] | None) -> dict[str, Any] | None:
        """*facts* without the timeline, counted toward the answer; a run that does
        not fit keeps only its small facts."""
        if facts is None:
            return None
        summary = {k: v for k, v in facts.items() if k != "timeline"}
        if not self._admit(ctx, summary):
            for name in _LARGE_RUN_FACTS:
                summary[name] = None
        return summary

    # -- the read API ---------------------------------------------------------

    def list_sessions(
        self,
        scope: Scope,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        wait: WaitMark | None = None,
    ) -> ReadAnswer:
        """Every session the scope sees, newest activity first, promptless runs included."""

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            rows = self._sessions(ctx, since)
            return self._page(ctx, rows, self._rows(limit, self.limits.sessions))

        return self._answer(scope, wait, build)

    def _sessions(self, ctx: _Ctx, since: datetime | None) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for item in self._collect(ctx, ScopeFilter(), Query(since=since), newest=True):
            if self._out_of_time(ctx):
                break
            e = item.entry
            session = e.session_id or (accept_session(e) if is_run_accept(e) else "")
            if not session:
                continue
            row = groups.get(session)
            if row is None:
                row = groups[session] = {
                    "session_id": session,
                    "persons": set(),
                    "identities": set(),
                    "first_ts": e.ts,
                    "last_ts": e.ts,
                    "entries": 0,
                    "prompt": None,
                    "run_ids": set(),
                    "terminal_session": "",
                }
            if item.person:
                row["persons"].add(item.person)
            row["identities"].add(e.identity)
            row["first_ts"] = min(row["first_ts"], e.ts)
            row["last_ts"] = max(row["last_ts"], e.ts)
            row["entries"] += 1
            run_id = run_id_of(e)
            if run_id:
                row["run_ids"].add(run_id)
            if not row["terminal_session"] and e.terminal_session:
                row["terminal_session"] = e.terminal_session
            body = item.body
            if (
                row["prompt"] is None
                and item.kind == KIND_PROMPT
                and not e.agent_id
                and body is not None
                and isinstance(body.get("text"), str)
                and not body.get("is_meta")
            ):
                row["prompt"] = body["text"]
        rows = []
        for row in groups.values():
            row["persons"] = sorted(row["persons"])
            row["identities"] = sorted(row["identities"])
            row["run_ids"] = sorted(row["run_ids"])
            rows.append(row)
        rows.sort(key=lambda r: (r["last_ts"], r["session_id"]), reverse=True)
        return rows

    def session_users(
        self, scope: Scope, *, since: datetime | None = None, wait: WaitMark | None = None
    ) -> ReadAnswer:
        """Every person with a session the scope sees: their session count and last activity."""

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            users: dict[str, dict[str, Any]] = {}
            for row in self._sessions(ctx, since):
                for person in row["persons"]:
                    user = users.setdefault(
                        person, {"person": person, "sessions": 0, "last_ts": row["last_ts"]}
                    )
                    user["sessions"] += 1
                    user["last_ts"] = max(user["last_ts"], row["last_ts"])
            return self._page(ctx, [users[p] for p in sorted(users)], len(users))

        return self._answer(scope, wait, build)

    def timeline(
        self,
        scope: Scope,
        session_id: str,
        *,
        after: str | None = None,
        limit: int | None = None,
        wait: WaitMark | None = None,
    ) -> ReadAnswer:
        """One session as one ordered stream (see the module docstring), paged by
        its top-level nodes so a call and what joins to it stay on one page;
        *after* is the ``next`` cursor of the page before."""
        if not is_token(session_id):
            raise ValueError(f"session id {session_id!r} is not a token")
        cursor = None if after is None else TimelineCursor.decode(after)

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            items = self._collect(ctx, ScopeFilter(session_ids=frozenset({session_id})))
            return self._timeline_page(ctx, items, cursor, limit)

        return self._answer(scope, wait, build)

    def agent_timeline(
        self,
        scope: Scope,
        session_id: str,
        agent_id: str,
        *,
        after: str | None = None,
        limit: int | None = None,
        wait: WaitMark | None = None,
    ) -> ReadAnswer:
        """One subagent's own stream within *session_id*, its nested subagents
        included, paged like :meth:`timeline`."""
        if not is_token(session_id) or not is_token(agent_id):
            raise ValueError(f"session {session_id!r} and agent {agent_id!r} must be tokens")
        cursor = None if after is None else TimelineCursor.decode(after)

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            items = self._collect(ctx, ScopeFilter(session_ids=frozenset({session_id})))
            return self._timeline_page(ctx, items, cursor, limit, agent_id)

        return self._answer(scope, wait, build)

    def full_entry(self, scope: Scope, key: str, *, wait: WaitMark | None = None) -> ReadAnswer:
        """One entry with its whole body; ``unavailable`` when its chunk set is incomplete,
        ``data: None`` when the scope sees no entry under *key*."""
        if not isinstance(key, str) or not key:
            raise ValueError("full_entry needs a key")

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            items = self._collect(ctx, ScopeFilter(), Query(keys=frozenset({key})))
            if items and items[0].incomplete:
                return items[0], False
            page, _ = self._page(ctx, items[:1], 1)
            return (page[0] if page else None), False

        answer = self._answer(scope, wait, build)
        if answer.available and isinstance(answer.data, RecordItem) and answer.data.incomplete:
            return ReadAnswer.unavailable(f"entry {key!r} is incomplete: a chunk is missing")
        return answer

    def run(self, scope: Scope, run_id: str, *, wait: WaitMark | None = None) -> ReadAnswer:
        """One run: its accept and terminal entries, the question of its bridge turn,
        the answer text of its session and its tool-call count."""

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            facts = self._runs(ctx, [run_id]).get(run_id)
            return self._run_summary(ctx, facts), False

        return self._answer(scope, wait, build)

    def run_timeline(
        self, scope: Scope, run_id: str, *, wait: WaitMark | None = None
    ) -> ReadAnswer:
        """The timeline of the run's session, complete only once its transcript is in."""

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            facts = self._runs(ctx, [run_id]).get(run_id)
            if facts is None:
                return None, False
            return self._page(ctx, facts["timeline"], self.limits.rows)

        return self._answer(scope, wait, build)

    def recent_audit(
        self,
        scope: Scope,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        surface: str | None = None,
        written_by: str | None = None,
        wait: WaitMark | None = None,
    ) -> ReadAnswer:
        """The newest audit entries, optionally of one surface.

        *written_by* keeps only the entries whose trusted ``identity`` is that
        identity; attribution and on-behalf rows never widen it.
        """

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            items = self._collect(
                ctx,
                ScopeFilter(kinds=frozenset({KIND_AUDIT})),
                Query(since=since, order=ORDER_NEWEST_FIRST),
            )
            if surface is not None:
                items = [i for i in items if i.entry.body.get(SURFACE_FIELD) == surface]
            if written_by is not None:
                items = [i for i in items if i.entry.identity == written_by]
            return self._page(ctx, items, self._rows(limit, self.limits.rows))

        return self._answer(scope, wait, build)

    def bridge_history(
        self,
        scope: Scope,
        *,
        conversation: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
        wait: WaitMark | None = None,
    ) -> ReadAnswer:
        """The newest bridge turns (at most :data:`BRIDGE_TURNS` by default), each
        with its run's status and answer text."""

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            turns = self._collect(
                ctx,
                ScopeFilter(kinds=frozenset({KIND_BRIDGE_TURN})),
                Query(since=since, order=ORDER_NEWEST_FIRST),
            )
            if conversation is not None:
                turns = [
                    t for t in turns if t.entry.body.get(BRIDGE_CONVERSATION_FIELD) == conversation
                ]
            rows = (
                min(BRIDGE_TURNS, self.limits.rows)
                if limit is None
                else self._rows(limit, self.limits.rows)
            )
            head, more = turns[:rows], len(turns) > rows
            facts = self._runs(ctx, (t.entry.run_id for t in head if t.entry.run_id), turns)
            history: list[dict[str, Any]] = []
            for turn in head:
                if not self._admit(ctx, turn):
                    return history, False
                run = facts.get(turn.entry.run_id) if turn.entry.run_id else None
                history.append({"turn": turn, "run": self._run_summary(ctx, run)})
            return history, more

        return self._answer(scope, wait, build)

    def entries(
        self,
        scope: Scope,
        kind: str,
        since: datetime | None = None,
        limit: int | None = None,
        *,
        source: str | None = None,
        wait: WaitMark | None = None,
    ) -> ReadAnswer:
        """The newest entries of one *kind* (feedback lists, hook debug).

        *source* keeps only the entries whose body names that ``source``; the
        match is made before *limit* counts.
        """
        check_kind(kind)

        def build(ctx: _Ctx) -> tuple[Any, bool]:
            items = self._collect(
                ctx,
                ScopeFilter(kinds=frozenset({kind})),
                Query(since=since, order=ORDER_NEWEST_FIRST),
            )
            if source is not None:
                items = [i for i in items if i.entry.body.get(SOURCE_FIELD) == source]
            return self._page(ctx, items, self._rows(limit, self.limits.rows))

        return self._answer(scope, wait, build)
