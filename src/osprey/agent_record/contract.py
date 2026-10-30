"""The record contract — the one interface every writer, ingest path, backend and reader shares.

**Entries.** The record is a set of immutable :class:`Entry` rows. The unit of
uniqueness is ``(key, chunk.index)`` on every backend: a backend checks whether
that pair already exists before it inserts, so a duplicate is never stored and
is counted in :class:`AppendResult` instead. Nothing is updated, merged or
re-attributed after append. Keys are deterministic per source and start with
the trusted source identity (:func:`transcript_key`, :func:`otel_key`,
:func:`spool_key`), so restarts, resumed sessions and transcript rewrites never
produce a second copy and no sender can collide with another sender's keys. The
two run keys (:func:`run_key`) are the one exception: a run's accept and
terminal entries are keyed by the run, so the startup sweep and the real
outcome share one terminal key and the first write wins.

**Kinds.** :data:`KINDS` is the only list of kinds. Every ingest route accepts
a fixed subset (:data:`ROUTE_KINDS`), and two kinds are further restricted to
one class of source (:data:`SOURCE_CLASS_KINDS`): ``run`` only from dispatch
identities, ``bridge_turn`` only from bridge identities. A kind a source may not
send is refused at ingest (:func:`kind_allowed`), and a refused spool line
still counts as consumed.

**Chunks.** A body larger than a backend's single-entry limit is split into
ordered chunks that share the key and differ in ``chunk.index``
(:func:`split_entry`). The chunk set whose chunk 0 was stored first defines
``total`` and the content; an index outside that set is ignored and counted,
and a set with a missing chunk is reported incomplete, never as a whole body
(:func:`reassemble`). Reassembly is byte-identical to the original body.

**Secrets.** Every ingest path passes each body through :func:`scrub` before
append, so no reader can ever return a secret.

**Ingest state is not a kind.** Transcript offsets, spool checkpoints and the
refused-line records live in the backend's separate state namespace
(``get_state``/``put_state``, last write wins; ``count_state`` counts the
names under a prefix, optionally only those stamped since a time), named by
:func:`transcript_state_name`, :func:`spool_state_name` and
:func:`refused_state_name`. Only the agent-record process writes it; no spool
line, OTLP event or transcript line can reach it.

**Scope** (:class:`Own`, :class:`All`, :class:`Reader`, :class:`Source`) is
applied by the reader, never by a backend. Which credential holds which scope
is fixed in :mod:`osprey.agent_record.identities`.

This module is **schema only**: it imports nothing outside the standard
library, opens no file and holds no backend code.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TypeGuard, runtime_checkable

# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------

KIND_PROMPT = "prompt"
KIND_ANSWER = "answer"
KIND_THINKING = "thinking"
KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"
KIND_SUBAGENT = "subagent"
KIND_USAGE = "usage"
KIND_ERROR = "error"
#: A tool decision (allow, deny, ask) reported by the CLI's OTLP events.
KIND_DECISION = "decision"
KIND_AUDIT = "audit"
KIND_RUN = "run"
KIND_BRIDGE_TURN = "bridge_turn"
KIND_FEEDBACK = "feedback"
KIND_DEBUG = "debug"
KIND_API_CALL = "api_call"

#: The kind registry — the only list of kinds. Reader scopes, named-reader
#: configuration and the history page all read it.
KINDS: tuple[str, ...] = (
    KIND_PROMPT,
    KIND_ANSWER,
    KIND_THINKING,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    KIND_SUBAGENT,
    KIND_USAGE,
    KIND_ERROR,
    KIND_DECISION,
    KIND_AUDIT,
    KIND_RUN,
    KIND_BRIDGE_TURN,
    KIND_FEEDBACK,
    KIND_DEBUG,
    KIND_API_CALL,
)
_KIND_SET = frozenset(KINDS)

#: The three ingest routes an entry can arrive by.
ROUTE_TRANSCRIPT = "transcript"
ROUTE_OTLP = "otlp"
ROUTE_SPOOL = "spool"
ROUTES = frozenset({ROUTE_TRANSCRIPT, ROUTE_OTLP, ROUTE_SPOOL})

#: The service route that answers each source's health.
HEALTH_SOURCES_PATH = "/health/sources"
#: The service route that runs the catch-up export on demand.
EXPORT_PATH = "/jobs/export"
#: How long a source may be live or hold a backlog without a commit before it is stopped.
STOPPED_AFTER = timedelta(minutes=5)
#: How far back :data:`HEALTH_SOURCES_PATH` counts a source's stored refusals.
REFUSALS_WINDOW = timedelta(hours=24)

#: The kinds each ingest route may produce. The transcript carries the content
#: of a session; OTLP carries only what the transcript lacks; the spool
#: carries the facts OSPREY components own. A receiver-owned spool directory
#: (:data:`OTLP_SPOOL_DIR`) is ingested under :data:`ROUTE_OTLP`.
ROUTE_KINDS: Mapping[str, frozenset[str]] = {
    ROUTE_TRANSCRIPT: frozenset(
        {
            KIND_PROMPT,
            KIND_ANSWER,
            KIND_THINKING,
            KIND_TOOL_CALL,
            KIND_TOOL_RESULT,
            KIND_SUBAGENT,
            KIND_DEBUG,
        }
    ),
    ROUTE_OTLP: frozenset({KIND_DECISION, KIND_USAGE, KIND_ERROR}),
    ROUTE_SPOOL: frozenset(
        {KIND_AUDIT, KIND_RUN, KIND_BRIDGE_TURN, KIND_FEEDBACK, KIND_DEBUG, KIND_API_CALL}
    ),
}

#: Classes of source identity, as :func:`osprey.agent_record.identities.identity_class`
#: assigns them.
SOURCE_TERMINAL = "terminal"
SOURCE_DISPATCH = "dispatch"
SOURCE_BRIDGE = "bridge"
SOURCE_SIDECAR = "sidecar"
SOURCE_HOST = "host"
SOURCE_HISTORY = "history"
SOURCE_OPERATOR = "operator"
SOURCE_READER = "reader"
#: Any other recording service (bluesky-web, a lane queueserver, …).
SOURCE_SERVICE = "service"
SOURCE_CLASSES = frozenset(
    {
        SOURCE_TERMINAL,
        SOURCE_DISPATCH,
        SOURCE_BRIDGE,
        SOURCE_SIDECAR,
        SOURCE_HOST,
        SOURCE_HISTORY,
        SOURCE_OPERATOR,
        SOURCE_READER,
        SOURCE_SERVICE,
    }
)

#: Kinds only one class of source may send. Every kind not named here is open
#: to any source its route admits.
SOURCE_CLASS_KINDS: Mapping[str, frozenset[str]] = {
    KIND_RUN: frozenset({SOURCE_DISPATCH}),
    KIND_BRIDGE_TURN: frozenset({SOURCE_BRIDGE}),
}

#: The source classes whose spool may carry the entries of their own
#: transcripts, under the transcript route and keys (:func:`forwarded_transcript`):
#: a host session forwards the transcripts Claude Code writes for the account it
#: runs as, which no container reads.
TRANSCRIPT_FORWARDING_CLASSES = frozenset({SOURCE_HOST})


def kind_allowed(kind: str, *, route: str, source_class: str) -> bool:
    """Whether a source of *source_class* may send *kind* over *route*.

    Raises:
        ValueError: If *route* or *source_class* is not a known name.
    """
    if route not in ROUTES:
        raise ValueError(f"unknown ingest route {route!r}; the routes are {sorted(ROUTES)}")
    if source_class not in SOURCE_CLASSES:
        raise ValueError(
            f"unknown source class {source_class!r}; the classes are {sorted(SOURCE_CLASSES)}"
        )
    if kind not in ROUTE_KINDS[route]:
        return False
    restricted = SOURCE_CLASS_KINDS.get(kind)
    return restricted is None or source_class in restricted


def check_kind(kind: str) -> str:
    """Return *kind*, or raise naming the registry when it is not a kind."""
    if kind not in _KIND_SET:
        raise ValueError(f"unknown kind {kind!r}; the kinds are {', '.join(KINDS)}")
    return kind


# ---------------------------------------------------------------------------
# Identities and tokens
# ---------------------------------------------------------------------------

#: Characters a source identity may not hold: it is a key prefix (``:``), a
#: directory name (``/``, ``\\``), a Basic login and an env-file value (``$``).
_IDENTITY_FORBIDDEN = "/\\:$\0"

#: A session id, terminal session, entry id or run id: one rule for all four.
TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def check_identity(identity: str) -> str:
    """Return *identity*, or raise when it cannot be a source identity.

    Raises:
        ValueError: If it is empty, padded, ``.``/``..``, or holds whitespace,
            ``/``, ``\\``, ``:``, ``$`` or NUL.
    """
    bad = (
        not isinstance(identity, str)
        or not identity
        or identity != identity.strip()
        or identity in (".", "..")
        or any(ch in identity for ch in _IDENTITY_FORBIDDEN)
        or any(ch.isspace() for ch in identity)
    )
    if bad:
        raise ValueError(
            f"{identity!r} cannot be a source identity: it may not be empty or hold "
            "whitespace, '/', '\\', ':' or '$'"
        )
    return identity


def is_token(value: object) -> TypeGuard[str]:
    """Whether *value* matches :data:`TOKEN_RE`."""
    return isinstance(value, str) and TOKEN_RE.match(value) is not None


def _check_token(value: str, what: str) -> str:
    if not is_token(value):
        raise ValueError(f"{what} {value!r} must match {TOKEN_RE.pattern}")
    return value


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

#: Hex digits of a content hash used inside a key.
_HASH_HEX = 32

TRANSCRIPT_KEY_PREFIX = "cc"
OTEL_KEY_PREFIX = "otel"
SPOOL_KEY_PREFIX = "sp"
RUN_KEY_PREFIX = "run"

RUN_ACCEPT = "accept"
RUN_TERMINAL = "terminal"
RUN_PHASES = frozenset({RUN_ACCEPT, RUN_TERMINAL})
#: The body field of a ``run`` spool line that names its phase.
RUN_PHASE_FIELD = "phase"
#: The body field a run accept names its session in, when the entry's own
#: ``session_id`` is empty.
RUN_SESSION_FIELD = "session_id"
#: The body field an audit entry names its surface in.
SURFACE_FIELD = "surface"

#: The OTLP attribute that restarts with every CLI process and so never
#: identifies an event's content.
OTEL_SEQUENCE_ATTRIBUTE = "event.sequence"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:_HASH_HEX]


def _key_part(value: str, what: str) -> str:
    if not isinstance(value, str) or not value or ":" in value:
        raise ValueError(f"{what} {value!r} cannot be part of a key: empty or holds ':'")
    return value


def transcript_key(
    identity: str,
    session_id: str,
    uuid: str | None,
    block_index: int,
    *,
    line: bytes | None = None,
) -> str:
    """``cc:<identity>:<sessionId>:<uuid>:<content block index>``.

    A line with no ``uuid`` is keyed by a hash of the whole line instead.

    Raises:
        ValueError: If neither *uuid* nor *line* is given, or a part is unusable.
    """
    if uuid:
        content = _key_part(uuid, "uuid")
    elif line is not None:
        content = _digest(line)
    else:
        raise ValueError("a transcript key needs the line's uuid or the line itself")
    if not isinstance(block_index, int) or block_index < 0:
        raise ValueError(f"block index {block_index!r} must be a non-negative int")
    return ":".join(
        (
            TRANSCRIPT_KEY_PREFIX,
            check_identity(identity),
            _key_part(session_id, "session id"),
            content,
            str(block_index),
        )
    )


def otel_content_id(attributes: Mapping[str, Any]) -> str:
    """A hash of every attribute except ``event.sequence``, independent of order."""
    kept = {str(k): v for k, v in attributes.items() if k != OTEL_SEQUENCE_ATTRIBUTE}
    canonical = json.dumps(kept, sort_keys=True, separators=(",", ":"), default=str)
    return _digest(canonical.encode("utf-8"))


def otel_key(
    identity: str,
    session: str,
    event_name: str,
    *,
    tool_use_id: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> str:
    """``otel:<identity>:<session>:<event name>:<content id>``.

    The content id is *tool_use_id* for tool decisions, otherwise
    :func:`otel_content_id` of *attributes*.

    Raises:
        ValueError: If neither *tool_use_id* nor *attributes* is given.
    """
    if tool_use_id:
        content = _key_part(tool_use_id, "tool_use_id")
    elif attributes is not None:
        content = otel_content_id(attributes)
    else:
        raise ValueError("an OTLP key needs the tool_use_id or the event's attributes")
    return ":".join(
        (
            OTEL_KEY_PREFIX,
            check_identity(identity),
            _key_part(session, "session"),
            _key_part(event_name, "event name"),
            content,
        )
    )


def spool_key(identity: str, entry_id: str) -> str:
    """``sp:<identity>:<entry_id>``."""
    return ":".join(
        (SPOOL_KEY_PREFIX, check_identity(identity), _check_token(entry_id, "entry_id"))
    )


def run_key(run_id: str, phase: str) -> str:
    """``run:<run_id>:accept`` or ``run:<run_id>:terminal``."""
    if phase not in RUN_PHASES:
        raise ValueError(f"run phase {phase!r} must be one of {sorted(RUN_PHASES)}")
    return ":".join((RUN_KEY_PREFIX, _check_token(run_id, "run_id"), phase))


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


class EntryInvalid(ValueError):
    """An entry that cannot be appended: a required field is missing or malformed."""


@dataclass(frozen=True)
class Chunk:
    """Position of one entry in its chunk set; ``(0, 1)`` for an unsplit body."""

    index: int = 0
    total: int = 1

    def __post_init__(self) -> None:
        if not (isinstance(self.index, int) and isinstance(self.total, int)):
            raise EntryInvalid(f"chunk ({self.index!r}, {self.total!r}) must be ints")
        if self.total < 1 or not 0 <= self.index < self.total:
            raise EntryInvalid(f"chunk ({self.index}, {self.total}) is outside its set")


UNSPLIT = Chunk(0, 1)


@dataclass(frozen=True)
class Entry:
    """One immutable fact in the record.

    ``identity`` is the trusted source identity (spool directory, OTLP login or
    transcript volume, mapped through :mod:`osprey.agent_record.identities`),
    never a value a sender wrote. ``on_behalf_of`` is kept only where the trust
    list allows it, else empty. ``inserted_at`` is the arrival time, set by
    agent-record; it is ``None`` until then. The person is never stored: the
    reader derives it.
    """

    key: str
    kind: str
    ts: datetime
    identity: str
    body: Mapping[str, Any] = field(hash=False)
    chunk: Chunk = UNSPLIT
    on_behalf_of: str = ""
    session_id: str = ""
    terminal_session: str = ""
    run_id: str = ""
    agent_id: str = ""
    parent_key: str = ""
    tool_use_id: str = ""
    inserted_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_entry(self)

    @property
    def unique_key(self) -> tuple[str, int]:
        """``(key, chunk.index)``, the unit of uniqueness on every backend."""
        return (self.key, self.chunk.index)


def run_id_of(entry: Entry) -> str:
    """The run *entry* belongs to: the run a ``run`` entry is keyed by, else its
    ``run_id`` field."""
    parts = entry.key.split(":")
    if entry.kind == KIND_RUN and len(parts) == 3 and parts[0] == RUN_KEY_PREFIX:
        return parts[1]
    return entry.run_id


def accept_session(entry: Entry) -> str:
    """The session a run accept *entry* names, or ``""`` when it names none."""
    session = entry.session_id or entry.body.get(RUN_SESSION_FIELD)
    return session if is_token(session) else ""


_OPTIONAL_STR_FIELDS = (
    "on_behalf_of",
    "session_id",
    "terminal_session",
    "run_id",
    "agent_id",
    "parent_key",
    "tool_use_id",
)


def validate_entry(entry: Entry) -> Entry:
    """Return *entry*, or raise :class:`EntryInvalid` naming what is wrong.

    An entry needs a ``key``, a registered ``kind``, a timezone-aware ``ts``, a
    usable ``identity`` and a JSON-object ``body``.
    """
    if not isinstance(entry.key, str) or not entry.key:
        raise EntryInvalid("entry has no key")
    if not isinstance(entry.kind, str) or not entry.kind:
        raise EntryInvalid(f"entry {entry.key!r} has no kind")
    if entry.kind not in _KIND_SET:
        raise EntryInvalid(
            f"entry {entry.key!r} has unknown kind {entry.kind!r}; the kinds are {', '.join(KINDS)}"
        )
    if not isinstance(entry.ts, datetime):
        raise EntryInvalid(f"entry {entry.key!r} has no ts")
    if entry.ts.tzinfo is None:
        raise EntryInvalid(f"entry {entry.key!r} has a ts without a timezone")
    if not isinstance(entry.identity, str) or not entry.identity:
        raise EntryInvalid(f"entry {entry.key!r} has no identity")
    try:
        check_identity(entry.identity)
    except ValueError as exc:
        raise EntryInvalid(f"entry {entry.key!r}: {exc}") from None
    if not isinstance(entry.body, Mapping):
        raise EntryInvalid(f"entry {entry.key!r} has a body that is not a JSON object")
    if not isinstance(entry.chunk, Chunk):
        raise EntryInvalid(f"entry {entry.key!r} has no chunk")
    for name in _OPTIONAL_STR_FIELDS:
        if not isinstance(getattr(entry, name), str):
            raise EntryInvalid(f"entry {entry.key!r} field {name} must be a string")
    if entry.inserted_at is not None and (
        not isinstance(entry.inserted_at, datetime) or entry.inserted_at.tzinfo is None
    ):
        raise EntryInvalid(f"entry {entry.key!r} has an inserted_at without a timezone")
    return entry


def parse_ts(value: Any) -> datetime:
    """A timezone-aware ``datetime`` from an ISO-8601 string or a ``datetime``.

    Raises:
        EntryInvalid: If *value* is neither, or carries no timezone.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise EntryInvalid(f"ts {value!r} is not ISO-8601") from None
    else:
        raise EntryInvalid(f"ts {value!r} is not a timestamp")
    if parsed.tzinfo is None:
        raise EntryInvalid(f"ts {value!r} has no timezone")
    return parsed


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

#: The one field of a chunk's body: its slice of the serialized whole body.
CHUNK_PART_FIELD = "part"
_CHUNK_OVERHEAD = len(json.dumps({CHUNK_PART_FIELD: ""}, separators=(",", ":")).encode())


def serialize_body(body: Mapping[str, Any]) -> str:
    """The canonical serialization chunking splits and reassembly restores.

    Key order is kept, so reassembly is byte-identical to this string.
    """
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


#: The stored entry columns, in table and export order, on every backend.
ENTRY_COLUMNS = (
    "key",
    "chunk_index",
    "chunk_total",
    "kind",
    "ts",
    "identity",
    "on_behalf_of",
    "session_id",
    "terminal_session",
    "run_id",
    "agent_id",
    "parent_key",
    "tool_use_id",
    "body",
    "inserted_at",
)


def entry_row(entry: Entry, now: datetime) -> dict[str, Any]:
    """*entry* as its stored columns, in :data:`ENTRY_COLUMNS` order.

    The chunk is split into its two columns, the body is its
    :func:`serialize_body` text, and an entry without ``inserted_at`` is
    stamped *now*. Times stay ``datetime``; a backend encodes them itself.
    """
    return {
        "key": entry.key,
        "chunk_index": entry.chunk.index,
        "chunk_total": entry.chunk.total,
        "kind": entry.kind,
        "ts": entry.ts,
        "identity": entry.identity,
        "on_behalf_of": entry.on_behalf_of,
        "session_id": entry.session_id,
        "terminal_session": entry.terminal_session,
        "run_id": entry.run_id,
        "agent_id": entry.agent_id,
        "parent_key": entry.parent_key,
        "tool_use_id": entry.tool_use_id,
        "body": serialize_body(entry.body),
        "inserted_at": entry.inserted_at or now,
    }


def entry_from_row(row: Mapping[str, Any]) -> Entry:
    """The :class:`Entry` a stored row of :data:`ENTRY_COLUMNS` holds.

    ``ts`` and ``inserted_at`` must already be aware ``datetime`` values.

    Raises:
        EntryInvalid: If the row does not make a valid entry.
        ValueError: If the body is not JSON.
    """
    return Entry(
        key=row["key"],
        kind=row["kind"],
        ts=row["ts"],
        identity=row["identity"],
        body=json.loads(row["body"]),
        chunk=Chunk(int(row["chunk_index"]), int(row["chunk_total"])),
        on_behalf_of=row["on_behalf_of"],
        session_id=row["session_id"],
        terminal_session=row["terminal_session"],
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        parent_key=row["parent_key"],
        tool_use_id=row["tool_use_id"],
        inserted_at=row["inserted_at"],
    )


def _encoded_size(body: Mapping[str, Any]) -> int:
    return len(serialize_body(body).encode("utf-8"))


def split_entry(entry: Entry, limit_bytes: int) -> list[Entry]:
    """*entry* as the ordered chunks a backend with *limit_bytes* can store.

    An entry whose serialized body fits is returned as itself, unsplit. Every
    chunk shares the key and every other field; its body is
    ``{"part": <slice>}`` and the slices concatenate to :func:`serialize_body`
    of the original body.

    Raises:
        ValueError: If *limit_bytes* cannot hold even one character per chunk.
    """
    if entry.chunk != UNSPLIT:
        raise ValueError(f"entry {entry.key!r} is already a chunk")
    if _encoded_size(entry.body) <= limit_bytes:
        return [entry]
    text = serialize_body(entry.body)
    pieces = _split_text(text, limit_bytes)
    total = len(pieces)
    return [
        replace(entry, chunk=Chunk(i, total), body={CHUNK_PART_FIELD: piece})
        for i, piece in enumerate(pieces)
    ]


def _split_text(text: str, limit_bytes: int) -> list[str]:
    budget = limit_bytes - _CHUNK_OVERHEAD

    def size(piece: str) -> int:
        return len(json.dumps(piece, ensure_ascii=False).encode("utf-8")) - 2

    pieces: list[str] = []
    pos = 0
    while pos < len(text):
        n = min(len(text) - pos, max(budget, 1))
        while n > 0:
            measured = size(text[pos : pos + n])
            if measured <= budget:
                break
            shrunk = n * budget // measured
            n = shrunk if shrunk < n else n - 1
        if n <= 0:
            raise ValueError(f"a chunk limit of {limit_bytes} bytes cannot hold one character")
        pieces.append(text[pos : pos + n])
        pos += n
    return pieces


@dataclass(frozen=True)
class Reassembled:
    """The outcome of :func:`reassemble`.

    ``body`` is ``None`` exactly when ``incomplete`` is true. ``ignored``
    counts chunks outside the defining set.
    """

    key: str
    body: Mapping[str, Any] | None = field(hash=False)
    incomplete: bool
    ignored: int
    first: Entry = field(hash=False)


def reassemble(chunks: Sequence[Entry]) -> Reassembled:
    """The whole body of one key from its stored chunks, in storage order.

    The chunk 0 stored first defines ``total``. A chunk whose index falls
    outside that set is ignored and counted; a chunk index already seen is
    ignored (the backend keeps only the first). With no chunk 0, or a missing
    index, the set is ``incomplete`` and no body is returned.

    Raises:
        ValueError: If *chunks* is empty or mixes keys.
    """
    if not chunks:
        raise ValueError("reassemble needs at least one chunk")
    key = chunks[0].key
    if any(c.key != key for c in chunks):
        raise ValueError("reassemble takes the chunks of exactly one key")
    first = next((c for c in chunks if c.chunk.index == 0), None)
    if first is None:
        return Reassembled(key, None, True, 0, chunks[0])
    total = first.chunk.total
    by_index: dict[int, Entry] = {}
    ignored = 0
    for c in chunks:
        if c.chunk.index >= total:
            ignored += 1
        elif c.chunk.index not in by_index:
            by_index[c.chunk.index] = c
    if len(by_index) < total:
        return Reassembled(key, None, True, ignored, first)
    if total == 1:
        return Reassembled(key, first.body, False, ignored, first)
    parts = []
    for i in range(total):
        part = by_index[i].body.get(CHUNK_PART_FIELD)
        if not isinstance(part, str):
            return Reassembled(key, None, True, ignored, first)
        parts.append(part)
    try:
        body = json.loads("".join(parts))
    except ValueError:
        return Reassembled(key, None, True, ignored, first)
    if not isinstance(body, dict):
        return Reassembled(key, None, True, ignored, first)
    return Reassembled(key, body, False, ignored, first)


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

#: Shorter values are never treated as secrets: masking them would mangle text.
MIN_SECRET_LEN = 8
SECRET_MASK = "***"  # noqa: S105 -- the mask that replaces a secret, holds no secret


def scrub(body: Any, secret_set: Iterable[str]) -> Any:
    """*body* with each secret replaced by ``***`` in every string leaf.

    Secrets shorter than :data:`MIN_SECRET_LEN` are ignored. The longest
    secret is masked first, so a secret that contains another is masked
    whole. Mapping keys are kept as they are. Returns a new structure; *body*
    is not changed.
    """
    secrets = sorted(
        {s for s in secret_set if isinstance(s, str) and len(s) >= MIN_SECRET_LEN},
        key=len,
        reverse=True,
    )
    if not secrets:
        return body
    return _scrub_value(body, secrets)


def _scrub_value(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            if secret in value:
                value = value.replace(secret, SECRET_MASK)
        return value
    if isinstance(value, Mapping):
        return {k: _scrub_value(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_value(v, secrets) for v in value]
    return value


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Own:
    """The entries attributed to one person."""

    person: str

    def __post_init__(self) -> None:
        if not isinstance(self.person, str) or not self.person:
            raise ValueError("Own needs a person")


@dataclass(frozen=True)
class All:
    """Every entry."""


@dataclass(frozen=True)
class Reader:
    """A named reader: the listed kinds, arrived within ``days_back`` days (``None``: all)."""

    name: str
    kinds: frozenset[str] = frozenset(KINDS)
    days_back: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Reader needs a name")
        object.__setattr__(self, "kinds", frozenset(check_kind(k) for k in self.kinds))
        if self.days_back is not None and (
            not isinstance(self.days_back, int)
            or isinstance(self.days_back, bool)
            or self.days_back < 1
        ):
            raise ValueError(f"Reader days_back {self.days_back!r} must be a positive int")


@dataclass(frozen=True)
class Narrow:
    """Restricts a :class:`Source` scope to one session or a set of run ids."""

    session_id: str | None = None
    run_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_ids", frozenset(self.run_ids))
        if (self.session_id is None) == (not self.run_ids):
            raise ValueError("Narrow names exactly one of session_id or run_ids")


@dataclass(frozen=True)
class Source:
    """What a service identity ingested itself, the run entries it wrote, and
    the run entries and session timelines of every ``run_id`` its own entries
    name, optionally narrowed."""

    identity: str
    narrow: Narrow | None = None

    def __post_init__(self) -> None:
        check_identity(self.identity)


Scope = Own | All | Reader | Source


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class RecordBackendError(Exception):
    """A backend failed; readers answer ``unavailable`` and ingest keeps its spools."""


class RecordBackendUnavailable(RecordBackendError):
    """The backend cannot be reached or opened right now."""


@dataclass(frozen=True)
class AppendResult:
    """``appended`` entries were stored; ``duplicates`` already existed and were not."""

    appended: int = 0
    duplicates: int = 0


#: The least step between two appends' ``inserted_at`` stamps.
STAMP_TICK = timedelta(microseconds=1)


def arrival_order(entry: Entry) -> tuple[datetime, datetime, str]:
    """Arrival order: ``inserted_at``, else the source time; then ts and key."""
    return (entry.inserted_at or entry.ts, entry.ts, entry.key)


def next_stamp(
    newest: datetime | None, now: datetime, entries: Iterable[Entry]
) -> tuple[datetime, datetime]:
    """The ``inserted_at`` of an append made at *now*, and the newest stamp after it.

    *newest* is the newest stamp the backend already holds (``None`` when it
    holds none). The append is stamped strictly after it even when the clock
    steps back, so stamps follow commit order. An entry that already carries
    ``inserted_at`` keeps it, and the next append is stamped after that too.
    """
    stamp = now if newest is None else max(now, newest + STAMP_TICK)
    given = [e.inserted_at.astimezone(UTC) for e in entries if e.inserted_at is not None]
    return stamp, max([stamp, *given])


@dataclass(frozen=True)
class BackendHealth:
    """One backend's own view of itself."""

    ok: bool
    backend: str
    detail: str = ""
    entries: int | None = None
    oldest_arrival_day: date | None = None
    newest_arrival_day: date | None = None


@dataclass(frozen=True)
class ScopeFilter:
    """The coarse row filter the reader derives from a scope and hands a backend.

    Empty sets mean "no restriction". The reader still applies the scope
    itself; a backend only narrows what it returns.
    """

    identities: frozenset[str] = frozenset()
    kinds: frozenset[str] = frozenset()
    session_ids: frozenset[str] = frozenset()
    run_ids: frozenset[str] = frozenset()
    arrived_since: datetime | None = None


ORDER_OLDEST_FIRST = "oldest_first"
ORDER_NEWEST_FIRST = "newest_first"


@dataclass(frozen=True)
class Query:
    """What to select inside a :class:`ScopeFilter`.

    ``keys`` and ``key_prefix`` match ``Entry.key``; ``since``/``until`` bound
    ``ts`` (``since`` inclusive, ``until`` exclusive); order is by ``ts`` then
    ``inserted_at``.
    """

    keys: frozenset[str] = frozenset()
    key_prefix: str | None = None
    tool_use_ids: frozenset[str] = frozenset()
    since: datetime | None = None
    until: datetime | None = None
    limit: int | None = None
    order: str = ORDER_OLDEST_FIRST


class ConditionOp(StrEnum):
    """The operator of a :class:`Condition`; the comparisons are spelled as SQL spells them."""

    IN = "in"
    STARTS_WITH = "starts_with"
    AT_LEAST = ">="
    BEFORE = "<"


@dataclass(frozen=True)
class Condition:
    """One predicate of a select on one stored column; each backend renders it.

    ``value`` is a non-empty ``frozenset[str]`` for :attr:`ConditionOp.IN`, a
    ``str`` for :attr:`ConditionOp.STARTS_WITH` and an aware ``datetime`` for
    the comparisons.
    """

    column: str
    op: ConditionOp
    value: Any


def select_conditions(scope_filter: ScopeFilter, query: Query) -> list[Condition]:
    """The predicates a select of *query* inside *scope_filter* applies, all ANDed."""
    conditions: list[Condition] = []

    def any_of(column: str, values: frozenset[str]) -> None:
        if values:
            conditions.append(Condition(column, ConditionOp.IN, values))

    def bound(column: str, op: ConditionOp, value: datetime | None) -> None:
        if value is not None:
            conditions.append(Condition(column, op, value))

    any_of("identity", scope_filter.identities)
    any_of("kind", scope_filter.kinds)
    any_of("session_id", scope_filter.session_ids)
    any_of("run_id", scope_filter.run_ids)
    bound("inserted_at", ConditionOp.AT_LEAST, scope_filter.arrived_since)
    any_of("key", query.keys)
    if query.key_prefix is not None:
        conditions.append(Condition("key", ConditionOp.STARTS_WITH, query.key_prefix))
    any_of("tool_use_id", query.tool_use_ids)
    bound("ts", ConditionOp.AT_LEAST, query.since)
    bound("ts", ConditionOp.BEFORE, query.until)
    return conditions


def select_order(query: Query) -> str:
    """The ``ORDER BY`` list of a select of *query*: ``ts``, then ``inserted_at``,
    then the unique key, all in the query's direction.

    Raises:
        RecordBackendError: For an order other than the two ``ORDER_*`` values.
    """
    if query.order == ORDER_OLDEST_FIRST:
        direction = "ASC"
    elif query.order == ORDER_NEWEST_FIRST:
        direction = "DESC"
    else:
        raise RecordBackendError(f"unknown select order {query.order!r}")
    return ", ".join(
        f"{column} {direction}" for column in ("ts", "inserted_at", "key", "chunk_index")
    )


@runtime_checkable
class StateStore(Protocol):
    """The ingest state namespace: last write wins, written only by agent-record."""

    def get_state(self, name: str) -> Mapping[str, Any] | None: ...

    def put_state(self, name: str, value: Mapping[str, Any]) -> None: ...

    def count_state(self, prefix: str, since: datetime | None = None) -> int:
        """How many states are named ``<prefix>…``; with *since*, only those whose
        :data:`STATE_TIME_FIELD` (written by :func:`state_time`) is at or after it."""
        ...


#: The key of a state value's time, the one :meth:`StateStore.count_state` filters on.
STATE_TIME_FIELD = "at"


def state_time(moment: datetime) -> str:
    """*moment* as a state value's time: fixed-width UTC ISO-8601 with microseconds.

    One width for every value, so a backend compares two times as text.
    """
    if moment.tzinfo is None:
        raise ValueError("a state time must be timezone-aware")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def encode_state(name: str, value: Mapping[str, Any]) -> str:
    """The stored text of the state *value* under *name*.

    Raises:
        RecordBackendError: If *name* is empty, or *value* is not a JSON mapping.
    """
    if not isinstance(name, str) or not name:
        raise RecordBackendError("state needs a name")
    if not isinstance(value, Mapping):
        raise RecordBackendError(f"state {name!r} must be a mapping, not {type(value).__name__}")
    try:
        return json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise RecordBackendError(f"state {name!r} is not JSON: {exc}") from exc


def decode_state(name: str, text: str, where: str | Path) -> Mapping[str, Any]:
    """The state value stored as *text* under *name* in the store *where*.

    Raises:
        RecordBackendError: If *text* is not a JSON object.
    """
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise RecordBackendError(f"state {name!r} in {where} is unreadable: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RecordBackendError(f"state {name!r} in {where} is not a JSON object")
    return value


@runtime_checkable
class RecordBackend(StateStore, Protocol):
    """Where the record is stored. Scope is applied by the reader, never here.

    ``append`` checks ``(key, chunk.index)`` before it inserts, so a duplicate
    is never stored and is counted, across arrival days as well as within
    them. ``delete_arrived_before`` drops whole arrival days (``inserted_at``)
    and is called only for days the archive manifest records as exported.
    Every method raises :class:`RecordBackendError` on failure.
    """

    def append(self, entries: Sequence[Entry]) -> AppendResult: ...

    def select(self, scope_filter: ScopeFilter, query: Query) -> list[Entry]: ...

    def delete_arrived_before(self, day: date) -> int: ...

    def export_parquet(self, arrival_day: date, dest: Path) -> int: ...

    def health(self) -> BackendHealth: ...


# ---------------------------------------------------------------------------
# Ingest state names (shared by the tailer, the spool consumer and the reader)
# ---------------------------------------------------------------------------

TRANSCRIPT_STATE_PREFIX = "transcript"
SPOOL_STATE_PREFIX = "spool"
REFUSED_STATE_PREFIX = "refused"


def check_projects_relpath(relpath: str) -> str:
    """Return *relpath*, or raise unless it is a plain path relative to ``projects/``."""
    if not isinstance(relpath, str) or not relpath or "\\" in relpath or "\0" in relpath:
        raise ValueError(f"transcript path {relpath!r} is not a relative POSIX path")
    path = PurePosixPath(relpath)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relpath.split("/")):
        raise ValueError(f"transcript path {relpath!r} must be relative to projects/")
    return relpath


def transcript_state_name(identity: str, relpath: str) -> str:
    """``transcript:<identity>:<path relative to projects/>``."""
    return f"{TRANSCRIPT_STATE_PREFIX}:{check_identity(identity)}:{check_projects_relpath(relpath)}"


def transcript_passed(
    state: StateStore, identity: str, relpath: str, generation: int, offset: int
) -> bool:
    """Whether *identity*'s committed position in the transcript *relpath* has
    passed a mark at ``(generation, offset)``; ``False`` when none is committed
    or it is unreadable.

    Raises:
        RecordBackendError: If the state store fails.
    """
    value = state.get_state(transcript_state_name(identity, relpath))
    if value is None:
        return False
    try:
        committed = TranscriptOffset.from_state(value)
    except ValueError:
        return False
    return committed.passes(generation, offset)


def spool_state_name(identity: str, segment: str, inode: int) -> str:
    """``spool:<identity>:<segment name>:<inode>``."""
    if parse_segment_name(segment) is None:
        raise ValueError(f"{segment!r} is not a spool segment name")
    if not isinstance(inode, int) or inode < 0:
        raise ValueError(f"inode {inode!r} must be a non-negative int")
    return f"{SPOOL_STATE_PREFIX}:{check_identity(identity)}:{segment}:{inode}"


def refused_state_name(identity: str, entry_id: str) -> str:
    """``refused:<identity>:<entry_id>`` — a spool line consumed but not stored."""
    return f"{REFUSED_STATE_PREFIX}:{check_identity(identity)}:{_check_token(entry_id, 'entry_id')}"


def _non_negative_int(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{what} {value!r} must be a non-negative int")
    return value


#: Set in a transcript state value once the tailer has read its generation to
#: the end of the file.
FULLY_READ_FIELD = "fully_read"


@dataclass(frozen=True)
class TranscriptOffset:
    """The tailer's committed position in one transcript file.

    ``generation`` increases each time the file is replaced or shrinks;
    ``offset`` is the byte offset read and committed in that generation;
    ``fully_read`` says the tailer reached the end of the file at that offset.
    The state value is ``{generation, offset}``, plus ``fully_read`` when set.
    """

    generation: int
    offset: int
    fully_read: bool = False

    def __post_init__(self) -> None:
        _non_negative_int(self.generation, "generation")
        _non_negative_int(self.offset, "offset")

    def to_state(self) -> dict[str, Any]:
        value: dict[str, Any] = {"generation": self.generation, "offset": self.offset}
        if self.fully_read:
            value[FULLY_READ_FIELD] = True
        return value

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> TranscriptOffset:
        return cls(
            value.get("generation"),  # type: ignore[arg-type]
            value.get("offset"),  # type: ignore[arg-type]
            bool(value.get(FULLY_READ_FIELD)),
        )

    def passes(self, generation: int, offset: int) -> bool:
        """Whether this position has passed a mark at ``(generation, offset)``."""
        if self.generation == generation:
            return self.offset >= offset
        return self.generation > generation and self.fully_read


@dataclass(frozen=True)
class SpoolCheckpoint:
    """The spool consumer's committed byte offset in one segment."""

    offset: int

    def __post_init__(self) -> None:
        _non_negative_int(self.offset, "offset")

    def to_state(self) -> dict[str, int]:
        return {"offset": self.offset}

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> SpoolCheckpoint:
        return cls(value.get("offset"))  # type: ignore[arg-type]


def refused_state_value(kind: str, reason: str, at: datetime) -> dict[str, str]:
    """The value recorded under :func:`refused_state_name`; *at* is when it was refused."""
    return {"kind": str(kind), "reason": str(reason), STATE_TIME_FIELD: state_time(at)}


# ---------------------------------------------------------------------------
# Spool
# ---------------------------------------------------------------------------

#: The spool root under the project, and the env var that overrides it.
SPOOL_DIR_ENV = "OSPREY_SPOOL_DIR"
#: The record off switch. Set to :data:`RECORD_OFF_VALUE`, every writer of the
#: record (the spool writer, the hook emitter, the API-call log) writes nothing
#: and touches no disk. Any other value leaves the writers on.
RECORD_OFF_ENV = "OSPREY_AGENT_RECORD_OFF"
RECORD_OFF_VALUE = "1"
#: The variable that switches Claude Code's OTLP export on.
TELEMETRY_SWITCH_ENV = "CLAUDE_CODE_ENABLE_TELEMETRY"


def reaches_a_record(key: str) -> bool:
    """Whether the environment variable *key* would carry a run into a record.

    The spool directory takes the run's spool lines; ``OTEL_*`` and the
    telemetry switch send its OTLP batches to the record's OTLP receiver.
    """
    return key.startswith("OTEL_") or key in (SPOOL_DIR_ENV, TELEMETRY_SWITCH_ENV)


def unrecorded_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* as the child of a run that records nothing receives it.

    For tooling runs (a reviewer, a benchmark), which are not the agent's
    work. Every variable that :func:`reaches_a_record` is dropped and the
    record off switch is set. A launch env is built from the project's config,
    its OTLP variables included, and the SDK overlays it on the parent's (a stdio
    MCP server takes it as its whole env), so scrubbing the parent alone would
    not keep the run out of the record.

    Returns:
        A new dict; *env* is not changed.
    """
    kept = {key: value for key, value in env.items() if not reaches_a_record(key)}
    return kept | {RECORD_OFF_ENV: RECORD_OFF_VALUE}


#: The receiver-owned directory under the spool root: ``.otlp/<login>/``.
OTLP_SPOOL_DIR = ".otlp"
#: A writer's failure log inside its own spool directory.
WRITER_ERRORS_FILE = ".errors"

OPEN_SUFFIX = ".open"
SEALED_SUFFIX = ".sealed"
#: A segment is sealed once it reaches this size …
SEGMENT_MAX_BYTES = 8 * 1024 * 1024
#: … or this age, checked before any write …
SEGMENT_MAX_AGE_S = 60
#: … or after this long without a write, and always at exit.
SEGMENT_IDLE_S = 60
#: An ``.open`` segment idle this long is reported as an abandoned writer.
ABANDONED_OPEN_S = 3600

_SEGMENT_RE = re.compile(
    r"^(?P<host>.+)-(?P<pid>[0-9]+)-(?P<start_ns>[0-9]+)(?P<suffix>\.open|\.sealed)$"
)


@dataclass(frozen=True)
class SegmentName:
    """``<host>-<pid>-<start ns>.open`` or ``….sealed``."""

    host: str
    pid: int
    start_ns: int
    sealed: bool = False

    def __post_init__(self) -> None:
        if not self.host or any(ch in self.host for ch in "/\\\0") or self.host in (".", ".."):
            raise ValueError(f"segment host {self.host!r} cannot be part of a file name")
        _non_negative_int(self.pid, "pid")
        _non_negative_int(self.start_ns, "start_ns")

    @property
    def name(self) -> str:
        suffix = SEALED_SUFFIX if self.sealed else OPEN_SUFFIX
        return f"{self.host}-{self.pid}-{self.start_ns}{suffix}"


def segment_name(host: str, pid: int, start_ns: int) -> str:
    """The name of a new, open segment."""
    return SegmentName(host, pid, start_ns).name


def sealed_name(open_name: str) -> str:
    """The name the open segment *open_name* takes when it is sealed."""
    return open_name[: -len(OPEN_SUFFIX)] + SEALED_SUFFIX


def parse_segment_name(name: str) -> SegmentName | None:
    """The parts of a segment file name, or ``None`` when *name* is not one."""
    if not isinstance(name, str) or "/" in name:
        return None
    match = _SEGMENT_RE.match(name)
    if match is None:
        return None
    try:
        return SegmentName(
            match["host"],
            int(match["pid"]),
            int(match["start_ns"]),
            sealed=match["suffix"] == SEALED_SUFFIX,
        )
    except ValueError:
        return None


class SpoolLineInvalid(ValueError):
    """A spool line that cannot be ingested."""


_SPOOL_TOKEN_FIELDS = ("session_id", "terminal_session", "run_id", "tool_use_id")
_SPOOL_TEXT_FIELDS = ("on_behalf_of", "agent_id", "parent_key")


@dataclass(frozen=True)
class SpoolLine:
    """One JSON object per spool line.

    ``{"entry_id", "kind", "ts", "key"?, "session_id"?, "terminal_session"?,
    "run_id"?, "tool_use_id"?, "on_behalf_of"?, "agent_id"?, "parent_key"?,
    "body"}``. Identity is never read from the line: it comes from the
    directory the line was found in, and an ``identity`` field in the line is
    ignored. ``agent_id`` and ``parent_key`` are kept only on a forwarded
    transcript entry (:func:`forwarded_transcript`).
    """

    entry_id: str
    kind: str
    ts: str
    body: Mapping[str, Any] = field(hash=False)
    key: str | None = None
    session_id: str | None = None
    terminal_session: str | None = None
    run_id: str | None = None
    tool_use_id: str | None = None
    on_behalf_of: str | None = None
    agent_id: str | None = None
    parent_key: str | None = None

    def __post_init__(self) -> None:
        if not is_token(self.entry_id):
            raise SpoolLineInvalid(
                f"spool line entry_id {self.entry_id!r} must match {TOKEN_RE.pattern}"
            )
        if self.kind not in _KIND_SET:
            raise SpoolLineInvalid(f"spool line {self.entry_id}: unknown kind {self.kind!r}")
        try:
            parse_ts(self.ts)
        except EntryInvalid as exc:
            raise SpoolLineInvalid(f"spool line {self.entry_id}: {exc}") from None
        if not isinstance(self.body, Mapping):
            raise SpoolLineInvalid(f"spool line {self.entry_id}: body is not a JSON object")
        if self.key is not None and (not isinstance(self.key, str) or not self.key):
            raise SpoolLineInvalid(f"spool line {self.entry_id}: key must be a non-empty string")
        for name in (*_SPOOL_TOKEN_FIELDS, *_SPOOL_TEXT_FIELDS):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise SpoolLineInvalid(f"spool line {self.entry_id}: {name} must be a string")

    def to_json(self) -> str:
        """The line as written, without a trailing newline; unset fields omitted."""
        out: dict[str, Any] = {"entry_id": self.entry_id, "kind": self.kind, "ts": self.ts}
        for name in ("key", *_SPOOL_TOKEN_FIELDS, *_SPOOL_TEXT_FIELDS):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out["body"] = dict(self.body)
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def parse_spool_line(raw: bytes | str) -> SpoolLine:
    """Parse one spool line.

    Raises:
        SpoolLineInvalid: If it is not a JSON object of the spool line schema.
    """
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SpoolLineInvalid(f"spool line is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise SpoolLineInvalid("spool line is not a JSON object")
    missing = [name for name in ("entry_id", "kind", "ts", "body") if name not in data]
    if missing:
        raise SpoolLineInvalid(f"spool line lacks {', '.join(missing)}")
    return SpoolLine(
        entry_id=data["entry_id"],
        kind=data["kind"],
        ts=data["ts"],
        body=data["body"],
        key=data.get("key"),
        session_id=data.get("session_id"),
        terminal_session=data.get("terminal_session"),
        run_id=data.get("run_id"),
        tool_use_id=data.get("tool_use_id"),
        on_behalf_of=data.get("on_behalf_of"),
        agent_id=data.get("agent_id"),
        parent_key=data.get("parent_key"),
    )


def forwarded_transcript(line: SpoolLine, identity: str) -> bool:
    """Whether *line* forwards an entry of *identity*'s own transcripts.

    Such a line carries the entry's transcript key (:func:`transcript_key`),
    which names *identity*, the identity of the directory it was found in; it is
    ingested under :data:`ROUTE_TRANSCRIPT`, and only from a source of
    :data:`TRANSCRIPT_FORWARDING_CLASSES`. The key is the one the transcript
    tailer gives the same entry, so an entry filed by either is filed once.
    """
    key = line.key
    return isinstance(key, str) and key.startswith(f"{TRANSCRIPT_KEY_PREFIX}:{identity}:")


def spool_line_of(entry: Entry, entry_id: str) -> SpoolLine:
    """The spool line that forwards the transcript *entry* of its own identity."""
    return SpoolLine(
        entry_id=entry_id,
        kind=entry.kind,
        ts=entry.ts.isoformat(),
        body=entry.body,
        key=entry.key,
        session_id=entry.session_id or None,
        tool_use_id=entry.tool_use_id or None,
        agent_id=entry.agent_id or None,
        parent_key=entry.parent_key or None,
    )


def spool_entry_key(line: SpoolLine, identity: str, *, receiver_owned: bool) -> str:
    """The key a spool line is stored under.

    The line's own ``key`` is honoured in a receiver-owned directory, and on a
    forwarded transcript entry of *identity* (:func:`forwarded_transcript`). A
    ``run`` line is keyed by its run and the phase its body names. Every other
    line is ``sp:<identity>:<entry_id>``.

    Raises:
        SpoolLineInvalid: If a receiver-owned line has no key, or a run line
            has no run id or phase.
    """
    if receiver_owned:
        if not line.key:
            raise SpoolLineInvalid(f"receiver-owned spool line {line.entry_id} has no key")
        return line.key
    if forwarded_transcript(line, identity):
        assert line.key is not None
        return line.key
    if line.kind == KIND_RUN:
        phase = line.body.get(RUN_PHASE_FIELD)
        if not line.run_id or phase not in RUN_PHASES:
            raise SpoolLineInvalid(
                f"run line {line.entry_id} needs a run_id and a body {RUN_PHASE_FIELD!r} "
                f"of {sorted(RUN_PHASES)}"
            )
        try:
            return run_key(line.run_id, phase)
        except ValueError as exc:
            raise SpoolLineInvalid(f"run line {line.entry_id}: {exc}") from None
    return spool_key(identity, line.entry_id)


def entry_from_spool_line(
    line: SpoolLine,
    *,
    identity: str,
    receiver_owned: bool,
    on_behalf_of_trusted: bool,
) -> Entry:
    """The entry a spool line becomes, before scrubbing and arrival stamping.

    *identity* is the trusted identity of the directory the line was found in.
    ``on_behalf_of`` is kept only when *on_behalf_of_trusted*, ``agent_id`` and
    ``parent_key`` only on a forwarded transcript entry; a token field that
    does not match :data:`TOKEN_RE` is dropped.
    """

    def token(value: str | None) -> str:
        return value if is_token(value) else ""

    forwarded = not receiver_owned and forwarded_transcript(line, identity)
    return Entry(
        key=spool_entry_key(line, identity, receiver_owned=receiver_owned),
        kind=line.kind,
        ts=parse_ts(line.ts),
        identity=identity,
        body=line.body,
        on_behalf_of=(line.on_behalf_of or "") if on_behalf_of_trusted else "",
        session_id=token(line.session_id),
        terminal_session=token(line.terminal_session),
        run_id=token(line.run_id),
        tool_use_id=token(line.tool_use_id),
        agent_id=token(line.agent_id) if forwarded else "",
        parent_key=(line.parent_key or "") if forwarded else "",
    )


#: What ``writer.write()`` returns: the ``entry_id`` on success, ``None`` on
#: failure (logged and reported to health). Callers test it for truthiness.
WriteResult = str | None


# ---------------------------------------------------------------------------
# Wait marks
# ---------------------------------------------------------------------------


class WaitMarkInvalid(ValueError):
    """A wait mark that does not follow the schema."""


@dataclass(frozen=True)
class TranscriptMark:
    """The transcript files of one session, each at ``(generation, byte offset)``.

    Paths are relative to ``projects/`` on both sides, never mount paths.
    """

    session_id: str
    files: Mapping[str, tuple[int, int]] = field(hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise WaitMarkInvalid("a transcript mark needs a session_id")
        if not isinstance(self.files, Mapping):
            raise WaitMarkInvalid("a transcript mark's files must be a mapping")
        normalized: dict[str, tuple[int, int]] = {}
        for path, position in self.files.items():
            try:
                check_projects_relpath(path)
                generation, offset = position
                _non_negative_int(generation, "generation")
                _non_negative_int(offset, "offset")
            except (TypeError, ValueError) as exc:
                raise WaitMarkInvalid(f"transcript mark file {path!r}: {exc}") from None
            normalized[path] = (generation, offset)
        object.__setattr__(self, "files", normalized)


@dataclass(frozen=True)
class Drain:
    """Wait until ingest has committed what the caller's scope had pending.

    The service, never the caller, resolves what that is: when the read
    arrives it snapshots the end of every spool segment and ``.otlp/`` batch
    of the identities the caller's scope sees and, with :attr:`session_id`,
    the size of that session's transcript files under the roots of those
    identities. The read waits until the consumer and the tailer have
    committed past the snapshot.
    """

    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.session_id is not None and not is_token(self.session_id):
            raise WaitMarkInvalid(f"drain session_id {self.session_id!r} is not a token")


@dataclass(frozen=True)
class WaitMark:
    """``{"drain": {"session_id": <str> | null}}`` or ``{"transcript": {...}}``.

    Exactly one of :attr:`drain` and :attr:`transcript` is set.
    """

    drain: Drain | None = None
    transcript: TranscriptMark | None = None

    def __post_init__(self) -> None:
        if (self.drain is None) == (self.transcript is None):
            raise WaitMarkInvalid("a wait mark names exactly one of drain or transcript")

    @classmethod
    def from_json(cls, data: Any) -> WaitMark:
        """Parse the JSON form, raising :class:`WaitMarkInvalid` on any deviation."""
        if isinstance(data, (str, bytes)):
            try:
                data = json.loads(data)
            except ValueError as exc:
                raise WaitMarkInvalid(f"wait mark is not JSON: {exc}") from None
        if not isinstance(data, Mapping) or len(data) != 1:
            raise WaitMarkInvalid("a wait mark is an object with one of drain or transcript")
        if "drain" in data:
            d = data["drain"]
            if not isinstance(d, Mapping) or set(d) - {"session_id"}:
                raise WaitMarkInvalid("a drain mark is an object with only a session_id")
            session_id = d.get("session_id")
            if session_id is not None and not isinstance(session_id, str):
                raise WaitMarkInvalid("a drain mark's session_id is a string or null")
            return cls(drain=Drain(session_id))
        if "transcript" in data:
            t = data["transcript"]
            if not isinstance(t, Mapping) or not isinstance(t.get("files"), Mapping):
                raise WaitMarkInvalid("a transcript mark needs session_id and files")
            files = {
                path: tuple(pos) if isinstance(pos, list) else pos
                for path, pos in t["files"].items()
            }
            return cls(transcript=TranscriptMark(t.get("session_id", ""), files))
        raise WaitMarkInvalid("a wait mark names drain or transcript")

    def to_json(self) -> dict[str, Any]:
        """The JSON form :meth:`from_json` reads back."""
        if self.drain is not None:
            return {"drain": {"session_id": self.drain.session_id}}
        assert self.transcript is not None
        return {
            "transcript": {
                "session_id": self.transcript.session_id,
                "files": {p: [g, o] for p, (g, o) in self.transcript.files.items()},
            }
        }

    def passed(self, state: StateStore, *, transcript_identity: str) -> bool:
        """Whether the tailer has passed this transcript mark.

        It passes when, for each file, the committed offset of
        *transcript_identity*'s copy is at least the mark in the same
        generation, or the file has since moved to a newer generation and that
        generation was fully read.

        Raises:
            ValueError: If this is a drain mark: only the service, which sees
                the spool and the transcript roots, resolves one.
            RecordBackendError: If the state store fails.
        """
        if self.transcript is None:
            raise ValueError("a drain mark is resolved by the service, not checked against state")
        return all(
            transcript_passed(state, transcript_identity, path, *mark)
            for path, mark in self.transcript.files.items()
        )
