"""The per-operator session ledger: which transcripts each session key has been on.

A session key outlives its conversations. A ``/clear`` moves the key to a fresh
transcript while every artifact saved under it keeps carrying the key, so the key
alone cannot say which conversation an artifact came from. The ledger is the
history that answers it: an append-only list of ``(session_key, transcript_id,
from_ts)`` rows, one per move the hub observed, in the order it observed them.
:func:`osprey.stores.artifact_store.owner_of` reads it to place an artifact under
the transcript its key was on when the artifact was first saved.

**One document, two readers.** The hub writes it (every turn report, every chat
turn end, every spawn); the artifacts app only reads it. Both find it the same
way, through the caller passing :func:`osprey_connectors.posture_store.state_dir`
— the rule the transcript map and the control-context record use — so this module
takes the directory as a parameter and resolves nothing itself. That keeps it a
pure function of its inputs: the artifacts app and the hub share one reader with
no app imported on either side.

**Append-only, never repeating.** A row whose transcript equals the key's latest
row is not appended, so a hook reporting the same transcript every turn costs no
write. :func:`append` answers whether it wrote, which is what lets each write
point send exactly one ``session_transcript`` frame per actual move.

**One exception to append-only.** Deleting a session takes its transcript out of
the operator's history: :func:`remove_transcript` drops every row naming it, and
the session meta's tombstone keeps it from being written back.

**A damaged file is not an outage.** :func:`load` answers an empty ledger for a
file that is unreadable, not JSON or the wrong shape, and logs it; the next
:func:`append` rewrites the file whole from what survived. The affected
artifacts then have no owner rather than failing a listing.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from osprey_connectors.safety_file import atomic_replace

logger = logging.getLogger(__name__)

__all__ = [
    "EPOCH",
    "STORE_FILENAME",
    "Ledger",
    "LedgerRow",
    "append",
    "append_many",
    "exists",
    "ledger_path",
    "load",
    "load_checked",
    "remove_transcript",
    "write_empty",
]

#: The document's name, beside ``session-transcripts.json`` and the session meta.
STORE_FILENAME = "session-ledger.json"

#: The ``from_ts`` of a key's first transcript when the moment is unknown (the
#: backfill). A key's first row has an open lower bound anyway, so any time at or
#: before every real save is the same statement.
EPOCH = "1970-01-01T00:00:00+00:00"

# One writer process (the hub) appends from the event loop and from worker
# threads; the read-modify-write must not interleave inside it.
_WRITE_LOCK = threading.Lock()


class LedgerRow(NamedTuple):
    """One observed move: *session_key* was on *transcript_id* from *from_ts* on."""

    session_key: str
    transcript_id: str
    from_ts: str


@dataclass(frozen=True)
class Ledger:
    """The ledger as read: rows in append order, with the look-ups callers need."""

    rows: Sequence[LedgerRow] = field(default_factory=tuple)

    def __iter__(self) -> Iterator[LedgerRow]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def keys(self) -> list[str]:
        """Every session key with at least one row, in first-seen order."""
        return list(dict.fromkeys(row.session_key for row in self.rows))

    def current(self, key: str) -> str | None:
        """The transcript *key*'s latest row names, or ``None`` for an unseen key."""
        for row in reversed(self.rows):
            if row.session_key == key:
                return row.transcript_id
        return None

    def key_of(self, transcript_id: str) -> str | None:
        """The key of the latest row naming *transcript_id*, or ``None``."""
        for row in reversed(self.rows):
            if row.transcript_id == transcript_id:
                return row.session_key
        return None

    def transcripts(self) -> list[str]:
        """Every transcript with at least one row, in first-seen order."""
        return list(dict.fromkeys(row.transcript_id for row in self.rows))


def ledger_path(state_dir: Path | None) -> Path | None:
    """Where the ledger lives under *state_dir*, or ``None`` with no location."""
    return None if state_dir is None else Path(state_dir) / STORE_FILENAME


def exists(state_dir: Path | None) -> bool:
    """Whether a ledger file is present (the backfill runs only when it is not)."""
    path = ledger_path(state_dir)
    return path is not None and path.exists()


def aware_time(value: object) -> datetime | None:
    """*value* (an ISO-8601 string or a datetime) as an aware datetime, else ``None``."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo is not None else None


def _iso(value: str | datetime) -> str | None:
    """*value* as an aware ISO-8601 string, or ``None`` when it is not aware time."""
    parsed = aware_time(value)
    return None if parsed is None else parsed.isoformat()


def _row(raw: object) -> LedgerRow | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    key, transcript_id, from_ts = raw
    if not (isinstance(key, str) and key and isinstance(transcript_id, str) and transcript_id):
        return None
    stamp = _iso(from_ts) if isinstance(from_ts, str) else None
    return None if stamp is None else LedgerRow(key, transcript_id, stamp)


def load(state_dir: Path | None) -> Ledger:
    """Read the ledger under *state_dir*; never raises.

    No location or no file is an empty ledger, silently. A file that cannot be
    read, is not JSON or is not ``{"rows": [...]}`` is an empty ledger too, and is
    logged: the next :func:`append` rewrites it. Malformed rows inside a good
    document are dropped one by one, so one bad row does not cost the rest.
    """
    return _read(state_dir)[0]


def load_checked(state_dir: Path | None) -> tuple[Ledger, bool]:
    """``(ledger, available)`` for the ledger under *state_dir*; never raises.

    The reader for callers that must not mistake an unreadable ledger for an
    empty one (an empty ledger silently re-owns every artifact). No location, a
    file that cannot be read, is not JSON or has no ``rows`` list answers an
    empty ledger and ``False``. A missing file is an empty, available ledger: no
    session has been recorded yet. A good document answers its rows and ``True``,
    malformed rows dropped one by one as in :func:`load`.
    """
    return _read(state_dir, no_location_available=False)


def _read(state_dir: Path | None, *, no_location_available: bool = True) -> tuple[Ledger, bool]:
    path = ledger_path(state_dir)
    if path is None:
        return Ledger(), no_location_available
    try:
        text = path.read_text()
    except FileNotFoundError:
        return Ledger(), True
    except OSError:
        logger.warning("The session ledger at %s cannot be read; treating it as empty", path)
        return Ledger(), False
    try:
        document = json.loads(text)
    except ValueError:
        logger.warning("The session ledger at %s is not JSON; treating it as empty", path)
        return Ledger(), False
    raw_rows = document.get("rows") if isinstance(document, dict) else None
    if not isinstance(raw_rows, list):
        logger.warning("The session ledger at %s has no rows list; treating it as empty", path)
        return Ledger(), False
    rows = tuple(row for row in (_row(raw) for raw in raw_rows) if row is not None)
    return Ledger(rows), True


#: Read by the artifacts app and the agent's MCP processes, a dispatch worker's among
#: them under another uid, through the state directory's setgid group.
_LEDGER_FILE_MODE = 0o640


def _write(path: Path, rows: Sequence[LedgerRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"rows": [list(row) for row in rows]}, indent=2)
    atomic_replace(path, text.encode("utf-8"), mode=_LEDGER_FILE_MODE, sync="best_effort")


def append_many(
    state_dir: Path | None, entries: Iterable[tuple[str, str, str | datetime]]
) -> list[LedgerRow]:
    """Append each ``(key, transcript_id, from_ts)`` in order, in one write.

    Each entry follows :func:`append`'s rules against the rows before it
    (including the ones appended earlier in the same call). Returns the rows
    actually appended and writes nothing when none were.

    A failed write is logged and answers ``[]``: the ledger is best-effort
    durable, like the transcript map, and the caller must not send a frame for a
    move nobody can read back.
    """
    path = ledger_path(state_dir)
    if path is None:
        return []
    with _WRITE_LOCK:
        rows = list(load(state_dir).rows)
        latest = {}
        for row in rows:
            latest[row.session_key] = row.transcript_id
        added: list[LedgerRow] = []
        for key, transcript_id, from_ts in entries:
            stamp = _iso(from_ts)
            if not key or not transcript_id or stamp is None:
                logger.debug("Ignoring a ledger row with a blank id or a naive time")
                continue
            if latest.get(key) == transcript_id:
                continue
            row = LedgerRow(key, transcript_id, stamp)
            rows.append(row)
            added.append(row)
            latest[key] = transcript_id
        if not added:
            return []
        try:
            _write(path, rows)
        except OSError:
            logger.warning("Could not persist the session ledger to %s", path, exc_info=True)
            return []
    return added


def append(
    state_dir: Path | None, key: str, transcript_id: str, from_ts: str | datetime | None = None
) -> bool:
    """Record that *key* moved to *transcript_id* at *from_ts* (default: now).

    Returns ``False`` without writing when there is no location, when the key's
    latest row already names *transcript_id*, when an id is blank or the time is
    not aware, or when the write fails (logged). ``True`` means a row is on disk.
    """
    stamp = datetime.now(UTC) if from_ts is None else from_ts
    return bool(append_many(state_dir, [(key, transcript_id, stamp)]))


def remove_transcript(state_dir: Path | None, transcript_id: str) -> int:
    """Drop every row naming *transcript_id* (a deleted session); the count dropped.

    Each key keeps its other rows, so a key that left the transcript is current
    for its later one, and one that had moved to it is current for its earlier
    one again. A failed write is logged and answers ``0``.
    """
    path = ledger_path(state_dir)
    if path is None or not transcript_id:
        return 0
    with _WRITE_LOCK:
        rows = list(load(state_dir).rows)
        kept = [row for row in rows if row.transcript_id != transcript_id]
        if len(kept) == len(rows):
            return 0
        try:
            _write(path, kept)
        except OSError:
            logger.warning("Could not persist the session ledger to %s", path, exc_info=True)
            return 0
    return len(rows) - len(kept)


def write_empty(state_dir: Path | None) -> bool:
    """Create an empty ledger file if none exists (marks the backfill as done)."""
    path = ledger_path(state_dir)
    if path is None:
        return False
    with _WRITE_LOCK:
        if path.exists():
            return False
        try:
            _write(path, [])
        except OSError:
            logger.warning("Could not create the session ledger at %s", path, exc_info=True)
            return False
    return True
