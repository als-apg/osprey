"""Who withdrew queued work, and what they withdrew.

Three routes take work off the queue: `DELETE /queue/items/{uid}` drops one
pending item, `DELETE /queue/items` drops all of them, and `POST /queue/abort`
stops the plan already moving hardware. Each carries the ``X-Osprey-Owner``
header naming the person it is for, exactly as an enqueue does. This module is
where that name is kept once the route has acted, so a queue row that
disappeared has the same kind of answer to "who" as the row that appeared.

It is a RECORD, not an audit trail and not an authorization check. The routes
stay ungated — withdrawing pending work arms nothing, and the emergency halt
is gated on nothing at all — and nothing here can refuse a removal or change
what one does. A caller that names nobody is legitimate (cron fires jobs that
way, and a token-only caller never had a name to send), so an owner-less
record is a first-class shape rather than a gap.

**What a record holds.** The clock time, which of the three actions it was,
the owner who asked for it, and enough of the item to recognize it: its queue
uid, its plan name, its item type, the owner it was ENQUEUED under, and the
OSPREY run id it carried. Never the plan's arguments — they are unbounded, and
the reserved owner kwarg among them is not a plan argument at all, so this file
would be a second place it could leak from. :func:`~.queue_backend.split_owner`
is read for the enqueuing owner alone and its stripped item is discarded, which
is the same public shaping ``queue.py``'s ``_public_item`` relies on.

**Bounded.** The newest :data:`MAX_RECORDS` are kept and older ones fall off
the end. A removal is a small, frequent event and this file is read whole on
every ``GET /queue/removals``, so it is capped by count rather than allowed to
grow with the deployment's age. Losing the oldest is the right loss to take:
the question the record answers is about what just happened to the queue an
operator is looking at.

**Persistence.** One JSON file inside the bridge's writable directory
(``session_dir.py`` resolves it), newest record first, written whole on every
append. The posture is the session-plan store's and `history_removals.py`'s:
durable across a bridge restart, lost on a container rebuild unless the deploy
points ``BLUESKY_SESSION_PLAN_DIR`` at a mounted path. An unreadable file
starts the log empty rather than failing the bridge's first removal, and a
failed write is a warning — the in-memory list is still this process's truth,
and only the restart guarantee is lost.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .queue_backend import run_id_of, split_owner
from .session_dir import resolve_session_plan_dir

logger = logging.getLogger(__name__)

_FILENAME = "queue-removals.json"

#: How many records the log keeps. Everything older falls off the end.
MAX_RECORDS = 200

#: One pending item dropped by uid.
ACTION_REMOVE = "remove"

#: Every pending item dropped at once; one record per item that was dropped.
ACTION_CLEAR = "clear"

#: The plan that was running stopped where it stood.
ACTION_ABORT = "abort"

#: Every key a record carries, so a consumer can read a record without
#: testing for presence: an unknown value is ``None``, never absent.
_RECORD_KEYS = ("at", "action", "owner", "uid", "name", "item_type", "item_owner", "run_id")


def _text(value: Any) -> str | None:
    """*value* when it is a non-empty string, else ``None``.

    Queue items are whatever the enqueuer put on the wire — the queue port is
    the facility's, so an item can reach it from outside OSPREY entirely — and
    a record must not report an empty name as a name.
    """
    return value if isinstance(value, str) and value else None


def build_record(action: str, owner: str | None, item: Any = None) -> dict[str, Any]:
    """One record: the action, who asked for it, and what it took off the queue.

    *item* is the manager's own object for the row that went away, or ``None``
    when the route could not name one — a clear whose queue read did not answer,
    or an abort where the running item could not be read. Every item-shaped key
    is then ``None``, which reads as "this happened, and what it happened to is
    not recorded" rather than as a fabricated row.

    ``at`` is UTC and ISO-8601 with an explicit offset, so a reader never has to
    guess which clock the bridge kept.
    """
    record: dict[str, Any] = dict.fromkeys(_RECORD_KEYS)
    record["at"] = datetime.now(UTC).isoformat()
    record["action"] = action
    record["owner"] = owner
    if not isinstance(item, Mapping):
        return record
    item = dict(item)
    # The stripped copy is discarded: only the owner it carried is wanted, and
    # the plan arguments it still holds are exactly what must not land here.
    _, item_owner = split_owner(item)
    record["uid"] = _text(item.get("item_uid"))
    record["name"] = _text(item.get("name"))
    record["item_type"] = _text(item.get("item_type"))
    record["item_owner"] = item_owner
    record["run_id"] = run_id_of(item)
    return record


class QueueRemovals:
    """The persisted, bounded log of withdrawn queue work, newest first."""

    def __init__(self, path: Path, *, max_records: int = MAX_RECORDS) -> None:
        self._path = path
        self._max_records = max_records
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            logger.warning("queue-removals file %s unreadable, starting empty: %s", self._path, exc)
            return []
        if not isinstance(raw, list):
            return []
        # Entry-by-entry tolerance, not all-or-nothing: a file damaged in place
        # or hand-edited costs the entries it mangled, not the whole log.
        records = [entry for entry in raw if isinstance(entry, dict)]
        return records[: self._max_records]

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._records))
        except OSError as exc:
            # The in-memory list is still the truth for this process; only the
            # restart guarantee is lost, and that is worth a log line.
            logger.warning("could not persist queue removals to %s: %s", self._path, exc)

    def __len__(self) -> int:
        return len(self._records)

    def append(self, action: str, owner: str | None, item: Any = None) -> dict[str, Any]:
        """Record one withdrawal and return the record written."""
        record = build_record(action, owner, item)
        with self._lock:
            self._records.insert(0, record)
            del self._records[self._max_records :]
            self._save()
        return record

    def records(self) -> list[dict[str, Any]]:
        """Every record held, newest first, as copies."""
        with self._lock:
            return [dict(record) for record in self._records]

    def clear(self) -> None:
        with self._lock:
            if not self._records:
                return
            self._records.clear()
            self._save()


_store: QueueRemovals | None = None


def removal_log() -> QueueRemovals:
    """The process's one log, built on first use from the bridge's writable directory."""
    global _store
    if _store is None:
        _store = QueueRemovals(resolve_session_plan_dir() / _FILENAME)
    return _store


def _clear() -> None:
    """Test hook: forget the process-level log so the next call rebuilds it."""
    global _store
    _store = None
