"""Durable inbox spool for Slack events that have been acked but not yet handled.

Socket Mode wants each envelope acked within about three seconds, while
``BridgeRuntime.handle_event`` claims an event and then blocks for the whole run.
An ack is a promise that Slack will not redeliver, so the event has to be on disk
before the ack goes out: the ingest thread :meth:`Inbox.put` s it, acks only when
that returned ``True``, and the serve loop :meth:`Inbox.remove` s it once
``handle_event`` has returned. A crash anywhere in between leaves the record, and
the next start replays it (the engine's dedup turns an already-claimed event into
a ``duplicate``; reconcile finishes a run that was in flight).

One file per record, named from the Events API ``event_id``, in a directory on the
bridge-owned state volume (the same volume as ``DEDUP_PATH``). What the store
guarantees, and deliberately nothing else:

  * **A put that returned True is on disk.** Written with
    :func:`~osprey_connectors.safety_file.atomic_replace` (temporary file, fsync,
    rename, directory fsync, ``sync="required"``). A put that cannot be made
    durable — the cap is reached, or the disk refused — returns ``False`` and the
    caller withholds the ack, so Slack redelivers later. Nothing is dropped
    silently.
  * **The cap counts live records only** (default 64). A redelivered ``event_id``
    already in the inbox is accepted without rewriting it, even at the cap: the
    record it promises is already durable.
  * **A record that keeps failing stops being retried.** :meth:`Inbox.note_failure`
    counts attempts in the record itself (so they survive a restart) and moves the
    record to ``quarantine/`` at ``max_attempts``. Quarantine is never deleted; it
    is the operator's to read.
  * **Unreadable records do not block the rest.** :meth:`Inbox.pending` moves a file
    it cannot read or parse to ``quarantine/`` and carries on.
  * **Oldest first.** Records carry a sequence number that continues across
    restarts.

One lock guards every operation, so the ingest thread and the serve pool can share
one instance.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from osprey_connectors.safety_file import atomic_replace

log = logging.getLogger(__name__)

QUARANTINE_DIR = "quarantine"
_SUFFIX = ".json"
_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-")


def _file_name(key: str) -> str:
    """Map ``key`` to a file name, injectively: ``[A-Za-z0-9-]`` passes through and
    every other byte (``_`` included) becomes ``_xx``. Real ``event_id`` values are
    alphanumeric and so stay readable; the name never starts with a dot (dot names
    are ``atomic_replace``'s temporary files) and never contains a separator."""
    if not key:
        raise ValueError("inbox key must be non-empty")
    out = []
    for ch in key:
        if ch in _SAFE:
            out.append(ch)
        else:
            out.extend(f"_{b:02x}" for b in ch.encode("utf-8"))
    return "".join(out) + _SUFFIX


#: Read back only by this bridge, from its own state volume, when it replays or quarantines.
_RECORD_FILE_MODE = 0o600


class Inbox:
    """A directory of one-file-per-event records; see the module docstring."""

    def __init__(self, directory: str | Path, cap: int = 64, max_attempts: int = 3):
        self.directory = Path(directory)
        self.cap = cap
        self.max_attempts = max_attempts
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._next_seq = 1 + max((self._peek_seq(p) for p in self._record_paths()), default=0)

    # -- public API -----------------------------------------------------------

    def put(self, key: str, payload: Any) -> bool:
        """Persist ``payload`` under ``key``; ``True`` only once it is durable."""
        path = self.directory / _file_name(key)
        with self._lock:
            try:
                if path.exists():
                    return True
                if len(self._record_paths()) >= self.cap:
                    log.warning("slack inbox full (%d records); not accepting %s", self.cap, key)
                    return False
                record = {"key": key, "seq": self._next_seq, "attempts": 0, "payload": payload}
                atomic_replace(
                    path,
                    json.dumps(record).encode("utf-8"),
                    mode=_RECORD_FILE_MODE,
                    sync="required",
                )
            except OSError as exc:
                log.warning("slack inbox could not persist %s: %s", key, exc)
                return False
            self._next_seq += 1
            return True

    def remove(self, key: str) -> None:
        """Drop the record for ``key`` (handled). Unknown keys are ignored."""
        path = self.directory / _file_name(key)
        with self._lock:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                # The record stays and is replayed; dedup makes that a duplicate.
                log.warning("slack inbox could not remove %s: %s", key, exc)

    def note_failure(self, key: str) -> None:
        """Count one failed attempt; quarantine the record at ``max_attempts``."""
        path = self.directory / _file_name(key)
        with self._lock:
            record = self._read(path)
            if record is None:
                if path.exists():
                    self._quarantine(path)
                return
            record["attempts"] = int(record.get("attempts", 0)) + 1
            if record["attempts"] >= self.max_attempts:
                log.warning(
                    "slack inbox record %s failed %d times; quarantined",
                    key,
                    record["attempts"],
                )
                self._write_then_quarantine(path, record)
                return
            try:
                atomic_replace(
                    path,
                    json.dumps(record).encode("utf-8"),
                    mode=_RECORD_FILE_MODE,
                    sync="required",
                )
            except OSError as exc:
                log.warning("slack inbox could not record a failure of %s: %s", key, exc)

    def pending(self) -> list[tuple[str, Any]]:
        """Every live record as ``(key, payload)``, oldest first. Files that cannot
        be read or parsed are quarantined and skipped."""
        with self._lock:
            found: list[tuple[int, str, str, Any]] = []
            for path in self._record_paths():
                record = self._read(path)
                if record is None:
                    self._quarantine(path)
                    continue
                seq = record.get("seq")
                found.append(
                    (
                        seq if isinstance(seq, int) else 0,
                        path.name,
                        str(record.get("key") or path.name.removesuffix(_SUFFIX)),
                        record["payload"],
                    )
                )
            found.sort(key=lambda item: (item[0], item[1]))
            return [(key, payload) for _, _, key, payload in found]

    # -- internals (caller holds the lock, except in __init__) ------------------

    def _record_paths(self) -> list[Path]:
        return [
            p
            for p in self.directory.iterdir()
            if p.name.endswith(_SUFFIX) and not p.name.startswith(".") and p.is_file()
        ]

    @staticmethod
    def _read(path: Path) -> dict | None:
        """The record at ``path``, or ``None`` when it is missing, unreadable or
        not a record."""
        try:
            record = json.loads(path.read_bytes())
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict) or "payload" not in record:
            return None
        return record

    def _peek_seq(self, path: Path) -> int:
        record = self._read(path)
        seq = record.get("seq") if record else None
        return seq if isinstance(seq, int) else 0

    def _write_then_quarantine(self, path: Path, record: dict) -> None:
        try:
            atomic_replace(
                path, json.dumps(record).encode("utf-8"), mode=_RECORD_FILE_MODE, sync="best_effort"
            )
        except OSError as exc:
            log.warning("slack inbox could not update %s before quarantine: %s", path.name, exc)
        self._quarantine(path)

    def _quarantine(self, path: Path) -> None:
        """Move ``path`` into ``quarantine/`` under a name that never collides."""
        target_dir = self.directory / QUARANTINE_DIR
        stem = path.name.removesuffix(_SUFFIX)
        target = target_dir / f"{stem}.{uuid.uuid4().hex}{_SUFFIX}"
        try:
            target_dir.mkdir(exist_ok=True)
            os.replace(path, target)
        except OSError as exc:
            log.warning("slack inbox could not quarantine %s: %s", path.name, exc)
        else:
            log.warning("slack inbox quarantined %s as %s", path.name, target.name)
