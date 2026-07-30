"""Crash-safe dedup / run store for a bridge.

Guards against two failure modes:

  * message *redelivery* — the same inbound message arriving more than once
    (Pub/Sub redelivery, an IMAP re-poll). The message is marked processed only
    after handling completes, so a crash mid-run redelivers it; the persisted
    claim prevents re-dispatching.
  * Bridge *restart* mid-run — on startup, :meth:`reconcile` reports the runs
    that were still in flight so the caller can re-attach (poll the worker to
    terminal + repost) instead of firing a brand-new dispatch.

Backed by a small JSON file on a bridge-owned volume (persistence/locking in
:class:`~osprey.bridges.core.store.JsonFileStore`). The store is a flat
``message_id -> {"run_id": ..., "status": ..., <extra>}`` mapping. Extra metadata
(space/thread/question, sender) is permitted so a restart can repost to the
right destination; the core contract is only ``run_id`` + ``status``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .config import TERMINAL_STATUSES
from .store import JsonFileStore

# How long a terminal entry lingers before :meth:`DedupStore.claim` reaps it.
# Chosen to exceed Pub/Sub's maximum redelivery window: as long as an entry can
# still be redelivered, its dedup claim must survive to suppress a re-dispatch,
# so pruning only after 7d preserves dedup correctness. A ``retry_now`` id that
# outlives this window surfaces as "too old to retry" (its entry is gone).
PRUNE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


class DedupStore(JsonFileStore):
    def __init__(self, path: str, *, now: Callable[[], float] = time.time) -> None:
        # Injectable clock (epoch seconds) so tests drive prune ages with a fake
        # clock, matching the ``now``-seam convention used across the core.
        self._now = now
        super().__init__(path)

    # --- Internal helpers --------------------------------------------------
    def _stamp_settled_if_terminal(self, entry: dict[str, Any], status: str) -> None:
        """Record ``settled_at`` when an entry crosses into a terminal status,
        marking the moment its 7d prune clock starts."""
        if status in TERMINAL_STATUSES:
            entry["settled_at"] = self._now()

    def _prune_locked(self) -> None:
        """Drop terminal entries older than :data:`PRUNE_MAX_AGE_SECONDS`.

        Caller MUST hold ``self._lock``. Only terminal entries are considered —
        ``pending`` / ``in_flight`` / ``queued`` are live work and never touched
        (verified same-lock safe: this runs inside :meth:`claim`'s existing lock,
        so it cannot race the drain or reconcile). An entry's age is measured
        from ``settled_at`` (when it went terminal), falling back to
        ``claimed_at`` (when it was first claimed).

        A terminal entry carrying NEITHER timestamp is a pre-feature (ts-less)
        entry: rather than delete it — which would drop a still-valid dedup claim
        — it is grandfathered by stamping ``settled_at = now`` (persisted), so it
        ages out one full window later instead of vanishing on first observation.
        """
        now = self._now()
        changed = False
        for message_id in list(self._data.keys()):
            entry = self._data[message_id]
            if entry.get("status") not in TERMINAL_STATUSES:
                continue
            ts = entry.get("settled_at")
            if not isinstance(ts, (int, float)) or isinstance(ts, bool):
                ts = entry.get("claimed_at")
            if not isinstance(ts, (int, float)) or isinstance(ts, bool):
                # Pre-feature ts-less terminal entry: grandfather to now.
                entry["settled_at"] = now
                changed = True
                continue
            if now - ts > PRUNE_MAX_AGE_SECONDS:
                del self._data[message_id]
                changed = True
        if changed:
            self._flush_locked()

    # --- API ---------------------------------------------------------------
    def seen(self, message_id: str) -> bool:
        """True if we have already accepted this message id."""
        with self._lock:
            return message_id in self._data

    def claim(self, message_id: str, **meta: Any) -> bool:
        """Atomically claim a message id for processing.

        Records a ``pending`` entry (``run_id=None``) and returns ``True`` iff
        this call is the first to see ``message_id``. Concurrent redeliveries of
        the same message therefore get exactly one ``True`` — the check and the
        write happen under one lock, closing the seen()-then-record() race.

        Prunes aged-out terminal entries first (see :meth:`_prune_locked`) under
        this same lock: a redelivery of a long-since-pruned message finds its old
        entry gone and re-claims cleanly.
        """
        with self._lock:
            self._prune_locked()
            if message_id in self._data:
                return False
            entry: dict[str, Any] = {
                "run_id": None,
                "status": "pending",
                "claimed_at": self._now(),
            }
            if meta:
                entry.update(meta)
            self._data[message_id] = entry
            self._flush_locked()
            return True

    def get(self, message_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data.get(message_id)
            return dict(entry) if entry is not None else None

    def record(
        self,
        message_id: str,
        run_id: str | None,
        status: str = "in_flight",
        **meta: Any,
    ) -> None:
        """Upsert an entry. Sets ``run_id`` + ``status``; merges any ``meta``
        (e.g. space/thread) while preserving previously stored fields.

        A terminal entry is final: once a message has reached a terminal status
        (see :data:`~osprey.bridges.core.config.TERMINAL_STATUSES`) this is a
        no-op, so a late or duplicate callback cannot resurrect it into a
        non-terminal (reconcilable) state and re-dispatch an already-answered
        message on the next restart.
        """
        with self._lock:
            entry = self._data.get(message_id)
            if entry is not None and entry.get("status") in TERMINAL_STATUSES:
                return
            if entry is None:
                entry = {}
            entry["run_id"] = run_id
            entry["status"] = status
            if meta:
                entry.update(meta)
            self._stamp_settled_if_terminal(entry, status)
            self._data[message_id] = entry
            self._flush_locked()

    def update_status(self, message_id: str, status: str) -> None:
        """Set the status of an existing entry (no-op if unknown)."""
        with self._lock:
            entry = self._data.get(message_id)
            if entry is None:
                return
            entry["status"] = status
            self._stamp_settled_if_terminal(entry, status)
            self._flush_locked()

    def transition(
        self,
        message_id: str,
        from_status: str,
        to_status: str,
        **meta: Any,
    ) -> bool:
        """Compare-and-set an entry's status under the single store lock.

        Flips ``message_id`` from ``from_status`` to ``to_status`` iff the entry
        exists AND is currently in ``from_status``, merging any ``meta`` on
        success. Returns ``True`` when the flip was applied (and persisted),
        ``False`` — with no write — when the entry is absent or its status does
        not match ``from_status``.

        This is the concurrency primitive of the retry queue: two threads racing
        over the same entry (e.g. a drain flipping ``queued -> in_flight`` versus
        a coalesce dropping ``queued -> superseded``) both read the same
        ``from_status``, but only the first to run under the lock matches it and
        wins; the loser sees the already-changed status and returns ``False``.
        """
        with self._lock:
            entry = self._data.get(message_id)
            if entry is None or entry.get("status") != from_status:
                return False
            entry["status"] = to_status
            if meta:
                entry.update(meta)
            self._stamp_settled_if_terminal(entry, to_status)
            self._data[message_id] = entry
            self._flush_locked()
            return True

    def queued(self) -> dict[str, dict[str, Any]]:
        """Return entries currently parked in the retry queue (``status ==
        'queued'``). These are deliberately EXCLUDED from :meth:`reconcile` — the
        drain loop owns them, not startup reconciliation."""
        with self._lock:
            return {
                mid: dict(entry)
                for mid, entry in self._data.items()
                if entry.get("status") == "queued"
            }

    def reconcile(self) -> dict[str, dict[str, Any]]:
        """Return entries that need re-attaching on startup: NOT terminal and
        NOT ``queued``.

        ``queued`` entries are retry-queue work owned by the drain loop, so they
        are excluded here even though they are non-terminal — reconcile must not
        re-dispatch them out from under the drain. ``queued`` is intentionally
        kept OUT of :data:`~osprey.bridges.core.config.TERMINAL_STATUSES` (other
        consumers key off that frozenset); the exclusion is explicit instead."""
        with self._lock:
            return {
                mid: dict(entry)
                for mid, entry in self._data.items()
                if entry.get("status") not in TERMINAL_STATUSES and entry.get("status") != "queued"
            }
