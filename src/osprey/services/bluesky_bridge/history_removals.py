"""The runs an operator has removed from OSPREY's history view.

The queue server's history is a list the manager will clear whole
(``history_clear``) or trim from an entry back to the oldest, but it has no
way to drop ONE entry from the middle. OSPREY's `GET /runs` is already its
own view of that list — only items the bridge enqueued appear there — so a
single-entry removal is recorded here, on the OSPREY side, and the runs
surface stops reporting the run: absent from `GET /runs`, 404 from
`GET /runs/{id}`, invisible to the panel and to the agent's tools alike.
The manager's own list is untouched, and the run's data stays in Tiled.

**Persistence.** The set lives in one JSON file inside the bridge's writable
directory (``session_dir.py`` resolves it), so a removal survives a bridge
restart. The posture is exactly the session-plan store's: durable across a
restart, lost on a container rebuild unless the deploy points
``BLUESKY_SESSION_PLAN_DIR`` at a mounted path.

**Pruning.** An id whose run the manager no longer holds — history cleared
out-of-band, or by `DELETE /history` — is dropped the next time the history
is read, so the set never grows past the history it masks.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from pathlib import Path

from .session_dir import resolve_session_plan_dir

logger = logging.getLogger(__name__)

_FILENAME = "removed-runs.json"


class RemovedRuns:
    """The persisted set of run ids removed from the history view."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._ids: set[str] = self._load()

    def _load(self) -> set[str]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return set()
        except (OSError, ValueError) as exc:
            logger.warning("removed-runs file %s unreadable, starting empty: %s", self._path, exc)
            return set()
        if not isinstance(raw, list):
            return set()
        return {value for value in raw if isinstance(value, str)}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(sorted(self._ids)))
        except OSError as exc:
            # The in-memory set is still the truth for this process; only the
            # restart guarantee is lost, and that is worth a log line.
            logger.warning("could not persist removed runs to %s: %s", self._path, exc)

    def __contains__(self, run_id: object) -> bool:
        return run_id in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._ids:
                return
            self._ids.add(run_id)
            self._save()

    def clear(self) -> None:
        with self._lock:
            if not self._ids:
                return
            self._ids.clear()
            self._save()

    def prune(self, present: Iterable[str | None]) -> None:
        """Forget every id the manager's history no longer carries."""
        keep = {run_id for run_id in present if isinstance(run_id, str)}
        with self._lock:
            stale = self._ids - keep
            if not stale:
                return
            self._ids -= stale
            self._save()


_store: RemovedRuns | None = None


def removed_runs() -> RemovedRuns:
    """The process's one store, built on first use from the bridge's writable directory."""
    global _store
    if _store is None:
        _store = RemovedRuns(resolve_session_plan_dir() / _FILENAME)
    return _store


def _clear() -> None:
    """Test hook: forget the process-level store so the next call rebuilds it."""
    global _store
    _store = None
