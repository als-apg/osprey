"""Bounded live-row buffer for run data still in flight or just completed.

Task 2.2's ``GET /runs/{id}/data`` route needs a source of truth for run data
before (or in place of) a Tiled server: the scan itself runs inside this
bridge process (``runs.do_promote`` owns the runner/RunEngine), so this
module subscribes a plain-dict document callback to that same RunEngine and
keeps translated rows in a bounded, run_uid-keyed buffer. Real RunEngine
wiring lands in task 2.7 (``real-runengine-integration``) — this module is
deliberately import-clean of bluesky so it can be built and unit-tested with
synthetic documents beforehand. Generalizes the concept in BELLA's
``services/experiment_config/live_rows.py``: this version has no
GEECS-specific legacy column mapping, and explicitly RETAINS completed runs
rather than evicting at a small run count, so a read arriving after the scan
finishes still succeeds without a Tiled server.

Document handling (bluesky's plain-dict document protocol):

- ``start``: begins a new buffer for ``doc["uid"]``, marked partial.
- ``event``: each event's ``doc["data"]`` becomes one row; columns are
  discovered incrementally in first-seen order across events. A key seen for
  the first time extends the column list and backfills every already-stored
  row with ``None`` in that column, so all stored rows stay aligned to the
  current column list; a later event missing an already-known column simply
  records ``None`` for it.
- ``stop``: flips ``partial`` to ``False``. This is the ONLY thing that ends
  the "still filling in" state — a buffer that never gets a stop doc (e.g.
  the process crashed mid-scan) stays partial forever, which is the honest
  answer.

Bounding uses two independent knobs:

- ``_MAX_ROWS_PER_RUN``: a hard cap on stored rows per run, so a runaway scan
  cannot grow the buffer without limit. ``total_seen`` keeps counting every
  event past this cap — Task 2.2's ``row_count`` reports this *true* total
  even when the tail beyond the cap was never stored (a documented trade-off
  for a pathological never-ending run, not the common case).
- ``_MAX_RUNS``: number of run buffers retained at all, oldest-evicted
  (insertion-ordered ``OrderedDict``, ``move_to_end`` on every write) once
  exceeded. This is what lets a completed run's data survive to be read
  later — unlike BELLA's demo-day ``_MAX_RUNS = 4``, Phase 2 sets this much
  higher since there is no Tiled fallback yet.

The recorder itself never raises: every document is handled inside a
try/except so a recorder bug can never abort a run (mirrors the runner
seam's own safety contract) — a bug here loses live-read fidelity for that
run, not the run itself.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# Retained run buffers (oldest evicted first). Considerably higher than
# BELLA's demo-day `_MAX_RUNS = 4` since Phase 2 has no Tiled fallback yet — a
# completed run must stay readable well past its own completion.
_MAX_RUNS = 50
# Hard per-run row storage cap — a safety valve against a runaway/never-ending
# scan, not a normal-case limit (see `total_seen` above).
_MAX_ROWS_PER_RUN = 10_000

_lock = Lock()
# run_uid -> {"columns": [...], "rows": [[...], ...], "partial": bool, "total_seen": int}
_buffers: OrderedDict[str, dict[str, Any]] = OrderedDict()


def get(run_uid: str) -> dict[str, Any] | None:
    """A snapshot of the live buffer for *run_uid* (or None if unknown)."""
    with _lock:
        buf = _buffers.get(run_uid)
        if buf is None:
            return None
        return {
            "columns": list(buf["columns"]),
            "rows": [list(row) for row in buf["rows"]],
            "partial": buf["partial"],
            "total_seen": buf["total_seen"],
        }


def _clear() -> None:
    """Drop all buffers (test isolation only)."""
    with _lock:
        _buffers.clear()


class LiveRowRecorder:
    """Bluesky document callback recording rows into the bounded live buffer.

    One instance per promoted run (mirrors BELLA's contract): subscribe it to
    the RunEngine between ``reinitialize()`` and ``start_run_thread()``
    (task 2.7) so the start doc is never missed.
    """

    def __init__(self) -> None:
        self._uid: str | None = None

    def __call__(self, name: str, doc: dict[str, Any]) -> None:
        try:
            if name == "start":
                self._on_start(doc)
            elif name == "event":
                self._on_event(doc)
            elif name == "stop":
                self._on_stop(doc)
        except Exception:
            # RunEngine thread — a recorder bug must never touch the scan.
            logger.warning("live-row recorder failed on %r doc", name, exc_info=True)

    def _on_start(self, doc: dict[str, Any]) -> None:
        uid = doc.get("uid")
        if not uid:
            return
        self._uid = uid
        with _lock:
            _buffers[uid] = {"columns": [], "rows": [], "partial": True, "total_seen": 0}
            _buffers.move_to_end(uid)
            while len(_buffers) > _MAX_RUNS:
                _buffers.popitem(last=False)

    def _on_event(self, doc: dict[str, Any]) -> None:
        if not self._uid:
            return
        data = doc.get("data") or {}
        with _lock:
            buf = _buffers.get(self._uid)
            if buf is None:
                return
            new_columns = [key for key in data if key not in buf["columns"]]
            if new_columns:
                buf["columns"].extend(new_columns)
                for row in buf["rows"]:
                    row.extend([None] * len(new_columns))
            buf["total_seen"] += 1
            if len(buf["rows"]) < _MAX_ROWS_PER_RUN:
                buf["rows"].append([data.get(col) for col in buf["columns"]])
            _buffers.move_to_end(self._uid)

    def _on_stop(self, doc: dict[str, Any]) -> None:
        if not self._uid:
            return
        with _lock:
            buf = _buffers.get(self._uid)
            if buf is not None:
                buf["partial"] = False
