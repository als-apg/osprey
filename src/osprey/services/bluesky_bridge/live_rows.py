"""Bounded live-row buffer for run data still in flight or just completed.

The ``GET /runs/{id}/data`` route needs a source of truth for run data before
(or in place of) a Tiled server. Scans run in the queueserver worker, which
publishes its document stream over 0MQ; the bridge's document plane
(``document_plane.py``) subscribes to that stream and feeds the plain-dict
callback below, which keeps translated rows in a bounded buffer. This module is
deliberately import-clean of bluesky, so it can be unit-tested with synthetic
documents and no worker at all. Generalizes the concept in BELLA's
``services/experiment_config/live_rows.py``: this version has no
GEECS-specific legacy column mapping, and explicitly RETAINS completed runs
rather than evicting at a small run count, so a read arriving after the plan
finishes still succeeds without a Tiled server.

Document handling (bluesky's plain-dict document protocol):

- ``start``: begins a new buffer for ``doc["uid"]`` — or for the recorder's
  explicit ``key``, when the caller keys buffers by something other than the
  RunEngine's own uid (see :class:`LiveRowRecorder`) — marked partial, and
  stamped with the recorder's opaque ``plan`` value if it was given one.
- ``event``: each event's ``doc["data"]`` becomes one row; columns are
  discovered incrementally in first-seen order across events. A key seen for
  the first time extends the column list and backfills every already-stored
  row with ``None`` in that column, so all stored rows stay aligned to the
  current column list; a later event missing an already-known column simply
  records ``None`` for it.
- ``stop``: flips ``partial`` to ``False``. This is the ONLY thing that ends
  the "still filling in" state — a buffer that never gets a stop doc (e.g.
  the process crashed mid-run) stays partial forever, which is the honest
  answer.

Bounding uses two independent knobs:

- :func:`max_rows_per_run`: a hard cap on stored rows per run, so a runaway
  plan cannot grow the buffer without limit. ``total_seen`` keeps counting
  every event past this cap — the ``row_count`` on the run-data route reports
  this *true* total
  even when the tail beyond the cap was never stored (a documented trade-off
  for a pathological never-ending run, not the common case).
- :func:`max_runs`: number of run buffers retained at all, oldest-evicted
  (insertion-ordered ``OrderedDict``, ``move_to_end`` on every write) once
  exceeded. This is what lets a completed run's data survive to be read later.

Both are authored per facility (``bluesky.live_max_rows_per_run`` and
``bluesky.live_max_runs`` in the build profile) and reach the bridge as env
vars, because how many runs are worth keeping in memory, and how long a run
gets, are properties of the plans a facility runs and of the memory its
deployment host has. A recorder resolves both at its start document and holds
them for that run, so a cap never moves under a buffer that is already filling.

The recorder itself never raises: every document is handled inside a
try/except so a recorder bug can never abort a run (mirrors the runner
seam's own safety contract) — a bug here loses live-read fidelity for that
run, not the run itself.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

#: Env var carrying how many run buffers this bridge retains at once.
#: Authored per facility in the build profile and rendered into the compose
#: file; the container cannot read `config.yml`, so the env var is the channel.
MAX_RUNS_ENV = "BLUESKY_LIVE_MAX_RUNS"

#: Used when the variable is unset — high enough that a completed run stays
#: readable well past its own completion.
DEFAULT_MAX_RUNS = 50

#: Env var carrying the hard per-run row storage cap.
MAX_ROWS_PER_RUN_ENV = "BLUESKY_LIVE_MAX_ROWS_PER_RUN"

#: Used when the variable is unset. A safety valve against a runaway or
#: never-ending plan, not a normal-case limit (see `total_seen` above).
DEFAULT_MAX_ROWS_PER_RUN = 10_000


def max_runs() -> int:
    """How many run buffers this bridge retains at once, oldest evicted first.

    Read from `os.environ` on every call — never cached at import — so a test
    (and a re-exec'd process) sees the value its environment declares.

    Raises:
        ValueError: if the variable is set to something that is not an integer,
            or to an integer below 1. Retaining zero runs would evict every
            buffer the moment it was created, which is a misconfiguration
            rather than a policy.
    """
    return _positive_int(MAX_RUNS_ENV, DEFAULT_MAX_RUNS)


def max_rows_per_run() -> int:
    """How many rows one run buffer stores before it stops growing.

    Rows past this cap are counted in ``total_seen`` and not stored, so the
    run-data route keeps reporting the true total over a truncated buffer.

    Raises:
        ValueError: if the variable is set to something that is not an integer,
            or to an integer below 1.
    """
    return _positive_int(MAX_ROWS_PER_RUN_ENV, DEFAULT_MAX_ROWS_PER_RUN)


def _positive_int(env_var: str, default: int) -> int:
    """Read *env_var* as an integer >= 1, falling back to *default* when unset."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{env_var}={raw!r} is not an integer; set it to a whole "
            f"number >= 1 (default {default})"
        ) from None
    if value < 1:
        raise ValueError(f"{env_var}={raw!r} must be >= 1")
    return value


_lock = Lock()
# run_uid -> {"columns": [...], "rows": [[...], ...], "partial": bool,
#             "total_seen": int, "plan": Any | None}
_buffers: OrderedDict[str, dict[str, Any]] = OrderedDict()


def get(run_uid: str) -> dict[str, Any] | None:
    """A snapshot of the live buffer for *run_uid* (or None if unknown).

    ``columns`` and ``rows`` are copies, so a caller cannot reach back into the
    buffer through them. ``plan`` is whatever the recorder was handed and is
    returned by reference — this module never inspects or mutates it, and
    neither may a caller.
    """
    with _lock:
        buf = _buffers.get(run_uid)
        if buf is None:
            return None
        return {
            "columns": list(buf["columns"]),
            "rows": [list(row) for row in buf["rows"]],
            "partial": buf["partial"],
            "total_seen": buf["total_seen"],
            "plan": buf["plan"],
        }


def _clear() -> None:
    """Drop all buffers (test isolation only)."""
    with _lock:
        _buffers.clear()


class LiveRowRecorder:
    """Bluesky document callback recording rows into the bounded live buffer.

    One instance per run (mirrors BELLA's contract): it must see the run's
    start document, so it is opened at that document rather than partway
    through the stream.

    Args:
        key: Buffer key to record under, instead of the start document's own
            ``uid``. The document plane (``document_plane.py``) sets it to the
            OSPREY run id carried in the start document, because a run executed
            in the queueserver container has a RunEngine uid the bridge never
            chose and cannot map back to the item an operator enqueued. Left
            ``None``, the run uid IS the identity the read path looks up —
            which is the honest key for a run that carries no OSPREY id.
        plan: Which plan produced this run, carried alongside its rows so a
            reader that has only a run id can tell what the columns mean. The
            value is opaque here: this module stores it and hands it back in
            :func:`get`, and never interprets, validates, or copies it. The
            document plane extracts it from the start document (mirroring
            ``key``), which keeps this module testable with synthetic
            documents. ``None`` for a run whose start document carried no plan
            stamp — an honest "not known", never a placeholder.
    """

    def __init__(self, key: str | None = None, plan: Any | None = None) -> None:
        self._key = key
        self._plan = plan
        self._uid: str | None = None
        self._max_rows: int = DEFAULT_MAX_ROWS_PER_RUN

    def __call__(self, name: str, doc: dict[str, Any]) -> None:
        try:
            if name == "start":
                self._on_start(doc)
            elif name == "event":
                self._on_event(doc)
            elif name == "stop":
                self._on_stop(doc)
        except Exception:
            # RunEngine thread — a recorder bug must never touch the plan.
            logger.warning("live-row recorder failed on %r doc", name, exc_info=True)

    def _on_start(self, doc: dict[str, Any]) -> None:
        uid = self._key or doc.get("uid")
        if not uid:
            return
        self._uid = uid
        # Both caps are resolved once, here, and not again for the rest of the
        # run: `_on_event` runs per event document on the RunEngine's thread,
        # and a cap that changes mid-run would move the boundary of a buffer
        # that is already half-filled under it.
        self._max_rows = max_rows_per_run()
        retained_runs = max_runs()
        with _lock:
            _buffers[uid] = {
                "columns": [],
                "rows": [],
                "partial": True,
                "total_seen": 0,
                "plan": self._plan,
            }
            _buffers.move_to_end(uid)
            while len(_buffers) > retained_runs:
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
            if len(buf["rows"]) < self._max_rows:
                buf["rows"].append([data.get(col) for col in buf["columns"]])
            _buffers.move_to_end(self._uid)

    def _on_stop(self, doc: dict[str, Any]) -> None:
        if not self._uid:
            return
        with _lock:
            buf = _buffers.get(self._uid)
            if buf is not None:
                buf["partial"] = False
