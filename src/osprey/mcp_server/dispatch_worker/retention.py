"""Opt-in retention sweep for dispatch run records and artifacts.

Disabled by default. Set the ``RETENTION_DAYS`` env var to a positive integer to
enable it; unset, empty, ``0``, or a non-integer all mean *disabled* (nothing is
ever deleted). When enabled, a periodic background task deletes:

  * persisted dispatch run records (``_agent_data/dispatch/{run_id}.json``), and
  * ``ArtifactStore`` entries (index row + on-disk file),

whose age exceeds the threshold. Age is measured from a record's completion (a
run record's ``completed_at``, falling back to ``created_at``; an artifact's
``timestamp``). A record is deleted only when it is strictly older than
``RETENTION_DAYS`` days — a record aged exactly ``N-1`` days survives, one aged
``N+1`` days is deleted.

In-flight runs are never swept regardless of age: a run record whose status is
not terminal is skipped, and any run id currently pending in the worker (passed
in as ``in_flight_run_ids``) is skipped along with the artifacts it produced.

This is a generic OSPREY-core capability with no channel awareness. The sweep
functions are pure (they take the log dir, an ``ArtifactStore``, and an injected
``now``) so they can be driven directly in tests without a clock or sleeps.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.stores.artifact_store import ArtifactStore

logger = logging.getLogger("osprey.mcp_server.dispatch_worker.retention")

_SECONDS_PER_DAY = 86400.0

# Only terminal run records are eligible for deletion. A non-terminal (pending)
# record represents an in-flight run and must survive regardless of age. Kept in
# sync with dispatch_api._TERMINAL_STATUSES.
_TERMINAL_STATUSES = frozenset({"completed", "error"})

# Default interval between periodic sweeps when enabled.
DEFAULT_SWEEP_INTERVAL_SEC = 3600.0


def retention_days_from_env(env: dict[str, str] | None = None) -> int:
    """Parse ``RETENTION_DAYS``. Unset, empty, ``0``, or non-integer → ``0`` (off).

    Only a positive integer enables retention; every other value disables it, so
    a typo can never silently start deleting data on a shorter horizon than
    intended.
    """
    raw = (env if env is not None else os.environ).get("RETENTION_DAYS", "").strip()
    if not raw:
        return 0
    try:
        days = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid RETENTION_DAYS=%r — retention disabled", raw)
        return 0
    if days <= 0:
        return 0
    return days


def _parse_iso_timestamp(value: str) -> float | None:
    """Convert an ISO-8601 artifact timestamp to epoch seconds, or ``None``."""
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def sweep_dispatch_runs(
    log_dir: str | Path,
    retention_days: int,
    now: float,
    in_flight_run_ids: Iterable[str] = (),
) -> int:
    """Delete persisted run records older than the threshold. Returns the count.

    Skips any record that is non-terminal or whose run id is in
    ``in_flight_run_ids`` (an in-flight run is never swept). Unreadable files are
    left in place. A no-op when ``retention_days <= 0`` or the dir is absent.
    """
    if retention_days <= 0:
        return 0
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return 0

    in_flight = frozenset(in_flight_run_ids)
    cutoff = now - retention_days * _SECONDS_PER_DAY
    deleted = 0

    for path in log_dir.glob("*.json"):
        run_id = path.name.removesuffix(".json")
        if run_id in in_flight:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Retention: could not read %s — skipping", path, exc_info=True)
            continue

        if data.get("status") not in _TERMINAL_STATUSES:
            # In-flight / non-terminal record — never sweep it.
            continue

        ts = data.get("completed_at")
        if ts is None:
            ts = data.get("created_at")
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            continue

        # Strictly older than the window: age exactly N days survives (ts == cutoff).
        if ts >= cutoff:
            continue

        try:
            path.unlink()
            deleted += 1
        except OSError:
            logger.warning("Retention: failed to delete %s", path, exc_info=True)

    return deleted


def sweep_artifacts(
    store: ArtifactStore,
    retention_days: int,
    now: float,
    in_flight_run_ids: Iterable[str] = (),
) -> int:
    """Delete artifact entries older than the threshold. Returns the count.

    An artifact produced by an in-flight run (its ``run_id`` tag is in
    ``in_flight_run_ids``) survives regardless of age. Both the index row and the
    on-disk file are removed via ``ArtifactStore.delete_entry``. A no-op when
    ``retention_days <= 0``.
    """
    if retention_days <= 0:
        return 0

    in_flight = frozenset(in_flight_run_ids)
    cutoff = now - retention_days * _SECONDS_PER_DAY
    deleted = 0

    # Snapshot the entry ids first: delete_entry mutates the index under a lock,
    # so iterate over a stable list rather than the live entry collection.
    for entry in list(store.list_entries()):
        if entry.run_id and entry.run_id in in_flight:
            continue
        ts = _parse_iso_timestamp(entry.timestamp)
        if ts is None or ts >= cutoff:
            continue
        if store.delete_entry(entry.id):
            deleted += 1

    return deleted


def run_sweep(
    log_dir: str | Path,
    store: ArtifactStore,
    retention_days: int,
    now: float | None = None,
    in_flight_run_ids: Iterable[str] = (),
) -> dict[str, int]:
    """Run one full retention sweep (run records + artifacts). Returns counts.

    Pure and directly callable in tests. Logs a single line with the deleted
    counts when anything was removed.
    """
    if retention_days <= 0:
        return {"runs": 0, "artifacts": 0}
    if now is None:
        now = time.time()

    in_flight = frozenset(in_flight_run_ids)
    runs_deleted = sweep_dispatch_runs(log_dir, retention_days, now, in_flight)
    artifacts_deleted = sweep_artifacts(store, retention_days, now, in_flight)

    if runs_deleted or artifacts_deleted:
        logger.info(
            "Retention sweep (older than %dd): deleted %d run record(s), %d artifact(s)",
            retention_days,
            runs_deleted,
            artifacts_deleted,
        )
    return {"runs": runs_deleted, "artifacts": artifacts_deleted}


async def retention_loop(
    log_dir: str | Path,
    store_factory: Callable[[], ArtifactStore],
    retention_days: int,
    in_flight_run_ids: Callable[[], Iterable[str]],
    interval_sec: float = DEFAULT_SWEEP_INTERVAL_SEC,
) -> None:
    """Periodically run :func:`run_sweep` every ``interval_sec`` seconds.

    ``store_factory`` builds a fresh ``ArtifactStore`` each cycle (the worker's
    module singleton is rooted at the wrong CWD — see
    ``osprey.agent_runner.artifact_resolve._get_store``). ``in_flight_run_ids`` is
    re-read each cycle so a run that starts mid-sweep-interval is protected.

    A failing sweep is logged and the loop continues — retention must never take
    the worker down. Returns only on cancellation.
    """
    import asyncio

    logger.info(
        "Retention sweep enabled: deleting records older than %d day(s), every %.0fs",
        retention_days,
        interval_sec,
    )
    while True:
        await asyncio.sleep(interval_sec)
        try:
            run_sweep(
                log_dir,
                store_factory(),
                retention_days,
                in_flight_run_ids=in_flight_run_ids(),
            )
        except Exception:
            logger.exception("Retention sweep cycle failed — continuing")
