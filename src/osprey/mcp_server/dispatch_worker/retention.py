"""Opt-in retention sweep for dispatch run records and artifacts.

Disabled by default. Set the ``RETENTION_DAYS`` env var to a positive integer to
enable it; unset, empty, ``0``, or a non-integer all mean *disabled* (nothing is
ever deleted). When enabled, a periodic background task deletes:

  * persisted dispatch run records (``<agent-data root>/dispatch/{run_id}.json``), and
  * ``ArtifactStore`` entries (index row + on-disk file),

whose age exceeds the threshold. Age is measured from a record's completion (a
run record's ``completed_at``, falling back to ``created_at``; an artifact's
``timestamp``). A record is deleted only when it is strictly older than
``RETENTION_DAYS`` days — a record aged exactly ``N-1`` days survives, one aged
``N+1`` days is deleted.

In-flight runs are never swept regardless of age: a run record whose status is
not terminal is skipped, and any run id currently pending in the worker (passed
in as ``in_flight_run_ids``) is skipped along with the artifacts it produced.

The same eligibility rule (:func:`record_is_deletable`) also backs the
dashboard's on-demand *clear history* action, which runs it with no age floor —
so a button click and a scheduled sweep agree, by construction, on what counts
as a finished run.

This is a generic OSPREY-core capability with no channel awareness. The sweep
functions are pure (they take the log dir, an ``ArtifactStore``, and an injected
``now``) so they can be driven directly in tests without a clock or sleeps.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def record_is_deletable(
    record: Mapping[str, Any],
    now: float,
    older_than_days: int = 0,
) -> bool:
    """True when a run record may be deleted. The one eligibility rule.

    Shared by the periodic retention sweep and the dashboard's clear-history
    action, so the two can never disagree about which runs are finished and safe
    to drop.

    A record qualifies when its status is terminal and — when ``older_than_days``
    is positive — its completion is strictly older than that window (a record
    aged exactly N days survives). ``older_than_days <= 0`` means *no age floor*:
    every terminal record qualifies, which is what the clear-history button
    asks for. A record whose timestamp cannot be read is never deletable under
    an age floor.

    In-flight protection is the caller's: a run the worker still holds as
    pending is filtered out by run id before this is consulted, because a
    pending run's record is not always on disk yet.
    """
    if record.get("status") not in _TERMINAL_STATUSES:
        return False
    if older_than_days <= 0:
        return True

    ts = record.get("completed_at")
    if ts is None:
        ts = record.get("created_at")
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return False

    # Strictly older than the window: age exactly N days survives (ts == cutoff).
    return ts < now - older_than_days * _SECONDS_PER_DAY


def delete_run_records(
    log_dir: str | Path,
    now: float,
    older_than_days: int = 0,
    in_flight_run_ids: Iterable[str] = (),
) -> list[str]:
    """Delete every eligible persisted run record. Returns the deleted run ids.

    Eligibility is :func:`record_is_deletable` plus the in-flight guard. With
    ``older_than_days`` unset there is no age floor and every terminal record
    goes. Unreadable files are left in place, as is any file whose unlink fails
    — a partial deletion is reported honestly rather than raising.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []

    in_flight = frozenset(in_flight_run_ids)
    deleted: list[str] = []

    for path in sorted(log_dir.glob("*.json")):
        run_id = path.name.removesuffix(".json")
        if run_id in in_flight:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s — leaving it in place", path, exc_info=True)
            continue

        if not record_is_deletable(data, now, older_than_days):
            continue

        try:
            path.unlink()
            deleted.append(run_id)
        except OSError:
            logger.warning("Failed to delete %s", path, exc_info=True)

    return deleted


def sweep_dispatch_runs(
    log_dir: str | Path,
    retention_days: int,
    now: float,
    in_flight_run_ids: Iterable[str] = (),
) -> int:
    """Delete persisted run records older than the threshold. Returns the count.

    The age-based face of :func:`delete_run_records`: skips any record that is
    non-terminal or whose run id is in ``in_flight_run_ids`` (an in-flight run is
    never swept). A no-op when ``retention_days <= 0`` or the dir is absent.
    """
    if retention_days <= 0:
        return 0
    return len(delete_run_records(log_dir, now, retention_days, in_flight_run_ids))


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

    from osprey.stores.artifact_store import artifact_mutation_actor

    # Snapshot the entry ids first: delete_entry mutates the index under a lock,
    # so iterate over a stable list rather than the live entry collection.
    # These deletes are maintenance, not agent actions — tag them so store
    # listeners don't report them as agent activity.
    with artifact_mutation_actor("system"):
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
    log_dir: str | Path | Callable[[], str | Path],
    store_factory: Callable[[], ArtifactStore],
    retention_days: int,
    in_flight_run_ids: Callable[[], Iterable[str]],
    interval_sec: float = DEFAULT_SWEEP_INTERVAL_SEC,
) -> None:
    """Periodically run :func:`run_sweep` every ``interval_sec`` seconds.

    ``log_dir`` may be a callable, and the worker passes one: the record
    directory is derived from the config the worker was pointed at, and this
    loop starts during application lifespan — before anything has established
    that the config is readable yet. Resolving once at startup would let a
    config that arrives moments later leave the sweep pointed at the fallback
    root for the life of the process, quietly aging out nothing while the writer
    filled a different directory. Re-resolved each cycle, that self-corrects.

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
                log_dir() if callable(log_dir) else log_dir,
                store_factory(),
                retention_days,
                in_flight_run_ids=in_flight_run_ids(),
            )
        except Exception:
            logger.exception("Retention sweep cycle failed — continuing")
