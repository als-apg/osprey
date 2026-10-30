"""Opt-in retention sweep for the dispatch worker's artifacts.

Disabled by default. Set the ``RETENTION_DAYS`` env var to a positive integer to
enable it; unset, empty, ``0``, or a non-integer all mean *disabled* (nothing is
ever deleted). When enabled, a periodic background task deletes the
``ArtifactStore`` entries (index row + on-disk file) whose ``timestamp`` is
strictly older than ``RETENTION_DAYS`` days — an artifact aged exactly ``N-1``
days survives, one aged ``N+1`` days is deleted.

The artifacts of an in-flight run are never swept regardless of age: any run id
currently pending in the worker (passed in as ``in_flight_run_ids``) is skipped
along with the artifacts it produced.

Run history is not this module's: it lives in the agent record, which ages it
out under ``agent_record.retention_days``.

This is a generic OSPREY-core capability with no channel awareness. The sweep
functions are pure (they take an ``ArtifactStore`` and an injected ``now``) so
they can be driven directly in tests without a clock or sleeps.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.stores.artifact_store import ArtifactStore

logger = logging.getLogger("osprey.mcp_server.dispatch_worker.retention")

_SECONDS_PER_DAY = 86400.0

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


def sweep_artifacts(
    store: ArtifactStore,
    retention_days: int,
    now: float | None = None,
    in_flight_run_ids: Iterable[str] = (),
) -> int:
    """Delete artifact entries older than the threshold. Returns the count.

    An artifact produced by an in-flight run (its ``run_id`` tag is in
    ``in_flight_run_ids``) survives regardless of age. Both the index row and the
    on-disk file are removed via ``ArtifactStore.delete_entry``. A no-op when
    ``retention_days <= 0``. Logs a single line with the count when anything
    was removed.
    """
    if retention_days <= 0:
        return 0

    in_flight = frozenset(in_flight_run_ids)
    cutoff = (time.time() if now is None else now) - retention_days * _SECONDS_PER_DAY
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

    if deleted:
        logger.info(
            "Retention sweep (older than %dd): deleted %d artifact(s)", retention_days, deleted
        )
    return deleted


async def retention_loop(
    store_factory: Callable[[], ArtifactStore],
    retention_days: int,
    in_flight_run_ids: Callable[[], Iterable[str]],
    interval_sec: float = DEFAULT_SWEEP_INTERVAL_SEC,
) -> None:
    """Periodically run :func:`sweep_artifacts` every ``interval_sec`` seconds.

    ``store_factory`` builds a fresh ``ArtifactStore`` each cycle: the store is
    rooted at the agent-data root of the config the worker was pointed at, and
    this loop starts during application lifespan, before anything has
    established that the config is readable yet, so a root resolved once at
    startup could stay on the fallback for the life of the process (see
    ``osprey.agent_runner.artifact_resolve._get_store``). ``in_flight_run_ids``
    is re-read each cycle so a run that starts mid-sweep-interval is protected.

    A failing sweep is logged and the loop continues — retention must never take
    the worker down. Returns only on cancellation.
    """
    logger.info(
        "Retention sweep enabled: deleting artifacts older than %d day(s), every %.0fs",
        retention_days,
        interval_sec,
    )
    while True:
        await asyncio.sleep(interval_sec)
        try:
            sweep_artifacts(
                store_factory(),
                retention_days,
                in_flight_run_ids=in_flight_run_ids(),
            )
        except Exception:
            logger.exception("Retention sweep cycle failed — continuing")
