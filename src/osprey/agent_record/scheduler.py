"""The service's jobs: export, then retention, at start-up and daily.

One asyncio task in the agent-record service's lifespan (:func:`run`) runs
:func:`run_jobs` once as soon as it starts, then once a day at
:data:`DAILY_AT`: ``archive.run_export(record, now, archive_root)``, then
``retention.run_retention(record, now, retention_days, archive_root)``. Both
are looked up on their modules by name at every run, so each job is
implemented in its own module and nothing here changes when it is.

Retention runs only after the export returned. An export that raises skips
that run's retention, because retention may only remove what the archive
already holds; the failure is logged, kept as :attr:`JobsState.last_error`
for the health report, and the next run tries both jobs again. A failed run
never ends the loop.

``now`` for every run comes from one clock: the service's override when it
sets one, the UTC wall clock otherwise. When the next daily run is due is
always decided by the wall clock, so an override moves which days the jobs
see as closed, never how often they run.

The jobs block (they read and write the backend), so they run in a worker
thread; the event loop keeps serving reads and ingest meanwhile, and start-up
never waits on them. A cancelled :func:`run` waits for a job thread still
running before it returns, so whoever closes the backend after it never
closes it under a job.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from osprey.agent_record import archive, retention
from osprey.agent_record.contract import RecordBackend

logger = logging.getLogger(__name__)

__all__ = [
    "DAILY_AT",
    "JobsState",
    "export",
    "fixed_clock",
    "next_run",
    "run",
    "run_jobs",
    "wall_clock",
]

#: The UTC time of day the daily jobs run.
DAILY_AT = time(0, 30, tzinfo=UTC)

Clock = Callable[[], datetime]


def wall_clock() -> datetime:
    """The UTC wall clock: the jobs' clock when the service sets no override."""
    return datetime.now(UTC)


def fixed_clock(when: datetime) -> Clock:
    """A clock that always reads *when* (made aware UTC if naive)."""
    value = when if when.tzinfo is not None else when.replace(tzinfo=UTC)
    return lambda: value


@dataclass
class JobsState:
    """What the jobs have done so far, for ``/health`` and ``/health/sources``.

    Attributes:
        ready: The start-up run has finished, whatever its outcome.
        last_error: The last export failure as ``"<type>: <message>"``;
            ``None`` before any failure and again after an export succeeds.
    """

    ready: bool = False
    last_error: str | None = None


def next_run(now: datetime, at: time = DAILY_AT) -> datetime:
    """The first moment strictly after *now* at the time of day *at* (UTC)."""
    now = now.astimezone(UTC)
    when = datetime.combine(now.date(), at.replace(tzinfo=None), tzinfo=UTC)
    if when <= now:
        when += timedelta(days=1)
    return when


def run_jobs(
    record: RecordBackend,
    now: datetime,
    *,
    archive_root: Path,
    retention_days: int,
    state: JobsState | None = None,
) -> None:
    """Run the export, then retention, for *now*.

    Args:
        record: The record backend the jobs act on.
        now: The run's clock reading (aware UTC).
        archive_root: The archive root the export writes into.
        retention_days: The record's retention window in days.
        state: Given, its ``last_error`` is set when the export fails and
            cleared when it succeeds.

    Raises:
        Exception: Whatever the export raises; retention is then not run.
            A failure of retention itself propagates too.
    """
    export(record, now, archive_root, state)
    retention.run_retention(record, now, retention_days, archive_root)


def export(
    record: RecordBackend, now: datetime, archive_root: Path, state: JobsState | None = None
) -> list[date]:
    """Run the export for *now*, keeping its outcome as ``state.last_error``.

    The one way an export runs, scheduled or on demand, so the health report
    reads both the same.

    Returns:
        The days the export wrote.

    Raises:
        Exception: Whatever the export raises, after it is kept in *state*.
    """
    try:
        days = archive.run_export(record, now, archive_root)
    except Exception as exc:
        if state is not None:
            state.last_error = f"{type(exc).__name__}: {exc}"
        raise
    if state is not None:
        state.last_error = None
    return days


def _run_logged(job: Callable[[], None]) -> None:
    """Run *job*, logging a failure instead of raising it: one run never ends the loop."""
    try:
        job()
    except Exception:
        logger.exception("agent-record jobs failed; retention is skipped when the export failed")


async def _in_thread(job: Callable[[], None]) -> None:
    """Run *job* in a worker thread; on cancel, wait for the thread first."""
    future = asyncio.get_running_loop().run_in_executor(None, job)
    try:
        await asyncio.shield(future)
    except asyncio.CancelledError:
        await asyncio.wait({future})
        raise


async def run(
    record: RecordBackend,
    *,
    archive_root: Path,
    retention_days: int,
    clock: Clock | None = None,
    state: JobsState | None = None,
    wall: Clock = wall_clock,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    at: time = DAILY_AT,
) -> None:
    """Run the jobs once now, then once a day at *at*, until cancelled.

    Args:
        record: The record backend the jobs act on.
        archive_root: The archive root the export writes into.
        retention_days: The record's retention window in days.
        clock: Where each run's ``now`` comes from; :func:`wall_clock` when
            ``None``.
        state: Updated as runs finish (``ready`` after the first, and
            ``last_error``); a private one when ``None``.
        wall: The wall clock that decides when the next daily run is due.
        sleep: The coroutine that waits a number of seconds.
        at: The UTC time of day the daily runs start.
    """
    jobs_clock = clock or wall_clock
    jobs_state = state if state is not None else JobsState()

    async def once() -> None:
        job = functools.partial(
            run_jobs,
            record,
            jobs_clock(),
            archive_root=archive_root,
            retention_days=retention_days,
            state=jobs_state,
        )
        await _in_thread(functools.partial(_run_logged, job))

    await once()
    jobs_state.ready = True
    while True:
        due = next_run(wall(), at)
        await sleep(max((due - wall()).total_seconds(), 0.0))
        await once()
