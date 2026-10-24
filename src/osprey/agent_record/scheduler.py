"""The service's daily jobs: export, then retention.

One asyncio task in the agent-record service's lifespan (:func:`run`) wakes
once a day at :data:`DAILY_AT` and runs :func:`run_jobs`:
``archive.run_export(record, now)``, then ``retention.run_retention(record,
now)``. Both are looked up on their modules by name at every run, so each job
is implemented in its own module and nothing here changes when it is.

Retention runs only after the export returned. An export that raises skips
that day's retention, because retention may only remove what the archive
already holds; the failure is logged and the next day runs both jobs again.
A failed day never ends the loop.

The jobs block (they read and write the backend), so they run in a worker
thread; the event loop keeps serving reads and ingest meanwhile.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, time, timedelta

from osprey.agent_record import archive, retention
from osprey.agent_record.contract import RecordBackend

logger = logging.getLogger(__name__)

__all__ = ["DAILY_AT", "next_run", "run", "run_jobs"]

#: The UTC time of day the jobs run.
DAILY_AT = time(0, 30, tzinfo=UTC)


def next_run(now: datetime, at: time = DAILY_AT) -> datetime:
    """The first moment strictly after *now* at the time of day *at* (UTC)."""
    now = now.astimezone(UTC)
    when = datetime.combine(now.date(), at.replace(tzinfo=None), tzinfo=UTC)
    if when <= now:
        when += timedelta(days=1)
    return when


def run_jobs(record: RecordBackend, now: datetime) -> None:
    """Run the export, then retention, for *now*.

    Raises:
        Exception: Whatever the export raises; retention is then not run.
            A failure of retention itself propagates too.
    """
    archive.run_export(record, now)
    retention.run_retention(record, now)


async def run(
    record: RecordBackend,
    *,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    at: time = DAILY_AT,
) -> None:
    """Run :func:`run_jobs` once a day at *at*, forever.

    Args:
        record: The record backend the jobs act on.
        now: The wall clock, returning an aware ``datetime``; UTC now by default.
        sleep: The coroutine that waits a number of seconds.
        at: The UTC time of day the jobs run.
    """
    clock = now or (lambda: datetime.now(UTC))
    while True:
        due = next_run(clock(), at)
        await sleep(max((due - clock()).total_seconds(), 0.0))
        try:
            await asyncio.to_thread(run_jobs, record, clock())
        except Exception:  # one failed day never ends the loop
            logger.exception("agent-record daily jobs failed; retention skipped for this day")
