"""The daily removal of arrival days past the retention window.

:func:`run_retention` is the scheduler's second daily job, run only after
:func:`osprey.agent_record.archive.run_export` returned
(:mod:`osprey.agent_record.scheduler`), which looks it up on this module by
name at every run.

The window is ``retention_days`` UTC days before the run's day: an arrival day
before ``today - retention_days`` is past retention. A day past retention is
removed only when the archive holds it, that is when it is in
:func:`osprey.agent_record.archive.exported_days`; the removal stops before the
oldest day the archive does not hold, so an export outage longer than the
window removes nothing and an unexported day is never skipped over. Removal
drops whole arrival days through
:meth:`~osprey.agent_record.contract.RecordBackend.delete_arrived_before`; the
ingest state is never removed.

A removed day stays out of the record: the transcript tailer's retention floor
is never later than the oldest arrival day the record still holds, so a
rewritten transcript's expired lines are skipped rather than inserted again,
and the export never revisits a day before the record's oldest arrival day.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from osprey.agent_record.archive import exported_days, oldest_arrival_day
from osprey.agent_record.contract import RecordBackend

__all__ = ["run_retention"]


def run_retention(
    record: RecordBackend, now: datetime, retention_days: int, archive_root: Path
) -> int:
    """Remove the exported arrival days of *record* that are past retention as of *now*.

    The days removed run from the record's oldest arrival day up to, not
    including, the earlier of ``retention_days`` days before *now*'s UTC day
    and the oldest day in that span the archive does not hold.

    Args:
        record: The record backend the service runs.
        now: The scheduler's clock reading for this run (aware UTC).
        retention_days: The retention window in days (at least 1).
        archive_root: The archive root whose exported days retention may remove.

    Returns:
        The number of entries removed.

    Raises:
        ValueError: If *retention_days* is below 1.
        RecordBackendError: If the record cannot say which days it holds, or
            the removal fails.
        OSError: If a manifest of the archive cannot be read.
    """
    if retention_days < 1:
        raise ValueError(f"retention_days must be at least 1, not {retention_days}")
    oldest = oldest_arrival_day(record)
    if oldest is None:
        return 0
    window = now.astimezone(UTC).date() - timedelta(days=retention_days)
    done = exported_days(archive_root)
    cutoff = oldest
    while cutoff < window and cutoff in done:
        cutoff += timedelta(days=1)
    if cutoff <= oldest:
        return 0
    return record.delete_arrived_before(cutoff)
