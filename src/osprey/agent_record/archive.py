"""The daily export of the record to the archive.

:func:`run_export` is the scheduler's first daily job
(:mod:`osprey.agent_record.scheduler`), which looks it up on this module by
name at every run.

Each closed arrival day (a UTC day before the run's day) is exported once,
into that day's own directory of the archive root:
``<root>/<day>/agent_record/entries.parquet`` holds every entry that arrived
that day, and the day's ``MANIFEST.jsonl`` gets a ``file`` line for it and then
an ``agent_record_day`` line, last. A day with no entries gets the day line with
``rows: 0`` and no file. The day line is the only mark of an exported day
(:func:`exported_days`); an ``agent_record_purged`` line clears it.

The export writes through the archive service's
:class:`~osprey.services.archive.run.ArchiveTree`: it holds the one archive
lock on ``.incoming`` from before it reads the marks until its last line is
written, so two runs never export a day twice, and every file is staged in
``.incoming`` and moved into place only when whole. A run that stopped between
placing a day's file and writing its day line left that day unmarked; the next
run exports the day again and replaces that file rather than adding a second.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from osprey.agent_record.contract import RecordBackend, RecordBackendError
from osprey.services.archive.manifest import AGENT_RECORD_DAY, load_state
from osprey.services.archive.run import ArchiveTree, file_record

__all__ = [
    "EXPORT_FILE",
    "EXPORT_SUBDIR",
    "exported_days",
    "oldest_arrival_day",
    "run_export",
    "unexported_days",
]

#: The directory of a day's export, under that day's directory.
EXPORT_SUBDIR = "agent_record"

#: A day's export file, under :data:`EXPORT_SUBDIR`.
EXPORT_FILE = "entries.parquet"


def exported_days(archive_root: Path) -> set[date]:
    """The arrival days whose latest record line in the archive is ``agent_record_day``.

    Raises:
        OSError: If a manifest exists but cannot be read.
    """
    return {date.fromisoformat(day) for day in load_state(archive_root).agent_record_days}


def unexported_days(oldest: date | None, today: date, done: set[date]) -> list[date]:
    """The closed days from *oldest* through the day before *today* not in *done*, oldest first.

    Empty when *oldest* is ``None``, a record that holds no entry.
    """
    days: list[date] = []
    day = oldest
    while day is not None and day < today:
        if day not in done:
            days.append(day)
        day += timedelta(days=1)
    return days


def oldest_arrival_day(record: RecordBackend) -> date | None:
    """The oldest arrival day *record* holds; ``None`` when it holds no entry.

    Raises:
        RecordBackendError: If the record's health is not ok, so it cannot say.
    """
    health = record.health()
    if not health.ok:
        raise RecordBackendError(f"the record cannot say which days it holds: {health.detail}")
    return health.oldest_arrival_day


def _export_day(record: RecordBackend, tree: ArchiveTree, day: date) -> None:
    label = day.isoformat()
    temp = tree.staging_path(".parquet")
    try:
        rows = record.export_parquet(day, temp)
        if rows:
            size, sha = tree.adopt(temp)
            path = tree.place(temp, f"{EXPORT_SUBDIR}/{EXPORT_FILE}", day=label, replace=True)
            tree.append(
                file_record(
                    path,
                    source=path,
                    size=size,
                    sha256=sha,
                    source_mtime_ns=None,
                    day=label,
                    rows=rows,
                ),
                day=label,
            )
    finally:
        tree.discard(temp)
    tree.append({"kind": AGENT_RECORD_DAY, "day": label, "rows": rows}, day=label)


def run_export(record: RecordBackend, now: datetime, archive_root: Path) -> list[date]:
    """Export every closed arrival day of *record* the archive does not hold yet.

    The days run from the record's oldest arrival day through the day before
    *now*'s UTC day, oldest first, skipping every day in
    :func:`exported_days`. The marks are read, and every day written, under the
    archive lock.

    Args:
        record: The record backend the service runs.
        now: The scheduler's clock reading for this run (aware UTC).
        archive_root: The archive root the days are exported into.

    Returns:
        The days exported, oldest first.

    Raises:
        RecordBackendError: If the record cannot be read; the failing day and
            every later one stay unmarked.
        osprey.services.archive.run.ArchiveDestinationError: If the archive
            root cannot be written.
        OSError: If the archive cannot be read or written part-way.
    """
    tree = ArchiveTree(archive_root, now)
    try:
        done = exported_days(archive_root)
        days = unexported_days(oldest_arrival_day(record), now.astimezone(UTC).date(), done)
        exported: list[date] = []
        for day in days:
            _export_day(record, tree, day)
            exported.append(day)
        return exported
    finally:
        tree.close()
