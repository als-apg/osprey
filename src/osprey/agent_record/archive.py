"""The daily export of the record to the archive.

:func:`run_export` is the scheduler's first daily job
(:mod:`osprey.agent_record.scheduler`). The scheduler looks it up on this
module by name at every run, so the export is implemented by replacing this
function's body; the scheduler and the service stay as they are.

The export is not implemented: :func:`run_export` does nothing. The record
keeps every entry until retention removes it, and retention removes nothing
yet either (:mod:`osprey.agent_record.retention`).
"""

from __future__ import annotations

from datetime import datetime

from osprey.agent_record.contract import RecordBackend

__all__ = ["run_export"]


def run_export(record: RecordBackend, now: datetime) -> None:
    """Export the arrival days that are due, as of *now*, from *record*.

    Does nothing: no export is implemented.

    Args:
        record: The record backend the service runs.
        now: The scheduler's clock reading for this run (aware UTC).
    """
    del record, now
