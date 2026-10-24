"""The daily removal of arrival days past the retention window.

:func:`run_retention` is the scheduler's second daily job, run only after
:func:`osprey.agent_record.archive.run_export` returned
(:mod:`osprey.agent_record.scheduler`). The scheduler looks it up on this
module by name at every run, so retention is implemented by replacing this
function's body; the scheduler and the service stay as they are.

Retention is not implemented: :func:`run_retention` removes nothing, so the
record keeps every entry it holds.
"""

from __future__ import annotations

from datetime import datetime

from osprey.agent_record.contract import RecordBackend

__all__ = ["run_retention"]


def run_retention(record: RecordBackend, now: datetime) -> None:
    """Remove the arrival days of *record* that are past retention as of *now*.

    Does nothing: no retention is implemented.

    Args:
        record: The record backend the service runs.
        now: The scheduler's clock reading for this run (aware UTC).
    """
    del record, now
