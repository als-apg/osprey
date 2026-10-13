"""The one state of this process's guarded run, behind one lock.

A process has one guarded run. The call it was approved for, the run lock it
holds, the journaled run it has open and the restore reports it filed are facts
of the process, not of the thread or task that took them, so a thread or task
the run starts sees the same answers as the code that started it.

Two invariants hold:

- Every fact of the guarded run is a field of :data:`STATE`, read and changed
  with :data:`LOCK` held.
- :data:`STATE` and :data:`LOCK` are spelled ``_run_state.STATE`` and
  ``_run_state.LOCK`` at the moment of use and never bound by an import, so a
  forked child (:func:`reset_in_child`) and a test can replace them.

This module is stdlib-only at import time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from osprey.runtime.journal import DurableJournal, Journal

#: The one lock every reader and writer of :data:`STATE` holds. Re-entrant, so
#: a guarded write can hold it across the journaling it calls.
LOCK = threading.RLock()


@dataclass(frozen=True)
class ApprovedCall:
    """The tool call this process runs for, as the approval hook bound it.

    Attributes:
        tool: The guarded tool's short name, as ``approval.tools`` keys it.
        journal_sha256: The approved journal digest,
            :data:`~osprey.runtime.guarded_run.APPROVED_NO_JOURNAL` when the
            prompt listed none, ``None`` when the call carries none.
        target: The approved control target, ``None`` when the call carries none.
    """

    tool: str
    journal_sha256: str | None
    target: str | None


@dataclass(frozen=True)
class Hold:
    """The run lock this process holds.

    Attributes:
        target: The control target's name.
        directory: The target's guarded-run directory.
        fd: The locked lock file's descriptor.
        started: When the lock was taken (ISO-8601), as the holder record says.
    """

    target: str
    directory: Path
    fd: int
    started: str


@dataclass
class OpenRun:
    """The journaled guarded run this process has open.

    Attributes:
        target: The control target's name.
        journal: The run's outermost journal, sinking into ``durable``.
        durable: The open durable journal file.
        deadline_guard: The run is guarded by the execution deadline.
        stopped_by: What escaped a run opened inside this one, ``None`` while
            nothing has; once set, the run accepts no further guarded write.
    """

    target: str
    journal: Journal
    durable: DurableJournal
    deadline_guard: bool
    stopped_by: BaseException | None = None


@dataclass
class RunState:
    """Every fact of this process's guarded run.

    Attributes:
        approved: The bound call, ``None`` in a process no guarded tool started.
        hold: The run lock held, ``None`` when none is.
        run: The journaled run open, ``None`` when none is.
        reports: Every restore report filed, as JSON objects, in print order.
    """

    approved: ApprovedCall | None = None
    hold: Hold | None = None
    run: OpenRun | None = None
    reports: list[dict[str, Any]] = field(default_factory=list)


#: The state of this process's guarded run.
STATE = RunState()


def reset_in_child() -> None:
    """Give a forked child a new lock and a state holding only the approved call.

    A lock another thread held at the fork is never released in the child, so
    the child never touches the parent's. The child holds no run lock and has
    no run open; the parent's descriptors are closed before this is called.
    """
    global LOCK, STATE
    LOCK = threading.RLock()
    STATE = RunState(approved=STATE.approved)
