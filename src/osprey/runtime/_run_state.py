"""The one state of this process's run lock, behind one lock.

The run lock a process holds is a fact of the process, not of the thread or
task that took it, so a thread or task the run starts sees the same answer as
the code that started it.

Three invariants hold:

- Every fact of the hold is a field of :data:`STATE`, read and changed with
  :data:`LOCK` held.
- :data:`STATE` and :data:`LOCK` are spelled ``_run_state.STATE`` and
  ``_run_state.LOCK`` at the moment of use and never bound by an import, so a
  forked child (:func:`reset_in_child`) and a test can replace them.
- :data:`LOCK` is held only across system calls, never across a deployment
  lookup, an import, a write or another lock's acquisition, so a fork may wait
  for it and a lock entry is never split by a fork.

This module is stdlib-only at import time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: The one lock every reader and writer of :data:`STATE` holds.
LOCK = threading.Lock()


@dataclass(frozen=True)
class Hold:
    """The run lock this process holds.

    Attributes:
        target: The control target's name.
        directory: The target's guarded-run directory.
        fd: The locked lock file's descriptor.
        dir_fd: The open descriptor of ``directory``; the lock is opened
            relative to it.
        started: When the lock was taken (ISO-8601), as the holder record says.
    """

    target: str
    directory: Path
    fd: int
    started: str
    dir_fd: int


@dataclass
class RunState:
    """Every fact of this process's run lock.

    Attributes:
        hold: The run lock held, ``None`` when none is.
    """

    hold: Hold | None = None


#: The state of this process's run lock.
STATE = RunState()


def reset_in_child() -> None:
    """Give a forked child a new lock and a state that holds nothing.

    The child's copy of :data:`LOCK` is held by the fork's own acquire and is
    replaced, never released. The parent's descriptors are closed before this
    is called.
    """
    global LOCK, STATE
    LOCK = threading.Lock()
    STATE = RunState()
