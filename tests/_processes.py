"""Process-table probes for suites that start, stop or abandon real processes."""

from __future__ import annotations

import contextlib
import os
import signal


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` still names a process (a zombie still counts)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def kill_group(pgid: int) -> None:
    """SIGKILL the process group ``pgid``; a group that is gone already is not an error."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
