"""Process-table probes for suites that start, stop or abandon real processes."""

from __future__ import annotations

import os


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` still names a process (a zombie still counts)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
