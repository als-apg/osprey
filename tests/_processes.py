"""Process-table probes for suites that start, stop or abandon real processes.

Two notions of "still there" live here, and they differ on purpose over zombies:

* :func:`pid_alive` asks whether the pid still names a process. A zombie still
  does, so a test that must see a process *reaped* (its parent waited for it,
  or it was reparented to init and init did) uses this one.
* :func:`running_state` / :func:`pid_gone` / :func:`wait_gone` ask whether the
  process still *runs*. A zombie does not, so a test that only cares that the
  code under test stopped a process, whoever reaps it later, uses these.

None of them reaps anything: a probe that called ``waitpid`` would steal the
exit status from the parent (an asyncio child watcher, a ``Popen``) that owns it.
A pid owned by another user (``PermissionError`` on signal 0) exists, so every
probe here treats it as present.

The probes ask the kernel about one pid at a time (signal 0, then
``psutil.Process``); none of them reads the product's process table, so they
stay a fair witness in a test that faults that table.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Iterable

import psutil

from tests._live import wait_until


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` still names a process (a zombie still counts)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_state(pid: int) -> str | None:
    """Return the psutil status of *pid* while it runs, ``None`` once it is gone or a zombie.

    The status is read only for a pid the kernel still answers for: that is the
    one case a signal probe cannot tell a running process from an exited one
    nobody has reaped.
    """
    if not pid_alive(pid):
        return None
    try:
        state = psutil.Process(pid).status()
    except psutil.NoSuchProcess:
        return None
    return None if state == psutil.STATUS_ZOMBIE else state


def pid_gone(pid: int, within: float = 5.0) -> bool:
    """Return whether *pid* is gone, or a zombie nobody has reaped yet, within *within* s.

    ``within=0.0`` looks once, so ``assert not pid_gone(pid, within=0.0)`` says
    the process is still running now.
    """
    try:
        wait_until(lambda: running_state(pid) is None, timeout=within)
    except AssertionError:
        return False
    return True


async def wait_gone(pids: Iterable[int], within: float = 5.0) -> None:
    """Wait until every one of *pids* is gone, or a zombie nobody has reaped yet.

    The wait of an async test: it yields to the loop between looks and reads
    the process table off it, so whatever the loop still has to finish is not held up by
    the waiting. All of *pids* share one deadline of *within* seconds. A pid
    still running at the deadline fails the wait by name, with the state
    the process table last reported for it.
    """
    started = time.monotonic()
    running = list(pids)
    while True:
        states = {pid: await asyncio.to_thread(running_state, pid) for pid in running}
        running = [pid for pid in running if states[pid] is not None]
        if not running:
            return
        waited = time.monotonic() - started
        if waited > within:
            pid = running[0]
            raise AssertionError(
                f"pid {pid} (ps state {states[pid]!r}) is still running {waited:.1f} s after "
                "the stop returned"
            )
        await asyncio.sleep(0.05)


def kill_quietly(pids: Iterable[int]) -> None:
    """SIGKILL every pid in *pids*, ignoring those already gone."""
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def kill_group(pgid: int) -> None:
    """SIGKILL the process group ``pgid``; a group that is gone already is not an error."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
