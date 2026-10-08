"""Putting down the child processes a supervisor owns.

Every child a supervisor starts, whether a long-lived connector host or a
short job worker, is put down the same way, and this module is where that way
is spelled once. It holds to these invariants:

1. Nothing signals a child before the event loop has had the chance to reap
   its exit.
2. Every exit code reported comes from a reap.

It imports only the standard library and knows nothing about the control
system its callers talk to.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

__all__ = [
    "DEFAULT_TERMINATE_GRACE_S",
    "REAP_WINDOW_S",
    "reap_exit_code",
    "terminate",
]

#: How long a child gets between ``SIGTERM`` and ``SIGKILL``.
DEFAULT_TERMINATE_GRACE_S = 2.0

#: Longest a supervisor waits for the event loop to collect a child that may
#: already have exited, before it treats the child as running. Collection
#: follows the exit within milliseconds; the window only bounds the cost to a
#: child that is in fact still running.
REAP_WINDOW_S = 0.5


async def reap_exit_code(process: Any, window_s: float) -> int | None:
    """Wait up to *window_s* for the event loop to collect *process*'s exit.

    Returns the exit code, or ``None`` if the child is still running when the
    window closes. Never signals the child and never raises.
    """
    if process.returncode is None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), window_s)
    returncode: int | None = process.returncode
    return returncode


async def terminate(process: Any, grace_s: float) -> int | None:
    """``SIGTERM``, then ``SIGKILL`` after the grace period. Never raises.

    A child that already exited is collected by the event loop before anything
    signals it: a signal sent to an exited, uncollected child reaps it out from
    under the loop's watcher, which then loses the child's own exit code and
    reports 255. The wait for that collection is bounded by
    ``min(grace_s, REAP_WINDOW_S)``, so it only delays putting down a child
    that is in fact still running.

    Returns:
        The reaped exit code, or ``None`` if the child outlived ``SIGKILL``'s
        grace period.
    """
    if await reap_exit_code(process, min(grace_s, REAP_WINDOW_S)) is not None:
        return _returncode(process)
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), grace_s)
        return _returncode(process)
    except TimeoutError:
        pass
    with contextlib.suppress(OSError):
        process.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), grace_s)
    return _returncode(process)


def _returncode(process: Any) -> int | None:
    returncode: int | None = process.returncode
    return returncode
