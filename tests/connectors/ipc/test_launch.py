"""Ending a connector-host child without losing what it exited with."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

import pytest

from osprey_connectors.ipc.launch import terminate_host

#: A child that exits on its own is often still unreaped when it is terminated;
#: enough rounds that a platform where that window is a race lands in it.
ROUNDS = 25


async def _exited_but_unreaped(code: int) -> asyncio.subprocess.Process:
    """A child that has exited with *code*, held where the loop has not yet reaped it.

    ``waitid(WNOWAIT)`` blocks until the child is a zombie without reaping it,
    and the loop is not yielded to in between. Where the loop's child watcher
    runs on another thread it may reap first; the round then shows nothing,
    which is why the test runs several.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", f"import os; os._exit({code})"
    )
    with contextlib.suppress(ChildProcessError):
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    return process


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="needs waitid(WNOWAIT)")
async def test_terminating_a_child_that_already_exited_keeps_its_own_exit_code():
    for _ in range(ROUNDS):
        process = await _exited_but_unreaped(3)
        await terminate_host(process, 5.0)
        assert process.returncode == 3
