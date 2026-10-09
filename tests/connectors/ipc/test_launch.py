"""Putting a connector-host child down: the exit status survives the supervisor.

The supervisor puts a child down as soon as its stream ends, and a child that
ended its stream has usually exited already. These tests hold that window open
on purpose, so the child has exited while the event loop has not yet seen it
when :func:`terminate_host` runs. Whatever :func:`terminate_host` does in that
window must leave the child for the loop's own watcher to reap, or asyncio
reports the exit status as 255.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time

import pytest

from osprey_connectors.ipc.launch import terminate_host

GRACE_S = 5.0

#: A child that exits on its own is often still unreaped when it is terminated;
#: enough rounds that a platform where that window is a race lands in it.
ROUNDS = 25


async def _spawn(code: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE
    )


async def _exited_but_unreaped(code: int) -> asyncio.subprocess.Process:
    """A child that has exited with *code*, held where the loop has not yet reaped it.

    ``waitid(WNOWAIT)`` blocks until the child is a zombie without reaping it,
    and the loop is not yielded to in between. Where the loop's child watcher
    runs on another thread it may reap first; the round then shows nothing,
    which is why the test runs several.
    """
    process = await _spawn(f"import os; os._exit({code})")
    with contextlib.suppress(ChildProcessError):
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    return process


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="needs waitid(WNOWAIT)")
async def test_terminating_a_child_that_already_exited_keeps_its_own_exit_code():
    for _ in range(ROUNDS):
        process = await _exited_but_unreaped(3)
        await terminate_host(process, GRACE_S)
        assert process.returncode == 3


async def test_a_child_that_exited_after_closing_its_stream_keeps_its_exit_code():
    process = await _spawn("import os, time; os.close(1); time.sleep(0.05); os._exit(3)")
    assert await process.stdout.read() == b""
    # Blocks the loop: the child exits now, and the loop has not seen it yet.
    time.sleep(0.5)
    assert process.returncode is None

    await terminate_host(process, GRACE_S)

    assert process.returncode == 3


async def test_a_running_child_is_terminated():
    process = await _spawn("import sys, time; print('up', flush=True); time.sleep(60)")
    assert await process.stdout.readline() == b"up\n"

    await terminate_host(process, GRACE_S)

    assert process.returncode == -signal.SIGTERM


async def test_a_child_that_ignores_sigterm_is_killed_after_the_grace():
    process = await _spawn(
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('up', flush=True); time.sleep(60)"
    )
    assert await process.stdout.readline() == b"up\n"

    await terminate_host(process, 0.5)

    assert process.returncode == -signal.SIGKILL


async def test_a_child_already_reaped_is_left_alone():
    process = await _spawn("import os; os._exit(4)")
    assert await process.wait() == 4

    await terminate_host(process, GRACE_S)

    assert process.returncode == 4
