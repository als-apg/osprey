"""Putting a connector-host child down: the exit status survives the supervisor.

The supervisor puts a child down as soon as its stream ends, and a child that
ended its stream has usually exited already. These tests hold that window open
on purpose: the child closes its stdout and only then exits, and the event loop
is kept from noticing the exit until :func:`terminate_host` has run. Whatever
:func:`terminate_host` does in that window must leave the child for the loop's
own watcher to reap, or asyncio reports the exit status as 255.
"""

import asyncio
import signal
import sys
import time

from osprey_connectors.ipc.launch import terminate_host

GRACE_S = 5.0


async def _spawn(code: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE
    )


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
