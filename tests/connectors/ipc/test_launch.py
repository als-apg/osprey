"""How a connector-host child is put down: exited children keep their own exit code."""

import asyncio
import signal
import sys

import pytest

from osprey_connectors.ipc.launch import reap_exit_code, terminate_host

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")


class _ExitedUnreapedChild:
    """A child that has exited but whose status the event loop has not collected yet.

    Signalling it models what a real signal does to such a child: the reap that
    precedes the signal steals the status, and the watcher records 255.
    """

    def __init__(self, code: int, after_s: float) -> None:
        self.returncode: int | None = None
        self.signals: list[str] = []
        self._code = code
        self._after_s = after_s

    async def wait(self) -> int:
        if self.returncode is None:
            await asyncio.sleep(self._after_s)
            if self.returncode is None:
                self.returncode = self._code
        return self.returncode

    def terminate(self) -> None:
        self.signals.append("terminate")
        self.returncode = 255

    def kill(self) -> None:
        self.signals.append("kill")
        self.returncode = 255


async def test_a_child_that_exited_unreaped_keeps_its_exit_code_and_is_not_signalled():
    child = _ExitedUnreapedChild(3, after_s=0.01)

    await terminate_host(child, 0.5)

    assert child.returncode == 3
    assert child.signals == []


async def test_reaping_an_exited_child_returns_its_exit_code():
    child = _ExitedUnreapedChild(5, after_s=0.01)

    assert await reap_exit_code(child, 0.5) == 5
    assert child.signals == []


async def test_reaping_a_running_child_gives_up_after_the_window_without_signalling_it():
    child = _ExitedUnreapedChild(5, after_s=60.0)

    assert await reap_exit_code(child, 0.05) is None
    assert child.signals == []


async def test_a_real_child_that_exits_on_its_own_reports_its_own_exit_code():
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import os; os._exit(3)")

    await terminate_host(process, 0.5)

    assert process.returncode == 3


@posix_only
async def test_a_running_child_is_still_terminated():
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)"
    )

    await terminate_host(process, 0.5)

    assert process.returncode == -signal.SIGTERM


@posix_only
async def test_a_child_that_ignores_sigterm_is_killed():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n",
        stdout=asyncio.subprocess.PIPE,
    )
    assert (await asyncio.wait_for(process.stdout.readline(), 10.0)).strip() == b"ready"

    await terminate_host(process, 0.5)

    assert process.returncode == -signal.SIGKILL
