"""How a supervised child is put down: exited children keep their own exit code."""

import asyncio
import os
import select
import signal
import subprocess
import sys

import pytest

from osprey_connectors.process import reap_exit_code, terminate
from osprey_connectors_tests._paths import PACKAGE_SRC

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

    await terminate(child, 0.5)

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
    """A child whose exit the loop has already collected keeps its own exit
    code: putting it down signals nothing.
    """
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import os; os._exit(3)")
    await process.wait()

    assert await terminate(process, 0.5) == 3

    assert process.returncode == 3


@posix_only
async def test_a_child_that_exited_after_closing_its_stream_keeps_its_exit_code():
    """A supervisor puts a child down as soon as its stream ends, and such a
    child has usually exited already. The blocked loop holds that window open:
    the child has exited and the loop has not yet seen it when it is put down.

    The child exits only when told to, and only after the loop is blocked; the
    parent learns of the exit from a pipe the child holds open until it dies,
    so no timing guess decides which side of the window the test lands on.
    """
    go_read, go_write = os.pipe()
    exited_read, exited_write = os.pipe()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import os, sys; os.close(1); os.read(int(sys.argv[1]), 1); os._exit(3)",
        str(go_read),
        stdout=asyncio.subprocess.PIPE,
        pass_fds=(go_read, exited_write),
    )
    os.close(go_read)
    os.close(exited_write)
    try:
        assert await process.stdout.read() == b""
        # From here the loop stays blocked until the child is gone.
        os.write(go_write, b"x")
        ready, _, _ = select.select([exited_read], [], [], 30.0)
        assert ready, "the child did not exit after it was told to"
        assert os.read(exited_read, 1) == b""
    finally:
        os.close(go_write)
        os.close(exited_read)
    assert process.returncode is None

    assert await terminate(process, 5.0) == 3

    assert process.returncode == 3


async def test_a_child_the_loop_already_collected_is_left_alone():
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import os; os._exit(4)")
    assert await process.wait() == 4

    assert await terminate(process, 0.5) == 4

    assert process.returncode == 4


@posix_only
async def test_a_running_child_is_still_terminated():
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)"
    )

    await terminate(process, 0.5)

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

    await terminate(process, 0.5)

    assert process.returncode == -signal.SIGKILL


#: Seconds a child that has closed its pipe gets to finish exiting on a loaded host.
_EXIT_WAIT_S = 30.0


@posix_only
async def test_terminate_returns_the_reaped_code():
    exited = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import os\nprint('exiting', flush=True)\nos._exit(3)\n",
        stdout=asyncio.subprocess.PIPE,
    )
    # The pipe closes while the child dies, before it has exited; the wait makes
    # terminate meet an exited child.
    assert await asyncio.wait_for(exited.stdout.read(), 10.0) == b"exiting\n"
    await asyncio.wait_for(exited.wait(), _EXIT_WAIT_S)
    assert await terminate(exited, 0.5) == 3

    sleeping = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)"
    )
    assert await terminate(sleeping, 0.5) == -signal.SIGTERM


def test_the_module_imports_only_the_standard_library():
    src = PACKAGE_SRC
    code = (
        "import sys\n"
        "import osprey_connectors.process\n"
        "leaked = [m for m in sys.modules\n"
        "          if m == 'numpy' or m == 'osprey' or m.startswith('osprey.')\n"
        "          or m.startswith('osprey_connectors.ipc')]\n"
        "print('LEAKED', leaked) if leaked else print('CLEAN')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(src)),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout, result.stdout
