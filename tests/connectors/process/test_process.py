"""How a supervised child is put down: exited children keep their own exit code."""

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from osprey_connectors.process import (
    ChildExit,
    ExitCause,
    JobSlots,
    reap_exit_code,
    terminate,
)

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
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import os; os._exit(3)")

    await terminate(process, 0.5)

    assert process.returncode == 3


@posix_only
async def test_a_child_that_exited_after_closing_its_stream_keeps_its_exit_code():
    """A supervisor puts a child down as soon as its stream ends, and such a
    child has usually exited already. The blocked loop holds that window open:
    the child has exited and the loop has not yet seen it when it is put down.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import os, time; os.close(1); time.sleep(0.05); os._exit(3)",
        stdout=asyncio.subprocess.PIPE,
    )
    assert await process.stdout.read() == b""
    time.sleep(0.5)
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


@posix_only
async def test_terminate_returns_the_reaped_code():
    exited = await asyncio.create_subprocess_exec(sys.executable, "-c", "import os; os._exit(3)")
    assert await terminate(exited, 0.5) == 3

    sleeping = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)"
    )
    assert await terminate(sleeping, 0.5) == -signal.SIGTERM


_SLEEP = "import time; time.sleep(60)"


def _python(code: str, *args: str) -> list[str]:
    return [sys.executable, "-c", code, *args]


async def test_a_job_that_exits_zero_is_completed():
    job = JobSlots().reserve("worker")
    await job.start(_python("pass"))

    exit_ = await job.wait()

    assert exit_ == ChildExit(job.job, ExitCause.COMPLETED, 0, "")


async def test_a_job_that_exits_nonzero_unasked_is_failed_with_its_code():
    job = JobSlots().reserve("worker")
    await job.start(_python("import os; os._exit(3)"))

    exit_ = await job.wait()

    assert (exit_.cause, exit_.returncode) == (ExitCause.FAILED, 3)


@posix_only
async def test_a_job_stopped_after_its_own_sigterm_is_cancelled_not_failed():
    job = JobSlots().reserve("worker")
    await job.start(_python(_SLEEP))

    exit_ = await job.stop()

    assert (exit_.cause, exit_.returncode) == (ExitCause.CANCELLED, -signal.SIGTERM)
    assert await job.wait() is exit_


@posix_only
async def test_a_job_past_its_deadline_is_timed_out():
    job = JobSlots().reserve("worker")
    await job.start(_python(_SLEEP))

    exit_ = await job.wait(deadline_s=0.2)

    assert exit_.cause is ExitCause.TIMED_OUT
    assert exit_.returncode == job.process.returncode == -signal.SIGTERM


@posix_only
async def test_the_first_asked_cause_wins():
    job = JobSlots().reserve("worker")
    await job.start(_python(_SLEEP))

    job.asked(ExitCause.TIMED_OUT)
    exit_ = await job.stop(ExitCause.CANCELLED)

    assert exit_.cause is ExitCause.TIMED_OUT


async def test_reserve_mints_the_job_id_before_start():
    job = JobSlots().reserve("worker")
    assert job.process is None
    assert job.pid is None

    await job.start(
        _python("import sys; print(sys.argv[1], flush=True)", str(job.job)),
        stdout=asyncio.subprocess.PIPE,
    )
    printed = await asyncio.wait_for(job.process.stdout.read(), 10.0)
    await job.wait()

    assert printed.strip().decode() == str(job.job)
    assert job.pid == job.process.pid


@posix_only
async def test_a_new_reservation_makes_the_old_job_non_current_and_reaps_it_before_the_new_spawn():
    slots = JobSlots()
    old = slots.reserve("worker")
    await old.start(_python(_SLEEP))
    old_pid = old.pid

    new = slots.reserve("worker")
    assert not slots.is_current(old)
    assert slots.is_current(new)
    assert slots.current("worker") is new

    # The new child probes the old pid as its first act, so it reports what
    # existed at the moment it was spawned.
    probe = (
        "import os, sys\n"
        "try:\n"
        "    os.kill(int(sys.argv[1]), 0)\n"
        "    print('alive', flush=True)\n"
        "except ProcessLookupError:\n"
        "    print('gone', flush=True)\n"
    )
    await new.start(_python(probe, str(old_pid)), stdout=asyncio.subprocess.PIPE)
    seen = await asyncio.wait_for(new.process.stdout.read(), 10.0)
    await new.wait()

    assert seen.strip() == b"gone"
    with pytest.raises(ProcessLookupError):
        os.kill(old_pid, 0)
    assert (await old.wait()).cause is ExitCause.CANCELLED


async def test_a_job_superseded_before_it_starts_spawns_nothing():
    slots = JobSlots()
    old = slots.reserve("worker")
    slots.reserve("worker")

    await old.start(_python(_SLEEP))

    assert old.process is None
    exit_ = await old.wait()
    assert (exit_.cause, exit_.returncode) == (ExitCause.CANCELLED, None)


@posix_only
async def test_stop_all_returns_only_after_every_child_is_reaped():
    slots = JobSlots()
    jobs = [slots.reserve(name) for name in ("a", "b", "c")]
    for job in jobs:
        await job.start(_python(_SLEEP))

    exits = await slots.stop_all()

    assert sorted(exit_.job for exit_ in exits) == sorted(job.job for job in jobs)
    assert all(job.process.returncode is not None for job in jobs)
    assert all(exit_.cause is ExitCause.CANCELLED for exit_ in exits)
    assert [slots.current(name) for name in ("a", "b", "c")] == [None, None, None]


def test_job_ids_are_unique_across_slots():
    first, second = JobSlots(), JobSlots()

    ids = [first.reserve("a").job, first.reserve("b").job, second.reserve("a").job]

    assert len(set(ids)) == len(ids)


async def test_a_job_writing_a_mebibyte_to_stderr_still_completes_and_keeps_its_tail():
    code = (
        "import sys\n"
        "data = bytes(65 + i % 26 for i in range(1 << 20))\n"
        "sys.stderr.buffer.write(data)\n"
        "sys.stderr.flush()\n"
    )
    expected = bytes(65 + i % 26 for i in range(1 << 20))[-500:].decode()
    job = JobSlots().reserve("worker")
    await job.start(_python(code), collect_stderr=500)

    exit_ = await job.wait(deadline_s=30.0)

    assert exit_.cause is ExitCause.COMPLETED
    assert exit_.stderr_tail == expected


def test_the_module_imports_only_the_standard_library():
    src = Path(__file__).resolve().parents[3] / "packages" / "osprey-connectors" / "src"
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
