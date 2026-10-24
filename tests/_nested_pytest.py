"""Run a nested pytest to its end, stopped only when it stops making progress.

A nested run's wall time grows with how loaded the host is, so a fixed deadline
fails a slow run that would have passed. A working run keeps writing: its
progress, its summary, and the reports plugins print after the summary. So the
run is waited on for as long as its stdout or stderr keeps growing, and stopped
only after ``STALL_SECONDS`` in which neither grew. The streams go to files
rather than pipes so their growth can be observed while the run is alive.

A nested run gets a temporary root of its own, named to it by
``PYTEST_DEBUG_TEMPROOT``, unless the caller shares one on purpose
(``temproot=``). A nested run that does not load ``tests/conftest.py`` numbers
its directories under ``pytest-of-<user>`` in that root and prunes the older
``pytest-N`` ones when it exits; the shared ``pytest-of-<user>`` of the host
temp dir holds what every other such run left behind, so pruning there would
make the nested run spend minutes deleting another run's files after its
summary, writing nothing. In its own root it prunes only itself.

A stopped run is reported with the Python stack of every thread of every
process it started: the run gets its own process group and ``faulthandler``,
and the stop is the product's ``osprey.deployment.subprocess_capture.stop_group``,
the sequence every bounded gate run is stopped with: a ``SIGABRT`` to that
group, which makes each process write its stacks to the same stderr before it
dies, then SIGTERM and SIGKILL. Whatever the run left in its group is stopped
the same way when it ends, so no worker it started outlives the calling test,
and one that survives SIGKILL fails the test by pid, and one it could not look
for fails it saying so.

A finished run reports how long it sat silent before it exited
(``NestedRun.quiet_before_exit``): the time between the last growth of its
streams and its exit, which for a pytest run is the time it took to end after
its last word.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from osprey.deployment.subprocess_capture import stop_group
from osprey_connectors_tests._paths import TESTS_ROOT as HARNESS_ROOT

#: How long a nested run may write nothing before it counts as hung. It spans
#: the run's silent phases (interpreter start, collection, worker start-up, the
#: live-thread report's grace), not the run as a whole.
STALL_SECONDS = 120.0

#: How often the run's streams are checked for growth.
_POLL_SECONDS = 0.2

#: How long a stopped run's processes get to write their stacks before the
#: group is sent SIGTERM.
_DUMP_SECONDS = 5.0

#: How long a stopped run gets after SIGTERM before the group is killed.
_GRACE_SECONDS = 5.0

#: Prefixes of the keys an outer pytest run sets for itself that must not reach
#: a nested one: its worker id, its options and its diagnostics directory.
_OUTER_RUN_KEYS = ("PYTEST_XDIST_", "PYTEST_ADDOPTS", "OSPREY_CI_DIAG_DIR")


class NestedRun(subprocess.CompletedProcess[str]):
    """A finished nested run, with how long it sat silent before it exited."""

    def __init__(
        self,
        args: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
        quiet_before_exit: float,
    ) -> None:
        super().__init__(list(args), returncode, stdout, stderr)
        #: Seconds between the last growth of stdout or stderr and the exit,
        #: to the resolution of the poll (``_POLL_SECONDS``).
        self.quiet_before_exit = quiet_before_exit


def outer_run_free_env(also_drop: Sequence[str] = ()) -> dict[str, str]:
    """This process's environment without the keys the outer pytest run set for itself.

    Args:
        also_drop: Further key prefixes the caller decides for itself.

    Returns:
        A copy of ``os.environ`` without any key starting with one of the
        outer run's prefixes or *also_drop*.
    """
    prefixes = (*_OUTER_RUN_KEYS, *also_drop)
    return {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}


def with_harness_path(env: dict[str, str]) -> dict[str, str]:
    """*env* with the shared test harness prepended to ``PYTHONPATH``.

    The outer run reaches ``osprey_connectors_tests`` through the root
    pyproject's ``pythonpath`` setting, which a child process does not read.
    """
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(HARNESS_ROOT), env.get("PYTHONPATH"))))
    return env


def run_nested_pytest(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str] | None,
    stall_seconds: float = STALL_SECONDS,
    temproot: Path | None = None,
) -> NestedRun:
    """Run ``argv`` to its exit, failing the calling test if it stalls.

    Args:
        argv: The command, normally ``[sys.executable, "-m", "pytest", ...]``.
        cwd: The directory it runs in.
        env: Its environment; ``None`` inherits this process's. Output is made
            unbuffered, so what the child writes is visible at once, and the
            run's temporary root is ``temproot``, either way.
        stall_seconds: How long it may write nothing before it is stopped.
        temproot: The run's ``PYTEST_DEBUG_TEMPROOT``; ``None`` gives it a root
            of its own inside the helper's scratch, which goes with the
            helper's return. A path is used as is, so several runs can share it.

    Returns:
        The finished run, with its stdout and stderr as text and how long it
        sat silent before it exited.
    """
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONFAULTHANDLER"] = "1"
    with tempfile.TemporaryDirectory(prefix="nested-pytest-") as scratch:
        if temproot is None:
            temproot = Path(scratch) / "temproot"
            temproot.mkdir()
        child_env["PYTEST_DEBUG_TEMPROOT"] = str(temproot)
        out_path = Path(scratch) / "stdout"
        err_path = Path(scratch) / "stderr"
        with out_path.open("w") as out, err_path.open("w") as err:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                written = -1
                last_progress = time.monotonic()
                while not _exited(proc):
                    size = out_path.stat().st_size + err_path.stat().st_size
                    now = time.monotonic()
                    if size != written:
                        written, last_progress = size, now
                    elif now - last_progress > stall_seconds:
                        survivors = stop_group(proc.pid, dump=_DUMP_SECONDS, grace=_GRACE_SECONDS)
                        pytest.fail(
                            f"nested run wrote no new output for {stall_seconds} s and was "
                            f"stopped{_survivor_note(survivors)}"
                            f"\n--- stdout ---\n{out_path.read_text()}"
                            f"\n--- stderr ---\n{err_path.read_text()}"
                        )
                    time.sleep(_POLL_SECONDS)
                exited = time.monotonic()
                if out_path.stat().st_size + err_path.stat().st_size != written:
                    last_progress = exited
            finally:
                leftover = stop_group(proc.pid, grace=_GRACE_SECONDS)
                proc.wait()
            if leftover is None:
                pytest.fail(
                    "could not tell whether the nested run left processes behind: "
                    "the process table could not be read"
                )
            elif leftover:
                pytest.fail(f"nested run left processes behind{_survivor_note(leftover)}")
        return NestedRun(
            argv,
            proc.returncode,
            out_path.read_text(),
            err_path.read_text(),
            quiet_before_exit=exited - last_progress,
        )


def _exited(proc: subprocess.Popen[bytes]) -> bool:
    """Whether ``proc`` has exited, observed without reaping it.

    The leader stays unreaped until the group stop has run, so its id cannot be
    taken by another process while the stop signals and reads the group.
    """
    return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _survivor_note(survivors: list[int] | None) -> str:
    """The clause naming a stop's survivors, or nothing when there are none."""
    if survivors is None:
        return "; the process table could not be read to name what is still running"
    if not survivors:
        return ""
    return "; still running after SIGKILL: pids " + " ".join(str(pid) for pid in survivors)
