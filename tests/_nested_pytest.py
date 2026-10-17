"""Run a nested pytest to its end, stopped only when it stops making progress.

A nested run's wall time grows with how loaded the host is, so a fixed deadline
fails a slow run that would have passed. A working run keeps writing: its
progress, its summary, and the reports plugins print after the summary. So the
run is waited on for as long as its stdout or stderr keeps growing, and stopped
only after ``STALL_SECONDS`` in which neither grew. The streams go to files
rather than pipes so their growth can be observed while the run is alive.

A nested run's temporary directories live under a root inside the helper's
scratch, named to it by ``PYTEST_DEBUG_TEMPROOT``. pytest prunes the older
numbered ``pytest-N`` directories of its root when it exits, and the shared
``pytest-of-<user>`` root holds what every other run on the host left behind,
often a whole suite's ``tmp_path`` tree. Pruning there would make the nested
run spend minutes deleting another run's files after its summary, writing
nothing. In its own root it prunes only itself.

A stopped run is reported with the Python stack of every thread of every
process it started: the run gets its own process group and ``faulthandler``,
and the stop is ``stop_group`` from ``scripts/run_bounded.py``, the sequence
every bounded gate run is stopped with: a ``SIGABRT`` to that group, which
makes each process write its stacks to the same stderr before it dies, then
SIGTERM and SIGKILL. Whatever the run left in its group is killed when it
ends, so no worker it started outlives the calling test.

A finished run reports how long it sat silent before it exited
(``NestedRun.quiet_before_exit``): the time between the last growth of its
streams and its exit, which for a pytest run is the time it took to end after
its last word.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

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

_RUN_BOUNDED = Path(__file__).resolve().parents[1] / "scripts" / "run_bounded.py"


def _load_stop_group():
    """Load ``stop_group`` from the gate scripts' bounded runner by path."""
    spec = importlib.util.spec_from_file_location("_run_bounded", _RUN_BOUNDED)
    assert spec is not None and spec.loader is not None, _RUN_BOUNDED
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stop_group


_stop_group = _load_stop_group()


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


def run_nested_pytest(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str] | None,
    stall_seconds: float = STALL_SECONDS,
) -> NestedRun:
    """Run ``argv`` to its exit, failing the calling test if it stalls.

    Args:
        argv: The command, normally ``[sys.executable, "-m", "pytest", ...]``.
        cwd: The directory it runs in.
        env: Its environment; ``None`` inherits this process's. Output is made
            unbuffered, so what the child writes is visible at once, and the
            run's temporary root is a directory of its own, either way.
        stall_seconds: How long it may write nothing before it is stopped.

    Returns:
        The finished run, with its stdout and stderr as text and how long it
        sat silent before it exited.
    """
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONFAULTHANDLER"] = "1"
    with tempfile.TemporaryDirectory(prefix="nested-pytest-") as scratch:
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
                while proc.poll() is None:
                    size = out_path.stat().st_size + err_path.stat().st_size
                    now = time.monotonic()
                    if size != written:
                        written, last_progress = size, now
                    elif now - last_progress > stall_seconds:
                        _stop_group(
                            proc.pid,
                            dump=_DUMP_SECONDS,
                            grace=_GRACE_SECONDS,
                            wait=proc.wait,
                        )
                        pytest.fail(
                            f"nested run wrote no new output for {stall_seconds} s and was "
                            f"stopped\n--- stdout ---\n{out_path.read_text()}"
                            f"\n--- stderr ---\n{err_path.read_text()}"
                        )
                    time.sleep(_POLL_SECONDS)
                exited = time.monotonic()
                if out_path.stat().st_size + err_path.stat().st_size != written:
                    last_progress = exited
            finally:
                _signal_group(proc.pid, signal.SIGKILL)
                proc.wait()
        return NestedRun(
            argv,
            proc.returncode,
            out_path.read_text(),
            err_path.read_text(),
            quiet_before_exit=exited - last_progress,
        )


def _signal_group(pgid: int, signum: signal.Signals) -> None:
    """Send ``signum`` to every process left in the run's group, if any."""
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass
