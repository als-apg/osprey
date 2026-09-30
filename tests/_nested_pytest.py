"""Run a nested pytest to its end, stopped only when it stops making progress.

A nested run's wall time grows with how loaded the host is, so a fixed deadline
fails a slow run that would have passed. A working run keeps writing: its
progress, its summary, and the reports plugins print after the summary. So the
run is waited on for as long as its stdout or stderr keeps growing, and stopped
only after ``STALL_SECONDS`` in which neither grew. The streams go to files
rather than pipes so their growth can be observed while the run is alive.

A stopped run is reported with the Python stack of every thread of every
process it started: the run gets its own process group and ``faulthandler``,
and the stop is a ``SIGABRT`` to that group, which makes each process write its
stacks to the same stderr before it dies. Whatever the run left in its group is
killed when it ends, so no worker it started outlives the calling test.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

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
#: group is killed outright.
_DUMP_SECONDS = 5.0


def run_nested_pytest(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str] | None,
    stall_seconds: float = STALL_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` to its exit, failing the calling test if it stalls.

    Args:
        argv: The command, normally ``[sys.executable, "-m", "pytest", ...]``.
        cwd: The directory it runs in.
        env: Its environment; ``None`` inherits this process's. Output is made
            unbuffered either way, so what the child writes is visible at once.
        stall_seconds: How long it may write nothing before it is stopped.

    Returns:
        The finished run, with its stdout and stderr as text.
    """
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONFAULTHANDLER"] = "1"
    with tempfile.TemporaryDirectory(prefix="nested-pytest-") as scratch:
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
                        _signal_group(proc.pid, signal.SIGABRT)
                        try:
                            proc.wait(_DUMP_SECONDS)
                        except subprocess.TimeoutExpired:
                            pass
                        _signal_group(proc.pid, signal.SIGKILL)
                        proc.wait()
                        pytest.fail(
                            f"nested run wrote no new output for {stall_seconds} s and was "
                            f"stopped\n--- stdout ---\n{out_path.read_text()}"
                            f"\n--- stderr ---\n{err_path.read_text()}"
                        )
                    time.sleep(_POLL_SECONDS)
            finally:
                _signal_group(proc.pid, signal.SIGKILL)
                proc.wait()
        return subprocess.CompletedProcess(
            list(argv), proc.returncode, out_path.read_text(), err_path.read_text()
        )


def _signal_group(pgid: int, signum: signal.Signals) -> None:
    """Send ``signum`` to every process left in the run's group, if any."""
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass
