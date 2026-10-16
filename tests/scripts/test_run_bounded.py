"""Tests for the wall-clock bound the local gate scripts put on each pytest run.

The wrapper's ``main`` is only ever run as a subprocess, never called here: it
installs signal handlers and exits the process, neither of which belongs in the
test process. Its ``stop_group`` is importable by path, since loading the
module installs no handler.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _REPO_ROOT / "scripts" / "run_bounded.py"
GATE_SCRIPTS = ("quick_check.sh", "premerge_check.sh", "ci_check.sh", "gate.sh")

_WRAPPER_PREFIX = re.compile(r"uv run python scripts/run_bounded\.py [0-9.]+ -- ")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _pid_is_gone(pid: int, within: float) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_the_command_exit_code_passes_through():
    result = _run("30", "--", sys.executable, "-c", "import sys; sys.exit(3)")
    assert result.returncode == 3


def test_a_command_past_its_bound_is_stopped_with_its_process_group(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    # The bound is the child's readiness window as well as the stop stimulus:
    # a shell is ready in milliseconds on an idle host and within a few seconds
    # on a loaded one, so three seconds is the window every bounded test here
    # gives its child.
    command = ["sh", "-c", f"sleep 60 & echo $! > {shlex.quote(str(pid_file))}; wait"]
    result = _run("3", "--", *command)
    assert result.returncode == 124
    assert "still running after 3 s" in result.stderr
    assert pid_file.exists(), "the child was not ready inside the 3 s bound: no grandchild pid file"
    pid = int(pid_file.read_text())
    try:
        assert _pid_is_gone(pid, within=5.0)
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_command_that_ignores_sigterm_is_killed_after_the_grace(tmp_path):
    marker = tmp_path / "sigterm-ignored"
    # An ignored signal stays ignored across exec, so the marker proves the
    # sleep that receives SIGABRT and SIGTERM is the one that ignores them.
    command = [
        "sh",
        "-c",
        f"trap '' TERM ABRT; : > {shlex.quote(str(marker))}; exec sleep 60",
    ]
    started = time.monotonic()
    result = _run("--dump", "1", "--grace", "1", "3", "--", *command)
    elapsed = time.monotonic() - started
    assert result.returncode == 124
    assert marker.exists(), (
        "the child was not ready inside the 3 s bound: SIGTERM was never ignored"
    )
    # Bound plus dump wait plus grace: anything shorter means the command died
    # of SIGABRT or SIGTERM and the SIGKILL path never ran.
    assert elapsed >= 5.0
    assert elapsed < 12


def test_a_command_past_its_bound_prints_its_python_stacks_before_it_is_stopped(tmp_path):
    marker = tmp_path / "ready"
    child = f"import pathlib, time; pathlib.Path({str(marker)!r}).touch(); time.sleep(60)"
    result = _run("--dump", "5", "--grace", "1", "3", "--", sys.executable, "-c", child)
    assert result.returncode == 124
    assert marker.exists(), "the child was not ready inside the 3 s bound"
    assert "most recent call first" in result.stderr


def test_a_stop_waits_for_the_whole_group_when_its_leader_dies_first(tmp_path):
    """The gate shape: a leader that dies of SIGABRT at once, Python beneath it.

    ``uv run`` leads every gate's group and dies of SIGABRT without writing
    anything, while pytest beneath it is still writing its stacks. A shell
    stands in for it here. The Python child takes a second to "write its
    stacks" and then ignores SIGTERM, so the stop is right only if it waits
    out the dump for the group rather than the leader, and kills what is left
    after the grace.
    """
    ready = tmp_path / "ready"
    dumped = tmp_path / "dumped"
    pid_file = tmp_path / "child.pid"
    child = textwrap.dedent(
        f"""
        import os, pathlib, signal, time
        def on_abort(*_):
            time.sleep(1.0)
            pathlib.Path({str(dumped)!r}).touch()
        signal.signal(signal.SIGABRT, on_abort)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
        pathlib.Path({str(ready)!r}).touch()
        while True:
            time.sleep(60)
        """
    )
    script = tmp_path / "child.py"
    script.write_text(child)
    # The child's streams are not the wrapper's pipes, so a child that
    # outlives the stop fails the assertions below instead of holding the
    # pipes open.
    leader = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} >/dev/null 2>&1; :"
    result = _run("--dump", "5", "--grace", "1", "3", "--", "sh", "-c", leader)
    try:
        assert result.returncode == 124
        assert ready.exists(), "the child was not ready inside the 3 s bound"
        assert dumped.exists(), "SIGTERM was sent before the group's stacks were written"
        pid = int(pid_file.read_text())
        assert _pid_is_gone(pid, within=1.0), "a process that survived SIGTERM was never killed"
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_sigint_reaches_the_command_and_the_wrapper_dies_of_it():
    child = textwrap.dedent(
        """
        import sys, time
        try:
            print("ready", flush=True)
            time.sleep(60)
        except KeyboardInterrupt:
            print("child-interrupted", flush=True)
            sys.exit(2)
        """
    )
    wrapper = subprocess.Popen(
        [sys.executable, str(SCRIPT), "30", "--", sys.executable, "-c", child],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert wrapper.stdout is not None
        assert wrapper.stdout.readline().strip() == "ready"
        wrapper.send_signal(signal.SIGINT)
        out, _ = wrapper.communicate(timeout=30)
        assert "child-interrupted" in out
        assert wrapper.returncode == -signal.SIGINT
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(10)


def _unbounded_pytest_lines(source: str) -> tuple[int, list[str]]:
    """Count the pytest invocations in *source* and return the ones with no bound."""
    invocations = [
        line
        for line in source.splitlines()
        if "uv run pytest " in line and not line.strip().startswith(("#", "echo"))
    ]
    return len(invocations), [line for line in invocations if "scripts/run_bounded.py" not in line]


@pytest.mark.parametrize("script", GATE_SCRIPTS)
def test_every_gate_script_bounds_its_pytest_runs(script):
    count, unbounded = _unbounded_pytest_lines((_REPO_ROOT / "scripts" / script).read_text())
    assert count >= 1, f"{script} has no pytest invocation"
    assert unbounded == [], f"{script} runs pytest without scripts/run_bounded.py"


@pytest.mark.parametrize("script", GATE_SCRIPTS)
def test_every_gate_script_bounds_its_pytest_runs__mutation_drops_the_wrapper(script):
    source = (_REPO_ROOT / "scripts" / script).read_text()
    mutated = _WRAPPER_PREFIX.sub("", source)
    assert mutated != source, f"{script} carries no wrapper prefix to drop"
    count, unbounded = _unbounded_pytest_lines(mutated)
    assert count >= 1
    assert unbounded, f"the check missed {script} with its wrapper removed"
