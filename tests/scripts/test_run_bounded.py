"""Tests for the wall-clock bound the local gate scripts put on each pytest run.

The wrapper is only ever run as a subprocess, never imported: it installs
signal handlers and exits the process, neither of which belongs in the test
process.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _REPO_ROOT / "scripts" / "run_bounded.py"
GATE_SCRIPTS = ("quick_check.sh", "premerge_check.sh", "ci_check.sh")

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
    child = textwrap.dedent(
        f"""
        import subprocess, time
        grandchild = subprocess.Popen(["sleep", "60"])
        with open({str(pid_file)!r}, "w") as f:
            f.write(str(grandchild.pid))
        time.sleep(60)
        """
    )
    result = _run("1", "--", sys.executable, "-c", child)
    assert result.returncode == 124
    assert "still running after 1 s" in result.stderr
    pid = int(pid_file.read_text())
    try:
        assert _pid_is_gone(pid, within=5.0)
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_command_that_ignores_sigterm_is_killed_after_the_grace():
    child = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    started = time.monotonic()
    result = _run("--grace", "1", "1", "--", sys.executable, "-c", child)
    assert result.returncode == 124
    assert time.monotonic() - started < 10


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
