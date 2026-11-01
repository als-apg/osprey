"""Tests for the wall-clock bound the check command puts on each pytest run.

The wrapper's ``main`` is only ever run as a subprocess, never called here: it
installs signal handlers and exits the process, neither of which belongs in the
test process. Its ``stop_group`` is importable by path, since loading the
module installs no handler, so the group stop is proved by calling it directly
on a group that has said it is ready: the readiness line is a handshake, not a
clock, and these tests run in the parallel sweep. The clocks left in them are
the stop's own: the dump test's child, already running and asleep, answers
SIGABRT inside the dump and SIGTERM inside the grace, and a child starved for
that long fails it.

Two wrapper tests keep a readiness window and carry the ``wallclock`` marker.
For them the bound is the only stimulus, the child must be running when it
fires, and the wrapper offers no hold between the two: a Python child prints
its stacks only once its interpreter is up with faulthandler on, and a child
ignores SIGTERM only once its trap is set. Every window and ceiling there is
written from the values the test passes to the wrapper.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests._live import wait_until

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _REPO_ROOT / "scripts" / "run_bounded.py"
CHECK_SCRIPT = _REPO_ROOT / "scripts" / "check.py"
WRAPPER = ["uv", "run", "python", "scripts/run_bounded.py"]

#: The bound handed to the wrapper; for the ``wallclock`` tests it is also the
#: child's readiness window.
BOUND_S = 3.0
#: The SIGABRT-to-SIGTERM wait passed to every stop.
DUMP_S = 0.5
#: The SIGTERM-to-SIGKILL wait passed to every stop.
GRACE_S = 1.0


def _load_by_path(name: str, path: Path) -> Iterator[ModuleType]:
    """The script at *path* loaded under *name*, since scripts/ is not a package.

    The module is registered in sys.modules before exec so @dataclass can
    resolve annotations through cls.__module__, under a name no import uses.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def check() -> Iterator[ModuleType]:
    """``scripts/check.py`` loaded by path."""
    yield from _load_by_path("_check_by_path_for_bounds", CHECK_SCRIPT)


@pytest.fixture(scope="module")
def run_bounded() -> Iterator[ModuleType]:
    """``scripts/run_bounded.py`` loaded by path; loading it installs no handler."""
    yield from _load_by_path("_run_bounded_by_path", SCRIPT)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _bounded(*command: str) -> subprocess.CompletedProcess[str]:
    """*command* run under the wrapper with this module's bound, dump and grace."""
    return _run("--dump", f"{DUMP_S:g}", "--grace", f"{GRACE_S:g}", f"{BOUND_S:g}", "--", *command)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@contextlib.contextmanager
def _group(argv: list[str]) -> Iterator[subprocess.Popen[str]]:
    """*argv* started as the leader of its own process group, stdout piped.

    The test stops the group itself; on the way out whatever is left of it is
    killed and the pipe is closed, so a failing assertion leaves nothing behind.
    """
    proc = subprocess.Popen(argv, process_group=0, stdout=subprocess.PIPE, text=True)
    try:
        yield proc
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        assert proc.stdout is not None
        proc.stdout.close()


def _stop(run_bounded: ModuleType, proc: subprocess.Popen[str]) -> float:
    """Stop *proc*'s group with this module's dump and grace; return the seconds it took."""
    started = time.monotonic()
    run_bounded.stop_group(proc.pid, dump=DUMP_S, grace=GRACE_S, wait=proc.wait)
    return time.monotonic() - started


def test_the_command_exit_code_passes_through():
    result = _run("30", "--", sys.executable, "-c", "import sys; sys.exit(3)")
    assert result.returncode == 3


def test_a_command_past_its_bound_exits_124_and_says_so():
    result = _bounded("sh", "-c", "sleep 60 & wait")
    assert result.returncode == 124
    assert f"still running after {BOUND_S:g} s" in result.stderr


def test_a_command_whose_leader_dies_first_exits_124_and_says_so(tmp_path):
    """The gate shape: a shell leader that dies of SIGABRT at once, Python beneath it."""
    script = tmp_path / "child.py"
    script.write_text("import time\nwhile True:\n    time.sleep(60)\n")
    # The child's streams are not the wrapper's pipes, so a child that
    # outlives the stop cannot hold them open.
    leader = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} >/dev/null 2>&1; :"
    result = _bounded("sh", "-c", leader)
    assert result.returncode == 124
    assert f"still running after {BOUND_S:g} s" in result.stderr


def test_a_stop_reaches_the_grandchildren_of_the_group(run_bounded):
    with _group(["sh", "-c", "sleep 60 & echo $!; wait"]) as proc:
        assert proc.stdout is not None
        grandchild = int(proc.stdout.readline())
        try:
            _stop(run_bounded, proc)
            # A stop that ends in SIGKILL waits for the leader alone, so an
            # orphan can outlast its return until init reaps it.
            wait_until(
                lambda: not _alive(grandchild), timeout=3 * GRACE_S, what="the grandchild gone"
            )
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(grandchild, signal.SIGKILL)


def test_a_stop_waits_out_the_dump_for_the_group_when_its_leader_dies_first(run_bounded, tmp_path):
    """The gate shape: a leader that dies of SIGABRT at once, Python beneath it.

    ``uv run`` leads every gate's group and dies of SIGABRT without writing
    anything, while pytest beneath it is still writing its stacks. A shell
    stands in for it here. The Python child starts "writing its stacks" on
    SIGABRT and never finishes, so SIGTERM finds the dump done only if the stop
    waited out the dump for the group rather than for the leader; a stop that
    waited for the leader alone sends no SIGTERM at all, and no record is
    written.
    """
    dumped = tmp_path / "dumped"
    termed = tmp_path / "termed"
    child = textwrap.dedent(
        f"""
        import os, pathlib, signal, time
        def on_abort(*_):
            pathlib.Path({str(dumped)!r}).touch()
            while True:
                time.sleep(60)
        def on_term(*_):
            pathlib.Path({str(termed)!r}).write_text(str(pathlib.Path({str(dumped)!r}).exists()))
        signal.signal(signal.SIGABRT, on_abort)
        signal.signal(signal.SIGTERM, on_term)
        print(os.getpid(), flush=True)
        while True:
            time.sleep(60)
        """
    )
    script = tmp_path / "child.py"
    script.write_text(child)
    leader = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}; :"
    with _group(["sh", "-c", leader]) as proc:
        assert proc.stdout is not None
        pid = int(proc.stdout.readline())
        try:
            elapsed = _stop(run_bounded, proc)
            assert termed.exists(), "the group was never sent SIGTERM"
            assert termed.read_text() == "True", "SIGTERM was sent before the stacks were written"
            assert elapsed >= DUMP_S + GRACE_S, "the stop did not wait out the dump and the grace"
            # This stop ends in SIGKILL, which leaves the orphan to init's reap.
            wait_until(lambda: not _alive(pid), timeout=3 * GRACE_S, what="the child gone")
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_a_stop_kills_a_group_that_ignores_sigterm_after_the_grace(run_bounded):
    child = textwrap.dedent(
        """
        import signal, time
        signal.signal(signal.SIGABRT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ready", flush=True)
        time.sleep(60)
        """
    )
    with _group([sys.executable, "-c", child]) as proc:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready"
        elapsed = _stop(run_bounded, proc)
        assert proc.returncode == -signal.SIGKILL
        assert elapsed >= DUMP_S + GRACE_S


@pytest.mark.wallclock
def test_the_wrapper_passes_its_dump_and_grace_to_the_stop(run_bounded, tmp_path):
    marker = tmp_path / "sigterm-ignored"
    # An ignored signal stays ignored across exec, so the marker proves the
    # sleep that receives SIGABRT and SIGTERM is the one that ignores them.
    command = [
        "sh",
        "-c",
        f"trap '' TERM ABRT; : > {shlex.quote(str(marker))}; exec sleep 60",
    ]
    defaults = run_bounded._parse([f"{BOUND_S:g}", "--", "true"])
    started = time.monotonic()
    result = _bounded(*command)
    elapsed = time.monotonic() - started
    assert result.returncode == 124
    assert marker.exists(), (
        f"the child was not ready inside the {BOUND_S:g} s bound: SIGTERM was never ignored"
    )
    # Bound plus dump wait plus grace: anything shorter means the command died
    # of SIGABRT or SIGTERM and the SIGKILL path never ran.
    assert elapsed >= BOUND_S + DUMP_S + GRACE_S
    # Each default exceeds the value passed, so a wrapper that dropped either
    # flag takes at least this long.
    assert elapsed < BOUND_S + defaults.dump + GRACE_S


@pytest.mark.wallclock
def test_a_command_past_its_bound_prints_its_python_stacks_before_it_is_stopped(tmp_path):
    marker = tmp_path / "ready"
    child = f"import pathlib, time; pathlib.Path({str(marker)!r}).touch(); time.sleep(60)"
    result = _bounded(sys.executable, "-c", child)
    assert result.returncode == 124
    assert marker.exists(), f"the child was not ready inside the {BOUND_S:g} s bound"
    assert "most recent call first" in result.stderr


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


def _pytest_checks(checks: tuple[Any, ...]) -> list[Any]:
    """The ``tests`` checks of *checks* whose command is a pytest run."""
    return [c for c in checks if c.kind == "tests" and "pytest" in c.argv]


def _unbounded_pytest_checks(check: ModuleType, checks: tuple[Any, ...]) -> list[str]:
    """What leaves a pytest check of *checks* unbounded, one line per defect.

    A check is bounded when it declares a bound and every tier that runs it
    resolves its command through ``scripts/run_bounded.py`` with a bound.
    """
    problems = []
    for declared in _pytest_checks(checks):
        if declared.bound_s <= 0:
            problems.append(f"{declared.id}: declares no bound")
        for tier in sorted(declared.tiers):
            context = check.Context(tier=tier)
            for argv in check.commands(declared, check.Selection(check.WHOLE), context):
                bounded = argv[:4] == WRAPPER and "--" in argv and float(argv[4]) > 0
                if not bounded:
                    problems.append(f"{declared.id}: {tier} runs it without {WRAPPER[-1]}")
    return problems


def test_every_pytest_check_resolves_through_the_bound(check):
    assert [c.id for c in _pytest_checks(check.CHECKS)] == [
        "pytest",
        "pytest-pty",
        "pytest-wallclock",
        "pytest-wallclock-unit",
    ]
    assert _unbounded_pytest_checks(check, check.CHECKS) == []


def test_every_pytest_check_resolves_through_the_bound__mutation_a_bound_of_zero_is_reported(
    check,
):
    mutated = tuple(
        dataclasses.replace(c, bound_s=0) if c.id == "pytest-pty" else c for c in check.CHECKS
    )
    assert _unbounded_pytest_checks(check, mutated) == [
        "pytest-pty: declares no bound",
        "pytest-pty: full runs it without scripts/run_bounded.py",
    ]
