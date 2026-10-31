"""Tests for the wall-clock bound the check command puts on each pytest run.

The wrapper's ``main`` is only ever run as a subprocess, never called here: it
installs signal handlers and exits the process, neither of which belongs in the
test process. The group stop it calls, ``stop_child``, lives in the product's
``osprey.deployment.subprocess_capture``; the stopped groups here are probed
with that module's ``group_is_gone``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
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

from osprey.deployment.subprocess_capture import group_is_gone
from tests._processes import kill_group

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _REPO_ROOT / "scripts" / "run_bounded.py"
CHECK_SCRIPT = _REPO_ROOT / "scripts" / "check.py"
#: The wrapper after ``uv run --frozen``, the prefix every bounded command carries.
WRAPPER = ["python", "scripts/run_bounded.py"]


@pytest.fixture(scope="module")
def check() -> Iterator[ModuleType]:
    """``scripts/check.py`` loaded by path, since scripts/ is not a package.

    The module is registered in sys.modules before exec so @dataclass can
    resolve annotations through cls.__module__, under a name no import uses.
    """
    name = "_check_by_path_for_bounds"
    spec = importlib.util.spec_from_file_location(name, CHECK_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _kill_recorded_group(pgid_file: Path) -> None:
    if pgid_file.exists() and pgid_file.read_text().strip():
        kill_group(int(pgid_file.read_text()))


def test_the_command_exit_code_passes_through():
    result = _run("30", "--", sys.executable, "-c", "import sys; sys.exit(3)")
    assert result.returncode == 3


def test_a_command_past_its_bound_is_stopped_with_its_process_group(tmp_path):
    group_file = tmp_path / "group.pid"
    # The bound is the child's readiness window as well as the stop stimulus:
    # a shell is ready in milliseconds on an idle host and within a few seconds
    # on a loaded one, so three seconds is the window every bounded test here
    # gives its child. The shell leads the group, so its own pid names it.
    command = ["sh", "-c", f"echo $$ > {shlex.quote(str(group_file))}; sleep 60 & wait"]
    try:
        result = _run("3", "--", *command)
        assert result.returncode == 124
        assert "still running after 3 s" in result.stderr
        assert group_file.exists(), "the child was not ready inside the 3 s bound: no group file"
        assert group_is_gone(int(group_file.read_text()))
    finally:
        _kill_recorded_group(group_file)


def test_a_command_that_ignores_sigterm_is_killed_after_the_grace(tmp_path):
    marker = tmp_path / "sigterm-ignored"
    # An ignored signal stays ignored across exec, so the marker proves the
    # sleep that receives SIGABRT and SIGTERM is the one that ignores them.
    # exec keeps the shell's pid, and the child leads its own group, so the
    # recorded pid names the group the sleep runs in.
    command = [
        "sh",
        "-c",
        f"trap '' TERM ABRT; echo $$ > {shlex.quote(str(marker))}; exec sleep 60",
    ]
    try:
        started = time.monotonic()
        result = _run("--dump", "1", "--grace", "1", "3", "--", *command)
        elapsed = time.monotonic() - started
        assert result.returncode == 124
        assert marker.exists(), (
            "the child was not ready inside the 3 s bound: SIGTERM was never ignored"
        )
        # A run as long as bound plus dump wait plus grace rules out a death by
        # SIGABRT or SIGTERM; a process that ignored both and is gone was ended
        # by the SIGKILL.
        assert elapsed >= 5.0
        assert group_is_gone(int(marker.read_text()))
    finally:
        _kill_recorded_group(marker)


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
    child = textwrap.dedent(
        f"""
        import pathlib, signal, time
        def on_abort(*_):
            time.sleep(1.0)
            pathlib.Path({str(dumped)!r}).touch()
        signal.signal(signal.SIGABRT, on_abort)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(ready)!r}).touch()
        while True:
            time.sleep(60)
        """
    )
    script = tmp_path / "child.py"
    script.write_text(child)
    group_file = tmp_path / "group.pid"
    # The child's streams are not the wrapper's pipes, so a child that
    # outlives the stop fails the assertions below instead of holding the
    # pipes open. The leader records its own pid, the group's id, first.
    leader = (
        f"echo $$ > {shlex.quote(str(group_file))}; "
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} >/dev/null 2>&1; :"
    )
    try:
        result = _run("--dump", "5", "--grace", "1", "3", "--", "sh", "-c", leader)
        assert result.returncode == 124
        assert ready.exists(), "the child was not ready inside the 3 s bound"
        assert dumped.exists(), "SIGTERM was sent before the group's stacks were written"
        assert group_is_gone(int(group_file.read_text())), (
            "a process that survived SIGTERM was never killed"
        )
        assert "still running after SIGKILL" not in result.stderr
    finally:
        _kill_recorded_group(group_file)


def test_a_survivor_of_the_stop_is_named_and_the_run_still_exits_124(tmp_path):
    marker = tmp_path / "ready"
    child = f"import pathlib, time; pathlib.Path({str(marker)!r}).touch(); time.sleep(60)"
    # The real stop runs, then reports a process it could not end.
    wrapper = textwrap.dedent(
        f"""
        import runpy, sys
        from osprey.deployment import subprocess_capture
        real = subprocess_capture.stop_child
        def stop_child(process, **kwargs):
            real(process, **kwargs)
            return subprocess_capture.StopOutcome(ended=(process.pid,), survivors=(999999,))
        subprocess_capture.stop_child = stop_child
        sys.argv = [{str(SCRIPT)!r}, "--dump", "1", "--grace", "1", "3", "--",
                    {sys.executable!r}, "-c", {child!r}]
        runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", wrapper], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 124
    assert marker.exists(), "the child was not ready inside the 3 s bound"
    assert "run_bounded: still running after SIGKILL: pid 999999" in result.stderr


def test_a_stop_that_could_not_read_the_table_says_so_and_the_run_still_exits_124(tmp_path):
    marker = tmp_path / "ready"
    child = f"import pathlib, time; pathlib.Path({str(marker)!r}).touch(); time.sleep(60)"
    # The real stop runs, then reports that it could not read the process table.
    wrapper = textwrap.dedent(
        f"""
        import runpy, sys
        from osprey.deployment import subprocess_capture
        real = subprocess_capture.stop_child
        def stop_child(process, **kwargs):
            real(process, **kwargs)
            return subprocess_capture.StopOutcome(ended=(process.pid,), unknown=(process.pid,))
        subprocess_capture.stop_child = stop_child
        sys.argv = [{str(SCRIPT)!r}, "--dump", "1", "--grace", "1", "3", "--",
                    {sys.executable!r}, "-c", {child!r}]
        runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", wrapper], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 124
    assert marker.exists(), "the child was not ready inside the 3 s bound"
    assert "run_bounded: process table unreadable: group" in result.stderr
    assert "still running after SIGKILL" not in result.stderr


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
                prefix = [*check.kit.UV_RUN, *WRAPPER]
                bounded = (
                    argv[: len(prefix)] == prefix and "--" in argv and float(argv[len(prefix)]) > 0
                )
                if not bounded:
                    problems.append(f"{declared.id}: {tier} runs it without {WRAPPER[-1]}")
    return problems


def test_every_pytest_check_resolves_through_the_bound(check):
    assert [c.id for c in _pytest_checks(check.CHECKS)] == [
        "pytest",
        "pytest-pty",
        "pytest-wallclock",
        "readonly-warmups-py314",
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
