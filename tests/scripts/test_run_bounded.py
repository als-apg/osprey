"""Tests for the wall-clock bound the check command puts on each pytest run.

The wrapper's ``main`` is only ever run as a subprocess, never called here: it
installs signal handlers and exits the process, neither of which belongs in the
test process. The group stop it calls, ``stop_child``, lives in the product's
``osprey.deployment.subprocess_capture``, whose own tests prove the stop's
signal ladder; the tests here prove only what the wrapper adds to it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import signal
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from osprey.deployment import subprocess_capture
from tests._by_path import load_module

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _REPO_ROOT / "scripts" / "run_bounded.py"
CHECK_SCRIPT = _REPO_ROOT / "scripts" / "check.py"
#: The wrapper after ``uv run --frozen``, the prefix every bounded command carries.
WRAPPER = ["python", "scripts/run_bounded.py"]

# scripts/ is not a package, so the wrapper is loaded by path, unregistered: its
# import installs no signal handler, and only its constants are read here.
rb = load_module("_run_bounded_by_path", SCRIPT)

#: The hang-ceiling multiple: a wait here is bounded by this many times the
#: waits it covers.
_CEILING_FACTOR = 3

#: Ceiling on reaping a child sent SIGKILL: the stop's own wait after SIGKILL.
_REAPED_S = _CEILING_FACTOR * subprocess_capture._KILL_WAIT_S


def _ceiling(
    bound: float, dump: float = rb.DEFAULT_DUMP_S, grace: float = rb.DEFAULT_GRACE_S
) -> float:
    """Ceiling on one wrapper run: its bound, the stop's ladder, and the last reads.

    After SIGKILL the stop waits up to ``_KILL_WAIT_S`` for the group, then up
    to as long again to read the process table for survivors.
    """
    return _CEILING_FACTOR * (bound + dump + grace + 2 * subprocess_capture._KILL_WAIT_S)


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


def _run(
    bound: float,
    *command: str,
    dump: float = rb.DEFAULT_DUMP_S,
    grace: float = rb.DEFAULT_GRACE_S,
) -> subprocess.CompletedProcess[str]:
    argv = ["--dump", f"{dump:g}", "--grace", f"{grace:g}", f"{bound:g}", "--", *command]
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        timeout=_ceiling(bound, dump, grace),
    )


def test_the_command_exit_code_passes_through():
    result = _run(30, sys.executable, "-c", "import sys; sys.exit(3)")
    assert result.returncode == 3


def test_a_command_past_its_bound_exits_124_and_says_so():
    bound = 1
    result = _run(bound, "sleep", "60", dump=1, grace=1)
    assert result.returncode == 124
    assert f"still running after {bound:g} s" in result.stderr


def _wrapped(stop_child_body: str, flags: list[str], command: list[str]) -> str:
    """A program that runs the wrapper as ``__main__`` with ``stop_child`` replaced.

    ``stop_child_body`` is the replacement's body; ``real`` names the product's
    stop and ``process`` and ``kwargs`` its arguments.
    """
    body = textwrap.indent(textwrap.dedent(stop_child_body), "    ")
    return (
        "import json, runpy, sys\n"
        "from osprey.deployment import subprocess_capture\n"
        "real = subprocess_capture.stop_child\n"
        "def stop_child(process, **kwargs):\n"
        f"{body}\n"
        "subprocess_capture.stop_child = stop_child\n"
        f"sys.argv = {[str(SCRIPT), *flags, '--', *command]!r}\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )


def test_dump_and_grace_reach_the_stop_and_default_to_the_named_constants():
    recording = """
    print("stop_child:" + json.dumps(kwargs), file=sys.stderr, flush=True)
    return real(process, **kwargs)
    """
    command = ["sleep", "60"]

    def kwargs_of(flags: list[str], dump: float, grace: float) -> dict[str, float]:
        result = subprocess.run(
            [sys.executable, "-c", _wrapped(recording, flags, command)],
            capture_output=True,
            text=True,
            timeout=_ceiling(1, dump, grace),
        )
        assert result.returncode == 124, result.stderr
        [line] = [ln for ln in result.stderr.splitlines() if ln.startswith("stop_child:")]
        return json.loads(line.removeprefix("stop_child:"))

    assert kwargs_of(["--dump", "1", "--grace", "2", "1"], 1, 2) == {"dump": 1.0, "grace": 2.0}
    assert kwargs_of(["1"], rb.DEFAULT_DUMP_S, rb.DEFAULT_GRACE_S) == {
        "dump": rb.DEFAULT_DUMP_S,
        "grace": rb.DEFAULT_GRACE_S,
    }


@pytest.mark.wallclock
def test_a_command_past_its_bound_prints_its_python_stacks_before_it_is_stopped(tmp_path):
    """The one test here whose bound is also the child's readiness window.

    faulthandler writes stacks only once the child's interpreter is up and
    blocked, which no bound can attest: the marker says the child was ready,
    and a loaded host that starts it after the bound fails the run instead.
    """
    marker = tmp_path / "ready"
    child = f"import pathlib, time; pathlib.Path({str(marker)!r}).touch(); time.sleep(60)"
    result = _run(3, sys.executable, "-c", child, dump=5, grace=1)
    assert result.returncode == 124
    assert marker.exists(), "the child was not ready inside the 3 s bound"
    assert "most recent call first" in result.stderr


def test_a_survivor_of_the_stop_is_named_and_the_run_still_exits_124():
    # The real stop runs, then reports a process it could not end.
    reporting = """
    real(process, **kwargs)
    return subprocess_capture.StopOutcome(ended=(process.pid,), survivors=(999999,))
    """
    program = _wrapped(reporting, ["--dump", "1", "--grace", "1", "3"], ["sleep", "60"])
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=_ceiling(3, 1, 1)
    )
    assert result.returncode == 124
    assert "run_bounded: still running after SIGKILL: pid 999999" in result.stderr


def test_a_stop_that_could_not_read_the_table_says_so_and_the_run_still_exits_124():
    # The real stop runs, then reports that it could not read the process table.
    reporting = """
    real(process, **kwargs)
    return subprocess_capture.StopOutcome(ended=(process.pid,), unknown=(process.pid,))
    """
    program = _wrapped(reporting, ["--dump", "1", "--grace", "1", "3"], ["sleep", "60"])
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=_ceiling(3, 1, 1)
    )
    assert result.returncode == 124
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
        out, _ = wrapper.communicate(timeout=_ceiling(30))
        assert "child-interrupted" in out
        assert wrapper.returncode == -signal.SIGINT
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(_REAPED_S)


def _pytest_checks(checks: tuple[Any, ...]) -> list[Any]:
    """The ``tests`` checks of *checks* whose command is a pytest run."""
    return [c for c in checks if c.kind == "tests" and "pytest" in c.argv]


def _unbounded_pytest_checks(check: ModuleType, checks: tuple[Any, ...]) -> list[str]:
    """What leaves a pytest check of *checks* unbounded, one line per defect.

    A check is bounded when it declares a bound and every tier that runs it
    resolves its command through ``scripts/run_bounded.py`` with a bound. The
    wrapper runs through ``uv run``, or ``uv run --no-sync`` for a check whose
    environment a sync would move.
    """
    problems = []
    for declared in _pytest_checks(checks):
        if declared.bound_s <= 0:
            problems.append(f"{declared.id}: declares no bound")
        for tier in sorted(declared.tiers):
            context = check.Context(tier=tier)
            for argv in check.commands(declared, check.Selection(check.WHOLE), context):
                prefix = [*check.kit.UV_RUN, *WRAPPER]
                # A check that keeps an installed environment as it is (the
                # CANOPY contract) adds ``--no-sync`` after the kit's ``uv run``.
                frozen = len(check.kit.UV_RUN)
                head = (
                    argv[:frozen] + argv[frozen + 1 :]
                    if argv[frozen : frozen + 1] == ["--no-sync"]
                    else argv
                )
                bounded = (
                    head[: len(prefix)] == prefix and "--" in head and float(head[len(prefix)]) > 0
                )
                if not bounded:
                    problems.append(f"{declared.id}: {tier} runs it without {WRAPPER[-1]}")
    return problems


def test_every_pytest_check_resolves_through_the_bound(check):
    assert [c.id for c in _pytest_checks(check.CHECKS)] == [
        "pytest",
        "pytest-pty",
        "pytest-wallclock",
        "pytest-wallclock-unit",
        "readonly-warmups-py314",
        "canopy-contract",
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
