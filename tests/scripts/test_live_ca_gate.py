"""Tests for the time limit the live Channel Access gate puts on each module.

None of these touches Channel Access. What the gate has to get right: a module
that never returns ends as one named error with its output so far, nothing it
started in its own session survives it, and a process that outlives a module
while holding its output open cannot keep the gate waiting.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import IO, Any

import pytest

from osprey.deployment.subprocess_capture import StopOutcome, group_is_gone
from tests._by_path import load_module
from tests._live import wait_until
from tests._processes import kill_group, kill_quietly, pid_alive

# scripts/ is not a package, so the gate is loaded by path. It is deliberately not
# registered in sys.modules: nothing in it resolves its own module name, and an
# import-time write to sys.modules is process state no fixture can undo.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "va" / "live_ca" / "gate.py"
gate = load_module("live_ca_gate", _MODULE_PATH)

#: How long the stand-in processes would run if nothing stopped them. Far longer
#: than any limit used here, so "returned quickly" can only mean "was not waited for".
_HANG_S = 60

#: Upper bound on a call that must not wait for the hang.
_PROMPT_S = 20

#: Limit for a child that must reach its first print before it is cut off: long
#: enough for the interpreter to start and print on a loaded host, far below
#: ``_HANG_S``, so a timed-out child here can only have been cut off in the hang.
_LIMIT_S = 5.0

#: A limit long enough for a child pytest to start, run one test and print its
#: counts on a host busy with a parallel suite, and still far short of the hang.
_REPORT_S = 20


def _wait_for_pid(pid_file: Path, within_s: float = 10.0) -> int:
    def written() -> int | None:
        text = pid_file.read_text() if pid_file.exists() else ""
        return int(text) if text.endswith("\n") else None

    return wait_until(written, timeout=within_s, what=f"{pid_file} to be written")


def _group_file(pid_file: Path) -> Path:
    return pid_file.with_suffix(".pgid")


def _spawning_child(pid_file: Path, *, new_session: bool, then: str) -> list[str]:
    """A command that starts a sleeping grandchild, records its pid, then runs ``then``.

    The child's own process group is written next to ``pid_file`` before it.
    """
    code = (
        "import os, pathlib, subprocess, sys, time\n"
        f"g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep({_HANG_S})'],"
        f" start_new_session={new_session})\n"
        f"pathlib.Path({str(_group_file(pid_file))!r}).write_text(str(os.getpgid(0)) + '\\n')\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(g.pid) + '\\n')\n"
        "print('child-done', flush=True)\n"
        f"{then}\n"
    )
    return [sys.executable, "-c", code]


def test_a_child_that_outlives_its_limit_is_killed_and_its_output_so_far_is_returned() -> None:
    cmd = [
        sys.executable,
        "-c",
        f"print('before-the-hang', flush=True); import time; time.sleep({_HANG_S})",
    ]

    started = time.monotonic()
    stdout, _stderr, returncode, timed_out, _outcome = gate._run_child(cmd, _LIMIT_S)
    elapsed = time.monotonic() - started

    assert timed_out is True
    assert "before-the-hang" in stdout
    assert returncode != 0
    assert elapsed < _PROMPT_S


def test_an_orphan_holding_the_output_open_cannot_block_the_read(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    cmd = _spawning_child(pid_file, new_session=True, then="sys.exit(0)")

    try:
        started = time.monotonic()
        stdout, _stderr, returncode, timed_out, _outcome = gate._run_child(cmd, _PROMPT_S)
        elapsed = time.monotonic() - started

        orphan = _wait_for_pid(pid_file)
        assert timed_out is False
        assert returncode == 0
        assert "child-done" in stdout
        assert elapsed < _PROMPT_S
        # The orphan really was still there, holding the inherited output open.
        assert pid_alive(orphan)
    finally:
        if pid_file.exists():
            kill_quietly([_wait_for_pid(pid_file)])


def test_a_grandchild_in_the_module_session_dies_with_the_module_on_timeout(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "grandchild.pid"
    cmd = _spawning_child(pid_file, new_session=False, then=f"time.sleep({_HANG_S})")

    try:
        _stdout, _stderr, _returncode, timed_out, outcome = gate._run_child(cmd, 2)

        grandchild_group = _wait_for_pid(_group_file(pid_file))
        assert timed_out is True
        assert outcome is not None and outcome.gone
        assert group_is_gone(grandchild_group), "the grandchild survived its module's timeout"
    finally:
        if _group_file(pid_file).exists():
            kill_group(_wait_for_pid(_group_file(pid_file)))


def test_a_hung_module_is_reported_as_one_error_with_a_timed_out_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = tmp_path / "test_hangs.py"
    module.write_text(
        f"import time\n\n\ndef test_never_returns():\n    time.sleep({_HANG_S})\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    counts, status = gate._run_module(str(module), 2)
    elapsed = time.monotonic() - started

    assert counts == {"passed": 0, "skipped": 0, "failed": 0, "error": 1}
    assert status == 1
    assert elapsed < _PROMPT_S
    out = capsys.readouterr().out
    assert "timed out after 2 s" in out
    assert str(module) in out


def _until_counts_reported(stdout: IO[str]) -> None:
    """Return once the child's output file holds the counts line, without moving its offset."""
    fd = stdout.fileno()

    def reported() -> bool:
        text = os.pread(fd, os.fstat(fd).st_size, 0).decode(errors="replace")
        return any(line.startswith(gate.COUNTS_SENTINEL) for line in text.splitlines())

    wait_until(reported, timeout=_PROMPT_S, interval=0.02, what="the module's counts line")


class _TimedFromCounts:
    """A real child whose timed wait starts once it has reported its counts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._real = subprocess.Popen(*args, **kwargs)
        self._stdout = kwargs["stdout"]

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            _until_counts_reported(self._stdout)
        return self._real.wait(timeout=timeout)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def test_a_module_that_reports_counts_and_then_hangs_is_still_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "test_hangs_after_reporting.py"
    # The hang starts after pytest has finished and the counts are printed.
    module.write_text(
        "import atexit\nimport time\n\n"
        f"atexit.register(time.sleep, {_HANG_S})\n\n\n"
        "def test_passes():\n    pass\n",
        encoding="utf-8",
    )
    # The limit is timed from the counts line, so a slow start cannot spend it.
    monkeypatch.setattr(
        gate,
        "subprocess",
        types.SimpleNamespace(Popen=_TimedFromCounts, TimeoutExpired=subprocess.TimeoutExpired),
    )

    counts, status = gate._run_module(str(module), _REPORT_S)

    out = capsys.readouterr().out
    assert "1 passed" in out, out
    assert counts == {"passed": 0, "skipped": 0, "failed": 0, "error": 1}
    assert status == 1
    assert f"timed out after {_REPORT_S} s" in out


def test_a_timed_out_module_whose_table_could_not_be_read_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        gate,
        "_run_child",
        lambda _cmd, _limit: ("", "", -9, True, StopOutcome(ended=(1,), unknown=(1,))),
    )

    counts, status = gate._run_module("m", 1)

    assert counts == {"passed": 0, "skipped": 0, "failed": 0, "error": 1}
    assert status == 1
    assert "process table unreadable: group 1 may still be running" in capsys.readouterr().out


def test_a_module_timeout_flag_is_not_fanned_out_as_a_target_and_a_bad_value_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _must_not_run(*_args: object) -> None:
        raise AssertionError("a refused limit must not run any module")

    monkeypatch.setattr(gate, "_run_module", _must_not_run)
    monkeypatch.delenv("OSPREY_LIVE_CA_MODULE_TIMEOUT", raising=False)

    for bad in ("0", "-3", "soon", "nan", "inf"):
        assert gate.main([f"--module-timeout={bad}"]) == 1
        assert bad in capsys.readouterr().out

    monkeypatch.setenv("OSPREY_LIVE_CA_MODULE_TIMEOUT", "never")
    assert gate.main([]) == 1
    assert "never" in capsys.readouterr().out


def test_the_limit_comes_from_the_flag_then_the_environment_then_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, float]] = []

    def _record(target: str, limit_s: float) -> tuple[dict[str, int], int]:
        seen.append((target, limit_s))
        return ({"passed": 1, "skipped": 0, "failed": 0, "error": 0}, 0)

    monkeypatch.setattr(gate, "_run_module", _record)

    monkeypatch.delenv("OSPREY_LIVE_CA_MODULE_TIMEOUT", raising=False)
    assert gate.main(["tests/a.py"]) == 0
    monkeypatch.setenv("OSPREY_LIVE_CA_MODULE_TIMEOUT", "45")
    assert gate.main(["tests/a.py"]) == 0
    assert gate.main(["--module-timeout=7.5", "tests/a.py"]) == 0

    assert seen == [
        ("tests/a.py", gate.MODULE_TIMEOUT_S),
        ("tests/a.py", 45.0),
        ("tests/a.py", 7.5),
    ]
