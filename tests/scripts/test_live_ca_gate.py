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
from contextlib import suppress
from pathlib import Path
from typing import IO, Any

import pytest

from osprey.deployment.subprocess_capture import _KILL_WAIT_S, StopOutcome, group_is_gone
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

#: A limit timed from the counts line (``_TimedFromCounts``): the child's start
#: is not spent from it, and what follows the line is a hang, so any limit far
#: short of ``_HANG_S`` can only expire in that hang.
_AFTER_COUNTS_S = 2

#: A limit for a child that need not print anything before it is cut off.
_ANY_LIMIT_S = 1.0

#: The hang-ceiling multiple: a wait here is bounded by this many times the
#: waits it covers.
_CEILING_FACTOR = 3


def _wait_for_pid(pid_file: Path, within_s: float) -> int:
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


def test_a_child_past_its_limit_has_its_group_stopped_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timer is the one path that stops a timed-out child; nothing stops it twice."""
    stopped: list[subprocess.Popen[str]] = []
    real_stop_child = gate.stop_child

    def recording_stop_child(proc, **kwargs):
        stopped.append(proc)
        return real_stop_child(proc, **kwargs)

    spawned: list[subprocess.Popen[str]] = []

    def recording_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        proc = subprocess.Popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(gate, "stop_child", recording_stop_child)
    monkeypatch.setattr(
        gate,
        "subprocess",
        types.SimpleNamespace(Popen=recording_popen, TimeoutExpired=subprocess.TimeoutExpired),
    )
    cmd = [sys.executable, "-c", f"import time; time.sleep({_HANG_S})"]

    _stdout, _stderr, _returncode, timed_out, outcome = gate._run_child(cmd, _ANY_LIMIT_S)

    assert timed_out is True
    assert outcome is not None
    assert len(spawned) == 1
    assert stopped == spawned


def test_an_orphan_holding_the_output_open_cannot_block_the_read(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    cmd = _spawning_child(pid_file, new_session=True, then="sys.exit(0)")

    try:
        started = time.monotonic()
        stdout, _stderr, returncode, timed_out, _outcome = gate._run_child(cmd, _PROMPT_S)
        elapsed = time.monotonic() - started

        orphan = _wait_for_pid(pid_file, within_s=_REPORT_S)
        assert timed_out is False
        assert returncode == 0
        assert "child-done" in stdout
        assert elapsed < _PROMPT_S
        # The orphan really was still there, holding the inherited output open.
        assert pid_alive(orphan)
    finally:
        if pid_file.exists():
            kill_quietly([_wait_for_pid(pid_file, within_s=_REPORT_S)])


def test_a_session_stop_takes_the_grandchild_in_the_module_session_with_it(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "grandchild.pid"
    cmd = _spawning_child(pid_file, new_session=False, then=f"time.sleep({_HANG_S})")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, start_new_session=True)

    try:
        # The child is ready once its group, which its grandchild shares, is on disk.
        group = _wait_for_pid(_group_file(pid_file), within_s=_REPORT_S)
        outcome = gate.stop_child(proc, grace=gate._STOP_GRACE_S)

        assert outcome.gone
        wait_until(
            lambda: group_is_gone(group),
            timeout=_CEILING_FACTOR * (gate._STOP_GRACE_S + _KILL_WAIT_S),
            what=f"the end of group {group} after its session was stopped",
        )
    finally:
        if _group_file(pid_file).exists():
            kill_group(_wait_for_pid(_group_file(pid_file), within_s=_REPORT_S))
        if proc.returncode is None:
            kill_group(proc.pid)
            proc.wait(timeout=_CEILING_FACTOR * _KILL_WAIT_S)


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


def _counts_reported(stdout: IO[str]) -> bool:
    """Whether the child's output file holds the counts line, read without moving its offset."""
    fd = stdout.fileno()
    text = os.pread(fd, os.fstat(fd).st_size, 0).decode(errors="replace")
    return any(line.startswith(gate.COUNTS_SENTINEL) for line in text.splitlines())


def _until_counts_reported(proc: subprocess.Popen[str], stdout: IO[str]) -> None:
    """Return once the child has printed its counts line, or has ended without one.

    No wall-clock budget: the child's own run bounds the wait, however slowly it
    starts on a loaded host, since before its counts line it either prints that
    line or exits; the hang comes only after the line.
    """
    while not _counts_reported(stdout):
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=0.02)
            return


class _TimedFromCounts:
    """A real child whose timed wait starts once it has reported its counts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._real = subprocess.Popen(*args, **kwargs)
        self._stdout = kwargs["stdout"]

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            _until_counts_reported(self._real, self._stdout)
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

    counts, status = gate._run_module(str(module), _AFTER_COUNTS_S)

    out = capsys.readouterr().out
    assert "1 passed" in out, out
    assert counts == {"passed": 0, "skipped": 0, "failed": 0, "error": 1}
    assert status == 1
    assert f"timed out after {_AFTER_COUNTS_S} s" in out


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


def test_pva_mode_runs_the_seam_timed_passes_injection_and_turn_suites_in_that_order() -> None:
    assert gate.PVA_SUITES == (
        "tests/va/test_lume_pva_seam.py",
        "tests/va/test_timed_passes_live.py",
        "tests/va/test_injection_live.py",
        "tests/va/test_turn_by_turn_live.py",
    )
    root = _MODULE_PATH.parents[3]
    assert all((root / suite).is_file() for suite in (*gate.LIVE_SUITES, *gate.PVA_SUITES))
