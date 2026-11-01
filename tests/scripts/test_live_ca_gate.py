"""Tests for the time limit the live Channel Access gate puts on each module.

None of these touches Channel Access. What the gate has to get right: a module
that never returns ends as one named error with its output so far, nothing it
started in its own session survives it, and a process that outlives a module
while holding its output open cannot keep the gate waiting.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests._by_path import load_module
from tests._live import wait_until
from tests._processes import pid_alive

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

#: The limit a hung child is killed at: room for an interpreter to start and
#: print on a loaded host, and still far inside :data:`_PROMPT_S`.
_KILL_LIMIT_S = 5

#: A limit long enough for a child pytest to start, run one test and print its
#: counts on a host busy with a parallel suite, and still far short of the hang.
_REPORT_S = 20


def _written_pid(pid_file: Path) -> int | None:
    """The pid in *pid_file* once its line is complete, else ``None``."""
    text = pid_file.read_text() if pid_file.exists() else ""
    return int(text) if text.endswith("\n") else None


def _read_pid(pid_file: Path) -> int:
    return wait_until(
        lambda: _written_pid(pid_file), timeout=_REPORT_S, what=f"a pid line in {pid_file}"
    )


def _kill(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _spawning_child(pid_file: Path, *, new_session: bool, then: str) -> list[str]:
    """A command that starts a sleeping grandchild, records its pid, then runs ``then``."""
    code = (
        "import subprocess, sys, time\n"
        f"g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep({_HANG_S})'],"
        f" start_new_session={new_session})\n"
        f"open({str(pid_file)!r}, 'w').write(str(g.pid) + '\\n')\n"
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
    stdout, _stderr, returncode, timed_out = gate._run_child(cmd, _KILL_LIMIT_S)
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
        stdout, _stderr, returncode, timed_out = gate._run_child(cmd, _PROMPT_S)
        elapsed = time.monotonic() - started

        orphan = _read_pid(pid_file)
        assert timed_out is False
        assert returncode == 0
        assert "child-done" in stdout
        assert elapsed < _PROMPT_S
        # The orphan really was still there, holding the inherited output open.
        assert pid_alive(orphan)
    finally:
        if pid_file.exists():
            _kill(_read_pid(pid_file))


def test_a_session_stop_takes_the_grandchild_in_the_module_session_with_it(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "grandchild.pid"
    cmd = _spawning_child(pid_file, new_session=False, then=f"time.sleep({_HANG_S})")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, start_new_session=True)

    try:
        # The child is ready once its grandchild's pid is on disk.
        grandchild = _read_pid(pid_file)
        gate._kill_session(proc)
        wait_until(
            lambda: not pid_alive(grandchild),
            timeout=_REPORT_S,
            what=f"the end of grandchild {grandchild} after its session was stopped",
        )
    finally:
        if proc.poll() is None:
            gate._kill_session(proc)
        if (grandchild_pid := _written_pid(pid_file)) is not None:
            _kill(grandchild_pid)


def test_a_child_past_its_limit_has_its_session_stopped_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[subprocess.Popen[bytes]] = []
    real_kill_session = gate._kill_session

    def _recording_kill_session(proc: subprocess.Popen[bytes]) -> None:
        stopped.append(proc)
        real_kill_session(proc)

    spawned: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def _recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(gate, "_kill_session", _recording_kill_session)
    monkeypatch.setattr(gate.subprocess, "Popen", _recording_popen)
    cmd = [sys.executable, "-c", f"import time; time.sleep({_HANG_S})"]

    _stdout, _stderr, _returncode, timed_out = gate._run_child(cmd, _KILL_LIMIT_S)

    assert timed_out is True
    assert stopped == spawned
    assert len(stopped) == 1


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


def test_a_module_that_reports_counts_and_then_hangs_is_still_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = tmp_path / "test_hangs_after_reporting.py"
    # The hang starts after pytest has finished and the counts are printed.
    module.write_text(
        "import atexit\nimport time\n\n"
        f"atexit.register(time.sleep, {_HANG_S})\n\n\n"
        "def test_passes():\n    pass\n",
        encoding="utf-8",
    )

    counts, status = gate._run_module(str(module), _REPORT_S)

    out = capsys.readouterr().out
    assert "1 passed" in out, out
    assert counts == {"passed": 0, "skipped": 0, "failed": 0, "error": 1}
    assert status == 1
    assert f"timed out after {_REPORT_S} s" in out


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
