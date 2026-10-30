"""Behavior of the subprocess capture helper.

Two modes are under test: the default, which spools a child's merged output to
``var/logs/`` and names that file on failure, and ``--verbose``, which must hand
the child this process's own stdio with no redirection at all.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from osprey.cli.phase_reporter import NullReporter, PhaseReporter, install_reporter
from osprey.deployment import process_table, subprocess_capture
from osprey.deployment.errors import CapturedProcessError, CapturedProcessTimeout
from osprey.deployment.process_table import ProcessTableUnreadable
from osprey.deployment.subprocess_capture import (
    SPOOL_RETENTION,
    StopOutcome,
    _mismatched_volumes,
    group_is_gone,
    observe_exit,
    run_captured,
    stop_child,
)
from tests._live import wait_until
from tests._processes import kill_group


@pytest.fixture(autouse=True)
def quiet_reporter():
    """Install a fresh quiet reporter and restore the previous one after."""
    previous = install_reporter(NullReporter(verbose=False))
    yield
    install_reporter(previous)


@pytest.fixture
def verbose_reporter():
    """Put the helper in pass-through mode for the duration of a test."""
    previous = install_reporter(NullReporter(verbose=True))
    yield
    install_reporter(previous)


@pytest.fixture
def recorded_run(monkeypatch):
    """Replace ``subprocess.run`` with a recorder, returning the call log."""
    calls: list[tuple[tuple, dict]] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr("osprey.deployment.subprocess_capture.subprocess.run", fake_run)
    return calls


def _echo(*lines: str) -> list[str]:
    """Argv for a child that writes ``lines`` to stdout, in order."""
    body = "\n".join(f"print({line!r}, flush=True)" for line in lines)
    return [sys.executable, "-c", body]


def _spools(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "var" / "logs").glob("*.log"))


class TestSpoolFile:
    """The default mode writes the child's output under ``var/logs/``."""

    def test_creates_spool_under_var_logs(self, tmp_path):
        run_captured(_echo("hello"), spool_name="build-worker", repo_root=tmp_path)

        spools = _spools(tmp_path)
        assert len(spools) == 1
        assert spools[0].name.startswith("build-worker-")
        assert spools[0].read_text() == "hello\n"

    def test_defaults_repo_root_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        run_captured(_echo("hi"), spool_name="compose-up")

        spools = _spools(tmp_path)
        assert len(spools) == 1
        assert spools[0].name.startswith("compose-up-")

    def test_merges_stderr_into_stdout_in_order(self, tmp_path):
        argv = [
            sys.executable,
            "-c",
            "import sys\n"
            "print('one', flush=True)\n"
            "print('two', file=sys.stderr, flush=True)\n"
            "print('three', flush=True)\n",
        ]

        run_captured(argv, spool_name="merged", repo_root=tmp_path)

        assert _spools(tmp_path)[0].read_text() == "one\ntwo\nthree\n"

    def test_second_run_in_the_same_second_gets_its_own_file(self, tmp_path):
        run_captured(_echo("first"), spool_name="twice", repo_root=tmp_path)
        run_captured(_echo("second"), spool_name="twice", repo_root=tmp_path)

        bodies = {p.read_text() for p in _spools(tmp_path)}
        assert bodies == {"first\n", "second\n"}

    def test_stdout_is_not_captured_into_the_result(self, tmp_path):
        completed = run_captured(_echo("hello"), spool_name="build", repo_root=tmp_path)

        assert completed.stdout is None
        assert completed.stderr is None


class TestRetention:
    """Old spools are pruned at entry, never failing the run."""

    @pytest.mark.usefixtures("recorded_run")
    def test_prunes_to_the_newest_twenty(self, tmp_path):
        logs = tmp_path / "var" / "logs"
        logs.mkdir(parents=True)
        for index in range(22):
            stale = logs / f"old-{index:02d}.log"
            stale.write_text(str(index))
            when = 1_700_000_000 + index
            os.utime(stale, (when, when))

        run_captured(["true"], spool_name="new", repo_root=tmp_path)

        names = sorted(p.name for p in _spools(tmp_path))
        assert len(names) == SPOOL_RETENTION + 1
        # The two oldest went; the newest kept ones and the new spool remain.
        assert "old-00.log" not in names
        assert "old-01.log" not in names
        assert "old-21.log" in names
        assert any(name.startswith("new-") for name in names)

    def test_unreadable_spool_does_not_fail_the_run(self, tmp_path, monkeypatch):
        def boom(_self):
            raise OSError("nope")

        monkeypatch.setattr(Path, "unlink", boom)
        logs = tmp_path / "var" / "logs"
        logs.mkdir(parents=True)
        for index in range(25):
            (logs / f"old-{index:02d}.log").write_text("x")

        completed = run_captured(_echo("ok"), spool_name="new", repo_root=tmp_path)

        assert completed.returncode == 0


class TestFailure:
    """A non-zero exit points the caller at the spool holding the output."""

    def test_raises_carrying_spool_path(self, tmp_path):
        argv = [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]

        with pytest.raises(CapturedProcessError) as excinfo:
            run_captured(argv, spool_name="failing", repo_root=tmp_path)

        error = excinfo.value
        assert error.returncode == 3
        assert error.cmd == argv
        assert error.spool_path == _spools(tmp_path)[0]
        assert error.spool_path.read_text() == "boom\n"
        assert str(error.spool_path) in str(error)

    def test_check_false_returns_completed_process(self, tmp_path):
        argv = [sys.executable, "-c", "raise SystemExit(4)"]

        completed = run_captured(argv, spool_name="tolerated", repo_root=tmp_path, check=False)

        assert isinstance(completed, subprocess.CompletedProcess)
        assert completed.returncode == 4


class TestReporterCoupling:
    """The open spool is handed to the reporter for interrupt reporting."""

    def test_notes_the_spool_on_the_open_phase(self, tmp_path):
        reporter = PhaseReporter(color=False)
        reporter.emit = lambda text, style=None: None  # type: ignore[method-assign]
        install_reporter(reporter)
        phase = reporter.phase("Building images")

        run_captured(_echo("x"), spool_name="noted", repo_root=tmp_path)

        assert phase.spool == _spools(tmp_path)[0]

    def test_spool_stays_set_after_the_child_exits(self, tmp_path):
        reporter = PhaseReporter(color=False)
        reporter.emit = lambda text, style=None: None  # type: ignore[method-assign]
        install_reporter(reporter)
        phase = reporter.phase("Building images")

        run_captured(_echo("x"), spool_name="noted", repo_root=tmp_path)
        run_captured(_echo("y"), spool_name="noted", repo_root=tmp_path)

        assert phase.spool is not None
        assert phase.spool.exists()


class TestOnLine:
    """``on_line`` tees the child's output to a callback, line by line.

    The seam that gives a long silent build per-step progress: the caller hands
    a parser, the child's merged stream still lands in the spool byte for byte,
    and each line also reaches the callback as decoded text. The callback is
    cosmetic by contract — nothing it does may change what the run itself
    produces or whether it fails.
    """

    def test_the_callback_sees_every_line_the_spool_records(self, tmp_path):
        seen: list[str] = []

        run_captured(_echo("one", "two"), spool_name="tee", repo_root=tmp_path, on_line=seen.append)

        assert seen == ["one", "two"]
        assert _spools(tmp_path)[0].read_text() == "one\ntwo\n"

    def test_stderr_reaches_the_callback_in_stream_order(self, tmp_path):
        argv = [
            sys.executable,
            "-c",
            "import sys\nprint('out', flush=True)\nprint('err', file=sys.stderr, flush=True)\n",
        ]
        seen: list[str] = []

        run_captured(argv, spool_name="tee", repo_root=tmp_path, on_line=seen.append)

        assert seen == ["out", "err"]

    def test_a_raising_callback_neither_fails_the_run_nor_loses_output(self, tmp_path):
        def boom(line: str) -> None:
            raise RuntimeError(line)

        completed = run_captured(
            _echo("one", "two"), spool_name="tee", repo_root=tmp_path, on_line=boom
        )

        assert completed.returncode == 0
        assert _spools(tmp_path)[0].read_text() == "one\ntwo\n"

    def test_the_failure_path_still_names_the_spool(self, tmp_path):
        argv = [sys.executable, "-c", "import sys; print('partial'); sys.exit(3)"]

        with pytest.raises(CapturedProcessError) as excinfo:
            run_captured(argv, spool_name="failing", repo_root=tmp_path, on_line=lambda _: None)

        assert excinfo.value.returncode == 3
        assert excinfo.value.spool_path == _spools(tmp_path)[0]
        assert excinfo.value.spool_path.read_text() == "partial\n"

    @pytest.mark.usefixtures("verbose_reporter")
    def test_verbose_mode_streams_raw_and_never_calls_back(self, tmp_path, recorded_run):
        """Verbose already puts the child's own lines on the terminal; step
        lines derived from the same stream would only duplicate them."""
        seen: list[str] = []

        run_captured(
            ["docker", "build", "."], spool_name="build", repo_root=tmp_path, on_line=seen.append
        )

        assert seen == []
        (args, kwargs) = recorded_run[0]
        assert args[0] == ["docker", "build", "."]
        assert "stdout" not in kwargs


class TestVerbosePassThrough:
    """SC7: verbose runs the identical argv with inherited stdio."""

    @pytest.mark.usefixtures("verbose_reporter")
    def test_same_argv_and_no_redirection_kwargs(self, tmp_path, recorded_run):
        argv = ["docker", "build", "."]

        run_captured(argv, env={"A": "1"}, spool_name="build", repo_root=tmp_path)

        (args, kwargs) = recorded_run[0]
        assert args[0] == argv
        assert "stdout" not in kwargs
        assert "stderr" not in kwargs
        assert "stdin" not in kwargs
        assert kwargs["env"] == {"A": "1"}

    @pytest.mark.usefixtures("recorded_run", "verbose_reporter")
    def test_writes_no_spool_file(self, tmp_path):
        run_captured(["docker", "build", "."], spool_name="build", repo_root=tmp_path)

        assert not (tmp_path / "var" / "logs").exists()

    def test_default_mode_uses_the_same_argv_plus_redirection(self, tmp_path, recorded_run):
        argv = ["docker", "build", "."]

        run_captured(argv, env={"A": "1"}, spool_name="build", repo_root=tmp_path)

        (args, kwargs) = recorded_run[0]
        assert args[0] == argv
        assert kwargs["stderr"] is subprocess.STDOUT
        assert Path(kwargs["stdout"].name) == _spools(tmp_path)[0]

    @pytest.mark.usefixtures("verbose_reporter")
    def test_failure_still_raises_captured_error_without_a_spool(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "osprey.deployment.subprocess_capture.subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 7),
        )

        with pytest.raises(CapturedProcessError) as excinfo:
            run_captured(["docker", "build", "."], spool_name="build", repo_root=tmp_path)

        # One exception type in both modes, so a verb's handler needs one except
        # clause. There is nothing to replay here — the output already streamed.
        assert excinfo.value.returncode == 7
        assert excinfo.value.spool_path is None
        assert not (tmp_path / "var" / "logs").exists()


class TestSpoolPathOnTheResult:
    """The result names its own spool file, in both modes.

    The reporter's copy only covers runs made while a phase is open, so an
    advisory caller (``check=False``, no phase) would otherwise have no way to
    point an operator at output that never reached the terminal.
    """

    def test_capture_mode_result_carries_the_spool_path(self, tmp_path):
        completed = run_captured(_echo("hello"), spool_name="build", repo_root=tmp_path)

        assert completed.spool_path == _spools(tmp_path)[0]
        assert completed.spool_path.read_text() == "hello\n"

    @pytest.mark.usefixtures("recorded_run", "verbose_reporter")
    def test_verbose_mode_result_carries_none(self, tmp_path):
        completed = run_captured(["docker", "build", "."], spool_name="build", repo_root=tmp_path)

        # Nothing was spooled — the output already went to the terminal.
        assert completed.spool_path is None


class TestWorkingDirectory:
    """``cwd`` is the child's directory, independent of the spool's location.

    A scaffolded script (``scripts/verify.sh``) is run from the project root so
    its own relative paths resolve the way they do when an operator runs it by
    hand, while its output still spools under the deployment repo. Both modes
    forward it: a run must not change directory just because ``--verbose`` was
    passed.
    """

    def test_default_mode_runs_the_child_in_cwd(self, tmp_path):
        elsewhere = tmp_path / "project"
        elsewhere.mkdir()
        repo = tmp_path / "repo"

        run_captured(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=elsewhere,
            spool_name="verify-script",
            repo_root=repo,
        )

        spool = _spools(repo)[0]
        assert Path(spool.read_text().strip()).resolve() == elsewhere.resolve()

    @pytest.mark.usefixtures("verbose_reporter")
    def test_verbose_mode_forwards_cwd(self, tmp_path, recorded_run):
        run_captured(
            ["bash", "scripts/verify.sh"],
            cwd=tmp_path,
            spool_name="verify-script",
            repo_root=tmp_path,
        )

        (_args, kwargs) = recorded_run[0]
        assert kwargs["cwd"] == tmp_path

    def test_omitting_cwd_leaves_the_child_in_this_process_directory(self, tmp_path, recorded_run):
        run_captured(["docker", "build", "."], spool_name="build", repo_root=tmp_path)

        (_args, kwargs) = recorded_run[0]
        assert kwargs["cwd"] is None


#: A child that reports whether its stdin is the null device and, if so, what a
#: read of it returns. It never reads a terminal, so a regression fails rather
#: than hanging the suite.
_STDIN_PROBE = [
    sys.executable,
    "-c",
    "import os, sys\n"
    "null = os.path.samestat(os.fstat(0), os.stat(os.devnull))\n"
    "print(null)\n"
    "print(repr(sys.stdin.read()) if null else 'not-read')\n",
]


@pytest.fixture
def typed_stdin():
    """Put a pipe holding an operator's answer on this process's fd 0.

    pytest already parks fd 0 on the null device, so without this a child that
    merely inherited stdin would look exactly like one given the null device.
    """
    saved = os.dup(0)
    read_end, write_end = os.pipe()
    os.write(write_end, b"y\n")
    os.close(write_end)
    os.dup2(read_end, 0)
    os.close(read_end)
    try:
        yield
    finally:
        os.dup2(saved, 0)
        os.close(saved)


@pytest.mark.usefixtures("typed_stdin")
class TestNoTerminalForCapturedChildren:
    """A captured child owns no terminal: a prompt reads EOF instead of blocking."""

    def test_captured_child_reads_eof_on_stdin(self, tmp_path):
        run_captured(_STDIN_PROBE, spool_name="probe", repo_root=tmp_path)

        assert _spools(tmp_path)[0].read_text() == "True\n''\n"

    def test_watched_child_reads_eof_on_stdin(self, tmp_path):
        seen: list[str] = []

        run_captured(_STDIN_PROBE, spool_name="probe", repo_root=tmp_path, on_line=seen.append)

        assert seen == ["True", "''"]


#: The line compose prints when a named volume's labels differ from the compose
#: file; with stdin at EOF it answers the prompt No and carries on.
_MISMATCH = (
    'Volume "{}" exists but doesn\'t match configuration in compose file. '
    "Recreate (data will be lost)?"
)


def _exits(code: int, *lines: str) -> list[str]:
    """Argv for a child that prints ``lines`` and exits with ``code``."""
    body = "".join(f"print({line!r})\n" for line in lines)
    return [sys.executable, "-c", f"{body}raise SystemExit({code})"]


class TestKeptVolumeWarning:
    """Compose keeping a mismatched volume is a silent success the run still reports."""

    def test_helper_names_each_volume_once_in_order(self):
        text = "\n".join(
            [
                _MISMATCH.format("demo_pgdata"),
                "Container demo-postgres-1  Started",
                _MISMATCH.format("demo_qmd-index"),
                _MISMATCH.format("demo_pgdata"),
            ]
        )

        assert _mismatched_volumes(text) == ["demo_pgdata", "demo_qmd-index"]

    def test_helper_finds_nothing_in_ordinary_output(self):
        assert _mismatched_volumes('Volume "demo_pgdata"  Created\n') == []

    def test_successful_run_warns_naming_volume_and_spool(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            result = run_captured(
                _exits(0, _MISMATCH.format("demo_pgdata"), "Container demo-pg-1  Started"),
                spool_name="compose-up",
                repo_root=tmp_path,
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "demo_pgdata" in message
        assert str(result.spool_path) in message
        assert "osprey status" in message

    def test_watched_run_warns_too(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            run_captured(
                _exits(0, _MISMATCH.format("demo_pgdata"), _MISMATCH.format("demo_mongo")),
                spool_name="compose-up",
                repo_root=tmp_path,
                on_line=lambda _line: None,
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "demo_pgdata" in warnings[0].getMessage()
        assert "demo_mongo" in warnings[0].getMessage()

    def test_failed_run_warns_before_raising(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING), pytest.raises(CapturedProcessError):
            run_captured(
                _exits(1, _MISMATCH.format("demo_pgdata")),
                spool_name="compose-up",
                repo_root=tmp_path,
            )

        assert "demo_pgdata" in caplog.text

    def test_no_marker_no_warning(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            run_captured(
                _exits(0, "Container demo-pg-1  Started"), spool_name="up", repo_root=tmp_path
            )

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# Bounded runs: a time limit and a watch.
#
# A `compose up` can wait forever (podman-compose retries a dependency's
# health once a second with no limit). A bounded run stops its child — and
# everything the child started — past its limit, and calls a watch while it
# runs so the operator sees what it is waiting on. These are real processes:
# the property under test is that a grandchild dies with its parent.
# ---------------------------------------------------------------------------


def _group_file(pid_file: Path) -> Path:
    return pid_file.with_suffix(".pgid")


def _spawner(pid_file: Path) -> list[str]:
    """Argv for a child that starts a long-lived grandchild and records its pid and group.

    The grandchild is what a `podman compose` dispatcher leaves behind: the
    provider process it started. It stays in the child's process group, whose
    id is written next to ``pid_file`` before ``started`` is printed.
    """
    body = (
        "import os, subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(_group_file(pid_file))!r}, 'w').write(str(os.getpgid(0)))\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "print('started', flush=True)\n"
        "time.sleep(60)\n"
    )
    return [sys.executable, "-c", body]


def _kill_recorded_group(pid_file: Path) -> None:
    group_file = _group_file(pid_file)
    if group_file.exists() and group_file.read_text():
        kill_group(int(group_file.read_text()))


def _wait_for_file(path: Path, timeout: float = 10.0, *, empty_ok: bool = False) -> None:
    wait_until(
        lambda: path.exists() and (empty_ok or path.read_text()),
        timeout=timeout,
        interval=0.02,
        what=f"a write to {path}",
    )


@pytest.mark.real_bounded_run
class TestBoundedRun:
    def test_a_run_inside_its_limit_returns_like_any_other(self, tmp_path):
        result = run_captured(_echo("done"), spool_name="bounded", repo_root=tmp_path, timeout=30)
        assert result.returncode == 0
        assert "done" in result.spool_path.read_text()

    def test_a_non_zero_exit_still_raises_under_check(self, tmp_path):
        with pytest.raises(CapturedProcessError) as excinfo:
            run_captured(
                [sys.executable, "-c", "raise SystemExit(3)"],
                spool_name="bounded",
                repo_root=tmp_path,
                timeout=30,
            )
        assert excinfo.value.returncode == 3

    def test_past_its_limit_the_child_and_its_grandchild_are_stopped(self, tmp_path):
        pid_file = tmp_path / "grandchild.pid"

        try:
            with pytest.raises(CapturedProcessTimeout) as excinfo:
                run_captured(
                    _spawner(pid_file), spool_name="bounded", repo_root=tmp_path, timeout=1.5
                )

            assert excinfo.value.timeout == 1.5
            assert excinfo.value.returncode is None
            assert "started" in excinfo.value.spool_path.read_text()
            group = int(_group_file(pid_file).read_text())
            assert group_is_gone(group), "the stopped run's grandchild outlived it"
        finally:
            _kill_recorded_group(pid_file)

    def test_an_interrupt_stops_the_group_and_propagates(self, tmp_path):
        pid_file = tmp_path / "grandchild.pid"

        def watch(_elapsed, _spool):
            _wait_for_file(pid_file)
            raise KeyboardInterrupt

        try:
            with pytest.raises(KeyboardInterrupt):
                run_captured(
                    _spawner(pid_file),
                    spool_name="bounded",
                    repo_root=tmp_path,
                    watch=watch,
                    watch_interval=0.2,
                )
            group = int(_group_file(pid_file).read_text())
            assert group_is_gone(group), "the interrupted run's grandchild outlived it"
        finally:
            _kill_recorded_group(pid_file)

    def test_the_watch_sees_the_time_run_and_the_spool(self, tmp_path):
        seen: list[tuple[float, Path | None]] = []
        run_captured(
            [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(0.7)"],
            spool_name="bounded",
            repo_root=tmp_path,
            watch=lambda elapsed, spool: seen.append((elapsed, spool)),
            watch_interval=0.2,
        )
        assert seen, "the watch never ran while the child did"
        elapsed = [e for e, _ in seen]
        assert elapsed == sorted(elapsed)
        assert all(spool is not None and spool.name.startswith("bounded-") for _, spool in seen)

    def test_a_watch_that_raises_changes_nothing(self, tmp_path):
        def watch(_elapsed, _spool):
            raise RuntimeError("cosmetic")

        result = run_captured(
            [sys.executable, "-c", "import time; time.sleep(0.5)"],
            spool_name="bounded",
            repo_root=tmp_path,
            watch=watch,
            watch_interval=0.1,
        )
        assert result.returncode == 0

    def test_a_line_watcher_cannot_be_combined_with_a_bound(self, tmp_path):
        with pytest.raises(TypeError):
            run_captured(
                _echo("x"),
                spool_name="bounded",
                repo_root=tmp_path,
                timeout=5,
                on_line=lambda line: None,
            )

    @pytest.mark.usefixtures("verbose_reporter")
    def test_verbose_mode_is_bounded_too(self, tmp_path):
        with pytest.raises(CapturedProcessTimeout) as excinfo:
            run_captured(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                spool_name="bounded",
                repo_root=tmp_path,
                timeout=0.5,
            )
        assert excinfo.value.spool_path is None


# ---------------------------------------------------------------------------
# Stopping a process group and proving it gone. Real processes: the property
# under test is what the kernel and the process table report.
class TestStopOutcome:
    def test_a_gone_outcome_is_truthy_and_gone(self):
        outcome: StopOutcome[int] = StopOutcome()
        assert bool(outcome) is True
        assert outcome.gone
        assert outcome.describe() == "gone"

    def test_survivors_or_unknown_are_not_gone(self):
        assert not StopOutcome(ended=(1,), survivors=(2,)).gone
        assert not StopOutcome(ended=(1,), unknown=(1,)).gone

    def test_describe_names_survivors_and_unknown_apart(self):
        outcome = StopOutcome(ended=(123,), survivors=(4242,), unknown=(123,))
        assert outcome.describe() == (
            "still running after SIGKILL: pid 4242; "
            "process table unreadable: group 123 may still be running"
        )

    def test_describe_takes_one_renderer_per_set(self):
        outcome = StopOutcome(ended=("a",), survivors=("b", "c"), unknown=("d",))
        assert outcome.describe(survivor=str.upper, unknown=lambda x: f"<{x}>") == (
            "still running after SIGKILL: B, C; process table unreadable: <d> may still be running"
        )


class TestObserveExit:
    def test_observe_exit_sees_an_exit_without_reaping_it(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        try:
            assert observe_exit(proc, within=5.0) == 0
            assert observe_exit(proc) == 0
            assert proc.returncode is None
        finally:
            assert proc.wait() == 0

    def test_observe_exit_answers_none_while_the_child_runs(self):
        proc = subprocess.Popen(["sleep", "60"])
        try:
            assert observe_exit(proc) is None
        finally:
            proc.kill()
            proc.wait(timeout=30)

    def test_observe_exit_reports_a_signal_as_popen_does(self):
        proc = subprocess.Popen(["sleep", "60"])
        try:
            proc.kill()
            assert observe_exit(proc, within=5.0) == -signal.SIGKILL
        finally:
            proc.wait(timeout=30)

    def test_observe_exit_blocks_until_the_exit_when_asked(self):
        proc = subprocess.Popen(["sleep", "0.3"])
        try:
            start = time.monotonic()
            assert observe_exit(proc, within=None) == 0
            assert time.monotonic() - start >= 0.25
        finally:
            proc.wait(timeout=30)

    def test_observe_exit_returns_the_status_a_reaped_child_already_has(self):
        proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
        proc.wait(timeout=30)
        assert observe_exit(proc) == 3

    def test_observe_exit_raises_when_something_else_reaped_the_child(self, monkeypatch):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        os.waitpid(proc.pid, 0)
        monkeypatch.setattr(subprocess_capture, "_REAP_SETTLE_S", 0.2)
        with pytest.raises(RuntimeError, match=f"pid {proc.pid} was reaped by something other"):
            observe_exit(proc)
        proc.returncode = 0  # the foreign reap took the status; keep Popen.__del__ quiet


# The kernel facts the probe rests on, read through each way this interpreter
# can call waitid(2): CPython's binding where it has one, and the C library
# directly on macOS, where CPython 3.11 and 3.12 bind none.
_READERS = [subprocess_capture._waitid] + (
    [subprocess_capture._libc_waitid] if sys.platform == "darwin" else []
)
_LOOK = os.WEXITED | os.WNOWAIT


@pytest.mark.parametrize("reader", _READERS, ids=lambda r: r.__name__)
class TestWaitidReader:
    def test_a_running_child_reads_none(self, reader):
        proc = subprocess.Popen(["sleep", "60"])
        try:
            assert reader(proc.pid, _LOOK | os.WNOHANG) is None
        finally:
            proc.kill()
            proc.wait(timeout=30)

    def test_an_exited_child_reads_its_status_twice_and_stays_unreaped(self, reader):
        proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
        first = reader(proc.pid, _LOOK)
        second = reader(proc.pid, _LOOK)
        assert first == second
        assert (first.si_pid, first.si_code, first.si_status) == (proc.pid, os.CLD_EXITED, 7)
        assert proc.wait() == 7

    def test_a_sigtermed_child_reads_killed_with_its_signal(self, reader):
        proc = subprocess.Popen(["sleep", "60"])
        proc.terminate()
        state = reader(proc.pid, _LOOK)
        assert (state.si_code, state.si_status) == (os.CLD_KILLED, signal.SIGTERM)
        assert proc.wait() == -signal.SIGTERM

    def test_a_reaped_child_is_not_a_child(self, reader):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=30)
        with pytest.raises(ChildProcessError):
            reader(proc.pid, _LOOK | os.WNOHANG)


# ---------------------------------------------------------------------------


def _exited_unreaped_leader() -> subprocess.Popen:
    """A group leader that has exited and that nobody has reaped yet."""
    leader = subprocess.Popen([sys.executable, "-c", "pass"], process_group=0)
    observe_exit(leader, within=None)
    return leader


def _sleeping_group(ready: Path, *, ignore_term: bool = False) -> subprocess.Popen:
    """A group of one sleep, returned once ``ready`` says its signal setup is done."""
    trap = "trap '' TERM; " if ignore_term else ""
    group = subprocess.Popen(
        ["sh", "-c", f"{trap}: > {shlex.quote(str(ready))}; exec sleep 60"], process_group=0
    )
    _wait_for_file(ready, empty_ok=True)
    return group


@pytest.mark.real_bounded_run
class TestProcessGroupStop:
    def test_an_exited_unreaped_leader_counts_as_gone(self):
        leader = _exited_unreaped_leader()
        try:
            os.kill(leader.pid, 0)  # still in the process table, as a zombie
            assert group_is_gone(leader.pid)
        finally:
            leader.wait(timeout=30)

    def test_a_group_with_a_running_member_is_not_gone(self, tmp_path):
        group = _sleeping_group(tmp_path / "ready")
        try:
            assert not group_is_gone(group.pid)
        finally:
            kill_group(group.pid)
            group.wait(timeout=30)

    def test_a_group_this_user_may_not_signal_counts_as_present(self, tmp_path, monkeypatch):
        group = _sleeping_group(tmp_path / "ready")
        real_killpg = os.killpg

        def refused(pgid, sig):
            if sig == 0:
                raise PermissionError(1, "Operation not permitted")
            real_killpg(pgid, sig)

        monkeypatch.setattr(subprocess_capture.os, "killpg", refused)
        try:
            assert not group_is_gone(group.pid)
        finally:
            monkeypatch.undo()
            kill_group(group.pid)
            group.wait(timeout=30)

    def test_group_is_gone_without_a_deadline_answers_at_once(self, tmp_path, monkeypatch):
        group = _sleeping_group(tmp_path / "ready")
        real_snapshot = process_table.snapshot
        reads: list[None] = []

        def counted():
            reads.append(None)
            return real_snapshot()

        monkeypatch.setattr(process_table, "snapshot", counted)
        no_sleep = types.SimpleNamespace(monotonic=time.monotonic, sleep=pytest.fail)
        monkeypatch.setattr(subprocess_capture, "time", no_sleep)
        try:
            assert not group_is_gone(group.pid)
            assert len(reads) == 1
        finally:
            monkeypatch.undo()
            kill_group(group.pid)
            group.wait(timeout=30)

    def test_group_is_gone_waits_for_a_group_that_ends_inside_its_deadline(self):
        group = subprocess.Popen(["sh", "-c", "sleep 0.5"], process_group=0)
        try:
            assert group_is_gone(group.pid, within=5.0)
        finally:
            kill_group(group.pid)
            group.wait(timeout=30)

    def test_group_is_gone_reads_a_retryable_failure_again_inside_its_deadline(self, monkeypatch):
        leader = _exited_unreaped_leader()
        real_snapshot = process_table.snapshot
        failures = [ProcessTableUnreadable(OSError("busy"))]

        def flaky():
            if failures:
                raise failures.pop()
            return real_snapshot()

        monkeypatch.setattr(process_table, "snapshot", flaky)
        try:
            assert group_is_gone(leader.pid, within=2.0)
        finally:
            leader.wait(timeout=30)

    def test_group_is_gone_says_it_could_not_look_when_the_table_is_unreadable(
        self, tmp_path, monkeypatch
    ):
        group = _sleeping_group(tmp_path / "ready")

        def refused():
            raise ProcessTableUnreadable(PermissionError("refused"))

        monkeypatch.setattr(process_table, "snapshot", refused)
        try:
            with pytest.raises(ProcessTableUnreadable):
                group_is_gone(group.pid)
        finally:
            monkeypatch.undo()
            kill_group(group.pid)
            group.wait(timeout=30)

    def test_an_unreadable_table_still_waits_out_the_grace_before_sigkill(self, monkeypatch):
        leader = _exited_unreaped_leader()
        sent: list[tuple[int, float]] = []

        def refused():
            raise ProcessTableUnreadable(PermissionError("refused"))

        monkeypatch.setattr(
            subprocess_capture,
            "_signal_group",
            lambda _pgid, sig: sent.append((sig, time.monotonic())),
        )
        monkeypatch.setattr(process_table, "snapshot", refused)
        monkeypatch.setattr(subprocess_capture, "_KILL_WAIT_S", 0.2)
        try:
            subprocess_capture._stop_group(leader.pid, grace=0.5)
        finally:
            leader.wait(timeout=30)
        times = dict(sent)
        assert times[signal.SIGKILL] - times[signal.SIGTERM] >= 0.45

    def test_a_retryable_table_failure_is_read_again_not_reported(self, monkeypatch):
        leader = _exited_unreaped_leader()
        real_snapshot = process_table.snapshot
        failures = [ProcessTableUnreadable(OSError("busy"))]

        def flaky():
            if failures:
                raise failures.pop()
            return real_snapshot()

        monkeypatch.setattr(process_table, "snapshot", flaky)
        try:
            assert subprocess_capture._stop_group(leader.pid, grace=1.0) == []
        finally:
            leader.wait(timeout=30)

    def test_stop_child_says_unknown_when_the_table_is_unreadable(self, monkeypatch):
        proc = _exited_unreaped_leader()

        def refused():
            raise ProcessTableUnreadable(PermissionError("refused"))

        monkeypatch.setattr(process_table, "snapshot", refused)
        monkeypatch.setattr(subprocess_capture, "_KILL_WAIT_S", 0.2)
        outcome = stop_child(proc, grace=0.2)
        assert outcome.unknown == (proc.pid,)
        assert outcome.survivors == ()
        assert not outcome.gone
        assert proc.returncode == 0

    def test_stop_child_names_a_leader_it_saw_alive_when_the_table_is_unreadable(
        self, tmp_path, monkeypatch
    ):
        proc = _sleeping_group(tmp_path / "ready")

        def refused():
            raise ProcessTableUnreadable(PermissionError("refused"))

        monkeypatch.setattr(subprocess_capture, "_signal_group", lambda _pgid, _sig: None)
        monkeypatch.setattr(process_table, "snapshot", refused)
        monkeypatch.setattr(subprocess_capture, "_KILL_WAIT_S", 0.2)
        try:
            outcome = stop_child(proc, grace=0.2)
            assert outcome.survivors == (proc.pid,)
            assert outcome.unknown == (proc.pid,)
            assert outcome.describe() == (
                f"still running after SIGKILL: pid {proc.pid}; "
                f"process table unreadable: group {proc.pid} may still be running"
            )
            assert proc.returncode is None
        finally:
            monkeypatch.undo()
            kill_group(proc.pid)
            proc.wait(timeout=30)

    def test_stop_child_refuses_a_reaped_leader(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"], process_group=0)
        proc.wait(timeout=30)
        with pytest.raises(ValueError, match="already reaped"):
            stop_child(proc, grace=0.2)

    def test_stop_child_refuses_a_child_that_does_not_lead_its_group(self):
        proc = subprocess.Popen(["sleep", "60"])
        try:
            with pytest.raises(ValueError, match="does not lead its own process group"):
                stop_child(proc, grace=0.2)
        finally:
            proc.kill()
            proc.wait(timeout=30)

    def test_stop_child_reaps_the_leader_it_stopped(self, tmp_path):
        proc = _sleeping_group(tmp_path / "ready", ignore_term=True)
        try:
            outcome = stop_child(proc, grace=0.5)
            assert outcome.gone
            assert outcome.ended == (proc.pid,)
            assert proc.returncode == -signal.SIGKILL
            with pytest.raises(ChildProcessError):
                subprocess_capture._waitid(proc.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        finally:
            kill_group(proc.pid)

    def test_stop_child_names_a_member_that_outlives_sigkill_and_leaves_the_leader_unwaited(
        self, tmp_path, monkeypatch
    ):
        proc = _sleeping_group(tmp_path / "ready", ignore_term=True)
        real_signal_group = subprocess_capture._signal_group

        def no_sigkill(pgid, sig):
            if sig != signal.SIGKILL:
                real_signal_group(pgid, sig)

        monkeypatch.setattr(subprocess_capture, "_signal_group", no_sigkill)
        monkeypatch.setattr(subprocess_capture, "_KILL_WAIT_S", 0.2)
        try:
            outcome = stop_child(proc, grace=0.2)
            assert outcome.survivors == (proc.pid,)
            assert outcome.unknown == ()
            assert proc.returncode is None
        finally:
            monkeypatch.undo()
            kill_group(proc.pid)
            proc.wait(timeout=30)

    def test_stop_child_stops_a_child_that_already_exited(self):
        proc = _exited_unreaped_leader()
        outcome = stop_child(proc, grace=0.5)
        assert outcome.gone
        assert proc.returncode == 0

    def test_a_stop_that_could_not_read_the_table_says_so(self, monkeypatch, caplog):
        proc = subprocess.Popen([sys.executable, "-c", "pass"], process_group=0)
        monkeypatch.setattr(
            subprocess_capture,
            "stop_child",
            lambda process, **_k: StopOutcome(ended=(process.pid,), unknown=(process.pid,)),
        )
        try:
            with caplog.at_level(logging.WARNING):
                subprocess_capture._stop(proc, own_group=True)
        finally:
            proc.wait(timeout=30)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(f"process table unreadable: group {proc.pid}" in w for w in warnings)


def test_a_stopped_runs_detail_is_its_failure_detail():
    """The caller's account of a stall is what the failure report prints."""
    from osprey.deployment.subprocess_capture import captured_failure_detail

    exc = CapturedProcessTimeout(["docker", "compose", "up"], 60.0, None)
    exc.detail = "Waiting on: x (starting)."
    assert captured_failure_detail(exc) == "Waiting on: x (starting)."
