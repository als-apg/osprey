"""Behavior of the subprocess capture helper.

Two modes are under test: the default, which spools a child's merged output to
``var/logs/`` and names that file on failure, and ``--verbose``, which must hand
the child this process's own stdio with no redirection at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.cli.phase_reporter import NullReporter, PhaseReporter, install_reporter
from osprey.deployment.errors import CapturedProcessError
from osprey.deployment.subprocess_capture import SPOOL_RETENTION, run_captured


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

    def test_prunes_to_the_newest_twenty(self, tmp_path, recorded_run):
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
        def boom(self):
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

    def test_verbose_mode_streams_raw_and_never_calls_back(
        self, tmp_path, recorded_run, verbose_reporter
    ):
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

    def test_same_argv_and_no_redirection_kwargs(self, tmp_path, recorded_run, verbose_reporter):
        argv = ["docker", "build", "."]

        run_captured(argv, env={"A": "1"}, spool_name="build", repo_root=tmp_path)

        (args, kwargs) = recorded_run[0]
        assert args[0] == argv
        assert "stdout" not in kwargs
        assert "stderr" not in kwargs
        assert kwargs["env"] == {"A": "1"}

    def test_writes_no_spool_file(self, tmp_path, recorded_run, verbose_reporter):
        run_captured(["docker", "build", "."], spool_name="build", repo_root=tmp_path)

        assert not (tmp_path / "var" / "logs").exists()

    def test_default_mode_uses_the_same_argv_plus_redirection(self, tmp_path, recorded_run):
        argv = ["docker", "build", "."]

        run_captured(argv, env={"A": "1"}, spool_name="build", repo_root=tmp_path)

        (args, kwargs) = recorded_run[0]
        assert args[0] == argv
        assert kwargs["stderr"] is subprocess.STDOUT
        assert Path(kwargs["stdout"].name) == _spools(tmp_path)[0]

    def test_failure_still_raises_captured_error_without_a_spool(
        self, tmp_path, monkeypatch, verbose_reporter
    ):
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

    def test_verbose_mode_result_carries_none(self, tmp_path, recorded_run, verbose_reporter):
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

    def test_verbose_mode_forwards_cwd(self, tmp_path, recorded_run, verbose_reporter):
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
