"""Deterministic shutdown of the streaming lifecycle step's drain thread.

A streamed lifecycle step reads its child's stdout on a daemon thread. A child
that outlives its step's timeout used to keep that thread printing into
whatever the CLI drew next; these tests pin that it stops instead -- bounded,
silent, and without hanging the caller on the pipe it was reading.
"""

from __future__ import annotations

import shlex
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from osprey.cli import build_lifecycle
from osprey.cli.build_profile_schema import LifecycleStep
from osprey.cli.phase_reporter import NullReporter, PhaseReporter, install_reporter
from osprey.errors import BuildProfileError

# Every wait here is a deadline an assertion polls to, never a sleep sized to
# what the child "should" take.
_POLL_INTERVAL = 0.02
_DEADLINE = 5.0


class _RecordingReporter(PhaseReporter):
    """A reporter that keeps the lines it is handed instead of printing them.

    Subclasses :class:`PhaseReporter` because that is the seam the drain thread
    calls: a bare stand-in would satisfy the calls this test makes and miss the
    ones the code makes.

    The two output classes are kept apart. A child's streamed lines arrive on
    ``echo`` (the never-silenced path a verb's own output takes) and the phase
    record's own lines on ``emit``, so a test that asserts on one of them is
    never handed the other.
    """

    def __init__(self) -> None:
        super().__init__(color=False)
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._emitted: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:
        with self._lock:
            self._emitted.append(text)

    def echo(self, text: str) -> None:
        with self._lock:
            self._lines.append(text)

    def lines(self) -> list[str]:
        """The child's lines so far -- safe to read while the thread runs."""
        with self._lock:
            return list(self._lines)

    def emitted(self) -> list[str]:
        """The phase-record lines so far, in order."""
        with self._lock:
            return list(self._emitted)


@pytest.fixture
def reporter() -> Iterator[_RecordingReporter]:
    """Install a recording reporter for the duration of one test."""
    recording = _RecordingReporter()
    previous = install_reporter(recording)
    try:
        yield recording
    finally:
        install_reporter(previous)


def _script_cmd(path: Path, body: str) -> str:
    """Write a child script and return the command that runs it unbuffered."""
    path.write_text(body)
    return f"{shlex.quote(sys.executable)} -u {shlex.quote(str(path))}"


def _drain_threads() -> list[threading.Thread]:
    """Every drain thread currently alive, from this step or any other."""
    return [t for t in threading.enumerate() if t.name == build_lifecycle._DRAIN_THREAD_NAME]


def _wait_until(predicate: Callable[[], bool], deadline: float = _DEADLINE) -> bool:
    """Poll ``predicate`` until it holds or ``deadline`` seconds have passed."""
    stop_at = time.monotonic() + deadline
    while time.monotonic() < stop_at:
        if predicate():
            return True
        time.sleep(_POLL_INTERVAL)
    return predicate()


def test_normal_child_emits_every_line_in_order(
    tmp_path: Path, reporter: _RecordingReporter
) -> None:
    """A child that finishes inside its timeout still gets every line through."""
    run = _script_cmd(
        tmp_path / "well_behaved.py",
        "for i in range(1, 4):\n    print(f'line {i}')\n",
    )
    steps = [LifecycleStep(name="stream ok", run=run, timeout=30, stream=True)]

    build_lifecycle._run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)

    assert reporter.lines() == ["    line 1", "    line 2", "    line 3"]
    assert _drain_threads() == []


@pytest.mark.parametrize(
    ("make_reporter", "record_visible"),
    [
        pytest.param(lambda: PhaseReporter(color=False), True, id="printing-reporter"),
        pytest.param(lambda: NullReporter(verbose=True), False, id="verbose-reporter"),
    ],
)
def test_a_streamed_child_is_visible_under_every_reporter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_reporter: Callable[[], PhaseReporter],
    record_visible: bool,
) -> None:
    """``--stream`` shows the child's lines even where the record is silenced.

    ``osprey --verbose build --stream`` installs the reporter that swallows the
    phase record, and the child's transcript is the one thing that invocation
    asks hardest for. It rides the echo path for that reason: the record around
    it goes quiet, the transcript does not.
    """
    run = _script_cmd(tmp_path / "chatty.py", "print('from the child')\n")
    steps = [LifecycleStep(name="stream visible", run=run, timeout=30, stream=True)]

    previous = install_reporter(make_reporter())
    try:
        build_lifecycle._run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)
    finally:
        install_reporter(previous)

    printed = capsys.readouterr().out
    assert "    from the child" in printed
    assert ("Running post_build commands" in printed) is record_visible


def test_overrunning_child_is_cut_off_at_its_timeout(
    tmp_path: Path, reporter: _RecordingReporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child sleeping past its timeout ends the step; its later line is lost."""
    monkeypatch.setattr(build_lifecycle, "_EXIT_GRACE_SECONDS", 0.3)
    run = _script_cmd(
        tmp_path / "overrunning.py",
        "import time\nprint('early')\ntime.sleep(30)\nprint('late')\n",
    )
    steps = [LifecycleStep(name="stream overrun", run=run, timeout=1, stream=True)]

    started = time.monotonic()
    with pytest.raises(BuildProfileError, match="stream overrun"):
        build_lifecycle._run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)
    elapsed = time.monotonic() - started

    # Bounded by the step's own timeout plus the grace, not the child's sleep.
    assert elapsed < 5
    assert reporter.lines() == ["    early"]
    assert _drain_threads() == []


def test_drain_thread_is_silenced_when_a_grandchild_holds_the_pipe(
    tmp_path: Path, reporter: _RecordingReporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A line written after the step ends is dropped, not emitted.

    The child leaves a grandchild holding the inherited write end, so killing
    the child does not close the pipe and the drain thread stays blocked in its
    read. This is the case the stop flag exists for -- and the case where
    closing the pipe from the caller would hang it on the reader's lock.
    """
    monkeypatch.setattr(build_lifecycle, "_EXIT_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(build_lifecycle, "_DRAIN_SHUTDOWN_SECONDS", 0.3)
    marker = tmp_path / "grandchild-wrote.txt"
    grandchild = tmp_path / "grandchild.py"
    # Sleeps well past the step's own end (timeout + grace, ~1.3s here), so the
    # line reaches the pipe only once the step has been reported.
    grandchild.write_text(
        "import time\n"
        "time.sleep(3)\n"
        "print('late from grandchild', flush=True)\n"
        f"open({str(marker)!r}, 'w').write('done')\n"
    )
    run = _script_cmd(
        tmp_path / "leaves_grandchild.py",
        "import subprocess, sys, time\n"
        "print('early')\n"
        f"subprocess.Popen([sys.executable, '-u', {str(grandchild)!r}])\n"
        "time.sleep(30)\n",
    )
    steps = [LifecycleStep(name="stream leaky", run=run, timeout=1, stream=True)]

    with pytest.raises(BuildProfileError, match="stream leaky"):
        build_lifecycle._run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)

    during_step = reporter.lines()
    assert during_step == ["    early"]

    # Wait for proof the grandchild actually put its line in the pipe, then
    # assert the drain thread read it and passed nothing on.
    assert _wait_until(marker.exists, deadline=15.0), "grandchild never wrote its line"
    assert _wait_until(lambda: not _drain_threads()), "drain thread never ended"
    assert reporter.lines() == during_step


def test_stop_drain_returns_promptly_when_the_reader_cannot_be_unblocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shutdown helper never waits on a read it cannot interrupt.

    Closing a pipe under a blocked ``readline`` blocks the caller until that
    read returns on its own. This pins the guard that keeps the CLI out of it.
    """
    monkeypatch.setattr(build_lifecycle, "_DRAIN_SHUTDOWN_SECONDS", 0.2)
    release = threading.Event()
    closed: list[bool] = []

    class _StuckPipe:
        def readline(self) -> str:
            release.wait(timeout=_DEADLINE)
            return ""

        def close(self) -> None:
            closed.append(True)

    stop = threading.Event()
    pipe = _StuckPipe()
    reader = threading.Thread(
        target=build_lifecycle._drain_stdout,
        args=(pipe, stop),
        name=build_lifecycle._DRAIN_THREAD_NAME,
        daemon=True,
    )
    reader.start()

    started = time.monotonic()
    build_lifecycle._stop_drain(reader, stop, pipe)  # type: ignore[arg-type]
    elapsed = time.monotonic() - started

    assert elapsed < 2, "shutdown waited on a read it cannot interrupt"
    assert stop.is_set()
    assert closed == [], "closed a pipe the reader is still blocked on"

    release.set()
    reader.join(timeout=_DEADLINE)
    assert not reader.is_alive()
