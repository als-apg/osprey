"""Tests for the phase reporter's build watchers and its heartbeat pass.

Every timestamp here is injected. Nothing in this file sleeps or reads a real
clock: the heartbeat decision takes ``now`` as an argument precisely so a
quarter-hour build can be replayed from a fixture in milliseconds. The monitor
thread's tests are the one place a real clock is unavoidable -- and even there
the wait is on an event with a timeout, never on a duration.
"""

import io
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from osprey.cli import phase_reporter
from osprey.cli.phase_reporter import (
    PhaseReporter,
    current_reporter,
    install_reporter,
)
from osprey.cli.styles import osprey_theme
from osprey.deployment.build_progress import BuildModel
from tests.deployment.test_build_progress import BOOKKEEPING, PINNED_CAPTIONS

FIXTURES = Path(__file__).resolve().parents[1] / "deployment" / "fixtures"

COLD_BUILD = "buildkit_cold_build.log"
PROJECT_IMAGE = "buildkit_project_image.log"


def fixture_lines(name: str) -> list[str]:
    """Read a captured spool the way ``run_captured`` delivers it to ``on_line``.

    Bytes split on newlines only: 188 of the cold build's lines carry embedded
    carriage returns, and Python's universal-newline text mode would shred each
    of those into several lines the child never emitted.
    """
    return (FIXTURES / name).read_bytes().decode().split("\n")


def vertex_lines(name: str, node: int) -> list[str]:
    """Every line one BuildKit vertex emitted, in order."""
    prefix = f"#{node} "
    return [line for line in fixture_lines(name) if line.startswith(prefix)]


#: One heartbeat line, split back into its parts. Every service in the cold
#: build carries a step by the time it first beats -- its opening line is a
#: header -- so the step is matched rather than allowed for, which keeps a
#: mis-split from being read as a caption.
_BEAT = re.compile(
    r"  · (?P<service>\S+) (?P<step>\d+/\d+|internal|auth|exporting) (?P<caption>.*)"
)


def beat_caption(beat: str) -> str:
    """The caption a heartbeat line put on screen."""
    parsed = _BEAT.fullmatch(beat.rsplit(" (running ", 1)[0])
    assert parsed is not None, beat
    return parsed["caption"]


def replay_heartbeats(name: str) -> list[str]:
    """Beat every service after every line of a fixture, and collect the lines.

    A beat is due 30s after the last one, so the replay clock advances 31s per
    line: every state a row passes through is put on screen exactly once, which
    is what makes "no heartbeat ever showed X" a claim about the whole build
    rather than about the moments a real clock happened to land on.
    """
    reporter = PhaseReporter(color=False)
    model = BuildModel()
    beats: list[str] = []
    with reporter.watch_build(model):
        for tick, line in enumerate(fixture_lines(name)):
            now = 1000.0 + 31.0 * tick
            model.feed(line, now)
            beats.extend(reporter._heartbeat_pass(now + 1.0))
    return beats


@pytest.fixture
def sink(monkeypatch):
    """Capture reporter output, as in ``test_phase_reporter``."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        phase_reporter,
        "console",
        Console(file=buffer, theme=osprey_theme, force_terminal=True, width=200),
    )
    return buffer


def monitor_threads() -> list[threading.Thread]:
    """Every live reporter monitor thread in the process.

    Named threads are what makes a leak visible: a monitor that outlived its
    reporter is still in this list long after the test that started it.
    """
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == phase_reporter._MONITOR_THREAD_NAME and thread.is_alive()
    ]


@pytest.fixture(autouse=True)
def parked_monitor(monkeypatch):
    """Park the monitor's tick, so only the tests about it feel it.

    Registering a build starts the monitor, and the monitor reads the real
    clock -- while every other test here feeds the model timestamps out of a
    fixture. One tick landing inside such a test would mark every service as
    beaten at "seconds since boot" and silence the assertions that follow it.
    The tests that exercise the thread install an interval of their own.
    """
    monkeypatch.setattr(phase_reporter, "_MONITOR_INTERVAL", 3600.0)


@pytest.fixture
def fake_clock(monkeypatch):
    """Drive the phase lines' own elapsed arithmetic from a scripted clock."""

    def install(*ticks):
        stream = iter(ticks)
        monkeypatch.setattr(phase_reporter, "time", SimpleNamespace(monotonic=lambda: next(stream)))

    return install


def test_a_long_gap_between_lines_beats_inside_the_gap():
    """The case the whole feature exists for: a silent step proves it is alive."""
    lines = vertex_lines(COLD_BUILD, 17)
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    beats = []
    with reporter.watch_build(model):
        model.feed(lines[0], 1000.0)
        # A 90s gap before the vertex's next line, ticked every 10s.
        for tick in range(1010, 1091, 10):
            beats.extend(reporter._heartbeat_pass(float(tick)))
        model.feed(lines[1], 1090.0)

    assert len(beats) == 2
    assert beats[0].startswith("  · virtual-accelerator 6/8 RUN ")
    assert beats[0].endswith(" (running 40.0s)")
    assert beats[1].endswith(" (running 1m20s)")


def test_a_service_that_keeps_talking_still_beats():
    """Heartbeats are independent of line arrival, per the non-TTY contract.

    A step that prints a line a second is not silent, but it is also not
    finished, and the operator still needs to see it move.
    """
    lines = vertex_lines(COLD_BUILD, 17)
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        for offset, line in enumerate(lines[:40]):
            model.feed(line, 1000.0 + offset)
        assert reporter._heartbeat_pass(1039.0) != []


def test_nothing_beats_before_the_interval_has_passed():
    lines = vertex_lines(COLD_BUILD, 17)
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        model.feed(lines[0], 1000.0)
        assert reporter._heartbeat_pass(1029.0) == []
        assert reporter._heartbeat_pass(1030.0) == []
        assert reporter._heartbeat_pass(1030.5) != []


def test_a_deregistered_build_never_beats_again():
    """The stale-heartbeat guard: a build that has returned goes quiet."""
    lines = vertex_lines(COLD_BUILD, 17)
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        model.feed(lines[0], 1000.0)

    assert reporter._heartbeat_pass(2000.0) == []
    assert reporter._heartbeat_pass(9999.0) == []


def test_deregistration_happens_when_the_block_raises():
    """A failed build must go quiet exactly as a successful one does."""
    lines = vertex_lines(COLD_BUILD, 17)
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with pytest.raises(RuntimeError):
        with reporter.watch_build(model):
            model.feed(lines[0], 1000.0)
            raise RuntimeError("compose build failed")

    assert reporter._heartbeat_pass(2000.0) == []


def test_watch_build_yields_the_model_and_emits_nothing_itself(sink):
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model) as watched:
        assert watched is model
        model.feed(vertex_lines(COLD_BUILD, 17)[0], 1000.0)
        reporter._heartbeat_pass(2000.0)

    assert sink.getvalue() == ""


def test_phase_lines_are_untouched_when_no_build_is_watched(sink, fake_clock):
    """The regression guard for the whole feature.

    Off a TTY the contract is that nothing about today's output changes except
    the heartbeats that were not there before -- so with no build registered,
    a phase's lines are byte-for-byte what they have always been.
    """
    fake_clock(0.0, 12.0, 90.0)

    reporter = PhaseReporter(color=False)
    with reporter.phase("Building images") as phase:
        phase.step("web-terminal")

    assert sink.getvalue() == (
        "→ Building images\n  · web-terminal (12.0s)\n  ✓ Building images (1m30s)\n"
    )


def test_a_finished_service_stops_beating():
    """``naming to ... done`` means the image is built, not running."""
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        for offset, line in enumerate(vertex_lines(COLD_BUILD, 23)):
            model.feed(line, 1000.0 + offset)
        assert reporter._heartbeat_pass(2000.0) == []


def test_the_export_pseudo_step_beats_like_any_other():
    """The longest stretch of a cold build carries no Dockerfile step index."""
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        model.feed("#23 [virtual-accelerator] exporting to image", 1000.0)
        (beat,) = reporter._heartbeat_pass(1100.0)

    assert beat == "  · virtual-accelerator exporting exporting to image (running 1m40s)"


def test_an_infrastructure_step_word_renders_without_a_step_index():
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        model.feed(vertex_lines(COLD_BUILD, 1)[0], 1000.0)
        (beat,) = reporter._heartbeat_pass(1031.0)

    assert beat == (
        "  · virtual-accelerator internal load build definition from Dockerfile (running 31.0s)"
    )


def test_a_single_image_build_beats_under_its_label():
    """A nameless ``[ 3/13]`` header belongs to the model's label."""
    reporter = PhaseReporter(color=False)
    model = BuildModel(label="myrepo-project:local")

    with reporter.watch_build(model):
        model.feed(vertex_lines(PROJECT_IMAGE, 10)[0], 1000.0)
        (beat,) = reporter._heartbeat_pass(1031.0)

    assert beat.startswith("  · myrepo-project:local 3/13 RUN apt-get update")


def test_every_service_of_a_compose_build_beats_once_per_interval():
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        for line in fixture_lines(COLD_BUILD)[:120]:
            model.feed(line, 1000.0)
        beats = reporter._heartbeat_pass(1031.0)

    services = [beat.split()[1] for beat in beats]
    assert services == [row.service for row in model.snapshot()]
    assert len(services) == len(set(services))
    assert reporter._heartbeat_pass(1032.0) == []


def test_each_watched_build_beats_on_its_own_schedule():
    reporter = PhaseReporter(color=False)
    services, project = BuildModel(), BuildModel(label="myrepo-project:local")

    with reporter.watch_build(services):
        services.feed(vertex_lines(COLD_BUILD, 17)[0], 1000.0)
        assert len(reporter._heartbeat_pass(1031.0)) == 1
        with reporter.watch_build(project):
            project.feed(vertex_lines(PROJECT_IMAGE, 10)[0], 1031.0)
            assert len(reporter._heartbeat_pass(1062.0)) == 2
        assert len(reporter._heartbeat_pass(1093.0)) == 1


def test_beat_state_dies_with_the_registration():
    """Re-registering a model starts from silence, not a stale schedule."""
    lines = vertex_lines(COLD_BUILD, 17)
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        model.feed(lines[0], 1000.0)
        assert len(reporter._heartbeat_pass(1031.0)) == 1
        assert reporter._heartbeat_pass(1032.0) == []

    with reporter.watch_build(model):
        assert len(reporter._heartbeat_pass(1033.0)) == 1


def test_a_long_run_caption_is_clipped_to_one_readable_line():
    """A step's opening caption is its whole shell command; the log is not."""
    lines = vertex_lines(COLD_BUILD, 17)
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        model.feed(lines[0], 1000.0)
        (beat,) = reporter._heartbeat_pass(1031.0)

    assert len(lines[0]) > 500
    assert len(beat) < 160
    assert "… (running 31.0s)" in beat


def test_no_heartbeat_line_ever_carries_buildkit_bookkeeping():
    """Off a TTY these lines are the only build output an operator sees, so a
    caption that is really the vertex's stopwatch or its status reads as the
    build itself: `· virtual-accelerator 6/8 DONE 0.0s (running 4m10s)`."""
    captions = [beat_caption(beat) for beat in replay_heartbeats(COLD_BUILD)]

    assert captions
    assert [caption for caption in captions if BOOKKEEPING.match(caption)] == []


def test_a_genuine_caption_reaches_the_heartbeat_line_verbatim():
    """The filter is between the model and both surfaces; this is the half of
    it that has to survive the trip to this one."""
    captions = {beat_caption(beat) for beat in replay_heartbeats(COLD_BUILD)}

    assert [caption for caption in PINNED_CAPTIONS if caption not in captions] == []


def test_a_stream_with_no_buildkit_headers_never_beats():
    """A podman provider's own progress format leaves the model empty."""
    reporter = PhaseReporter(color=False)
    model = BuildModel()

    with reporter.watch_build(model):
        for line in ("STEP 1/13: FROM python:3.11-slim", "Copying blob sha256:abc123"):
            model.feed(line, 1000.0)
        assert reporter._heartbeat_pass(2000.0) == []


def test_no_watchers_at_all_is_a_quiet_pass():
    assert PhaseReporter(color=False)._heartbeat_pass(2000.0) == []


# -- the monitor thread ------------------------------------------------------


class _RecordingReporter(PhaseReporter):
    """Records what was emitted, and what was still running when it was.

    ``running_at_emit`` is how the teardown ORDER is asserted rather than
    assumed: a line printed while the monitor was still alive is a line the
    monitor could have printed over.
    """

    def __init__(self) -> None:
        super().__init__(color=False)
        self.lines: list[str] = []
        self.running_at_emit: list[int] = []
        self.emitted = threading.Event()

    def emit(self, text: str, style: str | None = None) -> None:
        super().emit(text, style)
        self.lines.append(text)
        self.running_at_emit.append(len(monitor_threads()))
        self.emitted.set()


def test_the_monitor_runs_for_exactly_as_long_as_a_build_is_watched():
    """Off a TTY the watcher set owns the thread: first in starts, last out stops."""
    reporter = PhaseReporter(color=False)
    assert monitor_threads() == []

    with reporter.watch_build(BuildModel()):
        assert len(monitor_threads()) == 1
        with reporter.watch_build(BuildModel(label="myrepo-project:local")):
            assert len(monitor_threads()) == 1
        assert len(monitor_threads()) == 1

    assert monitor_threads() == []


def test_a_pinned_monitor_outlives_the_builds_it_watched():
    """The live region's monitor still has an open phase's elapsed to tick."""
    reporter = PhaseReporter(color=False)
    reporter._start_monitor(pinned=True)

    with reporter.watch_build(BuildModel()):
        assert len(monitor_threads()) == 1
    assert len(monitor_threads()) == 1

    reporter.stop_rendering()
    assert monitor_threads() == []


def test_the_monitor_thread_emits_every_heartbeat_the_pass_returns(sink, monkeypatch):
    """The integration the thread exists for, on a clock the test drives.

    The decision itself is covered above; what is proven here is that the loop
    calls it on its own, that the lines it returns reach ``emit`` (dropping
    them would silence the pass that followed), and that a beaten service is
    not beaten again on the next tick.
    """
    clock = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(phase_reporter, "time", SimpleNamespace(monotonic=lambda: clock.value))
    monkeypatch.setattr(phase_reporter, "_MONITOR_INTERVAL", 0.005)

    reporter = _RecordingReporter()
    model = BuildModel()

    with reporter.watch_build(model):
        model.feed(vertex_lines(COLD_BUILD, 17)[0], 1000.0)
        assert reporter.lines == []  # Nothing is due in the first instant.
        clock.value = 1031.0
        assert reporter.emitted.wait(timeout=10.0), "the monitor never emitted a heartbeat"

    assert len(reporter.lines) == 1
    assert reporter.lines[0].startswith("  · virtual-accelerator 6/8 RUN ")
    assert reporter.lines[0].endswith(" (running 31.0s)")
    assert reporter.lines[0] in sink.getvalue()


def test_a_failing_phase_stops_the_monitor_before_it_prints():
    """Nothing may print after ``✗``: the failure is the last word."""
    reporter = _RecordingReporter()
    reporter._start_monitor(pinned=True)

    with pytest.raises(RuntimeError):
        with reporter.phase("Building images"):
            assert len(monitor_threads()) == 1
            raise RuntimeError("compose build failed")

    assert monitor_threads() == []
    assert reporter.lines[-1].startswith("  ✗ Building images")
    assert reporter.running_at_emit[-1] == 0


def test_an_interrupted_phase_stops_the_monitor_before_it_prints():
    reporter = _RecordingReporter()
    reporter._start_monitor(pinned=True)

    with pytest.raises(KeyboardInterrupt):
        with reporter.phase("Building images"):
            raise KeyboardInterrupt

    assert monitor_threads() == []
    assert reporter.lines[-1].startswith("  ⚠ Building images interrupted")
    assert reporter.running_at_emit[-1] == 0


def test_a_direct_install_reporter_swap_leaves_no_thread_behind():
    """The path a test fixture or a library caller takes -- no verb involved."""
    reporter = PhaseReporter(color=False)
    previous = install_reporter(reporter)
    try:
        reporter._start_monitor(pinned=True)
        assert len(monitor_threads()) == 1
    finally:
        assert install_reporter(previous) is reporter

    assert monitor_threads() == []
    assert current_reporter() is previous


def test_a_verb_under_the_cli_runner_leaves_no_thread_behind():
    """The production path: ``lifecycle_reporter``'s uninstall quiesces the verb.

    The pinned start stands in for the live region a TTY run would mount --
    without it the watcher set would have stopped the monitor on its own, and
    the uninstall would be proving nothing.
    """
    seen = []

    @click.command()
    def verb():
        from osprey.cli.main import lifecycle_reporter

        with lifecycle_reporter() as reporter:
            reporter._start_monitor(pinned=True)
            with reporter.watch_build(BuildModel()):
                seen.append(len(monitor_threads()))

    original = current_reporter()
    result = CliRunner().invoke(verb, [])

    assert result.exit_code == 0, result.output
    assert seen == [1]
    assert monitor_threads() == []
    assert current_reporter() is original


def test_the_quiet_verbose_reporter_runs_no_monitor_at_all():
    """Under ``--verbose`` the child's own output streams; nothing to prove."""
    reporter = phase_reporter.NullReporter(verbose=True)

    with reporter.watch_build(BuildModel()):
        assert monitor_threads() == []

    reporter.stop_rendering()
    assert monitor_threads() == []
