"""Tests for the lifecycle phase reporter (plain-line progress output)."""

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from osprey.cli import phase_reporter
from osprey.cli.phase_reporter import (
    NullReporter,
    PhaseReporter,
    current_reporter,
    format_elapsed,
    install_reporter,
    is_verbose,
)
from osprey.cli.styles import osprey_theme


@pytest.fixture
def sink(monkeypatch):
    """Capture reporter output through a forced-terminal console.

    force_terminal mirrors the win32 console in ``styles``: it proves the
    no-ANSI guarantee comes from the reporter, not from Rich noticing a pipe.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(
        phase_reporter,
        "console",
        Console(
            file=buffer,
            theme=osprey_theme,
            force_terminal=True,
            color_system="standard",
            no_color=False,
            width=200,
        ),
    )
    return buffer


@pytest.fixture
def fake_clock(monkeypatch):
    """Drive the reporter's elapsed arithmetic from a scripted monotonic clock.

    Patches the module's own ``time`` reference, not ``time.monotonic``
    itself, so nothing else in the process sees the fake clock.
    """

    def install(*ticks):
        stream = iter(ticks)
        monkeypatch.setattr(phase_reporter, "time", SimpleNamespace(monotonic=lambda: next(stream)))

    return install


@pytest.fixture(autouse=True)
def restore_installed_reporter():
    """Leave the module singleton exactly as the test found it."""
    original = current_reporter()
    yield
    install_reporter(original)


def test_format_elapsed_seconds_and_minutes():
    assert format_elapsed(0.0) == "0.0s"
    assert format_elapsed(3.44) == "3.4s"
    assert format_elapsed(59.9) == "59.9s"
    assert format_elapsed(60) == "1m00s"
    assert format_elapsed(127.6) == "2m07s"


def test_phase_lifecycle_lines(sink):
    reporter = PhaseReporter(color=False)
    with reporter.phase("Rendering configuration") as phase:
        phase.step("compose")
        phase.done("6 files")

    lines = [line for line in sink.getvalue().splitlines() if line.strip()]
    assert lines[0] == "→ Rendering configuration"
    assert lines[1].startswith("  · compose")
    assert lines[2].startswith("  ✓ Rendering configuration (")
    assert lines[2].endswith("— 6 files")


def test_done_reports_elapsed_from_monotonic(sink, fake_clock):
    fake_clock(100.0, 190.5)

    reporter = PhaseReporter(color=False)
    reporter.phase("Building images").done()

    assert "✓ Building images (1m30s)" in sink.getvalue()


def test_step_reports_lap_and_suppresses_instant_laps(sink, fake_clock):
    # phase start, first step, second step
    fake_clock(0.0, 0.01, 12.0)

    reporter = PhaseReporter(color=False)
    phase = reporter.phase("Building images")
    phase.step("web-terminal")
    phase.step("project")

    lines = [line for line in sink.getvalue().splitlines() if line.strip()]
    assert lines[1] == "  · web-terminal"
    assert lines[2] == "  · project (12.0s)"


def test_non_tty_output_has_no_ansi_escapes(sink):
    reporter = PhaseReporter(color=False)
    with reporter.phase("Starting stack") as phase:
        phase.step("compose up")

    output = sink.getvalue()
    assert "\x1b[" not in output
    assert "→ Starting stack" in output
    assert "  · compose up" in output


def test_color_defaults_to_stdout_isatty(monkeypatch):
    monkeypatch.setattr(phase_reporter.sys, "stdout", io.StringIO())
    assert PhaseReporter().color is False


def test_tty_output_is_styled(sink):
    PhaseReporter(color=True).phase("Starting stack").done()
    assert "\x1b[" in sink.getvalue()


def test_teardown_runs_on_exception_and_replays_spool(sink, tmp_path):
    spool = tmp_path / "build-images.log"
    spool.write_text("step 1/3\nERROR: no such image\n")

    reporter = PhaseReporter(color=False)
    with pytest.raises(RuntimeError):
        with reporter.phase("Building images") as phase:
            phase.set_spool(spool)
            raise RuntimeError("compose failed")

    output = sink.getvalue()
    assert "✗ Building images (" in output
    assert "ERROR: no such image" in output
    assert str(spool) in output


def test_fail_dumps_replay_content_in_full(sink, tmp_path):
    spool = tmp_path / "up.log"
    spool.write_text("line one\nline two\nline three\n")

    reporter = PhaseReporter(color=False)
    reporter.phase("Starting stack").fail(spool)

    output = sink.getvalue()
    for line in ("line one", "line two", "line three"):
        assert line in output


def test_fail_without_replay_prints_only_the_failure_line(sink):
    reporter = PhaseReporter(color=False)
    reporter.phase("Provider check").fail(None)

    lines = [line for line in sink.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0] == "→ Provider check"
    assert lines[1].startswith("  ✗ Provider check (")


def test_fail_reports_unreadable_spool_without_raising(sink, tmp_path):
    reporter = PhaseReporter(color=False)
    reporter.phase("Starting stack").fail(tmp_path / "missing.log")

    assert "could not read spool" in sink.getvalue()


def test_explicit_fail_is_not_repeated_by_teardown(sink, tmp_path):
    spool = tmp_path / "build.log"
    spool.write_text("boom\n")

    reporter = PhaseReporter(color=False)
    with pytest.raises(RuntimeError):
        with reporter.phase("Building images") as phase:
            phase.fail(spool)
            raise RuntimeError("propagated")

    assert sink.getvalue().count("boom") == 1
    assert sink.getvalue().count("✗ Building images") == 1


@pytest.mark.parametrize("code", [0, None])
def test_a_clean_systemexit_closes_the_phase_as_a_success(sink, tmp_path, code):
    """``down`` ends by exiting on compose's status; exit 0 is not a failure.

    Without this the successful path prints ``✗`` and replays the entire spool
    it just captured — the exact noise the phase lines exist to remove.
    """
    spool = tmp_path / "down.log"
    spool.write_text("compose chatter\n")

    reporter = PhaseReporter(color=False)
    with pytest.raises(SystemExit):
        with reporter.phase("Stopping myrepo"):
            reporter.note_spool(spool)
            raise SystemExit(code)

    assert "✓ Stopping myrepo" in sink.getvalue()
    assert "✗" not in sink.getvalue()
    assert "compose chatter" not in sink.getvalue()


@pytest.mark.parametrize("code", [1, "boom"])
def test_a_nonzero_systemexit_still_fails_the_phase(sink, tmp_path, code):
    """Only a clean exit is a success; a failing one keeps the replay."""
    spool = tmp_path / "down.log"
    spool.write_text("compose chatter\n")

    reporter = PhaseReporter(color=False)
    with pytest.raises(SystemExit):
        with reporter.phase("Stopping myrepo"):
            reporter.note_spool(spool)
            raise SystemExit(code)

    assert "✗ Stopping myrepo" in sink.getvalue()
    assert "compose chatter" in sink.getvalue()


def test_interrupt_names_partial_spool_without_replaying(sink, tmp_path):
    spool = tmp_path / "up.log"
    spool.write_text("half a log\n")

    reporter = PhaseReporter(color=False)
    with pytest.raises(KeyboardInterrupt):
        with reporter.phase("Starting stack"):
            reporter.note_spool(spool)
            raise KeyboardInterrupt

    lines = [line for line in sink.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[1].startswith("  ⚠ Starting stack interrupted (")
    assert str(spool) in lines[1]
    assert "half a log" not in sink.getvalue()


def test_interrupt_without_spool_prints_one_line(sink):
    reporter = PhaseReporter(color=False)
    with pytest.raises(KeyboardInterrupt):
        with reporter.phase("Project environment"):
            raise KeyboardInterrupt

    lines = [line for line in sink.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[1] == lines[1].rstrip()
    assert "partial output" not in lines[1]


def test_current_phase_tracks_the_open_phase(sink):
    reporter = PhaseReporter(color=False)
    assert reporter.current_phase is None
    with reporter.phase("Repo created") as phase:
        assert reporter.current_phase is phase
    assert reporter.current_phase is None


def test_note_spool_without_an_open_phase_is_harmless(sink, tmp_path):
    reporter = PhaseReporter(color=False)
    reporter.note_spool(tmp_path / "orphan.log")
    assert sink.getvalue() == ""


def test_null_reporter_emits_nothing(sink, tmp_path):
    spool = tmp_path / "verbose.log"
    spool.write_text("streamed live already\n")

    reporter = NullReporter()
    with reporter.phase("Building images") as phase:
        phase.step("web-terminal")
        phase.set_spool(spool)
        phase.fail(spool)

    assert sink.getvalue() == ""


def test_null_reporter_swallows_interrupt_teardown(sink):
    reporter = NullReporter(verbose=True)
    with pytest.raises(KeyboardInterrupt):
        with reporter.phase("Starting stack"):
            raise KeyboardInterrupt

    assert sink.getvalue() == ""


def test_install_and_current_reporter_round_trip():
    reporter = PhaseReporter(color=False)
    previous = install_reporter(reporter)

    assert current_reporter() is reporter
    assert install_reporter(previous) is reporter
    assert current_reporter() is previous


def test_default_reporter_is_quiet_and_not_verbose():
    assert isinstance(current_reporter(), NullReporter)
    assert is_verbose() is False


def test_is_verbose_follows_the_installed_reporter():
    install_reporter(NullReporter(verbose=True))
    assert is_verbose() is True

    install_reporter(PhaseReporter(color=False))
    assert is_verbose() is False
