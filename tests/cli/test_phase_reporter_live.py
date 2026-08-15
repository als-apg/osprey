"""Tests for the terminal live region (``LiveReporter``).

The region is decoration over the plain lines, and every test here is about
that relationship rather than about how the region looks (which
``test_live_render.py`` owns): the same words still reach the scrollback, the
region never outlives the moment it is torn down, and nothing it does can
appear after a failure line.

Everything runs on a recording console with ``force_terminal=True``, because a
Rich ``Live`` on a non-terminal console silently renders nothing -- a test suite
that forgot this would assert against a region that was never drawn.
"""

import io
import logging
import re
import sys
import threading
import time
from importlib import import_module
from types import SimpleNamespace

import click
import pytest
from rich.console import Console
from rich.logging import RichHandler

from osprey.cli import phase_reporter
from osprey.cli.phase_reporter import (
    LiveReporter,
    NullReporter,
    PhaseReporter,
    current_reporter,
    install_reporter,
)
from osprey.cli.styles import Styles, osprey_theme
from osprey.deployment.build_progress import BuildModel
from osprey_connectors.logger import get_logger

#: The install site's module. Imported this way because ``osprey.cli`` exports a
#: ``main`` FUNCTION of its own, which shadows the submodule of the same name
#: under every ``import ... as`` spelling.
cli_main = import_module("osprey.cli.main")

#: Anything Rich writes that is not text: styles, cursor moves, erases.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

#: The tail of a heartbeat line -- what must never appear on a terminal, where
#: the region moving is the proof that the build is alive. Deliberately the same
#: words the hand-off's committed line uses: they are one format, and a test
#: that stopped matching both would go quietly half-blind.
_HEARTBEAT_MARK = "(running "


@pytest.fixture(autouse=True)
def parked_monitor(monkeypatch):
    """Park the monitor's tick, so only the tests about it feel it.

    Mounting the region starts a real thread on the real clock, while these
    tests drive phases and models from scripted timestamps. One tick landing
    mid-test repaints a region full of "seconds since boot" and silences every
    assertion after it -- an ordering-dependent flake with no visible cause.
    The tests that exercise the thread install an interval of their own.
    """
    monkeypatch.setattr(phase_reporter, "_MONITOR_INTERVAL", 3600.0)


@pytest.fixture(autouse=True)
def restore_installed_reporter():
    """Leave the module singleton exactly as the test found it."""
    original = current_reporter()
    yield
    install_reporter(original)


def recording_console() -> tuple[Console, io.StringIO]:
    """A themed console that records everything written to it, escapes included.

    ``force_terminal`` is what makes the ``Live`` real here: off a terminal Rich
    skips the repaint entirely. The theme is not optional either -- the region's
    styles are semantic tokens, and a bare ``Console()`` raises ``MissingStyle``
    on the first one.
    """
    buffer = io.StringIO()
    return (
        Console(file=buffer, theme=osprey_theme, force_terminal=True, width=100),
        buffer,
    )


@pytest.fixture
def live():
    """A started live reporter, its buffer, and a guaranteed teardown.

    ``color`` is off so that the only escape sequences in the buffer come from
    the region: with styled words every line would carry its own, and "nothing
    moved after the failure line" would be unassertable.
    """
    console, buffer = recording_console()
    reporter = LiveReporter(console=console)
    reporter.color = False
    reporter.start_rendering()
    try:
        yield reporter, buffer
    finally:
        reporter.stop_rendering()


def visible_lines(text: str) -> list[str]:
    """The words in ``text``, with escapes and blank lines dropped."""
    return [line for line in _ANSI.sub("", text).splitlines() if line.strip()]


def after(text: str, marker: str) -> str:
    """Everything written after the line carrying ``marker``."""
    index = text.index(marker)
    return text[text.index("\n", index) + 1 :]


def cursor_control(text: str) -> str | None:
    """The first cursor control in ``text``, or None if it is all words.

    An escape sequence OR a bare carriage return: a repaint opens its line with
    a lone ``\\r``, which is cursor control in the same sense as the escapes but
    matches none of their shapes. "Nothing moved after this line" asserted on
    the escape pattern alone would miss exactly the repaint it looks for.
    """
    match = _ANSI.search(text)
    if match is not None:
        return match.group()
    return "\r" if "\r" in text else None


def monitor_threads() -> list[threading.Thread]:
    """Every reporter monitor thread alive in the process."""
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == phase_reporter._MONITOR_THREAD_NAME and thread.is_alive()
    ]


def build_with_rows() -> BuildModel:
    """A model mid-build: two services, one of them several steps in."""
    model = BuildModel()
    model.feed("#1 [virtual-accelerator internal] load build definition", 1000.0)
    model.feed("#2 [event-dispatcher 3/13] RUN pip install -e .", 1000.0)
    model.feed("#2 1.234 Downloading torch (2.1 GB)", 1001.0)
    return model


# ---------------------------------------------------------------------------
# The lines are the record; the region is decoration over them
# ---------------------------------------------------------------------------


def test_the_scrollback_is_content_identical_to_the_plain_reporter(monkeypatch):
    """The point of the feature: a terminal gains a region, not different words.

    Run the same script through both reporters and compare what a reader would
    see. A phase line that gained a spinner, or a step that moved into the
    region instead of the scrollback, would show up here.
    """
    plain_console, plain_buffer = recording_console()
    monkeypatch.setattr(phase_reporter, "console", plain_console)
    plain = PhaseReporter(color=False)

    live_console, live_buffer = recording_console()
    rich = LiveReporter(console=live_console)
    rich.color = False
    rich.start_rendering()
    try:
        for reporter in (plain, rich):
            with reporter.phase("Building services") as phase:
                phase.step("compose config")
                phase.step("images built")
    finally:
        rich.stop_rendering()

    assert visible_lines(live_buffer.getvalue()) == visible_lines(plain_buffer.getvalue())
    assert [line.split(" (")[0] for line in visible_lines(plain_buffer.getvalue())] == [
        "→ Building services",
        "  · compose config",
        "  · images built",
        "  ✓ Building services",
    ]


def test_a_raw_stdout_write_is_proxied_through_the_console():
    """``redirect_stdout`` is stated policy: a helper that never heard of the
    reporter still prints above the region rather than into it.

    Mounted inside the test body rather than from the fixture: pytest reassigns
    ``sys.stdout`` between the setup and call phases, which would drop the
    proxy the Live installed and prove nothing.
    """
    console, buffer = recording_console()
    reporter = LiveReporter(console=console)
    reporter.color = False

    reporter.start_rendering()
    try:
        sys.stdout.write("straight to stdout\n")
        sys.stdout.flush()
    finally:
        reporter.stop_rendering()

    assert "straight to stdout" in visible_lines(buffer.getvalue())


def test_the_live_is_configured_as_the_only_refresher(live):
    """Each of these is load-bearing, so each is pinned.

    Rich's own refresh thread would race the monitor for the cursor; a
    non-transient region would freeze its last half-drawn frame into the
    scrollback; and a second console would repaint over the lines.
    """
    reporter, _ = live

    assert reporter._live.auto_refresh is False
    assert reporter._live.transient is True
    assert reporter._live._redirect_stdout is True
    assert reporter._live.console is reporter.out()


def test_starting_twice_mounts_one_region_and_one_thread(live):
    """``init --up`` chains verbs through one reporter; the second must not
    mount a region of its own over the first."""
    reporter, _ = live
    first = reporter._live

    reporter.start_rendering()

    assert reporter._live is first
    assert len(monitor_threads()) == 1


# ---------------------------------------------------------------------------
# What the region shows
# ---------------------------------------------------------------------------


def test_a_repaint_draws_the_ticker_and_the_rows_under_it(live):
    """One repaint, driven by hand: the ticker and every unfinished service.

    The title is asserted to appear EXACTLY ONCE — as the permanent line the
    phase opened with, never again in the region that repaints beneath it. A
    substring check would pass on that one line alone and would go on passing if
    the duplicate came back, which is the thing this pins.
    """
    reporter, buffer = live
    reporter.phase("Building services")

    with reporter.watch_build(build_with_rows()):
        assert reporter.refresh_live() is True

    # Carriage returns are cursor control, same as the escapes: a repaint opens
    # its line with one, and they are not part of what the operator reads.
    painted = _ANSI.sub("", buffer.getvalue()).replace("\r", "")
    assert painted.count("Building services") == 1
    assert "→ Building services" in painted
    assert re.search(r"^ {2}\S {1}\d+m\d\ds$", painted, re.MULTILINE) is not None
    assert "virtual-accelerator" in painted
    assert "event-dispatcher" in painted
    assert "Downloading torch (2.1 GB)" in painted


def test_two_registered_builds_render_as_one_table(live):
    """A compose run and a single image can be in flight at once, and the
    operator is waiting on both."""
    reporter, buffer = live
    reporter.phase("Building services")
    single = BuildModel(label="myrepo-project:local")
    single.feed("#4 [ 2/13] RUN uv sync", 1000.0)

    with reporter.watch_build(build_with_rows()), reporter.watch_build(single):
        reporter.refresh_live()

    painted = _ANSI.sub("", buffer.getvalue())
    assert "event-dispatcher" in painted
    assert "myrepo-project:local" in painted


def test_closing_a_phase_takes_its_region_down_with_it(live):
    """The ``✓`` must not print under a region still ticking for the phase it
    just reported finished.

    Rich appends the region to whatever print is passing through, so without a
    repaint at the state change the stale frame lands *after* the closing line
    and stays there until the next tick.
    """
    reporter, buffer = live
    with reporter.phase("Building services"):
        # The production shape: a build is watched for exactly one captured
        # subprocess, inside the phase that reports it.
        with reporter.watch_build(build_with_rows()):
            reporter.refresh_live()
    written = buffer.getvalue()

    assert "virtual-accelerator" in written
    assert "virtual-accelerator" not in after(written, "✓ Building services")


def test_a_styled_line_does_not_take_the_region_with_it(monkeypatch):
    """A ``style=`` argument applies to everything a print renders -- including
    the region Rich appends to it, which would repaint the whole build table in
    the colour of the line that happened to scroll past."""
    monkeypatch.setattr(phase_reporter, "time", SimpleNamespace(monotonic=lambda: 1000.0))
    regions = []

    for style in (Styles.SUCCESS, None):
        console, buffer = recording_console()
        reporter = LiveReporter(console=console)
        reporter.start_rendering()
        try:
            reporter.phase("Building services")
            with reporter.watch_build(build_with_rows()):
                reporter.refresh_live()
                marker = "  the line"
                reporter.emit(marker, style=style)
                regions.append(after(buffer.getvalue(), marker))
        finally:
            reporter.stop_rendering()

    assert regions[0] == regions[1]
    assert "virtual-accelerator" in regions[0]


def test_refresh_live_is_false_before_the_region_is_mounted():
    """False is what hands the tick to the heartbeat pass instead."""
    console, _ = recording_console()

    assert LiveReporter(console=console).refresh_live() is False


def test_refresh_live_is_false_again_once_the_region_is_stopped(live):
    reporter, _ = live
    reporter.stop_rendering()

    assert reporter.refresh_live() is False


def test_a_repaint_that_raises_degrades_to_plain_lines(live, monkeypatch):
    """The region is decoration over a deploy that is still running: losing it
    must not cost the phase lines and heartbeats behind it."""
    reporter, buffer = live

    def explode(*args, **kwargs):
        raise RuntimeError("render blew up")

    monkeypatch.setattr(phase_reporter, "render_live_region", explode)

    assert reporter.refresh_live() is False
    assert reporter._live is None

    reporter.emit("still reporting")
    assert "still reporting" in visible_lines(buffer.getvalue())


# ---------------------------------------------------------------------------
# The monitor drives the region -- and only the region
# ---------------------------------------------------------------------------


def test_the_monitor_repaints_instead_of_heartbeating(monkeypatch):
    """The whole reason ``refresh_live`` reports back: a tick does one or the
    other, and a table that visibly moves does not also need lines saying so.

    The build here is stalled well past the heartbeat interval, which off a
    terminal would print a line every pass.
    """
    monkeypatch.setattr(phase_reporter, "_MONITOR_INTERVAL", 0.01)
    console, buffer = recording_console()
    reporter = LiveReporter(console=console)
    reporter.color = False
    model = BuildModel()
    model.feed("#2 [event-dispatcher 3/13] RUN pip install -e .", time.monotonic() - 600)

    reporter.start_rendering()
    try:
        reporter.phase("Building services")
        with reporter.watch_build(model):
            time.sleep(0.1)
    finally:
        reporter.stop_rendering()

    painted = _ANSI.sub("", buffer.getvalue())
    assert "event-dispatcher" in painted
    assert _HEARTBEAT_MARK not in painted


def test_the_region_stops_before_the_reporter_lets_go(monkeypatch):
    """A leaked monitor outlives its verb and repaints over whatever runs next.

    Asserted through ``install_reporter``, because the swap -- not the verb's
    ``finally`` -- is the path every caller has in common.
    """
    monkeypatch.setattr(phase_reporter, "_MONITOR_INTERVAL", 0.01)
    console, _ = recording_console()
    reporter = LiveReporter(console=console)
    reporter.color = False

    previous = install_reporter(reporter)
    reporter.start_rendering()
    assert len(monitor_threads()) == 1

    install_reporter(previous)

    assert monitor_threads() == []
    assert reporter._live is None


# ---------------------------------------------------------------------------
# Teardown ordering: nothing moves after the last line
# ---------------------------------------------------------------------------


def test_teardown_stops_the_monitor_before_the_region():
    """The ordering contract itself, asserted as a sequence rather than as a
    symptom -- the symptom is a garbled line, which only appears if a tick
    happens to land inside the microsecond between the two steps.

    Taking the region down first leaves a tick free to repaint one that is no
    longer there, restoring the cursor onto the row the next plain line is
    about to be written on. The base class stops the monitor and the override
    adds the region, so this order is a property of the inheritance and not of
    any one of the four call sites.
    """
    console, _ = recording_console()
    order = []

    class _Order(LiveReporter):
        def _stop_monitor(self) -> None:
            order.append("monitor")
            super()._stop_monitor()

        def _close_live(self) -> None:
            order.append("region")
            super()._close_live()

    reporter = _Order(console=console)
    reporter.color = False
    reporter.start_rendering()
    reporter.stop_rendering()

    assert order == ["monitor", "region"]


def test_a_failure_line_is_the_last_thing_written(live, tmp_path):
    """The contract this whole task is judged on.

    After the ``✗`` the buffer must hold the replayed spool and nothing else:
    no escape sequence (the region is down, so nothing can repaint over the
    replay) and no heartbeat (the monitor is joined, so nothing can claim a
    build that just died is still running).
    """
    reporter, buffer = live
    spool = tmp_path / "build.log"
    spool.write_text("compose stderr line one\ncompose stderr line two\n")
    reporter.phase("Building services")

    with reporter.watch_build(build_with_rows()):
        reporter.refresh_live()
        reporter.current_phase.fail(spool)

    written = buffer.getvalue()
    tail = after(written, "✗ Building services")
    # The region really was painting up to the failure -- otherwise "nothing
    # moved afterwards" would be true of a run that never moved at all.
    assert _ANSI.search(written[: written.index("✗")]) is not None
    assert _ANSI.search(tail) is None
    assert _HEARTBEAT_MARK not in tail
    assert "compose stderr line two" in tail
    assert reporter._live is None
    assert monitor_threads() == []


def test_an_interrupt_line_is_the_last_thing_written(live):
    """Ctrl-C's answer is one line, and nothing may print after it."""
    reporter, buffer = live
    phase = reporter.phase("Starting containers")
    phase.set_spool(None)

    with reporter.watch_build(build_with_rows()):
        reporter.refresh_live()
        phase.interrupted()

    written = buffer.getvalue()
    tail = after(written, "⚠ Starting containers")
    assert _ANSI.search(written[: written.index("⚠")]) is not None
    assert _ANSI.search(tail) is None
    assert _HEARTBEAT_MARK not in tail
    assert reporter._live is None
    assert monitor_threads() == []


def test_the_spool_replay_never_passes_through_the_live():
    """Structural, not textual: the replay is a plain write by the time it runs.

    A replay rendered through a live region would be re-wrapped and repainted
    over -- and a failed build's spool is the one output an operator reads
    line by line.
    """
    console, _ = recording_console()
    seen = {}

    class _Spy(LiveReporter):
        def replay(self, path):
            seen["reporter_live"] = self._live
            seen["console_live"] = self.out()._live_stack
            super().replay(path)

    reporter = _Spy(console=console)
    reporter.color = False
    reporter.start_rendering()
    try:
        reporter.phase("Building services").fail(None)
    finally:
        reporter.stop_rendering()

    assert seen["reporter_live"] is None
    assert seen["console_live"] == []


def test_the_plain_reporter_still_has_no_region_to_stop(monkeypatch):
    """``stop_rendering`` and ``start_rendering`` are called unconditionally by
    the install site, on whichever reporter it installed."""
    console, _ = recording_console()
    monkeypatch.setattr(phase_reporter, "console", console)

    for reporter in (PhaseReporter(color=False), NullReporter(verbose=True)):
        reporter.start_rendering()
        reporter.stop_rendering()

    assert monitor_threads() == []


# ---------------------------------------------------------------------------
# Giving the terminal up: temporarily (``suspended``) and for good (``hand_off``)
# ---------------------------------------------------------------------------


def test_a_suspend_takes_the_region_down_and_puts_it_back(live):
    """The prompt case: a region left mounted repaints over the question while
    the operator is still reading it."""
    reporter, _ = live
    reporter.phase("Preparing configuration")

    with reporter.suspended():
        assert reporter._live is None
        assert monitor_threads() == []

    assert reporter._live is not None
    assert len(monitor_threads()) == 1


def test_a_suspend_restores_the_region_when_the_prompt_raises(live):
    """A prompt the operator Ctrl-Cs is exactly the one that must not leave the
    rest of the verb rendering nothing."""
    reporter, _ = live

    with pytest.raises(KeyboardInterrupt), reporter.suspended():
        raise KeyboardInterrupt

    assert reporter._live is not None
    assert len(monitor_threads()) == 1


def test_nested_suspends_stop_once_and_restart_once():
    """Reentrancy, asserted as counts and not only as an end state.

    An inner block that restarted on its own exit would remount the region on
    top of the prompt the outer block is still waiting on -- which leaves the
    region mounted at the end either way, so the end state alone proves nothing.
    """
    console, _ = recording_console()
    calls = []

    class _Counted(LiveReporter):
        def start_rendering(self) -> None:
            calls.append("start")
            super().start_rendering()

        def stop_rendering(self) -> None:
            calls.append("stop")
            super().stop_rendering()

    reporter = _Counted(console=console)
    reporter.color = False
    reporter.start_rendering()
    try:
        with reporter.suspended():
            with reporter.suspended():
                assert reporter._live is None
            # The inner block is out and the region is still down: the prompt
            # the outer block opened is still on the terminal.
            assert reporter._live is None
        assert reporter._live is not None
    finally:
        reporter.stop_rendering()

    assert calls == ["start", "stop", "start", "stop"]


def test_a_nested_suspend_leaves_the_region_alone_whatever_it_finds(live):
    """Nesting is COUNTED, not inferred from what is on the screen.

    Inferring it from "is a region mounted" holds only while nothing else
    mounts one -- and a chained verb sharing this reporter (``init --up``)
    does exactly that. An inner block acting on what it found would take that
    region down and hand it back while the outer block's prompt is still
    waiting for an answer.
    """
    reporter, _ = live

    with reporter.suspended():
        reporter.start_rendering()  # the chained verb, sharing this reporter
        mounted = reporter._live

        with reporter.suspended():
            assert reporter._live is mounted

        assert reporter._live is mounted


def test_suspending_twice_in_a_row_is_safe(live):
    """Two prompts in one verb -- reset asks its typed confirmation after the
    seed prompt has already been answered."""
    reporter, _ = live

    for _ in range(2):
        with reporter.suspended():
            pass

    assert reporter._live is not None
    assert len(monitor_threads()) == 1


def test_a_suspend_does_not_resurrect_a_region_that_degraded(live, monkeypatch):
    """A run that fell back to plain lines stays fallen back.

    ``refresh_live`` takes the region down when a repaint raises, and the run
    finishes without one. A prompt afterwards must not hand it a fresh region
    that the same failing repaint would only take down again.
    """
    reporter, _ = live
    monkeypatch.setattr(phase_reporter, "render_live_region", _explode)
    assert reporter.refresh_live() is False

    with reporter.suspended():
        pass

    assert reporter._live is None


def test_hand_off_stops_the_region_and_joins_the_monitor(live):
    """The terminal is about to belong to compose: a Live still mounted when
    ``os.execvpe`` lands leaves the cursor hidden and the last frame half-drawn
    under compose's own output, in a process that no longer exists to fix it."""
    reporter, _ = live
    reporter.phase("Starting osprey-project")

    with reporter.watch_build(build_with_rows()):
        reporter.refresh_live()
        reporter.hand_off()

    assert reporter._live is None
    assert monitor_threads() == []


def test_hand_off_commits_the_open_phase_as_a_plain_line(live):
    """The ticker was the only word on a phase that never prints a ``✓``.

    An attached ``up`` is replaced by compose with its start phase still open,
    so the hand-off erases the operator's only reading of it mid-frame. The
    committed line is the LAST thing written: compose's own output starts on
    the next row, and nothing of this process may repaint over it.
    """
    reporter, buffer = live
    reporter.phase("Starting osprey-project")

    with reporter.watch_build(build_with_rows()):
        reporter.refresh_live()
        reporter.hand_off()

    written = buffer.getvalue()
    committed = "  · Starting osprey-project (running "
    assert committed in _ANSI.sub("", written)
    # The region really was painting up to the hand-off -- otherwise "nothing
    # follows the committed line" would be true of a run that never drew one.
    assert _ANSI.search(written[: written.index("· Starting")]) is not None
    assert "virtual-accelerator" not in after(written, committed)
    assert cursor_control(after(written, committed)) is None


def test_handing_off_twice_commits_one_line(live):
    """Idempotent: ``up`` and ``restart`` both reach the exec point through
    helpers that each hand off, and one phase must not report itself twice."""
    reporter, buffer = live
    reporter.phase("Starting osprey-project")

    reporter.hand_off()
    reporter.hand_off()

    assert _ANSI.sub("", buffer.getvalue()).count(_HEARTBEAT_MARK) == 1


def test_start_rendering_after_a_hand_off_never_remounts(live):
    """The degrade is PERMANENT. Every later start -- an install site swapping
    a reporter in, a ``suspended()`` block that spanned the hand-off -- must
    leave the terminal to the process that owns it now."""
    reporter, _ = live
    reporter.hand_off()

    reporter.start_rendering()

    assert reporter._live is None
    assert monitor_threads() == []


def test_a_suspend_open_across_a_hand_off_does_not_remount(live):
    """The seams compose: reset prompts, hands off, and the block still exits."""
    reporter, _ = live
    reporter.phase("Starting osprey-project")

    with reporter.suspended():
        reporter.hand_off()

    assert reporter._live is None
    assert monitor_threads() == []


def test_a_failure_after_a_hand_off_prints_plain_lines_only(live, tmp_path):
    """``os.execvpe`` raises when the compose binary is missing, and the
    ``fail()`` that answers it runs in a process whose terminal has already
    been given away. It prints its ``✗`` and its spool, and mounts nothing."""
    reporter, buffer = live
    spool = tmp_path / "compose.log"
    spool.write_text("compose stderr line one\n")
    phase = reporter.phase("Starting osprey-project")
    reporter.hand_off()

    phase.fail(spool)

    written = buffer.getvalue()
    assert "✗ Starting osprey-project" in _ANSI.sub("", written)
    assert cursor_control(after(written, "✗ Starting osprey-project")) is None
    assert "compose stderr line one" in written
    assert reporter._live is None
    assert monitor_threads() == []


def test_the_seams_are_no_ops_with_nothing_to_render(monkeypatch):
    """Both are called unconditionally by their call sites, on whichever
    reporter the verb installed -- so a missing definition is an
    ``AttributeError`` in production, and a definition that printed something
    would put a line off a terminal that the terminal never shows."""
    console, buffer = recording_console()
    monkeypatch.setattr(phase_reporter, "console", console)

    for reporter in (PhaseReporter(color=False), NullReporter(verbose=True)):
        reporter.phase("Starting osprey-project")
        with reporter.suspended():
            with reporter.suspended():
                pass
        reporter.hand_off()
        reporter.hand_off()

    assert visible_lines(buffer.getvalue()) == ["→ Starting osprey-project"]
    assert monitor_threads() == []


def test_stdout_piped_while_stdin_is_a_terminal_prompts_with_no_region(monkeypatch):
    """``osprey up | tee``: the prompt is real (stdin is still the operator's
    terminal) while stdout is a pipe, so there is no region to pause for it.

    The whole point of the no-op semantics -- the install site picked the plain
    reporter, and the seams the prompt site wraps itself in must leave both the
    prompt and the piped scrollback exactly as they were.
    """
    piped = _PipedStdout()
    monkeypatch.setattr(cli_main.sys, "stdout", piped)
    reporter = cli_main._tty_aware_reporter()
    console, buffer = recording_console()
    monkeypatch.setattr(phase_reporter, "console", console)

    def answer(prompt: str) -> str:
        """Stand in for the operator's terminal, writing the prompt as
        ``input()`` does before it reads the reply."""
        sys.stdout.write(prompt)
        return "y"

    monkeypatch.setattr(click.termui, "visible_prompt_func", answer)

    reporter.phase("Preparing configuration")
    with reporter.suspended():
        answered = click.confirm("Create .env from the template")
    reporter.hand_off()
    reporter.start_rendering()

    assert answered is True
    assert "Create .env from the template [y/N]" in piped.getvalue()
    assert visible_lines(buffer.getvalue()) == ["→ Preparing configuration"]
    assert _ANSI.search(buffer.getvalue()) is None
    assert monitor_threads() == []


def _explode(*args, **kwargs):
    """A repaint that fails, for the degrade paths."""
    raise RuntimeError("render blew up")


class _PipedStdout(io.StringIO):
    """Stdout redirected to a file or a pipe: writable, but not a terminal."""

    def isatty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Which reporter the install site picks
# ---------------------------------------------------------------------------


def test_a_terminal_gets_the_live_reporter(monkeypatch):
    monkeypatch.setattr(cli_main.sys, "stdout", _FakeStdout(tty=True))

    assert isinstance(cli_main._tty_aware_reporter(), LiveReporter)


def test_a_pipe_gets_the_plain_reporter(monkeypatch):
    """Stdout piped while stdin is still a terminal (``osprey up | tee``) is
    the case that must NOT mount a region."""
    monkeypatch.setattr(cli_main.sys, "stdout", _FakeStdout(tty=False))

    reporter = cli_main._tty_aware_reporter()

    assert type(reporter) is PhaseReporter


def test_the_install_site_starts_the_region_it_installed(monkeypatch):
    """The region is mounted once, by the verb that owns the reporter."""
    started = []

    class _Spy(PhaseReporter):
        def start_rendering(self) -> None:
            started.append(current_reporter() is self)

    monkeypatch.setattr(cli_main, "_tty_aware_reporter", lambda: _Spy(color=False))

    with cli_main.lifecycle_reporter():
        pass

    assert started == [True]


class _FakeStdout:
    """Just enough stdout for the install site's one question."""

    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


# ---------------------------------------------------------------------------
# Log records while a region is mounted
# ---------------------------------------------------------------------------


@pytest.fixture
def log_handler():
    """A ``RichHandler`` on the ROOT logger, writing to a stream of its own.

    The shape ``configure_logging()`` installs, minus the parts that would make
    the assertions about wrapping: one handler, one console, bound to a stream
    that is emphatically NOT the reporter's -- which is the whole reason a record
    can garble the region in the first place.

    Removed again afterwards, level included: a handler left on the root logger
    prints every later test's log records, and a swapped console left on it would
    outlive the buffer it points at.
    """
    stream = io.StringIO()
    own_console = Console(file=stream, force_terminal=True, width=100)
    handler = RichHandler(
        console=own_console,
        show_time=False,
        show_path=False,
        markup=True,
    )
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        # The handler's OWN console is yielded rather than read back off it: by
        # the time a test runs the region may already have borrowed it, and
        # "restored" asserted against whatever is installed at that point would
        # be asserting the borrow against itself.
        yield handler, stream, own_console
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


@pytest.fixture
def live_over_logging(log_handler):
    """A live reporter mounted AFTER the log handler is on the root logger.

    The ordering is the point, and is why these tests do not reuse ``live``:
    the borrow happens when the region mounts, so a handler registered after
    that is never borrowed and every assertion below would pass vacuously.
    """
    console, buffer = recording_console()
    reporter = LiveReporter(console=console)
    reporter.color = False
    reporter.start_rendering()
    try:
        yield reporter, buffer, console
    finally:
        reporter.stop_rendering()


def test_a_log_record_lands_as_one_intact_line_above_the_region(live_over_logging, log_handler):
    """The symptom the swap exists for: a record written straight to stderr
    lands in the middle of a repainting region and the two interleave.

    Asserted on the reporter's buffer rather than on the absence of garbling,
    because "not garbled" is only visible on a real terminal -- here the proof
    is that the record went through the console the ``Live`` is mounted on, in
    one piece, with its own stream left empty.
    """
    reporter, buffer, _console = live_over_logging
    _handler, stderr, _own = log_handler
    reporter.phase("Building services")

    with reporter.watch_build(build_with_rows()):
        reporter.refresh_live()
        get_logger("deploy").key_info("pulling base image")
        reporter.refresh_live()

    written = buffer.getvalue()
    # Rich pads a log line out to the console width, so the trailing run of
    # spaces is the renderer's and not part of the record.
    assert [line.rstrip() for line in visible_lines(written) if "pulling base image" in line] == [
        "INFO     pulling base image"
    ]
    assert "pulling base image" not in stderr.getvalue()


def test_the_plain_reporter_leaves_the_log_handler_on_stderr(monkeypatch, log_handler):
    """Off a terminal the handler keeps the stderr it documents.

    Piped and CI consumers read records on stderr and program output on stdout,
    and there is no region off a TTY for a record to garble -- so the plain
    reporter has nothing to borrow for and must not touch the handler at all.
    """
    handler, stderr, original = log_handler
    console, buffer = recording_console()
    monkeypatch.setattr(phase_reporter, "console", console)

    reporter = PhaseReporter(color=False)
    reporter.start_rendering()
    reporter.phase("Building services")
    get_logger("deploy").key_info("pulling base image")
    reporter.stop_rendering()

    assert handler.console is original
    assert "pulling base image" in stderr.getvalue()
    assert "pulling base image" not in buffer.getvalue()


def test_the_handler_gets_its_own_console_back_on_uninstall(log_handler):
    """Restored by IDENTITY, through the swap every caller has in common.

    The handler's console is configured by whoever installed it -- stderr,
    ``force_terminal``, a fixed width -- so an equivalent rebuilt from the same
    arguments would not be giving it back. The borrowed one points at a buffer
    that dies with the verb, and a run that drew a region must leave the process
    logging exactly as it found it.
    """
    handler, _stderr, original = log_handler
    console, _buffer = recording_console()
    reporter = LiveReporter(console=console)
    reporter.color = False

    previous = install_reporter(reporter)
    reporter.start_rendering()
    borrowed = handler.console

    install_reporter(previous)

    assert borrowed is console
    assert handler.console is original


def test_the_handler_gets_its_own_console_back_on_hand_off(live_over_logging, log_handler):
    """``os.execvpe`` is a moment away: the region is gone and so is the borrow.

    A handler still pointing at a console whose ``Live`` no longer exists is the
    one state that outlives this process's ownership of the terminal.
    """
    reporter, _buffer, console = live_over_logging
    handler, _stderr, original = log_handler
    reporter.phase("Starting osprey-project")
    assert handler.console is console

    reporter.hand_off()

    assert handler.console is original


def test_the_handler_is_restored_when_the_phase_raises(live_over_logging, log_handler):
    """The failure path reaches teardown through ``Phase.fail``, not through a
    tidy return -- and that is the path an operator actually hits."""
    reporter, _buffer, console = live_over_logging
    handler, _stderr, original = log_handler
    assert original is not console

    with pytest.raises(RuntimeError, match="compose exploded"):
        with reporter.phase("Building services"):
            assert handler.console is console
            raise RuntimeError("compose exploded")

    assert handler.console is original


def test_a_root_logger_with_no_rich_handler_is_left_alone(monkeypatch):
    """Nothing to borrow is normal, not an error.

    A library caller or a test may never have called ``configure_logging()``,
    and then no record is heading for stderr for a repaint to collide with. A
    reporter that insisted on finding a handler would turn the region -- pure
    decoration -- into a reason a deploy cannot start.
    """
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])
    console, buffer = recording_console()
    reporter = LiveReporter(console=console)
    reporter.color = False

    reporter.start_rendering()
    try:
        assert reporter._borrowed_log_consoles == []
        reporter.phase("Building services")
        reporter.refresh_live()
    finally:
        reporter.stop_rendering()

    assert "→ Building services" in visible_lines(buffer.getvalue())
