"""What an operator actually sees, read off a real terminal.

Every other test in this repository reads the CLI's output through a capture
object: a ``CliRunner``, a recording console, a probe handler. Each of those is
a stand-in for a terminal, and each is right about a different half of it. None
of them can answer the question this feature is about — whether a deploy that
paints a live region over its own output leaves a screen a human can read.

So this module asks the terminal. Each scenario allocates a pty with
:func:`pty.openpty`, starts a real ``osprey`` verb on it as a subprocess against
the exemplar repo and the stub runtime, and keeps every byte the process wrote
to it, escape sequences included. The bytes are then replayed through
:class:`Terminal` — a small emulator that understands exactly the control
vocabulary Rich's ``Live`` uses — and the scenarios assert on the screen that
comes out, not on the byte stream.

**Garbling, operationally.** "The final screen is not garbled" is a judgement a
test cannot make, so it is spelled here as four mechanical properties, checked
together by :func:`assert_screen_is_intact`:

1. *Every escape sequence in the stream is whole and understood.* The emulator
   records any ``ESC`` that does not begin a complete sequence it models, and
   any complete sequence outside that vocabulary. Interleaving — a log record
   written into the middle of a repaint — shows up here first, because it
   splits one sequence across another's bytes.
2. *No region transient survives the teardown.* The region is ``transient``, so
   once it is down no spinner frame and no table rule may remain on the screen.
   A leftover row is a repaint the teardown did not take back.
3. *No line is written twice.* A region that repaints over scrollback instead of
   over itself duplicates the line under it; a permanent line may appear once.
4. *The cursor is given back.* ``Live`` hides it on mount; a run that ends
   without the matching show has left the operator's terminal broken.

**Frame polling, never sleeps.** Nothing here waits a fixed time for the screen
to reach a state. The reader loop polls the accumulated stream against a
predicate with a generous deadline (:data:`RUN_TIMEOUT`), which is what makes
the scenarios deterministic rather than merely usually-passing: a slow CI box
takes longer, not a different path. The one thing that must be normalised out
of any two-frame comparison is the spinner glyph — it advances on an 80 ms
cadence and is the only wall-clock character the region draws.

**Scenario 7 is the one an operator can see go wrong.** ``output.warn`` and
``output.fail`` carry a stderr contract, and a stderr console escapes ``Live``'s
redirection entirely — so a warning issued while a region is mounted used to be
written at whatever column the region had left the cursor on, and stayed there
with a dead spinner frame welded to its front. The renderer now hands those
lines to the region's own console for as long as one is up, and scenario 7 pins
that: the warning lands above the region, on its own row, in one piece.

POSIX only. ``pty`` is a POSIX module and the whole module skips elsewhere.
"""

from __future__ import annotations

import os
import re
import select
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.pty,
    pytest.mark.skipif(os.name != "posix", reason="a pty is a POSIX facility"),
]

# Guarded together, and for the same reason: all three are POSIX-only, and an
# import error is raised at *collection*, which is before any skipif can apply.
# An unguarded `import fcntl` would turn "this suite does not run on Windows"
# into "the whole test session fails to collect on Windows".
if os.name == "posix":  # pragma: no branch - the skip above covers the other side
    import fcntl
    import pty
    import termios

#: Terminal geometry every scenario runs in. Fixed so the screen a scenario
#: asserts on is the same one everywhere: Rich takes its width from the pty's
#: window size, and a developer's 210-column terminal would otherwise wrap
#: different lines than CI's 80.
TERMINAL_COLUMNS = 100
TERMINAL_ROWS = 40

#: How long any one scenario's process may take. Deliberately far above what
#: the work costs (the slowest scenario is ~9 s, and that is a timeout it is
#: *asserting* about): this is the harness giving up, and a generous ceiling is
#: what keeps a loaded CI box from turning latency into a failure.
RUN_TIMEOUT = 120.0

#: How long :meth:`PtyProcess.wait_for` gives one screen state to appear.
POLL_TIMEOUT = 60.0

#: The bootstrap every scenario's subprocess runs. ``-c`` rather than the
#: console script, so the run always uses the interpreter — and therefore the
#: checkout — the tests are running from.
CLI_BOOTSTRAP = "import sys; from osprey.cli.main import cli; sys.exit(cli())"

#: Criterion 1's own shape: a ``RichHandler`` line at INFO with the timestamp
#: column filled in.
INFO_LINE = re.compile(r"\[\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\]\s+INFO")

#: The same record when it is not the first of its second — ``RichHandler``
#: leaves the timestamp column blank and prints the level alone. Criterion 1's
#: regex alone would miss every INFO record but the first, so both are checked.
INFO_LEVEL_COLUMN = re.compile(r"(?:^|\s)INFO {2,}\S")

#: Rich's ``dots`` spinner, the region's only wall-clock character.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: Box-drawing characters a rendered table rule is made of. None of the CLI's
#: permanent lines use them, so one on the final screen is a region transient
#: that outlived its region.
BOX_DRAWING = "─│┌┐└┘├┤┬┴┼━┃╭╮╯╰"

# ---------------------------------------------------------------------------
# the terminal
# ---------------------------------------------------------------------------


class Terminal:
    """A screen, replayed from what a process wrote to a pty.

    Deliberately small. It models the control sequences Rich's ``Live`` and
    ``Console`` actually emit — carriage return, line feed, erase-in-line,
    erase-in-display, relative cursor motion, SGR, and cursor visibility — and
    records anything else in :attr:`unsupported` rather than ignoring it. That
    record is half of the garbling check: a stream this emulator cannot fully
    parse is, by definition, a stream a terminal would have rendered as
    something other than what the program meant.

    Lines grow without bound: this is scrollback, not a viewport. A region is
    drawn at the bottom of it and erased from the same place, which is exactly
    how the real thing behaves for a region shorter than the screen.
    """

    #: One complete CSI sequence: ``ESC [`` parameters, final byte.
    _CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")

    def __init__(self) -> None:
        self.lines: list[str] = [""]
        self.row = 0
        self.col = 0
        self.cursor_hidden = False
        #: Every ``ESC`` this emulator could not account for, as a repr of the
        #: bytes around it. Non-empty means the screen below is a guess.
        self.unsupported: list[str] = []
        #: Cursor-visibility transitions, in order: ``"hide"`` / ``"show"``.
        self.cursor_events: list[str] = []

    # -- feeding ------------------------------------------------------------

    def feed(self, data: str) -> None:
        """Replay *data* onto the screen."""
        index = 0
        while index < len(data):
            char = data[index]
            if char == "\x1b":
                index = self._escape(data, index)
                continue
            if char == "\r":
                self.col = 0
            elif char == "\n":
                self._newline()
            elif char == "\b":
                self.col = max(0, self.col - 1)
            elif char == "\t":
                self._write(" " * (8 - self.col % 8))
            elif char < " ":
                # Any other C0 control would move the cursor in a way this
                # emulator does not model; record it rather than drop it.
                self.unsupported.append(repr(char))
            else:
                self._write(char)
            index += 1

    def _escape(self, data: str, index: int) -> int:
        """Consume one escape sequence starting at *index*; return the next index."""
        match = self._CSI.match(data, index)
        if match is None:
            self.unsupported.append(repr(data[index : index + 12]))
            return index + 1
        params, final = match.group(1), match.group(2)
        self._csi(params, final, match.group(0))
        return match.end()

    def _csi(self, params: str, final: str, whole: str) -> None:
        """Apply one parsed CSI sequence."""
        if final == "m":  # SGR: styling only, no effect on the text
            return
        if params.startswith("?"):
            if params == "?25" and final in "hl":
                self.cursor_hidden = final == "l"
                self.cursor_events.append("hide" if final == "l" else "show")
                return
            self.unsupported.append(repr(whole))
            return
        count = int(params) if params.isdigit() else (0 if params == "" else -1)
        if count < 0:
            self.unsupported.append(repr(whole))
            return
        if final == "A":
            self.row = max(0, self.row - max(1, count))
        elif final == "B":
            for _ in range(max(1, count)):
                self._newline(carriage_return=False)
        elif final == "C":
            self.col += max(1, count)
        elif final == "D":
            self.col = max(0, self.col - max(1, count))
        elif final == "K":
            self._erase_line(count)
        elif final == "J":
            self._erase_display(count)
        else:
            self.unsupported.append(repr(whole))

    # -- primitives ---------------------------------------------------------

    def _newline(self, *, carriage_return: bool = False) -> None:
        self.row += 1
        while len(self.lines) <= self.row:
            self.lines.append("")
        if carriage_return:
            self.col = 0

    def _write(self, text: str) -> None:
        line = self.lines[self.row]
        if len(line) < self.col:
            line += " " * (self.col - len(line))
        self.lines[self.row] = line[: self.col] + text + line[self.col + len(text) :]
        self.col += len(text)

    def _erase_line(self, mode: int) -> None:
        line = self.lines[self.row]
        if mode == 0:
            self.lines[self.row] = line[: self.col]
        elif mode == 1:
            self.lines[self.row] = " " * min(self.col, len(line)) + line[self.col :]
        else:
            self.lines[self.row] = ""

    def _erase_display(self, mode: int) -> None:
        if mode in (0, 1):
            self.lines[self.row] = self.lines[self.row][: self.col]
            del self.lines[self.row + 1 :]
        else:
            self.lines = [""]
            self.row = self.col = 0

    # -- reading it back ----------------------------------------------------

    def scrollback(self) -> list[str]:
        """The screen as lines, right-stripped, with trailing blanks dropped.

        Right-stripped because a repaint pads to the region's width and an
        operator does not see the padding; trailing blanks dropped because the
        teardown leaves the erased region's row behind as an empty one.
        """
        lines = [line.rstrip() for line in self.lines]
        while lines and not lines[-1]:
            lines.pop()
        return lines


def strip_ansi(text: str) -> str:
    """*text* with every escape sequence removed, and nothing else changed."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b.", "", text)


def without_spinner_frames(text: str) -> str:
    """*text* with every spinner glyph normalised to a single placeholder.

    Two paints of the same region differ by exactly this one character (plus
    the ticker, which only moves once a second). Comparing paints without
    normalising it compares the wall clock.
    """
    return re.sub(f"[{SPINNER_FRAMES}]", "*", text)


# ---------------------------------------------------------------------------
# running a verb on one
# ---------------------------------------------------------------------------


@dataclass
class PtyRun:
    """One verb, run on a real terminal, and everything it wrote there."""

    argv: list[str]
    exit_code: int
    raw: bytes
    #: The screen after the last byte, as scrollback lines.
    screen: list[str] = field(default_factory=list)
    #: Everything ever written, escapes stripped — including region frames the
    #: teardown has since erased. This is the *stream*, not the screen.
    stream: str = ""
    #: The undecodable or unmodelled escapes the emulator met, if any.
    unsupported: list[str] = field(default_factory=list)
    #: Cursor hide/show transitions, in order.
    cursor_events: list[str] = field(default_factory=list)

    @property
    def screen_text(self) -> str:
        """The final screen as one string."""
        return "\n".join(self.screen)

    def describe(self) -> str:
        """A failure message worth reading: the exit code and the final screen."""
        return f"argv={self.argv} exit={self.exit_code}\n--- final screen ---\n" + self.screen_text


class PtyProcess:
    """A verb running on a pty, readable while it runs.

    Held open by :func:`run_on_pty` so a scenario can wait for something to
    appear on the screen and then type at it — the interactive shape a
    ``CliRunner`` cannot reproduce, because the prompt and the region are
    competing for the same terminal.
    """

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str], bootstrap: str) -> None:
        self.argv = argv
        self._master, slave = pty.openpty()
        fcntl.ioctl(
            slave, termios.TIOCSWINSZ, struct.pack("HHHH", TERMINAL_ROWS, TERMINAL_COLUMNS, 0, 0)
        )
        self._buffer = bytearray()
        self._master_closed = False
        self._process = subprocess.Popen(  # noqa: S603 - argv is built here, not by input
            [sys.executable, "-c", bootstrap, *argv],
            cwd=str(cwd),
            env=env,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            # Its own session, so the pty is the child's controlling terminal
            # and a signal aimed at the test runner's group cannot reach it.
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave)

    # -- reading ------------------------------------------------------------

    def _pump(self, timeout: float) -> bool:
        """Read whatever is available within *timeout*; False at end of stream."""
        try:
            ready, _, _ = select.select([self._master], [], [], timeout)
        except OSError:
            return False
        if not ready:
            return True
        try:
            chunk = os.read(self._master, 65536)
        except OSError:
            # Every slave descriptor is closed: on Linux that is EOF, on macOS
            # it is EIO. Both mean the same thing here.
            return False
        if not chunk:
            return False
        self._buffer.extend(chunk)
        return True

    def text(self) -> str:
        """Everything written so far, escapes stripped."""
        return strip_ansi(self._buffer.decode("utf-8", "replace"))

    def wait_for(self, pattern: str, *, timeout: float = POLL_TIMEOUT) -> None:
        """Poll the stream until *pattern* appears, or fail saying what did.

        The harness's whole substitute for sleeping. A scenario that needs the
        screen to have reached a state waits for the state, so a slow machine
        costs seconds rather than a red test.
        """
        deadline = time.monotonic() + timeout
        compiled = re.compile(pattern)
        while time.monotonic() < deadline:
            if compiled.search(self.text()):
                return
            if not self._pump(0.05) and not compiled.search(self.text()):
                break
        pytest.fail(
            f"{pattern!r} never appeared on the terminal within {timeout:.0f}s.\n"
            f"--- stream so far ---\n{self.text()}"
        )

    def send(self, data: str) -> None:
        """Type *data* at the terminal, as an operator's keyboard would."""
        os.write(self._master, data.encode("utf-8"))

    # -- ending -------------------------------------------------------------

    def _close_master(self) -> None:
        """Release the pty master, at most once."""
        if not self._master_closed:
            self._master_closed = True
            os.close(self._master)

    def abandon(self) -> None:
        """Kill the child and release the pty, for a run that will not finish.

        Every path out of a scenario that is not :meth:`finish` comes through
        here. It matters more than it looks: the child is started in its own
        session, so nothing aimed at the test runner's process group reaches
        it, and it would outlive not just the test but the whole pytest run.
        Scenario 6 is the sharp case — an ``osprey up -d`` waiting at a prompt
        that no one is going to answer would sit there indefinitely, holding
        its repo copy, its pty and its stub log open.
        """
        try:
            self._process.kill()
        except OSError:  # pragma: no cover - already reaped
            pass
        else:
            self._process.wait()
        self._close_master()

    def finish(self, *, timeout: float = RUN_TIMEOUT) -> PtyRun:
        """Read to end of stream, reap the process, and build the screen."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._pump(0.1):
                break
        else:
            self.abandon()
            pytest.fail(
                f"{self.argv} did not finish within {timeout:.0f}s.\n"
                f"--- stream so far ---\n{self.text()}"
            )
        self._close_master()
        try:
            exit_code = self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            self._process.kill()
            exit_code = self._process.wait()

        terminal = Terminal()
        terminal.feed(bytes(self._buffer).decode("utf-8", "replace"))
        return PtyRun(
            argv=self.argv,
            exit_code=exit_code,
            raw=bytes(self._buffer),
            screen=terminal.scrollback(),
            stream=strip_ansi(bytes(self._buffer).decode("utf-8", "replace")),
            unsupported=terminal.unsupported,
            cursor_events=terminal.cursor_events,
        )


def run_on_pty(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    bootstrap: str = CLI_BOOTSTRAP,
    interact: Callable[[PtyProcess], None] | None = None,
) -> PtyRun:
    """Run one verb on a fresh pty and return everything it painted.

    Args:
        argv: Arguments after ``osprey``.
        cwd: Directory to run in — a writable repo copy, never the exemplar.
        env: The child's whole environment, stub runtime included.
        bootstrap: The ``python -c`` program that starts the CLI. Scenarios that
            need to reach inside the process (the degraded-region one) pass
            their own, which is the only cross-process seam a subprocess has.
        interact: Called with the live process once it is started, for scenarios
            that answer a prompt. If it raises — a ``pytest.fail`` out of
            ``wait_for``, an ``OSError`` out of ``send`` — the child is killed
            and the pty released before the failure propagates. ``BaseException``
            rather than ``Exception`` on purpose: ``pytest.fail`` raises one, and
            catching the narrower class would leak the child on exactly the path
            most likely to be taken.
    """
    process = PtyProcess(list(argv), cwd, env, bootstrap)
    if interact is not None:
        try:
            interact(process)
        except BaseException:
            process.abandon()
            raise
    return process.finish()


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------


def assert_screen_is_intact(run: PtyRun) -> None:
    """The four mechanical properties of an ungarbled screen (see module docs).

    Held by every scenario without exception. There was one until the renderer
    stopped writing trouble past a mounted region — scenario 7's warnings came
    back with a spinner frame welded on — and the exemption that made room for
    it is gone with the defect, deliberately: an escape hatch nothing needs is a
    hatch the next damaged line slips through.
    """
    assert not run.unsupported, (
        f"the terminal stream contains escape bytes this emulator could not parse as whole "
        f"sequences, which is what an interleaved write looks like from outside: "
        f"{run.unsupported[:5]}\n{run.describe()}"
    )

    leftovers = [
        line
        for line in run.screen
        if any(glyph in line for glyph in SPINNER_FRAMES)
        or any(glyph in line for glyph in BOX_DRAWING)
    ]
    assert not leftovers, f"region transients survived the teardown: {leftovers}\n{run.describe()}"

    seen: dict[str, int] = {}
    for line in run.screen:
        if line.strip():
            seen[line] = seen.get(line, 0) + 1
    repeated = {line: count for line, count in seen.items() if count > 1}
    assert not repeated, (
        f"lines appear more than once on the final screen, which is what a repaint over "
        f"scrollback leaves behind: {repeated}\n{run.describe()}"
    )

    assert run.cursor_events, f"no cursor visibility control at all\n{run.describe()}"
    assert run.cursor_events[-1] == "show", (
        f"the run ended with the cursor still hidden ({run.cursor_events}); the operator's "
        f"terminal is left broken\n{run.describe()}"
    )


def info_shaped_lines(run: PtyRun) -> list[str]:
    """Every line on the terminal that reads as a rendered INFO record.

    Read off the whole *stream*, not the final screen: a record that flashed
    inside a region and was erased by the next repaint was still on the
    operator's terminal.
    """
    return [
        line
        for line in run.stream.splitlines()
        if INFO_LINE.search(line) or INFO_LEVEL_COLUMN.search(line)
    ]


def assert_no_info_lines(run: PtyRun) -> None:
    """Criterion 1: nothing INFO-shaped anywhere the operator could have seen it."""
    offenders = info_shaped_lines(run)
    assert not offenders, f"INFO-shaped lines in the default view: {offenders}\n{run.describe()}"


def assert_region_was_mounted_when(run: PtyRun, needle: str) -> None:
    """*needle* was written while the region was up, not before or after it.

    Asked of the raw bytes rather than of the screen, because the screen is
    what is left *after* the teardown and cannot say when a line arrived. A
    region is up exactly while a cursor-hide has no matching show, so the test
    is: count both in everything written before the needle.
    """
    text = run.raw.decode("utf-8", "replace")
    at = text.find(needle)
    assert at >= 0, f"{needle!r} is nowhere in the stream\n{run.describe()}"
    before = text[:at]
    assert before.count("\x1b[?25l") > before.count("\x1b[?25h"), (
        f"{needle!r} was written with no region mounted, so this scenario is not about "
        f"what it says it is about\n{run.describe()}"
    )


def region_frames(run: PtyRun) -> list[str]:
    """Every write that carried a spinner glyph — one per repaint.

    A repaint always starts with a carriage return (Rich returns to column
    zero, erases, and draws), so splitting the stream on ``\\r`` cuts it into
    candidate frames. The line feed a preceding permanent line ended with rides
    along on the front of the frame that follows it and is stripped: it belongs
    to the line above, not to the picture.
    """
    return [
        segment.strip("\n")
        for segment in run.stream.split("\r")
        if any(glyph in segment for glyph in SPINNER_FRAMES)
    ]


def assert_region_was_live(run: PtyRun) -> None:
    """The region mounted, repainted at least twice, and came down.

    "Repainted" is asserted through the spinner *advancing*: one frame drawn
    twice would be a static region, and the monitor thread is what this proves
    is running.
    """
    assert run.cursor_events[:1] == ["hide"], (
        f"the region never mounted (no cursor hide)\n{run.describe()}"
    )
    frames = region_frames(run)
    assert len(frames) >= 2, f"the region painted {len(frames)} frame(s)\n{run.describe()}"
    glyphs = {glyph for frame in frames for glyph in SPINNER_FRAMES if glyph in frame}
    assert len(glyphs) >= 2, (
        f"the region painted {len(frames)} frames but the spinner never advanced "
        f"({glyphs}), so nothing was repainting it\n{run.describe()}"
    )
    assert run.cursor_events[-1] == "show", f"the region never came down\n{run.describe()}"


# ---------------------------------------------------------------------------
# scenario 0 — the harness cleans up after itself
# ---------------------------------------------------------------------------


def test_the_harness_kills_a_child_it_gave_up_on(tmp_path: Path, pty_env: dict[str, str]) -> None:
    """A scenario that fails mid-interaction must not leave its verb running.

    This is about the harness rather than the CLI, and it is here because the
    failure mode is invisible from inside a test run: children are started in
    their own session, so nothing aimed at the runner's process group reaches
    them, and one left behind by a failed interactive scenario would outlive
    the whole pytest session holding a pty and a repo copy open. A developer
    would meet it later as a machine with stray ``osprey up`` processes on it,
    with nothing to connect them to the test that failed an hour earlier.

    Driven with a bare sleeping program rather than a verb: what is under test
    is the harness's cleanup path, and a real deploy would only make the same
    assertion slower and less certain.
    """
    captured: list[PtyProcess] = []

    def give_up(process: PtyProcess) -> None:
        captured.append(process)
        raise RuntimeError("this scenario gave up")

    with pytest.raises(RuntimeError, match="gave up"):
        run_on_pty(
            [],
            cwd=tmp_path,
            env=pty_env,
            bootstrap="import time; time.sleep(600)",
            interact=give_up,
        )

    assert captured, "the interaction never ran, so nothing was under test"
    process = captured[0]
    assert process._process.poll() is not None, (
        "the child outlived the scenario that gave up on it, and nothing in this "
        "session will ever reap it"
    )
    assert process._master_closed, "the pty master was leaked"
    with pytest.raises(OSError):
        os.read(process._master, 1)


# ---------------------------------------------------------------------------
# scenario 1 — the default view
# ---------------------------------------------------------------------------


def test_the_scripted_deploy_shows_no_info_shaped_lines(
    startable_repo: Path, pty_env: dict[str, str]
) -> None:
    """Criterion 1, on the terminal it is written about.

    The scripted deploy is ``osprey up -d`` against the stub runtime: it runs
    the real start path end to end, and every fact it decides to tell the
    operator is a printed line rather than a log record. Nothing INFO-shaped
    may reach the screen.
    """
    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env)

    assert run.exit_code == 0, run.describe()
    assert_no_info_lines(run)
    assert "✓ Preflight" in run.screen_text, run.describe()
    assert "running" in run.screen_text, run.describe()


def test_verbose_puts_the_transcript_back_on_the_same_terminal(
    startable_repo: Path, pty_env: dict[str, str]
) -> None:
    """The armed witness for the scenario above.

    Zero INFO lines is only evidence of a gate if the same harness, on the same
    verb, can see INFO lines when the gate is lifted. ``osprey -v up -d`` is
    that run: the gate is off, and the records the default view withheld are
    on the screen in the shape the regex is looking for.
    """
    run = run_on_pty(["-v", "up", "-d"], cwd=startable_repo, env=pty_env)

    assert run.exit_code == 0, run.describe()
    assert info_shaped_lines(run), (
        f"the harness saw no INFO-shaped line even with the gate lifted, so its absence "
        f"in the default view proves nothing\n{run.describe()}"
    )


# ---------------------------------------------------------------------------
# scenario 2 — the region's own life
# ---------------------------------------------------------------------------


def test_the_region_mounts_repaints_and_tears_down_clean(
    startable_repo: Path, pty_env: dict[str, str], unhurried_runtime: None
) -> None:
    """Mount, repaint, teardown — and a screen with nothing of the region left."""
    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env)

    assert run.exit_code == 0, run.describe()
    assert_region_was_live(run)
    assert_screen_is_intact(run)


def test_two_paints_of_the_region_differ_only_by_the_clock(
    startable_repo: Path, pty_env: dict[str, str], unhurried_runtime: None
) -> None:
    """Consecutive frames are the same picture, redrawn — not two pictures.

    With the spinner normalised out, two adjacent frames of the same open phase
    must be identical apart from the elapsed ticker. A frame that differs
    otherwise is one that painted something else into the region's row.
    """
    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env)

    assert run.exit_code == 0, run.describe()
    frames = [without_spinner_frames(frame).rstrip() for frame in region_frames(run)]
    assert len(frames) >= 2, run.describe()
    normalised = {re.sub(r"\d+m\d+s", "<elapsed>", frame) for frame in frames}
    assert len(normalised) == 1, (
        f"the region's frames are not repaints of one picture: {normalised}\n{run.describe()}"
    )


# ---------------------------------------------------------------------------
# scenario 3 — a record arriving while the region is up
# ---------------------------------------------------------------------------


def test_a_warning_lands_intact_above_a_mounted_region(
    startable_repo: Path, pty_env: dict[str, str], unhurried_runtime: None
) -> None:
    """A real warning from the start path, on a real terminal, above the region.

    Provoked rather than injected: the start path compares the shell's exports
    against the deployment's env chain and warns when an exported value
    disagrees with the pinned one for a variable the compose files interpolate.
    Seeding ``ZO_ROOT_USER_PASSWORD`` into the repo's ``.env`` and exporting a
    different value for the run is the operator mistake that produces it — and
    it produces it in the middle of the start phase, with the region mounted
    and the monitor thread repainting.

    The call site is promoted (:func:`osprey.cli.output.warn_fact`), so what
    must land is the renderer's trouble shape — marked summary, indented body —
    and NOT a ``RichHandler`` record: while the reporter owns the terminal the
    altitude gate keeps raw WARNING records off it, and this scenario is where
    that pair of facts is read off a real screen. Intact means: the block's
    lines are consecutive on the final screen with no region frame anywhere
    between them.
    """
    env_path = startable_repo / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8") + "ZO_ROOT_USER_PASSWORD=Chain5ide!Value\n",
        encoding="utf-8",
    )
    pty_env["ZO_ROOT_USER_PASSWORD"] = "Exp0rted!Value"

    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env)

    assert run.exit_code == 0, run.describe()
    assert_screen_is_intact(run)

    screen = run.screen
    heads = [
        index
        for index, line in enumerate(screen)
        if "shell export disagrees with this deployment's env chain" in line
    ]
    assert len(heads) == 1, f"expected exactly one promoted warning\n{run.describe()}"
    head = heads[0]
    assert screen[head].lstrip().startswith("⚠"), (
        f"the warning did not render as the renderer's warn shape: {screen[head]!r}\n"
        f"{run.describe()}"
    )
    # The summary names what diverged; the promotion must not have lost that.
    assert "ZO_ROOT_USER_PASSWORD" in screen[head], run.describe()
    assert_region_was_mounted_when(run, "shell export disagrees")

    # The gate's half of the contract: the record this warning also files must
    # NOT have painted as a RichHandler block — the promoted shape is the one
    # voice this warning gets on the terminal.
    assert not any(re.search(r"\bWARNING {2}\S", line) for line in screen), (
        f"a raw WARNING record reached the terminal past the altitude gate\n{run.describe()}"
    )

    # The block wraps at the terminal's width, so "intact" is about the block:
    # every line of it is consecutive and no region frame is drawn between any
    # two of its lines.
    body = screen[head : head + 6]
    assert all(not any(glyph in line for glyph in SPINNER_FRAMES) for line in body), (
        "a region frame is interleaved with the warning block:\n" + "\n".join(body)
    )

    # And it is above the region, not inside it: the phase the region was
    # drawing for closes after the block.
    closing = [index for index, line in enumerate(screen) if "✓ Starting" in line]
    assert closing and closing[0] > head, (
        f"the warning did not land above the open phase's region\n{run.describe()}"
    )


# ---------------------------------------------------------------------------
# scenario 4 — a step whose child outlives its timeout
# ---------------------------------------------------------------------------


#: A lifecycle step that leaves a grandchild holding its stdout, then hangs
#: until the timeout kills it. The grandchild writes long after the step has
#: been decided — which is the line that must never reach the terminal.
OVERRUNNING_STEP = """\
import subprocess
import sys
import time

subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import sys, time; time.sleep(8); "
        "sys.stdout.write('LATE-LINE-FROM-GRANDCHILD\\\\n'); sys.stdout.flush()",
    ]
)
sys.stdout.write("EARLY-LINE-FROM-STEP\\n")
sys.stdout.flush()
time.sleep(600)
"""

#: Appended to the copy's ``profile.yml``. ``stream: true`` is what puts the
#: step's own output on the terminal through the drain thread; ``timeout: 1``
#: is what makes it overrun.
OVERRUN_LIFECYCLE = """
lifecycle:
  post_build:
    - name: overrunning step
      run: {python} {script}
      timeout: 1
      stream: true
"""


def test_a_step_whose_child_overruns_its_timeout_never_reaches_the_screen(
    exemplar_copy: Path, pty_env: dict[str, str], tmp_path: Path
) -> None:
    """The stop-event fix, end to end on a terminal.

    A streamed lifecycle step spawns a grandchild that inherits its stdout and
    then hangs. The step's own timeout kills it; the grandchild survives,
    holding the pipe open with the drain thread still blocked in ``readline``,
    and writes a line seconds later — after the step has been reported failed.
    That line must be dropped, because by then the terminal belongs to whatever
    the CLI is drawing next.

    The step script lives outside the repo: the build warns about unrecognized
    top-level entries, and a stray file in the repo root would arm a different
    scenario's assertion.
    """
    script = tmp_path / "overrunning_step.py"
    script.write_text(OVERRUNNING_STEP, encoding="utf-8")
    profile = exemplar_copy / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8")
        + OVERRUN_LIFECYCLE.format(python=sys.executable, script=script),
        encoding="utf-8",
    )

    run = run_on_pty(["build", "--skip-deps"], cwd=exemplar_copy, env=pty_env)

    # The step failed, which is the point: it overran.
    assert run.exit_code != 0, run.describe()
    assert "EARLY-LINE-FROM-STEP" in run.stream, run.describe()
    assert "LATE-LINE-FROM-GRANDCHILD" not in run.stream, (
        f"a line from a step's orphaned grandchild reached the terminal after the step "
        f"was reported\n{run.describe()}"
    )
    assert "overrunning step' failed" in run.stream, run.describe()
    assert_no_info_lines(run)
    assert_screen_is_intact(run)


# ---------------------------------------------------------------------------
# scenario 5 — a repaint that raises
# ---------------------------------------------------------------------------


#: The degraded-region seam. A repaint failure has no product-level trigger —
#: it is Rich or the renderer going wrong — so the scenario reaches into the
#: subprocess the only way a subprocess can be reached: by choosing the program
#: that starts it. The patch is applied where ``PhaseReporter`` reads the name
#: (it imports it into its own namespace), before the CLI is imported at all,
#: and a marker file records that the injected failure really fired — without
#: it the scenario could pass by never having degraded anything.
DEGRADING_BOOTSTRAP = """
import os
import sys

from osprey.cli import phase_reporter

_real = phase_reporter.render_live_region
_paints = [0]


def _fail_after_the_second_paint(*args, **kwargs):
    _paints[0] += 1
    if _paints[0] > 2:
        with open(os.environ["OSPREY_PTY_DEGRADE_MARKER"], "w") as handle:
            handle.write(str(_paints[0]))
        raise RuntimeError("injected repaint failure")
    return _real(*args, **kwargs)


phase_reporter.render_live_region = _fail_after_the_second_paint

from osprey.cli.main import cli

sys.exit(cli())
"""


def test_a_repaint_that_raises_degrades_to_plain_lines(
    startable_repo: Path, pty_env: dict[str, str], tmp_path: Path, unhurried_runtime: None
) -> None:
    """The region dies; the deploy does not, and the screen stays readable.

    A repaint that raises takes the region down and leaves the run to finish as
    a plain-line run — decoration is the only thing an operator may lose. On a
    real terminal that means: the run still exits 0 and still says everything
    it was going to say, no half-drawn frame is left behind, and the cursor
    comes back even though nothing ever called the normal teardown.
    """
    marker = tmp_path / "degraded.marker"
    pty_env["OSPREY_PTY_DEGRADE_MARKER"] = str(marker)

    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env, bootstrap=DEGRADING_BOOTSTRAP)

    assert marker.is_file(), (
        f"the injected repaint failure never fired, so nothing degraded and this scenario "
        f"asserted nothing\n{run.describe()}"
    )
    assert run.exit_code == 0, run.describe()
    assert "✓ Preflight" in run.screen_text, run.describe()
    assert "running" in run.screen_text, run.describe()
    assert_no_info_lines(run)
    assert_screen_is_intact(run)


# ---------------------------------------------------------------------------
# scenario 6 — a prompt under a mounted region
# ---------------------------------------------------------------------------


def test_the_env_seed_prompt_is_readable_under_a_mounted_region(
    startable_repo: Path, pty_env: dict[str, str]
) -> None:
    """The suspended-prompt path, with a real keyboard answer.

    ``up`` refuses a repo with no ``.env``, and on an interactive terminal it
    offers to seed one from the shell first. That question is asked inside the
    Preflight phase, so the region is mounted and the monitor thread is
    repainting a spinner one line below where the operator is reading. The
    reporter suspends the region for the prompt; this asserts that it does,
    from the outside: the question is on the screen intact, the answer echoes
    after it, and no spinner is drawn over either.
    """
    (startable_repo / ".env").unlink()
    pty_env["ANTHROPIC_API_KEY"] = "sk-ant-exported-0000000000000000000000"

    def answer(process: PtyProcess) -> None:
        process.wait_for(r"Seed one from your shell")
        process.send("y\n")

    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env, interact=answer)

    assert run.exit_code == 0, run.describe()
    assert (startable_repo / ".env").is_file(), run.describe()

    asked = [line for line in run.screen if "Seed one from your shell" in line]
    assert len(asked) == 1, f"the prompt is not on the screen once\n{run.describe()}"
    assert "ANTHROPIC_API_KEY" in asked[0], run.describe()
    assert not any(glyph in asked[0] for glyph in SPINNER_FRAMES), (
        f"the region painted over the question\n{run.describe()}"
    )
    # The prompt's line carries the operator's own echoed answer, and the
    # deploy continues below it.
    assert asked[0].rstrip().endswith("y"), (
        f"the typed answer did not echo on the prompt line\n{run.describe()}"
    )
    assert_no_info_lines(run)
    assert_screen_is_intact(run)


# ---------------------------------------------------------------------------
# scenario 7 — warn()/fail() from inside a mounted phase
# ---------------------------------------------------------------------------


#: A validate-phase step that fails. The validate phase does not abort a build,
#: so the renderer's ``warn()`` runs with the region still mounted and the build
#: carries on to its summary card — which is what makes this the scenario that
#: characterizes the stderr-console bypass.
FAILING_VALIDATE_LIFECYCLE = """
lifecycle:
  validate:
    - name: failing check
      run: {python} -c "import sys; sys.stdout.write('CHECK-SAYS-NO\\n'); sys.exit(3)"
      timeout: 60
      stream: true
"""


def test_a_warn_from_inside_a_mounted_phase_does_not_take_the_screen_with_it(
    exemplar_copy: Path, pty_env: dict[str, str]
) -> None:
    """The known hazard, characterized on a real terminal.

    ``output.warn()`` and ``output.fail()`` print through ``styles.err_console``,
    and Rich unwraps its own ``FileProxy`` for a stderr console — so a mounted
    live region does not capture them and the line is written straight to the
    terminal, possibly mid-repaint. The region survives (it is transient and
    the next tick repaints it), but nothing places the line *above* the region
    the way a borrowed log console does.

    This pins what that actually looks like: a failing ``validate`` step, whose
    ``warn()`` lands while the build's region is mounted and the build then
    continues to its summary card. The assertion is the screen — the warning is
    on it, whole and on its own line, and the region left nothing behind.
    """
    profile = exemplar_copy / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8")
        + FAILING_VALIDATE_LIFECYCLE.format(python=sys.executable),
        encoding="utf-8",
    )

    run = run_on_pty(["build", "--skip-deps"], cwd=exemplar_copy, env=pty_env)

    assert run.exit_code == 0, run.describe()
    warned = [line for line in run.screen if "failing check' failed" in line]
    assert len(warned) == 1, f"the warning is not on the screen once\n{run.describe()}"
    assert warned[0].lstrip().startswith("⚠"), (
        f"the warning did not render as the renderer's warn shape\n{run.describe()}"
    )
    assert not any(glyph in warned[0] for glyph in SPINNER_FRAMES), (
        f"a region frame is interleaved with the warning line\n{run.describe()}"
    )
    assert "built" in run.screen_text, (
        f"the build did not finish after the warning\n{run.describe()}"
    )
    assert_region_was_mounted_when(run, "failing check' failed")
    assert_no_info_lines(run)
    assert_screen_is_intact(run)


#: The dangerous position, occupied on purpose. The scenario above is the one a
#: *shipped* call site can reach today, and it lands while the region is mounted
#: but drawing nothing, because the lifecycle phases run outside any open phase.
#: The position that used to weld is a ``warn()`` issued from inside an open
#: phase, and no shipped call site sits there yet. So this bootstrap puts one
#: there, at the worst instruction it could occupy: the return of ``Phase.step``,
#: one statement after the reporter has printed the sub-step line and the region
#: has redrawn itself under it. The cursor is parked at the end of that frame,
#: and a line written straight at the terminal would go on top of it.
#:
#: Synchronous, not a thread. A background burst reached the same instant only
#: for whichever warnings happened to fall right after a repaint — two of eight,
#: measured. Riding the step call reaches it on purpose, so the scenario asserts
#: a property of the code rather than of the scheduler.
HAZARD_BOOTSTRAP = """
import atexit
import pathlib
import sys

from osprey.cli import output, phase_reporter

_real_step = phase_reporter.Phase.step
_issued = []

atexit.register(
    lambda: pathlib.Path("probe-step-receipt").write_text(str(len(_issued)), encoding="utf-8")
)


def _step(self, name):
    _real_step(self, name)
    index = len(_issued)
    _issued.append(name)
    output.warn(f"probe warning {index}", f"detail {index}, one statement after a repaint")


phase_reporter.Phase.step = _step

from osprey.cli.main import cli

sys.exit(cli())
"""

#: The same injection point, handed straight to the reporter's console instead
#: of going through the renderer's trouble primitive. The control for the
#: scenario below: what differs between the two runs is the route the line
#: takes, and nothing else.
CONTROL_BOOTSTRAP = HAZARD_BOOTSTRAP.replace(
    'output.warn(f"probe warning {index}", f"detail {index}, one statement after a repaint")',
    'phase_reporter.current_reporter().out().print(f"probe warning {index}")',
)
assert CONTROL_BOOTSTRAP != HAZARD_BOOTSTRAP, (
    "the control's substitution missed, so it would run the hazard's own program and "
    "'proves the stream is what differs' would be measuring nothing"
)

#: The tail of the detail line :data:`HAZARD_BOOTSTRAP` prints under each
#: warning. Spelled here as well as inside the bootstrap so the placement
#: assertion can name it, with an import-time check that the two still agree —
#: a reworded bootstrap would otherwise leave the assertion looking for copy
#: nothing prints any more.
HAZARD_DETAIL_TAIL = "one statement after a repaint"
assert HAZARD_DETAIL_TAIL in HAZARD_BOOTSTRAP, (
    "the detail copy asserted on is not the copy the hazard bootstrap prints"
)

#: A burst spaced to straddle several of the monitor's 0.25 s repaints, for the
#: scenario that asks what concurrency alone does to the same line.
BURST_BOOTSTRAP = """
import sys
import threading
import time

from osprey.cli import output, phase_reporter

_real_step = phase_reporter.Phase.step
_started = []


def _burst():
    for index in range(8):
        output.warn(f"probe warning {index}", f"detail {index}, issued while the region drew")
        time.sleep(0.1)


def _step(self, name):
    _real_step(self, name)
    if not _started:
        _started.append(True)
        threading.Thread(target=_burst).start()


phase_reporter.Phase.step = _step

from osprey.cli.main import cli

sys.exit(cli())
"""

#: How many warnings :data:`BURST_BOOTSTRAP` issues. The thread is deliberately
#: NOT a daemon: a burst cut off by process exit would leave a half-written line
#: that looks exactly like the garbling these scenarios exist to detect, and the
#: harness must not manufacture its own evidence.
BURST_WARNINGS = 8


def probe_warning_rows(run: PtyRun) -> list[int]:
    """The screen row of every line carrying a probe warning's summary."""
    return [
        row
        for row, line in enumerate(run.screen)
        if re.search(r"probe warning \d+$", line.rstrip())
    ]


def probe_warning_lines(run: PtyRun) -> list[str]:
    """Every final-screen line carrying a probe warning's summary."""
    return [run.screen[row] for row in probe_warning_rows(run)]


def issued_step_count(repo: Path) -> int:
    """How many ``Phase.step`` calls the hazard bootstrap rode, from its receipt.

    The bootstraps that ride ``Phase.step`` issue exactly one warning per
    sub-step, so this is what their warning count is checked against — a
    scenario that hardcoded "two warnings" would go quietly vacuous the day the
    start path gains or loses a step. The count comes from the bootstrap's own
    receipt file rather than from counting ``·`` lines on the screen: promoted
    facts (:func:`osprey.cli.output.report_fact`) share that grammar on
    purpose, so the screen cannot tell a sub-step from a fact — and a proxy
    that miscounts would fail these scenarios for a reason that has nothing to
    do with how trouble is routed.
    """
    receipt = repo / "probe-step-receipt"
    assert receipt.exists(), (
        "the hazard bootstrap never wrote its step receipt, so the warning count "
        "has nothing sound to be checked against"
    )
    return int(receipt.read_text(encoding="utf-8"))


def test_a_warn_issued_from_inside_an_open_phase_lands_above_the_region(
    startable_repo: Path, pty_env: dict[str, str], unhurried_runtime: None
) -> None:
    """A warning from the worst instruction in the verb, read off the terminal.

    ``warn()`` and ``fail()`` carry a stderr contract, and Rich unwraps its own
    ``FileProxy`` for a stderr console — so a line written there while a region
    is mounted goes straight at the terminal, at whatever column the cursor was
    left at. One statement after a sub-step that column is the end of the frame
    the region has just redrawn, so the warning used to be printed onto the
    region's own row and left there: the region redraws a row lower and never
    takes that one back.

    The renderer now gives a mounted region the trouble line, the same borrow
    the log handler gets, and this is what that buys on a real terminal: every
    warning on its own row, its own first character first, its detail line
    intact underneath it, and — through the unexempted
    :func:`assert_screen_is_intact` — not one spinner frame left anywhere on the
    final screen.
    """
    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env, bootstrap=HAZARD_BOOTSTRAP)

    assert run.exit_code == 0, run.describe()
    warnings = probe_warning_lines(run)
    issued = issued_step_count(startable_repo)
    assert issued, f"the deploy ran no sub-step to ride\n{run.describe()}"
    assert len(warnings) == issued, (
        f"one warning per sub-step was issued but {len(warnings)} of {issued} reached the "
        f"screen\n{run.describe()}"
    )
    # The set asserted on below is pinned by an anchored pattern, so a warning
    # whose line came out a different shape would drop out of it rather than
    # fail it — the damage removing itself from the check. The two predicates
    # are made to agree here, on this run's own screen, before either is used.
    assert warnings == [line for line in run.screen if "probe warning" in line], (
        f"a line carries a probe warning but is not shaped like one, so it escaped the "
        f"assertions below\n{run.describe()}"
    )

    for index, line in enumerate(warnings):
        assert line.strip() == f"⚠ probe warning {index}", (
            f"the warning did not land on its own row in one piece: {line!r}. A region frame "
            f"on the front of it means trouble is being written past the region again\n"
            f"{run.describe()}"
        )

    # The detail belonging to each warning is on the row immediately below it,
    # in one piece. Without this the docstring's "its detail line lands under
    # it" would be a claim the test never checks.
    for index, row in enumerate(probe_warning_rows(run)):
        detail = run.screen[row + 1].strip()
        assert detail == f"detail {index}, {HAZARD_DETAIL_TAIL}", (
            f"the warning's detail line is not intact on the row under it: {detail!r}\n"
            f"{run.describe()}"
        )

    assert_region_was_mounted_when(run, "probe warning 0")
    assert_region_was_live(run)
    assert_screen_is_intact(run)


def test_control_the_same_line_through_the_reporter_console_never_welds(
    startable_repo: Path, pty_env: dict[str, str], unhurried_runtime: None
) -> None:
    """CONTROL for the scenario above: same instruction, straight at the console.

    Identical injection point, identical timing, but the line is handed to
    ``current_reporter().out()`` by the bootstrap rather than routed there by the
    renderer. That console is the one ``Live`` redirects, and it has always put a
    line above the region on its own row with no frame attached — which is what
    made it the specification the fix was written against.

    It stays here as the independent witness. If this run ever starts welding
    too, the cause is the pty, the emulator or the injection point rather than
    anything about how trouble is routed, and the scenario above should be read
    in that light.

    The copy differs — a reporter line has no ``⚠`` glyph and no detail line,
    because it is a console ``print`` rather than the renderer's trouble shape.
    That is why this is a control for *placement* and nothing else.
    """
    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env, bootstrap=CONTROL_BOOTSTRAP)

    assert run.exit_code == 0, run.describe()
    lines = probe_warning_lines(run)
    issued = issued_step_count(startable_repo)
    assert issued, f"the deploy ran no sub-step to ride\n{run.describe()}"
    assert len(lines) == issued, (
        f"the control must issue the same number of lines as the finding does, or the two "
        f"runs are not comparable: {len(lines)} of {issued}\n{run.describe()}"
    )
    for index, line in enumerate(lines):
        assert line.strip() == f"probe warning {index}", (
            f"a reporter-routed line did not land on its own row: {line!r}\n{run.describe()}"
        )
    assert_region_was_live(run)
    assert_screen_is_intact(run)


def test_a_burst_of_warnings_across_repaints_never_tears_a_line(
    startable_repo: Path, pty_env: dict[str, str], unhurried_runtime: None
) -> None:
    """What concurrency alone costs, once the routing is right.

    Eight warnings from their own thread at 0.1 s intervals, against a deploy
    slowed to a real runtime's latency, so the burst straddles several of the
    monitor's 0.25 s repaints. Nothing synchronises the two threads, so the
    question this answers is whether a warning can land in the middle of a
    repaint and split it — and whether the console the region lends out survives
    being written to from a thread that is not the one repainting it.

    It does. Each warning arrives whole, in order, on its own row, and the
    region survives every one of them. Two of the eight used to come back with a
    region frame welded on, run after run; none may now.
    """
    run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env, bootstrap=BURST_BOOTSTRAP)

    assert run.exit_code == 0, run.describe()
    warnings = probe_warning_lines(run)
    assert len(warnings) == BURST_WARNINGS, (
        f"expected {BURST_WARNINGS} probe warnings on the screen, found {len(warnings)}: "
        f"{warnings}\n{run.describe()}"
    )
    # Same reason as the scenario above: the count is pinned by an anchored
    # pattern, so a damaged line would leave the set instead of failing it. The
    # two predicates are made to agree first.
    assert warnings == [line for line in run.screen if "probe warning" in line], (
        f"a line carries a probe warning but is not shaped like one, so it escaped the "
        f"assertions below\n{run.describe()}"
    )
    for index, line in enumerate(warnings):
        assert line.strip() == f"⚠ probe warning {index}", (
            f"a warning issued mid-repaint did not land on its own row in one piece: "
            f"{line!r}\n{run.describe()}"
        )

    assert_region_was_mounted_when(run, "probe warning 0")
    assert_region_was_live(run)
    assert_screen_is_intact(run)
