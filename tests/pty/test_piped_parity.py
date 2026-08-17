"""The same deploy, on a terminal and down a pipe, says the same things.

A terminal gets one device and a redirect gets two files, and the CLI knows the
difference: on a terminal it mounts a live region, paints a spinner over its own
output and hands every stderr writer in the process to that region's console;
down a pipe it mounts nothing and writes plain lines to two separate streams.
Criterion 9 is the claim that none of that changes *what is said* — that the
decoration is decoration, and a redirect loses no line and gains none.

That claim is spelled here as an arithmetic one, which is the only form of it a
test can check without deciding for itself which lines "correspond":

    TTY scrollback  -  piped stderr  ==  piped stdout, in order

The left subtraction is over a multiset of lines: every line the piped run wrote
to stderr is struck off the terminal's scrollback once, and whatever is left has
to be the piped run's stdout, in the order it came out. Nothing is matched by
position, by prefix or by eye. A line that only a terminal prints leaves a
remainder stdout does not have; a line a redirect drops leaves stdout with one
the remainder does not; either way the assertion fails and the diff names it.

**The guard comes first.** The subtraction is only unambiguous if no single line
is written to both piped streams — otherwise striking off a stderr line could
strike a stdout line instead and the arithmetic would balance for the wrong
reason. So the first assertion is that the scripted deploy has no such line, and
a parity failure after it is therefore attributable to parity.

**Region transients are not filtered out, they are proven absent.** The region is
``transient``: it erases itself on teardown, so it is gone from the scrollback the
subtraction reads, and :func:`assert_screen_is_intact` is what establishes that on
this run rather than a filter that would hide a frame it failed to erase.

**Both runs are the same deploy, to the letter.** The two runs are made
comparable by construction rather than by normalisation wherever that is
possible: they run the same bootstrap program, against the same repository at the
same absolute path (restored from a pristine snapshot between them, so the second
run is not reading the first one's residue), on the same published ports, with
the same environment. Three things cannot be construction-equal and are
normalised, each for a measured reason:

* **The width.** Rich takes a pipe's width from ``COLUMNS`` and falls back to 80;
  a terminal's comes from the pty, which this harness fixes at
  :data:`~tests.pty.test_live_region_real_terminal.TERMINAL_COLUMNS`. The echo
  path itself never wraps, but anything rendered through a width-aware rich
  surface (a table, a record under ``-v``) would wrap differently without the
  pin — so the piped run is told the terminal's width, and the two runs wrap
  alike.
* **The clock.** A rendered log record is stamped with the wall clock and a
  phase line carries its own elapsed. Both are normalised to placeholders.
* **Sub-second laps.** A lap under 50 ms prints with no duration at all
  (``phase_reporter.Phase.step``), so against the instant stub the *presence* of a
  parenthetical — not just its value — would be a coin toss between the two runs.
  ``unhurried_runtime`` puts every runtime call an order of magnitude above that
  threshold, which is why it is not optional here. It is also what gives the
  terminal run a region worth comparing against: measured without it, the deploy
  finishes so fast that the spinner never advances once, and a parity assertion
  would then be comparing two runs that both printed plain lines.

**The stderr side is armed on purpose.** The scripted deploy is quiet on stderr,
and a subtraction with nothing to subtract would pass for no reason. So the run
provokes the env-chain disagreement warning the way scenario 3 does — a real
warning from the start path, not an injected line. Its call site is promoted
(:func:`osprey.cli.output.warn_fact`), so what lands on stderr is the renderer's
trouble block: a summary line, a wrapped multi-line detail and a remedy, which is
the strongest shape available — every one of those lines has to appear on the
terminal and be struck off, and any drift in how a redirect wraps or splits the
block shows up as a leftover.

POSIX only, like the rest of this directory.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.pty.conftest import StubRuntime

# The scenario runs on the sibling suite's harness rather than a second copy of
# it: its emulator, its pty runner and its terminal geometry. The fixtures both
# suites drive with -- ``startable_repo``, ``pty_env``, ``unhurried_runtime`` --
# live in ``conftest.py``, so pytest resolves them by name and neither module
# imports them.
from tests.pty.test_live_region_real_terminal import (
    CLI_BOOTSTRAP,
    SPINNER_FRAMES,
    TERMINAL_COLUMNS,
    TERMINAL_ROWS,
    assert_region_was_live,
    assert_screen_is_intact,
    run_on_pty,
    strip_ansi,
)

pytestmark = [
    pytest.mark.pty,
    pytest.mark.skipif(os.name != "posix", reason="a pty is a POSIX facility"),
]

#: The ``RichHandler`` timestamp column, which is wall-clock and therefore
#: differs between two runs seconds apart. Fixed width, so normalising it cannot
#: move where the record wraps.
TIMESTAMP = re.compile(r"\[\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\]")

#: What that column reads as once normalised. A rendered log record can still
#: reach stderr outside a reporter's tenure (and always under ``-v``), so the
#: normalisation keeps handling the stamp even though the armed warning no
#: longer carries one.
TIME_PLACEHOLDER = "[<time>]"

#: The first line of the armed warning's block. The call site is promoted
#: (:func:`osprey.cli.output.warn_fact`), so the shape on stderr is the
#: renderer's trouble block — ``⚠ summary`` over an indented body — rather than
#: a ``RichHandler`` record: while the reporter owns the run the altitude gate
#: keeps raw WARNING records off the terminal, and the promoted block is the
#: warning's one rendered voice. The anchor is the glyph and the phrase's first
#: words only — short enough that no width this suite runs at can wrap it —
#: because matching the whole head phrase line by line would report a wrapping
#: difference as a missing warning.
RECORD_HEAD = re.compile(r"^\s*⚠ shell export disagrees\b")

#: A phase line's or a sub-step's trailing duration, in both shapes
#: :func:`osprey.cli.phase_reporter.format_elapsed` produces.
ELAPSED = re.compile(r"\((?:\d+\.\d+s|\d+m\d{2}s)\)")

#: The variable the arming disagreement is about. Interpolated by the exemplar's
#: compose files, which is what makes the start path compare it at all.
ARMED_VARIABLE = "ZO_ROOT_USER_PASSWORD"

#: The value pinned in the repository's own chain, and the different one exported
#: into the run. Values are never printed by the warning, so neither is a secret
#: in any sense; they only have to disagree.
CHAIN_VALUE = "Chain5ide!Value"
EXPORTED_VALUE = "Exp0rted!Value"

#: The head of the record the disagreement produces. Asserted on the piped
#: stderr capture before the subtraction, so a run whose arming silently stopped
#: working fails as "nothing to subtract" rather than passing as parity.
#:
#: Looked for in the record with its wrapping collapsed, never line by line: the
#: phrase is long enough to wrap, and a width the run did not expect would then
#: fail the arming check and report a lost warning instead of the wrapping
#: difference that is the actual finding.
ARMED_WARNING_HEAD = "shell export disagrees with this deployment's env chain"

#: How many lines that record is expected to occupy, as a floor — and it is the
#: **record's** size, not the capture's. A floor over the whole of stderr would
#: be satisfied by any other chatter that happened to land there, which is the
#: same set-derived-check-without-a-size-pin soft spot the earlier reviews on
#: this suite closed: the set that is measured and the set whose copy is checked
#: have to be one set.
#:
#: Pinned as a floor rather than exactly, because what matters is that the
#: subtraction has the whole block to work on and the copy is free to grow. One
#: line would satisfy "non-empty" while testing almost nothing. The echo path
#: never wraps — a pipe carries each of the block's lines whole, and the
#: harness's scrollback reassembles a terminal's wrapped rows back into them —
#: so the floor is the promoted block's three logical lines: the summary, the
#: detail and the remedy.
ARMED_WARNING_MINIMUM_LINES = 3


@dataclass
class PipedRun:
    """One verb run with both streams redirected, and the two captures."""

    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str

    def describe(self) -> str:
        """A failure message worth reading: both captures, labelled."""
        return (
            f"argv={self.argv} exit={self.exit_code}\n"
            f"--- piped stdout ---\n{self.stdout}\n"
            f"--- piped stderr ---\n{self.stderr}"
        )


def run_piped(argv: list[str], *, cwd: Path, env: dict[str, str]) -> PipedRun:
    """Run one verb with stdout and stderr redirected to separate pipes.

    Deliberately spelled here rather than borrowed from ``conftest``'s render
    helper: the whole premise of this module is that the two runs are the same
    program on different devices, so both must demonstrably start from
    :data:`~tests.pty.test_live_region_real_terminal.CLI_BOOTSTRAP` — the same
    constant :class:`~tests.pty.test_live_region_real_terminal.PtyProcess` runs.
    A second copy of that program somewhere else could drift from it and the
    comparison would quietly become one between two different CLIs.

    ``stdin`` is ``/dev/null`` rather than the runner's own: a verb that reached
    a prompt here would read an immediate end of file and give up, which is a
    failure the scenario can see. Inheriting pytest's stdin would leave it
    waiting for a keystroke nobody is going to send.
    """
    result = subprocess.run(  # noqa: S603 - argv is built here, not by input
        [sys.executable, "-c", CLI_BOOTSTRAP, *argv],
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return PipedRun(
        argv=list(argv),
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def piped_env(env: dict[str, str]) -> dict[str, str]:
    """*env* with the terminal's own geometry pinned for a run that has none.

    The one environmental difference between the two runs, and it exists to
    remove a difference rather than to make one: Rich reads ``COLUMNS`` ahead of
    everything else, so this is how a process with no terminal is told the width
    the other run's terminal had. See the module docstring for what happens
    without it.
    """
    piped = dict(env)
    piped["COLUMNS"] = str(TERMINAL_COLUMNS)
    piped["LINES"] = str(TERMINAL_ROWS)
    return piped


def normalised(lines: list[str]) -> list[str]:
    """*lines* reduced to what an operator reads, with the two clocks removed.

    Escapes are stripped (the terminal's screen already is), lines are
    right-stripped — a pipe pads a wrapped record's lines out to the console
    width and a terminal's scrollback does not — and trailing blank lines are
    dropped, which is the same rule
    :meth:`~tests.pty.test_live_region_real_terminal.Terminal.scrollback` applies
    to the screen. Interior blank lines are kept: one is part of what the deploy
    prints, and both sides have to have it.
    """
    out = [
        ELAPSED.sub("(<elapsed>)", TIMESTAMP.sub(TIME_PLACEHOLDER, strip_ansi(line))).rstrip()
        for line in lines
    ]
    while out and not out[-1]:
        out.pop()
    return out


def subtract(scrollback: list[str], subtrahend: list[str]) -> tuple[list[str], list[str]]:
    """Strike each line of *subtrahend* off *scrollback* once.

    Returns the remainder in its original order, and whatever could not be
    struck off at all. A non-empty second element means the terminal never
    carried a line the pipe did, which is a different failure from a parity
    mismatch and is reported as one.
    """
    outstanding = Counter(subtrahend)
    remainder = []
    for line in scrollback:
        if outstanding[line] > 0:
            outstanding[line] -= 1
        else:
            remainder.append(line)
    return remainder, sorted(outstanding.elements())


def diff(expected: list[str], actual: list[str]) -> str:
    """A unified diff of two line lists, for a failure message."""
    return "\n".join(
        difflib.unified_diff(expected, actual, "piped stdout", "terminal remainder", lineterm="")
    )


def test_the_scripted_deploy_reads_the_same_piped_as_it_does_on_a_terminal(
    startable_repo: Path,
    stub_runtime: StubRuntime,
    pty_env: dict[str, str],
    tmp_path: Path,
    unhurried_runtime: None,
) -> None:
    """Criterion 9's comparison: two devices, one deploy, the same lines.

    ``osprey up -d`` twice over — once on a real pty, once with both streams
    redirected — against the same repository, restored to its pristine state in
    between. The terminal run mounts a live region and proxies the process's
    stderr onto it; the piped run mounts nothing and keeps the two streams apart.
    Subtracting the piped run's stderr from the terminal's scrollback must leave
    the piped run's stdout, in order.
    """
    # Arm the stderr side. A real disagreement between the shell's exports and
    # the deployment's env chain, provoked the way scenario 3 provokes it: the
    # repository pins a value for a variable the compose files interpolate, and
    # the run exports a different one.
    env_path = startable_repo / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8") + f"{ARMED_VARIABLE}={CHAIN_VALUE}\n",
        encoding="utf-8",
    )
    pty_env[ARMED_VARIABLE] = EXPORTED_VALUE

    # The state both runs start from, byte for byte. Snapshotting the tree is
    # what lets the second run use the *same absolute path* as the first: the
    # repository's own path and its directory name are printed (in the summary
    # card, and inside the wrapped warning), so two copies at two paths would
    # differ in the copy itself and no normalisation could put them back
    # together — the path is wrapped mid-string across several lines.
    pristine = tmp_path / "pristine"
    shutil.copytree(startable_repo, pristine, symlinks=True)

    terminal_run = run_on_pty(["up", "-d"], cwd=startable_repo, env=pty_env)

    assert terminal_run.exit_code == 0, terminal_run.describe()
    # The terminal run has to be the interesting one, or the comparison is
    # between two plain-line runs and says nothing about the region.
    assert_region_was_live(terminal_run)
    # And this is the "minus region transients" of the comparison: the region
    # took every frame back, so the scrollback below carries none.
    assert_screen_is_intact(terminal_run)

    shutil.rmtree(startable_repo)
    shutil.copytree(pristine, startable_repo, symlinks=True)
    # The runtime forgets what it started, so the second deploy is a first
    # deploy. Only the state file: the invocation log is what the ``stub_runtime``
    # fixture checks for unanswered seams at teardown, and it must keep both
    # runs' invocations in it.
    stub_runtime.state_path.unlink(missing_ok=True)

    piped_run = run_piped(["up", "-d"], cwd=startable_repo, env=piped_env(pty_env))

    assert piped_run.exit_code == 0, piped_run.describe()

    # The premise of the whole comparison: a pipe got no region. Asked of the raw
    # captures, since a mounted region is a cursor hide and a spinner glyph.
    for name, capture in (("stdout", piped_run.stdout), ("stderr", piped_run.stderr)):
        assert "\x1b[?25l" not in capture, (
            f"the piped run hid the cursor on {name}, so it mounted a region with no terminal "
            f"to mount it on\n{piped_run.describe()}"
        )
        assert not any(glyph in capture for glyph in SPINNER_FRAMES), (
            f"a spinner frame reached piped {name}\n{piped_run.describe()}"
        )

    scrollback = normalised(terminal_run.screen)
    stdout = normalised(piped_run.stdout.splitlines())
    stderr = normalised(piped_run.stderr.splitlines())

    # Armed control. Without a record on stderr the subtraction below would
    # subtract nothing and pass for that reason. Both halves of the check — how
    # big it is and what it says — are asked of the same slice: the record, from
    # its head line to the end of the capture, rather than of stderr as a whole.
    heads = [index for index, line in enumerate(stderr) if RECORD_HEAD.match(line)]
    assert len(heads) == 1, (
        f"expected exactly one rendered record on the piped run's stderr, found {len(heads)}. "
        f"The slice measured below runs from a record's head line to the end of the capture, "
        f"so a second record makes it ambiguous: scope the slice to the armed record rather "
        f"than dropping this check\n{piped_run.describe()}"
    )
    record = stderr[heads[0] :]
    assert len(record) >= ARMED_WARNING_MINIMUM_LINES, (
        f"the record on the piped run's stderr is {len(record)} line(s) long, so either the "
        f"arming stopped working or the record no longer wraps, and this test has next to "
        f"nothing to subtract\n{piped_run.describe()}"
    )
    unwrapped = " ".join(" ".join(record).split())
    assert ARMED_WARNING_HEAD in unwrapped and ARMED_VARIABLE in unwrapped, (
        f"the armed warning is not the record on stderr, so the subtraction is not the one "
        f"this test is about\n{piped_run.describe()}"
    )
    assert stdout, f"the piped run printed nothing at all\n{piped_run.describe()}"

    # THE GUARD, and it is first on purpose: the subtraction is only unambiguous
    # while no line is written to both streams. Were there one, striking a stderr
    # line off the scrollback could strike the stdout copy instead and the
    # arithmetic could balance for a reason that has nothing to do with parity.
    #
    # A BLANK LINE IS NOT EXEMPT, and must never be made one. The deploy prints
    # an interior blank on stdout; if stderr ever prints one too, the terminal's
    # blank cannot be attributed to either stream and the subtraction is exactly
    # as ambiguous as it would be for any shared line. Filtering blanks out of
    # the comparison would not fix that — it would hide it, and quietly excuse
    # the scrollback from carrying one.
    ambiguous = sorted(set(stdout) & set(stderr))
    assert not ambiguous, (
        f"these lines are on both piped streams, so subtracting stderr from the terminal's "
        f"scrollback cannot say which stream a scrollback line came from: {ambiguous}. This "
        f"guard firing is the guard working, including on a blank line — the fix is to tell "
        f"the two streams apart, never to exempt the line\n{piped_run.describe()}"
    )

    remainder, unstruck = subtract(scrollback, stderr)

    assert not unstruck, (
        f"the piped run wrote these to stderr but they are nowhere in the terminal's "
        f"scrollback, so a redirect is being told something a terminal is not: {unstruck}\n"
        f"{terminal_run.describe()}\n{piped_run.describe()}"
    )
    assert remainder == stdout, (
        f"the terminal's scrollback, minus everything the piped run said on stderr, is not "
        f"the piped run's stdout:\n{diff(stdout, remainder)}\n"
        f"{terminal_run.describe()}\n{piped_run.describe()}"
    )
