"""``osprey reset`` and the terminal it shares with the live region.

What reset *destroys* is pinned in ``tests/deployment/test_reset_scoping.py``:
the identity scoping, the refusal, the plan-equals-execution equality, the exit
codes. None of that is repeated here, and this file borrows that module's
fixtures rather than building a second deployment repo that could drift from it.

What is pinned here is narrower and entirely about output. On a terminal the
reporter mounts a live region that owns the cursor, so reset's plan is written
through the reporter's console instead of straight at stdout, and the typed
destruction confirmation takes the region down for as long as it is waiting on
an answer. Both of those are changes to WHERE bytes go, and the contract is that
off a terminal they change nothing at all:

* the plan is byte for byte the lines the deployment layer emitted -- not a
  substring of them, since a transport that re-wrapped at 80 columns or dropped
  a trailing newline would pass every substring assertion ever written;
* the interrupt notice is still on **stderr**. That one is asserted as a STREAM
  rather than as text: it is the line a piped or CI caller watches for, and
  moving it to stdout would break a consumer silently while every text
  assertion kept passing.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from osprey.cli import phase_reporter, styles
from osprey.cli.phase_reporter import LiveReporter, install_reporter
from osprey.cli.reset_cmd import reset as reset_command
from osprey.cli.styles import osprey_theme
from osprey.deployment import reset as reset_mod
from osprey.deployment.reset import confirmation_token, reset_deployment
from tests.deployment import test_reset_scoping as scoping

# The deployment repo, the recording runtime and the no-op ``down``, taken from
# the module that specifies them. Duplicating them here would give this file a
# second definition of "a deployment repo" to keep in step with the first.
# Bound as attributes rather than imported so that a test parameter named after
# a fixture is not read as shadowing an import.
FakeRuntime = scoping.FakeRuntime
make_probe = scoping.make_probe
ours = scoping.ours
repo = scoping.repo
no_down = scoping.no_down

#: Everything Rich writes that is not text: styles, cursor moves, erases -- and
#: the bare carriage return a repaint opens its line with, which the escape
#: pattern alone does not catch.
_NOISE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\r")

#: The interrupt notice, in full, exactly as the renderer's failure shape lays
#: it out: what failed, then why, then the one thing to do. Written out rather
#: than imported so that a reword has to come past these tests and re-affirm
#: both the wording and the stream it goes to.
INTERRUPT_NOTICE = (
    "✗ Reset interrupted while it was removing things\n"
    "  It does not roll back, so this deployment is in a partly-reset state.\n"
    "  → re-run `osprey reset` to finish; it re-reads what is actually there\n"
)

#: Wide enough that the region's table has room, narrow enough that the longest
#: plan lines are past it -- which is what makes "not wrapped" an observation
#: rather than a coincidence.
_WIDTH = 100


@pytest.fixture
def live_reporter(monkeypatch: pytest.MonkeyPatch):
    """A mounted live region over a recording console, installed as THE reporter.

    ``force_terminal`` is what makes the ``Live`` real: off a terminal Rich skips
    the repaint entirely and there would be no region for any of this to be
    about. The theme is not optional either -- the region's styles are semantic
    tokens, and a bare ``Console`` raises ``MissingStyle`` on the first frame.

    Installing it before the verb runs is what puts it in the verb's hands:
    ``lifecycle_reporter`` reuses a reporter it finds already installed, and
    therefore never mounts a second region -- so this fixture starts the one it
    yields. Swapping the previous reporter back at the end stops it.

    The monitor's tick is parked for the same reason every live test parks it:
    mounting the region starts a real thread on the real clock, and one tick
    landing mid-test repaints the region under the output being asserted on.
    """
    monkeypatch.setattr(phase_reporter, "_MONITOR_INTERVAL", 3600.0)
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        color_system="standard",
        no_color=False,
        width=_WIDTH,
        theme=osprey_theme,
        legacy_windows=False,
    )
    reporter = LiveReporter(console=console)
    previous = install_reporter(reporter)
    reporter.start_rendering()
    try:
        yield reporter, buffer
    finally:
        install_reporter(previous)


def rendered(buffer: io.StringIO) -> str:
    """What a reader would see in the recording console, without the cursor work."""
    return _NOISE.sub("", buffer.getvalue())


def probe_for(fake: FakeRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the CLI's reset at ``fake`` instead of a container runtime."""
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: make_probe(fake))


# ---------------------------------------------------------------------------
# Off a terminal: not one byte moves
# ---------------------------------------------------------------------------


def test_the_plan_off_a_terminal_is_byte_for_byte_the_lines_reset_emitted(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The console transport must be exactly ``click.echo`` where nothing renders.

    Both halves are the same ``--dry-run`` against the same repo, so the plan
    they describe is identical by construction and the only thing left to
    disagree about is the transport. Equality rather than containment: wrapping
    at the console's width, a swallowed blank line, a missing trailing newline
    -- the failures worth catching here are all invisible to a substring check.
    """
    fake = FakeRuntime(volumes={"mine-vol": ours(repo)})
    probe_for(fake, monkeypatch)

    emitted: list[str] = []
    reset_deployment(repo, probe=make_probe(fake), dry_run=True, emit=emitted.append)

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "".join(f"{line}\n" for line in emitted)
    # Guard on the guard: a plan of only short lines would let a re-wrapping
    # transport through this test unnoticed.
    assert max(len(line) for line in emitted) > _WIDTH


def test_the_interrupt_notice_off_a_terminal_is_still_on_stderr(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stream is the contract, so the stream is what is asserted.

    A piped ``osprey reset`` has a caller reading stdout for the plan and stderr
    for trouble. Routing this notice through the reporter's console would put it
    on stdout, where that caller never looks -- and every assertion about its
    WORDING would keep passing while it did.

    Read ``result.stdout`` and ``result.stderr``, never ``result.output``: that
    third one is the two streams CONCATENATED, and an assertion made against it
    cannot tell them apart at all. The equality below is what keeps this honest
    even so -- stderr must hold the notice and NOTHING else, so a capture that
    collapsed the streams would fail it on the plan rather than pass it by
    accident.
    """

    def interrupt_during_teardown(_root: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(reset_mod, "down_deployment", interrupt_during_teardown)
    probe_for(FakeRuntime(volumes={"mine-vol": ours(repo)}), monkeypatch)

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), "-y"])

    assert result.exit_code == 3
    assert result.stderr == INTERRUPT_NOTICE
    assert "partly-reset" not in result.stdout


# ---------------------------------------------------------------------------
# On a terminal: through the console that owns the region
# ---------------------------------------------------------------------------


def test_the_plan_is_written_through_the_console_that_owns_the_region(
    repo: Path, monkeypatch: pytest.MonkeyPatch, live_reporter
) -> None:
    """A raw write to stdout would land inside the region, not above it."""
    _, buffer = live_reporter
    probe_for(FakeRuntime(volumes={"mine-vol": ours(repo)}), monkeypatch)

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "WILL BE REMOVED" in rendered(buffer)
    assert "WILL BE REMOVED" not in result.stdout


def test_the_typed_confirmation_runs_with_the_region_taken_down(
    repo: Path, no_down, monkeypatch: pytest.MonkeyPatch, live_reporter
) -> None:
    """The prompt asks its question on a terminal nothing else is repainting.

    ``input()`` writes its prompt to the real stdout, underneath anything Rich
    has mounted there, so a region left up redraws over the question while the
    operator is reading it -- on the one prompt in OSPREY that destroys data.

    Both halves matter. Nothing may be rendering while the answer is being
    typed, and the region must be back afterwards: ``suspended()`` restores on
    the way out, and a reset that finished with its region gone would leave the
    verb's remaining output unsequenced.
    """
    reporter, _ = live_reporter
    at_the_prompt: dict[str, object] = {}

    def answer(_prompt: str) -> str:
        at_the_prompt["rendering"] = reporter.is_rendering()
        at_the_prompt["depth"] = reporter._suspend_depth
        return confirmation_token(repo)

    monkeypatch.setattr("builtins.input", answer)
    probe_for(FakeRuntime(volumes={"mine-vol": ours(repo)}), monkeypatch)

    result = CliRunner().invoke(reset_command, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert at_the_prompt == {"rendering": False, "depth": 1}
    assert reporter.is_rendering()


def test_the_interrupt_notice_goes_through_the_console_while_a_region_is_up(
    repo: Path, monkeypatch: pytest.MonkeyPatch, live_reporter
) -> None:
    """On a terminal the notice still belongs to the console that owns the region.

    stdout and stderr are the same device here, so nothing is lost by routing
    it; what would be lost by not routing it is the notice itself, written into
    a region that is about to repaint over it. The renderer's trouble path
    borrows the region's console for exactly that reason, so the notice is read
    out of the region's buffer and NOT out of the stderr console, which is
    replaced here with a recording one and asserted to have taken nothing.

    Asserted as the whole three-line block rather than as a substring: a
    transport that folded it at the console's width, or dropped the indent that
    makes the cause and the remedy subordinate, would pass every substring
    assertion ever written against it.
    """

    _, buffer = live_reporter
    err_buffer = io.StringIO()
    monkeypatch.setattr(
        styles, "err_console", Console(file=err_buffer, width=_WIDTH, theme=osprey_theme)
    )

    def interrupt_during_teardown(_root: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(reset_mod, "down_deployment", interrupt_during_teardown)
    probe_for(FakeRuntime(volumes={"mine-vol": ours(repo)}), monkeypatch)

    result = CliRunner().invoke(reset_command, ["--repo", str(repo), "-y"])

    assert result.exit_code == 3
    assert INTERRUPT_NOTICE in rendered(buffer)
    assert err_buffer.getvalue() == ""
