"""Tests for the CLI's core output primitives (``osprey.cli.output``).

Four things are being pinned here, and they are the whole contract eleven
migration tasks depend on:

* **Echo class.** Every primitive prints under every reporter, ``NullReporter``
  included -- ``osprey -v <verb>`` keeps its verb's own output even though the
  phase record is swallowed. Each primitive is exercised under
  ``PhaseReporter``, under a ``LiveReporter`` with a region actually mounted,
  and under ``NullReporter``.
* **Trouble's stream.** ``warn`` and ``fail`` take stderr, except while a live
  region is mounted: there they take the region's own console, which is the only
  one that can put the line ABOVE the region rather than onto the row it is
  about to redraw. The borrow ends with the region.
* **Machine mode.** Inside the block stdout stays empty and every line lands on
  stderr, so a ``--json`` caller reads one document.
* **Plainness off a terminal.** Piped output carries no escape sequences at
  all, so subordination has to be structural (indentation) rather than only
  dim.

The reporter fakes here subclass ``PhaseReporter`` rather than standing in for
it, per repo convention: a fake that only looks like one would not prove that
the real ``out()``/``echo`` seam is the one being used.
"""

import io
import re

import pytest
from rich.console import Console
from rich.text import Text

from osprey.cli import output, phase_reporter, styles
from osprey.cli.output import (
    fail,
    machine_mode,
    machine_mode_active,
    note,
    report,
    section,
    table,
    warn,
)
from osprey.cli.phase_reporter import (
    LiveReporter,
    NullReporter,
    PhaseReporter,
    current_reporter,
    install_reporter,
)
from osprey.cli.styles import data_table, osprey_theme, panel

#: Anything Rich writes that is not text: styles, cursor moves, erases.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

#: Just the color half of that. A line printed above a mounted region carries
#: the region's own erase (``\x1b[2K``) whether or not it is styled, so "is this
#: line colored" has to be asked of the select-graphic-rendition sequences alone.
_SGR = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def restore_installed_reporter():
    """Leave the module singleton exactly as the test found it."""
    original = current_reporter()
    yield
    install_reporter(original)


@pytest.fixture(autouse=True)
def machine_mode_off():
    """Prove no test leaks the process-scoped redirect into the next one."""
    yield
    assert output._machine_mode is False


@pytest.fixture(autouse=True)
def parked_monitor(monkeypatch):
    """Park the monitor's tick: these tests drive the region by hand.

    A real tick landing mid-test repaints the region and writes cursor control
    into the buffer the assertions read.
    """
    monkeypatch.setattr(phase_reporter, "_MONITOR_INTERVAL", 3600.0)


def recording_console() -> tuple[Console, io.StringIO]:
    """A themed terminal console that records everything written to it.

    ``force_terminal`` is what makes a ``Live`` real: off a terminal Rich skips
    the repaint entirely. The theme is not optional either -- the primitives'
    styles are semantic tokens, and a bare ``Console()`` raises ``MissingStyle``
    on the first one.
    """
    buffer = io.StringIO()
    return (
        Console(file=buffer, theme=osprey_theme, force_terminal=True, width=100),
        buffer,
    )


class RecordingReporter(PhaseReporter):
    """A plain-line reporter whose console is a buffer, not the terminal."""

    def __init__(self, console: Console) -> None:
        super().__init__(color=False)
        self._console = console

    def out(self) -> Console:
        return self._console


def visible(text: str) -> list[str]:
    """The words in ``text``, with escapes and blank lines dropped."""
    return [line for line in _ANSI.sub("", text).splitlines() if line.strip()]


def visible_raw(text: str) -> list[str]:
    """The same lines, with their escapes left ON.

    ``visible``'s counterpart for the tests that are about the styling rather
    than the words: whether a line is colored can only be asked of a line that
    still carries its escapes.
    """
    return [line for line in text.splitlines() if _ANSI.sub("", line).strip()]


@pytest.fixture
def err_buffer(monkeypatch):
    """Stand a recording terminal console in for ``styles.err_console``.

    The trouble primitives resolve stderr through the ``styles`` module alias at
    call time, so this is the seam -- and it is the only way to see what they
    would put on a *terminal*, which is where the colors are. It is also the one
    way to ask whether the stderr console was used AT ALL, which ``capsys``
    cannot: a ``Live`` swaps ``sys.stderr`` for a proxy of its own while a region
    is mounted, so the descriptor a line reached says nothing about the console
    it went through.

    It REPLACES the console object, which is safe here only because nothing in
    this module drives the CLI: ``styles.set_theme`` re-themes whatever console
    the module global holds at call time and pops before it pushes, so a run
    that reaches ``cli.main`` with a substituted console meets an empty theme
    stack. A test here that ever invokes a verb must redirect the real console's
    ``_file`` instead of swapping the object. The substitution is what buys the
    ``force_terminal`` console the color assertions need, which redirecting a
    non-terminal stderr console cannot.
    """
    console, buffer = recording_console()
    monkeypatch.setattr(styles, "err_console", console)
    return buffer


@pytest.fixture
def recording_err(err_buffer):
    """The same recording stderr console, under the plain-line reporter."""
    install_reporter(PhaseReporter(color=False))
    return err_buffer


@pytest.fixture
def live():
    """A ``LiveReporter`` with its region mounted, installed, torn down after."""
    console, buffer = recording_console()
    reporter = LiveReporter(console=console)
    install_reporter(reporter)
    reporter.start_rendering()
    try:
        yield reporter, buffer
    finally:
        reporter.stop_rendering()


# -- report -----------------------------------------------------------------


def test_report_prints_under_the_plain_reporter(capsys):
    install_reporter(PhaseReporter(color=False))
    report("deployment created at /srv/beamline")
    assert visible(capsys.readouterr().out) == ["deployment created at /srv/beamline"]


def test_report_prints_under_null_reporter(capsys):
    """The whole point of echo class: ``-v`` keeps the verb's own output."""
    install_reporter(NullReporter(verbose=True))
    report("deployment created at /srv/beamline")
    assert visible(capsys.readouterr().out) == ["deployment created at /srv/beamline"]


def test_null_reporter_still_swallows_the_phase_record(capsys):
    """The control for the test above -- emit class really is silenced here."""
    reporter = NullReporter(verbose=True)
    install_reporter(reporter)
    reporter.emit("→ preparing")
    report("deployment created at /srv/beamline")
    assert visible(capsys.readouterr().out) == ["deployment created at /srv/beamline"]


def test_report_prints_through_the_mounted_region(live, capsys):
    """On a terminal the reporter's console owns the cursor, so the line uses it."""
    reporter, buffer = live
    report("deployment created at /srv/beamline")
    assert "deployment created at /srv/beamline" in _ANSI.sub("", buffer.getvalue())
    # Nothing bypassed the region onto raw stdout.
    assert capsys.readouterr().out == ""
    # And the region survived the line rather than being torn down by it.
    assert reporter.is_rendering() is True


def test_report_style_rides_on_the_text_not_the_call():
    """A ``style=`` argument would repaint a mounted region in the line's color."""
    console, buffer = recording_console()
    install_reporter(RecordingReporter(console))
    report("osprey-demo — running", style="bold")
    assert "\x1b[1m" in buffer.getvalue()
    assert visible(buffer.getvalue()) == ["osprey-demo — running"]


def test_report_takes_prose_positionally():
    """The copy guard reads prose out of positional arguments only.

    Positional-only, so a call site cannot quietly hide a printed sentence from
    the guard by naming the argument.
    """
    with pytest.raises(TypeError):
        report(text="deployment created")  # type: ignore[call-arg]


def test_report_prints_a_blank_separator_line():
    console, buffer = recording_console()
    install_reporter(RecordingReporter(console))
    report("")
    assert _ANSI.sub("", buffer.getvalue()) == "\n"


# -- note -------------------------------------------------------------------


def test_note_is_indented_one_step(capsys):
    install_reporter(PhaseReporter(color=False))
    report("using profile als-demo")
    note("resolved from OSPREY_PROFILE")
    assert visible(capsys.readouterr().out) == [
        "using profile als-demo",
        "  resolved from OSPREY_PROFILE",
    ]


def test_note_prints_under_null_reporter(capsys):
    install_reporter(NullReporter(verbose=True))
    note("resolved from OSPREY_PROFILE")
    assert visible(capsys.readouterr().out) == ["  resolved from OSPREY_PROFILE"]


def test_note_prints_through_the_mounted_region(live, capsys):
    _reporter, buffer = live
    note("resolved from OSPREY_PROFILE")
    assert "  resolved from OSPREY_PROFILE" in _ANSI.sub("", buffer.getvalue())
    assert capsys.readouterr().out == ""


def test_note_is_dim_on_a_terminal():
    """Dim is the other half of subordinate -- the half a pipe cannot carry."""
    console, buffer = recording_console()
    install_reporter(RecordingReporter(console))
    note("resolved from OSPREY_PROFILE")
    assert "\x1b[" in buffer.getvalue()


def test_note_takes_prose_positionally():
    with pytest.raises(TypeError):
        note(text="resolved from OSPREY_PROFILE")  # type: ignore[call-arg]


# -- section ----------------------------------------------------------------


def test_section_aligns_its_values_into_a_column(capsys):
    install_reporter(PhaseReporter(color=False))
    section("endpoints", {"web": "http://localhost:8080", "openobserve": "http://localhost:5080"})
    assert visible(capsys.readouterr().out) == [
        "endpoints",
        "  web           http://localhost:8080",
        "  openobserve   http://localhost:5080",
    ]


def test_section_accepts_ordered_pairs_with_repeated_labels(capsys):
    install_reporter(PhaseReporter(color=False))
    section("mounts", [("volume", "osprey-data"), ("volume", "osprey-logs")])
    assert visible(capsys.readouterr().out) == [
        "mounts",
        "  volume   osprey-data",
        "  volume   osprey-logs",
    ]


def test_section_renders_non_string_values(capsys):
    install_reporter(PhaseReporter(color=False))
    section("ports", {"web": 8080, "detached": True})
    assert visible(capsys.readouterr().out) == [
        "ports",
        "  web        8080",
        "  detached   True",
    ]


def test_section_without_rows_prints_its_title_alone(capsys):
    install_reporter(PhaseReporter(color=False))
    section("endpoints", {})
    assert visible(capsys.readouterr().out) == ["endpoints"]


def test_section_without_a_title_prints_its_rows_alone(capsys):
    install_reporter(PhaseReporter(color=False))
    section("", {"web": "http://localhost:8080"})
    assert visible(capsys.readouterr().out) == ["  web   http://localhost:8080"]


def test_section_prints_under_null_reporter(capsys):
    install_reporter(NullReporter(verbose=True))
    section("endpoints", {"web": "http://localhost:8080"})
    assert visible(capsys.readouterr().out) == [
        "endpoints",
        "  web   http://localhost:8080",
    ]


def test_section_prints_through_the_mounted_region(live, capsys):
    _reporter, buffer = live
    section("endpoints", {"web": "http://localhost:8080"})
    assert visible(buffer.getvalue())[-2:] == [
        "endpoints",
        "  web   http://localhost:8080",
    ]
    assert capsys.readouterr().out == ""


def test_section_takes_its_title_positionally():
    with pytest.raises(TypeError):
        section(title="endpoints", items={})  # type: ignore[call-arg]


# -- warn -------------------------------------------------------------------


def test_warn_marks_its_summary_and_indents_its_detail(capsys):
    install_reporter(PhaseReporter(color=False))
    warn("no theme in the profile", "used the default theme instead")
    captured = capsys.readouterr()
    assert visible(captured.err) == [
        "⚠ no theme in the profile",
        "  used the default theme instead",
    ]
    assert captured.out == ""


def test_warn_without_detail_prints_the_summary_alone(capsys):
    install_reporter(PhaseReporter(color=False))
    warn("no theme in the profile")
    assert visible(capsys.readouterr().err) == ["⚠ no theme in the profile"]


def test_warn_treats_empty_detail_as_absent(capsys):
    install_reporter(PhaseReporter(color=False))
    warn("no theme in the profile", "")
    assert visible(capsys.readouterr().err) == ["⚠ no theme in the profile"]


def test_warn_indents_a_multi_line_detail_line_by_line(capsys):
    install_reporter(PhaseReporter(color=False))
    warn("two keys were ignored", "cli.colour is not a key\ncli.thème is not a key")
    assert visible(capsys.readouterr().err) == [
        "⚠ two keys were ignored",
        "  cli.colour is not a key",
        "  cli.thème is not a key",
    ]


def test_warn_is_colored_on_a_terminal_and_only_on_its_summary(recording_err):
    """One colored line per warning: the detail stays at reading contrast."""
    warn("no theme in the profile", "used the default theme instead")
    assert _ANSI.sub("", recording_err.getvalue()).splitlines() == [
        "⚠ no theme in the profile",
        "  used the default theme instead",
    ]
    lines = recording_err.getvalue().splitlines()
    assert "\x1b[" in lines[0]
    assert "\x1b[" not in lines[1]


def test_warn_takes_its_prose_positionally():
    with pytest.raises(TypeError):
        warn(summary="no theme in the profile")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        warn("no theme in the profile", detail="used the default")  # type: ignore[call-arg]


def test_warn_prints_under_null_reporter(capsys):
    """Trouble is echo class too: ``-v`` never swallows a warning."""
    install_reporter(NullReporter(verbose=True))
    warn("no theme in the profile")
    assert visible(capsys.readouterr().err) == ["⚠ no theme in the profile"]


# -- fail -------------------------------------------------------------------


def test_fail_prints_what_failed_then_why_then_what_to_do(capsys):
    """The whole shape, in one place -- FR6's single form for every refusal."""
    install_reporter(PhaseReporter(color=False))
    fail(
        "could not start the web service",
        "port 8080 is already bound",
        "free the port, or set deploy.web.port",
    )
    captured = capsys.readouterr()
    assert visible(captured.err) == [
        "✗ could not start the web service",
        "  port 8080 is already bound",
        "  → free the port, or set deploy.web.port",
    ]
    assert captured.out == ""


def test_fail_without_cause_or_remedy_prints_the_summary_alone(capsys):
    install_reporter(PhaseReporter(color=False))
    fail("could not start the web service")
    assert visible(capsys.readouterr().err) == ["✗ could not start the web service"]


def test_fail_with_a_cause_and_no_remedy(capsys):
    """Nothing honest to suggest is a reason to say nothing, not to invent one."""
    install_reporter(PhaseReporter(color=False))
    fail("could not start the web service", "port 8080 is already bound")
    assert visible(capsys.readouterr().err) == [
        "✗ could not start the web service",
        "  port 8080 is already bound",
    ]


def test_fail_with_a_remedy_and_no_cause(capsys):
    install_reporter(PhaseReporter(color=False))
    fail("no profile is selected", None, "pick one with osprey use <profile>")
    assert visible(capsys.readouterr().err) == [
        "✗ no profile is selected",
        "  → pick one with osprey use <profile>",
    ]


def test_fail_indents_a_multi_line_cause_line_by_line(capsys):
    install_reporter(PhaseReporter(color=False))
    fail("two services did not come up", "web: port 8080 is bound\napi: image not found", "retry")
    assert visible(capsys.readouterr().err) == [
        "✗ two services did not come up",
        "  web: port 8080 is bound",
        "  api: image not found",
        "  → retry",
    ]


def test_fail_treats_empty_cause_and_remedy_as_absent(capsys):
    install_reporter(PhaseReporter(color=False))
    fail("could not start the web service", "", "")
    assert visible(capsys.readouterr().err) == ["✗ could not start the web service"]


def test_fail_only_prints(capsys):
    """No raise, no exit, no exit code -- the caller keeps its own Abort."""
    install_reporter(PhaseReporter(color=False))
    assert fail("could not start the web service", "port 8080 is already bound") is None
    report("and the verb decided what to do about it")
    captured = capsys.readouterr()
    assert visible(captured.out) == ["and the verb decided what to do about it"]


def test_fail_is_colored_on_a_terminal_summary_and_remedy_arrow_only(recording_err):
    fail("could not start the web service", "port 8080 is already bound", "free the port")
    lines = recording_err.getvalue().splitlines()
    assert "\x1b[" in lines[0]
    assert "\x1b[" not in lines[1]
    # The remedy line's ARROW carries the accent; its prose stays uncolored.
    arrow, _, prose = lines[2].partition("→ ")
    assert "\x1b[" in arrow
    assert "\x1b[" not in prose.replace("\x1b[0m", "")
    assert _ANSI.sub("", recording_err.getvalue()).splitlines() == [
        "✗ could not start the web service",
        "  port 8080 is already bound",
        "  → free the port",
    ]


def test_fail_takes_its_prose_positionally():
    with pytest.raises(TypeError):
        fail(summary="could not start the web service")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        fail("could not start", cause="port 8080 is bound")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        fail("could not start", None, remedy="free the port")  # type: ignore[call-arg]


# -- fail without the mark ---------------------------------------------------


def test_an_unmarked_failure_is_the_marked_one_minus_its_glyph(capsys):
    """One failure wears one ``✗``, so the handler explaining it wears none.

    A refusal raised inside an open phase leaves through ``Phase.fail``, which
    has already printed ``✗ title`` for it. The handler that says WHY prints the
    same block with the mark left off -- same words, same indentation, same
    stream -- so the two read as one failure rather than two.
    """
    install_reporter(PhaseReporter(color=False))
    fail("--dev cannot be honored", "this render carries no wheel", "osprey build --dev")
    marked = visible(capsys.readouterr().err)
    fail(
        "--dev cannot be honored",
        "this render carries no wheel",
        "osprey build --dev",
        mark=False,
    )
    unmarked = visible(capsys.readouterr().err)

    assert unmarked == [
        "--dev cannot be honored",
        "  this render carries no wheel",
        "  → osprey build --dev",
    ]
    # The same block, and the mark is the whole of the difference.
    assert marked == [f"✗ {unmarked[0]}", *unmarked[1:]]


def test_an_unmarked_failure_prints_no_glyph_at_all(capsys):
    """The point of the form, asserted as the property it exists for."""
    install_reporter(PhaseReporter(color=False))
    fail("--dev cannot be honored", "this render carries no wheel", "osprey build --dev")
    fail("--dev cannot be honored", "this render carries no wheel", mark=False)
    assert capsys.readouterr().err.count("✗") == 1


def test_an_unmarked_failure_keeps_its_summary_colored(recording_err):
    """Losing the mark does not lower it out of the altitude an operator reads at."""
    fail("--dev cannot be honored", "this render carries no wheel", mark=False)
    lines = recording_err.getvalue().splitlines()
    assert "\x1b[" in lines[0]
    assert "\x1b[" not in lines[1]


def test_an_unmarked_failure_prints_under_null_reporter(capsys):
    """Echo class like the rest: ``osprey -v`` never swallows a refusal's reason."""
    install_reporter(NullReporter(verbose=True))
    fail("--dev cannot be honored", "this render carries no wheel", mark=False)
    assert visible(capsys.readouterr().err) == [
        "--dev cannot be honored",
        "  this render carries no wheel",
    ]


def test_an_unmarked_failure_lands_above_a_mounted_region(live, err_buffer, capsys):
    """It takes the region-aware routing whole, because it is the same seam."""
    reporter, buffer = live
    fail("--dev cannot be honored", "this render carries no wheel", mark=False)
    assert visible(buffer.getvalue())[-2:] == [
        "--dev cannot be honored",
        "  this render carries no wheel",
    ]
    assert err_buffer.getvalue() == ""
    assert capsys.readouterr().err == ""
    assert reporter.is_rendering() is True


def test_an_unmarked_failure_honors_machine_mode(capsys):
    """A ``--json`` caller reads one document, refusals included."""
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        fail("--dev cannot be honored", "this render carries no wheel", mark=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert visible(captured.err) == [
        "--dev cannot be honored",
        "  this render carries no wheel",
    ]


def test_fail_takes_its_mark_as_a_keyword():
    """Prose stays positional and the flag stays keyword: neither can pass as the other."""
    with pytest.raises(TypeError):
        fail("--dev cannot be honored", None, None, False)  # type: ignore[call-arg]


def test_fail_prints_under_null_reporter(capsys):
    install_reporter(NullReporter(verbose=True))
    fail("could not start the web service", "port 8080 is already bound", "free the port")
    assert visible(capsys.readouterr().err) == [
        "✗ could not start the web service",
        "  port 8080 is already bound",
        "  → free the port",
    ]


def test_trouble_lands_above_a_mounted_region_through_its_console(live, err_buffer, capsys):
    """A mounted region borrows the trouble lines, so they land above it whole.

    Writing at the real stderr while a region is up puts the line wherever the
    region left the cursor -- Rich unwraps its own ``FileProxy``
    (``rich_proxied_file``) for a stderr ``Console``, so ``Live`` never sees the
    write and never takes that row back. The region's own console is the only
    one that can place a line above it, which is the same borrow the log handler
    already gets, so trouble takes it too for as long as the region is up.
    """
    reporter, buffer = live
    warn("no theme in the profile")
    fail("could not start the web service", "port 8080 is already bound", "free the port")
    captured = capsys.readouterr()
    assert visible(buffer.getvalue())[-4:] == [
        "⚠ no theme in the profile",
        "✗ could not start the web service",
        "  port 8080 is already bound",
        "  → free the port",
    ]
    # Nothing went past the region: not through the stderr console it cannot
    # see, and not onto a raw stream either.
    assert err_buffer.getvalue() == ""
    assert captured.err == ""
    assert captured.out == ""
    # And the region survived the lines rather than being torn down by them.
    assert reporter.is_rendering() is True


def test_trouble_keeps_its_colors_through_a_mounted_region(live):
    """The borrow changes the stream, not the shape: one colored summary line."""
    _reporter, buffer = live
    warn("no theme in the profile", "used the default theme instead")
    summary, detail = visible_raw(buffer.getvalue())[-2:]
    # Named first, so the colors below are judged on the warning's own rows and
    # not on whatever the region happened to paint last.
    assert [_ANSI.sub("", line).strip() for line in (summary, detail)] == [
        "⚠ no theme in the profile",
        "used the default theme instead",
    ]
    assert _SGR.search(summary)
    assert not _SGR.search(detail)


def test_trouble_goes_back_to_stderr_once_the_region_comes_down(live, err_buffer):
    """The other half of the borrow: the console is given back at teardown.

    A verb that drew a region and then failed after it came down must put its
    ``✗`` back on the stream its contract names, rather than keep whichever
    console the region happened to leave behind.
    """
    reporter, buffer = live
    reporter.stop_rendering()
    before = buffer.getvalue()
    warn("no theme in the profile")
    assert visible(err_buffer.getvalue()) == ["⚠ no theme in the profile"]
    assert buffer.getvalue() == before


def test_machine_mode_keeps_trouble_off_a_mounted_region(live, err_buffer):
    """Stdout purity outranks the region for trouble as it does for report.

    A ``--json`` verb's document is the one thing on stdout, so inside the block
    a warning takes the stderr console even where the region would have had
    somewhere better to put it.
    """
    _reporter, buffer = live
    with machine_mode():
        warn("no theme in the profile")
    assert "no theme in the profile" not in buffer.getvalue()
    assert visible(err_buffer.getvalue()) == ["⚠ no theme in the profile"]


# -- table ------------------------------------------------------------------


def a_table():
    """One two-column data table whose cells are ``Text``, as the contract asks."""
    built = data_table()
    built.add_column("Service")
    built.add_column("Status")
    built.add_row("dispatch", Text("● Running", style="success"))
    return built


def test_table_prints_under_the_plain_reporter(capsys):
    install_reporter(PhaseReporter(color=False))
    table(a_table())
    printed = "\n".join(visible(capsys.readouterr().out))
    assert "Service" in printed
    assert "dispatch" in printed
    assert "● Running" in printed


def test_table_prints_under_null_reporter(capsys):
    """Echo class: ``osprey -v status`` keeps the table it ran the verb for."""
    install_reporter(NullReporter(verbose=True))
    table(a_table())
    assert "dispatch" in capsys.readouterr().out


def test_table_prints_through_the_mounted_region(live, capsys):
    """A table lands ABOVE the region, on the console that owns the cursor."""
    reporter, buffer = live
    table(a_table())
    assert "dispatch" in _ANSI.sub("", buffer.getvalue())
    assert capsys.readouterr().out == ""
    assert reporter.is_rendering() is True


def test_table_moves_to_stderr_in_machine_mode(capsys):
    """A ``--json`` verb's stdout is one document, tables included."""
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        table(a_table())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "dispatch" in captured.err


def test_table_renders_exactly_as_a_bare_console_print_would(capsys):
    """The module's ``soft_wrap`` changes nothing: a table sizes its own columns.

    Pinned because it is the reason this primitive can route through the same
    seam as the text ones instead of needing print keywords of its own.
    """
    install_reporter(PhaseReporter(color=False))
    table(a_table())
    through_the_primitive = capsys.readouterr().out

    # The same width the primitive just rendered at, read off the console it
    # used: a hard-coded 80 would disagree with any run that sets ``COLUMNS``.
    buffer = io.StringIO()
    Console(file=buffer, theme=osprey_theme, width=styles.console.width).print(a_table())

    assert through_the_primitive == buffer.getvalue()


def test_a_markup_string_cell_prints_its_brackets(capsys):
    """The trap this primitive's docstring exists for, pinned as behaviour.

    ``markup=False`` propagates into every nested renderable, so a cell built as
    a markup STRING renders its tags literally and drags the column widths out
    with them. Cells are ``Text``; there is no second way.
    """
    install_reporter(PhaseReporter(color=False))
    built = data_table()
    built.add_column("Status")
    built.add_row("[success]● Running[/success]")
    table(built)
    assert "[success]" in capsys.readouterr().out


def test_table_takes_its_renderable_positionally():
    """Uniform with the prose primitives, so every call in the module reads alike."""
    with pytest.raises(TypeError):
        table(renderable=a_table())  # type: ignore[call-arg]


def test_a_panel_goes_through_the_same_door(capsys):
    """Named for tables, but it is the generic renderable printer."""
    install_reporter(PhaseReporter(color=False))
    table(panel(Text("nothing is deployed")))
    assert "nothing is deployed" in capsys.readouterr().out


# -- machine mode -----------------------------------------------------------


def test_machine_mode_keeps_stdout_pure(capsys):
    """A ``--json`` verb's stdout is one document; the prose goes to stderr."""
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        report("deployment created at /srv/beamline")
        note("resolved from OSPREY_PROFILE")
        section("endpoints", {"web": "http://localhost:8080"})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert visible(captured.err) == [
        "deployment created at /srv/beamline",
        "  resolved from OSPREY_PROFILE",
        "endpoints",
        "  web   http://localhost:8080",
    ]


def test_machine_mode_redirects_even_with_a_region_mounted(live, capsys):
    """Stdout purity outranks the region: the block exists for the JSON reader."""
    _reporter, buffer = live
    with machine_mode():
        report("deployment created at /srv/beamline")
    assert "deployment created" not in buffer.getvalue()
    assert "deployment created at /srv/beamline" in capsys.readouterr().err


def test_machine_mode_restores_stdout_after_the_block(capsys):
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        report("inside")
    report("outside")
    captured = capsys.readouterr()
    assert visible(captured.out) == ["outside"]
    assert visible(captured.err) == ["inside"]


def test_machine_mode_restores_on_an_exception(capsys):
    install_reporter(PhaseReporter(color=False))
    with pytest.raises(RuntimeError), machine_mode():
        report("inside")
        raise RuntimeError("boom")
    report("outside")
    captured = capsys.readouterr()
    assert visible(captured.out) == ["outside"]
    assert visible(captured.err) == ["inside"]


def test_machine_mode_nests_without_ending_the_redirect_early(capsys):
    """A ``--json`` verb calling another one keeps the redirect for its body."""
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        with machine_mode():
            report("inner")
        report("after inner")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert visible(captured.err) == ["inner", "after inner"]


def test_machine_mode_leaves_trouble_where_it_already_was(capsys):
    """``warn``/``fail`` are stderr either way; the block changes nothing here."""
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        warn("no theme in the profile")
        fail("could not start the web service", "port 8080 is already bound", "free the port")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert visible(captured.err) == [
        "⚠ no theme in the profile",
        "✗ could not start the web service",
        "  port 8080 is already bound",
        "  → free the port",
    ]


# -- machine_mode_active -----------------------------------------------------


def test_machine_mode_active_is_false_outside_the_block():
    assert machine_mode_active() is False


def test_machine_mode_active_is_true_inside_the_block():
    """What a call site mounting its own console asks before it picks one."""
    with machine_mode():
        assert machine_mode_active() is True
    assert machine_mode_active() is False


def test_machine_mode_active_is_restored_after_an_exception():
    with pytest.raises(RuntimeError):
        with machine_mode():
            assert machine_mode_active() is True
            raise RuntimeError("the verb blew up mid-document")
    assert machine_mode_active() is False


# -- plainness off a terminal ------------------------------------------------


def test_every_primitive_is_plain_off_a_terminal(capsys):
    """Piped output carries no escapes at all, whatever the styles say."""
    install_reporter(PhaseReporter(color=False))
    report("deployment created", style="bold")
    note("resolved from OSPREY_PROFILE")
    section("endpoints", {"web": "http://localhost:8080"})
    warn("no theme in the profile", "used the default theme instead")
    fail("could not start the web service", "port 8080 is already bound", "free the port")
    captured = capsys.readouterr()
    assert "\x1b" not in captured.out
    assert "\x1b" not in captured.err


def test_trouble_keeps_its_glyphs_off_a_terminal(capsys):
    """``✗`` and ``→`` are content, not styling: a pipe still gets them."""
    install_reporter(PhaseReporter(color=False))
    warn("no theme in the profile")
    fail("could not start the web service", "port 8080 is already bound", "free the port")
    err = capsys.readouterr().err
    assert "⚠ no theme in the profile" in err
    assert "✗ could not start the web service" in err
    assert "  → free the port" in err


def test_machine_mode_output_is_plain_off_a_terminal(capsys):
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        report("deployment created", style="bold")
        note("resolved from OSPREY_PROFILE")
        section("endpoints", {"web": "http://localhost:8080"})
    assert "\x1b" not in capsys.readouterr().err


# -- report_fact and the run ledger ------------------------------------------


class _RecordingLogger:
    """The two methods the fact primitives call, recording what they were told."""

    def __init__(self) -> None:
        self.key_info_messages: list[str] = []
        self.warnings: list[str] = []

    def key_info(self, message: str) -> None:
        self.key_info_messages.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


@pytest.fixture(autouse=True)
def empty_ledger():
    """Start and leave every test with no collected rows."""
    output.clear_ledger()
    yield
    output.clear_ledger()


def test_report_fact_prints_one_line_in_the_step_column(capsys):
    """A fact wears the phase record's own ``  · `` shape, not a paragraph's."""
    install_reporter(PhaseReporter(color=False))
    logger = _RecordingLogger()
    output.report_fact(logger, "minted 2 service auth token(s) → .env")
    assert visible(capsys.readouterr().out) == ["  · minted 2 service auth token(s) → .env"]
    assert logger.key_info_messages == ["minted 2 service auth token(s) → .env"]


def test_report_fact_prints_under_null_reporter(capsys):
    """Echo class: ``-v`` keeps the facts with the rest of the verb's output."""
    install_reporter(NullReporter(verbose=True))
    output.report_fact(_RecordingLogger(), "dev mode: DEV_MODE set for the containers")
    assert visible(capsys.readouterr().out) == ["  · dev mode: DEV_MODE set for the containers"]


def test_report_fact_takes_its_prose_positionally():
    with pytest.raises(TypeError):
        output.report_fact(_RecordingLogger(), message="minted a token")  # type: ignore[call-arg]


def test_a_wrote_row_waits_for_the_flush(capsys):
    """The inline line prints now; the row prints once, at the flush, aligned."""
    install_reporter(PhaseReporter(color=False))
    logger = _RecordingLogger()
    output.report_fact(
        logger,
        "minted 2 service auth token(s) → .env",
        wrote=(".env", "TOKEN_A, TOKEN_B (gitignored; keep them secret)"),
    )
    before_flush = capsys.readouterr().out
    assert "TOKEN_A" not in before_flush

    output.flush_ledger()
    assert visible(capsys.readouterr().out) == [
        "This deploy wrote",
        "  .env   TOKEN_A, TOKEN_B (gitignored; keep them secret)",
    ]
    # The row reached the log sinks at collection time, not at the flush.
    assert any("TOKEN_A" in message for message in logger.key_info_messages)


def test_the_flush_empties_the_ledger(capsys):
    install_reporter(PhaseReporter(color=False))
    output.report_fact(_RecordingLogger(), "wrote a file", wrote=("file", "what it holds"))
    output.flush_ledger()
    capsys.readouterr()
    output.flush_ledger()
    assert capsys.readouterr().out == ""


def test_a_flush_with_nothing_collected_prints_nothing(capsys):
    install_reporter(PhaseReporter(color=False))
    output.flush_ledger()
    assert capsys.readouterr().out == ""


def test_clear_ledger_drops_rows_without_printing_them(capsys):
    """The next run must not inherit rows a failed run never flushed."""
    install_reporter(PhaseReporter(color=False))
    output.report_fact(_RecordingLogger(), "wrote a file", wrote=("file", "what it holds"))
    capsys.readouterr()
    output.clear_ledger()
    output.flush_ledger()
    assert capsys.readouterr().out == ""


def test_ledger_rows_keep_their_collection_order(capsys):
    install_reporter(PhaseReporter(color=False))
    logger = _RecordingLogger()
    output.report_fact(logger, "first", wrote=("b-label", "second row"))
    output.report_fact(logger, "second", wrote=("a-label", "first row"))
    capsys.readouterr()
    output.flush_ledger()
    lines = visible(capsys.readouterr().out)
    assert lines[1].startswith("  b-label")
    assert lines[2].startswith("  a-label")


def test_report_fact_honors_machine_mode(capsys):
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        output.report_fact(_RecordingLogger(), "minted a token", wrote=(".env", "TOKEN"))
        output.flush_ledger()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "minted a token" in captured.err
    assert "TOKEN" in captured.err


# -- warn_fact ----------------------------------------------------------------


def test_warn_fact_prints_the_indented_trouble_shape(capsys):
    """Summary one step in, body one step under it, remedy marked ``→``."""
    install_reporter(PhaseReporter(color=False))
    logger = _RecordingLogger()
    output.warn_fact(
        logger,
        "host timezone differs from the facility setting",
        "$TZ=US/Pacific but system.timezone=UTC",
        "set both to the same zone",
    )
    captured = capsys.readouterr()
    assert visible(captured.err) == [
        "  ⚠ host timezone differs from the facility setting",
        "    $TZ=US/Pacific but system.timezone=UTC",
        "    → set both to the same zone",
    ]
    assert captured.out == ""
    # One record for the sinks, carrying all three parts.
    assert logger.warnings == [
        "host timezone differs from the facility setting "
        "$TZ=US/Pacific but system.timezone=UTC set both to the same zone"
    ]


def test_warn_fact_without_detail_or_remedy_is_the_summary_alone(capsys):
    install_reporter(PhaseReporter(color=False))
    output.warn_fact(_RecordingLogger(), "this deployment is reachable from the network")
    assert visible(capsys.readouterr().err) == ["  ⚠ this deployment is reachable from the network"]


def test_warn_fact_prints_under_null_reporter(capsys):
    """Echo class, like every promoted fact: ``-v`` never swallows it."""
    install_reporter(NullReporter(verbose=True))
    output.warn_fact(_RecordingLogger(), "this deployment is reachable from the network")
    assert visible(capsys.readouterr().err) == ["  ⚠ this deployment is reachable from the network"]


def test_warn_fact_takes_its_prose_positionally():
    with pytest.raises(TypeError):
        output.warn_fact(_RecordingLogger(), summary="reachable")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        output.warn_fact(_RecordingLogger(), "reachable", detail="why")  # type: ignore[call-arg]


def test_warn_fact_honors_machine_mode(capsys):
    install_reporter(PhaseReporter(color=False))
    with machine_mode():
        output.warn_fact(_RecordingLogger(), "this deployment is reachable from the network")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "reachable from the network" in captured.err


def test_warn_fact_lands_above_a_mounted_region(live, err_buffer, capsys):
    """Promoted trouble takes the same region-aware seam as ``warn``."""
    reporter, buffer = live
    output.warn_fact(_RecordingLogger(), "this deployment is reachable from the network")
    assert "⚠ this deployment is reachable from the network" in _ANSI.sub("", buffer.getvalue())
    assert err_buffer.getvalue() == ""
    assert capsys.readouterr().err == ""
    assert reporter.is_rendering() is True
