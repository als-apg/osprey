"""The single output renderer for the OSPREY CLI.

Every primitive here is echo-class: it prints a verb's OWN output, so it
survives whichever reporter is installed -- including the ``NullReporter`` that
global ``--verbose`` installs, where the phase record itself is swallowed.
Phase, step and ticker decoration is emit-class and stays in
:mod:`osprey.cli.phase_reporter`; nothing in this module is ever used for it.

The primitives resolve their console at call time, never at import: on a
terminal the installed reporter owns the cursor, and its console is the only
one that can put a line ABOVE a mounted live region instead of inside it. That
holds for the trouble primitives too: :func:`warn` and :func:`fail` are stderr
off a terminal, and follow the region while one is mounted -- joining every
other stderr writer in the process, which a mounted ``Live`` has been proxying
onto the region's console all along. :func:`_target_console` states the
mechanism and what it costs. Under :func:`machine_mode` every primitive writes
to stderr instead, so a ``--json`` verb leaves stdout carrying one pure JSON
document and nothing else.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import PurePath
from typing import TYPE_CHECKING

from rich.text import Text

from . import styles
from .phase_reporter import current_reporter
from .styles import Styles

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rich.console import Console, RenderableType

    from osprey_connectors.logger import ComponentLogger

__all__ = [
    "clear_ledger",
    "fail",
    "flush_ledger",
    "machine_mode",
    "machine_mode_active",
    "note",
    "report",
    "report_fact",
    "section",
    "table",
    "warn",
    "warn_fact",
]

#: The one indentation step. Everything subordinate -- a note, a section's rows
#: -- is one step in from the line it belongs under, which is the same step the
#: phase record already uses for ``  · name`` and ``  ✓ title``. Structure, not
#: color: it is the half of "subordinate" that survives a pipe, where no style
#: reaches the reader at all.
_INDENT = "  "

#: The gap between a section's label column and its values. Wide enough that a
#: ragged label column still reads as two columns rather than as one sentence.
_GUTTER = "   "

#: The narrowest value column :func:`flush_ledger` will fold prose into. Below
#: this a fold produces a column of two-word fragments that is harder to read
#: than an over-long line, so a very narrow terminal gets the over-long line.
_LEDGER_MIN_VALUE_WIDTH = 32

#: What a failure or a refusal opens with -- the same mark
#: :meth:`Messages.error <osprey.cli.styles.Messages.error>` has always used, and
#: the one this module's trouble path spells. One reported failure wears exactly
#: one of these, which is the part that takes care -- see :func:`fail`'s
#: ``mark``, which is how a handler explains a failure the phase record has
#: already marked without putting a second one in front of it.
_FAIL_GLYPH = "✗"

#: What a warning opens with -- the mark the phase record's interrupted line
#: already uses, so a warning reads the same wherever it comes from.
_WARN_GLYPH = "⚠"

#: What marks the line telling the operator what to do about it. Content, not
#: decoration: it survives a pipe, and it is what makes a remedy findable in a
#: scrollback full of causes.
_REMEDY_GLYPH = "→"

#: What opens a fact line -- the same mark the phase record's ``  · name`` steps
#: wear, so a fact reads as part of the run it happened in rather than as a
#: paragraph interrupting it. A fact is not a step: it never touches the phase's
#: lap clock and never carries a duration.
_FACT_GLYPH = "·"

#: True for as long as a :func:`machine_mode` block is open. Process-scoped
#: rather than task- or thread-scoped: the CLI is single-threaded at these call
#: sites, and a ``--json`` verb wraps its whole body in one block.
_machine_mode = False

#: The run ledger: the ``(label, value)`` rows :func:`report_fact` collected for
#: the closing "what this run wrote" block. Process-scoped for the same reason
#: the reporter is a module singleton -- the facts are minted by helpers deep in
#: the call stack, and the verb that owns the run flushes them once at its end.
_ledger: list[tuple[str, str]] = []

#: The heading the flushed ledger prints under. One spelling, here, so the
#: closing block reads identically whichever verb flushes it.
_LEDGER_TITLE = "This deploy wrote"


#: What a URL looks like in a line the CLI prints. Whitespace-delimited, since
#: the values these appear in ("http://127.0.0.1:9080  (landing page)") pad the
#: address with spaces rather than punctuation.
_URL_RE = re.compile(r"https?://\S+")

#: Trailing characters that end a SENTENCE rather than the address in it, so
#: they are handed back to the surrounding text before the link is made.
_URL_TRAILING = ".,;:"


def _linkify(text: str) -> Text:
    """Render ``text`` with every URL in it as a terminal hyperlink.

    Semantic styling by TYPE, which is the rule :func:`section` already follows
    for paths: an address the CLI prints is the one thing in its output that a
    reader wants to ACT on, and a terminal that understands OSC 8 can open it on
    a click instead of on a copy, a paste and a guess about where the address
    ends. Terminals that do not understand the sequence show the same plain
    text, and a console that is not a terminal at all -- a pipe, a captured test
    run -- emits no escape at all, so nothing downstream of this sees a change.
    """
    rendered = Text()
    cursor = 0
    for match in _URL_RE.finditer(text):
        url = match.group().rstrip(_URL_TRAILING)
        rendered.append(text[cursor : match.start()])
        rendered.append(url, style=f"link {url}")
        cursor = match.start() + len(url)
    rendered.append(text[cursor:])
    return rendered


def _region_is_mounted() -> bool:
    """Whether the installed reporter is drawing a live region right now.

    The reporter's own predicate rather than a second copy of it:
    :meth:`PhaseReporter.is_rendering
    <osprey.cli.phase_reporter.PhaseReporter.is_rendering>` is the same answer
    :meth:`~osprey.cli.phase_reporter.PhaseReporter.suspended` branches on, and
    an answer maintained twice is an answer that will disagree with itself.
    Named for what this module is asking, which is where the region is, not what
    the reporter is doing about it.
    """
    return current_reporter().is_rendering()


def _target_console(*, err: bool) -> Console:
    """The console this call must print through.

    Machine mode wins over everything: the point of the block is that stdout
    carries the JSON document and nothing else, so human output goes to stderr
    even where a reporter would have had somewhere better to put it.

    Otherwise stdout output goes through the installed reporter's console --
    which on a terminal is the console the live region was built on, the only
    one Rich can place a line above the region with. Read per call, because
    installing a reporter is what changes the answer.

    Trouble takes ``styles.err_console``, because its stream is part of its
    contract -- EXCEPT while a live region is mounted, where writing at stderr
    damages the screen. Rich unwraps its own ``FileProxy`` for a stderr-bound
    console, so ``Live`` never sees that write: the line lands at whatever
    column the region left the cursor on and welds a dead spinner and a frozen
    ticker to its front, permanently, because the region redraws a row lower and
    never takes that one back. So a mounted region borrows the trouble line for
    as long as it is up, exactly as it borrows the log handler's console (see
    :meth:`LiveReporter._borrow_log_console
    <osprey.cli.phase_reporter.LiveReporter._borrow_log_console>`), and gives it
    back the moment the region comes down.

    THE MECHANISM, and the scope of what this changes: a mounted ``Live``
    already proxies the process's stderr onto the region's console. Everything
    that writes there -- a log record, a library's warning, a traceback on the
    way out -- has been arriving on the region's console, which is stdout's, for
    as long as regions have existed. ``styles.err_console`` was the single
    escapee, because Rich resolves a console's file through
    ``rich_proxied_file`` and so unwraps the very proxy that redirect installs;
    escaping is exactly what welded the line to the frame. This module stops
    escaping, rather than moving a stream that was otherwise being kept.

    The cost is therefore real but bounded, and it is the region's cost rather
    than this function's: while a region is mounted, ``osprey up 2>errors.log``
    on a terminal writes its trouble to the screen and not to the file -- as it
    already did for that run's log records and tracebacks. A region only mounts
    when stdout is a terminal, so a piped or redirected run, where the two
    streams are genuinely separate files and a reader is relying on the
    difference, never mounts one and never takes this path.

    The check and the print are not one atomic step, and that is affordable
    rather than overlooked. A region coming down between them sends the line to
    the console it was about to leave -- the same device stderr would have
    reached, since a region only exists on a terminal. The other direction
    cannot be reached at all: a region is mounted at the verb's entry, before
    anything that could call this from another thread exists. A lock would buy
    an ordering nobody can observe.
    """
    if _machine_mode:
        return styles.err_console
    if err and not _region_is_mounted():
        return styles.err_console
    return current_reporter().out()


def _echo(renderable: RenderableType, *, err: bool = False) -> None:
    """Print one renderable on the never-silenced echo path.

    The same console and the same print keywords :meth:`PhaseReporter.echo
    <osprey.cli.phase_reporter.PhaseReporter.echo>` uses, and for its reasons:
    no markup and no highlighting (paths, brackets and version numbers in a
    verb's output are text, not style tags), no wrapping (callers pass lines
    that are already the shape they want).

    Style rides on a pre-styled :class:`~rich.text.Text` rather than on a
    ``style=`` argument, which is :meth:`LiveReporter.emit
    <osprey.cli.phase_reporter.LiveReporter.emit>`'s precedent: Rich applies a
    ``style=`` to everything the call renders, so while a region is mounted it
    would repaint the table underneath in the line's color.

    ``err`` marks a line whose stream is part of its contract -- the seam the
    trouble primitives print through. It is a claim about the reader, not about
    a file descriptor: :func:`_target_console` still routes it through a mounted
    region's console, because there is no reader on a terminal for whom raw
    stderr would be the better answer.
    """
    _target_console(err=err).print(renderable, markup=False, highlight=False, soft_wrap=True)


def report(text: str, /, *, style: str | None = None) -> None:
    """Print ``text`` as a report line -- a verb's primary output.

    The default altitude for anything an operator ran the verb to find out:
    what was created, where it answers, what changed. Printed under every
    reporter, so ``osprey -v <verb>`` still ends with its verb's own report.

    Pass an empty string for a blank separator line.

    :param text: The line, already in the shape it should appear in.
        Positional-ONLY by API contract -- the copy guard reads prose out of
        positional arguments, so a printed sentence that arrived as a keyword
        would be a sentence the guard never saw. Declaring it positional-only
        makes that unwritable rather than merely discouraged.
    :param style: A semantic token from :class:`~osprey.cli.styles.Styles` for
        the whole line (a card's title line, say). Keyword-only because it is
        not prose; omitted, the line prints in the terminal's own color.
    """
    # The base style, not a span: a URL inside the line carries its own link
    # span from :func:`_linkify`, and a base style layers under both.
    line = _linkify(text)
    if style:
        line.style = style
    _echo(line)


def report_fact(
    logger: ComponentLogger, message: str, /, *, wrote: tuple[str, object] | None = None
) -> None:
    """Print an operator-critical fact in the run's own step column, and keep it.

    The promotion contract, spelled once for every module that carries facts a
    person running the verb has to read: credentials a deploy minted, secret
    files it wrote or renamed, host state it changed, a posture it chose. The
    fact prints as ONE short line in the same indented shape the phase record's
    steps use, so a run reads as one column rather than as steps interrupted by
    paragraphs. The line goes through ``logger.key_info`` too, so file and
    aggregation sinks keep the transcript they always kept; under ``-v`` it
    appears twice, once as the verb's own output and once in the transcript
    that flag asked for.

    ``wrote`` is the fact's second altitude: a ``(label, value)`` row for the
    closing ledger the owning verb flushes after its summary card (see
    :func:`flush_ledger`). The inline line says THAT it happened, in passing;
    the ledger row says exactly what, with the variable names and caveats an
    operator acts on later. Give a row to anything someone will need after the
    run has scrolled by; skip it for postures the inline line already states in
    full. Ledger rows travel in a keyword and a variable, so the copy-style
    guard never reads them; their prose keeps the house rules by review.

    :param logger: The calling module's logger, passed in rather than made here,
        so the record still names the module that holds the fact.
    :param message: The finished inline line, short enough to sit in the step
        column. Positional-ONLY, as for :func:`report`: the copy guard reads
        prose out of positional arguments.
    :param wrote: The ledger row, or ``None`` for an inline-only fact. The value
        is rendered with :func:`str`, like a :func:`section` value.
    """
    line = Text(f"{_INDENT}{_FACT_GLYPH} ", style=Styles.ACCENT)
    line.append(message)
    _echo(line)
    logger.key_info(message)
    if wrote is not None:
        label, value = wrote
        _ledger.append((label, str(value)))
        logger.key_info(f"{label}: {value}")


def warn_fact(
    logger: ComponentLogger, summary: str, detail: str | None = None, remedy: str | None = None, /
) -> None:
    """Print a warning in the run's own column, and keep its record for the sinks.

    :func:`warn`'s counterpart to :func:`report_fact`, for the warnings an
    operator must read while a lifecycle verb runs: the deploy posture that
    changed, the configuration that disagrees with itself. The block is the
    trouble shape -- marked summary, indented body, ``→ remedy`` -- indented one
    step so it reads as part of the run it happened in, and the record goes to
    ``logger.warning`` for the file and aggregation sinks.

    While a lifecycle reporter is installed, the altitude gate keeps raw WARNING
    records off the terminal (see :mod:`osprey.cli.altitude`), so this function
    is how a warning reaches the operator at all; under ``-v`` the gate is
    lifted and the warning appears in both shapes, exactly as a promoted fact
    does.

    :param logger: The calling module's logger, as for :func:`report_fact`.
    :param summary: What went sideways, in one line and without the glyph.
        Positional-ONLY, as for :func:`warn`: the copy guard reads prose out of
        positional arguments.
    :param detail: What an operator needs in order to judge it. Printed indented
        under the summary, line by line. Positional-only for the same reason.
    :param remedy: The one thing to do about it, printed as ``→ remedy`` under
        the detail. Omit it when there is nothing honest to suggest.
    """
    _echo(Text(f"{_INDENT}{_WARN_GLYPH} {summary}", style=Styles.WARNING), err=True)
    body = list(detail.splitlines()) if detail else []
    if remedy:
        body.append(f"{_REMEDY_GLYPH} {remedy}")
    for line in body:
        _echo(_body_text(f"{_INDENT}{_INDENT}", line), err=True)
    logger.warning(" ".join(part for part in (summary, detail, remedy) if part))


def flush_ledger() -> None:
    """Print the collected ledger rows as one closing block, and empty the ledger.

    Called by the verb that owns the run, BEFORE its summary card -- and, on the
    attached ``up`` path, immediately before the reporter's hand-off, because a
    process about to replace itself with compose has exactly one last chance to
    say what it wrote. A run that collected nothing prints nothing.

    Grouped by file rather than laid out as a key/value block, which is what
    :func:`section` renders and what this used to be. Two properties of these
    rows defeat that shape and no other block has either. Their labels REPEAT --
    one deploy writes four separate things into ``.env`` -- so the label column
    prints the same path four times and does no work. And their values are
    PROSE, wide enough to need folding, and a folded value in a two-column block
    is indistinguishable from the next row: the reader cannot tell a continued
    sentence from a new thing written. So each file is named once, on its own
    line, and the things written into it are the facts under it, in the same
    ``·`` shape the run's own steps wear. Consecutive rows sharing a label form
    one group, so the block still reads in the order the run did the writing;
    a file written to, left, and returned to is named twice, which is what
    happened.
    """
    if not _ledger:
        return
    rows, _ledger[:] = list(_ledger), []
    report("")
    _echo(Text(_LEDGER_TITLE, style=Styles.HEADER))
    body_indent = f"{_INDENT}{_INDENT}"
    fold_indent = f"{body_indent}  "
    width = max(_LEDGER_MIN_VALUE_WIDTH, _target_console(err=False).width - len(fold_indent))
    previous: str | None = None
    for label, value in rows:
        if label != previous:
            _echo(Text(f"{_INDENT}{label}", style=Styles.PATH))
            previous = label
        for index, chunk in enumerate(textwrap.wrap(value, width) or [""]):
            if index:
                line = Text(fold_indent)
            else:
                line = Text(body_indent)
                line.append(f"{_FACT_GLYPH} ", style=Styles.ACCENT)
            line.append_text(_linkify(chunk))
            _echo(line)


def clear_ledger() -> None:
    """Drop any rows a previous run left uncollected.

    The verb that installs the lifecycle reporter calls this at entry: the
    ledger is process-scoped, and a run that failed before its flush must not
    donate its rows to the next run in the same process (the test suite runs
    the CLI many times in one process).
    """
    _ledger.clear()


def note(text: str, /) -> None:
    """Print ``text`` as a note -- secondary detail under what came before.

    For the sentence an operator does not need in order to act, but is glad to
    have when they read: what a default was resolved from, what a skipped step
    would have done. Subordinate two ways, because only one of them survives a
    pipe -- indented one step, and dim on a terminal.

    :param text: The note. Positional-only by API contract, as for
        :func:`report`.
    """
    _echo(Text(f"{_INDENT}{text}", style=Styles.DIM))


def section(
    title: str,
    items: Mapping[str, object] | Sequence[tuple[str, object]],
    /,
    *,
    label_style: str = Styles.LABEL,
) -> None:
    """Print a key/value block: ``title``, then one indented row per item.

    The one shape for "here are the facts about this thing" -- an endpoint
    list, a summary card's rows, a config dump. Labels are padded to a common
    width so the values line up as a column; values are rendered with
    :func:`str`, so a caller may hand over paths, ports and booleans as they
    are. The title carries the theme's header token, the same anchor color the
    phase record's arrows use, and a value handed over as a
    :class:`~pathlib.PurePath` renders in the path token -- semantic styling
    by TYPE, never by guessing at strings.

    An empty ``items`` prints the title alone, and an empty ``title`` prints
    the rows alone -- a section that continues the line above it.

    :param title: The block's heading. Positional-only by API contract, as for
        :func:`report`.
    :param items: The rows, as a mapping or as an ordered sequence of
        ``(label, value)`` pairs. A sequence when the order is meaningful and
        labels may repeat; either way the rows print in iteration order.
    :param label_style: The style token for the label column. The default is
        the bold label; :func:`flush_ledger` passes the path token because its
        labels ARE paths by :func:`report_fact`'s ``wrote`` contract.
        Keyword-only: it is not prose, and the copy guard reads prose out of
        the positional arguments.
    """
    rows = list(items.items()) if isinstance(items, Mapping) else list(items)
    if title:
        _echo(Text(title, style=Styles.HEADER))
    width = max((len(label) for label, _ in rows), default=0)
    for label, value in rows:
        line = Text(_INDENT)
        line.append(label.ljust(width), style=label_style)
        line.append(_GUTTER)
        if isinstance(value, PurePath):
            line.append(str(value), style=Styles.PATH)
        else:
            line.append_text(_linkify(str(value)))
        _echo(line)


def table(renderable: RenderableType, /) -> None:
    """Print one built renderable -- a table, a panel, a syntax block.

    The echo-class way to put something that is not a line of text on screen.
    Named for the case that brought it: tables are what the CLI builds most, via
    :func:`~osprey.cli.styles.data_table` and
    :func:`~osprey.cli.styles.layout_table`. Anything Rich can render is
    accepted, and a :class:`~rich.panel.Panel` from
    :func:`~osprey.cli.styles.panel` goes through the same door.

    Printed through the same seam as every other primitive, so a table lands
    above a mounted live region rather than inside it, survives ``osprey -v``,
    and moves to stderr inside :func:`machine_mode` with the lines around it.

    **Cells must be :class:`~rich.text.Text`, never markup strings.** This
    primitive inherits the module's ``markup=False``, and Rich propagates that
    into every nested renderable: a cell holding ``"[success]OK[/success]"``
    prints those brackets literally and takes the column widths with it. Style a
    cell by building ``Text("OK", style=Styles.SUCCESS)``. With ``Text`` cells
    the output is byte-identical to a bare ``console.print(table)`` -- a table
    computes its own column widths, so the module's ``soft_wrap`` changes
    nothing about how it renders.

    :param renderable: A built Rich renderable. Positional-only for uniformity
        with the rest of the module, though there is no prose here for the copy
        guard to read.
    """
    _echo(renderable)


def _body_text(indent: str, line: str) -> Text:
    """Build one indented body line, with a remedy's arrow in the accent.

    The arrow is the one glyph in a trouble body that is an instruction rather
    than evidence, and the accent is what lets the eye find it without reading
    the block. The words are untouched: off a terminal the line prints exactly
    as composed.
    """
    text = Text(indent)
    remedy_head = f"{_REMEDY_GLYPH} "
    if line.startswith(remedy_head):
        text.append(remedy_head, style=Styles.ACCENT)
        text.append(line[len(remedy_head) :])
    else:
        text.append(line)
    return text


def _trouble(glyph: str, style: str, summary: str, body: Sequence[str]) -> None:
    """Print one trouble event: a marked summary line, then its indented body.

    The one shape for everything that went wrong, so that a reader who has seen
    one failure can read every other one. Only the summary line carries the
    color token -- a cause printed in red is a cause competing with the line
    that says what failed, and dim would push it below the altitude an operator
    reads at. Structure carries the subordination instead, which is also the
    half of it that survives a pipe.

    Stderr, because the stream is part of the contract for trouble: a caller
    piping stdout somewhere keeps its warnings and failures on the terminal.
    While a live region is mounted the lines go through the region's console
    instead, which is what puts them ABOVE it in one piece -- see
    :func:`_target_console` for the mechanism and what it costs.

    :param glyph: The mark opening the summary line, or an empty string for a
        failure that is already marked somewhere else -- see :func:`fail`'s
        ``mark``. The summary then opens with its own first word, and nothing
        else about the block changes.
    :param style: The :class:`~osprey.cli.styles.Styles` token for that line.
    :param summary: What happened, in one line.
    :param body: The already-prefixed lines under it, each printed one
        :data:`_INDENT` in.
    """
    _echo(Text(f"{glyph} {summary}" if glyph else summary, style=style), err=True)
    for line in body:
        _echo(_body_text(_INDENT, line), err=True)


def warn(summary: str, detail: str | None = None, /) -> None:
    """Print a warning: something the operator should know went sideways.

    For what the verb worked around rather than stopped for -- a key it could
    not read and defaulted, a step it skipped, a deprecation it honored anyway.
    Use :func:`fail` when the verb is not going to deliver what it was asked
    for.

    Prints to stderr under every reporter -- above a live region rather than
    onto it while one is mounted -- and inside :func:`machine_mode` like
    everything else here, so a ``--json`` caller never finds a warning in its
    document.

    :param summary: What went sideways, in one line and without the glyph.
        Positional-ONLY, as for :func:`report`: the copy guard reads prose out
        of positional arguments, so a keyword would hide a printed sentence
        from it.
    :param detail: What an operator needs in order to judge it -- the value
        that was rejected, what was used instead. Printed one indent step
        under the summary; multi-line detail is indented line by line.
        Positional-only for the same reason.
    """
    _trouble(_WARN_GLYPH, Styles.WARNING, summary, detail.splitlines() if detail else ())


def fail(
    summary: str,
    cause: str | None = None,
    remedy: str | None = None,
    /,
    *,
    mark: bool = True,
) -> None:
    """Print a failure or a refusal: ``✗ summary``, its cause, its remedy.

    The single shape for every way the CLI declines to deliver -- an error it
    caught, a precondition it will not proceed without, a write it refuses.
    Whichever it is, the operator reads the same three parts: what failed, why,
    and what to do about it.

    This function only prints. It does not raise, exit, or set an exit code:
    the caller stays in charge of its own :class:`click.Abort` and its own exit
    status, so adopting the shape never changes what a script sees.

    :param summary: What failed, in one line and without the glyph.
        Positional-ONLY, as for :func:`report`.
    :param cause: Why -- the bound port, the missing key, the exception's
        message. Printed one indent step under the summary; multi-line cause is
        indented line by line. Positional-only.
    :param remedy: The one thing to do about it, printed as ``→ remedy`` under
        the cause. Omit it when there is nothing honest to suggest -- a remedy
        that does not work costs more than none. Positional-only.
    :param mark: Whether to open with ``✗``. Pass ``False`` where the failure
        already wears one: a refusal raised inside an open phase leaves through
        :meth:`Phase.fail <osprey.cli.phase_reporter.Phase.fail>`, which has
        printed ``✗ title`` for it already. The mark means "here is a thing that
        failed", and there is one thing to say it about -- a second mark reads
        as a second failure, so the handler that says WHY prints the same block
        unmarked. What continues is the READING, not the record: the phase line
        is the reporter's and goes to stdout, this block is trouble and goes to
        stderr as it always does, so the two are one failure to somebody
        watching a terminal and two independent records to anything reading the
        streams apart. The block keeps this function's own geometry too --
        column-zero summary, cause one step in -- rather than nesting under the
        phase line it follows. Only the glyph is dropped. Keyword-only, because
        it is not prose and the copy guard reads prose out of the positional
        arguments.
    """
    body = list(cause.splitlines()) if cause else []
    if remedy:
        body.append(f"{_REMEDY_GLYPH} {remedy}")
    _trouble(_FAIL_GLYPH if mark else "", Styles.ERROR, summary, body)


@contextmanager
def machine_mode() -> Iterator[None]:
    """Send every renderer line to stderr for as long as the block runs.

    What a ``--json`` verb wraps its body in. The caller writes its document to
    stdout and everything else in the process -- a warning from three frames
    down, a note a shared helper prints -- lands on stderr, so the reader of
    stdout parses one document rather than a document with prose in it.

    Reentrant: a nested block restores the state it found rather than switching
    the mode off, so a ``--json`` verb calling another one keeps the redirect
    for its whole body.

    Process-scoped, and deliberately so -- the primitives are called from
    helpers deep in the call stack that hold no handle on the verb, which is
    the same reason the reporter itself is a module singleton.
    """
    global _machine_mode
    previous, _machine_mode = _machine_mode, True
    try:
        yield
    finally:
        _machine_mode = previous


def machine_mode_active() -> bool:
    """Whether a :func:`machine_mode` block is open right now.

    Lets a call site that mounts a console of its own -- a progress spinner, a
    live region -- pick the stderr console while a ``--json`` path is running,
    rather than painting over the document on stdout. The primitives here
    already do this for themselves; this is for the ones that cannot.
    """
    return _machine_mode
