"""The live build region: a build snapshot in, one Rich renderable out.

A cold ``compose build`` spends a quarter of an hour inside a single phase, and
the plain ``→ Building services`` line that opens it says nothing for the rest of
that time. On a terminal a live region is therefore drawn beneath that line: a
spinner with a ticking duration, and under it one row per service, fed from
:class:`osprey.deployment.build_progress.BuildModel`:

.. code-block:: text

    → Building services                    <- printed once, permanently
      · compose config validated (1.2s)    <- printed once, permanently
      ⠹ 4m12s                              <- from here down, the live region

        virtual-accelerator   6/8         Downloading torch (2.1 GB)…   4m02s
        bluesky-bridge        3/8         RUN pip install -e .          1m18s
        event-dispatcher      exporting   exporting layers              3m41s

The region never repeats the phase's title. The phase announced itself with a
permanent ``→ title`` line the moment it opened, and a title redrawn underneath
that line reads as a rendering glitch rather than as progress; what the region
adds is the part that could not be printed once — a spinner and a duration that
keep moving. Printing the title exactly once also keeps the terminal's scrollback
identical to the output of a run that was piped to a file.

The ticker sits at the indent of the ``  · step`` lines the phase reporter
prints, so it reads as belonging to the ``→ title`` above it, and the rows sit
one level deeper still, so the table reads as nested under the phase rather than
as a peer of it.

A service leaves the table the moment its image is finished. Its completion is
already in the scrollback as a step line of its own, permanently, so repeating
it here says nothing new while crowding out what the operator is actually
waiting on — and the table shrinks as the build lands each image. It keeps this
region honest against the off-terminal heartbeat too, which likewise stops
reporting a service once its image is built.

This module is the rendering half and nothing else: it owns no thread, no
:class:`~rich.live.Live`, no console and no clock. Both timings arrive as
arguments — the phase's ``elapsed`` and the caller's ``now`` — for the same
reason :meth:`BuildModel.feed` takes its timestamp: the caller measures time
once, and a fifteen-minute build renders identically from a fixture.

Every style here is a semantic token from :mod:`osprey.cli.styles`, resolved
through the active Rich theme. Nothing names a color, so a retheme (or
``osprey theme-lab``) reaches this region like it reaches the rest of the CLI.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from osprey.deployment.build_progress import EXPORT_STEP, BuildRow

from .styles import Styles, ThemeConfig

#: Columns the service rows are indented by. The phase reporter's sub-steps sit
#: at 2, so 4 nests the table one level below them.
_INDENT = 4

#: Columns the ticker is indented by — the reporter's own ``  · step`` indent,
#: so the ticker reads as one more line belonging to the phase above it rather
#: than as the start of something new.
_TICKER_INDENT = 2

#: Blank columns between two adjacent table columns.
_GUTTER = 3

#: The width the step column is held at. The step vocabulary is closed — a
#: Dockerfile index, a header word (``internal``, ``auth``), or
#: :data:`EXPORT_STEP` — and ``exporting`` is the widest of them. A build opens
#: on ``None``/``internal``/``auth`` and only reaches ``exporting`` a quarter of
#: an hour later, so a column sized from whatever has arrived would widen under
#: the operator's eyes and shift every caption with it. Pinning it costs a few
#: blank columns early and buys a table that never reflows.
_STEP_WIDTH = len(EXPORT_STEP)

#: The caption never shrinks below this, however narrow the terminal is. A
#: caption clipped to a single ellipsis says less than a clipped word does.
_MIN_CAPTION = 8

#: Shown for a service whose step is not known yet — ``status_display``'s
#: spelling for a cell with nothing in it.
_NO_STEP = "-"


def format_duration(seconds: float) -> str:
    """Format a duration for the live region, always as ``MmSSs``.

    Deliberately not :func:`osprey.cli.phase_reporter.format_elapsed`, which
    switches from ``0.4s`` to ``1m20s`` at a minute. That switch is right for a
    line printed once, and wrong for a counter repainted every fraction of a
    second: the width would jump around under the operator's eyes, and a build
    is measured in minutes anyway.

    Args:
        seconds: Elapsed seconds; negative values are treated as zero.

    Returns:
        The duration as ``4m02s``.
    """
    minutes, secs = divmod(max(0, int(seconds)), 60)
    return f"{minutes}m{secs:02d}s"


def render_live_region(
    rows: Sequence[BuildRow],
    *,
    phase_open: bool,
    elapsed: float,
    now: float,
    width: int,
) -> RenderableType:
    """Build the renderable for the live region.

    Pure: it reads the arguments and returns a renderable. It never calls
    :meth:`BuildModel.snapshot` itself — the caller passes the rows it already
    holds, which is what keeps the model's lock off the render path.

    Args:
        rows: One :class:`~osprey.deployment.build_progress.BuildRow` per
            service, in the order they should appear. Usually a
            :meth:`BuildModel.snapshot`. Rows whose image is already finished
            are dropped, so the table lists exactly what is still outstanding
            and shrinks as the build lands each image.
        phase_open: Whether a phase is open, and so whether the region carries a
            ticker at all. A boolean rather than the phase's title, because the
            title is not drawn here: it is already in the scrollback, printed
            once, by the line that opened the phase.
        elapsed: Seconds the open phase has been running, for its ticker.
        now: The caller's current :func:`time.monotonic` reading, against which
            each row's age is measured.
        width: Columns the region may occupy — normally ``console.width``. Only
            the caption gives way to it; the other columns are sized by content.

    Returns:
        A renderable of exactly zero height when there is no open phase and no
        rows, so an idle live region leaves no blank line behind.
    """
    parts: list[RenderableType] = []
    if phase_open:
        parts.append(_phase_line(elapsed))
    pending = [row for row in rows if row.finished is None]
    if pending:
        if parts:
            parts.append(Text(""))
        parts.append(Padding(_row_table(pending, now, width), (0, 0, 0, _INDENT), expand=False))
    return Group(*parts)


def _phase_line(elapsed: float) -> RenderableType:
    """The open phase's ticker: ``  <spinner> <elapsed>``.

    Title-less by design — see the module docstring: the phase already named
    itself in the scrollback, and this line exists for the two things that could
    not be printed once.

    A grid rather than a single :class:`~rich.text.Text` because the spinner is
    a renderable of its own; the spacing lives in the cells so the line reads
    the way it prints. The spinner matches the one the health suite already
    shows — same name, same themed style.
    """
    line = Table.grid()
    line.add_column()
    line.add_column()
    line.add_row(
        Spinner("dots", style=ThemeConfig.get_spinner_style()),
        Text(f" {format_duration(elapsed)}", style=Styles.DIM),
    )
    return Padding(line, (0, 0, 0, _TICKER_INDENT), expand=False)


def _row_table(rows: Sequence[BuildRow], now: float, width: int) -> RenderableType:
    """The borderless service table: service, step, caption, age.

    Column styles follow ``status_display._create_status_table``'s precedent —
    the identifying column in accent, the trailing detail dimmed.
    """
    cells = [
        (row.service, row.step or _NO_STEP, row.caption, format_duration(now - row.started_at))
        for row in rows
    ]
    table = Table(box=None, show_header=False, padding=(0, 0, 0, _GUTTER), pad_edge=False)
    table.add_column(style=Styles.ACCENT, no_wrap=True)
    table.add_column(style=Styles.SUCCESS, no_wrap=True, min_width=_STEP_WIDTH)
    table.add_column(
        style=Styles.INFO,
        no_wrap=True,
        overflow="ellipsis",
        max_width=_caption_width(cells, width),
    )
    table.add_column(style=Styles.DIM, no_wrap=True)
    for service, step, caption, age in cells:
        # Plain Text, never markup: a caption is BuildKit's output, and one
        # holding a `[...]` would otherwise be read as a style tag.
        table.add_row(Text(service), Text(step), Text(caption), Text(age))
    return table


def _caption_width(cells: Sequence[tuple[str, str, str, str]], width: int) -> int:
    """How much room the caption column gets once the other three are placed.

    The caption is the only column allowed to lose text: a truncated service
    name or duration is worse than useless, while a truncated caption still
    names what the step is doing. Rich clips it with an ellipsis from here.

    The step column is measured at its pinned width rather than its contents,
    so the caption budget stays put as the steps under it change shape.
    """
    service = max(len(cell[0]) for cell in cells)
    step = max(_STEP_WIDTH, max(len(cell[1]) for cell in cells))
    age = max(len(cell[3]) for cell in cells)
    return max(_MIN_CAPTION, width - _INDENT - service - step - age - 3 * _GUTTER)


__all__ = ["format_duration", "render_live_region"]
