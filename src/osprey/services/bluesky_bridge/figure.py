"""The figure spec: one drawing vocabulary shared by the panel and the agent.

A plan's ``render(window, params)`` returns a `Figure`: a list of `Panel`s, each
carrying exactly one mark (`LinesMark`, `HeatmapMark`, or `BarsMark`) alongside
its title, axis labels/units, and annotations. The bridge serves that object as
JSON from `GET /runs/{run_id}/figure`; the panel's ``figure-renderer.js`` draws
it and the MCP ``get_run_figure`` tool projects it for the agent. Both consumers
dispatch on ``mark.kind`` and read snake_case field names straight out of
`model_dump()`, so this module is the contract between them: neither can be
type-checked against it, and a renamed field breaks them silently.

Every figure is honest about what it shows. `Figure.partial` says whether the
run was still producing rows, `Figure.source` says which store those rows came
from, and `Figure.reason` -- one of the ``REASON_*`` constants -- says why a
default figure stands in for a plan's own view. Every mark reports
``decimated``/``source_points`` so a thinned series is never mistaken for the
whole dataset.

Non-finite floats (``NaN``, ``inf``) in nullable value fields are normalized to
``None`` during validation. They are nothing a renderer can draw, and
``json.dumps`` writes them as bare ``NaN``/``Infinity`` -- not valid JSON, and a
``JSON.parse`` failure in the browser rather than a visible gap in the plot. The
same value on a required axis coordinate is a `ValidationError` instead: a point
with no x has no place on an axis at all.

Kept pydantic-only (no bluesky/ophyd/tiled imports), like `plan_metadata.py`, so
plan modules and the bridge process can both import it without dragging the
RunEngine stack into either one. `plan_fields`, its one sibling import, holds
the same line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from .plan_fields import resolve_column

# --- Reason vocabulary -------------------------------------------------------
#
# Set on `Figure.reason` when the served figure is the bridge's default view
# rather than the one the plan asked for. Stable strings: the panel and the
# agent narrate from them, so they are part of the wire contract.

REASON_NO_RENDER = "no_render"
"""The plan declares no ``render()`` -- the default figure *is* its view."""

REASON_RENDER_FAILED = "render_failed"
"""The plan's ``render()`` raised; the default figure stands in for it."""

REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS = "render_not_supported_for_session_plans"
"""The plan came from the session/unreviewed layer, whose ``render()`` is never run."""

REASON_PARAMS_MISMATCH = "params_mismatch"
"""The stamped parameters did not validate against the plan's PARAMS schema."""

REASON_PLAN_IDENTITY_UNAVAILABLE = "plan_identity_unavailable"
"""The run could not be tied back to a known plan, so no ``render()`` applies."""

REASON_SOURCE_UNAVAILABLE = "source_unavailable"
"""Reading the rows failed; the figure is empty but the response is still a 200."""

FIGURE_REASONS: frozenset[str] = frozenset(
    {
        REASON_NO_RENDER,
        REASON_RENDER_FAILED,
        REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS,
        REASON_PARAMS_MISMATCH,
        REASON_PLAN_IDENTITY_UNAVAILABLE,
        REASON_SOURCE_UNAVAILABLE,
    }
)
"""Every reason the bridge may set. `Figure.reason` is left unvalidated against
this set so an unforeseen reason still serves a figure rather than a 500; use it
in tests and in narration lookups."""


# --- Numeric normalization ---------------------------------------------------


def _finite_or_none(value: Any) -> Any:
    """Coerce ``value`` to a plain ``float``, mapping non-finite results to ``None``.

    Numpy scalars and ints become Python floats so `model_dump()` yields
    JSON-native types; ``NaN``/``inf`` become ``None`` (a gap the renderer can
    draw); anything that isn't number-like is passed through untouched for
    pydantic to report as a type error.
    """
    if value is None or isinstance(value, bool):
        return value
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    return as_float if math.isfinite(as_float) else None


def _require_finite(value: Any) -> Any:
    """Coerce ``value`` to a plain ``float``, rejecting non-finite results.

    Used for axis coordinates, where a ``NaN`` is not a gap but a point with no
    position: better a `ValidationError` the render contract turns into the
    default figure than an unplottable coordinate on the wire.
    """
    if value is None or isinstance(value, bool):
        return value
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(as_float):
        raise ValueError("must be a finite number, got NaN or infinity")
    return as_float


NullableFloat = Annotated[float | None, BeforeValidator(_finite_or_none)]
"""A measured value, or ``None`` for a gap. Non-finite input becomes ``None``."""

FiniteFloat = Annotated[float, BeforeValidator(_require_finite)]
"""An axis coordinate. Non-finite input is a validation error."""


class _FigureNode(BaseModel):
    """Shared config for every node in the figure tree.

    ``extra="forbid"`` because a misspelled field is a mark the renderer draws
    wrong (or not at all) rather than an error anyone sees: caught here, it is a
    `ValidationError` inside the plan's ``render()`` and the run still gets its
    default figure.
    """

    model_config = ConfigDict(extra="forbid")


# --- Marks -------------------------------------------------------------------


class Point(_FigureNode):
    """One sample in a line series."""

    x: FiniteFloat
    y: NullableFloat = None
    """The measured value, or ``None`` for a gap the renderer draws as a break
    in the line rather than a jump through zero."""


class Series(_FigureNode):
    """One named line within a `LinesMark`."""

    label: str
    points: list[Point] = Field(default_factory=list)
    decimated: bool = False
    """Whether ``points`` was thinned from a larger set."""

    source_points: int
    """How many points this series was built from, before any decimation.
    Equals ``len(points)`` when ``decimated`` is false."""


class LinesMark(_FigureNode):
    """A set of x/y series drawn as lines."""

    kind: Literal["lines"] = "lines"
    series: list[Series] = Field(default_factory=list)
    decimated: bool = False
    """True when *any* series was thinned; per-series truth is on `Series`."""

    source_points: int
    """Total points across every series before decimation."""


class HeatmapMark(_FigureNode):
    """A labelled 2-D grid of values, drawn as colored cells."""

    kind: Literal["heatmap"] = "heatmap"
    x_labels: list[str] = Field(default_factory=list)
    y_labels: list[str] = Field(default_factory=list)
    values: list[list[NullableFloat]] = Field(default_factory=list)
    """Row-major: ``values[j][i]`` is the cell at ``y_labels[j]`` /
    ``x_labels[i]``. A ``None`` cell is one that was never measured and is drawn
    as empty, not as zero."""

    value_label: str = ""
    """What the cell values *are* -- the heatmap's third axis (e.g. ``"slope"``)."""

    value_units: str | None = None
    decimated: bool = False
    source_points: int
    """Number of cells the heatmap was built from, before any summarization."""


class BarsMark(_FigureNode):
    """One value per named category, drawn as bars."""

    kind: Literal["bars"] = "bars"
    label: str = ""
    categories: list[str] = Field(default_factory=list)
    values: list[NullableFloat] = Field(default_factory=list)
    """Positionally aligned with ``categories``; ``None`` is a missing bar."""

    decimated: bool = False
    source_points: int
    """Number of categories before any decimation."""


Mark = Annotated[LinesMark | HeatmapMark | BarsMark, Field(discriminator="kind")]
"""The mark kinds a panel may carry, tagged by ``kind`` so a dumped figure
round-trips back through `Figure.model_validate` as the same mark type."""


# --- Panels and figures ------------------------------------------------------


class Panel(_FigureNode):
    """One plot: a mark plus everything needed to read it."""

    title: str
    x_label: str = ""
    y_label: str = ""
    x_units: str | None = None
    y_units: str | None = None
    annotations: list[str] = Field(default_factory=list)
    """Short operator-facing notes about what the panel does *not* show -- a
    decimation cap, dropped rows, series omitted past a limit. Rendered as text
    beside the plot, so they must read as sentences, not as codes."""

    mark: Mark
    """Exactly one mark. A panel showing two things is two panels."""

    section: str | None = None
    """Groups this panel with every other panel carrying the same string into
    one collapsed disclosure below the unsectioned panels, in first-appearance
    order. ``None`` -- the default, and what every other plan emits -- renders
    the panel inline, exactly as before this field existed.

    A section holding more than one panel shows *one at a time*, behind a
    picker: sections exist for detail a reader consults about a chosen device,
    not for a wall of plots to scroll past."""

    series_picker: bool = False
    """Whether the reader may choose which of this panel's series are drawn.
    Only meaningful for a `LinesMark` with labelled series. The panel ships
    every series either way -- the picker filters what is *drawn*, so toggling
    it costs no fetch."""


class Figure(_FigureNode):
    """A complete plan view: the payload of `GET /runs/{run_id}/figure`.

    JSON-serializable via `model_dump()`, which is how both the panel and the
    MCP projection consume it.
    """

    panels: list[Panel] = Field(default_factory=list)
    partial: bool
    """True while the run may still produce rows -- the panel keeps polling."""

    source: Literal["live", "tiled"]
    """Which store the rows came from: the in-process live buffer, or Tiled."""

    reason: str | None = None
    """Why this is the default figure instead of the plan's own view. ``None``
    means the plan rendered it; otherwise one of `FIGURE_REASONS`."""


# --- Decimation --------------------------------------------------------------

DEFAULT_MAX_POINTS = 2000
"""Points per series the panel-facing route serves. The MCP projection spends
the same number across the *whole* figure instead."""


class DecimatedPoints(NamedTuple):
    """What `decimate` kept, and the truth about what it dropped.

    The field names are `Series`'s, so a caller can splat the result::

        Series(label="bpm3", **decimate(points)._asdict())
    """

    points: list[Point]
    decimated: bool
    source_points: int


def decimate(
    points: Sequence[Point],
    max_points: int = DEFAULT_MAX_POINTS,
    *,
    source_points: int | None = None,
    decimated: bool = False,
) -> DecimatedPoints:
    """Thin ``points`` to at most ``max_points``, keeping both endpoints.

    A run with 200k rows is not 200k pixels wide, and neither the browser nor
    the agent's context can afford the difference. This picks an evenly spaced
    subset instead: the first and last points verbatim, so the series still
    spans the x range it actually covered, and one representative from each of
    the ``max_points - 2`` equal windows in between.

    **Gaps survive.** A window whose points include a ``y is None`` gap is
    represented *by that gap* rather than by a measured value, so missing data
    is never smoothed over by the stride -- the drawn line breaks wherever the
    source had nothing to draw. This over-reports gap width (one dropout in a
    window of ten renders as a window-wide break) and never under-reports it,
    which is the direction that keeps the picture honest: the figure may say
    "no data here" where the data was merely sparse, but it never claims a
    reading the run did not take. The first gap in the window wins; the two
    endpoints are always the source's own endpoints, gap or not.

    **Stride is positional, not by x value.** Points are taken every ``n /
    max_points`` samples in sequence order, which is even in x for the
    acquisition-ordered series this module builds. A series whose x samples
    bunch up keeps that bunching -- decimation preserves the shape of the
    sampling, it does not resample onto a uniform axis.

    **Re-decimation composes** only if the caller passes the prior truth in.
    ``source_points`` always means the count *before any* decimation, so the
    MCP projection re-striding an already-thinned series to a smaller budget
    must hand back what it was told::

        thinner = decimate(
            series.points,
            budget,
            source_points=series.source_points,
            decimated=series.decimated,
        )

    Without those two arguments the input is taken to be the raw, untouched
    data, and the returned ``source_points`` would shrink to the already-thinned
    count -- a figure claiming it drew everything it had.

    Args:
        points: The series' points, in the order they were acquired.
        max_points: Most points to keep. Must be at least 1. Budgets of 1 and 2
            are degenerate but legal (the MCP projection can allocate them when
            a figure carries more series than points): 1 keeps the first point,
            2 keeps both endpoints, and neither leaves room for the gap rule.
        source_points: How many points the input was itself built from, when it
            has already been decimated. Defaults to ``len(points)``.
        decimated: Whether the input has already been thinned. The returned flag
            is this ``or`` whatever this call does.

    Returns:
        A `DecimatedPoints` with the kept points -- a new list, never the input
        object -- and the composed ``decimated`` / ``source_points`` truth.

    Raises:
        ValueError: If ``max_points`` is below 1, or ``source_points`` is
            smaller than the number of points given (which would report a
            series as larger after decimation than before it).
    """
    if max_points < 1:
        raise ValueError(f"max_points must be at least 1, got {max_points}")

    count = len(points)
    original = count if source_points is None else source_points
    if original < count:
        raise ValueError(
            f"source_points ({original}) is smaller than the {count} points given; "
            "it counts the points before decimation, not after"
        )

    if count <= max_points:
        return DecimatedPoints(list(points), decimated, original)

    kept = [points[0]]
    windows = max_points - 2  # <= 0 for the degenerate budgets, skipping the loop
    span = count - 2  # every interior window holds at least one point
    for i in range(windows):
        start = 1 + (i * span) // windows
        stop = 1 + ((i + 1) * span) // windows
        gap = next((j for j in range(start, stop) if points[j].y is None), None)
        kept.append(points[gap if gap is not None else start + (stop - start) // 2])
    if max_points >= 2:
        kept.append(points[-1])

    return DecimatedPoints(kept, True, original)


# --- Row windows -------------------------------------------------------------


class RowWindow(NamedTuple):
    """The rows a figure is being built from, and how much of the run they are.

    ``rows_complete`` is the load-bearing field: it says whether these rows are
    the run's *whole* dataset or a window onto it. A renderer that indexes rows
    positionally -- the orm render deriving which corrector a row belongs to
    from its position in the sweep -- is only sound over a complete set, and
    must fall back to something position-free when this is ``False``.
    """

    rows: list[dict[str, Any]]
    columns: list[str]
    rows_complete: bool
    total_seen: int


def rows_from_columnar(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    total_seen: int,
) -> RowWindow:
    """Turn the bridge's columnar row payload into per-row dicts.

    Both row sources -- the in-process live buffer and Tiled -- reach a plan's
    ``render()`` through here, so this is where their two spellings of "no
    reading" are reconciled: the live buffer stores a missing value as ``None``
    (an Event doc simply had no key for that column), while a Tiled table, being
    a numeric array, has no ``None`` to store and surfaces the same absence as a
    float ``NaN``. Downstream everything -- the plan renders, the default
    figure, the live-vs-Tiled parity the panel depends on -- sees ``None`` and
    only ``None``, so a run replayed from Tiled draws the same figure it drew
    live. Infinities are left alone: an ``inf`` is a value the run produced, not
    a reading it failed to take, and the figure models null it at the drawing
    edge anyway.

    **Emission order is the caller's to establish.** Rows are mapped in the
    order given and never sorted -- the live buffer appends in Event order, and
    a Tiled caller must read its table sorted by ``seq_num`` -- because sorting
    here would need a key column this payload deliberately projects away.

    Args:
        columns: The column names, in the order each row's values are stored.
        rows: The window's rows, each a sequence of values positionally
            matching ``columns``.
        total_seen: How many rows the run has produced in total, which is more
            than ``len(rows)`` when the window is paginated or the buffer hit
            its retention cap.

    Returns:
        A `RowWindow` whose ``rows_complete`` is exact: ``True`` only when the
        rows given are every row the run produced.

    Raises:
        ValueError: If a row's width does not match ``columns`` (a corrupt
            buffer, not a figure), or if ``total_seen`` is smaller than the
            number of rows given (a window larger than the run it windows).
    """
    if total_seen < len(rows):
        raise ValueError(
            f"total_seen ({total_seen}) is smaller than the {len(rows)} rows given; "
            "it counts every row the run produced, not just this window's"
        )

    mapped = [
        {
            name: (None if isinstance(value, float) and math.isnan(value) else value)
            for name, value in zip(columns, values, strict=True)
        }
        for values in rows
    ]

    return RowWindow(mapped, list(columns), len(mapped) == total_seen, total_seen)


# --- Default figure ----------------------------------------------------------

MAX_DEFAULT_SERIES = 12
"""How many columns the default figure draws. Past this the panels stop being
readable and the payload stops being cheap; the raw table below the plot still
lists every column, and the figure says how many it left out."""


def _numeric_value(value: Any) -> float | None:
    """Return ``value`` as a finite ``float``, or ``None`` if it is not one.

    ``None`` here means "nothing to draw at this row": a missing reading, a
    string, a boolean flag, a ``NaN`` that survived the adapter, or an ``inf``
    -- which is a value the run produced but not one an axis has a place for.
    Anything number-like (including numpy scalars and ints) converts.
    """
    if value is None or isinstance(value, bool | str | bytes):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return as_float if math.isfinite(as_float) else None


def _role_columns(names: Sequence[str], columns: Sequence[str]) -> list[str]:
    """The distinct data columns *names* resolve to, in the order they are named.

    A channel that has no column in this window -- never read, or read after
    the window was taken -- contributes nothing, so the result is shorter than
    the input rather than carrying a hole.
    """
    ordered: dict[str, None] = {}
    for name in names:
        column = resolve_column(name, columns)
        if column is not None:
            ordered.setdefault(column, None)
    return list(ordered)


def _x_axis_column(window: RowWindow, movable: Sequence[str]) -> str | None:
    """The column the run's own x axis lives in, or ``None`` for row index.

    Only a run with exactly ONE movable column has an x axis the default figure
    can name. A plan that moves several channels -- the orm's serial sweeps,
    where every corrector in turn is driven through the same run -- has no
    single independent variable, and plotting against one of them would draw
    the other sweeps as vertical smears through it. Row index is the honest
    answer there, and it is also the answer when the declaration resolves to
    nothing: a run stamped before roles existed, a plan that declares none, a
    channel whose column never appeared.

    The one column must also hold at least one finite number somewhere in the
    window. A movable that reads back as a string, or not at all, is a column
    with no positions in it, and every point drawn against it would be dropped.
    """
    resolved = _role_columns(movable, window.columns)
    if len(resolved) != 1:
        return None
    column = resolved[0]
    if not any(_numeric_value(row.get(column)) is not None for row in window.rows):
        return None
    return column


def default_figure(
    window: RowWindow,
    *,
    reason: str | None,
    partial: bool,
    source: Literal["live", "tiled"],
    movable: Sequence[str] = (),
    readable: Sequence[str] = (),
) -> Figure:
    """Build the bridge's stand-in view: every numeric column against the run's x.

    This is what a run gets when no plan ``render()`` produced a figure for it
    -- the plan declares none, its render raised, its parameters did not
    validate, its identity was lost. It answers the one question a plan-agnostic
    view can: what did each recorded number do over the course of the run.

    **A column is drawn when at least one row holds a finite number for it**, and
    it is drawn over *every* row: a column that reads values through row 49 and
    ``None`` afterwards is one series with a gap, never a series that quietly
    ends and never a dropped column. The alternative -- deciding numericness
    from the rows currently in hand -- makes the figure's shape depend on when
    the panel polled, so a column would appear and disappear between ticks. Rows
    that hold a string, a boolean, or a non-finite float are gaps for the same
    reason: the renderer breaks the line rather than inventing a position.

    **x is the plan's own movable column when the plan declared exactly one**,
    and the row's index in ``window.rows`` otherwise. The declaration is the
    only thing that can say which column is the independent one -- there is no
    reading a column's values can do it -- so ``movable``/``readable`` are the
    channel names the run's parameters supplied for those roles
    (`plan_fields.collect_channels`), matched to columns by
    `plan_fields.resolve_column`. Pass nothing and the figure is exactly what it
    was before roles existed: every numeric column against row index, in column
    order. See `_x_axis_column` for when a declaration is *not* usable.

    Role context applies as a unit. With a usable x column the movable's own
    column becomes the axis rather than a panel, the columns the plan declared
    readable are drawn first (so they are the ones that survive the series cap),
    and the rest of the numeric columns follow in column order behind them --
    dropped from the figure never, reordered only. Rows holding no reading for
    the x column have no position on that axis and are left out of every series,
    which the first panel says in as many rows.

    Points stay in acquisition order rather than being sorted by x: a plan that
    sweeps its movable more than once in a run draws the retrace it actually
    performed instead of a line stitched across passes.

    Series past `MAX_DEFAULT_SERIES` are dropped in that order and the count of
    what was dropped is annotated on the first panel, so the figure never
    implies it is showing every column. A window that is only part of its run
    (``rows_complete`` false) is annotated the same way.

    **Never raises.** Every fallback path in the figure route ends here,
    including the one that got here *because* something else raised, so an empty
    window, a run with no columns, or a table of nothing but strings all produce
    a valid `Figure` with no panels rather than a second failure.

    Args:
        window: The rows to draw, their column order, and how much of the run
            they are.
        reason: Why the default figure is standing in -- one of the ``REASON_*``
            constants, or ``None`` if some caller genuinely means "no reason".
            Required rather than defaulted: a default figure that does not say
            why it is the default is the thing this vocabulary exists to prevent.
        partial: Whether the run may still produce rows. The caller owns this:
            it knows whether the stop document has arrived, and this function
            only sees a window.
        source: Which store the rows came from. On a source-read failure, pass
            the store that was being read -- the empty figure still says where
            it was looking.
        movable: Channel names the run's parameters supplied for the movable
            role, in declaration order. Empty -- the default -- means no
            declaration reached here and x is the row index.
        readable: Channel names supplied for the readable role. Used only to
            order the panels, and only when ``movable`` yielded an x column.

    Returns:
        A `Figure` of one single-series lines `Panel` per drawn column, each
        series decimated to `DEFAULT_MAX_POINTS`.
    """
    x_column = _x_axis_column(window, movable)

    drawn = [
        name
        for name in window.columns
        if name != x_column
        and any(_numeric_value(row.get(name)) is not None for row in window.rows)
    ]
    if x_column is not None:
        drawable = set(drawn)
        leading = [name for name in _role_columns(readable, window.columns) if name in drawable]
        led = set(leading)
        drawn = leading + [name for name in drawn if name not in led]
    omitted = len(drawn) - MAX_DEFAULT_SERIES

    positions = (
        (row, float(index) if x_column is None else _numeric_value(row.get(x_column)))
        for index, row in enumerate(window.rows)
    )
    placed = [(row, x) for row, x in positions if x is not None]
    unplaced = len(window.rows) - len(placed)

    annotations: list[str] = []
    if omitted > 0:
        annotations.append(
            f"{omitted} more numeric column{'s' if omitted > 1 else ''} not shown; "
            "the table below lists every column."
        )
    if not window.rows_complete:
        annotations.append(
            f"Showing {len(window.rows)} of {window.total_seen} rows the run produced."
        )
    if unplaced:
        annotations.append(
            f"Not drawing {unplaced} of {len(window.rows)} rows with no {x_column} reading."
        )

    panels: list[Panel] = []
    for name in drawn[:MAX_DEFAULT_SERIES]:
        points = [Point(x=x, y=_numeric_value(row.get(name))) for row, x in placed]
        kept = decimate(points)
        panels.append(
            Panel(
                title=name,
                x_label="Row" if x_column is None else x_column,
                y_label=name,
                annotations=annotations if not panels else [],
                mark=LinesMark(
                    series=[Series(label=name, **kept._asdict())],
                    decimated=kept.decimated,
                    source_points=kept.source_points,
                ),
            )
        )

    return Figure(panels=panels, partial=partial, source=source, reason=reason)
