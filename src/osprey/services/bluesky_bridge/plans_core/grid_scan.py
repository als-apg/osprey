"""Shipped plan: ``grid_scan``, an n-dimensional rectangular grid scan.

Discovered via the layered directory catalog's ``shipped`` tier (a folder
scan of this package's ``plans_core/`` dir — see ``plan_loader.py``): step a
set of setpoint devices over a rectangular grid, reading a set of detectors
at every grid point, wrapping ``bluesky.plans.grid_scan``.

Device-agnostic: ``setpoints``/``detectors`` are resolved by string name
against whatever ``devices`` dict the bridge passes in; nothing here names a
facility PV or a fixed device set.

`render` at the bottom is this plan's own view of its rows, served by
``GET /runs/{run_id}/figure``: lines for a 1-D sweep, a heatmap for a 2-D
grid, one line per outer-axis combination above that.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from bluesky import plans as bp
from pydantic import BaseModel, Field, model_validator

from osprey.services.bluesky_bridge.figure import (
    Figure,
    HeatmapMark,
    LinesMark,
    Panel,
    Point,
    Series,
    decimate,
)

logger = logging.getLogger("osprey.services.bluesky_bridge.plans_core.grid_scan")

PLAN_METADATA = {
    "name": "grid_scan",
    "description": (
        "Scan a rectangular grid of setpoint devices, reading all detectors at "
        "every grid point (bluesky bp.grid_scan)."
    ),
    "category": "accelerator",
    "required_devices": ["setpoints", "detectors"],
    "writes": True,
}


class GridAxis(BaseModel):
    """One setpoint device's sweep range and point count, for one grid dimension."""

    setpoint: str
    start: float
    stop: float
    num_points: int = Field(..., ge=2, description="Points along this axis.")


class PARAMS(BaseModel):
    """Parameters for ``grid_scan``: one `GridAxis` per grid dimension.

    Steps each axis's setpoint device over its own ``[start, stop]`` range in
    ``num_points`` evenly-spaced steps, reading every device in ``detectors``
    at each combination of axis positions (a rectangular grid of
    ``prod(num_points)`` total points).
    """

    detectors: list[str] = Field(
        ..., min_length=1, description="Device names to read at each grid point."
    )
    axes: list[GridAxis] = Field(..., min_length=1, description="One entry per grid dimension.")
    snake_axes: bool = Field(
        default=False, description="Snake back-and-forth across successive axes."
    )

    @model_validator(mode="after")
    def _setpoints_and_detectors_disjoint(self) -> PARAMS:
        """Reject a device named as both a driven setpoint and a read detector."""
        setpoints = {axis.setpoint for axis in self.axes}
        overlap = setpoints & set(self.detectors)
        if overlap:
            raise ValueError(
                f"setpoints and detectors must be disjoint (overlap: {sorted(overlap)})"
            )
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    """Build the n-dimensional grid-scan generator.

    Mirrors the built-in ``grid_scan`` plan's (``plans.py``) idiom: resolves
    each axis's setpoint/detector by string name against ``devices`` and
    hands the flattened ``(device, start, stop, num_points)`` triples straight
    to ``bp.grid_scan``.
    """
    detectors = [devices[name] for name in params.detectors]
    args: list[Any] = []
    for axis in params.axes:
        args.extend([devices[axis.setpoint], axis.start, axis.stop, axis.num_points])
    return bp.grid_scan(detectors, *args, snake_axes=params.snake_axes)


# --- Rendering ---------------------------------------------------------------
#
# What a grid scan *is* decides how it should be drawn, and the plan knows that
# where the bridge does not: a 1-D sweep is a line, a 2-D grid is an image, and
# anything deeper is a family of lines along the fast axis. Everything below is
# total -- `render` never raises, because a figure that cannot be drawn should
# cost the operator the panel's default view, never the run.

_MAX_SERIES = 12
"""Most lines drawn on one panel for a 3-or-more-axis scan. A 5x5x5 grid has 25
outer-axis combinations and no reader can tell 25 lines apart; the excess is
reported in an annotation rather than drawn."""

_GRID_REL_TOL = 1e-6
_GRID_ABS_TOL = 1e-8
"""How far a recorded axis value may sit from a grid point and still be counted
as that point, as a fraction of the axis span and as an absolute floor.

Rows carry axis *readbacks*, not the commanded setpoints: the bridge's
``ConnectorSettable`` waits for a device to settle within its deadband (1e-9)
and the value recorded is whatever the device then reported. Keying heatmap
cells on those floats directly would give a 10x10 scan up to 100 distinct
"columns" that each hold one point, so cell indices are derived from the grid
geometry instead and the readback only has to land near its own grid point."""


def _resolve_column(rows: list[dict[str, Any]], device_name: str) -> str | None:
    """The event-data column holding *device_name*'s readings, if there is one.

    Same name-fuzzy rule the ORM analysis uses (`orm_analysis._match_value`):
    ophyd-async keys a hinted signal as either the bare device name or
    ``f"{device}-{signal}"`` depending on how many readable children the device
    exposes, so match the exact key first and the ``"<device>-"`` prefix second.
    Only the first row is inspected -- every row of a window carries every
    column (the shared row adapter's contract).
    """
    if not rows:
        return None
    keys = rows[0]
    if device_name in keys:
        return device_name
    prefix = f"{device_name}-"
    for key in keys:
        if key.startswith(prefix):
            return key
    return None


def _numeric(value: Any) -> float | None:
    """*value* as a plain finite float, or ``None`` if it is not one.

    ``None`` covers every un-drawable reading with one answer: a missing key, a
    Tiled ``NaN``, an ``inf``, a string an odd device wrote. On a y value that
    is a gap in the line; on an x value the row has no position and is dropped.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def _axis_step(axis: GridAxis) -> float:
    """The spacing between two adjacent points on *axis*.

    ``num_points >= 2`` is enforced by `GridAxis`, so this never divides by
    zero. It can legitimately *be* zero, for the degenerate ``start == stop``
    axis, which `_grid_index` handles separately.
    """
    return (axis.stop - axis.start) / (axis.num_points - 1)


def _grid_index(axis: GridAxis, value: float) -> int | None:
    """Which point of *axis* the readback *value* belongs to, or ``None``.

    Nearest grid point by ``round((value - start) / step)``, clamped into range
    so a readback that overshot the last point still lands on it, then accepted
    only if it really is within `_GRID_REL_TOL` / `_GRID_ABS_TOL` of that
    point's commanded position. A value between two grid points -- a device
    caught mid-move, a scan whose axis was disturbed -- is ``None``: better a
    dropped row the panel counts than a reading filed under a position it was
    never taken at.
    """
    step = _axis_step(axis)
    tolerance = max(_GRID_REL_TOL * abs(axis.stop - axis.start), _GRID_ABS_TOL)
    if step == 0.0:
        index = 0
    else:
        try:
            index = round((value - axis.start) / step)
        except (OverflowError, ValueError):  # pragma: no cover - denormal step
            return None
        index = min(max(index, 0), axis.num_points - 1)
    if abs(value - (axis.start + index * step)) > tolerance:
        return None
    return index


def _format_number(value: float) -> str:
    """A short, human-readable spelling of an axis position for a label."""
    return f"{value:g}"


def _dropped_note(dropped: int, total: int) -> str:
    return (
        f"{dropped} of {total} rows are not shown: their axis readings did not land "
        "on a grid point."
    )


def _thinned_note() -> str:
    return "Lines are thinned to a fixed number of points for display."


def _lines_mark(series: list[Series]) -> LinesMark:
    """A `LinesMark` carrying *series*, with the aggregate truth filled in.

    Nothing cross-validates mark-level ``decimated``/``source_points`` against
    the series', so summing them here is the one place it happens.
    """
    return LinesMark(
        series=series,
        decimated=any(entry.decimated for entry in series),
        source_points=sum(entry.source_points for entry in series),
    )


def _one_axis_panels(rows: list[dict[str, Any]], params: PARAMS) -> list[Panel]:
    """One detector-versus-setpoint line panel per detector.

    ``snake_axes`` is a no-op with a single axis (there is no outer axis to
    snake against), so the sweep is one monotone pass and the panel is one line.
    Points are drawn against the axis *readback*, not its commanded position:
    with only one dimension there is no cell to key, so the honest x is the
    value the device actually reported.
    """
    axis = params.axes[0]
    axis_column = _resolve_column(rows, axis.setpoint)
    if axis_column is None:
        return []

    panels: list[Panel] = []
    for detector in params.detectors:
        detector_column = _resolve_column(rows, detector)
        if detector_column is None:
            continue

        points: list[Point] = []
        dropped = 0
        for row in rows:
            x = _numeric(row.get(axis_column))
            if x is None:
                dropped += 1
                continue
            points.append(Point(x=x, y=_numeric(row.get(detector_column))))
        points.sort(key=lambda point: point.x)

        kept = decimate(points)
        annotations: list[str] = []
        if dropped:
            annotations.append(_dropped_note(dropped, len(rows)))
        if kept.decimated:
            annotations.append(_thinned_note())

        panels.append(
            Panel(
                title=f"{detector} vs {axis.setpoint}",
                x_label=axis.setpoint,
                y_label=detector,
                annotations=annotations,
                mark=_lines_mark([Series(label=detector, **kept._asdict())]),
            )
        )
    return panels


def _two_axis_panels(rows: list[dict[str, Any]], params: PARAMS) -> list[Panel]:
    """One heatmap panel per detector over the two-axis grid.

    The inner (fast) axis runs along x and the outer (slow) axis along y, the
    orientation bluesky's own ``LiveGrid`` uses, so a snaked scan fills the
    image row by row. Cells are keyed by grid *index* (see `_grid_index`), which
    is what makes a snaked run and a monodirectional one produce the same
    image: traversal order never reaches the cell key. A cell measured twice
    keeps the later row, and a cell never measured stays ``None`` -- drawn
    empty, not as a zero the run never read.
    """
    outer, inner = params.axes[0], params.axes[1]
    outer_column = _resolve_column(rows, outer.setpoint)
    inner_column = _resolve_column(rows, inner.setpoint)
    if outer_column is None or inner_column is None:
        return []

    outer_step, inner_step = _axis_step(outer), _axis_step(inner)
    y_labels = [_format_number(outer.start + j * outer_step) for j in range(outer.num_points)]
    x_labels = [_format_number(inner.start + i * inner_step) for i in range(inner.num_points)]

    panels: list[Panel] = []
    for detector in params.detectors:
        detector_column = _resolve_column(rows, detector)
        if detector_column is None:
            continue

        cells: dict[tuple[int, int], float | None] = {}
        dropped = 0
        for row in rows:
            outer_value = _numeric(row.get(outer_column))
            inner_value = _numeric(row.get(inner_column))
            j = None if outer_value is None else _grid_index(outer, outer_value)
            i = None if inner_value is None else _grid_index(inner, inner_value)
            if j is None or i is None:
                dropped += 1
                continue
            cells[(j, i)] = _numeric(row.get(detector_column))

        values: list[list[float | None]] = [
            [cells.get((j, i)) for i in range(inner.num_points)] for j in range(outer.num_points)
        ]
        annotations = [_dropped_note(dropped, len(rows))] if dropped else []

        panels.append(
            Panel(
                title=f"{detector} over the {outer.setpoint} × {inner.setpoint} grid",
                x_label=inner.setpoint,
                y_label=outer.setpoint,
                annotations=annotations,
                mark=HeatmapMark(
                    x_labels=x_labels,
                    y_labels=y_labels,
                    values=values,
                    value_label=detector,
                    source_points=len(cells),
                ),
            )
        )
    return panels


def _combination_label(axes: list[GridAxis], indices: tuple[int, ...]) -> str:
    """``"m1=0.5, m2=2"`` -- an outer-axis combination named by grid position.

    The commanded positions, not the readbacks: two rows in the same series
    have slightly different readbacks and only one label between them.
    """
    return ", ".join(
        f"{axis.setpoint}={_format_number(axis.start + index * _axis_step(axis))}"
        for axis, index in zip(axes, indices, strict=True)
    )


def _multi_axis_panels(rows: list[dict[str, Any]], params: PARAMS) -> list[Panel]:
    """One panel per detector: lines along the innermost axis.

    A 3-D grid has no honest single picture, so this draws the scan the way it
    was run -- one line per pass along the fast axis -- and names each line by
    where the outer axes were parked for it. Lines are ordered by outer grid
    index (never by arrival), so the legend reads the same for a partial run as
    for the finished one, and capped at `_MAX_SERIES` with the excess counted
    in an annotation rather than quietly missing.
    """
    inner = params.axes[-1]
    outer_axes = params.axes[:-1]
    inner_column = _resolve_column(rows, inner.setpoint)
    resolved = [_resolve_column(rows, axis.setpoint) for axis in outer_axes]
    if inner_column is None or any(column is None for column in resolved):
        return []
    outer_columns = [column for column in resolved if column is not None]

    panels: list[Panel] = []
    for detector in params.detectors:
        detector_column = _resolve_column(rows, detector)
        if detector_column is None:
            continue

        groups: dict[tuple[int, ...], list[Point]] = {}
        dropped = 0
        for row in rows:
            x = _numeric(row.get(inner_column))
            indices: list[int] = []
            for axis, column in zip(outer_axes, outer_columns, strict=True):
                value = _numeric(row.get(column))
                index = None if value is None else _grid_index(axis, value)
                if index is None:
                    break
                indices.append(index)
            if x is None or len(indices) != len(outer_axes):
                dropped += 1
                continue
            groups.setdefault(tuple(indices), []).append(
                Point(x=x, y=_numeric(row.get(detector_column)))
            )

        ordered = sorted(groups.items())
        series: list[Series] = []
        for combination, points in ordered[:_MAX_SERIES]:
            points.sort(key=lambda point: point.x)
            kept = decimate(points)
            series.append(
                Series(label=_combination_label(outer_axes, combination), **kept._asdict())
            )

        annotations = []
        if dropped:
            annotations.append(_dropped_note(dropped, len(rows)))
        if len(ordered) > _MAX_SERIES:
            annotations.append(
                f"Showing {_MAX_SERIES} of {len(ordered)} combinations of the outer axes; "
                "the rest are not drawn."
            )
        if any(entry.decimated for entry in series):
            annotations.append(_thinned_note())

        panels.append(
            Panel(
                title=f"{detector} vs {inner.setpoint}",
                x_label=inner.setpoint,
                y_label=detector,
                annotations=annotations,
                mark=_lines_mark(series),
            )
        )
    return panels


def render(rows: list[dict[str, Any]], params: PARAMS) -> Figure:
    """Draw a grid scan's rows: lines for 1-D, a heatmap for 2-D, lines above.

    Args:
        rows: The run's rows so far, one dict per grid point, keyed by event
            column (the shared row adapter's output). May be a partial run, or
            empty.
        params: The parameters the run was launched with, which is where the
            grid geometry -- and therefore every cell index and axis label --
            comes from.

    Returns:
        A `Figure` with one panel per detector, or no panels at all when the
        rows carry nothing this plan can place on a grid. ``partial`` and
        ``source`` are placeholders: the figure route knows which store the
        rows came from and whether the run is still producing them, and stamps
        both onto the returned figure. ``partial=True`` is the safe standing
        value -- a panel that keeps polling a settled run costs a request,
        while one that stops polling a live run shows a half-finished scan and
        calls it the whole thing.

    Never raises. A malformed row, a device that never reported, a column the
    scan did not record: each costs at most the panel it belonged to, and an
    unforeseen failure costs the panels but still returns a figure, because the
    alternative is an operator watching a running scan see an error where the
    picture should be.
    """
    panels: list[Panel] = []
    try:
        if len(params.axes) == 1:
            panels = _one_axis_panels(rows, params)
        elif len(params.axes) == 2:
            panels = _two_axis_panels(rows, params)
        else:
            panels = _multi_axis_panels(rows, params)
    except Exception:
        logger.warning("grid_scan render failed; serving an empty figure", exc_info=True)
        panels = []
    return Figure(panels=panels, partial=True, source="live")


_RENDER_CONFORMANCE: Callable[[list[dict[str, Any]], PARAMS], Figure] = render
"""Binds `render` to the signature `PlanSpec.render` declares, so a drifting
annotation is a type error here rather than a surprise in the figure route.
`PlanSpec` is generic over its params model but the loader holds
``PlanSpec[Any]``, which makes a module-level ``render`` invisible to mypy
unless the module says what it should be -- this is the module saying it."""
