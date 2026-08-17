"""Coverage for ``plans_core/grid_scan.py``'s ``render()``.

Three shapes, one per grid dimensionality: lines for a 1-D sweep, a heatmap for
a 2-D grid, one line per outer-axis combination above that. The load-bearing
property throughout is that **cell and series keys come from the grid geometry,
not from the recorded floats**. Rows carry axis READBACKS -- the bridge's
`ConnectorSettable` records what the device reported after settling, not the
value it was commanded to -- so the fixtures here perturb every axis reading by
+/-1e-9 before handing it to `render`. Without that perturbation a
float-equality-keyed implementation passes every test in this file while
producing ``num_points**2`` one-point cells against a real stack; with it, the
same implementation fails immediately.

Runs in the normal unit lane -- the bluesky stack is a core dependency -- with
the same `importorskip` guard as its siblings so a slimmed install skips rather
than errors.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("bluesky")

from osprey.services.bluesky_bridge.figure import (  # noqa: E402
    DEFAULT_MAX_POINTS,
    Figure,
    HeatmapMark,
    LinesMark,
)
from osprey.services.bluesky_bridge.plans_core import grid_scan  # noqa: E402
from osprey.services.bluesky_bridge.plans_core.grid_scan import (  # noqa: E402
    PARAMS,
    GridAxis,
    render,
)

# =========================================================================
# Fixture builders: a grid scan's rows, as the bridge would have recorded them
# =========================================================================


def _axis(setpoint: str, start: float, stop: float, num_points: int) -> GridAxis:
    return GridAxis(setpoint=setpoint, start=start, stop=stop, num_points=num_points)


def _nominal(axis: GridAxis, index: int) -> float:
    """The position axis *index* was commanded to."""
    return axis.start + index * (axis.stop - axis.start) / (axis.num_points - 1)


def _traverse(axes: list[GridAxis], snake: bool, flip: bool = False) -> Iterator[tuple[int, ...]]:
    """Grid indices in emission order, outermost axis slowest.

    With ``snake`` every axis but the outermost reverses direction each time
    the axis outside it steps -- the boustrophedon path ``bp.grid_scan`` drives
    with ``snake_axes=True``.
    """
    if not axes:
        yield ()
        return
    head, rest = axes[0], axes[1:]
    order = range(head.num_points - 1, -1, -1) if flip else range(head.num_points)
    for step, index in enumerate(order):
        for tail in _traverse(rest, snake, flip=snake and step % 2 == 1):
            yield (index, *tail)


def _detector_value(indices: tuple[int, ...]) -> float:
    """A value unique to one grid point, so a mis-keyed cell is unmistakable."""
    return sum(index * 100**position for position, index in enumerate(reversed(indices))) + 0.5


def _rows(
    axes: list[GridAxis],
    detectors: dict[str, str],
    *,
    snake: bool = False,
    jitter: float = 1e-9,
    axis_columns: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The rows a run over *axes* would have produced.

    Axis columns hold the commanded position plus an alternating ``+/-jitter``:
    a real readback is never bit-identical to its setpoint, and a cell keyed on
    the float itself would fan out into one cell per row.

    Args:
        axes: The grid, outermost first.
        detectors: Detector device name -> the event column it is recorded in
            (ophyd-async names a hinted child ``"<device>-<signal>"``).
        snake: Whether the inner axes reverse direction each outer step.
        jitter: How far each readback sits from its commanded position.
        axis_columns: Setpoint device name -> event column, for the axes whose
            column is not the bare device name. Defaults to bare names.
    """
    axis_columns = axis_columns or {}
    rows: list[dict[str, Any]] = []
    for sign, indices in enumerate(_traverse(axes, snake)):
        row: dict[str, Any] = {}
        for axis, index in zip(axes, indices, strict=True):
            offset = jitter if sign % 2 == 0 else -jitter
            row[axis_columns.get(axis.setpoint, axis.setpoint)] = _nominal(axis, index) + offset
        for column in detectors.values():
            row[column] = _detector_value(indices)
        rows.append(row)
    return rows


def _sole_panel(figure: Figure):
    assert len(figure.panels) == 1, f"expected one panel, got {[p.title for p in figure.panels]}"
    return figure.panels[0]


# =========================================================================
# One axis: detector-vs-setpoint lines
# =========================================================================


def test_one_axis_draws_one_line_panel_per_detector_against_the_readback() -> None:
    axes = [_axis("motor1", 0.0, 4.0, 5)]
    params = PARAMS(readbacks=["det1", "det2"], axes=axes)
    rows = _rows(axes, {"det1": "det1", "det2": "det2"})

    figure = render(rows, params)

    assert [panel.title for panel in figure.panels] == ["det1 vs motor1", "det2 vs motor1"]
    panel = figure.panels[0]
    assert (panel.x_label, panel.y_label) == ("motor1", "det1")
    mark = panel.mark
    assert isinstance(mark, LinesMark)
    assert len(mark.series) == 1

    # x is the readback the run recorded, jitter and all -- with one dimension
    # there is no cell to key, so nothing is rounded to the commanded position.
    points = mark.series[0].points
    assert [point.x for point in points] == [row["motor1"] for row in rows]
    assert [point.y for point in points] == [row["det1"] for row in rows]
    assert mark.source_points == len(rows)
    assert mark.decimated is False
    assert panel.annotations == []


def test_one_axis_orders_points_by_x_whatever_order_the_rows_arrived_in() -> None:
    axes = [_axis("motor1", 0.0, 4.0, 5)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = list(reversed(_rows(axes, {"det1": "det1"})))

    mark = _sole_panel(render(rows, params)).mark

    assert isinstance(mark, LinesMark)
    xs = [point.x for point in mark.series[0].points]
    assert xs == sorted(xs)


def test_one_axis_keeps_a_missing_detector_reading_as_a_gap_not_a_zero() -> None:
    axes = [_axis("motor1", 0.0, 2.0, 3)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})
    rows[1]["det1"] = None
    rows[2]["det1"] = float("nan")  # how Tiled spells the same absence

    mark = _sole_panel(render(rows, params)).mark

    assert isinstance(mark, LinesMark)
    assert [point.y for point in mark.series[0].points] == [0.5, None, None]


def test_one_axis_drops_rows_with_no_usable_axis_reading_and_counts_them() -> None:
    axes = [_axis("motor1", 0.0, 4.0, 5)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})
    rows[1]["motor1"] = None
    rows[3]["motor1"] = float("nan")

    panel = _sole_panel(render(rows, params))
    mark = panel.mark

    assert isinstance(mark, LinesMark)
    assert len(mark.series[0].points) == 3
    assert panel.annotations == [
        "2 of 5 rows are not shown: their axis readings did not land on a grid point."
    ]


def test_one_axis_thins_a_long_sweep_and_says_so() -> None:
    total = DEFAULT_MAX_POINTS * 2
    axes = [_axis("motor1", 0.0, 1.0, total)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})

    panel = _sole_panel(render(rows, params))
    mark = panel.mark

    assert isinstance(mark, LinesMark)
    series = mark.series[0]
    assert len(series.points) == DEFAULT_MAX_POINTS
    assert series.decimated is True
    assert series.source_points == total
    assert mark.decimated is True
    assert mark.source_points == total
    assert "thinned" in " ".join(panel.annotations).lower()


def test_snake_axes_is_a_no_op_for_a_single_axis() -> None:
    axes = [_axis("motor1", 0.0, 4.0, 5)]
    rows = _rows(axes, {"det1": "det1"})

    straight = render(rows, PARAMS(readbacks=["det1"], axes=axes, snake_axes=False))
    snaked = render(rows, PARAMS(readbacks=["det1"], axes=axes, snake_axes=True))

    assert straight.model_dump() == snaked.model_dump()


# =========================================================================
# Two axes: a heatmap keyed by grid geometry
# =========================================================================


@pytest.mark.parametrize("snake", [False, True])
def test_two_axis_heatmap_cells_match_their_source_rows(snake: bool) -> None:
    axes = [_axis("motor1", 0.0, 2.0, 3), _axis("motor2", -1.0, 1.0, 5)]
    params = PARAMS(readbacks=["det1"], axes=axes, snake_axes=snake)
    rows = _rows(axes, {"det1": "det1-value"}, snake=snake)

    panel = _sole_panel(render(rows, params))
    mark = panel.mark
    assert isinstance(mark, HeatmapMark)

    # Every (i, j, value) triple is the one its source row carried: the outer
    # axis indexes rows of `values`, the inner axis indexes columns, and
    # traversal order never reaches the key.
    for (j, i), value in zip(
        _traverse(axes, snake), [row["det1-value"] for row in rows], strict=True
    ):
        assert mark.values[j][i] == value, f"cell (j={j}, i={i}) holds the wrong row's reading"

    assert len(mark.values) == 3
    assert all(len(row) == 5 for row in mark.values)
    assert mark.source_points == 15
    assert panel.annotations == []


def test_two_axis_heatmap_is_the_same_picture_snaked_or_not() -> None:
    axes = [_axis("motor1", 0.0, 2.0, 3), _axis("motor2", -1.0, 1.0, 5)]

    straight = render(
        _rows(axes, {"det1": "det1"}, snake=False),
        PARAMS(readbacks=["det1"], axes=axes, snake_axes=False),
    )
    snaked = render(
        _rows(axes, {"det1": "det1"}, snake=True),
        PARAMS(readbacks=["det1"], axes=axes, snake_axes=True),
    )

    assert straight.model_dump() == snaked.model_dump()


def test_two_axis_heatmap_labels_the_commanded_grid_positions() -> None:
    axes = [_axis("motor1", 0.0, 2.0, 3), _axis("motor2", -1.0, 1.0, 5)]
    params = PARAMS(readbacks=["det1"], axes=axes)

    panel = _sole_panel(render(_rows(axes, {"det1": "det1"}), params))
    mark = panel.mark

    assert isinstance(mark, HeatmapMark)
    assert mark.y_labels == ["0", "1", "2"]
    assert mark.x_labels == ["-1", "-0.5", "0", "0.5", "1"]
    assert (panel.x_label, panel.y_label) == ("motor2", "motor1")
    assert mark.value_label == "det1"
    assert panel.title == "det1 over the motor1 × motor2 grid"


def test_two_axis_heatmap_does_not_fan_out_when_readbacks_miss_their_setpoints() -> None:
    """The regression this whole file exists for.

    A 3x5 grid whose readbacks are all perturbed must still produce 15 cells in
    a 3x5 image. Keying on the recorded float instead would produce 15 distinct
    "columns" -- one row of the image per reading.
    """
    axes = [_axis("motor1", 0.0, 2.0, 3), _axis("motor2", -1.0, 1.0, 5)]
    params = PARAMS(readbacks=["det1"], axes=axes, snake_axes=True)
    rows = _rows(axes, {"det1": "det1"}, snake=True, jitter=5e-9)

    mark = _sole_panel(render(rows, params)).mark

    assert isinstance(mark, HeatmapMark)
    assert mark.source_points == 15
    assert all(cell is not None for row in mark.values for cell in row)


def test_two_axis_unmeasured_cells_stay_empty_on_a_partial_run() -> None:
    axes = [_axis("motor1", 0.0, 2.0, 3), _axis("motor2", 0.0, 1.0, 3)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})[:4]  # the run is still on its second pass

    mark = _sole_panel(render(rows, params)).mark

    assert isinstance(mark, HeatmapMark)
    assert mark.source_points == 4
    assert mark.values[0] == [0.5, 1.5, 2.5]
    assert mark.values[1] == [100.5, None, None]
    assert mark.values[2] == [None, None, None]


def test_two_axis_drops_a_row_caught_between_grid_points_and_counts_it() -> None:
    axes = [_axis("motor1", 0.0, 2.0, 3), _axis("motor2", 0.0, 2.0, 3)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})
    rows[4]["motor2"] = 0.5  # a device caught mid-move, halfway between points

    panel = _sole_panel(render(rows, params))
    mark = panel.mark

    assert isinstance(mark, HeatmapMark)
    assert mark.source_points == 8
    assert mark.values[1][1] is None
    assert panel.annotations == [
        "1 of 9 rows are not shown: their axis readings did not land on a grid point."
    ]


def test_two_axis_keeps_the_axis_span_tolerance_proportional() -> None:
    """A wide axis tolerates a proportionally wider readback error."""
    axes = [_axis("motor1", 0.0, 1.0, 2), _axis("motor2", 0.0, 1e6, 3)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"}, jitter=0.0)
    rows[0]["motor2"] = 0.4  # 4e-7 of the span: noise on a metre-scale stage

    mark = _sole_panel(render(rows, params)).mark

    assert isinstance(mark, HeatmapMark)
    assert mark.values[0][0] == 0.5


def test_two_axis_records_the_later_row_when_a_cell_is_measured_twice() -> None:
    axes = [_axis("motor1", 0.0, 1.0, 2), _axis("motor2", 0.0, 1.0, 2)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})
    revisit = dict(rows[0])
    revisit["det1"] = 42.0
    rows.append(revisit)

    mark = _sole_panel(render(rows, params)).mark

    assert isinstance(mark, HeatmapMark)
    assert mark.values[0][0] == 42.0
    assert mark.source_points == 4


def test_two_axis_draws_one_heatmap_per_detector() -> None:
    axes = [_axis("motor1", 0.0, 1.0, 2), _axis("motor2", 0.0, 1.0, 2)]
    params = PARAMS(readbacks=["det1", "det2"], axes=axes)
    rows = _rows(axes, {"det1": "det1", "det2": "det2-value"})

    figure = render(rows, params)

    assert [panel.mark.value_label for panel in figure.panels] == ["det1", "det2"]  # type: ignore[union-attr]


# =========================================================================
# Three or more axes: lines along the innermost axis
# =========================================================================


def test_three_axes_draw_one_line_per_outer_combination() -> None:
    axes = [
        _axis("motor1", 0.0, 1.0, 2),
        _axis("motor2", 0.0, 2.0, 3),
        _axis("motor3", 0.0, 3.0, 4),
    ]
    params = PARAMS(readbacks=["det1"], axes=axes, snake_axes=True)
    rows = _rows(axes, {"det1": "det1"}, snake=True)

    panel = _sole_panel(render(rows, params))
    mark = panel.mark
    assert isinstance(mark, LinesMark)

    assert len(mark.series) == 6  # 2 x 3 outer combinations
    assert [series.label for series in mark.series] == [
        "motor1=0, motor2=0",
        "motor1=0, motor2=1",
        "motor1=0, motor2=2",
        "motor1=1, motor2=0",
        "motor1=1, motor2=1",
        "motor1=1, motor2=2",
    ]
    assert (panel.x_label, panel.y_label) == ("motor3", "det1")
    assert panel.title == "det1 vs motor3"

    # Each line holds its own pass along the fast axis, left to right, and
    # every y is the reading its own grid point produced.
    for series in mark.series:
        assert len(series.points) == 4
        xs = [point.x for point in series.points]
        assert xs == sorted(xs)
    assert [point.y for point in mark.series[0].points] == [0.5, 1.5, 2.5, 3.5]
    assert [point.y for point in mark.series[5].points] == [10200.5, 10201.5, 10202.5, 10203.5]
    assert mark.source_points == 24
    assert panel.annotations == []


def test_three_axes_cap_the_lines_and_count_what_is_not_drawn() -> None:
    axes = [
        _axis("motor1", 0.0, 3.0, 4),
        _axis("motor2", 0.0, 4.0, 5),
        _axis("motor3", 0.0, 1.0, 2),
    ]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})

    panel = _sole_panel(render(rows, params))
    mark = panel.mark

    assert isinstance(mark, LinesMark)
    assert len(mark.series) == grid_scan._MAX_SERIES == 12
    assert mark.series[0].label == "motor1=0, motor2=0"
    assert panel.annotations == [
        "Showing 12 of 20 combinations of the outer axes; the rest are not drawn."
    ]
    # The aggregate counts what the mark actually carries, not the whole run.
    assert mark.source_points == 24


def test_four_axes_still_group_by_every_outer_axis() -> None:
    axes = [
        _axis("motor1", 0.0, 1.0, 2),
        _axis("motor2", 0.0, 1.0, 2),
        _axis("motor3", 0.0, 1.0, 2),
        _axis("motor4", 0.0, 1.0, 2),
    ]
    params = PARAMS(readbacks=["det1"], axes=axes, snake_axes=True)
    rows = _rows(axes, {"det1": "det1"}, snake=True)

    mark = _sole_panel(render(rows, params)).mark

    assert isinstance(mark, LinesMark)
    assert len(mark.series) == 8
    assert mark.series[0].label == "motor1=0, motor2=0, motor3=0"


def test_three_axes_drop_rows_whose_outer_readbacks_are_off_grid() -> None:
    axes = [
        _axis("motor1", 0.0, 1.0, 2),
        _axis("motor2", 0.0, 1.0, 2),
        _axis("motor3", 0.0, 1.0, 2),
    ]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})
    rows[0]["motor2"] = 0.5
    rows[1]["motor3"] = None

    panel = _sole_panel(render(rows, params))
    mark = panel.mark

    assert isinstance(mark, LinesMark)
    assert mark.source_points == 6
    assert panel.annotations == [
        "2 of 8 rows are not shown: their axis readings did not land on a grid point."
    ]


# =========================================================================
# Column resolution, degradation, and the render contract
# =========================================================================


def test_columns_resolve_through_the_ophyd_async_child_suffix() -> None:
    axes = [_axis("motor1", 0.0, 1.0, 2)]
    params = PARAMS(readbacks=["det1"], axes=axes)
    rows = _rows(
        axes,
        {"det1": "det1-value"},
        axis_columns={"motor1": "motor1-readback"},
    )

    mark = _sole_panel(render(rows, params)).mark

    assert isinstance(mark, LinesMark)
    assert [point.y for point in mark.series[0].points] == [0.5, 1.5]


def test_a_detector_that_never_reported_costs_only_its_own_panel() -> None:
    axes = [_axis("motor1", 0.0, 1.0, 2)]
    params = PARAMS(readbacks=["det1", "det_offline"], axes=axes)
    rows = _rows(axes, {"det1": "det1"})

    figure = render(rows, params)

    assert [panel.title for panel in figure.panels] == ["det1 vs motor1"]


def test_an_axis_column_that_is_absent_leaves_the_figure_empty() -> None:
    axes = [_axis("motor1", 0.0, 1.0, 2), _axis("motor2", 0.0, 1.0, 2)]
    params = PARAMS(readbacks=["det1"], axes=axes)

    figure = render([{"det1": 1.0}, {"det1": 2.0}], params)

    assert figure.panels == []


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([], id="no rows yet"),
        pytest.param([{}], id="a row with nothing in it"),
        pytest.param([{"motor1": "not a number", "det1": object()}], id="unreadable values"),
        pytest.param([{"motor1": True, "det1": [1, 2]}], id="wrong types entirely"),
    ],
)
def test_render_never_raises_on_malformed_rows(rows: list[dict[str, Any]]) -> None:
    for axes in (
        [_axis("motor1", 0.0, 1.0, 2)],
        [_axis("motor1", 0.0, 1.0, 2), _axis("motor2", 0.0, 1.0, 2)],
        [_axis("motor1", 0.0, 1.0, 2), _axis("motor2", 0.0, 1.0, 2), _axis("motor3", 0.0, 1.0, 2)],
    ):
        figure = render(rows, PARAMS(readbacks=["det1"], axes=axes))
        assert isinstance(figure, Figure)


def test_an_unforeseen_failure_costs_the_panels_not_the_response(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise RuntimeError("something no fixture anticipated")

    monkeypatch.setattr(grid_scan, "_one_axis_panels", _boom)
    axes = [_axis("motor1", 0.0, 1.0, 2)]

    with caplog.at_level(logging.WARNING):
        figure = render(_rows(axes, {"det1": "det1"}), PARAMS(readbacks=["det1"], axes=axes))

    assert figure.panels == []
    assert figure.partial is True
    assert "grid_scan render failed" in caplog.text


def test_the_figure_round_trips_through_its_own_wire_form() -> None:
    axes = [_axis("motor1", 0.0, 2.0, 3), _axis("motor2", 0.0, 2.0, 3)]
    params = PARAMS(readbacks=["det1"], axes=axes)

    figure = render(_rows(axes, {"det1": "det1"}), params)
    revalidated = Figure.model_validate(figure.model_dump())

    assert revalidated.model_dump() == figure.model_dump()
    assert isinstance(revalidated.panels[0].mark, HeatmapMark)


def test_render_is_bound_to_the_signature_planspec_declares() -> None:
    assert grid_scan._RENDER_CONFORMANCE is render
