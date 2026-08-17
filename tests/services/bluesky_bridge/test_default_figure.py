"""Coverage for `default_figure`.

The default figure is what a run gets when no plan render produced one -- and
several of the paths that land here landed here *because* something already
failed, so these tests pin two things above the drawing itself: the figure never
misrepresents what it is showing (a column that stops reading is a gap, not a
dropped column; a capped or windowed figure says so), and no input makes it
raise.

Pure pydantic -- no bluesky import, so this runs in the fast import-clean lane.
"""

from __future__ import annotations

import math

import pytest

from osprey.services.bluesky_bridge.figure import (
    DEFAULT_MAX_POINTS,
    MAX_DEFAULT_SERIES,
    REASON_NO_RENDER,
    REASON_SOURCE_UNAVAILABLE,
    Figure,
    LinesMark,
    RowWindow,
    default_figure,
    rows_from_columnar,
)


def _window(columns, rows, total_seen=None):
    """A `RowWindow` from columnar values, complete unless told otherwise."""
    return rows_from_columnar(columns, rows, len(rows) if total_seen is None else total_seen)


def _series(figure: Figure, index: int):
    mark = figure.panels[index].mark
    assert isinstance(mark, LinesMark)
    return mark.series[0]


def _ys(figure: Figure, index: int) -> list[float | None]:
    return [point.y for point in _series(figure, index).points]


# --- Which columns become series ---------------------------------------------


def test_one_panel_per_numeric_column_in_column_order():
    figure = default_figure(
        _window(["bpm1", "bpm2"], [[1.0, 4.0], [2.0, 5.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert [panel.title for panel in figure.panels] == ["bpm1", "bpm2"]
    assert [_series(figure, i).label for i in (0, 1)] == ["bpm1", "bpm2"]


def test_each_panel_carries_exactly_one_series():
    figure = default_figure(
        _window(["bpm1", "bpm2"], [[1.0, 4.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    for panel in figure.panels:
        assert isinstance(panel.mark, LinesMark)
        assert len(panel.mark.series) == 1


def test_x_is_the_row_index_when_no_roles_are_declared():
    figure = default_figure(
        _window(["bpm1"], [[1.0], [2.0], [3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert [point.x for point in _series(figure, 0).points] == [0.0, 1.0, 2.0]
    assert figure.panels[0].x_label == "Row"


def test_a_column_numeric_in_one_row_only_is_still_drawn():
    figure = default_figure(
        _window(["mostly_missing"], [[None], [None], [7.0], [None]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert [panel.title for panel in figure.panels] == ["mostly_missing"]
    assert _ys(figure, 0) == [None, None, 7.0, None]


def test_a_column_that_stops_reading_is_a_gapped_series_not_a_dropped_column():
    # The pinned case: numeric through row 49, nothing afterwards. A predicate
    # that asked "is this column numeric in the rows I have?" of the tail window
    # would drop it; the column must survive as one series with a long gap.
    values = [[float(i), 0.0] for i in range(50)] + [[None, 0.0] for _ in range(50)]
    figure = default_figure(
        _window(["stops_early", "steady"], values),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert [panel.title for panel in figure.panels] == ["stops_early", "steady"]
    ys = _ys(figure, 0)
    assert len(ys) == 100
    assert ys[:50] == [float(i) for i in range(50)]
    assert ys[50:] == [None] * 50


def test_a_column_that_starts_reading_late_is_drawn_from_row_zero():
    figure = default_figure(
        _window(["starts_late"], [[None], [None], [3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert _ys(figure, 0) == [None, None, 3.0]


@pytest.mark.parametrize(
    "value",
    [None, "reading", b"reading", True, False, math.inf, -math.inf],
    ids=["none", "str", "bytes", "true", "false", "inf", "neg-inf"],
)
def test_non_numeric_entries_are_gaps_not_values(value):
    figure = default_figure(
        _window(["mixed"], [[1.0], [value], [3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert _ys(figure, 0) == [1.0, None, 3.0]


@pytest.mark.parametrize(
    "values",
    [["a", "b"], [True, False], [None, None], [math.inf, -math.inf]],
    ids=["strings", "bools", "nones", "infinities"],
)
def test_a_column_never_finite_numeric_is_not_drawn(values):
    figure = default_figure(
        _window(["never_numeric"], [[value] for value in values]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert figure.panels == []


def test_nan_from_a_tiled_table_draws_the_same_figure_as_none_from_the_live_buffer():
    live = default_figure(
        _window(["bpm1"], [[1.0], [None], [3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )
    tiled = default_figure(
        _window(["bpm1"], [[1.0], [float("nan")], [3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="tiled",
    )

    assert _ys(live, 0) == _ys(tiled, 0) == [1.0, None, 3.0]


def test_integers_and_integer_strings_are_told_apart():
    figure = default_figure(
        _window(["count", "label"], [[3, "12"], [4, "13"]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert [panel.title for panel in figure.panels] == ["count"]
    assert _ys(figure, 0) == [3.0, 4.0]


def test_a_row_missing_a_column_key_is_a_gap():
    window = RowWindow([{"bpm1": 1.0}, {}, {"bpm1": 3.0}], ["bpm1"], True, 3)

    figure = default_figure(window, reason=REASON_NO_RENDER, partial=False, source="live")

    assert _ys(figure, 0) == [1.0, None, 3.0]


def test_keys_outside_the_declared_columns_are_not_drawn():
    window = RowWindow([{"bpm1": 1.0, "undeclared": 2.0}], ["bpm1"], True, 1)

    figure = default_figure(window, reason=REASON_NO_RENDER, partial=False, source="live")

    assert [panel.title for panel in figure.panels] == ["bpm1"]


# --- The declared x axis ------------------------------------------------------
#
# The two branches SC-5 pins: a run whose plan declared exactly one movable
# channel is drawn against that channel's column, and everything else -- a
# serial sweep over several movables, a channel with no column, a run that
# declared nothing -- keeps the row-index figure this module served before roles
# existed.


def test_a_single_movable_channel_becomes_the_x_axis():
    figure = default_figure(
        _window(["motor1", "bpm1"], [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
        readable=["bpm1"],
    )

    assert [panel.title for panel in figure.panels] == ["bpm1"]
    assert figure.panels[0].x_label == "motor1"
    assert [point.x for point in _series(figure, 0).points] == [10.0, 20.0, 30.0]
    assert _ys(figure, 0) == [1.0, 2.0, 3.0]


def test_the_movable_column_is_the_axis_not_a_panel_of_its_own():
    figure = default_figure(
        _window(["motor1", "bpm1"], [[10.0, 1.0], [20.0, 2.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
    )

    assert [panel.title for panel in figure.panels] == ["bpm1"]


def test_undeclared_numeric_columns_are_still_drawn_against_the_declared_x():
    # The default figure never drops a column: a reading the plan did not
    # declare readable is still a reading the run recorded.
    figure = default_figure(
        _window(["motor1", "bpm1", "temperature"], [[10.0, 1.0, 300.0], [20.0, 2.0, 301.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
        readable=["bpm1"],
    )

    assert [panel.title for panel in figure.panels] == ["bpm1", "temperature"]
    assert all(panel.x_label == "motor1" for panel in figure.panels)


def test_declared_readable_columns_are_drawn_before_the_rest():
    # Column order puts the undeclared column first; the declaration reorders,
    # which is what decides who survives the series cap.
    figure = default_figure(
        _window(["motor1", "temperature", "bpm1"], [[10.0, 300.0, 1.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
        readable=["bpm1"],
    )

    assert [panel.title for panel in figure.panels] == ["bpm1", "temperature"]


def test_a_channel_read_through_a_child_signal_resolves_to_its_column():
    # A mock device reads back under "<channel>-<signal>"; the connector reads
    # back under the bare channel name. Both must find their column.
    figure = default_figure(
        _window(["motor1-readback", "bpm1-readback"], [[10.0, 1.0], [20.0, 2.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
        readable=["bpm1"],
    )

    assert figure.panels[0].x_label == "motor1-readback"
    assert [point.x for point in _series(figure, 0).points] == [10.0, 20.0]


def test_one_channel_named_twice_is_still_one_movable():
    figure = default_figure(
        _window(["motor1", "bpm1"], [[10.0, 1.0], [20.0, 2.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1", "motor1"],
    )

    assert figure.panels[0].x_label == "motor1"


def test_a_serial_sweep_over_several_movables_keeps_the_row_index():
    # The orm's shape: one run drives every corrector in turn, so no single
    # column is the independent variable.
    figure = default_figure(
        _window(["cor1", "cor2", "bpm1"], [[1.0, 0.0, 5.0], [2.0, 0.0, 6.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["cor1", "cor2"],
        readable=["bpm1"],
    )

    assert [panel.title for panel in figure.panels] == ["cor1", "cor2", "bpm1"]
    assert figure.panels[0].x_label == "Row"
    assert [point.x for point in _series(figure, 2).points] == [0.0, 1.0]


def test_a_movable_with_no_column_in_this_window_keeps_the_row_index():
    figure = default_figure(
        _window(["bpm1"], [[1.0], [2.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
        readable=["bpm1"],
    )

    assert [panel.title for panel in figure.panels] == ["bpm1"]
    assert figure.panels[0].x_label == "Row"


def test_a_movable_column_holding_no_number_keeps_the_row_index():
    figure = default_figure(
        _window(["motor1", "bpm1"], [["parked", 1.0], ["parked", 2.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
        readable=["bpm1"],
    )

    assert [panel.title for panel in figure.panels] == ["bpm1"]
    assert figure.panels[0].x_label == "Row"


def test_readable_order_is_not_applied_without_a_declared_x_axis():
    # Roles apply as a unit: with no usable movable the figure is exactly the
    # pre-roles one, column order included.
    figure = default_figure(
        _window(["temperature", "bpm1"], [[300.0, 1.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        readable=["bpm1"],
    )

    assert [panel.title for panel in figure.panels] == ["temperature", "bpm1"]


def test_rows_with_no_x_reading_are_left_out_and_said_so():
    figure = default_figure(
        _window(["motor1", "bpm1"], [[10.0, 1.0], [None, 2.0], [30.0, 3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
    )

    assert [point.x for point in _series(figure, 0).points] == [10.0, 30.0]
    assert _ys(figure, 0) == [1.0, 3.0]
    assert figure.panels[0].annotations == ["Not drawing 1 of 3 rows with no motor1 reading."]


def test_points_stay_in_acquisition_order_over_a_retraced_sweep():
    figure = default_figure(
        _window(["motor1", "bpm1"], [[10.0, 1.0], [20.0, 2.0], [10.0, 3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
        movable=["motor1"],
    )

    assert [point.x for point in _series(figure, 0).points] == [10.0, 20.0, 10.0]


def test_declared_roles_over_a_degenerate_window_do_not_raise():
    figure = default_figure(
        RowWindow([], ["motor1"], False, 400),
        reason=REASON_SOURCE_UNAVAILABLE,
        partial=True,
        source="tiled",
        movable=["motor1"],
        readable=["bpm1"],
    )

    assert figure.panels == []


# --- The series cap ----------------------------------------------------------


def test_forty_numeric_columns_give_twelve_panels_and_an_annotation_naming_the_rest():
    columns = [f"bpm{i}" for i in range(40)]
    figure = default_figure(
        _window(columns, [[float(i) for i in range(40)]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert len(figure.panels) == MAX_DEFAULT_SERIES == 12
    assert [panel.title for panel in figure.panels] == columns[:12]
    assert any("28" in note for note in figure.panels[0].annotations)


def test_the_cap_annotation_lands_on_the_first_panel_only():
    columns = [f"bpm{i}" for i in range(40)]
    figure = default_figure(
        _window(columns, [[float(i) for i in range(40)]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert len(figure.panels[0].annotations) == 1
    assert all(panel.annotations == [] for panel in figure.panels[1:])


def test_the_cap_counts_drawable_columns_not_declared_ones():
    # 40 declared columns, half of them never numeric: 20 drawable, 8 omitted.
    columns = [f"col{i}" for i in range(40)]
    row = [float(i) if i % 2 else "text" for i in range(40)]
    figure = default_figure(
        _window(columns, [row]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert len(figure.panels) == 12
    assert any("8 more" in note for note in figure.panels[0].annotations)


def test_exactly_twelve_columns_are_not_annotated():
    columns = [f"bpm{i}" for i in range(12)]
    figure = default_figure(
        _window(columns, [[float(i) for i in range(12)]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert len(figure.panels) == 12
    assert figure.panels[0].annotations == []


def test_a_single_omitted_column_reads_as_singular():
    columns = [f"bpm{i}" for i in range(13)]
    figure = default_figure(
        _window(columns, [[float(i) for i in range(13)]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert figure.panels[0].annotations == [
        "1 more numeric column not shown; the table below lists every column."
    ]


# --- Incomplete windows ------------------------------------------------------


def test_a_windowed_run_says_how_many_rows_it_is_showing():
    figure = default_figure(
        _window(["bpm1"], [[1.0], [2.0]], total_seen=500),
        reason=REASON_NO_RENDER,
        partial=True,
        source="live",
    )

    assert figure.panels[0].annotations == ["Showing 2 of 500 rows the run produced."]


def test_a_complete_window_is_not_annotated():
    figure = default_figure(
        _window(["bpm1"], [[1.0], [2.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    assert figure.panels[0].annotations == []


def test_a_capped_and_windowed_figure_reports_both():
    columns = [f"bpm{i}" for i in range(40)]
    figure = default_figure(
        _window(columns, [[float(i) for i in range(40)]], total_seen=9),
        reason=REASON_NO_RENDER,
        partial=True,
        source="tiled",
    )

    assert len(figure.panels[0].annotations) == 2


# --- Decimation --------------------------------------------------------------


def test_a_long_series_is_decimated_and_says_so():
    rows = [[float(i)] for i in range(DEFAULT_MAX_POINTS * 3)]
    figure = default_figure(
        _window(["bpm1"], rows),
        reason=REASON_NO_RENDER,
        partial=False,
        source="tiled",
    )

    series = _series(figure, 0)
    mark = figure.panels[0].mark
    assert len(series.points) == DEFAULT_MAX_POINTS
    assert series.decimated is True
    assert series.source_points == DEFAULT_MAX_POINTS * 3
    assert mark.decimated is True
    assert mark.source_points == DEFAULT_MAX_POINTS * 3


def test_decimation_keeps_the_first_and_last_rows():
    rows = [[float(i)] for i in range(DEFAULT_MAX_POINTS * 3)]
    figure = default_figure(
        _window(["bpm1"], rows),
        reason=REASON_NO_RENDER,
        partial=False,
        source="tiled",
    )

    points = _series(figure, 0).points
    assert points[0].x == 0.0
    assert points[-1].x == float(DEFAULT_MAX_POINTS * 3 - 1)


def test_a_gap_survives_decimation():
    rows = [[float(i)] for i in range(DEFAULT_MAX_POINTS * 3)]
    rows[7] = [None]
    figure = default_figure(
        _window(["bpm1"], rows),
        reason=REASON_NO_RENDER,
        partial=False,
        source="tiled",
    )

    assert any(point.y is None for point in _series(figure, 0).points)


def test_an_undecimated_series_reports_its_own_length():
    figure = default_figure(
        _window(["bpm1"], [[1.0], [2.0], [3.0]]),
        reason=REASON_NO_RENDER,
        partial=False,
        source="live",
    )

    series = _series(figure, 0)
    mark = figure.panels[0].mark
    assert series.decimated is False
    assert series.source_points == 3
    assert mark.decimated is False
    assert mark.source_points == 3


# --- The figure envelope -----------------------------------------------------


def test_partial_source_and_reason_are_the_callers_to_state():
    figure = default_figure(
        _window(["bpm1"], [[1.0]]),
        reason=REASON_SOURCE_UNAVAILABLE,
        partial=True,
        source="tiled",
    )

    assert figure.partial is True
    assert figure.source == "tiled"
    assert figure.reason == REASON_SOURCE_UNAVAILABLE


def test_reason_may_be_none():
    figure = default_figure(
        _window(["bpm1"], [[1.0]]),
        reason=None,
        partial=False,
        source="live",
    )

    assert figure.reason is None


def test_the_figure_round_trips_through_its_own_wire_form():
    columns = [f"bpm{i}" for i in range(40)]
    figure = default_figure(
        _window(columns, [[float(i) for i in range(40)], [None] * 40]),
        reason=REASON_NO_RENDER,
        partial=True,
        source="live",
    )

    assert Figure.model_validate(figure.model_dump()) == figure


# --- Never raises ------------------------------------------------------------


@pytest.mark.parametrize(
    ("columns", "rows"),
    [
        ([], []),
        ([], [[]]),
        (["bpm1"], []),
        (["bpm1"], [[None]]),
        (["bpm1", "bpm2"], [["a", "b"], ["c", "d"]]),
    ],
    ids=["nothing", "empty-rows", "no-rows", "all-none", "all-strings"],
)
def test_degenerate_windows_give_an_empty_figure_rather_than_raising(columns, rows):
    figure = default_figure(
        _window(columns, rows),
        reason=REASON_SOURCE_UNAVAILABLE,
        partial=False,
        source="tiled",
    )

    assert figure.panels == []
    assert figure.reason == REASON_SOURCE_UNAVAILABLE


def test_an_unhelpful_value_type_is_a_gap_not_an_exception():
    class Opaque:
        pass

    window = RowWindow([{"bpm1": 1.0}, {"bpm1": Opaque()}], ["bpm1"], True, 2)

    figure = default_figure(window, reason=REASON_NO_RENDER, partial=False, source="live")

    assert _ys(figure, 0) == [1.0, None]


def test_an_empty_window_over_a_nonempty_run_still_draws_nothing_and_says_so():
    figure = default_figure(
        RowWindow([], ["bpm1"], False, 400),
        reason=REASON_SOURCE_UNAVAILABLE,
        partial=True,
        source="tiled",
    )

    # No panel exists to carry the annotation -- an empty figure shows nothing
    # to qualify -- but the envelope still reports partial/source/reason.
    assert figure.panels == []
    assert figure.partial is True
