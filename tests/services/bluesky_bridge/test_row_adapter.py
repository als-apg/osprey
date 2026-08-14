"""Coverage for `rows_from_columnar` / `RowWindow`.

Every figure -- a plan's own `render()`, the default figure, the panel-facing
route, and the MCP projection -- is built from rows that came through here, and
they arrive from two stores that spell "no reading" differently: the live buffer
stores ``None``, a Tiled table stores float ``NaN``. So these tests pin the two
properties downstream depends on rather than the mechanics of the conversion:
the same run replayed from Tiled produces the same rows it produced live, and
``rows_complete`` is exact -- the orm render only trusts row *position* when it
is ``True``.

Pure pydantic -- no bluesky import, so this runs in the fast import-clean lane.
"""

from __future__ import annotations

import math

import pytest

from osprey.services.bluesky_bridge.figure import RowWindow, rows_from_columnar

_COLUMNS = ["corrector_1", "bpm3"]


# --- The conversion ----------------------------------------------------------


def test_columns_and_values_are_zipped_positionally():
    window = rows_from_columnar(_COLUMNS, [[1.0, -0.5], [2.0, 0.25]], 2)

    assert window.rows == [
        {"corrector_1": 1.0, "bpm3": -0.5},
        {"corrector_1": 2.0, "bpm3": 0.25},
    ]


def test_returns_a_row_window_carrying_the_columns():
    window = rows_from_columnar(_COLUMNS, [[1.0, -0.5]], 1)

    assert isinstance(window, RowWindow)
    assert window.columns == _COLUMNS
    assert window.total_seen == 1


def test_columns_are_copied_not_aliased():
    columns = list(_COLUMNS)
    window = rows_from_columnar(columns, [[1.0, -0.5]], 1)

    columns.append("added_later")

    assert window.columns == _COLUMNS


def test_no_rows_gives_an_empty_window():
    window = rows_from_columnar(_COLUMNS, [], 0)

    assert window.rows == []
    assert window.columns == _COLUMNS
    assert window.rows_complete is True


def test_a_run_with_no_columns_yet_maps_to_empty_dicts():
    """A start doc landed but no Event did: real rows, no keys (Tiled's shape)."""
    window = rows_from_columnar([], [], 0)

    assert window.rows == []
    assert window.columns == []


def test_non_float_values_pass_through_untouched():
    window = rows_from_columnar(
        ["seq", "label", "flag", "missing"],
        [[3, "cycle-a", True, None]],
        1,
    )

    assert window.rows == [{"seq": 3, "label": "cycle-a", "flag": True, "missing": None}]


# --- NaN is the other store's None -------------------------------------------


def test_nan_becomes_none():
    window = rows_from_columnar(_COLUMNS, [[1.0, float("nan")]], 1)

    assert window.rows == [{"corrector_1": 1.0, "bpm3": None}]


def test_tiled_nan_rows_equal_live_none_rows():
    """The parity the panel depends on: replaying a run from Tiled produces the
    same rows the live buffer produced, so it draws the same figure."""
    live = rows_from_columnar(_COLUMNS, [[1.0, None], [2.0, 0.25]], 2)
    tiled = rows_from_columnar(_COLUMNS, [[1.0, float("nan")], [2.0, 0.25]], 2)

    assert tiled.rows == live.rows


def test_infinities_are_left_alone():
    """An ``inf`` is a value the run produced, not a reading it failed to take
    -- the figure models null it at the drawing edge, this adapter does not."""
    window = rows_from_columnar(_COLUMNS, [[float("inf"), float("-inf")]], 1)

    assert window.rows[0]["corrector_1"] == math.inf
    assert window.rows[0]["bpm3"] == -math.inf


def test_a_nan_string_is_not_a_nan():
    window = rows_from_columnar(["note"], [["nan"]], 1)

    assert window.rows == [{"note": "nan"}]


# --- Emission order is the caller's, and is preserved ------------------------


def test_emission_order_is_preserved_and_never_sorted():
    """Rows are mapped in the order given -- the adapter has no ``seq_num`` to
    sort on, so a Tiled caller must read its table already sorted."""
    out_of_order = [[2.0, 0.2], [0.0, 0.0], [1.0, 0.1]]

    window = rows_from_columnar(_COLUMNS, out_of_order, 3)

    assert [row["corrector_1"] for row in window.rows] == [2.0, 0.0, 1.0]


# --- rows_complete is exact, not a heuristic ---------------------------------


def test_rows_complete_when_the_window_holds_every_row():
    window = rows_from_columnar(_COLUMNS, [[1.0, 0.1], [2.0, 0.2]], 2)

    assert window.rows_complete is True


def test_rows_incomplete_when_the_run_produced_more_than_the_window_holds():
    """A paginated window, or a buffer that hit its retention cap: index
    arithmetic over these rows is unsound, so the orm render falls back."""
    window = rows_from_columnar(_COLUMNS, [[1.0, 0.1], [2.0, 0.2]], 5000)

    assert window.rows_complete is False
    assert window.total_seen == 5000


def test_an_empty_window_over_a_run_that_produced_rows_is_incomplete():
    window = rows_from_columnar(_COLUMNS, [], 7)

    assert window.rows_complete is False


# --- Structural bugs are loud ------------------------------------------------


def test_a_window_larger_than_its_run_raises():
    with pytest.raises(ValueError, match="smaller than the 2 rows given"):
        rows_from_columnar(_COLUMNS, [[1.0, 0.1], [2.0, 0.2]], 1)


def test_a_short_row_raises():
    with pytest.raises(ValueError):
        rows_from_columnar(_COLUMNS, [[1.0]], 1)


def test_a_long_row_raises():
    with pytest.raises(ValueError):
        rows_from_columnar(_COLUMNS, [[1.0, 0.1, 99.0]], 1)
