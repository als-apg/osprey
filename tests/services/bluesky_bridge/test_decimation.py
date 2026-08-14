"""Coverage for `decimate`.

Three callers share this function -- the default figure builder, the
panel-facing figure route, and the MCP projection that re-strides an already
thinned figure onto a smaller budget -- and each of them publishes its result
as `Series.decimated` / `Series.source_points`, which an operator and an agent
both read as fact. So these tests pin the honesty properties, not just the
count: the endpoints survive, the x order survives, gaps survive, and
``source_points`` keeps meaning "before *any* decimation" across a second pass.

Pure pydantic -- no bluesky import, so this runs in the fast import-clean lane.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from osprey.services.bluesky_bridge.figure import (
    DEFAULT_MAX_POINTS,
    LinesMark,
    Point,
    Series,
    decimate,
)


def _ramp(count: int, *, gaps: set[int] | None = None) -> list[Point]:
    """``count`` points along x = index, with ``y=None`` at the ``gaps`` indices."""
    gaps = gaps or set()
    return [Point(x=float(i), y=None if i in gaps else float(i) * 2.0) for i in range(count)]


def _xs(points: list[Point]) -> list[float]:
    return [p.x for p in points]


# --- Under budget: nothing is touched, nothing is claimed --------------------


@pytest.mark.parametrize("count", [0, 1, 2, 7, 10])
def test_at_or_under_budget_keeps_every_point(count: int) -> None:
    points = _ramp(count)

    result = decimate(points, 10)

    assert result.points == points
    assert result.decimated is False
    assert result.source_points == count


def test_under_budget_returns_a_new_list() -> None:
    points = _ramp(5)

    result = decimate(points, 10)

    assert result.points is not points
    result.points.append(Point(x=99.0, y=99.0))
    assert len(points) == 5


def test_default_budget_is_two_thousand() -> None:
    assert DEFAULT_MAX_POINTS == 2000
    assert decimate(_ramp(2000)).decimated is False
    assert len(decimate(_ramp(2001)).points) == 2000


def test_input_is_never_mutated() -> None:
    points = _ramp(50, gaps={13})
    before = [(p.x, p.y) for p in points]

    decimate(points, 7)

    assert [(p.x, p.y) for p in points] == before


# --- Over budget: the stride ------------------------------------------------


@pytest.mark.parametrize(("count", "budget"), [(11, 10), (100, 10), (2001, 2000), (99999, 2000)])
def test_over_budget_spends_exactly_the_budget(count: int, budget: int) -> None:
    result = decimate(_ramp(count), budget)

    assert len(result.points) == budget
    assert result.decimated is True
    assert result.source_points == count


@pytest.mark.parametrize(("count", "budget"), [(11, 10), (100, 10), (12345, 2000), (99999, 3)])
def test_endpoints_and_x_order_survive(count: int, budget: int) -> None:
    points = _ramp(count)

    kept = decimate(points, budget).points

    assert kept[0] == points[0]
    assert kept[-1] == points[-1]
    assert _xs(kept) == sorted(_xs(kept)), "stride must not reorder the series"
    assert len(set(_xs(kept))) == len(kept), "no point is emitted twice"


def test_stride_is_evenly_spaced() -> None:
    # 10 points onto 5: both endpoints plus one from each of three equal
    # interior windows -- the same selection an even stride would make.
    kept = decimate(_ramp(10), 5).points

    assert _xs(kept) == [0.0, 2.0, 4.0, 7.0, 9.0]


def test_stride_spacing_stays_regular_on_a_large_series() -> None:
    kept = decimate(_ramp(100_000), 1000).points

    steps = [b - a for a, b in pairwise(_xs(kept))]
    # 100 samples per window; only the endpoint-adjacent windows differ, and
    # then by less than one window.
    assert max(steps) - min(steps) < 100


# --- Gaps are data, not noise ------------------------------------------------


def test_a_lone_gap_survives_a_heavy_stride() -> None:
    # An even stride over 1000 points onto 10 would sample none of index 500.
    result = decimate(_ramp(1000, gaps={500}), 10)

    gaps = [p for p in result.points if p.y is None]
    assert len(gaps) == 1
    assert gaps[0].x == 500.0


def test_every_window_holding_a_gap_reports_one() -> None:
    # 10 points onto 5 -> interior windows [1,3), [3,6), [6,9). Gaps in the
    # first and last of those; the middle window has none.
    kept = decimate(_ramp(10, gaps={2, 8}), 5).points

    assert [(p.x, p.y) for p in kept] == [
        (0.0, 0.0),
        (2.0, None),  # window [1,3): the gap, not the midpoint
        (4.0, 8.0),  # window [3,6): no gap, so the midpoint
        (8.0, None),  # window [6,9): the gap, not the midpoint 7
        (9.0, 18.0),
    ]


def test_the_first_gap_in_a_window_wins() -> None:
    kept = decimate(_ramp(10, gaps={1, 2}), 5).points

    assert kept[1].x == 1.0
    assert kept[1].y is None


def test_an_all_gap_series_decimates_to_gaps() -> None:
    result = decimate(_ramp(500, gaps=set(range(500))), 20)

    assert len(result.points) == 20
    assert all(p.y is None for p in result.points)
    assert result.source_points == 500


def test_endpoints_win_over_the_gap_rule_but_are_still_truthful() -> None:
    # A gap at index 1 sits in the first interior window, so it is kept there;
    # index 0 stays index 0 because the series' x range is not negotiable.
    kept = decimate(_ramp(10, gaps={0, 1, 9}), 5).points

    assert (kept[0].x, kept[0].y) == (0.0, None), "an endpoint that IS a gap stays a gap"
    assert (kept[1].x, kept[1].y) == (1.0, None)
    assert (kept[-1].x, kept[-1].y) == (9.0, None)


def test_a_gap_never_displaces_a_measured_window() -> None:
    kept = decimate(_ramp(100, gaps={5}), 10).points

    assert [p for p in kept if p.y is None] == [Point(x=5.0, y=None)]
    assert sum(1 for p in kept if p.y is not None) == 9


# --- Re-decimation composes -------------------------------------------------


def test_second_pass_carries_the_original_count_forward() -> None:
    first = decimate(_ramp(50_000), 2000)
    assert first.source_points == 50_000

    second = decimate(
        first.points,
        200,
        source_points=first.source_points,
        decimated=first.decimated,
    )

    assert len(second.points) == 200
    assert second.decimated is True
    assert second.source_points == 50_000, "source_points means before ANY decimation"
    assert second.points[0] == first.points[0]
    assert second.points[-1] == first.points[-1]


def test_a_second_pass_under_budget_still_reports_the_first() -> None:
    first = decimate(_ramp(5000), 100)

    second = decimate(
        first.points,
        500,
        source_points=first.source_points,
        decimated=first.decimated,
    )

    assert second.points == first.points
    assert second.decimated is True, "the series is still a thinned one"
    assert second.source_points == 5000


def test_forgetting_the_prior_truth_reports_only_this_pass() -> None:
    # The trap the composing callers must avoid: without the two keyword
    # arguments, an already-thinned input looks like raw data.
    first = decimate(_ramp(50_000), 2000)

    naive = decimate(first.points, 200)

    assert naive.source_points == 2000
    assert naive.decimated is True


def test_source_points_smaller_than_the_input_is_a_bug_not_a_figure() -> None:
    with pytest.raises(ValueError, match="before decimation"):
        decimate(_ramp(100), 10, source_points=50)


def test_source_points_may_exceed_the_input_without_a_second_stride() -> None:
    result = decimate(_ramp(10), 100, source_points=999, decimated=True)

    assert result.source_points == 999
    assert result.decimated is True


# --- Degenerate budgets ------------------------------------------------------


def test_budget_of_two_keeps_only_the_endpoints() -> None:
    result = decimate(_ramp(100), 2)

    assert _xs(result.points) == [0.0, 99.0]
    assert result.source_points == 100


def test_budget_of_one_keeps_the_first_point() -> None:
    result = decimate(_ramp(100), 1)

    assert _xs(result.points) == [0.0]
    assert result.decimated is True


@pytest.mark.parametrize("budget", [0, -1])
def test_a_budget_below_one_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        decimate(_ramp(10), budget)


def test_empty_input_needs_no_budget_argument() -> None:
    result = decimate([])

    assert result.points == []
    assert result.decimated is False
    assert result.source_points == 0


# --- Feeding the models ------------------------------------------------------


def test_the_result_splats_into_a_series() -> None:
    result = decimate(_ramp(9000, gaps={4500}), 2000)

    series = Series(label="bpm3", **result._asdict())

    assert series.decimated is True
    assert series.source_points == 9000
    assert len(series.points) == 2000
    assert any(p.y is None for p in series.points)


def test_mark_level_truth_aggregates_the_per_series_truth() -> None:
    thinned = decimate(_ramp(5000), 2000)
    untouched = decimate(_ramp(10))

    mark = LinesMark(
        series=[
            Series(label="thinned", **thinned._asdict()),
            Series(label="untouched", **untouched._asdict()),
        ],
        decimated=thinned.decimated or untouched.decimated,
        source_points=thinned.source_points + untouched.source_points,
    )

    assert mark.decimated is True
    assert mark.source_points == 5010
