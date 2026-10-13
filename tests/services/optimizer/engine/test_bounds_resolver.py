"""Unit tests for ``resolve_bounds``."""

from __future__ import annotations

import math

import pytest

from osprey.services.optimizer.models.config import VariableOverride
from osprey.services.optimizer.services.bounds_resolver import (
    effective_step_size,
    resolve_bounds,
    snap_value_to_grid,
)


def test_effective_step_size_prefers_override_then_hardware() -> None:
    """The single step source shared by the run and the cockpit preview: override step
    when set, else the hardware step. Keeping it in one place is what prevents the two
    code paths from snapping to different grids."""
    assert effective_step_size(VariableOverride(step_size=2.0), 0.1) == 2.0
    # No override, or an override without a step, both fall back to the hardware step.
    assert effective_step_size(None, 0.1) == 0.1
    assert effective_step_size(VariableOverride(min=1.0, max=9.0), 0.1) == 0.1


@pytest.mark.parametrize(
    "override,current_value,safety_min,safety_max,expected_min,expected_max",
    [
        (None, 5.0, 0.0, 10.0, 0.0, 10.0),
        (VariableOverride(min=2.0, max=8.0), 5.0, 0.0, 10.0, 2.0, 8.0),
        (VariableOverride(min=-1.0), 0.0, -5.0, 5.0, -1.0, 5.0),
        # Absolute mode does not consult current_value, so NaN must not poison the result.
        (VariableOverride(min=2.0, max=8.0), math.nan, 0.0, 10.0, 2.0, 8.0),
    ],
    ids=[
        "no_override",
        "absolute_min_max",
        "partial_min_falls_back_to_safety_max",
        "nan_current_value_tolerated",
    ],
)
def test_absolute_bounds_unclamped(
    override: VariableOverride | None,
    current_value: float,
    safety_min: float,
    safety_max: float,
    expected_min: float,
    expected_max: float,
) -> None:
    resolved = resolve_bounds(
        override, current_value=current_value, safety_min=safety_min, safety_max=safety_max
    )
    assert resolved.min == expected_min
    assert resolved.max == expected_max
    assert resolved.was_clamped is False


@pytest.mark.parametrize(
    "override,current_value,safety_min,safety_max,expected_min,expected_max",
    [
        (VariableOverride(window_fraction=0.1), 10.0, 0.0, 100.0, 5.0, 15.0),
        (VariableOverride(window_abs=2.0), 5.0, 0.0, 10.0, 3.0, 7.0),
        (VariableOverride(window_fraction=0.1), 0.0, -5.0, 5.0, -0.5, 0.5),
        (VariableOverride(window_fraction=0.1), -10.0, -100.0, 0.0, -15.0, -5.0),
        # No step_size argument at all: the resolved window is left unsnapped.
        (VariableOverride(window_abs=0.123), 5.0, 0.0, 10.0, 4.877, 5.123),
    ],
    ids=[
        "fraction_within_safety",
        "abs_within_safety",
        "fraction_independent_of_current_magnitude",
        "fraction_with_negative_current",
        "no_step_size_leaves_bounds_unsnapped",
    ],
)
def test_window_mode_within_safety_unclamped(
    override: VariableOverride,
    current_value: float,
    safety_min: float,
    safety_max: float,
    expected_min: float,
    expected_max: float,
) -> None:
    resolved = resolve_bounds(
        override, current_value=current_value, safety_min=safety_min, safety_max=safety_max
    )
    assert resolved.min == pytest.approx(expected_min)
    assert resolved.max == pytest.approx(expected_max)
    assert resolved.was_clamped is False


@pytest.mark.parametrize(
    (
        "override",
        "current_value",
        "safety_min",
        "safety_max",
        "expected_min",
        "expected_max",
        "req_min",
        "req_max",
    ),
    [
        # range=12, fraction=0.5 → width 6 → requested current ± 3 = [7, 13]; clamped to [7, 12]
        pytest.param(
            VariableOverride(window_fraction=0.5),
            10.0,
            0.0,
            12.0,
            7.0,
            12.0,
            7.0,
            13.0,
            id="window_fraction_clamps_max",
        ),
        # Window centered entirely outside safety: collapse to the nearest safety endpoint.
        pytest.param(
            VariableOverride(window_abs=1.0),
            100.0,
            0.0,
            10.0,
            10.0,
            10.0,
            99.0,
            101.0,
            id="collapses_above_safety",
        ),
        pytest.param(
            VariableOverride(window_abs=1.0),
            -100.0,
            0.0,
            10.0,
            0.0,
            0.0,
            -101.0,
            -99.0,
            id="collapses_below_safety",
        ),
        pytest.param(
            VariableOverride(min=-100.0, max=100.0),
            0.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            -100.0,
            100.0,
            id="absolute_override_clamped",
        ),
    ],
)
def test_clamped_against_safety(
    override: VariableOverride,
    current_value: float,
    safety_min: float,
    safety_max: float,
    expected_min: float,
    expected_max: float,
    req_min: float,
    req_max: float,
) -> None:
    """Anything reaching outside the safety range is clamped to it, and the
    originally-requested span is echoed back so the caller can see the clamp."""
    resolved = resolve_bounds(
        override, current_value=current_value, safety_min=safety_min, safety_max=safety_max
    )
    assert resolved.min == pytest.approx(expected_min)
    assert resolved.max == pytest.approx(expected_max)
    assert resolved.was_clamped is True
    assert resolved.requested_min == pytest.approx(req_min)
    assert resolved.requested_max == pytest.approx(req_max)


@pytest.mark.parametrize(
    ("override", "current_value"),
    [
        pytest.param(
            VariableOverride(window_fraction=0.1), math.nan, id="nan_in_window_fraction_mode"
        ),
        pytest.param(VariableOverride(window_abs=1.0), math.inf, id="inf_in_window_abs_mode"),
    ],
)
def test_non_finite_current_value_in_window_mode_raises(
    override: VariableOverride, current_value: float
) -> None:
    """Window modes centre on the current reading, so a non-finite one must refuse
    rather than produce a bogus range."""
    with pytest.raises(ValueError, match="finite current_value"):
        resolve_bounds(override, current_value=current_value, safety_min=0.0, safety_max=10.0)


@pytest.mark.parametrize(
    ("safety_min", "safety_max", "match"),
    [
        pytest.param(10.0, 0.0, "inverted", id="inverted"),
        pytest.param(0.0, math.inf, "non-finite", id="inf_safety_max"),
        pytest.param(math.nan, 10.0, "non-finite", id="nan_safety_min"),
    ],
)
def test_invalid_safety_range_raises(safety_min: float, safety_max: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_bounds(None, current_value=0.0, safety_min=safety_min, safety_max=safety_max)


class TestStepSnapping:
    """Snap the resolved interval to the step grid anchored at safety_min."""

    def test_step_snap_shrinks_window_inward(self) -> None:
        # Phase-tuning shape: safety [-165, 165], step 0.1, window_fraction=0.1 around 12.345.
        # half-width = 0.1 * 330 / 2 = 16.5 → requested [-4.155, 28.845] → snapped [-4.1, 28.8]
        override = VariableOverride(window_fraction=0.1)
        resolved = resolve_bounds(
            override,
            current_value=12.345,
            safety_min=-165.0,
            safety_max=165.0,
            step_size=0.1,
        )
        assert resolved.min == pytest.approx(-4.1)
        assert resolved.max == pytest.approx(28.8)
        assert resolved.requested_min == pytest.approx(-4.155)
        assert resolved.requested_max == pytest.approx(28.845)
        assert resolved.was_clamped is True

    @pytest.mark.parametrize(
        "override,current_value,safety_min,safety_max,step_size,expected_min,expected_max",
        [
            (None, 0.0, 0.0, 10.0, 0.5, 0.0, 10.0),
            (VariableOverride(min=2.0, max=8.0), 5.0, 0.0, 10.0, 1.0, 2.0, 8.0),
            (None, 2.5, 2.5, 2.5, 0.1, 2.5, 2.5),
        ],
        ids=[
            "safety_endpoints_preserved_when_aligned",
            "does_not_shrink_when_already_aligned",
            "zero_safety_range_returns_single_point",
        ],
    )
    def test_step_snap_unclamped(
        self,
        override: VariableOverride | None,
        current_value: float,
        safety_min: float,
        safety_max: float,
        step_size: float,
        expected_min: float,
        expected_max: float,
    ) -> None:
        resolved = resolve_bounds(
            override,
            current_value=current_value,
            safety_min=safety_min,
            safety_max=safety_max,
            step_size=step_size,
        )
        assert resolved.min == expected_min
        assert resolved.max == expected_max
        assert resolved.was_clamped is False

    @pytest.mark.parametrize(
        ("override", "current_value", "step_size", "expected_points"),
        [
            # Requested [2.3, 2.7] with step=1.0 has no grid point inside → collapse to nearest grid point
            # of midpoint (2.5). round(2.5) is 2 under banker's rounding.
            pytest.param(
                VariableOverride(min=2.3, max=2.7), 2.5, 1.0, (2.0, 3.0), id="no_grid_point_inside"
            ),
            # Override fully outside safety collapses to safety_max; that point is on-grid (10.0 is multiple of 0.5).
            pytest.param(
                VariableOverride(min=20.0, max=30.0), 25.0, 0.5, (10.0,), id="after_safety_collapse"
            ),
        ],
    )
    def test_step_snap_collapses_to_single_grid_point(
        self,
        override: VariableOverride,
        current_value: float,
        step_size: float,
        expected_points: tuple[float, ...],
    ) -> None:
        resolved = resolve_bounds(
            override,
            current_value=current_value,
            safety_min=0.0,
            safety_max=10.0,
            step_size=step_size,
        )
        assert resolved.min == resolved.max
        assert resolved.min in expected_points
        assert resolved.was_clamped is True

    def test_step_snap_tolerates_floating_point_noise(self) -> None:
        # 0.1 + 0.2 == 0.30000000000000004 → without an epsilon, this would snap down to 0.2.
        override = VariableOverride(min=0.1 + 0.2, max=0.9)
        resolved = resolve_bounds(
            override, current_value=0.5, safety_min=0.0, safety_max=1.0, step_size=0.1
        )
        assert resolved.min == pytest.approx(0.3)
        assert resolved.max == pytest.approx(0.9)

    @pytest.mark.parametrize("bad_step", [0.0, -0.1, math.nan, math.inf, -math.inf])
    def test_invalid_step_size_raises(self, bad_step: float) -> None:
        with pytest.raises(ValueError, match="step_size must be a finite positive number"):
            resolve_bounds(
                None, current_value=5.0, safety_min=0.0, safety_max=10.0, step_size=bad_step
            )


class TestSnapValueToGrid:
    """Quantize a single candidate value onto the step grid (used at the eval seam)."""

    @pytest.mark.parametrize(
        ("value", "vmin", "vmax", "step", "expected"),
        [
            (2.0, 0.0, 10.0, 1.0, 2.0),
            (2.4, 0.0, 10.0, 1.0, 2.0),
            (2.6, 0.0, 10.0, 1.0, 3.0),
            (2.5, 0.0, 10.0, 1.0, 2.0),  # banker's rounding -> 2.0
            (-3.0, 0.0, 10.0, 1.0, 0.0),  # clamped below min
            (99.0, 0.0, 10.0, 1.0, 10.0),  # clamped above max
            # Anchor is vmin (2.3), so grid is 2.3, 2.8, 3.3, ...; 3.0 -> 2.8.
            (3.0, 2.3, 5.3, 0.5, 2.8),
        ],
    )
    def test_snaps_value_onto_grid(
        self, value: float, vmin: float, vmax: float, step: float, expected: float
    ) -> None:
        assert snap_value_to_grid(value, vmin=vmin, vmax=vmax, step=step) == pytest.approx(expected)

    def test_is_idempotent(self) -> None:
        once = snap_value_to_grid(7.31, vmin=0.0, vmax=10.0, step=0.2)
        assert snap_value_to_grid(once, vmin=0.0, vmax=10.0, step=0.2) == pytest.approx(once)
