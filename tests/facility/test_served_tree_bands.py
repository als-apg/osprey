"""The band a refused write is checked against, derived from the served tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey_connectors.simulation.view import SimulatorView
from tests.facility.served_tree import (
    LEAK_VISIBILITY_FACTOR,
    READBACK_FLOOR,
    assert_unmoved,
    assert_write_visible,
    served_active,
    served_tree,
    unmoved_band,
)

READBACK = "T:Q1:CUR:RB"


def test_a_channel_with_no_declared_motion_gets_the_floor(tmp_path: Path) -> None:
    view = SimulatorView.open(served_tree(tmp_path, readings=[READBACK]))

    assert unmoved_band(view, READBACK, served_active(view)) == READBACK_FLOOR


def test_the_band_is_twice_the_declared_motion_plus_the_floor(tmp_path: Path) -> None:
    view = SimulatorView.open(
        served_tree(
            tmp_path,
            readings=[READBACK],
            channels={READBACK: {"simulation": {"noise": {"absolute": 0.01}}}},
        )
    )
    active = served_active(view)
    motion = view.motion_envelope(READBACK, active=active)

    assert motion > 0.0
    assert unmoved_band(view, READBACK, active) == 2 * motion + READBACK_FLOOR


def test_a_tree_with_no_active_state_serves_nominal(tmp_path: Path) -> None:
    view = SimulatorView.open(served_tree(tmp_path, readings=[READBACK]))

    assert served_active(view) == ["nominal"]


def test_a_demand_inside_the_visibility_margin_is_refused() -> None:
    band = 0.01
    with pytest.raises(AssertionError, match="too close"):
        assert_write_visible(1.0 + LEAK_VISIBILITY_FACTOR * band / 2, 1.0, band)
    assert_write_visible(1.0 + LEAK_VISIBILITY_FACTOR * band, 1.0, band)


def test_a_half_landed_write_is_not_unmoved() -> None:
    before, written, after = 0.0, 5.0, 2.0

    # A distance comparison (|after-before| < |after-written|) takes this triple
    # for unmoved; the band does not.
    assert abs(after - before) < abs(after - written)
    with pytest.raises(AssertionError, match=READBACK):
        assert_unmoved(READBACK, before, after, READBACK_FLOOR)
