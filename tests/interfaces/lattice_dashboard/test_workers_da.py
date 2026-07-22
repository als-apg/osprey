"""Tests for the dynamic aperture worker (DA boundary via tracking)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from osprey.interfaces.lattice_dashboard.workers.da import (
    build_figure,
    compute_da,
    find_da_at_angle,
)


class TestFindDaAtAngle:
    """Bisection converges toward amp_max (survives) or amp_min (lost)."""

    def test_all_survive_converges_high(self):
        ring = MagicMock()
        ring.track.return_value = (np.zeros((6, 1, 1)), {}, {})  # finite → survives
        amp = find_da_at_angle(ring, angle_rad=0.0, nturns=8, amp_min=0.0, amp_max=1.0, n_bisect=10)
        # Every step keeps the upper half → converges close to amp_max
        assert amp > 0.99

    def test_all_lost_converges_low(self):
        ring = MagicMock()
        ring.track.return_value = (np.full((6, 1, 1), np.nan), {}, {})  # lost
        amp = find_da_at_angle(ring, angle_rad=0.0, amp_min=0.0, amp_max=1.0, nturns=8, n_bisect=10)
        assert amp < 0.01

    def test_track_exception_treated_as_lost(self):
        """A tracking payload that makes np.isfinite raise counts as lost."""
        ring = MagicMock()
        # object-dtype array survives unpack_tracking but np.isfinite raises TypeError
        ring.track.return_value = np.array([[object()]], dtype=object)
        amp = find_da_at_angle(ring, angle_rad=0.0, amp_min=0.0, amp_max=1.0, nturns=8, n_bisect=6)
        assert amp < 0.02


class TestComputeDa:
    """compute_da builds a closed boundary and a positive area."""

    def test_boundary_shape_and_area(self, fodo_ring):
        da_x, da_y, amps, area = compute_da(
            fodo_ring, nturns=64, n_angles=5, amp_max=0.01, n_bisect=6
        )
        # Boundary is mirrored + closed: 2*n_angles + 1 points
        assert len(da_x) == len(da_y) == 2 * 5 + 1
        assert len(amps) == 5
        assert area > 0
        assert np.all(np.isfinite(da_x))
        assert np.all(np.isfinite(da_y))

    def test_boundary_is_closed(self, fodo_ring):
        da_x, da_y, _, _ = compute_da(fodo_ring, nturns=32, n_angles=4, amp_max=0.01, n_bisect=5)
        assert da_x[0] == da_x[-1]
        assert da_y[0] == da_y[-1]


class TestDaBuildFigure:
    """build_figure draws the DA boundary with an area annotation."""

    def _sample(self):
        theta = np.linspace(0, 2 * np.pi, 20)
        return 0.01 * np.cos(theta), 0.01 * np.sin(theta)

    def test_area_annotation_and_title(self):
        da_x, da_y = self._sample()
        fig = build_figure(da_x, da_y, area_mm2=42.5, nturns=256)
        assert "256 turns" in fig.layout.title.text
        ann_texts = [a.text for a in fig.layout.annotations]
        assert any("42.5" in t for t in ann_texts)
        # y-axis is scale-locked to x for an undistorted aspect ratio
        assert fig.layout.yaxis.scaleanchor == "x"

    def test_single_trace_without_baseline(self):
        da_x, da_y = self._sample()
        fig = build_figure(da_x, da_y, area_mm2=42.5)
        assert len(fig.data) == 1
        assert "42" in fig.data[0].name

    def test_baseline_adds_dashed_trace(self):
        da_x, da_y = self._sample()
        baseline = (da_x * 0.8, da_y * 0.8, 30.0)
        fig = build_figure(da_x, da_y, area_mm2=42.5, baseline=baseline)
        assert len(fig.data) == 2
        dashed = [t for t in fig.data if t.line and t.line.dash == "dash"]
        assert len(dashed) == 1
        assert "Baseline" in dashed[0].name
