"""Tests for the optics worker (beta functions + dispersion)."""

from __future__ import annotations

import numpy as np

from osprey.interfaces.lattice_dashboard.workers.optics import build_figure, compute_optics


class TestComputeOptics:
    """compute_optics returns Twiss arrays from a real pyAT ring."""

    def test_returns_four_aligned_arrays(self, fodo_ring):
        s_pos, beta_x, beta_y, eta_x, _ = compute_optics(fodo_ring)

        n = len(fodo_ring) + 1  # refpts = range(len(ring)+1)
        assert len(s_pos) == n
        assert len(beta_x) == len(beta_y) == len(eta_x) == n

    def test_summary_updates_match_the_tracked_ring(self, fodo_ring):
        """The header-summary block is computed from the same optics call."""
        _, beta_x, beta_y, _, summary = compute_optics(fodo_ring)

        assert set(summary) == {"tunes", "chromaticity", "beta_max"}
        assert len(summary["tunes"]) == len(summary["chromaticity"]) == 2
        assert summary["beta_max"] == [float(np.max(beta_x)), float(np.max(beta_y))]
        assert all(np.isfinite(v) for v in summary["tunes"])

    def test_summary_updates_follow_a_magnet_change(self, make_fodo):
        """Detuning the quads moves the tunes the chips display."""
        _, _, _, _, design = compute_optics(make_fodo())
        _, _, _, _, detuned = compute_optics(make_fodo(kf=1.1))

        assert design["tunes"] != detuned["tunes"]

    def test_physical_values(self, fodo_ring):
        s_pos, beta_x, beta_y, eta_x, _ = compute_optics(fodo_ring)

        # Beta functions are strictly positive and finite
        assert np.all(beta_x > 0)
        assert np.all(beta_y > 0)
        assert np.all(np.isfinite(eta_x))
        # s runs monotonically from 0 to the circumference
        assert s_pos[0] == 0.0
        assert np.all(np.diff(s_pos) >= 0)


class TestOpticsBuildFigure:
    """build_figure assembles the two-panel beta/dispersion subplot."""

    def _sample(self, n=5):
        s = np.linspace(0, 20, n)
        return s, np.full(n, 10.0), np.full(n, 5.0), np.full(n, 0.1)

    def test_basic_traces(self):
        s, bx, by, ex = self._sample()
        fig = build_figure(s, bx, by, ex)

        names = [t.name for t in fig.data]
        assert any("β" in n and "x" in n for n in names)  # beta_x present
        assert fig.layout.title.text == "Optics"
        # Three solid traces: beta_x, beta_y, eta_x
        assert len(fig.data) == 3

    def test_baseline_adds_dashed_traces(self):
        s, bx, by, ex = self._sample()
        baseline = (s, bx * 0.9, by * 0.9, ex * 0.9)
        fig = build_figure(s, bx, by, ex, baseline_data=baseline)

        dashed = [t for t in fig.data if t.line and t.line.dash == "dash"]
        assert len(dashed) == 3
        # Current + baseline traces
        assert len(fig.data) == 6
