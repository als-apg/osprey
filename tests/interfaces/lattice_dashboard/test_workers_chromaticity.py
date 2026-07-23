"""Tests for the chromaticity worker (tune vs. momentum deviation)."""

from __future__ import annotations

import numpy as np
import pytest

from osprey.interfaces.lattice_dashboard.workers import chromaticity as chrom_mod
from osprey.interfaces.lattice_dashboard.workers.chromaticity import (
    build_figure,
    compute_chromaticity,
)


class TestComputeChromaticity:
    """compute_chromaticity sweeps dp/p and returns aligned tune arrays."""

    def test_shapes_and_dp_grid(self, fodo_ring):
        dp, nux, nuy = compute_chromaticity(fodo_ring, dp_min=-0.01, dp_max=0.01, n_steps=7)
        assert len(dp) == len(nux) == len(nuy) == 7
        assert dp[0] == -0.01
        assert dp[-1] == 0.01
        assert np.all(np.isfinite(nux))
        assert np.all(np.isfinite(nuy))

    def test_slope_matches_pyat_chromaticity(self, fodo_ring):
        """The fitted slope dnu/d(dp) should match pyAT's reported chromaticity."""
        import at

        _, rd, _ = at.get_optics(fodo_ring, get_chrom=True)
        xi_x_expected, xi_y_expected = rd.chromaticity

        dp, nux, nuy = compute_chromaticity(fodo_ring, dp_min=-0.005, dp_max=0.005, n_steps=11)
        xi_x = np.polyfit(dp, nux, 1)[0]
        xi_y = np.polyfit(dp, nuy, 1)[0]

        assert xi_x == pytest.approx(xi_x_expected, rel=0.15)
        assert xi_y == pytest.approx(xi_y_expected, rel=0.15)

    def test_unstable_point_yields_nan(self, monkeypatch):
        """A dp where get_optics raises must produce NaN, not crash."""

        def boom(*args, **kwargs):
            raise RuntimeError("unstable")

        monkeypatch.setattr(chrom_mod.at, "get_optics", boom)
        dp, nux, nuy = compute_chromaticity(object(), n_steps=5)
        assert np.all(np.isnan(nux))
        assert np.all(np.isnan(nuy))


class TestChromaticityBuildFigure:
    """build_figure renders tune curves and fits the chromaticity into the title."""

    def _sample(self, n=11):
        dp = np.linspace(-0.03, 0.03, n)
        nux = 0.48 - 5.0 * dp  # xi_x = -5
        nuy = 0.10 - 3.0 * dp  # xi_y = -3
        return dp, nux, nuy

    def test_title_reports_fitted_chromaticity(self):
        dp, nux, nuy = self._sample()
        fig = build_figure(dp, nux, nuy)
        title = fig.layout.title.text
        assert "Chromaticity" in title
        # Fitted slopes recovered into the title
        assert "-5.0" in title
        assert "-3.0" in title

    def test_two_traces_without_baseline(self):
        dp, nux, nuy = self._sample()
        fig = build_figure(dp, nux, nuy)
        assert len(fig.data) == 2

    def test_baseline_adds_dashed_traces(self):
        dp, nux, nuy = self._sample()
        baseline = (dp, nux + 0.001, nuy + 0.001)
        fig = build_figure(dp, nux, nuy, baseline=baseline)
        dashed = [t for t in fig.data if t.line and t.line.dash == "dash"]
        assert len(dashed) == 2
        assert len(fig.data) == 4

    def test_all_nan_skips_fit(self):
        dp = np.linspace(-0.03, 0.03, 11)
        nan = np.full(11, np.nan)
        fig = build_figure(dp, nan, nan)
        # No fit → bare "Chromaticity" title (no xi values)
        assert fig.layout.title.text == "Chromaticity"
