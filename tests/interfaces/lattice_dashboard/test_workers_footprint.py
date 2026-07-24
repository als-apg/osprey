"""Tests for footprint worker helpers not covered by test_lattice_tracking.

Covers get_tune_fft frequency recovery and the baseline/design-tune overlay
branches of build_figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from osprey.interfaces.lattice_dashboard.workers.footprint import build_figure, get_tune_fft


class TestGetTuneFft:
    """get_tune_fft recovers the dominant frequency of a turn-by-turn signal."""

    def test_recovers_known_tune(self):
        nturns = 256
        true_tune = 0.237
        turns = np.arange(nturns)
        signal = np.sin(2 * np.pi * true_tune * turns)
        recovered = get_tune_fft(signal)
        # Zero-padded FFT resolves the fractional tune to within a bin
        assert recovered == pytest.approx(true_tune, abs=2e-3)

    def test_returns_fraction_in_unit_interval(self):
        turns = np.arange(128)
        signal = np.cos(2 * np.pi * 0.41 * turns)
        recovered = get_tune_fft(signal)
        assert 0.0 <= recovered <= 0.5


class TestFootprintBuildFigureOverlays:
    """build_figure overlay branches: baseline footprint and baseline tune."""

    def _data(self, n=4):
        rng = np.random.default_rng(0)
        nux = 0.25 + rng.normal(0, 1e-3, n)
        nuy = 0.15 + rng.normal(0, 1e-3, n)
        amps = np.linspace(0.5, 2.0, n)
        diffusion = np.linspace(-10, -4, n)
        return nux, nuy, amps, diffusion

    def test_survival_title(self):
        nux, nuy, amps, diffusion = self._data(4)
        fig = build_figure(nux, nuy, amps, diffusion, n_total=9)
        assert "4/9 survived" in fig.layout.title.text

    def test_baseline_footprint_trace(self):
        nux, nuy, amps, diffusion = self._data()
        baseline = (nux + 0.01, nuy + 0.01, amps)
        fig = build_figure(nux, nuy, amps, diffusion, baseline=baseline)
        assert any(t.name == "Baseline footprint" for t in fig.data)

    def test_baseline_and_design_tune_markers(self):
        nux, nuy, amps, diffusion = self._data()
        fig = build_figure(
            nux,
            nuy,
            amps,
            diffusion,
            design_tune=(0.26, 0.16),
            baseline_tune=(0.25, 0.15),
        )
        names = [t.name for t in fig.data]
        assert "Design tune" in names
        assert "Baseline tune" in names

    def test_empty_footprint_no_resonance_overlay(self):
        """Zero survivors must not raise and adds no resonance lines."""
        empty = np.array([])
        fig = build_figure(empty, empty, empty, empty, n_total=9)
        # No survivors → no gray resonance line traces
        assert not any(t.mode == "lines" and t.showlegend is False for t in fig.data)
        assert "0/9 survived" in fig.layout.title.text
