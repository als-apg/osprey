"""Tests for tracking worker fixes — tuple unpacking and sextupole detection.

Covers:
- unpack_tracking(): pyAT tuple → ndarray extraction + 4D→3D squeeze
- state.initialize(): sextupole H vs quadrupole K detection order
- compute_footprint(): merged diffusion computation
- build_figure() for footprint: diffusion coloring + resonance overlay
- add_resonance_overlay(): shared helper from _base
- LMA worker: compute_lma, extract_lattice_elements, build_figure
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from osprey.interfaces.lattice_dashboard.workers._base import (
    add_resonance_overlay,
    unpack_tracking,
)


class TestUnpackTracking:
    """Verify unpack_tracking() handles both old and new pyAT return formats."""

    def test_tuple_return(self):
        """New pyAT API returns (data, params, losses) tuple."""
        data = np.random.default_rng(1).random((6, 1, 256))
        params = {"some": "params"}
        losses = {"lost": False}
        result = (data, params, losses)

        unpacked = unpack_tracking(result)

        assert isinstance(unpacked, np.ndarray)
        np.testing.assert_array_equal(unpacked, data)

    def test_ndarray_return(self):
        """Old pyAT API returns plain ndarray — backward compat."""
        data = np.random.default_rng(2).random((6, 1, 256))

        unpacked = unpack_tracking(data)

        assert isinstance(unpacked, np.ndarray)
        np.testing.assert_array_equal(unpacked, data)

    def test_squeeze_single_particle(self):
        """4D array with nparticles=1 should be squeezed to 3D."""
        # Shape: (6, nrefpts=1, nparticles=1, nturns=256)
        data_4d = np.random.default_rng(3).random((6, 1, 1, 256))
        result = (data_4d, {}, {})

        unpacked = unpack_tracking(result)

        assert unpacked.ndim == 3
        assert unpacked.shape == (6, 1, 256)
        np.testing.assert_array_equal(unpacked, data_4d[:, :, 0, :])

    def test_no_squeeze_multi_particle(self):
        """4D array with nparticles>1 should NOT be squeezed."""
        data_4d = np.random.default_rng(4).random((6, 1, 5, 256))
        result = (data_4d, {}, {})

        unpacked = unpack_tracking(result)

        assert unpacked.ndim == 4
        assert unpacked.shape == (6, 1, 5, 256)


class TestSextupoleDetection:
    """Verify sextupoles are classified by H, not misclassified as K=0 quads."""

    @pytest.fixture
    def state(self, tmp_path):
        from osprey.interfaces.lattice_dashboard.state import LatticeState

        return LatticeState(tmp_path / "lattice")

    def _make_sextupole_elem(self, fam_name: str, h_val: float, k_val: float = 0.0):
        """Create mock element with both K and H attributes (like real sextupoles)."""
        elem = MagicMock()
        elem.FamName = fam_name
        elem.K = k_val  # Sextupoles have K=0 (quad component)
        elem.H = h_val  # Sextupoles have nonzero H
        # getattr should work naturally with MagicMock attributes
        return elem

    def _make_quadrupole_elem(self, fam_name: str, k_val: float):
        """Create mock element with K but no H attribute."""
        elem = MagicMock(spec=["FamName", "K"])
        elem.FamName = fam_name
        elem.K = k_val
        return elem

    def test_sextupole_classified_by_h(self, state):
        """Sextupoles with H!=0 and K=0 must be classified as sextupole."""
        sext = self._make_sextupole_elem("SD", h_val=21.54)
        quad = self._make_quadrupole_elem("QF", k_val=2.5)

        ring = MagicMock()
        ring.__len__ = lambda self: 2
        ring.__iter__ = lambda self: iter([sext, quad])
        ring.energy = 2e9
        ring.get_s_pos.return_value = np.array([100.0])

        rd = SimpleNamespace(tune=np.array([0.3, 0.2]), chromaticity=np.array([1.0, 1.5]))
        ld = SimpleNamespace(beta=np.array([[10.0, 5.0], [12.0, 6.0], [10.0, 5.0]]))

        mock_at = MagicMock()
        mock_at.load_m.return_value = ring
        mock_at.get_optics.return_value = (None, rd, ld)

        with patch.dict(sys.modules, {"at": mock_at}):
            result = state.initialize("/fake/lattice.m")

        families = result["families"]
        assert "SD" in families
        assert families["SD"]["type"] == "sextupole"
        assert families["SD"]["param"] == "H"
        assert families["SD"]["value"] == pytest.approx(21.54)

        assert "QF" in families
        assert families["QF"]["type"] == "quadrupole"
        assert families["QF"]["param"] == "K"


# ── Merged footprint tests ──────────────────────────────


class TestComputeFootprint:
    """Verify merged footprint returns 4 arrays including diffusion."""

    def test_compute_footprint_returns_diffusion(self):
        """compute_footprint must return (nux, nuy, amps, diffusion)."""
        from osprey.interfaces.lattice_dashboard.workers.footprint import (
            compute_footprint,
        )

        # Create mock ring with simple oscillatory tracking data
        ring = MagicMock()
        rng = np.random.default_rng(42)

        def mock_track(rin, nturns, refpts=None):
            """Return sinusoidal TBT data with known tune."""
            turns = np.arange(nturns)
            # Tune ~ 0.25 with slight drift between halves
            nux = 0.25 + rng.normal(0, 1e-6)
            nuy = 0.15 + rng.normal(0, 1e-6)
            x = 1e-3 * np.sin(2 * np.pi * nux * turns)
            y = 1e-3 * np.sin(2 * np.pi * nuy * turns)
            data = np.zeros((6, 1, nturns))
            data[0, 0, :] = x
            data[2, 0, :] = y
            return (data, {}, {})

        ring.track = mock_track

        nux, nuy, amps, diffusion = compute_footprint(ring, n_amp=3, n_half=128)

        # Should have some surviving particles (3x3 = 9 max)
        assert len(nux) > 0
        assert len(nux) == len(nuy) == len(amps) == len(diffusion)
        # Diffusion should be finite negative values (log10 of small number)
        assert np.all(np.isfinite(diffusion))
        assert np.all(diffusion < 0)
        # Amps should be positive (in mm)
        assert np.all(amps > 0)

    def test_footprint_tracks_double_turns(self):
        """compute_footprint should track 2*n_half turns."""
        from osprey.interfaces.lattice_dashboard.workers.footprint import (
            compute_footprint,
        )

        ring = MagicMock()
        tracked_nturns = []

        def mock_track(rin, nturns, refpts=None):
            tracked_nturns.append(nturns)
            turns = np.arange(nturns)
            data = np.zeros((6, 1, nturns))
            data[0, 0, :] = 1e-3 * np.sin(2 * np.pi * 0.25 * turns)
            data[2, 0, :] = 1e-3 * np.sin(2 * np.pi * 0.15 * turns)
            return (data, {}, {})

        ring.track = mock_track

        compute_footprint(ring, n_amp=2, n_half=64)

        # Every call should use 2*n_half = 128
        assert all(n == 128 for n in tracked_nturns)


class TestFootprintBuildFigure:
    """Verify build_figure uses diffusion for marker coloring."""

    def test_diffusion_coloring(self):
        """Figure markers should use Viridis_r colorscale with diffusion values."""
        from osprey.interfaces.lattice_dashboard.workers.footprint import build_figure

        nux = np.array([0.25, 0.251, 0.252])
        nuy = np.array([0.15, 0.151, 0.152])
        amps = np.array([1.0, 2.0, 3.0])
        diffusion = np.array([-10.0, -6.0, -3.0])

        fig = build_figure(nux, nuy, amps, diffusion)

        # Find the footprint trace (should be the first/only non-resonance trace)
        footprint_trace = None
        for trace in fig.data:
            if trace.name == "Footprint":
                footprint_trace = trace
                break

        assert footprint_trace is not None
        # Plotly resolves named colorscales to tuples; check the reversed order
        # (Viridis_r starts with yellow #fde725, ends with purple #440154)
        cs = footprint_trace.marker.colorscale
        assert cs[0][1] == "#fde725"  # yellow at start = reversed Viridis
        np.testing.assert_array_equal(footprint_trace.marker.color, diffusion)
        assert footprint_trace.marker.cmin == -12
        assert footprint_trace.marker.cmax == -2

    def test_design_tune_marker(self):
        """Design tune should appear as a red star marker."""
        from osprey.interfaces.lattice_dashboard.workers.footprint import build_figure

        nux = np.array([0.25])
        nuy = np.array([0.15])
        amps = np.array([1.0])
        diffusion = np.array([-8.0])

        fig = build_figure(nux, nuy, amps, diffusion, design_tune=(0.26, 0.16))

        design_traces = [t for t in fig.data if t.name == "Design tune"]
        assert len(design_traces) == 1
        assert design_traces[0].marker.color == "red"
        assert design_traces[0].marker.symbol == "star"

    def test_resonance_overlay_present(self):
        """Figure should include resonance line traces."""
        from osprey.interfaces.lattice_dashboard.workers.footprint import build_figure

        nux = np.array([0.25, 0.26])
        nuy = np.array([0.15, 0.16])
        amps = np.array([1.0, 2.0])
        diffusion = np.array([-8.0, -6.0])

        fig = build_figure(nux, nuy, amps, diffusion)

        # Should have resonance lines (many gray line traces)
        resonance_traces = [t for t in fig.data if t.mode == "lines" and t.showlegend is False]
        assert len(resonance_traces) > 0


# ── Resonance overlay shared helper ─────────────────────


class TestResonanceOverlay:
    """Verify add_resonance_overlay works from _base module."""

    def test_adds_line_traces(self):
        """add_resonance_overlay should add Scatter traces with mode='lines'."""
        import plotly.graph_objects as go

        fig = go.Figure()
        add_resonance_overlay(fig, (0.2, 0.3), (0.1, 0.2))

        assert len(fig.data) > 0
        for trace in fig.data:
            assert trace.mode == "lines"
            assert trace.showlegend is False

    def test_empty_range_no_crash(self):
        """Degenerate range should not crash, just produce no traces."""
        import plotly.graph_objects as go

        fig = go.Figure()
        add_resonance_overlay(fig, (0.5, 0.5), (0.5, 0.5))
        # Should not raise


# ── LMA worker tests ────────────────────────────────────


class TestComputeLMA:
    """Verify LMA computation with mocked pyAT bisection."""

    def test_lma_compute_basic(self):
        """compute_lma should return s_pos, dp_plus, dp_minus arrays."""
        from osprey.interfaces.lattice_dashboard.workers.lma import compute_lma

        # Rotated ring mock: track() returns finite data (particle survives)
        rotated = MagicMock()
        survived_data = np.zeros((6, 1, 1))  # 4D shape, squeezed by unpack_tracking
        rotated.track.return_value = survived_data

        ring = MagicMock()
        ring.get_s_pos.return_value = np.linspace(0, 200, 101)
        ring.__len__ = lambda self: 100
        ring.rotate.return_value = rotated

        n_refpts = 20
        dp_max = 0.05
        s, dpp, dpm = compute_lma(
            ring,
            n_refpts=n_refpts,
            dp_max=dp_max,
            n_bisect=5,
            sector_length=100.0,
        )

        assert len(s) == n_refpts
        assert len(s) == len(dpp) == len(dpm)
        # All particles survive → bisection converges near dp_max
        assert np.all(dpp > 0)
        assert np.all(dpm > 0)

    def test_lma_all_lost(self):
        """compute_lma should return near-zero acceptance when all particles are lost."""
        from osprey.interfaces.lattice_dashboard.workers.lma import compute_lma

        # Rotated ring mock: track() returns NaN (particle lost)
        rotated = MagicMock()
        lost_data = np.full((6, 1, 1), np.nan)
        rotated.track.return_value = lost_data

        ring = MagicMock()
        ring.get_s_pos.return_value = np.linspace(0, 200, 101)
        ring.__len__ = lambda self: 100
        ring.rotate.return_value = rotated

        s, dpp, dpm = compute_lma(ring, n_refpts=10, n_bisect=5, sector_length=100.0)

        assert len(s) == 10
        # All particles lost → bisection converges to 0
        np.testing.assert_allclose(dpp, 0.0, atol=1e-10)
        np.testing.assert_allclose(dpm, 0.0, atol=1e-10)


class TestExtractLatticeElements:
    """Verify lattice element extraction for the lattice strip."""

    def test_extracts_element_types(self):
        """Should correctly classify dipoles, quads, and sextupoles."""
        from osprey.interfaces.lattice_dashboard.workers.lma import (
            extract_lattice_elements,
        )

        dipole = MagicMock(spec=["FamName", "BendingAngle", "Length"])
        dipole.FamName = "BM"
        dipole.BendingAngle = 0.1

        quad = MagicMock(spec=["FamName", "K", "Length"])
        quad.FamName = "QF"
        quad.K = 2.5

        sext = MagicMock(spec=["FamName", "H", "K", "Length"])
        sext.FamName = "SD"
        sext.H = 21.5
        sext.K = 0.0  # sextupoles have K=0

        drift = MagicMock(spec=["FamName", "Length"])
        drift.FamName = "DR"

        ring = MagicMock()
        ring.__iter__ = lambda self: iter([dipole, quad, sext, drift])
        # s-positions: 0, 1, 2, 3, 4
        ring.get_s_pos.return_value = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        elements = extract_lattice_elements(ring, sector_length=10.0)

        types = [e["type"] for e in elements]
        assert "dipole" in types
        assert "quadrupole" in types
        assert "sextupole" in types
        # Drift should not appear
        assert "drift" not in types

        # Check dipole
        bm = next(e for e in elements if e["name"] == "BM")
        assert bm["type"] == "dipole"
        assert bm["s_start"] == 0.0
        assert bm["s_end"] == 1.0


class TestLMABuildFigure:
    """Verify LMA figure construction."""

    def test_build_figure_basic(self):
        """build_figure should create a subplot figure with traces."""
        from osprey.interfaces.lattice_dashboard.workers.lma import build_figure

        s_pos = np.linspace(0, 50, 20)
        dp_plus = np.full(20, 0.03)
        dp_minus = np.full(20, 0.025)
        elements = [
            {"s_start": 0, "s_end": 2, "type": "dipole", "name": "BM", "strength": 0.1},
            {"s_start": 5, "s_end": 6, "type": "quadrupole", "name": "QF", "strength": 2.0},
        ]

        fig = build_figure(s_pos, dp_plus, dp_minus, elements)

        # Should have dp+ and dp- traces
        assert len(fig.data) >= 2
        trace_names = [t.name for t in fig.data]
        assert "dp+" in trace_names
        assert "dp-" in trace_names
        # Title should mention sector
        assert "sector" in fig.layout.title.text.lower()

    def test_build_figure_with_baseline(self):
        """Baseline should add dashed traces."""
        from osprey.interfaces.lattice_dashboard.workers.lma import build_figure

        s_pos = np.linspace(0, 50, 10)
        dp_plus = np.full(10, 0.03)
        dp_minus = np.full(10, 0.025)
        baseline = (s_pos, np.full(10, 0.028), np.full(10, 0.022))

        fig = build_figure(s_pos, dp_plus, dp_minus, [], baseline=baseline)

        # Should have baseline traces (dashed)
        dashed_traces = [t for t in fig.data if t.line and t.line.dash == "dash"]
        assert len(dashed_traces) == 2

    def test_lattice_strip_shapes(self):
        """Lattice elements should produce shapes in the figure."""
        from osprey.interfaces.lattice_dashboard.workers.lma import build_figure

        s_pos = np.linspace(0, 50, 10)
        dp_plus = np.full(10, 0.03)
        dp_minus = np.full(10, 0.025)
        elements = [
            {"s_start": 0, "s_end": 2, "type": "dipole", "name": "BM", "strength": 0.1},
            {"s_start": 5, "s_end": 6, "type": "quadrupole", "name": "QF", "strength": 2.0},
            {"s_start": 10, "s_end": 10.5, "type": "sextupole", "name": "SD", "strength": 20.0},
        ]

        fig = build_figure(s_pos, dp_plus, dp_minus, elements)

        # Should have shapes for the lattice strip
        shapes = fig.layout.shapes
        assert shapes is not None
        assert len(shapes) == 3
