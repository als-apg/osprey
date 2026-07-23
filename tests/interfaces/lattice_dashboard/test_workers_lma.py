"""Tests for the LMA worker exercised against a real pyAT ring.

The tracking-mock cases for compute_lma / build_figure / extract_lattice_elements
live in tests/interfaces/test_lattice_tracking.py; here we cover the bisection
helper and the real-ring compute/extract paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from osprey.interfaces.lattice_dashboard.workers.lma import (
    compute_lma,
    extract_lattice_elements,
    find_ma_at_refpt,
)


class TestFindMaAtRefpt:
    """find_ma_at_refpt bisects the positive and negative dp acceptance."""

    def test_all_survive_converges_high(self):
        rotated = MagicMock()
        rotated.track.return_value = (np.zeros((6, 1, 1)), {}, {})
        ring = MagicMock()
        ring.rotate.return_value = rotated

        dp_plus, dp_minus = find_ma_at_refpt(ring, refpt=0, nturns=8, dp_max=0.05, n_bisect=10)
        assert dp_plus > 0.049
        assert dp_minus > 0.049
        ring.rotate.assert_called_once_with(0)

    def test_all_lost_converges_zero(self):
        rotated = MagicMock()
        rotated.track.return_value = (np.full((6, 1, 1), np.nan), {}, {})
        ring = MagicMock()
        ring.rotate.return_value = rotated

        dp_plus, dp_minus = find_ma_at_refpt(ring, refpt=3, nturns=8, dp_max=0.05, n_bisect=10)
        assert dp_plus < 1e-3
        assert dp_minus < 1e-3


class TestComputeLmaRealRing:
    """compute_lma over a real ring returns aligned, physical acceptance arrays."""

    def test_shapes_and_positivity(self, fodo_ring):
        s, dpp, dpm = compute_lma(
            fodo_ring, n_refpts=6, nturns=64, dp_max=0.05, n_bisect=6, sector_length=None
        )
        assert len(s) == len(dpp) == len(dpm)
        assert len(s) <= 6
        # Acceptance is non-negative and bounded by dp_max
        assert np.all(dpp >= 0) and np.all(dpp <= 0.05 + 1e-9)
        assert np.all(dpm >= 0) and np.all(dpm <= 0.05 + 1e-9)
        # s-positions are sorted along the sector
        assert np.all(np.diff(s) >= 0)

    def test_sector_length_limits_refpts(self, fodo_ring):
        """A half-ring sector should yield fewer refpts than the full ring."""
        circ = float(fodo_ring.get_s_pos(len(fodo_ring))[0])
        s_full, _, _ = compute_lma(
            fodo_ring, n_refpts=500, nturns=16, dp_max=0.02, n_bisect=3, sector_length=circ
        )
        s_half, _, _ = compute_lma(
            fodo_ring, n_refpts=500, nturns=16, dp_max=0.02, n_bisect=3, sector_length=circ / 2
        )
        assert len(s_half) < len(s_full)
        assert np.all(s_half < circ / 2)


class TestExtractLatticeElementsRealRing:
    """extract_lattice_elements classifies real pyAT elements, skipping drifts."""

    def test_classifies_and_skips_drifts(self, fodo_ring_sext):
        circ = float(fodo_ring_sext.get_s_pos(len(fodo_ring_sext))[0])
        elements = extract_lattice_elements(fodo_ring_sext, sector_length=circ)

        types = {e["type"] for e in elements}
        assert {"dipole", "quadrupole", "sextupole"} <= types
        assert "drift" not in types
        # Each element carries a bounded, ordered s-span within the sector
        for e in elements:
            assert e["s_start"] < e["s_end"] <= circ + 1e-9
            assert e["name"]

    def test_sector_truncation(self, fodo_ring_sext):
        """Only elements starting before sector_length are returned."""
        circ = float(fodo_ring_sext.get_s_pos(len(fodo_ring_sext))[0])
        half = extract_lattice_elements(fodo_ring_sext, sector_length=circ / 2)
        assert all(e["s_start"] < circ / 2 for e in half)
