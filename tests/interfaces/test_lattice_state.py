"""Tests for LatticeState.initialize() — empty refpts crash (GH bug).

Reproduces the ValueError when at.get_optics() returns an empty ld.beta
array (shape (0, 2)), which happens on the ALSU AR lattice when refpts
is not explicitly passed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from osprey.interfaces.lattice_dashboard.state import LatticeState


@pytest.fixture
def state(tmp_path):
    return LatticeState(tmp_path / "lattice")


def _make_mock_ring(n_elements: int = 10, energy: float = 2e9):
    """Create a mock AT ring with n_elements."""
    ring = MagicMock()
    ring.__len__ = lambda self: n_elements
    ring.energy = energy
    ring.get_s_pos.return_value = np.array([100.0])
    ring.__iter__ = lambda self: iter([])
    return ring


def _make_lindata(n_refpts: int):
    """Create a mock lindata object with n_refpts rows."""
    if n_refpts == 0:
        beta = np.empty((0, 2))
    else:
        beta = np.random.default_rng(42).uniform(1, 30, size=(n_refpts, 2))
    return SimpleNamespace(beta=beta)


def _make_mock_at(ring, rd, ld):
    """Create a mock 'at' module with load_lattice and get_optics."""
    mock_at = MagicMock()
    mock_at.load_lattice.return_value = ring
    mock_at.get_optics.return_value = (None, rd, ld)
    return mock_at


class TestInitializeRefpts:
    """Verify that initialize() passes explicit refpts to get_optics."""

    def test_empty_beta_no_crash(self, state):
        """Previously crashed with ValueError on np.max of empty array."""
        ring = _make_mock_ring()
        ld_empty = _make_lindata(0)
        rd = SimpleNamespace(tune=np.array([0.3, 0.2]), chromaticity=np.array([1.0, 1.5]))
        mock_at = _make_mock_at(ring, rd, ld_empty)

        with patch.dict(sys.modules, {"at": mock_at}):
            result = state.initialize("/fake/lattice.m")

        assert result["summary"]["beta_max"] == [0.0, 0.0]

    def test_normal_beta(self, state):
        """Normal case: non-empty beta array produces correct max values."""
        n_elements = 10
        ring = _make_mock_ring(n_elements)
        ld = _make_lindata(n_elements + 1)
        rd = SimpleNamespace(tune=np.array([0.3, 0.2]), chromaticity=np.array([1.0, 1.5]))
        mock_at = _make_mock_at(ring, rd, ld)

        with patch.dict(sys.modules, {"at": mock_at}):
            result = state.initialize("/fake/lattice.m")

        expected_bx = float(np.max(ld.beta[:, 0]))
        expected_by = float(np.max(ld.beta[:, 1]))
        assert result["summary"]["beta_max"] == pytest.approx([expected_bx, expected_by])

    def test_refpts_passed_to_get_optics(self, state):
        """Verify that initialize() passes refpts=range(len(ring)+1)."""
        n_elements = 10
        ring = _make_mock_ring(n_elements)
        ld = _make_lindata(n_elements + 1)
        rd = SimpleNamespace(tune=np.array([0.3, 0.2]), chromaticity=np.array([1.0, 1.5]))
        mock_at = _make_mock_at(ring, rd, ld)

        with patch.dict(sys.modules, {"at": mock_at}):
            state.initialize("/fake/lattice.m")

        call_kwargs = mock_at.get_optics.call_args
        assert call_kwargs.kwargs.get("refpts") == range(n_elements + 1)
        assert call_kwargs.kwargs.get("get_chrom") is True
