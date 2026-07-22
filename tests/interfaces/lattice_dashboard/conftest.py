"""Shared fixtures for lattice dashboard physics-worker tests.

Provides small, in-memory pyAT FODO rings so the compute workers exercise
the real pyAT code paths (get_optics, tracking, rotate) without loading a
full facility lattice.  The rings are deliberately tiny and stable so that
tracking-based workers (DA, LMA, footprint) run in well under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

at = pytest.importorskip("at")


def _build_fodo(kf: float = 1.0, with_sextupole: bool = False) -> at.Lattice:
    """Construct a small, stable FODO ring.

    Two identical cells of focusing/defocusing quads, drifts and dipoles.
    kf=1.0 yields a stable working point (tune ~ [0.48, 0.10]).
    """
    d = at.Drift("DR", 0.5)
    qf = at.Quadrupole("QF", 0.2, kf)
    qd = at.Quadrupole("QD", 0.2, -kf)
    b = at.Dipole("BM", 0.5, np.pi / 8)
    if with_sextupole:
        sf = at.Sextupole("SF", 0.1, 5.0)
        cell = [qf, d, sf, b, d, qd, d, b, d]
    else:
        cell = [qf, d, b, d, qd, d, b, d]
    return at.Lattice(cell * 2, name="FODO", energy=2e9)


@pytest.fixture
def make_fodo():
    """Factory returning fresh FODO rings (avoids cross-test mutation)."""
    return _build_fodo


@pytest.fixture
def fodo_ring():
    """A fresh, stable FODO ring without sextupoles."""
    return _build_fodo()


@pytest.fixture
def fodo_ring_sext():
    """A fresh, stable FODO ring including a sextupole family."""
    return _build_fodo(with_sextupole=True)
