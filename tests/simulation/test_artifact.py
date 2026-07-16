"""Tests for the canonical ``.mat`` build artifact of the shared ALS-U AR ring.

Covers a full serialize/reload round-trip (`build_ring` -> `save_canonical_mat`
-> `at.load_mat`) and a regeneration-equality check that guards against a stale
committed artifact by comparing a freshly built+reloaded ring against the
committed one.
"""

from __future__ import annotations

import re

import at
import numpy as np

from osprey.simulation.lattice import (
    build_ring,
    load_canonical_ring,
    save_canonical_mat,
)

# Flat-naming family prefixes (``{fam}{id:02d}``) whose ``PolynomB`` we assert
# preserved. DIPOLE is excluded here — dipoles are ``at.Dipole`` and already
# checked separately by ``_dipoles``.
_AR_MAGNET_TYPES = frozenset({"QF", "QD", "QFA", "SF", "SD", "SHF", "SHD"})
_MAGNET_FAM_RE = re.compile(r"^([A-Z]+)\d{2,}$")


def _fam_names(ring):
    return [e.FamName for e in ring]


def _pass_methods(ring):
    return [e.PassMethod for e in ring]


def _monitors(ring):
    return [e for e in ring if isinstance(e, at.Monitor)]


def _correctors(ring):
    return [e for e in ring if isinstance(e, at.Corrector)]


def _dipoles(ring):
    return [e for e in ring if isinstance(e, at.Dipole)]


def _is_ar_magnet(e):
    if not hasattr(e, "PolynomB"):
        return False
    match = _MAGNET_FAM_RE.match(e.FamName)
    return match is not None and match.group(1) in _AR_MAGNET_TYPES


def _ar_magnets(ring):
    return [e for e in ring if _is_ar_magnet(e)]


def _assert_rings_equal(a, b):
    """Assert two rings match on the canonical structural + physics fields."""
    # Order-preserving family names and pass methods.
    assert _fam_names(a) == _fam_names(b)
    assert _pass_methods(a) == _pass_methods(b)

    # Monitors.
    a_mon, b_mon = _monitors(a), _monitors(b)
    assert [e.FamName for e in a_mon] == [e.FamName for e in b_mon]

    # Correctors: length-2 KickAngle vectors.
    a_cor, b_cor = _correctors(a), _correctors(b)
    assert len(a_cor) == len(b_cor)
    for ea, eb in zip(a_cor, b_cor, strict=True):
        assert np.allclose(
            np.asarray(ea.KickAngle).ravel(), np.asarray(eb.KickAngle).ravel()
        )

    # Dipoles: entrance/exit edge angles.
    a_dip, b_dip = _dipoles(a), _dipoles(b)
    assert len(a_dip) == len(b_dip)
    for ea, eb in zip(a_dip, b_dip, strict=True):
        assert np.allclose(ea.EntranceAngle, eb.EntranceAngle)
        assert np.allclose(ea.ExitAngle, eb.ExitAngle)

    # AR magnet PolynomB.
    a_mag, b_mag = _ar_magnets(a), _ar_magnets(b)
    assert len(a_mag) == len(b_mag)
    for ea, eb in zip(a_mag, b_mag, strict=True):
        assert np.allclose(
            np.asarray(ea.PolynomB).ravel(), np.asarray(eb.PolynomB).ravel()
        )


def test_canonical_mat_roundtrip(tmp_path):
    """build_ring -> save_canonical_mat -> at.load_mat preserves all fields."""
    ring = build_ring()

    dest = tmp_path / "rt.mat"
    written = save_canonical_mat(ring, dest)
    assert written == dest

    reloaded = at.load_mat(str(dest), use="RING")

    # Field-by-field round-trip fidelity.
    assert _fam_names(reloaded) == _fam_names(ring)
    assert _pass_methods(reloaded) == _pass_methods(ring)

    ring_mon, reloaded_mon = _monitors(ring), _monitors(reloaded)
    assert [e.FamName for e in reloaded_mon] == [e.FamName for e in ring_mon]
    assert len(reloaded_mon) == 72

    ring_cor, reloaded_cor = _correctors(ring), _correctors(reloaded)
    assert len(reloaded_cor) == 144
    assert len(ring_cor) == 144
    for ea, eb in zip(ring_cor, reloaded_cor, strict=True):
        assert np.allclose(
            np.asarray(ea.KickAngle).ravel(), np.asarray(eb.KickAngle).ravel()
        )

    ring_dip, reloaded_dip = _dipoles(ring), _dipoles(reloaded)
    assert len(reloaded_dip) == 36
    assert len(ring_dip) == 36
    for ea, eb in zip(ring_dip, reloaded_dip, strict=True):
        assert np.allclose(ea.EntranceAngle, eb.EntranceAngle)
        assert np.allclose(ea.ExitAngle, eb.ExitAngle)

    ring_mag, reloaded_mag = _ar_magnets(ring), _ar_magnets(reloaded)
    assert len(reloaded_mag) == len(ring_mag)
    assert ring_mag  # non-empty guard
    for ea, eb in zip(ring_mag, reloaded_mag, strict=True):
        assert np.allclose(
            np.asarray(ea.PolynomB).ravel(), np.asarray(eb.PolynomB).ravel()
        )


def test_committed_artifact_matches_fresh_build(tmp_path):
    """Guard against a stale committed .mat: fresh build must equal committed."""
    committed = load_canonical_ring()

    fresh = build_ring()
    dest = tmp_path / "fresh.mat"
    save_canonical_mat(fresh, dest)
    reloaded_fresh = at.load_mat(str(dest), use="RING")

    _assert_rings_equal(reloaded_fresh, committed)

    assert len(committed) == 802
    assert np.isclose(committed.circumference, 182.1219508800, atol=1e-6)
