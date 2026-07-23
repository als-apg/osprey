"""Ground-truth locks for the hand-ported ALS-U AR lattice.

These tests assert measured values against the deterministic outputs of
:func:`osprey.simulation.lattice.superperiod` and
:func:`osprey.simulation.lattice.build_ring`. They are fully offline: no
network, no MATLAB, no soft-IOC. pyAT (``import at``) is the only physics
dependency.
"""

from collections import Counter

import at
import pytest

from osprey.simulation.lattice import build_ring, superperiod


def _type_census(elements):
    """Census of AT element class names over an iterable of elements."""
    return dict(Counter(type(e).__name__ for e in elements))


def _summed_length(elements):
    return sum(float(getattr(e, "Length", 0.0)) for e in elements)


def test_superperiod_census_and_length():
    """superperiod('01C') element count, type census, and summed length."""
    sup = superperiod("01C")

    assert len(sup) == 51

    census = _type_census(sup)
    assert census == {
        "Monitor": 6,
        "Quadrupole": 6,
        "Sextupole": 8,
        "Dipole": 3,
        "Drift": 28,
    }

    assert _summed_length(sup) == pytest.approx(10.37682924, abs=1e-6)


def test_assemble_full_ring_ground_truth():
    """build_ring() global metadata, type census, cavity, and markers."""
    ring = build_ring()

    assert isinstance(ring, at.Lattice)
    assert len(ring) == 802
    assert ring.circumference == pytest.approx(182.1219508800, abs=1e-6)
    assert ring.energy == 2.0e9
    assert ring.periodicity == 1
    assert ring.is_6d is True

    census = _type_census(ring)
    assert census == {
        "Marker": 13,
        "Drift": 368,
        "Monitor": 72,
        "Corrector": 144,
        "Sextupole": 96,
        "Quadrupole": 72,
        "Dipole": 36,
        "RFCavity": 1,
    }

    cavities = [e for e in ring if type(e).__name__ == "RFCavity"]
    assert len(cavities) == 1
    cavity = cavities[0]
    assert cavity.Frequency == pytest.approx(500416928.281479, rel=1e-6)
    assert cavity.HarmNumber == 304

    markers = {e.FamName for e in ring if type(e).__name__ == "Marker"}
    expected_markers = {f"SECT{i}" for i in range(1, 13)} | {"INJ"}
    assert markers == expected_markers


def test_physics_linear_stability_and_optics():
    """Linear stability (SC4), optics, tune, and circumference of the 4D ring."""
    r = build_ring().deepcopy()
    r.disable_6d()
    assert r.is_6d is False

    m44, _ = at.find_m44(r, dp=0.0)
    # Trace of each 2x2 transverse block within (-2, 2) -> stable betatron motion.
    assert abs(m44[0, 0] + m44[1, 1]) < 2
    assert abs(m44[2, 2] + m44[3, 3]) < 2

    # get_optics with chromaticity must complete without error. It returns a
    # 3-tuple; ringdata (index 1) carries 'tune' and 'chromaticity' fields.
    optics = at.get_optics(r, get_chrom=True, dp=1e-6)
    assert len(optics) == 3
    ringdata = optics[1]
    assert ringdata["tune"] is not None
    assert ringdata["chromaticity"] is not None

    tune = at.get_tune(r)
    assert len(tune) == 2
    assert all(v == v and abs(v) != float("inf") for v in tune)

    assert r.circumference == pytest.approx(182.12, abs=1e-2)
