"""Ground-truth locks for the example facility's committed SR deck.

These tests assert measured values against the deck as it is committed. They
are fully offline: no network, no MATLAB, no soft-IOC. pyAT (``import at``) is
the only physics dependency.
"""

from collections import Counter

import at
import pytest

from tests.simulation._sr_deck import load_sr_deck, load_sr_deck_4d


def _type_census(elements):
    """Census of AT element class names over an iterable of elements."""
    return dict(Counter(type(e).__name__ for e in elements))


def test_deck_ground_truth():
    """The deck's global metadata, type census, cavity, and markers."""
    deck = load_sr_deck()

    assert isinstance(deck, at.Lattice)
    assert len(deck) == 802
    assert deck.circumference == pytest.approx(182.1219508800, abs=1e-6)
    assert deck.energy == 2.0e9
    assert deck.periodicity == 1
    assert deck.is_6d is True

    census = _type_census(deck)
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

    cavities = [e for e in deck if type(e).__name__ == "RFCavity"]
    assert len(cavities) == 1
    cavity = cavities[0]
    assert cavity.Frequency == pytest.approx(500416928.281479, rel=1e-6)
    assert cavity.HarmNumber == 304

    markers = {e.FamName for e in deck if type(e).__name__ == "Marker"}
    expected_markers = {f"SECT{i}" for i in range(1, 13)} | {"INJ"}
    assert markers == expected_markers


def test_4d_deck_is_linearly_stable():
    """Linear stability of the 4D deck.

    Tunes and chromaticity are checked in ``test_fidelity.py``, and the
    circumference to 1e-6 in ``test_deck_ground_truth``.
    """
    r = load_sr_deck_4d()
    assert r.is_6d is False

    m44, _ = at.find_m44(r, dp=0.0)
    # Trace of each 2x2 transverse block within (-2, 2) -> stable betatron motion.
    assert abs(m44[0, 0] + m44[1, 1]) < 2
    assert abs(m44[2, 2] + m44[3, 3]) < 2
