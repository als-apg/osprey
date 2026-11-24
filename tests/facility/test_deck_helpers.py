"""The generators' deck helpers build a deck pyAT loads and tracks.

``scripts/facility_demo/_deck.py`` holds the element constructors the
generators share, the deck document and its file text, and
:func:`fodo_cells`, a line of focusing-defocusing cells stated by a few
numbers. A line built from them loads in pyAT with every element uniquely
named, carries a particle on its axis to the end, and loses a particle a
corrector kicks past the aperture its quadrupoles state.
"""

from __future__ import annotations

from pathlib import Path

import at
import numpy as np
import pytest

from tests.facility.test_transfer_wiring import _module

deck = _module("facility_demo_deck", "_deck.py")

CELLS = 4
APERTURE = (0.02, 0.02)


@pytest.fixture
def line(tmp_path: Path) -> at.Lattice:
    elements = deck.fodo_cells(
        CELLS, quad_length=0.3, drift_length=1.2, k=1.2, steps=10, aperture=APERTURE
    )
    path = tmp_path / "CELLS.json"
    path.write_text(deck.text(deck.document("CELLS", 1.0e9, elements)), encoding="utf-8")
    return at.load_lattice(str(path))


def _lost(line: at.Lattice) -> bool:
    out = at.lattice_track(line, np.zeros((6, 1)), losses=True)[2]
    return bool(out["loss_map"]["islost"][0])


def test_the_cells_load_with_every_element_uniquely_named(line: at.Lattice) -> None:
    names = [element.FamName for element in line]
    assert len(names) == len(set(names)) == 7 * CELLS
    assert [type(element).__name__ for element in line[:7]] == [
        "Monitor",
        "Corrector",
        "Corrector",
        "Quadrupole",
        "Drift",
        "Quadrupole",
        "Drift",
    ]
    assert line.energy == 1.0e9
    assert line.periodicity == 1


def test_every_quadrupole_alternates_its_strength_and_states_the_aperture(
    line: at.Lattice,
) -> None:
    quadrupoles = [element for element in line if isinstance(element, at.Quadrupole)]
    assert [element.K for element in quadrupoles] == [1.2, -1.2] * CELLS
    assert all(list(element.EApertures) == list(APERTURE) for element in quadrupoles)


def test_a_quadrupole_without_an_aperture_states_none() -> None:
    assert "EApertures" not in deck.quadrupole("Q", 0.3, 1.2, steps=10)


def test_a_particle_on_the_axis_reaches_the_end(line: at.Lattice) -> None:
    assert not _lost(line)


def test_a_kick_past_the_aperture_loses_the_particle(line: at.Lattice) -> None:
    line[line.get_uint32_index("HCM01")[0]].KickAngle = np.array([0.05, 0.0])
    assert _lost(line)


def test_a_calibration_refuses_a_zero_gain() -> None:
    assert deck.linear_calibration(2.0, 1.0) == {
        "curve": {"linear": {"gain": 2.0, "offset": 1.0}},
        "energy_scaling": "none",
    }
    with pytest.raises(ValueError, match="no inverse"):
        deck.linear_calibration(0.0)
