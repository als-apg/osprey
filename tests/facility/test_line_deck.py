"""The example facility's transfer line deck ``decks/LINE.json``.

The committed deck is the generator's ``line`` output, loads in pyAT, holds the
stated element counts over about 12 m with unique names and an elliptical
aperture on every quadrupole, and carries a zero-kick particle to its end.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

at = pytest.importorskip("at")

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts/facility_demo/generate.py"
DECK = REPO_ROOT / "src/osprey/templates/facilities/example/decks/LINE.json"


@pytest.fixture(scope="module")
def lattice() -> object:
    """The committed deck loaded by pyAT."""
    return at.load_lattice(str(DECK))


def test_the_committed_deck_is_the_generators_output(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "line", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    assert (tmp_path / "decks/LINE.json").read_bytes() == DECK.read_bytes()


def test_the_deck_holds_the_stated_element_counts(lattice: object) -> None:
    counts = Counter(type(element).__name__ for element in lattice)
    assert counts["Quadrupole"] == 8
    assert counts["Corrector"] == 8
    assert counts["Monitor"] == 4
    assert counts["Marker"] == 2
    names = [element.FamName for element in lattice]
    assert sum(name.startswith("HCM") for name in names) == 4
    assert sum(name.startswith("VCM") for name in names) == 4
    assert names[0] == "START"
    assert names[-1] == "END"


def test_every_element_name_is_unique(lattice: object) -> None:
    names = [element.FamName for element in lattice]
    assert len(names) == len(set(names))


def test_the_line_is_about_twelve_metres(lattice: object) -> None:
    assert lattice.circumference == pytest.approx(12.0)


def test_every_quadrupole_carries_the_elliptical_aperture(lattice: object) -> None:
    quadrupoles = [element for element in lattice if isinstance(element, at.Quadrupole)]
    for quadrupole in quadrupoles:
        assert list(quadrupole.EApertures) == [0.02, 0.02]


def test_a_zero_kick_particle_reaches_the_end(lattice: object) -> None:
    r_in = np.zeros((6, 1))
    r_out, _, lost = at.lattice_track(
        lattice, r_in, nturns=1, refpts=len(lattice), losses=True, in_place=False
    )
    assert not lost["loss_map"]["islost"][0]
    assert np.all(np.isfinite(r_out))
