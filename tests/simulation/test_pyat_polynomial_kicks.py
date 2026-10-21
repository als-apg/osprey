"""The pyat engine's polynomial-kick copy of a deck.

A reader that takes a corrector's strength from ``PolynomB[0]`` and
``PolynomA[0]`` integrated over the element's length, or from the terms
themselves on a zero-length element (pyAML), reads nothing off an element that
carries its kick as ``KickAngle``. ``polynomial_kicks`` writes a copy of the
deck in which each named element carries the same integrated kick as its
polynomials: the deck's file is never written to, every element keeps the
deck's length and place, a zero-length corrector becomes a zero-length thin
multipole, and the orbit the copy's correctors move is the orbit the deck's
move.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.simulation.engines import pyat as engine  # noqa: E402

#: A kick small enough to stay linear, large enough to read.
KICK = 1.0e-5


def _cell() -> list[object]:
    """A stable bending cell: two thin correctors stacked after a BPM, then a drift."""
    return [
        at.Quadrupole("QF", 0.25, 1.2),
        at.Drift("D1", 0.5),
        at.Dipole("B1", 1.0, math.pi / 8),
        at.Drift("D2", 0.2),
        at.Quadrupole("QD", 0.5, -1.2),
        at.Monitor("BPM1"),
        at.Corrector("HC", 0.0, [0.0, 0.0]),
        at.Corrector("VC", 0.0, [0.0, 0.0]),
        at.Drift("D3", 0.2),
        at.Dipole("B2", 1.0, math.pi / 8),
        at.Drift("D4", 0.5),
        at.Monitor("BPM2"),
        at.Quadrupole("QF2", 0.25, 1.2),
    ]


def _deck(tmp_path: Path, elements: list[object], name: str = "deck.json") -> Path:
    path = tmp_path / name
    lattice = at.Lattice(elements, name="cell", energy=3.0e9, periodicity=8)
    lattice.save(str(path))
    return path


def _copy(text: str, tmp_path: Path) -> at.Lattice:
    path = tmp_path / "copy.json"
    path.write_text(text, encoding="utf-8")
    return at.load_lattice(str(path))


def _element(lattice: at.Lattice, name: str) -> at.Element:
    (found,) = [element for element in lattice if element.FamName == name]
    return found


def _orbit(lattice: at.Lattice) -> np.ndarray:
    monitors = [index for index, element in enumerate(lattice) if isinstance(element, at.Monitor)]
    _, orbit = at.find_orbit4(lattice, refpts=monitors)
    return orbit[:, [0, 2]]


def test_a_thin_corrector_becomes_a_thin_multipole_and_every_element_keeps_its_place(
    tmp_path: Path,
) -> None:
    deck = _deck(tmp_path, _cell())
    copied = _copy(engine.polynomial_kicks(deck, ["HC", "VC"]), tmp_path)
    original = at.load_lattice(str(deck))
    for name in ("HC", "VC"):
        element = _element(copied, name)
        assert element.PassMethod == "ThinMPolePass"
        assert element.Length == 0.0
    for kept, before in zip(copied, original, strict=True):
        assert kept.FamName == before.FamName
        assert kept.Length == before.Length
    assert copied.circumference == original.circumference


def test_the_copy_carries_the_deck_s_kicks_as_polynomials(tmp_path: Path) -> None:
    elements = _cell()
    elements[6].KickAngle = np.array([2.0e-5, 0.0])
    elements[7].KickAngle = np.array([0.0, -3.0e-5])
    copied = _copy(engine.polynomial_kicks(_deck(tmp_path, elements), ["HC", "VC"]), tmp_path)
    for name, kick in (("HC", (2.0e-5, 0.0)), ("VC", (0.0, -3.0e-5))):
        element = _element(copied, name)
        assert element.PolynomB[0] == -kick[0]
        assert element.PolynomA[0] == kick[1]
        assert not hasattr(element, "KickAngle")


def test_a_thick_corrector_keeps_its_length(tmp_path: Path) -> None:
    elements = _cell()
    elements[6] = at.Corrector("HC", 0.1, [4.0e-5, 1.0e-5], PolynomB=[0.0, 0.5])
    elements[8] = at.Drift("D3", 0.1)
    copied = _copy(engine.polynomial_kicks(_deck(tmp_path, elements), ["HC"]), tmp_path)
    element = _element(copied, "HC")
    assert element.Length == 0.1
    assert list(element.PolynomB) == pytest.approx([-4.0e-4])
    assert list(element.PolynomA) == pytest.approx([1.0e-4])
    assert _element(copied, "D3").Length == 0.1


def test_a_multipole_s_kick_is_folded_into_its_polynomials(tmp_path: Path) -> None:
    """A multipole pass applies ``sin(KickAngle)`` over the length to its dipole terms."""
    elements = _cell()
    sextupole = at.Sextupole("SX", 0.2, 3.0, KickAngle=np.array([1.0e-4, 2.0e-4]))
    elements.insert(9, sextupole)
    copied = _copy(engine.polynomial_kicks(_deck(tmp_path, elements), ["SX"]), tmp_path)
    element = _element(copied, "SX")
    assert element.PassMethod == sextupole.PassMethod
    assert element.PolynomB[0] == pytest.approx(-math.sin(1.0e-4) / 0.2)
    assert element.PolynomA[0] == pytest.approx(math.sin(2.0e-4) / 0.2)
    assert element.PolynomB[2] == 3.0
    assert list(element.KickAngle) == [0.0, 0.0]


def test_a_thin_corrector_with_no_drift_beside_it_is_converted_in_place(tmp_path: Path) -> None:
    elements = [element for element in _cell() if element.FamName != "D3"]
    deck = _deck(tmp_path, elements)
    copied = _copy(engine.polynomial_kicks(deck, ["HC", "VC"]), tmp_path)
    original = at.load_lattice(str(deck))
    for name in ("HC", "VC"):
        element = _element(copied, name)
        assert element.PassMethod == "ThinMPolePass"
        assert element.Length == 0.0
    for kept, before in zip(copied, original, strict=True):
        assert kept.FamName == before.FamName
        assert kept.Length == before.Length


def test_a_zero_length_multipole_s_kick_is_folded_into_its_dipole_terms(tmp_path: Path) -> None:
    """A thin multipole pass applies ``sin(KickAngle)`` as its integrated dipole terms."""
    kx, ky = 2.0e-5, -3.0e-5
    elements = _cell()
    elements.insert(8, at.ThinMultipole("TM", [0.0], [0.0], KickAngle=np.array([kx, ky])))
    deck = _deck(tmp_path, elements)
    copied = _copy(engine.polynomial_kicks(deck, ["TM"]), tmp_path)
    element = _element(copied, "TM")
    assert element.PassMethod == "ThinMPolePass"
    assert element.Length == 0.0
    assert element.PolynomB[0] == pytest.approx(-math.sin(kx), rel=1e-12)
    assert element.PolynomA[0] == pytest.approx(math.sin(ky), rel=1e-12)
    assert list(element.KickAngle) == [0.0, 0.0]
    assert _orbit(copied) == pytest.approx(_orbit(at.load_lattice(str(deck))), rel=1e-9)


def test_an_element_carrying_no_kick_is_left_alone(tmp_path: Path) -> None:
    deck = _deck(tmp_path, _cell())
    copied = engine.polynomial_kicks(deck, ["QF", "NOT_THERE"])
    assert json.loads(copied) == json.loads(engine.polynomial_kicks(deck, []))


def test_the_deck_s_file_is_not_written(tmp_path: Path) -> None:
    deck = _deck(tmp_path, _cell())
    before = deck.read_bytes()
    engine.polynomial_kicks(deck, ["HC", "VC"])
    assert deck.read_bytes() == before


def test_the_copy_is_the_same_text_every_time(tmp_path: Path) -> None:
    deck = _deck(tmp_path, _cell())
    first = engine.polynomial_kicks(deck, ["VC", "HC"])
    assert engine.polynomial_kicks(deck, ["HC", "VC"]) == first
    assert first.endswith("\n")
    assert "at_version" not in json.loads(first)


@pytest.mark.parametrize(("name", "plane"), [("HC", 0), ("VC", 1)])
def test_a_copied_corrector_moves_the_orbit_as_the_deck_s_kick_does(
    tmp_path: Path, name: str, plane: int
) -> None:
    deck = _deck(tmp_path, _cell())
    original = at.load_lattice(str(deck))
    copied = _copy(engine.polynomial_kicks(deck, ["HC", "VC"]), tmp_path)
    kicked = original.deepcopy()
    angle = np.zeros(2)
    angle[plane] = KICK
    _element(kicked, name).KickAngle = angle
    served = _orbit(kicked) - _orbit(original)
    base = _orbit(copied)
    element = _element(copied, name)
    if plane == 0:
        element.PolynomB[0] = -KICK
    else:
        element.PolynomA[0] = KICK
    measured = _orbit(copied) - base
    assert np.max(np.abs(served)) > 0
    assert measured == pytest.approx(served, rel=1e-9, abs=1e-15)
