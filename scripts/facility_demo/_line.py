"""The example facility's transfer line ``LINE``, built from the constants below.

The deck is a single-pass FODO line: a start marker, ``CELLS`` cells, an end
marker. Each cell holds one BPM, one horizontal and one vertical corrector, a
focusing and a defocusing quadrupole, and a drift after each quadrupole. Every
quadrupole carries an elliptical aperture, so a large enough corrector kick
loses the particle before the end of the line. Every element name is unique in
the deck.

The deck is written in the pyAT JSON format: two-space indentation, one
trailing newline, elements in beam order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The model's name: the deck file is ``decks/<MODEL>.json``.
MODEL = "LINE"

#: Beam energy in eV.
ENERGY = 2.0e9

#: Number of FODO cells.
CELLS = 4

#: Quadrupole length in m.
QUAD_LENGTH = 0.3

#: Drift length after each quadrupole in m.
DRIFT_LENGTH = 1.2

#: Focusing quadrupole strength in 1/m^2; the defocusing one carries its negative.
QUAD_K = 1.2

#: Horizontal and vertical semi-axes of every quadrupole's elliptical aperture in m.
QUAD_APERTURE = (0.02, 0.02)

#: Integration steps of every quadrupole.
QUAD_STEPS = 10

#: Name of the marker at the start of the line.
START_MARKER = "START"

#: Name of the marker at the end of the line.
END_MARKER = "END"


def _marker(name: str) -> dict[str, Any]:
    return {"FamName": name, "Length": 0.0, "PassMethod": "IdentityPass", "Class": "Marker"}


def _monitor(name: str) -> dict[str, Any]:
    return {"FamName": name, "Length": 0.0, "PassMethod": "IdentityPass", "Class": "Monitor"}


def _corrector(name: str) -> dict[str, Any]:
    return {
        "FamName": name,
        "Length": 0.0,
        "PassMethod": "CorrectorPass",
        "KickAngle": [0.0, 0.0],
        "Class": "Corrector",
    }


def _quadrupole(name: str, k: float) -> dict[str, Any]:
    return {
        "FamName": name,
        "Length": QUAD_LENGTH,
        "PassMethod": "StrMPoleSymplectic4Pass",
        "MaxOrder": 1,
        "NumIntSteps": QUAD_STEPS,
        "PolynomA": [0.0, 0.0],
        "PolynomB": [0.0, k],
        "K": k,
        "EApertures": list(QUAD_APERTURE),
        "Class": "Quadrupole",
    }


def _drift(name: str) -> dict[str, Any]:
    return {"FamName": name, "Length": DRIFT_LENGTH, "PassMethod": "DriftPass", "Class": "Drift"}


def elements() -> list[dict[str, Any]]:
    """The line's elements in beam order."""
    line = [_marker(START_MARKER)]
    for cell in range(1, CELLS + 1):
        n = f"{cell:02d}"
        line += [
            _monitor(f"BPM{n}"),
            _corrector(f"HCM{n}"),
            _corrector(f"VCM{n}"),
            _quadrupole(f"QF{n}", QUAD_K),
            _drift(f"DF{n}"),
            _quadrupole(f"QD{n}", -QUAD_K),
            _drift(f"DD{n}"),
        ]
    line.append(_marker(END_MARKER))
    return line


def deck() -> dict[str, Any]:
    """The pyAT JSON document of the line."""
    return {
        "atjson": 1,
        "elements": elements(),
        "properties": {
            "name": MODEL,
            "energy": ENERGY,
            "particle": {"name": "relativistic", "rest_energy": 0.0, "charge": -1.0},
            "periodicity": 1,
        },
    }


def deck_text() -> str:
    """The deck file's exact text."""
    return json.dumps(deck(), indent=2) + "\n"


def write_deck(tree: Path) -> Path:
    """Write ``decks/LINE.json`` under the facility tree ``tree`` and return its path."""
    path = tree / "decks" / f"{MODEL}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(deck_text(), encoding="utf-8")
    return path
