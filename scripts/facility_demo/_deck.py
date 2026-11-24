"""Building a pyAT JSON deck, shared by the generators.

An element is a dict in the pyAT JSON form, its keys in the order the deck
files hold them. :func:`document` wraps a line of elements into the deck
document and :func:`text` gives the file's exact text: two-space indentation,
one trailing newline, elements in beam order.

:func:`fodo_cells` builds a line of focusing-defocusing cells from a few
numbers, for a deck stated by constants instead of read from MML.
"""

from __future__ import annotations

import json
from typing import Any

#: The particle of every deck.
PARTICLE: dict[str, Any] = {"name": "relativistic", "rest_energy": 0.0, "charge": -1.0}


def marker(name: str) -> dict[str, Any]:
    """A marker."""
    return {"FamName": name, "Length": 0.0, "PassMethod": "IdentityPass", "Class": "Marker"}


def monitor(name: str) -> dict[str, Any]:
    """A beam position monitor."""
    return {"FamName": name, "Length": 0.0, "PassMethod": "IdentityPass", "Class": "Monitor"}


def drift(name: str, length: float) -> dict[str, Any]:
    """A drift of ``length`` m."""
    return {"FamName": name, "Length": length, "PassMethod": "DriftPass", "Class": "Drift"}


def corrector(name: str, length: float = 0.0) -> dict[str, Any]:
    """A corrector of ``length`` m at zero kick."""
    return {
        "FamName": name,
        "Length": length,
        "PassMethod": "CorrectorPass",
        "KickAngle": [0.0, 0.0],
        "Class": "Corrector",
    }


def quadrupole(
    name: str,
    length: float,
    k: float,
    *,
    steps: int,
    aperture: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """A quadrupole of strength ``k`` (1/m^2), integrated in ``steps`` steps.

    ``aperture`` is the horizontal and vertical semi-axes, in m, of an
    elliptical aperture; without one the quadrupole stops nothing.
    """
    element: dict[str, Any] = {
        "FamName": name,
        "Length": length,
        "PassMethod": "StrMPoleSymplectic4Pass",
        "MaxOrder": 1,
        "NumIntSteps": steps,
        "PolynomA": [0.0, 0.0],
        "PolynomB": [0.0, k],
        "K": k,
    }
    if aperture is not None:
        element["EApertures"] = list(aperture)
    element["Class"] = "Quadrupole"
    return element


def bend(
    name: str,
    length: float,
    angle: float,
    *,
    entrance_angle: float,
    exit_angle: float,
    k: float = 0.0,
    steps: int,
) -> dict[str, Any]:
    """A bend of ``angle`` rad with its edge angles and gradient ``k`` (1/m^2)."""
    return {
        "FamName": name,
        "Length": length,
        "PassMethod": "BndMPoleSymplectic4Pass",
        "BendingAngle": angle,
        "EntranceAngle": entrance_angle,
        "ExitAngle": exit_angle,
        "MaxOrder": 1,
        "NumIntSteps": steps,
        "PolynomA": [0.0, 0.0],
        "PolynomB": [0.0, k],
        "K": k,
        "Class": "Bend",
    }


def fodo_cells(
    cells: int,
    *,
    quad_length: float,
    drift_length: float,
    k: float,
    steps: int,
    aperture: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """``cells`` focusing-defocusing cells in beam order, every element uniquely named.

    Cell ``nn`` holds a monitor ``BPMnn``, a horizontal and a vertical
    corrector ``HCMnn`` and ``VCMnn``, a focusing quadrupole ``QFnn`` of
    strength ``k``, a drift ``DFnn``, a defocusing quadrupole ``QDnn`` of
    strength ``-k`` and a drift ``DDnn``.
    """
    line: list[dict[str, Any]] = []
    for cell in range(1, cells + 1):
        n = f"{cell:02d}"
        line += [
            monitor(f"BPM{n}"),
            corrector(f"HCM{n}"),
            corrector(f"VCM{n}"),
            quadrupole(f"QF{n}", quad_length, k, steps=steps, aperture=aperture),
            drift(f"DF{n}", drift_length),
            quadrupole(f"QD{n}", quad_length, -k, steps=steps, aperture=aperture),
            drift(f"DD{n}", drift_length),
        ]
    return line


def document(
    name: str, energy: float, elements: list[dict[str, Any]], *, periodicity: int = 1
) -> dict[str, Any]:
    """The pyAT JSON document of the deck ``name`` at ``energy`` eV."""
    return {
        "atjson": 1,
        "elements": elements,
        "properties": {
            "name": name,
            "energy": energy,
            "particle": dict(PARTICLE),
            "periodicity": periodicity,
        },
    }


def text(deck: dict[str, Any]) -> str:
    """The deck file's exact text."""
    return json.dumps(deck, indent=2) + "\n"


def linear_calibration(gain: float, offset: float = 0.0, scaling: str = "none") -> dict[str, Any]:
    """A wiring record's calibration: a linear curve and its energy scaling."""
    if gain == 0.0:
        raise ValueError("a calibration gain of 0 has no inverse")
    return {"curve": {"linear": {"gain": gain, "offset": offset}}, "energy_scaling": scaling}
