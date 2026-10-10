"""The example facility's transfer line ``LINE``, built from the constants below.

The deck is a single-pass FODO line: a start marker, ``CELLS`` cells, an end
marker. Each cell holds one BPM, one horizontal and one vertical corrector, a
focusing and a defocusing quadrupole, and a drift after each quadrupole. Every
quadrupole carries an elliptical aperture, so a large enough corrector kick
loses the particle before the end of the line. Every element name is unique in
the deck.

The deck is written in the pyAT JSON format: two-space indentation, one
trailing newline, elements in beam order.

The line's records are its place, spanning the deck from the start marker; a
device per BPM, corrector and quadrupole, each placed by that span; the BPMs'
position readbacks and every magnet's current setpoint and readback; the
groups its measurement file names; and its model, solved single-pass from the
start Twiss parameters. The line carries no seed and no limits record.
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

#: Twiss parameters at the start of the line, ``(x, y)``: the periodic
#: solution of one cell, so the optics repeat cell to cell.
TWISS_BETA = (7.448848, 4.786082)
TWISS_ALPHA = (-1.333207, 0.885765)

#: Kick angle in rad per A of corrector current.
CORRECTOR_GAIN = 1.0e-3

#: Integrated quadrupole strength in 1/m^2 per A of quadrupole current; the
#: defocusing quadrupoles carry its negative.
QUAD_GAIN = 1.0e-2

#: A horizontal-corrector kick, in the corrector's setpoint unit (A), that
#: loses the beam on the single-pass line.
LOSING_KICK = 5.0

#: How close a corrector's and a quadrupole's readback must come to its
#: setpoint, in A.
CORRECTOR_TOLERANCE = 0.02
QUAD_TOLERANCE = 0.05

#: Each device family: its class and its address subsystem.
_FAMILIES: dict[str, tuple[str, str]] = {
    "BPM": ("BeamPositionMonitor", "DIAG"),
    "HCM": ("HCorrector", "MAG"),
    "VCM": ("VCorrector", "MAG"),
    "QF": ("Quadrupole", "MAG"),
    "QD": ("Quadrupole", "MAG"),
}

#: The groups the measurement file names, each with its family and description.
_GROUPS: dict[str, tuple[str, str]] = {
    "bpm": ("BPM", "Transfer line BPMs: measure the beam trajectory along the line."),
    "hcor": ("HCM", "Transfer line horizontal correctors: steer the beam trajectory."),
    "vcor": ("VCM", "Transfer line vertical correctors: steer the beam trajectory."),
}


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


def _cells() -> list[str]:
    return [f"{cell:02d}" for cell in range(1, CELLS + 1)]


def _address(family: str, n: str, tail: str) -> str:
    return f"{MODEL}:{_FAMILIES[family][1]}:{family}:{n}:{tail}"


def place() -> dict[str, Any]:
    """The line's place: one machine spanning the deck from its start marker."""
    return {
        "id": MODEL,
        "level": "machine",
        "description": (
            f"Transfer line ({MODEL}): single-pass line of {CELLS} focusing-defocusing cells, "
            "each with one BPM, one horizontal and one vertical corrector and two quadrupoles."
        ),
        "span": {"model": MODEL, "from_marker": START_MARKER},
    }


def devices() -> list[dict[str, Any]]:
    """The line's devices, sorted by id."""
    return sorted(
        (
            {
                "id": f"{MODEL}/{family}{n}",
                "class": klass,
                "label": f"{MODEL} {family} {n}",
            }
            for family, (klass, _) in _FAMILIES.items()
            for n in _cells()
        ),
        key=lambda device: device["id"],
    )


def _setpoint_pair(family: str, n: str, tolerance: float) -> list[dict[str, Any]]:
    device = f"{MODEL}/{family}{n}"
    readback = _address(family, n, "CURRENT:RB")
    return [
        {"id": readback, "on": {"device": device}, "signal": "current_readback", "unit": "A"},
        {
            "id": _address(family, n, "CURRENT:SP"),
            "role": "setpoint",
            "pair": readback,
            "tolerance": {"absolute": tolerance},
            "on": {"device": device},
            "signal": "current_setpoint",
            "unit": "A",
        },
    ]


def channels() -> list[dict[str, Any]]:
    """The line's channels, sorted by address."""
    rows: list[dict[str, Any]] = []
    for n in _cells():
        for axis in ("X", "Y"):
            rows.append(
                {
                    "id": _address("BPM", n, f"POSITION:{axis}"),
                    "on": {"device": f"{MODEL}/BPM{n}"},
                    "signal": f"position_{axis.lower()}_readback",
                    "unit": "m",
                }
            )
        for family in ("HCM", "VCM"):
            rows += _setpoint_pair(family, n, CORRECTOR_TOLERANCE)
        for family in ("QF", "QD"):
            rows += _setpoint_pair(family, n, QUAD_TOLERANCE)
    return sorted(rows, key=lambda row: row["id"])


def groups() -> list[dict[str, Any]]:
    """The line's groups, the ones its measurement file names, sorted by id."""
    return sorted(
        (
            {
                "id": f"{MODEL}/{family}",
                "description": description,
                "members": [f"{MODEL}/{family}{n}" for n in _cells()],
            }
            for family, description in _GROUPS.values()
        ),
        key=lambda group: group["id"],
    )


def _calibration(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def _wiring() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in _cells():
        for axis in ("X", "Y"):
            rows.append(
                {
                    "address": _address("BPM", n, f"POSITION:{axis}"),
                    "element": f"BPM{n}",
                    "engine": {"axis": axis.lower()},
                    "calibration": _calibration(1.0),
                }
            )
        for family, engine, gain in (
            ("HCM", {"attribute": "KickAngle", "index": 0}, CORRECTOR_GAIN),
            ("VCM", {"attribute": "KickAngle", "index": 1}, CORRECTOR_GAIN),
            ("QF", {"attribute": "PolynomB", "index": 1}, QUAD_GAIN),
            ("QD", {"attribute": "PolynomB", "index": 1}, -QUAD_GAIN),
        ):
            for tail in ("CURRENT:RB", "CURRENT:SP"):
                rows.append(
                    {
                        "address": _address(family, n, tail),
                        "element": f"{family}{n}",
                        "engine": dict(engine),
                        "calibration": _calibration(gain),
                    }
                )
    return sorted(rows, key=lambda row: row["address"])


def model() -> dict[str, Any]:
    """The line's model: the deck solved single-pass from the start Twiss, and its wiring."""
    return {
        "name": MODEL,
        "engine": "pyat",
        "deck": f"decks/{MODEL}.json",
        "settings": {
            "pyat": {
                "solve": "single_pass",
                "twiss_in": {"beta": list(TWISS_BETA), "alpha": list(TWISS_ALPHA)},
            }
        },
        "wiring": _wiring(),
    }


def measurement() -> dict[str, Any]:
    """The line's measurement file: an orbit response over its BPMs and correctors."""
    return {
        "kinds": ["orm"],
        "groups": {role: f"{MODEL}/{family}" for role, (family, _) in _GROUPS.items()},
    }


def _dump(data: Any) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=float("inf"))


def _load(path: Path) -> list[dict[str, Any]]:
    import yaml

    if not path.is_file():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def _merge(path: Path, records: list[dict[str, Any]], key: str, *, ordered: bool) -> None:
    """Replace ``records`` in the YAML list at ``path`` by ``key``.

    A record already in the file keeps its position; a new one is appended.
    With ``ordered`` the list is sorted by ``key``.
    """
    new = {record[key]: record for record in records}
    merged = [new.pop(row[key], row) for row in _load(path)]
    merged += new.values()
    if ordered:
        merged.sort(key=lambda row: row[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(merged), encoding="utf-8")


def write_sources(tree: Path) -> None:
    """Merge the line into the facility tree ``tree``.

    The place, devices, channels and groups go into ``records/``, the model
    into ``models.yaml``, and the measurement file is written whole. Every
    other record keeps its text; ``limits.yaml`` and ``seeds.yaml`` are not
    touched.
    """
    records = tree / "records"
    _merge(records / "places.yaml", [place()], "id", ordered=False)
    _merge(records / "devices.yaml", devices(), "id", ordered=True)
    _merge(records / "channels.yaml", channels(), "id", ordered=True)
    _merge(records / "groups.yaml", groups(), "id", ordered=True)
    _merge(tree / "models.yaml", [model()], "name", ordered=True)
    path = tree / "measurement" / f"{MODEL}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(measurement()), encoding="utf-8")
