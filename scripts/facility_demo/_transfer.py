"""The example facility's injector transfer lines ``LTB`` and ``BTS``, built from MML.

Each deck is the MML lattice of its section, as :mod:`_mml` reads it, written in
the pyAT JSON format (two-space indentation, one trailing newline, elements in
beam order):

``LTB``
    linac to booster, ``gtblattice.m`` LTB1 followed by LTB2.
``BTS``
    booster to storage ring, ``btslattice.m`` BTS.

Every element keeps the MML optics verbatim: lengths, normalized quadrupole
strength ``K`` (1/m^2) and bend, entrance and exit angles (rad). Only the deck
energy is the demo's own -- 150 MeV for LTB and 2.0 GeV for BTS, where MML
states 50 MeV and 1.9 GeV. Normalized strengths are field over rigidity, so
keeping ``K`` and the angles keeps the optics; the magnet currents that produce
them scale with the energy ratio, which the models' calibration carries.

Element kinds map to pyAT classes: an MML ``corrector`` becomes a ``Corrector``
of the same length (``CorrectorPass``, zero kick); ``quadrupole`` a
``Quadrupole``; ``rbend``/``sbend`` a ``Bend`` with MML's edge angles as given
(MML's ``rbend`` sets them explicitly); a ``BPM`` marker a ``Monitor``; every
other marker, and BTS's ``AP`` aperture (+-10 m, which stops nothing), a
``Marker``; ``drift`` a ``Drift``. Each deck opens with a ``START`` and closes
with an ``END`` marker; BTS, whose MML array has no end marker, gets one after
its last element.

Every non-drift element has a unique name in its deck: the MML token when the
section holds it once, otherwise the token followed by its occurrence number,
so ``LTB:BPM#3`` is ``BPM3`` and ``LTB:Q11#1`` is ``Q11``. :func:`element_name`
maps an element table ``deck_element`` to that name. Drifts keep their MML
token, which may repeat.

The four septa -- BTS ``SEN``, ``SEK``, ``SIK`` and ``SIN`` -- are ordinary
bends here at their MML angles: fixed deck elements that no channel drives.

The launch Twiss parameters are MML's, from ``GTB/updateatindex.m`` (start of
LTB) and ``BTS/updateatindex.m`` (start of BTS). MML marks the LTB launch beta
``% ???``: a value nobody measured. The demo adopts it as stated, 12 m in both
planes with zero alpha and zero dispersion -- a round, waist-like beam after
the linac -- because it carries a nominal particle and propagates finite optics
through the whole line, and any replacement would be just as unmeasured.

The same module turns the element table's LTB and BTS rows into records, in
the table's beam order: the bends, quadrupoles and correctors with their
current setpoint, readback and on-status; the beam position monitors with
their sum signal and validity; the screens (paddle command, paddle position,
lamp); the integrating current transformer; and, in BTS, the storage ring's
first screen (SR01C TV1, ``BTS:DIAG:TV:07``) and the "all paddles out" count
``BTS:DIAG:PADDLES:01:COUNT:INSERTED``, an int whose nominal 0 means no screen
is in the beam. Both sit in BTS, where MML places the storage ring screen.

BTS predates the table: its correctors, focusing quadrupoles and monitors 01
to 06 already carry 72 frozen channels on 24 frozen devices, and the groups
``BTS/BPM``, ``BTS/HCM`` and ``BTS/VCM``. Those channels and devices are never
emitted, so they keep their text; the table's other rows on the same devices
(on-statuses, sum signals) are added beside them, and the three groups grow
by the new devices while keeping their descriptions. Places LTB and BTS are
machine places spanning their deck from its start marker; BTS keeps its
description and names. Each setpoint pairs with the readback of its field.

Each line is a model: its deck solved single-pass from the launch Twiss, with
a wiring record per table row the deck drives (``ElementRow.wired``) -- the
magnets' setpoints and readbacks and the monitors' positions; ``FIXED_ELEMENTS``
are held at their deck strength. The calibration maps the table's nominal,
the golden current ``I_golden``, onto the deck's own optics at the demo
energy, so the deck holds the golden machine:

- a quadrupole drives ``PolynomB[1]`` with gain ``K / I_golden``, offset 0;
- a bend drives ``PolynomB[0]``, the field error on its design bend, with gain
  ``(angle / L) / I_golden`` and offset ``-angle / L``, zero at golden;
- a corrector drives ``KickAngle`` with the line's gain per ampere and offset
  ``-gain * I_golden``, a zero kick at golden;
- a monitor reads the orbit in millimetres: its curve maps mm to m (gain
  ``1e-3``), so a reading is ``1e3`` times the solved orbit.

Quadrupole and bend strengths are physics at the deck energy and scale with
rigidity (``brho``); the kicks and positions do not. Every deck corrector,
quadrupole, monitor and bend is wired by a row or listed in
``FIXED_ELEMENTS``; an uncovered one stops generation, named. A wired channel
carries no seed nominal -- its operating point is the deck's -- so a setpoint,
an on-status and an unpaired readback carry the table's nominal only when the
deck does not drive them, and a paired readback follows its setpoint. Each
line has an orbit-response measurement file over its BPM, HCM and VCM groups.
No limits record is written.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import _elements
import _line
import _mml
import _standin

#: The transfer lines this module builds, in beam order.
SECTIONS = ("LTB", "BTS")

#: Each deck's beam energy in eV, the demo's own.
ENERGY: dict[str, float] = {"LTB": 150.0e6, "BTS": 2.0e9}

#: Name of the marker at the start of each deck.
START_MARKER = "START"

#: Name of the marker at the end of each deck.
END_MARKER = "END"

#: Integration steps of every quadrupole and bend.
STEPS = 10

#: Launch Twiss parameters at each deck's start marker, ``(x, y)`` per plane
#: and ``(eta_x, eta_px, eta_y, eta_py)`` for dispersion, from MML's
#: ``updateatindex.m``. LTB's beta is MML's ``% ???`` value, adopted as stated
#: (see the module docstring).
TWISS_IN: dict[str, dict[str, list[float]]] = {
    "LTB": {
        "beta": [12.0, 12.0],
        "alpha": [0.0, 0.0],
        "dispersion": [0.0, 0.0, 0.0, 0.0],
    },
    "BTS": {
        "beta": [0.63, 9.6],
        "alpha": [-0.32, 2.42],
        "dispersion": [0.074, 0.018, 0.0, 0.0],
    },
}


def twiss_in(section: str) -> dict[str, list[float]]:
    """The launch Twiss parameters of ``section``, as a model's ``pyat.twiss_in``."""
    return {key: list(values) for key, values in TWISS_IN[section].items()}


def _named(section: _mml.Section) -> list[tuple[_mml.Element, str]]:
    """Every element of ``section`` with its deck name."""
    counts = Counter(element.name for element in section.elements)
    named = []
    for element in section.elements:
        if element.type == "drift" or counts[element.name] == 1:
            name = element.name
        else:
            name = f"{element.name}{element.occurrence}"
        named.append((element, name))
    return named


def element_name(deck_element: str, section: _mml.Section) -> str:
    """The deck name of the element table key ``deck_element`` in ``section``."""
    for element, name in _named(section):
        if element.deck_element == deck_element:
            return name
    raise ValueError(f"{deck_element} is not an element of {section.name}")


def _marker(name: str) -> dict[str, Any]:
    return {"FamName": name, "Length": 0.0, "PassMethod": "IdentityPass", "Class": "Marker"}


def _monitor(name: str) -> dict[str, Any]:
    return {"FamName": name, "Length": 0.0, "PassMethod": "IdentityPass", "Class": "Monitor"}


def _drift(name: str, length: float) -> dict[str, Any]:
    return {"FamName": name, "Length": length, "PassMethod": "DriftPass", "Class": "Drift"}


def _corrector(name: str, length: float) -> dict[str, Any]:
    return {
        "FamName": name,
        "Length": length,
        "PassMethod": "CorrectorPass",
        "KickAngle": [0.0, 0.0],
        "Class": "Corrector",
    }


def _quadrupole(name: str, length: float, k: float) -> dict[str, Any]:
    return {
        "FamName": name,
        "Length": length,
        "PassMethod": "StrMPoleSymplectic4Pass",
        "MaxOrder": 1,
        "NumIntSteps": STEPS,
        "PolynomA": [0.0, 0.0],
        "PolynomB": [0.0, k],
        "K": k,
        "Class": "Quadrupole",
    }


def _bend(name: str, definition: _mml.Definition) -> dict[str, Any]:
    return {
        "FamName": name,
        "Length": definition.length,
        "PassMethod": "BndMPoleSymplectic4Pass",
        "BendingAngle": definition.angle,
        "EntranceAngle": definition.entrance_angle,
        "ExitAngle": definition.exit_angle,
        "MaxOrder": 1,
        "NumIntSteps": STEPS,
        "PolynomA": [0.0, 0.0],
        "PolynomB": [0.0, definition.k],
        "K": definition.k,
        "Class": "Bend",
    }


def _element(element: _mml.Element, name: str) -> dict[str, Any]:
    definition = element.definition
    if element.type == "drift":
        return _drift(name, definition.length)
    if element.type == "corrector":
        return _corrector(name, definition.length)
    if element.type == "quadrupole":
        return _quadrupole(name, definition.length, definition.k)
    if element.type in ("rbend", "sbend"):
        return _bend(name, definition)
    if element.kind == "monitor":
        return _monitor(name)
    if element.type in ("marker", "aperture"):
        return _marker(name)
    raise ValueError(f"{element.deck_element}: no deck element for MML type {element.type!r}")


def elements(section: _mml.Section) -> list[dict[str, Any]]:
    """``section``'s deck elements in beam order, between its start and end markers."""
    line = [_element(element, name) for element, name in _named(section)]
    if line[0]["FamName"] != START_MARKER or line[0]["Class"] != "Marker":
        raise ValueError(f"{section.name}: the MML line does not open with a START marker")
    if line[-1]["FamName"] != END_MARKER:
        line.append(_marker(END_MARKER))
    names = [item["FamName"] for item in line if item["Class"] != "Drift"]
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"{section.name}: element names repeat: {', '.join(repeated)}")
    return line


def deck(section: _mml.Section) -> dict[str, Any]:
    """The pyAT JSON document of ``section``'s line at the demo energy."""
    return {
        "atjson": 1,
        "elements": elements(section),
        "properties": {
            "name": section.name,
            "energy": ENERGY[section.name],
            "particle": {"name": "relativistic", "rest_energy": 0.0, "charge": -1.0},
            "periodicity": 1,
        },
    }


def deck_text(section: _mml.Section) -> str:
    """The deck file's exact text."""
    return json.dumps(deck(section), indent=2) + "\n"


def write_decks(tree: Path, root: Path | None = _mml.MML_ROOT) -> list[Path]:
    """Write ``decks/LTB.json`` and ``decks/BTS.json`` under ``tree`` from the MML at ``root``."""
    paths = []
    for name in SECTIONS:
        path = tree / "decks" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(deck_text(_mml.load(name, root)), encoding="utf-8")
        paths.append(path)
    return paths


# --- records ---------------------------------------------------------------------

#: Each section's name in prose, its place description, and its place names.
SECTION_TEXT: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "LTB": (
        "Linac-to-booster line",
        "Linac-to-Booster Line (LTB): carries the 150 MeV linac beam to booster injection, "
        "with bends, quadrupoles, correctors, beam position monitors and screens.",
        ("linac-to-booster line", "linac to booster"),
    ),
    "BTS": ("Booster-to-storage transfer line", "", ()),
}

#: Each family: its name in prose, its group description after the section
#: name, and its group names.
FAMILY_TEXT: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "HCM": (
        "horizontal corrector",
        "horizontal correctors: steering magnets.",
        ("horizontal corrector",),
    ),
    "VCM": (
        "vertical corrector",
        "vertical correctors: steering magnets.",
        ("vertical corrector",),
    ),
    "QF": (
        "focusing quadrupole",
        "focusing quadrupoles: horizontally focusing quadrupoles.",
        ("focusing quadrupole",),
    ),
    "QD": (
        "defocusing quadrupole",
        "defocusing quadrupoles: horizontally defocusing quadrupoles.",
        ("defocusing quadrupole",),
    ),
    "DIPOLE": (
        "bending magnet",
        "bending magnets: dipoles that steer the line onto its next section.",
        ("bending magnet", "dipole"),
    ),
    "BPM": (
        "beam position monitor",
        "beam position monitors: measure beam position and the sum signal.",
        ("beam position monitor",),
    ),
    "TV": (
        "screen",
        "screens: insertable paddles that image the beam and block it while in.",
        ("screen", "TV"),
    ),
    "PADDLES": (
        "paddle summary",
        "paddle summary: counts the screens in the beam; 0 means all paddles are out.",
        ("all paddles out", "paddle summary"),
    ),
    "ICT": (
        "integrating current transformer",
        "integrating current transformers: measure the transmitted charge per pulse.",
        ("integrating current transformer", "charge monitor"),
    ),
}

#: What each role's channel reads, for its description: the stand-ins' text,
#: plus the roles only the transfer lines carry.
ROLE_TEXT: dict[str, str] = {
    **_standin.ROLE_TEXT,
    "dipole_sp": "current setpoint (read-write): the commanded bending magnet current.",
    "dipole_rb": "current readback (read-only): the measured bending magnet current.",
    "dipole_on": "on status (read-only): the bending magnet supply is on.",
    "paddle_count": (
        "inserted screen count (read-only): how many screens are in the beam; "
        "0 means all paddles are out."
    ),
}

#: The BTS families whose devices 01 to 06 predate the element table.
FROZEN_FAMILIES = ("HCM", "VCM", "QF", "BPM")

#: The roles of the frozen BTS channels on those devices.
FROZEN_ROLES = frozenset(
    {
        "hcm_sp",
        "hcm_rb",
        "hcm_ready",
        "vcm_sp",
        "vcm_rb",
        "vcm_ready",
        "quad_sp",
        "quad_rb",
        "quad_ready",
        "bpm_x",
        "bpm_y",
        "bpm_valid",
    }
)

#: The BTS groups that predate the table: their description and names stay,
#: their members grow.
FROZEN_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "BTS/BPM": (
        "Transfer Line BPMs: Monitor beam trajectory during injection.",
        ("beam position monitor",),
    ),
    "BTS/HCM": (
        "Horizontal Correctors: Horizontal steering magnets in the transfer line.",
        ("horizontal corrector",),
    ),
    "BTS/VCM": (
        "Vertical Correctors: Vertical steering magnets in the transfer line.",
        ("vertical corrector",),
    ),
}


def _family(row: _elements.ElementRow) -> str:
    return row.demo_name.split(":")[2]


def _number(row: _elements.ElementRow) -> str:
    return row.demo_name.split(":")[3]


def _frozen_device(row: _elements.ElementRow) -> bool:
    """Whether the row's device is one of the 24 that predate the table."""
    return row.section == "BTS" and _family(row) in FROZEN_FAMILIES and int(_number(row)) <= 6


def frozen(row: _elements.ElementRow) -> bool:
    """Whether the row is one of the 72 frozen BTS channels, which are never emitted."""
    return _frozen_device(row) and row.role in FROZEN_ROLES


def table_rows() -> list[_elements.ElementRow]:
    """Every LTB and BTS row of the table, one per address, in beam order, frozen ones too.

    A channel bound by several legacy names has several rows; the first one
    stands for the channel.
    """
    seen: dict[str, _elements.ElementRow] = {}
    for row in _elements.load_elements():
        if row.section in SECTIONS:
            seen.setdefault(row.demo_name, row)
    return list(seen.values())


def rows() -> list[_elements.ElementRow]:
    """The LTB and BTS rows this module emits: every table row but the frozen ones."""
    return [row for row in table_rows() if not frozen(row)]


def _pairs() -> dict[str, str]:
    """Each emitted setpoint's readback: the ``RB`` address of the same field."""
    addresses = {row.demo_name for row in table_rows()}
    out: dict[str, str] = {}
    for row in rows():
        if row.template.writable:
            readback = f"{row.demo_name.rsplit(':', 1)[0]}:RB"
            if readback not in addresses:
                raise ValueError(f"{row.demo_name}: no readback {readback} in the table")
            out[row.demo_name] = readback
    return out


def devices() -> list[dict[str, Any]]:
    """The LTB and BTS devices the table adds, sorted by id; the frozen ones are kept as they are."""
    out: dict[str, dict[str, Any]] = {}
    for row in table_rows():
        key = _standin.device_id(row)
        device_class = row.template.device_class
        if key in out:
            if out[key]["class"] != device_class:
                raise ValueError(f"{key}: rows of classes {out[key]['class']} and {device_class}")
            continue
        family, number = _family(row), _number(row)
        out[key] = {
            "id": key,
            "class": device_class,
            "description": f"{SECTION_TEXT[row.section][0]} {FAMILY_TEXT[family][0]} {number}.",
            "names": [f"{family}{number}"],
            "label": f"{family} {number}",
            "place": row.section,
        }
    frozen_ids = {_standin.device_id(row) for row in table_rows() if _frozen_device(row)}
    return [out[key] for key in sorted(out) if key not in frozen_ids]


def channels() -> list[dict[str, Any]]:
    """The LTB and BTS channels the table adds, sorted by address."""
    pairs = _pairs()
    out: list[dict[str, Any]] = []
    for row in rows():
        template = row.template
        record: dict[str, Any] = {"id": row.demo_name}
        if template.writable:
            record["role"] = "setpoint"
            record["pair"] = pairs[row.demo_name]
            if template.tolerance is not None:
                record["tolerance"] = {"absolute": template.tolerance}
        record["on"] = {"device": _standin.device_id(row)}
        record["signal"] = template.signal
        if template.value_type != "float":
            record["value_type"] = template.value_type
        if template.unit is not None:
            record["unit"] = template.unit
        section_name = SECTION_TEXT[row.section][0]
        family_name = FAMILY_TEXT[_family(row)][0]
        record["description"] = f"{section_name} {family_name} {_number(row)} {ROLE_TEXT[row.role]}"
        out.append(record)
    return sorted(out, key=lambda record: record["id"])


def groups() -> list[dict[str, Any]]:
    """One group per LTB and BTS family, sorted by id, frozen devices among the members."""
    members: dict[str, list[str]] = {}
    for row in table_rows():
        key = f"{row.section}/{_family(row)}"
        device = _standin.device_id(row)
        if device not in members.setdefault(key, []):
            members[key].append(device)
    out = []
    for key in sorted(members):
        section, family = key.split("/")
        if key in FROZEN_GROUPS:
            description, names = FROZEN_GROUPS[key]
        else:
            _, text, names = FAMILY_TEXT[family]
            description = f"{SECTION_TEXT[section][0]} {text}"
        out.append(
            {
                "id": key,
                "description": description,
                "names": list(names),
                "members": sorted(members[key]),
            }
        )
    return out


def _span(section: str) -> dict[str, str]:
    return {"model": section, "from_marker": START_MARKER}


def places() -> list[dict[str, Any]]:
    """Place LTB, a machine place spanning the LTB deck; BTS is already a place."""
    _, description, names = SECTION_TEXT["LTB"]
    return [
        {
            "id": "LTB",
            "level": "machine",
            "description": description,
            "names": list(names),
            "span": _span("LTB"),
        }
    ]


def seeds() -> dict[str, dict[str, Any]]:
    """The emitted LTB and BTS seed nominals by address.

    A setpoint, an on-status and an unpaired readback the deck does not drive
    carry the table's nominal; a wired channel and a paired readback carry
    none. The frozen BTS channels keep the seeds they have, but a nominal.
    """
    paired = set(_pairs().values())
    out: dict[str, dict[str, Any]] = {}
    for row in rows():
        value = row.nominal_value()
        if value is not None and not row.wired and row.demo_name not in paired:
            out[row.demo_name] = {"nominal": value}
    return out


# --- models ----------------------------------------------------------------------

#: Each line's corrector kick per ampere, in radians at the deck energy.
CORRECTOR_GAIN: dict[str, float] = {"LTB": 1.0e-3, "BTS": 1.0e-4}

#: A monitor's curve: millimetres to the solved orbit's metres.
BPM_GAIN = 1.0e-3

#: The corrector step of each line's orbit-response measurement.
CORRECTOR_DELTA = 1.0e-5

#: The deck classes every one of which a row wires or ``FIXED_ELEMENTS`` holds.
COVERED_CLASSES = ("Corrector", "Quadrupole", "Monitor", "Bend")

#: Each wired role's deck class and engine block.
ENGINE: dict[str, tuple[str, dict[str, Any]]] = {
    "hcm": ("Corrector", {"attribute": "KickAngle", "index": 0}),
    "vcm": ("Corrector", {"attribute": "KickAngle", "index": 1}),
    "quad": ("Quadrupole", {"attribute": "PolynomB", "index": 1}),
    "dipole": ("Bend", {"attribute": "PolynomB", "index": 0}),
    "bpm_x": ("Monitor", {"axis": "x"}),
    "bpm_y": ("Monitor", {"axis": "y"}),
}


def _engine_key(row: _elements.ElementRow) -> str:
    return row.role if row.role.startswith("bpm_") else row.role.split("_", 1)[0]


def read_deck(tree: Path, section: str) -> dict[str, Any]:
    """The committed deck ``decks/<section>.json`` under ``tree``."""
    return json.loads((tree / "decks" / f"{section}.json").read_text(encoding="utf-8"))


def deck_element_name(deck_element: str, names: set[str]) -> str:
    """The deck name of the element table key ``deck_element`` among the deck's ``names``.

    The deck names an element by its MML token when the section holds it once,
    otherwise by the token and its occurrence (see :func:`element_name`).
    """
    match = _elements.DECK_ELEMENT.match(deck_element)
    if match is None:
        raise ValueError(f"{deck_element} is not a deck element")
    token, occurrence = match["family"], int(match["n"])
    if occurrence == 1 and token in names:
        return token
    if f"{token}{occurrence}" in names:
        return f"{token}{occurrence}"
    raise ValueError(f"{deck_element} is not an element of the deck")


def _golden(rows: list[_elements.ElementRow]) -> dict[str, float]:
    """Each wired deck element's golden current: its setpoint row's nominal."""
    out: dict[str, float] = {}
    for row in rows:
        if row.wired and row.template.writable:
            value = row.nominal_value()
            if not isinstance(value, float):
                raise ValueError(f"{row.demo_name}: a wired setpoint needs a numeric nominal")
            out[row.deck_element] = value
    return out


def _calibration(gain: float, offset: float, scaling: str) -> dict[str, Any]:
    if gain == 0.0:
        raise ValueError("a calibration gain of 0 has no inverse")
    return {"curve": {"linear": {"gain": gain, "offset": offset}}, "energy_scaling": scaling}


def _row_calibration(
    row: _elements.ElementRow, element: dict[str, Any], golden: float | None
) -> dict[str, Any]:
    key = _engine_key(row)
    if key.startswith("bpm_"):
        return _calibration(BPM_GAIN, 0.0, "none")
    if golden is None or (golden == 0.0 and key in ("quad", "dipole")):
        raise ValueError(f"{row.demo_name}: no nonzero golden current for {row.deck_element}")
    if key in ("hcm", "vcm"):
        gain = CORRECTOR_GAIN[row.section]
        return _calibration(gain, -gain * golden if golden else 0.0, "none")
    if key == "quad":
        return _calibration(element["PolynomB"][1] / golden, 0.0, "brho")
    curvature = element["BendingAngle"] / element["Length"]
    return _calibration(curvature / golden, -curvature, "brho")


def wiring(
    section: str, deck_document: dict[str, Any], table: list[_elements.ElementRow] | None = None
) -> list[dict[str, Any]]:
    """The wiring of ``section``'s model, sorted by address.

    ``table`` defaults to the section's table rows. Every deck corrector,
    quadrupole, monitor and bend must be wired by a row or listed in
    ``FIXED_ELEMENTS``; an uncovered one raises, named.
    """
    if table is None:
        table = table_rows()
    table = [row for row in table if row.section == section]
    by_name = {element["FamName"]: element for element in deck_document["elements"]}
    names = set(by_name)
    golden = _golden(table)
    covered = {
        deck_element_name(key, names)
        for key in _elements.FIXED_ELEMENTS
        if key.startswith(f"{section}:")
    }
    out: list[dict[str, Any]] = []
    for row in table:
        if not row.wired:
            continue
        name = deck_element_name(row.deck_element, names)
        element = by_name[name]
        deck_class, engine = ENGINE[_engine_key(row)]
        if element["Class"] != deck_class:
            raise ValueError(
                f"{row.demo_name}: {row.deck_element} is a {element['Class']}, not a {deck_class}"
            )
        covered.add(name)
        out.append(
            {
                "address": row.demo_name,
                "element": name,
                "engine": dict(engine),
                "calibration": _row_calibration(row, element, golden.get(row.deck_element)),
            }
        )
    uncovered = [
        element["FamName"]
        for element in deck_document["elements"]
        if element["Class"] in COVERED_CLASSES and element["FamName"] not in covered
    ]
    if uncovered:
        raise ValueError(
            f"{section}: deck elements neither wired nor fixed: {', '.join(uncovered)}"
        )
    return sorted(out, key=lambda record: record["address"])


def model(section: str, deck_document: dict[str, Any]) -> dict[str, Any]:
    """``section``'s model: its deck solved single-pass from the launch Twiss, and its wiring."""
    return {
        "name": section,
        "engine": "pyat",
        "deck": f"decks/{section}.json",
        "settings": {"pyat": {"solve": "single_pass", "twiss_in": twiss_in(section)}},
        "wiring": wiring(section, deck_document),
    }


def measurement(section: str) -> dict[str, Any]:
    """``section``'s measurement file: an orbit response over its BPMs and correctors."""
    return {
        "kinds": ["orm"],
        "groups": {"bpm": f"{section}/BPM", "hcor": f"{section}/HCM", "vcor": f"{section}/VCM"},
        "corrector_delta": CORRECTOR_DELTA,
    }


def wired_addresses() -> set[str]:
    """Every LTB and BTS address a model wires."""
    return {row.demo_name for row in table_rows() if row.wired}


def _strip_nominals(path: Path, addresses: set[str]) -> None:
    """Remove ``nominal`` from the seeds of ``addresses``; drop an entry left empty."""
    import yaml

    seeds_dir = str(Path(__file__).resolve().parents[1] / "demo_seeds")
    if seeds_dir not in sys.path:
        sys.path.insert(0, seeds_dir)
    import reseed

    committed: dict[str, dict[str, Any]] = dict(reseed.load(path) or {})
    document: dict[str, dict[str, Any]] = {}
    for address in sorted(committed):
        entry = dict(committed[address])
        if address in addresses:
            entry.pop("nominal", None)
        if entry:
            document[address] = {key: entry[key] for key in reseed.KEY_ORDER if key in entry}
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def write_sources(tree: Path) -> None:
    """Merge the LTB and BTS models into the facility tree ``tree``.

    The models go into ``models.yaml``, each measurement file is written
    whole, places LTB and BTS gain their span, and the wired channels' seed
    nominals are removed. Every other record keeps its text.
    """
    decks = {section: read_deck(tree, section) for section in SECTIONS}
    _line._merge(
        tree / "models.yaml",
        [model(section, decks[section]) for section in SECTIONS],
        "name",
        ordered=True,
    )
    for section in SECTIONS:
        path = tree / "measurement" / f"{section}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_line._dump(measurement(section)), encoding="utf-8")
    places_path = tree / "records" / "places.yaml"
    spanned = []
    for record in _line._load(places_path):
        if record["id"] in SECTIONS:
            spanned.append({**record, "span": _span(record["id"])})
    _line._merge(places_path, spanned, "id", ordered=False)
    _strip_nominals(tree / "seeds.yaml", wired_addresses())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: _transfer.py <facility tree>")
    for written in write_decks(Path(sys.argv[1])):
        print(written)
    write_sources(Path(sys.argv[1]))
