"""The example facility's booster ring ``BR``, from MML's booster lattice.

The deck is MML's ``Booster/boosterlattice.m`` array ``BOOSTER``, element for
element and in MML's order, read by ``_mml.py``: four cells of six bending
magnets, four focusing and four defocusing quadrupoles (each written as two
half-length elements), four horizontal and four vertical correctors, eight
monitor markers, two ``SF`` and three ``SD`` sextupoles; three screen markers;
an aperture and the RF cavity at the start. Every length, angle, strength,
the cavity's voltage, frequency and harmonic number are MML's. A repeated
element is named by its MML token and its occurrence (``QF1`` to ``QF32``,
``BU1`` to ``BU24``), as the transfer-line decks name theirs.

Two things are stated differently from the MML file:

- The beam energy is the booster-to-storage line's (``_transfer.ENERGY``),
  the demo's 2.0 GeV that the booster's place text also states, where MML's
  file defaults to 1.9 GeV. The normalized strengths and angles, and so the
  optics, are MML's.
- MML writes the sextupoles with zero length and zero strength, so they act
  on nothing; the deck holds them as markers, and a sextupole MML gives a
  length or a strength is refused.

The bends are those of the MML file's active branch: sector bends with zero
edge angles. The booster-to-storage line's launch Twiss, also MML's, matches
the 7.5 degree edges of the branch that file disables; neither is altered.

The ring is 75 m round and closes (24 bends of 15 degrees). It is solved as a
periodic ring with its cavity on. The model is static: one energy point.
Nothing ramps, and the ramp gain and offset channels stay texture stand-ins.

The records follow MML's booster families (``Booster/boosterinit.m``):

- ``BEND``: 24 magnets on one supply. The demo's bend supply is device
  ``DIPOLE-RAMP 01``; its current setpoint and readback are shared by the 24
  ``DIPOLE`` devices and drive all 24 deck bends, a slice each.
- ``QF`` and ``QD``: 16 magnets each, one supply per family (``QF-RAMP 01``
  and ``QD-RAMP 01``), driving both halves of every magnet.
- ``HCM`` and ``VCM``: 16 each, powered one by one, MML range -6 A to 6 A,
  written as their limits.
- ``BPMx``/``BPMy``: 20 monitors, devices 2, 3, 4, 6 and 8 of each sector,
  which MML binds to lattice monitor ``8 * (sector - 1) + device``. The
  other 12 monitor markers of the lattice are no device and are not wired.

MML states no current-to-strength calibration for the booster (its
``booster2at`` is the identity), so the golden currents (``GOLDEN``) and the
corrector kick per ampere (``CORRECTOR_GAIN``) are the demo's own, chosen
inside MML's supply ranges. The calibration maps each golden current onto
the deck's MML strength, as the transfer lines' does:

- a quadrupole supply drives ``PolynomB[1]`` with gain ``K / I_golden``;
- the bend supply drives ``PolynomB[0]``, the field error on the design bend,
  with gain ``(angle / L) / I_golden`` and offset ``-angle / L``;
- a corrector drives ``KickAngle`` with ``CORRECTOR_GAIN`` rad per ampere;
- a monitor reads the orbit in millimetres (gain ``1e-3``).

Strengths scale with rigidity (``brho``); kicks and positions do not. A wired
channel carries no seed nominal.

MML's booster also has an ``RF`` family (a power control in DAC volts, no
frequency channel), ``SF`` and ``SD`` supplies (one per sector) and no tune
channel. None of them is a record here, so the measurement file lists the
orbit response alone.

The records the booster had before, eight bends and four and four
quadrupoles with a supply each, are removed with their seeds. The monitors,
the family groups and the two system groups keep the text they had.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import _deck
import _elements
import _mml
import _transfer
import _tree

from osprey_facility import source_tree

#: The model's name, the booster's section.
NAME = "BR"

#: The deck's beam energy in eV: the energy the booster hands to the
#: booster-to-storage line.
ENERGY = _transfer.ENERGY["BTS"]

#: The booster's name in prose.
TEXT = "Booster ring"

#: MML's sectors, and the monitors of each that are devices
#: (``AO.BPMx.DeviceList`` of ``boosterinit.m``).
SECTORS = 4
BPM_DEVICES = (2, 3, 4, 6, 8)
BPMS_PER_SECTOR = 8

#: The supply device of each family on one supply, and the MML token of the
#: deck elements it drives.
SUPPLIES: dict[str, tuple[str, str]] = {
    "DIPOLE": ("DIPOLE-RAMP", "BU"),
    "QF": ("QF-RAMP", "QF"),
    "QD": ("QD-RAMP", "QD"),
}

#: Each supply's golden current in A: the demo's own, inside MML's supply
#: ranges (bend 0 to 1000 A, quadrupoles 0 to 130 A).
GOLDEN: dict[str, float] = {"DIPOLE": 800.0, "QF": 110.0, "QD": 105.0}

#: A corrector's kick per ampere in rad at the deck energy: the demo's own,
#: the booster-to-storage line's.
CORRECTOR_GAIN = 1.0e-4

#: MML's corrector range in A (``AO.HCM.Setpoint.Range``).
CORRECTOR_RANGE = 6.0

#: The families every device of which this module emits, and each one's
#: device class, name in prose, group text and group names.
FAMILIES: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "DIPOLE": (
        "Dipole",
        "bending magnet",
        "Booster Dipole Magnets: Main bending magnets, ramped during acceleration cycle to "
        "increase beam energy.",
        ("dipole bending magnet", "bending magnet"),
    ),
    "QF": (
        "Quadrupole",
        "focusing quadrupole",
        "Booster Focusing Quadrupoles: Maintain beam focusing during energy ramp.",
        ("focusing quadrupole",),
    ),
    "QD": (
        "Quadrupole",
        "defocusing quadrupole",
        "Booster Defocusing Quadrupoles: Maintain beam focusing during energy ramp.",
        ("defocusing quadrupole",),
    ),
    "HCM": (
        "HCorrector",
        "horizontal corrector",
        "Booster Horizontal Correctors (HCM): the 16 steering magnets, four per sector, "
        "each on its own supply, that correct the horizontal orbit.",
        ("horizontal corrector",),
    ),
    "VCM": (
        "VCorrector",
        "vertical corrector",
        "Booster Vertical Correctors (VCM): the 16 steering magnets, four per sector, "
        "each on its own supply, that correct the vertical orbit.",
        ("vertical corrector",),
    ),
    "BPM": (
        "BeamPositionMonitor",
        "beam position monitor",
        "Booster BPMs: Measure beam position during acceleration. 20 BPMs for orbit measurement.",
        ("beam position monitor",),
    ),
}

#: The groups over several families: their text and member families; the
#: current monitor ``DCCT 01`` stays a member of the diagnostics.
SYSTEM_GROUPS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "BR/MAG": (
        "Booster Magnet System: Ramped magnets that track the beam energy during "
        "acceleration. Dipoles define the orbit, quadrupoles maintain focusing.",
        ("DIPOLE", "QD", "QF", "HCM", "VCM"),
        (),
    ),
    "BR/DIAG": (
        "Booster Diagnostics: Beam measurement during acceleration cycle.",
        ("BPM",),
        ("BR/DCCT01",),
    ),
}

#: What each supply current reads, for its description.
SUPPLY_TEXT: dict[str, str] = {
    "DIPOLE": "the one supply that feeds the 24 bending magnets in series",
    "QF": "the one supply that feeds the 16 focusing quadrupoles",
    "QD": "the one supply that feeds the 16 defocusing quadrupoles",
}

#: The channels of a corrector and of a monitor: role and address tail.
CORRECTOR_ROLES = (("sp", "CURRENT:SP"), ("rb", "CURRENT:RB"), ("on", "STATUS:ON"))
MONITOR_ROLES = (("bpm_x", "POSITION:X"), ("bpm_y", "POSITION:Y"), ("bpm_valid", "STATUS:VALID"))

#: What a monitor channel reads, for its description: the text the booster's
#: monitors had before the lattice was MML's.
MONITOR_TEXT: dict[str, str] = {
    "bpm_x": "position horizontal. Beam position during booster cycle (millimeters). "
    "Horizontal Position (read-only): Beam X position.",
    "bpm_y": "position vertical. Beam position during booster cycle (millimeters). "
    "Vertical Position (read-only): Beam Y position.",
    "bpm_valid": "status valid. BPM status. Valid Data (read-only): BPM reading is valid.",
}


# --- the deck ---------------------------------------------------------------------


def _element(element: _mml.Element, name: str) -> dict[str, Any]:
    definition = element.definition
    if element.type == "rfcavity":
        return _deck.cavity(
            name,
            voltage=definition.voltage,
            frequency=definition.frequency,
            harmonic=int(definition.harmonic),
            energy=ENERGY,
        )
    if element.type == "sextupole":
        if definition.length or definition.k:
            raise ValueError(f"{element.deck_element}: a sextupole with a length or a strength")
        return _deck.marker(name)
    return _transfer._element(element, name, NAME)


def elements(section: _mml.Section) -> list[dict[str, Any]]:
    """The booster's deck elements in MML's order."""
    line = [_element(element, name) for element, name in _transfer._named(section)]
    if line[0]["FamName"] != _transfer.START_MARKER:
        raise ValueError(f"{NAME}: the MML ring does not open with a START marker")
    return line


def deck(section: _mml.Section) -> dict[str, Any]:
    """The pyAT JSON document of the ring at the demo energy."""
    line = elements(section)
    (harmonic,) = [element["HarmNumber"] for element in line if element["Class"] == "RFCavity"]
    return _deck.document(NAME, ENERGY, line, harmonic_number=harmonic)


def deck_text(section: _mml.Section) -> str:
    """The deck file's exact text."""
    return _deck.text(deck(section))


def write_deck(tree: Path, root: Path | None = _mml.MML_ROOT) -> Path:
    """Write ``simulation/decks/BR.json`` under ``tree`` from the MML at ``root``."""
    path = tree / _transfer.deck_path(NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(deck_text(_mml.load(NAME, root)), encoding="utf-8")
    return path


# --- devices and their deck elements ----------------------------------------------


def _count(deck_document: dict[str, Any], token: str) -> int:
    names = {element["FamName"] for element in deck_document["elements"]}
    count = 0
    while f"{token}{count + 1}" in names:
        count += 1
    return count


def bound(deck_document: dict[str, Any]) -> dict[str, dict[str, tuple[str, ...]]]:
    """Each family's devices, by two-digit number, and the deck elements each one is.

    A bend is one element, a quadrupole its two consecutive halves, a
    corrector one element, and monitor ``n`` of the 20 the lattice monitor MML
    binds it to.
    """
    order = [element["FamName"] for element in deck_document["elements"]]
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    out["DIPOLE"] = {f"{n:02d}": (f"BU{n}",) for n in range(1, _count(deck_document, "BU") + 1)}
    for family in ("QF", "QD"):
        halves = _count(deck_document, family)
        if halves % 2:
            raise ValueError(f"{NAME}: {halves} {family} elements do not pair into magnets")
        out[family] = {}
        for n in range(1, halves // 2 + 1):
            first, second = f"{family}{2 * n - 1}", f"{family}{2 * n}"
            if order.index(second) != order.index(first) + 1:
                raise ValueError(f"{NAME}: {first} and {second} are not one magnet's halves")
            out[family][f"{n:02d}"] = (first, second)
    for family in ("HCM", "VCM"):
        out[family] = {
            f"{n:02d}": (f"{family}{n}",) for n in range(1, _count(deck_document, family) + 1)
        }
    monitors = [
        BPMS_PER_SECTOR * (sector - 1) + device
        for sector in range(1, SECTORS + 1)
        for device in BPM_DEVICES
    ]
    if _count(deck_document, "BPM") != BPMS_PER_SECTOR * SECTORS:
        raise ValueError(f"{NAME}: the deck does not hold {BPMS_PER_SECTOR * SECTORS} monitors")
    by_position = sorted(monitors, key=lambda n: order.index(f"BPM{n}"))
    out["BPM"] = {f"{i:02d}": (f"BPM{n}",) for i, n in enumerate(by_position, start=1)}
    return out


def _device_id(family: str, number: str) -> str:
    return f"{NAME}/{family}{number}"


def _mml_name(family: str, deck_name: str) -> str | None:
    """MML's common name of a monitor: ``BPM(sector,device)``."""
    if family != "BPM":
        return None
    index = int(deck_name.removeprefix("BPM")) - 1
    return f"BPM({index // BPMS_PER_SECTOR + 1},{index % BPMS_PER_SECTOR + 1})"


def devices(deck_document: dict[str, Any]) -> list[dict[str, Any]]:
    """The booster's lattice devices, sorted by id."""
    out = []
    for family, numbered in bound(deck_document).items():
        device_class, prose, _, _ = FAMILIES[family]
        for number, deck_names in numbered.items():
            names = [f"{family}{number}"]
            mml_name = _mml_name(family, deck_names[0])
            if mml_name is not None:
                names.append(mml_name)
            out.append(
                {
                    "id": _device_id(family, number),
                    "class": device_class,
                    "description": f"{TEXT} {prose} {number}.",
                    "names": names,
                    "label": f"{family} {number}",
                    "place": NAME,
                }
            )
    return sorted(out, key=lambda record: record["id"])


def _channel(
    address: str, role: str, device: str, description: str, pair: str | None = None
) -> dict[str, Any]:
    template = _elements.ROLE_TEMPLATES[role]
    record: dict[str, Any] = {"id": address}
    if template.writable:
        record["role"] = "setpoint"
        record["pair"] = pair
        if template.tolerance is not None:
            record["tolerance"] = {"absolute": template.tolerance}
    record["on"] = {"device": device}
    record["signal"] = template.signal
    if template.value_type != "float":
        record["value_type"] = template.value_type
    if template.unit is not None:
        record["unit"] = template.unit
    record["description"] = description
    return record


def supply_address(family: str, sub: str) -> str:
    """The current setpoint (``SP``) or readback (``RB``) of ``family``'s one supply."""
    return f"{NAME}:MAG:{SUPPLIES[family][0]}:01:CURRENT:{sub}"


def channels(deck_document: dict[str, Any]) -> list[dict[str, Any]]:
    """The booster's lattice channels, sorted by address.

    Each supply's current pair is on its supply device alone, so every address
    is bound once; which magnets it feeds is the model wiring's to say. A
    corrector and a monitor carry their own.
    """
    devices_of = bound(deck_document)
    out: list[dict[str, Any]] = []
    for family, (supply, _) in SUPPLIES.items():
        role = "dipole" if family == "DIPOLE" else "quad"
        for sub, text in (
            ("SP", "current setpoint (read-write): the commanded current of"),
            ("RB", "current readback (read-only): the measured current of"),
        ):
            out.append(
                _channel(
                    supply_address(family, sub),
                    f"{role}_{sub.lower()}",
                    _device_id(supply, "01"),
                    f"{TEXT} {FAMILIES[family][1]} supply {text} {SUPPLY_TEXT[family]}.",
                    pair=supply_address(family, "RB"),
                )
            )
    for family in ("HCM", "VCM"):
        for number in devices_of[family]:
            stem = f"{NAME}:MAG:{family}:{number}"
            for role, tail in CORRECTOR_ROLES:
                key = f"{family.lower()}_{role}"
                out.append(
                    _channel(
                        f"{stem}:{tail}",
                        key,
                        _device_id(family, number),
                        f"{TEXT} {FAMILIES[family][1]} {number} {_transfer.ROLE_TEXT[key]}",
                        pair=f"{stem}:CURRENT:RB",
                    )
                )
    for number in devices_of["BPM"]:
        for role, tail in MONITOR_ROLES:
            out.append(
                _channel(
                    f"{NAME}:DIAG:BPM:{number}:{tail}",
                    role,
                    _device_id("BPM", number),
                    f"{TEXT} {FAMILIES['BPM'][1]} {number} {MONITOR_TEXT[role]}",
                )
            )
    return sorted(out, key=lambda record: record["id"])


def valid_flags(deck_document: dict[str, Any]) -> list[str]:
    """Every monitor's ``STATUS:VALID`` address, in device order."""
    return [f"{NAME}:DIAG:BPM:{number}:STATUS:VALID" for number in bound(deck_document)["BPM"]]


def groups(deck_document: dict[str, Any]) -> list[dict[str, Any]]:
    """One group per lattice family and the two system groups, sorted by id."""
    devices_of = bound(deck_document)
    members = {
        family: sorted(_device_id(family, number) for number in numbered)
        for family, numbered in devices_of.items()
    }
    out = [
        {
            "id": f"{NAME}/{family}",
            "description": FAMILIES[family][2],
            "names": list(FAMILIES[family][3]),
            "members": members[family],
        }
        for family in FAMILIES
    ]
    for key, (description, families, further) in SYSTEM_GROUPS.items():
        listed = sorted([*further, *(device for family in families for device in members[family])])
        out.append({"id": key, "description": description, "members": listed})
    return sorted(out, key=lambda record: record["id"])


def seeds(deck_document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The seed nominals of the channels no model drives: every status, true."""
    return {
        record["id"]: {"nominal": "TRUE"}
        for record in channels(deck_document)
        if record.get("value_type") == "bool"
    }


def limits(deck_document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every corrector setpoint's range, MML's, sorted by address."""
    devices_of = bound(deck_document)
    return [
        {
            "address": f"{NAME}:MAG:{family}:{number}:CURRENT:SP",
            "min_value": -CORRECTOR_RANGE,
            "max_value": CORRECTOR_RANGE,
        }
        for family in ("HCM", "VCM")
        for number in devices_of[family]
    ]


# --- the model --------------------------------------------------------------------


def _supply_calibration(element: dict[str, Any], golden: float) -> dict[str, Any]:
    if element["Class"] == "Quadrupole":
        return _deck.linear_calibration(element["PolynomB"][1] / golden, 0.0, "brho")
    curvature = element["BendingAngle"] / element["Length"]
    return _deck.linear_calibration(curvature / golden, -curvature, "brho")


def wiring(deck_document: dict[str, Any]) -> list[dict[str, Any]]:
    """The model's wiring, sorted by address.

    Each supply's setpoint and readback drive every element of its family, a
    slice each naming its magnet; every corrector's and every device
    monitor's channels are wired to their element. Every bend, quadrupole and
    corrector of the deck is driven.
    """
    by_name = {element["FamName"]: element for element in deck_document["elements"]}
    devices_of = bound(deck_document)
    out: list[dict[str, Any]] = []
    driven: set[str] = set()
    for family in SUPPLIES:
        role = "dipole" if family == "DIPOLE" else "quad"
        _, engine = _transfer.ENGINE[role]
        slices = [
            {"element": name, "device": _device_id(family, number)}
            for number, deck_names in devices_of[family].items()
            for name in deck_names
        ]
        driven.update(piece["element"] for piece in slices)
        first = by_name[slices[0]["element"]]
        out.extend(
            {
                "address": supply_address(family, sub),
                "slices": [dict(piece) for piece in slices],
                "engine": dict(engine),
                "calibration": _supply_calibration(first, GOLDEN[family]),
            }
            for sub in ("RB", "SP")
        )
    for family in ("HCM", "VCM"):
        _, engine = _transfer.ENGINE[family.lower()]
        for number, (name,) in devices_of[family].items():
            driven.add(name)
            out.extend(
                {
                    "address": f"{NAME}:MAG:{family}:{number}:CURRENT:{sub}",
                    "element": name,
                    "engine": dict(engine),
                    "calibration": _deck.linear_calibration(CORRECTOR_GAIN, 0.0, "none"),
                }
                for sub in ("RB", "SP")
            )
    for number, (name,) in devices_of["BPM"].items():
        for axis in ("X", "Y"):
            _, engine = _transfer.ENGINE[f"bpm_{axis.lower()}"]
            out.append(
                {
                    "address": f"{NAME}:DIAG:BPM:{number}:POSITION:{axis}",
                    "element": name,
                    "engine": dict(engine),
                    "calibration": _deck.linear_calibration(_transfer.BPM_GAIN, 0.0, "none"),
                }
            )
    undriven = [
        element["FamName"]
        for element in deck_document["elements"]
        if element["Class"] in ("Corrector", "Quadrupole", "Bend")
        and element["FamName"] not in driven
    ]
    if undriven:
        raise ValueError(f"{NAME}: deck magnets not driven: {', '.join(undriven)}")
    return sorted(out, key=lambda record: record["address"])


def model(deck_document: dict[str, Any]) -> dict[str, Any]:
    """The booster's model: its deck solved as a periodic ring, and its wiring."""
    return {
        "name": NAME,
        "engine": "pyat",
        "deck": _transfer.deck_path(NAME),
        "settings": {"pyat": {"solve": "periodic"}},
        "wiring": wiring(deck_document),
    }


def measurement() -> dict[str, Any]:
    """The booster's measurement file: an orbit response over its monitors and correctors."""
    return _transfer.measurement(NAME)


def wired_addresses(deck_document: dict[str, Any]) -> set[str]:
    """Every booster address the model wires."""
    return {record["address"] for record in wiring(deck_document)}


def span() -> dict[str, str]:
    """Place ``BR``'s span: the whole deck, from its start marker."""
    return {"model": NAME, "from_marker": _transfer.START_MARKER}


# --- writing ----------------------------------------------------------------------


def owned(record_id: str) -> bool:
    """Whether a device, channel or group id is one this module emits or replaces.

    Every record of a lattice family, in the id forms ``BR/<FAMILY><NN>``,
    ``BR:<SYS>:<FAMILY>:<NN>:...`` and ``BR/<FAMILY>``; the two system groups;
    and each supply's current pair.
    """
    if record_id in SYSTEM_GROUPS:
        return True
    if record_id in {supply_address(family, sub) for family in SUPPLIES for sub in ("SP", "RB")}:
        return True
    if record_id.startswith(f"{NAME}/"):
        return record_id.removeprefix(f"{NAME}/").rstrip("0123456789") in FAMILIES
    parts = record_id.split(":")
    return len(parts) == 6 and parts[0] == NAME and parts[2] in FAMILIES


def _replace(path: Path, records: list[dict[str, Any]], *, ordered: bool) -> None:
    """Drop every owned record of the list at ``path`` and merge ``records`` in."""
    kept = [row for row in _tree.load(path) if not owned(str(row["id"]))]
    path.write_text(_tree.dump(kept), encoding="utf-8")
    _tree.merge(path, records, "id", ordered=ordered)


def _write_seeds(path: Path, deck_document: dict[str, Any]) -> None:
    import yaml

    seeds_dir = str(Path(__file__).resolve().parents[1] / "demo_seeds")
    if seeds_dir not in sys.path:
        sys.path.insert(0, seeds_dir)
    import reseed

    committed: dict[str, dict[str, Any]] = dict(reseed.load(path) or {})
    emitted = {record["id"] for record in channels(deck_document)}
    document = {
        address: entry
        for address, entry in committed.items()
        if not owned(address) or address in emitted
    }
    for address, entry in seeds(deck_document).items():
        document[address] = {**document.get(address, {}), **entry}
    ordered = {
        address: {
            key: document[address][key] for key in reseed.KEY_ORDER if key in document[address]
        }
        for address in sorted(document)
    }
    path.write_text(
        _tree.origin_line() + yaml.safe_dump(ordered, sort_keys=False), encoding="utf-8"
    )
    _transfer.strip_nominals(path, wired_addresses(deck_document))


def _limit_line(record: dict[str, Any]) -> str:
    return (
        f"- {{address: {record['address']}, min_value: {record['min_value']}, "
        f"max_value: {record['max_value']}}}"
    )


def _write_limits(path: Path, deck_document: dict[str, Any]) -> None:
    """Put the booster's corrector ranges, one line each, at the end of the limits file."""
    prefix = f"- {{address: {NAME}:"
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith(prefix)
    ]
    lines += [_limit_line(record) for record in limits(deck_document)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sources(tree: Path) -> None:
    """Write the booster's lattice records and model into the facility tree ``tree``.

    The records this module owns are replaced whole; the model goes into
    ``simulation/models.yaml``, the measurement file is written whole, place ``BR``
    gains its span, the corrector ranges go into ``limits.yaml`` and the
    seeds lose every removed channel and every wired nominal. Every other
    record keeps its text.
    """
    deck_document = _transfer.read_deck(tree, NAME)
    records = tree / "records"
    _replace(records / "devices.yaml", devices(deck_document), ordered=True)
    _replace(records / "channels.yaml", channels(deck_document), ordered=True)
    _replace(records / "groups.yaml", groups(deck_document), ordered=True)
    _tree.merge(tree / source_tree.path("models"), [model(deck_document)], "name", ordered=True)
    measured = tree / source_tree.keyed_file("measurement", NAME)
    measured.parent.mkdir(parents=True, exist_ok=True)
    measured.write_text(_tree.dump(measurement()), encoding="utf-8")
    places_path = records / "places.yaml"
    spanned = [
        {**record, "span": span()} for record in _tree.load(places_path) if record["id"] == NAME
    ]
    if not spanned:
        raise ValueError(f"{places_path}: no place {NAME}")
    _tree.merge(places_path, spanned, "id", ordered=False)
    _write_limits(tree / "limits.yaml", deck_document)
    _write_seeds(tree / source_tree.path("seeds"), deck_document)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: _booster.py <facility tree>")
    print(write_deck(Path(sys.argv[1])))
    write_sources(Path(sys.argv[1]))
