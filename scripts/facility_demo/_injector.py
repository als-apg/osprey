"""The example facility's injector records, built from the element table.

``_elements.load_elements()`` reads ``injector_elements.tsv``; this module turns
the rows of the sections it emits into records and merges them into a facility
tree, the way ``_line.py`` merges the transfer line: a record already in a file
keeps its position, a new one is added, every other record keeps its text.

It emits the storage ring's two injection septa, SEK (``SR:MAG:SEPTUM:01``) and
SEN (``SR:MAG:SEPTUM:02``):

- a device of class ``Septum`` per septum, in the injection sector;
- each septum's current setpoint, readback and on-status, and on SEK the
  sampled current pulse, a 500-point waveform;
- the group ``SR/SEPTUM`` holding both;
- a seed per setpoint and on-status carrying the table's nominal, and on the
  SEK waveform a ``pulse`` driven by the SEK setpoint and enabled by its
  on-status.

The septa are fixed deck elements: no model wires them, the texture serves
every one of their channels from its seed. The readbacks' noise and the
setpoints' tolerances are the rule table's (``scripts/demo_seeds/rules.yaml``);
the tolerance written here is the role's, which the rule table restates.

It emits the booster's panel rows the same way, as texture rows of place
``BR``: the injection kicker (class ``Kicker``), the extraction septa SEN
(``BR:MAG:SEPTUM:01``) and SEK (``BR:MAG:SEPTUM:02``), the five booster
screens, and the gain/offset stand-in setpoints of the bending magnet and
quadrupole energy ramps. Each family gets a group; each setpoint pairs with
its readback; every setpoint, on-status and unpaired readback carries the
table's nominal. The booster septa sit on the BTS deck as fixed elements, so
no model wires them and none carries a pulse. A channel bound by several
legacy names (the septum supplies' interlock bits) is one channel. The
booster's existing records -- its bends, quadrupoles, monitors, current
monitor and system groups -- are not emitted and keep their text.

The gun-to-linac line and the linac are texture stand-ins built by
``_standin.py``, and the linac-to-booster and booster-to-storage lines are
built by ``_transfer.py``; ``write_sources`` merges their places, devices,
channels, groups and seeds beside the septa's and the booster's. The machine
places GTL, LN, LTB and BTS and every family group the table adds carry the
hand-authored descriptions and names of ``PLACE_TEXT`` and ``GROUP_TEXT``;
the BTS groups that predate the table keep their text.

It owns the injector's one gate chain, GTL -> LN -> LTB -> BR -> BTS -> SR
injection, in the table's beam order, and writes it as explicit ``gate`` seeds:

- every BPM sum and every transmission/charge proxy (ICT charge) reads a
  hundredth of its value, and every BPM ``STATUS:VALID`` reads ``FALSE``, while
  the gun's high voltage or heater on-status readback is ``FALSE`` or the
  paddle readback of any screen upstream of it in beam order is ``TRUE``;
- each screen's lamp readback reads ``TRUE`` while its own paddle is in;
- the "all paddles out" count reads 1 while any paddle is in.

A BPM's positions carry no gate, and no storage ring channel does: the SR01C
screen and the paddle count sit in BTS. This module writes every ``gate`` key
of ``seeds.yaml``; the runtime reads the inputs as written and does no ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import _elements
import _line
import _standin
import _transfer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_seeds"))

import reseed

#: The sections whose rows this module emits.
SECTIONS = ("SR",)

#: The device classes this module emits within those sections.
CLASSES = ("Septum",)

#: Each septum's two-digit number, its name, and what it does.
SEPTA: dict[str, tuple[str, str]] = {
    "01": (
        "SEK",
        "pulsed injection septum; its current pulse is sampled as a waveform",
    ),
    "02": ("SEN", "the second injection septum; its current is read as a scalar"),
}

#: The place every septum sits in: the injection straight of sector 1.
PLACE = "SR/SECT1"

#: The group of the septa.
GROUP = "SR/SEPTUM"

#: What each septum channel's field and subfield read, for its description.
_FIELD_TEXT: dict[str, str] = {
    "CURRENT:SP": (
        "current setpoint (read-write): the commanded peak current of the septum pulse."
    ),
    "CURRENT:RB": "current readback (read-only): the measured peak current of the septum pulse.",
    "STATUS:ON": "on status (read-only): the septum supply is energized and pulsing.",
    "CURRENT:WAVEFORM": (
        "current pulse waveform (read-only): the sampled time trace of one current pulse, "
        "500 samples in A, refreshed about once a second. Its peak follows the current "
        "setpoint; it is flat zero while the septum is off."
    ),
}

#: What a channel adds to its description on a septum whose pulse is sampled.
_PULSE_TEXT: dict[str, str] = {
    "CURRENT:SP": " Scales the sampled current pulse.",
    "STATUS:ON": " When false the sampled current pulse is a flat zero trace.",
}


def _number(row: _elements.ElementRow) -> str:
    return row.demo_name.split(":")[3]


def _tail(row: _elements.ElementRow) -> str:
    return ":".join(row.demo_name.split(":")[4:])


def _device_id(row: _elements.ElementRow) -> str:
    return f"{row.section}/SEPTUM{_number(row)}"


def rows() -> list[_elements.ElementRow]:
    """The table's emitted rows, one per address, in table order.

    A channel bound by several legacy names has several rows; the first one
    stands for the channel.
    """
    seen: dict[str, _elements.ElementRow] = {}
    for row in _elements.load_elements():
        if row.section in SECTIONS and row.template.device_class in CLASSES:
            seen.setdefault(row.demo_name, row)
    return list(seen.values())


def devices() -> list[dict[str, Any]]:
    """The septa's devices, sorted by id."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows():
        n = _number(row)
        name, role = SEPTA[n]
        out[_device_id(row)] = {
            "id": _device_id(row),
            "class": row.template.device_class,
            "description": f"Storage ring injection septum {name}: {role}.",
            "names": [f"SEPTUM{n}", name],
            "label": name,
            "place": PLACE,
        }
    return [out[key] for key in sorted(out)]


def _pulsed(number: str) -> bool:
    """Whether the septum numbered ``number`` has a sampled current pulse."""
    return any(_number(row) == number and row.template.value_type == "waveform" for row in rows())


def _description(row: _elements.ElementRow) -> str:
    name, _ = SEPTA[_number(row)]
    tail = _tail(row)
    extra = _PULSE_TEXT.get(tail, "") if _pulsed(_number(row)) else ""
    return f"Storage ring injection septum {name} {_FIELD_TEXT[tail]}{extra}"


def channels() -> list[dict[str, Any]]:
    """The septa's channels, sorted by address."""
    by_number = {(_number(row), row.template.site): row.demo_name for row in rows()}
    out: list[dict[str, Any]] = []
    for row in rows():
        template = row.template
        record: dict[str, Any] = {"id": row.demo_name}
        if template.writable:
            record["role"] = "setpoint"
            record["pair"] = by_number[(_number(row), "rb")]
            record["tolerance"] = {"absolute": template.tolerance}
        record["on"] = {"device": _device_id(row)}
        record["signal"] = template.signal
        if template.value_type != "float":
            record["value_type"] = template.value_type
        if template.shape is not None:
            record["shape"] = list(template.shape)
        if template.unit is not None:
            record["unit"] = template.unit
        record["description"] = _description(row)
        out.append(record)
    return sorted(out, key=lambda record: record["id"])


def groups() -> list[dict[str, Any]]:
    """The septa's group."""
    return [
        {
            "id": GROUP,
            "description": (
                "Injection Septa: the storage ring's two injection septum magnets, SEK and "
                "SEN, which bend the beam arriving from the booster-to-storage-ring line "
                "onto the ring's injection orbit while shielding the stored beam. SEK is "
                "pulsed, and its current pulse is sampled as a waveform."
            ),
            "names": ["injection septum"],
            "members": [device["id"] for device in devices()],
        }
    ]


def seeds() -> dict[str, dict[str, Any]]:
    """The septa's seed keys by address: a nominal, or the SEK waveform's pulse."""
    out: dict[str, dict[str, Any]] = {}
    sites = {(_number(row), row.template.site): row.demo_name for row in rows()}
    for row in rows():
        value = row.nominal_value()
        if row.template.value_type == "waveform":
            n = _number(row)
            out[row.demo_name] = {
                "pulse": {"driver": sites[(n, "sp")], "enable": sites[(n, "status")]}
            }
        elif value is not None and (row.template.writable or row.template.site == "status"):
            out[row.demo_name] = {"nominal": value}
    return out


# --- the booster ------------------------------------------------------------------

#: The booster's section, as the table names it.
BOOSTER = "BR"

#: The booster's name in prose, as its existing records write it.
BOOSTER_TEXT = "Booster ring"

#: Each booster family the table adds, and its name in prose.
BOOSTER_FAMILY_TEXT: dict[str, str] = {
    "KICKER": "injection kicker",
    "SEPTUM": "extraction septum",
    "TV": "screen",
    "DIPOLE-RAMP": "bending magnet ramp",
    "QF-RAMP": "focusing quadrupole ramp",
    "QD-RAMP": "defocusing quadrupole ramp",
}

#: Each booster septum's two-digit number and its name.
BOOSTER_SEPTA: dict[str, str] = {"01": "SEN", "02": "SEK"}

#: What each energy-ramp role's channel reads, keyed by the role's tail.
_RAMP_TEXT: dict[str, str] = {
    "gain_sp": "gain setpoint (read-write): the commanded scale of the energy ramp; 1 is "
    "the nominal ramp.",
    "gain_rb": "gain readback (read-only): the applied scale of the energy ramp.",
    "offset_sp": "offset setpoint (read-write): the commanded current offset added to the "
    "energy ramp.",
    "offset_rb": "offset readback (read-only): the applied current offset of the energy ramp.",
    "on": "on status (read-only): the ramped supply is on.",
}

#: What each booster channel reads, for its description.
BOOSTER_ROLE_TEXT: dict[str, str] = {
    **_transfer.ROLE_TEXT,
    "kicker_sp": "voltage setpoint (read-write): the commanded kicker pulse voltage.",
    "kicker_rb": "voltage readback (read-only): the measured kicker pulse voltage.",
    "kicker_on": "on status (read-only): the kicker pulser is on.",
    "septum_sp": "current setpoint (read-write): the commanded septum current.",
    "septum_rb": "current readback (read-only): the measured septum current.",
    "septum_on": "on status (read-only): the septum supply is on.",
    **{
        f"{prefix}_{key}": text
        for prefix in ("dipole_ramp", "quad_ramp")
        for key, text in _RAMP_TEXT.items()
    },
}


def booster_rows() -> list[_elements.ElementRow]:
    """The table's booster rows, one per address, in table order.

    A channel bound by several legacy names has several rows; the first one
    stands for the channel.
    """
    seen: dict[str, _elements.ElementRow] = {}
    for row in _elements.load_elements():
        if row.section == BOOSTER:
            seen.setdefault(row.demo_name, row)
    return list(seen.values())


def _family(row: _elements.ElementRow) -> str:
    return row.demo_name.split(":")[2]


def _booster_pairs() -> dict[str, str]:
    """Each booster setpoint's readback: the ``RB`` address of the same field."""
    addresses = {row.demo_name for row in booster_rows()}
    out: dict[str, str] = {}
    for row in booster_rows():
        if row.template.writable:
            readback = f"{row.demo_name.rsplit(':', 1)[0]}:RB"
            if readback not in addresses:
                raise ValueError(f"{row.demo_name}: no readback {readback} in the table")
            out[row.demo_name] = readback
    return out


def booster_devices() -> list[dict[str, Any]]:
    """The booster devices the table adds, sorted by id."""
    out: dict[str, dict[str, Any]] = {}
    for row in booster_rows():
        key = _standin.device_id(row)
        device_class = row.template.device_class
        if key in out:
            if out[key]["class"] != device_class:
                raise ValueError(f"{key}: rows of classes {out[key]['class']} and {device_class}")
            continue
        family, number = _family(row), _number(row)
        prose = BOOSTER_FAMILY_TEXT[family]
        record: dict[str, Any] = {"id": key, "class": device_class}
        if family == "SEPTUM":
            name = BOOSTER_SEPTA[number]
            record["description"] = f"{BOOSTER_TEXT} {prose} {name}."
            record["names"] = [f"{family}{number}", name]
            record["label"] = name
        else:
            record["description"] = f"{BOOSTER_TEXT} {prose} {number}."
            record["names"] = [f"{family}{number}"]
            record["label"] = f"{family} {number}"
        record["place"] = BOOSTER
        out[key] = record
    return [out[key] for key in sorted(out)]


def booster_channels() -> list[dict[str, Any]]:
    """The booster channels the table adds, sorted by address."""
    pairs = _booster_pairs()
    out: list[dict[str, Any]] = []
    for row in booster_rows():
        template = row.template
        record: dict[str, Any] = {"id": row.demo_name}
        if template.writable:
            record["role"] = "setpoint"
            record["pair"] = pairs[row.demo_name]
            # A tolerance is in the channel's unit, so a unitless setpoint (a ramp
            # gain) states none and settles on its value.
            if template.tolerance is not None and template.unit is not None:
                record["tolerance"] = {"absolute": template.tolerance}
        record["on"] = {"device": _standin.device_id(row)}
        record["signal"] = template.signal
        if template.value_type != "float":
            record["value_type"] = template.value_type
        if template.unit is not None:
            record["unit"] = template.unit
        family, number = _family(row), _number(row)
        prose = BOOSTER_FAMILY_TEXT[family]
        name = BOOSTER_SEPTA[number] if family == "SEPTUM" else number
        record["description"] = f"{BOOSTER_TEXT} {prose} {name} {BOOSTER_ROLE_TEXT[row.role]}"
        out.append(record)
    return sorted(out, key=lambda record: record["id"])


def booster_groups() -> list[dict[str, Any]]:
    """One group per booster family the table adds, sorted by id.

    Their descriptions and names are ``GROUP_TEXT``'s.
    """
    members: dict[str, list[str]] = {}
    for row in booster_rows():
        group = members.setdefault(f"{BOOSTER}/{_family(row)}", [])
        device = _standin.device_id(row)
        if device not in group:
            group.append(device)
    return [_described({"id": key, "members": sorted(members[key])}) for key in sorted(members)]


def booster_seeds() -> dict[str, dict[str, Any]]:
    """The booster seed nominals by address.

    A setpoint, an on-status and an unpaired readback carry the table's
    nominal; a paired readback follows its setpoint and carries none.
    """
    paired = set(_booster_pairs().values())
    out: dict[str, dict[str, Any]] = {}
    for row in booster_rows():
        value = row.nominal_value()
        if value is not None and row.demo_name not in paired:
            out[row.demo_name] = {"nominal": value}
    return out


# --- descriptions ------------------------------------------------------------------

#: The injector's machine places in beam order: each one's description and
#: names. The stand-ins emit GTL and LN and the transfer lines LTB; these texts
#: replace theirs. BTS predates the table and is emitted here, its span kept and
#: its text grown to the line it holds.
PLACE_TEXT: dict[str, tuple[str, tuple[str, ...]]] = {
    "GTL": (
        "Gun-to-Linac Line (GTL): the start of the injector. A thermionic electron gun "
        "produces the beam; bucking coils cancel the solenoid field on its cathode, two "
        "sub-harmonic bunchers compress the long gun pulse into bunches the linac can "
        "capture, and solenoids hold the low-energy beam together on its way to the linac. "
        "Correctors steer it, beam position monitors and screens show where it is, and an "
        "integrating current transformer measures the charge the gun delivers.",
        ("gun-to-linac line", "gun to linac", "GTL", "electron gun line", "injector front end"),
    ),
    "LN": (
        "Linac (LN): the linear accelerator that takes the bunched beam from the "
        "gun-to-linac line to 150 MeV. A buncher cavity and two travelling-wave "
        "accelerating sections, phased from the master RF and held at temperature by their "
        "cooling water, provide the energy gain; solenoids and quadrupoles focus the beam, "
        "correctors steer it, and beam position monitors, screens and an integrating "
        "current transformer measure it.",
        ("linac", "linear accelerator", "LN", "injector linac"),
    ),
    "LTB": (
        "Linac-to-Booster Line (LTB): the transfer line that carries the 150 MeV linac beam "
        "to injection into the booster. Bending magnets route it, focusing and defocusing "
        "quadrupoles match its optics to the booster, correctors steer it, and beam "
        "position monitors, screens and an integrating current transformer follow the beam "
        "and its transmitted charge along the way.",
        ("linac-to-booster line", "linac to booster", "LTB", "booster injection line"),
    ),
    "BTS": (
        "Booster-to-Storage Transfer Line (BTS): the transfer line, about 35 m long, that "
        "carries the 2.0 GeV beam extracted from the booster to injection into the storage "
        "ring. Bending magnets route it, focusing and defocusing quadrupoles match its "
        "optics, and horizontal and vertical correctors steer it onto the injection septa. "
        "Beam position monitors with sum signals, an integrating current transformer, and "
        "screens up to the storage ring's first screen follow the beam; a paddle summary "
        "counts the screens in the beam.",
        (
            "booster-to-storage transfer line",
            "booster to storage",
            "BTS",
            "storage ring injection line",
        ),
    ),
}

#: The injector's family groups: each one's description and names. Every group
#: the stand-ins, the transfer lines and the booster rows emit has one, except
#: the BTS groups that predate the table, which keep their text.
GROUP_TEXT: dict[str, tuple[str, tuple[str, ...]]] = {
    # The gun-to-linac line.
    "GTL/EGUN": (
        "Electron Gun (EGUN): the thermionic gun at the start of the injector. Its high "
        "voltage and cathode heater are switched and set here; with either off, no beam "
        "leaves the gun.",
        ("electron gun", "gun", "thermionic gun"),
    ),
    "GTL/BC": (
        "Gun-to-Linac Bucking Coils (BC): coils around the gun that cancel the solenoid "
        "field on the cathode, so the beam leaves the gun without a net rotation.",
        ("bucking coil", "gun bucking coil"),
    ),
    "GTL/SHB": (
        "Gun-to-Linac Sub-Harmonic Bunchers (SHB): RF cavities at a sub-harmonic of the "
        "linac frequency that compress the long gun pulse into short bunches the linac can "
        "capture.",
        ("sub-harmonic buncher", "buncher", "SHB"),
    ),
    "GTL/SOL": (
        "Gun-to-Linac Solenoids (SOL): focus the low-energy gun beam in both planes on its "
        "way to the linac.",
        ("solenoid", "gun-to-linac solenoid"),
    ),
    "GTL/HCM": (
        "Gun-to-Linac Horizontal Correctors (HCM): small dipoles that steer the gun beam "
        "horizontally into the linac.",
        ("horizontal corrector", "gun-to-linac horizontal corrector"),
    ),
    "GTL/VCM": (
        "Gun-to-Linac Vertical Correctors (VCM): small dipoles that steer the gun beam "
        "vertically into the linac.",
        ("vertical corrector", "gun-to-linac vertical corrector"),
    ),
    "GTL/BPM": (
        "Gun-to-Linac Beam Position Monitors (BPM): measure the horizontal and vertical "
        "position of the gun beam, and the sum signal that tracks its charge.",
        ("beam position monitor", "gun-to-linac BPM"),
    ),
    "GTL/TV": (
        "Gun-to-Linac Screens (TV): insertable paddles that image the gun beam on a camera. "
        "A screen in the beam stops it, so everything downstream sees no beam.",
        ("screen", "TV", "gun-to-linac screen"),
    ),
    "GTL/ICT": (
        "Gun-to-Linac Integrating Current Transformer (ICT): measures the charge per pulse "
        "that the gun delivers into the line.",
        ("integrating current transformer", "charge monitor", "gun charge monitor"),
    ),
    # The linac.
    "LN/MASTER": (
        "Linac Master RF (MASTER): the master oscillator whose phase is the reference for "
        "the bunchers and the accelerating sections.",
        ("master RF", "master oscillator", "linac master oscillator"),
    ),
    "LN/WATER": (
        "Linac RF Cooling Water (WATER): holds the linac structures at the temperature "
        "their RF frequency is tuned for.",
        ("RF cooling water", "linac cooling water"),
    ),
    "LN/BUNCHER": (
        "Linac Buncher Cavity (BUNCHER): the RF structure at the linac entrance that "
        "completes the bunching of the gun beam before acceleration.",
        ("buncher cavity", "linac buncher"),
    ),
    "LN/ACC": (
        "Linac Accelerating Sections (ACC): the travelling-wave structures that accelerate "
        "the bunched beam to 150 MeV.",
        ("accelerating section", "accelerating structure", "linac section"),
    ),
    "LN/SOL": (
        "Linac Solenoids (SOL): focus the beam in both planes while its energy is still "
        "low, at the start of the linac.",
        ("solenoid", "linac solenoid"),
    ),
    "LN/QF": (
        "Linac Focusing Quadrupoles (QF): quadrupoles that focus the linac beam "
        "horizontally and defocus it vertically.",
        ("focusing quadrupole", "linac focusing quadrupole"),
    ),
    "LN/QD": (
        "Linac Defocusing Quadrupoles (QD): quadrupoles that defocus the linac beam "
        "horizontally and focus it vertically.",
        ("defocusing quadrupole", "linac defocusing quadrupole"),
    ),
    "LN/HCM": (
        "Linac Horizontal Correctors (HCM): small dipoles that steer the beam horizontally "
        "through the linac.",
        ("horizontal corrector", "linac horizontal corrector"),
    ),
    "LN/VCM": (
        "Linac Vertical Correctors (VCM): small dipoles that steer the beam vertically "
        "through the linac.",
        ("vertical corrector", "linac vertical corrector"),
    ),
    "LN/BPM": (
        "Linac Beam Position Monitors (BPM): measure the horizontal and vertical beam "
        "position in the linac, and the sum signal that tracks the beam charge.",
        ("beam position monitor", "linac BPM"),
    ),
    "LN/TV": (
        "Linac Screens (TV): insertable paddles that image the linac beam on a camera. A "
        "screen in the beam stops it, so everything downstream sees no beam.",
        ("screen", "TV", "linac screen"),
    ),
    "LN/ICT": (
        "Linac Integrating Current Transformer (ICT): measures the charge per pulse that "
        "the linac transmits.",
        ("integrating current transformer", "charge monitor", "linac charge monitor"),
    ),
    # The linac-to-booster line.
    "LTB/DIPOLE": (
        "Linac-to-Booster Bending Magnets (DIPOLE): dipoles that route the 150 MeV linac "
        "beam along the line to the booster injection point.",
        ("bending magnet", "dipole", "linac-to-booster bend"),
    ),
    "LTB/QF": (
        "Linac-to-Booster Focusing Quadrupoles (QF): quadrupoles that focus the beam "
        "horizontally and, with the defocusing ones, match its optics to the booster.",
        ("focusing quadrupole", "linac-to-booster focusing quadrupole"),
    ),
    "LTB/QD": (
        "Linac-to-Booster Defocusing Quadrupoles (QD): quadrupoles that focus the beam "
        "vertically and, with the focusing ones, match its optics to the booster.",
        ("defocusing quadrupole", "linac-to-booster defocusing quadrupole"),
    ),
    "LTB/HCM": (
        "Linac-to-Booster Horizontal Correctors (HCM): small dipoles that steer the beam "
        "horizontally along the line and into the booster.",
        ("horizontal corrector", "linac-to-booster horizontal corrector"),
    ),
    "LTB/VCM": (
        "Linac-to-Booster Vertical Correctors (VCM): small dipoles that steer the beam "
        "vertically along the line and into the booster.",
        ("vertical corrector", "linac-to-booster vertical corrector"),
    ),
    "LTB/BPM": (
        "Linac-to-Booster Beam Position Monitors (BPM): measure the beam trajectory along "
        "the line in millimetres, and the sum signal that tracks the beam charge.",
        ("beam position monitor", "linac-to-booster BPM"),
    ),
    "LTB/TV": (
        "Linac-to-Booster Screens (TV): insertable paddles that image the beam on a camera "
        "along the line. A screen in the beam stops it, so the booster sees no beam.",
        ("screen", "TV", "linac-to-booster screen"),
    ),
    "LTB/ICT": (
        "Linac-to-Booster Integrating Current Transformer (ICT): measures the charge per "
        "pulse that reaches the booster injection point.",
        ("integrating current transformer", "charge monitor", "linac-to-booster charge monitor"),
    ),
    # The booster's families the table adds.
    "BR/KICKER": (
        "Booster Injection Kicker (KICKER): the pulsed magnet that puts the beam arriving "
        "from the linac-to-booster line onto the booster orbit.",
        ("injection kicker", "kicker", "booster injection kicker"),
    ),
    "BR/SEPTUM": (
        "Booster Extraction Septa (SEPTUM): SEN and SEK, the septum magnets that bend the "
        "extracted 2.0 GeV beam out of the booster into the booster-to-storage line.",
        ("extraction septum", "booster extraction septum"),
    ),
    "BR/TV": (
        "Booster Screens (TV): insertable paddles that image the beam on a camera at "
        "injection, in the booster and at extraction. A screen in the beam stops it, so "
        "everything downstream sees no beam.",
        ("screen", "TV", "booster screen"),
    ),
    "BR/DIPOLE-RAMP": (
        "Booster Bending Magnet Ramp (DIPOLE-RAMP): the gain and offset applied to the "
        "bending magnet supply's energy ramp from 150 MeV to 2.0 GeV.",
        ("bending magnet ramp", "dipole ramp"),
    ),
    "BR/QF-RAMP": (
        "Booster Focusing Quadrupole Ramp (QF-RAMP): the gain and offset applied to the "
        "focusing quadrupole supply's energy ramp, which keeps the focusing in step with "
        "the bending field.",
        ("focusing quadrupole ramp", "QF ramp"),
    ),
    "BR/QD-RAMP": (
        "Booster Defocusing Quadrupole Ramp (QD-RAMP): the gain and offset applied to the "
        "defocusing quadrupole supply's energy ramp, which keeps the focusing in step with "
        "the bending field.",
        ("defocusing quadrupole ramp", "QD ramp"),
    ),
    # The booster-to-storage line's families the table adds.
    "BTS/DIPOLE": (
        "Booster-to-Storage Bending Magnets (DIPOLE): dipoles that route the 2.0 GeV "
        "extracted beam from the booster to the storage ring injection septa.",
        ("bending magnet", "dipole", "booster-to-storage bend"),
    ),
    "BTS/QF": (
        "Booster-to-Storage Focusing Quadrupoles (QF): quadrupoles that focus the beam "
        "horizontally and, with the defocusing ones, match its optics to the storage ring.",
        ("focusing quadrupole", "booster-to-storage focusing quadrupole"),
    ),
    "BTS/QD": (
        "Booster-to-Storage Defocusing Quadrupoles (QD): quadrupoles that focus the beam "
        "vertically and, with the focusing ones, match its optics to the storage ring.",
        ("defocusing quadrupole", "booster-to-storage defocusing quadrupole"),
    ),
    "BTS/TV": (
        "Booster-to-Storage Screens (TV): insertable paddles that image the beam on a "
        "camera along the line, up to the storage ring's first screen. A screen in the beam "
        "stops it, so the storage ring sees no injected beam.",
        ("screen", "TV", "booster-to-storage screen"),
    ),
    "BTS/PADDLES": (
        "Booster-to-Storage Paddle Summary (PADDLES): counts the injector screens in the "
        "beam; 0 means all paddles are out and the beam can reach the storage ring.",
        ("all paddles out", "paddle summary", "paddle count"),
    ),
    "BTS/ICT": (
        "Booster-to-Storage Integrating Current Transformer (ICT): measures the charge per "
        "pulse extracted from the booster and carried to the storage ring.",
        (
            "integrating current transformer",
            "charge monitor",
            "booster-to-storage charge monitor",
        ),
    ),
}


def _described(record: dict[str, Any]) -> dict[str, Any]:
    """The group ``record`` with ``GROUP_TEXT``'s description and names."""
    if record["id"] not in GROUP_TEXT:
        raise ValueError(f"{record['id']}: no description in GROUP_TEXT")
    description, names = GROUP_TEXT[record["id"]]
    return {
        "id": record["id"],
        "description": description,
        "names": list(names),
        "members": record["members"],
    }


def places() -> list[dict[str, Any]]:
    """The injector's machine places in beam order, with ``PLACE_TEXT``'s text."""
    emitted = {record["id"]: record for line in (_standin, _transfer) for record in line.places()}
    emitted["BTS"] = {
        "id": "BTS",
        "level": "machine",
        "span": {"model": "BTS", "from_marker": _transfer.START_MARKER},
    }
    out = []
    for key, (description, names) in PLACE_TEXT.items():
        record = {**emitted.pop(key), "description": description, "names": list(names)}
        out.append(
            {k: record[k] for k in ("id", "level", "description", "names", "span") if k in record}
        )
    if emitted:
        raise ValueError(f"places without text in PLACE_TEXT: {sorted(emitted)}")
    return out


def line_groups() -> list[dict[str, Any]]:
    """The stand-ins' and the transfer lines' groups, with ``GROUP_TEXT``'s text.

    The BTS groups that predate the table keep their text.
    """
    return [
        record if record["id"] in _transfer.FROZEN_GROUPS else _described(record)
        for line in (_standin, _transfer)
        for record in line.groups()
    ]


# --- the gate chain ---------------------------------------------------------------

#: The gun's on-status readbacks: either one ``FALSE`` stops the beam at the gun.
GUN_ON_ROLES = ("gun_hv_on_rb", "gun_heater_on_rb")

#: What a beam reading downstream of a blocking input reads, by role: a float
#: reading's factor, a bool reading's label. No gated reading is exactly 0.
BEAM_GATES: dict[str, Any] = {"bpm_sum": 0.01, "ict_charge": 0.01, "bpm_valid": "FALSE"}

#: What a screen's lamp readback reads while its own paddle is in.
LAMP_LIT = "TRUE"

#: What the "all paddles out" count reads while any paddle is in.
PADDLES_IN = 1


def _chain() -> list[_elements.ElementRow]:
    """Every table row, one per address, sections in beam order, rows in table order."""
    seen: dict[str, _elements.ElementRow] = {}
    for row in _elements.load_elements():
        seen.setdefault(row.demo_name, row)
    order = {section: i for i, section in enumerate(_elements.SECTIONS)}
    return sorted(seen.values(), key=lambda row: order[row.section])


def _paddle(lamp: str) -> str:
    """The paddle readback of the screen whose lamp readback is ``lamp``."""
    return lamp.removesuffix("LAMP:RB") + "PADDLE:RB"


def gates() -> dict[str, dict[str, Any]]:
    """The chain's ``gate`` seeds by address: ``{inputs: {address: label}, gated}``.

    A beam reading's inputs are the gun's on-status readbacks and every paddle
    readback before it in beam order; a lamp's input is its own paddle; the
    paddle count's inputs are every paddle. A reading with no input carries no
    gate, and a storage ring reading carries none.
    """
    rows = _chain()
    names = {row.demo_name for row in rows}
    upstream: dict[str, str] = {}
    paddles: dict[str, str] = {}
    out: dict[str, dict[str, Any]] = {}
    counts: list[str] = []
    for row in rows:
        address = row.demo_name
        if row.role in GUN_ON_ROLES:
            upstream[address] = "FALSE"
        elif row.role == "tv_paddle_rb":
            upstream[address] = paddles[address] = "TRUE"
        elif row.role in BEAM_GATES and row.section != "SR" and upstream:
            out[address] = {"inputs": dict(sorted(upstream.items())), "gated": BEAM_GATES[row.role]}
        elif row.role == "tv_lamp_rb":
            paddle = _paddle(address)
            if paddle not in names:
                raise ValueError(f"{address}: no paddle readback {paddle} in the element table")
            out[address] = {"inputs": {paddle: "TRUE"}, "gated": LAMP_LIT}
        elif row.role == "paddle_count":
            counts.append(address)
    for address in counts:
        out[address] = {"inputs": dict(sorted(paddles.items())), "gated": PADDLES_IN}
    if stray := sorted(a for a in out if a.split(":", 1)[0] == "SR"):
        raise ValueError(f"storage ring channels in the gate chain: {stray}")
    return out


# --- merging ----------------------------------------------------------------------

#: The per-entry seed keys this module owns; every other key of an entry is kept.
_SEED_KEYS = ("nominal", "pulse")

#: The seed key this module owns in every entry.
_GATE_KEY = "gate"


def _merge_seeds(
    path: Path, entries: dict[str, dict[str, Any]], gate_seeds: dict[str, dict[str, Any]]
) -> None:
    """Set the owned keys in ``seeds.yaml``, written as ``reseed`` writes it.

    Each of ``entries`` gets its ``_SEED_KEYS`` replaced; every entry loses its
    ``gate`` and the addresses of ``gate_seeds`` get theirs.
    """
    import yaml

    committed: dict[str, dict[str, Any]] = reseed.load(path) if path.is_file() else {}
    committed = {
        address: {k: v for k, v in entry.items() if k != _GATE_KEY}
        for address, entry in (committed or {}).items()
    }
    for address, keys in entries.items():
        kept = {k: v for k, v in committed.get(address, {}).items() if k not in _SEED_KEYS}
        committed[address] = {**kept, **keys}
    for address, gate in gate_seeds.items():
        committed.setdefault(address, {})[_GATE_KEY] = gate
    document = {
        address: {
            key: committed[address][key] for key in reseed.KEY_ORDER if key in committed[address]
        }
        for address in sorted(committed)
        if committed[address]
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def write_sources(tree: Path) -> None:
    """Merge the emitted injector records into the facility tree ``tree``.

    The septa's, the booster's, the stand-ins' and the transfer lines' places,
    devices, channels and groups go into ``records/``, the places and groups
    with the hand-authored text of ``PLACE_TEXT`` and ``GROUP_TEXT``, and their
    seed keys and the gate chain into ``seeds.yaml``; every other record keeps
    its text.
    """
    records = tree / "records"
    lines = (_standin, _transfer)
    _line._merge(records / "places.yaml", places(), "id", ordered=False)
    emitters = (
        ("devices", devices() + booster_devices() + [r for x in lines for r in x.devices()]),
        ("channels", channels() + booster_channels() + [r for x in lines for r in x.channels()]),
        ("groups", groups() + booster_groups() + line_groups()),
    )
    for name, emitted in emitters:
        _line._merge(records / f"{name}.yaml", emitted, "id", ordered=True)
    _merge_seeds(
        tree / "seeds.yaml",
        {**seeds(), **booster_seeds(), **_standin.seeds(), **_transfer.seeds()},
        gates(),
    )
