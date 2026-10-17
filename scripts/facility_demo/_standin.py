"""The example facility's gun-to-linac line and linac, as texture stand-ins.

``_elements.load_elements()`` reads ``injector_elements.tsv``; this module turns
its GTL (gun to linac) and LN (linac) rows into records, in the table's beam
order. No deck models either section: the texture serves every channel from
its seed, a readback follows its setpoint, and a beam position monitor reads
its seeded position with the rule table's noise.

What the rows describe:

- the electron gun: high voltage, heater and auxiliary supply on-offs, the
  high voltage, the heater current, the gun phase and the grid bias;
- the RF stand-ins: the sub-harmonic bunchers' phase and voltage with their
  on-offs, the linac master phase, the second accelerating section's phase,
  the RF cooling water temperature, and the cavity temperatures;
- the magnets: solenoids, bucking coils, quadrupoles and correctors, each with
  its current setpoint, readback and on-status;
- the diagnostics: beam position monitors with their sum signal and validity,
  insertable screens (paddle command, paddle position, lamp), and the
  integrating current transformer that stands in for transmitted charge.

Each setpoint pairs with the readback of the same field; a setpoint, an
on-status and an unpaired readback carry the table's nominal as their seed,
while a paired readback follows its setpoint and carries none. Setpoints are
writable by default: no limits record is written. Every device sits in the
place of its section, and every family gets a group.
"""

from __future__ import annotations

from typing import Any

import _elements

#: The sections whose rows this module emits.
SECTIONS = ("GTL", "LN")

#: Each section's name in prose, its place description, and its place names.
SECTION_TEXT: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "GTL": (
        "Gun-to-linac line",
        "Gun-to-Linac Line (GTL): the electron gun and the low-energy line that bunches and "
        "steers its beam into the linac, with sub-harmonic bunchers, solenoids, correctors, "
        "beam position monitors and screens.",
        ("gun-to-linac line", "gun to linac"),
    ),
    "LN": (
        "Linac",
        "Linac (LN): the travelling-wave linear accelerator that takes the bunched gun beam "
        "to 150 MeV for the linac-to-booster line, with its RF stand-ins, solenoids, "
        "quadrupoles, correctors and diagnostics.",
        ("linac", "linear accelerator"),
    ),
}

#: Each family: its name in prose, its group description, and its group names.
FAMILY_TEXT: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "EGUN": (
        "electron gun",
        "Electron Gun: the thermionic gun that produces the injector beam.",
        ("electron gun",),
    ),
    "BC": (
        "bucking coil",
        "Bucking Coils: cancel the solenoid field on the gun cathode.",
        ("bucking coil",),
    ),
    "HCM": (
        "horizontal corrector",
        "Horizontal Correctors: horizontal steering magnets.",
        ("horizontal corrector",),
    ),
    "VCM": (
        "vertical corrector",
        "Vertical Correctors: vertical steering magnets.",
        ("vertical corrector",),
    ),
    "SOL": (
        "solenoid",
        "Solenoids: focus the low-energy beam.",
        ("solenoid",),
    ),
    "SHB": (
        "sub-harmonic buncher",
        "Sub-Harmonic Bunchers: compress the gun pulse into bunches for the linac.",
        ("sub-harmonic buncher", "buncher"),
    ),
    "BPM": (
        "beam position monitor",
        "Beam Position Monitors: measure beam position and the sum signal.",
        ("beam position monitor",),
    ),
    "TV": (
        "screen",
        "Screens: insertable paddles that image the beam and block it while in.",
        ("screen", "TV"),
    ),
    "ICT": (
        "integrating current transformer",
        "Integrating Current Transformers: measure the transmitted charge per pulse.",
        ("integrating current transformer", "charge monitor"),
    ),
    "MASTER": (
        "master RF",
        "Master RF: the linac's master oscillator phase.",
        ("master RF", "master oscillator"),
    ),
    "WATER": (
        "RF cooling water",
        "RF Cooling Water: holds the linac structures at their working temperature.",
        ("RF cooling water",),
    ),
    "BUNCHER": (
        "buncher cavity",
        "Buncher Cavity: the linac's bunching structure.",
        ("buncher cavity",),
    ),
    "ACC": (
        "accelerating section",
        "Accelerating Sections: the linac's travelling-wave structures.",
        ("accelerating section",),
    ),
    "QF": (
        "focusing quadrupole",
        "Focusing Quadrupoles: horizontally focusing quadrupoles.",
        ("focusing quadrupole",),
    ),
    "QD": (
        "defocusing quadrupole",
        "Defocusing Quadrupoles: horizontally defocusing quadrupoles.",
        ("defocusing quadrupole",),
    ),
}

#: What each role's channel reads, for its description.
ROLE_TEXT: dict[str, str] = {
    "gun_voltage_sp": "high voltage setpoint (read-write): the commanded gun high voltage.",
    "gun_voltage_rb": "high voltage readback (read-only): the measured gun high voltage.",
    "gun_hv_on_sp": "high voltage on command (read-write): switches the gun high voltage.",
    "gun_hv_on_rb": "high voltage on status (read-only): the gun high voltage is on.",
    "gun_heater_sp": "heater current setpoint (read-write): the commanded cathode heater current.",
    "gun_heater_rb": "heater current readback (read-only): the measured cathode heater current.",
    "gun_heater_on_sp": "heater on command (read-write): switches the cathode heater.",
    "gun_heater_on_rb": "heater on status (read-only): the cathode heater is on.",
    "gun_aux_on_sp": "auxiliary supply on command (read-write): switches the auxiliary supply.",
    "gun_aux_on_rb": "auxiliary supply on status (read-only): the auxiliary supply is on.",
    "gun_phase_sp": "phase setpoint (read-write): the commanded gun timing phase.",
    "gun_phase_rb": "phase readback (read-only): the measured gun timing phase.",
    "gun_bias_sp": "grid bias setpoint (read-write): the commanded grid bias voltage.",
    "gun_bias_rb": "grid bias readback (read-only): the measured grid bias voltage.",
    "bucking_sp": "current setpoint (read-write): the commanded coil current.",
    "bucking_rb": "current readback (read-only): the measured coil current.",
    "bucking_on": "on status (read-only): the coil supply is on.",
    "solenoid_sp": "current setpoint (read-write): the commanded solenoid current.",
    "solenoid_rb": "current readback (read-only): the measured solenoid current.",
    "solenoid_on": "on status (read-only): the solenoid supply is on.",
    "hcm_sp": "current setpoint (read-write): the commanded corrector current.",
    "hcm_rb": "current readback (read-only): the measured corrector current.",
    "hcm_on": "on status (read-only): the corrector supply is on.",
    "vcm_sp": "current setpoint (read-write): the commanded corrector current.",
    "vcm_rb": "current readback (read-only): the measured corrector current.",
    "vcm_on": "on status (read-only): the corrector supply is on.",
    "quad_sp": "current setpoint (read-write): the commanded quadrupole current.",
    "quad_rb": "current readback (read-only): the measured quadrupole current.",
    "quad_on": "on status (read-only): the quadrupole supply is on.",
    "shb_phase_sp": "phase setpoint (read-write): the commanded buncher phase.",
    "shb_phase_rb": "phase readback (read-only): the measured buncher phase.",
    "shb_phase_on_sp": "phase loop on command (read-write): switches the buncher phase loop.",
    "shb_phase_on_rb": "phase loop on status (read-only): the buncher phase loop is on.",
    "shb_voltage_sp": "RF voltage setpoint (read-write): the commanded buncher gap voltage.",
    "shb_voltage_rb": "RF voltage readback (read-only): the measured buncher gap voltage.",
    "shb_hv_on_sp": "high voltage on command (read-write): switches the buncher amplifier.",
    "shb_hv_on_rb": "high voltage on status (read-only): the buncher amplifier is on.",
    "bpm_x": "position horizontal (read-only): beam X position.",
    "bpm_y": "position vertical (read-only): beam Y position.",
    "bpm_sum": "sum signal (read-only): the summed button signal, relative to full beam.",
    "bpm_valid": "valid status (read-only): the monitor sees beam and its reading is valid.",
    "tv_paddle_sp": "paddle in command (read-write): inserts the screen into the beam.",
    "tv_paddle_rb": "paddle in status (read-only): the screen is in the beam and blocks it.",
    "tv_lamp_rb": "lamp status (read-only): the screen lamp is lit.",
    "ict_charge": "charge readback (read-only): the transmitted charge per pulse.",
    "master_phase_sp": "phase setpoint (read-write): the commanded master RF phase.",
    "master_phase_rb": "phase readback (read-only): the measured master RF phase.",
    "master_on": "on status (read-only): the master RF drive is on.",
    "water_temperature_sp": "temperature setpoint (read-write): the commanded water temperature.",
    "water_temperature_rb": "temperature readback (read-only): the measured water temperature.",
    "water_heater_on": "heater on status (read-only): the water heater is on.",
    "water_interlock": "interlock status (read-only): the water interlock has tripped.",
    "buncher_temperature": "temperature readback (read-only): the buncher body temperature.",
    "acc_temperature": "temperature readback (read-only): the section body temperature.",
    "acc_phase_sp": "phase setpoint (read-write): the commanded section RF phase.",
    "acc_phase_rb": "phase readback (read-only): the measured section RF phase.",
    "acc_on": "on status (read-only): the section RF drive is on.",
}


def _tokens(row: _elements.ElementRow) -> list[str]:
    return row.demo_name.split(":")


def _family(row: _elements.ElementRow) -> str:
    return _tokens(row)[2]


def _number(row: _elements.ElementRow) -> str:
    return _tokens(row)[3]


def _field(row: _elements.ElementRow) -> str:
    """The address up to its last token: the device's field."""
    return row.demo_name.rsplit(":", 1)[0]


def device_id(row: _elements.ElementRow) -> str:
    """The id of the device a row's channel is on: ``<SEC>/<FAMILY><NN>``."""
    return f"{row.section}/{_family(row)}{_number(row)}"


def rows() -> list[_elements.ElementRow]:
    """The table's GTL and LN rows, one per address, in beam order.

    A channel bound by several legacy names has several rows; the first one
    stands for the channel.
    """
    seen: dict[str, _elements.ElementRow] = {}
    for row in _elements.load_elements():
        if row.section in SECTIONS:
            seen.setdefault(row.demo_name, row)
    return list(seen.values())


def _pairs() -> dict[str, str]:
    """Each setpoint's readback: the ``RB`` address of the same field."""
    addresses = {row.demo_name for row in rows()}
    out: dict[str, str] = {}
    for row in rows():
        if row.template.writable:
            readback = f"{_field(row)}:RB"
            if readback not in addresses:
                raise ValueError(f"{row.demo_name}: no readback {readback} in the table")
            out[row.demo_name] = readback
    return out


def devices() -> list[dict[str, Any]]:
    """The GTL and LN devices, sorted by id."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows():
        key = device_id(row)
        device_class = row.template.device_class
        if key in out:
            if out[key]["class"] != device_class:
                raise ValueError(f"{key}: rows of classes {out[key]['class']} and {device_class}")
            continue
        family, number = _family(row), _number(row)
        section_name = SECTION_TEXT[row.section][0]
        out[key] = {
            "id": key,
            "class": device_class,
            "description": f"{section_name} {FAMILY_TEXT[family][0]} {number}.",
            "names": [f"{family}{number}"],
            "label": f"{family} {number}",
            "place": row.section,
        }
    return [out[key] for key in sorted(out)]


def _description(row: _elements.ElementRow) -> str:
    section_name = SECTION_TEXT[row.section][0]
    return f"{section_name} {FAMILY_TEXT[_family(row)][0]} {_number(row)} {ROLE_TEXT[row.role]}"


def channels() -> list[dict[str, Any]]:
    """The GTL and LN channels, sorted by address."""
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
        record["on"] = {"device": device_id(row)}
        record["signal"] = template.signal
        if template.value_type != "float":
            record["value_type"] = template.value_type
        if template.unit is not None:
            record["unit"] = template.unit
        record["description"] = _description(row)
        out.append(record)
    return sorted(out, key=lambda record: record["id"])


def groups() -> list[dict[str, Any]]:
    """One group per GTL and LN family, sorted by id."""
    members: dict[str, list[str]] = {}
    for device in devices():
        section = device["place"]
        family = device["id"].split("/", 1)[1].rstrip("0123456789")
        members.setdefault(f"{section}/{family}", []).append(device["id"])
    out = []
    for key in sorted(members):
        section, family = key.split("/")
        _, text, names = FAMILY_TEXT[family]
        out.append(
            {
                "id": key,
                "description": f"{SECTION_TEXT[section][0]} {text[0].lower()}{text[1:]}",
                "names": list(names),
                "members": members[key],
            }
        )
    return out


def places() -> list[dict[str, Any]]:
    """The GTL and LN machine places, in beam order."""
    return [
        {
            "id": section,
            "level": "machine",
            "description": SECTION_TEXT[section][1],
            "names": list(SECTION_TEXT[section][2]),
        }
        for section in SECTIONS
    ]


def seeds() -> dict[str, dict[str, Any]]:
    """The GTL and LN seed nominals by address.

    A setpoint, an on-status and an unpaired readback carry the table's
    nominal; a paired readback follows its setpoint and carries none.
    """
    paired = set(_pairs().values())
    out: dict[str, dict[str, Any]] = {}
    for row in rows():
        value = row.nominal_value()
        if value is not None and row.demo_name not in paired:
            out[row.demo_name] = {"nominal": value}
    return out
