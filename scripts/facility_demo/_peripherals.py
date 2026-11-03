"""The storage ring's peripheral devices of the example facility, sector by sector.

The example facility's storage-ring sector screens draw its peripherals beside the
lattice, the BPMs and correctors, the injection septa and the RF straight. This
module emits their records, laid out by the example facility's own design:

* ion pumps, six in each arc (72);
* ion gauges, one in each arc and one or two in each straight;
* sector gate valves, two per sector (one at the straight's upstream end and one at
  the arc's downstream end), with a moving flag and an open/close setpoint;
* insertion devices in the straights that hold them, named by their kind and their
  number round the ring (``EPU7``, the device behind ``SR:ID:07``);
* front-end photon shutters, a fixed count per sector;
* absorber thermocouples, two rows of ten under each straight that has them;
* scrapers in sectors 1, 2, 3 and 12;
* the tunnel's air temperature and its air-handler valves;
* three passive harmonic cavities in straight 2;
* a packed interlock word on each injection septum.

The vacuum records read pressures in Torr, pump currents in uA and pump voltages
in kV.

Every record is emitted whole and merged by id, so running it twice writes the same
files. Seeds get their ``nominal`` (and the valves' closed-switch ``gate``) here;
``scripts/demo_seeds/reseed.py`` stamps the noise and the tolerances.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import _line

SECTORS = range(1, 13)

#: The facility-added device classes, each the child of a vocabulary class.
CLASSES: tuple[dict[str, Any], ...] = (
    {
        "class": "InsertionDevice",
        "parent": "Magnet",
        "aliases": ["insertion device", "undulator", "wiggler", "EPU", "ID"],
        "description": "Undulator or wiggler in a straight section; its magnetic gap sets "
        "the photon energy of the beamline it serves.",
    },
    {
        "class": "PhotonShutter",
        "parent": "Actuated",
        "aliases": ["photon shutter", "front-end shutter", "beamline shutter"],
        "description": "Front-end shutter that admits or blocks the photon beam of one port.",
    },
    {
        "class": "TemperatureSensor",
        "parent": "Instrumentation",
        "aliases": ["thermocouple", "temperature sensor", "TC"],
        "description": "Thermocouple or air-temperature sensor.",
    },
    {
        "class": "AirHandler",
        "parent": "AcceleratorDevice",
        "aliases": ["air handler", "AHU", "air handling unit"],
        "description": "Air-handling unit that conditions the tunnel air.",
    },
    {
        "class": "HarmonicCavity",
        "parent": "BunchingCavity",
        "aliases": ["harmonic cavity", "third harmonic cavity", "bunch lengthening cavity"],
        "description": "Passive higher-harmonic RF cavity that lengthens the bunches.",
    },
)

ION_PUMPS_PER_SECTOR = 6
#: Straights with two ion gauges; every other straight has one.
TWO_GAUGE_STRAIGHTS = frozenset({1, 3, 4, 6, 11})
#: Valve ``n`` (1..12) sits at the upstream end of straight ``n``; valve ``12 + n`` at the
#: downstream end of arc ``n``.
VALVES_PER_SECTOR = 2

#: The insertion devices in beam order: sector, kind, period (mm), elliptically
#: polarising, nominal gap (mm), nominal row offset (mm), beamline owns it.
INSERTION_DEVICES: tuple[tuple[int, str, int, bool, float, float, bool], ...] = (
    (2, "U", 50, False, 18.0, 0.0, True),
    (4, "EPU", 50, True, 16.5, 0.0, True),
    (4, "EPU", 90, True, 22.0, 12.0, True),
    (5, "W", 114, False, 14.2, 0.0, True),
    (6, "IVU", 23, False, 6.8, 0.0, True),
    (6, "EPU", 50, True, 15.4, -8.5, True),
    (7, "EPU", 50, True, 20.1, 0.0, True),
    (7, "EPU", 50, True, 17.8, 25.0, True),
    (8, "U", 80, False, 31.0, 0.0, False),
    (9, "U", 100, False, 42.5, 0.0, True),
    (10, "U", 50, False, 24.6, 0.0, True),
    (11, "EPU", 50, True, 19.3, 0.0, True),
    (11, "EPU", 50, True, 205.0, 0.0, False),
    (12, "U", 50, False, 16.9, 0.0, True),
)
KIND_TEXT = {
    "U": "undulator",
    "EPU": "elliptically polarising undulator",
    "W": "wiggler",
    "IVU": "in-vacuum undulator",
}
#: Spoken names of the kinds, the long form of a device's name.
KIND_NAME = {
    "U": "Undulator",
    "EPU": "EPU",
    "W": "Wiggler",
    "IVU": "In-vacuum undulator",
}

#: Front-end shutters per sector.
SHUTTERS = {1: 0, 2: 2, 3: 0, 4: 2, 5: 11, 6: 6, 7: 6, 8: 10, 9: 4, 10: 3, 11: 6, 12: 6}
#: Every ninth shutter reads closed, a beamline that is not taking beam.
CLOSED_SHUTTER_EVERY = 9

#: Straights whose absorbers carry thermocouples: rows by straight, ten per row.
TC_STRAIGHTS = (2, 5, 6, 7, 8, 9, 10, 11, 12)
TC_ROWS = ("U", "L")
TC_PER_ROW = 10
#: Straight 4 has a single thermocouple.
TC_SINGLE = 4
TC_LIMIT_C = 80.0

#: The scrapers: family, sector, position in the sector (1 the straight's start, 4 the
#: middle of the arc, 6 its end), nominal insertion (mm).
SCRAPERS: tuple[tuple[str, int, int, float], ...] = (
    ("top", 1, 1, 0.0),
    ("bottom", 1, 1, 0.0),
    ("top", 1, 4, 1.8),
    ("bottom", 1, 4, 0.0),
    ("top", 2, 6, 0.0),
    ("top", 3, 1, 0.0),
    ("bottom", 3, 1, 0.0),
    ("inside", 3, 1, 0.0),
    ("top", 12, 6, 0.6),
)
#: Where a scraper sits, by its index in the sector: the straight's start, the
#: middle of the arc, the arc's end.
SCRAPER_WHERE = {1: "straight scraper", 4: "mid-arc scraper", 6: "arc-end scraper"}
SCRAPER_TEXT = {
    "top": "Top scraper: blade inserted from above",
    "bottom": "Bottom scraper: blade inserted from below",
    "inside": "Inside scraper: horizontal blade inserted from the inner side",
}

#: Tunnel air-temperature sensors and air-handler valves, by sector.
AIR_SECTORS = (1, 3, 4, 6, 7, 8, 9, 11, 12)
AHU_SECTORS = (1, 3, 4, 6, 12)

HARMONIC_CAVITIES = 3
HARMONIC_STRAIGHT = 2

#: The injection septa's interlock word, bit by bit from bit 0.
INTERLOCK_BITS = ("Vacuum OK", "Crash-off OK", "HV cover OK", "Over-temp OK", "HV ready", "HV on")
INTERLOCK_HEALTHY = (1 << len(INTERLOCK_BITS)) - 1

#: The radiation monitors' sectors, monitor n in sector RADIATION_SECTORS[n - 1].
RADIATION_SECTORS = (1, 4, 7, 10)

SR = "Storage ring"


def _place(sector: int) -> str:
    return f"SR/SECT{sector}"


def _channel(
    address: str,
    device: str,
    description: str,
    *,
    signal: str | None = None,
    unit: str | None = None,
    value_type: str | None = None,
    precision: int | None = None,
    pair: str | None = None,
    setpoint: bool = False,
) -> dict[str, Any]:
    """One channel record, keys in the order the committed file writes them.

    A setpoint names its readback in ``pair``; one with ``setpoint`` and no pair is a
    command that pairs with itself.
    """
    record: dict[str, Any] = {"id": address}
    if pair is not None or setpoint:
        record["role"] = "setpoint"
    if pair is not None:
        record["pair"] = pair
    record["on"] = {"device": device}
    if signal is not None:
        record["signal"] = signal
    if unit is not None:
        record["unit"] = unit
    if value_type is not None:
        record["value_type"] = value_type
    if precision is not None:
        record["precision"] = precision
    record["description"] = description
    return record


def _device(
    device: str,
    cls: str,
    description: str,
    names: list[str],
    label: str,
    place: str,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "id": device,
        "class": cls,
        "description": description,
        "names": names,
        "label": label,
        "place": place,
    }
    if attributes:
        record["attributes"] = attributes
    return record


# --------------------------------------------------------------------------- ion pumps


def ion_pump_numbers(sector: int) -> list[int]:
    first = (sector - 1) * ION_PUMPS_PER_SECTOR + 1
    return list(range(first, first + ION_PUMPS_PER_SECTOR))


def _ion_pump_pressure_torr(number: int) -> float:
    """A pump's nominal pressure: 0.5 to 1.5 nTorr, higher next to the straights."""
    k = (number - 1) % ION_PUMPS_PER_SECTOR
    edge = 0.45 if k in (0, ION_PUMPS_PER_SECTOR - 1) else 0.0
    scatter = 0.1 * math.sin(1.7 * number)
    return round((0.65 + edge + scatter) * 1e-9, 12)


def _ion_pumps() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for sector in SECTORS:
        for n in ion_pump_numbers(sector):
            device = f"SR/ION-PUMP{n:02d}"
            head = f"SR:VAC:ION-PUMP:{n:02d}"
            devices.append(
                _device(
                    device,
                    "Pump",
                    "Ion Pumps: Sputter-ion pumps that provide UHV pumping in the storage ring. "
                    "No moving parts, clean operation, excellent for UHV. Use high voltage (kV "
                    "range) to ionize and trap gas molecules.",
                    [f"ION-PUMP{n:02d}", "ion pump"],
                    f"ION-PUMP {n:02d}",
                    _place(sector),
                )
            )
            what = f"{SR} ion pump {n:02d}"
            channels += [
                _channel(
                    f"{head}:CURRENT:RB",
                    device,
                    f"{what} current readback. Ion Pump Current (microAmperes): Current "
                    "proportional to gas pumping rate. Low current (<1 μA) indicates good "
                    "vacuum. Current Readback (read-only): Measured ion pump current.",
                    signal="current_readback",
                    unit="uA",
                    precision=2,
                ),
                _channel(
                    f"{head}:PRESSURE:RB",
                    device,
                    f"{what} pressure readback. Vacuum Pressure (Torr): Local chamber "
                    "pressure calculated from pump current. Target: <1×10^-9 Torr. Pressure "
                    "Readback (read-only): Calculated vacuum pressure from pump current.",
                    signal="vacuum_readback",
                    unit="Torr",
                ),
                _channel(
                    f"{head}:STATUS:FAULT",
                    device,
                    f"{what} status fault. Pump Operational Status: Health and fault "
                    "monitoring. Fault Status (read-only): Pump fault detected.",
                    signal="fault_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:STATUS:ON",
                    device,
                    f"{what} status on. Pump Operational Status: Health and fault monitoring. "
                    "On Status (read-only): Pump high voltage enabled.",
                    signal="power_on_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:VOLTAGE:RB",
                    device,
                    f"{what} voltage readback. High Voltage Supply (Volts): Operating voltage "
                    "for ion pump (typically 3-7 kV). Voltage Readback (read-only): Actual "
                    "measured ion pump operating voltage. Reads in kV.",
                    signal="voltage_readback",
                    unit="kV",
                    precision=2,
                ),
                _channel(
                    f"{head}:VOLTAGE:SP",
                    device,
                    f"{what} voltage setpoint. High Voltage Supply (Volts): Operating voltage "
                    "for ion pump (typically 3-7 kV). Voltage Setpoint (read-write): Commanded "
                    "ion pump high voltage. Reads in kV.",
                    signal="voltage_setpoint",
                    unit="kV",
                    precision=2,
                    pair=f"{head}:VOLTAGE:RB",
                ),
            ]
            pressure = _ion_pump_pressure_torr(n)
            seeds[f"{head}:CURRENT:RB"] = {"nominal": round(pressure * 1e9 * 0.8, 3)}
            seeds[f"{head}:PRESSURE:RB"] = {"nominal": pressure, "clamp": [0.0, None]}
            seeds[f"{head}:STATUS:FAULT"] = {"nominal": "FALSE"}
            seeds[f"{head}:STATUS:ON"] = {"nominal": "TRUE"}
            seeds[f"{head}:VOLTAGE:RB"] = {"nominal": 5.0}
            seeds[f"{head}:VOLTAGE:SP"] = {"nominal": 5.0}
    return devices, channels, seeds


# --------------------------------------------------------------------------- ion gauges


def gauge_names(sector: int) -> list[tuple[str, str]]:
    """The sector's gauges in beam order: (name, where) -- the straight's, then the arc's."""
    letters = "AB" if sector in TWO_GAUGE_STRAIGHTS else "A"
    straight = [(f"ST{sector:02d}{letter}", "straight") for letter in letters]
    return [*straight, (f"SR{sector:02d}", "arc")]


def _gauges() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for sector in SECTORS:
        for name, where in gauge_names(sector):
            device = f"SR/GAUGE{name}"
            head = f"SR:VAC:GAUGE:{name}"
            devices.append(
                _device(
                    device,
                    "Gauge",
                    # The arc gauges keep the description the channel finder's
                    # gauge family has always carried.
                    "Cold cathode ionization gauges, one per storage-ring sector."
                    if where == "arc"
                    else f"Cold cathode ionization gauge in straight {sector}; a trip removes "
                    "the sector valves' permission to open.",
                    [f"GAUGE{name}", "vacuum gauge", "ion gauge"],
                    f"GAUGE {name}",
                    _place(sector),
                )
            )
            what = f"{SR} vacuum gauge {name}"
            channels += [
                _channel(
                    f"{head}:PRESSURE:RB",
                    device,
                    f"{what} pressure readback. Measured Pressure (Torr): Direct pressure "
                    "reading from ionization gauge. Pressure Readback (read-only): Direct "
                    "gauge measurement of local vacuum pressure.",
                    signal="vacuum_readback",
                    unit="Torr",
                ),
                _channel(
                    f"{head}:RESET:SP",
                    device,
                    f"{what} trip reset command (read-write): clears a latched gauge trip once "
                    "the pressure is back below the trip level.",
                    signal="reset_command",
                    value_type="bool",
                    setpoint=True,
                ),
                _channel(
                    f"{head}:STATUS:ON",
                    device,
                    f"{what} status on. Gauge Operational Status: Health and validity "
                    "monitoring. On Status (read-only): Gauge powered and measuring.",
                    signal="power_on_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:STATUS:TRIP",
                    device,
                    f"{what} trip status (read-only): the pressure crossed the gauge's trip "
                    "level and the sector valves lost their permission to open.",
                    signal="interlock_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:STATUS:VALID",
                    device,
                    f"{what} status valid. Gauge Operational Status: Health and validity "
                    "monitoring. Valid Status (read-only): Pressure reading is valid and "
                    "within range.",
                    signal="status",
                    value_type="bool",
                ),
            ]
            torr = 3.2e-10 if where == "arc" else 5.5e-10
            torr = round(torr * (1 + 0.15 * math.sin(2.3 * sector + len(name))), 13)
            seeds[f"{head}:PRESSURE:RB"] = {"nominal": torr, "clamp": [0.0, None]}
            seeds[f"{head}:RESET:SP"] = {"nominal": "FALSE"}
            seeds[f"{head}:STATUS:ON"] = {"nominal": "TRUE"}
            seeds[f"{head}:STATUS:TRIP"] = {"nominal": "FALSE"}
            seeds[f"{head}:STATUS:VALID"] = {"nominal": "TRUE"}
    return devices, channels, seeds


# --------------------------------------------------------------------------- valves


def valve_numbers(sector: int) -> list[int]:
    """The sector's valves in beam order: the straight's upstream one, then the arc's."""
    return [sector, len(SECTORS) + sector]


def _valves() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for sector in SECTORS:
        for n in valve_numbers(sector):
            device = f"SR/VALVE{n:02d}"
            head = f"SR:VAC:VALVE:{n:02d}"
            where = (
                f"upstream end of straight {sector}"
                if n == sector
                else f"downstream end of arc {sector}"
            )
            devices.append(
                _device(
                    device,
                    "Valve",
                    "Vacuum Valves: Isolation and control valves for vacuum system. Gate valves "
                    "or all-metal valves for UHV compatibility. Used to isolate sectors during "
                    "maintenance." + ("" if n == sector else f" This one sits at the {where}."),
                    [f"VALVE{n:02d}", "gate valve"],
                    f"VALVE {n:02d}",
                    _place(sector),
                )
            )
            what = f"{SR} gate valve {n:02d}"
            channels += [
                _channel(
                    f"{head}:CONTROL:CLOSE",
                    device,
                    f"{what} control close. Valve Control Commands: Commands to actuate valve. "
                    "Close Command: Command channel to close valve.",
                    signal="valve_open_command",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:CONTROL:OPEN",
                    device,
                    f"{what} control open. Valve Control Commands: Commands to actuate valve. "
                    "Open Command: Command channel to open valve.",
                    signal="valve_open_command",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:OPEN:SP",
                    device,
                    f"{what} open setpoint (read-write): TRUE opens the valve, FALSE closes "
                    "it; the open and closed limit switches follow.",
                    signal="valve_open_command",
                    value_type="bool",
                    pair=f"{head}:POSITION:OPEN",
                ),
                _channel(
                    f"{head}:POSITION:CLOSED",
                    device,
                    f"{what} position closed. Valve Position State: Current valve position "
                    "(open/closed). Closed Status (read-only): Valve fully closed limit switch "
                    "active.",
                    signal="valve_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:POSITION:OPEN",
                    device,
                    f"{what} position open. Valve Position State: Current valve position "
                    "(open/closed). Open Status (read-only): Valve fully open limit switch "
                    "active.",
                    signal="valve_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:STATUS:FAULT",
                    device,
                    f"{what} status fault. Valve Operational Status: Ready and fault "
                    "monitoring. Fault Status (read-only): Valve fault or timeout condition.",
                    signal="fault_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:STATUS:MOVING",
                    device,
                    f"{what} moving status (read-only): the valve is between its open and "
                    "closed limit switches.",
                    signal="ramping_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:STATUS:READY",
                    device,
                    f"{what} status ready. Valve Operational Status: Ready and fault "
                    "monitoring. Ready Status (read-only): Valve controller ready.",
                    signal="ready_status",
                    value_type="bool",
                ),
            ]
            seeds[f"{head}:CONTROL:CLOSE"] = {"nominal": "TRUE"}
            seeds[f"{head}:CONTROL:OPEN"] = {"nominal": "TRUE"}
            seeds[f"{head}:OPEN:SP"] = {"nominal": "TRUE"}
            seeds[f"{head}:POSITION:CLOSED"] = {
                "nominal": "FALSE",
                "gate": {"inputs": {f"{head}:POSITION:OPEN": "FALSE"}, "gated": "TRUE"},
            }
            seeds[f"{head}:POSITION:OPEN"] = {"nominal": "TRUE"}
            seeds[f"{head}:STATUS:FAULT"] = {"nominal": "FALSE"}
            seeds[f"{head}:STATUS:MOVING"] = {"nominal": "FALSE"}
            seeds[f"{head}:STATUS:READY"] = {"nominal": "TRUE"}
    return devices, channels, seeds


# --------------------------------------------------------------------------- insertion devices


def insertion_devices(sector: int) -> list[int]:
    """The numbers of the insertion devices in straight ``sector``, in beam order."""
    return [n for n, row in enumerate(INSERTION_DEVICES, 1) if row[0] == sector]


def id_name(number: int) -> str:
    """The device's short name: its kind and its number round the ring (``EPU7``)."""
    return f"{INSERTION_DEVICES[number - 1][1]}{number}"


def id_long_name(number: int) -> str:
    """The device's spoken name (``EPU 7``, ``Undulator 9``)."""
    return f"{KIND_NAME[INSERTION_DEVICES[number - 1][1]]} {number}"


def id_kind(number: int) -> str:
    sector, kind, period, *_ = INSERTION_DEVICES[number - 1]
    return f"{kind}{period}"


def is_epu(number: int) -> bool:
    return INSERTION_DEVICES[number - 1][3]


def _insertion_devices() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for n, (sector, kind, period, epu, gap, offset, grant) in enumerate(INSERTION_DEVICES, 1):
        name, spoken = id_name(n), id_long_name(n)
        device = f"SR/ID{n:02d}"
        head = f"SR:ID:{n:02d}"
        devices.append(
            _device(
                device,
                "InsertionDevice",
                f"{spoken} ({name}): {KIND_TEXT[kind]} with a {period} mm period in straight "
                f"{sector}. "
                "Its gap (and an elliptically polarising one's row offset) sets the photon "
                "energy and polarisation of the beamline it serves; the beamline holds the "
                "gap while control is granted to it, operations otherwise.",
                [f"ID{n:02d}", name, spoken, f"{kind}{period}", KIND_TEXT[kind]],
                f"{name} ({period} mm)",
                _place(sector),
            )
        )
        what = f"{SR} insertion device {name} ({kind}{period})"
        channels += [
            _channel(
                f"{head}:FF:ENABLE",
                device,
                f"{what} feed-forward enable status (read-only): the corrector feed-forward "
                "that cancels the gap's orbit and tune kicks is on.",
                signal="status",
                value_type="bool",
            ),
            _channel(
                f"{head}:GAP:RB",
                device,
                f"{what} gap readback (read-only): the measured magnetic gap in mm; the gap "
                "opens to about 200 mm.",
                signal="gap_readback",
                unit="mm",
                precision=2,
            ),
            _channel(
                f"{head}:GAP:SP",
                device,
                f"{what} gap setpoint (read-write): the commanded magnetic gap in mm.",
                signal="gap_setpoint",
                unit="mm",
                precision=2,
                pair=f"{head}:GAP:RB",
            ),
            _channel(
                f"{head}:STATUS:GRANT",
                device,
                f"{what} beamline control status (read-only): TRUE while the beamline holds "
                "control of the gap, FALSE while operations holds it.",
                signal="local_mode_status",
                value_type="bool",
            ),
            _channel(
                f"{head}:STATUS:MOVING",
                device,
                f"{what} moving status (read-only): the gap or the row offset is moving.",
                signal="ramping_status",
                value_type="bool",
            ),
        ]
        seeds[f"{head}:FF:ENABLE"] = {"nominal": "TRUE"}
        seeds[f"{head}:GAP:RB"] = {"nominal": gap}
        seeds[f"{head}:GAP:SP"] = {"nominal": gap}
        seeds[f"{head}:STATUS:GRANT"] = {"nominal": "TRUE" if grant else "FALSE"}
        seeds[f"{head}:STATUS:MOVING"] = {"nominal": "FALSE"}
        if epu:
            channels += [
                _channel(
                    f"{head}:OFFSET:RB",
                    device,
                    f"{what} row offset readback (read-only): the longitudinal shift of the "
                    "magnet rows in mm, from -25 to 25; it sets the polarisation.",
                    signal="offset_readback",
                    unit="mm",
                    precision=2,
                ),
                _channel(
                    f"{head}:OFFSET:SP",
                    device,
                    f"{what} row offset setpoint (read-write): the commanded row shift in mm.",
                    signal="offset_setpoint",
                    unit="mm",
                    precision=2,
                    pair=f"{head}:OFFSET:RB",
                ),
            ]
            seeds[f"{head}:OFFSET:RB"] = {"nominal": offset}
            seeds[f"{head}:OFFSET:SP"] = {"nominal": offset}
    return devices, channels, seeds


# --------------------------------------------------------------------------- shutters


def shutter_numbers(sector: int) -> list[int]:
    first = sum(SHUTTERS[s] for s in SECTORS if s < sector) + 1
    return list(range(first, first + SHUTTERS[sector]))


def shutter_ports(sector: int) -> list[tuple[int, str]]:
    """The sector's shutters in beam order: (number, port), the straight's ID ports first."""
    numbers = shutter_numbers(sector)
    ids = min(len(insertion_devices(sector)), len(numbers))
    return [(n, "insertion device" if i < ids else "bend") for i, n in enumerate(numbers)]


def _shutters() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for sector in SECTORS:
        for n, port in shutter_ports(sector):
            device = f"SR/SHUTTER{n:02d}"
            devices.append(
                _device(
                    device,
                    "PhotonShutter",
                    f"Front-end photon shutter of a {port} port in storage-ring sector {sector}.",
                    [f"SHUTTER{n:02d}", "photon shutter", "front-end shutter", f"{port} port"],
                    f"SHUTTER {n:02d}",
                    _place(sector),
                )
            )
            address = f"SR:FE:SHUTTER:{n:02d}:OPEN:RB"
            channels.append(
                _channel(
                    address,
                    device,
                    f"{SR} front-end shutter {n:02d} open status (read-only): the shutter of "
                    f"this {port} port in sector {sector} is open and the beamline takes "
                    "photons.",
                    signal="status",
                    value_type="bool",
                )
            )
            closed = n % CLOSED_SHUTTER_EVERY == 0
            seeds[address] = {"nominal": "FALSE" if closed else "TRUE"}
    return devices, channels, seeds


# --------------------------------------------------------------------------- thermocouples


def thermocouples(sector: int) -> list[str]:
    """The names of straight ``sector``'s absorber thermocouples: upper row, then lower."""
    if sector == TC_SINGLE:
        return [f"{sector:02d}U01"]
    if sector not in TC_STRAIGHTS:
        return []
    return [f"{sector:02d}{row}{i:02d}" for row in TC_ROWS for i in range(1, TC_PER_ROW + 1)]


def _tc_temperature(name: str) -> float:
    """A thermocouple's nominal: a heat bump along the absorber, peaking near its middle."""
    sector, row, i = int(name[:2]), name[2], int(name[3:])
    peak = 14.0 + 3.0 * math.sin(1.3 * sector)
    bump = peak * math.exp(-(((i - 6) / 2.4) ** 2))
    return round(27.0 + bump - (3.5 if row == "L" else 0.0) + 0.3 * math.sin(i + sector), 2)


def _thermocouples() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for sector in SECTORS:
        for name in thermocouples(sector):
            row = "upper" if name[2] == "U" else "lower"
            device = f"SR/TC{name}"
            head = f"SR:VAC:TC:{name}"
            devices.append(
                _device(
                    device,
                    "TemperatureSensor",
                    f"Absorber thermocouple {int(name[3:])} of the {row} row in straight "
                    f"{sector}: watches the photon absorber for mis-steered light.",
                    [f"TC{name}", "absorber thermocouple", "thermocouple"],
                    f"TC {name}",
                    _place(sector),
                )
            )
            what = f"{SR} absorber thermocouple {name}"
            channels += [
                _channel(
                    f"{head}:LIMIT:RB",
                    device,
                    f"{what} temperature limit (read-only): the absorber temperature, in degC, "
                    "above which the over-temperature flag trips and beam is dumped.",
                    signal="temperature_limit_readback",
                    unit="degC",
                    precision=1,
                ),
                _channel(
                    f"{head}:STATUS:OVERTEMP",
                    device,
                    f"{what} over-temperature status (read-only): the temperature is above "
                    "its limit.",
                    signal="over_temperature_status",
                    value_type="bool",
                ),
                _channel(
                    f"{head}:TEMPERATURE:RB",
                    device,
                    f"{what} temperature readback (read-only): the absorber temperature in degC.",
                    signal="temperature_readback",
                    unit="degC",
                    precision=1,
                ),
            ]
            seeds[f"{head}:LIMIT:RB"] = {"nominal": TC_LIMIT_C}
            seeds[f"{head}:STATUS:OVERTEMP"] = {"nominal": "FALSE"}
            seeds[f"{head}:TEMPERATURE:RB"] = {"nominal": _tc_temperature(name)}
    return devices, channels, seeds


# --------------------------------------------------------------------------- scrapers


def scraper_numbers(sector: int) -> list[int]:
    return [n for n, row in enumerate(SCRAPERS, 1) if row[1] == sector]


def scraper_family(number: int) -> str:
    return SCRAPERS[number - 1][0]


def scraper_position(number: int) -> int:
    return SCRAPERS[number - 1][2]


def _scrapers() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for n, (family, sector, index, insertion) in enumerate(SCRAPERS, 1):
        device = f"SR/SCRAPER{n:02d}"
        head = f"SR:DIAG:SCRAPER:{n:02d}"
        devices.append(
            _device(
                device,
                "Aperture",
                f"{SCRAPER_TEXT[family]}, at position {index} of storage-ring sector {sector}; "
                "the depth at which it starts to intercept beam measures the halo.",
                [f"SCRAPER{n:02d}", f"{family} scraper", "scraper", SCRAPER_WHERE[index]],
                f"SCRAPER {n:02d} ({family})",
                _place(sector),
            )
        )
        what = f"{SR} {family} scraper {n:02d}"
        channels += [
            _channel(
                f"{head}:POSITION:RB",
                device,
                f"{what} position readback (read-only): the blade's insertion into the "
                "aperture in mm, 0 fully retracted.",
                signal="blade_position_readback",
                unit="mm",
                precision=2,
            ),
            _channel(
                f"{head}:POSITION:SP",
                device,
                f"{what} position setpoint (read-write): the commanded blade insertion in mm, "
                "from 0 to 6.4.",
                signal="blade_position_setpoint",
                unit="mm",
                precision=2,
                pair=f"{head}:POSITION:RB",
            ),
            _channel(
                f"{head}:STATUS:HOMED",
                device,
                f"{what} homed status (read-only): the axis has found its home switch since "
                "power-up.",
                signal="status",
                value_type="bool",
            ),
            _channel(
                f"{head}:STATUS:LIMIT",
                device,
                f"{what} limit status (read-only): a travel limit switch is active.",
                signal="interlock_status",
                value_type="bool",
            ),
            _channel(
                f"{head}:STATUS:MOVING",
                device,
                f"{what} moving status (read-only): the blade is moving.",
                signal="ramping_status",
                value_type="bool",
            ),
        ]
        seeds[f"{head}:POSITION:RB"] = {"nominal": insertion}
        seeds[f"{head}:POSITION:SP"] = {"nominal": insertion}
        seeds[f"{head}:STATUS:HOMED"] = {"nominal": "TRUE"}
        seeds[f"{head}:STATUS:LIMIT"] = {"nominal": "FALSE"}
        seeds[f"{head}:STATUS:MOVING"] = {"nominal": "FALSE"}
    return devices, channels, seeds


# --------------------------------------------------------------------------- tunnel


def _tunnel() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for sector in AIR_SECTORS:
        device = f"SR/AIR{sector:02d}"
        address = f"SR:ENV:AIR:{sector:02d}:TEMPERATURE:RB"
        devices.append(
            _device(
                device,
                "TemperatureSensor",
                f"Tunnel air-temperature sensor of storage-ring sector {sector}.",
                [f"AIR{sector:02d}", "tunnel air temperature", "tunnel temperature"],
                f"AIR {sector:02d}",
                _place(sector),
            )
        )
        channels.append(
            _channel(
                address,
                device,
                f"{SR} tunnel air temperature, sector {sector:02d} (read-only): the "
                "air-handling zone temperature in degC; the magnets' and the vacuum "
                "chamber's thermal stability follows it.",
                signal="temperature_readback",
                unit="degC",
                precision=1,
            )
        )
        seeds[address] = {"nominal": round(24.2 + 0.6 * math.sin(0.9 * sector), 2)}
    for sector in AHU_SECTORS:
        device = f"SR/AHU{sector:02d}"
        address = f"SR:ENV:AHU:{sector:02d}:VALVE:RB"
        devices.append(
            _device(
                device,
                "AirHandler",
                f"Air-handling unit of storage-ring sector {sector}.",
                [f"AHU{sector:02d}", "air handler", "air handling unit"],
                f"AHU {sector:02d}",
                _place(sector),
            )
        )
        channels.append(
            _channel(
                address,
                device,
                f"{SR} air-handler {sector:02d} chilled-water valve position (read-only): "
                "how far the valve is open, in percent; it opens as the tunnel warms.",
                signal="valve_position_readback",
                unit="%",
                precision=0,
            )
        )
        seeds[address] = {"nominal": round(45.0 + 12.0 * math.sin(1.1 * sector), 1)}
    return devices, channels, seeds


# --------------------------------------------------------------------------- harmonic cavities


def _harmonic_cavities() -> tuple[list, list, dict]:
    devices, channels, seeds = [], [], {}
    for n in range(1, HARMONIC_CAVITIES + 1):
        device = f"SR/HARMONIC{n:02d}"
        head = f"SR:RF:HARMONIC:{n:02d}"
        devices.append(
            _device(
                device,
                "HarmonicCavity",
                f"Passive third-harmonic cavity {n} in straight {HARMONIC_STRAIGHT}: the "
                "stored beam drives it, and its tuner sets how much it lengthens the "
                "bunches.",
                [f"HARMONIC{n:02d}", "harmonic cavity", "third harmonic cavity"],
                f"HARMONIC {n:02d}",
                _place(HARMONIC_STRAIGHT),
            )
        )
        what = f"{SR} harmonic cavity {n:02d}"
        channels += [
            _channel(
                f"{head}:FLOW:RB",
                device,
                f"{what} cooling-water flow readback (read-only), in L/min.",
                signal="coolant_flow_readback",
                unit="L/min",
                precision=1,
            ),
            _channel(
                f"{head}:POWER:RB",
                device,
                f"{what} beam-induced power readback (read-only), in kW.",
                signal="power_readback",
                unit="kW",
                precision=1,
            ),
            _channel(
                f"{head}:STATUS:LOOP",
                device,
                f"{what} tuning loop status (read-only): the tuner loop is closed and holds "
                "the cavity's detuning.",
                signal="status",
                value_type="bool",
            ),
            _channel(
                f"{head}:STATUS:TEMP_OK",
                device,
                f"{what} temperature OK status (read-only): the cavity body is inside its "
                "temperature band.",
                signal="thermal_status",
                value_type="bool",
            ),
            _channel(
                f"{head}:TEMPERATURE:RB",
                device,
                f"{what} body temperature readback (read-only), in degC.",
                signal="temperature_readback",
                unit="degC",
                precision=1,
            ),
            _channel(
                f"{head}:TUNER:RB",
                device,
                f"{what} tuner position readback (read-only), in mm.",
                signal="tuner_position_readback",
                unit="mm",
                precision=2,
            ),
            _channel(
                f"{head}:TUNER:SP",
                device,
                f"{what} tuner position setpoint (read-write), in mm.",
                signal="tuner_position_setpoint",
                unit="mm",
                precision=2,
                pair=f"{head}:TUNER:RB",
            ),
        ]
        tuner = round(12.4 + 0.7 * n, 2)
        seeds[f"{head}:FLOW:RB"] = {"nominal": round(18.5 + 0.6 * n, 1)}
        seeds[f"{head}:POWER:RB"] = {"nominal": round(4.2 + 0.4 * n, 1)}
        seeds[f"{head}:STATUS:LOOP"] = {"nominal": "TRUE"}
        seeds[f"{head}:STATUS:TEMP_OK"] = {"nominal": "TRUE"}
        seeds[f"{head}:TEMPERATURE:RB"] = {"nominal": round(36.0 + 0.8 * n, 1)}
        seeds[f"{head}:TUNER:RB"] = {"nominal": tuner}
        seeds[f"{head}:TUNER:SP"] = {"nominal": tuner}
    return devices, channels, seeds


# --------------------------------------------------------------------------- septa


def _septa() -> tuple[list, list, dict]:
    channels, seeds = [], {}
    bits = ", ".join(f"bit {i} {text}" for i, text in enumerate(INTERLOCK_BITS))
    for n, name in ((1, "SEK"), (2, "SEN")):
        address = f"SR:MAG:SEPTUM:{n:02d}:INTERLOCK:WORD"
        channels.append(
            _channel(
                address,
                f"SR/SEPTUM{n:02d}",
                f"{SR} injection septum {name} interlock word (read-only): one bit per "
                f"permit, set while it is satisfied: {bits}. {INTERLOCK_HEALTHY} when every "
                "permit is met.",
                signal="interlock_status",
                value_type="int",
            )
        )
        seeds[address] = {"nominal": INTERLOCK_HEALTHY}
    return [], channels, seeds


# --------------------------------------------------------------------------- assembly


def radiation_places() -> dict[str, str]:
    """The radiation monitors' places: monitor ``n`` in sector ``RADIATION_SECTORS[n-1]``."""
    out = {}
    for kind in ("GAMMA", "NEUTRON"):
        for n, sector in enumerate(RADIATION_SECTORS, 1):
            out[f"SR/{kind}{n:02d}"] = _place(sector)
    return out


#: The radiation monitors' gamma dose rate outside the shield wall, mrem/hr, by
#: sector: highest at injection; the neutron monitors read a fraction of it.
GAMMA_MREM_PER_HR = {1: 0.35, 4: 0.08, 7: 0.06, 10: 0.07}
NEUTRON_FRACTION = 0.4


def radiation_seeds() -> dict[str, dict[str, Any]]:
    """Plausible dose rates for the radiation monitors: a fraction of a mrem/hr."""
    out = {}
    for n, sector in enumerate(RADIATION_SECTORS, 1):
        gamma = GAMMA_MREM_PER_HR[sector]
        for kind, rate in (("GAMMA", gamma), ("NEUTRON", round(gamma * NEUTRON_FRACTION, 3))):
            for tail in ("INST", "AVG_1MIN", "AVG_1HR"):
                out[f"SR:DIAG:{kind}:{n:02d}:DOSE_RATE:{tail}"] = {"nominal": rate}
    return out


#: The superbends: the middle dipole of arcs 4, 8 and 12.
SUPERBEND_SECTORS = (4, 8, 12)
DIPOLES_PER_SECTOR = 3


def superbends(committed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The committed dipoles that are superbends, named and described as such."""
    numbers = {(s - 1) * DIPOLES_PER_SECTOR + 2: s for s in SUPERBEND_SECTORS}
    out = []
    for device in committed:
        m = re.fullmatch(r"SR/DIPOLE(\d+)", device["id"])
        if m and int(m.group(1)) in numbers:
            sector = numbers[int(m.group(1))]
            out.append(
                {
                    **device,
                    "description": f"Superbend: the superconducting dipole in the middle of arc "
                    f"{sector}, whose high field serves hard X-ray bend ports.",
                    "names": sorted({*device.get("names", []), "superbend"}),
                }
            )
    return out


FAMILIES = (
    _ion_pumps,
    _gauges,
    _valves,
    _insertion_devices,
    _shutters,
    _thermocouples,
    _scrapers,
    _tunnel,
    _harmonic_cavities,
    _septa,
)


def emitted() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Every family's devices, channels and seed entries."""
    devices, channels, seeds = [], [], {}
    for family in FAMILIES:
        d, c, s = family()
        devices += d
        channels += c
        seeds.update(s)
    return devices, channels, seeds


#: Groups the peripherals join, and the new groups.
GROUP_TEXT = {
    "SR/ID": "Insertion Devices (ID): undulators and wigglers in the storage-ring straights; "
    "each one's gap sets the photon energy of its beamline.",
    "SR/FE": "Front Ends (FE): the photon shutters of the beamline ports.",
    "SR/TC": "Absorber Thermocouples (TC): temperatures of the photon absorbers under the "
    "straights.",
    "SR/ENV": "Tunnel Environment (ENV): tunnel air temperatures and air-handler valves.",
    "SR/SCRAPER": "Scrapers (SCRAPER): insertable blades that measure the beam halo.",
    "SR/HC": "Harmonic Cavities (HC): passive third-harmonic cavities in straight 2 that "
    "lengthen the bunches.",
}


def groups(devices: list[dict[str, Any]], committed: list[dict[str, Any]]) -> list[dict]:
    """``SR/VAC``, ``SR/RF`` and ``SR/DIAG`` with their new members, and the new groups."""
    by_id = {g["id"]: g for g in committed}
    members: dict[str, list[str]] = {gid: [] for gid in (*GROUP_TEXT, "SR/VAC")}
    for device in devices:
        cls = device["class"]
        target = {
            "Pump": "SR/VAC",
            "Gauge": "SR/VAC",
            "Valve": "SR/VAC",
            "InsertionDevice": "SR/ID",
            "PhotonShutter": "SR/FE",
            "Aperture": "SR/SCRAPER",
            "AirHandler": "SR/ENV",
            "HarmonicCavity": "SR/HC",
        }.get(cls)
        if cls == "TemperatureSensor":
            target = "SR/ENV" if device["id"].startswith("SR/AIR") else "SR/TC"
        if target is None:
            raise ValueError(f"no group for {device['id']} of class {cls}")
        members[target].append(device["id"])
    out = []
    for gid, added in members.items():
        if gid in by_id:
            group = dict(by_id[gid])
            group["members"] = sorted(set(group.get("members", [])) | set(added))
        else:
            group = {"id": gid, "description": GROUP_TEXT[gid], "members": sorted(added)}
        out.append(group)
    return out


def write_sources(tree: Path) -> None:
    """Merge the peripherals into the facility tree ``tree``.

    The devices, channels and groups go into ``records/``, the facility-added classes
    into ``classes.yaml`` (written whole), and each channel's ``nominal`` (and the
    valves' closed-switch ``gate``) into ``seeds.yaml``; the radiation monitors move to
    their sectors. Every other record keeps its text.
    """
    import _injector
    import yaml

    records = tree / "records"
    devices, channels, seeds = emitted()
    seeds.update(radiation_seeds())
    committed_devices = _line._load(records / "devices.yaml")
    moved = radiation_places()
    placed = [{**d, "place": moved[d["id"]]} for d in committed_devices if d["id"] in moved]
    placed += superbends(committed_devices)
    _line._merge(records / "devices.yaml", devices + placed, "id", ordered=True)
    tagged = {c["id"]: c["tags"] for c in _line._load(records / "channels.yaml") if "tags" in c}
    channels = [{**c, "tags": tagged[c["id"]]} if c["id"] in tagged else c for c in channels]
    _line._merge(records / "channels.yaml", channels, "id", ordered=True)
    _line._merge(
        records / "groups.yaml",
        groups(devices, _line._load(records / "groups.yaml")),
        "id",
        ordered=True,
    )
    (tree / "classes.yaml").write_text(_line._dump(list(CLASSES)), encoding="utf-8")
    committed = _injector.reseed.load(tree / "seeds.yaml")
    for address, entry in seeds.items():
        kept = {k: v for k, v in committed.get(address, {}).items() if k not in entry}
        committed[address] = {**kept, **entry}
    document = {
        address: {
            key: committed[address][key]
            for key in _injector.reseed.KEY_ORDER
            if key in committed[address]
        }
        for address in sorted(committed)
        if committed[address]
    }
    (tree / "seeds.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
