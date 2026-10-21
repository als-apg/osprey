"""The injector's element table: every channel of the gun, linac, transfer lines and
injection septa, and the panel binding each one replaces.

``injector_elements.tsv`` beside this module is the table. Each row is one
(channel, panel binding) pair in five tab-separated columns:

``legacy_pv``
    The name a GTLView/BTSView binding site reads or writes today, or blank for
    a channel no panel names (a BPM's ``SUM``, a screen's lamp with no panel
    binding, the removed BTS corrector 06). A legacy name appears with each
    role at most once, so the panel re-map is keyed by (legacy name, role); a
    channel bound by several legacy names (a magnet's status read both from its
    supply and from an interlock input) has one row per name.
``role``
    A key of ``ROLE_TEMPLATES``, which states the channel's device class,
    signal, value type, unit, writability and tolerance, and the panel site
    (setpoint, readback, status, position, readout) that resolves to it.
``demo_name``
    The channel's address in the demo grammar
    ``{SEC}:{SYS}:{FAMILY}:{NN}:{FIELD}:{SUB}``, every token from the alphabets
    below.
``deck_element``
    ``<SEC>:<FAMILY>#<n>``: the ``n``-th occurrence of the lattice-array token
    ``FAMILY`` (the MML variable name, not the element's ``FamName``) in that
    section's array, ``LTB`` being ``gtblattice.m`` LTB1 followed by LTB2 and
    ``BTS`` being ``btslattice.m`` BTS. It is set on the rows the deck binds:
    the magnet setpoints and readbacks and the BPM positions. A row whose
    element is in ``FIXED_ELEMENTS`` (the four septa) names its element but is
    not wired: the texture serves it. Every other row is blank.
``nominal``
    The channel's held value: a number, or ``TRUE``/``FALSE`` for a bool; blank
    for the waveform, whose trace is generated.

Rows run in beam order within each section (gun to linac to booster to storage
injection); a line starting with ``#`` is a section banner.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

#: The checked-in element table.
TABLE = Path(__file__).resolve().with_name("injector_elements.tsv")

#: The table's columns, in order.
COLUMNS = ("legacy_pv", "role", "demo_name", "deck_element", "nominal")

# --- the demo grammar --------------------------------------------------------------

#: Sections, in beam order.
SECTIONS = ("GTL", "LN", "LTB", "BR", "BTS", "SR")

#: Systems.
SYSTEMS = ("MAG", "DIAG", "RF", "GUN")

#: Device families.
FAMILIES = (
    "HCM",
    "VCM",
    "QF",
    "QD",
    "DIPOLE",
    "SOL",
    "BC",
    "SEPTUM",
    "KICKER",
    "DIPOLE-RAMP",
    "QF-RAMP",
    "QD-RAMP",
    "BPM",
    "TV",
    "PADDLES",
    "ICT",
    "DCCT",
    "EGUN",
    "SHB",
    "MASTER",
    "ACC",
    "BUNCHER",
    "WATER",
)

#: Fields.
FIELDS = (
    "CURRENT",
    "STATUS",
    "POSITION",
    "SUM",
    "PADDLE",
    "LAMP",
    "COUNT",
    "CHARGE",
    "VOLTAGE",
    "HV_ON",
    "HEATER",
    "HEATER_ON",
    "AUX_ON",
    "PHASE",
    "PHASE_ON",
    "BIAS",
    "TEMPERATURE",
    "GAIN",
    "OFFSET",
)

#: Subfields. ``SP`` is the only writable one and pairs with the ``RB`` of the
#: same field.
SUBFIELDS = ("SP", "RB", "ON", "READY", "VALID", "X", "Y", "INTERLOCK", "INSERTED", "WAVEFORM")

#: The six-token grammar; the device number is two digits from ``01``.
DEMO_NAME = re.compile(
    r"^(?P<sec>[A-Z]+):(?P<sys>[A-Z]+):(?P<family>[A-Z][A-Z-]*):(?P<nn>(?!00)\d{2}):"
    r"(?P<field>[A-Z][A-Z_]*):(?P<sub>[A-Z]+)$"
)

# --- the deck ----------------------------------------------------------------------

#: The sections whose lattice arrays the deck elements index.
DECK_SECTIONS = ("LTB", "BTS")

#: An occurrence-indexed deck element.
DECK_ELEMENT = re.compile(r"^(?P<sec>LTB|BTS):(?P<family>[A-Za-z][A-Za-z0-9_]*)#(?P<n>[1-9]\d*)$")

#: Every deck magnet no row wires, held at its MML strength. The four septa are
#: bound to their texture rows by name (SR SEK -> SIK, SR SEN -> SIN, BR SEN ->
#: SEN, BR SEK -> SEK) and stay unwired. ``BX`` is an unpowered zero-angle bend;
#: ``Q13`` shares the ``Q11`` supply (LTB Q1.1) and is held, not driven;
#: ``BR1QD1``, ``BRKI`` and ``BR1QF2`` are the booster's own magnets at the end
#: of the line; ``SRBMP3``/``SRBMP4`` are storage-ring bumps.
FIXED_ELEMENTS = (
    "LTB:BX#1",
    "LTB:Q13#1",
    "LTB:BR1QD1#1",
    "LTB:BRKI#1",
    "LTB:BR1QF2#1",
    "BTS:SEN#1",
    "BTS:SEK#1",
    "BTS:SIK#1",
    "BTS:SIN#1",
    "BTS:SRBMP3#1",
    "BTS:SRBMP4#1",
)

# --- the roles ---------------------------------------------------------------------

#: The panel sites a role answers: a control (``sp``), a readback (``rb``), a
#: status colour or lamp (``status``), a BPM plot (``position``), or a plain
#: read (``readout``).
SITES = ("sp", "rb", "status", "position", "readout")


@dataclass(frozen=True)
class RoleTemplate:
    """What a role states about every channel that carries it."""

    #: The vocabulary class of the channel's device.
    device_class: str
    #: The vocabulary signal role of the channel.
    signal: str
    #: The panel site that resolves to the channel.
    site: str
    #: ``float``, ``bool``, ``int`` or ``waveform``.
    value_type: str = "float"
    #: Engineering unit, or ``None`` for a unitless value.
    unit: str | None = None
    #: Whether the channel is a writable setpoint.
    writable: bool = False
    #: Absolute setpoint tolerance, or ``None``.
    tolerance: float | None = None
    #: Sample count of a waveform, or ``None``.
    shape: tuple[int, ...] | None = None


def _magnet(device_class: str, tolerance: float, unit: str = "A") -> dict[str, RoleTemplate]:
    """Setpoint, readback and on-status of a current-driven magnet."""
    return {
        "sp": RoleTemplate(
            device_class, "current_setpoint", "sp", unit=unit, writable=True, tolerance=tolerance
        ),
        "rb": RoleTemplate(device_class, "current_readback", "rb", unit=unit),
        "on": RoleTemplate(device_class, "power_on_status", "status", value_type="bool"),
    }


def _pair(
    device_class: str, setpoint: str, readback: str, unit: str | None, tolerance: float | None
) -> dict[str, RoleTemplate]:
    """A writable setpoint and its readback."""
    value_type = "float" if unit is not None or tolerance is not None else "bool"
    return {
        "sp": RoleTemplate(
            device_class,
            setpoint,
            "sp",
            value_type=value_type,
            unit=unit,
            writable=True,
            tolerance=tolerance,
        ),
        "rb": RoleTemplate(device_class, readback, "rb", value_type=value_type, unit=unit),
    }


def _switch(device_class: str) -> dict[str, RoleTemplate]:
    """A writable on/off command and its state, read as a status."""
    return {
        "sp": RoleTemplate(
            device_class, "power_on_command", "sp", value_type="bool", writable=True
        ),
        "rb": RoleTemplate(device_class, "power_on_status", "status", value_type="bool"),
    }


def _templates() -> dict[str, RoleTemplate]:
    roles: dict[str, RoleTemplate] = {}

    def add(prefix: str, parts: dict[str, RoleTemplate]) -> None:
        for suffix, template in parts.items():
            roles[f"{prefix}_{suffix}"] = template

    for prefix, device_class, tolerance in (
        ("hcm", "HCorrector", 0.02),
        ("vcm", "VCorrector", 0.02),
        ("quad", "Quadrupole", 0.05),
        ("dipole", "Dipole", 0.05),
        ("solenoid", "Solenoid", 0.05),
        ("bucking", "BuckingCoil", 0.02),
        ("septum", "Septum", 1.0),
    ):
        add(prefix, _magnet(device_class, tolerance))
    # The frozen BTS magnets also report supply-ready.
    for prefix, device_class in (
        ("hcm", "HCorrector"),
        ("vcm", "VCorrector"),
        ("quad", "Quadrupole"),
    ):
        roles[f"{prefix}_ready"] = RoleTemplate(
            device_class, "ready_status", "status", value_type="bool"
        )
    roles["septum_waveform"] = RoleTemplate(
        "Septum", "pulse_waveform", "readout", value_type="waveform", unit="A", shape=(500,)
    )
    add("kicker", _pair("Kicker", "voltage_setpoint", "voltage_readback", "kV", 0.05))
    roles["kicker_on"] = RoleTemplate("Kicker", "power_on_status", "status", value_type="bool")
    for prefix, device_class in (("dipole_ramp", "Dipole"), ("quad_ramp", "Quadrupole")):
        add(
            f"{prefix}_gain",
            _pair(device_class, "ramp_gain_setpoint", "ramp_gain_readback", None, 0.001),
        )
        add(
            f"{prefix}_offset",
            _pair(device_class, "ramp_offset_setpoint", "ramp_offset_readback", "A", 0.05),
        )
        roles[f"{prefix}_on"] = RoleTemplate(
            device_class, "power_on_status", "status", value_type="bool"
        )

    # Beam position monitors: millimetres on every injector BPM.
    roles["bpm_x"] = RoleTemplate(
        "BeamPositionMonitor", "position_x_readback", "position", unit="mm"
    )
    roles["bpm_y"] = RoleTemplate(
        "BeamPositionMonitor", "position_y_readback", "position", unit="mm"
    )
    roles["bpm_sum"] = RoleTemplate("BeamPositionMonitor", "bpm_sum_readback", "readout")
    roles["bpm_valid"] = RoleTemplate("BeamPositionMonitor", "status", "status", value_type="bool")

    # Screens: the paddle is a writable bool with its position readback; the
    # lamp lights when the paddle is in.
    roles["tv_paddle_sp"] = RoleTemplate(
        "ScreenDiagnostic", "screen_in_command", "sp", value_type="bool", writable=True
    )
    roles["tv_paddle_rb"] = RoleTemplate(
        "ScreenDiagnostic", "screen_in_status", "rb", value_type="bool"
    )
    roles["tv_lamp_rb"] = RoleTemplate("ScreenDiagnostic", "lamp_status", "rb", value_type="bool")
    roles["paddle_count"] = RoleTemplate("ScreenDiagnostic", "status", "status", value_type="int")

    # Beam intensity: the per-section charge proxy and the stored-beam current.
    roles["ict_charge"] = RoleTemplate(
        "BeamCurrentMonitor", "beam_intensity_readback", "readout", unit="nC"
    )
    roles["dcct_current"] = RoleTemplate(
        "BeamCurrentMonitor", "beam_intensity_readback", "readout", unit="A"
    )

    # The electron gun.
    add(
        "gun_voltage",
        _pair("ElectronGun", "gun_voltage_setpoint", "gun_voltage_readback", "kV", 0.1),
    )
    add("gun_hv_on", _switch("ElectronGun"))
    add("gun_heater", _pair("ElectronGun", "current_setpoint", "current_readback", "A", 0.01))
    add("gun_heater_on", _switch("ElectronGun"))
    add("gun_aux_on", _switch("ElectronGun"))
    add("gun_phase", _pair("ElectronGun", "phase_setpoint", "phase_readback", "deg", 0.1))
    add("gun_bias", _pair("ElectronGun", "voltage_setpoint", "voltage_readback", "V", 0.1))

    # Sub-harmonic bunchers.
    add("shb_phase", _pair("SubHarmonicBuncher", "phase_setpoint", "phase_readback", "deg", 0.1))
    add("shb_phase_on", _switch("SubHarmonicBuncher"))
    add("shb_voltage", _pair("SubHarmonicBuncher", "rf_hv_setpoint", "rf_hv_readback", "kV", 0.05))
    add("shb_hv_on", _switch("SubHarmonicBuncher"))

    # The linac's RF stand-ins.
    add("master_phase", _pair("RadioFrequency", "phase_setpoint", "phase_readback", "deg", 0.1))
    roles["master_on"] = RoleTemplate(
        "RadioFrequency", "power_on_status", "status", value_type="bool"
    )
    add("acc_phase", _pair("TravellingWaveCavity", "phase_setpoint", "phase_readback", "deg", 0.1))
    roles["acc_on"] = RoleTemplate(
        "TravellingWaveCavity", "power_on_status", "status", value_type="bool"
    )
    roles["acc_temperature"] = RoleTemplate(
        "TravellingWaveCavity", "temperature_readback", "readout", unit="degC"
    )
    roles["buncher_temperature"] = RoleTemplate(
        "BunchingCavity", "temperature_readback", "readout", unit="degC"
    )
    add(
        "water_temperature",
        _pair("RadioFrequency", "temperature_setpoint", "temperature_readback", "degC", 0.1),
    )
    roles["water_heater_on"] = RoleTemplate(
        "RadioFrequency", "power_on_status", "status", value_type="bool"
    )
    roles["water_interlock"] = RoleTemplate(
        "RadioFrequency", "interlock_status", "status", value_type="bool"
    )
    return roles


#: Every role of the table.
ROLE_TEMPLATES: dict[str, RoleTemplate] = _templates()

# --- the rows ----------------------------------------------------------------------


@dataclass(frozen=True)
class ElementRow:
    """One row of the element table; blank cells read as ``None``."""

    legacy_pv: str | None
    role: str
    demo_name: str
    deck_element: str | None
    nominal: str | None

    @property
    def template(self) -> RoleTemplate:
        """The row's role template."""
        return ROLE_TEMPLATES[self.role]

    @property
    def section(self) -> str:
        """The demo name's section token."""
        return self.demo_name.split(":", 1)[0]

    @property
    def wired(self) -> bool:
        """Whether the deck drives this channel: bound, and not a fixed element."""
        return self.deck_element is not None and self.deck_element not in FIXED_ELEMENTS

    def nominal_value(self) -> float | int | str | None:
        """The nominal typed by the role: a number, a bool label, or ``None``."""
        if self.nominal is None:
            return None
        value_type = self.template.value_type
        if value_type == "int":
            return int(self.nominal)
        if value_type == "bool":
            return self.nominal
        return float(self.nominal)


def load_elements(path: Path = TABLE) -> list[ElementRow]:
    """Read the element table, skipping ``#`` banner lines."""
    with path.open(newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    reader = csv.reader(lines, delimiter="\t")
    header = next(reader)
    if tuple(header) != COLUMNS:
        raise ValueError(f"{path}: header is {header}, expected {list(COLUMNS)}")
    rows = []
    for number, cells in enumerate(reader, start=2):
        if len(cells) != len(COLUMNS):
            raise ValueError(f"{path}: data row {number} has {len(cells)} cells")
        legacy, role, name, deck, nominal = (cell.strip() or None for cell in cells)
        if role is None or name is None:
            raise ValueError(f"{path}: data row {number} needs a role and a demo_name")
        rows.append(ElementRow(legacy, role, name, deck, nominal))
    return rows
