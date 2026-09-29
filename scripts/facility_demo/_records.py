"""The demo's authored records, derived from its hand-written sources.

Every record comes from one of the demo's committed sources:

* the demo TTL -- devices (id, class, bare name), the channels each device
  binds (address, role, description) and the signal each binding reads or
  writes;
* the tier-3 in_context database -- each channel's common names;
* the tier-1 in_context database -- the addresses tagged ``in_context``;
* the tier-3 hierarchical database -- each channel's value type, and the
  machine, system, family, field and subfield descriptions;
* the tier-3 middle-layer database -- the machine and family labels, and each
  device's common name, ``DeviceList`` and ``ElementList`` entries;
* the virtual accelerator's bindings -- which devices a model wires, so only
  the others carry a hand place;
* the committed deck -- the sector holding the RF cavity element;
* the fingerprint additions golden -- the instruments the demo serves beyond
  the TTL's addresses, each on the deck machine.

The records are plain dicts in the shape ``data/facility/`` stores them,
sorted by id (places in tree order), so writing them is deterministic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any


def _sibling(name: str) -> ModuleType:
    """``<name>.py`` beside this file, under a name no other module takes."""
    qualified = f"facility_demo_{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, Path(__file__).with_name(f"{name}.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


fp = _sibling("fingerprint")

MIDDLE_LAYER = f"{fp.TIER3_DIR}/middle_layer.json"
VA_BINDINGS = f"{fp._CA_DATA}/simulation/va_bindings.json"
SR_DECK = f"{fp._CA_DATA}/facility/decks/SR.json"
ADDITIONS = "tests/facility/golden/demo_fingerprint_additions.json"

#: The machines of the demo, in tree order.
MACHINES = ("SR", "BR", "BTS")

#: The machine whose deck places devices, and the model that deck belongs to.
DECK_MACHINE = "SR"

#: The number of sectors of the deck machine; sector n spans [SECTn, SECTn+1).
SECTORS = 12

#: Families of the deck machine with one device per sector, the sector being
#: the first entry of the device's ``DeviceList``.
SECTOR_INDEXED_FAMILIES = frozenset({"GAUGE", "VALVE"})

#: Families of the deck machine that sit in the sector holding the deck's RF
#: cavity element.
RF_CAVITY_FAMILIES = frozenset({"CAVITY"})

#: The deck's class for an RF cavity element.
RF_CAVITY_CLASS = "RFCavity"

#: The tag marking a channel the in_context channel finder lists.
IN_CONTEXT_TAG = "in_context"

#: The vocabulary signal role each TTL signal means; every TTL signal has one.
SIGNAL_ROLE: dict[str, str] = {
    "bpm_golden_x": "position_x_golden_readback",
    "bpm_golden_y": "position_y_golden_readback",
    "bpm_offset_x": "position_offset",
    "bpm_offset_y": "position_offset",
    "bpm_position_x": "position_x_readback",
    "bpm_position_y": "position_y_readback",
    "bpm_status_connected": "status",
    "bpm_status_valid": "status",
    "cavity_frequency_rb": "frequency_readback",
    "cavity_frequency_sp": "frequency_setpoint",
    "cavity_power_fwd": "rf_forward_power_readback",
    "cavity_power_net": "power_readback",
    "cavity_power_rev": "rf_reflected_power_readback",
    "cavity_status_fault": "fault_status",
    "cavity_status_interlock": "interlock_status",
    "cavity_status_ready": "ready_status",
    "cavity_temperature_rb": "temperature_readback",
    "cavity_tuner_rb": "tuner_position_readback",
    "cavity_tuner_sp": "tuner_position_setpoint",
    "cavity_voltage_golden": "voltage_golden_readback",
    "cavity_voltage_rb": "voltage_readback",
    "cavity_voltage_sp": "voltage_setpoint",
    "dcct_current_rb": "beam_intensity_readback",
    "dcct_status_valid": "status",
    "dipole_current_golden": "current_golden_readback",
    "dipole_current_rb": "current_readback",
    "dipole_current_sp": "current_setpoint",
    "dipole_status_fault": "fault_status",
    "dipole_status_on": "power_on_status",
    "dipole_status_ready": "ready_status",
    "gamma_dose_rate_avg_1hr": "dose_rate_readback",
    "gamma_dose_rate_avg_1min": "dose_rate_readback",
    "gamma_dose_rate_inst": "dose_rate_readback",
    "gamma_status_alarm": "status",
    "gamma_status_valid": "status",
    "gauge_pressure_rb": "vacuum_readback",
    "gauge_status_on": "power_on_status",
    "gauge_status_valid": "status",
    "hcm_current_golden": "current_golden_readback",
    "hcm_current_rb": "current_readback",
    "hcm_current_sp": "current_setpoint",
    "hcm_status_on": "power_on_status",
    "hcm_status_ready": "ready_status",
    "ion_pump_current_rb": "current_readback",
    "ion_pump_pressure_rb": "vacuum_readback",
    "ion_pump_status_fault": "fault_status",
    "ion_pump_status_on": "power_on_status",
    "ion_pump_voltage_rb": "voltage_readback",
    "ion_pump_voltage_sp": "voltage_setpoint",
    "klystron_power_rb": "power_readback",
    "klystron_status_fault": "fault_status",
    "klystron_status_on": "power_on_status",
    "klystron_status_ready": "ready_status",
    "klystron_voltage_rb": "rf_hv_readback",
    "klystron_voltage_sp": "rf_hv_setpoint",
    "neutron_dose_rate_avg_1hr": "dose_rate_readback",
    "neutron_dose_rate_avg_1min": "dose_rate_readback",
    "neutron_dose_rate_inst": "dose_rate_readback",
    "neutron_status_alarm": "status",
    "neutron_status_valid": "status",
    "qd_current_golden": "current_golden_readback",
    "qd_current_rb": "current_readback",
    "qd_current_sp": "current_setpoint",
    "qd_status_fault": "fault_status",
    "qd_status_on": "power_on_status",
    "qd_status_ready": "ready_status",
    "qf_current_golden": "current_golden_readback",
    "qf_current_rb": "current_readback",
    "qf_current_sp": "current_setpoint",
    "qf_status_fault": "fault_status",
    "qf_status_on": "power_on_status",
    "qf_status_ready": "ready_status",
    "qfa_current_golden": "current_golden_readback",
    "qfa_current_rb": "current_readback",
    "qfa_current_sp": "current_setpoint",
    "qfa_status_fault": "fault_status",
    "qfa_status_on": "power_on_status",
    "qfa_status_ready": "ready_status",
    "sd_current_golden": "current_golden_readback",
    "sd_current_rb": "current_readback",
    "sd_current_sp": "current_setpoint",
    "sd_status_fault": "fault_status",
    "sd_status_on": "power_on_status",
    "sd_status_ready": "ready_status",
    "sf_current_golden": "current_golden_readback",
    "sf_current_rb": "current_readback",
    "sf_current_sp": "current_setpoint",
    "sf_status_fault": "fault_status",
    "sf_status_on": "power_on_status",
    "sf_status_ready": "ready_status",
    "shd_current_golden": "current_golden_readback",
    "shd_current_rb": "current_readback",
    "shd_current_sp": "current_setpoint",
    "shd_status_fault": "fault_status",
    "shd_status_on": "power_on_status",
    "shd_status_ready": "ready_status",
    "shf_current_golden": "current_golden_readback",
    "shf_current_rb": "current_readback",
    "shf_current_sp": "current_setpoint",
    "shf_status_fault": "fault_status",
    "shf_status_on": "power_on_status",
    "shf_status_ready": "ready_status",
    "valve_control_close": "valve_open_command",
    "valve_control_open": "valve_open_command",
    "valve_position_closed": "valve_status",
    "valve_position_open": "valve_status",
    "valve_status_fault": "fault_status",
    "valve_status_ready": "ready_status",
    "vcm_current_golden": "current_golden_readback",
    "vcm_current_rb": "current_readback",
    "vcm_current_sp": "current_setpoint",
    "vcm_status_on": "power_on_status",
    "vcm_status_ready": "ready_status",
}


class RecordsError(RuntimeError):
    """The demo sources disagree with each other; the records cannot be derived."""


@dataclass(frozen=True)
class Binding:
    """One TTL binding.

    Attributes:
        address: The channel's full address.
        role: ``setpoint`` or ``readback``.
        description: The binding's description.
        ttl_signal: The local name of the signal the binding reads or writes.
    """

    address: str
    role: str
    description: str
    ttl_signal: str


@dataclass(frozen=True)
class TtlDevice:
    """One TTL device and its bindings.

    Attributes:
        machine: The address's first segment.
        system: The address's second segment.
        family: The address's third segment.
        name: The device's bare name.
        cls: The device's vocabulary class.
        bindings: The device's bindings, sorted by address.
    """

    machine: str
    system: str
    family: str
    name: str
    cls: str
    bindings: tuple[Binding, ...]

    @property
    def id(self) -> str:
        return f"{self.machine}/{self.name}"


@dataclass(frozen=True)
class Records:
    """The demo's authored records.

    Attributes:
        identity: ``identity.yaml``.
        classes: ``classes.yaml``.
        places: ``records/places.yaml``.
        devices: ``records/devices.yaml``.
        channels: ``records/channels.yaml``.
        groups: ``records/groups.yaml``.
    """

    identity: dict[str, Any]
    classes: list[dict[str, Any]]
    places: list[dict[str, Any]]
    devices: list[dict[str, Any]]
    channels: list[dict[str, Any]]
    groups: list[dict[str, Any]]


def _json(path: str) -> Any:
    return json.loads(fp._rel(path).read_text(encoding="utf-8"))


def _local(term: Any) -> str:
    return str(term).rsplit("/", 1)[-1]


@cache
def ttl_devices() -> tuple[TtlDevice, ...]:
    """Every device of the demo TTL with its bindings, sorted by id."""
    import rdflib

    prop = rdflib.Namespace(fp._NARAD_PROPERTY)
    graph = rdflib.Graph()
    graph.parse(fp._rel(fp.DEMO_TTL), format="turtle")
    devices = []
    for subject, name in graph.subject_objects(prop.sourceName):
        bindings = []
        for binding in graph.objects(subject, prop.hasBinding):
            written = graph.value(binding, prop.writesSignal)
            read = graph.value(binding, prop.readsSignal)
            if (written is None) == (read is None):
                raise RecordsError(f"{binding}: must either read or write one signal")
            description = graph.value(binding, prop.description)
            bindings.append(
                Binding(
                    address=str(graph.value(binding, prop.fullPv)),
                    role="setpoint" if written is not None else "readback",
                    description="" if description is None else str(description),
                    ttl_signal=_local(written if written is not None else read),
                )
            )
        bindings.sort(key=lambda b: b.address)
        segments = {tuple(b.address.split(":")[:3]) for b in bindings}
        if len(segments) != 1:
            raise RecordsError(f"{name}: bindings span {sorted(segments)}")
        machine, system, family = segments.pop()
        devices.append(
            TtlDevice(
                machine=machine,
                system=system,
                family=family,
                name=str(name),
                cls=_local(graph.value(subject, rdflib.RDF.type)),
                bindings=tuple(bindings),
            )
        )
    devices.sort(key=lambda d: d.id)
    ids = [d.id for d in devices]
    if len(set(ids)) != len(ids):
        raise RecordsError("two TTL devices share a <machine>/<name> id")
    return tuple(devices)


def signal_role(ttl_signal: str) -> str | None:
    """The vocabulary signal role a TTL signal means, or ``None``."""
    return SIGNAL_ROLE.get(ttl_signal)


@cache
def wired_addresses() -> frozenset[str]:
    """Every address the virtual accelerator's bindings wire to the deck."""
    document = _json(VA_BINDINGS)
    addresses = set()
    for binding in document["bindings"]:
        for key in ("setpoint_address", "readback_address"):
            if binding.get(key):
                addresses.add(binding[key])
    return frozenset(addresses)


def wired_device_ids() -> frozenset[str]:
    """The devices owning at least one wired address."""
    wired = wired_addresses()
    return frozenset(
        device.id for device in ttl_devices() if any(b.address in wired for b in device.bindings)
    )


@cache
def rf_cavity_sector() -> int:
    """The deck sector holding the deck's RF cavity element."""
    sector = None
    found = []
    for element in _json(SR_DECK)["elements"]:
        name = str(element.get("FamName", ""))
        if element.get("Class") == "Marker" and name.startswith("SECT"):
            sector = int(name.removeprefix("SECT"))
        elif element.get("Class") == RF_CAVITY_CLASS:
            found.append(sector)
    if len(found) != 1 or found[0] is None:
        raise RecordsError(f"{SR_DECK}: expected one RF cavity inside a sector, found {found}")
    return found[0]


def _hierarchical_tree() -> dict[str, Any]:
    tree: dict[str, Any] = _json(fp.TIER3_HIERARCHICAL)["tree"]
    return tree


def _children(node: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for key, value in node.items():
        if not key.startswith("_"):
            yield key, value


def _field_sentences(family_node: dict[str, Any]) -> dict[str, str]:
    """``<field>`` and ``<field>/<subfield>`` -> the hierarchical description there."""
    sentences: dict[str, str] = {}
    for field, field_node in _children(family_node["DEVICE"]):
        sentences[field] = field_node["_description"]
        for subfield, subfield_node in _children(field_node):
            sentences[f"{field}/{subfield}"] = subfield_node["_description"]
    return sentences


def _setup_by_device() -> dict[str, dict[str, Any]]:
    """Device id -> its middle-layer common name, DeviceList and ElementList entries."""
    by_address = {b.address: d.id for d in ttl_devices() for b in d.bindings}
    entries: dict[str, dict[str, Any]] = {}
    middle_layer = _json(MIDDLE_LAYER)
    for machine, machine_node in _children(middle_layer):
        for family, family_node in _children(machine_node):
            setup = family_node["_setup"]
            owners = _column_owners(family_node, by_address, len(setup["CommonNames"]))
            for index, device_id in enumerate(owners):
                if device_id in entries:
                    raise RecordsError(f"{machine}/{family}: {device_id} set up twice")
                entries[device_id] = {
                    "common_name": setup["CommonNames"][index],
                    "DeviceList": setup["DeviceList"][index],
                    "ElementList": setup["ElementList"][index],
                }
    return entries


def _column_owners(family_node: dict[str, Any], by_address: dict[str, str], size: int) -> list[str]:
    """The device owning each ``_setup`` position, from every field's ChannelNames."""
    owners: list[str | None] = [None] * size
    for _field, field_node in _children(family_node):
        for _subfield, leaf in _children(field_node):
            names = leaf["ChannelNames"]
            if len(names) != size:
                raise RecordsError(f"{names[0]}: {len(names)} ChannelNames for {size} devices")
            for index, address in enumerate(names):
                owner = by_address[address]
                if owners[index] not in (None, owner):
                    raise RecordsError(f"{address}: position {index} names two devices")
                owners[index] = owner
    if any(owner is None for owner in owners):
        raise RecordsError("a middle-layer family has a position no channel names")
    return [str(owner) for owner in owners]


def _hand_place(device: TtlDevice, setup: dict[str, Any]) -> str:
    """The place an unwired device is authored in."""
    if device.machine != DECK_MACHINE:
        return device.machine
    if device.family in SECTOR_INDEXED_FAMILIES:
        return f"{DECK_MACHINE}/SECT{setup['DeviceList'][0]}"
    if device.family in RF_CAVITY_FAMILIES:
        return f"{DECK_MACHINE}/SECT{rf_cavity_sector()}"
    return DECK_MACHINE


def _identity(*, standalone: bool) -> dict[str, Any]:
    identity: dict[str, Any] = {"code": "ca"}
    if standalone:
        identity["name"] = "Example Research Facility"
    return identity


def _places() -> list[dict[str, Any]]:
    tree = _hierarchical_tree()
    labels = _json(MIDDLE_LAYER)
    places: list[dict[str, Any]] = []
    for machine in MACHINES:
        places.append(
            {
                "id": machine,
                "level": "machine",
                "description": tree[machine]["_description"],
                "names": [labels[machine]["_description"]],
            }
        )
        if machine != DECK_MACHINE:
            continue
        for n in range(1, SECTORS + 1):
            span: dict[str, Any] = {"model": DECK_MACHINE, "from_marker": f"SECT{n}"}
            if n < SECTORS:
                span["to_marker"] = f"SECT{n + 1}"
            places.append({"id": f"{machine}/SECT{n}", "level": "sector", "span": span})
    return places


def _devices() -> list[dict[str, Any]]:
    setups = _setup_by_device()
    wired = wired_device_ids()
    devices = []
    for device in ttl_devices():
        setup = setups[device.id]
        record: dict[str, Any] = {
            "id": device.id,
            "class": device.cls,
            "names": [device.name, setup["common_name"]],
        }
        if device.id not in wired:
            record["place"] = _hand_place(device, setup)
        record["attributes"] = {
            "DeviceList": setup["DeviceList"],
            "ElementList": setup["ElementList"],
        }
        devices.append(record)
    return devices


def _channels() -> list[dict[str, Any]]:
    names = fp._in_context_names()
    value_types = fp._value_types()
    in_context = {row["address"] for row in _json(fp.TIER1_IN_CONTEXT)["channels"]}
    channels = []
    for device in ttl_devices():
        for binding in device.bindings:
            record: dict[str, Any] = {"id": binding.address}
            if binding.role == "setpoint":
                record["role"] = "setpoint"
            record["on"] = {"device": device.id}
            role = signal_role(binding.ttl_signal)
            if role is not None:
                record["signal"] = role
            if value_types[binding.address] != "float":
                record["value_type"] = value_types[binding.address]
            record["names"] = list(names[binding.address])
            record["description"] = binding.description
            if binding.address in in_context:
                record["tags"] = [IN_CONTEXT_TAG]
            channels.append(record)
    for row in _json(ADDITIONS)["rows"]:
        channels.append(
            {
                "id": row["address"],
                "on": {"place": DECK_MACHINE},
                "names": list(row["names"]),
                "description": row["description"],
            }
        )
    channels.sort(key=lambda record: record["id"])
    return channels


def _groups() -> list[dict[str, Any]]:
    tree = _hierarchical_tree()
    labels = _json(MIDDLE_LAYER)
    members: dict[tuple[str, str], list[str]] = {}
    for device in ttl_devices():
        members.setdefault((device.machine, device.family), []).append(device.id)
        members.setdefault((device.machine, device.system), []).append(device.id)
    groups = []
    for machine, machine_node in _children(tree):
        for system, system_node in _children(machine_node):
            groups.append(
                {
                    "id": f"{machine}/{system}",
                    "description": system_node["_description"],
                    "members": sorted(members[(machine, system)]),
                }
            )
            for family, family_node in _children(system_node):
                groups.append(
                    {
                        "id": f"{machine}/{family}",
                        "description": family_node["_description"],
                        "names": [labels[machine][family]["_description"]],
                        "members": sorted(members[(machine, family)]),
                        "signals": _field_sentences(family_node),
                    }
                )
    ids = [group["id"] for group in groups]
    if len(set(ids)) != len(ids):
        raise RecordsError("a family and a system share a <machine>/<name> group id")
    groups.sort(key=lambda group: group["id"])
    return groups


def build_records(*, standalone: bool = False) -> Records:
    """Derive the demo's authored records from its committed sources.

    Args:
        standalone: Write the standalone presets' identity, which adds a name.

    Returns:
        The records, each list in its emission order.
    """
    return Records(
        identity=_identity(standalone=standalone),
        classes=[],
        places=_places(),
        devices=_devices(),
        channels=_channels(),
        groups=_groups(),
    )
