"""The demo's committed ``data/facility/`` records against its other sources.

The control-assistant preset commits the demo's facility records as authored
sources. These tests read them as the facility loader reads them and hold them
to the demo's other sources: the channel rows joined as the frozen fingerprint
joins them, the hand places, the value types, the signal roles, the in_context
tags, the groups and the places.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.sources import load_sources, read_yaml
from tests.facility.test_cf_view_parity import (
    FINGERPRINT_ROW_KEYS,
    fingerprint_rows,
    load_golden,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CA_DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
FACILITY_TREE = CA_DATA / "facility"
STANDALONE_TREE = REPO_ROOT / "src/osprey/templates/apps/channel_finder_standalone/data/facility"
DEMO_TTL = CA_DATA / "demo_machine.ttl"
TIER1_IN_CONTEXT = CA_DATA / "channel_databases/tiers/tier1/in_context.json"
TIER3_HIERARCHICAL = CA_DATA / "channel_databases/tiers/tier3/hierarchical.json"
VA_BINDINGS = CA_DATA / "simulation/va_bindings.json"
MIDDLE_LAYER = CA_DATA / "channel_databases/tiers/tier3/middle_layer.json"
VOCABULARY = REPO_ROOT / "src/osprey/facility/schema/_generated/vocabulary.json"

_NARAD_PROPERTY = "https://narad.example.org/property/"

#: The general quantity roles the demo's channels need beyond the seed roles.
NEW_ROLES = frozenset(
    {
        "dose_rate_readback",
        "frequency_readback",
        "frequency_setpoint",
        "position_offset",
        "power_readback",
        "temperature_readback",
        "tuner_position_readback",
        "tuner_position_setpoint",
        "voltage_golden_readback",
        "voltage_readback",
        "voltage_setpoint",
    }
)


@cache
def committed_files(standalone: bool = False) -> dict[str, str]:
    """Relative path -> text of every YAML source of the committed tree.

    Args:
        standalone: Read a standalone preset's tree instead.

    Returns:
        Each file's path relative to ``data/facility`` and its text.
    """
    root = STANDALONE_TREE if standalone else FACILITY_TREE
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.yaml"))
    }


@cache
def committed(name: str) -> Any:
    """One committed file, parsed as the facility loader parses it."""
    return read_yaml(committed_files()[name])


def records_by_id(kind: str) -> dict[str, dict[str, Any]]:
    """``records/<kind>s.yaml`` keyed by id."""
    return {record["id"]: record for record in committed(f"records/{kind}s.yaml")}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def ttl_bindings() -> dict[str, dict[str, str]]:
    """Address -> ``{device, signal}``: the binding's device and TTL signal local name."""
    import rdflib

    prop = rdflib.Namespace(_NARAD_PROPERTY)
    graph = rdflib.Graph()
    graph.parse(DEMO_TTL, format="turtle")
    bindings = {}
    for device, binding in graph.subject_objects(prop.hasBinding):
        machine = str(graph.value(device, prop.sectionCode))
        signal = graph.value(binding, prop.readsSignal) or graph.value(binding, prop.writesSignal)
        bindings[str(graph.value(binding, prop.fullPv))] = {
            "device": f"{machine}/{graph.value(device, prop.sourceName)}",
            "signal": str(signal).rsplit("/", 1)[-1],
        }
    return bindings


@cache
def wired_devices() -> frozenset[str]:
    """The devices owning an address the virtual accelerator's bindings wire."""
    addresses = {
        address
        for binding in _json(VA_BINDINGS)["bindings"]
        for address in (binding["setpoint_address"], binding["readback_address"])
        if address
    }
    assert len(addresses) == 840
    bindings = ttl_bindings()
    return frozenset(bindings[address]["device"] for address in addresses)


def _joined_rows() -> list[dict[str, Any]]:
    """The channel records as fingerprint rows: schema defaults filled, sorted."""
    rows = [
        {
            "address": channel["id"],
            "role": channel.get("role", "readback"),
            "value_type": channel.get("value_type", "float"),
            "names": channel["names"],
            "description": channel["description"],
        }
        for channel in committed("records/channels.yaml")
    ]
    return sorted(rows, key=lambda row: row["address"])


def test_committed_tree_loads_without_a_stop() -> None:
    result = load_sources(FACILITY_TREE)
    assert result.errors == []
    assert result.sources.identity == {"code": "ca"}
    assert result.sources.classes == []


def test_standalone_identity_adds_the_facility_name() -> None:
    assert read_yaml(committed_files(standalone=True)["identity.yaml"]) == {
        "code": "ca",
        "name": "Example Research Facility",
    }


def test_joined_channels_equal_the_fingerprint_and_its_additions() -> None:
    additions = load_golden("demo_fingerprint_additions.json")["rows"]
    expected = sorted(fingerprint_rows() + additions, key=lambda row: row["address"])
    assert _joined_rows() == expected


def test_additions_are_the_four_deck_machine_instruments() -> None:
    rows = load_golden("demo_fingerprint_additions.json")["rows"]
    assert [row["address"] for row in rows] == [
        "SR:DIAG:CHROM:X",
        "SR:DIAG:CHROM:Y",
        "SR:DIAG:TUNE:X",
        "SR:DIAG:TUNE:Y",
    ]
    assert all(list(row) == FINGERPRINT_ROW_KEYS for row in rows)
    assert {(row["role"], row["value_type"]) for row in rows} == {("readback", "float")}
    channels = records_by_id("channel")
    assert all(channels[row["address"]]["on"] == {"place": "SR"} for row in rows)


def test_every_bi_channel_is_bool() -> None:
    channels = records_by_id("channel")
    bools = {row["address"] for row in fingerprint_rows() if row["value_type"] == "bool"}
    assert len(bools) == 1246
    assert {a for a, c in channels.items() if c.get("value_type") == "bool"} == bools
    assert {c.get("value_type") for c in channels.values()} == {None, "bool"}


def test_every_ttl_signal_maps_to_one_vocabulary_role() -> None:
    roles = {role["name"] for role in _json(VOCABULARY)["signal_roles"]}
    channels = records_by_id("channel")
    bindings = ttl_bindings()
    assert len(bindings) == 2908
    by_signal: dict[str, set[str | None]] = {}
    for address, binding in bindings.items():
        by_signal.setdefault(binding["signal"], set()).add(channels[address].get("signal"))
    assert {signal for signal, mapped in by_signal.items() if len(mapped) != 1} == set()
    mapped = {role for roles_of in by_signal.values() for role in roles_of}
    assert None not in mapped
    assert mapped <= roles
    assert not mapped & set(by_signal)


def test_the_first_vocabulary_roles_are_left_as_they_were() -> None:
    roles = [role["name"] for role in _json(VOCABULARY)["signal_roles"]]
    for name in ("current_setpoint", "status", "valve_open_command", "gradient_readback"):
        assert name in roles
    assert len(roles) == 89 + len(NEW_ROLES)
    assert NEW_ROLES <= set(roles)


def test_every_bound_setpoint_pairs_with_its_binding_readback() -> None:
    channels = records_by_id("channel")
    bound = [b for b in _json(VA_BINDINGS)["bindings"] if b["readback_address"]]
    assert len(bound) == 348
    for binding in bound:
        assert channels[binding["setpoint_address"]]["pair"] == binding["readback_address"]


def test_pairs_name_the_same_device_readback_and_only_setpoints_carry_one() -> None:
    channels = records_by_id("channel")
    for address, channel in channels.items():
        if channel.get("role", "readback") != "setpoint":
            assert "pair" not in channel, address
            continue
        rb = address.removesuffix(":SP") + ":RB"
        if address.endswith(":SP") and rb in channels:
            assert channel["pair"] == rb, address
            assert channels[rb].get("role", "readback") == "readback", address
            assert channels[rb]["on"] == channel["on"], address
        else:
            assert "pair" not in channel, address


@cache
def leaf_units() -> dict[str, str]:
    """Address -> the middle-layer leaf's HWUnits, for every leaf naming one."""
    units = {}
    for machine, machine_node in _json(MIDDLE_LAYER).items():
        if machine.startswith("_"):
            continue
        for family, family_node in machine_node.items():
            if family.startswith("_"):
                continue
            for field, field_node in family_node.items():
                if field.startswith("_"):
                    continue
                for subfield, leaf in field_node.items():
                    if subfield.startswith("_") or not leaf.get("HWUnits"):
                        continue
                    for address in leaf["ChannelNames"]:
                        units[address] = leaf["HWUnits"]
    return units


def test_every_channel_carries_its_leaf_unit() -> None:
    channels = records_by_id("channel")
    units = leaf_units()
    assert units
    assert {a: c["unit"] for a, c in channels.items() if "unit" in c} == units
    assert channels["SR:MAG:QF:01:CURRENT:SP"]["unit"] == "A"


def test_every_channel_is_on_its_ttl_device() -> None:
    channels = records_by_id("channel")
    for address, binding in ttl_bindings().items():
        assert channels[address]["on"] == {"device": binding["device"]}, address


def test_the_tier1_addresses_are_tagged_in_context() -> None:
    tier1 = {row["address"] for row in _json(TIER1_IN_CONTEXT)["channels"]}
    assert len(tier1) == 569
    channels = records_by_id("channel")
    assert {a for a, c in channels.items() if c.get("tags")} == tier1
    assert all(channels[a]["tags"] == ["in_context"] for a in tier1)


def test_only_unwired_devices_carry_a_hand_place() -> None:
    devices = records_by_id("device")
    wired = wired_devices()
    assert len(devices) == 512
    assert len(wired) == 420
    assert wired <= set(devices)
    assert all("place" not in devices[d] for d in wired)
    unwired = set(devices) - wired
    assert len(unwired) == 92
    assert all("place" in devices[d] for d in unwired)


def test_hand_places_sit_on_their_machine() -> None:
    devices = records_by_id("device")
    places = records_by_id("place")
    for device_id, device in devices.items():
        if "place" not in device:
            continue
        machine = device_id.split("/", 1)[0]
        assert device["place"] in places, device_id
        assert device["place"].split("/", 1)[0] == machine, device_id
        if machine != "SR":
            assert device["place"] == machine, device_id
    assert devices["SR/CAVITY01"]["place"] == "SR/SECT11"
    assert devices["SR/GAUGESR07"]["place"] == "SR/SECT7"
    assert devices["SR/VALVE12"]["place"] == "SR/SECT12"


def test_devices_carry_their_bare_name_and_class() -> None:
    import rdflib

    prop = rdflib.Namespace(_NARAD_PROPERTY)
    graph = rdflib.Graph()
    graph.parse(DEMO_TTL, format="turtle")
    devices = records_by_id("device")
    for subject, name in graph.subject_objects(prop.sourceName):
        device = devices[f"{graph.value(subject, prop.sectionCode)}/{name}"]
        assert device["names"][0] == str(name)
        assert device["class"] == str(graph.value(subject, rdflib.RDF.type)).rsplit("/", 1)[-1]
        assert set(device["attributes"]) == {"DeviceList", "ElementList"}


def test_places_are_the_machines_and_the_deck_machine_sectors() -> None:
    places = committed("records/places.yaml")
    machines = [p["id"] for p in places if p["level"] == "machine"]
    assert machines == ["SR", "BR", "BTS"]
    sectors = [p for p in places if p["level"] == "sector"]
    assert [p["id"] for p in sectors] == [f"SR/SECT{n}" for n in range(1, 13)]
    for n, place in enumerate(sectors, start=1):
        span = {"model": "SR", "from_marker": f"SECT{n}"}
        if n < 12:
            span["to_marker"] = f"SECT{n + 1}"
        assert place["span"] == span
    assert all("span" not in p for p in places if p["level"] == "machine")


def test_family_and_system_groups() -> None:
    tree = _json(TIER3_HIERARCHICAL)["tree"]
    groups = records_by_id("group")
    by_family: dict[str, set[str]] = {}
    by_system: dict[str, set[str]] = {}
    for address, binding in ttl_bindings().items():
        machine, system, family = address.split(":")[:3]
        by_family.setdefault(f"{machine}/{family}", set()).add(binding["device"])
        by_system.setdefault(f"{machine}/{system}", set()).add(binding["device"])
    assert not set(by_family) & set(by_system)
    assert set(groups) == set(by_family) | set(by_system)
    for gid, members in by_family.items():
        assert set(groups[gid]["members"]) == members, gid
        assert "signals" in groups[gid], gid
    for gid, members in by_system.items():
        machine, system = gid.split("/")
        assert set(groups[gid]["members"]) == members, gid
        assert groups[gid]["description"] == tree[machine][system]["_description"], gid
        assert "signals" not in groups[gid], gid
    assert "SR/DIPOLE" in groups and "BR/DIPOLE" in groups


@pytest.mark.parametrize("name", ["records/devices.yaml", "records/channels.yaml"])
def test_records_are_sorted_by_id(name: str) -> None:
    ids = [record["id"] for record in committed(name)]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
