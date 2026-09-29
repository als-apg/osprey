"""The demo generator's records against the demo's hand-written sources.

``scripts/facility_demo/generate.py`` writes the demo's ``data/facility/``
records from its TTL and channel databases. These tests read the records back
from the YAML the generator writes and hold them to the sources: the channel
rows joined as the fingerprint joins them, the hand places, the value types,
the signal roles, the in_context tags, the groups and the places.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from osprey.facility.sources import load_sources, read_yaml
from tests.facility.test_cf_view_parity import (
    FINGERPRINT_ROW_KEYS,
    fingerprint_rows,
    load_golden,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "facility_demo" / "generate.py"
CA_DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
DEMO_TTL = CA_DATA / "demo_machine.ttl"
TIER1_IN_CONTEXT = CA_DATA / "channel_databases/tiers/tier1/in_context.json"
TIER3_HIERARCHICAL = CA_DATA / "channel_databases/tiers/tier3/hierarchical.json"
VA_BINDINGS = CA_DATA / "simulation/va_bindings.json"
VOCABULARY = REPO_ROOT / "src/osprey/facility/schema/_generated/vocabulary.json"

_NARAD_PROPERTY = "https://narad.example.org/property/"


@cache
def generator() -> ModuleType:
    """``scripts/facility_demo/generate.py`` as a module."""
    spec = importlib.util.spec_from_file_location("facility_demo_generate", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records_module() -> ModuleType:
    """``scripts/facility_demo/_records.py``, as the generator loaded it."""
    records: ModuleType = generator()._records
    return records


@cache
def generated_files(standalone: bool = False) -> dict[str, str]:
    """Relative path -> text of every file the generator writes."""
    module = generator()
    files: dict[str, str] = module.files(module._records.build_records(standalone=standalone))
    return files


@cache
def generated(name: str) -> Any:
    """One generated file, parsed back as the facility loader parses it."""
    return read_yaml(generated_files()[name])


def records_by_id(kind: str) -> dict[str, dict[str, Any]]:
    """``records/<kind>s.yaml`` keyed by id."""
    return {record["id"]: record for record in generated(f"records/{kind}s.yaml")}


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
        for channel in generated("records/channels.yaml")
    ]
    return sorted(rows, key=lambda row: row["address"])


def test_written_tree_loads_without_a_stop(tmp_path: Path) -> None:
    assert generator().main(["--out", str(tmp_path)]) == 0
    written = {
        path.relative_to(tmp_path).as_posix(): path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.yaml")
    }
    assert written == generated_files()
    result = load_sources(tmp_path)
    assert result.errors == []
    assert result.sources.identity == {"code": "ca"}
    assert result.sources.classes == []


def test_generation_is_deterministic() -> None:
    module = generator()
    assert module.files(module._records.build_records()) == generated_files()


def test_standalone_identity_adds_the_facility_name() -> None:
    assert read_yaml(generated_files(standalone=True)["identity.yaml"]) == {
        "code": "ca",
        "name": "Example Research Facility",
    }
    standalone = {k: v for k, v in generated_files(standalone=True).items() if k != "identity.yaml"}
    assert standalone == {k: v for k, v in generated_files().items() if k != "identity.yaml"}


def test_loading_the_generator_leaves_the_import_path_and_top_level_names_alone() -> None:
    path = list(sys.path)
    spec = importlib.util.spec_from_file_location("facility_demo_generate_isolated", GENERATOR)
    assert spec and spec.loader
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert sys.path == path
    assert "fingerprint" not in sys.modules
    assert "_records" not in sys.modules


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


def test_every_fingerprint_channel_carries_the_role_its_ttl_signal_means() -> None:
    table = records_module().SIGNAL_ROLE
    roles = {role["name"] for role in _json(VOCABULARY)["signal_roles"]}
    channels = records_by_id("channel")
    bindings = ttl_bindings()
    assert len(bindings) == 2908
    assert set(table) <= {binding["signal"] for binding in bindings.values()}
    for address, binding in bindings.items():
        channel = channels[address]
        if binding["signal"] in table:
            assert channel["signal"] == table[binding["signal"]], address
        else:
            assert "signal" not in channel, address
    signals = {c["signal"] for c in channels.values() if "signal" in c}
    assert signals <= roles
    assert not signals & {binding["signal"] for binding in bindings.values()}


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
    places = generated("records/places.yaml")
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
    ids = [record["id"] for record in generated(name)]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
