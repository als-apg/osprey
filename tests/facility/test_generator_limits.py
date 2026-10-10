"""The demo's committed ``limits.yaml`` and ``measurement/SR.yaml``.

The demo limits three setpoints, each record teaching one shape, and leaves
every other channel to the deployment's limits mode. Its measurement file
names the deck machine's family groups and instruments and pyAML's step and
settle keys. These tests read both files back as the facility loader parses
them and hold them to the committed records and the limits golden.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from osprey.facility.build import build_facility
from osprey.facility.sources import read_yaml, slot_names
from osprey.facility.views.limits import limits_document
from tests._builds import BuiltProject
from tests.facility.test_cf_view_parity import load_golden

REPO_ROOT = Path(__file__).resolve().parents[2]
FACILITY_TREE = REPO_ROOT / "src/osprey/templates/facilities/example"

_NARAD_PROPERTY = "https://narad.example.org/property/"

#: The three records, as the file states them.
TEACHING_RECORDS = [
    {"address": "SR:MAG:HCM:01:CURRENT:SP", "min_value": -12.0, "max_value": 12.0},
    {
        "address": "SR:RF:CAVITY:01:FREQUENCY:SP",
        "min_value": 500.0,
        "max_value": 500.8,
        "max_step": 0.01,
    },
    {"address": "SR:VAC:ION-PUMP:01:VOLTAGE:SP", "writable": False},
]

#: The three records resolved, keyed by address.
RESOLVED_ROWS = {
    "SR:MAG:HCM:01:CURRENT:SP": {
        "min_value": -12.0,
        "max_value": 12.0,
        "max_step": None,
        "writable": True,
        "confirm": True,
    },
    "SR:RF:CAVITY:01:FREQUENCY:SP": {
        "min_value": 500.0,
        "max_value": 500.8,
        "max_step": 0.01,
        "writable": True,
        "confirm": True,
    },
    "SR:VAC:ION-PUMP:01:VOLTAGE:SP": {
        "min_value": None,
        "max_value": None,
        "max_step": None,
        "writable": False,
        "confirm": True,
    },
}

#: The deck's RF frequency, in MHz, that the wired cavity's band surrounds.
CAVITY_OPERATING_MHZ = 500.417

#: The measurement keys that are not pyAML step or settle keys.
_MEASUREMENT_ROLES = {"kinds", "groups", "instruments"}


@cache
def committed_files() -> dict[str, str]:
    """Relative path -> text of every YAML source of the committed tree.

    Returns:
        Each file's path relative to the facility tree and its text.
    """
    return {
        path.relative_to(FACILITY_TREE).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(FACILITY_TREE.rglob("*.yaml"))
    }


@cache
def committed(name: str) -> Any:
    """One committed file, parsed as the facility loader parses it."""
    return read_yaml(committed_files()[name])


def records_by_id(kind: str) -> dict[str, dict[str, Any]]:
    """``records/<kind>s.yaml`` keyed by id."""
    return {record["id"]: record for record in committed(f"records/{kind}s.yaml")}


def wired_devices(view: Path, facility: dict[str, Any]) -> frozenset[str]:
    """The devices owning an address the facility's ``SR`` model wires.

    Args:
        view: The graph view the build writes.
        facility: The build's facility file.

    Returns:
        The device id of every device-bound binding of a wired address.
    """
    import rdflib

    prop = rdflib.Namespace(_NARAD_PROPERTY)
    graph = rdflib.Graph()
    graph.parse(view, format="turtle")
    device_of = {}
    for device, binding in graph.subject_objects(prop.hasBinding):
        if graph.value(device, prop.sourceName) is None:
            continue
        device_of[str(graph.value(binding, prop.fullPv))] = str(graph.value(device, prop.deviceId))
    [model] = [model for model in facility["models"] if model["name"] == "SR"]
    wired = {record["address"] for record in model["wiring"]}
    return frozenset(device_of[address] for address in wired if address in device_of)


def measurement() -> dict[str, Any]:
    """``measurement/SR.yaml`` as the loader parses it."""
    document: dict[str, Any] = committed("measurement/SR.yaml")
    return document


# --- limits ----------------------------------------------------------------------


def test_limits_hold_exactly_the_three_teaching_records() -> None:
    assert committed("limits.yaml") == {"records": TEACHING_RECORDS}


def test_each_limits_record_sits_under_its_comment() -> None:
    lines = committed_files()["limits.yaml"].splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("- address: ")]
    assert len(starts) == 3
    for start, record in zip(starts, TEACHING_RECORDS, strict=True):
        assert lines[start] == f"- address: {record['address']}"
        assert lines[start - 1].startswith("# ")


def test_every_limited_address_is_a_setpoint() -> None:
    channels = records_by_id("channel")
    for record in TEACHING_RECORDS:
        assert channels[record["address"]].get("role") == "setpoint"


def test_cavity_band_holds_the_deck_operating_point() -> None:
    cavity = RESOLVED_ROWS["SR:RF:CAVITY:01:FREQUENCY:SP"]
    assert cavity["min_value"] < CAVITY_OPERATING_MHZ < cavity["max_value"]


def test_limits_golden_holds_the_three_resolved_rows() -> None:
    golden = load_golden("limits.json")
    assert "defaults" not in golden
    assert golden["count"] == 3
    assert golden["channels"] == RESOLVED_ROWS


def test_limits_golden_resolves_the_committed_records() -> None:
    """The limits view of the committed tree is the golden, a bound left unstated absent."""
    resolved = limits_document(build_facility(FACILITY_TREE, project_name="ca"))
    del resolved["_version"]
    golden = {
        address: {slot: value for slot, value in row.items() if value is not None}
        for address, row in load_golden("limits.json")["channels"].items()
    }
    assert resolved == golden


# --- measurement -----------------------------------------------------------------


def test_measurement_allows_the_five_kinds() -> None:
    assert measurement()["kinds"] == [
        "orm",
        "dispersion",
        "trm",
        "crm",
        "chromaticity_monitor",
    ]


def test_each_measurement_group_is_a_deck_machine_family_with_a_wired_member(
    built_control_assistant: BuiltProject,
) -> None:
    groups = records_by_id("group")
    wired = wired_devices(
        built_control_assistant.build_dir / "data" / "graph" / "facility.ttl",
        built_control_assistant.facility,
    )
    named = measurement()["groups"]
    assert sorted(named) == ["bpm", "hcor", "quad", "sext", "vcor"]
    for role, group_id in named.items():
        machine, _, family = group_id.partition("/")
        assert machine == "SR" and family, f"{role}: {group_id} is not an SR/<family> group"
        assert group_id in groups, f"{role}: {group_id} is not a committed group"
        assert set(groups[group_id]["members"]) & wired, f"{role}: {group_id} has no wired member"


def test_measurement_instruments_are_served_channels() -> None:
    channels = records_by_id("channel")
    instruments = measurement()["instruments"]
    assert instruments == {
        "tune": "SR:DIAG:TUNE:X",
        "chromaticity": "SR:DIAG:CHROM:X",
        "rf": "SR:RF:CAVITY:01:FREQUENCY:SP",
    }
    for address in instruments.values():
        assert address in channels
    assert channels[instruments["rf"]].get("role") == "setpoint"


def test_measurement_carries_every_pyaml_step_and_settle_key() -> None:
    document = measurement()
    tuning = {key: value for key, value in document.items() if key not in _MEASUREMENT_ROLES}
    assert set(tuning) == slot_names("Measurement") - _MEASUREMENT_ROLES
    assert tuning == {
        "n_step": 5,
        "n_avg_meas": 1,
        "fit_order": 2,
        "sleep_between_step": 0.0,
        "sleep_between_meas": 0.0,
        "corrector_delta": 1.0e-5,
        "quad_delta": 1.0e-3,
        "sextu_delta": 1.0e-2,
        "frequency_delta": 100.0,
    }


def test_the_tree_commits_both_files() -> None:
    assert {"limits.yaml", "measurement/SR.yaml"} <= set(committed_files())
