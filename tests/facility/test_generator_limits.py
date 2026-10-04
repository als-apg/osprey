"""The demo's committed ``limits.yaml`` and ``measurement/SR.yaml``.

The demo limits three setpoints, each record teaching one shape, and leaves
every other channel to the deployment's limits mode. Its measurement file
names the deck machine's family groups and instruments and pyAML's step and
settle keys. These tests read both files back as the facility loader parses
them and hold them to the committed records and the limits golden.
"""

from __future__ import annotations

import importlib.util
import json
from functools import cache
from types import ModuleType
from typing import Any

import pytest

from osprey.facility.sources import slot_names
from tests._builds import BuiltProject
from tests.facility.test_cf_view_parity import load_golden
from tests.facility.test_generator_records import (
    REPO_ROOT,
    committed,
    committed_files,
    records_by_id,
    wired_devices,
)

LIMITS_MODULE = REPO_ROOT / "scripts" / "facility_demo" / "_limits.py"

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
def limits_module() -> ModuleType:
    """``scripts/facility_demo/_limits.py`` as a module."""
    spec = importlib.util.spec_from_file_location("facility_demo_limits_test", LIMITS_MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    module = limits_module()
    roles = {
        address: record.get("role", "readback")
        for address, record in records_by_id("channel").items()
    }
    resolved = {
        record["address"]: module.resolve(record, roles[record["address"]])
        for record in committed("limits.yaml")["records"]
    }
    assert resolved == load_golden("limits.json")["channels"]


def test_limits_golden_is_what_its_reproduce_command_writes() -> None:
    golden_path = REPO_ROOT / "tests" / "facility" / "golden" / "limits.json"
    assert limits_module().golden() == golden_path.read_bytes()
    assert json.loads(golden_path.read_text(encoding="utf-8"))["_reproduce"] == (
        "uv run python scripts/facility_demo/_limits.py --write tests/facility/golden/limits.json"
    )


# --- measurement -----------------------------------------------------------------


def test_measurement_allows_the_five_kinds() -> None:
    assert measurement()["kinds"] == [
        "orm",
        "dispersion",
        "trm",
        "crm",
        "chromaticity_monitor",
    ]


@pytest.mark.xdist_group("built_control_assistant")
def test_each_measurement_group_is_a_deck_machine_family_with_a_wired_member(
    built_control_assistant: BuiltProject,
) -> None:
    groups = records_by_id("group")
    wired = wired_devices(built_control_assistant.build_dir / "data" / "graph" / "facility.ttl")
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
        "singular_values": 16,
        "sleep_between_step": 0.0,
        "sleep_between_meas": 0.0,
        "corrector_delta": 1.0e-5,
        "quad_delta": 1.0e-3,
        "sextu_delta": 1.0e-2,
        "frequency_delta": 100.0,
    }


def test_the_tree_commits_both_files() -> None:
    assert {"limits.yaml", "measurement/SR.yaml"} <= set(committed_files())
