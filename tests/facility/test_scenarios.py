"""The demo's scenarios, and scenario faults through the build."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from osprey.facility.build import build_facility
from osprey.facility.errors import FacilityBuildError
from osprey.facility.scenarios import _value_errors
from osprey_connectors.simulation import STILL_ALL
from osprey_connectors.simulation.envelope import active_envelopes
from tests.facility._synthetic_trees import plain_tree, write_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
FACILITY = REPO_ROOT / "src/osprey/templates/facilities/example"

#: The demo's physics model.
MODEL = "SR"


def _scenario(name: str) -> dict[str, Any]:
    return yaml.safe_load((FACILITY / "scenarios" / f"{name}.yaml").read_text(encoding="utf-8"))


def test_an_unplaned_monitor_error_lands_on_both_readings() -> None:
    faults = _scenario("bpm-polarity")["faults"][MODEL]
    assert faults == {
        "SR:DIAG:BPM:17:POSITION:X": {"polarity": -1},
        "SR:DIAG:BPM:17:POSITION:Y": {"polarity": -1},
    }


def test_a_corrector_gain_is_its_setpoint_calibration() -> None:
    faults = _scenario("orm-dual-fault")["faults"][MODEL]
    assert faults["SR:MAG:HCM:01:CURRENT:SP"] == {"cal_factor": 0.5}


def test_the_demo_build_carries_every_scenario_verbatim(built_control_assistant: Any) -> None:
    built = {s["name"]: s for s in built_control_assistant.facility["scenarios"]}
    for path in sorted((FACILITY / "scenarios").glob("*.yaml")):
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert built[path.stem] == {"name": path.stem, **source}


# --- a Tango address --------------------------------------------------------------

#: A monitor reading whose address holds ``/``, as a Tango attribute does.
TANGO = "sr/d-bpm/1/x"


def test_a_tango_address_fault_round_trips(tmp_path: Path) -> None:
    tree = plain_tree()
    tree["records/channels.yaml"].append({"id": TANGO, "on": {"device": "SR/BPM1"}})
    tree["models.yaml"] = [
        {
            "name": "optics",
            "engine": "pyat",
            "wiring": [
                {"address": "Q1:SP", "element": "Q1"},
                {"address": TANGO, "element": "BPM1", "engine": {"axis": "x"}},
            ],
        }
    ]
    faults = {"optics": {TANGO: {"offset": 1.0e-4, "roll": 0.01}}}
    tree["scenarios/tango.yaml"] = {"faults": faults, "channel_faults": {"Q1:SP": "stuck"}}

    document = build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")

    assert [s["faults"] for s in document["scenarios"]] == [faults]


# --- the still sentinel -----------------------------------------------------------


def test_the_still_value_the_build_accepts_is_the_one_the_simulator_stills_on(
    tmp_path: Path,
) -> None:
    tree = plain_tree()
    tree["scenarios/hush.yaml"] = {"still": STILL_ALL}

    document = build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")

    hush = {s["name"]: s for s in document["scenarios"]}["hush"]
    assert hush["still"] == STILL_ALL
    seeds = {"BPM1:X": {"nominal": 1.0, "noise": {"absolute": 0.1}}}
    assert active_envelopes(seeds, {"hush": hush}, ["hush"]) == {"BPM1:X": 0.0}


def test_a_still_value_other_than_the_sentinel_stops_the_build(tmp_path: Path) -> None:
    tree = plain_tree()
    tree["scenarios/hush.yaml"] = {"still": STILL_ALL.upper()}

    with pytest.raises(FacilityBuildError) as caught:
        build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")

    assert "neither `all` nor a list of addresses" in str(caught.value)


# --- stop lines carry every digit -------------------------------------------------


def test_an_override_off_its_band_prints_every_digit_it_carries(tmp_path: Path) -> None:
    tree = plain_tree()
    tree["limits.yaml"] = {
        "records": [{"address": "Q1:SP", "min_value": 0.0, "max_value": 1.2345678}]
    }
    tree["scenarios/warm.yaml"] = {"overrides": {"Q1:SP": 1.23456789}}

    with pytest.raises(FacilityBuildError) as caught:
        build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")

    assert "`overrides.Q1:SP` 1.23456789 lies above `max_value` 1.2345678" in str(caught.value)


def test_a_fault_value_off_its_range_prints_every_digit_it_carries() -> None:
    fault = SimpleNamespace(options=None, value_range=(0.0, 0.123456789))

    errors = list(_value_errors("warm", ["scenarios/warm.yaml"], "offset", 0.1234567891, fault))

    assert [(error.detail, error.remedy) for error in errors] == [
        (
            "`offset` 0.1234567891 lies above its range [0, 0.123456789]",
            "write a value inside [0, 0.123456789]",
        )
    ]
