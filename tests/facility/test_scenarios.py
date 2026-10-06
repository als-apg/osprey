"""The demo's scenarios, and scenario faults through the build."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.facility.build import build_facility
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
    faults = {"optics": {TANGO: {"offset": 1.0e-4, "roll": 0.01}, "Q1:SP": "stuck"}}
    tree["scenarios/tango.yaml"] = {"faults": faults}

    document = build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")

    assert [s["faults"] for s in document["scenarios"]] == [faults]
