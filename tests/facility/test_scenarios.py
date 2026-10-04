"""The demo's translated scenarios, and scenario faults through the build.

A monitor fault written per element in a simulation bundle and the same fault
written per address in ``data/facility/scenarios/`` must reach the same
readings: each translated fault, mapped back through the wiring to its
element, equals the per-plane fields the bundle's error fans out to today.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import build_facility
from osprey.simulation.apply import _bpm_error_env_fields
from osprey_connectors.simulation.machine import _parse_physics_fault
from tests.facility._synthetic_trees import plain_tree, write_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
FACILITY = DATA / "facility"
BUNDLES = DATA / "simulation" / "scenarios"

#: The demo's physics model.
MODEL = "SR"


def _wiring() -> dict[str, dict[str, Any]]:
    models = yaml.safe_load((FACILITY / "models.yaml").read_text(encoding="utf-8"))
    (model,) = [m for m in models if m["name"] == MODEL]
    return {record["address"]: record for record in model["wiring"]}


def _scenario(name: str) -> dict[str, Any]:
    return yaml.safe_load((FACILITY / "scenarios" / f"{name}.yaml").read_text(encoding="utf-8"))


def _bundle_physics(name: str) -> Any:
    raw = json.loads((BUNDLES / name / "scenario.json").read_text(encoding="utf-8"))
    return _parse_physics_fault(name, raw["physics"])


def _today(name: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """The bundle's monitor errors as today's per-plane fields, and its corrector gains."""
    physics = _bundle_physics(name)
    monitors = {
        element: _bpm_error_env_fields(spec) for element, spec in physics.bpm_errors.items()
    }
    return monitors, dict(physics.corrector_gain)


def _translated(name: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """The translated faults, mapped back to elements through the wiring."""
    wiring = _wiring()
    monitors: dict[str, dict[str, float]] = {}
    gains: dict[str, float] = {}
    for address, fields in _scenario(name)["faults"][MODEL].items():
        record = wiring[address]
        axis = record["engine"].get("axis")
        for field, value in fields.items():
            if field == "cal_factor":
                gains[record["element"]] = float(value)
                continue
            key = field if field == "roll" else f"{field}_{axis}"
            monitors.setdefault(record["element"], {})[key] = float(value)
    return monitors, gains


@pytest.mark.parametrize("name", ["bpm-polarity", "orm-dual-fault"])
def test_a_translated_physics_bundle_faults_what_the_bundle_faults(name: str) -> None:
    assert _translated(name) == _today(name)


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
