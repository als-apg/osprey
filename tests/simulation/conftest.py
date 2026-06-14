"""Shared fixtures for simulation engine tests.

Uses a small inline test machine (a handful of channels plus scenarios),
not the full VL-1 model.
"""

import copy
import json
import shutil
from pathlib import Path

import pytest

# Shipped control_assistant simulation: shared channels + scenario bundle tree.
TEMPLATE_SIM = (
    Path(__file__).parents[2] / "src/osprey/templates/apps/control_assistant/data/simulation"
)

TEST_MACHINE = {
    "name": "TestRig",
    "description": "Tiny inline test machine",
    "channels": {
        "T:Q1:CUR:SP": {
            "value": 42.0,
            "units": "A",
            "noise": 0.0,
            "description": "Test quad current setpoint (nominal 42.0 A)",
        },
        "T:Q1:CUR:RB": {
            "expr": "ch('T:Q1:CUR:SP')",
            "units": "A",
            "noise": 0.0,
            "description": "Test quad current readback",
        },
        "T:RF:STATUS": {"value": 1, "noise": 0.0, "description": "RF status (1=on, 0=off)"},
        "T:TRANS": {
            "expr": ("max(0.0, 98.5 - 0.85 * abs(ch('T:Q1:CUR:SP') - 42.0)) * ch('T:RF:STATUS')"),
            "units": "%",
            "noise": 0.0,
            "description": "Beam transmission",
        },
        "T:MODE": {"value": "CW", "description": "Operating mode"},
        "T:NOISY": {"value": 100.0, "noise": 0.05, "description": "Noisy diagnostic"},
        "T:VAC": {"value": 8.0e-9, "units": "Torr", "noise": 0.0, "description": "Vacuum"},
    },
    "scenarios": {
        "nominal": {"description": "All systems nominal."},
        "quad-drift": {
            "description": "Q1 left at a stale setpoint.",
            "overrides": {"T:Q1:CUR:SP": 28.4},
            "archiver": [
                {
                    "channel": "T:Q1:CUR:SP",
                    "events": [{"shape": "step", "at": 0.35, "to": 28.4}],
                }
            ],
        },
        "vac-leak": {
            "description": "Vacuum leak ramps up until an RF trip.",
            "overrides": {"T:VAC": 3.8e-7, "T:RF:STATUS": 0, "T:MODE": "FAULT"},
            "archiver": [
                {
                    "channel": "T:VAC",
                    "events": [
                        {"shape": "ramp", "at": 0.15, "until": 0.88, "to": 9.5e-8},
                        {"shape": "step", "at": 0.88, "to": 3.8e-7},
                    ],
                },
                {
                    "channel": "T:RF:STATUS",
                    "events": [
                        {"shape": "spike", "at": 0.5, "amplitude": -0.2, "width": 0.01},
                        {"shape": "step", "at": 0.88, "to": 0},
                    ],
                },
                {
                    "channel": "T:MODE",
                    "events": [{"shape": "step", "at": 0.88, "to": "FAULT"}],
                },
            ],
        },
    },
}


@pytest.fixture
def machine_dict():
    """Deep copy of the inline test machine, safe to mutate per test."""
    return copy.deepcopy(TEST_MACHINE)


@pytest.fixture
def make_machine_file(tmp_path):
    """Factory writing a machine dict to a JSON file under tmp_path."""

    def _make(machine, name="machine.json"):
        path = tmp_path / name
        path.write_text(json.dumps(machine))
        return path

    return _make


@pytest.fixture
def machine_file(machine_dict, make_machine_file):
    """The inline test machine written to a tmp file."""
    return make_machine_file(machine_dict)


@pytest.fixture
def engine_factory(tmp_path):
    """Build the shipped control_assistant engine under given scenario(s).

    Copies the shipped ``machine.json`` and its ``scenarios/`` bundle tree into
    a temp dir alongside an ``active_scenarios`` state file; the engine re-reads
    the state file on mtime change, so the returned engine is already pinned to
    the requested set (``nominal`` is always implicitly active). Variadic, so
    composition can be exercised: ``make('vacuum-burst', 'rf-thermal')``.
    """
    from osprey.simulation import SimulationEngine

    def make(*names: str) -> "SimulationEngine":
        machine = tmp_path / "machine.json"
        shutil.copy(TEMPLATE_SIM / "machine.json", machine)
        shutil.copytree(TEMPLATE_SIM / "scenarios", tmp_path / "scenarios")
        active = names or ("nominal",)
        (tmp_path / "active_scenarios").write_text("\n".join(active) + "\n")
        return SimulationEngine.from_file(machine)

    return make
