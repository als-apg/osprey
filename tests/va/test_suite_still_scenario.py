"""The VA live-container suite starts every container in its own scenario set.

The suite's containers serve monitor readings without their declared motion, so
a monitor reading is the solved orbit. That stilling is a scenario the suite
adds beside the shipped ones and activates the product's way; the staged
``nominal`` scenario is the shipped one. This module renders what the suite's
containers mount and checks both, without Docker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey_connectors.simulation.state import read_active_state, resolve_active_scenarios
from osprey_connectors.simulation.view import VIEW_RELPATH, SimulatorView
from tests.va.e2e import conftest as e2e_conftest

SHIPPED_NOMINAL = e2e_conftest.PRESET_FACILITY_DIR / "scenarios" / "nominal.yaml"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> e2e_conftest.VaProject:
    """The deployment repo the suite's session container serves."""
    return e2e_conftest.stage_va_project(tmp_path_factory.mktemp("suite_still"))


@pytest.fixture(scope="module")
def view(project: e2e_conftest.VaProject) -> SimulatorView:
    return SimulatorView.open(project.data_dir / VIEW_RELPATH.name)


@pytest.fixture(scope="module")
def monitors() -> list[str]:
    return e2e_conftest._monitor_addresses(e2e_conftest.PRESET_FACILITY_DIR)


def test_the_session_project_starts_in_the_suite_set(project: e2e_conftest.VaProject) -> None:
    active = resolve_active_scenarios(read_active_state(project.state_dir)[0])

    assert active == ["nominal", *e2e_conftest.SUITE_SCENARIOS]


def test_the_staged_nominal_scenario_is_the_shipped_one(
    project: e2e_conftest.VaProject,
) -> None:
    from osprey.simulation.apply import view_scenarios

    scenarios = view_scenarios(project.project_dir)
    assert scenarios is not None
    nominal = {key: value for key, value in scenarios["nominal"].items() if key != "name"}

    assert "still" not in nominal
    assert nominal == yaml.safe_load(SHIPPED_NOMINAL.read_text(encoding="utf-8"))


def test_the_suite_scenario_stills_exactly_the_monitor_readings(
    view: SimulatorView, monitors: list[str]
) -> None:
    scenarios = {str(scenario["name"]): scenario for scenario in view.scenarios()}

    assert list(scenarios[e2e_conftest.STILL_MONITORS_SCENARIO_NAME]["still"]) == monitors


def test_every_monitor_is_still_under_the_suite_set(
    view: SimulatorView, monitors: list[str]
) -> None:
    moving = {
        address: band
        for address in monitors
        if (band := view.motion_envelope(address, active=e2e_conftest.SUITE_SCENARIOS)) != 0.0
    }

    assert monitors
    assert moving == {}


def test_a_monitor_moves_under_nominal_alone(view: SimulatorView, monitors: list[str]) -> None:
    assert any(view.motion_envelope(address, active=()) != 0.0 for address in monitors)


def test_the_demo_data_mount_carries_the_suite_state() -> None:
    args = e2e_conftest.demo_data_run_args()
    mounts = [args[index + 1] for index, arg in enumerate(args) if arg == "-v"]
    state = [
        Path(mount.split(":")[0]) for mount in mounts if mount.endswith(":/state/simulation:ro")
    ]

    assert len(state) == 1
    assert "VA_STATE_DIR=/state/simulation" in args
    active = resolve_active_scenarios(read_active_state(state[0])[0])
    assert active == ["nominal", *e2e_conftest.SUITE_SCENARIOS]
