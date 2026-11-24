"""Applying a scenario that stops an IOC the render cannot stop is refused.

The simulator view keeps the name of an IOC whose model the render does not serve
out of the scenario's ``stopped_servers`` and lists it under ``inactive_servers``;
``apply_scenarios`` refuses the scenario rather than activate it without the
stop it describes, and writes no state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.simulation.apply import apply_scenarios
from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME
from osprey_connectors.workspace import resolve_simulation_state_dir
from tests._simulator_view import write_scenarios_view


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_scenarios_view(
        tmp_path,
        {
            "ioc-down": {
                "description": "Stops an IOC whose model this render does not serve.",
                "stopped_servers": [],
                "inactive_servers": {"line-ioc": "model not served"},
            },
            "ioc-served": {
                "description": "Stops an IOC this render serves.",
                "stopped_servers": ["ring-ioc"],
            },
        },
    )
    (tmp_path / "config.yml").write_text(yaml.safe_dump({}), encoding="utf-8")
    return tmp_path


def _state_file(project: Path) -> Path:
    return resolve_simulation_state_dir({}, project) / ACTIVE_SCENARIOS_FILENAME


def test_applying_a_scenario_stopping_an_unserved_server_is_refused(project: Path) -> None:
    with pytest.raises(
        ValueError, match=r"'ioc-down' stops IOC\(s\).*line-ioc \(model not served\)"
    ):
        apply_scenarios(project, ["ioc-down"], seed_logbook=False, seed_archive=False)

    assert not _state_file(project).exists()


def test_the_refusal_holds_inside_a_set(project: Path) -> None:
    with pytest.raises(ValueError, match="line-ioc"):
        apply_scenarios(project, ["ioc-served", "ioc-down"], seed_logbook=False, seed_archive=False)

    assert not _state_file(project).exists()


def test_a_scenario_stopping_only_served_servers_applies(project: Path) -> None:
    result = apply_scenarios(project, ["ioc-served"], seed_logbook=False, seed_archive=False)

    assert result.active == ("nominal", "ioc-served")
