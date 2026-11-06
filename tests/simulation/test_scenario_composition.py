"""Scenario sets compose only on disjoint targets.

The built-in ``still`` scenario activates and composes as any listed scenario
does. Archiver events are not targets, so the shipped pair vacuum-burst +
rf-thermal, which both script the beam current, composes; still + rf-thermal-live,
which disagree on the motion of one reading, is refused.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.records import FacilityRecords
from osprey.simulation.apply import apply_scenarios, view_scenarios
from osprey_connectors.simulation.state import (
    read_active_state,
    scenario_targets,
    validate_composition,
)
from osprey_connectors.workspace import resolve_simulation_state_dir
from tests._facility_file import demo_facility_document

TEMPLATE_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"

#: The config the example's view is rendered with: every model served.
VIEW_CONFIG: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project whose render holds the example facility's simulator view, as a build writes it."""
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    root = tmp_path_factory.mktemp("project")
    doc = demo_facility_document()
    write_simulator_view(
        root / "data" / "simulator",
        ViewInputs(
            records=FacilityRecords(doc),
            rendered_config=VIEW_CONFIG,
            facility_dir=TEMPLATE_FACILITY,
            served=resolve_served(VIEW_CONFIG, FacilityRecords(doc)),
            reported=None,
        ),
    )
    (root / "config.yml").write_text(yaml.safe_dump({}), encoding="utf-8")
    return root


def _active(project: Path) -> list[str]:
    return read_active_state(resolve_simulation_state_dir({}, project))[0]


def test_sim_apply_activates_the_built_in_still(project: Path) -> None:
    result = apply_scenarios(project, ["still"], seed_logbook=False, seed_archive=False)

    assert result.active == ("nominal", "still")
    assert _active(project) == ["still"]


def test_scenarios_that_clash_are_refused_together(project: Path) -> None:
    apply_scenarios(project, ["still"], seed_logbook=False, seed_archive=False)

    with pytest.raises(ValueError, match="both set the motion of"):
        apply_scenarios(
            project, ["still", "rf-thermal-live"], seed_logbook=False, seed_archive=False
        )

    assert _active(project) == ["still"]


def test_vacuum_burst_and_rf_thermal_compose(project: Path) -> None:
    # Both script the beam current; the archive merges their events in set order.
    result = apply_scenarios(
        project, ["vacuum-burst", "rf-thermal"], seed_logbook=False, seed_archive=False
    )

    assert result.active == ("nominal", "vacuum-burst", "rf-thermal")
    assert _active(project) == ["vacuum-burst", "rf-thermal"]


def test_every_pair_that_composed_with_archiver_channels_as_targets_still_composes(
    project: Path,
) -> None:
    scenarios = view_scenarios(project)
    assert scenarios
    targets = {name: scenario_targets(scenario) for name, scenario in scenarios.items()}
    # The rule before archiver events composed: each entry's channel was a target too.
    before = {
        name: targets[name] | {str(entry["channel"]) for entry in scenario.get("archiver") or []}
        for name, scenario in scenarios.items()
    }

    composed = [
        pair
        for pair in combinations(sorted(scenarios), 2)
        if not validate_composition(before, pair)
    ]

    assert composed
    assert [pair for pair in composed if validate_composition(targets, pair)] == []
