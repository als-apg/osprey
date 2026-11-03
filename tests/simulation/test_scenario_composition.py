"""Scenario sets compose only on disjoint targets.

The built-in ``still`` scenario activates and composes as any listed scenario
does, and the shipped pair vacuum-burst + rf-thermal, which both script the beam
current, is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.records import FacilityRecords
from osprey.simulation.apply import apply_scenarios
from osprey_connectors.simulation.state import read_active_state
from osprey_connectors.workspace import resolve_simulation_state_dir

TEMPLATE_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"

#: The config the example's view is rendered with: every model served.
VIEW_CONFIG: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project whose render holds the example facility's simulator view, as a build writes it."""
    from osprey.facility.build import build_facility
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    root = tmp_path_factory.mktemp("project")
    doc = build_facility(TEMPLATE_FACILITY, project_name="example")
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


@pytest.mark.parametrize(
    ("scenarios", "match"),
    [
        pytest.param(
            ["still", "rf-thermal-live"], "both set the motion of", id="still-and-rf-live"
        ),
        # Both script the beam current, so the pair touches one channel twice.
        pytest.param(
            ["vacuum-burst", "rf-thermal"],
            r"SR:DIAG:DCCT:01:CURRENT:RB.*disjoint channel sets",
            id="vacuum-burst-and-rf-thermal",
        ),
    ],
)
def test_scenarios_that_clash_are_refused_together(project: Path, scenarios, match) -> None:
    apply_scenarios(project, ["still"], seed_logbook=False, seed_archive=False)

    with pytest.raises(ValueError, match=match):
        apply_scenarios(project, scenarios, seed_logbook=False, seed_archive=False)

    assert _active(project) == ["still"]
