"""``apply_scenarios`` reads the render's simulator view under every control-system type.

This file pins:

- ``apply_scenarios`` activates a set the simulator view lists under every
  type, and refuses a render without a view.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from osprey.simulation.apply import apply_scenarios
from tests._simulator_view import write_scenarios_view

TEMPLATE_SIM = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/simulation"
)

MOCK_CS = {
    "type": "mock",
    "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
}
MOCK_MISSING_CS = {"type": "mock", "connector": {"mock": {}}}
VA_CS = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"simulation_file": "data/simulation/machine.json"}},
}
UNKNOWN_CS = {"type": "bogus", "connector": {}}


def _stage_project(tmp_path: Path, control_system: dict) -> Path:
    """Copy the shipped simulation tree and write a config.yml with the given
    ``control_system`` block.

    The flat shape: ``config.yml`` beside ``data/``. This is what a *container*
    project directory looks like — its root is the render — and it is the shape
    the direct-API callers below are handed.
    """
    sim_dst = tmp_path / "data" / "simulation"
    sim_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_SIM, sim_dst)
    config = {"control_system": control_system}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return tmp_path


# ---------------------------------------------------------------------------
# apply_scenarios (simulation/apply.py)
# ---------------------------------------------------------------------------


class TestApplyScenariosReadsTheSimulatorView:
    @pytest.mark.parametrize("control_system", [MOCK_CS, MOCK_MISSING_CS, VA_CS, UNKNOWN_CS])
    def test_a_set_the_view_lists_is_applied_under_every_type(self, tmp_path, control_system):
        project = _stage_project(tmp_path, control_system)
        write_scenarios_view(project, {"nominal": {}, "rf-thermal": {}})

        result = apply_scenarios(project, ["rf-thermal"], seed_logbook=False)

        assert result.active == ("nominal", "rf-thermal")

    @pytest.mark.parametrize("control_system", [MOCK_CS, UNKNOWN_CS])
    def test_a_render_without_a_simulator_view_is_refused(self, tmp_path, control_system):
        project = _stage_project(tmp_path, control_system)

        with pytest.raises(ValueError) as raised:
            apply_scenarios(project, ["rf-thermal"], seed_logbook=False)

        assert str(raised.value) == (
            f"Project {project} has no simulator view in {project / 'data' / 'simulator'}; "
            "`sim apply` only applies to simulation-backed projects (guards a real DB). "
            "Run 'osprey build'."
        )
