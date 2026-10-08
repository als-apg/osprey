"""``apply_scenarios`` reads the render's simulator view under every control-system type.

This file pins:

- ``apply_scenarios`` activates a set the simulator view lists under every
  type, and refuses a render without a view.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.simulation.apply import apply_scenarios
from tests._simulator_view import write_scenarios_view
from tests.simulation.conftest import stage_sim_project

MOCK_CS = {"type": "mock", "connector": {"mock": {}}}
VA_CS = {"type": "virtual_accelerator", "connector": {"virtual_accelerator": {}}}
UNKNOWN_CS = {"type": "bogus", "connector": {}}


def _stage_project(tmp_path: Path, control_system: dict) -> Path:
    """Write a config.yml with the given ``control_system`` block.

    The flat shape: ``config.yml`` beside ``data/``. This is what a *container*
    project directory looks like — its root is the render — and it is the shape
    the direct-API callers below are handed.
    """
    return stage_sim_project(tmp_path, control_system=control_system)


# ---------------------------------------------------------------------------
# apply_scenarios (simulation/apply.py)
# ---------------------------------------------------------------------------


class TestApplyScenariosReadsTheSimulatorView:
    @pytest.mark.parametrize("control_system", [MOCK_CS, VA_CS, UNKNOWN_CS])
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
