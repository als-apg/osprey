"""Type-aware simulation-file lookup for `osprey sim apply`.

``osprey_connectors.simulation.engine.resolve_simulation_file`` resolves the
simulation-model file for the active ``control_system.type``;
``apply_scenarios`` (simulation/apply.py) reads the render's simulator view
instead, whatever the type. This file pins:

- mock resolution reads ``connector.mock.simulation_file``, and a missing key
  is refused with the mock wording.
- VA resolution reads ``connector.virtual_accelerator.simulation_file``, and
  falls back to the mock key when its own key is unset (the mock fallback is
  a real key, not an informational one).
- an unknown/unsupported ``control_system.type`` fails cleanly, naming both
  the type-specific key and the mock fallback key that were tried.
- ``apply_scenarios`` activates a set the simulator view lists under every
  type, and refuses a render without a view.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from osprey.simulation.apply import apply_scenarios
from osprey_connectors.simulation.engine import resolve_simulation_file
from tests._simulator_view import write_scenarios_view

TEMPLATE_SIM = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/simulation"
)

MOCK_TYPE_KEY = "control_system.connector.mock.simulation_file"

MOCK_CS = {
    "type": "mock",
    "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
}
MOCK_MISSING_CS = {"type": "mock", "connector": {"mock": {}}}
VA_CS = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"simulation_file": "data/simulation/machine.json"}},
}
VA_FALLS_BACK_CS = {
    "type": "virtual_accelerator",
    "connector": {
        "virtual_accelerator": {},
        "mock": {"simulation_file": "data/simulation/machine.json"},
    },
}
VA_MISSING_CS = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {}, "mock": {}},
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
# resolve_simulation_file (the shared helper)
# ---------------------------------------------------------------------------


class TestResolveSimulationFile:
    def test_mock_resolves_from_mock_key(self, tmp_path):
        project = _stage_project(tmp_path, MOCK_CS)
        path, active_type, type_key, mock_key = resolve_simulation_file(
            {"control_system": MOCK_CS}, project
        )
        assert path == project / "data/simulation/machine.json"
        assert active_type == "mock"
        assert type_key == MOCK_TYPE_KEY
        assert mock_key == MOCK_TYPE_KEY

    def test_mock_missing_key_returns_none(self, tmp_path):
        path, active_type, _, _ = resolve_simulation_file(
            {"control_system": MOCK_MISSING_CS}, tmp_path
        )
        assert path is None
        assert active_type == "mock"

    def test_type_defaults_to_mock_when_unset(self, tmp_path):
        """No `control_system.type` key at all -> resolves as mock (today's implicit default)."""
        config = {
            "control_system": {
                "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
            }
        }
        path, active_type, _, _ = resolve_simulation_file(config, tmp_path)
        assert active_type == "mock"
        assert path == tmp_path / "data/simulation/machine.json"

    def test_va_resolves_from_va_key(self, tmp_path):
        project = _stage_project(tmp_path, VA_CS)
        path, active_type, type_key, mock_key = resolve_simulation_file(
            {"control_system": VA_CS}, project
        )
        assert path == project / "data/simulation/machine.json"
        assert active_type == "virtual_accelerator"
        assert type_key == "control_system.connector.virtual_accelerator.simulation_file"
        assert mock_key == MOCK_TYPE_KEY

    def test_va_falls_back_to_mock_key_when_own_key_unset(self, tmp_path):
        project = _stage_project(tmp_path, VA_FALLS_BACK_CS)
        path, active_type, _, _ = resolve_simulation_file(
            {"control_system": VA_FALLS_BACK_CS}, project
        )
        assert path == project / "data/simulation/machine.json"
        assert active_type == "virtual_accelerator"

    def test_va_missing_both_keys_returns_none(self, tmp_path):
        path, active_type, type_key, mock_key = resolve_simulation_file(
            {"control_system": VA_MISSING_CS}, tmp_path
        )
        assert path is None
        assert active_type == "virtual_accelerator"
        assert type_key == "control_system.connector.virtual_accelerator.simulation_file"
        assert mock_key == MOCK_TYPE_KEY

    def test_unknown_type_returns_none_naming_both_keys(self, tmp_path):
        path, active_type, type_key, mock_key = resolve_simulation_file(
            {"control_system": UNKNOWN_CS}, tmp_path
        )
        assert path is None
        assert active_type == "bogus"
        assert type_key == "control_system.connector.bogus.simulation_file"
        assert mock_key == MOCK_TYPE_KEY


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
