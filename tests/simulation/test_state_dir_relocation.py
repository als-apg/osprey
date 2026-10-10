"""Scenario state lives under the agent-data root, never in build-owned ``data/``.

``active_scenarios`` is the one simulation file that changes after a build. A
project's ``data/`` tree is re-rendered from the profile on every build and
checksummed into the manifest, so a scenario switch landing there reads as
project drift and is erased by ``osprey build``. These tests pin the
relocation: where the state directory resolves, and that no writer touches
``data/`` any more.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.simulation.apply import apply_scenarios
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR
from osprey_connectors.workspace import resolve_simulation_state_dir
from tests._simulator_view import write_scenarios_view
from tests.simulation.conftest import stage_sim_project


def _make_project(tmp_path: Path, **config_extra) -> Path:
    """A sim-backed project: build-owned ``data/simulation/`` plus config.yml."""
    project = stage_sim_project(tmp_path, **config_extra)
    write_scenarios_view(project, {"nominal": {}, "vacuum-burst": {}})
    return project


class TestResolveStateDir:
    def test_defaults_under_the_agent_data_root(self, tmp_path):
        assert (
            resolve_simulation_state_dir({}, tmp_path)
            == tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / "simulation"
        )

    def test_follows_a_relocated_agent_data_root(self, tmp_path):
        config = {"agent_data": {"base_dir": "./workspace"}}

        assert (
            resolve_simulation_state_dir(config, tmp_path) == tmp_path / "workspace" / "simulation"
        )

    def test_explicit_config_key_wins(self, tmp_path):
        config = {"simulation": {"state_dir": "run/state"}}

        assert resolve_simulation_state_dir(config, tmp_path) == tmp_path / "run" / "state"

    def test_explicit_absolute_path_is_kept(self, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        config = {"simulation": {"state_dir": str(elsewhere)}}

        assert resolve_simulation_state_dir(config, tmp_path) == elsewhere

    @pytest.mark.parametrize("bad", [None, 42, "", ["run/state"], {"path": "run/state"}])
    def test_a_mistyped_key_falls_through_to_the_default(self, tmp_path, bad):
        """Matches how ``find_runtime_write_paths_under_data`` reads the same key:
        anything but a non-empty string is not a path, so it is ignored."""
        config = {"simulation": {"state_dir": bad}}

        assert (
            resolve_simulation_state_dir(config, tmp_path)
            == tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / "simulation"
        )

    def test_a_non_mapping_section_falls_through_to_the_default(self, tmp_path):
        assert (
            resolve_simulation_state_dir({"simulation": "not-a-section"}, tmp_path)
            == tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / "simulation"
        )

    def test_the_config_key_is_the_one_the_drift_check_warns_about(self):
        """One spelling, or a project passes the check while still writing into data/."""
        from osprey.utils.config import RUNTIME_WRITE_PATH_KEYS
        from osprey_connectors.workspace import SIMULATION_STATE_DIR_CONFIG_KEY

        assert SIMULATION_STATE_DIR_CONFIG_KEY in RUNTIME_WRITE_PATH_KEYS


class TestApplyLeavesDataUntouched:
    @pytest.fixture
    def project(self, tmp_path):
        return _make_project(tmp_path)

    def _snapshot(self, root: Path) -> dict[str, bytes]:
        return {
            str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
        }

    def test_apply_writes_state_under_agent_data(self, project):
        apply_scenarios(project, ["vacuum-burst"], seed_logbook=False)

        state_file = project / DEFAULT_AGENT_DATA_BASE_DIR / "simulation" / "active_scenarios"
        assert state_file.exists()
        assert "vacuum-burst" in state_file.read_text()

    def test_apply_does_not_modify_the_data_tree(self, project):
        before = self._snapshot(project / "data")

        apply_scenarios(project, ["vacuum-burst"], seed_logbook=False)

        assert self._snapshot(project / "data") == before

    def test_apply_honours_an_explicit_state_dir(self, tmp_path):
        project = _make_project(tmp_path, simulation={"state_dir": "run/scenarios"})

        apply_scenarios(project, ["vacuum-burst"], seed_logbook=False)

        assert (project / "run" / "scenarios" / "active_scenarios").exists()
