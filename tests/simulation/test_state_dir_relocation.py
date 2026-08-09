"""Scenario state lives under ``_agent_data/``, never in the build-owned ``data/``.

``active_scenarios`` is the one simulation file that changes after a build. A
project's ``data/`` tree is re-rendered from the profile on every build and
checksummed into the manifest, so a scenario switch landing there reads as
project drift and is erased by ``osprey build --force``. These tests pin the
relocation: where the state directory resolves, and that no writer touches
``data/`` any more.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from osprey.simulation.apply import apply_scenarios
from osprey.simulation.engine import SimulationEngine, default_state_dir, resolve_state_dir

TEMPLATE_SIM = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/simulation"
)


def _make_project(tmp_path: Path, **config_extra) -> Path:
    """A sim-backed project: build-owned ``data/simulation/`` plus config.yml."""
    shutil.copytree(TEMPLATE_SIM, tmp_path / "data" / "simulation")
    config = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
        **config_extra,
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return tmp_path


class TestResolveStateDir:
    def test_defaults_under_the_agent_data_root(self, tmp_path):
        assert resolve_state_dir({}, tmp_path) == tmp_path / "_agent_data" / "simulation"

    def test_follows_a_relocated_agent_data_root(self, tmp_path):
        config = {"agent_data": {"base_dir": "./workspace"}}

        assert resolve_state_dir(config, tmp_path) == tmp_path / "workspace" / "simulation"

    def test_explicit_config_key_wins(self, tmp_path):
        config = {"simulation": {"state_dir": "run/state"}}

        assert resolve_state_dir(config, tmp_path) == tmp_path / "run" / "state"

    def test_explicit_absolute_path_is_kept(self, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        config = {"simulation": {"state_dir": str(elsewhere)}}

        assert resolve_state_dir(config, tmp_path) == elsewhere

    @pytest.mark.parametrize("bad", [None, 42, "", ["run/state"], {"path": "run/state"}])
    def test_a_mistyped_key_falls_through_to_the_default(self, tmp_path, bad):
        """Matches how ``find_runtime_write_paths_under_data`` reads the same key:
        anything but a non-empty string is not a path, so it is ignored."""
        config = {"simulation": {"state_dir": bad}}

        assert resolve_state_dir(config, tmp_path) == tmp_path / "_agent_data" / "simulation"

    def test_a_non_mapping_section_falls_through_to_the_default(self, tmp_path):
        assert (
            resolve_state_dir({"simulation": "not-a-section"}, tmp_path)
            == tmp_path / "_agent_data" / "simulation"
        )

    def test_the_config_key_is_the_one_the_drift_check_warns_about(self):
        """One spelling, or a project passes the check while still writing into data/."""
        from osprey.simulation.engine import STATE_DIR_CONFIG_KEY
        from osprey.utils.config import RUNTIME_WRITE_PATH_KEYS

        assert STATE_DIR_CONFIG_KEY in RUNTIME_WRITE_PATH_KEYS

    def test_no_config_at_all_still_resolves(self, tmp_path, monkeypatch):
        """A connector in a project with no loadable config must not crash."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        from osprey.utils.workspace import reset_config_cache

        reset_config_cache()
        try:
            assert default_state_dir() == tmp_path / "_agent_data" / "simulation"
        finally:
            reset_config_cache()


class TestEngineWritesOutsideData:
    def test_activation_writes_into_the_state_dir(self, machine_file, state_dir):
        engine = SimulationEngine.from_file(machine_file, state_dir=state_dir)

        engine.set_active_scenario("vac-leak")

        assert (state_dir / "active_scenarios").read_text().strip() == "vac-leak"
        assert not (machine_file.parent / "active_scenarios").exists()

    def test_state_dir_is_created_on_demand(self, machine_file, tmp_path):
        missing = tmp_path / "made" / "up" / "state"
        engine = SimulationEngine.from_file(machine_file, state_dir=missing)

        engine.set_active_scenario("vac-leak")

        assert (missing / "active_scenarios").exists()

    def test_a_state_file_left_beside_the_machine_is_ignored(self, machine_file, state_dir):
        """Stale state in a rebuilt ``data/`` tree must not resurrect a scenario."""
        (machine_file.parent / "active_scenarios").write_text("vac-leak\n")
        engine = SimulationEngine.from_file(machine_file, state_dir=state_dir)

        assert engine.active_scenarios() == ("nominal",)

    def test_two_state_dirs_do_not_share_a_cached_engine(self, machine_file, tmp_path):
        one, two = tmp_path / "one", tmp_path / "two"
        engine_one = SimulationEngine.from_file(machine_file, state_dir=one)
        engine_two = SimulationEngine.from_file(machine_file, state_dir=two)

        assert engine_one is not engine_two
        engine_one.set_active_scenario("vac-leak")
        assert engine_two.active_scenarios() == ("nominal",)

    def test_two_spellings_of_one_state_dir_share_a_cached_engine(self, machine_file, state_dir):
        """The cache key resolves, like the machine path — otherwise two engines
        would hold divergent session state over a single state file."""
        indirect = state_dir / ".." / state_dir.name

        assert SimulationEngine.from_file(machine_file, state_dir=state_dir) is (
            SimulationEngine.from_file(machine_file, state_dir=indirect)
        )


class TestStateFileIsReplacedAtomically:
    """The IOC polls this file from a bind mount once a second.

    A truncate-in-place write is visible to the reader mid-write; an atomic
    rename is not. The VA's ``EngineSource`` detects the swap by content, and
    the run contract in ``docker/virtual-accelerator/README.md`` promises the
    inode swap is what reaches the container.
    """

    def test_activation_replaces_the_inode_rather_than_truncating(self, machine_file, state_dir):
        engine = SimulationEngine.from_file(machine_file, state_dir=state_dir)
        engine.set_active_scenario("vac-leak")
        state_file = state_dir / "active_scenarios"
        before = state_file.stat().st_ino

        engine.set_active_scenario("quad-drift")

        assert state_file.stat().st_ino != before
        assert state_file.read_text().strip() == "quad-drift"

    def test_no_temp_file_is_left_behind(self, machine_file, state_dir):
        engine = SimulationEngine.from_file(machine_file, state_dir=state_dir)

        engine.set_active_scenario("vac-leak")

        assert [p.name for p in state_dir.iterdir()] == ["active_scenarios"]


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

        state_file = project / "_agent_data" / "simulation" / "active_scenarios"
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
