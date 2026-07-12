"""FR5 deploy-time render step: a scenario's ``physics`` block -> VA_* ``.env`` vars.

Tests two things per bundle: the ``physics`` block parses into the ``Scenario``
(task 4.1's schema), and :func:`render_scenario_physics_env` (task 4.2) emits
the exact ``VA_QUAD_MISALIGN``/``VA_BPM_ERRORS``/``VA_CORR_GAIN`` strings the
VA entrypoint parses -- round-tripped through the entrypoint's own parse
helpers (not just asserted as a string) so the two-party contract actually
holds, not merely "looks right". Also covers backward compatibility: a bundle
with no ``physics`` block still parses and renders nothing.

Uses the two shipped discovery scenarios (``errant-quad``, ``bpm-polarity``)
as real fixtures, and ``rf-thermal`` (no ``physics`` block) for the
backward-compat case -- the same shipped ``control_assistant`` bundle tree
``test_apply_timezone.py`` stages into a temp project.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from osprey.services.virtual_accelerator import entrypoint
from osprey.simulation.apply import render_scenario_physics_env
from osprey.simulation.engine import SimulationEngine

TEMPLATE_SIM = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/simulation"
)


def _make_project(tmp_path: Path) -> Path:
    """Stage a minimal sim-backed project from the shipped bundle tree."""
    sim_dst = tmp_path / "data" / "simulation"
    sim_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_SIM, sim_dst)
    config = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return tmp_path


def _make_inline_project(tmp_path: Path, scenarios: dict) -> Path:
    """Stage a minimal sim-backed project from an inline (constructed) machine.json.

    Unlike ``_make_project``, this doesn't depend on the shipped bundle tree --
    it writes a bare-bones machine description with no ``channels`` and just
    the given ``scenarios``, for tests that need full control over a
    scenario's ``physics`` block rather than whatever the shipped fixtures
    happen to declare.
    """
    sim_dir = tmp_path / "data" / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    machine = {"channels": {}, "scenarios": scenarios}
    (sim_dir / "machine.json").write_text(json.dumps(machine))
    config = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return tmp_path


class TestErrantQuadScenario:
    """physics.quad_misalign -> VA_QUAD_MISALIGN, round-tripped through the entrypoint."""

    def test_physics_block_parses_into_scenario(self, tmp_path):
        project = _make_project(tmp_path)
        engine = SimulationEngine.from_file(project / "data/simulation/machine.json")
        engine.set_active_scenario("errant-quad")
        scenario = engine._scenarios[
            "errant-quad"
        ]  # private: no public accessor for a scenario's parsed physics block
        assert scenario.physics is not None
        assert scenario.physics.quad_misalign == {"QF07": pytest.approx(3.0e-4)}
        assert scenario.physics.bpm_errors == {}
        assert scenario.physics.corrector_gain == {}

    def test_render_emits_va_quad_misalign(self, tmp_path):
        project = _make_project(tmp_path)
        rendered = render_scenario_physics_env(project, ["errant-quad"])

        assert set(rendered) == {"VA_QUAD_MISALIGN"}
        env_text = (project / ".env").read_text()
        assert f"VA_QUAD_MISALIGN={rendered['VA_QUAD_MISALIGN']}" in env_text

    def test_rendered_value_round_trips_through_entrypoint_parser(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path)
        rendered = render_scenario_physics_env(project, ["errant-quad"])

        monkeypatch.setenv("VA_QUAD_MISALIGN", rendered["VA_QUAD_MISALIGN"])
        parsed = entrypoint._parse_device_float_map(  # exercising the entrypoint's own parse helper directly
            "VA_QUAD_MISALIGN", bound=entrypoint.MAX_QUAD_MISALIGN_DX_M
        )
        assert parsed == {"QF07": pytest.approx(3.0e-4)}


class TestBpmPolarityScenario:
    """physics.bpm_errors -> VA_BPM_ERRORS, round-tripped through the entrypoint."""

    def test_physics_block_parses_into_scenario(self, tmp_path):
        project = _make_project(tmp_path)
        engine = SimulationEngine.from_file(project / "data/simulation/machine.json")
        engine.set_active_scenario("bpm-polarity")
        scenario = engine._scenarios[
            "bpm-polarity"
        ]  # private: no public accessor for a scenario's parsed physics block
        assert scenario.physics is not None
        assert scenario.physics.bpm_errors["BPM17"].polarity == -1
        assert scenario.physics.bpm_errors["BPM17"].offset == 0.0
        assert scenario.physics.quad_misalign == {}
        assert scenario.physics.corrector_gain == {}

    def test_render_emits_va_bpm_errors_isotropic_fanout(self, tmp_path):
        project = _make_project(tmp_path)
        rendered = render_scenario_physics_env(project, ["bpm-polarity"])

        assert set(rendered) == {"VA_BPM_ERRORS"}
        # Only the non-identity field is emitted, fanned out to both planes.
        assert rendered["VA_BPM_ERRORS"] == "BPM17:polarity_x=-1.0,polarity_y=-1.0"

    def test_rendered_value_round_trips_through_entrypoint_parser(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path)
        rendered = render_scenario_physics_env(project, ["bpm-polarity"])

        monkeypatch.setenv("VA_BPM_ERRORS", rendered["VA_BPM_ERRORS"])
        parsed = (
            entrypoint._parse_bpm_errors()
        )  # exercising the entrypoint's own parse helper directly
        assert parsed == {
            "BPM17": {"polarity_x": pytest.approx(-1.0), "polarity_y": pytest.approx(-1.0)}
        }


class TestBackwardCompatibility:
    """A bundle without a ``physics`` block still parses, and renders nothing."""

    def test_scenario_without_physics_block_parses_with_none(self, tmp_path):
        project = _make_project(tmp_path)
        engine = SimulationEngine.from_file(project / "data/simulation/machine.json")
        engine.set_active_scenario("rf-thermal")
        scenario = engine._scenarios[
            "rf-thermal"
        ]  # private: no public accessor for a scenario's parsed physics block
        assert scenario.physics is None

    def test_render_of_physics_free_scenario_yields_nothing(self, tmp_path):
        project = _make_project(tmp_path)
        rendered = render_scenario_physics_env(project, ["rf-thermal"])
        assert rendered == {}

    def test_render_of_physics_free_scenario_writes_no_env_file(self, tmp_path):
        project = _make_project(tmp_path)
        render_scenario_physics_env(project, ["rf-thermal"])
        assert not (project / ".env").is_file()

    def test_nominal_only_renders_nothing(self, tmp_path):
        project = _make_project(tmp_path)
        rendered = render_scenario_physics_env(project, [])
        assert rendered == {}
        assert not (project / ".env").is_file()


class TestEnvReconciliation:
    """Re-rendering after a scenario switch clears a prior render's stale VA_* vars."""

    def test_unrelated_env_content_is_preserved(self, tmp_path):
        project = _make_project(tmp_path)
        (project / ".env").write_text("SOME_OTHER_VAR=keep-me\n")

        render_scenario_physics_env(project, ["errant-quad"])

        env_text = (project / ".env").read_text()
        assert "SOME_OTHER_VAR=keep-me" in env_text
        assert "VA_QUAD_MISALIGN=" in env_text

    def test_switching_to_a_physics_free_scenario_clears_the_prior_fault(self, tmp_path):
        project = _make_project(tmp_path)
        render_scenario_physics_env(project, ["errant-quad"])
        assert "VA_QUAD_MISALIGN=" in (project / ".env").read_text()

        rendered = render_scenario_physics_env(project, ["rf-thermal"])

        assert rendered == {}
        env_text = (project / ".env").read_text()
        assert "VA_QUAD_MISALIGN" not in env_text

    def test_switching_to_a_different_fault_replaces_not_accumulates(self, tmp_path):
        project = _make_project(tmp_path)
        render_scenario_physics_env(project, ["errant-quad"])

        rendered = render_scenario_physics_env(project, ["bpm-polarity"])

        assert set(rendered) == {"VA_BPM_ERRORS"}
        env_text = (project / ".env").read_text()
        assert "VA_QUAD_MISALIGN" not in env_text
        assert env_text.count("VA_BPM_ERRORS=") == 1


class TestErrors:
    def test_unknown_scenario_name_raises(self, tmp_path):
        project = _make_project(tmp_path)
        with pytest.raises(ValueError, match="Unknown scenario"):
            render_scenario_physics_env(project, ["no-such-scenario"])

    def test_non_simulation_backed_project_raises(self, tmp_path):
        (tmp_path / "config.yml").write_text(yaml.safe_dump({"control_system": {}}))
        with pytest.raises(ValueError, match="simulation-backed"):
            render_scenario_physics_env(tmp_path, ["nominal"])

    def test_two_active_scenarios_faulting_the_same_device_raises(self, tmp_path):
        """Mirrors validate_composition's disjointness rule for physics devices."""
        project = _make_inline_project(
            tmp_path,
            {
                "scenario-a": {"physics": {"quad_misalign": {"QF07": 1.0e-4}}},
                "scenario-b": {"physics": {"quad_misalign": {"QF07": 2.0e-4}}},
            },
        )
        with pytest.raises(ValueError, match="disjoint"):
            render_scenario_physics_env(project, ["scenario-a", "scenario-b"])


class TestEmptyBpmErrorsRenderGuard:
    """An all-identity bpm_errors device renders "", which must not become a key/line."""

    def test_all_identity_device_renders_no_va_bpm_errors_key(self, tmp_path):
        project = _make_inline_project(
            tmp_path,
            {"silent-fault": {"physics": {"bpm_errors": {"BPM01": {}}}}},
        )

        rendered = render_scenario_physics_env(project, ["silent-fault"])

        assert "VA_BPM_ERRORS" not in rendered

    def test_all_identity_device_writes_no_env_file(self, tmp_path):
        project = _make_inline_project(
            tmp_path,
            {"silent-fault": {"physics": {"bpm_errors": {"BPM01": {}}}}},
        )

        render_scenario_physics_env(project, ["silent-fault"])

        assert not (project / ".env").is_file()
