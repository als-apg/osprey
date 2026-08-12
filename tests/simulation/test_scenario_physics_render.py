"""FR5 deploy-time render step: a scenario's ``physics`` block -> VA_* ``.env`` vars.

Tests two things per bundle: the ``physics`` block parses into the ``Scenario``
(task 4.1's schema), and :func:`render_scenario_physics_env` (task 4.2) emits
the exact ``VA_BPM_ERRORS``/``VA_CORR_GAIN`` strings the
VA entrypoint parses -- round-tripped through the entrypoint's own parse
helpers (not just asserted as a string) so the two-party contract actually
holds, not merely "looks right". Also covers backward compatibility: a bundle
with no ``physics`` block still parses and renders nothing.

Uses the shipped ``bpm-polarity`` discovery scenario as a real fixture, and
``rf-thermal`` (no ``physics`` block) for the backward-compat case -- the same
shipped ``control_assistant`` bundle tree ``test_apply_timezone.py`` stages
into a temp project. ``corrector_gain`` has no shipped scenario fixture, so
its coverage uses ``_make_inline_project``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from osprey.services.virtual_accelerator import entrypoint
from osprey.simulation.apply import (
    compute_scenario_physics_env,
    render_scenario_physics_env,
    write_scenario_physics_env,
)
from osprey.simulation.engine import SimulationEngine, resolve_active_scenarios

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


class TestOrmDualFaultScenario:
    """physics.bpm_errors + physics.corrector_gain together on disjoint devices.

    ``orm-dual-fault`` is the combined-fault bundle the ORM agentic e2e
    activates: a BPM 17 polarity flip (reusing ``bpm-polarity``'s shape) plus
    a bounded HCM01 gain deficit, on two distinct devices so the disjoint-
    device claim in ``_render_physics_vars`` never trips within one scenario.
    """

    def test_physics_block_parses_both_faults(self, tmp_path):
        project = _make_project(tmp_path)
        engine = SimulationEngine.from_file(project / "data/simulation/machine.json")
        engine.set_active_scenario("orm-dual-fault")
        scenario = engine._scenarios[
            "orm-dual-fault"
        ]  # private: no public accessor for a scenario's parsed physics block
        assert scenario.physics is not None
        assert scenario.physics.bpm_errors["BPM17"].polarity == -1
        assert scenario.physics.corrector_gain == {"HCM01": 0.5}

    def test_render_emits_both_va_vars_on_disjoint_devices(self, tmp_path):
        project = _make_project(tmp_path)
        rendered = render_scenario_physics_env(project, ["orm-dual-fault"])

        assert set(rendered) == {"VA_BPM_ERRORS", "VA_CORR_GAIN"}
        assert rendered["VA_BPM_ERRORS"] == "BPM17:polarity_x=-1.0,polarity_y=-1.0"
        assert rendered["VA_CORR_GAIN"] == "HCM01=0.5"

    def test_rendered_values_round_trip_through_entrypoint_parsers(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path)
        rendered = render_scenario_physics_env(project, ["orm-dual-fault"])

        monkeypatch.setenv("VA_BPM_ERRORS", rendered["VA_BPM_ERRORS"])
        monkeypatch.setenv("VA_CORR_GAIN", rendered["VA_CORR_GAIN"])
        bpm_errors = (
            entrypoint._parse_bpm_errors()
        )  # exercising the entrypoint's own parse helpers directly
        corr_gain = entrypoint._parse_device_float_map(
            "VA_CORR_GAIN", bound=entrypoint.MAX_CORR_GAIN_FACTOR
        )
        assert bpm_errors == {
            "BPM17": {"polarity_x": pytest.approx(-1.0), "polarity_y": pytest.approx(-1.0)}
        }
        assert corr_gain == {"HCM01": pytest.approx(0.5)}

    def test_disjoint_devices_satisfy_the_claim_guard_alongside_another_fault(self, tmp_path):
        """A second active scenario faulting different devices doesn't trip
        ``_render_physics_vars``'s disjoint-device claim -- mirrors
        ``TestErrors``'s same-device negative case, but disjoint."""
        project = _make_inline_project(
            tmp_path,
            {
                "orm-dual-fault": {
                    "physics": {
                        "bpm_errors": {"BPM17": {"polarity": -1}},
                        "corrector_gain": {"HCM01": 0.5},
                    }
                },
                "other-fault": {"physics": {"corrector_gain": {"HCM02": 0.8}}},
            },
        )
        rendered = render_scenario_physics_env(project, ["orm-dual-fault", "other-fault"])

        assert rendered["VA_BPM_ERRORS"] == "BPM17:polarity_x=-1.0,polarity_y=-1.0"
        assert rendered["VA_CORR_GAIN"] == "HCM01=0.5,HCM02=0.8"


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


_RECONCILIATION_SCENARIOS = {
    "corr-fault": {"physics": {"corrector_gain": {"HCM01": 1.15}}},
    "bpm-fault": {"physics": {"bpm_errors": {"BPM17": {"polarity": -1}}}},
    "no-fault": {},
}


class TestEnvReconciliation:
    """Re-rendering after a scenario switch clears a prior render's stale VA_* vars."""

    def test_unrelated_env_content_is_preserved(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        (project / ".env").write_text("SOME_OTHER_VAR=keep-me\n")

        render_scenario_physics_env(project, ["corr-fault"])

        env_text = (project / ".env").read_text()
        assert "SOME_OTHER_VAR=keep-me" in env_text
        assert "VA_CORR_GAIN=" in env_text

    def test_switching_to_a_physics_free_scenario_clears_the_prior_fault(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        render_scenario_physics_env(project, ["corr-fault"])
        assert "VA_CORR_GAIN=" in (project / ".env").read_text()

        rendered = render_scenario_physics_env(project, ["no-fault"])

        assert rendered == {}
        env_text = (project / ".env").read_text()
        assert "VA_CORR_GAIN" not in env_text

    def test_switching_to_a_different_fault_replaces_not_accumulates(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        render_scenario_physics_env(project, ["corr-fault"])

        rendered = render_scenario_physics_env(project, ["bpm-fault"])

        assert set(rendered) == {"VA_BPM_ERRORS"}
        env_text = (project / ".env").read_text()
        assert "VA_CORR_GAIN" not in env_text
        assert env_text.count("VA_BPM_ERRORS=") == 1


def _apply_physics(project: Path, names: list[str]) -> bool:
    """Compute then write, the way ``osprey sim apply`` composes the two steps.

    Returns the writer's changed signal.
    """
    return write_scenario_physics_env(project, compute_scenario_physics_env(project, names))


class TestSharedActiveSetResolution:
    """The engine and the renderer resolve one active set, from one definition.

    ``osprey sim apply`` validates and renders the resolved set *before*
    ``set_active_scenarios`` activates it. If the two resolutions could drift,
    the CLI would validate one set and apply another -- the exact failure FR1's
    ordering exists to prevent -- and nothing downstream would notice.
    """

    @pytest.mark.parametrize(
        "names",
        [
            [],
            ["corr-fault"],
            ["corr-fault", "bpm-fault"],
            ["nominal", "corr-fault"],  # explicit nominal must not duplicate
            ["corr-fault", "corr-fault"],  # repeat must collapse
        ],
    )
    def test_engine_validates_exactly_the_resolver_set(self, tmp_path, names, monkeypatch):
        """Spy on ``validate_composition`` rather than on the returned active
        tuple: the return value is re-derived from the state file, which
        normalizes ordering and would hide a divergence. The set handed to
        validation is the one that matters -- it is what the CLI renders
        physics for.

        Only the *first* call is the resolved set; activation validates again
        after re-reading the state file it just wrote.
        """
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        engine = SimulationEngine.from_file(project / "data/simulation/machine.json")
        seen: list[list[str]] = []
        real_validate = engine.validate_composition

        def spy(candidate):
            seen.append(list(candidate))
            return real_validate(candidate)

        monkeypatch.setattr(engine, "validate_composition", spy)

        engine.set_active_scenarios(names)

        assert seen, "set_active_scenarios never validated the set it activates"
        assert seen[0] == resolve_active_scenarios(names)

    @pytest.mark.parametrize(
        "names",
        [
            [],
            ["corr-fault"],
            ["corr-fault", "bpm-fault"],
            ["nominal", "corr-fault"],  # explicit nominal must not duplicate
            ["corr-fault", "corr-fault"],  # repeat must collapse
        ],
    )
    def test_engine_activates_exactly_what_the_resolver_returns(self, tmp_path, names):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        engine = SimulationEngine.from_file(project / "data/simulation/machine.json")

        active = engine.set_active_scenarios(names)

        assert list(active) == resolve_active_scenarios(names)

    def test_duplicates_and_explicit_nominal_collapse(self):
        assert resolve_active_scenarios(["a", "a", "nominal", "b"]) == ["nominal", "a", "b"]


class TestComputeIsPure:
    """The compute step validates and renders without touching the filesystem.

    ``osprey sim apply`` runs it ahead of the purge-confirmation prompt so that a
    rejected set -- or a user who answers "no" -- leaves the project untouched;
    that guarantee only holds if computing really writes nothing.
    """

    def test_compute_writes_no_env_file(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)

        rendered = compute_scenario_physics_env(project, ["corr-fault"])

        assert rendered == {"VA_CORR_GAIN": "HCM01=1.15"}
        assert not (project / ".env").is_file()

    def test_compute_matches_what_the_composed_entry_point_writes(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)

        computed = compute_scenario_physics_env(project, ["corr-fault"])
        written = render_scenario_physics_env(project, ["corr-fault"])

        assert computed == written

    def test_device_collision_raises_without_writing(self, tmp_path):
        project = _make_inline_project(
            tmp_path,
            {
                "scenario-a": {"physics": {"corrector_gain": {"HCM01": 1.1}}},
                "scenario-b": {"physics": {"corrector_gain": {"HCM01": 1.2}}},
            },
        )
        (project / ".env").write_text("SOME_OTHER_VAR=keep-me\n")

        with pytest.raises(ValueError, match="disjoint"):
            compute_scenario_physics_env(project, ["scenario-a", "scenario-b"])

        assert (project / ".env").read_text() == "SOME_OTHER_VAR=keep-me\n"

    def test_unknown_scenario_raises_without_writing(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)

        with pytest.raises(ValueError, match="Unknown scenario"):
            compute_scenario_physics_env(project, ["no-such-scenario"])

        assert not (project / ".env").is_file()


class TestChangedSignal:
    """The writer reports whether ``.env`` *content* changed, not that it wrote.

    It reconciles unconditionally, so "a write happened" would be true on every
    call. ``osprey sim apply`` gates its "run osprey up" notice on this
    signal, so re-applying the same scenario must stay quiet while clearing a
    stale fault must not.
    """

    def test_first_render_reports_changed(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        assert _apply_physics(project, ["corr-fault"]) is True

    def test_identical_re_render_reports_unchanged(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        _apply_physics(project, ["corr-fault"])

        assert _apply_physics(project, ["corr-fault"]) is False

    def test_switching_to_a_different_fault_reports_changed(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        _apply_physics(project, ["corr-fault"])

        assert _apply_physics(project, ["bpm-fault"]) is True

    def test_clearing_a_prior_fault_reports_changed(self, tmp_path):
        """A cleared fault is still live in the running VA -- it must report
        changed even though nothing is rendered."""
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        _apply_physics(project, ["corr-fault"])

        assert _apply_physics(project, ["no-fault"]) is True

    def test_clearing_when_already_clean_reports_unchanged(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        _apply_physics(project, ["corr-fault"])
        _apply_physics(project, ["no-fault"])

        assert _apply_physics(project, ["no-fault"]) is False

    def test_physics_free_scenario_on_a_fresh_project_reports_unchanged(self, tmp_path):
        """Nothing to render and no ``.env`` to clean up: no write, no change."""
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)

        assert _apply_physics(project, ["no-fault"]) is False
        assert not (project / ".env").is_file()

    def test_re_render_preserves_unrelated_content_byte_for_byte(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        (project / ".env").write_text("SOME_OTHER_VAR=keep-me\n")
        _apply_physics(project, ["corr-fault"])
        first = (project / ".env").read_text()

        _apply_physics(project, ["corr-fault"])

        assert (project / ".env").read_text() == first

    def test_re_render_does_not_accumulate_the_block_header(self, tmp_path):
        """The header comment is owned by the writer, so it is reconciled like
        the keys under it -- otherwise every rewrite would append a fresh one
        and no re-render could ever compare equal."""
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        _apply_physics(project, ["corr-fault"])
        _apply_physics(project, ["corr-fault"])

        assert (project / ".env").read_text().count("# Scenario physics fault") == 1

    def test_incidental_normalization_alone_reports_unchanged(self, tmp_path):
        """A hand-edited ``.env`` with no final newline gets normalized on the
        next write. That is a byte change but not a *physics* change, and
        reporting it would make ``sim apply`` announce a fault that never
        existed -- so the signal compares the physics block, not the file."""
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        (project / ".env").write_text("A=1")  # no trailing newline

        changed = _apply_physics(project, ["no-fault"])

        assert changed is False
        assert (project / ".env").read_text() == "A=1\n"  # still normalized

    def test_trailing_blank_lines_alone_report_unchanged(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        (project / ".env").write_text("A=1\n\n\n")

        assert _apply_physics(project, ["no-fault"]) is False

    def test_normalization_does_not_mask_a_real_physics_change(self, tmp_path):
        """The same un-normalized file *with* a fault to render still reports
        changed -- the fix must not swallow the signal it exists to carry."""
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        (project / ".env").write_text("A=1")

        assert _apply_physics(project, ["corr-fault"]) is True

    def test_signal_ignores_whitespace_around_an_unchanged_value(self, tmp_path):
        """A hand-spaced ``VA_CORR_GAIN = HCM01=1.15`` names the same fault."""
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        (project / ".env").write_text("VA_CORR_GAIN = HCM01=1.15\n")

        assert _apply_physics(project, ["corr-fault"]) is False

    def test_clearing_removes_the_block_header_too(self, tmp_path):
        project = _make_inline_project(tmp_path, _RECONCILIATION_SCENARIOS)
        (project / ".env").write_text("SOME_OTHER_VAR=keep-me\n")
        _apply_physics(project, ["corr-fault"])

        _apply_physics(project, ["no-fault"])

        assert (project / ".env").read_text() == "SOME_OTHER_VAR=keep-me\n"


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
                "scenario-a": {"physics": {"corrector_gain": {"HCM01": 1.1}}},
                "scenario-b": {"physics": {"corrector_gain": {"HCM01": 1.2}}},
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
