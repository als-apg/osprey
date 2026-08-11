"""``osprey sim apply``'s physics-fault stage: validate pure, write last, then notify.

The CLI is the only production path that activates a scenario, so it is where
the FR1/FR2 contract has to hold:

* every check that can reject the requested set (channel collisions reported by
  ``validate_composition``, physics-device collisions raised by the pure compute
  step) runs *before* the first ``.env`` write, and so does the purge prompt --
  a rejected set or an aborted prompt leaves the project byte-identical;
* the "run ``osprey deploy up``" notice fires exactly when the ``.env``'s
  physics block actually changed, in either direction, and only when a virtual
  accelerator is deployed to consume it.

Companion to ``tests/simulation/test_scenario_physics_render.py``, which pins the
same renderer at the function level. Between them they are the whole proof: no
e2e activates a physics fault. Every deployed scan-stack e2e boots a HEALTHY
machine -- ``_orm_stack.write_scan_env`` writes no physics block, so
``tests/e2e/test_orm_roundtrip.py`` and ``tests/e2e/test_scan_stack_agentic.py``
exercise only this mechanism's no-fault path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.sim import sim_group

# Two physics scenarios on disjoint devices, one physics-free scenario to clear
# back to, one same-device pair (physics collision) and one same-channel pair
# (override collision -- what validate_composition reports).
_SCENARIOS = {
    "corr-fault": {"physics": {"corrector_gain": {"HCM01": 1.15}}},
    "bpm-fault": {"physics": {"bpm_errors": {"BPM17": {"polarity": -1}}}},
    "no-fault": {},
    "phys-a": {"physics": {"corrector_gain": {"HCM09": 1.1}}},
    "phys-b": {"physics": {"corrector_gain": {"HCM09": 1.2}}},
    "over-a": {"overrides": {"T:BPM1:X": 1.0}},
    "over-b": {"overrides": {"T:BPM1:X": 2.0}},
}
# The one channel the override scenarios collide on -- an override must name a
# declared channel, so the machine cannot be channel-less like the pure-physics
# fixtures in tests/simulation/test_scenario_physics_render.py.
_CHANNELS = {"T:BPM1:X": {"value": 0.0, "units": "mm", "noise": 0.0}}


def _make_project(tmp_path: Path, *, va_deployed: bool = True, ariel: bool = False) -> Path:
    """Stage a sim-backed project rooted at ``tmp_path`` and return its path.

    ``va_deployed`` controls the notice's gate (``deployed_services``
    membership, not ``control_system.type`` -- the reference preset's default
    shape is mock-type *with* the VA deployed). ``ariel`` adds the config key
    that turns on the purge-confirmation prompt.
    """
    sim_dir = tmp_path / "data" / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / "machine.json").write_text(
        json.dumps({"channels": _CHANNELS, "scenarios": _SCENARIOS})
    )

    deployed = ["postgresql"] + (["virtual_accelerator"] if va_deployed else [])
    config: dict = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
        "deployed_services": deployed,
    }
    if ariel:
        config["ariel"] = {"database": {"host": "localhost"}}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return tmp_path


def _apply(project: Path, monkeypatch, *args: str, **kwargs):
    """Invoke ``sim apply`` with the project as the working directory."""
    monkeypatch.chdir(project)
    return CliRunner().invoke(sim_group, ["apply", *args], **kwargs)


def _env_lines(project: Path) -> list[str]:
    env = project / ".env"
    return (
        [ln for ln in env.read_text().splitlines() if ln.startswith("VA_")] if env.is_file() else []
    )


@pytest.fixture(autouse=True)
def _contain_env_written_by_the_cli():
    """Keep what ``sim apply`` writes to the environment inside the test that wrote it.

    ``sim apply`` builds a config *after* writing the project ``.env``, and
    ``ConfigBuilder`` loads that ``.env`` into ``os.environ`` -- a passthrough
    that is load-bearing in production (it feeds ``${VAR}`` expansion) but that
    here publishes this file's faults to every later test in the session. Left
    unchecked, ``VA_BPM_ERRORS``/``VA_CORR_GAIN`` reach the virtual
    accelerator's boot tests, where a lattice-physics fault without
    ``VA_LATTICE='builtin'`` is a fatal configuration.

    ``monkeypatch`` cannot undo this: it restores only the variables it set
    itself, and these are written by product code at runtime. Snapshotting the
    whole environment also covers the non-``VA_`` keys a hand-written ``.env``
    fixture puts on the same passthrough.
    """
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


class TestRenderAndNotice:
    """A physics scenario reaches ``.env``, and the user is told to redeploy."""

    def test_apply_renders_physics_vars_into_env(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path)

        result = _apply(project, monkeypatch, "corr-fault", "--no-seed")

        assert result.exit_code == 0, result.output
        assert _env_lines(project) == ["VA_CORR_GAIN=HCM01=1.15"]

    def test_notice_names_deploy_up_and_warns_that_restart_is_not_enough(
        self, tmp_path, monkeypatch
    ):
        project = _make_project(tmp_path)

        result = _apply(project, monkeypatch, "corr-fault", "--no-seed")

        assert "osprey deploy up" in result.output
        # A plain restart reuses the old environment -- the user has to be told
        # that specifically, or they will "restart" and see the old physics.
        assert "docker restart" in result.output

    def test_identical_reapply_emits_no_notice(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path)
        _apply(project, monkeypatch, "corr-fault", "--no-seed")

        result = _apply(project, monkeypatch, "corr-fault", "--no-seed")

        assert result.exit_code == 0, result.output
        assert "deploy up" not in result.output
        assert _env_lines(project) == ["VA_CORR_GAIN=HCM01=1.15"]

    def test_clearing_a_fault_emits_the_notice(self, tmp_path, monkeypatch):
        """A cleared fault is still live in the running container, so switching
        to a physics-free scenario has to raise the notice too."""
        project = _make_project(tmp_path)
        _apply(project, monkeypatch, "corr-fault", "--no-seed")

        result = _apply(project, monkeypatch, "no-fault", "--no-seed")

        assert result.exit_code == 0, result.output
        assert "osprey deploy up" in result.output
        assert _env_lines(project) == []

    def test_switching_faults_emits_the_notice(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path)
        _apply(project, monkeypatch, "corr-fault", "--no-seed")

        result = _apply(project, monkeypatch, "bpm-fault", "--no-seed")

        assert "osprey deploy up" in result.output
        assert _env_lines(project) == ["VA_BPM_ERRORS=BPM17:polarity_x=-1.0,polarity_y=-1.0"]

    def test_notice_is_silent_when_no_va_is_deployed(self, tmp_path, monkeypatch):
        """Nothing consumes the rendered vars, so the notice would be noise --
        but the ``.env`` is still the source of truth and is written anyway."""
        project = _make_project(tmp_path, va_deployed=False)

        result = _apply(project, monkeypatch, "corr-fault", "--no-seed")

        assert result.exit_code == 0, result.output
        assert "deploy up" not in result.output
        assert _env_lines(project) == ["VA_CORR_GAIN=HCM01=1.15"]

    def test_no_physics_apply_on_an_unnormalized_env_stays_silent(self, tmp_path, monkeypatch):
        """A hand-edited ``.env`` (no final newline) is normalized by the write.
        That is a byte change but not a physics change -- announcing "cleared
        the previous scenario's physics fault" here would name a fault that
        never existed."""
        project = _make_project(tmp_path)
        (project / ".env").write_text("A=1")  # no trailing newline

        result = _apply(project, monkeypatch, "no-fault", "--no-seed")

        assert result.exit_code == 0, result.output
        assert "deploy up" not in result.output
        assert "Cleared" not in result.output
        assert (project / ".env").read_text() == "A=1\n"  # normalized anyway

    def test_notice_survives_a_failing_apply(self, tmp_path, monkeypatch):
        """The notice is emitted between the write and ``apply_scenarios``, so a
        downstream seeding failure can never swallow it."""
        project = _make_project(tmp_path)

        with patch(
            "osprey.simulation.apply.apply_scenarios", side_effect=ValueError("logbook exploded")
        ):
            result = _apply(project, monkeypatch, "corr-fault", "--no-seed")

        assert result.exit_code != 0
        assert "osprey deploy up" in result.output
        assert _env_lines(project) == ["VA_CORR_GAIN=HCM01=1.15"]


class TestZeroWritesOnRejection:
    """A rejected set never reaches the writer."""

    def test_channel_collision_exits_nonzero_without_writing(self, tmp_path, monkeypatch):
        """``validate_composition`` *returns* its problems; the CLI is what
        turns a non-empty list into a non-zero exit."""
        project = _make_project(tmp_path)

        with patch("osprey.simulation.apply.apply_scenarios") as apply_mock:
            result = _apply(project, monkeypatch, "over-a", "over-b", "--no-seed")

        assert result.exit_code != 0
        assert "disjoint channel sets" in result.output
        assert not (project / ".env").is_file()
        apply_mock.assert_not_called()

    def test_physics_device_collision_exits_nonzero_without_writing(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path)

        with patch("osprey.simulation.apply.apply_scenarios") as apply_mock:
            result = _apply(project, monkeypatch, "phys-a", "phys-b", "--no-seed")

        assert result.exit_code != 0
        assert "disjoint physics-fault devices" in result.output
        assert not (project / ".env").is_file()
        apply_mock.assert_not_called()

    def test_collision_leaves_a_previously_rendered_fault_intact(self, tmp_path, monkeypatch):
        """The writer reconciles unconditionally, so a collision that slipped
        past the pure step would silently wipe the live fault."""
        project = _make_project(tmp_path)
        _apply(project, monkeypatch, "corr-fault", "--no-seed")
        before = (project / ".env").read_text()

        result = _apply(project, monkeypatch, "phys-a", "phys-b", "--no-seed")

        assert result.exit_code != 0
        assert (project / ".env").read_text() == before

    def test_unknown_scenario_exits_nonzero_without_writing(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path)

        result = _apply(project, monkeypatch, "no-such-scenario", "--no-seed")

        assert result.exit_code != 0
        assert "Unknown scenario" in result.output
        assert not (project / ".env").is_file()


class TestPromptAbort:
    """Answering "no" at the purge prompt leaves the project untouched."""

    @staticmethod
    def _stub_purge_info():
        async def fake_get_purge_info(_config):
            return SimpleNamespace(entry_count=7)

        return patch(
            "osprey.services.ariel_search.cli_operations.get_purge_info", fake_get_purge_info
        )

    def test_abort_writes_no_env_file(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path, ariel=True)

        with self._stub_purge_info():
            result = _apply(project, monkeypatch, "corr-fault", input="n\n")

        assert result.exit_code != 0
        assert "Aborted" in result.output
        assert not (project / ".env").is_file()

    def test_abort_leaves_a_previously_rendered_fault_intact(self, tmp_path, monkeypatch):
        project = _make_project(tmp_path, ariel=True)
        _apply(project, monkeypatch, "corr-fault", "--no-seed")
        before = (project / ".env").read_text()

        with self._stub_purge_info():
            result = _apply(project, monkeypatch, "bpm-fault", input="n\n")

        assert result.exit_code != 0
        assert (project / ".env").read_text() == before

    def test_confirming_proceeds_to_write(self, tmp_path, monkeypatch):
        """The abort test only means something if the same setup writes on "yes"."""
        project = _make_project(tmp_path, ariel=True)

        with self._stub_purge_info(), patch("osprey.simulation.apply.apply_scenarios") as mock:
            mock.return_value = SimpleNamespace(active=["nominal", "corr-fault"], logbook_seeded=3)
            result = _apply(project, monkeypatch, "corr-fault", input="y\n")

        assert result.exit_code == 0, result.output
        assert _env_lines(project) == ["VA_CORR_GAIN=HCM01=1.15"]


class TestNonSimulationProject:
    """A project with no simulation file keeps its original error path."""

    def test_non_sim_project_errors_without_writing(self, tmp_path, monkeypatch):
        (tmp_path / "config.yml").write_text(yaml.safe_dump({"control_system": {}}))

        result = _apply(tmp_path, monkeypatch, "nominal", "--no-seed")

        assert result.exit_code != 0
        assert "simulation-backed" in result.output
        assert not (tmp_path / ".env").is_file()


@pytest.mark.parametrize("names", [("corr-fault",), ("bpm-fault",), ("no-fault",)])
def test_apply_is_idempotent_on_repeat(tmp_path, monkeypatch, names):
    """Re-applying any scenario twice converges: same ``.env``, quiet the second time."""
    project = _make_project(tmp_path)
    _apply(project, monkeypatch, *names, "--no-seed")
    first = (project / ".env").read_text() if (project / ".env").is_file() else None

    result = _apply(project, monkeypatch, *names, "--no-seed")

    assert result.exit_code == 0, result.output
    assert "deploy up" not in result.output
    assert ((project / ".env").read_text() if (project / ".env").is_file() else None) == first
