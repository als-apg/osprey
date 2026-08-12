"""``data/`` is byte-stable across every runtime writer (FR-8b regression).

A project's ``data/`` tree is build-owned: rendered from the profile, checksummed
into ``.osprey-manifest.json``, and cleared by ``osprey build``. Anything
written there while the system runs therefore reads as project drift and is
erased by the next forced rebuild — silently taking an operator's scenario
selection or accumulated channel-finder feedback with it.

Unit tests pin each writer's own target directory. This one pins the property
that matters across all of them at once: build a real project, run every known
runtime writer against it, and assert :func:`calculate_file_checksums` reports
byte-identical ``data/`` entries afterwards. A new runtime writer that defaults
into ``data/`` fails here even if nobody thought to write it its own test.

Each test also asserts its writer actually wrote something under the agent-data
root — an assertion that nothing changed is worthless if the writer no-opped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey.cli.templates.manifest import calculate_file_checksums
from osprey.interfaces.channel_finder.app import FEEDBACK_DIR
from osprey.utils.config import (
    ConfigBuilder,
    find_runtime_write_paths_under_data,
)
from osprey.utils.workspace import agent_data_base_dir

PRESET = "control-assistant"
PROJECT_NAME = "checksum-stability"
_PRESET_PATH = Path(__file__).resolve().parents[2] / "src/osprey/profiles/presets" / f"{PRESET}.yml"


def _data_checksums(project: Path) -> dict[str, str]:
    """The build-owned subset of the project's checksums."""
    return {
        path: digest
        for path, digest in calculate_file_checksums(project).items()
        if Path(path).parts[0] == "data"
    }


@pytest.fixture(scope="module")
def built_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real render, shared by every test here.

    The RENDER, not the repo: ``data/`` is build output, and what these tests
    watch is whether anything writes back into it after the build.

    Built once. The writers below are only ever supposed to touch the agent-data
    root, so they cannot interfere with each other's ``data/`` assertions, and a
    per-test rebuild would pay the build cost for nothing.
    """
    repo = tmp_path_factory.mktemp("projects") / PROJECT_NAME

    created = CliRunner().invoke(init, [str(repo), "--preset", PRESET, "--no-git"])
    assert created.exit_code == 0, created.output

    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo / "build"


@pytest.fixture(scope="module")
def baseline(built_project: Path) -> dict[str, str]:
    """``data/`` checksums as the build left them."""
    checksums = _data_checksums(built_project)
    assert checksums, "the built project has no data/ tree — this test would be vacuous"
    return checksums


@pytest.fixture
def project_config(built_project: Path) -> dict:
    return yaml.safe_load((built_project / "config.yml").read_text(encoding="utf-8"))


class TestTheBuildItself:
    def test_no_scenario_state_ships_in_data(self, built_project: Path) -> None:
        assert not (built_project / "data" / "simulation" / "active_scenarios").exists()

    def test_the_shipped_config_points_no_runtime_writer_into_data(
        self, built_project: Path, project_config: dict
    ) -> None:
        """The warning 6.1 added must stay silent for a project we ship."""
        assert find_runtime_write_paths_under_data(project_config, built_project) == []


class TestRuntimeWriters:
    """Every writer that runs after the build, against one real project."""

    def test_activating_a_scenario_leaves_data_untouched(
        self, built_project: Path, baseline: dict[str, str], project_config: dict
    ) -> None:
        from osprey.simulation.apply import apply_scenarios
        from osprey.utils.workspace import resolve_simulation_state_dir

        # Both side-seedings are off for the same reason: each one reaches a
        # service this test does not run and is not about. `seed_logbook` wants
        # ARIEL's Postgres; `seed_archive` wants the MongoDB store the
        # control-assistant preset now declares, and refuses outright without
        # the password `osprey up` mints -- correctly, since a rewrite it
        # cannot perform would leave the archive contradicting the scenario.
        # What is under test here is where the ACTIVATION writes: the state file
        # belongs in `_agent_data/`, never in the build-owned `data/` tree.
        apply_scenarios(built_project, ["vacuum-burst"], seed_logbook=False, seed_archive=False)

        # Resolved from the config rather than spelled out, because where the
        # agent-data root sits is a config decision: pinning the literal here
        # would turn a deliberate move of the state zone into a test failure
        # that reads like a regression.
        state_dir = resolve_simulation_state_dir(project_config, built_project)
        state_file = state_dir / "active_scenarios"
        assert "vacuum-burst" in state_file.read_text(encoding="utf-8")
        assert _data_checksums(built_project) == baseline

    def test_recording_channel_finder_feedback_leaves_data_untouched(
        self, built_project: Path, baseline: dict[str, str], project_config: dict
    ) -> None:
        """Both feedback stores, resolved the way the app's lifespan resolves them."""
        from osprey.services.channel_finder.feedback.pending_store import PendingReviewStore
        from osprey.services.channel_finder.feedback.store import FeedbackStore

        feedback = (
            project_config.get("channel_finder", {})
            .get("pipelines", {})
            .get("hierarchical", {})
            .get("feedback", {})
        )
        store_path = feedback.get("store_path", f"{FEEDBACK_DIR}/hierarchical_feedback.json")
        # Under the agent-data root, wherever the config puts it. Asserted
        # against the configured root rather than a literal, because the point
        # is the ZONE: a feedback store outside it sits in the tracked source
        # zone, where it dirties `git status` and a rebuild erases it.
        assert store_path.startswith(f"{agent_data_base_dir(project_config)}/"), (
            f"the shipped config points the feedback store at {store_path}, outside "
            f"the agent-data root — runtime writes there are not durable state"
        )
        hierarchical = built_project / store_path
        hierarchical.parent.mkdir(parents=True, exist_ok=True)

        FeedbackStore(hierarchical).record_success(
            query="show me the storage ring correctors",
            facility="ALS",
            selections={"system": "MAG"},
            channel_count=42,
        )
        pending = built_project / FEEDBACK_DIR / "pending_reviews.json"
        pending.parent.mkdir(parents=True, exist_ok=True)
        PendingReviewStore(pending).capture(
            {
                "query": "show me the storage ring correctors",
                "facility": "ALS",
                "tool_name": "mcp__channel-finder__build_channels",
                "channel_count": 42,
                "selections": {"system": "MAG"},
            }
        )

        assert json.loads(hierarchical.read_text(encoding="utf-8"))["entries"]
        assert json.loads(pending.read_text(encoding="utf-8"))["items"]
        assert _data_checksums(built_project) == baseline

    def test_deploy_prep_leaves_data_untouched(
        self, built_project: Path, baseline: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rendering the compose stack regenerates ``build/``, never ``data/``.

        The Virtual Accelerator's channel manifest is written into
        ``data/simulation/`` by the *build*, and deploy prep must not rewrite it:
        a second write with a different byte layout would read as drift on a
        project nobody touched.
        """
        from osprey.deployment.compose_generator import prepare_compose_files

        monkeypatch.chdir(built_project)
        _, compose_files = prepare_compose_files(str(built_project / "config.yml"))

        assert compose_files, "deploy prep rendered no compose files — assertion is vacuous"
        assert _data_checksums(built_project) == baseline


class TestTheInstrumentHasTeeth:
    """Negative control: the comparison above must actually fail on a real write.

    Without this, every assertion in this file would still pass if
    ``calculate_file_checksums`` silently stopped covering ``data/`` — the exact
    regression 6.1 removed the ``data`` exclusion to prevent.
    """

    def test_a_writer_pointed_into_data_is_caught(
        self, built_project: Path, tmp_path: Path
    ) -> None:
        import shutil

        from osprey.simulation.apply import apply_scenarios

        project = tmp_path / "misconfigured"
        shutil.copytree(built_project, project)
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        config["simulation"] = {"state_dir": "data/simulation"}
        (project / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
        before = _data_checksums(project)

        # Same two opt-outs as the positive case above, for the same reason —
        # this negative control is about where the state file lands, not about
        # ARIEL or the archive.
        apply_scenarios(project, ["vacuum-burst"], seed_logbook=False, seed_archive=False)

        assert (project / "data" / "simulation" / "active_scenarios").exists()
        assert _data_checksums(project) != before


class TestTheLoaderRejectsUnknownProfileKeys:
    """A schema removal has to fail somewhere, and the hash cannot be that somewhere.

    ``compute_profile_hash`` resolves a profile but never parses it, so deleting a
    key from the schema does not move the hash — a profile still naming the
    retired key would hash identically and build as if nothing changed. The
    loader's unknown-key rejection is the pin that notices, so it needs a test
    the hash path cannot mask.
    """

    def test_a_shipped_preset_loads(self, tmp_path: Path) -> None:
        """Control: the rejection below is about the added key, not the fixture."""
        from osprey.cli.build_profile import load_profile

        assert load_profile(_PRESET_PATH) is not None

    def test_an_unknown_top_level_key_is_rejected(self, tmp_path: Path) -> None:
        from osprey.cli.build_profile import load_profile
        from osprey.errors import BuildProfileError

        profile = yaml.safe_load(_PRESET_PATH.read_text(encoding="utf-8"))
        profile["definitely_not_a_profile_key"] = "x"
        path = tmp_path / "profile.yml"
        path.write_text(yaml.safe_dump(profile), encoding="utf-8")

        with pytest.raises(BuildProfileError) as excinfo:
            load_profile(path)

        assert "definitely_not_a_profile_key" in str(excinfo.value)


class TestTheWarningFiresWhenAWriterIsPointedBackIntoData:
    """The remaining way ``data/`` can still be written: a config that asks for it."""

    def test_loading_such_a_config_warns(
        self, built_project: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = yaml.safe_load((built_project / "config.yml").read_text(encoding="utf-8"))
        config["project_root"] = str(tmp_path)
        config["simulation"] = {"state_dir": "data/simulation"}
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        with caplog.at_level("WARNING", logger="CONFIG"):
            ConfigBuilder(str(config_path), load_env=False)

        assert "simulation.state_dir" in caplog.text
