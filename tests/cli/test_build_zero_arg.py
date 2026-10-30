"""``osprey build``: the zero-argument render of a deployment repo's ``build/``.

The zone contract: the manifest at ``build/.osprey-manifest.json`` and nowhere
else, ``project_root`` naming the repo rather than the render, the repo's
``.env`` never copied into a directory documented as safe to delete, and the
``var/`` skeleton created by whichever command runs first in a fresh clone.
The swap that makes a build atomic is tested in ``test_build_swap.py``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli import build_cmd
from osprey.cli.build_cmd import build
from osprey.deployment import staleness
from tests.cli._lifecycle_build import rendered_config
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo


def _build(runner: CliRunner, cwd: Path, *args: str):
    """Run ``osprey build`` from *cwd*, as an operator standing there would."""
    # The flags spelled out rather than as `*CI_FLAGS`: tests/test_offline_build_env.py
    # reads build argv from source and credits only a literal `--skip-deps`.
    with contextlib.chdir(cwd):
        return runner.invoke(build, ["--skip-deps", "--skip-lifecycle", *args])


@pytest.fixture(scope="module")
def built_exemplar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The exemplar after one default build, once per module; read only.

    A test that edits the repo or varies the build uses its own ``lifecycle_repo``.
    """
    repo = build_exemplar_repo(tmp_path_factory.mktemp("built-exemplar") / EXEMPLAR_DIRNAME)
    result = _build(CliRunner(), repo)
    assert result.exit_code == 0, result.output
    return repo


def _manifest(repo: Path) -> dict:
    return json.loads(staleness.build_manifest_path(repo).read_text(encoding="utf-8"))


class TestRendersTheOutputZone:
    """What a successful zero-argument build puts on disk."""

    def test_build_is_the_rendered_project_with_no_nested_directory(self, built_exemplar):
        build_dir = built_exemplar / "build"
        assert (build_dir / "config.yml").is_file()
        assert (build_dir / ".mcp.json").is_file()
        assert (build_dir / ".claude").is_dir()
        # The deployment name is the repo's; it names the project, never a
        # directory inside build/.
        assert not (build_dir / built_exemplar.name).exists()
        assert rendered_config(built_exemplar)["project_name"] == built_exemplar.name

    def test_manifest_lands_where_the_drift_check_looks(self, built_exemplar):
        manifest_path = staleness.build_manifest_path(built_exemplar)
        assert manifest_path.is_file()
        assert manifest_path.parent == built_exemplar / "build"

    def test_compose_files_are_rendered_by_the_same_build(self, built_exemplar):
        """One build does it all: build/ describes the whole deployment."""
        rendered = list((built_exemplar / "build").rglob("docker-compose*.yml"))
        assert rendered, "no compose file was rendered"

    def test_state_zone_is_created_when_absent(self, runner, lifecycle_repo_factory, tmp_path):
        """A fresh clone carries no git-ignored directory; the build makes them."""
        repo = lifecycle_repo_factory(tmp_path / "clone")
        (repo / "var" / "agent_data").rmdir()
        (repo / "var").rmdir()

        assert _build(runner, repo).exit_code == 0

        assert (repo / "var" / "agent_data").is_dir()
        # The pre-upgrade audit tree is only ever read and purged, never made.
        assert not (repo / "var" / "audit").exists()

    def test_state_zone_creation_keeps_what_is_already_there(self, runner, lifecycle_repo):
        keeper = lifecycle_repo / "var" / "agent_data" / "memory.json"
        keeper.write_text('{"remembered": true}', encoding="utf-8")

        assert _build(runner, lifecycle_repo).exit_code == 0

        assert keeper.read_text(encoding="utf-8") == '{"remembered": true}'

    def test_no_staging_directory_survives_the_build(self, built_exemplar):
        assert not (built_exemplar / "build" / ".tmp").exists()
        assert not (built_exemplar / "var" / ".osprey-build-incoming").exists()
        assert not (built_exemplar / "var" / ".osprey-build-outgoing").exists()

    def test_nothing_records_the_path_the_render_was_staged_at(self, built_exemplar):
        """The swap moves the tree; anything naming ``build/.tmp`` would break on arrival."""
        offenders = [
            path
            for path in (built_exemplar / "build").rglob("*")
            if path.is_file()
            and path.suffix in {".yml", ".yaml", ".json", ".md", ".toml"}
            and "build/.tmp" in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert offenders == []


class TestRepoIdentity:
    """``project_root`` is the repo, not the directory being written."""

    def test_project_root_is_the_repo_not_the_render(self, built_exemplar):
        assert rendered_config(built_exemplar)["project_root"] == str(built_exemplar)

    def test_runtime_root_substitutes_the_path_a_container_sees(self, runner, lifecycle_repo):
        _build(runner, lifecycle_repo, "--runtime-root", "/app/als-exemplar")

        assert rendered_config(lifecycle_repo)["project_root"] == "/app/als-exemplar"

    def test_runs_from_any_subdirectory(self, runner, lifecycle_repo):
        result = _build(runner, lifecycle_repo / "data" / "facility" / "knowledge")

        assert result.exit_code == 0, result.output
        assert (lifecycle_repo / "build" / "config.yml").is_file()

    def test_repo_flag_targets_another_checkout(self, runner, lifecycle_repo, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        result = _build(runner, outside, "--repo", str(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert (lifecycle_repo / "build" / "config.yml").is_file()

    def test_outside_a_repo_is_a_named_refusal(self, runner, tmp_path):
        outside = tmp_path / "nowhere"
        outside.mkdir()

        result = _build(runner, outside)

        assert result.exit_code != 0
        assert "profile.yml" in result.output

    def test_the_repos_env_is_never_copied_into_the_disposable_zone(
        self, runner, lifecycle_repo_factory, tmp_path
    ):
        """``rm -rf build/`` is documented as safe; a copy of .env would make it a lie."""
        repo = lifecycle_repo_factory(tmp_path / "seeded", seed_env=True)
        secret = "sk-ant-exemplar-not-a-real-key"
        assert secret in (repo / ".env").read_text(encoding="utf-8")

        assert _build(runner, repo).exit_code == 0

        assert secret in (repo / ".env").read_text(encoding="utf-8")
        leaked = [
            path
            for path in (repo / "build").rglob("*")
            if path.is_file() and secret in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert leaked == []


class TestFingerprint:
    """The stamp the drift check reads, and the detail that explains a refusal."""

    def test_a_fresh_build_reads_as_clean(self, built_exemplar):
        report = staleness.check_drift(built_exemplar)

        assert report.state is staleness.DriftState.CLEAN

    def test_the_stamp_is_the_existing_profile_hash(self, built_exemplar):
        from osprey.cli.build_profile import compute_profile_hash

        creation = _manifest(built_exemplar)["creation"]
        assert creation["preset_hash"] == compute_profile_hash(built_exemplar / "profile.yml")
        assert creation["osprey_version"]

    def test_per_key_digests_are_stamped_so_a_refusal_can_name_the_change(
        self, runner, lifecycle_repo
    ):
        _build(runner, lifecycle_repo)
        digests = _manifest(lifecycle_repo)["creation"][staleness.KEY_DIGESTS_MANIFEST_KEY]
        assert staleness.MATERIAL_KEY in digests

        profile = lifecycle_repo / "profile.yml"
        profile.write_text(
            profile.read_text(encoding="utf-8") + "\nmodel: claude-sonnet-5\n",
            encoding="utf-8",
        )
        report = staleness.check_drift(lifecycle_repo)

        assert report.state is staleness.DriftState.DRIFT
        assert "model" in report.changed_keys

    def test_the_reproducible_command_is_the_zero_argument_one(self, built_exemplar):
        """The repo IS the invocation — there is no name or path to replay."""
        assert _manifest(built_exemplar)["reproducible_command"] == "osprey build"


class TestRunningDeploymentWarning:
    """A build renders files. It never reaches a container that is already up."""

    @staticmethod
    def _fake_runtime(monkeypatch, tmp_path, rows) -> dict:
        """Answer the ownership probe with *rows*; return the config the build passes.

        ``rows`` are ``(name, repo_id, working_dir)`` triples; ``repo_id`` of
        ``None`` is a container an older OSPREY stamped no label onto.
        """
        from osprey.deployment.compose_generator import REPO_ID_LABEL
        from osprey.deployment.container_ownership import Resource

        monkeypatch.setattr(
            "osprey.deployment.runtime_helper.get_runtime_command",
            lambda config=None: ["docker", "compose"],
        )

        class _Probe:
            runtime = "docker"

            def __init__(self, *args, **kwargs):
                pass

            def containers_for_project(self, project, *, include_stopped=True):  # noqa: ARG002
                return [
                    Resource(
                        kind="container",
                        name=name,
                        labels={
                            "com.docker.compose.project": project,
                            "com.docker.compose.project.working_dir": working_dir,
                            **({REPO_ID_LABEL: repo_id} if repo_id else {}),
                        },
                    )
                    for name, repo_id, working_dir in rows
                ]

            def volumes_for_project(self, project):  # noqa: ARG002
                return []

        monkeypatch.setattr("osprey.deployment.reset.RuntimeProbe", _Probe)
        return {"project_name": "als-exemplar", "project_root": str(tmp_path)}

    def test_running_containers_are_named_with_when_the_build_takes_effect(
        self, caplog, monkeypatch, tmp_path
    ):
        from osprey.deployment.compose_generator import repo_identity

        mine = repo_identity(tmp_path)
        config = self._fake_runtime(
            monkeypatch,
            tmp_path,
            [
                ("als-exemplar-bluesky-1", mine, str(tmp_path)),
                ("als-exemplar-nginx-1", None, str(tmp_path)),
            ],
        )

        with caplog.at_level(logging.WARNING):
            build_cmd._warn_if_deployment_running(config, "als-exemplar")

        assert "als-exemplar-nginx-1" in caplog.text
        assert "als-exemplar-bluesky-1" in caplog.text
        assert "osprey up" in caplog.text
        assert "another checkout" not in caplog.text

    def test_another_checkouts_containers_are_named_as_that_checkouts(
        self, caplog, monkeypatch, tmp_path
    ):
        """A scratch clone's build must not claim a colleague's running stack."""
        other = tmp_path / "elsewhere"
        config = self._fake_runtime(
            monkeypatch,
            tmp_path,
            [("als-exemplar-nginx-1", "0123456789ab", str(other))],
        )

        with caplog.at_level(logging.WARNING):
            build_cmd._warn_if_deployment_running(config, "als-exemplar")

        assert "another checkout" in caplog.text
        assert str(other) in caplog.text
        assert "takes effect" not in caplog.text

    def test_a_stopped_deployment_is_not_warned_about(self, caplog, monkeypatch, tmp_path):
        config = self._fake_runtime(monkeypatch, tmp_path, [])

        with caplog.at_level(logging.WARNING):
            build_cmd._warn_if_deployment_running(config, "als-exemplar")

        assert "takes effect" not in caplog.text
        assert "another checkout" not in caplog.text

    def test_no_container_runtime_is_not_a_build_failure(self, caplog, monkeypatch):
        def unavailable(_config=None):
            raise RuntimeError("no container runtime with compose support found")

        monkeypatch.setattr("osprey.deployment.runtime_helper.get_runtime_command", unavailable)

        with caplog.at_level(logging.WARNING):
            build_cmd._warn_if_deployment_running(None, "als-exemplar")

        assert "takes effect" not in caplog.text


class TestCarriedOverFlags:
    def test_skip_deps_renders_no_venv(self, built_exemplar):
        assert not (built_exemplar / "build" / ".venv").exists()

    def test_skip_deps_records_the_interpreter_running_the_build(self, built_exemplar):
        """No venv was made, so the only interpreter known to have osprey is this one."""
        assert sys.executable in (built_exemplar / "build" / ".mcp.json").read_text(
            encoding="utf-8"
        )

    def test_stream_is_accepted_as_the_short_flag(self, runner, lifecycle_repo):
        result = _build(runner, lifecycle_repo, "-s")

        assert result.exit_code == 0, result.output
