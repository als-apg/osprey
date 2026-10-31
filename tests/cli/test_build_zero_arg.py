"""``osprey build``: the zero-argument render of a deployment repo's ``build/``.

The property these tests exist for is atomicity. ``build/`` is what stops a
running deployment — ``osprey down`` reads its compose files — so a build that
fails, or one an operator interrupts, must leave the previous ``build/``
untouched and usable. Every render therefore lands in ``build/.tmp`` and is
swapped in by rename only once it is complete, and the tests below take the
render apart at each of the points where it can fail.

The rest is the zone contract: the manifest at ``build/.osprey-manifest.json``
and nowhere else, ``project_root`` naming the repo rather than the render, the
repo's ``.env`` never copied into a directory documented as safe to delete, and
the ``var/`` skeleton created by whichever command runs first in a fresh clone.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import shutil
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from osprey.cli import build_cmd
from osprey.cli.build_cmd import build
from osprey.cli.build_environment import _ProjectVenvPlan
from osprey.deployment import staleness
from osprey_connectors.errors import BuildProfileError
from tests.cli._lifecycle_build import CI_FLAGS, rendered_config
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

# A plan with nothing to install, for builds whose environment step is faked.
_EMPTY_PLAN = _ProjectVenvPlan(
    base_python=None,
    base_python_arg=sys.executable,
    osprey_spec="osprey-framework",
    osprey_label="test",
    extra_deps=[],
    uv_path=None,
    member_requirements=[],
)


def _build(runner: CliRunner, cwd: Path, *args: str, flags: list[str] | None = None):
    """Run ``osprey build`` from *cwd*, as an operator standing there would."""
    with contextlib.chdir(cwd):
        return runner.invoke(build, [*(CI_FLAGS if flags is None else flags), *args])


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


class TestAFailedRenderPreservesThePreviousBuild:
    """The core property: only a complete render ever replaces ``build/``."""

    @staticmethod
    def _mark(repo: Path) -> Path:
        """Leave a fingerprint in the current build/ so a survivor is identifiable."""
        marker = repo / "build" / "previous-render.marker"
        marker.write_text("first build", encoding="utf-8")
        return marker

    def test_a_render_that_raises_leaves_the_previous_build_intact(
        self, runner, lifecycle_repo, monkeypatch
    ):
        assert _build(runner, lifecycle_repo).exit_code == 0
        marker = self._mark(lifecycle_repo)
        before = (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8")

        def explode(*args, **kwargs):
            raise RuntimeError("render failed")

        monkeypatch.setattr(build_cmd, "_apply_conventions", explode)
        result = _build(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert marker.is_file(), "the previous render was replaced by a failed build"
        assert (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8") == before
        assert not (lifecycle_repo / "build" / ".tmp").exists()

    def test_an_interrupt_leaves_the_previous_build_intact(
        self, runner, lifecycle_repo, monkeypatch
    ):
        """Ctrl-C is not an exception the build catches — the guarantee is structural."""
        assert _build(runner, lifecycle_repo).exit_code == 0
        marker = self._mark(lifecycle_repo)

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(build_cmd, "_render_compose_files", interrupt)
        result = _build(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert marker.is_file()
        assert not (lifecycle_repo / "build" / ".tmp").exists()

    def test_a_profile_that_no_longer_validates_cannot_replace_a_good_build(
        self, runner, lifecycle_repo
    ):
        assert _build(runner, lifecycle_repo).exit_code == 0
        marker = self._mark(lifecycle_repo)

        profile = lifecycle_repo / "profile.yml"
        profile.write_text(
            profile.read_text(encoding="utf-8").replace(
                "provider: anthropic", "provider: nonexistent-provider"
            ),
            encoding="utf-8",
        )
        result = _build(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert marker.is_file()

    def test_the_second_build_replaces_the_first_wholesale(self, runner, lifecycle_repo):
        assert _build(runner, lifecycle_repo).exit_code == 0
        stale = lifecycle_repo / "build" / "left-over-from-the-last-render.txt"
        stale.write_text("gone next time", encoding="utf-8")

        assert _build(runner, lifecycle_repo).exit_code == 0

        assert not stale.exists()
        assert (lifecycle_repo / "build" / "config.yml").is_file()


# A tree holding this file refuses to be deleted, for the whole of a test that
# has asked for it. Stands in for the real reasons a leftover survives a sweep
# — a subtree written by a container running as root, an NFS `.nfs*`
# sillyrename — none of which a test can produce as one unprivileged user on
# whatever filesystem the checkout happens to sit on.
_UNDELETABLE_MARKER = "undeletable.marker"


def _refuse_to_delete_marked_trees(monkeypatch, *, silently: bool = False) -> None:
    """Make every marked tree undeletable, and take the retry wait to zero.

    Two flavours, because both happen: the filesystem raises (an NFS
    sillyrename left in an otherwise-swept directory reports ENOTEMPTY), or the
    removal reports success and the tree is still standing anyway — which is
    what ``ignore_errors=True`` made of the first flavour for as long as the
    sweep used it.
    """
    real_rmtree = shutil.rmtree

    def rmtree(target, *args, **kwargs):
        if (Path(target) / _UNDELETABLE_MARKER).exists():
            if silently:
                return None
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(target))
        return real_rmtree(target, *args, **kwargs)

    monkeypatch.setattr(build_cmd.shutil, "rmtree", rmtree)
    monkeypatch.setattr(build_cmd, "_LEFTOVER_RETRY_SECONDS", 0)


def _held_aside(repo: Path, leftover: Path) -> list[Path]:
    """The copies of *leftover* the repair parked in the state zone."""
    prefix = f"{leftover.name}{build_cmd._HELD_ASIDE_SUFFIX}"
    return sorted(entry for entry in (repo / "var").iterdir() if entry.name.startswith(prefix))


class TestSwapRecovery:
    """The one window where ``build/`` does not exist, and how it is closed."""

    def test_an_incoming_render_is_adopted_when_build_is_gone(self, lifecycle_repo):
        zones = build_cmd._render_zones(lifecycle_repo, "als-exemplar")
        zones.incoming.parent.mkdir(parents=True, exist_ok=True)
        zones.incoming.mkdir()
        (zones.incoming / "config.yml").write_text("new render\n", encoding="utf-8")

        build_cmd._repair_interrupted_swap(zones)

        assert (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8") == (
            "new render\n"
        )
        assert not zones.incoming.exists()

    def test_an_outgoing_render_is_adopted_when_nothing_else_is_left(self, lifecycle_repo):
        zones = build_cmd._render_zones(lifecycle_repo, "als-exemplar")
        zones.outgoing.parent.mkdir(parents=True, exist_ok=True)
        zones.outgoing.mkdir()
        (zones.outgoing / "config.yml").write_text("old render\n", encoding="utf-8")

        build_cmd._repair_interrupted_swap(zones)

        assert (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8") == (
            "old render\n"
        )

    def test_a_present_build_wins_over_leftovers(self, lifecycle_repo):
        """A build that never reported success never happened."""
        zones = build_cmd._render_zones(lifecycle_repo, "als-exemplar")
        zones.build_dir.mkdir()
        (zones.build_dir / "config.yml").write_text("the build in place\n", encoding="utf-8")
        zones.incoming.parent.mkdir(parents=True, exist_ok=True)
        zones.incoming.mkdir()
        (zones.incoming / "config.yml").write_text("never announced\n", encoding="utf-8")

        build_cmd._repair_interrupted_swap(zones)

        assert (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8") == (
            "the build in place\n"
        )
        assert not zones.incoming.exists()

    def test_an_undeletable_leftover_is_moved_aside_rather_than_left_in_the_way(
        self, lifecycle_repo, monkeypatch
    ):
        zones = build_cmd._render_zones(lifecycle_repo, "als-exemplar")
        zones.build_dir.mkdir()
        zones.incoming.parent.mkdir(parents=True, exist_ok=True)
        zones.incoming.mkdir()
        (zones.incoming / _UNDELETABLE_MARKER).write_text("", encoding="utf-8")
        _refuse_to_delete_marked_trees(monkeypatch)

        build_cmd._repair_interrupted_swap(zones)

        assert not zones.incoming.exists(), "the name the next swap renames onto is still taken"
        aside = _held_aside(lifecycle_repo, zones.incoming)
        assert len(aside) == 1
        assert (aside[0] / _UNDELETABLE_MARKER).is_file(), "the tree was not carried across intact"

    def test_a_build_recovers_from_a_leftover_that_cannot_be_deleted(
        self, runner, lifecycle_repo, monkeypatch
    ):
        """The shipped failure: one undeletable leftover broke EVERY later build.

        The swap renames onto ``var/.osprey-build-outgoing``, so a leftover
        there that the repair could not remove used to surface — hundreds of
        lines later, and on every build from then on — as a bare ENOTEMPTY
        naming neither the path nor the reason.
        """
        assert _build(runner, lifecycle_repo).exit_code == 0
        zones = build_cmd._render_zones(lifecycle_repo, "als-exemplar")
        zones.outgoing.mkdir(parents=True)
        (zones.outgoing / _UNDELETABLE_MARKER).write_text("", encoding="utf-8")
        _refuse_to_delete_marked_trees(monkeypatch, silently=True)

        result = _build(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        assert (lifecycle_repo / "build" / "config.yml").is_file()
        assert len(_held_aside(lifecycle_repo, zones.outgoing)) == 1

    def test_a_leftover_that_cannot_even_be_moved_names_the_path_and_the_reason(
        self, lifecycle_repo, monkeypatch
    ):
        stuck = lifecycle_repo / "var" / "stuck-zone"
        stuck.mkdir(parents=True)
        (stuck / _UNDELETABLE_MARKER).write_text("", encoding="utf-8")
        _refuse_to_delete_marked_trees(monkeypatch)
        blocked = lifecycle_repo / "var" / "occupied-by-a-file"
        blocked.write_text("not somewhere a directory can be parked\n", encoding="utf-8")

        with pytest.raises(click.ClickException) as raised:
            build_cmd._clear_leftover(stuck, aside_root=blocked)

        assert str(stuck) in str(raised.value)
        assert "Directory not empty" in str(raised.value)


@pytest.mark.slow
def test_a_killed_build_leaves_a_build_that_can_still_stop_the_stack(
    runner, lifecycle_repo, tmp_path
):
    """The acceptance gate, run for real: SIGKILL mid-render, previous build/ survives.

    ``SIGKILL`` cannot be caught, so no cleanup handler of ours runs — which is
    the point. The signal is raised from inside a render step rather than after
    a sleep, so the process dies at a known point in the middle of the render
    rather than at whatever point a timer happens to land on.
    """
    assert _build(runner, lifecycle_repo).exit_code == 0
    marker = lifecycle_repo / "build" / "previous-render.marker"
    marker.write_text("first build", encoding="utf-8")
    compose_before = sorted(
        str(path.relative_to(lifecycle_repo)) for path in lifecycle_repo.rglob("docker-compose.yml")
    )

    script = tmp_path / "killed_build.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, signal
            from osprey.cli import build_cmd

            def die(*args, **kwargs):
                os.kill(os.getpid(), signal.SIGKILL)

            build_cmd._apply_conventions = die
            os.chdir({str(lifecycle_repo)!r})
            build_cmd._build_repo(
                None, stream=False, skip_lifecycle=True, skip_deps=True, runtime_root=None
            )
            """
        ),
        encoding="utf-8",
    )
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)

    assert completed.returncode == -signal.SIGKILL, completed.stderr
    assert marker.is_file(), "SIGKILL mid-render destroyed the previous build"
    assert (lifecycle_repo / "build" / "config.yml").is_file()
    assert (
        sorted(
            str(path.relative_to(lifecycle_repo))
            for path in lifecycle_repo.rglob("docker-compose.yml")
        )
        == compose_before
    ), "the compose files `osprey down` reads did not survive"


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

    def test_the_venv_is_created_at_the_path_the_render_records(
        self, runner, lifecycle_repo, monkeypatch
    ):
        """It records its own location, so it is created where it will be read from."""
        created: dict[str, Path] = {}

        def fake_venv(project_path: Path, _plan) -> None:
            created["at"] = Path(project_path)
            bin_dir = Path(project_path) / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "python").write_text("#!/bin/sh\n", encoding="utf-8")
            (Path(project_path) / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

        monkeypatch.setattr(
            build_cmd, "_plan_project_venv", lambda _profile, *, install: _EMPTY_PLAN
        )
        monkeypatch.setattr(build_cmd, "_create_project_venv", fake_venv)
        result = _build(runner, lifecycle_repo, flags=["--skip-lifecycle"])

        assert result.exit_code == 0, result.output
        assert created["at"] == lifecycle_repo / "build"
        # And the swap carried it — and its dependency record — into the render.
        assert (lifecycle_repo / "build" / ".venv" / "bin" / "python").is_file()
        assert (lifecycle_repo / "build" / "pyproject.toml").is_file()

    def test_a_venv_survives_a_failed_rebuild(self, runner, lifecycle_repo, monkeypatch):
        """It is the one artifact outside the atomic set — and not one `down` needs."""

        def fake_venv(project_path: Path, _plan) -> None:
            (Path(project_path) / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
            (Path(project_path) / ".venv" / "bin" / "python").write_text(
                "#!/bin/sh\n", encoding="utf-8"
            )

        monkeypatch.setattr(
            build_cmd, "_plan_project_venv", lambda _profile, *, install: _EMPTY_PLAN
        )
        monkeypatch.setattr(build_cmd, "_create_project_venv", fake_venv)
        assert _build(runner, lifecycle_repo, flags=["--skip-lifecycle"]).exit_code == 0

        def explode(*args, **kwargs):
            raise RuntimeError("render failed")

        monkeypatch.setattr(build_cmd, "_apply_conventions", explode)
        assert _build(runner, lifecycle_repo, flags=["--skip-lifecycle"]).exit_code != 0

        assert (lifecycle_repo / "build" / ".venv" / "bin" / "python").is_file()
        assert (lifecycle_repo / "build" / "config.yml").is_file()

    def test_an_unreproducible_base_leaves_the_venv_as_it_was(
        self, runner, lifecycle_repo, monkeypatch
    ):
        """A build the plan refuses answers before the environment is touched."""

        def fake_venv(project_path: Path, _plan) -> None:
            (Path(project_path) / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
            (Path(project_path) / ".venv" / "bin" / "python").write_text(
                "#!/bin/sh\n", encoding="utf-8"
            )

        monkeypatch.setattr(
            build_cmd, "_plan_project_venv", lambda _profile, *, install: _EMPTY_PLAN
        )
        monkeypatch.setattr(build_cmd, "_create_project_venv", fake_venv)
        assert _build(runner, lifecycle_repo, flags=["--skip-lifecycle"]).exit_code == 0
        python = lifecycle_repo / "build" / ".venv" / "bin" / "python"
        before = (python.read_bytes(), python.stat().st_mtime_ns)

        def refuse(_profile, *, install):  # noqa: ARG001
            raise BuildProfileError("probe")

        created: list[Path] = []
        monkeypatch.setattr(build_cmd, "_plan_project_venv", refuse)
        monkeypatch.setattr(
            build_cmd, "_create_project_venv", lambda path, _plan: created.append(path)
        )
        assert _build(runner, lifecycle_repo, flags=["--skip-lifecycle"]).exit_code != 0

        assert created == []
        assert (python.read_bytes(), python.stat().st_mtime_ns) == before

    def test_stream_is_accepted_as_the_short_flag(self, runner, lifecycle_repo):
        result = _build(runner, lifecycle_repo, "-s")

        assert result.exit_code == 0, result.output
