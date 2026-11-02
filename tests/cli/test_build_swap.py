"""``osprey build``: only a complete render ever replaces ``build/``.

``build/`` is what stops a running deployment — ``osprey down`` reads its
compose files — so a build that fails, or one an operator interrupts, must
leave the previous ``build/`` untouched and usable. Every render therefore
lands in ``build/.tmp`` and is swapped in by rename only once it is complete,
and the tests below take the render apart at each of the points where it can
fail. They move real directories with ``os.replace`` and kill a real child, so
they run on macOS too; ``test_build_zero_arg.py`` holds the rest of the build.
"""

from __future__ import annotations

import contextlib
import errno
import os
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
from osprey_connectors.errors import BuildProfileError
from tests.cli._lifecycle_build import CI_FLAGS

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


class TestServedFilesSurviveABuild:
    """A build never removes what a running stack is serving from ``build/``.

    The web-terminal artifacts (``build/nginx/``, ``build/docker-compose.web.yml``)
    are written by ``osprey up`` and the roster verbs, not by the build, so a
    swap that only brought the build's own render across left the running nginx
    bind-mounting paths that no longer existed.
    """

    @staticmethod
    def _serve(repo: Path) -> dict[Path, tuple[str, int]]:
        nginx = repo / "build" / "nginx"
        (nginx / "templates").mkdir(parents=True)
        (nginx / "landing.html").write_text("<h1>landing</h1>\n", encoding="utf-8")
        (nginx / "nginx.conf").write_text("server {}\n", encoding="utf-8")
        (nginx / "templates" / "secret-alice.conf.template").write_text("x\n", encoding="utf-8")
        compose = repo / "build" / "docker-compose.web.yml"
        compose.write_text("services: {}\n", encoding="utf-8")
        return {
            path: (path.read_text(encoding="utf-8") if path.is_file() else "", path.stat().st_ino)
            for path in (
                nginx,
                nginx / "templates",
                nginx / "landing.html",
                nginx / "nginx.conf",
                nginx / "templates" / "secret-alice.conf.template",
                compose,
            )
        }

    def test_the_served_files_are_intact_after_a_build(self, runner, lifecycle_repo):
        assert _build(runner, lifecycle_repo).exit_code == 0
        served = self._serve(lifecycle_repo)

        result = _build(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        for path, (content, inode) in served.items():
            assert path.exists(), f"the build removed {path.relative_to(lifecycle_repo)}"
            assert path.stat().st_ino == inode, "the mounted file was replaced, not carried over"
            if path.is_file():
                assert path.read_text(encoding="utf-8") == content

    def test_the_served_files_are_there_while_the_old_render_is_replaced(
        self, runner, lifecycle_repo, monkeypatch
    ):
        """Checked at the swap's last rename, where the old tree is about to go."""
        assert _build(runner, lifecycle_repo).exit_code == 0
        self._serve(lifecycle_repo)
        real_replace = os.replace
        seen: list[bool] = []

        def replace(src, dst):
            if Path(dst) == lifecycle_repo / "build" and Path(src).name == (
                build_cmd._INCOMING_DIRNAME
            ):
                seen.append((Path(src) / "nginx" / "landing.html").is_file())
            return real_replace(src, dst)

        monkeypatch.setattr(build_cmd.os, "replace", replace)

        assert _build(runner, lifecycle_repo).exit_code == 0
        assert seen == [True]

    def test_a_first_build_has_nothing_to_carry(self, runner, lifecycle_repo):
        assert _build(runner, lifecycle_repo).exit_code == 0
        assert not (lifecycle_repo / "build" / "nginx").exists()


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
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )

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


class TestTheVenvRidesTheSwap:
    """The venv is carried across the swap, never rebuilt by it."""

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

        monkeypatch.setattr(build_cmd, "_plan_project_venv", lambda _profile: _EMPTY_PLAN)
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

        monkeypatch.setattr(build_cmd, "_plan_project_venv", lambda _profile: _EMPTY_PLAN)
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

        monkeypatch.setattr(build_cmd, "_plan_project_venv", lambda _profile: _EMPTY_PLAN)
        monkeypatch.setattr(build_cmd, "_create_project_venv", fake_venv)
        assert _build(runner, lifecycle_repo, flags=["--skip-lifecycle"]).exit_code == 0
        python = lifecycle_repo / "build" / ".venv" / "bin" / "python"
        before = (python.read_bytes(), python.stat().st_mtime_ns)

        def refuse(_profile):
            raise BuildProfileError("probe")

        created: list[Path] = []
        monkeypatch.setattr(build_cmd, "_plan_project_venv", refuse)
        monkeypatch.setattr(
            build_cmd, "_create_project_venv", lambda path, _plan: created.append(path)
        )
        assert _build(runner, lifecycle_repo, flags=["--skip-lifecycle"]).exit_code != 0

        assert created == []
        assert (python.read_bytes(), python.stat().st_mtime_ns) == before
