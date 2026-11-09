"""``osprey build`` hands each lifecycle phase the venv it runs on.

The profile reference states the contract: in ``post_build`` and ``validate``
the project venv's ``bin/`` leads ``PATH``, so ``python`` is the project's own
interpreter; ``pre_build`` runs before that venv exists, and under
``--skip-deps`` no step gets one. ``post_build`` and ``validate`` steps see the
repository's env chain, secrets in ``.env`` included; so do ``pre_build`` steps.

Each build here runs in process with the install replaced by a stub that writes
a ``bin/python`` into the directory it is handed, so the venv a step sees is
compared with the one the install was given, and no install runs.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_environment import _ProjectVenvPlan
from osprey_connectors.dotenv import ENV_LOCAL_FILENAME
from osprey_connectors.workspace import BUILD_DIR_NAME
from tests._builds import init_project

pytestmark = [pytest.mark.slow]

#: The repo's directory name, which is the project name.
REPO = "demo"

#: The phases that dump their environment, in the order the build runs them.
PHASES = ("pre_build", "post_build", "validate")

#: A key the env chain carries and the shell-only list does not name.
PROBE_KEY = "LIFECYCLE_PROBE"

#: The profile reference's sentence on the env chain, which
#: ``test_steps_see_the_repo_env_chain`` proves.
ENV_CHAIN_SENTENCE = (
    "``post_build`` and ``validate`` steps see the repository's env chain, ``.env`` "
    "over ``.env.shared``, secrets in ``.env`` included; so do ``pre_build`` steps."
)

PROFILE_REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "source"
    / "reference"
    / "configuration"
    / "profile.rst"
)


class _Repo:
    """A hello-world repo whose lifecycle dumps each phase's environment."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from osprey.cli import build_cmd

        self.dumps = tmp_path / "dumps"
        self.dumps.mkdir()
        self.which_python = self.dumps / "which-python.txt"
        self.install_dirs: list[Path] = []
        self.repo = init_project(tmp_path, "hello-world", REPO)

        profile_path = self.repo / "profile.yml"
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        # The pipe is what makes the runner use a shell, which is what makes
        # `tee` land the environment in a file this test reads back.
        profile["lifecycle"] = {
            phase: [{"name": f"dump {phase}", "run": f"env | tee {self._dump_path(phase)}"}]
            for phase in PHASES
        }
        profile["lifecycle"]["validate"].append({"name": "which python", "run": "python"})
        profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

        which_python = self.which_python

        def install_stub(project_path: Path, _plan: _ProjectVenvPlan) -> None:
            self.install_dirs.append(project_path)
            python = project_path / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text(f'#!/bin/sh\nprintf "%s" "$0" > {which_python}\n', encoding="utf-8")
            python.chmod(python.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        monkeypatch.setattr(build_cmd, "_create_project_venv", install_stub)

    def _dump_path(self, phase: str) -> Path:
        return self.dumps / f"{phase}-env.txt"

    def build(self, *extra: str) -> None:
        for phase in PHASES:
            self._dump_path(phase).unlink(missing_ok=True)
        result = CliRunner().invoke(build, ["--repo", str(self.repo), *extra])
        assert result.exit_code == 0, result.output

    def environment(self, phase: str) -> dict[str, str]:
        lines = self._dump_path(phase).read_text(encoding="utf-8").splitlines()
        return dict(line.split("=", 1) for line in lines if "=" in line)

    def path_entries(self, phase: str) -> list[Path]:
        return [Path(entry) for entry in self.environment(phase)["PATH"].split(os.pathsep)]

    @property
    def build_dir(self) -> Path:
        return self.repo / BUILD_DIR_NAME


def _under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


@pytest.fixture
def lifecycle_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Repo:
    return _Repo(tmp_path, monkeypatch)


def test_post_build_and_validate_run_on_the_build_venv(lifecycle_repo: _Repo) -> None:
    lifecycle_repo.build()

    assert len(lifecycle_repo.install_dirs) == 1
    venv = lifecycle_repo.install_dirs[0] / ".venv"
    assert venv.resolve() == (lifecycle_repo.build_dir / ".venv").resolve()
    for phase in ("post_build", "validate"):
        assert lifecycle_repo.path_entries(phase)[0].resolve() == (venv / "bin").resolve(), phase
    ran = Path(lifecycle_repo.which_python.read_text(encoding="utf-8"))
    assert ran.resolve() == (venv / "bin" / "python").resolve()


def test_pre_build_sees_the_shell_path(lifecycle_repo: _Repo) -> None:
    lifecycle_repo.build()

    entries = lifecycle_repo.path_entries("pre_build")
    assert not [entry for entry in entries if _under(entry, lifecycle_repo.build_dir)]


def test_a_skip_deps_rebuild_puts_no_venv_on_path(lifecycle_repo: _Repo) -> None:
    lifecycle_repo.build()
    leftover = lifecycle_repo.build_dir / ".venv"
    assert leftover.is_dir()

    lifecycle_repo.build("--skip-deps")

    entries = lifecycle_repo.path_entries("validate")
    assert not [entry for entry in entries if _under(entry, lifecycle_repo.build_dir)]
    assert leftover.is_dir()


def test_steps_see_the_repo_env_chain(lifecycle_repo: _Repo) -> None:
    env_file = lifecycle_repo.repo / ENV_LOCAL_FILENAME
    existing = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    separator = "" if not existing or existing.endswith("\n") else "\n"
    env_file.write_text(f"{existing}{separator}{PROBE_KEY}=1\n", encoding="utf-8")

    lifecycle_repo.build()

    for phase in PHASES:
        assert lifecycle_repo.environment(phase).get(PROBE_KEY) == "1", phase


def test_the_profile_reference_states_the_env_chain_the_steps_see() -> None:
    reference = " ".join(PROFILE_REFERENCE.read_text(encoding="utf-8").split())

    assert ENV_CHAIN_SENTENCE in reference
