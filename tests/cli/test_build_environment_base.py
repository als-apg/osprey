"""Tests for the custom base interpreter and ``environment.packages`` install.

Covers :func:`osprey.cli.build_environment._create_project_venv` threading
``profile.environment.python`` into venv creation (both the ``uv`` and the
stdlib-``venv`` branch) and folding ``profile.environment.packages`` into the
single resolver pass alongside ``profile.dependencies``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from osprey.cli.build_environment import _create_project_venv
from osprey.cli.build_profile import BuildProfile, EnvironmentConfig


def _profile(
    *,
    dependencies: list[str] | None = None,
    python: str | None = None,
    packages: list[str] | None = None,
) -> BuildProfile:
    """Build a minimal profile carrying the environment block under test."""
    return BuildProfile(
        name="test",
        dependencies=list(dependencies or []),
        osprey_install="osprey-framework==1.2.3",
        environment=EnvironmentConfig(python=python, packages=list(packages or [])),
    )


@pytest.fixture
def calls(monkeypatch) -> list[list[str]]:
    """Record every subprocess command and report success for each."""
    recorded: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)
    return recorded


@pytest.fixture
def with_uv(monkeypatch) -> str:
    """Force the uv branch with a deterministic uv path."""
    monkeypatch.setenv("UV", "/opt/bin/uv")
    return "/opt/bin/uv"


@pytest.fixture
def without_uv(monkeypatch) -> None:
    """Force the stdlib-venv branch by making uv undiscoverable."""
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)


class TestCustomBaseInterpreter:
    """``environment.python`` selects the interpreter the project venv is based on."""

    def test_uv_branch_uses_custom_base(self, calls, with_uv, tmp_path):
        base = tmp_path / "other-venv" / "bin" / "python"

        _create_project_venv(tmp_path, _profile(python=str(base)))

        venv_cmd = calls[0]
        assert venv_cmd[:3] == [with_uv, "venv", str(tmp_path / ".venv")]
        assert venv_cmd[venv_cmd.index("--python") + 1] == str(base)

    def test_uv_branch_defaults_to_build_interpreter(self, calls, with_uv, tmp_path):
        _create_project_venv(tmp_path, _profile())

        venv_cmd = calls[0]
        assert venv_cmd[venv_cmd.index("--python") + 1] == sys.executable

    def test_stdlib_branch_uses_custom_base(self, calls, without_uv, tmp_path):
        base = tmp_path / "bare" / "bin" / "python3.12"

        _create_project_venv(tmp_path, _profile(python=str(base)))

        venv_cmd = calls[0]
        assert venv_cmd == [str(base), "-m", "venv", str(tmp_path / ".venv")]

    def test_stdlib_branch_defaults_to_build_interpreter(self, calls, without_uv, tmp_path):
        _create_project_venv(tmp_path, _profile())

        assert calls[0] == [sys.executable, "-m", "venv", str(tmp_path / ".venv")]

    def test_tilde_in_base_is_expanded(self, calls, with_uv, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

        _create_project_venv(tmp_path, _profile(python="~/envs/rig/bin/python"))

        venv_cmd = calls[0]
        assert venv_cmd[venv_cmd.index("--python") + 1] == str(
            Path("~/envs/rig/bin/python").expanduser()
        )

    def test_venv_base_takes_the_same_path_as_a_bare_interpreter(self, calls, with_uv, tmp_path):
        """A venv's python is just an interpreter — no mode flag, no branch."""
        venv_root = tmp_path / "base-venv"
        (venv_root / "bin").mkdir(parents=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        base = venv_root / "bin" / "python"
        base.touch()

        _create_project_venv(tmp_path, _profile(python=str(base)))

        venv_cmd = calls[0]
        assert venv_cmd[venv_cmd.index("--python") + 1] == str(base)
        # No inheritance is assumed: only the declared requirements are
        # installed. A declared base is additionally probed and frozen (see
        # tests/cli/test_project_pyproject_record.py), which is what carries a
        # venv base's packages over — as explicit pins, never implicitly.
        assert calls[-1][-1:] == ["osprey-framework==1.2.3"]


class TestEnvironmentPackages:
    """``environment.packages`` joins ``dependencies`` in the one resolver pass."""

    def test_uv_install_carries_dependencies_and_packages(self, calls, with_uv, tmp_path):
        _create_project_venv(
            tmp_path,
            _profile(dependencies=["numpy>=1.24"], packages=["cothread", "h5py==3.11.0"]),
        )

        assert len(calls) == 2, "environment.packages must not add a second install pass"
        install_cmd = calls[1]
        assert install_cmd[:3] == [with_uv, "pip", "install"]
        assert install_cmd[-4:] == [
            "osprey-framework==1.2.3",
            "numpy>=1.24",
            "cothread",
            "h5py==3.11.0",
        ]

    def test_stdlib_install_carries_dependencies_and_packages(self, calls, without_uv, tmp_path):
        _create_project_venv(
            tmp_path, _profile(dependencies=["numpy>=1.24"], packages=["cothread"])
        )

        assert len(calls) == 2
        install_cmd = calls[1]
        assert install_cmd[0] == str(tmp_path / ".venv" / "bin" / "python")
        assert install_cmd[-3:] == ["osprey-framework==1.2.3", "numpy>=1.24", "cothread"]

    def test_packages_alone_reach_the_resolver(self, calls, with_uv, tmp_path):
        _create_project_venv(tmp_path, _profile(packages=["pyepics"]))

        assert calls[1][-2:] == ["osprey-framework==1.2.3", "pyepics"]

    def test_empty_environment_block_changes_nothing(self, calls, with_uv, tmp_path):
        _create_project_venv(tmp_path, _profile(dependencies=["pandas"]))

        assert calls[1][-2:] == ["osprey-framework==1.2.3", "pandas"]
