"""Tests for the custom base interpreter and ``environment.packages`` install.

Covers :func:`osprey.cli.build_environment._resolve_project_requirements`
folding ``profile.environment.packages`` into the one requirement list
alongside ``profile.dependencies``, and
:func:`osprey.cli.build_environment._create_project_venv` threading
``profile.environment.python`` into venv creation (both the ``uv`` and the
stdlib-``venv`` branch) and installing that list in a single resolver pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from osprey.cli.build_environment import (
    _create_project_venv,
    _resolve_project_requirements,
)
from osprey.cli.build_profile import BuildProfile, EnvironmentConfig
from osprey.deployment.members import WORKSPACE_MEMBERS


def _profile(
    *,
    dependencies: list[str] | None = None,
    python: str | None = None,
    packages: list[str] | None = None,
    osprey_install: str = "osprey-framework==1.2.3",
) -> BuildProfile:
    """Build a minimal profile carrying the environment block under test."""
    return BuildProfile(
        name="test",
        dependencies=list(dependencies or []),
        osprey_install=osprey_install,
        environment=EnvironmentConfig(python=python, packages=list(packages or [])),
    )


def _install(project_path: Path, profile: BuildProfile) -> None:
    """Resolve *profile*'s requirement list and create the venv from it."""
    _create_project_venv(project_path, profile, _resolve_project_requirements(profile))


def _venv_command(calls: list[list[str]]) -> list[str]:
    """The venv-creation call; a declared base's freeze probe runs before it."""
    return next(cmd for cmd in calls if "venv" in cmd)


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

        _install(tmp_path, _profile(python=str(base)))

        venv_cmd = _venv_command(calls)
        assert venv_cmd[:3] == [with_uv, "venv", str(tmp_path / ".venv")]
        assert venv_cmd[venv_cmd.index("--python") + 1] == str(base)

    @pytest.mark.usefixtures("with_uv")
    def test_uv_branch_defaults_to_build_interpreter(self, calls, tmp_path):
        _install(tmp_path, _profile())

        venv_cmd = _venv_command(calls)
        assert venv_cmd[venv_cmd.index("--python") + 1] == sys.executable

    @pytest.mark.usefixtures("without_uv")
    def test_stdlib_branch_uses_custom_base(self, calls, tmp_path):
        base = tmp_path / "bare" / "bin" / "python3.12"

        _install(tmp_path, _profile(python=str(base)))

        venv_cmd = _venv_command(calls)
        assert venv_cmd == [str(base), "-m", "venv", str(tmp_path / ".venv")]

    @pytest.mark.usefixtures("without_uv")
    def test_stdlib_branch_defaults_to_build_interpreter(self, calls, tmp_path):
        _install(tmp_path, _profile())

        assert _venv_command(calls) == [sys.executable, "-m", "venv", str(tmp_path / ".venv")]

    @pytest.mark.usefixtures("with_uv")
    def test_tilde_in_base_is_expanded(self, calls, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

        _install(tmp_path, _profile(python="~/envs/rig/bin/python"))

        venv_cmd = _venv_command(calls)
        assert venv_cmd[venv_cmd.index("--python") + 1] == str(
            Path("~/envs/rig/bin/python").expanduser()
        )

    @pytest.mark.usefixtures("with_uv")
    def test_venv_base_takes_the_same_path_as_a_bare_interpreter(self, calls, tmp_path):
        """A venv's python is just an interpreter — no mode flag, no branch."""
        venv_root = tmp_path / "base-venv"
        (venv_root / "bin").mkdir(parents=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        base = venv_root / "bin" / "python"
        base.touch()

        _install(tmp_path, _profile(python=str(base)))

        venv_cmd = _venv_command(calls)
        assert venv_cmd[venv_cmd.index("--python") + 1] == str(base)
        # No inheritance is assumed: only the declared requirements are
        # installed. A declared base is additionally probed and frozen (see
        # tests/cli/test_project_pyproject_record.py), which is what carries a
        # venv base's packages over — as explicit pins, never implicitly.
        assert calls[-1][-1:] == ["osprey-framework==1.2.3"]


class TestProjectRequirements:
    """The requirement list is fixed before anything is installed."""

    def test_packages_and_dependencies_form_one_list_without_a_venv(self, tmp_path):
        requirements = _resolve_project_requirements(
            _profile(dependencies=["pandas"], packages=["pyepics"])
        )

        assert requirements.extra_deps == ["pandas", "pyepics"]
        assert requirements.base_python_arg == sys.executable
        assert list(tmp_path.iterdir()) == []

    def test_packages_alone_reach_the_list(self):
        assert _resolve_project_requirements(_profile(packages=["pyepics"])).extra_deps == [
            "pyepics"
        ]

    def test_empty_environment_block_changes_nothing(self):
        assert _resolve_project_requirements(_profile(dependencies=["pandas"])).extra_deps == [
            "pandas"
        ]

    def test_a_requirement_named_twice_is_listed_once(self):
        requirements = _resolve_project_requirements(
            _profile(dependencies=["numpy>=1.24"], packages=["numpy>=1.24", "cothread"])
        )

        assert requirements.extra_deps == ["numpy>=1.24", "cothread"]

    def test_declared_base_is_the_venv_interpreter(self, tmp_path, monkeypatch):
        base = tmp_path / "bare" / "bin" / "python3.12"
        monkeypatch.setattr(
            "osprey.cli.build_environment.freeze_base_environment", lambda *a, **k: []
        )

        assert _resolve_project_requirements(_profile(python=str(base))).base_python_arg == str(
            base
        )


class TestEnvironmentPackages:
    """``environment.packages`` joins ``dependencies`` in the one resolver pass."""

    def test_uv_install_carries_dependencies_and_packages(self, calls, with_uv, tmp_path):
        _install(
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

    @pytest.mark.usefixtures("without_uv")
    def test_stdlib_install_carries_dependencies_and_packages(self, calls, tmp_path):
        _install(tmp_path, _profile(dependencies=["numpy>=1.24"], packages=["cothread"]))

        assert len(calls) == 2
        install_cmd = calls[1]
        assert install_cmd[0] == str(tmp_path / ".venv" / "bin" / "python")
        assert install_cmd[-3:] == ["osprey-framework==1.2.3", "numpy>=1.24", "cothread"]


class TestWorkspaceMembersFromSource:
    """A source-tree osprey spec carries its workspace members on the pip path.

    uv resolves the members through the checkout's ``[tool.uv.sources]``; pip
    reads no such table, so the pip command must name each member directory.
    """

    @pytest.mark.usefixtures("with_uv")
    def test_uv_command_is_unchanged_for_a_source_spec(self, calls, tmp_path):
        root = tmp_path / "osprey-src"

        _install(tmp_path, _profile(osprey_install=str(root), packages=["pyepics"]))

        install_cmd = calls[1]
        assert install_cmd[-2:] == [str(root), "pyepics"]
        assert not any("packages/" in arg for arg in install_cmd)

    @pytest.mark.usefixtures("without_uv")
    def test_pip_command_appends_every_member_for_a_source_spec(self, calls, tmp_path):
        root = tmp_path / "osprey-src"

        _install(tmp_path, _profile(osprey_install=str(root), packages=["pyepics"]))

        install_cmd = calls[1]
        members = [str(root / "packages" / name) for name in WORKSPACE_MEMBERS]
        assert len(members) == 2
        assert install_cmd[-(2 + len(members)) :] == [str(root), "pyepics", *members]

    @pytest.mark.usefixtures("without_uv")
    def test_pip_command_appends_no_member_for_a_pinned_spec(self, calls, tmp_path):
        _install(tmp_path, _profile(packages=["pyepics"]))

        install_cmd = calls[1]
        assert install_cmd[-2:] == ["osprey-framework==1.2.3", "pyepics"]
        assert not any("packages/" in arg for arg in install_cmd)

    _VCS_URL = "git+https://example.invalid/osprey.git@0123abc"

    @classmethod
    def _vcs_members(cls) -> list[str]:
        return [f"{m} @ {cls._VCS_URL}#subdirectory=packages/{m}" for m in WORKSPACE_MEMBERS]

    @pytest.mark.usefixtures("without_uv")
    @pytest.mark.parametrize(
        "spec",
        [
            _VCS_URL,
            f"{_VCS_URL}#egg=osprey-framework",
            f"osprey-framework @ {_VCS_URL}",
        ],
        ids=["bare-vcs-url", "vcs-url-with-fragment", "direct-reference"],
    )
    def test_pip_command_names_every_member_subdirectory_for_a_vcs_spec(
        self, calls, tmp_path, spec
    ):
        """The members live in the same repository, one subdirectory each, so
        they come from the same commit rather than by name from PyPI."""
        _install(tmp_path, _profile(osprey_install=spec))

        install_cmd = calls[1]
        members = self._vcs_members()
        assert install_cmd[-(1 + len(members)) :] == [spec, *members]

    @pytest.mark.usefixtures("without_uv")
    @pytest.mark.parametrize("form", ["direct-reference", "bare-url"])
    def test_pip_command_names_every_member_directory_for_a_file_url(self, calls, tmp_path, form):
        root = tmp_path / "osprey src"
        url = root.as_uri()
        spec = f"osprey-framework @ {url}" if form == "direct-reference" else url

        _install(tmp_path, _profile(osprey_install=spec))

        install_cmd = calls[1]
        members = [str(root / "packages" / name) for name in WORKSPACE_MEMBERS]
        assert install_cmd[-(1 + len(members)) :] == [spec, *members]

    @pytest.mark.usefixtures("without_uv", "calls")
    @pytest.mark.parametrize(
        "spec",
        [
            "osprey-framework @ https://example.invalid/osprey_framework-1.0-py3-none-any.whl",
            "https://example.invalid/osprey_framework-1.0.tar.gz",
            "hg+https://example.invalid/osprey",
        ],
    )
    def test_pip_command_refuses_a_spec_whose_members_it_cannot_locate(self, tmp_path, spec):
        """A framework fetched from an arbitrary URL names no place its members
        live, and resolving them by name from PyPI may pick up a different
        project's package."""
        from osprey_connectors.errors import BuildProfileError

        with pytest.raises(BuildProfileError, match="workspace member"):
            _install(tmp_path, _profile(osprey_install=spec))

    @pytest.mark.usefixtures("without_uv")
    def test_pip_command_appends_no_member_for_a_version_range(self, calls, tmp_path):
        _install(tmp_path, _profile(osprey_install="osprey-framework>=2026.9"))

        assert calls[1][-1] == "osprey-framework>=2026.9"
