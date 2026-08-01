"""Tests for the built project's dependency record (``pyproject.toml``).

The record is what a rebuild elsewhere — most importantly a container image —
resolves from, so the property under test throughout is that *what was
installed is what was recorded*. Anything installed but not recorded gets
pruned by the first ``uv sync``; anything recorded but not installed makes the
image diverge from the host it was built on.

Assertions read the emitted file back as parsed TOML rather than as text, so
the explanatory comments in it cannot collide with assertions about its tables
— except where the comment itself is the subject.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from osprey.cli import build_environment
from osprey.cli.build_environment import _create_project_venv
from osprey.cli.build_profile import BuildProfile, EnvironmentConfig
from osprey.errors import BuildProfileError

pytestmark = pytest.mark.unit

OSPREY_SPEC = "osprey-framework==1.2.3"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _profile(
    *,
    dependencies: list[str] | None = None,
    python: str | None = None,
    packages: list[str] | None = None,
    inherit_exclude: list[str] | None = None,
) -> BuildProfile:
    """A minimal profile carrying the environment block under test."""
    return BuildProfile(
        name="test",
        dependencies=list(dependencies or []),
        osprey_install=OSPREY_SPEC,
        environment=EnvironmentConfig(
            python=python,
            packages=list(packages or []),
            inherit_exclude=list(inherit_exclude or []),
        ),
    )


def _record(name: str, version: str, direct_url: dict | None = None) -> dict:
    """A probe record as :func:`_probe_base_distributions` returns them."""
    return {
        "name": name,
        "version": version,
        "direct_url": None if direct_url is None else json.dumps(direct_url),
    }


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record every subprocess command and report success for each."""
    recorded: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)
    monkeypatch.setenv("UV", "/opt/bin/uv")
    return recorded


@pytest.fixture
def base_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Declare a venv base and control the distributions it reports.

    Returns a callable taking the base's installed distributions and yielding
    the interpreter path to declare as ``environment.python``.
    """

    def declare(*records: dict) -> str:
        monkeypatch.setattr(build_environment, "_base_is_venv", lambda _python: True)
        monkeypatch.setattr(
            build_environment, "_probe_base_distributions", lambda _python: list(records)
        )
        return str(tmp_path / "base-venv" / "bin" / "python")

    return declare


def build(project_path: Path, profile: BuildProfile) -> dict:
    """Build into *project_path* and return the emitted record, parsed."""
    project_path.mkdir(parents=True, exist_ok=True)
    _create_project_venv(project_path, profile)
    return tomllib.loads((project_path / "pyproject.toml").read_text(encoding="utf-8"))


def recorded_deps(data: dict) -> list[str]:
    return data["project"]["dependencies"]


def installed_deps(calls: list[list[str]]) -> list[str]:
    """The requirements handed to the resolver, in order.

    The install command is the last one issued; its requirement tail starts
    after the flags, at the osprey spec.
    """
    install_cmd = calls[-1]
    return install_cmd[install_cmd.index(OSPREY_SPEC) :]


# --------------------------------------------------------------------------
# What gets recorded
# --------------------------------------------------------------------------


class TestRecordedDependencies:
    """The record covers the frozen base, ``dependencies`` and ``packages``."""

    def test_records_frozen_base_dependencies_and_packages(self, calls, base_venv, tmp_path):
        python = base_venv(_record("lmfit", "1.3.2"), _record("cothread", "2.20"))

        data = build(
            tmp_path / "project",
            _profile(python=python, dependencies=["numpy>=1.24"], packages=["h5py==3.11.0"]),
        )

        assert recorded_deps(data) == [
            OSPREY_SPEC,
            "cothread==2.20",
            "lmfit==1.3.2",
            "numpy>=1.24",
            "h5py==3.11.0",
        ]

    def test_installed_set_is_the_recorded_set(self, calls, base_venv, tmp_path):
        """The divergence this task exists to close: one list, two consumers."""
        python = base_venv(_record("lmfit", "1.3.2"))

        data = build(
            tmp_path / "project",
            _profile(python=python, dependencies=["numpy>=1.24"], packages=["cothread"]),
        )

        assert installed_deps(calls) == recorded_deps(data)

    def test_frozen_packages_reach_the_single_resolver_pass(self, calls, base_venv, tmp_path):
        python = base_venv(_record("lmfit", "1.3.2"))

        build(tmp_path / "project", _profile(python=python))

        # The probe seam is patched out here, so the recorded commands are
        # exactly venv creation and one install.
        assert len(calls) == 2, "the frozen set must not add a second install pass"
        assert installed_deps(calls) == [OSPREY_SPEC, "lmfit==1.3.2"]

    def test_inherit_exclude_drops_a_package_from_the_record(self, calls, base_venv, tmp_path):
        python = base_venv(_record("lmfit", "1.3.2"), _record("Private_Pkg", "0.1.0"))

        data = build(tmp_path / "project", _profile(python=python, inherit_exclude=["private-pkg"]))

        assert recorded_deps(data) == [OSPREY_SPEC, "lmfit==1.3.2"]

    def test_freeze_is_computed_once(self, calls, monkeypatch, tmp_path):
        """Two probes mean two chances for the record and the venv to disagree."""
        probes: list[Path] = []
        monkeypatch.setattr(build_environment, "_base_is_venv", lambda _python: True)
        monkeypatch.setattr(
            build_environment,
            "_probe_base_distributions",
            lambda python: (probes.append(python), [_record("lmfit", "1.3.2")])[1],
        )

        build(
            tmp_path / "project",
            _profile(python=str(tmp_path / "base-venv" / "bin" / "python")),
        )

        assert len(probes) == 1

    def test_freeze_failure_aborts_before_anything_is_recorded(self, calls, base_venv, tmp_path):
        """A half-built project with an honest-looking record is the worst outcome."""
        python = base_venv(
            _record("mine", "0.1.0", {"url": "file:///src/mine", "dir_info": {"editable": True}})
        )
        project_path = tmp_path / "project"

        with pytest.raises(BuildProfileError, match="cannot be reproduced"):
            build(project_path, _profile(python=python))

        assert not (project_path / "pyproject.toml").exists()


class TestNoFrozenBase:
    """Without a venv base to freeze, the record is what it always was."""

    def test_undeclared_base_records_osprey_and_profile_deps_only(self, calls, tmp_path):
        data = build(tmp_path / "project", _profile(dependencies=["numpy>=1.24"]))

        assert recorded_deps(data) == [OSPREY_SPEC, "numpy>=1.24"]

    def test_undeclared_base_is_never_probed(self, calls, tmp_path):
        """The build's own interpreter is an install accident, not a curated base.

        Freezing it would carry osprey's test tooling into every built project,
        so an undeclared base is not enumerated at all.
        """
        build(tmp_path / "project", _profile(dependencies=["numpy>=1.24"]))

        assert len(calls) == 2, "venv creation and one install — no base probe"

    def test_non_venv_declared_base_records_the_declared_requirements(
        self, calls, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(build_environment, "_base_is_venv", lambda _python: False)

        data = build(
            tmp_path / "project",
            _profile(
                python=str(tmp_path / "bare" / "bin" / "python3.12"),
                dependencies=["numpy>=1.24"],
                packages=["cothread"],
            ),
        )

        assert recorded_deps(data) == [OSPREY_SPEC, "numpy>=1.24", "cothread"]


# --------------------------------------------------------------------------
# De-duplication
# --------------------------------------------------------------------------


class TestDeduplication:
    """One entry per distribution; the more explicit requirement wins."""

    def test_pin_from_the_freeze_beats_a_bare_profile_name(self, calls, base_venv, tmp_path):
        python = base_venv(_record("numpy", "2.2.6"))

        data = build(tmp_path / "project", _profile(python=python, dependencies=["numpy"]))

        assert recorded_deps(data) == [OSPREY_SPEC, "numpy==2.2.6"]

    def test_pin_beats_a_bare_name_declared_earlier(self, calls, tmp_path):
        data = build(
            tmp_path / "project",
            _profile(dependencies=["numpy"], packages=["numpy==2.1.0"]),
        )

        assert recorded_deps(data) == [OSPREY_SPEC, "numpy==2.1.0"]

    def test_bare_name_never_displaces_a_pin_declared_earlier(self, calls, tmp_path):
        data = build(
            tmp_path / "project",
            _profile(dependencies=["numpy==2.1.0"], packages=["numpy"]),
        )

        assert recorded_deps(data) == [OSPREY_SPEC, "numpy==2.1.0"]

    def test_profile_pin_overrides_a_conflicting_frozen_pin(self, calls, base_venv, tmp_path):
        """Pin vs pin: the profile is a declaration, the freeze is inherited."""
        python = base_venv(_record("numpy", "2.2.6"))

        data = build(tmp_path / "project", _profile(python=python, packages=["numpy==2.1.0"]))

        assert recorded_deps(data) == [OSPREY_SPEC, "numpy==2.1.0"]
        assert installed_deps(calls) == recorded_deps(data)

    def test_collision_is_by_canonical_name(self, calls, base_venv, tmp_path):
        """``Alpha_Pkg`` and ``alpha-pkg`` are one distribution to the resolver."""
        python = base_venv(_record("Alpha_Pkg", "0.1.0"))

        data = build(tmp_path / "project", _profile(python=python, packages=["alpha-pkg"]))

        assert recorded_deps(data) == [OSPREY_SPEC, "Alpha_Pkg==0.1.0"]

    def test_direct_url_requirement_beats_a_bare_name(self, calls, tmp_path):
        url_req = "mypkg @ https://example.invalid/mypkg-1.0-py3-none-any.whl"

        data = build(tmp_path / "project", _profile(dependencies=["mypkg"], packages=[url_req]))

        assert recorded_deps(data) == [OSPREY_SPEC, url_req]

    def test_osprey_requirement_stays_authoritative(self, calls, base_venv, tmp_path):
        """osprey's own spec is recorded once, first, and is never overridden."""
        python = base_venv(_record("lmfit", "1.3.2"))

        data = build(
            tmp_path / "project",
            _profile(python=python, dependencies=["osprey-framework==9.9.9"]),
        )

        assert recorded_deps(data) == [OSPREY_SPEC, "lmfit==1.3.2"]

    def test_every_recorded_entry_parses_as_a_requirement(self, calls, base_venv, tmp_path):
        """An unparseable entry makes the whole record unusable to any resolver."""
        from packaging.requirements import Requirement

        python = base_venv(_record("lmfit", "1.3.2"))

        data = build(
            tmp_path / "project",
            _profile(python=python, dependencies=["numpy>=1.24"], packages=["h5py==3.11.0"]),
        )

        for dep in recorded_deps(data):
            Requirement(dep)


# --------------------------------------------------------------------------
# Shape of the emitted file
# --------------------------------------------------------------------------


class TestFileShape:
    """The file is written whole, and the base interpreter is provenance only."""

    def test_rebuild_rewrites_rather_than_appends(self, calls, base_venv, tmp_path):
        """An appended emission would redefine ``[project]`` — a TOML error.

        A successful parse of the second build is therefore itself an assertion.
        """
        python = base_venv(_record("lmfit", "1.3.2"))
        project_path = tmp_path / "project"
        profile = _profile(python=python, dependencies=["numpy>=1.24"])

        build(project_path, profile)
        data = build(project_path, profile)

        assert recorded_deps(data).count("lmfit==1.3.2") == 1
        assert recorded_deps(data).count("numpy>=1.24") == 1

    def test_base_interpreter_is_recorded_as_a_comment(self, calls, base_venv, tmp_path):
        python = base_venv(_record("lmfit", "1.3.2"))
        project_path = tmp_path / "project"

        data = build(project_path, _profile(python=python))
        text = (project_path / "pyproject.toml").read_text(encoding="utf-8")

        assert f"# base interpreter: {python}" in text
        mentions = [line for line in text.splitlines() if python in line]
        assert mentions and all(line.startswith("#") for line in mentions)
        assert python not in json.dumps(data), "provenance must not be a parseable key"

    def test_undeclared_base_records_the_build_interpreter(self, calls, tmp_path):
        project_path = tmp_path / "project"

        build(project_path, _profile())
        text = (project_path / "pyproject.toml").read_text(encoding="utf-8")

        assert f"# base interpreter: {sys.executable}" in text

    def test_no_build_system_table(self, calls, base_venv, tmp_path):
        """A ``[build-system]`` would make every ``uv run`` try to build the project."""
        python = base_venv(_record("lmfit", "1.3.2"))

        data = build(tmp_path / "project", _profile(python=python))

        assert "build-system" not in data
