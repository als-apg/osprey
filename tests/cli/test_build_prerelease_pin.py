"""A pre-release osprey pin has to reach the resolver as a pre-release resolve.

``osprey build`` installs ``osprey-framework==<version>`` into the project venv.
uv admits a pre-release for a requirement that names one, but the framework's
own ``osprey-connectors`` requirement carries no pre-release and osprey only
ever ships the two as a pair, so a beta pin resolves to nothing unless the
resolve as a whole admits pre-releases. The same applies to the recorded
``pyproject.toml``: a later ``uv sync`` in the built directory re-resolves from
it and must reach the same conclusion.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from osprey.cli.build_environment import _create_project_venv
from osprey.cli.build_profile import BuildProfile, EnvironmentConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    recorded: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", fake_run)
    monkeypatch.setenv("UV", "/opt/bin/uv")
    return recorded


def _build(project_path: Path, osprey_install: str) -> dict:
    project_path.mkdir(parents=True, exist_ok=True)
    profile = BuildProfile(
        name="test", osprey_install=osprey_install, environment=EnvironmentConfig()
    )
    _create_project_venv(project_path, profile)
    return tomllib.loads((project_path / "pyproject.toml").read_text(encoding="utf-8"))


def _install_cmd(calls: list[list[str]]) -> list[str]:
    return calls[-1]


class TestAPrereleasePin:
    def test_the_install_admits_prereleases(self, calls, tmp_path):
        _build(tmp_path / "project", "osprey-framework==2026.9.0b1")

        cmd = _install_cmd(calls)
        assert cmd[:3] == ["/opt/bin/uv", "pip", "install"]
        assert "--prerelease" in cmd and cmd[cmd.index("--prerelease") + 1] == "allow"

    def test_the_record_admits_prereleases_for_a_later_sync(self, calls, tmp_path):
        data = _build(tmp_path / "project", "osprey-framework==2026.9.0b1")

        assert data["tool"]["uv"]["prerelease"] == "allow"


class TestAStablePin:
    def test_the_resolve_stays_strict(self, calls, tmp_path):
        data = _build(tmp_path / "project", "osprey-framework==2026.9.0")

        assert "--prerelease" not in _install_cmd(calls)
        assert "tool" not in data

    def test_a_pin_that_only_excludes_a_prerelease_stays_strict(self, calls, tmp_path):
        data = _build(tmp_path / "project", "osprey-framework>=2026.6.2,!=2026.6.2a0")

        assert "--prerelease" not in _install_cmd(calls)
        assert "tool" not in data
