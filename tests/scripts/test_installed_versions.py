"""The stale-install reader compares what is installed with what the tree describes.

``scripts/installed_versions.py`` derives the package list and each package's
describe command from the pyprojects, so these run it over a small git workspace
holding both tag streams, with only the installed metadata faked.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests._by_path import load_module

REPO_ROOT = Path(__file__).resolve().parents[2]
iv = load_module("installed_versions", REPO_ROOT / "scripts" / "installed_versions.py")

ROOT_PYPROJECT = """\
[project]
name = "app"

[tool.uv.workspace]
members = ["packages/lib"]

[tool.hatch.version]
source = "vcs"
raw-options = { git_describe_command = "git describe --dirty --tags --long --match v[0-9]*" }
"""

MEMBER_PYPROJECT = """\
[project]
name = "lib"

[tool.hatch.version]
source = "vcs"
raw-options = { root = "../..", scm = { git = { describe_command = \
"git describe --dirty --tags --long --match osprey-connectors-v2[0-9][0-9][0-9].*" } } }
"""


def _git(root: Path, *args: str) -> None:
    identity = ("-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false")
    subprocess.run(["git", *identity, *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A committed workspace with no tag yet."""
    (tmp_path / "pyproject.toml").write_text(ROOT_PYPROJECT, encoding="utf-8")
    member = tmp_path / "packages" / "lib"
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text(MEMBER_PYPROJECT, encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def _installed(monkeypatch: pytest.MonkeyPatch, **versions: str) -> None:
    def version(dist: str) -> str:
        if dist not in versions:
            raise iv.PackageNotFoundError(dist)
        return versions[dist]

    monkeypatch.setattr(iv, "version", version)


def test_a_stale_install_is_a_row_on_either_stream(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(workspace, "tag", "v2026.10.0")
    _git(workspace, "tag", "osprey-connectors-v2026.10.0b3")
    _installed(monkeypatch, app="2026.9.0.post4+gabcdef012", lib="2026.9.0b3")
    assert iv.stale_rows(workspace) == [
        ("app", "2026.9.0.post4+gabcdef012", "2026.10.0"),
        ("lib", "2026.9.0b3", "2026.10.0b3"),
    ]


def test_an_install_past_the_tag_is_current(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(workspace, "tag", "v2026.10.0")
    _git(workspace, "tag", "osprey-connectors-v2026.10.0b3")
    _git(workspace, "commit", "-q", "--allow-empty", "-m", "second")
    (workspace / "dirty").write_text("", encoding="utf-8")
    _git(workspace, "add", "dirty")
    _installed(monkeypatch, app="2026.10.0.post5+g012345678.d20261101", lib="2026.10.0b3.dev1")
    assert iv.stale_rows(workspace) == []


def test_a_beta_is_not_its_final(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(workspace, "tag", "v2026.10.0")
    _installed(monkeypatch, app="2026.10.0b3")
    assert iv.stale_rows(workspace) == [("app", "2026.10.0b3", "2026.10.0")]


def test_a_package_not_installed_is_not_judged(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(workspace, "tag", "v2026.10.0")
    _git(workspace, "tag", "osprey-connectors-v2026.10.0")
    _installed(monkeypatch, app="2026.10.0")
    assert iv.stale_rows(workspace) == []


def test_a_tag_that_does_not_describe_is_not_judged(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed(monkeypatch, app="2026.9.0", lib="2026.9.0")
    assert iv.stale_rows(workspace) == []


def test_a_package_with_no_describe_command_is_not_judged(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "packages" / "lib" / "pyproject.toml").write_text(
        '[project]\nname = "lib"\nversion = "1.0"\n', encoding="utf-8"
    )
    _git(workspace, "tag", "v2026.10.0")
    _installed(monkeypatch, app="2026.10.0", lib="0.5")
    assert iv.stale_rows(workspace) == []


def test_the_script_prints_a_tab_separated_row_per_stale_package(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(workspace, "tag", "v2026.10.0")
    _installed(monkeypatch, app="2026.9.0")
    monkeypatch.setattr(iv, "REPO_ROOT", workspace)
    assert iv.main() == 0
    assert capsys.readouterr().out == "app\t2026.9.0\t2026.10.0\n"
