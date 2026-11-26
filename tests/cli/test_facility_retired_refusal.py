"""Every command that reads or writes a deployment's facility refuses a retired layout.

A deployment whose facility tree still holds a source at a retired path is
stopped before anything is read or written: ``osprey build``, ``osprey
validate``, each ``osprey facility import`` verb, ``osprey facility validate``
and ``osprey scaffold pull`` all print the refusal once, in full, and exit 1.
Nothing in the deployment changes apart from the state zone a build creates
first, and no ``pre_build`` step runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey import repo_layout
from osprey.cli.main import cli
from osprey_facility import source_tree
from osprey_facility.errors import RetiredLayoutRefused, _revalidate_command
from osprey_facility.retired import refuse_retired
from tests._builds import init_project

pytestmark = [pytest.mark.slow]

#: The project name, and the repo's directory name.
PROJECT = "demo"

#: The state zone: the one directory a build may create before it refuses.
STATE_ZONE = "var"

#: The file and the directory moved back to where the retired layout kept them.
RETIRED_FILE = "seeds"
RETIRED_DIR = "scenarios"


def _facility(repo: Path) -> Path:
    return repo / repo_layout.entry("facility").path


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo deployment with one file and one directory at their retired paths.

    Its profile carries a ``pre_build`` step that leaves a marker outside the
    deployment, so a test can tell whether the hook ran.
    """
    repo = init_project(tmp_path_factory.mktemp("retired"), "demo", PROJECT)
    facility = _facility(repo)
    for slot in (RETIRED_FILE, RETIRED_DIR):
        current = facility / source_tree.path(slot)
        shutil.move(str(current), facility / current.name)
    return repo


@pytest.fixture
def repo(initialised: Path, tmp_path: Path) -> Path:
    """A fresh copy of the old-layout deployment, with a marking ``pre_build`` step."""
    copy = tmp_path / PROJECT
    shutil.copytree(initialised, copy, symlinks=True)
    profile = copy / "profile.yml"
    data = yaml.safe_load(profile.read_text(encoding="utf-8"))
    steps = data.setdefault("lifecycle", {}).setdefault("pre_build", [])
    steps.insert(0, {"name": "mark", "run": f"touch {_marker(copy)}"})
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return copy


def _marker(repo: Path) -> Path:
    return repo.parent / "pre_build_ran"


def _snapshot(repo: Path) -> dict[str, bytes | None]:
    """Every path in the deployment outside the state zone; files with their bytes."""
    snapshot: dict[str, bytes | None] = {}
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo)
        if relative.parts[0] == STATE_ZONE:
            continue
        snapshot[relative.as_posix()] = (
            path.read_bytes() if path.is_file() and not path.is_symlink() else None
        )
    return snapshot


def _expected_message(repo: Path) -> str:
    with pytest.raises(RetiredLayoutRefused) as refusal:
        refuse_retired(_facility(repo))
    return str(refusal.value)


def _assert_refused(repo: Path, args: list[str]) -> None:
    expected = _expected_message(repo)
    before = _snapshot(repo)
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert result.output.count(expected) == 1, result.output
    assert result.output.count("still holds sources at retired paths") == 1, result.output
    assert "Unexpected error" not in result.output
    assert _snapshot(repo) == before
    assert not _marker(repo).exists()


def test_the_fixture_holds_a_retired_file_and_directory(initialised: Path) -> None:
    facility = _facility(initialised)
    assert (facility / Path(source_tree.path(RETIRED_FILE)).name).is_file()
    assert (facility / Path(source_tree.path(RETIRED_DIR)).name).is_dir()
    assert not (facility / source_tree.path(RETIRED_FILE)).exists()


def test_build_refuses_before_writing_or_running_pre_build(repo: Path) -> None:
    _assert_refused(repo, ["build", "--repo", str(repo)])
    assert not (repo / "build").exists()


def test_validate_refuses(repo: Path) -> None:
    _assert_refused(repo, ["validate", "--repo", str(repo)])


def test_facility_validate_refuses(repo: Path) -> None:
    _assert_refused(repo, ["facility", "validate", "--repo", str(repo)])


def test_import_mml_refuses(repo: Path, tmp_path: Path) -> None:
    export = tmp_path / "machine.ring.ao.json"
    export.write_text("{}", encoding="utf-8")
    _assert_refused(repo, ["facility", "import", "mml", str(export), "--repo", str(repo)])


def test_import_list_refuses(repo: Path, tmp_path: Path) -> None:
    listing = tmp_path / "channels.csv"
    listing.write_text("address\nSR:NEW:CHANNEL\n", encoding="utf-8")
    _assert_refused(repo, ["facility", "import", "list", str(listing), "--repo", str(repo)])


def test_import_relations_refuses(repo: Path, tmp_path: Path) -> None:
    table = tmp_path / "relations.csv"
    table.write_text("type,subject,object\n", encoding="utf-8")
    _assert_refused(
        repo,
        ["facility", "import", "relations", str(table), "--layer", "cables", "--repo", str(repo)],
    )


def test_scaffold_pull_refuses_whatever_is_pulled(repo: Path) -> None:
    # A pull that would otherwise write: one knowledge index is missing.
    index = _facility(repo) / source_tree.path("knowledge") / "index.md"
    index.unlink()
    _assert_refused(
        repo,
        [
            "scaffold",
            "pull",
            f"demo:{repo_layout.entry('facility').path}/{source_tree.path('knowledge')}",
            "--force",
            "--repo",
            str(repo),
        ],
    )
    assert not index.exists()


def test_scaffold_pull_names_the_deployment_when_its_profile_does_not_resolve(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The deployment sits inside another work tree, which is not the repo to migrate.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    shutil.rmtree(repo / ".git", ignore_errors=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    profile = repo / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8") + "not_a_profile_key: true\n", encoding="utf-8"
    )
    result = CliRunner().invoke(
        cli,
        [
            "scaffold",
            "pull",
            f"demo:{repo_layout.entry('facility').path}/{source_tree.path('knowledge')}",
            "--force",
            "--repo",
            str(repo),
        ],
    )
    assert result.exit_code == 1, result.output
    assert result.output.count("still holds sources at retired paths") == 1, result.output
    assert result.output.splitlines()[-1] == _revalidate_command(repo.resolve()), result.output


def test_scaffold_pull_list_does_not_refuse(repo: Path) -> None:
    result = CliRunner().invoke(cli, ["scaffold", "pull", "demo", "--list", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
    assert "retired paths" not in result.output
