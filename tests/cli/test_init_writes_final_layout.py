"""A fresh ``osprey init`` writes the final layout, which validates and is not refused.

The facility lands at the facility row and the web content at the web row's
children; nothing lands under the dissolved ``data`` name. The tree is then
accepted by every reader that judges a deployment root before reading it:
``osprey validate``, the profile's own validation and the retired-layout
refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey import repo_layout
from osprey.cli.build_profile_load import load_profile
from osprey.cli.main import cli
from osprey.repo_retired import find_retired_root, refuse_retired_layout
from tests._builds import init_project

pytestmark = pytest.mark.usefixtures("container_runtime_is_up")


@pytest.fixture(autouse=True)
def _no_harvested_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's exported provider keys out of the repos these tests make."""
    from osprey.cli.templates.scaffolding import provider_api_key_entries

    for entry in provider_api_key_entries():
        monkeypatch.delenv(entry["var"], raising=False)


def _web(repo: Path, key: str) -> Path:
    return repo_layout.child_path(repo / repo_layout.entry("web").path, "web", key)


def _facility(repo: Path) -> Path:
    return repo / repo_layout.entry("facility").path


def _init(tmp_path_factory: pytest.TempPathFactory, preset: str, name: str) -> Path:
    """Init *preset* once for the module, with no exported key and no runtime probe."""
    from osprey.cli.templates.scaffolding import provider_api_key_entries

    with pytest.MonkeyPatch.context() as patch:
        for entry in provider_api_key_entries():
            patch.delenv(entry["var"], raising=False)
        patch.setattr(
            "osprey.deployment.runtime_helper.verify_runtime_is_running",
            lambda config=None, **_: (True, ""),
        )
        return init_project(tmp_path_factory.mktemp("final-layout"), preset, name)


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _init(tmp_path_factory, "demo", "demo")


@pytest.fixture(scope="module")
def hello_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _init(tmp_path_factory, "hello-world", "hello")


# --- what lands where ------------------------------------------------------------


def test_demo_lands_its_facility_at_the_facility_row(demo: Path) -> None:
    assert (_facility(demo) / "identity.yaml").is_file()


def test_demo_lands_its_web_content_at_the_web_children(demo: Path) -> None:
    for key in ("canopy-library", "canopy-collections", "canopy-theme", "landing"):
        tree = _web(demo, key)
        assert tree.is_dir(), key
        assert any(path.is_file() for path in tree.rglob("*")), key


def test_the_web_content_is_the_packaged_content_byte_for_byte(demo: Path) -> None:
    from osprey.cli.templates.manager import TemplateManager

    packaged = Path(TemplateManager().template_root) / "apps" / "demo" / "web"
    landed = demo / repo_layout.entry("web").path
    packaged_files = {
        path.relative_to(packaged).as_posix(): path.read_bytes()
        for path in packaged.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    landed_files = {
        path.relative_to(landed).as_posix(): path.read_bytes()
        for path in landed.rglob("*")
        if path.is_file()
    }
    assert packaged_files
    assert landed_files == packaged_files


@pytest.mark.parametrize("repo_name", ["demo", "hello_world"])
def test_nothing_lands_under_the_dissolved_name(repo_name: str, request) -> None:
    repo = request.getfixturevalue(repo_name)
    for name in repo_layout.dissolved_names():
        assert not (repo / name).exists(), name


@pytest.mark.parametrize("repo_name", ["demo", "hello_world"])
def test_no_retired_path_is_occupied(repo_name: str, request) -> None:
    repo = request.getfixturevalue(repo_name)
    for move in repo_layout.retired_moves():
        assert not (repo / move.old).exists(), move.old


def test_hello_world_has_a_facility_and_no_web_tree(hello_world: Path) -> None:
    assert _facility(hello_world).is_dir()
    assert not (hello_world / repo_layout.entry("web").path).exists()


# --- the tree is accepted --------------------------------------------------------


@pytest.mark.parametrize("repo_name", ["demo", "hello_world"])
def test_the_fresh_tree_is_not_refused(repo_name: str, request) -> None:
    repo = request.getfixturevalue(repo_name)
    found = find_retired_root(repo)
    assert found.empty, found
    refuse_retired_layout(repo)


@pytest.mark.parametrize("repo_name", ["demo", "hello_world"])
def test_the_fresh_tree_validates(repo_name: str, request) -> None:
    repo = request.getfixturevalue(repo_name)
    result = CliRunner().invoke(cli, ["validate", str(repo)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "retired" not in result.output.lower(), result.output


def test_a_profile_without_data_validates(hello_world: Path, tmp_path: Path) -> None:
    """``data:`` names no required tree: a profile that drops the key still validates."""
    import shutil

    repo = tmp_path / "hello"
    shutil.copytree(hello_world, repo, symlinks=True)
    profile = repo / repo_layout.entry("profile.yml").path
    document = yaml.safe_load(profile.read_text(encoding="utf-8"))
    document.pop("data", None)
    profile.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    load_profile(profile).validate(repo)
