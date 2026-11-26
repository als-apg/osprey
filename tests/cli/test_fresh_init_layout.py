"""A fresh ``osprey init`` of each shipped preset lands its bundled content at the final layout.

``facility/`` always, ``web/`` only for a preset that ships web content, and
never the dissolved ``data/``. The repo paths the written profile names (the
CANOPY theme, the knowledge bundle, the landing notices) point into those
trees and exist there, and no file the init writes tells the operator to edit
a ``data/`` source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey import repo_layout
from tests._builds import init_project

pytestmark = pytest.mark.usefixtures("container_runtime_is_up")

PROJECT_README_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "project"
    / "README.md.j2"
)


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
        return init_project(tmp_path_factory.mktemp("fresh-init"), preset, name)


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _init(tmp_path_factory, "demo", "demo")


@pytest.fixture(scope="module")
def hello_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _init(tmp_path_factory, "hello-world", "hello")


def _profile(repo: Path) -> dict:
    return yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))


def _notices(node: object) -> list[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "notices" and isinstance(value, list):
                return value
            found = _notices(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _notices(item)
            if found:
                return found
    return []


FACILITY = repo_layout.entry("facility").path
WEB = repo_layout.entry("web").path
DISSOLVED = repo_layout.dissolved_names()[0]


@pytest.mark.parametrize("repo_name", ["demo", "hello_world"])
def test_a_fresh_init_creates_a_facility_and_no_data_tree(repo_name: str, request) -> None:
    repo: Path = request.getfixturevalue(repo_name)
    assert (repo / FACILITY / "identity.yaml").is_file()
    assert not (repo / DISSOLVED).exists()


def test_demo_creates_a_web_tree(demo: Path) -> None:
    assert (repo := demo / WEB).is_dir()
    assert any(path.is_file() for path in repo.rglob("*"))


def test_hello_world_creates_no_web_tree(hello_world: Path) -> None:
    assert not (hello_world / WEB).exists()


@pytest.mark.parametrize("repo_name", ["demo", "hello_world"])
def test_the_written_profile_carries_no_data_key(repo_name: str, request) -> None:
    repo: Path = request.getfixturevalue(repo_name)
    assert "data" not in _profile(repo)


def test_every_repo_path_the_demo_profile_names_exists(demo: Path) -> None:
    profile = _profile(demo)
    config = profile.get("config") or {}
    paths = [
        config["web.panels.canopy.theme_file"],
        config["facility_knowledge.bundle_path"],
        *_notices(profile),
    ]
    assert "web/landing/working-safely.md" in paths
    for rel in paths:
        assert rel.split("/", 1)[0] in {FACILITY, WEB}, rel
        assert (demo / rel).exists(), rel


@pytest.mark.parametrize("repo_name", ["demo", "hello_world"])
def test_no_written_text_file_names_a_data_source_tree(repo_name: str, request) -> None:
    """No root file sends the operator to ``data/README.md`` or a ``data/`` source path."""
    repo: Path = request.getfixturevalue(repo_name)
    offenders = []
    for path in sorted(repo.iterdir()):
        if not path.is_file() or path.suffix not in {".md", ".yml", ".yaml", ""}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ("data/README.md", "data/facility/", "data/landing/", "data/canopy/"):
            if needle in text:
                offenders.append(f"{path.relative_to(repo)}: {needle}")
    assert offenders == []


def test_the_render_readme_maps_the_source_trees() -> None:
    text = PROJECT_README_TEMPLATE.read_text(encoding="utf-8")
    assert f"- `{FACILITY}/`" in text
    assert f"- `{WEB}/`" in text
    assert f"- `{DISSOLVED}/` —" not in text
    assert "data/README.md" not in text
    for needle in ("data/facility/", "data/landing/", "data/canopy/"):
        assert needle not in text, needle
