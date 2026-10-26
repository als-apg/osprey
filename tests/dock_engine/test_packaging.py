"""How ``dock_engine`` reaches an install: inside the framework wheel, never on its own.

``packages/dock-engine`` is not a uv workspace member and is published nowhere.
The root wheel target bundles its module and package data, so ``import
dock_engine`` works wherever the framework is installed, and nothing in the
framework's metadata can ask an index for a distribution that does not exist.
Each rule here is the one that, left unpinned, lets the member creep back in as
a fourth wheel the release workflow would then try to publish.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_ROOT = Path(__file__).resolve().parents[2]
_ENGINE = _ROOT / "packages" / "dock-engine"


@pytest.fixture(scope="module")
def root_pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_dock_engine_is_not_a_workspace_member(root_pyproject: dict) -> None:
    """A member is built by ``uv build --all-packages`` and published by the
    release workflow's stage-by-exclusion; the engine must stay out of both."""
    members = root_pyproject["tool"]["uv"]["workspace"]["members"]
    assert "packages/dock-engine" not in members, members
    assert "osprey-dock-engine" not in root_pyproject["tool"]["uv"].get("sources", {})


def test_nothing_in_the_framework_requires_a_dock_engine_distribution(
    root_pyproject: dict,
) -> None:
    """No ``Requires-Dist`` may name ``osprey-dock-engine``: no index serves it, so
    a framework wheel carrying the requirement would be uninstallable from PyPI."""
    project = root_pyproject["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    for group in root_pyproject.get("dependency-groups", {}).values():
        requirements.extend(entry for entry in group if isinstance(entry, str))
    names = {canonicalize_name(Requirement(r).name) for r in requirements}
    assert "osprey-dock-engine" not in names, sorted(names)


def test_the_framework_wheel_bundles_the_module_and_its_package_data(
    root_pyproject: dict,
) -> None:
    """The wheel target lists the engine's import root beside ``src/osprey`` and
    force-includes the case corpus and the manifest where ``dock_engine._data``
    reads them."""
    wheel = root_pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "packages/dock-engine/src/dock_engine" in wheel["packages"], wheel["packages"]
    assert wheel["force-include"] == {
        "packages/dock-engine/cases": "dock_engine/cases",
        "packages/dock-engine/dist/manifest.json": "dock_engine/manifest.json",
    }
    for source in wheel["force-include"]:
        assert (_ROOT / source).exists(), f"{source} is force-included but missing"


def test_the_engine_directory_still_installs_from_git_on_its_own() -> None:
    """CANOPY installs ``packages/dock-engine`` with ``#subdirectory=`` and
    without the framework, so the directory keeps a ``pyproject.toml`` whose
    wheel carries the same module and package data, and depends on nothing."""
    member = tomllib.loads((_ENGINE / "pyproject.toml").read_text(encoding="utf-8"))
    assert member["project"]["name"] == "osprey-dock-engine"
    assert member["project"]["dependencies"] == []
    wheel = member["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/dock_engine"]
    assert wheel["force-include"] == {
        "cases": "dock_engine/cases",
        "dist/manifest.json": "dock_engine/manifest.json",
    }
