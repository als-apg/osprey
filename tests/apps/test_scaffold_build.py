"""The scaffold application in a real build: its own package, carrying its panel and skill.

Read off the shared demo build with the scaffold added by ``osprey app new``.
That build skips the venv install, and a build makes the application's wheel
either way. The full install of the wheel beside osprey, offline, is the
``demo-app`` case of ``tests/integration/test_build_boot.py``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml

from osprey.apps.rules import deployment_packages
from osprey.deployment.wheel_build import DEPLOYMENT_PACKAGES_DIRNAME
from tests._builds import BuiltProject
from tests.fixtures.lifecycle_repo import LISTED_APP

pytestmark = pytest.mark.slow

PACKAGE = LISTED_APP.replace("-", "_")


def _listed_apps(repo: Path) -> list[str]:
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    return sorted(profile.get("apps") or {})


def test_the_deployment_packages_exactly_this_app(built_demo_app: BuiltProject) -> None:
    repo = built_demo_app.repo
    assert _listed_apps(repo) == [LISTED_APP]
    assert sorted(path.name for path in (repo / "apps").iterdir()) == [LISTED_APP]
    assert deployment_packages(repo, [], apps=_listed_apps(repo)) == [
        repo / "apps" / LISTED_APP / "pyproject.toml"
    ]


def test_the_build_makes_the_wheel_with_its_panel_and_skill(built_demo_app: BuiltProject) -> None:
    """The one wheel holds the modules, the panel and the skill, no tests, and no dependency."""
    [wheel] = (built_demo_app.build_dir / DEPLOYMENT_PACKAGES_DIRNAME).glob("*.whl")
    assert wheel.name.startswith(f"{PACKAGE}-0.1.0-")
    names = zipfile.ZipFile(wheel).namelist()
    assert {
        f"{PACKAGE}/__init__.py",
        f"{PACKAGE}/server.py",
        f"{PACKAGE}/tools.py",
        f"{PACKAGE}/panel/index.html",
        f"{PACKAGE}/panel/panel.js",
        f"{PACKAGE}/panel/panel.css",
        f"{PACKAGE}/skills/{LISTED_APP}/SKILL.md",
    } <= set(names)
    assert not [
        name
        for name in names
        if "/tests/" in name or "__pycache__" in name or name.endswith(".test.js")
    ]
    metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
    assert not [
        line
        for line in zipfile.ZipFile(wheel).read(metadata).decode().splitlines()
        if line.startswith("Requires-Dist: ")
    ]


def test_a_skip_deps_build_records_the_package_for_the_image(built_demo_app: BuiltProject) -> None:
    dockerfile = (built_demo_app.repo / "build" / "Dockerfile").read_text()
    assert f"{LISTED_APP}==0.1.0" in dockerfile
