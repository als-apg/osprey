"""The application's package: built from the app tree alone, carrying its panel and skill.

The wheel is read off the shared demo build with the app turned on. That build
skips the venv install, and a build makes the app's wheel either way. The full
install of the wheel beside osprey, offline, is the ``demo-orbit-bump`` case of
``tests/integration/test_build_boot.py``.
"""

from __future__ import annotations

import zipfile

import pytest
from orbit_bump import APP_NAME

from osprey.deployment.wheel_build import DEPLOYMENT_PACKAGES_DIRNAME
from tests._builds import BuiltProject

pytestmark = pytest.mark.slow


def test_the_build_makes_the_wheel_with_its_panel_and_skill(
    built_demo_orbit_bump: BuiltProject,
) -> None:
    """The one wheel holds the modules, the panel and the skill, no tests, and the app's own deps."""
    [wheel] = (built_demo_orbit_bump.build_dir / DEPLOYMENT_PACKAGES_DIRNAME).glob("*.whl")
    assert wheel.name.startswith("orbit_bump-0.1.0-")
    names = zipfile.ZipFile(wheel).namelist()
    assert {
        "orbit_bump/__init__.py",
        "orbit_bump/server.py",
        "orbit_bump/tools.py",
        "orbit_bump/solver.py",
        "orbit_bump/orm.py",
        "orbit_bump/plans.py",
        "orbit_bump/ring.py",
        "orbit_bump/panel/index.html",
        "orbit_bump/panel/panel.js",
        "orbit_bump/panel/panel.css",
        f"orbit_bump/skills/{APP_NAME}/SKILL.md",
    } <= set(names)
    assert not [
        name
        for name in names
        if "/tests/" in name or "__pycache__" in name or name.endswith(".test.js")
    ]
    metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
    requires = [
        line.removeprefix("Requires-Dist: ")
        for line in zipfile.ZipFile(wheel).read(metadata).decode().splitlines()
        if line.startswith("Requires-Dist: ")
    ]
    assert requires == ["numpy", "scipy"]
