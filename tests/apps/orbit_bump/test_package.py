"""The application's package: built from the app tree alone, carrying its panel and skill."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from orbit_bump import APP_NAME

from osprey.cli.build_cmd import build
from osprey.deployment.wheel_build import DEPLOYMENT_PACKAGES_DIRNAME
from osprey.profiles import presets
from tests._builds import init_project
from tests.fixtures.lifecycle_repo import LISTED_APPLICATIONS
from tests.test_offline_build_env import NO_ROUTE

pytestmark = pytest.mark.slow

APP_TREE = Path(presets.__file__).parent / "apps" / APP_NAME

#: A profile's ``apps:`` block listing this application and nothing else.
LISTED_ORBIT_BUMP = f"apps:\n  {APP_NAME}: {{}}\n\n"


def _list_only_this_app(repo: Path) -> None:
    """Make *repo* list this application alone, with its tree copied under ``apps/``."""
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert text.count(LISTED_APPLICATIONS) == 1
    profile.write_text(text.replace(LISTED_APPLICATIONS, LISTED_ORBIT_BUMP), encoding="utf-8")
    shutil.rmtree(repo / "apps")
    shutil.copytree(
        APP_TREE,
        repo / "apps" / APP_NAME,
        ignore=shutil.ignore_patterns("__pycache__", "*.py[co]", "node_modules"),
    )


def test_a_build_with_no_route_to_an_index_makes_the_wheel(
    offline_build_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A full build, the offline proof of the wheel, with no route to an index."""
    for name, value in {**offline_build_env, **NO_ROUTE}.items():
        monkeypatch.setitem(os.environ, name, value)
    repo = init_project(tmp_path, "demo", "demo")
    _list_only_this_app(repo)

    result = CliRunner().invoke(build, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    [wheel] = (repo / "build" / DEPLOYMENT_PACKAGES_DIRNAME).glob("*.whl")
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
    assert requires == ["numpy"]
