"""The preset ships this application alone, and re-initialising drops an application it no longer ships."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from orbit_bump import APP_NAME

from osprey.apps.rules import deployment_packages
from osprey.cli.build_profile_drift import APP_PROVENANCE_FILENAME
from osprey.cli.init_cmd import init
from osprey.profiles import presets
from tests._builds import BuiltProject, init_project

pytestmark = pytest.mark.slow

APP_TREE = Path(presets.__file__).parent / "apps" / APP_NAME

#: An application an earlier release of the preset shipped, spelled so no
#: tracked file carries the name whole.
OLD_APP = "knob" + "-example"


def _listed_apps(repo: Path) -> list[str]:
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    return sorted(profile.get("apps") or {})


def test_an_initialised_deployment_packages_exactly_this_app(
    built_demo: BuiltProject,
) -> None:
    repo = built_demo.repo
    assert _listed_apps(repo) == [APP_NAME]
    assert sorted(path.name for path in (repo / "apps").iterdir()) == [APP_NAME]
    assert deployment_packages(repo, [], apps=_listed_apps(repo)) == [
        repo / "apps" / APP_NAME / "pyproject.toml"
    ]


def test_a_skip_deps_build_records_the_package_for_the_image(
    built_demo: BuiltProject,
) -> None:
    dockerfile = (built_demo.repo / "build" / "Dockerfile").read_text()
    assert f"{APP_NAME}==0.1.0" in dockerfile


def test_a_forced_init_drops_an_application_the_preset_no_longer_ships(tmp_path: Path) -> None:
    repo = init_project(tmp_path, "demo", "demo")
    stale = repo / "apps" / OLD_APP
    shutil.copytree(
        APP_TREE,
        stale,
        ignore=shutil.ignore_patterns("__pycache__", "*.py[co]", "node_modules"),
    )
    (stale / APP_PROVENANCE_FILENAME).write_text(
        (repo / "apps" / APP_NAME / APP_PROVENANCE_FILENAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    profile = repo / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            f"apps:\n  {APP_NAME}: {{}}\n", f"apps:\n  {OLD_APP}: {{}}\n"
        ),
        encoding="utf-8",
    )
    assert _listed_apps(repo) == [OLD_APP]

    result = CliRunner().invoke(init, [str(repo), "--preset", "demo", "--no-git", "--force"])

    assert result.exit_code == 0, result.output
    assert not stale.exists()
    assert sorted(path.name for path in (repo / "apps").iterdir()) == [APP_NAME]
    assert (repo / "apps" / APP_NAME / APP_PROVENANCE_FILENAME).is_file()
    assert _listed_apps(repo) == [APP_NAME]
    assert OLD_APP not in profile.read_text(encoding="utf-8")
    assert OLD_APP not in result.output
