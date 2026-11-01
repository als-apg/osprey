"""The example's package: found by the deployment, and carrying its panel in the wheel."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from knob_example import APP_NAME

from osprey.apps.rules import deployment_packages
from osprey.cli.build_cmd import build
from osprey.deployment.wheel_build import DEPLOYMENT_PACKAGES_DIRNAME
from tests._builds import BuiltProject, init_project
from tests.test_offline_build_env import NO_ROUTE

pytestmark = pytest.mark.slow


def test_an_initialised_deployment_lists_the_package(
    built_control_assistant: BuiltProject,
) -> None:
    repo = built_control_assistant.repo
    assert deployment_packages(repo, [], apps=[APP_NAME]) == [
        repo / "apps" / APP_NAME / "pyproject.toml"
    ]


def test_a_skip_deps_build_records_the_package_for_the_image(
    built_control_assistant: BuiltProject,
) -> None:
    dockerfile = (built_control_assistant.repo / "build" / "Dockerfile").read_text()
    assert f"{APP_NAME}==0.1.0" in dockerfile


def test_a_build_with_no_route_to_an_index_makes_the_wheel(
    offline_build_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name, value in {**offline_build_env, **NO_ROUTE}.items():
        monkeypatch.setitem(os.environ, name, value)
    repo = init_project(tmp_path, "control-assistant", "demo")

    result = CliRunner().invoke(build, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    [wheel] = (repo / "build" / DEPLOYMENT_PACKAGES_DIRNAME).glob("*.whl")
    assert wheel.name.startswith("knob_example-")
    names = zipfile.ZipFile(wheel).namelist()
    assert {
        "knob_example/__init__.py",
        "knob_example/server.py",
        "knob_example/tools.py",
        "knob_example/panel/index.html",
        "knob_example/panel/panel.js",
        "knob_example/panel/panel.css",
    } <= set(names)
    assert not [name for name in names if "/tests/" in name or "__pycache__" in name]
