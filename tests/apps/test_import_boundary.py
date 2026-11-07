"""Application code reaches OSPREY only through its public surfaces.

The checker reads a package by source path, so it covers packages no ``pyproject.toml``
contract can name: import-linter refuses a listed module that is absent from its graph,
and the fixtures and a deployment's own packages are not installed there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from osprey.apps.boundary import check_package

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_APP = REPO_ROOT / "tests/apps/fixtures/boundary_fixture_app"
LATER_PACKAGES = (
    "src/osprey/profiles/presets/apps/knob-example/knob_example",
    "tests/facility/fixtures/fixture_kind",
)


def _package(tmp_path: Path, source: str, name: str = "module.py") -> Path:
    package = tmp_path / "some_app"
    target = package / name
    target.parent.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    target.write_text(source + "\n")
    return package


def test_the_fixture_app_stays_inside_the_boundary() -> None:
    assert check_package(FIXTURE_APP) == []


@pytest.mark.parametrize("relative", LATER_PACKAGES)
def test_a_shipped_package_stays_inside_the_boundary(relative: str) -> None:
    package = REPO_ROOT / relative
    if not package.is_dir():
        pytest.skip(f"{relative} is not in this tree")
    assert check_package(package) == []


def test_a_private_import_added_to_the_fixture_app_fails(tmp_path: Path) -> None:
    package = tmp_path / FIXTURE_APP.name
    shutil.copytree(FIXTURE_APP, package)
    (package / "leak.py").write_text("from osprey.apps.sdk._approval import x\n")

    [violation] = check_package(package)

    assert violation.field == "osprey.apps.sdk._approval.x"
    assert violation.source == "boundary_fixture_app/leak.py:1"


@pytest.mark.parametrize(
    "source",
    [
        "import osprey.apps.sdk",
        "import osprey.apps.sdk as sdk",
        "from osprey.apps.sdk import serve_tools",
        "from osprey.apps import sdk",
        "import osprey.facility.kinds",
        "from osprey.facility import kinds",
        "from osprey.facility.kinds import declare",
        "import osprey_connectors",
        "from osprey_connectors import types",
        "from osprey_connectors.types import tier_writes",
        "from osprey_connectors.control_system import write_door",
        "from osprey_connectors.control_system.write_door import door",
        "from osprey_connectors.simulation.state import x",
        "from .sibling import x",
        "from . import sibling",
        "import numpy",
        "from some_app.sibling import x",
        "import ospreyish.apps.loader",
    ],
)
def test_allowed_import(tmp_path: Path, source: str) -> None:
    assert check_package(_package(tmp_path, source)) == []


@pytest.mark.parametrize(
    ("source", "imported"),
    [
        ("import osprey", "osprey"),
        ("import osprey.apps", "osprey.apps"),
        ("import osprey.apps.loader", "osprey.apps.loader"),
        ("import osprey.apps.sdk._approval", "osprey.apps.sdk._approval"),
        ("from osprey import apps", "osprey.apps"),
        ("from osprey import *", "osprey"),
        ("from osprey.apps import loader", "osprey.apps.loader"),
        ("from osprey.apps.manifest import AppManifest", "osprey.apps.manifest.AppManifest"),
        ("from osprey.apps.sdk import _approval", "osprey.apps.sdk._approval"),
        ("from osprey.facility import other", "osprey.facility.other"),
        ("from osprey.facility.kinds_extra import x", "osprey.facility.kinds_extra.x"),
        ("from osprey_connectors import _version", "osprey_connectors._version"),
        (
            "from osprey_connectors.control_system._base import ControlSystemConnector",
            "osprey_connectors.control_system._base.ControlSystemConnector",
        ),
        ("import osprey_connectors.ipc._host", "osprey_connectors.ipc._host"),
        ("def f():\n    import osprey.cli", "osprey.cli"),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from osprey.registry import x",
            "osprey.registry.x",
        ),
    ],
)
def test_refused_import(tmp_path: Path, source: str, imported: str) -> None:
    [violation] = check_package(_package(tmp_path, source))

    assert violation.field == imported
    assert violation.rule == "import_boundary"
    assert imported in str(violation)
    assert "osprey.apps.sdk" in violation.message


def test_only_the_refused_name_of_a_mixed_import_is_reported(tmp_path: Path) -> None:
    package = _package(tmp_path, "from osprey.apps import sdk, loader")

    assert [v.field for v in check_package(package)] == ["osprey.apps.loader"]


def test_a_violation_names_the_file_line_and_application(tmp_path: Path) -> None:
    package = _package(tmp_path, "\n\nimport osprey.cli", name="deep/module.py")

    [violation] = check_package(package, app="knob")

    assert violation.app == "knob"
    assert violation.source == "some_app/deep/module.py:3"
    assert str(violation).startswith("some_app/deep/module.py:3: osprey.cli: ")


def test_a_relative_import_inside_the_package_is_its_own(tmp_path: Path) -> None:
    package = _package(tmp_path, "from ...other import x", name="a/b/module.py")

    assert check_package(package) == []


def test_a_relative_import_above_the_package_is_refused(tmp_path: Path) -> None:
    package = _package(tmp_path, "from ...other import x", name="a/module.py")

    [violation] = check_package(package)

    assert violation.rule == "relative_import_escape"
    assert violation.field == "...other"


def test_a_file_that_cannot_be_read_fails_instead_of_passing(tmp_path: Path) -> None:
    [violation] = check_package(_package(tmp_path, "def broken(:"))

    assert violation.rule == "unreadable_source"
    assert violation.source == "some_app/module.py"


def test_a_missing_package_is_refused_not_passed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a package directory"):
        check_package(tmp_path / "absent")
