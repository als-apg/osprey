"""Application code reaches OSPREY only through its public surfaces.

The checker reads a package by source path, so it covers packages no ``pyproject.toml``
contract can name: import-linter refuses a listed module that is absent from its graph,
and the fixtures and a deployment's own packages are not installed there.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.apps.boundary import check_package

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_APP = REPO_ROOT / "tests/apps/fixtures/boundary_fixture_app"
SHIPPED_PACKAGES = (
    "src/osprey/apps/scaffold-app/scaffold_app",
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


@pytest.mark.parametrize("relative", SHIPPED_PACKAGES)
def test_a_shipped_package_stays_inside_the_boundary(relative: str) -> None:
    assert check_package(REPO_ROOT / relative) == []


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
        "import osprey.apps.sdk.testing",
        "from osprey.apps.sdk.testing import synthetic_deployment",
        "from osprey.apps.sdk import testing",
        "import osprey_facility.kinds",
        "from osprey_facility import kinds",
        "from osprey_facility.kinds import KindDeclaration",
        "import osprey_connectors",
        "from osprey_connectors import types",
        "from osprey_connectors.types import tier_writes",
        "from osprey_connectors.control_system import write_door",
        "from osprey_connectors.control_system.write_door import open_door",
        "from osprey_connectors.simulation.state import read_active_state",
        "from osprey_connectors.control_system.write_door import *",
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
        ("import osprey.apps.sdk.testing._facility", "osprey.apps.sdk.testing._facility"),
        (
            "from osprey.apps.sdk.testing._build import invoke_build",
            "osprey.apps.sdk.testing._build.invoke_build",
        ),
        ("from osprey.facility import other", "osprey.facility.other"),
        ("from osprey.facility.kinds_extra import x", "osprey.facility.kinds_extra.x"),
        ("import osprey_facility", "osprey_facility"),
        ("import osprey_facility.records", "osprey_facility.records"),
        ("from osprey_facility import records", "osprey_facility.records"),
        (
            "from osprey_facility.records import FacilityRecords",
            "osprey_facility.records.FacilityRecords",
        ),
        (
            "from osprey_facility.reading import load_facility",
            "osprey_facility.reading.load_facility",
        ),
        ("from osprey_facility.kinds_extra import x", "osprey_facility.kinds_extra.x"),
        ("import osprey_facility._version", "osprey_facility._version"),
        ("from osprey_facility import _version", "osprey_facility._version"),
        ("from osprey_facility.kinds._private import x", "osprey_facility.kinds._private.x"),
        ("import osprey_facility.kinds._private", "osprey_facility.kinds._private"),
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


# --- no application package lives under a framework root ---------------------------


def _nested_package(tmp_path: Path, parent: str, source: str) -> Path:
    """A package placed at ``<parent>/fixture`` of a site directory."""
    package = tmp_path.joinpath(*parent.split("."), "fixture")
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "helpers.py").write_text("")
    (package / "tools.py").write_text(source + "\n")
    return package


def test_a_nested_package_imports_itself_by_its_dotted_name(tmp_path: Path) -> None:
    package = _nested_package(tmp_path, "knob_dist", "from knob_dist.fixture.helpers import x")

    assert check_package(package, app="fixture") == []


def test_a_package_under_a_framework_root_has_no_imports_of_its_own(tmp_path: Path) -> None:
    """One rule for every origin: an import under ``osprey`` is OSPREY's, never the app's."""
    package = _nested_package(tmp_path, "osprey.apps", "from osprey.apps.fixture import helpers")

    [violation] = check_package(package, app="fixture")

    assert violation.rule == "import_boundary"
    assert violation.field == "osprey.apps.fixture.helpers"


def test_a_nested_packages_relative_escape_is_measured_from_its_directory(
    tmp_path: Path,
) -> None:
    package = _nested_package(tmp_path, "knob_dist", "from . import helpers\nfrom .. import loader")

    [violation] = check_package(package)

    assert violation.rule == "relative_import_escape"
    assert violation.field == ".."
    assert violation.source == "fixture/tools.py:2"


# --- an allowed import resolves ----------------------------------------------------


@pytest.mark.parametrize(
    ("source", "imported"),
    [
        ("from osprey.apps.sdk import run", "osprey.apps.sdk.run"),
        ("from osprey_facility.kinds import declare", "osprey_facility.kinds.declare"),
        (
            "from osprey_connectors.control_system.write_door import door",
            "osprey_connectors.control_system.write_door.door",
        ),
        (
            "from osprey_connectors.simulation.state import x",
            "osprey_connectors.simulation.state.x",
        ),
        ("from osprey_connectors import absent_module", "osprey_connectors.absent_module"),
    ],
)
def test_a_name_its_module_does_not_define_is_refused(
    tmp_path: Path, source: str, imported: str
) -> None:
    [violation] = check_package(_package(tmp_path, source))

    assert violation.rule == "import_missing_name"
    assert violation.field == imported
    assert violation.source == "some_app/module.py:1"


@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("import osprey_connectors.absent_module", "osprey_connectors.absent_module"),
        ("from osprey_connectors.absent_module import x", "osprey_connectors.absent_module"),
        ("from osprey_connectors.absent_module import *", "osprey_connectors.absent_module"),
        (
            "from osprey_connectors.absent_package.state import x",
            "osprey_connectors.absent_package.state",
        ),
        ("import osprey.apps.sdk.absent_module", "osprey.apps.sdk.absent_module"),
        ("import osprey_facility.kinds.absent_module", "osprey_facility.kinds.absent_module"),
    ],
)
def test_a_module_that_does_not_exist_is_refused(tmp_path: Path, source: str, module: str) -> None:
    [violation] = check_package(_package(tmp_path, source))

    assert violation.rule == "import_missing_module"
    assert violation.field == module


def test_a_refused_import_is_not_also_resolved(tmp_path: Path) -> None:
    [violation] = check_package(_package(tmp_path, "from osprey.apps.nothing import x"))

    assert violation.rule == "import_boundary"


@pytest.mark.parametrize(
    "source",
    [
        "from some_app.sibling import x",
        "import some_app.sibling",
        "import numpy",
        "from numpy import not_a_numpy_name",
        "import not_an_installed_module",
        "from .sibling import x",
        "from . import sibling",
    ],
)
def test_an_import_outside_the_framework_is_never_resolved(tmp_path: Path, source: str) -> None:
    assert check_package(_package(tmp_path, source)) == []


def test_an_import_that_fails_for_another_reason_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osprey_connectors

    site = tmp_path / "site"
    (site / "broken_package").mkdir(parents=True)
    (site / "broken_module.py").write_text("raise RuntimeError('broken at import')\n")
    (site / "broken_package" / "__init__.py").write_text("raise RuntimeError('broken')\n")
    monkeypatch.setattr(osprey_connectors, "__path__", [*osprey_connectors.__path__, str(site)])
    for name in ("broken_module", "broken_package", "broken_package.state"):
        monkeypatch.delitem(sys.modules, f"osprey_connectors.{name}", raising=False)
    source = "\n".join(
        [
            "import osprey_connectors.broken_module",
            "from osprey_connectors.broken_module import anything",
            "from osprey_connectors.broken_package.state import anything",
        ]
    )

    assert check_package(_package(tmp_path / "app", source)) == []


#: Records every subprocess start and socket use, then checks the package in argv[1].
_NO_SIDE_EFFECTS = """
import json, socket, subprocess, sys

calls = []

class RecordingPopen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        calls.append(("Popen", repr(args)))
        super().__init__(*args, **kwargs)

subprocess.Popen = RecordingPopen

def recorder(name, original):
    def record(*args, **kwargs):
        calls.append((name, repr(args)))
        return original(*args, **kwargs)
    return record

for attr in ("connect", "connect_ex", "sendto"):
    setattr(socket.socket, attr, recorder(attr, getattr(socket.socket, attr)))
socket.create_connection = recorder("create_connection", socket.create_connection)
socket.getaddrinfo = recorder("getaddrinfo", socket.getaddrinfo)

from pathlib import Path
from osprey.apps.boundary import check_package

violations = [str(v) for v in check_package(Path(sys.argv[1]))]
print(json.dumps({"violations": violations, "calls": calls}))
"""


def test_resolving_imports_starts_no_process_and_opens_no_connection(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "from osprey.apps.sdk import serve_tools",
            "from osprey.apps.sdk.testing import synthetic_deployment",
            "from osprey_facility.kinds import KindDeclaration",
            "from osprey_connectors.types import tier_writes",
            "from osprey_connectors.config import get_config_value",
            "from osprey_connectors.control_system.write_door import open_door",
            "from osprey_connectors.simulation.state import read_active_state",
            "import osprey_connectors.posture_store",
        ]
    )
    package = _package(tmp_path, source)

    result = subprocess.run(
        [sys.executable, "-c", _NO_SIDE_EFFECTS, str(package)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])

    assert report == {"violations": [], "calls": []}
