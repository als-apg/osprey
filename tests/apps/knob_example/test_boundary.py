"""The example imports OSPREY only through the application surfaces."""

from __future__ import annotations

import shutil
from pathlib import Path

import knob_example

from osprey.apps.boundary import check_package

PACKAGE = Path(knob_example.__file__).resolve().parent


def test_the_package_stays_inside_the_boundary() -> None:
    assert sorted(path.name for path in PACKAGE.glob("*.py")) == [
        "__init__.py",
        "server.py",
        "tools.py",
    ]
    assert check_package(PACKAGE) == []


def test_the_check_reads_the_package_it_is_pointed_at(tmp_path: Path) -> None:
    copy = tmp_path / PACKAGE.name
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "tools.py").write_text("import osprey.cli\n")

    [violation] = check_package(copy)

    assert violation.field == "osprey.cli"
