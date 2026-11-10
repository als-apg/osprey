"""The application imports OSPREY only through the application surfaces."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import orbit_bump

from osprey.apps.boundary import check_package

PACKAGE = Path(orbit_bump.__file__).resolve().parent


def test_the_package_stays_inside_the_boundary() -> None:
    assert sorted(path.name for path in PACKAGE.glob("*.py")) == [
        "__init__.py",
        "orm.py",
        "plans.py",
        "ring.py",
        "server.py",
        "solver.py",
        "tools.py",
    ]
    assert check_package(PACKAGE) == []


def test_the_check_reads_the_package_it_is_pointed_at(tmp_path: Path) -> None:
    copy = tmp_path / PACKAGE.name
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "tools.py").write_text("import osprey.cli\n")

    [violation] = check_package(copy)

    assert violation.field == "osprey.cli"


def test_the_tool_server_does_not_load_the_backend() -> None:
    tree = ast.parse((PACKAGE / "tools.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "orbit_bump.server" not in imported
    assert "numpy" not in imported
