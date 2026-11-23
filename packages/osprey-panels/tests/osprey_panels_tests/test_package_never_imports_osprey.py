"""No module in the panels distribution imports the framework or the facility package.

``osprey-panels`` is installed and run where neither the framework nor the
facility package is present: a fork or tool that draws screens from a built
``facility.json``, and the ``osprey-panels`` command. The two packages meet only
in that file. A single ``osprey``, ``osprey_facility`` or ``osprey_connectors``
import anywhere in the engine breaks those hosts, but only on the code path that
reaches it, so an import inside a function or a rarely taken branch passes every
test that merely imports the module.

This module reads the source instead of running it. Every ``import`` and
``from ... import`` statement is checked wherever it sits -- module level,
inside a function, under ``if TYPE_CHECKING:`` -- together with
``importlib.import_module`` and ``__import__`` calls whose module name is a
string literal. ``osprey_panels`` is a distinct top-level package and relative
imports stay inside the package, so neither is an offender.

The panels test suite is held to the same scan, and may additionally import
neither the framework's ``tests`` package nor the connectors or facility
suites (``osprey_connectors_tests``, ``osprey_facility_tests``): on any host but
the standalone lane the monorepo would quietly provide them, so a stray import
would pass until then. A helper the suite needs belongs in
``osprey_panels_tests._harness``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from osprey_panels_tests._harness._paths import PACKAGE_SRC, REPO_ROOT, TESTS_ROOT

PACKAGE = PACKAGE_SRC / "osprey_panels"
FORBIDDEN = frozenset({"osprey", "osprey_facility", "osprey_connectors"})
SUITE_FORBIDDEN = FORBIDDEN | {"tests", "osprey_connectors_tests", "osprey_facility_tests"}

# Below each tree's module count, so a walk rooted at the wrong directory, or
# one that finds nothing, fails here. The package is flat: 19 modules, and a
# 20th, ``_version.py``, once a build has written it. The suite holds 41: the
# conftest above the package, the ``_harness`` modules and the panel tests; a
# walk rooted at ``osprey_panels_tests`` misses the conftest anchor.
MODULE_FLOOR = 15
SUITE_MODULE_FLOOR = 30

_IMPORT_CALLS = frozenset({"import_module", "__import__"})


def _literal_module_name(call: ast.Call) -> str | None:
    """Return the module a dynamic-import call names as a string literal."""
    func = call.func
    if isinstance(func, ast.Attribute):
        called = func.attr
    elif isinstance(func, ast.Name):
        called = func.id
    else:
        return None
    if called not in _IMPORT_CALLS:
        return None
    if call.args:
        target = call.args[0]
    else:
        target = next((kw.value for kw in call.keywords if kw.arg == "name"), None)
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value
    return None


def framework_imports(
    source: str, filename: str = "<source>", forbidden: frozenset[str] = FORBIDDEN
) -> list[tuple[int, str]]:
    """Return ``(line, module)`` for every import of a *forbidden* package in *source*."""

    def named(module: str) -> bool:
        return module.split(".", 1)[0] in forbidden

    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names if named(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and named(node.module):
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            module = _literal_module_name(node)
            if module is not None and named(module):
                found.append((node.lineno, module))
    return sorted(found)


def _modules(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _modules(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize(
    ("root", "forbidden"),
    [
        pytest.param(PACKAGE, FORBIDDEN, id="package"),
        pytest.param(TESTS_ROOT, SUITE_FORBIDDEN, id="test-suite"),
    ],
)
def test_no_module_imports_a_forbidden_package(root: Path, forbidden: frozenset[str]) -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path in _modules(root)
        for line, module in framework_imports(
            path.read_text(encoding="utf-8"), str(path), forbidden
        )
    ]
    assert not offenders, f"{root} imports {sorted(forbidden)}:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    ("root", "anchor", "floor"),
    [
        pytest.param(PACKAGE, PACKAGE / "__init__.py", MODULE_FLOOR, id="package"),
        pytest.param(TESTS_ROOT, TESTS_ROOT / "conftest.py", SUITE_MODULE_FLOOR, id="test-suite"),
    ],
)
def test_the_walk_covers_the_whole_tree(root: Path, anchor: Path, floor: int) -> None:
    modules = _modules(root)
    assert anchor in modules, f"the walk under {root} did not reach {anchor}"
    assert len(modules) >= floor, f"walked {len(modules)} modules under {root}, expected {floor}+"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param("import osprey\n", [(1, "osprey")], id="import"),
        pytest.param("import os, osprey.errors as e\n", [(1, "osprey.errors")], id="import-dotted"),
        pytest.param("from osprey import errors\n", [(1, "osprey")], id="from-import"),
        pytest.param(
            "from osprey.utils.config import get\n", [(1, "osprey.utils.config")], id="from-dotted"
        ),
        pytest.param(
            "def read():\n    from osprey.registry import X\n    return X\n",
            [(2, "osprey.registry")],
            id="function-local",
        ),
        pytest.param(
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import osprey.cli\n",
            [(3, "osprey.cli")],
            id="type-checking",
        ),
        pytest.param(
            "import importlib\nm = importlib.import_module('osprey.facility')\n",
            [(2, "osprey.facility")],
            id="importlib-literal",
        ),
        pytest.param(
            "from importlib import import_module\nm = import_module(name='osprey')\n",
            [(2, "osprey")],
            id="import-module-keyword",
        ),
        pytest.param(
            "m = __import__('osprey.errors')\n", [(1, "osprey.errors")], id="dunder-import"
        ),
        pytest.param(
            "from osprey_facility.validate import FACILITY_HEADER\n",
            [(1, "osprey_facility.validate")],
            id="facility",
        ),
        pytest.param("import osprey_facility\n", [(1, "osprey_facility")], id="facility-import"),
        pytest.param(
            "def read():\n    from osprey_connectors import errors\n",
            [(2, "osprey_connectors")],
            id="connectors",
        ),
    ],
)
def test_every_import_form_is_caught(source: str, expected: list[tuple[int, str]]) -> None:
    assert framework_imports(source) == expected


def test_the_suite_may_not_import_another_suite() -> None:
    source = (
        "from tests._builds import FRAMEWORK_CHECKS\n"
        "from osprey_connectors_tests._paths import REPO_ROOT\n"
        "from osprey_facility_tests._harness import _paths\n"
        "from osprey_panels_tests import _harness\n"
    )
    assert framework_imports(source, forbidden=SUITE_FORBIDDEN) == [
        (1, "tests._builds"),
        (2, "osprey_connectors_tests._paths"),
        (3, "osprey_facility_tests._harness"),
    ]


def test_only_the_suite_is_refused_the_framework_tests_package() -> None:
    """Control: ``tests`` is a forbidden root for the suite, not for the package."""
    assert framework_imports("import tests\n") == []


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("import osprey_panels.emit\n", id="own-package"),
        pytest.param("from osprey_panels import layout\n", id="own-package-from"),
        pytest.param("from . import layout\nfrom .osprey import x\n", id="relative"),
        pytest.param("import ospreyish\nimport osprey_facilityish\n", id="prefix-lookalike"),
        pytest.param("import importlib\nm = importlib.import_module(path)\n", id="non-literal"),
        pytest.param("NAME = 'osprey_facility.validate'\n", id="plain-string"),
    ],
)
def test_in_package_and_non_import_forms_pass(source: str) -> None:
    assert framework_imports(source) == []
