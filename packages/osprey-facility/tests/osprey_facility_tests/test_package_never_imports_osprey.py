"""No module in the facility distribution imports ``osprey``.

``osprey-facility`` is installed and run where the framework is absent: a fork
or tool that builds or reads a description, a measurement-kind plug-in, an app
that reads kinds. A single ``osprey`` import anywhere in it breaks those hosts,
but only on the code path that reaches it, so an import inside a function or a
rarely taken branch passes every test that merely imports the module.

This module reads the source instead of running it. Every ``import`` and
``from ... import`` statement is checked wherever it sits -- module level,
inside a function, under ``if TYPE_CHECKING:`` -- together with
``importlib.import_module`` and ``__import__`` calls whose module name is a
string literal. ``osprey_facility`` and ``osprey_connectors`` are distinct
top-level packages and relative imports stay inside the package, so none of
them is an offender.

The facility test suite is held to the same scan, and may import neither the
framework's ``tests`` package nor the connectors suite's
``osprey_connectors_tests``: on any host but the standalone lane the monorepo
would quietly provide them, so a stray import would pass until then. A helper
the suite needs belongs in ``osprey_facility_tests._harness``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from osprey_facility_tests._harness._paths import PACKAGE_SRC, REPO_ROOT, TESTS_ROOT

PACKAGE = PACKAGE_SRC / "osprey_facility"
FRAMEWORK = frozenset({"osprey"})
SUITE_FORBIDDEN = frozenset({"osprey", "tests", "osprey_connectors_tests"})

# Below each tree's module count and above that of any one subpackage, so a
# walk rooted at the wrong directory, or one that finds nothing, fails here.
# The package holds 60 modules, its largest subpackage (``layers``) 29. The
# suite holds 58: the conftest, the ``_harness`` modules, the helper halves the
# root suite imports, and the description tests moved in from ``tests/facility``.
MODULE_FLOOR = 40
SUITE_MODULE_FLOOR = 50

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
    source: str, filename: str = "<source>", forbidden: frozenset[str] = FRAMEWORK
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


@pytest.mark.parametrize(
    ("root", "forbidden"),
    [
        pytest.param(PACKAGE, FRAMEWORK, id="package"),
        pytest.param(TESTS_ROOT, SUITE_FORBIDDEN, id="test-suite"),
    ],
)
def test_no_module_imports_the_framework(root: Path, forbidden: frozenset[str]) -> None:
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
        pytest.param(
            TESTS_ROOT,
            TESTS_ROOT / "osprey_facility_tests" / "_harness" / "_guards.py",
            SUITE_MODULE_FLOOR,
            id="test-suite",
        ),
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
    ],
)
def test_every_import_form_is_caught(source: str, expected: list[tuple[int, str]]) -> None:
    assert framework_imports(source) == expected


def test_the_suite_may_not_import_the_framework_or_connectors_tests() -> None:
    source = (
        "from tests._builds import FRAMEWORK_CHECKS\n"
        "from osprey_connectors_tests._paths import REPO_ROOT\n"
        "from osprey_facility_tests import _harness\n"
    )
    assert framework_imports(source, forbidden=SUITE_FORBIDDEN) == [
        (1, "tests._builds"),
        (2, "osprey_connectors_tests._paths"),
    ]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("import osprey_facility.build\n", id="own-package"),
        pytest.param("from osprey_facility import kinds\n", id="own-package-from"),
        pytest.param("from osprey_connectors import errors\n", id="connectors"),
        pytest.param("from . import kinds\nfrom .osprey import x\n", id="relative"),
        pytest.param("import ospreyish\n", id="prefix-lookalike"),
        pytest.param("import importlib\nm = importlib.import_module(path)\n", id="non-literal"),
        pytest.param("NAME = 'osprey.facility'\n", id="plain-string"),
    ],
)
def test_in_package_and_non_import_forms_pass(source: str) -> None:
    assert framework_imports(source) == []
