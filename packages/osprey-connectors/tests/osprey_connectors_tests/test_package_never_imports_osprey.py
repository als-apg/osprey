"""No module in the connectors distribution imports ``osprey``.

``osprey-connectors`` is installed and run where the framework is absent:
connector-host children, executor sandboxes, notebook kernels and external
consumers of the distribution alone. A single ``osprey`` import anywhere in
it breaks those hosts, but only on the code path that reaches it, so an
import inside a function or a rarely taken branch passes every test that
merely imports the module (``test_import_isolation.py`` proves the runtime
side for the modules it loads).

This module reads the source instead of running it. Every ``import`` and
``from ... import`` statement is checked wherever it sits -- module level,
inside a function, under ``if TYPE_CHECKING:`` -- together with
``importlib.import_module`` and ``__import__`` calls whose module name is a
string literal. ``osprey_connectors`` is the package itself and relative
imports stay inside it, so neither is an offender.

The connectors test suite is held to the same scan, and may not import the
framework's ``tests`` package either: on any host but the standalone lane the
monorepo would quietly provide both, so a stray import would pass until then.
A helper both suites need belongs in ``osprey_connectors_tests._harness``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from osprey_connectors_tests._paths import PACKAGE_SRC, REPO_ROOT, TESTS_ROOT

PACKAGE = PACKAGE_SRC / "osprey_connectors"
FRAMEWORK = frozenset({"osprey"})
SUITE_FORBIDDEN = frozenset({"osprey", "tests"})

# Below each tree's module count and above that of any one subpackage, so a
# walk rooted at the wrong directory, or one that finds nothing, fails here.
MODULE_FLOOR = 40
SUITE_MODULE_FLOOR = 60

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
            TESTS_ROOT / "osprey_connectors_tests" / "_harness" / "_guards.py",
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
            "def read():\n    from osprey.errors import X\n    return X\n",
            [(2, "osprey.errors")],
            id="function-local",
        ),
        pytest.param(
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import osprey.simulation\n",
            [(3, "osprey.simulation")],
            id="type-checking",
        ),
        pytest.param(
            "import importlib\nm = importlib.import_module('osprey.simulation')\n",
            [(2, "osprey.simulation")],
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


def test_the_suite_may_not_import_the_framework_tests_package() -> None:
    source = "from tests._ports import reserved_port\nfrom osprey_connectors_tests import _paths\n"
    assert framework_imports(source, forbidden=SUITE_FORBIDDEN) == [(1, "tests._ports")]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("import osprey_connectors.errors\n", id="own-package"),
        pytest.param("from osprey_connectors import config\n", id="own-package-from"),
        pytest.param("from . import errors\nfrom .osprey import x\n", id="relative"),
        pytest.param("import ospreyish\n", id="prefix-lookalike"),
        pytest.param("import importlib\nm = importlib.import_module(path)\n", id="non-literal"),
        pytest.param("NAME = 'osprey.errors'\n", id="plain-string"),
    ],
)
def test_in_package_and_non_import_forms_pass(source: str) -> None:
    assert framework_imports(source) == []
