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
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "packages" / "osprey-connectors" / "src" / "osprey_connectors"
FRAMEWORK = "osprey"

# Below the package's module count and above that of any one subpackage, so a
# walk rooted at the wrong directory, or one that finds nothing, fails here.
MODULE_FLOOR = 40

_IMPORT_CALLS = frozenset({"import_module", "__import__"})


def _names_framework(module: str) -> bool:
    return module == FRAMEWORK or module.startswith(FRAMEWORK + ".")


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


def framework_imports(source: str, filename: str = "<source>") -> list[tuple[int, str]]:
    """Return ``(line, module)`` for every import of ``osprey`` in *source*."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            found.extend(
                (node.lineno, alias.name) for alias in node.names if _names_framework(alias.name)
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _names_framework(node.module):
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            module = _literal_module_name(node)
            if module is not None and _names_framework(module):
                found.append((node.lineno, module))
    return sorted(found)


def _package_modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_no_module_in_the_connectors_package_imports_osprey() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path in _package_modules()
        for line, module in framework_imports(path.read_text(encoding="utf-8"), str(path))
    ]
    assert not offenders, "the connectors package imports osprey:\n" + "\n".join(offenders)


def test_the_walk_covers_the_whole_package() -> None:
    modules = _package_modules()
    assert PACKAGE / "__init__.py" in modules, f"no package at {PACKAGE}"
    assert len(modules) >= MODULE_FLOOR, (
        f"walked {len(modules)} modules under {PACKAGE}, expected at least {MODULE_FLOOR}"
    )


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
