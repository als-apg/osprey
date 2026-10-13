"""No module in the connectors distribution imports ``osprey`` by name at run time.

``osprey-connectors`` is installed and run where the framework is absent:
connector-host children, executor sandboxes, notebook kernels and external
consumers of the distribution alone. A single ``osprey`` import anywhere in
it breaks those hosts, but only on the code path that reaches it.

Every ``import`` and ``from ... import`` statement, wherever it sits, is held
by the ``connectors-standalone`` contract that ``lint-imports`` checks (see
``[tool.importlinter]`` in the root ``pyproject.toml``). What an import graph
cannot see is a module named in a string, so this file reads the source for
``importlib.import_module`` and ``__import__`` calls whose module name is a
string literal. ``osprey_connectors`` is the package itself, so it is not an
offender.
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
    """Return ``(line, module)`` for every dynamic import of ``osprey`` in *source*."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Call):
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
            "def load():\n    return __import__('osprey.errors')\n",
            [(2, "osprey.errors")],
            id="dunder-import-in-function",
        ),
    ],
)
def test_every_dynamic_form_is_caught(source: str, expected: list[tuple[int, str]]) -> None:
    assert framework_imports(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("m = __import__('osprey_connectors.errors')\n", id="own-package"),
        pytest.param("m = __import__('ospreyish')\n", id="prefix-lookalike"),
        pytest.param("import importlib\nm = importlib.import_module(path)\n", id="non-literal"),
        pytest.param("NAME = 'osprey.errors'\n", id="plain-string"),
        # A statement is lint-imports' to judge, not this file's.
        pytest.param("import osprey\n", id="statement"),
    ],
)
def test_in_package_and_non_import_forms_pass(source: str) -> None:
    assert framework_imports(source) == []
