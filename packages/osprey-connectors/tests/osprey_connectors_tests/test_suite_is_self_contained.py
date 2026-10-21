"""This suite imports nothing from outside its own tree.

The connectors suite has to pass where only ``osprey-connectors`` is installed,
so it can reach neither the framework (``osprey``) nor the framework's test
package (``tests``, the repository root's). An import of either only fails on
the host that lacks it, which is the standalone lane alone; everywhere else
the monorepo quietly provides it. The scan here refuses it on every host, the
same way ``test_package_never_imports_osprey.py`` does for the distribution
itself: every ``import`` and ``from ... import`` wherever it sits, and every
``importlib.import_module`` or ``__import__`` call that names its module as a
string literal.

The other way in is the package's own pytest configuration. A ``pythonpath``
entry that reaches past ``packages/osprey-connectors`` puts the checkout on
``sys.path`` and makes ``tests`` importable again, so every entry must resolve
inside the package.

The framework's suite importing ``osprey_connectors_tests`` is the permitted
direction and is not checked here.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

from osprey_connectors_tests._paths import PACKAGE_ROOT, REPO_ROOT, TESTS_ROOT
from osprey_connectors_tests.test_package_never_imports_osprey import _literal_module_name

#: The top-level packages this suite may not import.
FORBIDDEN = frozenset({"osprey", "tests"})

# Below the suite's module count and above that of any one subpackage, so a
# walk rooted at the wrong directory, or one that finds nothing, fails here.
MODULE_FLOOR = 60


def _forbidden(module: str) -> bool:
    return module.split(".", 1)[0] in FORBIDDEN


def outside_imports(source: str, filename: str = "<source>") -> list[tuple[int, str]]:
    """Return ``(line, module)`` for every import of ``osprey`` or ``tests`` in *source*."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            found.extend(
                (node.lineno, alias.name) for alias in node.names if _forbidden(alias.name)
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _forbidden(node.module):
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            module = _literal_module_name(node)
            if module is not None and _forbidden(module):
                found.append((node.lineno, module))
    return sorted(found)


def _suite_modules() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_module_in_the_suite_imports_the_framework_or_its_tests() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path in _suite_modules()
        for line, module in outside_imports(path.read_text(encoding="utf-8"), str(path))
    ]
    assert not offenders, (
        "the connectors suite imports from outside its tree — a helper both suites "
        "need belongs in osprey_connectors_tests._harness:\n" + "\n".join(offenders)
    )


def test_the_walk_covers_the_whole_suite() -> None:
    modules = _suite_modules()
    for anchor in (
        TESTS_ROOT / "conftest.py",
        TESTS_ROOT / "osprey_connectors_tests" / "_harness" / "_guards.py",
    ):
        assert anchor in modules, f"the walk under {TESTS_ROOT} did not reach {anchor}"
    assert len(modules) >= MODULE_FLOOR, (
        f"walked {len(modules)} modules under {TESTS_ROOT}, expected at least {MODULE_FLOOR}"
    )


def test_the_pytest_pythonpath_stays_inside_the_package() -> None:
    config = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = config["tool"]["pytest"]["ini_options"].get("pythonpath", [])
    if isinstance(entries, str):
        entries = entries.split()
    outside = [
        entry
        for entry in entries
        if not (PACKAGE_ROOT / entry).resolve().is_relative_to(PACKAGE_ROOT)
    ]
    assert not outside, (
        f"pythonpath entries in {PACKAGE_ROOT / 'pyproject.toml'} reach outside the "
        f"package: {outside}"
    )


#: The framework's test package, spelled once so the sources below do not read
#: as imports of it to a plain-text search of this tree.
_TESTS = "tests"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(f"import {_TESTS}\n", [(1, _TESTS)], id="import-tests"),
        pytest.param(
            f"from {_TESTS}._ports import reserved_port\n",
            [(1, f"{_TESTS}._ports")],
            id="from-tests",
        ),
        pytest.param(f"from {_TESTS} import _ports\n", [(1, _TESTS)], id="from-tests-package"),
        pytest.param("import osprey.errors\n", [(1, "osprey.errors")], id="import-osprey"),
        pytest.param(
            f"def f():\n    from {_TESTS}.conftest import x\n",
            [(2, f"{_TESTS}.conftest")],
            id="local",
        ),
        pytest.param(
            f"import importlib\nimportlib.import_module('{_TESTS}._ports')\n",
            [(2, f"{_TESTS}._ports")],
            id="importlib-literal",
        ),
    ],
)
def test_every_outside_import_is_caught(source: str, expected: list[tuple[int, str]]) -> None:
    assert outside_imports(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from osprey_connectors_tests._harness import _ports\n", id="own-suite"),
        pytest.param("import osprey_connectors\n", id="the-package"),
        pytest.param("from . import _paths\n", id="relative"),
        pytest.param(f"import {_TESTS}uite\n", id="prefix-lookalike"),
        pytest.param("ROOT = 'tests'\n", id="plain-string"),
    ],
)
def test_the_suite_s_own_imports_pass(source: str) -> None:
    assert outside_imports(source) == []
