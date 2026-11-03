"""The ``canopy`` marker: which tests carry it, and what a run says about it.

The rule is the ``canopy`` marker's registered text in pyproject.toml; this module implements it.

Nothing here imports CANOPY: the root conftest imports this module in every
lane, and most lanes have no CANOPY installed. Which CANOPY is installed, and
whether it is the proven one, is read by :mod:`osprey.integrations.canopy.proven`
alone, which imports nothing outside the standard library.
"""

from __future__ import annotations

import ast
import functools
import importlib.metadata
import importlib.util
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from osprey.integrations.canopy import proven
from tests._tree_scan import TESTS_ROOT, python_sources

#: The top-level packages CANOPY and its converter install.
CANOPY_PACKAGES = ("canopy", "canopy_mcp", "pydmconverter")

MARKER = "canopy"

#: A well-formed commit that is not a proven one, for a test that needs another.
OTHER_COMMIT = "e5e62dad7" + "0" * 31

#: The name of the one item that stands in for a module that cannot be imported.
ABSENT_ITEM_NAME = "canopy_not_installed"

ABSENT_TEXT = (
    "CANOPY is not installed: install it at the commit osprey.integrations.canopy pins, "
    'or deselect with -m "not canopy"'
)

SKIP_REFUSED_TEXT = "a CANOPY contract test may not skip"

NOT_EXERCISED_TEXT = (
    "CANOPY contract not exercised here: run uv run python scripts/check.py run canopy-contract"
)

#: The stats categories that hold collected items rather than reports of a run.
_NOT_RUN_CATEGORIES = ("deselected",)

#: The stats categories whose reports are a test that did not pass.
_FAILED_CATEGORIES = ("failed", "error")

#: The report attribute that says its test carries the ``canopy`` mark.
_MARKED_ATTRIBUTE = "canopy_marked"

_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _imported_top_levels(node: ast.AST) -> Iterator[str]:
    """Top-level package names imported by *node* outside any function body."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _FUNCTION_SCOPES):
            continue
        if isinstance(child, ast.Import):
            for alias in child.names:
                yield alias.name.split(".", 1)[0]
        elif isinstance(child, ast.ImportFrom):
            if child.level == 0 and child.module:
                yield child.module.split(".", 1)[0]
        else:
            yield from _imported_top_levels(child)


def source_needs_canopy(source: str) -> bool:
    """Whether *source* imports a CANOPY package at import time.

    Module-level and class-body imports count; an import inside a function body
    does not, because collecting the module does not run it.
    """
    return any(name in CANOPY_PACKAGES for name in _imported_top_levels(ast.parse(source)))


def needs_canopy(path: Path) -> bool:
    """Whether the module at *path* cannot be imported without CANOPY.

    A module whose bytes name no CANOPY package is never parsed, so collecting
    the whole suite costs one read per module. A module that does not decode
    or parse is left to pytest, which reports its own collection error.
    """
    data = Path(path).read_bytes()
    if not any(name.encode() in data for name in CANOPY_PACKAGES):
        return False
    try:
        return source_needs_canopy(data.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False


def dotted(node: ast.AST) -> str:
    """``pytest.mark.canopy`` for the attribute chain that spells it, else ``""``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    return ".".join([node.id, *reversed(parts)])


def hand_marks(tree: ast.Module) -> set[str]:
    """Qualified names of the test functions and classes decorated ``pytest.mark.canopy``."""
    found: set[str] = set()
    spelled = f"pytest.mark.{MARKER}"

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if any(dotted(decorator) == spelled for decorator in node.decorator_list):
                found.add(prefix + node.name)
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    visit(tree.body, "")
    return found


def _membership(source: str) -> str | None:
    """``"derived"``, ``"hand"`` or ``None`` for a module with text *source*."""
    names_canopy = any(name in source for name in CANOPY_PACKAGES)
    if not names_canopy and f"mark.{MARKER}" not in source:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if names_canopy and any(name in CANOPY_PACKAGES for name in _imported_top_levels(tree)):
        return "derived"
    return "hand" if hand_marks(tree) else None


@functools.cache
def contract_members(tests_root: Path) -> dict[str, str]:
    """Every test module the ``canopy`` marker reaches, and how.

    Maps each module's path relative to *tests_root* (posix) to ``"derived"``
    when it imports CANOPY at import time, else to ``"hand"`` when it decorates
    a test with the mark; a module with neither is absent. This is the one
    table of the contract's members: the root conftest hooks, the contract
    scope computation and the marker guards all read it. *tests_root* is the
    tests directory this module lives in.
    """
    if Path(tests_root).resolve() != TESTS_ROOT:
        raise ValueError(f"{tests_root} is not the tests directory {TESTS_ROOT}")
    members: dict[str, str] = {}
    for relative, source in python_sources(Path(__file__)):
        kind = _membership(source)
        if kind is not None:
            members[Path(relative).as_posix()] = kind
    return members


def canopy_installed() -> bool:
    """Whether every CANOPY package is importable in this environment.

    A bare directory named like one of them is a namespace package with no
    origin, and does not count.
    """
    for name in CANOPY_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return False
        if spec is None or spec.origin is None:
            return False
    return True


class CanopyAbsentItem(pytest.Item):
    """Stands in for a CANOPY module's tests: selected without CANOPY, it fails."""

    def runtest(self) -> None:
        pytest.fail(ABSENT_TEXT, pytrace=False)

    def reportinfo(self) -> tuple[Path, None, str]:
        return self.path, None, self.name


class CanopyAbsent(pytest.File):
    """Collects a CANOPY module without importing it, as one failing item."""

    def collect(self) -> Iterator[CanopyAbsentItem]:
        item = CanopyAbsentItem.from_parent(self, name=ABSENT_ITEM_NAME)
        item.add_marker(MARKER)
        yield item


def module_node(module_path: Path, parent: pytest.Collector) -> pytest.Collector | None:
    """The collector for *module_path*, or ``None`` for pytest's own.

    A module that needs CANOPY is marked ``canopy`` as a whole where CANOPY is
    installed, and is replaced by its stand-in where it is not, so collection
    imports CANOPY only where it exists.
    """
    if not needs_canopy(module_path):
        return None
    if canopy_installed():
        module = pytest.Module.from_parent(parent, path=module_path)
        module.add_marker(MARKER)
        return module
    return CanopyAbsent.from_parent(parent, path=module_path)


def refuse_skip(item: pytest.Item, report: pytest.TestReport) -> None:
    """Turn a skip of a ``canopy``-marked *item* into a failure, in place."""
    if not report.skipped or hasattr(report, "wasxfail"):
        return
    if item.get_closest_marker(MARKER) is None:
        return
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
    report.outcome = "failed"
    report.longrepr = f"{SKIP_REFUSED_TEXT}: {reason}"


def deselects_canopy(markexpr: str | None) -> bool:
    """Whether ``-m`` *markexpr* leaves out an item whose only marker is ``canopy``."""
    if not markexpr or not markexpr.strip():
        return False
    from _pytest.mark.expression import Expression

    def carries(name: str, /, **kwargs: Any) -> bool:
        return name == MARKER

    return not Expression.compile(markexpr).evaluate(carries)


def stamp(item: pytest.Item, report: pytest.TestReport) -> None:
    """Record on *report* that *item* carries the ``canopy`` mark.

    The process that prints the summary may hold no collected item: in a
    parallel run the workers collect and the controller only receives their
    reports. A report keeps this attribute on its way between the two.
    """
    if item.get_closest_marker(MARKER) is not None:
        setattr(report, _MARKED_ATTRIBUTE, True)


def _marked(entries: Any) -> bool:
    """Whether any of *entries* is the report of a ``canopy``-marked test.

    The mark is read from the stamp, never from a report's keywords: every
    item below a directory node named ``canopy`` carries that word as a
    keyword, marked or not.
    """
    return any(getattr(entry, _MARKED_ATTRIBUTE, False) is True for entry in entries)


def _ran_canopy(stats: Mapping[str, Any]) -> bool:
    """Whether any report of the run belongs to a ``canopy``-marked test."""
    return any(
        _marked(entries)
        for category, entries in stats.items()
        if category not in _NOT_RUN_CATEGORIES
    )


def _canopy_failed(stats: Mapping[str, Any]) -> bool:
    """Whether a ``canopy``-marked test of the run failed or errored."""
    return any(_marked(stats.get(category, ())) for category in _FAILED_CATEGORIES)


def summary_lines(config: pytest.Config, stats: Mapping[str, Any]) -> list[str]:
    """What the end of a run says about the CANOPY contract."""
    lines = []
    if deselects_canopy(config.option.markexpr):
        lines.append(NOT_EXERCISED_TEXT)
    if _ran_canopy(stats):
        lines.append(proven.line(proven.judge(), ran=not _canopy_failed(stats)))
    return lines
