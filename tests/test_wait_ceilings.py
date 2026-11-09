"""Every ``wait_until_ready(..., ceiling=)`` site under tests/, declared once.

A ceiling lets a readiness wait run past its caller's own timeout, so each one
has to sit under something that still ends the test: the timeout mark of every
test that reaches it, or the full check's per-test cap. ``CEILING_SITES`` is
the one table of those sites. A row names the marks that cover the site, or
the reason none has to; the time-budget tests read this table and check the
ceilings' values against those bounds.

A site is keyed by its file and the ceiling expression's source text, never by
line, so an edit above it does not turn a row stale. These tests read the
source and start nothing: an undeclared site and a row that names no site both
fail, naming the file and line or the row.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests._tree_scan import python_sources

#: A site's key: its repository-relative file and the ceiling's source text.
SiteKey = tuple[str, str]


@dataclass(frozen=True)
class Covered:
    """A ceiling that sits under the timeout marks of the tests that reach it."""

    marks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.marks:
            raise ValueError("a covered ceiling names at least one mark")


@dataclass(frozen=True)
class Exempt:
    """A ceiling no mark has to cover, because it fits the per-test cap."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("an exempt ceiling states its reason")


CEILING_SITES: dict[SiteKey, Covered | Exempt] = {
    ("tests/integration/_qmd_ariel_support.py", "SIDECAR_HEALTH_CEILING"): Covered(
        ("ARIEL module", "OKF module", "deploy qmd test")
    ),
    ("tests/integration/_qmd_ariel_support.py", "SIDECAR_REINDEX_CEILING"): Covered(
        ("ARIEL module", "OKF module", "deploy qmd test")
    ),
    ("tests/e2e/test_deploy_lifecycle.py", "SIDECAR_HEALTH_CEILING"): Covered(("deploy qmd test",)),
    ("tests/_graphdb_container.py", "STORE_RECOVERY_S"): Exempt(
        "a store's recovery fits one test's cap"
    ),
    ("tests/test_container_support.py", "0.05"): Exempt(
        "the ceiling a unit test drives the wait to"
    ),
}


def ceiling_sites(source: str, path: str) -> list[tuple[SiteKey, int]]:
    """Each ``wait_until_ready`` call in *source* that passes ``ceiling=``, with its line.

    Args:
        source: A Python module's text.
        path: The module's repository-relative path, the key's first half.

    Returns:
        ``((path, ceiling source text), line)`` per call, in walk order.
    """
    sites = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "wait_until_ready":
            continue
        for keyword in node.keywords:
            if keyword.arg == "ceiling":
                text = ast.get_source_segment(source, keyword.value)
                sites.append(((path, text), node.lineno))
    return sites


def undeclared(sites: list[tuple[SiteKey, int]], table: dict[SiteKey, object]) -> list[str]:
    """``file:line ceiling=<expr>`` for every site *table* does not declare."""
    return [
        f"{path}:{line} ceiling={text}" for (path, text), line in sites if (path, text) not in table
    ]


def stale(sites: list[tuple[SiteKey, int]], table: dict[SiteKey, object]) -> list[str]:
    """``file ceiling=<expr>`` for every row of *table* that names no site."""
    found = {key for key, _ in sites}
    return [f"{path} ceiling={text}" for path, text in sorted(set(table) - found)]


def tree_ceiling_sites() -> list[tuple[SiteKey, int]]:
    """Every ``wait_until_ready(..., ceiling=)`` site in tests/."""
    sites = []
    for relative, source in python_sources(exempt=Path(__file__)):
        if "wait_until_ready" in source:
            sites += ceiling_sites(source, f"tests/{Path(relative).as_posix()}")
    return sites


def test_every_ceiling_site_is_declared():
    assert undeclared(tree_ceiling_sites(), CEILING_SITES) == []


def test_every_declared_row_names_a_site():
    assert stale(tree_ceiling_sites(), CEILING_SITES) == []


@pytest.mark.parametrize(
    "call",
    ["wait_until_ready(p, 'x', ceiling=X)", "support.wait_until_ready(p, 'x', ceiling=X)"],
    ids=["name", "attribute"],
)
def test_an_undeclared_ceiling_is_named_by_its_line(call):
    source = f"from tests import support\n\n\n{call}\n"

    assert undeclared(ceiling_sites(source, "tests/sample.py"), CEILING_SITES) == [
        "tests/sample.py:4 ceiling=X"
    ]


def test_a_row_with_no_site_is_named():
    table = {**CEILING_SITES, ("tests/sample.py", "X"): Exempt("gone")}

    assert stale(tree_ceiling_sites(), table) == ["tests/sample.py ceiling=X"]
