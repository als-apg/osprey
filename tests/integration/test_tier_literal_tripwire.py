"""The tier is three-valued: nothing reads it by comparing to the literal ``"allowed"`` (D3).

The tier key ``control_system.writes`` reads ``never``, ``asked`` or
``allowed``, and both ``asked`` and ``allowed`` arm. A reader that asks
``tier == "allowed"`` treats an ``asked`` tier as no tier at all, so every such
reader was moved to the three-word rule (``tier_allows``, or the normalised
word compared to ``never``). This module pins that across the shipped code:
no Python file under ``src/`` or a package's source tree compares a tier word
to the literal ``"allowed"``, except :mod:`osprey_connectors.types`, where
:func:`~osprey_connectors.types.tier_writes` defines the words.

A *tier word* is an expression that names a tier or a ceiling (an identifier
spelling ``tier`` or ``ceiling``) or reads the tier key itself (``["writes"]``,
``.get("writes")``, or the same through ``WRITES_KEY``). Comparing a switch
*position* to ``"allowed"`` is not a tier read and is left alone, as is a
membership test that names both arming words.

Test trees under ``packages/`` are not shipped code and are not scanned: a test
asserting ``tier_writes(...) == "allowed"`` is stating the rule, not reading it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: The one file allowed to spell the tier words against a value: it defines them.
DEFINING_FILE = REPO / "packages" / "osprey-connectors" / "src" / "osprey_connectors" / "types.py"

#: The literal no tier reader may compare to, and the names that stand for it.
ALLOWED = "allowed"
ALLOWED_NAMES = frozenset({"TIER_ALLOWED", "_TIER_ALLOWED"})
ASKED = "asked"
ASKED_NAMES = frozenset({"TIER_ASKED", "_TIER_ASKED"})

#: How the tier key is spelled when a reader indexes a section with it.
WRITES_KEYS = frozenset({"WRITES_KEY", "_WRITES_LEAF"})

_EQUALITY = (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)
_MEMBERSHIP = (ast.In, ast.NotIn)


def _identifier(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_word(node: ast.AST, value: str, names: frozenset[str]) -> bool:
    if isinstance(node, ast.Constant):
        return node.value == value
    return _identifier(node) in names


def _is_writes_key(node: ast.AST) -> bool:
    return _is_word(node, "writes", WRITES_KEYS)


def _names_a_tier(node: ast.AST) -> bool:
    """Whether *node* reads a tier word: a tier/ceiling name, or the tier key itself."""
    for child in ast.walk(node):
        name = _identifier(child)
        if name is not None and ("tier" in name.lower() or "ceiling" in name.lower()):
            return True
        if isinstance(child, ast.Subscript) and _is_writes_key(child.slice):
            return True
        if (
            isinstance(child, ast.Call)
            and _identifier(child.func) == "get"
            and child.args
            and _is_writes_key(child.args[0])
        ):
            return True
    return False


def _single_word_container(node: ast.AST) -> bool:
    """A literal container naming ``allowed`` without ``asked``: ``allowed`` alone arms."""
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return False
    elements = node.elts
    return any(_is_word(e, ALLOWED, ALLOWED_NAMES) for e in elements) and not any(
        _is_word(e, ASKED, ASKED_NAMES) for e in elements
    )


def literal_tier_reads(source: str) -> list[tuple[int, str]]:
    """``(line, expression)`` for every comparison of a tier word to ``"allowed"``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, right in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, _EQUALITY):
                hit = (_is_word(right, ALLOWED, ALLOWED_NAMES) and _names_a_tier(left)) or (
                    _is_word(left, ALLOWED, ALLOWED_NAMES) and _names_a_tier(right)
                )
            elif isinstance(op, _MEMBERSHIP):
                hit = _single_word_container(right) and _names_a_tier(left)
            else:
                hit = False
            if hit:
                found.append((node.lineno, ast.unparse(node)))
            left = right
    return found


def _shipped_sources() -> list[Path]:
    roots = [REPO / "src", *sorted((REPO / "packages").glob("*/src"))]
    return sorted(
        path
        for root in roots
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
    )


def test_the_scan_reads_the_shipped_trees():
    sources = _shipped_sources()

    assert DEFINING_FILE in sources, "the scan does not reach the connectors package"
    assert any(path.is_relative_to(REPO / "src" / "osprey") for path in sources)


def test_no_shipped_file_reads_the_tier_against_the_literal_allowed():
    offenders = []
    for path in _shipped_sources():
        if path == DEFINING_FILE:
            continue
        for line, expression in literal_tier_reads(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(REPO)}:{line}: {expression}")

    assert not offenders, (
        "a tier word compared to the literal 'allowed' misses the 'asked' tier; "
        "use tier_allows() or compare the normalised word to 'never':\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "source",
    [
        'tier_writes(section) == "allowed"',
        'tier != "allowed"',
        '"allowed" == self.ceiling',
        'section.get("writes") == "allowed"',
        'section.get(WRITES_KEY) is "allowed"',
        'section["writes"] == "allowed"',
        'owner_tier_term(section, login) in ("allowed",)',
        "tier == TIER_ALLOWED",
        'login_tier not in {"allowed", "skip"}',
    ],
)
def test_the_tripwire_catches_a_literal_tier_read(source):
    assert literal_tier_reads(source), source


@pytest.mark.parametrize(
    "source",
    [
        'tier in ("asked", "allowed")',
        "value in (_TIER_ASKED, _TIER_ALLOWED)",
        'tier != "never"',
        "tier_allows(section)",
        'position == "allowed"',
        'verdict.position == "allowed"',
        'stored == "allowed" and tier == "asked"',
    ],
)
def test_the_tripwire_leaves_the_three_word_rule_and_positions_alone(source):
    assert not literal_tier_reads(source), source
