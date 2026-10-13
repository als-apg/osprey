"""Where a guarded run's files live, and what a journal says, is spelled once in the suite.

:mod:`tests._guarded_run` derives a target's guarded-run directory, journal and
lock from the product's names, and plants a well-formed journal through the
product's writer. A test that types the names itself, or writes a journal
header by hand, keeps a private copy of the layout or the format that the
product can move away from while the test stays green.

Read off the source: every module but the owner is held to three rules, and
the few that state the name or the format on purpose are listed with their
reason. A listed exception that no longer matches anything fails too, so the
list cannot outlive what it excuses.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from osprey.runtime.guarded_run import GUARDED_RUN_DIR, JOURNAL_FILE_NAME, LOCK_FILE_NAME
from tests._tree_scan import python_sources

#: The owner module, which spells the layout and the format for every test.
OWNER = "_guarded_run.py"

#: The on-disk names a test takes from the owner rather than typing.
NAMES = frozenset({GUARDED_RUN_DIR, JOURNAL_FILE_NAME, LOCK_FILE_NAME})

#: A journal header written as text.
HEADER_TEXT = re.compile(r"""["']header["']\s*:\s*\{\s*["']target["']""")

#: ``(module relative to tests/, enclosing top-level name or None for the
#: whole module)`` that states a name or a header on purpose, and why.
ALLOWED: dict[tuple[str, str | None], str] = {
    (
        "runtime/test_guarded_run_dir.py",
        "test_the_exported_names",
    ): "the one statement of the on-disk name as product",
    ("runtime/test_journal.py", None): "the writer's and parser's own format cases",
    (
        "hooks/test_guarded_run_constants_drift.py",
        None,
    ): "the two parsers' shared table of malformed journals",
    (
        "runtime/test_run_lock.py",
        "test_a_header_only_journal_is_cleared",
    ): "malformed content, through plant_raw",
    (
        "hooks/test_approval_hook.py",
        "test_no_pending_record_binds_none",
    ): "malformed content, through plant_raw_at",
    (
        "hooks/test_approval_hook.py",
        "test_an_unreadable_journal_is_denied",
    ): "malformed content, through plant_raw_at",
}


def _key(node: ast.expr | None) -> object:
    return node.value if isinstance(node, ast.Constant) else None


def _rule(node: ast.AST) -> str | None:
    """The rule *node* breaks, or None."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str) and node.value in NAMES:
            return f"the name {node.value!r}"
        if isinstance(node.value, (str, bytes)):
            text = node.value.decode("latin-1") if isinstance(node.value, bytes) else node.value
            if HEADER_TEXT.search(text):
                return "a journal header as text"
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                _key(key) == "header"
                and isinstance(value, ast.Dict)
                and "target" in {_key(inner) for inner in value.keys}
            ):
                return "a journal header as a dict"
    return None


def spellings(text: str) -> list[tuple[int, str | None, str]]:
    """Every hand spelling in the module source *text*.

    Returns:
        ``(line, enclosing top-level name or None, rule)`` per hit, in source order.
    """
    found: list[tuple[int, str | None, str]] = []
    for statement in ast.parse(text).body:
        name = getattr(statement, "name", None)
        for node in ast.walk(statement):
            rule = _rule(node)
            if rule is not None:
                found.append((node.lineno, name, rule))
    return sorted(found, key=lambda hit: hit[0])


def test_a_path_join_literal_is_reported() -> None:
    text = 'def test_x(tmp_path):\n    path = tmp_path / "var" / "guarded_run" / "va" / "run.journal"\n'

    assert [(line, name) for line, name, _ in spellings(text)] == [(2, "test_x"), (2, "test_x")]


def test_a_hand_written_header_is_reported() -> None:
    text = (
        "LINES = [\n"
        '    {"header": {"target": "va", "pid": 1}},\n'
        "]\n"
        'RAW = b\'{"header": {"target": "va"}}\\n\'\n'
    )

    assert [(line, name) for line, name, _ in spellings(text)] == [(2, None), (4, None)]


def test_the_suite_takes_the_layout_and_the_format_from_the_owner() -> None:
    unlisted: list[str] = []
    matched: set[tuple[str, str | None]] = set()
    for path, text in python_sources(Path(__file__)):
        if path == OWNER:
            continue
        for line, name, rule in spellings(text):
            for key in ((path, name), (path, None)):
                if key in ALLOWED:
                    matched.add(key)
                    break
            else:
                unlisted.append(f"{path}:{line}: {rule}; take it from tests/{OWNER}")

    assert unlisted == []
    assert sorted(set(ALLOWED) - matched, key=str) == [], "allowed spellings that match nothing"
