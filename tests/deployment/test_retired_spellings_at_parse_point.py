"""The retired roster spellings live at the parse point, and nowhere after it.

A profile written in the old roster words is rewritten once, when it is
parsed (``osprey.cli.build_profile_document``). Every reader after that point
knows only ``tier``, ``default_tier`` and ``renders``; a reader that still
falls back to a retired key would keep a second spelling alive behind the
rewrite. This guard walks the source and the tests as syntax trees and
refuses any use of the three retired-key constants outside their homes: the
parse point, the migration verb, the deploy refusal that names the new home,
the hash label, and the tests that are about the old spelling itself.
"""

from __future__ import annotations

import ast
import doctest
from pathlib import Path

import pytest

import tests._renders

REPO = Path(__file__).resolve().parents[2]

RETIRED_NAMES = frozenset({"RETIRED_TIER_KEY", "RETIRED_DEFAULT_TIER_KEY", "RETIRED_CATALOG_KEY"})

#: The ``osprey.cli`` modules that may read a retired spelling: the parse point,
#: the migration verb, the deploy refusal and the hash label.
CLI_HOMES = frozenset(
    {"build_profile_document", "init_revalidate", "build_profile_deploy", "build_profile_merge"}
)

#: The tests whose subject is the old spelling.
TEST_HOMES = frozenset(
    {
        "tests/cli/test_retired_roster_keys.py",
        "tests/cli/test_init_revalidate.py",
        "tests/cli/test_init_revalidate_fill.py",
        "tests/cli/test_profile_hash_folding.py",
    }
)
TEST_HOME_DIRS = ("tests/fixtures/previous_release_repo/",)


def retired_uses(source: str) -> list[tuple[int, str]]:
    """``(line, name)`` for every import or attribute read of a retired constant."""
    uses: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            uses.extend((node.lineno, a.name) for a in node.names if a.name in RETIRED_NAMES)
        elif isinstance(node, ast.Attribute) and node.attr in RETIRED_NAMES:
            uses.append((node.lineno, node.attr))
    return uses


def _offenders(files: list[Path]) -> list[str]:
    found: list[str] = []
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        for line, name in retired_uses(path.read_text(encoding="utf-8")):
            found.append(f"{rel}:{line} {name}")
    return found


def _source_files() -> list[Path]:
    deployment = sorted((REPO / "src/osprey/deployment").rglob("*.py"))
    cli = [
        path
        for path in sorted((REPO / "src/osprey/cli").rglob("*.py"))
        if path.stem not in CLI_HOMES
    ]
    return deployment + cli


def _test_files() -> list[Path]:
    files = []
    for path in sorted((REPO / "tests").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in TEST_HOMES or rel.startswith(TEST_HOME_DIRS):
            continue
        files.append(path)
    return files


def test_no_reader_after_the_parse_point_reads_a_retired_spelling() -> None:
    offenders = _offenders(_source_files())
    assert offenders == [], (
        "a reader after the parse point uses a retired roster spelling; read "
        "`tier`, `default_tier` and `renders` alone:\n" + "\n".join(offenders)
    )


def test_no_test_outside_the_old_spelling_homes_uses_a_retired_spelling() -> None:
    offenders = _offenders(_test_files())
    assert offenders == [], (
        "a test outside the retired-spelling homes uses a retired roster spelling; "
        "spell the catalog through tests._renders, or move a test about the old "
        "spelling to tests/cli/test_retired_roster_keys.py:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "source",
    [
        "from osprey.cli.build_profile_document import RETIRED_CATALOG_KEY\n",
        "from osprey.cli.build_profile_document import (\n    x,\n    RETIRED_TIER_KEY,\n)\n",
        "from osprey.cli import build_profile_document as d\nd.RETIRED_DEFAULT_TIER_KEY\n",
    ],
)
def test_the_scan_sees_every_spelling_of_a_retired_use(source: str) -> None:
    assert retired_uses(source)


def test_the_scan_ignores_other_retired_words() -> None:
    """The retired top-level ``tier`` profile key and the retired control-system
    types are other words, never this guard's."""
    source = (
        "from osprey.cli.build_profile_load import RETIRED_CONTROL_SYSTEM_TYPES\n"
        "_RETIRED_TIER_KEY = 'tier'\n"
    )
    assert retired_uses(source) == []


def test_the_render_builders_spell_the_projection_entry() -> None:
    result = doctest.testmod(tests._renders, verbose=False)
    assert result.attempted and not result.failed
