"""No facility-tree path is spelt by hand outside the facility tree's declaration.

``osprey_facility.source_tree`` is the one place the facility tree's paths are
spelt; every other module in ``src/`` and ``packages/*/src`` reads them from
it. This guard reads the shipped Python for the spellings of the slots a later
layout moves and fails on one it finds outside the declaration, unless the
site is listed in :data:`ALLOWED` with its count and the reason it may stay.
:data:`ALLOWED` is therefore the inventory of the Python sites that change by
hand when those paths move. What counts as a spelling is
:mod:`tests._layout_spelling`'s rule.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests._layout_spelling import (
    CATEGORIES,
    EQUAL_TO_SOURCE,
    GENERATED_SCHEMA,
    PUBLIC_TEST_HELPER,
    Site,
    compare,
    scan,
    scan_source,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The declaration, the one file that spells the facility tree's paths.
DECLARATION = "packages/osprey-facility/src/osprey_facility/source_tree.py"

#: The spellings of the facility tree's slots that are not ordinary words: the
#: files, the keyed and deck directories in their slash form, and the benchmark
#: queries' directory. The grouping directories (``vocabulary``, ``provenance``,
#: ``simulation``) are ordinary words and are not guarded on their own; a path
#: under one is still caught by the name of the file or directory it ends in.
GUARDED = (
    "classes.yaml",
    "signals.yaml",
    "documents.yaml",
    "policy.yaml",
    "fixes.yaml",
    "models.yaml",
    "seeds.yaml",
    "scenarios/",
    "measurement/",
    "decks/",
    "benchmarks/channel_finder",
)

_GENERATED = "packages/osprey-facility/src/osprey_facility/schema/_generated/core.py"
_SIMULATION_VIEW = "packages/osprey-connectors/src/osprey_connectors/simulation/view.py"

#: ``(file, constant text) -> (count, category)`` for each site still allowed
#: to spell a guarded path.
ALLOWED: Mapping[tuple[str, str], tuple[int, str]] = {
    # The simulator view's output names, under the render's data/simulator/.
    (_SIMULATION_VIEW, "decks"): (1, EQUAL_TO_SOURCE),
    (_SIMULATION_VIEW, "scenarios"): (1, EQUAL_TO_SOURCE),
    # The build's stage name and the import's record-kind label.
    ("packages/osprey-facility/src/osprey_facility/build.py", "measurement"): (1, EQUAL_TO_SOURCE),
    ("src/osprey/cli/facility_cmd.py", "measurement"): (1, EQUAL_TO_SOURCE),
    # Field descriptions generated from the facility schema.
    (_GENERATED, "The fixes applied to the record, in fixes.yaml order."): (1, GENERATED_SCHEMA),
    (
        _GENERATED,
        "`[{entry_id, when, author, title, text, tags?, categories?, loto_tag?, extra?, "
        "attachments?}]`, the entries the scenario narrates. `attachments` lists the entry's "
        "pictures, each `{path: <picture file>}` or `{plot: <plot spec .json>}`, relative to "
        "the scenario's own directory `simulation/scenarios/<name>/`.",
    ): (1, GENERATED_SCHEMA),
}


@functools.cache
def _sites() -> tuple[Site, ...]:
    """Every guarded spelling in the shipped Python, read once per process."""
    return tuple(scan(REPO_ROOT, GUARDED, exempt={DECLARATION}))


def test_no_facility_path_is_spelt_by_hand() -> None:
    problems = compare(_sites(), ALLOWED)
    assert problems == [], (
        f"read these paths from {DECLARATION} instead of spelling them; a site that "
        "may keep its spelling goes in ALLOWED with its count and category:\n" + "\n".join(problems)
    )


def test_every_allowed_site_has_a_category() -> None:
    assert {category for _count, category in ALLOWED.values()} <= CATEGORIES


def test_every_guarded_spelling_is_a_declared_path() -> None:
    """Each spelling is a path segment, or a run of them, of a slot the declaration holds."""
    from osprey_facility import source_tree

    paths = [f"/{source_tree.path(row.name)}/" for row in source_tree.slots()]
    for spelling in GUARDED:
        assert any(f"/{spelling.rstrip('/')}/" in path for path in paths), spelling


# --- the scanner finds every form a spelling takes ---------------------------------------

_PLANTED = '''
"""A docstring naming fixes.yaml is prose, not code."""
LITERAL = "classes.yaml"
JOINED = Path("data") / "facility" / "signals.yaml"
DIRECTORY = root / "scenarios"
SENTENCE = "Correct the record in fixes.yaml and build again."
FORMATTED = f"measurement/{model}.yaml"
X = "measurement"
'''

#: The six forms, each with the line it is planted on.
_FORMS = {
    "a literal": (3, "classes.yaml"),
    "a joined path": (4, "signals.yaml"),
    'root / "scenarios"': (5, "scenarios/"),
    "a prose sentence naming fixes.yaml": (6, "fixes.yaml"),
    "an f-string measurement/{x}.yaml": (7, "measurement/"),
    'X = "measurement"': (8, "measurement/"),
}


@pytest.mark.parametrize("form", list(_FORMS))
def test_the_scanner_finds_a_planted_spelling(form: str) -> None:
    found = {(site.line, site.spelling) for site in scan_source(_PLANTED, "planted.py", GUARDED)}
    assert _FORMS[form] in found


def test_the_scanner_finds_the_six_forms_and_nothing_else() -> None:
    found = [(site.line, site.spelling) for site in scan_source(_PLANTED, "planted.py", GUARDED)]
    assert found == sorted(_FORMS.values())


_IDENTITIES = """
document["scenarios"]
{"measurement": block}
kind == "decks"
source_tree.path("scenarios")
name in ("measurement", "decks")
SCHEMA = "osprey.facility.scenarios/3"
PATTERN = "my_classes.yaml"
"""


def test_an_identity_is_no_spelling() -> None:
    assert scan_source(_IDENTITIES, "identities.py", GUARDED) == []


def test_a_directory_named_after_a_slash_is_a_spelling() -> None:
    (site,) = scan_source('WHERE = "data/facility/decks"\n', "where.py", GUARDED)
    assert site.spelling == "decks/"


_GROUPED = """
QUERIES = Path("data") / "facility" / "benchmarks" / "channel_finder"
MODELS = "simulation/models.yaml"
DECK = f"{root}/simulation/decks/{name}.json"
WORD = "simulation"
"""


def test_a_path_under_a_grouping_directory_is_caught_by_its_last_name() -> None:
    found = [(site.line, site.spelling) for site in scan_source(_GROUPED, "grouped.py", GUARDED)]
    assert found == [(2, "benchmarks/channel_finder"), (3, "models.yaml"), (4, "decks/")]


def test_a_stale_or_uncounted_allowance_is_reported() -> None:
    sites = scan_source('A = "fixes.yaml"\nB = "fixes.yaml"\n', "f.py", GUARDED)
    assert compare(sites, {("f.py", "fixes.yaml"): (2, PUBLIC_TEST_HELPER)}) == []
    assert compare(sites, {("f.py", "fixes.yaml"): (1, PUBLIC_TEST_HELPER)})
    assert compare(sites, {("f.py", "fixes.yaml"): (2, "convenient")})
    assert compare([], {("f.py", "fixes.yaml"): (2, PUBLIC_TEST_HELPER)})
    assert compare(sites, {})
