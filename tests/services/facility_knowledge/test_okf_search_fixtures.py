"""Property tests for the curated OKF query fixture set.

The fixtures in :mod:`tests.fixtures.qmd_okf` exist to make one acceptance
criterion mechanical: *ranked search must place the expected concept in its top
3*.  That criterion is only meaningful if the queries are genuinely hard — if
they can be answered by a plain substring scan, passing them proves nothing.

So the fixture set is built to a construction rule: **no fixture query returns
its expected concept at rank 1 under substring search**.  This module asserts
that rule against the live bundle, so a fixture that quietly becomes easy (an
edit to a bundle document, a reworded query) fails here rather than silently
weakening the acceptance gate downstream.

Each fixture also declares *how* it defeats the scan — nothing matches at all,
or a decoy concept is returned first — and that declaration is checked too.  A
fixture that degrades from a hard jargon mismatch into a trivially unmatched
sentence is a real loss of coverage, and this module refuses to let it pass
unnoticed.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.qmd_okf import (
    OKF_QUERY_FIXTURES,
    DefeatMode,
    OKFQueryFixture,
    load_fixture_bundle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranked_concept_ids(results: list[Any]) -> list[str]:
    """Return concept IDs from a search result list, in rank order.

    Tolerates both search result shapes so this module does not pin the return
    type of ``OKFBundle.search()``: the ``(concept_id, document)`` tuples of the
    substring implementation, and any ranked result object exposing a
    ``concept_id`` attribute.

    Args:
        results: Whatever ``OKFBundle.search()`` returned.

    Returns:
        Concept IDs in the order the search returned them.
    """
    ids: list[str] = []
    for item in results:
        if isinstance(item, tuple):
            ids.append(item[0])
        else:
            ids.append(item.concept_id)
    return ids


def _fixture_id(fixture: OKFQueryFixture) -> str:
    """Return a readable pytest parameter ID for *fixture*."""
    return f"{fixture.defeat_mode.value}-{fixture.expected_concept_id}"


_FIXTURE_PARAMS = pytest.mark.parametrize(
    "fixture",
    OKF_QUERY_FIXTURES,
    ids=[_fixture_id(f) for f in OKF_QUERY_FIXTURES],
)


# ---------------------------------------------------------------------------
# Bundle sanity
# ---------------------------------------------------------------------------


def test_bundle_concepts_all_parse() -> None:
    """Every concept in the checked-in bundle parses as an OKF document."""
    bundle = load_fixture_bundle()
    entries = bundle.list_concepts()
    assert entries, "fixture bundle is empty"

    for entry in entries:
        doc = bundle.read_concept(entry.concept_id)
        assert doc.frontmatter.get("type"), f"{entry.concept_id} has no type"
        assert doc.body.strip(), f"{entry.concept_id} has an empty body"


def test_bundle_index_covers_every_concept_file() -> None:
    """The bundle's ``index.md`` lists every non-reserved document on disk."""
    bundle = load_fixture_bundle()
    indexed = {e.concept_id for e in bundle.list_concepts()}
    on_disk = {
        md.relative_to(bundle.root).with_suffix("").as_posix()
        for md in bundle.root.rglob("*.md")
        if md.name not in {"index.md", "log.md"}
    }
    assert indexed == on_disk


# ---------------------------------------------------------------------------
# Fixture set shape
# ---------------------------------------------------------------------------


def test_fixture_queries_are_unique() -> None:
    """No query appears twice — duplicates would double-count the gate."""
    queries = [f.query for f in OKF_QUERY_FIXTURES]
    assert len(set(queries)) == len(queries)


def test_fixture_set_spans_many_distinct_concepts() -> None:
    """The set exercises a broad slice of the bundle, not one corner of it."""
    expected = {f.expected_concept_id for f in OKF_QUERY_FIXTURES}
    assert len(expected) >= 12


def test_both_defeat_modes_are_well_represented() -> None:
    """Neither defeat mode may collapse to a token presence.

    A set made only of unmatched paraphrases would be cheap to defeat; a set
    made only of decoy collisions would not exercise natural-language recall.
    """
    modes = [f.defeat_mode for f in OKF_QUERY_FIXTURES]
    for mode in DefeatMode:
        assert modes.count(mode) >= 4, f"only {modes.count(mode)} fixtures use {mode.value}"


@_FIXTURE_PARAMS
def test_fixture_expected_concept_exists(fixture: OKFQueryFixture) -> None:
    """Each fixture points at a concept that is actually in the bundle."""
    bundle = load_fixture_bundle()
    assert bundle.resolve_concept_path(fixture.expected_concept_id).exists()


@_FIXTURE_PARAMS
def test_fixture_decoy_field_agrees_with_defeat_mode(fixture: OKFQueryFixture) -> None:
    """A decoy concept is declared exactly when the defeat mode implies one."""
    bundle = load_fixture_bundle()
    if fixture.defeat_mode is DefeatMode.WRONG_CONCEPT_FIRST:
        assert fixture.decoy_concept_id, "decoy-first fixture must name its decoy"
        assert bundle.resolve_concept_path(fixture.decoy_concept_id).exists()
        assert fixture.decoy_concept_id != fixture.expected_concept_id
    else:
        assert fixture.decoy_concept_id == ""


@_FIXTURE_PARAMS
def test_fixture_has_a_rationale(fixture: OKFQueryFixture) -> None:
    """Each fixture records why its expected concept is the right answer."""
    assert len(fixture.rationale.split()) >= 10


# ---------------------------------------------------------------------------
# The construction rule
# ---------------------------------------------------------------------------


@_FIXTURE_PARAMS
def test_fixture_defeats_substring_search_at_rank_one(fixture: OKFQueryFixture) -> None:
    """Substring search must not answer this fixture correctly at rank 1.

    This is the property that makes the downstream "qmd top 3" acceptance
    criterion meaningful.  If it fails, the fixture no longer discriminates
    between ranked retrieval and a plain scan, and must be re-authored rather
    than relaxed.
    """
    bundle = load_fixture_bundle()
    ranked = _ranked_concept_ids(bundle.search(fixture.query))

    assert ranked[:1] != [fixture.expected_concept_id], (
        f"substring search already returns {fixture.expected_concept_id!r} at rank 1 "
        f"for {fixture.query!r}; this fixture proves nothing about ranking"
    )


@_FIXTURE_PARAMS
def test_fixture_defeats_substring_search_the_declared_way(fixture: OKFQueryFixture) -> None:
    """The *mechanism* by which substring search fails matches the declaration.

    Checking the mechanism, not just the outcome, catches the silent failure
    mode where a jargon-mismatch fixture decays into a query that simply matches
    nothing — still "not rank 1", but a much weaker test.
    """
    bundle = load_fixture_bundle()
    ranked = _ranked_concept_ids(bundle.search(fixture.query))

    if fixture.defeat_mode is DefeatMode.NO_SUBSTRING_MATCH:
        assert ranked == [], (
            f"{fixture.query!r} is declared to match nothing, but substring search "
            f"returned {ranked}"
        )
    else:
        assert ranked, f"{fixture.query!r} is declared to hit a decoy, but matched nothing"
        assert ranked[0] == fixture.decoy_concept_id, (
            f"{fixture.query!r} is declared to return {fixture.decoy_concept_id!r} at "
            f"rank 1, but substring search returned {ranked}"
        )
