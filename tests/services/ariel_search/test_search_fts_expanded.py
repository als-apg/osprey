"""Tests for the vocabulary-expanded tsquery builder.

`build_expanded_tsquery` is the sibling of `build_tsquery`: it decomposes a
parsed keyword query into non-overlapping spans and emits one ``&&``-joined leg
per span, so expansion widens what satisfies a span without ever dropping a
term. These tests pin the leg count, the placeholder count and the exact
parameter order for the shapes the proposal names.
"""

from pathlib import Path

import pytest

from osprey.services.ariel_search.database.search_fts import build_expanded_tsquery
from osprey.services.ariel_search.search.base import (
    ExpansionGroup,
    ParsedKeywordQuery,
    QueryExpansion,
)
from osprey.services.ariel_search.vocabulary import expand_query, load_vocabulary

CONTROL_ASSISTANT_VOCABULARY = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
      - ts
  - canonical: beam position monitor
    kind: acronym
    forms:
      - BPM
"""


def parsed(search_text: str, *phrases: str) -> ParsedKeywordQuery:
    """Build a `ParsedKeywordQuery` with only the fields this builder reads.

    Args:
        search_text: The query text left after filters, phrases and patterns.
        *phrases: Quoted phrases in query order, without their quotes.

    Returns:
        The parsed query.
    """
    return ParsedKeywordQuery(search_text=search_text, phrases=tuple(phrases))


def expansion(*groups: tuple[str, tuple[str, ...]]) -> QueryExpansion:
    """Build a `QueryExpansion` from ``(original, alternatives)`` pairs.

    Args:
        *groups: One pair per matched span, in query order. `original` must be
            spelled as it appears in normalized text, as the real expander
            spells it.

    Returns:
        The expansion. `flattened_text` is unused by the builder and left empty.
    """
    return QueryExpansion(
        groups=tuple(
            ExpansionGroup(original=original, alternatives=alts) for original, alts in groups
        ),
        flattened_text="",
    )


def legs(sql: str) -> list[str]:
    """Split an emitted tsquery into its ``&&``-joined legs."""
    return sql.split(" && ")


class TestSpanDecomposition:
    """Every span of the query becomes a leg -- no term is dropped."""

    def test_three_legs_five_placeholders_for_two_concepts_and_a_trailing_run(self):
        """`ts bpm undulator` pins the criterion's leg count, placeholders and params."""
        sql, params = build_expanded_tsquery(
            parsed("ts bpm undulator"),
            expansion(("ts", ("troubleshoot",)), ("bpm", ("beam position monitor",))),
        )

        assert len(legs(sql)) == 3
        assert sql.count("%s") == 5
        assert params == ["ts", "troubleshoot", "bpm", "beam position monitor", "undulator"]

    def test_concept_legs_are_parenthesized_alternations(self):
        """A concept span alternates its original with its alternatives."""
        sql, _params = build_expanded_tsquery(
            parsed("ts bpm undulator"),
            expansion(("ts", ("troubleshoot",)), ("bpm", ("beam position monitor",))),
        )

        assert legs(sql)[0] == (
            "(plainto_tsquery('english', %s) || plainto_tsquery('english', %s))"
        )
        assert legs(sql)[2] == "plainto_tsquery('english', %s)"

    def test_ambiguous_form_is_one_leg_over_every_binding(self):
        """`ts fault` with `ts` bound to two concepts: 2 legs, 4 placeholders."""
        sql, params = build_expanded_tsquery(
            parsed("ts fault"),
            expansion(("ts", ("troubleshoot", "timing system"))),
        )

        assert len(legs(sql)) == 2
        assert sql.count("%s") == 4
        assert params == ["ts", "troubleshoot", "timing system", "fault"]

    def test_leading_and_middle_unmatched_runs_are_one_leg_each(self):
        """Each maximal run of unmatched tokens collapses into a single leg."""
        sql, params = build_expanded_tsquery(
            parsed("fix ts now bpm later"),
            expansion(("ts", ("troubleshoot",)), ("bpm", ("beam position monitor",))),
        )

        assert len(legs(sql)) == 5
        assert params == [
            "fix",
            "ts",
            "troubleshoot",
            "now",
            "bpm",
            "beam position monitor",
            "later",
        ]

    def test_unmatched_run_text_is_normalized(self):
        """Runs are bound as normalized text (lowercased, `-` to space, collapsed)."""
        _sql, params = build_expanded_tsquery(parsed("Beam-Loss   Monitor"), expansion())

        assert params == ["beam loss monitor"]

    def test_multi_word_concept_span_consumes_all_its_tokens(self):
        """A three-token span is located as a unit, not token by token."""
        sql, params = build_expanded_tsquery(
            parsed("check beam position monitor offset"),
            expansion(("beam position monitor", ("bpm",))),
        )

        assert len(legs(sql)) == 3
        assert params == ["check", "beam position monitor", "bpm", "offset"]


class TestDirectionGates:
    """Canonical-to-acronym on and off produce the two-leg shapes the criteria name."""

    def test_canonical_expands_to_one_two_placeholder_leg(self):
        """`beam position monitor` with `bpm` as an alternative is one alternation."""
        sql, params = build_expanded_tsquery(
            parsed("beam position monitor"),
            expansion(("beam position monitor", ("bpm",))),
        )

        assert len(legs(sql)) == 1
        assert sql.count("%s") == 2
        assert params == ["beam position monitor", "bpm"]

    def test_no_groups_is_one_single_placeholder_leg(self):
        """With the gate off nothing expands: one plain leg over the whole text."""
        sql, params = build_expanded_tsquery(parsed("beam position monitor"), expansion())

        assert len(legs(sql)) == 1
        assert sql.count("%s") == 1
        assert sql == "plainto_tsquery('english', %s)"
        assert params == ["beam position monitor"]


class TestPhrases:
    """Quoted phrases stay phrase predicates, expanded only as a whole."""

    def test_plain_phrase_is_a_phraseto_leg(self):
        """An unexpanded phrase emits exactly today's phrase predicate."""
        sql, params = build_expanded_tsquery(parsed("", "beam loss"), expansion())

        assert sql == "phraseto_tsquery('english', %s)"
        assert params == ["beam loss"]

    def test_phrase_that_resolved_to_a_concept_alternates_phraseto_with_plainto(self):
        """A whole-phrase concept keeps `phraseto` for the phrase, `plainto` for alternatives."""
        sql, params = build_expanded_tsquery(
            parsed("", "beam position monitor"),
            expansion(("beam position monitor", ("bpm",))),
        )

        assert sql == ("(phraseto_tsquery('english', %s) || plainto_tsquery('english', %s))")
        assert params == ["beam position monitor", "bpm"]

    def test_phrase_group_is_not_double_emitted_with_search_text(self):
        """A group living only in a phrase produces one leg, not one per surface."""
        sql, params = build_expanded_tsquery(
            parsed("ts", "beam position monitor"),
            expansion(("ts", ("troubleshoot",)), ("beam position monitor", ("bpm",))),
        )

        assert len(legs(sql)) == 2
        assert params == ["ts", "troubleshoot", "beam position monitor", "bpm"]

    def test_phrase_is_bound_as_typed_not_normalized(self):
        """Phrase parameters keep the user's spelling; only runs are normalized."""
        _sql, params = build_expanded_tsquery(parsed("", "Beam Loss Monitor"), expansion())

        assert params == ["Beam Loss Monitor"]

    def test_group_inside_a_longer_phrase_leaves_the_phrase_unexpanded(self):
        """Partial phrase matches are a documented simplification: phrase stays exact."""
        sql, params = build_expanded_tsquery(
            parsed("", "bpm offset drift"),
            expansion(("bpm", ("beam position monitor",))),
        )

        assert sql == "phraseto_tsquery('english', %s)"
        assert params == ["bpm offset drift"]


class TestEmptyAndDegenerate:
    """Shapes that must mirror `build_tsquery`."""

    def test_empty_query_mirrors_build_tsquery(self):
        """No search text and no phrases yields the empty tsquery with no params."""
        sql, params = build_expanded_tsquery(parsed(""), expansion())

        assert sql == "plainto_tsquery('english', '')"
        assert params == []

    def test_whitespace_only_search_text_is_empty(self):
        """Whitespace normalizes away, leaving no legs."""
        sql, params = build_expanded_tsquery(parsed("   "), expansion())

        assert sql == "plainto_tsquery('english', '')"
        assert params == []

    def test_group_absent_from_both_surfaces_is_ignored(self):
        """A group matching neither search text nor a phrase never invents a leg."""
        sql, params = build_expanded_tsquery(
            parsed("undulator"),
            expansion(("ts", ("troubleshoot",))),
        )

        assert sql == "plainto_tsquery('english', %s)"
        assert params == ["undulator"]


class TestRealVocabulary:
    """The builder consumes what the real expander produces, unmodified."""

    def test_expand_query_output_drives_the_builder(self, tmp_path: Path):
        """A loaded vocabulary + `expand_query` reproduce the pinned `ts bpm undulator` shape."""
        path = tmp_path / "vocabulary.yml"
        path.write_text(CONTROL_ASSISTANT_VOCABULARY, encoding="utf-8")
        vocabulary, errors, _warnings = load_vocabulary(
            path, canonical_to_acronym=True, canonical_to_shorthand=False
        )
        assert errors == []
        assert vocabulary is not None

        resolved = expand_query(
            "TS BPM undulator",
            vocabulary,
            canonical_to_acronym=True,
            canonical_to_shorthand=False,
        )
        sql, params = build_expanded_tsquery(parsed("TS BPM undulator"), resolved)

        assert len(legs(sql)) == 3
        assert sql.count("%s") == 5
        assert params == ["ts", "troubleshoot", "bpm", "beam position monitor", "undulator"]

    def test_slash_form_stays_one_token(self, tmp_path: Path):
        """`t/s` survives normalization as a single token and locates in `search_text`."""
        path = tmp_path / "vocabulary.yml"
        path.write_text(CONTROL_ASSISTANT_VOCABULARY, encoding="utf-8")
        vocabulary, errors, _warnings = load_vocabulary(
            path, canonical_to_acronym=True, canonical_to_shorthand=False
        )
        assert errors == []
        assert vocabulary is not None

        resolved = expand_query(
            "t/s bpm",
            vocabulary,
            canonical_to_acronym=True,
            canonical_to_shorthand=False,
        )
        _sql, params = build_expanded_tsquery(parsed("t/s bpm"), resolved)

        assert params == ["t/s", "troubleshoot", "bpm", "beam position monitor"]


PLACEHOLDER_CASES = [
    ("ts bpm undulator", (), (("ts", ("troubleshoot",)), ("bpm", ("beam position monitor",)))),
    ("ts fault", (), (("ts", ("troubleshoot", "timing system")),)),
    ("beam position monitor", (), (("beam position monitor", ("bpm",)),)),
    ("beam position monitor", (), ()),
    ("fix ts now bpm later", (), (("ts", ("troubleshoot",)), ("bpm", ("beam position monitor",)))),
    ("", ("beam loss",), ()),
    ("", ("beam position monitor",), (("beam position monitor", ("bpm",)),)),
    (
        "ts",
        ("beam position monitor",),
        (("ts", ("troubleshoot",)), ("beam position monitor", ("bpm",))),
    ),
    ("undulator", (), (("ts", ("troubleshoot",)),)),
    ("", (), ()),
]


@pytest.mark.parametrize(("search_text", "phrases", "groups"), PLACEHOLDER_CASES)
def test_placeholder_count_equals_param_count(
    search_text: str,
    phrases: tuple[str, ...],
    groups: tuple[tuple[str, tuple[str, ...]], ...],
):
    """Every emitted statement binds exactly as many params as it has placeholders."""
    sql, params = build_expanded_tsquery(parsed(search_text, *phrases), expansion(*groups))

    assert sql.count("%s") == len(params)
