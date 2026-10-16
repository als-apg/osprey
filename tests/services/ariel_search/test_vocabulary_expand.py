"""Tests for deterministic query expansion against a facility vocabulary."""

import itertools
from pathlib import Path

import pytest

from osprey.services.ariel_search.search.base import ExpansionGroup, QueryExpansion
from osprey.services.ariel_search.vocabulary import (
    Concept,
    Vocabulary,
    expand_query,
    load_vocabulary,
    normalize,
)
from osprey.services.ariel_search.vocabulary.expand import Token, match_tokens
from osprey.services.ariel_search.vocabulary.model import _EDGE_TEXT

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

AMBIGUOUS_VOCABULARY = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - ts
  - canonical: timing system
    kind: acronym
    forms:
      - ts
"""

SYNONYM_VOCABULARY = """
concepts:
  - canonical: beam loss
    kind: synonym
    forms:
      - lost beam
      - beam dump
"""

BPM_PLURAL_VOCABULARY = """
concepts:
  - canonical: beam position monitor
    kind: acronym
    forms:
      - bpm
      - bpms
"""

#: The three members of `SYNONYM_VOCABULARY`, each with the other two.
SYNONYM_MEMBERS = [
    ("beam loss", ("lost beam", "beam dump")),
    ("lost beam", ("beam loss", "beam dump")),
    ("beam dump", ("beam loss", "lost beam")),
]


def build_vocabulary(concepts: list[Concept]) -> Vocabulary:
    """Build a `Vocabulary` directly, the way the loader does.

    Args:
        concepts: Concepts whose strings are already normalized.

    Returns:
        A vocabulary with the lookup tables and width the scan relies on.
    """
    forms_to_concepts: dict[str, list[Concept]] = {}
    canonical_to_concept: dict[str, Concept] = {}
    widths = [1]
    for concept in concepts:
        canonical_to_concept[concept.canonical] = concept
        widths.append(len(concept.canonical.split()))
        for form in concept.forms:
            forms_to_concepts.setdefault(form, []).append(concept)
            widths.append(len(form.split()))
    return Vocabulary(
        concepts=tuple(concepts),
        forms_to_concepts=forms_to_concepts,
        canonical_to_concept=canonical_to_concept,
        max_form_tokens=max(widths),
    )


def load(tmp_path: Path, text: str, **gates: bool) -> Vocabulary:
    """Load a vocabulary from YAML text, asserting it is error-free.

    Args:
        tmp_path: pytest temporary directory.
        text: The vocabulary file contents.
        **gates: Direction gates forwarded to `load_vocabulary`.

    Returns:
        The loaded vocabulary.
    """
    path = tmp_path / "vocabulary.yml"
    path.write_text(text, encoding="utf-8")
    vocabulary, errors, _warnings = load_vocabulary(path, **gates)
    assert errors == []
    assert vocabulary is not None
    return vocabulary


@pytest.fixture
def control_assistant(tmp_path: Path) -> Vocabulary:
    """The control-assistant-style example vocabulary, loaded from YAML."""
    return load(tmp_path, CONTROL_ASSISTANT_VOCABULARY)


@pytest.fixture
def ambiguous(tmp_path: Path) -> Vocabulary:
    """A vocabulary binding `ts` to two concepts."""
    return load(tmp_path, AMBIGUOUS_VOCABULARY)


@pytest.fixture
def synonyms(tmp_path: Path) -> Vocabulary:
    """A three-member `synonym` concept: one canonical and two forms."""
    return load(tmp_path, SYNONYM_VOCABULARY)


def expand(text: str, vocabulary: Vocabulary, **gates: bool) -> QueryExpansion:
    """Expand `text`, defaulting the gates to the shipped config defaults.

    Args:
        text: Query text.
        vocabulary: The vocabulary to match against.
        **gates: Overrides for `canonical_to_acronym` / `canonical_to_shorthand`.

    Returns:
        The resulting expansion.
    """
    return expand_query(
        text,
        vocabulary,
        canonical_to_acronym=gates.get("canonical_to_acronym", True),
        canonical_to_shorthand=gates.get("canonical_to_shorthand", False),
    )


def pairs(expansion: QueryExpansion) -> list[tuple[str, tuple[str, ...]]]:
    """Flatten groups to `(original, alternatives)` tuples for assertions.

    Args:
        expansion: The expansion to inspect.

    Returns:
        One tuple per group, in text order.
    """
    return [(group.original, group.alternatives) for group in expansion.groups]


class TestFormToCanonical:
    """The always-on direction: a typed form expands to its canonical."""

    def test_two_forms_yield_two_groups(self, control_assistant: Vocabulary) -> None:
        expansion = expand("ts bpm", control_assistant)

        assert pairs(expansion) == [
            ("ts", ("troubleshoot",)),
            ("bpm", ("beam position monitor",)),
        ]
        assert expansion.flattened_text == "ts bpm troubleshoot beam position monitor"

    def test_groups_are_expansion_group_instances(self, control_assistant: Vocabulary) -> None:
        expansion = expand("ts", control_assistant)

        assert isinstance(expansion, QueryExpansion)
        assert isinstance(expansion.groups[0], ExpansionGroup)
        assert expansion.groups[0].to_dict() == {
            "original": "ts",
            "alternatives": ["troubleshoot"],
        }

    def test_slash_form_survives_normalization(self, control_assistant: Vocabulary) -> None:
        expansion = expand("t/s bpm", control_assistant)

        assert pairs(expansion) == [
            ("t/s", ("troubleshoot",)),
            ("bpm", ("beam position monitor",)),
        ]

    def test_unmatched_tokens_are_left_alone(self, control_assistant: Vocabulary) -> None:
        expansion = expand("check the ts on shift", control_assistant)

        assert pairs(expansion) == [("ts", ("troubleshoot",))]
        assert expansion.flattened_text == "check the ts on shift troubleshoot"

    def test_no_match_leaves_query_untouched(self, control_assistant: Vocabulary) -> None:
        expansion = expand("undulator gap", control_assistant)

        assert expansion.groups == ()
        assert expansion.flattened_text == "undulator gap"

    def test_empty_text(self, control_assistant: Vocabulary) -> None:
        expansion = expand("", control_assistant)

        assert expansion.groups == ()
        assert expansion.flattened_text == ""


class TestAmbiguousForms:
    """A form bound to more than one concept expands to every binding."""

    def test_one_group_with_both_canonicals(self, ambiguous: Vocabulary) -> None:
        expansion = expand("ts fault", ambiguous)

        assert pairs(expansion) == [("ts", ("troubleshoot", "timing system"))]

    def test_flattened_text_carries_each_canonical_once(self, ambiguous: Vocabulary) -> None:
        expansion = expand("ts fault", ambiguous)

        assert expansion.flattened_text == "ts fault troubleshoot timing system"

    def test_alternatives_follow_file_order(self, tmp_path: Path) -> None:
        reversed_file = """
concepts:
  - canonical: timing system
    kind: acronym
    forms:
      - ts
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - ts
"""
        vocabulary = load(tmp_path, reversed_file)

        expansion = expand("ts", vocabulary)

        assert pairs(expansion) == [("ts", ("timing system", "troubleshoot"))]


class TestDirectionGates:
    """Canonical-to-form expansion is gated per concept kind."""

    def test_canonical_to_acronym_on(self, control_assistant: Vocabulary) -> None:
        expansion = expand("beam position monitor", control_assistant, canonical_to_acronym=True)

        assert pairs(expansion) == [("beam position monitor", ("bpm",))]
        assert expansion.flattened_text == "beam position monitor bpm"

    def test_canonical_to_acronym_off(self, control_assistant: Vocabulary) -> None:
        expansion = expand("beam position monitor", control_assistant, canonical_to_acronym=False)

        assert expansion.groups == ()
        assert expansion.flattened_text == "beam position monitor"

    def test_shorthand_canonical_gated_off_by_default(self, control_assistant: Vocabulary) -> None:
        expansion = expand("troubleshoot the offset", control_assistant)

        assert expansion.groups == ()
        assert expansion.flattened_text == "troubleshoot the offset"

    def test_shorthand_canonical_gated_on(self, control_assistant: Vocabulary) -> None:
        expansion = expand(
            "troubleshoot the offset", control_assistant, canonical_to_shorthand=True
        )

        assert pairs(expansion) == [("troubleshoot", ("t/s", "ts"))]
        assert expansion.flattened_text == "troubleshoot the offset t/s ts"

    def test_acronym_gate_does_not_leak_to_shorthand(self, control_assistant: Vocabulary) -> None:
        expansion = expand(
            "troubleshoot beam position monitor",
            control_assistant,
            canonical_to_acronym=True,
            canonical_to_shorthand=False,
        )

        assert pairs(expansion) == [("beam position monitor", ("bpm",))]


class TestSpanSelection:
    """Longest-form-first, left-to-right, non-overlapping."""

    def test_longer_form_wins_over_shorter_overlapping_form(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="beam position", kind="acronym", forms=("bp",)),
                Concept(canonical="beam position monitor", kind="acronym", forms=("bpm",)),
            ]
        )

        expansion = expand("beam position monitor", vocabulary)

        assert pairs(expansion) == [("beam position monitor", ("bpm",))]

    def test_prefix_form_does_not_fire_inside_longer_match(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="beam current", kind="shorthand", forms=("beam",)),
                Concept(
                    canonical="monitored beam position",
                    kind="shorthand",
                    forms=("beam position monitor",),
                ),
            ]
        )

        expansion = expand("beam position monitor", vocabulary)

        assert pairs(expansion) == [("beam position monitor", ("monitored beam position",))]

    def test_prefix_form_still_fires_when_the_longer_form_does_not_match(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="beam current", kind="shorthand", forms=("beam",)),
                Concept(
                    canonical="monitored beam position",
                    kind="shorthand",
                    forms=("beam position monitor",),
                ),
            ]
        )

        expansion = expand("beam position offset", vocabulary)

        assert pairs(expansion) == [("beam", ("beam current",))]

    def test_matched_span_is_consumed(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="orbit response matrix", kind="acronym", forms=("orm",)),
                Concept(canonical="response", kind="shorthand", forms=("resp",)),
            ]
        )

        expansion = expand("orbit response matrix", vocabulary)

        assert pairs(expansion) == [("orbit response matrix", ("orm",))]


class TestDeduplication:
    """New terms are added once; query terms are never repeated."""

    def test_repeated_form_yields_two_groups_but_one_new_term(
        self, control_assistant: Vocabulary
    ) -> None:
        expansion = expand("ts ts", control_assistant)

        assert pairs(expansion) == [("ts", ("troubleshoot",)), ("ts", ("troubleshoot",))]
        assert expansion.flattened_text == "ts ts troubleshoot"

    def test_canonical_already_in_query_is_not_appended(
        self, control_assistant: Vocabulary
    ) -> None:
        expansion = expand("ts troubleshoot", control_assistant)

        assert pairs(expansion) == [("ts", ("troubleshoot",))]
        assert expansion.flattened_text == "ts troubleshoot"

    def test_multiword_canonical_already_in_query_is_not_appended(
        self, control_assistant: Vocabulary
    ) -> None:
        expansion = expand("bpm beam position monitor", control_assistant)

        assert pairs(expansion)[0] == ("bpm", ("beam position monitor",))
        assert expansion.flattened_text == "bpm beam position monitor"

    def test_partial_token_overlap_is_not_a_containment(
        self, control_assistant: Vocabulary
    ) -> None:
        expansion = expand("bpm beam position", control_assistant)

        assert expansion.flattened_text == "bpm beam position beam position monitor"

    def test_trailing_punctuation_keeps_the_surface_and_appends_the_canonical(
        self, control_assistant: Vocabulary
    ) -> None:
        """The group's `original` is the stripped token; the surface `?` stays put."""
        expansion = expand("bpm?", control_assistant)

        assert pairs(expansion) == [("bpm", ("beam position monitor",))]
        assert expansion.flattened_text == "bpm? beam position monitor"

    def test_punctuated_query_terms_count_as_present(self, tmp_path: Path) -> None:
        """Behaviour change: dedupe compares stripped match tokens, not raw split words.

        Before, `bpm,` and `monitor.` were whole words that never equalled `bpm`
        or `monitor`, so both `bpm` and `beam position monitor` were appended
        although the query already said them. Now only the genuinely new `bpms`
        is added.
        """
        vocabulary = load(tmp_path, BPM_PLURAL_VOCABULARY)

        expansion = expand("bpm, beam position monitor.", vocabulary)

        assert pairs(expansion) == [
            ("bpm", ("beam position monitor",)),
            ("beam position monitor", ("bpm", "bpms")),
        ]
        assert expansion.flattened_text == "bpm, beam position monitor. bpms"

    def test_bracketed_query_term_counts_as_present(self, control_assistant: Vocabulary) -> None:
        """Behaviour change: `(troubleshoot)` already contains `troubleshoot`.

        Before, the raw word `(troubleshoot)` hid the canonical from the
        containment check and `troubleshoot` was appended a second time.
        """
        expansion = expand("ts (troubleshoot)", control_assistant)

        assert expansion.flattened_text == "ts (troubleshoot)"

    def test_span_is_never_its_own_alternative(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="ts", kind="acronym", forms=("timing sys",)),
                Concept(canonical="timing sys", kind="shorthand", forms=("ts",)),
            ]
        )

        expansion = expand("ts", vocabulary, canonical_to_acronym=True)

        assert pairs(expansion) == [("ts", ("timing sys",))]
        assert expansion.flattened_text == "ts timing sys"

    def test_span_with_no_surviving_alternative_produces_no_group(self) -> None:
        vocabulary = build_vocabulary(
            [Concept(canonical="ts", kind="acronym", forms=("ts alias",))]
        )
        vocabulary.forms_to_concepts["ts"] = [vocabulary.concepts[0]]

        expansion = expand("ts", vocabulary, canonical_to_acronym=False)

        assert expansion.groups == ()
        assert expansion.flattened_text == "ts"


class TestNormalization:
    """`original` is the matched span as it appears in the NORMALIZED text."""

    def test_hyphen_and_case_are_normalized_in_the_group(
        self, control_assistant: Vocabulary
    ) -> None:
        expansion = expand("Beam-Position   Monitor", control_assistant)

        assert pairs(expansion) == [("beam position monitor", ("bpm",))]

    def test_flattened_text_keeps_the_original_surface(self, control_assistant: Vocabulary) -> None:
        expansion = expand("Beam-Position   Monitor", control_assistant)

        assert expansion.flattened_text == "Beam-Position   Monitor bpm"

    def test_uppercase_form_matches(self, control_assistant: Vocabulary) -> None:
        expansion = expand("BPM offset", control_assistant)

        assert pairs(expansion) == [("bpm", ("beam position monitor",))]

    def test_normalize_is_the_shared_rule(self) -> None:
        assert normalize("Beam-Position   Monitor") == "beam position monitor"


class TestMatchTokens:
    """The shared tokenizer: normalize's split, edge stripping, original offsets."""

    def test_match_tokens_hyphen_word_with_trailing_comma(self) -> None:
        """`RF-trip,` splits at the hyphen; both tokens end their word at `p`."""
        assert match_tokens("RF-trip,") == [
            Token("rf", 0, 2, opens=False, closes=False, word_end=7),
            Token("trip", 3, 7, opens=False, closes=True, word_end=7),
        ]

    def test_match_tokens_bracketed_acronym_with_period(self) -> None:
        """`(BPM).` strips both edges; the core sits strictly inside them."""
        text = "(BPM)."
        assert match_tokens(text) == [Token("bpm", 1, 4, opens=True, closes=True, word_end=4)]
        assert text[1:4] == "BPM"

    def test_match_tokens_agrees_with_normalize_without_edges(self) -> None:
        """Without edge characters the joined norms are exactly `normalize`."""
        text = "  Beam-Position   Monitor t/s i/o\tSR01C "
        tokens = match_tokens(text)

        assert " ".join(t.norm for t in tokens) == normalize(text)
        assert not any(t.opens or t.closes for t in tokens)
        assert [text[t.start : t.end] for t in tokens] == [
            "Beam",
            "Position",
            "Monitor",
            "t/s",
            "i/o",
            "SR01C",
        ]

    def test_match_tokens_edge_only_token_is_dropped_and_breaks(self) -> None:
        """A spaced `(` / `...` / `:` vanishes but closes left and opens right."""
        tokens = match_tokens("beam ... position ( monitor :")

        assert [t.norm for t in tokens] == ["beam", "position", "monitor"]
        assert [(t.opens, t.closes) for t in tokens] == [
            (False, True),
            (True, True),
            (True, True),
        ]

    def test_match_tokens_interior_punctuation_is_kept(self) -> None:
        """Only the edges strip: `a.b` keeps its dot; `vert.` closes."""
        tokens = match_tokens("vert. orbit a.b")

        assert [t.norm for t in tokens] == ["vert", "orbit", "a.b"]
        assert [(t.opens, t.closes) for t in tokens] == [
            (False, True),
            (False, False),
            (False, False),
        ]

    def test_match_tokens_every_edge_character_strips(self) -> None:
        """Each character named in `_EDGE_TEXT` is stripped from both ends."""
        for edge in _EDGE_TEXT.split(" "):
            assert match_tokens(f"{edge}x{edge}") == [
                Token("x", 1, 2, opens=True, closes=True, word_end=2)
            ], edge

    def test_match_tokens_edge_text_renders_space_joined(self) -> None:
        assert _EDGE_TEXT == ", . : ; ! ? ( ) [ ] \" '"

    def test_match_tokens_empty_and_edge_only_text(self) -> None:
        assert match_tokens("") == []
        assert match_tokens("  ... ( ) - ") == []

    def test_match_tokens_offsets_index_the_original_string(self) -> None:
        """Offsets are code points into the untouched input, case and all."""
        text = "Check the BPM. RF-BPM, ok"
        tokens = match_tokens(text)

        assert [text[t.start : t.end] for t in tokens] == [
            "Check",
            "the",
            "BPM",
            "RF",
            "BPM",
            "ok",
        ]
        assert [text[: t.word_end].rsplit(" ", 1)[-1] for t in tokens] == [
            "Check",
            "the",
            "BPM",
            "RF-BPM",
            "RF-BPM",
            "ok",
        ]

    def test_match_tokens_is_reexported_from_expand(self) -> None:
        from osprey.services.ariel_search.vocabulary import expand, model

        assert expand.match_tokens is model.match_tokens
        assert expand.Token is model.Token


class TestTextAgnostic:
    """The function expands whatever string it is given — nothing more."""

    def test_field_filter_prefix_is_the_callers_problem(
        self, control_assistant: Vocabulary
    ) -> None:
        # `author:ts` is one whitespace token, so it does not match the form
        # `ts`. The keyword caller strips field filters before calling; this
        # function never parses search syntax.
        expansion = expand("author:ts bpm", control_assistant)

        assert pairs(expansion) == [("bpm", ("beam position monitor",))]

    def test_pattern_body_is_not_stripped_here(self, control_assistant: Vocabulary) -> None:
        # A caller that wants pattern bodies excluded must not pass them in.
        expansion = expand("ts /SR01C___BPM[0-9]+/", control_assistant)

        assert pairs(expansion) == [("ts", ("troubleshoot",))]

    def test_a_form_inside_a_pattern_body_would_match_if_passed(
        self, control_assistant: Vocabulary
    ) -> None:
        expansion = expand("/ ts /", control_assistant)

        assert pairs(expansion) == [("ts", ("troubleshoot",))]


class TestSynonymConcepts:
    """`synonym` is bidirectional and ungated: every member reaches the others."""

    @pytest.mark.parametrize(("member", "expected"), SYNONYM_MEMBERS)
    def test_each_member_reaches_the_other_two(
        self, synonyms: Vocabulary, member: str, expected: tuple[str, ...]
    ) -> None:
        expansion = expand(member, synonyms)

        assert pairs(expansion) == [(member, expected)]

    @pytest.mark.parametrize(
        ("to_acronym", "to_shorthand"), itertools.product([True, False], repeat=2)
    )
    @pytest.mark.parametrize(("member", "expected"), SYNONYM_MEMBERS)
    def test_no_gate_combination_affects_synonym(
        self,
        synonyms: Vocabulary,
        member: str,
        expected: tuple[str, ...],
        to_acronym: bool,
        to_shorthand: bool,
    ) -> None:
        expansion = expand(
            member,
            synonyms,
            canonical_to_acronym=to_acronym,
            canonical_to_shorthand=to_shorthand,
        )

        assert pairs(expansion) == [(member, expected)]

    def test_canonical_flattens_to_every_form_with_gates_off(self, synonyms: Vocabulary) -> None:
        expansion = expand(
            "beam loss",
            synonyms,
            canonical_to_acronym=False,
            canonical_to_shorthand=False,
        )

        assert expansion.flattened_text == "beam loss lost beam beam dump"

    def test_form_query_yields_the_canonical_first(self, synonyms: Vocabulary) -> None:
        expansion = expand("lost beam today", synonyms)

        assert pairs(expansion) == [("lost beam", ("beam loss", "beam dump"))]
        assert expansion.flattened_text == "lost beam today beam loss beam dump"

    def test_shorthand_form_query_still_yields_only_its_canonical(
        self, control_assistant: Vocabulary
    ) -> None:
        # `troubleshoot` also has the sibling form `t/s`; sibling forms are a
        # synonym-only direction and must not leak to the other kinds.
        expansion = expand("ts", control_assistant, canonical_to_shorthand=True)

        assert pairs(expansion) == [("ts", ("troubleshoot",))]

    def test_acronym_form_query_still_yields_only_its_canonical(self) -> None:
        vocabulary = build_vocabulary(
            [Concept(canonical="orbit response matrix", kind="acronym", forms=("orm", "or matrix"))]
        )

        expansion = expand("orm", vocabulary, canonical_to_acronym=True)

        assert pairs(expansion) == [("orm", ("orbit response matrix",))]

    def test_form_bound_to_two_kinds_lists_canonicals_before_siblings(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="orbit response matrix", kind="acronym", forms=("orm",)),
                Concept(canonical="beam loss", kind="synonym", forms=("orm", "lost beam")),
            ]
        )

        expansion = expand("orm", vocabulary)

        assert pairs(expansion) == [("orm", ("orbit response matrix", "beam loss", "lost beam"))]


PUNCTUATED_FORM_VOCABULARY = """
concepts:
  - canonical: vertical orbit
    kind: shorthand
    forms:
      - vert. orbit
"""


def scan(
    text: str,
    vocabulary: Vocabulary,
    *,
    canonical_to_acronym: bool = True,
    canonical_to_shorthand: bool = False,
    limit: int | None = None,
) -> list[tuple[str, int, int, tuple[str, ...], tuple[str, ...]]]:
    """Run the private scan over `text` and flatten each span for assertions.

    Args:
        text: Text to tokenize and scan.
        vocabulary: The vocabulary to match against.
        canonical_to_acronym: Gate for `acronym` concepts.
        canonical_to_shorthand: Gate for `shorthand` concepts.
        limit: Optional code-point cap on a span's last token end.

    Returns:
        One `(key, start, stop, alternatives, concepts)` tuple per span.
    """
    from osprey.services.ariel_search.vocabulary.expand import _Gates, _scan

    tokens = match_tokens(text)
    gates = _Gates(
        canonical_to_acronym=canonical_to_acronym,
        canonical_to_shorthand=canonical_to_shorthand,
    )
    return [
        (
            " ".join(token.norm for token in tokens[span.start : span.stop]),
            span.start,
            span.stop,
            tuple(span.alternatives),
            tuple(span.concepts),
        )
        for span in _scan(tokens, vocabulary, gates, limit=limit)
    ]


class TestScan:
    """The one shared scan: spans, bound concepts, interior patterns, limit."""

    def test_scan_yields_token_range_alternatives_and_concepts(
        self, control_assistant: Vocabulary
    ) -> None:
        assert scan("check the BPM now", control_assistant) == [
            ("bpm", 2, 3, ("beam position monitor",), ("beam position monitor",)),
        ]

    def test_scan_ambiguous_form_binds_every_canonical_in_file_order(
        self, ambiguous: Vocabulary
    ) -> None:
        assert scan("ts", ambiguous) == [
            ("ts", 0, 1, ("troubleshoot", "timing system"), ("troubleshoot", "timing system")),
        ]

    def test_scan_matched_canonical_binds_itself(self, control_assistant: Vocabulary) -> None:
        assert scan("Beam-Position Monitor", control_assistant) == [
            ("beam position monitor", 0, 3, ("bpm",), ("beam position monitor",)),
        ]

    def test_scan_gated_off_canonical_is_not_a_span(self, control_assistant: Vocabulary) -> None:
        assert scan("troubleshoot", control_assistant, canonical_to_shorthand=False) == []
        assert scan("troubleshoot", control_assistant, canonical_to_shorthand=True) == [
            ("troubleshoot", 0, 1, ("t/s", "ts"), ("troubleshoot",)),
        ]

    def test_scan_span_both_form_and_canonical_binds_deduped_in_file_order(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="orbit", kind="synonym", forms=("orb",)),
                Concept(canonical="closed orbit", kind="synonym", forms=("orbit",)),
            ]
        )

        assert scan("orbit", vocabulary) == [
            ("orbit", 0, 1, ("closed orbit", "orb"), ("closed orbit", "orbit")),
        ]

    def test_scan_synonym_form_alternatives_include_siblings(self, synonyms: Vocabulary) -> None:
        assert scan("lost beam", synonyms) == [
            ("lost beam", 0, 2, ("beam loss", "beam dump"), ("beam loss",)),
        ]

    def test_scan_empty_alternative_span_is_still_yielded(self) -> None:
        vocabulary = build_vocabulary(
            [Concept(canonical="beam", kind="synonym", forms=("beam",))],
        )

        assert scan("beam position", vocabulary) == [("beam", 0, 1, (), ("beam",))]

    def test_scan_spans_do_not_overlap_and_longest_wins(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="beam current", kind="shorthand", forms=("beam",)),
                Concept(canonical="bpm", kind="shorthand", forms=("beam position monitor",)),
            ]
        )

        assert [span[:3] for span in scan("beam position monitor beam", vocabulary)] == [
            ("beam position monitor", 0, 3),
            ("beam", 3, 4),
        ]

    def test_interior_period_after_first_token_matches_nothing(
        self, control_assistant: Vocabulary
    ) -> None:
        assert scan("beam. position monitor", control_assistant) == []
        assert pairs(expand("beam. position monitor", control_assistant)) == []

    def test_interior_spaced_period_matches_nothing(self, control_assistant: Vocabulary) -> None:
        assert scan("beam . position monitor", control_assistant) == []
        assert pairs(expand("beam . position monitor", control_assistant)) == []

    def test_interior_comma_inside_canonical_matches_nothing(
        self, control_assistant: Vocabulary
    ) -> None:
        assert scan("beam position, monitor", control_assistant) == []

    def test_interior_outer_edges_do_not_block_a_span(self, control_assistant: Vocabulary) -> None:
        assert [span[:3] for span in scan("(beam position monitor).", control_assistant)] == [
            ("beam position monitor", 0, 3),
        ]
        assert pairs(expand("(BPM), ts.", control_assistant)) == [
            ("bpm", ("beam position monitor",)),
            ("ts", ("troubleshoot",)),
        ]

    def test_interior_pattern_of_form_matches_its_own_punctuation(self, tmp_path: Path) -> None:
        vocabulary = load(tmp_path, PUNCTUATED_FORM_VOCABULARY)

        assert scan("vert. orbit drift", vocabulary) == [
            ("vert orbit", 0, 2, ("vertical orbit",), ("vertical orbit",)),
        ]
        assert pairs(expand("vert. orbit drift", vocabulary)) == [
            ("vert orbit", ("vertical orbit",)),
        ]

    def test_interior_pattern_of_form_rejects_unpunctuated_text(self, tmp_path: Path) -> None:
        vocabulary = load(tmp_path, PUNCTUATED_FORM_VOCABULARY)

        assert scan("vert orbit drift", vocabulary) == []
        assert pairs(expand("vert orbit drift", vocabulary)) == []

    def test_interior_fallback_for_hand_built_multi_token_form(self) -> None:
        vocabulary = build_vocabulary(
            [Concept(canonical="bpm", kind="shorthand", forms=("beam position monitor",))],
        )
        assert vocabulary.interior_patterns == {}

        assert scan("beam position monitor", vocabulary) == [
            ("beam position monitor", 0, 3, ("bpm",), ("bpm",)),
        ]
        assert scan("beam, position monitor", vocabulary) == []
        assert scan("beam position. monitor", vocabulary) == []

    def test_interior_rejected_span_falls_through_to_narrower_widths(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="bpm", kind="shorthand", forms=("beam position monitor",)),
                Concept(canonical="mon", kind="shorthand", forms=("monitor",)),
            ]
        )

        assert [span[:3] for span in scan("beam. position monitor", vocabulary)] == [
            ("monitor", 2, 3),
        ]

    def test_limit_none_scans_the_whole_text(self, control_assistant: Vocabulary) -> None:
        assert [span[0] for span in scan("ts then bpm", control_assistant)] == ["ts", "bpm"]

    def test_limit_admits_span_ending_exactly_at_limit(self, control_assistant: Vocabulary) -> None:
        text = "ts then bpm"
        end = text.index("bpm") + len("bpm")

        assert [span[0] for span in scan(text, control_assistant, limit=end)] == ["ts", "bpm"]
        assert [span[0] for span in scan(text, control_assistant, limit=end - 1)] == ["ts"]

    def test_limit_counts_the_core_not_trailing_edges(self, control_assistant: Vocabulary) -> None:
        text = "bpm. more"

        assert [span[0] for span in scan(text, control_assistant, limit=3)] == ["bpm"]

    def test_limit_crossing_span_falls_back_to_narrower_width(self) -> None:
        vocabulary = build_vocabulary(
            [
                Concept(canonical="bp", kind="shorthand", forms=("beam position",)),
                Concept(canonical="bpm", kind="shorthand", forms=("beam position monitor",)),
            ]
        )
        text = "beam position monitor"
        limit = text.index("monitor") + 2

        assert [span[:3] for span in scan(text, vocabulary, limit=limit)] == [
            ("beam position", 0, 2),
        ]

    def test_limit_zero_yields_nothing(self, control_assistant: Vocabulary) -> None:
        assert scan("bpm", control_assistant, limit=0) == []
