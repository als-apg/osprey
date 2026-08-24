"""Tests for deterministic query expansion against a facility vocabulary."""

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
