"""Tests for the ARIEL vocabulary model, loader and vendored stopword list."""

from pathlib import Path

import pytest

from osprey.services.ariel_search.vocabulary import (
    CONCEPT_KINDS,
    Concept,
    ConceptKind,
    Vocabulary,
    load_vocabulary,
    normalize,
)
from osprey.services.ariel_search.vocabulary.stopwords import _ENGLISH_FTS_STOPWORDS

VALID_VOCABULARY = """
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
      - beam-position monitor unit
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

# Exactly three errors: an unknown kind, an empty canonical, and a form that is
# its own canonical.
THREE_ERROR_VOCABULARY = """
concepts:
  - canonical: troubleshoot
    kind: verb
    forms:
      - t/s
  - canonical: ""
    kind: acronym
    forms:
      - bpm
  - canonical: beam position monitor
    kind: acronym
    forms:
      - Beam-Position Monitor
"""


def write_vocabulary(tmp_path: Path, text: str, name: str = "vocabulary.yml") -> Path:
    """Write a vocabulary file into ``tmp_path`` and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestNormalize:
    """Tests for the shared normalization rule (FR2)."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("T/S", "t/s"),
            ("Beam-Position Monitor", "beam position monitor"),
            ("beam   position \n monitor", "beam position monitor"),
            ("  BPM  ", "bpm"),
            ("I/O", "i/o"),
            ("multi--dash", "multi dash"),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_normalization_examples(self, raw: str, expected: str) -> None:
        """Lowercase, dash to space, slash kept, whitespace collapsed."""
        assert normalize(raw) == expected

    def test_normalization_is_idempotent(self) -> None:
        """Normalizing an already-normalized string changes nothing."""
        once = normalize("Beam-Position  Monitor")
        assert normalize(once) == once


class TestLoadValidVocabulary:
    """Tests for a vocabulary that loads cleanly."""

    def test_loads_concepts_in_file_order(self, tmp_path: Path) -> None:
        """Concepts are returned in file order with normalized strings."""
        vocab, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, VALID_VOCABULARY))

        assert errors == []
        assert warnings == []
        assert vocab is not None
        assert isinstance(vocab, Vocabulary)
        assert vocab.concept_count == 2
        assert vocab.concepts[0] == Concept(
            canonical="troubleshoot", kind="shorthand", forms=("t/s", "ts")
        )
        assert vocab.concepts[1] == Concept(
            canonical="beam position monitor",
            kind="acronym",
            forms=("bpm", "beam position monitor unit"),
        )

    def test_lookup_tables(self, tmp_path: Path) -> None:
        """Forms and canonicals are keyed by their normalized spelling."""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, VALID_VOCABULARY))

        assert vocab is not None
        assert vocab.concepts_for_form("t/s") == [vocab.concepts[0]]
        assert vocab.concepts_for_form("bpm") == [vocab.concepts[1]]
        assert vocab.concepts_for_form("nothing") == []
        assert vocab.forms_to_concepts["ts"] == [vocab.concepts[0]]
        assert vocab.concept_for_canonical("beam position monitor") is vocab.concepts[1]
        assert vocab.concept_for_canonical("BPM") is None

    def test_max_form_tokens_covers_canonicals_and_forms(self, tmp_path: Path) -> None:
        """The n-gram width spans every matchable string, not just the forms."""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, VALID_VOCABULARY))

        assert vocab is not None
        # "beam position monitor unit" is the longest matchable string.
        assert vocab.max_form_tokens == 4

    def test_max_form_tokens_from_canonical_alone(self, tmp_path: Path) -> None:
        """A long canonical with short forms still sets the n-gram width."""
        text = """
concepts:
  - canonical: beam position monitor
    kind: acronym
    forms:
      - bpm
"""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is not None
        assert vocab.max_form_tokens == 3

    def test_repeated_form_inside_one_concept_is_kept_once(self, tmp_path: Path) -> None:
        """A form listed twice under one concept is de-duplicated silently."""
        text = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
      - T/S
"""
        vocab, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == []
        assert vocab is not None
        assert vocab.concepts[0].forms == ("t/s",)

    @pytest.mark.parametrize("kind", CONCEPT_KINDS)
    def test_every_declared_kind_loads(self, tmp_path: Path, kind: ConceptKind) -> None:
        """Every kind in CONCEPT_KINDS loads cleanly, gates on or off.

        Pins the whole pipeline to the declared tuple: a kind added to
        ``CONCEPT_KINDS`` without teaching the loader about it fails here
        instead of at the first vocabulary file that uses it.
        """
        text = f"""
concepts:
  - canonical: beam loss
    kind: {kind}
    forms:
      - lost beam
"""
        path = write_vocabulary(tmp_path, text)
        for canonical_to_acronym in (True, False):
            for canonical_to_shorthand in (True, False):
                vocab, errors, warnings = load_vocabulary(
                    path,
                    canonical_to_acronym=canonical_to_acronym,
                    canonical_to_shorthand=canonical_to_shorthand,
                )

                assert errors == []
                assert warnings == []
                assert vocab is not None
                assert vocab.concepts[0] == Concept(
                    canonical="beam loss", kind=kind, forms=("lost beam",)
                )

    def test_all_three_kinds_in_one_file(self, tmp_path: Path) -> None:
        """A file mixing every kind loads with no errors and no warnings."""
        text = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
  - canonical: beam position monitor
    kind: acronym
    forms:
      - bpm
  - canonical: beam loss
    kind: synonym
    forms:
      - lost beam
      - beam dump
"""
        vocab, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == []
        assert vocab is not None
        assert {concept.kind for concept in vocab.concepts} == set(CONCEPT_KINDS)


class TestAmbiguousForm:
    """A form bound to more than one concept is legal and warned about."""

    def test_loads_with_one_warning_and_no_errors(self, tmp_path: Path) -> None:
        """The two-binding form loads: one warning, zero errors."""
        vocab, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, AMBIGUOUS_VOCABULARY))

        assert errors == []
        assert vocab is not None
        assert len(warnings) == 1

    def test_warning_text_names_every_canonical(self, tmp_path: Path) -> None:
        """The warning matches the shape pinned by the proposal."""
        _, _, warnings = load_vocabulary(write_vocabulary(tmp_path, AMBIGUOUS_VOCABULARY))

        assert warnings == [
            'form "ts" is bound to 2 concepts: troubleshoot, timing system '
            '— queries containing "ts" expand to both'
        ]

    def test_lookup_returns_both_concepts(self, tmp_path: Path) -> None:
        """The form lookup keeps every binding, in file order."""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, AMBIGUOUS_VOCABULARY))

        assert vocab is not None
        bound = vocab.concepts_for_form("ts")
        assert [concept.canonical for concept in bound] == ["troubleshoot", "timing system"]
        assert vocab.forms_to_concepts["ts"] == bound

    def test_three_way_ambiguity_names_all_of_them(self, tmp_path: Path) -> None:
        """More than two bindings drop the two-way wording."""
        text = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms: [ts]
  - canonical: timing system
    kind: acronym
    forms: [ts]
  - canonical: thermal shield
    kind: acronym
    forms: [ts]
"""
        vocab, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert vocab is not None
        assert len(vocab.concepts_for_form("ts")) == 3
        assert warnings == [
            'form "ts" is bound to 3 concepts: troubleshoot, timing system, thermal shield '
            '— queries containing "ts" expand to all of them'
        ]


class TestPrefixCollisionWarning:
    """A form that is a strict token prefix of another concept's form."""

    def test_warns_naming_both_concepts(self, tmp_path: Path) -> None:
        """The shorter form can never fire where the longer one matches."""
        text = """
concepts:
  - canonical: beam position monitor
    kind: acronym
    forms: [bpm]
  - canonical: beam position monitor offset
    kind: acronym
    forms: [bpm offset]
"""
        vocab, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert vocab is not None
        assert len(warnings) == 1
        assert 'form "bpm" (concept "beam position monitor")' in warnings[0]
        assert 'form "bpm offset" (concept "beam position monitor offset")' in warnings[0]
        assert "strict prefix" in warnings[0]

    def test_no_warning_for_forms_of_the_same_concept(self, tmp_path: Path) -> None:
        """One concept's own forms may shadow each other harmlessly."""
        text = """
concepts:
  - canonical: beam position monitor
    kind: acronym
    forms:
      - bpm
      - bpm unit
"""
        _, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == []

    def test_no_warning_for_a_non_token_prefix(self, tmp_path: Path) -> None:
        """Prefixing is token-wise, not character-wise."""
        text = """
concepts:
  - canonical: beam position monitor
    kind: acronym
    forms: [bp]
  - canonical: beam position monitor offset
    kind: acronym
    forms: [bpm offset]
"""
        _, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == []


class TestStopwordScope:
    """Only strings that reach ``plainto_tsquery`` are checked (FR1)."""

    SHORTHAND_STOPWORD_FORM = """
concepts:
  - canonical: current setpoint
    kind: shorthand
    forms:
      - can
"""

    STOPWORD_CANONICAL = """
concepts:
  - canonical: can
    kind: acronym
    forms:
      - cavity number
"""

    def test_shorthand_form_warns_only_when_its_gate_is_on(self, tmp_path: Path) -> None:
        """A shorthand form is checked iff ``canonical_to_shorthand``."""
        path = write_vocabulary(tmp_path, self.SHORTHAND_STOPWORD_FORM)

        _, errors_on, warnings_on = load_vocabulary(path, canonical_to_shorthand=True)
        _, errors_off, warnings_off = load_vocabulary(path, canonical_to_shorthand=False)

        assert errors_on == [] and errors_off == []
        assert warnings_off == []
        assert len(warnings_on) == 1
        assert 'form "can" of concept "current setpoint"' in warnings_on[0]
        assert "stopword" in warnings_on[0]

    def test_acronym_form_warns_only_when_its_gate_is_on(self, tmp_path: Path) -> None:
        """An acronym form is checked iff ``canonical_to_acronym``."""
        text = """
concepts:
  - canonical: current setpoint
    kind: acronym
    forms:
      - can
"""
        path = write_vocabulary(tmp_path, text)

        _, _, warnings_on = load_vocabulary(path, canonical_to_acronym=True)
        _, _, warnings_off = load_vocabulary(path, canonical_to_acronym=False)

        assert len(warnings_on) == 1
        assert warnings_off == []

    @pytest.mark.parametrize("canonical_to_acronym", [True, False])
    @pytest.mark.parametrize("canonical_to_shorthand", [True, False])
    def test_stopword_canonical_warns_under_both_settings(
        self, tmp_path: Path, canonical_to_acronym: bool, canonical_to_shorthand: bool
    ) -> None:
        """A canonical always reaches the tsquery, so it is always checked."""
        path = write_vocabulary(tmp_path, self.STOPWORD_CANONICAL)

        vocab, errors, warnings = load_vocabulary(
            path,
            canonical_to_acronym=canonical_to_acronym,
            canonical_to_shorthand=canonical_to_shorthand,
        )

        assert errors == []
        assert vocab is not None
        assert len(warnings) == 1
        assert 'canonical "can"' in warnings[0]
        assert "stopword" in warnings[0]

    def test_multi_word_string_with_one_real_word_is_not_a_stopword(self, tmp_path: Path) -> None:
        """``plainto_tsquery`` keeps the non-stop token, so nothing is dead."""
        text = """
concepts:
  - canonical: out of tolerance reading
    kind: acronym
    forms: [otr]
"""
        _, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == []

    @pytest.mark.parametrize("canonical_to_acronym", [True, False])
    @pytest.mark.parametrize("canonical_to_shorthand", [True, False])
    def test_synonym_form_warns_under_both_settings(
        self, tmp_path: Path, canonical_to_acronym: bool, canonical_to_shorthand: bool
    ) -> None:
        """A synonym form always reaches the tsquery, so it is always checked."""
        text = """
concepts:
  - canonical: beam loss
    kind: synonym
    forms:
      - the
"""
        path = write_vocabulary(tmp_path, text)

        vocab, errors, warnings = load_vocabulary(
            path,
            canonical_to_acronym=canonical_to_acronym,
            canonical_to_shorthand=canonical_to_shorthand,
        )

        assert errors == []
        assert vocab is not None
        assert len(warnings) == 1
        assert 'form "the" of concept "beam loss"' in warnings[0]
        assert "stopword" in warnings[0]


class TestFileLevelErrors:
    """Errors about the file itself rather than its content."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing file is one error, not an exception."""
        missing = tmp_path / "nope.yml"

        vocab, errors, warnings = load_vocabulary(missing)

        assert vocab is None
        assert warnings == []
        assert len(errors) == 1
        assert "not found" in errors[0]
        assert str(missing) in errors[0]

    def test_unreadable_file(self, tmp_path: Path) -> None:
        """A directory in place of the file is reported, not raised."""
        directory = tmp_path / "vocabulary.yml"
        directory.mkdir()

        vocab, errors, _ = load_vocabulary(directory)

        assert vocab is None
        assert len(errors) == 1
        assert "could not be read" in errors[0]

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """A YAML syntax error is reported with the parser's detail."""
        path = write_vocabulary(tmp_path, "concepts: [\n  - canonical: x\n")

        vocab, errors, _ = load_vocabulary(path)

        assert vocab is None
        assert len(errors) == 1
        assert "not valid YAML" in errors[0]
        assert "\n" not in errors[0]

    @pytest.mark.parametrize(
        ("text", "expected_type"),
        [("- a\n- b\n", "list"), ("just a string\n", "str"), ("", "NoneType")],
    )
    def test_non_mapping_document(self, tmp_path: Path, text: str, expected_type: str) -> None:
        """A document that is not a mapping is a single error."""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert errors == [
            "vocabulary file must contain a YAML mapping with a 'concepts' key, "
            f"got {expected_type}"
        ]

    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        """Anything besides ``concepts`` at the top level is refused."""
        text = """
version: 1
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms: [t/s]
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert errors == ["unknown top-level key(s): version (only 'concepts' is allowed)"]

    def test_missing_concepts_key(self, tmp_path: Path) -> None:
        """A mapping without ``concepts`` is refused."""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, "other: 1\n"))

        assert vocab is None
        assert "missing required top-level key 'concepts'" in errors

    @pytest.mark.parametrize("text", ["concepts: []\n", "concepts: {}\n"])
    def test_concepts_must_be_a_non_empty_list(self, tmp_path: Path, text: str) -> None:
        """An empty list, or a non-list, is refused."""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert len(errors) == 1
        assert "'concepts' must be a non-empty list" in errors[0]


class TestConceptErrors:
    """Per-concept errors, each naming the entry it belongs to."""

    def test_non_mapping_entry(self, tmp_path: Path) -> None:
        """A scalar list item is refused, naming its index."""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, "concepts:\n  - bpm\n"))

        assert vocab is None
        assert errors == ["concepts[0]: must be a mapping, got str"]

    def test_unknown_concept_key(self, tmp_path: Path) -> None:
        """An unknown key names the entry and the allowed keys."""
        text = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms: [t/s]
    aliases: [ts]
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert errors == [
            "concepts[0] ('troubleshoot'): unknown key(s): aliases "
            "(allowed: canonical, forms, kind)"
        ]

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ("    kind: shorthand\n    forms: [t/s]\n", "missing required key 'canonical'"),
            (
                "    canonical: ''\n    kind: shorthand\n    forms: [t/s]\n",
                "'canonical' must be a non-empty string",
            ),
            (
                "    canonical: 7\n    kind: shorthand\n    forms: [t/s]\n",
                "'canonical' must be a non-empty string",
            ),
            ("    canonical: troubleshoot\n    forms: [t/s]\n", "missing required key 'kind'"),
            (
                "    canonical: troubleshoot\n    kind: verb\n    forms: [t/s]\n",
                "unknown kind 'verb' (expected one of: acronym, shorthand, synonym)",
            ),
            ("    canonical: troubleshoot\n    kind: shorthand\n", "missing required key 'forms'"),
            (
                "    canonical: troubleshoot\n    kind: shorthand\n    forms: []\n",
                "'forms' must be a non-empty list",
            ),
            (
                "    canonical: troubleshoot\n    kind: shorthand\n    forms: t/s\n",
                "'forms' must be a non-empty list, got str",
            ),
            (
                "    canonical: troubleshoot\n    kind: shorthand\n    forms: [7]\n",
                "forms[0] must be a string, got int",
            ),
            (
                "    canonical: troubleshoot\n    kind: shorthand\n    forms: ['']\n",
                "forms[0] must be a non-empty string",
            ),
        ],
    )
    def test_single_concept_error(self, tmp_path: Path, entry: str, expected: str) -> None:
        """Each malformed field yields exactly one error naming the entry."""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, f"concepts:\n  -\n{entry}"))

        assert vocab is None
        assert len(errors) == 1
        assert errors[0].startswith("concepts[0]")
        assert expected in errors[0]

    def test_form_equal_to_its_canonical(self, tmp_path: Path) -> None:
        """A form is only useful if it differs from the canonical."""
        text = """
concepts:
  - canonical: beam position monitor
    kind: acronym
    forms:
      - Beam-Position   Monitor
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert len(errors) == 1
        assert "is the same as its canonical" in errors[0]
        assert "beam position monitor" in errors[0]

    def test_duplicate_canonical_after_normalization(self, tmp_path: Path) -> None:
        """Two entries cannot claim the same canonical."""
        text = """
concepts:
  - canonical: beam position monitor
    kind: acronym
    forms: [bpm]
  - canonical: Beam-Position Monitor
    kind: shorthand
    forms: [bpms]
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert len(errors) == 1
        assert "duplicate canonical 'beam position monitor'" in errors[0]
        assert "concepts[0]" in errors[0]
        assert errors[0].startswith("concepts[1]")

    def test_every_error_is_collected(self, tmp_path: Path) -> None:
        """A three-error file yields exactly three errors in one pass."""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, THREE_ERROR_VOCABULARY))

        assert vocab is None
        assert len(errors) == 3
        assert "unknown kind 'verb'" in errors[0]
        assert "'canonical' must be a non-empty string" in errors[1]
        assert "is the same as its canonical" in errors[2]

    def test_errors_suppress_the_vocabulary_but_not_the_warnings(self, tmp_path: Path) -> None:
        """Warnings from the well-formed entries still surface alongside errors."""
        text = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms: [ts]
  - canonical: timing system
    kind: acronym
    forms: [ts]
  - canonical: beam position monitor
    kind: telescope
    forms: [bpm]
"""
        vocab, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert len(errors) == 1
        assert len(warnings) == 1
        assert 'form "ts" is bound to 2 concepts' in warnings[0]


class TestStopwordList:
    """The vendored PostgreSQL ``english.stop`` list."""

    def test_length_is_pinned(self) -> None:
        """PostgreSQL's english.stop has 127 words."""
        assert len(_ENGLISH_FTS_STOPWORDS) == 127

    def test_is_a_frozenset_of_lowercase_words(self) -> None:
        """The list is immutable and already normalized."""
        assert isinstance(_ENGLISH_FTS_STOPWORDS, frozenset)
        assert all(word == word.lower() and word.strip() == word for word in _ENGLISH_FTS_STOPWORDS)

    @pytest.mark.parametrize("word", ["the", "a", "is", "not", "don", "s", "t", "now"])
    def test_known_members(self, word: str) -> None:
        """Spot-check words the english configuration discards."""
        assert word in _ENGLISH_FTS_STOPWORDS

    @pytest.mark.parametrize("word", ["bpm", "troubleshoot", "beam", "monitor"])
    def test_known_non_members(self, word: str) -> None:
        """Facility words are not stopwords."""
        assert word not in _ENGLISH_FTS_STOPWORDS


PUNCTUATED_VOCABULARY = """
concepts:
  - canonical: vertical orbit
    kind: shorthand
    forms:
      - vert. orbit
      - Vert-Orbit
  - canonical: beam position monitor
    kind: acronym
    forms:
      - BPM
      - "(BPM)"
"""


class TestInteriorPatterns:
    """Tests for tokenizer-norm keys, raw spellings and the interior-pattern table."""

    def test_interior_pattern_records_a_stripped_period(self, tmp_path: Path) -> None:
        """``vert. orbit`` keys as ``vert orbit`` with a closing break after ``vert``."""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, PUNCTUATED_VOCABULARY))

        assert errors == []
        assert vocab is not None
        # "vert. orbit" and "Vert-Orbit" share one key; the union keeps both patterns.
        assert vocab.interior_patterns["vert orbit"] == frozenset(
            {((True, False),), ((False, False),)}
        )

    def test_interior_pattern_of_a_punctuation_free_phrase(self, tmp_path: Path) -> None:
        """A three-token spelling with no edge characters has two no-break boundaries."""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, PUNCTUATED_VOCABULARY))

        assert vocab is not None
        assert vocab.interior_patterns["beam position monitor"] == frozenset(
            {((False, False), (False, False))}
        )

    def test_interior_pattern_of_a_single_token_is_empty(self, tmp_path: Path) -> None:
        """A one-token key has no interior boundary, whatever its edge characters."""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, PUNCTUATED_VOCABULARY))

        assert vocab is not None
        assert vocab.interior_patterns["bpm"] == frozenset({()})

    def test_interior_patterns_cover_every_key(self, tmp_path: Path) -> None:
        """Every form and canonical key has an entry in the table."""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, PUNCTUATED_VOCABULARY))

        assert vocab is not None
        keys = set(vocab.forms_to_concepts) | set(vocab.canonical_to_concept)
        assert set(vocab.interior_patterns) == keys

    def test_interior_patterns_union_across_concepts(self, tmp_path: Path) -> None:
        """A key bound by two concepts unions the patterns of both spellings."""
        text = """
concepts:
  - canonical: vertical orbit
    kind: shorthand
    forms:
      - vert. orbit
  - canonical: vertical orbit feedback
    kind: shorthand
    forms:
      - vert, orbit
      - vert orbit
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert vocab is not None
        assert [c.canonical for c in vocab.forms_to_concepts["vert orbit"]] == [
            "vertical orbit",
            "vertical orbit feedback",
        ]
        assert vocab.interior_patterns["vert orbit"] == frozenset(
            {((True, False),), ((False, False),)}
        )

    def test_interior_patterns_default_empty_on_hand_built_vocabulary(self) -> None:
        """A ``Vocabulary`` built without the table gets an empty one."""
        concept = Concept(canonical="beam position monitor", kind="acronym", forms=("bpm",))
        vocab = Vocabulary(
            concepts=(concept,),
            forms_to_concepts={"bpm": [concept]},
            canonical_to_concept={"beam position monitor": concept},
            max_form_tokens=3,
        )
        assert vocab.interior_patterns == {}

    def test_keys_use_stripped_tokenizer_norm(self, tmp_path: Path) -> None:
        """Forms and canonicals are keyed with edge characters stripped."""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, PUNCTUATED_VOCABULARY))

        assert vocab is not None
        assert vocab.concepts[0].forms == ("vert orbit",)
        assert vocab.concepts_for_form("vert orbit") == [vocab.concepts[0]]
        assert "vert. orbit" not in vocab.forms_to_concepts
        assert vocab.concepts[1].forms == ("bpm",)
        assert "(bpm)" not in vocab.forms_to_concepts

    def test_stripped_canonical_is_the_lookup_key(self, tmp_path: Path) -> None:
        """A canonical carrying edge characters is keyed by its stripped norm."""
        text = """
concepts:
  - canonical: "Vert. Orbit"
    kind: shorthand
    forms:
      - vo
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert vocab is not None
        assert vocab.concepts[0].canonical == "vert orbit"
        assert vocab.concept_for_canonical("vert orbit") is vocab.concepts[0]
        assert vocab.interior_patterns["vert orbit"] == frozenset({((True, False),)})

    def test_canonical_empty_after_stripping_is_an_error(self, tmp_path: Path) -> None:
        """A canonical of edge characters only reuses the non-empty-string error."""
        text = """
concepts:
  - canonical: "..."
    kind: shorthand
    forms:
      - x
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert errors == ["concepts[0] ('...'): 'canonical' must be a non-empty string"]

    def test_canonicals_colliding_after_stripping_are_duplicates(self, tmp_path: Path) -> None:
        """Two canonicals equal once edge characters are stripped collide."""
        text = """
concepts:
  - canonical: vert orbit
    kind: shorthand
    forms:
      - vo
  - canonical: "vert. orbit"
    kind: shorthand
    forms:
      - v.o
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert errors == [
            "concepts[1] ('vert. orbit'): duplicate canonical 'vert orbit' after "
            "normalization — already defined by concepts[0]"
        ]

    def test_form_equal_to_canonical_after_stripping_is_an_error(self, tmp_path: Path) -> None:
        """A form that strips down to its canonical reuses the same-as-canonical error."""
        text = """
concepts:
  - canonical: bpm
    kind: acronym
    forms:
      - "(BPM)"
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert errors == [
            "concepts[0] ('bpm'): forms[0] '(BPM)' is the same as its canonical "
            "'bpm' after normalization"
        ]

    def test_form_empty_after_stripping_is_an_error(self, tmp_path: Path) -> None:
        """A form of edge characters only reuses the non-empty-string error."""
        text = """
concepts:
  - canonical: bpm
    kind: acronym
    forms:
      - "?!"
"""
        vocab, errors, _ = load_vocabulary(write_vocabulary(tmp_path, text))

        assert vocab is None
        assert errors == ["concepts[0] ('bpm'): forms[0] must be a non-empty string"]

    def test_raw_spellings_retained_in_file_order(self, tmp_path: Path) -> None:
        """Every accepted raw spelling is kept verbatim, in file order."""
        vocab, _, _ = load_vocabulary(write_vocabulary(tmp_path, PUNCTUATED_VOCABULARY))

        assert vocab is not None
        assert vocab.concepts[0].raw_canonical == "vertical orbit"
        assert vocab.concepts[0].raw_forms == ("vert. orbit", "Vert-Orbit")
        assert vocab.concepts[1].raw_canonical == "beam position monitor"
        assert vocab.concepts[1].raw_forms == ("BPM", "(BPM)")

    def test_raw_spellings_do_not_affect_equality(self) -> None:
        """Concepts that differ only in raw spellings compare equal."""
        bare = Concept(canonical="vert orbit", kind="shorthand", forms=("vo",))
        spelled = Concept(
            canonical="vert orbit",
            kind="shorthand",
            forms=("vo",),
            raw_canonical="Vert. Orbit",
            raw_forms=("V.O.",),
        )
        assert bare == spelled
        assert bare.raw_canonical == ""
        assert bare.raw_forms == ()


class TestEdgePunctuationWarnings:
    """Warnings for raw forms whose edge punctuation changes how they match."""

    EDGE_WARNING = (
        'form "(BPM)" (concept "beam position monitor") is matched as "bpm" '
        "— edge punctuation (, . : ; ! ? ( ) [ ] \" ') is ignored in queries and entries"
    )
    INTERIOR_WARNING = (
        'form "vert. orbit" (concept "vertical orbit"): its interior punctuation is '
        "required — only text that breaks the same way matches"
    )

    def test_edge_and_interior_warnings_exact(self, tmp_path: Path) -> None:
        """Outer-only stripping and an interior break each get their own warning."""
        vocab, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, PUNCTUATED_VOCABULARY))

        assert errors == []
        assert vocab is not None
        assert warnings == [self.INTERIOR_WARNING, self.EDGE_WARNING]

    def test_edge_warning_interpolates_the_edge_character_set(self) -> None:
        """The edge warning names exactly the characters the tokenizer strips."""
        from osprey.services.ariel_search.vocabulary.model import _EDGE_TEXT

        assert f"({_EDGE_TEXT})" in self.EDGE_WARNING

    def test_hyphen_and_interior_slash_forms_have_no_edge_warning(self, tmp_path: Path) -> None:
        """``-`` splits like whitespace and ``t/s`` keeps its ``/``: neither warns."""
        text = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
      - Trouble-Shoot now
"""
        _, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == []

    def test_trailing_period_form_is_an_edge_warning(self, tmp_path: Path) -> None:
        """A trailing period alone is outer-edge stripping, not an interior break."""
        text = """
concepts:
  - canonical: vertical orbit
    kind: shorthand
    forms:
      - vert orbit.
"""
        _, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == [
            'form "vert orbit." (concept "vertical orbit") is matched as "vert orbit" '
            "— edge punctuation (, . : ; ! ? ( ) [ ] \" ') is ignored in queries and entries"
        ]

    def test_interior_break_wins_over_outer_edges(self, tmp_path: Path) -> None:
        """A form with both interior and outer edges gets only the interior warning."""
        text = """
concepts:
  - canonical: vertical orbit
    kind: shorthand
    forms:
      - "(vert. orbit)"
"""
        _, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == [
            'form "(vert. orbit)" (concept "vertical orbit"): its interior punctuation is '
            "required — only text that breaks the same way matches"
        ]

    def test_spaced_edge_only_token_is_an_interior_break(self, tmp_path: Path) -> None:
        """A dropped edge-only token between two words still breaks the form."""
        text = """
concepts:
  - canonical: vertical orbit
    kind: shorthand
    forms:
      - "vert : orbit"
"""
        _, errors, warnings = load_vocabulary(write_vocabulary(tmp_path, text))

        assert errors == []
        assert warnings == [
            'form "vert : orbit" (concept "vertical orbit"): its interior punctuation is '
            "required — only text that breaks the same way matches"
        ]
