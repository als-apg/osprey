"""Tests for entry-text expansion against the unit-test vocabulary."""

from __future__ import annotations

import pytest

from osprey.services.ariel_search.vocabulary.entry import (
    CAUSE_PHRASES,
    classify_build_keys,
)
from osprey.services.ariel_search.vocabulary.expand import (
    EntryExpansion,
    Insertion,
    expand_entry_text,
    match_tokens,
)
from osprey.services.ariel_search.vocabulary.loader import load_vocabulary
from tests.services.ariel_search.conftest import UNIT_VOCABULARY

AMBIGUOUS_EXTRA = """  - canonical: timing system
    kind: acronym
    forms: [ts]
"""


@pytest.fixture
def vocabulary(unit_vocabulary_file):
    loaded, errors, _warnings = load_vocabulary(unit_vocabulary_file)
    assert errors == []
    assert loaded is not None
    return loaded


@pytest.fixture
def ambiguous_vocabulary(tmp_path):
    path = tmp_path / "ambiguous.yml"
    path.write_text(UNIT_VOCABULARY + AMBIGUOUS_EXTRA, encoding="utf-8")
    loaded, errors, _warnings = load_vocabulary(path)
    assert errors == []
    assert loaded is not None
    return loaded


def expand(text, vocabulary, *, acronym=True, shorthand=False, **kwargs):
    return expand_entry_text(
        text,
        vocabulary,
        canonical_to_acronym=acronym,
        canonical_to_shorthand=shorthand,
        **kwargs,
    )


class TestExpandEntryTextInsertions:
    """expand_entry_text inserts each span's alternatives after its word."""

    def test_expand_entry_text_sentence_example(self, vocabulary):
        text = "Reset the BPM, then RF trip (again)."
        result = expand(text, vocabulary)
        assert result.text == (
            "Reset the BPM beam position monitor, then RF radio frequency trip (again)."
        )
        assert result.insertions == (
            Insertion(anchor=13, text="beam position monitor", concepts=("beam position monitor",)),
            Insertion(anchor=22, text="radio frequency", concepts=("radio frequency",)),
        )
        assert text[: result.insertions[0].anchor].endswith("BPM")
        assert text[: result.insertions[1].anchor].endswith("RF")

    def test_expand_entry_text_insertion_text_has_no_separator(self, vocabulary):
        result = expand("BPM", vocabulary)
        assert result.insertions[0].text == "beam position monitor"
        assert result.text == "BPM beam position monitor"

    def test_expand_entry_text_hyphenated_word_anchors_after_word(self, vocabulary):
        result = expand("RF-trip", vocabulary)
        assert result.text == "RF-trip radio frequency"
        assert result.insertions == (
            Insertion(anchor=7, text="radio frequency", concepts=("radio frequency",)),
        )

    def test_expand_entry_text_insertions_at_one_word_concatenate(self, vocabulary):
        result = expand("RF-BPM,", vocabulary)
        assert result.text == "RF-BPM radio frequency beam position monitor,"
        assert [ins.anchor for ins in result.insertions] == [6, 6]
        assert [ins.concepts for ins in result.insertions] == [
            ("radio frequency",),
            ("beam position monitor",),
        ]

    def test_expand_entry_text_ambiguous_form_lists_every_concept(self, ambiguous_vocabulary):
        result = expand("ts again", ambiguous_vocabulary)
        assert result.text == "ts troubleshoot timing system again"
        assert result.insertions[0].concepts == ("troubleshoot", "timing system")

    def test_expand_entry_text_canonical_expands_under_gate(self, vocabulary):
        result = expand("radio frequency", vocabulary)
        assert result.text == "radio frequency rf"
        assert result.insertions[0].concepts == ("radio frequency",)

    def test_expand_entry_text_shorthand_canonical_gated_off_by_default(self, vocabulary):
        text = "quadrupole"
        assert expand(text, vocabulary).text is text
        assert expand(text, vocabulary, shorthand=True).text == "quadrupole quad"

    def test_expand_entry_text_slash_form_is_one_token(self, vocabulary):
        assert expand("t/s done", vocabulary).text == "t/s troubleshoot done"
        assert expand("check i/o", vocabulary).text == "check i/o input output"

    def test_expand_entry_text_interior_punctuated_form(self, vocabulary):
        assert expand("vert. orbit drift", vocabulary).text == ("vert. orbit vertical orbit drift")
        text = "vert orbit drift"
        assert expand(text, vocabulary).text is text

    def test_expand_entry_text_clause_break_blocks_span(self, vocabulary):
        for text in ("the beam. Position monitor", "beam . position monitor"):
            result = expand(text, vocabulary)
            assert result.text is text
            assert result.insertions == ()


class TestExpandEntryTextNoMatch:
    """expand_entry_text returns the input object untouched when nothing is added."""

    def test_expand_entry_text_no_match_returns_identical_object(self, vocabulary):
        text = "".join(["nothing ", "to see here"])
        result = expand(text, vocabulary)
        assert result == EntryExpansion(text=text, insertions=())
        assert result.text is text

    def test_expand_entry_text_empty_text(self, vocabulary):
        text = ""
        assert expand(text, vocabulary).text is text

    def test_expand_entry_text_all_deduped_returns_identical_object(self, vocabulary):
        text = "".join(["BPM beam ", "position monitor"])
        assert expand(text, vocabulary).text is text


class TestExpandEntryTextDedupe:
    """A span already spelled out next to its alternative gains nothing."""

    @pytest.mark.parametrize(
        "text",
        [
            "BPM beam position monitor",
            "beam position monitor (BPM)",
            "quad (quadrupole)",
            "the BPM (beam position monitor) failed",
            "bpms, beam position monitor",
        ],
    )
    def test_expand_entry_text_adjacent_member_drops_insertion(self, vocabulary, text):
        result = expand(text, vocabulary)
        assert result.insertions == ()
        assert result.text is text

    def test_expand_entry_text_dedupe_independent_of_acronym_gate(self, vocabulary):
        text = "beam position monitor (BPM)"
        result = expand(text, vocabulary, acronym=False)
        assert result.insertions == ()
        assert result.text is text

    def test_expand_entry_text_dedupe_ambiguous_form(self, ambiguous_vocabulary):
        text = "ts troubleshoot"
        result = expand(text, ambiguous_vocabulary)
        assert result.insertions == ()
        assert result.text is text

    def test_expand_entry_text_non_adjacent_member_keeps_insertion(self, vocabulary):
        result = expand("BPM failed; beam position monitor replaced", vocabulary, acronym=False)
        assert result.text == "BPM beam position monitor failed; beam position monitor replaced"

    def test_expand_entry_text_dedupe_drops_only_matching_span(self, vocabulary):
        result = expand("RF BPM beam position monitor", vocabulary)
        assert result.text == "RF radio frequency BPM beam position monitor"
        assert [ins.concepts for ins in result.insertions] == [("radio frequency",)]


class TestExpandEntryTextLimit:
    """``limit`` excludes spans and dedupe partners past the cut."""

    def test_expand_entry_text_limit_inside_token_adds_nothing(self, vocabulary):
        text = "check the RFQ"
        limit = text.index("RFQ") + 2
        result = expand(text, vocabulary, limit=limit)
        assert result.insertions == ()
        assert result.text is text

    def test_expand_entry_text_limit_at_period_keeps_span(self, vocabulary):
        text = "the BPM."
        result = expand(text, vocabulary, limit=text.index("."))
        assert result.text == "the BPM beam position monitor."

    def test_expand_entry_text_limit_after_form_ignores_cut_partner(self, vocabulary):
        text = "BPM beam position monitor"
        result = expand(text, vocabulary, limit=3)
        assert result.insertions == (
            Insertion(anchor=3, text="beam position monitor", concepts=("beam position monitor",)),
        )

    def test_expand_entry_text_limit_excludes_later_spans(self, vocabulary):
        text = "BPM then RF"
        result = expand(text, vocabulary, limit=text.index("then"))
        assert [ins.concepts for ins in result.insertions] == [("beam position monitor",)]


class TestExpandEntryTextTokens:
    """A caller-supplied token list is used instead of re-tokenizing."""

    def test_expand_entry_text_precomputed_tokens_match(self, vocabulary):
        text = "Reset the BPM, then RF trip (again)."
        tokens = match_tokens(text)
        assert expand(text, vocabulary, tokens=tokens) == expand(text, vocabulary)

    def test_expand_entry_text_uses_supplied_tokens(self, vocabulary):
        text = "BPM and RF"
        tokens = match_tokens(text)[:1]
        result = expand(text, vocabulary, tokens=tokens)
        assert [ins.concepts for ins in result.insertions] == [("beam position monitor",)]


# ---------------------------------------------------------------------------
# EntryExpander: fingerprint, key, expand, call
# ---------------------------------------------------------------------------


def _expander_for(tmp_path, yaml_text, *, acronym=True, shorthand=False, name="v.yml"):
    from osprey.services.ariel_search.vocabulary import EntryExpander

    path = tmp_path / name
    path.write_text(yaml_text, encoding="utf-8")
    loaded, errors, _warnings = load_vocabulary(
        path, canonical_to_acronym=acronym, canonical_to_shorthand=shorthand
    )
    assert errors == []
    assert loaded is not None
    return EntryExpander(loaded, canonical_to_acronym=acronym, canonical_to_shorthand=shorthand)


def _edit(old, new):
    assert old in UNIT_VOCABULARY
    return UNIT_VOCABULARY.replace(old, new, 1)


class TestEntryExpanderFingerprintStable:
    """The fingerprint ignores everything that does not shape expanded text."""

    def test_fingerprint_is_sha256_hex(self, tmp_path):
        fp = _expander_for(tmp_path, UNIT_VOCABULARY).fingerprint
        assert len(fp) == 64
        int(fp, 16)

    def test_fingerprint_deterministic_across_loads(self, tmp_path):
        a = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        b = _expander_for(tmp_path, UNIT_VOCABULARY, name="b.yml")
        assert a.fingerprint == b.fingerprint

    def test_fingerprint_ignores_yaml_comments(self, tmp_path):
        commented = "# facility vocabulary\n" + _edit(
            "    forms: [rf]\n", "    forms: [rf]  # radio\n"
        )
        base = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        other = _expander_for(tmp_path, commented, name="b.yml")
        assert base.fingerprint == other.fingerprint

    def test_fingerprint_ignores_yaml_layout(self, tmp_path):
        block = _edit("    forms: [bpm, bpms]\n", "    forms:\n      - bpm\n      - bpms\n")
        flow = _edit(
            "  - canonical: quadrupole\n    kind: shorthand\n    forms: [quad]\n",
            '  - {canonical: "quadrupole", kind: shorthand, forms: ["quad"]}\n',
        )
        base = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        assert _expander_for(tmp_path, block, name="b.yml").fingerprint == base.fingerprint
        assert _expander_for(tmp_path, flow, name="c.yml").fingerprint == base.fingerprint

    def test_fingerprint_ignores_inner_whitespace_runs(self, tmp_path):
        spaced = _edit("canonical: beam position monitor", 'canonical: "beam   position  monitor"')
        spaced = spaced.replace("forms: [vert. orbit]", 'forms: ["vert.   orbit"]', 1)
        base = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        other = _expander_for(tmp_path, spaced, name="b.yml")
        assert base.fingerprint == other.fingerprint


class TestEntryExpanderFingerprintChanges:
    """Any edit that can change expanded text changes the fingerprint."""

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            pytest.param(
                "  - canonical: quadrupole\n    kind: shorthand\n    forms: [quad]\n",
                "",
                id="concept-removed",
            ),
            pytest.param(
                "concepts:\n",
                "concepts:\n  - canonical: timing system\n    kind: acronym\n    forms: [tsys]\n",
                id="concept-added",
            ),
            pytest.param(
                "canonical: radio frequency\n    kind: acronym",
                "canonical: radio frequency\n    kind: synonym",
                id="kind",
            ),
            pytest.param("forms: [bpm, bpms]", "forms: [bpm, bpms, bpm's]", id="raw-form-added"),
            pytest.param("forms: [bpm, bpms]", "forms: [bpm]", id="raw-form-removed"),
            pytest.param("forms: [bpm, bpms]", "forms: [bpm, BPMs]", id="raw-form-case"),
            pytest.param("forms: [vert. orbit]", "forms: [vert orbit]", id="form-punctuation"),
            pytest.param("forms: [vert. orbit]", "forms: [vert.orbit]", id="form-punct-space"),
            pytest.param(
                "canonical: quadrupole", "canonical: quadrupole.", id="canonical-punctuation"
            ),
            pytest.param("canonical: quadrupole", "canonical: Quadrupole", id="canonical-case"),
            pytest.param("canonical: input output", "canonical: input/output", id="canonical-raw"),
        ],
    )
    def test_fingerprint_changes_with_content(self, tmp_path, old, new):
        base = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        other = _expander_for(tmp_path, _edit(old, new), name="b.yml")
        assert base.fingerprint != other.fingerprint
        assert base.key != other.key

    def test_fingerprint_changes_with_concept_order(self, tmp_path):
        rf = "  - canonical: radio frequency\n    kind: acronym\n    forms: [rf]\n"
        reordered = _edit(rf, "").replace("concepts:\n", "concepts:\n" + rf, 1)
        base = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        other = _expander_for(tmp_path, reordered, name="b.yml")
        assert base.fingerprint != other.fingerprint

    @pytest.mark.parametrize(
        ("acronym", "shorthand"), [(False, False), (True, True), (False, True)]
    )
    def test_fingerprint_changes_with_gates(self, tmp_path, acronym, shorthand):
        base = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        other = _expander_for(
            tmp_path, UNIT_VOCABULARY, acronym=acronym, shorthand=shorthand, name="b.yml"
        )
        assert base.fingerprint != other.fingerprint

    def test_fingerprint_distinguishes_the_two_gates(self, tmp_path):
        only_acronym = _expander_for(
            tmp_path, UNIT_VOCABULARY, acronym=True, shorthand=False, name="a.yml"
        )
        only_shorthand = _expander_for(
            tmp_path, UNIT_VOCABULARY, acronym=False, shorthand=True, name="b.yml"
        )
        assert only_acronym.fingerprint != only_shorthand.fingerprint

    def test_fingerprint_and_key_change_with_expansion_rule(self, tmp_path, monkeypatch):
        from osprey.services.ariel_search.vocabulary import entry

        base = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        base_fingerprint, base_key = base.fingerprint, base.key
        monkeypatch.setattr(entry, "EXPANSION_RULE", entry.EXPANSION_RULE + 1)
        bumped = _expander_for(tmp_path, UNIT_VOCABULARY, name="b.yml")
        assert base_fingerprint != bumped.fingerprint
        assert base_key != bumped.key
        assert bumped.key.startswith(f"e{entry.EXPANSION_RULE}:")


class TestEntryExpanderKey:
    """``key`` is ``e<rule>:<first 16 hex digits of the fingerprint>``."""

    def test_key_shape(self, tmp_path):
        from osprey.services.ariel_search.vocabulary import EXPANSION_RULE

        expander = _expander_for(tmp_path, UNIT_VOCABULARY)
        assert expander.key == f"e{EXPANSION_RULE}:{expander.fingerprint[:16]}"
        assert EXPANSION_RULE == 1

    def test_key_stable_across_layout(self, tmp_path):
        block = _edit("    forms: [bpm, bpms]\n", "    forms:\n      - bpm\n      - bpms\n")
        a = _expander_for(tmp_path, UNIT_VOCABULARY, name="a.yml")
        b = _expander_for(tmp_path, block, name="b.yml")
        assert a.key == b.key


class TestEntryExpanderExpand:
    """expand and __call__ delegate to expand_entry_text with the bound gates."""

    @pytest.mark.parametrize(("acronym", "shorthand"), [(True, False), (False, True)])
    def test_expand_matches_expand_entry_text(self, tmp_path, acronym, shorthand):
        expander = _expander_for(tmp_path, UNIT_VOCABULARY, acronym=acronym, shorthand=shorthand)
        text = "Reset the BPM, troubleshoot the quadrupole and the radio frequency."
        expected = expand(text, expander.vocabulary, acronym=acronym, shorthand=shorthand)
        assert expander.expand(text) == expected
        assert expander(text) == expected.text

    def test_expand_passes_limit_and_tokens(self, tmp_path):
        expander = _expander_for(tmp_path, UNIT_VOCABULARY)
        text = "BPM and RF and BPM"
        tokens = match_tokens(text)
        result = expander.expand(text, limit=4, tokens=tokens)
        assert isinstance(result, EntryExpansion)
        assert result == expand(text, expander.vocabulary, limit=4, tokens=tokens)

    def test_call_without_matches_returns_text(self, tmp_path):
        expander = _expander_for(tmp_path, UNIT_VOCABULARY)
        assert expander("nothing to see here") == "nothing to see here"


FP_A = "0123456789abcdef"
FP_B = "fedcba9876543210"


class TestClassifyBuildKeys:
    """classify_build_keys sums stale counts by the first differing component."""

    def test_classify_three_keys_in_one_table(self):
        counts = {"r1": 5, f"r1:e1:{FP_A}": 3, f"r1:e1:{FP_B}": 7}
        assert classify_build_keys(counts, f"r1:e1:{FP_B}") == {
            "unexpanded": 5,
            "vocabulary": 3,
        }

    def test_classify_running_key_only_is_not_stale(self):
        assert classify_build_keys({"r1": 4}, "r1") == {}
        assert classify_build_keys({f"r1:e1:{FP_A}": 4}, f"r1:e1:{FP_A}") == {}
        assert classify_build_keys({}, "r1") == {}

    @pytest.mark.parametrize(
        ("stored", "running", "code"),
        [
            ("r2", "r1", "input_rule"),
            ("r1", "r2", "input_rule"),
            (f"r2:e1:{FP_A}", f"r1:e1:{FP_A}", "input_rule"),
            (f"r1:e1:{FP_A}", f"r2:e1:{FP_A}", "input_rule"),
            ("r1", f"r1:e1:{FP_A}", "unexpanded"),
            (f"r1:e1:{FP_A}", "r1", "expanded"),
            (f"r1:e2:{FP_A}", f"r1:e1:{FP_A}", "expansion_rule"),
            (f"r1:e1:{FP_A}", f"r1:e2:{FP_A}", "expansion_rule"),
            (f"r1:e1:{FP_B}", f"r1:e1:{FP_A}", "vocabulary"),
            (f"r1:e1:{FP_A}", f"r1:e1:{FP_B}", "vocabulary"),
        ],
    )
    def test_classify_single_component_difference(self, stored, running, code):
        assert classify_build_keys({stored: 6}, running) == {code: 6}

    @pytest.mark.parametrize(
        ("stored", "running", "code"),
        [
            # The input rule outranks every expansion difference.
            (f"r2:e2:{FP_B}", f"r1:e1:{FP_A}", "input_rule"),
            ("r2", f"r1:e1:{FP_A}", "input_rule"),
            # Expansion presence outranks the rule and fingerprint.
            ("r1", f"r1:e2:{FP_A}", "unexpanded"),
            # The expansion rule outranks the fingerprint.
            (f"r1:e2:{FP_B}", f"r1:e1:{FP_A}", "expansion_rule"),
        ],
    )
    def test_classify_precedence(self, stored, running, code):
        assert classify_build_keys({stored: 2}, running) == {code: 2}

    @pytest.mark.parametrize(
        "stored",
        [
            "",
            "x1",
            "r",
            "r0",
            "r01",
            "rx",
            "r1:e1",
            f"r1:e1:{FP_A}:extra",
            "r1:ex:abc",
            "r1:e1:",
            "r1:e1:XYZ",
            "R1",
            " r1",
        ],
    )
    def test_classify_unparseable_key_is_unrecognized(self, stored):
        assert classify_build_keys({stored: 9}, "r1") == {"unrecognized": 9}

    def test_classify_sums_counts_per_code(self):
        counts = {
            f"r1:e1:{FP_A}": 3,
            f"r1:e1:{'1' * 16}": 4,
            "bogus": 1,
            "also-bogus": 2,
            f"r1:e1:{FP_B}": 100,
        }
        assert classify_build_keys(counts, f"r1:e1:{FP_B}") == {
            "vocabulary": 7,
            "unrecognized": 3,
        }

    def test_classify_running_key_from_expander(self, tmp_path):
        expander = _expander_for(tmp_path, UNIT_VOCABULARY)
        running = f"r1:{expander.key}"
        counts = {running: 10, "r1": 2, f"r1:e1:{FP_A}": 1}
        assert classify_build_keys(counts, running) == {"unexpanded": 2, "vocabulary": 1}

    def test_classify_rejects_unparseable_running_key(self):
        with pytest.raises(ValueError, match="unrecognized running build key"):
            classify_build_keys({"r1": 1}, "nope")

    def test_classify_cause_phrases_cover_every_code(self):
        assert CAUSE_PHRASES == {
            "input_rule": "with a different input rule",
            "unexpanded": "without entry expansion",
            "expanded": "with entry expansion",
            "expansion_rule": "with a different expansion rule",
            "vocabulary": "with a different vocabulary",
            "unrecognized": "with an unrecognized build key",
        }
