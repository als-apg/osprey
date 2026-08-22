r"""Tests for ``parse_keyword_query`` (ARIEL keyword whole-query parser).

Every example pinned by the proposal's "Patterns" success criteria that is
observable at the parser level is asserted here: glob and ``/.../`` span
acceptance, the trailing-``?`` ordering rule, slash-form collisions (``t/s``),
the 3-literal-character acceptance rule, truncation, and the
``patterns_enabled=False`` escape hatch.
"""

from __future__ import annotations

import pytest

from osprey.services.ariel_search.models import DiagnosticLevel
from osprey.services.ariel_search.search import ParsedKeywordQuery, parse_keyword_query
from osprey.services.ariel_search.search.patterns import (
    EMPTY_PATTERN_NOTICE,
    TOO_GENERIC_REASON,
)


def _pattern_messages(parsed: ParsedKeywordQuery) -> list[str]:
    return [d.message for d in parsed.diagnostics if d.category == "pattern"]


class TestGlobTokens:
    """Glob tokens (``*``) become anchored ARE pattern spans."""

    def test_trailing_star_glob_accepted(self):
        parsed = parse_keyword_query("SR01C___BPM*")

        assert len(parsed.pattern_spans) == 1
        span = parsed.pattern_spans[0]
        assert span.body == r"\mSR01C___BPM\S*"
        assert span.source == "glob"
        assert span.original == "SR01C___BPM*"
        # The glob token no longer contributes to the FTS text.
        assert parsed.search_text == ""
        assert parsed.diagnostics == ()

    def test_question_mark_inside_glob_is_single_char_wildcard(self):
        parsed = parse_keyword_query("SR0?C___BPM*")

        assert len(parsed.pattern_spans) == 1
        assert parsed.pattern_spans[0].body == r"\mSR0\SC___BPM\S*"
        assert parsed.pattern_spans[0].source == "glob"
        assert parsed.search_text == ""

    def test_lone_question_mark_token_is_not_a_glob(self):
        parsed = parse_keyword_query("SR01C___BPM?")

        assert parsed.pattern_spans == ()
        # The trailing '?' is stripped before parsing.
        assert parsed.search_text == "SR01C___BPM"

    def test_glob_beside_ordinary_text(self):
        parsed = parse_keyword_query("trip SR01C___BPM* today")

        assert len(parsed.pattern_spans) == 1
        assert parsed.search_text == "trip today"


class TestRegexSpans:
    """``/.../`` spans are token-anchored and carry their body verbatim."""

    def test_regex_span_accepted(self):
        parsed = parse_keyword_query(r"/SR0[1-4]C___BPM\d+/")

        assert len(parsed.pattern_spans) == 1
        span = parsed.pattern_spans[0]
        assert span.body == r"SR0[1-4]C___BPM\d+"
        assert span.source == "regex"
        assert span.original == r"/SR0[1-4]C___BPM\d+/"
        assert parsed.search_text == ""

    def test_regex_span_may_contain_spaces(self):
        parsed = parse_keyword_query("/beam loss/")

        assert len(parsed.pattern_spans) == 1
        assert parsed.pattern_spans[0].body == "beam loss"
        assert parsed.pattern_spans[0].source == "regex"

    def test_trailing_question_stripped_before_span_extraction(self):
        parsed = parse_keyword_query(r"/SR0[1-4]C___BPM\d+/?")

        assert len(parsed.pattern_spans) == 1
        assert parsed.pattern_spans[0].body == r"SR0[1-4]C___BPM\d+"

    def test_internal_question_mark_survives(self):
        parsed = parse_keyword_query("/BPMs?/")

        assert len(parsed.pattern_spans) == 1
        assert parsed.pattern_spans[0].body == "BPMs?"

    def test_empty_pattern_is_dropped_with_info(self):
        parsed = parse_keyword_query("//")

        assert parsed.pattern_spans == ()
        assert parsed.search_text == ""
        assert _pattern_messages(parsed) == [EMPTY_PATTERN_NOTICE]
        assert parsed.diagnostics[0].level is DiagnosticLevel.INFO
        assert parsed.diagnostics[0].source == "keyword"


class TestNonPatterns:
    """Ordinary operator prose never becomes a pattern."""

    @pytest.mark.parametrize(
        "query",
        [
            "BPM (SR01)",
            "C++",
            "did the BPM trip?",
        ],
    )
    def test_zero_pattern_spans(self, query):
        parsed = parse_keyword_query(query)

        assert parsed.pattern_spans == ()

    def test_trailing_question_is_stripped_from_prose(self):
        parsed = parse_keyword_query("did the BPM trip?")

        assert parsed.search_text == "did the BPM trip"


class TestSlashFormCollision:
    """A ``/`` inside a shorthand form can neither open nor close a span."""

    def test_shorthand_slashes_are_not_patterns(self):
        parsed = parse_keyword_query("t/s i/o")

        assert parsed.pattern_spans == ()
        assert parsed.search_text == "t/s i/o"
        assert parsed.diagnostics == ()

    def test_shorthand_beside_a_real_span(self):
        parsed = parse_keyword_query(r"t/s /SR0[1-4]C___BPM\d+/")

        assert len(parsed.pattern_spans) == 1
        assert parsed.pattern_spans[0].body == r"SR0[1-4]C___BPM\d+"
        assert parsed.search_text == "t/s"


class TestLiteralRunRule:
    """Patterns below the 3-literal-character floor are searched as text."""

    @pytest.mark.parametrize(
        "query",
        [
            "test UNION SELECT * FROM users --",
            "5 * 3",
        ],
    )
    def test_bare_star_is_never_a_pattern(self, query):
        parsed = parse_keyword_query(query)

        assert parsed.pattern_spans == ()
        # The rejected glob token stays in the FTS text.
        assert "*" in parsed.search_text.split()
        assert _pattern_messages(parsed) == [f'"*": {TOO_GENERIC_REASON}']

    def test_too_generic_span_is_reinserted_as_text(self):
        parsed = parse_keyword_query("/ab/")

        assert parsed.pattern_spans == ()
        assert parsed.search_text == "ab"
        assert _pattern_messages(parsed) == [f'"/ab/": {TOO_GENERIC_REASON}']
        assert parsed.diagnostics[0].level is DiagnosticLevel.INFO
        assert parsed.diagnostics[0].category == "pattern"

    def test_arithmetic_slashes_keep_their_operand(self):
        parsed = parse_keyword_query("flow rate 5 / 6 / 7")

        assert parsed.pattern_spans == ()
        assert "6" in parsed.search_text.split()

    def test_long_literal_glob_is_still_accepted(self):
        parsed = parse_keyword_query("SR01C___BPM*")

        assert len(parsed.pattern_spans) == 1


class TestFieldFiltersAndPhrases:
    """The existing ``parse_query`` still owns fields and quoted phrases."""

    def test_field_filter_extracted(self):
        parsed = parse_keyword_query("author:ts bpm")

        assert parsed.field_filters == {"author": "ts"}
        assert parsed.search_text == "bpm"
        assert parsed.phrases == ()
        assert parsed.pattern_spans == ()

    def test_quoted_phrase_beside_a_glob(self):
        parsed = parse_keyword_query('"beam loss" SR01C___BPM* today')

        assert parsed.phrases == ("beam loss",)
        assert len(parsed.pattern_spans) == 1
        assert parsed.pattern_spans[0].source == "glob"
        assert parsed.search_text == "today"

    def test_empty_query(self):
        parsed = parse_keyword_query("")

        assert parsed.search_text == ""
        assert parsed.field_filters == {}
        assert parsed.phrases == ()
        assert parsed.pattern_spans == ()
        assert parsed.diagnostics == ()


class TestTruncation:
    """Truncation happens first, on the raw query."""

    def test_span_straddling_the_limit_is_not_a_pattern(self):
        # A '/regex/' whose closing slash sits past MAX_QUERY_LENGTH.
        prefix = "beam " * 199  # 995 chars
        head = prefix + r"/SR0[1-4]C___BPM\d+/ "  # 1016 chars
        query = head + "tail " * 36 + "abcd"
        assert len(query) == 1200
        assert query.index("/SR0") < 1000 < query.index(r"\d+/")

        parsed = parse_keyword_query(query)

        assert parsed.pattern_spans == ()
        assert _pattern_messages(parsed) == []
        warnings = [d for d in parsed.diagnostics if d.level is DiagnosticLevel.WARNING]
        assert len(warnings) == 1
        assert warnings[0].category == "truncation"
        assert warnings[0].source == "keyword"
        assert warnings[0].message == "Query truncated from 1200 to 1000 characters"

    def test_short_query_has_no_truncation_diagnostic(self):
        parsed = parse_keyword_query("beam loss")

        assert parsed.diagnostics == ()


class TestPatternsDisabled:
    """``patterns_enabled=False`` turns the whole pattern machinery off."""

    def test_glob_is_ordinary_text(self):
        parsed = parse_keyword_query("SR01C___BPM*", patterns_enabled=False)

        assert parsed.pattern_spans == ()
        assert parsed.search_text == "SR01C___BPM*"
        assert parsed.diagnostics == ()

    def test_span_is_ordinary_text(self):
        parsed = parse_keyword_query(r"/SR0[1-4]C___BPM\d+/", patterns_enabled=False)

        assert parsed.pattern_spans == ()
        assert parsed.search_text == r"/SR0[1-4]C___BPM\d+/"

    def test_trailing_question_left_to_parse_query(self):
        parsed = parse_keyword_query("did the BPM trip?", patterns_enabled=False)

        assert parsed.search_text == "did the BPM trip?"

    def test_truncation_still_applies(self):
        parsed = parse_keyword_query("a" * 1200, patterns_enabled=False)

        assert len(parsed.search_text) == 1000
        assert parsed.diagnostics[0].category == "truncation"
