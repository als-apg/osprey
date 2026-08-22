r"""Unit tests for ARIEL keyword-search pattern helpers.

Every example named in the feature proposal's "Patterns" success criteria is
pinned here: ``t/s i/o``, ``t/s /SR0[1-4]C___BPM\d+/``, ``/regex/?``, ``/BPMs?/``,
``flow rate 5 / 6 / 7``, ``//``, ``5 * 3``, ``SR0?C___BPM*`` and ``SR01C___BPM?``.
"""

from __future__ import annotations

import pytest

from osprey.services.ariel_search.search.patterns import (
    EMPTY_PATTERN_NOTICE,
    MIN_LITERAL_RUN,
    TOO_GENERIC_REASON,
    Accepted,
    Rejected,
    accept_pattern,
    extract_pattern_spans,
    glob_to_are,
    is_glob,
    longest_literal_run,
    strip_trailing_question,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# strip_trailing_question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("did the BPM trip?", "did the BPM trip"),
        ("did the BPM trip?  ", "did the BPM trip"),
        (r"/SR0[1-4]C___BPM\d+/?", r"/SR0[1-4]C___BPM\d+/"),
        ("/BPMs?/", "/BPMs?/"),
        ("ts bpm", "ts bpm"),
        ("", ""),
        ("?", ""),
    ],
)
def test_strip_trailing_question(query: str, expected: str) -> None:
    assert strip_trailing_question(query) == expected


def test_strip_trailing_question_removes_at_most_one() -> None:
    assert strip_trailing_question("what??") == "what?"


# ---------------------------------------------------------------------------
# extract_pattern_spans
# ---------------------------------------------------------------------------


def test_slash_forms_do_not_open_or_close_a_span() -> None:
    """``t/s i/o`` must yield no spans -- the criterion's slash-form collision."""
    result = extract_pattern_spans("t/s i/o")

    assert result.spans == ()
    assert result.notices == ()
    assert result.residual_text == "t/s i/o"


def test_span_beside_a_slash_form() -> None:
    result = extract_pattern_spans(r"t/s /SR0[1-4]C___BPM\d+/")

    assert result.residual_text == "t/s"
    assert len(result.spans) == 1
    assert result.spans[0].body == r"SR0[1-4]C___BPM\d+"
    assert result.spans[0].original == r"/SR0[1-4]C___BPM\d+/"
    assert result.notices == ()


def test_span_at_start_of_query() -> None:
    result = extract_pattern_spans(r"/SR0[1-4]C___BPM\d+/ trip")

    assert result.residual_text == "trip"
    assert [span.body for span in result.spans] == [r"SR0[1-4]C___BPM\d+"]


def test_span_in_the_middle_leaves_clean_residual() -> None:
    result = extract_pattern_spans("alpha /SR01C___BPM/ omega")

    assert result.residual_text == "alpha omega"
    assert [span.body for span in result.spans] == ["SR01C___BPM"]


def test_span_may_contain_spaces() -> None:
    result = extract_pattern_spans("/beam loss/")

    assert [span.body for span in result.spans] == ["beam loss"]
    assert result.residual_text == ""


def test_span_may_contain_escaped_slashes() -> None:
    result = extract_pattern_spans(r"/a\/b/")

    assert [span.body for span in result.spans] == [r"a\/b"]
    assert result.residual_text == ""


def test_multiple_spans_in_order() -> None:
    result = extract_pattern_spans("/alpha/ mid /omega/")

    assert [span.body for span in result.spans] == ["alpha", "omega"]
    assert result.residual_text == "mid"


def test_empty_pattern_is_dropped_with_a_notice() -> None:
    result = extract_pattern_spans("// bpm")

    assert result.spans == ()
    assert result.notices == (EMPTY_PATTERN_NOTICE,)
    assert result.residual_text == "bpm"


def test_whitespace_only_body_is_also_an_empty_pattern() -> None:
    result = extract_pattern_spans("/ /")

    assert result.spans == ()
    assert result.notices == (EMPTY_PATTERN_NOTICE,)


def test_flow_rate_slashes_are_not_a_pattern_after_acceptance() -> None:
    """``flow rate 5 / 6 / 7`` must keep ``6`` in the FTS leg.

    The span regex does match ``/ 6 /``, but its body has no 3-character literal
    run, so acceptance rejects it and the caller re-inserts the original text.
    """
    result = extract_pattern_spans("flow rate 5 / 6 / 7")

    assert len(result.spans) == 1
    assert result.spans[0].body == " 6 "
    decision = accept_pattern(result.spans[0].body)
    assert isinstance(decision, Rejected)
    assert "6" in result.spans[0].original


def test_no_spans_leaves_text_untouched() -> None:
    result = extract_pattern_spans("BPM (SR01) C++")

    assert result.spans == ()
    assert result.notices == ()
    assert result.residual_text == "BPM (SR01) C++"


def test_trailing_question_strip_then_extract() -> None:
    """The documented order: strip one ``?``, then extract spans."""
    stripped = strip_trailing_question(r"/SR0[1-4]C___BPM\d+/?")
    result = extract_pattern_spans(stripped)

    assert [span.body for span in result.spans] == [r"SR0[1-4]C___BPM\d+"]


def test_internal_question_mark_survives() -> None:
    stripped = strip_trailing_question("/BPMs?/")
    result = extract_pattern_spans(stripped)

    assert [span.body for span in result.spans] == ["BPMs?"]
    assert isinstance(accept_pattern("BPMs?"), Accepted)


# ---------------------------------------------------------------------------
# is_glob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("SR01C___BPM*", True),
        ("SR0?C___BPM*", True),
        ("*", True),
        ("SR01C___BPM?", False),
        ("BPM", False),
        ("C++", False),
        ("5", False),
        ("", False),
    ],
)
def test_is_glob(token: str, expected: bool) -> None:
    assert is_glob(token) is expected


# ---------------------------------------------------------------------------
# glob_to_are
# ---------------------------------------------------------------------------


def test_glob_to_are_trailing_star_omits_word_end() -> None:
    assert glob_to_are("SR01C___BPM*") == r"\mSR01C___BPM\S*"


def test_glob_to_are_internal_question_mark() -> None:
    assert glob_to_are("SR0?C___BPM*") == r"\mSR0\SC___BPM\S*"


def test_glob_to_are_without_trailing_star_wraps_word_end() -> None:
    assert glob_to_are("BPM*3") == r"\mBPM\S*3\M"


def test_glob_to_are_escapes_metacharacters() -> None:
    assert glob_to_are("a.b+c(d)e*") == r"\ma\.b\+c\(d\)e\S*"


def test_glob_to_are_leaves_underscore_literal() -> None:
    """``\\_`` is not a defined ARE escape -- underscores must stay bare."""
    assert "\\_" not in glob_to_are("SR01C___BPM*")


def test_glob_to_are_bare_star() -> None:
    assert glob_to_are("*") == r"\m\S*"


# ---------------------------------------------------------------------------
# longest_literal_run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (r"\mSR01C___BPM\S*", 11),
        (r"\mSR0\SC___BPM\S*", 7),
        (r"SR0[1-4]C___BPM\d+", 7),
        ("ab", 2),
        ("BPMs?", 3),
        ("beam loss", 4),
        (" 6 ", 1),
        (r"\m\S*", 0),
        ("", 0),
        (r"a\.bc", 4),
        ("(abc|de)", 3),
        (r"\d{4,}", 0),
        ("^bpm$", 3),
    ],
)
def test_longest_literal_run(pattern: str, expected: int) -> None:
    assert longest_literal_run(pattern) == expected


# ---------------------------------------------------------------------------
# accept_pattern
# ---------------------------------------------------------------------------


def test_min_literal_run_is_three() -> None:
    assert MIN_LITERAL_RUN == 3


def test_accept_pattern_accepts_regex_body() -> None:
    decision = accept_pattern(r"SR0[1-4]C___BPM\d+")

    assert isinstance(decision, Accepted)
    assert decision.are == r"SR0[1-4]C___BPM\d+"


def test_accept_pattern_accepts_space_bearing_body() -> None:
    assert accept_pattern("beam loss") == Accepted(are="beam loss")


def test_accept_pattern_rejects_two_literal_body() -> None:
    decision = accept_pattern("ab")

    assert isinstance(decision, Rejected)
    assert decision.reason == TOO_GENERIC_REASON
    assert "3 consecutive literal characters" in decision.reason


def test_accept_pattern_accepts_glob_are() -> None:
    assert accept_pattern(glob_to_are("SR01C___BPM*")) == Accepted(are=r"\mSR01C___BPM\S*")


def test_accept_pattern_accepts_question_glob_are() -> None:
    assert isinstance(accept_pattern(glob_to_are("SR0?C___BPM*")), Accepted)


def test_accept_pattern_rejects_bare_star_glob() -> None:
    """``5 * 3`` and ``... SELECT * FROM ...``: the lone ``*`` never survives."""
    assert is_glob("*") is True
    decision = accept_pattern(glob_to_are("*"))

    assert isinstance(decision, Rejected)
    assert decision.reason == TOO_GENERIC_REASON


@pytest.mark.parametrize("token", ["5", "3", "test", "UNION", "SELECT", "FROM", "users", "--"])
def test_injection_and_arithmetic_tokens_are_not_globs(token: str) -> None:
    assert is_glob(token) is False


def test_question_only_token_is_searched_as_text() -> None:
    """``SR01C___BPM?`` produces zero ``~*`` clauses."""
    assert is_glob("SR01C___BPM?") is False
