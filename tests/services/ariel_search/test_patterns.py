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


@pytest.mark.parametrize(
    "text",
    [
        # ``t/s i/o`` must yield no spans -- the criterion's slash-form collision.
        pytest.param("t/s i/o", id="slash-forms"),
        pytest.param("BPM (SR01) C++", id="no-slashes"),
    ],
)
def test_text_without_a_span_is_left_untouched(text) -> None:
    result = extract_pattern_spans(text)

    assert result.spans == ()
    assert result.notices == ()
    assert result.residual_text == text


def test_span_beside_a_slash_form() -> None:
    result = extract_pattern_spans(r"t/s /SR0[1-4]C___BPM\d+/")

    assert result.residual_text == "t/s"
    assert len(result.spans) == 1
    assert result.spans[0].body == r"SR0[1-4]C___BPM\d+"
    assert result.spans[0].original == r"/SR0[1-4]C___BPM\d+/"
    assert result.notices == ()


@pytest.mark.parametrize(
    ("query", "residual", "body"),
    [
        pytest.param(r"/SR0[1-4]C___BPM\d+/ trip", "trip", r"SR0[1-4]C___BPM\d+", id="at-start"),
        pytest.param("alpha /SR01C___BPM/ omega", "alpha omega", "SR01C___BPM", id="in-the-middle"),
    ],
)
def test_a_span_leaves_a_clean_residual(query, residual, body) -> None:
    result = extract_pattern_spans(query)

    assert result.residual_text == residual
    assert [span.body for span in result.spans] == [body]


@pytest.mark.parametrize(
    ("query", "body"),
    [
        pytest.param("/beam loss/", "beam loss", id="spaces"),
        pytest.param(r"/a\/b/", r"a\/b", id="escaped-slashes"),
    ],
)
def test_span_may_contain_spaces_and_escaped_slashes(query, body) -> None:
    result = extract_pattern_spans(query)

    assert [span.body for span in result.spans] == [body]
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
        pytest.param("5", False, id="arithmetic-5"),
        pytest.param("3", False, id="arithmetic-3"),
        pytest.param("test", False, id="plain-word"),
        pytest.param("UNION", False, id="sql-union"),
        pytest.param("SELECT", False, id="sql-select"),
        pytest.param("FROM", False, id="sql-from"),
        pytest.param("users", False, id="sql-table-name"),
        pytest.param("--", False, id="sql-comment-marker"),
        # ``SR01C___BPM?`` produces zero ``~*`` clauses.
        pytest.param("SR01C___BPM?", False, id="question-only-token-is-searched-as-text"),
    ],
)
def test_is_glob(token: str, expected: bool) -> None:
    assert is_glob(token) is expected


# ---------------------------------------------------------------------------
# glob_to_are
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("glob", "are"),
    [
        pytest.param("SR01C___BPM*", r"\mSR01C___BPM\S*", id="trailing_star_omits_word_end"),
        pytest.param("SR0?C___BPM*", r"\mSR0\SC___BPM\S*", id="internal_question_mark"),
        pytest.param("BPM*3", r"\mBPM\S*3\M", id="without_trailing_star_wraps_word_end"),
        pytest.param("a.b+c(d)e*", r"\ma\.b\+c\(d\)e\S*", id="escapes_metacharacters"),
    ],
)
def test_glob_to_are(glob: str, are: str) -> None:
    assert glob_to_are(glob) == are


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
