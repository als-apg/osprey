r"""Pattern-token helpers for ARIEL keyword search.

Operators write control-system names with wildcards (``SR01C___BPM*``) or supply an
explicit regular expression between slashes (``/SR0[1-4]C___BPM\d+/``). This module
holds the pure, dependency-free functions that recognise those forms and turn them
into PostgreSQL Advanced Regular Expressions (ARE) suitable for a ``raw_text ~* %s``
predicate. Nothing here touches SQL, configuration or the database.

Processing order (owned by ``parse_keyword_query`` in ``keyword.py``, which calls
these helpers in sequence):

1. truncate to ``MAX_QUERY_LENGTH``
2. :func:`strip_trailing_question` -- remove at most one trailing ``?``
3. :func:`extract_pattern_spans` -- pull ``/.../`` spans out of the text
4. quoted phrases and whitespace tokens (the existing ``parse_query``)
5. :func:`is_glob` / :func:`glob_to_are` over the remaining search text
6. :func:`accept_pattern` on every candidate ARE (both forms)

**Ownership note.** :func:`extract_pattern_spans` returns :class:`RawSpan` objects
carrying the body *as written*, before acceptance. The downstream parser
(``parse_keyword_query``) is what converts an accepted candidate into the
``PatternSpan`` dataclass defined in ``search/base.py``; this module deliberately
does not import from ``base`` so it stays a leaf with no ARIEL dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "EMPTY_PATTERN_NOTICE",
    "MIN_LITERAL_RUN",
    "TOO_GENERIC_REASON",
    "Accepted",
    "PatternDecision",
    "PatternExtraction",
    "RawSpan",
    "Rejected",
    "accept_pattern",
    "extract_pattern_spans",
    "glob_to_are",
    "is_glob",
    "longest_literal_run",
    "strip_trailing_question",
]

# Token-anchored ``/.../`` span. The leading ``(?:(?<=\s)|^)`` and trailing
# ``(?=\s|$)`` mean a slash inside a shorthand form such as ``t/s`` or ``i/o`` can
# neither open nor close a span. Spaces and backslash-escaped slashes are allowed
# inside the body.
PATTERN_SPAN = re.compile(r"(?:(?<=\s)|^)/((?:[^/\\]|\\.)+)/(?=\s|$)")

# ``PATTERN_SPAN`` requires a non-empty body, so a bare ``//`` never matches it.
# It is recognised separately and dropped with an INFO notice -- the only case
# where query text disappears instead of falling back to ordinary FTS text.
EMPTY_PATTERN_SPAN = re.compile(r"(?:(?<=\s)|^)//(?=\s|$)")

#: Minimum number of consecutive literal characters for a pattern to be usable by
#: the ``pg_trgm`` GIN index on ``raw_text``.
MIN_LITERAL_RUN = 3

#: INFO text for a pattern that cannot be index-accelerated.
TOO_GENERIC_REASON = (
    "pattern needs at least 3 consecutive literal characters to be indexable "
    "— searched as text instead"
)

#: INFO text for ``//`` (or a whitespace-only body).
EMPTY_PATTERN_NOTICE = "empty pattern ignored"

# ARE metacharacters that must be backslash-escaped when a glob's literal
# character happens to be one of them. ``_`` is NOT included: it is an ordinary
# literal in ARE and ``\_`` is not a defined escape.
_ARE_METACHARS = frozenset("\\.^$|()[]{}+?*")

# Characters that are structural in a regular expression and therefore never count
# towards a literal run. Quantifiers are consumed separately.
_REGEX_METACHARS = frozenset("^$.|()[]{}*+?")


@dataclass(frozen=True, slots=True)
class RawSpan:
    """A ``/.../`` span lifted out of a query, before acceptance.

    Attributes:
        body: The text between the delimiting slashes, exactly as written.
        original: The full span including its slashes, for re-insertion as
            ordinary FTS text if the body is later rejected.
    """

    body: str
    original: str


@dataclass(frozen=True, slots=True)
class PatternExtraction:
    """Result of :func:`extract_pattern_spans`.

    Attributes:
        residual_text: The query with every recognised span removed and
            whitespace collapsed.
        spans: The extracted spans, in query order.
        notices: INFO diagnostic texts the caller should surface (currently only
            :data:`EMPTY_PATTERN_NOTICE`, one per dropped empty pattern).
    """

    residual_text: str
    spans: tuple[RawSpan, ...]
    notices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Accepted:
    """A pattern that may be executed as a ``raw_text ~* %s`` predicate.

    Attributes:
        are: The PostgreSQL Advanced Regular Expression to bind as a parameter.
    """

    are: str


@dataclass(frozen=True, slots=True)
class Rejected:
    """A pattern that must be searched as ordinary text instead.

    Attributes:
        reason: INFO diagnostic text explaining why (see
            :data:`TOO_GENERIC_REASON`).
    """

    reason: str


PatternDecision = Accepted | Rejected


def strip_trailing_question(query: str) -> str:
    r"""Strip trailing whitespace and at most one trailing ``?`` from a query.

    Runs before span extraction so ``/SR0[1-4]C___BPM\d+/?`` yields the body
    ``SR0[1-4]C___BPM\d+`` while ``/BPMs?/`` keeps its internal ``?``.

    Args:
        query: Raw user query.

    Returns:
        The query without its trailing whitespace and one trailing question mark.
    """
    trimmed = query.rstrip()
    if trimmed.endswith("?"):
        return trimmed[:-1]
    return trimmed


def extract_pattern_spans(query: str) -> PatternExtraction:
    """Extract token-anchored ``/.../`` spans from a query.

    Uses :data:`PATTERN_SPAN` (plus :data:`EMPTY_PATTERN_SPAN` for ``//``) with
    ``finditer`` and a single removal pass over the matched positions, so the two
    never disagree about what was removed.

    Args:
        query: Query text, already truncated and question-stripped.

    Returns:
        A :class:`PatternExtraction`. Empty or whitespace-only bodies produce a
        notice and no span; every other span's body is returned verbatim for the
        caller to run through :func:`accept_pattern`.
    """
    matches: list[tuple[int, int, str | None]] = [
        (m.start(), m.end(), m.group(1)) for m in PATTERN_SPAN.finditer(query)
    ]
    matches.extend((m.start(), m.end(), None) for m in EMPTY_PATTERN_SPAN.finditer(query))
    matches.sort(key=lambda item: item[0])

    spans: list[RawSpan] = []
    notices: list[str] = []
    pieces: list[str] = []
    cursor = 0

    for start, end, body in matches:
        if start < cursor:  # pragma: no cover - the two regexes cannot overlap
            continue
        pieces.append(query[cursor:start])
        cursor = end
        if body is None or not body.strip():
            notices.append(EMPTY_PATTERN_NOTICE)
        else:
            spans.append(RawSpan(body=body, original=query[start:end]))

    pieces.append(query[cursor:])
    residual_text = " ".join("".join(pieces).split())
    return PatternExtraction(
        residual_text=residual_text,
        spans=tuple(spans),
        notices=tuple(notices),
    )


def is_glob(token: str) -> bool:
    """Report whether a whitespace token should be treated as a glob.

    Only ``*`` promotes a token to a glob. A lone ``?`` is ordinary text --
    ``SR01C___BPM?`` is searched as text, and ``?`` acts as a single-character
    wildcard only *inside* a token that already contains a ``*``.

    Args:
        token: A single whitespace-delimited token from the search text.

    Returns:
        True if the token contains ``*``.
    """
    return "*" in token


def glob_to_are(glob: str) -> str:
    r"""Translate a glob token into a PostgreSQL Advanced Regular Expression.

    ``*`` becomes ``\S*``, ``?`` becomes ``\S``, every other ARE metacharacter is
    backslash-escaped, and the result is wrapped in the word-boundary anchors
    ``\m`` ... ``\M``. The closing ``\M`` is omitted when the glob ends in ``*``
    so ``SR01C___BPM*`` still matches the bare name ``SR01C___BPM``.

    Args:
        glob: A token for which :func:`is_glob` returned True.

    Returns:
        The equivalent ARE source string.
    """
    out: list[str] = ["\\m"]
    for char in glob:
        if char == "*":
            out.append("\\S*")
        elif char == "?":
            out.append("\\S")
        elif char in _ARE_METACHARS:
            out.append("\\" + char)
        else:
            out.append(char)
    if not glob.endswith("*"):
        out.append("\\M")
    return "".join(out)


def _skip_bracket_expression(pattern: str, start: int) -> int:
    """Return the index just past the bracket expression opening at ``start``."""
    index = start + 1
    if index < len(pattern) and pattern[index] == "^":
        index += 1
    if index < len(pattern) and pattern[index] == "]":
        index += 1  # a leading ']' is a literal member of the class
    while index < len(pattern):
        if pattern[index] == "\\":
            index += 2
            continue
        if pattern[index] == "]":
            return index + 1
        index += 1
    return len(pattern)


def _consume_quantifier(pattern: str, index: int) -> tuple[int, bool]:
    """Consume a quantifier at ``index``.

    Args:
        pattern: The ARE source string being scanned.
        index: Position immediately after the element the quantifier would bind to.

    Returns:
        The new index and whether the quantifier makes the preceding element
        optional (``?``, ``*`` and any ``{...}`` bound are treated as optional;
        ``+`` guarantees at least one occurrence).
    """
    length = len(pattern)
    if index >= length:
        return index, False
    char = pattern[index]
    if char in "*?":
        index += 1
        if index < length and pattern[index] in "*?+":
            index += 1  # non-greedy / possessive marker
        return index, True
    if char == "+":
        index += 1
        if index < length and pattern[index] in "*?":
            index += 1
        return index, False
    if char == "{":
        close = pattern.find("}", index)
        if close != -1:
            return close + 1, True
    return index, False


def longest_literal_run(pattern: str) -> int:
    r"""Length of the longest run of guaranteed consecutive literal characters.

    Metacharacters, bracket expressions (``[...]``), backslash escape *classes*
    (``\d``, ``\S``, ``\m``, ``\M``, ...), whitespace and optionally-quantified
    characters all break a run; a backslash-escaped metacharacter (``\.``) counts
    as one literal character. This is the quantity the ``pg_trgm`` index needs at
    least :data:`MIN_LITERAL_RUN` of in order to accelerate a ``~*`` predicate.

    Args:
        pattern: An ARE source string (a ``/.../`` body, or the output of
            :func:`glob_to_are`).

    Returns:
        The longest such run, 0 if the pattern has no literal characters.
    """
    best = 0
    run = 0
    index = 0
    length = len(pattern)

    while index < length:
        char = pattern[index]
        literal: str | None = None

        if char == "\\":
            if index + 1 < length:
                following = pattern[index + 1]
                # \d, \S, \w, \m, \M, \b, backreferences ... are classes/anchors;
                # \. \/ \* ... are escaped literals.
                literal = None if following.isalnum() else following
                index += 2
            else:
                index += 1
        elif char == "[":
            index = _skip_bracket_expression(pattern, index)
        elif char in _REGEX_METACHARS:
            index += 1
        else:
            literal = char
            index += 1

        index, optional = _consume_quantifier(pattern, index)

        if literal is None or optional or literal.isspace():
            best = max(best, run)
            run = 0
        else:
            run += 1
            best = max(best, run)

    return max(best, run)


def accept_pattern(body: str) -> PatternDecision:
    """Decide whether a candidate pattern may run as a ``~*`` predicate.

    Applies the shared acceptance rule for both syntaxes: a pattern is accepted
    only when :func:`longest_literal_run` is at least :data:`MIN_LITERAL_RUN`.
    Callers pass the ARE they intend to execute -- the raw body for a ``/.../``
    span, or :func:`glob_to_are` output for a glob token.

    Args:
        body: The candidate ARE source string.

    Returns:
        :class:`Accepted` carrying the ARE, or :class:`Rejected` carrying
        :data:`TOO_GENERIC_REASON` (the caller re-inserts the token's original
        text into the FTS stream and emits the reason as an INFO diagnostic).
    """
    if longest_literal_run(body) >= MIN_LITERAL_RUN:
        return Accepted(are=body)
    return Rejected(reason=TOO_GENERIC_REASON)
