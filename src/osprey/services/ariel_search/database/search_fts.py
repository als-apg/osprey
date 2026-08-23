"""Shared ARIEL keyword-search full-text expressions.

The query predicates and expression indexes must parse to the same PostgreSQL
expression, or the planner cannot use the index. Keep those strings here rather
than letting migrations and search code drift independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osprey.services.ariel_search.vocabulary.model import normalize

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.search.base import (
        ExpansionGroup,
        ParsedKeywordQuery,
        QueryExpansion,
    )

RAW_TEXT_SEARCH_DOCUMENT = "raw_text"
RAW_TEXT_FTS_EXPRESSION = f"to_tsvector('english', {RAW_TEXT_SEARCH_DOCUMENT})"
SEMANTIC_KEYWORDS_DOCUMENT = "osprey_text_array_to_string(keywords)"
SEMANTIC_TEXT_SEARCH_DOCUMENT = (
    f"raw_text || ' ' || COALESCE(summary, '') || ' ' || COALESCE({SEMANTIC_KEYWORDS_DOCUMENT}, '')"
)
SEMANTIC_FTS_EXPRESSION = f"to_tsvector('english', {SEMANTIC_TEXT_SEARCH_DOCUMENT})"


def keyword_search_expressions(config: ARIELConfig) -> tuple[str, str]:
    """Return the ranking FTS expression and headline document for keyword search."""
    if config.is_enhancement_module_enabled("semantic_processor"):
        return SEMANTIC_FTS_EXPRESSION, SEMANTIC_TEXT_SEARCH_DOCUMENT
    return RAW_TEXT_FTS_EXPRESSION, RAW_TEXT_SEARCH_DOCUMENT


def keyword_fts_expression(config: ARIELConfig) -> str:
    """Return the FTS predicate expression available for the configured schema."""
    expression, _document = keyword_search_expressions(config)
    return expression


PLAIN_TSQUERY_LEG = "plainto_tsquery('english', %s)"
PHRASE_TSQUERY_LEG = "phraseto_tsquery('english', %s)"
EMPTY_TSQUERY = "plainto_tsquery('english', '')"


def build_expanded_tsquery(
    parsed: ParsedKeywordQuery, expansion: QueryExpansion
) -> tuple[str, list[str]]:
    """Build a vocabulary-expanded tsquery covering every term of a parsed query.

    The sibling of :func:`~osprey.services.ariel_search.search.keyword.build_tsquery`,
    used only when an expansion was resolved for the request. `build_tsquery`
    keeps its ``-> str`` signature and its callers; this function returns the SQL
    *and* the parameters, because an expansion needs one placeholder per
    alternative rather than one per query.

    The query is decomposed into non-overlapping spans and every span becomes a
    leg, so no term is ever dropped:

    * a **concept span** (one :class:`ExpansionGroup`) becomes an
      ``||``-alternation of ``plainto_tsquery`` over the matched span followed by
      its alternatives — matching the span *or* any alternative satisfies the leg
    * each **maximal run of unmatched tokens** between concept spans becomes one
      ``plainto_tsquery`` leg over that run
    * each **quoted phrase** becomes a ``phraseto_tsquery`` leg, or an
      alternation of that phrase with the alternatives of the group it resolved to

    All legs are joined with ``&&``, so the expansion widens *what counts as a
    match for one span* without loosening the requirement that every span match.

    Boolean-operator queries never reach this function: the caller routes them to
    the unexpanded `build_tsquery` path, since ``websearch_to_tsquery`` syntax
    cannot express the grouped alternation the expansion needs.

    Span location: `expansion.groups` are produced over normalized text, so this
    function normalizes `parsed.search_text` the same way and walks the token
    list left to right, locating each group's `original` at or after the previous
    match. Unmatched runs and concept spans are therefore bound as *normalized*
    text; phrases are bound exactly as the user quoted them. Normalization only
    lowercases, turns ``-`` into a space and collapses whitespace, all of which
    ``plainto_tsquery`` does anyway, so it cannot change what matches.

    Phrase attribution (deliberate simplification): a group is attributed to a
    phrase only when the group's `original` equals the normalized phrase in full.
    A group matching *inside* a longer phrase leaves that phrase unexpanded — a
    phrase is an exact-sequence requirement, and folding an alternative into part
    of it would change what the phrase means.

    Args:
        parsed: The parsed keyword query whose `search_text` and `phrases`
            drive the decomposition.
        expansion: The resolved expansion, whose `groups` are in query order,
            non-overlapping, and spelled as in normalized text.

    Returns:
        ``(sql, params)`` where `sql` is the ``&&``-joined tsquery expression and
        `params` are its bind values in placeholder order — one occurrence each,
        so ``len(params) == sql.count("%s")``. An empty query (no search text and
        no phrases) yields ``("plainto_tsquery('english', '')", [])``, mirroring
        `build_tsquery`.
    """
    legs: list[str] = []
    params: list[str] = []

    tokens = normalize(parsed.search_text).split()
    unlocated: list[ExpansionGroup] = []
    cursor = 0

    for group in expansion.groups:
        span_tokens = group.original.split()
        position = _locate_span(tokens, span_tokens, cursor)
        if position is None:
            # Not in search_text: the service expanded over search_text + phrases,
            # so this group belongs to a phrase (handled below).
            unlocated.append(group)
            continue
        if position > cursor:
            legs.append(PLAIN_TSQUERY_LEG)
            params.append(" ".join(tokens[cursor:position]))
        terms = [group.original, *group.alternatives]
        legs.append(_alternation([PLAIN_TSQUERY_LEG] * len(terms)))
        params.extend(terms)
        cursor = position + len(span_tokens)

    if cursor < len(tokens):
        legs.append(PLAIN_TSQUERY_LEG)
        params.append(" ".join(tokens[cursor:]))

    for phrase in parsed.phrases:
        phrase_group = _pop_phrase_group(unlocated, phrase)
        if phrase_group is None:
            legs.append(PHRASE_TSQUERY_LEG)
            params.append(phrase)
            continue
        alternatives = phrase_group.alternatives
        legs.append(_alternation([PHRASE_TSQUERY_LEG, *([PLAIN_TSQUERY_LEG] * len(alternatives))]))
        params.append(phrase)
        params.extend(alternatives)

    if not legs:
        return EMPTY_TSQUERY, []
    return " && ".join(legs), params


def _alternation(legs: list[str]) -> str:
    """Join tsquery legs into one ``||`` alternation, parenthesized when needed."""
    if len(legs) == 1:
        return legs[0]
    return "(" + " || ".join(legs) + ")"


def _locate_span(tokens: list[str], span_tokens: list[str], start: int) -> int | None:
    """Find a contiguous token sequence at or after `start`.

    Args:
        tokens: The normalized token list being walked.
        span_tokens: The tokens of the span to locate.
        start: Index to start searching from (never look back — groups are in
            text order and non-overlapping).

    Returns:
        The index where the span begins, or None when it is not present.
    """
    width = len(span_tokens)
    if width == 0:
        return None
    for index in range(start, len(tokens) - width + 1):
        if tokens[index : index + width] == span_tokens:
            return index
    return None


def _pop_phrase_group(unlocated: list[ExpansionGroup], phrase: str) -> ExpansionGroup | None:
    """Take the group that expanded a whole phrase, if there is one.

    Args:
        unlocated: Groups not found in `search_text`, mutated in place when one
            is consumed, so a group is never emitted twice.
        phrase: The quoted phrase as the user typed it.

    Returns:
        The matching group, removed from `unlocated`, or None when the phrase
        expanded to nothing (or only in part — see `build_expanded_tsquery`).
    """
    normalized_phrase = normalize(phrase)
    for group in unlocated:
        if group.original == normalized_phrase:
            unlocated.remove(group)
            return group
    return None
