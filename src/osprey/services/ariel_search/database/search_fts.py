"""Shared ARIEL keyword-search full-text expressions.

The query predicates and expression indexes must parse to the same PostgreSQL
expression, or the planner cannot use the index. Keep those strings here rather
than letting migrations and search code drift independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

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
