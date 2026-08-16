"""ARIEL search modules.

This module provides keyword, semantic, qmd sidecar, and SQL search
implementations for the ARIEL search service.
"""

from osprey.services.ariel_search.search.base import SearchToolDescriptor
from osprey.services.ariel_search.search.keyword import (
    ALLOWED_FIELD_PREFIXES,
    ALLOWED_OPERATORS,
    MAX_QUERY_LENGTH,
    KeywordSearchInput,
    format_keyword_result,
    keyword_search,
    parse_query,
)
from osprey.services.ariel_search.search.qmd import (
    ARIEL_COLLECTION,
    QmdSearchInput,
    QmdSearchSettings,
    format_qmd_result,
    qmd_search,
)
from osprey.services.ariel_search.search.semantic import (
    SemanticSearchInput,
    format_semantic_result,
    semantic_search,
)
from osprey.services.ariel_search.search.sql_query import (
    SqlQueryInput,
    format_sql_result,
    sql_query,
    validate_sql_query,
)

__all__ = [
    "ALLOWED_FIELD_PREFIXES",
    "ALLOWED_OPERATORS",
    "ARIEL_COLLECTION",
    "KeywordSearchInput",
    "MAX_QUERY_LENGTH",
    "QmdSearchInput",
    "QmdSearchSettings",
    "SearchToolDescriptor",
    "SemanticSearchInput",
    "SqlQueryInput",
    "format_keyword_result",
    "format_qmd_result",
    "format_semantic_result",
    "format_sql_result",
    "keyword_search",
    "parse_query",
    "qmd_search",
    "semantic_search",
    "sql_query",
    "validate_sql_query",
]
