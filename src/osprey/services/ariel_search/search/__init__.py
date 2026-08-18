"""ARIEL search modules.

This module provides keyword, semantic, hybrid (qmd-sidecar-backed), and SQL
search implementations for the ARIEL search service.
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
    HybridSearchInput,
    HybridSearchSettings,
    format_qmd_result,
    hybrid_search,
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
    "HybridSearchInput",
    "HybridSearchSettings",
    "KeywordSearchInput",
    "MAX_QUERY_LENGTH",
    "SearchToolDescriptor",
    "SemanticSearchInput",
    "SqlQueryInput",
    "format_keyword_result",
    "format_qmd_result",
    "format_semantic_result",
    "format_sql_result",
    "hybrid_search",
    "keyword_search",
    "parse_query",
    "semantic_search",
    "sql_query",
    "validate_sql_query",
]
