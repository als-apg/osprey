"""ARIEL search modules.

This module provides keyword and semantic search implementations
for the ARIEL search service.

RAG is implemented via the RAGPipeline in osprey.services.ariel_search.rag:
    hybrid retrieval → RRF fusion → context assembly → LLM generation
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
from osprey.services.ariel_search.search.semantic import (
    SemanticSearchInput,
    format_semantic_result,
    semantic_search,
)

__all__ = [
    "ALLOWED_FIELD_PREFIXES",
    "ALLOWED_OPERATORS",
    "KeywordSearchInput",
    "MAX_QUERY_LENGTH",
    "SearchToolDescriptor",
    "SemanticSearchInput",
    "format_keyword_result",
    "format_semantic_result",
    "keyword_search",
    "parse_query",
    "semantic_search",
]
