"""ARIEL search modules.

This module provides keyword and semantic search implementations
for the ARIEL search service.

RAG is now implemented via the Pipeline abstraction:
    SemanticRetriever → ContextWindowAssembler → SingleLLMProcessor → CitationFormatter
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

# Maps config module name → module path for lazy import by the agent executor.
# To add a new search module, add one line here.
SEARCH_MODULE_REGISTRY: dict[str, str] = {
    "keyword": "osprey.services.ariel_search.search.keyword",
    "semantic": "osprey.services.ariel_search.search.semantic",
}

__all__ = [
    "ALLOWED_FIELD_PREFIXES",
    "ALLOWED_OPERATORS",
    "KeywordSearchInput",
    "MAX_QUERY_LENGTH",
    "SEARCH_MODULE_REGISTRY",
    "SearchToolDescriptor",
    "SemanticSearchInput",
    "format_keyword_result",
    "format_semantic_result",
    "keyword_search",
    "parse_query",
    "semantic_search",
]
