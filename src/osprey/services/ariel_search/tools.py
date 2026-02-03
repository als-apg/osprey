"""ARIEL search tools for the ReAct agent.

This module provides LangChain StructuredTool wrappers for the search modules.

See 03_AGENTIC_REASONING.md Section 2.6 for specification.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools import StructuredTool

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import ARIELSearchRequest, EnhancedLogbookEntry

logger = logging.getLogger(__name__)


# === Input Schemas for Tools ===


class KeywordSearchInput(BaseModel):
    """Input schema for keyword search tool."""

    query: str = Field(
        description="Search terms. Supports phrases in quotes, AND/OR/NOT operators."
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Filter entries created after this time (inclusive)",
    )
    end_date: datetime | None = Field(
        default=None,
        description="Filter entries created before this time (inclusive)",
    )


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""

    query: str = Field(
        description="Natural language description of what to find"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return",
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1)",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Filter entries created after this time (inclusive)",
    )
    end_date: datetime | None = Field(
        default=None,
        description="Filter entries created before this time (inclusive)",
    )


class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool."""

    query: str = Field(
        description="Natural language question to answer"
    )
    max_entries: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum entries for context",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Filter entries created after this time (inclusive)",
    )
    end_date: datetime | None = Field(
        default=None,
        description="Filter entries created before this time (inclusive)",
    )


# === Tool Output Formatting ===


def format_keyword_result(
    entry: EnhancedLogbookEntry,
    score: float,
    highlights: list[str],
) -> dict[str, Any]:
    """Format a keyword search result for agent consumption.

    Args:
        entry: EnhancedLogbookEntry
        score: Relevance score
        highlights: Highlighted snippets

    Returns:
        Formatted dict for agent
    """
    timestamp = entry.get("timestamp")
    return {
        "entry_id": entry.get("entry_id"),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "author": entry.get("author"),
        "text": entry.get("raw_text", "")[:500],  # Truncate for agent
        "title": entry.get("metadata", {}).get("title"),
        "score": score,
        "highlights": highlights,
    }


def format_semantic_result(
    entry: EnhancedLogbookEntry,
    similarity: float,
) -> dict[str, Any]:
    """Format a semantic search result for agent consumption.

    Args:
        entry: EnhancedLogbookEntry
        similarity: Cosine similarity score

    Returns:
        Formatted dict for agent
    """
    timestamp = entry.get("timestamp")
    return {
        "entry_id": entry.get("entry_id"),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "author": entry.get("author"),
        "text": entry.get("raw_text", "")[:500],
        "title": entry.get("metadata", {}).get("title"),
        "similarity": similarity,
    }


# === Tool Creation ===


def create_search_tools(
    config: ARIELConfig,
    repository: ARIELRepository,
    embedder_loader: Callable[[], BaseEmbeddingProvider],
    request: ARIELSearchRequest,
) -> list[StructuredTool]:
    """Create LangChain tools from enabled search modules.

    Search modules are plain functions. This function creates closures that
    capture repository, config, embedder, and request context, making them
    available to search functions when called.

    Time Range Resolution (3-tier priority):
    1. Tool call parameter (highest) - Agent explicitly passes start_date/end_date
    2. Request context - From ARIELSearchRequest.time_range (default for session)
    3. No filter (lowest) - Search all entries

    Args:
        config: ARIEL configuration
        repository: Database repository instance
        embedder_loader: Callable that returns embedding model (lazy-loaded)
        request: ARIELSearchRequest containing query and optional default time_range

    Returns:
        List of StructuredTool instances for enabled search modules
    """
    from langchain_core.tools import StructuredTool

    tools: list[StructuredTool] = []

    def _resolve_time_range(
        tool_start: datetime | None,
        tool_end: datetime | None,
    ) -> tuple[datetime | None, datetime | None]:
        """Resolve time range with 3-tier priority.

        Agent can override by passing explicit dates, or omit to use defaults.
        """
        # Explicit tool params override request context
        if tool_start is not None or tool_end is not None:
            return (tool_start, tool_end)
        # Fall back to request context
        if request.time_range:
            return request.time_range
        # No filtering
        return (None, None)

    # Keyword Search Tool
    if config.is_search_module_enabled("keyword"):
        async def _keyword_search(
            query: str,
            max_results: int = 10,
            start_date: datetime | None = None,
            end_date: datetime | None = None,
        ) -> list[dict[str, Any]]:
            """Execute keyword search with captured dependencies."""
            from osprey.services.ariel_search.search.keyword import keyword_search

            resolved_start, resolved_end = _resolve_time_range(start_date, end_date)

            results = await keyword_search(
                query=query,
                repository=repository,
                config=config,
                max_results=max_results,
                start_date=resolved_start,
                end_date=resolved_end,
            )

            return [
                format_keyword_result(entry, score, highlights)
                for entry, score, highlights in results
            ]

        tools.append(
            StructuredTool.from_function(
                func=_keyword_search,
                coroutine=_keyword_search,
                name="keyword_search",
                description=(
                    "Fast text-based lookup using full-text search. "
                    "Use for specific terms, equipment names, PV names, or phrases. "
                    "Supports quoted phrases and AND/OR/NOT operators."
                ),
                args_schema=KeywordSearchInput,
            )
        )

    # Semantic Search Tool
    if config.is_search_module_enabled("semantic"):
        async def _semantic_search(
            query: str,
            max_results: int = 10,
            similarity_threshold: float = 0.7,
            start_date: datetime | None = None,
            end_date: datetime | None = None,
        ) -> list[dict[str, Any]]:
            """Execute semantic search with captured dependencies."""
            from osprey.services.ariel_search.search.semantic import semantic_search

            resolved_start, resolved_end = _resolve_time_range(start_date, end_date)
            embedder = embedder_loader()

            results = await semantic_search(
                query=query,
                repository=repository,
                config=config,
                embedder=embedder,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                start_date=resolved_start,
                end_date=resolved_end,
            )

            return [
                format_semantic_result(entry, similarity)
                for entry, similarity in results
            ]

        tools.append(
            StructuredTool.from_function(
                func=_semantic_search,
                coroutine=_semantic_search,
                name="semantic_search",
                description=(
                    "Find conceptually related entries using AI embeddings. "
                    "Use for queries describing concepts, situations, or events "
                    "where exact words may not match."
                ),
                args_schema=SemanticSearchInput,
            )
        )

    # RAG Search Tool
    if config.is_search_module_enabled("rag"):
        async def _rag_search(
            query: str,
            max_entries: int = 5,
            start_date: datetime | None = None,
            end_date: datetime | None = None,
        ) -> dict[str, Any]:
            """Execute RAG search with captured dependencies."""
            from osprey.services.ariel_search.search.rag import rag_search

            resolved_start, resolved_end = _resolve_time_range(start_date, end_date)
            embedder = embedder_loader()

            answer, source_entries = await rag_search(
                query=query,
                repository=repository,
                config=config,
                embedder=embedder,
                max_entries=max_entries,
                start_date=resolved_start,
                end_date=resolved_end,
            )

            return {
                "answer": answer,
                "sources": [e.get("entry_id") for e in source_entries],
                "entry_count": len(source_entries),
            }

        tools.append(
            StructuredTool.from_function(
                func=_rag_search,
                coroutine=_rag_search,
                name="rag_search",
                description=(
                    "Generate an answer to a question using retrieved logbook entries. "
                    "Use for direct questions expecting a synthesized answer "
                    "or when reasoning over multiple entries is needed."
                ),
                args_schema=RAGSearchInput,
            )
        )

    return tools


__all__ = [
    "KeywordSearchInput",
    "RAGSearchInput",
    "SemanticSearchInput",
    "create_search_tools",
    "format_keyword_result",
    "format_semantic_result",
]
