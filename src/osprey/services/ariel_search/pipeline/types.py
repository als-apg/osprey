"""Shared data types for ARIEL RAP pipeline.

These dataclasses define the data structures passed between pipeline stages.

See 05_RAP_ABSTRACTION.md Sections 2.2-2.5 for specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from osprey.services.ariel_search.models import EnhancedLogbookEntry


# === Retriever Types ===


@dataclass
class RetrievedItem:
    """A single item retrieved from the database.

    This is the unified return type for all retrievers, normalizing
    the different return formats from keyword, semantic, and RAG search.

    Attributes:
        entry: The logbook entry
        score: Relevance/similarity score (0-1 normalized)
        source: Which retriever found this (e.g., "keyword", "semantic")
        metadata: Retriever-specific metadata
    """

    entry: EnhancedLogbookEntry
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval.

    Attributes:
        max_results: Maximum items to retrieve
        start_date: Filter entries after this time
        end_date: Filter entries before this time
        similarity_threshold: Minimum similarity for semantic search
        include_highlights: Include highlighted snippets for keyword search
        fuzzy_fallback: Fall back to fuzzy search if no exact matches
    """

    max_results: int = 10
    start_date: datetime | None = None
    end_date: datetime | None = None
    similarity_threshold: float = 0.7
    include_highlights: bool = True
    fuzzy_fallback: bool = True


# === Assembler Types ===


@dataclass
class AssembledContext:
    """Context assembled from retrieved items.

    Attributes:
        items: Items that made it into context (may be subset of retrieved)
        text: Formatted context string for LLM consumption
        total_chars: Total character count of the context
        truncated: Whether any items were truncated or omitted
    """

    items: list[RetrievedItem]
    text: str
    total_chars: int
    truncated: bool


@dataclass
class AssemblyConfig:
    """Configuration for assembly.

    Attributes:
        max_items: Maximum items to include in context
        max_chars: Maximum total characters in context
        max_chars_per_item: Maximum characters per individual item
        separator: Separator between items in text
    """

    max_items: int = 10
    max_chars: int = 12000
    max_chars_per_item: int = 2000
    separator: str = "\n---\n"


# === Processor Types ===


@dataclass
class ProcessedResult:
    """Result from processing.

    Attributes:
        answer: Generated answer text (None for identity processing)
        items: Items used in processing
        reasoning: Explanation of processing (if any)
        citations: Entry IDs that were cited
    """

    answer: str | None
    items: list[RetrievedItem]
    reasoning: str | None = None
    citations: list[str] = field(default_factory=list)


@dataclass
class ProcessorConfig:
    """Configuration for processing.

    Attributes:
        temperature: LLM temperature for generation
        max_tokens: Maximum tokens in generated response
        provider: LLM provider name
        model_id: LLM model identifier
        base_url: Optional base URL for API
    """

    temperature: float = 0.1
    max_tokens: int = 1024
    provider: str = "ollama"
    model_id: str = "llama3.2"
    base_url: str | None = None


# === Formatter Types ===


@dataclass
class FormattedResponse:
    """Final formatted response.

    Attributes:
        content: Response content (string, dict, or async iterator)
        format_type: Type of format ('text', 'json', 'stream')
        metadata: Format-specific metadata
    """

    content: str | dict[str, Any] | AsyncIterator[str]
    format_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AssembledContext",
    "AssemblyConfig",
    "FormattedResponse",
    "ProcessedResult",
    "ProcessorConfig",
    "RetrievalConfig",
    "RetrievedItem",
]
