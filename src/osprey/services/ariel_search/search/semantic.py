"""ARIEL semantic search module.

This module provides embedding-based similarity search using pgvector.

See 02_SEARCH_MODULES.md Section 4 for specification.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.models import SearchMode
from osprey.services.ariel_search.search.base import ParameterDescriptor, SearchToolDescriptor
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")

# Default similarity threshold
DEFAULT_SIMILARITY_THRESHOLD = 0.7


async def semantic_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    embedder: BaseEmbeddingProvider,
    *,
    max_results: int = 10,
    similarity_threshold: float | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
    **kwargs: Any,
) -> list[tuple[EnhancedLogbookEntry, float]]:
    """Execute semantic similarity search.

    Generates an embedding for the query and finds similar entries
    using cosine similarity.

    Args:
        query: Natural language query
        repository: ARIEL database repository
        config: ARIEL configuration
        embedder: Embedding provider (Ollama or other)
        max_results: Maximum entries to return (default: 10)
        similarity_threshold: Minimum similarity score (default: 0.7).
            Can be overridden per-query, then falls back to config,
            then to hardcoded default.
        start_date: Filter entries after this time
        end_date: Filter entries before this time
        author: Filter by author name (ILIKE match)
        source_system: Filter by source system (exact match)

    Returns:
        List of (entry, similarity_score) tuples sorted by similarity
    """
    if not query.strip():
        return []

    logger.info(
        f"semantic_search: query={query!r}, max_results={max_results}, "
        f"threshold={similarity_threshold}, start_date={start_date}, end_date={end_date}"
    )

    # Resolve similarity threshold using 3-tier resolution
    # 1. Per-query parameter (highest priority)
    # 2. Config value
    # 3. Hardcoded default (lowest priority)
    threshold = similarity_threshold
    if threshold is None:
        semantic_config = config.search_modules.get("semantic")
        if semantic_config and semantic_config.settings:
            threshold = semantic_config.settings.get(
                "similarity_threshold",
                DEFAULT_SIMILARITY_THRESHOLD,
            )
        else:
            threshold = DEFAULT_SIMILARITY_THRESHOLD

    # Get the model to use for embedding
    model_name = config.get_search_model()
    if not model_name:
        logger.warning("No semantic search model configured")
        return []

    # Get provider config for credentials (api_key, base_url)
    # Priority: search module provider > embedding provider > default
    semantic_module = config.search_modules.get("semantic")
    provider_name = (
        (semantic_module.provider if semantic_module else None)
        or config.embedding.provider
        or "ollama"
    )

    # Resolve provider credentials via Osprey's config system
    # This may fail in test environments without config.yml
    try:
        from osprey.utils.config import get_provider_config

        provider_config = get_provider_config(provider_name)
    except FileNotFoundError:
        # Test environment without config.yml - use empty config
        logger.debug(f"No config.yml found, using empty provider config for '{provider_name}'")
        provider_config = {}

    base_url = provider_config.get("base_url") or embedder.default_base_url
    api_key = provider_config.get("api_key")

    # Generate query embedding
    try:
        embeddings = embedder.execute_embedding(
            texts=[query],
            model_id=model_name,
            base_url=base_url,
            api_key=api_key,
        )
        if not embeddings or not embeddings[0]:
            logger.error("Failed to generate query embedding")
            return []

        query_embedding = embeddings[0]

        # Check for dimension mismatch (GAP-C008)
        # Get expected dimension from config if available
        semantic_config = config.search_modules.get("semantic")
        if semantic_config and semantic_config.settings:
            expected_dim = semantic_config.settings.get("embedding_dimension")
            if expected_dim and len(query_embedding) != expected_dim:
                logger.warning(
                    f"Embedding dimension mismatch: query embedding has "
                    f"{len(query_embedding)} dimensions but config expects "
                    f"{expected_dim}. This may cause incorrect similarity scores."
                )

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return []

    # Execute semantic search via repository
    results = await repository.semantic_search(
        query_embedding=query_embedding,
        model_name=model_name,
        max_results=max_results,
        similarity_threshold=threshold,
        start_date=start_date,
        end_date=end_date,
        author=author,
        source_system=source_system,
    )

    logger.info(f"semantic_search: returning {len(results)} results")
    return results


# === Tool descriptor for agent auto-discovery ===


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""

    query: str = Field(description="Natural language description of what to find")
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


def get_parameter_descriptors() -> list[ParameterDescriptor]:
    """Return tunable parameter descriptors for the capabilities API."""
    return [
        ParameterDescriptor(
            name="similarity_threshold",
            label="Similarity Threshold",
            description="Minimum cosine similarity score for results (0-1)",
            param_type="float",
            default=0.7,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            section="Retrieval",
        ),
    ]


def get_tool_descriptor() -> SearchToolDescriptor:
    """Return the descriptor for auto-discovery by the agent executor."""
    return SearchToolDescriptor(
        name="semantic_search",
        description=(
            "Find conceptually related entries using AI embeddings. "
            "Use for queries describing concepts, situations, or events "
            "where exact words may not match."
        ),
        search_mode=SearchMode.SEMANTIC,
        args_schema=SemanticSearchInput,
        execute=semantic_search,
        format_result=format_semantic_result,
        needs_embedder=True,
    )
