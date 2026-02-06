"""ARIEL semantic search module.

This module provides embedding-based similarity search using pgvector.

See 02_SEARCH_MODULES.md Section 4 for specification.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = logging.getLogger(__name__)

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

    Returns:
        List of (entry, similarity_score) tuples sorted by similarity
    """
    if not query.strip():
        return []

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
    )

    return results
