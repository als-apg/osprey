"""Semantic retriever implementation for ARIEL RAP pipeline.

Wraps embedding-based similarity search functionality.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from osprey.services.ariel_search.pipeline.types import (
    RetrievalConfig,
    RetrievedItem,
)

if TYPE_CHECKING:
    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Retriever using embedding similarity search.

    Wraps the existing semantic_search() function and normalizes
    the return type to list[RetrievedItem].

    Attributes:
        repository: ARIEL database repository
        ariel_config: ARIEL configuration
        embedder: Embedding provider for query vectorization
    """

    def __init__(
        self,
        repository: ARIELRepository,
        ariel_config: ARIELConfig,
        embedder: BaseEmbeddingProvider,
    ) -> None:
        """Initialize the semantic retriever.

        Args:
            repository: Database repository for queries
            ariel_config: ARIEL configuration
            embedder: Embedding provider for query vectorization
        """
        self._repository = repository
        self._ariel_config = ariel_config
        self._embedder = embedder

    @property
    def name(self) -> str:
        """Unique identifier for this retriever."""
        return "semantic"

    async def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[RetrievedItem]:
        """Retrieve items matching the query using semantic similarity.

        Args:
            query: Natural language query string
            config: Retrieval configuration

        Returns:
            List of retrieved items sorted by similarity score
        """
        from osprey.services.ariel_search.search.semantic import semantic_search

        if not query.strip():
            return []

        # Execute semantic search using existing function
        results = await semantic_search(
            query=query,
            repository=self._repository,
            config=self._ariel_config,
            embedder=self._embedder,
            max_results=config.max_results,
            similarity_threshold=config.similarity_threshold,
            start_date=config.start_date,
            end_date=config.end_date,
        )

        # Normalize to RetrievedItem format
        retrieved_items: list[RetrievedItem] = []
        for entry, similarity in results:
            retrieved_items.append(
                RetrievedItem(
                    entry=entry,
                    score=similarity,
                    source=self.name,
                    metadata={"similarity": similarity},
                )
            )

        return retrieved_items


__all__ = ["SemanticRetriever"]
