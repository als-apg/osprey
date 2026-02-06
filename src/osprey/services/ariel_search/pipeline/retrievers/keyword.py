"""Keyword retriever implementation for ARIEL RAP pipeline.

Wraps PostgreSQL full-text search functionality.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from osprey.services.ariel_search.pipeline.types import (
    RetrievalConfig,
    RetrievedItem,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """Retriever using PostgreSQL full-text search.

    Wraps the existing keyword_search() function and normalizes
    the return type to list[RetrievedItem].

    Attributes:
        repository: ARIEL database repository
        ariel_config: ARIEL configuration
    """

    def __init__(
        self,
        repository: ARIELRepository,
        ariel_config: ARIELConfig,
    ) -> None:
        """Initialize the keyword retriever.

        Args:
            repository: Database repository for queries
            ariel_config: ARIEL configuration
        """
        self._repository = repository
        self._ariel_config = ariel_config

    @property
    def name(self) -> str:
        """Unique identifier for this retriever."""
        return "keyword"

    async def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[RetrievedItem]:
        """Retrieve items matching the query using full-text search.

        Args:
            query: Search query string
            config: Retrieval configuration

        Returns:
            List of retrieved items sorted by relevance score
        """
        from osprey.services.ariel_search.search.keyword import keyword_search

        if not query.strip():
            return []

        # Execute keyword search using existing function
        results = await keyword_search(
            query=query,
            repository=self._repository,
            config=self._ariel_config,
            max_results=config.max_results,
            start_date=config.start_date,
            end_date=config.end_date,
            include_highlights=config.include_highlights,
            fuzzy_fallback=config.fuzzy_fallback,
        )

        # Normalize to RetrievedItem format
        retrieved_items: list[RetrievedItem] = []
        for entry, score, highlights in results:
            retrieved_items.append(
                RetrievedItem(
                    entry=entry,
                    score=score,
                    source=self.name,
                    metadata={"highlights": highlights},
                )
            )

        return retrieved_items


__all__ = ["KeywordRetriever"]
