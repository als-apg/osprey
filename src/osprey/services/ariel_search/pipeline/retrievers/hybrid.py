"""Hybrid retriever implementation for ARIEL RAP pipeline.

Combines multiple retrievers with configurable fusion strategies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from osprey.services.ariel_search.pipeline.types import (
    RetrievalConfig,
    RetrievedItem,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.pipeline.protocols import (
        FusionStrategy,
        Retriever,
    )

logger = logging.getLogger(__name__)


class RRFFusion:
    """Reciprocal Rank Fusion strategy.

    Combines results from multiple retrievers using RRF scoring.
    RRF is effective for combining results with different score distributions.

    Formula: RRF(d) = sum(1 / (k + rank(d))) for each retriever
    where k is a constant (default 60) and rank(d) is the document's rank.
    """

    def __init__(self, k: int = 60) -> None:
        """Initialize RRF fusion.

        Args:
            k: Ranking constant (default 60)
        """
        self.k = k

    def fuse(
        self,
        results: list[list[RetrievedItem]],
    ) -> list[RetrievedItem]:
        """Fuse results from multiple retrievers using RRF.

        Args:
            results: List of result lists, one per retriever

        Returns:
            Combined and re-ranked list of retrieved items
        """
        # Map entry_id -> (item, cumulative RRF score)
        scores: dict[str, tuple[RetrievedItem, float]] = {}

        for retriever_results in results:
            for rank, item in enumerate(retriever_results):
                entry_id = item.entry["entry_id"]
                rrf_score = 1.0 / (self.k + rank + 1)  # +1 for 0-indexed rank

                if entry_id in scores:
                    # Update score, keep first item seen (preserves metadata)
                    existing_item, existing_score = scores[entry_id]
                    # Merge sources into metadata
                    sources = existing_item.metadata.get("sources", [existing_item.source])
                    if item.source not in sources:
                        sources.append(item.source)
                    existing_item.metadata["sources"] = sources
                    scores[entry_id] = (existing_item, existing_score + rrf_score)
                else:
                    # Create new entry with hybrid source
                    hybrid_item = RetrievedItem(
                        entry=item.entry,
                        score=rrf_score,
                        source="hybrid",
                        metadata={
                            "sources": [item.source],
                            "original_score": item.score,
                            **item.metadata,
                        },
                    )
                    scores[entry_id] = (hybrid_item, rrf_score)

        # Sort by RRF score descending
        sorted_items = sorted(scores.values(), key=lambda x: x[1], reverse=True)

        # Update scores in items and return
        result_items = []
        for item, score in sorted_items:
            item.score = score
            result_items.append(item)

        return result_items


class WeightedFusion:
    """Weighted fusion strategy.

    Combines results using weighted score averaging.
    Useful when retrievers have comparable score distributions.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialize weighted fusion.

        Args:
            weights: Dict mapping retriever name to weight (default: equal weights)
        """
        self.weights = weights or {}

    def fuse(
        self,
        results: list[list[RetrievedItem]],
    ) -> list[RetrievedItem]:
        """Fuse results from multiple retrievers using weighted averaging.

        Args:
            results: List of result lists, one per retriever

        Returns:
            Combined and re-ranked list of retrieved items
        """
        # Map entry_id -> (item, weighted score sum, weight sum)
        scores: dict[str, tuple[RetrievedItem, float, float]] = {}

        for retriever_results in results:
            if not retriever_results:
                continue

            # Get retriever name from first item
            retriever_name = retriever_results[0].source
            weight = self.weights.get(retriever_name, 1.0)

            for item in retriever_results:
                entry_id = item.entry["entry_id"]
                weighted_score = item.score * weight

                if entry_id in scores:
                    existing_item, score_sum, weight_sum = scores[entry_id]
                    # Merge sources
                    sources = existing_item.metadata.get("sources", [existing_item.source])
                    if item.source not in sources:
                        sources.append(item.source)
                    existing_item.metadata["sources"] = sources
                    scores[entry_id] = (
                        existing_item,
                        score_sum + weighted_score,
                        weight_sum + weight,
                    )
                else:
                    hybrid_item = RetrievedItem(
                        entry=item.entry,
                        score=weighted_score,
                        source="hybrid",
                        metadata={
                            "sources": [item.source],
                            "original_score": item.score,
                            **item.metadata,
                        },
                    )
                    scores[entry_id] = (hybrid_item, weighted_score, weight)

        # Calculate final weighted average scores
        result_items = []
        for item, score_sum, weight_sum in scores.values():
            item.score = score_sum / weight_sum if weight_sum > 0 else 0.0
            result_items.append(item)

        # Sort by score descending
        return sorted(result_items, key=lambda x: x.score, reverse=True)


class HybridRetriever:
    """Retriever that combines multiple retrievers with a fusion strategy.

    Executes multiple retrievers in parallel and combines results
    using the configured fusion strategy.

    Attributes:
        retrievers: List of retrievers to combine
        fusion_strategy: Strategy for combining results
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        fusion_strategy: FusionStrategy | None = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            retrievers: List of retrievers to combine
            fusion_strategy: Strategy for combining results (default: RRF)
        """
        self._retrievers = retrievers
        self._fusion_strategy = fusion_strategy or RRFFusion()

    @property
    def name(self) -> str:
        """Unique identifier for this retriever."""
        return "hybrid"

    async def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[RetrievedItem]:
        """Retrieve items by running multiple retrievers and fusing results.

        Args:
            query: Search query string
            config: Retrieval configuration

        Returns:
            Combined list of retrieved items
        """
        if not query.strip():
            return []

        if not self._retrievers:
            return []

        # Run all retrievers in parallel
        all_results = await asyncio.gather(
            *[r.retrieve(query, config) for r in self._retrievers],
            return_exceptions=True,
        )

        # Filter out exceptions and log them
        valid_results: list[list[RetrievedItem]] = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                retriever_name = getattr(self._retrievers[i], "name", f"retriever_{i}")
                logger.warning(f"Retriever '{retriever_name}' failed: {result}")
            else:
                valid_results.append(result)

        if not valid_results:
            return []

        # Fuse results using strategy
        fused = self._fusion_strategy.fuse(valid_results)

        # Respect max_results limit
        return fused[: config.max_results]


__all__ = [
    "HybridRetriever",
    "RRFFusion",
    "WeightedFusion",
]
