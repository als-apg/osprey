"""
Example: Custom Simple Pipeline

This example shows the minimal code needed to create a custom channel finder pipeline.
It implements a simple keyword-based search without using LLMs.
"""

import logging
from typing import Any

from ..core.base_pipeline import BasePipeline
from ..core.models import ChannelFinderResult, ChannelInfo

logger = logging.getLogger(__name__)


class KeywordSearchPipeline(BasePipeline):
    """
    Simple keyword-based channel search (no LLM required).

    Searches channel names and descriptions for keyword matches.
    Fast and deterministic, good for simple queries or debugging.
    """

    def __init__(
        self,
        database,
        model_config: dict,
        case_sensitive: bool = False,
        min_score: float = 0.5,
        **kwargs,
    ):
        """
        Initialize keyword search pipeline.

        Args:
            database: Channel database (BaseDatabase)
            model_config: Not used (kept for interface compatibility)
            case_sensitive: Whether to match case-sensitively
            min_score: Minimum score threshold (0-1)
        """
        super().__init__(database, model_config, **kwargs)

        self.case_sensitive = case_sensitive
        self.min_score = min_score

        logger.info(f"Initialized KeywordSearchPipeline (case_sensitive={case_sensitive})")

    @property
    def pipeline_name(self) -> str:
        return "Keyword Search (No LLM)"

    async def process_query(self, query: str) -> ChannelFinderResult:
        """
        Search channels using simple keyword matching.

        Args:
            query: Search query with keywords

        Returns:
            ChannelFinderResult with matching channels
        """
        if not query or not query.strip():
            return ChannelFinderResult(
                query=query, channels=[], total_channels=0, processing_notes="Empty query provided"
            )

        logger.info(f"Keyword search query: {query}")

        # Extract keywords
        keywords = self._extract_keywords(query)
        logger.info(f"Extracted keywords: {keywords}")

        # Get all channels
        all_channels = self.database.get_all_channels()

        # Score each channel
        scored_channels = []
        for ch in all_channels:
            score = self._score_channel(ch, keywords)
            if score >= self.min_score:
                scored_channels.append((score, ch))

        # Sort by score (descending)
        scored_channels.sort(key=lambda x: x[0], reverse=True)

        # Build result
        channel_infos = []
        for _, ch in scored_channels:
            channel_infos.append(
                ChannelInfo(
                    channel=ch["channel"], address=ch["address"], description=ch.get("description")
                )
            )

        notes = (
            f"Keyword search found {len(channel_infos)} channels "
            f"matching keywords: {', '.join(keywords)}"
        )

        logger.info(f"Found {len(channel_infos)} channels")

        return ChannelFinderResult(
            query=query,
            channels=channel_infos,
            total_channels=len(channel_infos),
            processing_notes=notes,
        )

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query."""
        # Simple whitespace split
        if self.case_sensitive:
            return query.split()
        else:
            return query.lower().split()

    def _score_channel(self, channel: dict, keywords: list[str]) -> float:
        """
        Score channel based on keyword matches.

        Scoring:
        - Keyword in channel name: +0.5
        - Keyword in description: +0.3
        - Normalized by number of keywords
        """
        name = channel["channel"]
        description = channel.get("description", "")

        if not self.case_sensitive:
            name = name.lower()
            description = description.lower()

        score = 0.0
        matches = 0

        for keyword in keywords:
            # Check name
            if keyword in name:
                score += 0.5
                matches += 1

            # Check description
            if keyword in description:
                score += 0.3
                matches += 1

        # Normalize by number of keywords
        if keywords:
            score = score / len(keywords)

        return min(score, 1.0)  # Cap at 1.0

    def get_statistics(self) -> dict[str, Any]:
        """Return pipeline statistics."""
        db_stats = self.database.get_statistics()
        return {
            "total_channels": db_stats.get("total_channels", 0),
            "case_sensitive": self.case_sensitive,
            "min_score": self.min_score,
            "uses_llm": False,
        }


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of custom pipeline registration.

    This would typically go in your application's __init__.py or registry module.
    """

    from services.channel_finder import ChannelFinderService

    # Register the custom pipeline
    ChannelFinderService.register_pipeline("keyword", KeywordSearchPipeline)

    print("✓ Registered 'keyword' pipeline")
    print("\nAvailable pipelines:")
    for name, desc in ChannelFinderService.list_available_pipelines().items():
        print(f"  - {name}: {desc}")

    print("\nNow you can use it in config.yml:")
    print(
        """
    channel_finder:
      pipeline_mode: "keyword"
      pipelines:
        keyword:
          database:
            type: "template"
            path: "data/channels.json"
          processing:
            case_sensitive: false
            min_score: 0.5
    """
    )
