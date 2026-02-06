"""Identity processor implementation for ARIEL RAP pipeline.

Pass-through processor that returns context unchanged.
Used for keyword and semantic search where no LLM processing is needed.
"""

from __future__ import annotations

from osprey.services.ariel_search.pipeline.types import (
    AssembledContext,
    ProcessedResult,
    ProcessorConfig,
)


class IdentityProcessor:
    """Processor that passes context through unchanged.

    Used for keyword and semantic search results where
    the retrieved items should be returned directly without
    LLM processing.
    """

    @property
    def processor_type(self) -> str:
        """Type of processor."""
        return "identity"

    async def process(
        self,
        query: str,
        context: AssembledContext,
        config: ProcessorConfig,
    ) -> ProcessedResult:
        """Pass context through unchanged.

        Args:
            query: Original query string (unused)
            context: Assembled context
            config: Processing configuration (unused)

        Returns:
            ProcessedResult with no answer, just the items
        """
        # Extract entry IDs for citations
        citations = [item.entry["entry_id"] for item in context.items]

        return ProcessedResult(
            answer=None,
            items=context.items,
            reasoning=None,
            citations=citations,
        )


__all__ = ["IdentityProcessor"]
