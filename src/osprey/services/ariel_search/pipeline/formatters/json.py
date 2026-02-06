"""JSON formatter implementation for ARIEL RAP pipeline.

Formats results as structured JSON for API responses.
"""

from __future__ import annotations

from typing import Any

from osprey.services.ariel_search.pipeline.types import (
    FormattedResponse,
    ProcessedResult,
)


class JSONFormatter:
    """Formatter that produces structured JSON output.

    Used for API responses and tool outputs where structured
    data is needed.
    """

    def __init__(
        self,
        max_text_length: int = 500,
        include_metadata: bool = True,
    ) -> None:
        """Initialize JSON formatter.

        Args:
            max_text_length: Maximum characters for text snippets
            include_metadata: Whether to include entry metadata
        """
        self._max_text_length = max_text_length
        self._include_metadata = include_metadata

    def format(
        self,
        result: ProcessedResult,
        config: dict[str, Any],
    ) -> FormattedResponse:
        """Format result as structured JSON.

        Args:
            result: Processed result to format
            config: Format configuration overrides

        Returns:
            FormattedResponse with JSON dict content
        """
        # Allow config overrides
        max_text = config.get("max_text_length", self._max_text_length)
        include_meta = config.get("include_metadata", self._include_metadata)

        # Format entries
        entries = []
        for item in result.items:
            entry = item.entry
            timestamp = entry.get("timestamp")

            formatted_entry: dict[str, Any] = {
                "entry_id": entry.get("entry_id"),
                "timestamp": timestamp.isoformat() if timestamp else None,
                "author": entry.get("author"),
                "text": entry.get("raw_text", "")[:max_text],
                "score": item.score,
                "source": item.source,
            }

            if include_meta:
                metadata = entry.get("metadata", {})
                if metadata.get("title"):
                    formatted_entry["title"] = metadata["title"]

            # Include retriever-specific metadata
            if item.metadata.get("highlights"):
                formatted_entry["highlights"] = item.metadata["highlights"]
            if item.metadata.get("similarity"):
                formatted_entry["similarity"] = item.metadata["similarity"]

            entries.append(formatted_entry)

        # Build response dict
        content: dict[str, Any] = {
            "entries": entries,
            "entry_count": len(entries),
            "sources": result.citations,
        }

        # Add answer if present (for RAG)
        if result.answer is not None:
            content["answer"] = result.answer

        # Add reasoning if present
        if result.reasoning:
            content["reasoning"] = result.reasoning

        return FormattedResponse(
            content=content,
            format_type="json",
            metadata={
                "entry_count": len(entries),
                "has_answer": result.answer is not None,
            },
        )


__all__ = ["JSONFormatter"]
