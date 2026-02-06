"""Citation formatter implementation for ARIEL RAP pipeline.

Formats results with [#id] citations for RAG answers.
"""

from __future__ import annotations

from typing import Any

from osprey.services.ariel_search.pipeline.types import (
    FormattedResponse,
    ProcessedResult,
)


class CitationFormatter:
    """Formatter that produces text with [#id] citations.

    Used for RAG answers where citations reference source entries.
    """

    def format(
        self,
        result: ProcessedResult,
        config: dict[str, Any],
    ) -> FormattedResponse:
        """Format result with citations.

        Args:
            result: Processed result to format
            config: Format configuration (currently unused)

        Returns:
            FormattedResponse with citation-annotated text
        """
        # If there's an answer, use it as the main content
        if result.answer:
            content = result.answer
        else:
            # For non-RAG results, format the entries
            parts = []
            for item in result.items:
                entry = item.entry
                entry_id = entry.get("entry_id", "unknown")
                timestamp = entry.get("timestamp")
                timestamp_str = timestamp.isoformat() if timestamp else "Unknown"
                author = entry.get("author", "Unknown")
                text = entry.get("raw_text", "")[:500]

                parts.append(f"[#{entry_id}] ({timestamp_str}, {author})\n{text}")

            content = "\n\n".join(parts) if parts else "No results found."

        # Build sources list for metadata
        sources = result.citations or [item.entry["entry_id"] for item in result.items]

        return FormattedResponse(
            content=content,
            format_type="text",
            metadata={
                "citations": result.citations,
                "sources": sources,
                "entry_count": len(result.items),
            },
        )


__all__ = ["CitationFormatter"]
