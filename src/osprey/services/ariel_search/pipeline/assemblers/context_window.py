"""Context window assembler implementation for ARIEL RAP pipeline.

Token-aware assembly for RAG context preparation.
Extracted from search/rag.py format_entry_for_context() and assembly loop.
"""

from __future__ import annotations

from osprey.services.ariel_search.pipeline.types import (
    AssembledContext,
    AssemblyConfig,
    RetrievedItem,
)


class ContextWindowAssembler:
    """Assembler that respects context window limits for LLM consumption.

    Formats entries in a structured way suitable for RAG prompts,
    respecting both per-entry and total context character limits.

    This implements the ENTRY #id format per spec Section 5.5.2.
    """

    def _format_entry_for_context(
        self,
        item: RetrievedItem,
        max_chars_per_item: int,
    ) -> str:
        """Format a single entry for inclusion in RAG context.

        Uses ENTRY #id format per spec Section 5.5.2.

        Args:
            item: Retrieved item to format
            max_chars_per_item: Maximum characters for this entry

        Returns:
            Formatted string for the entry
        """
        entry = item.entry
        metadata = entry.get("metadata", {})
        title = metadata.get("title", "")
        author = entry.get("author", "Unknown")
        timestamp = entry.get("timestamp")
        timestamp_str = timestamp.isoformat() if timestamp else "Unknown"

        # Use ENTRY #id | timestamp | Author: name format per spec Section 5.5.2
        header = f"ENTRY #{entry['entry_id']} | {timestamp_str} | Author: {author}"
        if title:
            header += f" | {title}"

        content = entry.get("raw_text", "")
        # Truncate long entries
        if len(content) > max_chars_per_item:
            content = content[:max_chars_per_item] + "..."

        return f"{header}\n{content}\n"

    def assemble(
        self,
        items: list[RetrievedItem],
        config: AssemblyConfig,
    ) -> AssembledContext:
        """Assemble items into context respecting character limits.

        Args:
            items: Retrieved items to assemble
            config: Assembly configuration with character limits

        Returns:
            Assembled context with formatted text for LLM
        """
        if not items:
            return AssembledContext(
                items=[],
                text="",
                total_chars=0,
                truncated=False,
            )

        # Sort by score descending
        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)

        # Build context respecting limits
        context_parts: list[str] = []
        included_items: list[RetrievedItem] = []
        total_chars = 0
        truncated = False

        for item in sorted_items:
            # Check max items limit
            if len(included_items) >= config.max_items:
                truncated = True
                break

            # Format the entry
            formatted = self._format_entry_for_context(
                item,
                config.max_chars_per_item,
            )

            # Check if adding this would exceed total limit
            if total_chars + len(formatted) > config.max_chars:
                # Try to fit a truncated version
                remaining = config.max_chars - total_chars
                if remaining > 100:  # Only add if meaningful content remains
                    formatted = formatted[:remaining] + "..."
                    context_parts.append(formatted)
                    total_chars += len(formatted)
                    included_items.append(item)
                truncated = True
                break

            context_parts.append(formatted)
            total_chars += len(formatted)
            included_items.append(item)

        # Join with separator
        text = config.separator.join(context_parts)

        return AssembledContext(
            items=included_items,
            text=text,
            total_chars=total_chars,
            truncated=truncated,
        )


__all__ = ["ContextWindowAssembler"]
