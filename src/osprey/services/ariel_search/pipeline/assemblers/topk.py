"""TopK assembler implementation for ARIEL RAP pipeline.

Simple top-K selection by score for keyword/semantic search.
"""

from __future__ import annotations

from osprey.services.ariel_search.pipeline.types import (
    AssembledContext,
    AssemblyConfig,
    RetrievedItem,
)


class TopKAssembler:
    """Assembler that selects top K items by score.

    Simple assembler for keyword and semantic search results
    that don't need LLM-oriented context formatting.
    """

    def assemble(
        self,
        items: list[RetrievedItem],
        config: AssemblyConfig,
    ) -> AssembledContext:
        """Assemble items by selecting top K by score.

        Args:
            items: Retrieved items to assemble
            config: Assembly configuration

        Returns:
            Assembled context with top K items
        """
        if not items:
            return AssembledContext(
                items=[],
                text="",
                total_chars=0,
                truncated=False,
            )

        # Sort by score descending and take top K
        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)
        top_items = sorted_items[: config.max_items]

        # Build simple text representation
        text_parts = []
        total_chars = 0

        for item in top_items:
            entry = item.entry
            entry_id = entry.get("entry_id", "unknown")
            timestamp = entry.get("timestamp")
            timestamp_str = timestamp.isoformat() if timestamp else "Unknown"
            author = entry.get("author", "Unknown")
            raw_text = entry.get("raw_text", "")

            # Format entry
            header = f"[{entry_id}] {timestamp_str} - {author}"
            content = raw_text[: config.max_chars_per_item]
            if len(raw_text) > config.max_chars_per_item:
                content += "..."

            entry_text = f"{header}\n{content}"
            text_parts.append(entry_text)
            total_chars += len(entry_text)

        text = config.separator.join(text_parts)
        truncated = len(sorted_items) > config.max_items

        return AssembledContext(
            items=top_items,
            text=text,
            total_chars=total_chars,
            truncated=truncated,
        )


__all__ = ["TopKAssembler"]
