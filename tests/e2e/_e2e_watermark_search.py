"""E2E custom search module: watermark search.

A minimal search module that queries for entries stamped with a
watermark in their ``metadata`` JSONB column.  Used by
``TestCustomModuleIntegration`` to prove that user-provided search
modules are discovered, loaded, and invoked by the ReAct agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.models import SearchMode, enhanced_entry_from_row
from osprey.services.ariel_search.search.base import SearchToolDescriptor

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry


class WatermarkSearchInput(BaseModel):
    """Input schema for the watermark_search tool."""

    query: str = Field(
        description="Search query — any text.  The tool returns entries whose "
        "metadata contains an 'e2e_watermark' key."
    )


async def watermark_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    *,
    max_results: int = 10,
    **kwargs: Any,
) -> list[tuple[EnhancedLogbookEntry, float, list[str]]]:
    """Return entries that have a non-null ``e2e_watermark`` metadata key."""
    from psycopg.rows import dict_row

    async with repository.pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT *
                FROM enhanced_entries
                WHERE metadata->>'e2e_watermark' IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                [max_results],
            )
            rows = await cur.fetchall()

    results: list[tuple[EnhancedLogbookEntry, float, list[str]]] = []
    for row in rows:
        entry = enhanced_entry_from_row(row)
        watermark = entry["metadata"].get("e2e_watermark", "")
        results.append((entry, 1.0, [f"watermark={watermark}"]))
    return results


def format_watermark_result(
    entry: EnhancedLogbookEntry,
    score: float,
    highlights: list[str],
) -> dict[str, Any]:
    """Format a watermark search result for agent consumption."""
    timestamp = entry.get("timestamp")
    return {
        "entry_id": entry.get("entry_id"),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "author": entry.get("author"),
        "text": entry.get("raw_text", "")[:500],
        "watermark": entry.get("metadata", {}).get("e2e_watermark"),
        "score": score,
        "highlights": highlights,
    }


def get_tool_descriptor() -> SearchToolDescriptor:
    """Return the descriptor for auto-discovery by the agent executor."""
    return SearchToolDescriptor(
        name="watermark_search",
        description=(
            "Search for logbook entries that have been stamped with a watermark. "
            "Use this tool when the user asks about watermarked entries."
        ),
        search_mode=SearchMode.KEYWORD,
        args_schema=WatermarkSearchInput,
        execute=watermark_search,
        format_result=format_watermark_result,
        needs_embedder=False,
    )
