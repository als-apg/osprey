"""MCP tools: ariel_browse + ariel_filter_options — browse and filter logbook entries.

PROMPT-PROVIDER: Tool docstrings are static prompts visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_logbook_search_prompt_builder()
  Facility-customizable: filter field descriptions, source system examples
"""

import json
import logging

from osprey.interfaces.ariel.mcp.registry import get_ariel_registry
from osprey.interfaces.ariel.mcp.server import make_error, mcp, parse_date_filters, serialize_entry

logger = logging.getLogger("osprey.interfaces.ariel.mcp.tools.browse")


@mcp.tool()
async def ariel_browse(
    page_size: int = 20,
    start_date: str | None = None,
    end_date: str | None = None,
    author: str | None = None,
    source_system: str | None = None,
) -> str:
    """Browse recent logbook entries with optional filtering.

    Returns the most recent N entries, optionally filtered by date range,
    author, or source system. Does not support offset pagination — use
    date filters to navigate through history.

    Args:
        page_size: Number of entries to return (default 20, max 100).
        start_date: Filter entries after this ISO-8601 date.
        end_date: Filter entries before this ISO-8601 date.
        author: Filter by author name (case-insensitive partial match).
        source_system: Filter by source system (exact match).

    Returns:
        JSON with entries and total_count.
    """
    try:
        registry = get_ariel_registry()
        service = await registry.service()

        # Clamp page_size
        page_size = max(1, min(page_size, 100))

        # Parse dates
        parsed_start, parsed_end = parse_date_filters(start_date, end_date)

        # Fetch entries via repository
        entries = await service.repository.search_by_time_range(
            start=parsed_start,
            end=parsed_end,
            limit=page_size,
        )

        # Post-filter by author/source_system (repository doesn't support these filters)
        if author:
            author_lower = author.lower()
            entries = [e for e in entries if author_lower in e.get("author", "").lower()]
        if source_system:
            entries = [e for e in entries if e["source_system"] == source_system]

        # Get total count
        total_count = await service.repository.count_entries()

        # Serialize entries (TypedDict — use dict access)
        entries_out = [serialize_entry(e, text_limit=300) for e in entries]

        return json.dumps(
            {
                "entries": entries_out,
                "returned": len(entries_out),
                "total_count": total_count,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("ariel_browse failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Browse failed: {exc}",
                ["Check ARIEL database connectivity."],
            )
        )


@mcp.tool()
async def ariel_filter_options(
    field: str = "authors",
) -> str:
    """Get distinct values for a filterable field.

    Useful for discovering available authors or source systems
    before filtering with ariel_browse or ariel_search.

    Args:
        field: Field to get options for — "authors" or "source_systems".

    Returns:
        JSON with field name and list of options.
    """
    try:
        registry = get_ariel_registry()
        service = await registry.service()

        field_methods = {
            "authors": "get_distinct_authors",
            "source_systems": "get_distinct_source_systems",
        }

        method_name = field_methods.get(field)
        if not method_name:
            return json.dumps(
                make_error(
                    "validation_error",
                    f"Unknown filter field: {field}",
                    [f"Available fields: {', '.join(field_methods)}"],
                )
            )

        method = getattr(service.repository, method_name)
        values = await method()

        return json.dumps(
            {
                "field": field,
                "options": values,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("ariel_filter_options failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Filter options failed: {exc}",
                ["Check ARIEL database connectivity."],
            )
        )
