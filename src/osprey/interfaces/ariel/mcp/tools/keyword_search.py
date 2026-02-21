"""MCP tool: keyword_search — keyword search the ARIEL logbook.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Facility-customizable: field prefix examples, operator guidance
"""

import json
import logging

from osprey.interfaces.ariel.mcp.registry import get_ariel_registry
from osprey.interfaces.ariel.mcp.server import (
    make_error,
    mcp,
    parse_date_filters,
    serialize_entry,
)

logger = logging.getLogger("osprey.interfaces.ariel.mcp.tools.keyword_search")


@mcp.tool()
async def keyword_search(
    query: str,
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    author: str | None = None,
    source_system: str | None = None,
    exclude_entry_ids: list[str] | None = None,
) -> str:
    """Search the ARIEL logbook using PostgreSQL full-text keyword search.

    Fast, exact-matching search. Supports quoted phrases and AND/OR/NOT operators.
    Best for specific terms, equipment names, PV names, or known phrases.

    Args:
        query: Search terms. Supports phrases in quotes, AND/OR/NOT operators.
        max_results: Maximum number of results (1-100, default 10).
        start_date: Filter entries after this ISO-8601 date (e.g. "2024-01-15").
        end_date: Filter entries before this ISO-8601 date.
        author: Filter by author name (partial match).
        source_system: Filter by source system (exact match).
        exclude_entry_ids: Entry IDs to exclude from results (for iterative search).

    Returns:
        JSON with matching entries, scores, and workspace file path.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "Empty search query.",
                ["Provide search terms describing what you are looking for."],
            )
        )

    try:
        from osprey.services.ariel_search.models import SearchMode

        registry = get_ariel_registry()
        service = await registry.service()

        # Parse dates
        parsed_start, parsed_end = parse_date_filters(start_date, end_date)
        time_range = (parsed_start, parsed_end) if parsed_start or parsed_end else None

        # Build advanced params
        adv: dict = {}
        if author:
            adv["author"] = author
        if source_system:
            adv["source_system"] = source_system

        # Over-fetch to compensate for post-filtering excluded IDs
        exclude_ids = set(exclude_entry_ids or [])
        fetch_count = max_results + len(exclude_ids) if exclude_ids else max_results

        # Execute search
        result = await service.search(
            query,
            max_results=fetch_count,
            time_range=time_range,
            mode=SearchMode.KEYWORD,
            advanced_params=adv,
        )

        # Post-filter excluded entry IDs
        entries = [e for e in result.entries if e["entry_id"] not in exclude_ids]
        entries = entries[:max_results]

        # Serialize
        entries_out = [serialize_entry(e, text_limit=500) for e in entries]

        response = {
            "query": query,
            "mode": "keyword",
            "results_found": len(entries_out),
            "reasoning": result.reasoning,
            "sources": list(result.sources),
            "entries": entries_out,
        }

        return json.dumps(response, default=str)

    except Exception as exc:
        logger.exception("keyword_search failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"ARIEL keyword search failed: {exc}",
                [
                    "Check ARIEL service configuration in config.yml.",
                    "Verify the ARIEL database is reachable.",
                ],
            )
        )
