"""MCP tool: semantic_search — semantic similarity search the ARIEL logbook.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Facility-customizable: similarity threshold guidance, embedding model info
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.mcp_server.ariel.server import (
    make_error,
    mcp,
    parse_date_filters,
)
from osprey.mcp_server.ariel.server_context import get_ariel_context
from osprey.mcp_server.ariel.tools.search_envelope import (
    ResultWindow,
    advanced_params,
    diagnostics,
    raise_for_fault_exception,
    raise_for_vocabulary_error,
    raise_on_statement_fault,
    success_envelope,
)
from osprey.services.ariel_search.exceptions import (
    PatternError,
    SearchTimeoutError,
    VocabularyError,
)

logger = logging.getLogger("osprey.mcp_server.ariel.tools.semantic_search")


@mcp.tool()
async def semantic_search(
    query: str,
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    author: str | None = None,
    source_system: str | None = None,
    similarity_threshold: float | None = None,
    exclude_entry_ids: list[str] | None = None,
    expand_query: bool | None = None,
) -> str:
    """Search the ARIEL logbook using embedding-based semantic similarity.

    Finds conceptually similar entries even when exact terms don't match.
    Best for natural language descriptions, concepts, or situations.

    Args:
        query: Natural language description of what to find.
        max_results: Maximum number of results (1-100, default 10).
        start_date: Filter entries after this ISO-8601 date (e.g. "2024-01-15").
        end_date: Filter entries before this ISO-8601 date.
        author: Filter by author name (partial match).
        source_system: Filter by source system (exact match).
        similarity_threshold: Minimum similarity score (0-1). Higher = stricter.
        exclude_entry_ids: Entry IDs to exclude from results (for iterative search).
        expand_query: Apply the facility vocabulary (shorthand/acronym expansion).
            None = the configured default; see capabilities().vocabulary.expand_by_default.

    Returns:
        JSON with matching entries, similarity scores, the vocabulary expansion
        applied, and any diagnostics the search reported.
    """
    if not query or not query.strip():
        return make_error(
            "validation_error",
            "Empty search query.",
            ["Provide a natural language description of what you are looking for."],
        )

    try:
        registry = get_ariel_context()
        service = await registry.service()

        parsed_start, parsed_end = parse_date_filters(start_date, end_date)
        time_range = (parsed_start, parsed_end) if parsed_start or parsed_end else None

        window = ResultWindow.build(max_results, exclude_entry_ids)

        result = await service.search(
            query,
            max_results=window.fetch_count,
            time_range=time_range,
            mode="semantic",
            advanced_params=advanced_params(
                author=author,
                source_system=source_system,
                similarity_threshold=similarity_threshold,
                expand_query=expand_query,
            ),
        )

        raise_on_statement_fault(result, "semantic")

        response = success_envelope(query, "semantic", result, window.select(result.entries))
        response["diagnostics"] = diagnostics(result)

        return json.dumps(response, default=str)

    except ToolError:
        raise
    except (PatternError, SearchTimeoutError) as exc:
        raise_for_fault_exception(exc, "semantic")
    except VocabularyError as exc:
        raise_for_vocabulary_error(exc, "semantic")
    except Exception as exc:
        logger.exception("semantic_search failed")
        return make_error(
            "internal_error",
            f"ARIEL semantic search failed: {exc}",
            [
                "Check ARIEL service configuration in config.yml.",
                "Verify the ARIEL database is reachable.",
            ],
        )
