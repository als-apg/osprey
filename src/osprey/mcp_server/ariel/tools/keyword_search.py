"""MCP tool: keyword_search — keyword search the ARIEL logbook.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Facility-customizable: field prefix examples, operator guidance
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

logger = logging.getLogger("osprey.mcp_server.ariel.tools.keyword_search")


@mcp.tool()
async def keyword_search(
    query: str,
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    author: str | None = None,
    source_system: str | None = None,
    exclude_entry_ids: list[str] | None = None,
    expand_query: bool | None = None,
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
        expand_query: Apply the facility vocabulary (shorthand/acronym expansion).
            None = the configured default; see capabilities().vocabulary.expand_by_default.

    Returns:
        JSON with matching entries, scores, the vocabulary expansion applied,
        and any diagnostics the search reported.
    """
    if not query or not query.strip():
        return make_error(
            "validation_error",
            "Empty search query.",
            ["Provide search terms describing what you are looking for."],
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
            mode="keyword",
            advanced_params=advanced_params(
                author=author, source_system=source_system, expand_query=expand_query
            ),
        )

        raise_on_statement_fault(result, "keyword")

        response = success_envelope(query, "keyword", result, window.select(result.entries))
        response["diagnostics"] = diagnostics(result)

        return json.dumps(response, default=str)

    except ToolError:
        raise
    except (PatternError, SearchTimeoutError) as exc:
        raise_for_fault_exception(exc, "keyword")
    except VocabularyError as exc:
        raise_for_vocabulary_error(exc, "keyword")
    except Exception as exc:
        logger.exception("keyword_search failed")
        return make_error(
            "internal_error",
            f"ARIEL keyword search failed: {exc}",
            [
                "Check ARIEL service configuration in config.yml.",
                "Verify the ARIEL database is reachable.",
            ],
        )
