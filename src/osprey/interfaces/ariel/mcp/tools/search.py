"""MCP tool: ariel_search — search the ARIEL logbook.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_logbook_search_prompt_builder()
  Facility-customizable: search mode descriptions, mode recommendations,
  facility-specific filter options, advanced parameter guidance
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

logger = logging.getLogger("osprey.interfaces.ariel.mcp.tools.search")


@mcp.tool()
async def ariel_search(
    query: str,
    mode: str = "rag",
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    author: str | None = None,
    source_system: str | None = None,
    advanced_params: dict | None = None,
) -> str:
    """Search the ARIEL logbook using keyword, semantic, RAG, or agent modes.

    Modes:
    - keyword: PostgreSQL full-text search (fastest, exact matching)
    - semantic: Embedding similarity search (conceptual matching)
    - rag: Retrieval-augmented generation — hybrid search + LLM answer (default)
    - agent: Agentic ReAct search with multi-step reasoning

    Args:
        query: Natural-language search query.
        mode: Search mode — "keyword", "semantic", "rag", or "agent".
        max_results: Maximum number of results (1-100, default 10).
        start_date: Filter entries after this ISO-8601 date (e.g. "2024-01-15").
        end_date: Filter entries before this ISO-8601 date.
        author: Filter by author name (partial match).
        source_system: Filter by source system (exact match).
        advanced_params: Mode-specific parameters (e.g. similarity_threshold, temperature).

    Returns:
        JSON with answer (if RAG/agent), reasoning, sources, entry summaries,
        and workspace file path for full results.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "Empty search query.",
                ["Provide a search query describing what you are looking for."],
            )
        )

    try:
        from osprey.services.ariel_search.models import SearchMode

        registry = get_ariel_registry()
        service = await registry.service()

        # Parse mode
        mode_map = {
            "keyword": SearchMode.KEYWORD,
            "semantic": SearchMode.SEMANTIC,
            "rag": SearchMode.RAG,
            "agent": SearchMode.AGENT,
        }
        search_mode = mode_map.get(mode.lower())
        if search_mode is None:
            return json.dumps(
                make_error(
                    "validation_error",
                    f"Unknown search mode: {mode}",
                    [f"Valid modes: {', '.join(mode_map)}"],
                )
            )

        # Parse dates
        parsed_start, parsed_end = parse_date_filters(start_date, end_date)
        time_range = (parsed_start, parsed_end) if parsed_start or parsed_end else None

        # Build advanced_params with author/source_system filters
        adv = dict(advanced_params or {})
        if author:
            adv["author"] = author
        if source_system:
            adv["source_system"] = source_system

        # Execute search
        result = await service.search(
            query,
            max_results=max_results,
            time_range=time_range,
            mode=search_mode,
            advanced_params=adv,
        )

        # Serialize entries (TypedDict — use dict access)
        entries_out = [serialize_entry(e, text_limit=500) for e in result.entries]

        # Build full payload for data context
        full_payload = {
            "query": query,
            "mode": mode,
            "answer": result.answer,
            "reasoning": result.reasoning,
            "sources": list(result.sources),
            "search_modes_used": [m.value for m in result.search_modes_used],
            "total_results": len(entries_out),
            "entries": entries_out,
        }

        # Save via DataContext (unified with other MCP tools)
        from osprey.mcp_server.data_context import get_data_context

        ctx = get_data_context()
        entry = ctx.save(
            tool="ariel_search",
            data=full_payload,
            description=f"ARIEL search: {query}",
            summary={
                "query": query,
                "mode": mode,
                "results_found": len(entries_out),
                "top_sources": list(result.sources)[:3],
            },
            access_details={
                "search_modes_used": [m.value for m in result.search_modes_used],
                "has_answer": result.answer is not None,
                "entry_fields": [
                    "entry_id", "timestamp", "author",
                    "source_system", "raw_text", "summary",
                ],
            },
            data_type="search_results",
        )

        # Return compact DataContext response with answer/reasoning/sources
        response = entry.to_tool_response()
        response["query"] = query
        response["mode"] = mode
        response["results_found"] = len(entries_out)
        response["answer"] = result.answer
        response["reasoning"] = result.reasoning
        response["sources"] = list(result.sources)
        response["entries"] = entries_out[:max_results]

        return json.dumps(response, default=str)

    except Exception as exc:
        logger.exception("ariel_search failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"ARIEL search failed: {exc}",
                [
                    "Check ARIEL service configuration in config.yml.",
                    "Verify the ARIEL database is reachable.",
                ],
            )
        )
