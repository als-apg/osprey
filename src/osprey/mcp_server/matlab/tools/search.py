"""MCP tool: mml_search — FTS5 BM25-ranked search of MML functions."""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.matlab.db import get_connection
from osprey.mcp_server.matlab.server import mcp

logger = logging.getLogger("osprey.mcp_server.matlab.tools.search")


@mcp.tool()
async def mml_search(
    query: str,
    group: str | None = None,
    type: str | None = None,
    limit: int = 20,
) -> str:
    """Search MML functions using full-text search (BM25 ranking).

    Searches across function names, docstrings, source code, and group names.
    Returns results ranked by relevance with text snippets.

    Args:
        query: Search terms (e.g. "orbit correction", "BPM calibration").
        group: Filter by group (e.g. "StorageRing", "BTS", "Common").
        type: Filter by type ("defined" or "script").
        limit: Maximum results to return (1-100, default 20).

    Returns:
        JSON with matching functions, BM25 scores, and text snippets.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error("validation_error", "Empty search query.", ["Provide search terms."])
        )

    limit = max(1, min(100, limit))

    try:
        conn = get_connection()
        safe_query = query.strip()

        where_parts: list[str] = []
        params: list = []

        if group:
            where_parts.append("f.group_name = ?")
            params.append(group)
        if type:
            where_parts.append("f.type = ?")
            params.append(type)

        where_clause = ""
        if where_parts:
            where_clause = "AND " + " AND ".join(where_parts)

        sql = f"""
            SELECT f.function_name, f.group_name, f.type,
                   f.in_degree, f.out_degree,
                   snippet(functions_fts, 1, '>>>', '<<<', '...', 40) as snippet,
                   rank
            FROM functions_fts fts
            JOIN functions f ON f.rowid = fts.rowid
            WHERE functions_fts MATCH ?
            {where_clause}
            ORDER BY rank
            LIMIT ?
        """

        rows = conn.execute(sql, [safe_query] + params + [limit]).fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "function_name": row["function_name"],
                    "group": row["group_name"],
                    "type": row["type"],
                    "in_degree": row["in_degree"],
                    "out_degree": row["out_degree"],
                    "snippet": row["snippet"],
                    "bm25_score": round(row["rank"], 4),
                }
            )

        conn.close()

        return json.dumps(
            {
                "query": query,
                "filters": {
                    k: v for k, v in {"group": group, "type": type}.items() if v is not None
                },
                "results_found": len(results),
                "functions": results,
            }
        )

    except Exception as exc:
        logger.exception("mml_search failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Search failed: {exc}",
                ["Check that the MATLAB MML database has been indexed."],
            )
        )
