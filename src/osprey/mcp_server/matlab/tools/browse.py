"""MCP tool: mml_browse — browse functions by group/type with sorting."""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.matlab.db import get_connection
from osprey.mcp_server.matlab.server import mcp

logger = logging.getLogger("osprey.mcp_server.matlab.tools.browse")

VALID_SORT_FIELDS = {"in_degree", "out_degree", "function_name"}
VALID_SORT_ORDERS = {"asc", "desc"}


@mcp.tool()
async def mml_browse(
    group: str | None = None,
    type: str | None = None,
    sort_by: str = "in_degree",
    order: str = "desc",
    limit: int = 20,
    offset: int = 0,
) -> str:
    """Browse MML functions by group or type with sorting.

    Unlike mml_search, this does NOT use full-text search — it filters on
    indexed columns for fast, structured browsing. At least one filter is required.

    Args:
        group: Filter by group (e.g. "StorageRing", "BTS", "Common", "MML").
        type: Filter by type ("defined" or "script").
        sort_by: Sort field: "in_degree", "out_degree", "function_name" (default: in_degree).
        order: Sort direction: "asc" or "desc" (default: desc).
        limit: Maximum results (1-100, default 20).
        offset: Skip first N results for pagination (default 0).

    Returns:
        JSON with matching functions and total count.
    """
    if not any([group, type]):
        return json.dumps(
            make_error(
                "validation_error",
                "At least one filter is required for browsing.",
                [
                    "Specify group or type.",
                    "Use mml_search for full-text search instead.",
                ],
            )
        )

    if sort_by not in VALID_SORT_FIELDS:
        sort_by = "in_degree"
    if order.lower() not in VALID_SORT_ORDERS:
        order = "desc"
    limit = max(1, min(100, limit))
    offset = max(0, offset)

    try:
        conn = get_connection()

        where_parts: list[str] = []
        params: list = []

        if group:
            where_parts.append("group_name = ?")
            params.append(group)
        if type:
            where_parts.append("type = ?")
            params.append(type)

        where_clause = "WHERE " + " AND ".join(where_parts)

        # Get total count
        count_row = conn.execute(
            f"SELECT COUNT(*) as c FROM functions {where_clause}", params
        ).fetchone()
        total = count_row["c"]

        # Get results
        sql = f"""
            SELECT function_name, group_name, type,
                   in_degree, out_degree, docstring
            FROM functions
            {where_clause}
            ORDER BY {sort_by} {order}
            LIMIT ? OFFSET ?
        """

        rows = conn.execute(sql, params + [limit, offset]).fetchall()

        results = []
        for row in rows:
            docstring = row["docstring"] or ""
            if len(docstring) > 200:
                docstring = docstring[:200] + "..."
            results.append(
                {
                    "function_name": row["function_name"],
                    "group": row["group_name"],
                    "type": row["type"],
                    "in_degree": row["in_degree"],
                    "out_degree": row["out_degree"],
                    "docstring": docstring,
                }
            )

        conn.close()

        return json.dumps(
            {
                "filters": {
                    k: v for k, v in {"group": group, "type": type}.items() if v is not None
                },
                "sort_by": sort_by,
                "order": order,
                "total_matching": total,
                "offset": offset,
                "results_returned": len(results),
                "functions": results,
            }
        )

    except Exception as exc:
        logger.exception("mml_browse failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Browse failed: {exc}",
                ["Check that the MATLAB MML database has been indexed."],
            )
        )
