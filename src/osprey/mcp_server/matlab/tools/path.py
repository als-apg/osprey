"""MCP tool: mml_path — BFS shortest path between functions."""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.matlab.db import get_connection
from osprey.mcp_server.matlab.server import mcp

logger = logging.getLogger("osprey.mcp_server.matlab.tools.path")


@mcp.tool()
async def mml_path(
    source: str,
    target: str,
    max_depth: int = 10,
) -> str:
    """Find the shortest call path between two functions.

    Uses BFS via recursive CTE to find the shortest chain of function calls
    connecting source to target. Self-loops and cycles are excluded.

    Args:
        source: Starting function name.
        target: Target function name.
        max_depth: Maximum path length to search (1-10, default 10).

    Returns:
        JSON with the shortest path or indication that no path exists.
    """
    if not source or not source.strip():
        return json.dumps(
            make_error("validation_error", "Empty source function name.", ["Provide a source."])
        )
    if not target or not target.strip():
        return json.dumps(
            make_error("validation_error", "Empty target function name.", ["Provide a target."])
        )

    source = source.strip()
    target = target.strip()
    max_depth = max(1, min(10, max_depth))

    if source == target:
        return json.dumps(
            {
                "source": source,
                "target": target,
                "path": [source],
                "length": 0,
            }
        )

    try:
        conn = get_connection()

        # Verify both functions exist
        for fn in (source, target):
            row = conn.execute(
                "SELECT function_name FROM functions WHERE function_name = ?",
                [fn],
            ).fetchone()
            if row is None:
                conn.close()
                return json.dumps(
                    make_error(
                        "not_found",
                        f"Function not found: {fn}",
                        ["Check spelling.", "Use mml_search to find functions."],
                    )
                )

        # BFS shortest path via recursive CTE
        path_sql = """
            WITH RECURSIVE paths(fn, path, depth) AS (
                SELECT ?, ?, 0
                UNION ALL
                SELECT d.callee, p.path || ',' || d.callee, p.depth + 1
                FROM dependencies d JOIN paths p ON d.caller = p.fn
                WHERE p.depth < ? AND d.caller != d.callee
                  AND INSTR(p.path, d.callee) = 0
            )
            SELECT path FROM paths WHERE fn = ? ORDER BY depth LIMIT 1
        """

        row = conn.execute(path_sql, [source, source, max_depth, target]).fetchone()
        conn.close()

        if row is None:
            return json.dumps(
                {
                    "source": source,
                    "target": target,
                    "path": None,
                    "length": None,
                    "message": f"No path found within {max_depth} steps.",
                }
            )

        path_list = row["path"].split(",")
        return json.dumps(
            {
                "source": source,
                "target": target,
                "path": path_list,
                "length": len(path_list) - 1,
            }
        )

    except Exception as exc:
        logger.exception("mml_path failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Path search failed: {exc}",
                ["Check that the MATLAB MML database has been indexed."],
            )
        )
