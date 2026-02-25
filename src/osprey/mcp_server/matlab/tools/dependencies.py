"""MCP tool: mml_dependencies — recursive call graph traversal."""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.matlab.db import get_connection
from osprey.mcp_server.matlab.server import mcp

logger = logging.getLogger("osprey.mcp_server.matlab.tools.dependencies")

VALID_DIRECTIONS = {"callers", "callees", "both"}


@mcp.tool()
async def mml_dependencies(
    function_name: str,
    direction: str = "both",
    depth: int = 1,
) -> str:
    """Get the dependency graph for a function using recursive traversal.

    Traces callers (who calls this function) and/or callees (what this function
    calls) up to a specified depth. Self-loops are filtered by default.

    Args:
        function_name: The function to trace dependencies for.
        direction: "callers", "callees", or "both" (default: both).
        depth: How many levels deep to traverse (1-3, default 1).

    Returns:
        JSON with dependency lists organized by depth level.
    """
    if not function_name or not function_name.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "Empty function name.",
                ["Provide a function name."],
            )
        )

    if direction not in VALID_DIRECTIONS:
        return json.dumps(
            make_error(
                "validation_error",
                f"Invalid direction: {direction}",
                ["Use 'callers', 'callees', or 'both'."],
            )
        )

    depth = max(1, min(3, depth))
    fn = function_name.strip()

    try:
        conn = get_connection()

        # Verify function exists
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

        result: dict = {
            "function_name": fn,
            "direction": direction,
            "depth": depth,
        }

        if direction in ("callees", "both"):
            callees_sql = """
                WITH RECURSIVE callees(fn, depth) AS (
                    SELECT callee, 1 FROM dependencies
                    WHERE caller = ? AND caller != callee
                    UNION
                    SELECT d.callee, c.depth + 1
                    FROM dependencies d JOIN callees c ON d.caller = c.fn
                    WHERE c.depth < ? AND d.caller != d.callee
                )
                SELECT DISTINCT fn, MIN(depth) as depth FROM callees
                GROUP BY fn ORDER BY depth, fn
            """
            rows = conn.execute(callees_sql, [fn, depth]).fetchall()
            result["callees"] = [{"function_name": r["fn"], "depth": r["depth"]} for r in rows]

        if direction in ("callers", "both"):
            callers_sql = """
                WITH RECURSIVE callers(fn, depth) AS (
                    SELECT caller, 1 FROM dependencies
                    WHERE callee = ? AND caller != callee
                    UNION
                    SELECT d.caller, c.depth + 1
                    FROM dependencies d JOIN callers c ON d.callee = c.fn
                    WHERE c.depth < ? AND d.caller != d.callee
                )
                SELECT DISTINCT fn, MIN(depth) as depth FROM callers
                GROUP BY fn ORDER BY depth, fn
            """
            rows = conn.execute(callers_sql, [fn, depth]).fetchall()
            result["callers"] = [{"function_name": r["fn"], "depth": r["depth"]} for r in rows]

        conn.close()

        return json.dumps(result)

    except Exception as exc:
        logger.exception("mml_dependencies failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Dependencies lookup failed: {exc}",
                ["Check that the MATLAB MML database has been indexed."],
            )
        )
