"""MCP tool: mml_get — retrieve full function details by name."""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.matlab.db import get_connection
from osprey.mcp_server.matlab.server import mcp

logger = logging.getLogger("osprey.mcp_server.matlab.tools.get_function")


@mcp.tool()
async def mml_get(
    function_name: str,
    include_source: bool = True,
) -> str:
    """Get full details for a specific MML function by name.

    Returns metadata, docstring, source code, and immediate callers/callees
    from the dependency graph.

    Args:
        function_name: The function name (e.g. "getbpm", "setsp").
        include_source: Include source code (default True).

    Returns:
        JSON with function details and immediate dependencies.
    """
    if not function_name or not function_name.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "Empty function name.",
                ["Provide a function name."],
            )
        )

    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM functions WHERE function_name = ?",
            [function_name.strip()],
        ).fetchone()

        if row is None:
            conn.close()
            return json.dumps(
                make_error(
                    "not_found",
                    f"Function not found: {function_name}",
                    [
                        "Check the function name spelling.",
                        "Use mml_search to find functions.",
                    ],
                )
            )

        result = {
            "function_name": row["function_name"],
            "file_path": row["file_path"],
            "docstring": row["docstring"],
            "group": row["group_name"],
            "type": row["type"],
            "in_degree": row["in_degree"],
            "out_degree": row["out_degree"],
        }

        if include_source and row["source_code"]:
            result["source_code"] = row["source_code"]

        # Immediate callers (functions that call this one)
        callers = conn.execute(
            "SELECT caller FROM dependencies WHERE callee = ? ORDER BY caller",
            [function_name.strip()],
        ).fetchall()
        result["callers"] = [r["caller"] for r in callers]

        # Immediate callees (functions called by this one)
        callees = conn.execute(
            "SELECT callee FROM dependencies WHERE caller = ? ORDER BY callee",
            [function_name.strip()],
        ).fetchall()
        result["callees"] = [r["callee"] for r in callees]

        conn.close()

        return json.dumps(result, default=str)

    except Exception as exc:
        logger.exception("mml_get failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to retrieve function: {exc}",
                ["Check that the MATLAB MML database has been indexed."],
            )
        )
