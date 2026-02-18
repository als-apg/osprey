"""MCP tool: mml_list_groups — list groups with function counts."""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.matlab.db import get_connection
from osprey.mcp_server.matlab.server import mcp

logger = logging.getLogger("osprey.mcp_server.matlab.tools.list_groups")


@mcp.tool()
async def mml_list_groups() -> str:
    """List all groups in the MML codebase with function counts.

    Returns groups sorted by function count descending.

    Returns:
        JSON with group names and function counts.
    """
    try:
        conn = get_connection()

        rows = conn.execute(
            """
            SELECT group_name, COUNT(*) as function_count
            FROM functions
            WHERE group_name != ''
            GROUP BY group_name
            ORDER BY function_count DESC
            """
        ).fetchall()

        groups = [
            {"group": row["group_name"], "function_count": row["function_count"]}
            for row in rows
        ]

        conn.close()

        return json.dumps({
            "groups_found": len(groups),
            "groups": groups,
        })

    except Exception as exc:
        logger.exception("mml_list_groups failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"List groups failed: {exc}",
                ["Check that the MATLAB MML database has been indexed."],
            )
        )
