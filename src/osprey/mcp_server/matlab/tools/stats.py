"""MCP tool: mml_stats — pre-computed database statistics."""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.matlab.db import get_connection
from osprey.mcp_server.matlab.server import mcp

logger = logging.getLogger("osprey.mcp_server.matlab.tools.stats")


@mcp.tool()
async def mml_stats() -> str:
    """Get summary statistics for the MML function database.

    Returns pre-computed statistics including total functions, total edges,
    group breakdown, type breakdown, top called functions, and top dependent
    functions. This is an instant response — no search required.

    Returns:
        JSON with database statistics.
    """
    try:
        conn = get_connection()

        rows = conn.execute("SELECT key, value FROM stats").fetchall()
        if not rows:
            conn.close()
            return json.dumps(
                make_error(
                    "not_found",
                    "No statistics available — database may not be indexed.",
                    ["Run the indexer first: python -m osprey.mcp_server.matlab index"],
                )
            )

        stats = {}
        for row in rows:
            key = row["key"]
            value = row["value"]
            # Parse JSON values (groups, types, top_called, top_dependent)
            if value.startswith("{"):
                try:
                    stats[key] = json.loads(value)
                except json.JSONDecodeError:
                    stats[key] = value
            elif value.isdigit():
                stats[key] = int(value)
            else:
                stats[key] = value

        conn.close()

        return json.dumps({"statistics": stats})

    except Exception as exc:
        logger.exception("mml_stats failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Stats retrieval failed: {exc}",
                ["Check that the MATLAB MML database has been indexed."],
            )
        )
