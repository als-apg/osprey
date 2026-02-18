"""MCP tool: papers_stats — pre-computed database statistics."""

import json
import logging

from osprey.mcp_server.accelpapers.db import get_connection
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.common import make_error

logger = logging.getLogger("osprey.mcp_server.accelpapers.tools.stats")


@mcp.tool()
async def papers_stats() -> str:
    """Get summary statistics for the accelerator physics paper database.

    Returns pre-computed statistics including total papers, year range,
    conference count, author count, citation totals, and document type breakdown.
    This is an instant response — no search required.

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
                    ["Run the indexer first: python -m osprey.mcp_server.accelpapers index"],
                )
            )

        stats = {}
        for row in rows:
            key = row["key"]
            value = row["value"]
            # Parse JSON values (document_types, top_conferences)
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
        logger.exception("papers_stats failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Stats retrieval failed: {exc}",
                ["Check that the AccelPapers database has been indexed."],
            )
        )
