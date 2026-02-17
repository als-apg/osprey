"""MCP tool: cf_ic_statistics -- report database statistics."""

import json
import logging

from osprey.services.channel_finder.mcp.in_context.registry import get_cf_ic_registry
from osprey.services.channel_finder.mcp.in_context.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.in_context.tools.statistics")


@mcp.tool()
def cf_ic_statistics() -> str:
    """Get database statistics (total channels, format, chunk info).

    Returns:
        JSON with database statistics including total channels, database
        format, and the number of chunks at the default chunk size of 50.
    """
    try:
        registry = get_cf_ic_registry()
        db = registry.database

        stats = db.get_statistics()

        # Include chunking info at default chunk size
        chunks = db.chunk_database(50)
        stats["total_chunks_at_50"] = len(chunks)
        stats["facility_name"] = registry.facility_name

        return json.dumps(stats)

    except Exception as exc:
        logger.exception("cf_ic_statistics failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get statistics: {exc}",
                [
                    "Check that the channel database is loaded correctly.",
                    "Verify config.yml channel_finder.pipelines.in_context.database settings.",
                ],
            )
        )
