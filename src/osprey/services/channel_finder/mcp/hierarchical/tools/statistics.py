"""MCP tool: statistics — get database statistics."""

import json
import logging

from osprey.services.channel_finder.mcp.hierarchical.registry import get_cf_hier_registry
from osprey.services.channel_finder.mcp.hierarchical.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.hierarchical.tools.statistics")


@mcp.tool()
def statistics() -> str:
    """Get database statistics including total channels and hierarchy levels.

    Returns:
        JSON with total channel count, hierarchy level names, and per-system breakdowns.
    """
    try:
        registry = get_cf_hier_registry()
        db = registry.database

        stats = db.get_statistics()

        return json.dumps(stats)

    except Exception as exc:
        logger.exception("statistics failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get statistics: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
