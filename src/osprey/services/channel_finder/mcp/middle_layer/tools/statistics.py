"""MCP tool: statistics — get database statistics."""

import json
import logging

from osprey.services.channel_finder.mcp.middle_layer.registry import get_cf_ml_registry
from osprey.services.channel_finder.mcp.middle_layer.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.middle_layer.tools.statistics")


@mcp.tool()
def statistics() -> str:
    """Get database statistics (total channels, systems, families).

    Returns:
        JSON with database statistics including total channel count,
        number of systems, and number of unique families.
    """
    try:
        registry = get_cf_ml_registry()
        stats = registry.database.get_statistics()

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
