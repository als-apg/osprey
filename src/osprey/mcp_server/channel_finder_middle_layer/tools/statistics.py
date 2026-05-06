"""MCP tool: statistics — get database statistics."""

from fastmcp.exceptions import ToolError
import json
import logging

from osprey.mcp_server.channel_finder_middle_layer.server import make_error, mcp
from osprey.mcp_server.channel_finder_middle_layer.server_context import get_cf_ml_context

logger = logging.getLogger("osprey.mcp_server.channel_finder_middle_layer.tools.statistics")


@mcp.tool()
def statistics() -> str:
    """Get database statistics (total channels, systems, families).

    Returns:
        JSON with database statistics including total channel count,
        number of systems, and number of unique families.
    """
    try:
        registry = get_cf_ml_context()
        stats = registry.database.get_statistics()

        return json.dumps(stats)

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("statistics failed")
        return make_error(
                "internal_error",
                f"Failed to get statistics: {exc}",
                ["Check that the channel finder database is configured."],
            )
