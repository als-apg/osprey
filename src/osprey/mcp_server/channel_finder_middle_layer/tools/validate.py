"""MCP tool: validate — validate that channel names exist in the database."""

import json
import logging

from osprey.mcp_server.channel_finder_middle_layer.registry import get_cf_ml_registry
from osprey.mcp_server.channel_finder_middle_layer.server import make_error, mcp

logger = logging.getLogger("osprey.mcp_server.channel_finder_middle_layer.tools.validate")


@mcp.tool()
def validate(channels: list[str]) -> str:
    """Validate that channel names exist in the database.

    Checks each channel name against the database and reports whether
    it is valid (exists) or not.

    Args:
        channels: List of channel names to validate.

    Returns:
        JSON with validation results for each channel.
    """
    if not channels:
        return json.dumps(
            make_error(
                "validation_error",
                "Empty channel list provided.",
                ["Provide one or more channel names to validate."],
            )
        )

    try:
        registry = get_cf_ml_registry()
        results = []
        for ch in channels:
            results.append(
                {
                    "channel": ch,
                    "valid": registry.database.validate_channel(ch),
                }
            )

        return json.dumps({"results": results, "total": len(results)})

    except Exception as exc:
        logger.exception("validate failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to validate channels: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
