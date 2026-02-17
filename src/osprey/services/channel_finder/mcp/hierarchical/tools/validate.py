"""MCP tool: cf_hier_validate — validate channel names against the database."""

import json
import logging

from osprey.services.channel_finder.mcp.hierarchical.registry import get_cf_hier_registry
from osprey.services.channel_finder.mcp.hierarchical.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.hierarchical.tools.validate")


@mcp.tool()
def cf_hier_validate(channels: list[str]) -> str:
    """Validate that channel names exist in the database.

    Checks each provided channel name against the hierarchical database
    and reports which are valid and which are not found.

    Args:
        channels: List of channel names to validate.

    Returns:
        JSON with validation results per channel and summary counts.
    """
    try:
        registry = get_cf_hier_registry()
        db = registry.database

        results = []
        valid_count = 0
        invalid_count = 0

        for channel in channels:
            is_valid = db.validate_channel(channel)
            results.append({"channel": channel, "valid": is_valid})
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

        return json.dumps(
            {
                "results": results,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "total": len(channels),
            }
        )

    except ValueError as exc:
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                ["Provide a list of channel name strings to validate."],
            )
        )
    except Exception as exc:
        logger.exception("cf_hier_validate failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to validate channels: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
