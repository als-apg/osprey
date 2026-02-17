"""MCP tool: cf_ic_validate -- validate channel names against the database."""

import json
import logging

from osprey.services.channel_finder.mcp.in_context.registry import get_cf_ic_registry
from osprey.services.channel_finder.mcp.in_context.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.in_context.tools.validate")


@mcp.tool()
def cf_ic_validate(channels: list[str]) -> str:
    """Validate that channel names exist in the database.

    Args:
        channels: List of channel names to validate.

    Returns:
        JSON with validation results for each channel, including
        counts of valid/invalid channels and the detailed results.
    """
    if not channels:
        return json.dumps(
            make_error(
                "validation_error",
                "Empty channel list.",
                ["Provide at least one channel name to validate."],
            )
        )

    try:
        registry = get_cf_ic_registry()
        db = registry.database

        results = db.validate_channels(channels)
        valid = db.get_valid_channels(results)
        invalid = db.get_invalid_channels(results)

        return json.dumps(
            {
                "total": len(channels),
                "valid_count": len(valid),
                "invalid_count": len(invalid),
                "valid_channels": valid,
                "invalid_channels": invalid,
                "results": results,
            }
        )

    except ValueError as exc:
        return json.dumps(make_error("validation_error", str(exc)))
    except Exception as exc:
        logger.exception("cf_ic_validate failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to validate channels: {exc}",
                [
                    "Check that the channel database is loaded correctly.",
                    "Verify config.yml channel_finder.pipelines.in_context.database settings.",
                ],
            )
        )
