"""MCP tool: build_channels — build channel addresses from selections.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_hierarchical_prompt_builder()
  Facility-customizable: selection dict examples, level name examples
"""

import json
import logging

from osprey.mcp_server.channel_finder_hierarchical.server import make_error, mcp
from osprey.mcp_server.channel_finder_hierarchical.server_context import get_cf_hier_context

logger = logging.getLogger("osprey.mcp_server.channel_finder_hierarchical.tools.build_channels")


@mcp.tool()
def build_channels(selections: dict) -> str:
    """Build channel addresses from hierarchy selections.

    After navigating the hierarchy with get_options, use this tool
    to construct the final channel addresses from your selections.

    Args:
        selections: Dict mapping level names to selected values.
            Values can be strings or lists of strings for multi-select.
            Example: {"ring": "SR", "system": "MAG", "family": "DIPOLE", "device": "B05", "field": "CURRENT", "subfield": "SP"}
            All levels from the hierarchy must be included — omitting a required level returns no channels.

    Returns:
        JSON with list of constructed channel addresses and total count.
    """
    try:
        registry = get_cf_hier_context()
        db = registry.database

        channels = db.build_channels_from_selections(selections)

        valid = []
        invalid = []
        for ch in channels:
            if db.validate_channel(ch):
                valid.append(ch)
            else:
                invalid.append(ch)

        result = {
            "channels": channels,
            "total": len(channels),
            "valid": valid,
            "invalid": invalid,
            "valid_count": len(valid),
            "invalid_count": len(invalid),
        }
        return json.dumps(result)

    except ValueError as exc:
        return make_error(
                "validation_error",
                str(exc),
                [
                    "Use get_options to discover hierarchy levels and valid options.",
                ],
            )
    except Exception as exc:
        logger.exception("build_channels failed")
        return make_error(
                "internal_error",
                f"Failed to build channels: {exc}",
                ["Check that the channel finder database is configured."],
            )
