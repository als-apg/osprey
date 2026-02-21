"""MCP tool: build_channels — build channel addresses from selections.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_hierarchical_prompt_builder()
  Facility-customizable: selection dict examples, level name examples
"""

import json
import logging

from osprey.services.channel_finder.mcp.hierarchical.registry import get_cf_hier_registry
from osprey.services.channel_finder.mcp.hierarchical.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.hierarchical.tools.build_channels")


@mcp.tool()
def build_channels(selections: dict) -> str:
    """Build channel addresses from hierarchy selections.

    After navigating the hierarchy with get_options, use this tool
    to construct the final channel addresses from your selections.

    Args:
        selections: Dict mapping level names to selected values.
            Values can be strings or lists of strings for multi-select.
            Example: {"system": "SR", "family": "BPM", "device": "01"}

    Returns:
        JSON with list of constructed channel addresses and total count.
    """
    try:
        registry = get_cf_hier_registry()
        db = registry.database

        channels = db.build_channels_from_selections(selections)

        result = {
            "channels": channels,
            "total": len(channels),
        }
        return json.dumps(result)

    except ValueError as exc:
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                [
                    "Use hierarchy_info to see required hierarchy levels.",
                    "Use get_options to see valid options at each level.",
                ],
            )
        )
    except Exception as exc:
        logger.exception("build_channels failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to build channels: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
