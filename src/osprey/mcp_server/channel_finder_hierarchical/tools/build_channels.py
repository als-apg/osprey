"""MCP tool: build_channels — build channel addresses from selections.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_hierarchical_prompt_builder()
  Facility-customizable: selection dict examples, level name examples
"""

import json
import logging

from fastmcp.exceptions import ToolError

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
            Example: {"machine": "SR", "sector": "SECT1", "class": "Gauge", "device": "SR/GAUGESR01", "leaf": "vacuum_readback"}
            All levels from the hierarchy must be included — omitting a required level returns a validation error naming it.

    Returns:
        JSON with:
        - channels: the addresses that exist in the database, each once
        - total: their count
        - invalid_count: number of selection combinations that hold no channel
        - invalid_examples: up to 10 of those combinations, as level -> selected value.
          They are selections, not addresses missing from the database: a list at
          several levels is combined in every way, and a combination that names no
          leaf (a device under a sector it is not in) holds nothing.
    """
    try:
        registry = get_cf_hier_context()
        return json.dumps(registry.database.build_existing_channels(selections))

    except ValueError as exc:
        return make_error(
            "validation_error",
            str(exc),
            [
                "Use get_options to discover hierarchy levels and valid options.",
            ],
        )
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("build_channels failed")
        return make_error(
            "internal_error",
            f"Failed to build channels: {exc}",
            ["Check that the channel finder database is configured."],
        )
