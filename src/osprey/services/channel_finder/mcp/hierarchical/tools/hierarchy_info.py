"""MCP tool: hierarchy_info — get hierarchy structure information.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_hierarchical_prompt_builder()
  Facility-customizable: level name examples, tree-vs-instances explanation
"""

import json
import logging

from osprey.services.channel_finder.mcp.hierarchical.registry import get_cf_hier_registry
from osprey.services.channel_finder.mcp.hierarchical.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.hierarchical.tools.hierarchy_info")


@mcp.tool()
def hierarchy_info() -> str:
    """Get hierarchy structure information: level names, types, and naming pattern.

    Returns the hierarchy levels, their configuration (tree vs instances, optional),
    and the naming pattern used to construct channel addresses.

    Returns:
        JSON with hierarchy_levels, hierarchy_config, naming_pattern, and facility_name.
    """
    try:
        registry = get_cf_hier_registry()
        db = registry.database

        return json.dumps(
            {
                "hierarchy_levels": db.hierarchy_levels,
                "hierarchy_config": db.hierarchy_config,
                "naming_pattern": db.naming_pattern,
                "facility_name": registry.facility_name,
            }
        )

    except Exception as exc:
        logger.exception("hierarchy_info failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get hierarchy info: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
