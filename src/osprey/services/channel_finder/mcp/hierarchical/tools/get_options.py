"""MCP tool: cf_hier_get_options — get available options at a hierarchy level.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_hierarchical_prompt_builder()
  Facility-customizable: level name examples, selection dict examples
"""

import json
import logging

from osprey.services.channel_finder.mcp.hierarchical.registry import get_cf_hier_registry
from osprey.services.channel_finder.mcp.hierarchical.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.hierarchical.tools.get_options")


@mcp.tool()
def cf_hier_get_options(level: str, selections: dict | None = None) -> str:
    """Get available options at a specific hierarchy level.

    Use cf_hier_hierarchy_info first to learn the level names and order.
    Then call this tool iteratively, passing previous selections to drill down.

    Args:
        level: Hierarchy level name to get options for (e.g., "system", "device").
        selections: Dict mapping previous level names to selected values.
            Example: {"system": "SR"} when querying the "family" level.

    Returns:
        JSON with level name, list of options (name + description), and total count.
    """
    try:
        registry = get_cf_hier_registry()
        db = registry.database

        options = db.get_options_at_level(level, selections or {})

        return json.dumps(
            {
                "level": level,
                "options": options,
                "total": len(options),
            }
        )

    except ValueError as exc:
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                [
                    "Use cf_hier_hierarchy_info to see available hierarchy levels.",
                    "Ensure previous level selections are valid.",
                ],
            )
        )
    except Exception as exc:
        logger.exception("cf_hier_get_options failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get options: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
