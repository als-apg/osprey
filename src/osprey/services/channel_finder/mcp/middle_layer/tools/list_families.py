"""MCP tool: cf_ml_list_families — list device families in a system.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_middle_layer_prompt_builder()
  Facility-customizable: tool description, system name examples (e.g., "SR" for Storage Ring)
"""

import json
import logging

from osprey.services.channel_finder.mcp.middle_layer.registry import get_cf_ml_registry
from osprey.services.channel_finder.mcp.middle_layer.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.middle_layer.tools.list_families")


@mcp.tool()
def cf_ml_list_families(system: str) -> str:
    """List all device families in a system with their descriptions.

    Args:
        system: System name (e.g., "SR" for Storage Ring).

    Returns:
        JSON with list of families and total count.
    """
    try:
        registry = get_cf_ml_registry()
        families = registry.database.list_families(system)

        return json.dumps({"families": families, "total": len(families)})

    except ValueError as exc:
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                ["Use cf_ml_list_systems to see available systems."],
            )
        )
    except Exception as exc:
        logger.exception("cf_ml_list_families failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to list families: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
