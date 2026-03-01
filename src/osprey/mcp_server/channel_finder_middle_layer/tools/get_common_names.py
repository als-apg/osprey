"""MCP tool: get_common_names — get common/friendly names for devices.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_middle_layer_prompt_builder()
  Facility-customizable: device name examples (e.g., "BPM 1", "BPM 2")
"""

import json
import logging

from osprey.mcp_server.channel_finder_middle_layer.registry import get_cf_ml_registry
from osprey.mcp_server.channel_finder_middle_layer.server import make_error, mcp

logger = logging.getLogger("osprey.mcp_server.channel_finder_middle_layer.tools.get_common_names")


@mcp.tool()
def get_common_names(system: str, family: str) -> str:
    """Get common/friendly names for devices in a family.

    Returns the human-readable names (e.g., "BPM 1", "BPM 2") that correspond
    to the devices in a family, if available in the database.

    Args:
        system: System name (e.g., "SR").
        family: Family name (e.g., "BPM").

    Returns:
        JSON with list of common names, or null with a message if not available.
    """
    try:
        registry = get_cf_ml_registry()
        common_names = registry.database.get_common_names(system, family)

        if common_names is not None:
            return json.dumps({"common_names": common_names})
        else:
            return json.dumps(
                {
                    "common_names": None,
                    "message": (
                        f"No common names available for '{system}:{family}'. "
                        f"Not all families have CommonNames defined in the database."
                    ),
                }
            )

    except Exception as exc:
        logger.exception("get_common_names failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get common names: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
