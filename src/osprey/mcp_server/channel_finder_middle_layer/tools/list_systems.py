"""MCP tool: list_systems — list all systems in the channel database.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_middle_layer_prompt_builder()
  Facility-customizable: tool description with facility-specific system examples
"""

import json
import logging

from osprey.mcp_server.channel_finder_middle_layer.server import make_error, mcp
from osprey.mcp_server.channel_finder_middle_layer.server_context import get_cf_ml_context

logger = logging.getLogger("osprey.mcp_server.channel_finder_middle_layer.tools.list_systems")


@mcp.tool()
def list_systems() -> str:
    """List all systems in the channel database with their descriptions.

    Returns:
        JSON with list of systems and total count.
    """
    try:
        registry = get_cf_ml_context()
        systems = registry.database.list_systems()

        return json.dumps({"systems": systems, "total": len(systems)})

    except Exception as exc:
        logger.exception("list_systems failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to list systems: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
