"""MCP tool: get_naming_patterns — return curated DB naming summary as context."""

import json
import logging

from osprey.mcp_server.direct_channel_finder.server import make_error, mcp

logger = logging.getLogger("osprey.mcp_server.direct_channel_finder.tools.get_naming_patterns")


@mcp.tool()
def get_naming_patterns() -> str:
    """Get a summary of PV naming conventions from the curated channel database.

    Returns a compact markdown summary of naming patterns, device types,
    and useful search examples. This helps you understand the PV namespace
    structure when searching for channels.

    Returns:
        JSON with naming_summary (markdown) and facility_name.
    """
    try:
        from osprey.mcp_server.direct_channel_finder.server_context import get_dcf_context

        registry = get_dcf_context()

        from osprey.services.channel_finder.utils.naming_summary import (
            generate_naming_summary,
        )

        summary = generate_naming_summary(registry.raw_config)

        return json.dumps(
            {
                "naming_summary": summary,
                "facility_name": registry.facility_name,
            }
        )

    except Exception as exc:
        logger.exception("get_naming_patterns failed")
        return json.dumps(
            make_error(
                "naming_patterns_error",
                f"Could not generate naming patterns: {exc}",
                suggestions=[
                    "This tool requires a curated channel database to be configured",
                    "The direct channel finder still works without this — use search_pvs",
                ],
            )
        )
