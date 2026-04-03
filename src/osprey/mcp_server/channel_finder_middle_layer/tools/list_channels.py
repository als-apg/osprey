"""MCP tool: list_channels — get channel names for a system/family/field path.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_middle_layer_prompt_builder()
  Facility-customizable: hierarchy navigation description, field/subfield examples,
  parameter descriptions
"""

import json
import logging

from osprey.mcp_server.channel_finder_middle_layer.server import make_error, mcp
from osprey.mcp_server.channel_finder_middle_layer.server_context import get_cf_ml_context

logger = logging.getLogger("osprey.mcp_server.channel_finder_middle_layer.tools.list_channels")


@mcp.tool()
def list_channels(
    system: str,
    family: str,
    field: str,
    subfield: str | None = None,
    sectors: list[int] | None = None,
    devices: list[int] | None = None,
) -> str:
    """Get channel names for a specific system/family/field path.

    Navigate the hierarchy (System -> Family -> Field -> Subfield) to get
    the actual EPICS PV channel names. Optionally filter by sector and/or
    device number.

    Args:
        system: System name (e.g., "SR").
        family: Family name (e.g., "BPM").
        field: Field name (e.g., "Monitor", "Setpoint").
        subfield: Optional subfield name for nested structures (e.g., "X", "Y").
        sectors: Optional list of sector numbers to filter by.
        devices: Optional list of device numbers to filter by.

    Returns:
        JSON with list of channel names and total count.
    """
    try:
        registry = get_cf_ml_context()
        channels = registry.database.list_channel_names(
            system, family, field, subfield, sectors, devices
        )

        result = {"channels": channels, "total": len(channels)}
        return json.dumps(result)

    except ValueError as exc:
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                [
                    "Use list_systems to see available systems.",
                    "Use list_families to see families in a system.",
                    "Use inspect_fields to see available fields.",
                ],
            )
        )
    except Exception as exc:
        logger.exception("list_channels failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to list channels: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
