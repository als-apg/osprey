"""MCP tool: cf_ml_list_channels — get channel names for a system/family/field path.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_middle_layer_prompt_builder()
  Facility-customizable: hierarchy navigation description, field/subfield examples,
  parameter descriptions
"""

import json
import logging

from osprey.services.channel_finder.mcp.middle_layer.registry import get_cf_ml_registry
from osprey.services.channel_finder.mcp.middle_layer.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.middle_layer.tools.list_channels")


@mcp.tool()
def cf_ml_list_channels(
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
        registry = get_cf_ml_registry()
        channels = registry.database.list_channel_names(
            system, family, field, subfield, sectors, devices
        )

        result = {"channels": channels, "total": len(channels)}

        try:
            from osprey.mcp_server.data_context import get_data_context

            selections = {
                "system": system,
                "family": family,
                "field": field,
            }
            if subfield is not None:
                selections["subfield"] = subfield
            if sectors is not None:
                selections["sectors"] = sectors
            if devices is not None:
                selections["devices"] = devices

            data_ctx = get_data_context()
            entry = data_ctx.save(
                tool="channel_find",
                data={"selections": selections, "channels": channels},
                description=f"Found {len(channels)} channel(s) via middle-layer lookup",
                summary={
                    "channels_found": len(channels),
                    "selections": selections,
                    "channels": channels[:10],
                },
                access_details={
                    "format": "channel_list",
                    "fields": ["channels"],
                    "pipeline": "middle_layer",
                },
                data_type="channel_addresses",
            )
            return json.dumps(entry.to_tool_response(), default=str)
        except Exception:
            logger.debug("DataContext save skipped (workspace not initialised)")
            return json.dumps(result)

    except ValueError as exc:
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                [
                    "Use cf_ml_list_systems to see available systems.",
                    "Use cf_ml_list_families to see families in a system.",
                    "Use cf_ml_inspect_fields to see available fields.",
                ],
            )
        )
    except Exception as exc:
        logger.exception("cf_ml_list_channels failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to list channels: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
