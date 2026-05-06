"""MCP tool: channel_read — read one or more control-system channels."""

import json
import logging

from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.tools.channel_read")


@mcp.tool()
async def channel_read(
    channels: list[str],
    include_metadata: bool = True,
) -> str:
    """Read current values from one or more control-system channels.

    Args:
        channels: List of channel/PV addresses to read.
        include_metadata: If True, include units, alarm status, and limits alongside values.

    Returns:
        JSON with a compact summary of channel values and a data file path for full details.
    """
    if not channels:
        return make_error(
                "validation_error",
                "No channels provided.",
                ["Provide at least one channel address."],
            )

    async with connector_error_handler("channel_read"):
        from osprey.mcp_server.control_system.server_context import get_server_context

        registry = get_server_context()
        connector = await registry.control_system()

        if len(channels) == 1:
            cv = await connector.read_channel(channels[0])
            readings = {channels[0]: cv}
        else:
            readings = await connector.read_multiple_channels(channels)

        results = {}
        for addr, cv in readings.items():
            entry: dict = {"value": cv.value, "timestamp": str(cv.timestamp)}
            if include_metadata and cv.metadata:
                md = cv.metadata
                entry["metadata"] = {
                    "units": md.units,
                    "precision": md.precision,
                    "alarm_status": md.alarm_status,
                    "description": md.description,
                    "min_value": md.min_value,
                    "max_value": md.max_value,
                }
            results[addr] = entry

        readings_summary: dict = {}
        for addr, cv in readings.items():
            readings_summary[addr] = {
                "value": cv.value,
                "timestamp": str(cv.timestamp),
            }
            if cv.metadata and hasattr(cv.metadata, "units"):
                readings_summary[addr]["units"] = cv.metadata.units

        summary = {
            "channels_read": len(results),
            "readings": readings_summary,
        }
        access_details = {
            "fields_per_entry": (
                ["value", "timestamp"] + (["metadata"] if include_metadata else [])
            ),
        }

        # Return ephemeral result (no persistent storage for channel reads)
        return json.dumps(
            {
                "status": "success",
                "description": f"Read {len(results)} channel(s)",
                "summary": summary,
                "access_details": access_details,
            },
            default=str,
        )
