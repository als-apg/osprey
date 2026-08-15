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
        include_metadata: If True, also report units, precision, alarm status and
            description for each channel. Use channel_limits for the bounds a write
            is checked against - the control system does not report those here.

    Returns:
        JSON with a summary of channel values, one entry per channel.
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

        # Only fields the connectors actually populate. No connector reports the
        # channel's display range, so this tool does not claim to either — and a
        # write bound is a limits-database question, not a control-system read.
        metadata_fields = ("units", "precision", "alarm_status", "description")

        readings_summary: dict = {}
        for addr, cv in readings.items():
            entry: dict = {"value": cv.value, "timestamp": str(cv.timestamp)}
            if include_metadata:
                for field in metadata_fields:
                    entry[field] = getattr(cv.metadata, field, None)
            readings_summary[addr] = entry

        summary = {
            "channels_read": len(readings_summary),
            "readings": readings_summary,
        }
        access_details = {
            "fields_per_entry": (
                ["value", "timestamp"] + (list(metadata_fields) if include_metadata else [])
            ),
        }

        # Return ephemeral result (no persistent storage for channel reads)
        return json.dumps(
            {
                "status": "success",
                "description": f"Read {len(readings_summary)} channel(s)",
                "summary": summary,
                "access_details": access_details,
            },
            default=str,
        )
