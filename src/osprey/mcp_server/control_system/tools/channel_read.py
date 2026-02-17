"""MCP tool: channel_read — read one or more control-system channels.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_channel_read_prompt_builder()
  Facility-customizable: tool description, parameter examples (PV name format),
  protocol terminology (EPICS vs Tango vs LabVIEW)
"""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.control_system.error_handling import ToolError, connector_error_handler
from osprey.mcp_server.control_system.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.channel_read")


@mcp.tool()
async def channel_read(
    channels: list[str],
    include_metadata: bool = True,
) -> str:
    """Read current values from one or more EPICS control-system channels.

    Args:
        channels: List of channel/PV addresses to read (e.g. ["SR:C01-MG:G01A{Quad:01}Fld-I"]).
        include_metadata: If True, include units, alarm status, and limits alongside values.

    Returns:
        JSON with a compact summary of channel values and a data file path for full details.
    """
    if not channels:
        return json.dumps(
            make_error(
                "validation_error",
                "No channels provided.",
                ["Provide at least one channel address."],
            )
        )

    try:
        async with connector_error_handler("channel_read"):
            from osprey.mcp_server.control_system.registry import get_mcp_registry

            registry = get_mcp_registry()
            connector = await registry.control_system()

            if len(channels) == 1:
                cv = await connector.read_channel(channels[0])
                readings = {channels[0]: cv}
            else:
                readings = await connector.read_multiple_channels(channels)

            # Build serialisable readings dict for the data file
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

            # Build compact summary inline
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

            # Save via DataContext
            from osprey.mcp_server.data_context import get_data_context

            data_ctx = get_data_context()
            entry = data_ctx.save(
                tool="channel_read",
                data={"channels_read": len(results), "readings": results},
                description=f"Read {len(results)} channel(s)",
                summary=summary,
                access_details=access_details,
                data_type="channel_values",
            )

            return json.dumps(entry.to_tool_response(), default=str)

    except ToolError as exc:
        return exc.response
