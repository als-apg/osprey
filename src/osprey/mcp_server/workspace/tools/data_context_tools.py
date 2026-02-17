"""MCP tool: data_context_list — query the OSPREY data context index.

Provides Claude with a structured way to see what data has been collected
across all tool invocations, with optional filtering by tool or data type.
"""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.data_context_tools")


@mcp.tool()
async def data_context_list(
    tool_filter: str | None = None,
    data_type_filter: str | None = None,
    last_n: int | None = None,
) -> str:
    """List all data currently available in the OSPREY workspace.

    This is the primary way to see what data has been collected by OSPREY
    tools. Each entry includes a compact summary and the file path where
    full data can be read.

    Args:
        tool_filter: Only show entries from this tool (e.g. "archiver_read").
        data_type_filter: Only show entries of this type (e.g. "timeseries").
        last_n: Show only the most recent N entries.

    Returns:
        JSON with the list of data context entries.
    """
    try:
        from osprey.mcp_server.data_context import get_data_context

        ctx = get_data_context()
        entries = ctx.list_entries(
            tool_filter=tool_filter,
            data_type_filter=data_type_filter,
            last_n=last_n,
        )

        return json.dumps(
            {
                "total_entries": len(entries),
                "filters_applied": {
                    "tool": tool_filter,
                    "data_type": data_type_filter,
                    "last_n": last_n,
                },
                "entries": [e.to_dict() for e in entries],
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("data_context_list failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to list data context: {exc}",
                ["Check that the osprey-workspace directory is accessible."],
            )
        )


@mcp.tool()
async def data_context_delete(entry_id: int) -> str:
    """Delete a data context entry from the OSPREY workspace.

    Removes both the data file and its index entry.

    Args:
        entry_id: ID of the data context entry to delete.

    Returns:
        JSON confirmation of deletion.
    """
    try:
        from osprey.mcp_server.data_context import get_data_context

        ctx = get_data_context()
        deleted = ctx.delete_entry(entry_id)

        if not deleted:
            return json.dumps(
                make_error(
                    "not_found",
                    f"Data context entry {entry_id} not found.",
                    ["Check the entry_id from a previous tool response."],
                )
            )

        return json.dumps(
            {
                "status": "success",
                "entry_id": entry_id,
                "message": f"Data context entry {entry_id} deleted.",
            }
        )

    except Exception as exc:
        logger.exception("data_context_delete failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to delete data context entry: {exc}",
                ["Check that the osprey-workspace directory is accessible."],
            )
        )
