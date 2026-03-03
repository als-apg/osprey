"""MCP tools: data_list, data_read, data_delete.

Provides Claude with structured access to the OSPREY data store —
listing entries, reading full data content, and deleting entries.

All data is stored in the unified ArtifactStore.
"""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.data_context_tools")


@mcp.tool()
async def data_list(
    tool_filter: str | None = None,
    category_filter: str | None = None,
    last_n: int | None = None,
    source_agent_filter: str | None = None,
) -> str:
    """List all data currently available in the OSPREY workspace.

    This is the primary way to see what data has been collected by OSPREY
    tools. Each entry includes a compact summary and metadata.

    Args:
        tool_filter: Only show entries from this tool (e.g. "archiver_read").
        category_filter: Only show entries of this category (e.g. "archiver_data").
        last_n: Show only the most recent N entries.
        source_agent_filter: Only show entries from this agent
            (e.g. "logbook-search").

    Returns:
        JSON with the list of data entries.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entries = store.list_entries(
            tool_filter=tool_filter,
            category_filter=category_filter,
            last_n=last_n,
            source_agent_filter=source_agent_filter,
        )

        return json.dumps(
            {
                "total_entries": len(entries),
                "filters_applied": {
                    "tool": tool_filter,
                    "category": category_filter,
                    "last_n": last_n,
                    "source_agent": source_agent_filter,
                },
                "entries": [e.to_dict() for e in entries],
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("data_list failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to list data: {exc}",
                ["Check that the osprey-workspace directory is accessible."],
            )
        )


@mcp.tool()
async def data_read(entry_id: str) -> str:
    """Read the full data content of a data entry.

    Returns the complete JSON payload stored in the data file for the given
    entry. Use ``data_list`` first to discover available entry IDs.

    Args:
        entry_id: ID of the artifact/data entry to read.

    Returns:
        The full JSON data content of the entry.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.get_entry(entry_id)

        if entry is None:
            return json.dumps(
                make_error(
                    "not_found",
                    f"Data entry '{entry_id}' not found.",
                    ["Use data_list to see available entries."],
                )
            )

        # Get the data file path
        file_path = store.get_file_path(entry_id)
        if file_path is None or not file_path.exists():
            return json.dumps(
                make_error(
                    "file_not_found",
                    f"Data file for entry '{entry_id}' is missing from disk.",
                    [
                        "The data file may have been deleted externally.",
                        "Use data_list to find other entries.",
                    ],
                )
            )

        # Guard against very large files (>5 MB)
        size = file_path.stat().st_size
        if size > 5 * 1024 * 1024:
            return json.dumps(
                make_error(
                    "file_too_large",
                    f"Data file for entry '{entry_id}' is {size:,} bytes (>5 MB).",
                    [
                        "Read the file directly for large datasets.",
                        f"File path: {file_path}",
                    ],
                )
            )

        content = file_path.read_text(encoding="utf-8")
        return content

    except Exception as exc:
        logger.exception("data_read failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to read data entry: {exc}",
                ["Check that the osprey-workspace directory is accessible."],
            )
        )


@mcp.tool()
async def data_delete(entry_id: str) -> str:
    """Delete a data entry from the OSPREY workspace.

    Removes both the data file and its index entry.

    Args:
        entry_id: ID of the data entry to delete.

    Returns:
        JSON confirmation of deletion.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        deleted = store.delete_entry(entry_id)

        if not deleted:
            return json.dumps(
                make_error(
                    "not_found",
                    f"Data entry '{entry_id}' not found.",
                    ["Check the entry_id from a previous tool response."],
                )
            )

        return json.dumps(
            {
                "status": "success",
                "entry_id": entry_id,
                "message": f"Data entry '{entry_id}' deleted.",
            }
        )

    except Exception as exc:
        logger.exception("data_delete failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to delete data entry: {exc}",
                ["Check that the osprey-workspace directory is accessible."],
            )
        )
