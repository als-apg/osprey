"""MCP tools: artifact_focus, context_focus, memory_focus.

Each tool brings a specific item into the Focus View in the Artifact Gallery.
These tools share the same pattern: look up an entry, return an error if not
found, POST a focus notification to the gallery, and return success.
"""

import json
import logging

from osprey.mcp_server.common import gallery_url, make_error, post_json
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.focus")


@mcp.tool()
async def artifact_focus(artifact_id: str) -> str:
    """Set the Focus View in the Artifact Gallery to display a specific artifact.

    Brings the given artifact into the gallery's Focus View, making it the
    prominently displayed item. Useful after saving an artifact to direct
    attention to it.

    Args:
        artifact_id: ID of the artifact to focus (from artifact_save response).

    Returns:
        JSON with status and gallery URL.
    """
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.get_entry(artifact_id)
    if not entry:
        return json.dumps(
            make_error(
                "not_found",
                f"Artifact '{artifact_id}' not found.",
                ["Check the artifact_id from a previous artifact_save response."],
            )
        )

    base_url = gallery_url()
    post_json(f"{base_url}/api/focus", {"artifact_id": artifact_id})

    return json.dumps(
        {
            "status": "success",
            "artifact_id": artifact_id,
            "title": entry.title,
            "gallery_url": f"{base_url}#focus",
        }
    )


@mcp.tool()
async def context_focus(entry_id: int) -> str:
    """Set the Focus View in the Artifact Gallery to display a specific context entry.

    Brings the given data context entry into the gallery's Focus View, making it
    the prominently displayed item. Useful after a tool produces data to direct
    attention to it.

    Args:
        entry_id: ID of the context entry to focus (from tool response context_entry_id).

    Returns:
        JSON with status and gallery URL.
    """
    from osprey.mcp_server.data_context import get_data_context

    ctx = get_data_context()
    entry = ctx.get_entry(entry_id)
    if not entry:
        return json.dumps(
            make_error(
                "not_found",
                f"Context entry {entry_id} not found.",
                ["Check the context_entry_id from a previous tool response."],
            )
        )

    base_url = gallery_url()
    post_json(f"{base_url}/api/context/focus", {"entry_id": entry_id})

    return json.dumps(
        {
            "status": "success",
            "entry_id": entry_id,
            "description": entry.description,
            "gallery_url": f"{base_url}#context",
        }
    )


@mcp.tool()
async def memory_focus(memory_id: int) -> str:
    """Set the Focus View in the Artifact Gallery to display a specific memory entry.

    Brings the given memory entry into the gallery's Focus View, making it
    the prominently displayed item. Useful after a tool produces a memory to direct
    attention to it.

    Args:
        memory_id: ID of the memory entry to focus (from tool response memory_id).

    Returns:
        JSON with status and gallery URL.
    """
    from osprey.mcp_server.memory_store import get_memory_store

    store = get_memory_store()
    entry = store.get_entry(memory_id)
    if not entry:
        return json.dumps(
            make_error(
                "not_found",
                f"Memory entry {memory_id} not found.",
                ["Check the memory_id from a previous tool response."],
            )
        )

    base_url = gallery_url()
    post_json(f"{base_url}/api/memory/focus", {"memory_id": memory_id})

    return json.dumps(
        {
            "status": "success",
            "memory_id": memory_id,
            "content": entry.content[:100],
            "gallery_url": f"{base_url}#memory/focus",
        }
    )
