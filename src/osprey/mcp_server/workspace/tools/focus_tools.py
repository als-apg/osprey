"""MCP tools: artifact_focus, artifact_pin.

artifact_focus brings a specific item into the Focus View in the Artifact Gallery.
artifact_pin toggles the pinned flag for quick access filtering.
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
async def artifact_pin(artifact_id: str, pinned: bool = True) -> str:
    """Pin or unpin an artifact for quick access in the gallery.

    Pinned artifacts appear at the top of the sidebar and can be filtered
    via the pin filter chip.

    Args:
        artifact_id: ID of the artifact to pin/unpin.
        pinned: True to pin, False to unpin. Defaults to True.

    Returns:
        JSON with status and updated pinned state.
    """
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.set_pinned(artifact_id, pinned)
    if not entry:
        return json.dumps(
            make_error(
                "not_found",
                f"Artifact '{artifact_id}' not found.",
                ["Check the artifact_id from a previous artifact_save response."],
            )
        )

    # Notify gallery of the update
    base_url = gallery_url()
    post_json(f"{base_url}/api/artifacts/{artifact_id}/pin", {"pinned": pinned})

    return json.dumps(
        {
            "status": "success",
            "artifact_id": artifact_id,
            "title": entry.title,
            "pinned": entry.pinned,
        }
    )
