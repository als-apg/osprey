"""MCP tools: artifact_focus, artifact_pin.

artifact_focus selects a specific artifact in the gallery so the user sees it.
artifact_pin toggles the pinned flag for quick access filtering.
"""

import functools
import json
import logging

import anyio

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import gallery_url, notify_agent_activity, post_json
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.focus")


@mcp.tool()
async def artifact_focus(artifact_id: str, fullscreen: bool = False) -> str:
    """Select an artifact in the gallery so the user sees it.

    Sets the gallery's current selection to the given artifact, scrolling
    it into view and showing the preview pane. Also updates focus_state.txt
    so the agent-awareness hook includes this artifact in context.

    Args:
        artifact_id: ID of the artifact to select (from artifact_save response).
        fullscreen: If True, open the artifact in immersive fullscreen mode
            (hides sidebar, header, filters). Defaults to False.

    Returns:
        JSON with status and gallery URL.
    """
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.get_entry(artifact_id)
    if not entry:
        return make_error(
            "not_found",
            f"Artifact '{artifact_id}' not found.",
            ["Check the artifact_id from a previous artifact_save response."],
        )

    base_url = gallery_url()
    payload = {"artifact_id": artifact_id}
    if fullscreen:
        payload["fullscreen"] = True
    post_json(f"{base_url}/api/focus", payload)

    # Agent-activity highlight for the host activity strip. The gallery itself
    # is unchanged — it already self-signals via its own focus SSE above.
    # notify_agent_activity never raises; the blocking call runs off the loop.
    await anyio.to_thread.run_sync(
        functools.partial(
            notify_agent_activity,
            "artifact_focus",
            "artifact",
            detail=entry.title or artifact_id,
        )
    )

    return json.dumps(
        {
            "status": "success",
            "artifact_id": artifact_id,
            "title": entry.title,
            "fullscreen": fullscreen,
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
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.set_pinned(artifact_id, pinned)
    if not entry:
        return make_error(
            "not_found",
            f"Artifact '{artifact_id}' not found.",
            ["Check the artifact_id from a previous artifact_save response."],
        )

    # Notify gallery of the update
    base_url = gallery_url()
    post_json(f"{base_url}/api/artifacts/{artifact_id}/pin", {"pinned": pinned})

    # Agent-activity highlight (same contract as artifact_focus above).
    await anyio.to_thread.run_sync(
        functools.partial(
            notify_agent_activity,
            "artifact_pin",
            "artifact",
            detail=entry.title or artifact_id,
        )
    )

    return json.dumps(
        {
            "status": "success",
            "artifact_id": artifact_id,
            "title": entry.title,
            "pinned": entry.pinned,
        }
    )
