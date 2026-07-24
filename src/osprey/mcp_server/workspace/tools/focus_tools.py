"""MCP tools: artifact_focus, artifact_pin.

artifact_focus selects a specific artifact in the gallery so the user sees it.
artifact_pin toggles the pinned flag for quick access filtering.

Both tools report gallery outcomes honestly instead of fire-and-forget:

* ``artifact_focus``: the entire effect is gallery-side, so a failed or
  rejected POST is a tool error (``gallery_unreachable`` / ``gallery_error``
  — ``make_error`` raises a fastmcp ``ToolError``).
* ``artifact_pin``: the durable pin lands in the shared index first
  (``set_pinned``), so the response stays ``success`` but reports whether the
  gallery was actually notified via ``gallery_notified``.
"""

import json
import logging
import urllib.error

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import _post_json_with_response, gallery_url
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.focus")


@mcp.tool()
async def artifact_focus(artifact_id: str, fullscreen: bool = False) -> str:
    """Select an artifact in the gallery so the user sees it.

    Sets the gallery's current selection to the given artifact, scrolling
    it into view and showing the preview pane. Also updates focus_state.txt
    so the agent-awareness hook includes this artifact in context.

    The focus effect lives entirely in the gallery, so a gallery that is
    unreachable or rejects the request is reported as a tool error
    (``gallery_unreachable`` / ``gallery_error``) — never as success.

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
    payload: dict = {"artifact_id": artifact_id}
    if fullscreen:
        payload["fullscreen"] = True

    try:
        status, body = _post_json_with_response(f"{base_url}/api/focus", payload)
    except (urllib.error.URLError, OSError) as exc:
        file_path = store.get_file_path(artifact_id)
        return make_error(
            "gallery_unreachable",
            f"Could not reach the artifact gallery at {base_url}: {exc}",
            [
                "Check that the artifact gallery server is running.",
                f"The artifact file is available directly at: {file_path} "
                "— it can be opened in a browser.",
            ],
        )

    if not 200 <= status < 300:
        detail = body.get("detail", "") if isinstance(body, dict) else str(body)
        return make_error(
            "gallery_error",
            f"Gallery rejected the focus request (HTTP {status})"
            + (f": {detail}" if detail else "."),
            ["Verify the gallery serves the same artifact store this session saved to."],
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

    The pin flag is persisted in the shared artifact index first (the
    gallery's index watcher picks it up eventually), so the tool reports
    ``status: success`` even when the live gallery notification fails —
    ``gallery_notified`` records whether the gallery acknowledged the update.

    Args:
        artifact_id: ID of the artifact to pin/unpin.
        pinned: True to pin, False to unpin. Defaults to True.

    Returns:
        JSON with status, updated pinned state, and gallery_notified.
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

    # Notify the gallery; never raise — the durable state is already set.
    base_url = gallery_url()
    gallery_notified = False
    try:
        status, _body = _post_json_with_response(
            f"{base_url}/api/artifacts/{artifact_id}/pin", {"pinned": pinned}
        )
        gallery_notified = 200 <= status < 300
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Gallery pin notification failed (non-fatal): %s", exc)

    return json.dumps(
        {
            "status": "success",
            "artifact_id": artifact_id,
            "title": entry.title,
            "pinned": entry.pinned,
            "gallery_notified": gallery_notified,
        }
    )
