"""MCP tool: artifact_focus.

artifact_focus selects a specific artifact in the gallery so the user sees it.
The entire effect is gallery-side, so a failed or rejected POST is a tool error
(``gallery_unreachable`` / ``gallery_error`` — ``make_error`` raises a fastmcp
``ToolError``).

The focus records the session it was set in (ruling 15): the tool resolves the
session scope once, at the store's workspace root, and the POST carries
``transcript_id`` only when a transcript resolved; a dispatch run or a process
in no session sends none, so its focus reaches no session's prompt. The tool
acts by id and is not scoped.
"""

import json
import urllib.error

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import (
    _post_json_with_response,
    gallery_url,
    notify_agent_activity_async,
)
from osprey.mcp_server.workspace.server import mcp


def _current_transcript(store) -> str | None:
    """The transcript this process is in, resolved at *store*'s workspace root.

    ``None`` for a dispatch run or a process in no session.
    """
    from osprey.stores import artifact_scope

    return artifact_scope.current_scope(root=store._workspace).transcript


@mcp.tool()
async def artifact_focus(artifact_id: str, fullscreen: bool = False) -> str:
    """Select an artifact in the gallery so the user sees it.

    Sets the gallery's current selection to the given artifact, scrolling
    it into view and showing the preview pane. The focus carries the current
    session's transcript, so the gallery files it under that session (the
    gallery, not this tool, writes the focus state the agent-awareness hook
    reads); with no current session it reaches no prompt.

    The focus effect lives entirely in the gallery, so a gallery that is
    unreachable or rejects the request is reported as a tool error
    (``gallery_unreachable`` / ``gallery_error``) — never as success.

    Args:
        artifact_id: ID of the artifact to select (from artifact_register response).
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
            ["Check the artifact_id from a previous artifact_register response."],
        )

    base_url = gallery_url()
    payload: dict = {"artifact_id": artifact_id}
    if fullscreen:
        payload["fullscreen"] = True
    transcript = _current_transcript(store)
    if transcript is not None:
        payload["transcript_id"] = transcript

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

    # Agent-activity highlight for the host activity strip. The gallery itself
    # is unchanged — it already self-signals via its own focus SSE above.
    # notify_agent_activity_async never raises; the blocking call runs off the loop.
    await notify_agent_activity_async(
        "artifact_focus", "artifact", detail=entry.title or artifact_id
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
