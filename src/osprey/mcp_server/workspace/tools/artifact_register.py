"""MCP tools: artifact_register, artifact_delete, artifact_delete_all.

Register files or inline content as gallery artifacts, and delete them (one by
id, or this session's artifacts in one scope).

Listing artifacts and reading one (metadata, file path and content) live in
``artifact_query`` (``artifact_list`` / ``artifact_read``).
"""

import json
import logging
from pathlib import Path

from fastmcp.exceptions import ToolError

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import gallery_url
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.artifact_register")

# Content-type to artifact_type mapping
_CONTENT_TYPE_MAP = {
    "markdown": ("markdown", "text/markdown", ".md"),
    "html": ("html", "text/html", ".html"),
    "text": ("text", "text/plain", ".txt"),
    "json": ("json", "application/json", ".json"),
}


@mcp.tool()
async def artifact_register(
    title: str,
    description: str = "",
    file_path: str | None = None,
    content: str | None = None,
    content_type: str | None = None,
) -> str:
    """Register a file on disk, or literal text, as a gallery artifact.

    Use this for things that already exist as a file or as text: a
    screenshot, a CSV or PDF a tool left on disk, a markdown summary you
    wrote. The content is stored as given — nothing is serialized.

    A live Python object (a figure, a DataFrame, a dict of computed numbers)
    cannot be passed here; save it from the code that produced it with
    ``save_artifact()`` inside ``execute``.

    Exactly one of ``file_path`` or ``content`` must be provided.

    Args:
        title: Human-readable title for the artifact.
        description: Optional longer description.
        file_path: Path to an existing file to register as an artifact.
        content: Literal text to store (markdown, HTML, plain text, or JSON).
        content_type: Required with ``content`` — "markdown", "html", "text",
                      or "json". Ignored when file_path is provided.

    Returns:
        JSON with artifact ID and gallery URL.
    """
    if file_path and content:
        return make_error(
            "validation_error",
            "Provide exactly one of file_path or content, not both.",
            [
                "Use file_path to register an existing file.",
                "Use content for inline markdown/HTML/text/JSON.",
            ],
        )

    if not file_path and not content:
        return make_error(
            "validation_error",
            "Provide either file_path or content.",
            [
                "Use file_path to register an existing file.",
                "Use content for inline markdown/HTML/text/JSON.",
            ],
        )

    if content and content_type is None:
        return make_error(
            "validation_error",
            "content requires content_type.",
            [f"Pass content_type as one of: {', '.join(_CONTENT_TYPE_MAP)}."],
        )

    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()

    try:
        if file_path:
            source = Path(file_path)
            if not source.is_absolute():
                # Anchored on the deployment repo root, NOT this server's cwd.
                # The two are different processes with different working
                # directories: the agent CLI is spawned with cwd `build/` on a
                # host launch and this server inherits it, while the agent's own
                # code runs with cwd at the repo root and the artifact store
                # hands out repo-root-relative pointers. A relative path the
                # agent produced therefore resolved one zone too deep here —
                # and only in containers, where the two coincide, did it work.
                from osprey_connectors.workspace import load_osprey_config, resolve_project_root

                source = resolve_project_root(load_osprey_config()) / source
            entry = store.save_from_path(
                source_path=source,
                title=title,
                description=description,
                tool_source="artifact_register",
            )
        else:
            # Inline content
            if content_type not in _CONTENT_TYPE_MAP:
                return make_error(
                    "validation_error",
                    f"Unknown content_type '{content_type}'.",
                    [f"Valid types: {', '.join(_CONTENT_TYPE_MAP)}"],
                )

            a_type, mime, ext = _CONTENT_TYPE_MAP[content_type]
            from osprey.stores.artifact_store import _slugify

            filename = f"{_slugify(title)}{ext}"

            assert content is not None  # exactly one of file_path/content, checked above
            entry = store.save_file(
                file_content=content.encode(),
                filename=filename,
                artifact_type=a_type,
                title=title,
                description=description,
                mime_type=mime,
                tool_source="artifact_register",
            )

        url = gallery_url()
        return json.dumps(entry.to_tool_response(gallery_url=url), default=str)

    except FileNotFoundError as exc:
        return make_error(
            "file_not_found",
            str(exc),
            ["Check the file path exists.", "Use an absolute path or path relative to CWD."],
        )
    except PermissionError as exc:
        logger.exception("artifact_register permission denied")
        return make_error(
            "permission_denied",
            f"Cannot write artifact: {exc}",
            [
                "The process does not have write access to the artifact directory.",
                "Check ownership of _agent_data/artifacts/ — all writers must share a UID or be in a group with write permission.",
                "If running via dispatch sidecar, confirm supervisord.conf and the interactive web process run as the same user.",
            ],
        )
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("artifact_register failed")
        return make_error(
            "internal_error",
            f"Artifact save failed: {exc}",
            ["Check MCP server logs for details."],
        )


@mcp.tool()
async def artifact_delete(artifact_id: str) -> str:
    """Delete an artifact from the OSPREY gallery.

    The same rule as the gallery's delete: deleting a run's head deletes its
    whole run; any other artifact goes alone. Files and index entries are both
    removed. The id is an explicit reference, so it is not scoped to the
    current session.

    Args:
        artifact_id: ID of the artifact to delete.

    Returns:
        JSON confirmation listing every deleted id.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        removed = store.delete_with_group(artifact_id)

        if not removed:
            return make_error(
                "not_found",
                f"Artifact {artifact_id} not found.",
                ["Check the artifact_id from a previous artifact_register response."],
            )

        deleted_ids = [e.id for e in removed]
        return json.dumps(
            {
                "status": "success",
                "artifact_id": artifact_id,
                "deleted_ids": deleted_ids,
                "message": f"Deleted {len(deleted_ids)} artifact(s).",
            }
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("artifact_delete failed")
        return make_error(
            "internal_error",
            f"Failed to delete artifact: {exc}",
            ["Check MCP server logs for details."],
        )


#: ``scope`` value that means every form this session owns. Spelled out so a
#: bulk delete can never be reached by omitting an argument.
_SCOPE_EVERYTHING = "everything"


@mcp.tool()
async def artifact_delete_all(scope: str) -> str:
    """Permanently delete this session's artifacts in one scope, in one atomic call.

    Reaches only artifacts THIS session made: other sessions' artifacts, the
    shipped example, watchdog judgements and dispatch-run output are never
    touched. To delete another session's artifact, pass its id to
    ``artifact_delete``. With no current session it deletes nothing and says
    why in ``reason``.

    Deletion is NOT recoverable — files are unlinked, there is no trash.
    ``scope`` is required and names exactly what gets destroyed:

    - a form — "runs", "plots", "trends", "tables", "documents",
      "dashboards", "images" or "data" — or a registered extension type's
      key: this session's artifacts listed under it; every other group stays.
      "runs" deletes each script run with its code and outputs; any other
      form deletes a run's outputs in it, never the run. Use
      ``artifact_list`` first to see what each form holds.
    - "everything" — every artifact this session owns, across all forms,
      including archiver datasets and run results. Only use it when the user
      asked to wipe this session's artifacts.

    Prefer this over many sequential ``artifact_delete`` calls: it acquires
    the index lock once and notifies listeners consistently.

    Args:
        scope: "everything", a form, or an extension type key.

    Returns:
        JSON with the scope acted on, the session, the number of artifacts
        destroyed and their IDs, plus ``reason`` when no session resolved.
    """
    try:
        from osprey.mcp_server.workspace.tools.artifact_query import (
            extension_registry,
            require_form,
        )
        from osprey.stores.artifact_scope import current_scope
        from osprey.stores.artifact_store import get_artifact_store

        registry = extension_registry()
        require_form(scope, registry, extra=(_SCOPE_EVERYTHING,))

        from osprey.mcp_server.artifact_activity import (
            notify_bulk_delete,
            suppress_delete_frames,
        )

        store = get_artifact_store()
        session = current_scope(root=store._workspace)
        form = None if scope == _SCOPE_EVERYTHING else scope
        # One action, one activity frame: the store fires its delete listener
        # once per removed entry, and a scope bigger than the web terminal's
        # 50-frame history buffer would evict the buffer's whole content with
        # copies of the same delete. Suppress the per-entry frames and report
        # the summary instead.
        with suppress_delete_frames():
            deleted = store.delete_owned(session, form, registry)
        notify_bulk_delete(scope, len(deleted))

        response = {
            "status": "success",
            "scope": scope,
            "session": session.transcript,
            "deleted_count": len(deleted),
            "artifact_ids": [e.id for e in deleted],
            "message": (
                f"Permanently deleted {len(deleted)} artifact(s) of this session "
                f"in scope '{scope}'."
            ),
        }
        if session.transcript is None:
            response["reason"] = session.reason
            response["message"] = f"Deleted nothing: {session.reason}."
        return json.dumps(response)

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("artifact_delete_all failed")
        return make_error(
            "internal_error",
            f"Failed to delete artifacts in scope '{scope}': {exc}",
            ["Check MCP server logs for details."],
        )
