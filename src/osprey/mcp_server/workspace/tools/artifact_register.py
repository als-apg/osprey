"""MCP tools: artifact_register, artifact_delete, artifact_delete_all, artifact_get.

Register files or inline content as gallery artifacts, delete them (one at a
time or a whole scope), or look up artifact metadata and file paths.

Listing and reading artifact content live in ``artifact_query``.
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
    category: str = "",
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
        category: Optional category for gallery grouping. See the type
                  registry for valid categories.

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
                from osprey.utils.workspace import load_osprey_config, resolve_project_root

                source = resolve_project_root(load_osprey_config()) / source
            entry = store.save_from_path(
                source_path=source,
                title=title,
                description=description,
                tool_source="artifact_register",
                category=category,
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

            entry = store.save_file(
                file_content=content.encode(),
                filename=filename,
                artifact_type=a_type,
                title=title,
                description=description,
                mime_type=mime,
                tool_source="artifact_register",
                category=category,
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

    Removes both the artifact file and its index entry.

    Args:
        artifact_id: ID of the artifact to delete.

    Returns:
        JSON confirmation of deletion.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        deleted = store.delete_entry(artifact_id)

        if not deleted:
            return make_error(
                "not_found",
                f"Artifact {artifact_id} not found.",
                ["Check the artifact_id from a previous artifact_register response."],
            )

        return json.dumps(
            {
                "status": "success",
                "artifact_id": artifact_id,
                "message": f"Artifact {artifact_id} deleted.",
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


#: ``scope`` value that means the whole store, every category. Spelled out
#: so a bulk delete can never be reached by omitting an argument.
_SCOPE_EVERYTHING = "everything"

#: ``scope`` value for entries with no category (plots, screenshots and other
#: artifacts saved without one). The store spells this ``category=""``.
_SCOPE_UNCATEGORIZED = "uncategorized"


@mcp.tool()
async def artifact_delete_all(scope: str) -> str:
    """Permanently delete a whole scope of artifacts in one atomic call.

    Deletion is NOT recoverable — files are unlinked, there is no trash.
    ``scope`` is required and names exactly what gets destroyed:

    - a category key (e.g. "visualization", "archiver_data") — deletes only
      that category and leaves every other one intact. Use
      ``artifact_list`` first to see which categories exist.
    - "uncategorized" — only artifacts saved without a category.
    - "everything" — EVERY artifact in the store, across all categories.
      This includes archiver datasets, run results and other stored data,
      not just plots and documents. Only use it when the user asked to wipe
      the whole workspace.

    Prefer this over many sequential ``artifact_delete`` calls: it acquires
    the index lock once and notifies listeners consistently.

    Args:
        scope: "everything", "uncategorized", or a single category key.

    Returns:
        JSON with the scope acted on, the number of artifacts destroyed,
        their IDs, and a per-category breakdown of what was destroyed.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store
        from osprey.stores.type_registry import valid_category_keys

        valid_scopes = valid_category_keys() | {_SCOPE_EVERYTHING, _SCOPE_UNCATEGORIZED}
        if scope not in valid_scopes:
            return make_error(
                "validation_error",
                f"Unknown scope '{scope}'.",
                [
                    f'Use "{_SCOPE_EVERYTHING}" to delete every artifact in the store.',
                    f'Use "{_SCOPE_UNCATEGORIZED}" for artifacts saved without a category.',
                    f"Valid category keys: {', '.join(sorted(valid_category_keys()))}",
                ],
            )

        store = get_artifact_store()
        if scope == _SCOPE_EVERYTHING:
            deleted = store.delete_everything()
        elif scope == _SCOPE_UNCATEGORIZED:
            deleted = store.delete_category("")
        else:
            deleted = store.delete_category(scope)

        by_category: dict[str, int] = {}
        for entry in deleted:
            key = entry.category or _SCOPE_UNCATEGORIZED
            by_category[key] = by_category.get(key, 0) + 1

        return json.dumps(
            {
                "status": "success",
                "scope": scope,
                "deleted_count": len(deleted),
                "artifact_ids": [e.id for e in deleted],
                "deleted_by_category": by_category,
                "message": (f"Permanently deleted {len(deleted)} artifact(s) in scope '{scope}'."),
            }
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("artifact_delete_all failed")
        return make_error(
            "internal_error",
            f"Failed to delete artifacts in scope '{scope}': {exc}",
            ["Check MCP server logs for details."],
        )


@mcp.tool()
async def artifact_get(artifact_id: str) -> str:
    """Look up an artifact by ID to get its file path and metadata.

    Returns metadata including the on-disk file path, which can be passed
    to tools like ``graph_extract(image_path=...)``. Does not return the
    file content inline (artifacts can be large binaries) — use
    ``artifact_read`` for that.

    Args:
        artifact_id: ID of the artifact to look up.

    Returns:
        JSON with artifact metadata and file path.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.get_entry(artifact_id)

        if entry is None:
            return make_error(
                "not_found",
                f"Artifact {artifact_id} not found.",
                ["Use artifact_list or check a previous artifact_register response."],
            )

        file_path = store.get_file_path(artifact_id)
        url = gallery_url()

        result = {
            "artifact_id": entry.id,
            "title": entry.title,
            "description": entry.description,
            "artifact_type": entry.artifact_type,
            "mime_type": entry.mime_type,
            "size_bytes": entry.size_bytes,
            "timestamp": entry.timestamp,
            "file_path": str(file_path) if file_path else None,
            "gallery_url": url,
        }
        return json.dumps(result, default=str)

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("artifact_get failed")
        return make_error(
            "internal_error",
            f"Failed to get artifact: {exc}",
            ["Check MCP server logs for details."],
        )
