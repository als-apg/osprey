"""MCP tools: artifact_save, artifact_delete, artifact_get.

Register files or inline content as gallery artifacts, delete them,
or look up artifact metadata and file paths.
"""

import json
import logging
from pathlib import Path

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import gallery_url
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.artifact_save")

# Content-type to artifact_type mapping
_CONTENT_TYPE_MAP = {
    "markdown": ("markdown", "text/markdown", ".md"),
    "html": ("html", "text/html", ".html"),
    "text": ("text", "text/plain", ".txt"),
    "json": ("json", "application/json", ".json"),
}


@mcp.tool()
async def artifact_save(
    title: str,
    description: str = "",
    file_path: str | None = None,
    content: str | None = None,
    content_type: str = "markdown",
    category: str = "",
) -> str:
    """Save an artifact to the OSPREY gallery for interactive viewing.

    Creates a gallery artifact from either an existing file on disk or
    inline string content. Use this to register screenshots, markdown
    summaries, HTML content, or any file Claude has produced.

    Exactly one of ``file_path`` or ``content`` must be provided.

    Args:
        title: Human-readable title for the artifact.
        description: Optional longer description.
        file_path: Path to an existing file to register as an artifact.
        content: Inline string content (markdown, HTML, text, or JSON).
        content_type: Type of inline content — "markdown", "html", "text",
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

    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()

    try:
        if file_path:
            source = Path(file_path)
            if not source.is_absolute():
                source = Path.cwd() / source
            entry = store.save_from_path(
                source_path=source,
                title=title,
                description=description,
                tool_source="artifact_save",
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
                tool_source="artifact_save",
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
        logger.exception("artifact_save permission denied")
        return make_error(
                "permission_denied",
                f"Cannot write artifact: {exc}",
                [
                    "The process does not have write access to the artifact directory.",
                    "Check ownership of _agent_data/artifacts/ — all writers must share a UID or be in a group with write permission.",
                    "If running via dispatch sidecar, confirm supervisord.conf and the interactive web process run as the same user.",
                ],
            )
    except Exception as exc:
        logger.exception("artifact_save failed")
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
                    ["Check the artifact_id from a previous artifact_save response."],
                )

        return json.dumps(
            {
                "status": "success",
                "artifact_id": artifact_id,
                "message": f"Artifact {artifact_id} deleted.",
            }
        )

    except Exception as exc:
        logger.exception("artifact_delete failed")
        return make_error(
                "internal_error",
                f"Failed to delete artifact: {exc}",
                ["Check MCP server logs for details."],
            )


@mcp.tool()
async def artifact_get(artifact_id: str) -> str:
    """Look up an artifact by ID to get its file path and metadata.

    Returns metadata including the on-disk file path, which can be passed
    to tools like ``graph_extract(image_path=...)``. Does not return the
    file content inline (artifacts can be large binaries).

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
                    ["Use data_list or check a previous artifact_save response."],
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

    except Exception as exc:
        logger.exception("artifact_get failed")
        return make_error(
                "internal_error",
                f"Failed to get artifact: {exc}",
                ["Check MCP server logs for details."],
            )
