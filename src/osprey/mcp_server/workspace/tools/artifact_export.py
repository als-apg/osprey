"""MCP tool: artifact_export — convert HTML artifacts to images (PNG/JPEG)."""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import gallery_url
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.artifact_export")


@mcp.tool()
async def artifact_export(
    artifact_id: str,
    format: str = "png",
    width: int = 1200,
    height: int = 800,
) -> str:
    """Export an HTML artifact to a static image (PNG or JPEG).

    Converts interactive HTML artifacts (e.g. Plotly plots) into static
    images suitable for logbook attachments or reports. If the artifact
    is already in the target format, returns it without conversion.

    Args:
        artifact_id: ID of the artifact to export.
        format: Target image format — "png" or "jpeg".
        width: Viewport width in pixels for rendering.
        height: Viewport height in pixels for rendering.

    Returns:
        JSON with the new (or existing) artifact ID and gallery URL.
    """
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.get_entry(artifact_id)

    if entry is None:
        return json.dumps(
            make_error(
                "not_found",
                f"Artifact '{artifact_id}' not found.",
                ["Use artifact_save or execute to create artifacts first."],
            )
        )

    target_mime = f"image/{format}"

    # Already in target format — return as-is
    if entry.mime_type == target_mime:
        url = gallery_url()
        return json.dumps(entry.to_tool_response(gallery_url=url))

    # Only HTML artifacts can be converted
    if entry.mime_type != "text/html":
        return json.dumps(
            make_error(
                "conversion_not_supported",
                f"Cannot convert artifact type '{entry.mime_type}' to {format}.",
                [
                    "Only HTML artifacts (e.g. Plotly plots) can be exported to images.",
                    "Images and other formats are already in a static format.",
                ],
            )
        )

    # Convert HTML → image via Playwright
    try:
        from osprey.mcp_server.export.converter import (
            PlaywrightNotInstalledError,
            convert_html_to_image,
        )
    except ImportError:
        return json.dumps(
            make_error(
                "dependency_missing",
                "Export module not available.",
                ["Ensure osprey.mcp_server.export is installed."],
            )
        )

    html_path = store.get_file_path(artifact_id)
    if html_path is None or not html_path.exists():
        return json.dumps(
            make_error(
                "file_not_found",
                f"Artifact file not found on disk for '{artifact_id}'.",
                ["The artifact index may be stale. Try saving the artifact again."],
            )
        )

    output_path = html_path.with_suffix(f".{format}")

    try:
        await convert_html_to_image(
            html_path=html_path,
            output_path=output_path,
            fmt=format,
            width=width,
            height=height,
        )
    except PlaywrightNotInstalledError as exc:
        return json.dumps(
            make_error(
                "dependency_missing",
                str(exc),
                [
                    "Install Playwright: pip install playwright",
                    "Install browser: playwright install chromium",
                ],
            )
        )
    except Exception as exc:
        logger.exception("artifact_export conversion failed")
        return json.dumps(
            make_error(
                "conversion_error",
                f"HTML-to-image conversion failed: {exc}",
                ["Check MCP server logs for details."],
            )
        )

    # Save converted image as a new artifact
    try:
        image_bytes = output_path.read_bytes()
        new_entry = store.save_file(
            file_content=image_bytes,
            filename=f"{entry.title.replace(' ', '_')}.{format}",
            artifact_type="image",
            title=f"{entry.title} ({format.upper()})",
            description=f"Exported from HTML artifact {artifact_id}",
            mime_type=target_mime,
            tool_source="artifact_export",
            metadata={"source_artifact_id": artifact_id},
        )

        url = gallery_url()
        return json.dumps(new_entry.to_tool_response(gallery_url=url))

    except Exception as exc:
        logger.exception("artifact_export save failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to save exported image: {exc}",
                ["Check MCP server logs for details."],
            )
        )
