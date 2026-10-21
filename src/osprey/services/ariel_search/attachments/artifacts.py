"""Put an already-prepared ARIEL picture into the Workspace artifact store.

The web interface and the ARIEL MCP server share this operation.  It accepts
only the prepared rendition selected by their respective access layers; it
does not fetch a source URL, process an upload, or focus a browser panel.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from osprey.mcp_server.http import gallery_url
from osprey.services.ariel_search.attachments.compose import FILENAME_MAX_CHARS, _inert

#: The gallery category used for pictures retained from ARIEL.
PICTURE_ARTIFACT_CATEGORY = "visualization"

# Keep the gallery source stable without making this service-layer helper
# depend on the ARIEL MCP server's registration module.
ATTACHMENT_TO_ARTIFACT_TOOL = "attachment_to_artifact"

_RENDITION_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def artifact_filename(filename: Any, rendition_mime: str) -> str:
    """Return a safe gallery filename for a stored display rendition."""
    from osprey.stores.artifact_store import _slugify

    stem = PurePosixPath(str(filename or "picture").replace("\\", "/")).stem
    return f"{_slugify(stem) or 'picture'}{_RENDITION_EXTENSIONS.get(rendition_mime, '.img')}"


def picture_provenance(entry_id: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimal logbook provenance recorded with the artifact."""
    provenance: dict[str, Any] = {
        "entry_id": entry_id,
        "attachment_id": summary.get("attachment_id"),
        "filename": summary.get("filename"),
    }
    if summary.get("caption"):
        provenance["caption"] = summary["caption"]
        provenance["caption_source"] = summary.get("caption_source")
    return provenance


def picture_description(provenance: Mapping[str, Any]) -> str:
    """Build the human-readable gallery description from safe provenance."""
    lines = [
        f"Logbook picture {provenance['attachment_id']} from entry {provenance['entry_id']} "
        f"({_inert(provenance.get('filename'), FILENAME_MAX_CHARS)})."
    ]
    if provenance.get("caption"):
        source = provenance.get("caption_source") or "unknown"
        lines.append(f"Caption ({_inert(source, 40)}): {provenance['caption']}")
    return "\n".join(lines)


def save_viewable_picture_as_artifact(
    row: Mapping[str, Any], entry: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Store a prepared display rendition in Workspace and return its metadata.

    The caller establishes that ``row`` is viewable before calling.  This
    routine deliberately performs no gallery focus: the MCP caller may focus
    the gallery for an agent action, while a browser caller activates Workspace
    only in the requesting browser.
    """
    from osprey.stores.artifact_store import get_artifact_store

    entry_id = str(entry["entry_id"])
    attachment_id = str(summary["attachment_id"])
    rendition = bytes(row["rendition_bytes"])
    rendition_mime = str(row["rendition_mime"])
    sha256 = hashlib.sha256(rendition).hexdigest()
    provenance = picture_provenance(entry_id, summary)
    filename = _inert(summary.get("filename"), FILENAME_MAX_CHARS) or "picture"

    store = get_artifact_store()
    known = {entry.id for entry in store.list_entries(tool_filter=ATTACHMENT_TO_ARTIFACT_TOOL)}
    artifact = store.save_or_touch_by_sha256(
        sha256,
        origin="",
        save_kwargs={
            "file_content": rendition,
            "filename": artifact_filename(summary.get("filename"), rendition_mime),
            "artifact_type": "image",
            "title": f"{filename} (entry {entry_id})",
            "description": picture_description(provenance),
            "mime_type": rendition_mime,
            "tool_source": ATTACHMENT_TO_ARTIFACT_TOOL,
            "metadata": {"sha256": sha256, "logbook_picture": provenance},
            "category": PICTURE_ARTIFACT_CATEGORY,
        },
    )
    response = artifact.to_tool_response(gallery_url=gallery_url())
    response.update(
        {
            "entry_id": entry_id,
            "attachment_id": attachment_id,
            "created": artifact.id not in known,
        }
    )
    return response
