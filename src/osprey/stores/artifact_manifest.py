"""The execution-side half of the artifact store.

Agent code runs in a subprocess that has no artifact store: it cannot reach the
index, the gallery server, or the listeners. So ``save_artifact()`` inside that
subprocess writes files into ``<execution_folder>/artifacts/`` and appends an
entry to ``artifacts/manifest.json``; the parent reads the manifest back with
:func:`collect_artifacts` and registers each file properly.

Both ends of that contract live here. :data:`SAVE_ARTIFACT_SOURCE` is the
source text every execution wrapper injects — the python executor's and the
visualization sandbox's alike — and it delegates type detection to
:func:`osprey.stores.artifact_store.serialize_object`, the same serializer the
in-process store uses. There is one answer to "how is this object saved",
whichever path saved it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("osprey.stores.artifact_manifest")

#: Name of the per-execution manifest inside ``artifacts/``.
MANIFEST_FILENAME = "manifest.json"

#: ``save_artifact()`` as injected into an execution subprocess. Plain source
#: text, no template placeholders: the wrappers splice it in verbatim. It reads
#: ``_execution_dir`` from the module globals the wrapper defines and falls back
#: to the working directory.
SAVE_ARTIFACT_SOURCE = '''\
# save_artifact() — the execution-side half of the artifact store
def save_artifact(obj, title="Untitled", description="", artifact_type=None, category=""):
    """Save an object as a gallery artifact.

    Supported types:
      - plotly Figure -> interactive HTML
      - bokeh Model -> dashboard HTML
      - matplotlib Figure -> PNG image
      - pandas DataFrame -> HTML table
      - str -> markdown or HTML (auto-detected)
      - dict / list -> JSON
      - bytes -> binary file
      - numpy ndarray -> .npy file

    Args:
        obj: The object to save.
        title: Human-readable title shown in the gallery.
        description: Optional longer description.
        artifact_type: Override the auto-detected type.
        category: Optional category key for gallery grouping.
    """
    import json as _json
    import uuid as _uuid
    from pathlib import Path as _Path

    from osprey.stores.artifact_store import serialize_object as _serialize_object

    _art_base = _Path(globals().get("_execution_dir", _Path.cwd()))
    artifacts_dir = _art_base / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    art_id = _uuid.uuid4().hex[:12]
    content, detected_type, filename, mime_type = _serialize_object(obj, title)
    # Prefix with the id so two saves under one title never overwrite each other.
    filename = f"{art_id}_{filename}"
    final_type = artifact_type or detected_type

    (artifacts_dir / filename).write_bytes(content)

    manifest_path = artifacts_dir / "manifest.json"
    manifest = _json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    entry = {
        "id": art_id,
        "filename": filename,
        "title": title,
        "description": description,
        "artifact_type": final_type,
        "mime_type": mime_type,
        "size_bytes": len(content),
    }
    if category:
        entry["category"] = category
    manifest.append(entry)
    manifest_path.write_text(_json.dumps(manifest, indent=2))

    print(f"Artifact saved: {title} ({final_type}, {len(content)} bytes)")
'''


def collect_artifacts(execution_folder: Path) -> list[dict]:
    """Read back what ``save_artifact()`` wrote during one execution.

    Returns one dict per manifest entry whose file still exists, with ``path``
    resolved to the file on disk; the caller registers each with the store.
    A missing or unreadable manifest means nothing was saved.
    """
    manifest_path = execution_folder / "artifacts" / MANIFEST_FILENAME
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read artifact manifest", exc_info=True)
        return []

    artifacts = []
    for entry in manifest:
        file_path = execution_folder / "artifacts" / entry["filename"]
        if not file_path.exists():
            logger.debug("Artifact file missing: %s", file_path)
            continue
        art = {
            "path": file_path,
            "title": entry.get("title", "Untitled"),
            "description": entry.get("description", ""),
            "artifact_type": entry.get("artifact_type", "file"),
            "mime_type": entry.get("mime_type", "application/octet-stream"),
        }
        if entry.get("category"):
            art["category"] = entry["category"]
        artifacts.append(art)

    return artifacts
