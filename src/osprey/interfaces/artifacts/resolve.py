"""Read-only resolution of a dispatch run's artifacts to renderable bytes.

Generic OSPREY-core capability: given a headless dispatch ``run_id``, discover
the artifacts the run produced (via the artifact store's write-time ``run_id``
tag — ``ArtifactStore.list_entries(run_filter=...)`` / ``get_run_entry``) and
resolve each to a file on disk that an external caller can fetch as image/PDF
bytes.

Association is **created-by, not referenced-by**: a run owns exactly the
artifacts it *wrote* (tagged from ``OSPREY_DISPATCH_RUN_ID`` at save time),
never merely the ones its tool-call results happen to mention. Authorization
therefore never reads agent-controllable data, so a prompt-injected run cannot
widen its own artifact set or reach another run's bytes.

Resolution rules per artifact MIME type:
  * ``image/*`` and ``application/pdf`` — passthrough, served as-is.
  * HTML / Markdown / notebook / JSON / text — converted to PNG via the shared
    converter registry, and the temp PNG path is returned.
  * If the converter or one of its runtime dependencies (e.g. Playwright) is
    unavailable, the original file is returned with ``convertible=False`` rather
    than failing the whole call.

Depends only on ``osprey.stores`` (artifact index/files) and the artifact
converter registry (``osprey.mcp_server.ariel.converters``). It deliberately
does NOT import the ARIEL FastMCP server, so it is safe to import and call from
the dispatch worker without coupling the worker to that server.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.stores.artifact_store import ArtifactStore

logger = logging.getLogger("osprey.interfaces.artifacts.resolve")


@dataclass
class ArtifactRef:
    """A resolved, fetchable reference to one artifact produced by a run.

    Attributes:
        artifact_id: The artifact store ID.
        filename: Basename of the file that will be served (post-conversion).
        mime_type: MIME type of the bytes at ``abs_path`` (``image/png`` for a
            converted artifact; the original type for a passthrough/fallback).
        abs_path: Absolute path to the file to stream to the caller.
        convertible: ``True`` when the artifact is directly renderable — either a
            passthrough image/PDF or a successful conversion to PNG. ``False``
            when conversion was attempted but a converter/dependency was
            unavailable, in which case ``abs_path`` is the original file.
    """

    artifact_id: str
    filename: str
    mime_type: str
    abs_path: str
    convertible: bool


# ---------------------------------------------------------------------------
# Run-record + store location
# ---------------------------------------------------------------------------


def _agent_data_root() -> Path:
    """Return the ``_agent_data`` root for the deployed project.

    Mirrors the dispatch worker's persistence layout (see
    ``dispatch_api._LOG_DIR``): both derive from ``OSPREY_PROJECT_DIR`` so a
    resolver running in the worker process (whose CWD is the image WORKDIR, not
    the project dir) reads the same location the dispatched agent wrote to.
    """
    project_dir = os.environ.get("OSPREY_PROJECT_DIR", "/app/project")
    return Path(project_dir) / "_agent_data"


def _dispatch_log_dir() -> Path:
    return _agent_data_root() / "dispatch"


def _run_artifacts_dir(run_id: str) -> Path:
    """Deterministic, reused per-run scratch dir for converted (PNG) artifacts.

    Keyed by ``run_id`` rather than a fresh ``mkdtemp`` per call, so repeated
    resolution of the same run overwrites the same files (bounded by run count,
    not request count) instead of leaking a new temp dir on every request.
    """
    out = _agent_data_root() / "dispatch" / "artifacts" / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_run_record(run_id: str) -> dict[str, Any] | None:
    """Load a persisted dispatch run record by ID, or ``None`` if absent.

    Used only for the run-status disk fallback (a completed run evicted from the
    worker's in-memory cap is still readable on disk). It is NOT used for
    artifact association — that is the store's write-time ``run_id`` tag.
    """
    path = _dispatch_log_dir() / f"{run_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read dispatch run record %s", path, exc_info=True)
        return None


def _get_store() -> ArtifactStore:
    """Build an ArtifactStore rooted at the project's ``_agent_data``.

    Not the module-level singleton: the worker process CWD differs from the
    project dir, so the singleton's default root would point at the wrong place.
    """
    from osprey.stores.artifact_store import ArtifactStore

    return ArtifactStore(workspace_root=_agent_data_root())


def get_run_store() -> ArtifactStore:
    """Public accessor for the run-scoped artifact store used by dispatch.

    The single source of truth for the store root shared by everything that
    reads or writes a dispatch run's artifacts — the resolver's descriptors and
    byte route here, and the worker's pre-run input-file ingestion. Routing both
    through one constructor keeps them on the same ``_agent_data`` root, so a
    file the worker writes is exactly the file the byte route later serves.
    """
    return _get_store()


# ---------------------------------------------------------------------------
# Delivery prediction (render-free)
# ---------------------------------------------------------------------------


def predict_delivery(source_mime: str) -> tuple[str, bool]:
    """Predict how the byte route will deliver an artifact, WITHOUT rendering.

    Returns ``(delivered_mime, convertible)``:
      * passthrough types (``image/*``, ``application/pdf``,
        ``application/octet-stream``) → ``(source_mime, True)`` — served as-is.
      * everything else → ``("image/png", True)`` — rendered to PNG on fetch.

    ``convertible`` is optimistic: it reflects the intended (success-path)
    outcome. If a converter dependency (e.g. Playwright) is unavailable at fetch
    time, the byte route falls back to the original bytes with the ref's
    ``convertible`` set to ``False`` — the byte route's actual Content-Type is
    always authoritative. Kept render-free so listing a run's artifacts never
    triggers O(N) conversions.
    """
    from osprey.mcp_server.ariel.converters import get_converter, passthrough

    if get_converter(source_mime) is passthrough:
        return source_mime, True
    return "image/png", True


def _predicted_filename(entry_filename: str, delivered_mime: str) -> str:
    """Best-effort delivered basename for descriptors (display/caption only).

    The byte route's Content-Disposition filename (post-conversion) is
    authoritative; this is a render-free prediction: a rendered artifact is
    delivered as ``<stem>.png``, a passthrough keeps its stored name.
    """
    name = Path(entry_filename).name
    if delivered_mime == "image/png" and not name.lower().endswith(".png"):
        name = Path(name).stem + ".png"
    return name


def describe_run_artifacts(run_id: str) -> list[dict[str, Any]]:
    """Render-free descriptors of the artifacts a run produced, oldest first.

    Feeds BOTH the run-status ``artifacts`` list and the list route, so they can
    never disagree. Reads only the store index (via the strict ``run_filter``
    created-by tag) and predicts delivery — it never invokes a converter, so it
    is cheap regardless of artifact count. Returns ``[]`` for an unknown/empty
    ``run_id`` or a run that produced no artifacts.

    Each descriptor is ``{artifact_id, filename, source_mime, delivered_mime,
    convertible}``. ``source_mime`` is retained so a consumer always knows the
    real original bytes even when ``delivered_mime`` differs (rendered to PNG).
    """
    if not run_id:
        return []
    try:
        entries = _get_store().list_entries(run_filter=run_id)
    except Exception:
        logger.warning("Could not read artifact store for run %s", run_id, exc_info=True)
        return []
    out: list[dict[str, Any]] = []
    for e in entries:
        # Caller-supplied inputs (origin="input") are not things the run
        # produced — they are surfaced separately (see describe_run_input_artifacts)
        # so a consumer never mistakes an ingested image for agent output.
        if e.origin == "input":
            continue
        delivered_mime, convertible = predict_delivery(e.mime_type)
        out.append(
            {
                "artifact_id": e.id,
                "filename": _predicted_filename(e.filename, delivered_mime),
                "source_mime": e.mime_type,
                "delivered_mime": delivered_mime,
                "convertible": convertible,
            }
        )
    return out


def describe_run_input_artifacts(run_id: str) -> list[dict[str, Any]]:
    """Descriptors of the caller-supplied files ingested into a run, oldest first.

    The counterpart to :func:`describe_run_artifacts`: it returns only the
    ``origin="input"`` entries (the files the worker stored before the agent
    ran), which that function excludes. Reads only the store index via the
    strict created-by ``run_filter`` tag, so it is cheap and cannot cross runs.
    Returns ``[]`` for an unknown/empty run or one with no ingested inputs.

    Each descriptor is exactly ``{entry_id, filename, mime}`` — ``entry_id`` is
    the store id the byte route resolves, ``filename`` the caller's (sanitised)
    name, ``mime`` the stored content type. These keys are a consumed contract;
    do not rename them.
    """
    if not run_id:
        return []
    try:
        entries = _get_store().list_entries(run_filter=run_id)
    except Exception:
        logger.warning("Could not read artifact store for run %s", run_id, exc_info=True)
        return []
    out: list[dict[str, Any]] = []
    for e in entries:
        if e.origin != "input":
            continue
        out.append(
            {
                "entry_id": e.id,
                "filename": e.metadata.get("input_filename", e.filename),
                "mime": e.mime_type,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Single-artifact resolution
# ---------------------------------------------------------------------------


async def resolve_artifact_path(store: ArtifactStore, artifact_id: str, output_dir: Path) -> str:
    """Resolve one artifact ID to a renderable file path (strict).

    Shared core used by callers that want failure to surface (e.g. the ARIEL
    logbook attachment path). Images/PDFs pass through; rendered content types
    are converted to PNG in ``output_dir``.

    Raises:
        ValueError: If the artifact ID or its on-disk file is missing.

    Converter errors (e.g. a missing Playwright dependency) propagate to the
    caller unchanged.
    """
    from osprey.mcp_server.ariel.converters import get_converter

    entry = store.get_entry(artifact_id)
    if entry is None:
        raise ValueError(f"Artifact '{artifact_id}' not found.")

    file_path = store.get_file_path(artifact_id)
    if file_path is None or not file_path.exists():
        raise ValueError(f"Artifact file not found on disk for '{artifact_id}'.")

    converter = get_converter(entry.mime_type)
    result_path = await converter(file_path, output_dir)
    return str(result_path)


async def _resolve_one(
    store: ArtifactStore, artifact_id: str, output_dir: Path
) -> ArtifactRef | None:
    """Resolve one artifact to an :class:`ArtifactRef` (lenient).

    Returns ``None`` if the artifact or its file is missing (the run's tag
    survives but the file was deleted). Converter failures are caught and yield a
    ``convertible=False`` ref pointing at the original bytes rather than aborting
    the whole run resolution.
    """
    from osprey.mcp_server.ariel.converters import get_converter

    entry = store.get_entry(artifact_id)
    if entry is None:
        logger.warning("Artifact %s not found in store", artifact_id)
        return None

    file_path = store.get_file_path(artifact_id)
    if file_path is None or not file_path.exists():
        logger.warning("Artifact %s has no file on disk", artifact_id)
        return None

    converter = get_converter(entry.mime_type)
    try:
        result_path = Path(await converter(file_path, output_dir))
    except Exception:
        # Converter or a runtime dependency (e.g. Playwright) is unavailable —
        # fall back to the original bytes so the caller still gets something.
        logger.warning(
            "Artifact %s conversion failed; serving original bytes",
            artifact_id,
            exc_info=True,
        )
        return ArtifactRef(
            artifact_id=artifact_id,
            filename=file_path.name,
            mime_type=entry.mime_type,
            abs_path=str(file_path),
            convertible=False,
        )

    if result_path == file_path:
        # Passthrough (image/PDF) — served as-is with its original MIME type.
        mime_type = entry.mime_type
    else:
        # Rendered to PNG by the converter registry.
        mime_type = "image/png"

    return ArtifactRef(
        artifact_id=artifact_id,
        filename=result_path.name,
        mime_type=mime_type,
        abs_path=str(result_path),
        convertible=True,
    )


# ---------------------------------------------------------------------------
# Run resolution (created-by tag)
# ---------------------------------------------------------------------------


async def resolve_run_artifacts(run_id: str) -> list[ArtifactRef]:
    """Resolve every artifact a dispatch run produced to a fetchable file.

    Association is the store's write-time ``run_id`` tag (created-by), NOT the
    run's tool-call results. Returns an empty list for an unknown/empty run or a
    run that produced no artifacts. Artifacts whose files are missing from the
    store are skipped. Converts one PNG per rendered artifact, so callers that
    only need metadata should use :func:`describe_run_artifacts` instead.
    """
    if not run_id:
        return []

    store = _get_store()
    entries = store.list_entries(run_filter=run_id)
    if not entries:
        return []

    output_dir = _run_artifacts_dir(run_id)
    refs: list[ArtifactRef] = []
    for entry in entries:
        ref = await _resolve_one(store, entry.id, output_dir)
        if ref is not None:
            refs.append(ref)
    return refs


async def resolve_single_run_artifact(run_id: str, artifact_id: str) -> ArtifactRef | None:
    """Resolve ONE artifact of a run, converting only that artifact.

    The cross-run gate is the store's created-by tag: ``get_run_entry`` returns
    the entry only when it was *written* by ``run_id`` (non-empty tag, exact
    match). An unknown run, an artifact created by another run (or by no run), a
    traversal-shaped ``artifact_id``, or a missing on-disk file all return
    ``None`` — indistinguishable, so this is never a cross-run existence oracle.
    """
    store = _get_store()
    if store.get_run_entry(run_id, artifact_id) is None:
        return None

    output_dir = _run_artifacts_dir(run_id)
    return await _resolve_one(store, artifact_id, output_dir)
