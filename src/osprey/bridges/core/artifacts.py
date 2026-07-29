"""Fetch agent-generated plot artifacts from the dispatch worker.

This is the channel-agnostic *fetch* half of artifact delivery: given a run and
an artifact id, pull the bytes over the worker's authenticated byte route and
apply the size / PNG-magic guards. What a channel then DOES with those bytes —
republish them as a public object for an image widget, attach them as MIME
parts, share them over WebDAV — stays in the channel package.

``artifact_ids`` normalizes the run-status body's ``artifacts`` field (osprey
#363 descriptor dicts, or bare id strings from an older worker) to a plain list
of artifact-id strings.

Nothing here raises to the caller: a fetch failure must degrade to fewer (or
zero) images, never break the text answer. NO channel imports, NO osprey
dispatch imports.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import CoreConfig

logger = logging.getLogger(__name__)

# PNG magic bytes: any payload not starting with this is rejected.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Oversize guard: a large image renders as a *broken* widget in most chat
# surfaces, which is strictly worse than posting no image at all. This is a
# rendering bound, not a memory bound — the body is already fully read by the
# time it is checked.
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024

# Same kind of bound as MAX_ARTIFACT_BYTES, but for non-image documents: a
# rendering/sanity bound, not a memory bound — fetch_artifact reads the full
# body before the size check runs.
MAX_DOC_BYTES = 10 * 1024 * 1024


def artifact_ids(artifacts: list[Any] | None) -> list[str]:
    """Normalize a run's ``artifacts`` field to a list of artifact-id strings.

    Accepts either osprey #363 descriptor dicts (``{"artifact_id": ...}``) or
    bare id strings (an older worker still answering during the deploy window),
    in any mix. Anything else — a dict without a usable ``artifact_id``, a
    non-str/non-dict element — is skipped rather than raised on, so a malformed
    entry costs at most one image, never the whole answer.
    """
    ids: list[str] = []
    for a in artifacts or []:
        if isinstance(a, str):
            if a:
                ids.append(a)
        elif isinstance(a, dict):
            aid = a.get("artifact_id")
            if isinstance(aid, str) and aid:
                ids.append(aid)
        # else: unknown shape -> skip
    return ids


def split_artifacts(artifacts: list[Any] | None) -> tuple[list[str], list[dict]]:
    """Split a run's ``artifacts`` field into ``(image_ids, doc_descriptors)``.

    Mirrors ``artifact_ids``'s tolerance for the same input shapes (osprey #363
    descriptor dicts, bare id strings from an older worker, or a mix), but
    additionally routes each entry by ``delivered_mime``: PNG images go to
    ``image_ids`` (id strings, matching the inline-image path); every other
    artifact — including a descriptor with no ``delivered_mime`` key at all —
    goes to ``doc_descriptors`` as the *full* dict, since a document delivery
    path needs ``filename``/``delivered_mime`` that a bare id string can't
    carry. A bare string is back-compat for an older worker with no mime at
    all, so it is treated as the existing PNG image path. Anything else — a
    dict without a usable ``artifact_id``, a non-str/non-dict element — is
    skipped rather than raised on, same as ``artifact_ids``. Every id ends up
    in exactly one bucket.
    """
    image_ids: list[str] = []
    doc_descriptors: list[dict] = []
    for a in artifacts or []:
        if isinstance(a, str):
            if a:
                image_ids.append(a)
        elif isinstance(a, dict):
            aid = a.get("artifact_id")
            if isinstance(aid, str) and aid:
                if a.get("delivered_mime") == "image/png":
                    image_ids.append(aid)
                else:
                    doc_descriptors.append(a)
        # else: unknown shape -> skip
    return image_ids, doc_descriptors


# Fixed allowlist: an extension is derived ONLY from the mime, never a
# worker-supplied filename, so no "/" or ".." can enter a storage object key.
_EXT_BY_MIME = {
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "application/json": ".json",
    "image/jpeg": ".jpg",
    "image/svg+xml": ".svg",
}


def ext_for_mime(mime: str | None) -> str:
    """Map a ``delivered_mime`` to a file extension via a fixed allowlist.

    Unknown or ``None`` mimes fall back to ``.bin`` rather than raising or
    deriving anything from worker-supplied data.
    """
    if mime is None:
        return ".bin"
    return _EXT_BY_MIME.get(mime, ".bin")


def safe_label(name: str | None, fallback: str) -> str:
    """Strip CR/LF (a header-injection / serialization hazard) and bound the
    length of a worker-supplied name.

    Used for a user-visible label (never a storage object key) and for an
    attachment filename.
    """
    cleaned = (name or "").replace("\r", "").replace("\n", "").strip()
    return (cleaned or fallback)[:200]


def fetch_artifact(
    http: httpx.Client,
    cfg: CoreConfig,
    run_id: str,
    artifact_id: str,
    *,
    max_bytes: int = MAX_ARTIFACT_BYTES,
    require_png: bool = True,
) -> bytes | None:
    """Fetch one artifact's bytes from the worker; ``None`` on any failure or a
    payload that fails the size guard (or the PNG-magic guard when required).

    ``require_png`` defaults to ``True`` for image widgets that only render PNG.
    A channel that delivers arbitrary MIME (attaching the worker's
    ``delivered_mime`` bytes verbatim) passes ``require_png=False`` and a
    suitable ``max_bytes``, keeping only the size guard.

    The caller owns the ``httpx.Client``; build it with
    ``httpx.Client(trust_env=cfg.trust_env)`` so proxy handling comes from
    config rather than from whatever the ambient environment happens to set.
    """
    url = f"{cfg.worker_url}/dispatch/{run_id}/artifacts/{artifact_id}"
    headers = {"Authorization": f"Bearer {cfg.dispatch_worker_token}"}
    try:
        resp = http.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.content
    except Exception as exc:
        # Deliberately broad: ``httpx.InvalidURL`` does NOT subclass
        # ``httpx.HTTPError`` and is raised at URL-construction time, before
        # any transport error can occur. ``artifact_id`` is worker-supplied, so
        # a control character in it would otherwise escape and cost the user
        # the whole text answer. BaseException still propagates.
        logger.warning("artifact fetch failed for %s/%s: %s", run_id, artifact_id, exc)
        return None
    if len(data) > max_bytes:
        logger.warning(
            "artifact %s/%s oversize (%d bytes > %d), skipping",
            run_id,
            artifact_id,
            len(data),
            max_bytes,
        )
        return None
    if require_png and not data.startswith(PNG_MAGIC):
        logger.warning("artifact %s/%s is not a PNG, skipping", run_id, artifact_id)
        return None
    return data
