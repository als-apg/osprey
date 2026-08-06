"""Fetch agent-generated artifacts from the dispatch worker.

This is the channel-agnostic *fetch* half of artifact delivery: given a run and
an artifact id, pull the bytes over the worker's authenticated byte route and
apply the size guard. What a channel then DOES with those bytes — republish them
as a public object for an image widget, attach them as MIME parts, share them
over WebDAV — stays in the channel package.

A run's ``artifacts`` descriptors are a *prediction*, written when the run
completes: they promise ``delivered_mime: image/png`` for everything the worker
intends to render, but whether the render happens is only known at fetch time
(a converter dependency can be missing, a renderer can crash), and the byte
route then serves the original bytes under their real Content-Type. So a
descriptor may name and label an artifact, and must never decide what it *is*:
:class:`FetchedArtifact` carries the two facts that cannot lie — the bytes
(``is_png``) and the Content-Type the worker served them under. Routing on the
prediction instead loses artifacts silently (osprey #503).

``artifact_ids`` and ``artifact_descriptors`` normalize the run-status body's
``artifacts`` field (osprey #363 descriptor dicts, or bare id strings from an
older worker) to ids and to descriptor dicts respectively.

Nothing here raises to the caller: a fetch failure must degrade to fewer (or
zero) attachments, never break the text answer. NO channel imports, NO osprey
dispatch imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class FetchedArtifact:
    """One artifact as the worker actually served it.

    Attributes:
        data: The bytes, already past the size guard.
        content_type: The served Content-Type with any parameters stripped
            (``text/csv; charset=utf-8`` -> ``text/csv``), or ``None`` when the
            worker sent none. Authoritative over the run descriptor's
            ``delivered_mime``, which only predicted what a fetch would produce.
    """

    data: bytes
    content_type: str | None

    @property
    def is_png(self) -> bool:
        """Whether the bytes really are a PNG — magic bytes, not a header claim."""
        return self.data.startswith(PNG_MAGIC)


def _bare_mime(content_type: str | None) -> str | None:
    """Strip parameters and case from a Content-Type; ``None`` stays ``None``."""
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower() or None


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


def artifact_descriptors(artifacts: list[Any] | None) -> list[dict]:
    """Normalize a run's ``artifacts`` field to descriptor dicts, in order.

    Mirrors ``artifact_ids``'s tolerance for the same input shapes (osprey #363
    descriptor dicts, bare id strings from an older worker, or a mix), but keeps
    the *whole* dict so a delivery path also gets the ``filename`` /
    ``delivered_mime`` naming hints an id string cannot carry; a bare string
    becomes ``{"artifact_id": <id>}``, a descriptor with no hints at all.
    Anything else — a dict without a usable ``artifact_id``, a non-str/non-dict
    element — is skipped rather than raised on, so a malformed entry costs at
    most one artifact, never the whole answer.

    It deliberately does NOT route by ``delivered_mime``: that field is a
    prediction made before anything was rendered, and an artifact whose
    conversion failed at fetch time arrives as its original bytes. Route on what
    :func:`fetch_artifact` actually returns instead (osprey #503).
    """
    out: list[dict] = []
    for a in artifacts or []:
        if isinstance(a, str):
            if a:
                out.append({"artifact_id": a})
        elif isinstance(a, dict):
            aid = a.get("artifact_id")
            if isinstance(aid, str) and aid:
                out.append(a)
        # else: unknown shape -> skip
    return out


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
    max_bytes: int = MAX_DOC_BYTES,
) -> FetchedArtifact | None:
    """Fetch one artifact from the worker; ``None`` on any failure or a payload
    that fails the size guard.

    The only content the fetch rejects is content it could not get or could not
    fit: what the bytes turn out to BE is the caller's decision, taken from the
    returned :class:`FetchedArtifact`. A fetcher that dropped a payload for
    contradicting the run's ``delivered_mime`` prediction would make an artifact
    vanish with no error anywhere (osprey #503).

    ``max_bytes`` defaults to the document ceiling; a caller that will only
    accept an image applies :data:`MAX_ARTIFACT_BYTES` to the returned bytes,
    which are read in full either way.

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
        content_type = resp.headers.get("content-type")
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
    return FetchedArtifact(data=data, content_type=_bare_mime(content_type))
