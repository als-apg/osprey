"""Republish a run's artifacts as public GCS objects, for Chat's ``cardsV2`` widgets.

The widget requires a **publicly fetchable** URL: Google fetches the object
server-side to render the card, so a signed URL that expires — or anything behind
auth — does not render, and an object deleted later breaks an already-posted
message. That is why every object written here is addressed by the fixed public
form ``https://storage.googleapis.com/{bucket}/{name}``, built as a literal rather
than read back off the storage client.

This module is the Google-specific *publish* half of artifact delivery. The
channel-agnostic *fetch* half lives in :mod:`osprey.bridges.core.artifacts`, which
returns the bytes together with the ``Content-Type`` the worker actually served —
the authoritative type of those bytes, because the worker answers the byte route
over the file it really wrote while the run-status descriptor only records what
was *intended*. Both publishers here decide on that served type and nothing else:

*  :func:`publish_artifacts` uploads only what is genuinely a renderable PNG, so a
   conversion that fell back to HTML cannot be posted as an image widget (which
   Chat renders as a broken image);
*  :func:`publish_documents` derives BOTH the object key's extension and the
   upload's ``content_type`` from the served type. Neither the worker's
   ``filename`` nor the descriptor's ``delivered_mime`` can influence them: the
   filename because a name containing ``..`` or ``/`` would otherwise choose where
   the object is written, and ``delivered_mime`` because an object served under a
   type its bytes do not have is exactly the broken-render this routing exists to
   prevent. The sanitized filename survives only as the card's display label.

Neither function raises. Any failure — fetch, oversize payload, mislabelled
payload, upload error, missing credentials — costs at most the one artifact, and
an unset ``gcs_bucket`` costs all of them, because the text answer must still go
out. Nothing here is a startup requirement.
"""

from __future__ import annotations

import functools
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import httpx

from osprey.bridges.core import (
    MAX_DELIVERED_DOC_BYTES,
    MAX_DELIVERED_IMAGE_BYTES,
    ext_for_mime,
    fetch_artifact,
    safe_label,
)

from .config import GoogleChatBridgeConfig

logger = logging.getLogger(__name__)

GCS_SCOPES = ["https://www.googleapis.com/auth/devstorage.read_write"]
"""Bucket-write scope for the bridge's own service account.

``devstorage.read_write`` is the narrowest scope that can write an object — GCS
publishes no write-only scope, so the read half comes along whether or not this
bridge ever uses it (it does not; every call here is an upload). The scope is
therefore NOT what bounds this credential. The IAM role does: the account is
granted ``roles/storage.objectCreator`` on the bucket, which permits creating
objects and nothing else — no read, no overwrite, no delete. Narrowing the
*credential* means changing that role, not this list."""

PUBLIC_URL_HOST = "https://storage.googleapis.com"
"""Origin of GCS's anonymous read route. See the module docstring for why the
public form is built here rather than taken from the storage client."""

IMAGE_PREFIX = "plots/"
"""Key prefix for objects delivered as image widgets."""

DOCUMENT_PREFIX = "docs/"
"""Key prefix for objects delivered as document links."""

IMAGE_CONTENT_TYPE = "image/png"
"""The only type :func:`publish_artifacts` writes. What qualifies is decided by
:attr:`~osprey.bridges.core.FetchedArtifact.is_png` — the bytes, not the header."""

DEFAULT_CONTENT_TYPE = "application/octet-stream"
"""Upload type for a document the worker served with no ``Content-Type`` at all.
An absent type is never guessed from the bytes — it is reported as unknown."""

REQUEST_TIMEOUT = 30.0
"""Budget for one artifact fetch from the worker."""

ClientFactory = Callable[["GoogleChatBridgeConfig"], Any]
"""How a ``google.cloud.storage.Client`` is obtained. The default builds a real
one; tests inject a fake, which is what keeps ``google-cloud-storage`` out of the
test path entirely."""

_Published = TypeVar("_Published")


def _default_gcs_client(cfg: GoogleChatBridgeConfig) -> Any:
    """Build a storage client from the bridge's own service-account key.

    The same identity that posts to Chat owns the bucket writes, so no second
    credential is configured. Both Google imports are **lazy**: this module must
    import on a machine that has no Google libraries installed at all (a dev
    checkout, test collection), and it is only a caller that has actually
    configured a bucket which reaches this function.

    Args:
        cfg: Supplies the service-account key path and the owning project.

    Returns:
        A ``google.cloud.storage.Client``.

    Raises:
        Exception: Whatever the Google libraries raise when they are absent or
            the key is unusable. Callers here treat that as "no image delivery"
            rather than propagating it.
    """
    from google.cloud import storage
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        cfg.sa_key,  # gitleaks:allow - a filesystem path to the key, not a secret
        scopes=GCS_SCOPES,
    )
    return storage.Client(project=cfg.gcs_project or None, credentials=credentials)


def _public_url(bucket: str, name: str) -> str:
    """The anonymous read URL of an object, in the form Chat must be handed."""
    return f"{PUBLIC_URL_HOST}/{bucket}/{name}"


def _upload(
    client: Any,
    cfg: GoogleChatBridgeConfig,
    data: bytes,
    *,
    name: str,
    content_type: str,
) -> str | None:
    """Upload ``data`` to ``name`` in the configured bucket. Never raises.

    Args:
        client: The storage client.
        cfg: Supplies the destination bucket.
        data: Bytes to write.
        name: Full object key, including its prefix and extension.
        content_type: Type the object is served back under. It describes the
            bytes actually being written, never what a descriptor claimed.

    Returns:
        The object's public URL, or ``None`` if the upload failed.
    """
    try:
        bucket = client.bucket(cfg.gcs_bucket)
        blob = bucket.blob(name)
        blob.upload_from_string(data, content_type=content_type)
    except Exception:
        logger.warning("GCS upload failed for %s", name, exc_info=True)
        return None
    return _public_url(cfg.gcs_bucket, name)


def _publish_each(
    cfg: GoogleChatBridgeConfig,
    entries: Sequence[Any],
    *,
    http: httpx.Client | None,
    client_factory: ClientFactory,
    publish_one: Callable[[httpx.Client, Any, Any], _Published | None],
) -> list[_Published]:
    """Run ``publish_one`` over ``entries`` against one worker client and one GCS
    client. Never raises.

    This is the scaffolding both publishers share: the disabled-bucket
    short-circuit, the two clients' lifetimes, and the degradation rules. What an
    entry *is* — a bare artifact id or a descriptor dict — and what publishing it
    yields are entirely ``publish_one``'s business.

    Args:
        cfg: Supplies the bucket and, through ``cfg.core``, the worker route.
        entries: The artifacts to publish, in the order they should be posted.
        http: Client for the worker's byte route. ``None`` builds one for the
            duration of the call; an injected one is left open for its owner.
        client_factory: How to obtain the storage client. Called at most once,
            and never at all when there is nothing to publish.
        publish_one: Called as ``(http_client, gcs_client, entry)``; returns the
            published result, or ``None`` for an entry that could not be
            published. It must not raise.

    Returns:
        One result per successfully published entry, in input order — possibly
        empty, and shorter than ``entries`` when any of them failed.
    """
    if not cfg.gcs_bucket or not entries:
        # No bucket means image delivery is switched off for this deployment: an
        # answer still posts, text-only. Returning before the factory runs is
        # what keeps google-cloud-storage from being needed at all.
        return []

    owns_http = http is None
    http_client: httpx.Client | None = None
    published: list[_Published] = []
    try:
        # Built inside the try because constructing a client is itself a live
        # failure mode: under ``trust_env`` httpx reads proxy settings from the
        # environment and raises on a malformed one.
        http_client = http or httpx.Client(timeout=REQUEST_TIMEOUT, trust_env=cfg.core.trust_env)
        try:
            gcs_client = client_factory(cfg)
        except Exception:
            logger.warning("failed to build GCS client, skipping all artifacts", exc_info=True)
            return []
        for entry in entries:
            result = publish_one(http_client, gcs_client, entry)
            if result is not None:
                published.append(result)
    except Exception:
        # Keep whatever already uploaded: those objects exist and their URLs are
        # valid, so discarding them would orphan them for nothing.
        logger.warning("artifact publication aborted, posting what we have", exc_info=True)
        return published
    finally:
        if owns_http and http_client is not None:
            http_client.close()
    return published


def _publish_image(
    cfg: GoogleChatBridgeConfig,
    run_id: str,
    http_client: httpx.Client,
    gcs_client: Any,
    artifact_id: str,
) -> str | None:
    """Fetch one artifact and publish it as a public PNG. Never raises.

    Args:
        cfg: Supplies the worker route and the destination bucket.
        run_id: Run the artifact belongs to.
        http_client: Client for the worker's byte route.
        gcs_client: The storage client.
        artifact_id: Worker artifact id to fetch.

    Returns:
        The object's public URL, or ``None`` when the artifact could not be
        fetched, is not a renderable PNG, or failed to upload.
    """
    fetched = fetch_artifact(
        http_client, cfg.core, run_id, artifact_id, max_bytes=MAX_DELIVERED_IMAGE_BYTES
    )
    if fetched is None:
        return None  # fetch_artifact logged why: transport, HTTP status, or oversize.
    if not fetched.is_png:
        # Not dropped for being unwanted — dropped for being undeliverable on
        # THIS path. A caller that wants it anyway can offer the same artifact to
        # publish_documents, which makes no claim about the payload.
        #
        # The bytes decide, not the served header: this object is written to a
        # public bucket labelled image/png, and the magic bytes are what make that
        # label true. A real PNG served with no Content-Type at all is still a PNG
        # and is still publishable — dropping it over a missing header is the
        # silent artifact loss #503 exists to prevent.
        logger.warning(
            "artifact %s/%s is not a PNG (served as %s); no image card",
            run_id,
            artifact_id,
            fetched.content_type or "no content type",
        )
        return None
    return _upload(
        gcs_client,
        cfg,
        fetched.data,
        name=f"{IMAGE_PREFIX}{uuid.uuid4()}.png",
        content_type=IMAGE_CONTENT_TYPE,
    )


def _publish_document(
    cfg: GoogleChatBridgeConfig,
    run_id: str,
    http_client: httpx.Client,
    gcs_client: Any,
    descriptor: Mapping[str, Any],
) -> dict[str, str] | None:
    """Fetch one artifact and publish it as a public document. Never raises.

    The object key is settled only once the bytes are in hand, because its
    extension comes from the served type — the whole point of the routing.

    Args:
        cfg: Supplies the worker route and the destination bucket.
        run_id: Run the artifact belongs to.
        http_client: Client for the worker's byte route.
        gcs_client: The storage client.
        descriptor: The artifact's run-status descriptor. Only ``artifact_id``
            (what to fetch) and ``filename`` (the display label) are read;
            ``delivered_mime`` is deliberately ignored — see the module
            docstring.

    Returns:
        ``{"url", "filename", "mime"}`` for the published object, or ``None``
        when the descriptor names no artifact, or the fetch or upload failed.
    """
    artifact_id = descriptor.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        return None
    fetched = fetch_artifact(
        http_client, cfg.core, run_id, artifact_id, max_bytes=MAX_DELIVERED_DOC_BYTES
    )
    if fetched is None:
        return None  # fetch_artifact logged why: transport, HTTP status, or oversize.
    content_type = fetched.content_type or DEFAULT_CONTENT_TYPE
    extension = ext_for_mime(fetched.content_type)
    url = _upload(
        gcs_client,
        cfg,
        fetched.data,
        name=f"{DOCUMENT_PREFIX}{uuid.uuid4()}{extension}",
        content_type=content_type,
    )
    if url is None:
        return None
    return {
        "url": url,
        # The one place a worker-supplied name is allowed, and only sanitized:
        # a label is shown to a person, it does not address anything.
        "filename": safe_label(descriptor.get("filename"), f"artifact{extension}"),
        "mime": content_type,
    }


def publish_artifacts(
    cfg: GoogleChatBridgeConfig,
    run_id: str,
    artifact_ids: Sequence[str],
    *,
    http: httpx.Client | None = None,
    client_factory: ClientFactory = _default_gcs_client,
) -> list[str]:
    """Publish a run's images as public GCS objects. Never raises.

    Each artifact is fetched from the worker and, if it is genuinely a renderable
    PNG (``FetchedArtifact.is_png``), written to ``plots/{uuid4}.png`` and returned as
    a public URL for a ``cardsV2`` image widget. An artifact that is not — the
    fallback-conversion case, where the descriptor said PNG and the worker served
    something else — yields no URL here: it is **not deliverable as an image**,
    and posting its URL anyway would put a broken widget in the space. Offering
    the same artifact to :func:`publish_documents` publishes it under the type it
    actually has, at the cost of fetching it a second time.

    Results are in input order, and a failure of any kind costs exactly one URL,
    so the caller can always still post the text answer. Note that a short result
    is therefore not positionally alignable with ``artifact_ids``; a caller that
    needs the id-to-URL correspondence must check the counts match first.

    Args:
        cfg: Bridge config. An unset ``gcs_bucket`` disables image delivery and
            returns ``[]`` without building any client.
        run_id: Run whose artifacts these are.
        artifact_ids: Worker artifact ids, in the order they should be posted.
        http: Client for the worker's byte route. Tests inject an
            :class:`httpx.MockTransport`-backed client; production leaves it
            ``None`` and one is built and closed here.
        client_factory: How to obtain the storage client. Tests inject a fake, so
            no credentials, no network, and no Google libraries are involved.

    Returns:
        Public URLs of the objects written, possibly empty.
    """
    return _publish_each(
        cfg,
        artifact_ids,
        http=http,
        client_factory=client_factory,
        publish_one=functools.partial(_publish_image, cfg, run_id),
    )


def publish_documents(
    cfg: GoogleChatBridgeConfig,
    run_id: str,
    doc_descriptors: Sequence[Mapping[str, Any]],
    *,
    http: httpx.Client | None = None,
    client_factory: ClientFactory = _default_gcs_client,
) -> list[dict[str, str]]:
    """Publish a run's non-image artifacts as public GCS objects. Never raises.

    Each descriptor's artifact is fetched from the worker and written to
    ``docs/{uuid4}{ext}``, where both ``ext`` and the upload's ``content_type``
    come from the ``Content-Type`` the worker served with the bytes — an
    unrecognized type keeps its own ``content_type`` under a ``.bin`` key, and no
    type at all is written as ``application/octet-stream``. Nothing about the
    object key comes from the descriptor or from the worker's ``filename``; see
    the module docstring for why both are excluded.

    Because ``delivered_mime`` is never read, a descriptor from the image bucket
    can be offered here as-is when :func:`publish_artifacts` declined it, and it
    is published under whatever it really is.

    Results are in input order, and any per-document failure costs exactly one
    entry — including a descriptor carrying no usable ``artifact_id``, which is
    skipped rather than raised on. As with :func:`publish_artifacts`, a short
    result is not positionally alignable with the input.

    Args:
        cfg: Bridge config. An unset ``gcs_bucket`` disables document delivery
            and returns ``[]`` without building any client.
        run_id: Run whose artifacts these are.
        doc_descriptors: Run-status descriptors, in the order they should be
            posted.
        http: Client for the worker's byte route; see :func:`publish_artifacts`.
        client_factory: How to obtain the storage client; see
            :func:`publish_artifacts`.

    Returns:
        One ``{"url", "filename", "mime"}`` mapping per published document,
        possibly empty. ``filename`` is the display label only, and ``mime`` is
        the type the object is served back under.
    """
    return _publish_each(
        cfg,
        doc_descriptors,
        http=http,
        client_factory=client_factory,
        publish_one=functools.partial(_publish_document, cfg, run_id),
    )
