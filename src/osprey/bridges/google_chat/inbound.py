"""Download the files a user attached to an inbound Google Chat message.

When a human @mentions (or DMs) the app with files attached, each *uploaded*
attachment on the Chat ``Message`` resource carries an
``attachmentDataRef.resourceName``:opaque handle the app can fetch with its own
``chat.bot`` credential. :func:`~osprey.bridges.google_chat.events.parse_attachments`
already turned those into JSON-plain reference dicts (and split the Drive ones off
by name); this module performs the second step, the authenticated media GET
(``GET /v1/media/{resourceName}?alt=media``), and shapes the bytes into the
worker-payload dicts the engine folds into ``input_files``.

The GET is issued through an injected session whose ``.get()`` returns a response
exposing ``.content`` and ``.raise_for_status()`` — in production an
``AuthorizedSession`` built by :func:`build_media_session` (the ``chat.bot`` bearer
baked in, the same identity that posts replies); in tests an ``httpx.Client`` over a
``MockTransport``. No network and no credential is involved in a test run, and
``google-auth`` is imported lazily so this module imports without it installed.

:func:`download_attachments` NEVER raises. Any per-file failure — an unsupported
type, an oversize payload, a bad reference, a transport error, an unexpected
exception — degrades that ONE file to a ``{"filename", "reason"}`` skip note that
rides the payload so the agent can relay why; every other attachment is still
processed. Callers get an answer even when every attachment is declined.

What this module owns, and what it does not:

*   OWNED — the MIME allowlist and the per-type byte caps. Chat accepts any file
    type; the agent ingests images (rendered) and small text/data files (read as
    context), so anything else is declined by name rather than downloaded and
    dropped. The caps are per *file* and per *type*.
*   NOT OWNED — the per-message file-count and total-bytes budget, and the
    ``input_files`` capability probe. Those belong to the engine
    (:data:`~osprey.bridges.core.pipeline.MAX_FILES`,
    :data:`~osprey.bridges.core.pipeline.MAX_TOTAL_BYTES`), which applies one shared
    budget across the own and quoted provenance buckets of ``ops.download_inputs``.
    An adapter that re-applied either would double-count a budget the engine already
    enforces, or probe the dispatch pair from the adapter — which it must never do.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from osprey.bridges.core import safe_label

# Re-exported, never redeclared: the media download runs under the app's OWN identity —
# the very one it posts with — so it has to be the same scope list, not a second copy
# that happens to match today. The media a user sent *this app* is readable under it, so
# no user OAuth token is ever involved. Widening the app's scopes is therefore one edit,
# and cannot leave the download leg authorized for less than the posting leg.
from .client import SA_SCOPES

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FILENAME",
    "DOWNLOAD_FAILED_REASON",
    "DRIVE_SKIP_REASON",
    "IMAGE_MAX_BYTES",
    "IMAGE_MIMES",
    "MEDIA_BASE",
    "NO_REFERENCE_REASON",
    "SA_SCOPES",
    "TEXT_MAX_BYTES",
    "TEXT_MIMES",
    "UNPROCESSABLE_REASON",
    "MediaResponse",
    "MediaSession",
    "build_media_session",
    "download_attachments",
]


MEDIA_BASE = "https://chat.googleapis.com/v1/media"
"""Chat media download endpoint. The attachment's ``resourceName`` is appended as a
path segment (it already contains its own ``/``, so it is not escaped) and
``alt=media`` asks for the raw bytes rather than the metadata resource."""

IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
"""Image types the agent can look at. Deliberately the same set as the engine's
:data:`~osprey.bridges.core.pipeline.IMAGE_MIMES`, which decides which prior images
are replayed into a follow-up dispatch: ingesting a type the engine would never
replay would mean an image the agent can see now and silently cannot see one turn
later. The equality is pinned by a test rather than an import, since an adapter
reads the engine's public surface (``osprey.bridges.core``) and not its internals."""

IMAGE_MAX_BYTES = 5 * 1024 * 1024
"""Per-image cap. Generous: an image is the payload, and the engine's total budget
is what bounds a whole message.

Deliberately equal to the worker's own per-image ingest cap
(``input_files_policy.MAX_IMAGE_BYTES``), and pinned by a test rather than imported —
same reasoning as :data:`IMAGE_MIMES`. Accepting a file the worker will reject wastes a
download and turns a clean "too large" note into a fatal 400 mid-dispatch; accepting
less than it would makes the bridge the stricter party for no reason."""

TEXT_MIMES = frozenset({"text/csv", "text/plain", "text/markdown", "application/json"})
"""Text/data types the agent reads as context."""

TEXT_MAX_BYTES = 100 * 1024
"""Per-text-file cap. Tight: these land in the prompt as context, where a large
file costs far more than it carries.

Equal to ``input_files_policy.MAX_TEXT_BYTES`` on purpose, and pinned by the same
test as :data:`IMAGE_MAX_BYTES`."""

DEFAULT_FILENAME = "attachment"
"""Display name for an attachment Chat sent without a usable ``contentName``. Every
skip note needs *some* name, or the agent cannot tell the user which file it lost."""

DRIVE_SKIP_REASON = "Drive files can't be downloaded by the app"
"""Reason for a Drive-backed attachment: it exposes only a ``driveDataRef``, which
the app's own ``chat.bot`` credential cannot read, so it is never fetched.
:func:`~osprey.bridges.google_chat.events.parse_attachments` splits these out by
name before they reach here, and the ops layer pairs each name with this reason —
the source check below is the second line of defence, so the two skip paths read
identically to the user."""

NO_REFERENCE_REASON = "no download reference available"
"""Reason for a reference carrying no ``resource_name``: there is nothing to GET."""

DOWNLOAD_FAILED_REASON = "download failed"
"""Reason for a media GET that errored — a non-2xx status, a transport failure, a
deleted message. Deliberately not the exception text: a user-facing note should not
carry a URL or a stack detail."""

UNPROCESSABLE_REASON = "could not process attachment"
"""Reason attached by the per-file backstop, so an unexpected failure costs one
attachment rather than the whole batch."""


class MediaResponse(Protocol):
    """The response shape :func:`download_attachments` needs from a media GET.

    Satisfied by both ``httpx.Response`` (tests) and ``requests.Response`` (what an
    ``AuthorizedSession`` returns in production), so neither library has to be named
    in a signature.
    """

    @property
    def content(self) -> bytes:
        """The raw response body."""

    def raise_for_status(self) -> object:
        """Raise if the response carried an error status."""


class MediaSession(Protocol):
    """An authorized, GET-capable session for the Chat media endpoint.

    Injected rather than constructed, which is what keeps the download path
    hermetic under test: production passes :func:`build_media_session`'s
    ``AuthorizedSession``, tests pass an ``httpx.Client`` over a ``MockTransport``.
    """

    def get(self, url: str, *, params: Mapping[str, str]) -> MediaResponse:
        """GET ``url`` with ``params`` as the query string."""


def build_media_session(sa_key: str) -> MediaSession:
    """Build an authorized media-download session from the service-account key.

    ``google.auth`` is imported here rather than at module scope so this module —
    and the ops layer that imports it — loads without ``google-auth`` installed,
    which is what lets the whole inbound path be developed and tested on a machine
    that has none of the Google libraries.

    Args:
        sa_key: Path to the service-account JSON key file (the ``GCHAT_SA_KEY``
            deployment setting).

    Returns:
        A session whose ``.get()`` carries the ``chat.bot`` bearer automatically.
    """
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(sa_key, scopes=list(SA_SCOPES))
    session: MediaSession = AuthorizedSession(creds)
    return session


def download_attachments(
    session: MediaSession, refs: Iterable[Mapping[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Download one bucket of inbound Chat attachments. Never raises.

    Args:
        session: An authorized GET-capable session (see :class:`MediaSession`).
        refs: The reference dicts :func:`~osprey.bridges.google_chat.events.parse_attachments`
            built — ``content_name`` / ``content_type`` / ``resource_name`` /
            ``source`` — read straight off the persisted entry, so the retry drain
            and a post-restart reconcile fetch exactly what the live path did.
            ``None`` (and an empty list) is the no-op.

    Returns:
        ``(files, skipped)``. Each item of ``files`` is a worker-payload dict —
        ``{"filename", "mime", "content_b64", "ingest": True}`` — and each item of
        ``skipped`` is a ``{"filename", "reason"}`` note. Every reference lands in
        exactly one of the two, in the order Chat listed them, so a declined file is
        reported rather than silently absent.

    Guards, in order — a file that trips one is skipped, and is never downloaded
    past the point it fails: a Drive-backed source is not fetchable at all; the MIME
    must be on the allowlist (:data:`IMAGE_MIMES` / :data:`TEXT_MIMES`); the
    reference must carry a ``resource_name``; the GET must succeed; and the payload
    must be within its per-type cap. The engine applies the per-message count and
    total-bytes budget afterwards, across this bucket and the quoted one together.
    """
    files: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    try:
        ref_list = list(refs or [])
    except TypeError:
        # A non-iterable is a caller bug, but this function still must not raise.
        logger.warning("download_attachments got non-iterable references; treating as none")
        return files, skipped

    for ref in ref_list:
        if not isinstance(ref, Mapping):
            # Malformed persisted state, not a file the user sent: there is no name
            # to report it under, so it is dropped rather than turned into a note.
            logger.warning("dropping a non-mapping inbound attachment reference: %r", ref)
            continue

        # Inside the try, and seeded first: the backstop below needs a name to
        # report under even when reading the reference is what failed.
        filename = DEFAULT_FILENAME
        try:
            filename = safe_label(_text(ref, "content_name"), DEFAULT_FILENAME)

            # The three guards answerable from the reference alone come first, so an
            # undownloadable, unsupported or unreferenced file costs no round trip.
            if _is_drive(_text(ref, "source")):
                skipped.append({"filename": filename, "reason": DRIVE_SKIP_REASON})
                continue

            mime = _text(ref, "content_type")
            cap = _cap_for(mime)
            if cap is None:
                skipped.append(
                    {"filename": filename, "reason": f"unsupported type {mime or 'unknown'}"}
                )
                continue

            resource_name = _text(ref, "resource_name")
            if not resource_name:
                skipped.append({"filename": filename, "reason": NO_REFERENCE_REASON})
                continue

            try:
                data = _fetch(session, resource_name)
            except Exception as exc:
                logger.warning("attachment download failed for %r: %s", filename, exc)
                skipped.append({"filename": filename, "reason": DOWNLOAD_FAILED_REASON})
                continue

            # ``>``, not ``>=``: a file exactly at the cap is within it.
            if len(data) > cap:
                skipped.append(
                    {
                        "filename": filename,
                        "reason": f"too large ({_human_size(len(data))} > "
                        f"{_human_size(cap)} limit)",
                    }
                )
                continue

            files.append(
                {
                    "filename": filename,
                    "mime": mime or "",
                    "content_b64": base64.b64encode(data).decode("ascii"),
                    # Fresh input, as opposed to a replayed prior artifact: the worker
                    # budgets these against its fresh-file caps.
                    "ingest": True,
                }
            )
        except Exception:
            # Backstop: no single attachment may break the whole batch.
            logger.warning("unexpected error handling attachment %r", filename, exc_info=True)
            skipped.append({"filename": filename, "reason": UNPROCESSABLE_REASON})

    return files, skipped


def _cap_for(mime: str | None) -> int | None:
    """Per-file byte cap for ``mime``, or ``None`` if the type is not allowed.

    The single source of truth for both halves of the question "may this file be
    ingested, and how big may it be", so the allowlist cannot drift from the caps.
    """
    if mime in IMAGE_MIMES:
        return IMAGE_MAX_BYTES
    if mime in TEXT_MIMES:
        return TEXT_MAX_BYTES
    return None


def _fetch(session: MediaSession, resource_name: str) -> bytes:
    """Media GET for one ``resourceName``; returns the raw bytes or raises."""
    response = session.get(f"{MEDIA_BASE}/{resource_name}", params={"alt": "media"})
    response.raise_for_status()
    return response.content


def _is_drive(source: str | None) -> bool:
    """True for a Drive-backed attachment, across the source spellings Chat uses."""
    return "DRIVE" in (source or "").upper()


def _text(ref: Mapping[str, Any], key: str) -> str | None:
    """``ref[key]`` when it is a string, else ``None``.

    The references are JSON off the persisted entry, so a non-string is untrusted
    shape rather than a programming error and reads as absent.
    """
    value = ref.get(key)
    return value if isinstance(value, str) else None


def _human_size(nbytes: int) -> str:
    """Compact size for a user-facing reason: ``7.2MB``, ``100KB`` (no ``.0``)."""
    if nbytes >= 1024 * 1024:
        value, unit = nbytes / (1024 * 1024), "MB"
    else:
        value, unit = nbytes / 1024, "KB"
    return f"{f'{value:.1f}'.rstrip('0').rstrip('.')}{unit}"
