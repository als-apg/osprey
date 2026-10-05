"""Turn attachment bytes into the stored picture fields: the one rendition creator.

:func:`prepare_picture` is called by the copy step and by the native writers,
always with no database connection held. It first classifies the bytes by magic
number in-process (:func:`~osprey.imaging.formats.sniff`, no Pillow): a non-image
or a reserved format is answered at once with its skip reason and sniffed MIME
type, and no render worker is spawned. Only an accepted picture is handed to the
isolated render worker (:func:`~osprey.imaging.render.render_isolated`).

This module reads and writes no database state; callers persist the returned
fields. :class:`~osprey.imaging.render.RenderUnavailable` propagates, so the
caller decides how an unavailable worker is recorded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from osprey.imaging import render as _render
from osprey.imaging.formats import sniff
from osprey.imaging.render import RenderUnavailable

__all__ = ["PreparedPicture", "RenderUnavailable", "prepare_picture"]


@dataclass(frozen=True)
class PreparedPicture:
    """The picture fields of one ``attachment_files`` row.

    Attributes:
        mime_type: The sniffed MIME type of the original bytes.
        skip_reason: ``None`` when a rendition was made, else the content skip
            reason (``not_an_image``, ``reserved_format`` or a render refusal).
        rendition_bytes: The encoded rendition, or ``None``.
        rendition_mime: ``image/png`` or ``image/jpeg``, or ``None``.
        rendition_w: Rendition width in pixels, or ``None``.
        rendition_h: Rendition height in pixels, or ``None``.
        rendition_sha256: Hex sha256 of ``rendition_bytes``, or ``None``.
    """

    mime_type: str
    skip_reason: str | None
    rendition_bytes: bytes | None = None
    rendition_mime: str | None = None
    rendition_w: int | None = None
    rendition_h: int | None = None
    rendition_sha256: str | None = None

    @property
    def has_rendition(self) -> bool:
        """Whether a rendition was produced."""
        return self.rendition_bytes is not None


async def prepare_picture(data: bytes, *, task_id: str | None = None) -> PreparedPicture:
    """Sniff ``data`` and, for an accepted picture, render it in the isolated worker.

    Args:
        data: The original attachment bytes.
        task_id: Identity of the picture for the worker's failure rules; defaults
            to the sha256 of ``data``.

    Returns:
        A :class:`PreparedPicture`: rendition fields for a rendered picture, or
        the sniffed MIME type with a skip reason and no rendition.

    Raises:
        RenderUnavailable: The render worker could not be run; no content verdict
            was reached.
    """
    sniffed = sniff(data)
    if not sniffed.is_image:
        return PreparedPicture(mime_type=sniffed.mime, skip_reason=sniffed.skip_reason)

    outcome = await _render.render_isolated(data, task_id=task_id)
    rendition = outcome.rendition
    if rendition is None:
        return PreparedPicture(mime_type=sniffed.mime, skip_reason=outcome.reason)
    return PreparedPicture(
        mime_type=sniffed.mime,
        skip_reason=None,
        rendition_bytes=rendition.data,
        rendition_mime=rendition.mime,
        rendition_w=rendition.width,
        rendition_h=rendition.height,
        rendition_sha256=hashlib.sha256(rendition.data).hexdigest(),
    )
