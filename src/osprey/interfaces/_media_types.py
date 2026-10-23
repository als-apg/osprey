"""The content types OSPREY's web interfaces serve static assets as.

``mimetypes`` reads the host's own tables (``/etc/mime.types`` and friends)
on top of Python's defaults, so the same file can be served with a different
type on a different machine: RHEL maps ``.js`` to ``application/javascript``
where Python alone says ``text/javascript``. Every route that serves OSPREY's
assets looks its type up here first, so the answer does not depend on the host.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import FileResponse, Response
from starlette.staticfiles import NotModifiedResponse, StaticFiles
from starlette.types import Scope

#: Text asset suffixes, mapped to the content type to serve them as. The
#: web terminal's design-system panel route additionally rewrites the bodies
#: of the files listed here.
STATIC_TEXT_TYPES = {
    ".css": "text/css",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".html": "text/html",
    ".json": "application/json",
    ".map": "application/json",
}

#: Font suffixes, mapped to the content type to serve them as.
STATIC_FONT_TYPES = {".ttf": "font/ttf"}


def asset_media_type(candidate: Path) -> str:
    """The content type to serve *candidate* as.

    :data:`STATIC_TEXT_TYPES` and :data:`STATIC_FONT_TYPES` first — they are
    the deliberate mapping — then ``mimetypes`` for images and the rest, then
    a bytes-shaped fallback so an unknown suffix is still served rather than
    guessed wrong as text.
    """
    suffix = candidate.suffix.lower()
    mapped = STATIC_TEXT_TYPES.get(suffix) or STATIC_FONT_TYPES.get(suffix)
    if mapped is not None:
        return mapped
    guessed, _ = mimetypes.guess_type(candidate.name)
    return guessed or "application/octet-stream"


class PinnedTypeStaticFiles(StaticFiles):
    """``StaticFiles`` that serves the suffixes mapped above with their mapped type.

    Everything else about the response — conditional requests, ranges, the
    unmapped suffixes — is Starlette's own.
    """

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        suffix = Path(full_path).suffix.lower()
        media_type = STATIC_TEXT_TYPES.get(suffix) or STATIC_FONT_TYPES.get(suffix)
        if media_type is None:
            return super().file_response(full_path, stat_result, scope, status_code)
        response = FileResponse(
            full_path, status_code=status_code, stat_result=stat_result, media_type=media_type
        )
        if self.is_not_modified(response.headers, Headers(scope=scope)):
            return NotModifiedResponse(response.headers)
        return response
