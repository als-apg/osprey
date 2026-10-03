"""The one HTTP response every ARIEL attachment download goes through.

The bytes being sent decide how they are served; a stored or declared content
type never does. Only a raster picture the format registry accepts is answered
``inline`` with its sniffed type. Anything else — a PDF, an SVG, a web page
named like a picture — is an ``application/octet-stream`` download, so a
browser never renders or executes it on this origin.
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi.responses import Response

from osprey.imaging.formats import OCTET_STREAM, SNIFF_BYTES, sniff

#: Applied to every attachment response: no script, no plugin, no subresource.
ATTACHMENT_CSP = "sandbox; default-src 'none'"


def attachment_response(data: bytes, filename: str) -> Response:
    """Build the response that serves ``data`` as the attachment ``filename``.

    The first :data:`~osprey.imaging.formats.SNIFF_BYTES` bytes are sniffed. A
    raster picture is served ``inline`` with its sniffed MIME type; everything
    else is served as an ``application/octet-stream`` ``attachment``. Every
    response carries ``X-Content-Type-Options: nosniff`` and a sandboxing
    Content-Security-Policy, and names the file only through the RFC 5987
    ``filename*`` parameter, so no quote or line break in a name reaches a
    header unencoded.

    Args:
        data: The bytes to send.
        filename: The name the browser should give the file.

    Returns:
        The response carrying ``data``.
    """
    result = sniff(data[:SNIFF_BYTES])
    if result.is_image:
        media_type, disposition = result.mime, "inline"
    else:
        media_type, disposition = OCTET_STREAM, "attachment"
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(filename, safe='')}",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": ATTACHMENT_CSP,
        },
    )
