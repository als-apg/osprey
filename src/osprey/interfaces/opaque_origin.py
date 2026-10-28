"""The opaque origin every agent-written document is served under.

An HTML page, an SVG or an XML document the agent wrote must never run script
on the console's own origin: if it did, it could send a same-origin request
that arms writes. Each such response therefore carries a sandboxing
Content-Security-Policy *without* ``allow-same-origin``, which gives the
document an opaque origin — its requests arrive with ``Origin: null`` and
without the console's cookies, and are refused.

:data:`OPAQUE_CSP` is the one spelling of that header. Every surface that
serves agent-writable bytes imports it (or :func:`opaque_headers`) rather than
writing its own CSP string. The sandbox keeps what an agent-written document
legitimately needs — its own scripts, popups, downloads, and modals (printing
from a sandboxed frame needs ``allow-modals``) — and never grants
``allow-same-origin``, ``allow-forms`` or ``allow-top-navigation``.

A closed list of inert types (:data:`INERT_TYPES`) is exempt: no browser runs
script in a PDF, plain text, JSON, CSV or a raster picture, and Chrome's PDF
viewer does not render under a sandbox CSP. Those keep their existing headers
plus ``X-Content-Type-Options: nosniff``. Anything not on the list — HTML,
XHTML, SVG, any ``+xml`` type, XSL, an unknown or guessed type — is opaque.
"""

from __future__ import annotations

__all__ = ["INERT_TYPES", "OPAQUE_CSP", "document_headers", "is_inert", "opaque_headers"]

#: The sandbox every agent-written, non-inert document is served under.
OPAQUE_CSP = "sandbox allow-scripts allow-modals allow-popups allow-downloads"

#: The closed list of types served without the sandbox. Nothing is added to it
#: by inference: a type not named here is opaque.
INERT_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "text/plain",
        "application/json",
        "text/csv",
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }
)


def opaque_headers() -> dict[str, str]:
    """Return the headers an opaque document carries, as a fresh dict.

    Returns:
        ``Content-Security-Policy`` set to :data:`OPAQUE_CSP` and
        ``X-Content-Type-Options: nosniff``.
    """
    return {
        "Content-Security-Policy": OPAQUE_CSP,
        "X-Content-Type-Options": "nosniff",
    }


def is_inert(mime: str | None) -> bool:
    """Tell whether a response of type ``mime`` may skip the opaque sandbox.

    The type is compared case-insensitively with its parameters (for example
    ``; charset=utf-8``) removed. ``None``, an empty string and every type not
    in :data:`INERT_TYPES` are not inert.

    Args:
        mime: The response's media type, as served.

    Returns:
        ``True`` only for a type on the closed inert list.
    """
    if not mime:
        return False
    essence = mime.split(";", 1)[0].strip().lower()
    return essence in INERT_TYPES


def document_headers(mime: str | None) -> dict[str, str]:
    """Return the headers agent-written bytes served as ``mime`` carry, as a fresh dict.

    Args:
        mime: The response's media type, as served.

    Returns:
        ``X-Content-Type-Options: nosniff`` alone for an inert type, else
        :func:`opaque_headers`.
    """
    if is_inert(mime):
        return {"X-Content-Type-Options": "nosniff"}
    return opaque_headers()
