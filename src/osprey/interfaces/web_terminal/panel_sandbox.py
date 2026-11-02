"""The terminal's default-deny opaque origin for everything it serves as a panel.

Every panel the terminal serves — proxied under ``/panel/<id>/``, framework or
sidecar, and every discovered bundle under ``/panel-static/<id>/`` — shares the
terminal's own origin. A document served there that an agent could have written
(an artifact, a user's ``.html`` file, a page a runtime registration points at,
a discovered bundle) would run with the operator's session and same-origin reach
into every terminal API. So the terminal answers such responses with
:data:`~osprey.interfaces.opaque_origin.OPAQUE_CSP` (the policy the artifact
gallery's content sandbox applies too), which puts the document in an opaque origin
(``self.origin === 'null'``): it runs, it may offer a download, and it holds no
cookie and no same-origin reach.

Three rules decide, and every one of them fails toward the sandbox:

* **Content type.** Only a closed allowlist of non-document types
  (:data:`NON_DOCUMENT_TYPES`: JSON, PNG, JPEG, GIF, WebP, CSS, JavaScript,
  fonts) and the opaque origin's inert list
  (:data:`~osprey.interfaces.opaque_origin.INERT_TYPES`: PDF, plain text, CSV,
  which no browser runs script in and Chrome's PDF viewer does not render
  under a sandbox) go out untouched. Everything else is a document — HTML,
  and also SVG, XML and a missing type, which a browser renders or sniffs into
  one — and carries the sandbox plus ``X-Content-Type-Options: nosniff``.
* **Entry pages.** A panel's own application documents must stay same-origin to
  drive their API, so a response whose path matches one of the panel's declared
  entry-page templates is exempt. The templates are declared per panel:
  :attr:`~osprey.profiles.web_panels.SidecarPanel.entry_pages` for a
  sidecar, :data:`FRAMEWORK_ENTRY_PAGES` for a companion server, and a
  config-declared panel's own ``path`` (an operator declaration). A runtime
  registration and ``/panel-static`` declare none, ever.
* **Normalisation.** The path is matched only after percent-decoding (to a fixed
  point, :func:`osprey.interfaces.panel_urls.decoded_readings`) and dot-segment
  removal, with ``\\`` read as ``/``, so a spelling a
  backend resolves elsewhere (``lab/%2e%2e/files/x.html``) cannot borrow an
  entry page's exemption. A path is exempt only if *every* reading of it
  matches.

The header is **its own policy**: appended as a separate
``Content-Security-Policy`` header, never merged into an upstream one, so an
upstream ``sandbox allow-scripts allow-same-origin`` intersects with it and can
never loosen it.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable

from osprey.interfaces.opaque_origin import INERT_TYPES, OPAQUE_CSP
from osprey.interfaces.panel_urls import decoded_readings
from osprey.profiles.web_panels import SIDECAR_PANELS

#: The response header the directive travels in.
CSP_HEADER = "content-security-policy"

#: The header that stops a browser sniffing a document out of a sandboxed body.
NOSNIFF_HEADER = "x-content-type-options"

#: Content types (lower-case, parameters stripped) a browser never renders as a
#: document. Closed on purpose: a type not named here is treated as a document.
NON_DOCUMENT_TYPES = frozenset(
    {
        # JSON
        "application/json",
        # Raster images
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        # Stylesheets
        "text/css",
        # JavaScript
        "text/javascript",
        "application/javascript",
        "application/x-javascript",
        "application/ecmascript",
        "text/ecmascript",
        # Fonts
        "font/woff",
        "font/woff2",
        "font/ttf",
        "font/otf",
        "font/collection",
        "application/font-woff",
        "application/font-woff2",
        "application/x-font-ttf",
        "application/x-font-otf",
        "application/vnd.ms-fontobject",
    }
)

#: The entry documents of each companion-server panel, keyed by panel id. Each
#: is the application page its server answers at ``/``; every other document
#: such a server returns (an artifact, a rendered notebook, an attachment, an
#: export) is content and is sandboxed.
FRAMEWORK_ENTRY_PAGES: dict[str, tuple[str, ...]] = {
    # The gallery shell (artifacts/app.py ``@not_content`` root).
    "artifacts": ("/",),
    # The ARIEL logbook SPA (ariel/app.py root).
    "ariel": ("/",),
    # The channel-finder SPA (channel_finder/app.py root).
    "channel-finder": ("/",),
    # The lattice dashboard (lattice_dashboard/app.py root template).
    "lattice": ("/",),
    # The OKF knowledge reader (okf_panel/app.py root).
    "okf": ("/",),
    # The system-health dashboard (health/app.py root).
    "system-health": ("/",),
}

#: One MIME essence, ``type/subtype``, in RFC 9110 token characters only.
_MIME_ESSENCE_RE = re.compile(r"[!#$%&'*+.^_`|~0-9a-z-]+/[!#$%&'*+.^_`|~0-9a-z-]+")


def base_content_type(content_type: str | None) -> str:
    """*content_type* lower-cased with its parameters dropped (``""`` when absent)."""
    if not content_type:
        return ""
    # HTTP whitespace only: any other byte left in the essence fails the grammar.
    return content_type.split(";", 1)[0].strip(" \t").lower()


def is_non_document_type(content_type: str | None) -> bool:
    """Whether *content_type* is on the closed allowlist of non-document types.

    ``application/<x>+json`` counts as JSON. A missing type is a document, and
    so is any value that is not one well-formed MIME type: a browser reads a
    comma-separated ``Content-Type`` as a list and takes the *last* type it can
    parse (``application/json;x=,text/html`` renders as HTML), so a comma
    anywhere, or an essence outside the token grammar, fails closed.
    """
    if content_type is None or "," in content_type:
        return False
    base = base_content_type(content_type)
    if not _MIME_ESSENCE_RE.fullmatch(base):
        return False
    if base in NON_DOCUMENT_TYPES or base in INERT_TYPES:
        return True
    return base.startswith("application/") and base.endswith("+json")


def _normalise_once(path: str) -> str | None:
    """One reading of *path*: ``\\`` as ``/``, empty segments collapsed, dots removed.

    Returns ``None`` for a path carrying a control character, which no entry
    page contains and which a backend may read in ways this cannot predict.
    """
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        return None
    path = path.replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    trailing = path.endswith(("/", "/.", "/.."))
    normalised = posixpath.normpath(path)
    # normpath keeps a leading '//' (POSIX implementation-defined); collapse it.
    while normalised.startswith("//"):
        normalised = normalised[1:]
    if trailing and normalised != "/":
        normalised += "/"
    return normalised


def path_readings(path: str) -> tuple[str, ...] | None:
    """Every normalised reading of *path* a backend might act on.

    The path as received (already decoded once by the ASGI server) and the path
    percent-decoded to a fixed point, each normalised by :func:`_normalise_once`.
    ``None`` when a reading is unusable (a control character, or decoding that
    never settles), which callers treat as "matches no entry page".
    """
    decoded = decoded_readings(path)
    if decoded is None:
        return None
    readings: list[str] = []
    for current in decoded:
        reading = _normalise_once(current)
        if reading is None:
            return None
        readings.append(reading)
    return tuple(dict.fromkeys(readings))


def _template_matches(template: str, reading: str) -> bool:
    """Whether one normalised *reading* matches one entry-page *template*.

    A template is a root-relative path. A final ``*`` segment matches the rest
    of the path, any depth (``/s/*`` matches ``/s/a`` and ``/s/a/b``); any
    other template matches its own path exactly, a trailing ``/`` ignored.
    """
    if template.endswith("/*"):
        stem = template[:-1]  # keep the slash: '/s/*' -> '/s/'
        return reading.startswith(stem) or reading == stem.rstrip("/")
    return reading.rstrip("/") == template.rstrip("/")


def validate_entry_template(template: str) -> str | None:
    """Why *template* is not a valid entry-page template, or ``None`` when it is."""
    if not isinstance(template, str) or not template.startswith("/"):
        return f"entry page {template!r} must be a root-relative path"
    if "*" in template[:-1] or (template.endswith("*") and not template.endswith("/*")):
        return f"entry page {template!r}: '*' is allowed only as the final segment"
    if template.startswith("/panel-static") or "%" in template or "\\" in template:
        return f"entry page {template!r} is not a plain panel path"
    normalised = _normalise_once(template.removesuffix("*"))
    if normalised is None or normalised.rstrip("/") != template.removesuffix("*").rstrip("/"):
        return f"entry page {template!r} is not in normal form"
    return None


def matches_entry_page(path: str, entry_pages: Iterable[str]) -> bool:
    """Whether *path* (below the panel's mount) is one of its entry pages.

    Exempt only when every reading of the path (:func:`path_readings`) matches
    some template; an invalid template matches nothing.
    """
    templates = [t for t in entry_pages if validate_entry_template(t) is None]
    if not templates:
        return False
    readings = path_readings(path)
    if not readings:
        return False
    return all(any(_template_matches(t, r) for t in templates) for r in readings)


def needs_panel_sandbox(path: str, content_type: str | None, entry_pages: Iterable[str]) -> bool:
    """Whether a panel response at *path* with *content_type* gets the sandbox."""
    if is_non_document_type(content_type):
        return False
    return not matches_entry_page(path, entry_pages)


def builtin_entry_pages(panel_id: str) -> tuple[str, ...]:
    """The entry pages a built-in panel declares: a sidecar's field, else the framework table."""
    sidecar = SIDECAR_PANELS.get(panel_id)
    if sidecar is not None:
        return tuple(sidecar.entry_pages)
    return FRAMEWORK_ENTRY_PAGES.get(panel_id, ())


def sandbox_header_pairs(raw_headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """The raw header pairs to append to a sandboxed response with *raw_headers*.

    The policy, as its own header, unless the response already carries that
    very policy (the opaque origin's, which an OSPREY route sets itself): an
    identical second copy restricts nothing more. ``nosniff`` unless the
    response already carries an ``X-Content-Type-Options`` of its own.
    """
    policy = OPAQUE_CSP.encode("latin-1")
    pairs: list[tuple[bytes, bytes]] = []
    if not any(
        name.lower() == CSP_HEADER.encode("latin-1") and value.strip() == policy
        for name, value in raw_headers
    ):
        pairs.append((CSP_HEADER.encode("latin-1"), policy))
    if not any(name.lower() == NOSNIFF_HEADER.encode("latin-1") for name, _ in raw_headers):
        pairs.append((NOSNIFF_HEADER.encode("latin-1"), b"nosniff"))
    return pairs


def apply_panel_sandbox(response: object, path: str, entry_pages: Iterable[str]) -> bool:
    """Append the sandbox to a Starlette *response* when the rules above call for it.

    Reads the response's own ``content-type`` and appends to its
    ``raw_headers``, so the policy is a separate header beside anything the
    upstream sent. Returns whether the sandbox was applied.
    """
    raw_headers: list[tuple[bytes, bytes]] = response.raw_headers  # type: ignore[attr-defined]
    types = [
        value.decode("latin-1") for name, value in raw_headers if name.lower() == b"content-type"
    ]
    # Two Content-Type headers are one comma-joined list to a browser: a document.
    content_type = types[0] if len(types) == 1 else None
    if not needs_panel_sandbox(path, content_type, entry_pages):
        return False
    raw_headers.extend(sandbox_header_pairs(raw_headers))
    return True
