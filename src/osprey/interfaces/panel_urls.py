"""How a browser and a server read a URL bound for a panel.

Two readers judge URLs that end up in a panel: the web terminal (the
navigation guard on an agent's ``open`` and on a runtime registration, and the
panel sandbox's entry-page match) and the workspace MCP server's ``open_panel``
tool (its agent-hop refusal). Each must read a URL the way the browser or the
backend will, or a spelling the judge reads one way and the browser another
gets through. This module is that one reading, kept free of any web framework
so the MCP server imports it without loading the terminal.

* :func:`normalise_navigation_url` and :func:`navigation_url_refusal` are the
  navigation guard. A panel-focus URL ends up as an iframe ``src`` in the
  operator's own page, and a runtime registration URL is fetched by the proxy
  and embedded the same way. Browsers read both with the WHATWG URL parser,
  which is laxer than ``urlparse``: it drops leading and trailing C0 controls
  and spaces, deletes every tab and newline, and reads ``\\`` as ``/`` in an
  http(s) context. A check that judged the raw string would pass
  ``"\\tjavascript:..."`` or ``"/\\\\evil.example"`` while the browser loads a
  script URL or a cross-origin document. So the URL is normalised the way the
  browser will read it, judged in that form, and only that form travels on.
  The client's ``buildEmbedSrc`` (``panel-iframe-sync.js``) applies the same
  rule.
* :func:`decoded_readings` and :func:`resolved_path_segments` read a path the
  way a server may: percent-escapes unwrapped to a fixed point, at most
  :data:`MAX_DECODE_ROUNDS` times.
* :func:`member_navigation_url` is the guard's second step, per panel: an
  admitted URL must stay under the panel's own mount, or sit on the panel's own
  upstream origin, in which case it is rewritten into the mount so the browser
  only ever loads the proxied form. The client's ``isUrlWithinMount`` applies
  the mount rule again before an iframe moves.
"""

from __future__ import annotations

import re
from urllib.parse import quote, unquote, urlparse, urlsplit

#: How many percent-decoding rounds a path may need before it is stable. A path
#: still changing after that many is read as nothing (callers refuse it).
MAX_DECODE_ROUNDS = 8

_C0_CONTROL_OR_SPACE = "".join(chr(code) for code in range(0x21))
_TAB_OR_NEWLINE = {ord("\t"): None, ord("\n"): None, ord("\r"): None}
_URL_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_NAVIGABLE_SCHEMES = ("http", "https")


def normalise_navigation_url(raw: str) -> str:
    """Read *raw* the way the browser's WHATWG URL parser will.

    Strips leading and trailing C0 controls and spaces, removes every tab and
    newline, reads every ``\\`` as ``/`` and lower-cases the scheme, so that
    ``"  HTTPS://host/"`` and ``"\\tjava\\nscript:x"`` reach the guard as
    ``"https://host/"`` and ``"javascript:x"``. Reading ``\\`` as ``/``
    everywhere can only make a URL look *more* like a host reference, never
    less, so the guard errs closed.

    Args:
        raw: The URL as the caller sent it.

    Returns:
        The normalised URL; identical to *raw* for an ordinary URL.
    """
    url = raw.strip(_C0_CONTROL_OR_SPACE).translate(_TAB_OR_NEWLINE).replace("\\", "/")
    match = _URL_SCHEME_RE.match(url)
    if match:
        url = match.group(1).lower() + url[match.end(1) :]
    return url


def navigation_url_refusal(url: str) -> str | None:
    """Why a normalised navigation URL may not be loaded into a panel, or ``None``.

    The guard admits exactly two shapes: a root-relative path (``/...``, but
    never the scheme-relative ``//...``, which names another host) and an
    ``http``/``https`` URL with a host, the scheme rule runtime panel
    registration applies too. Everything else (``javascript:``, ``data:``,
    ``vbscript:``, ``blob:``, a bare relative path, an empty string) is refused.

    Args:
        url: A URL already passed through :func:`normalise_navigation_url`.

    Returns:
        A human-readable refusal, or ``None`` when the URL may be navigated to.
    """
    if not url:
        return "URL is empty"
    if url.startswith("//"):
        return (
            "URL must not be scheme-relative ('//host'); use a root-relative path or an http(s) URL"
        )
    if url.startswith("/"):
        return None
    match = _URL_SCHEME_RE.match(url)
    if match is None:
        return "URL must be a root-relative path (starting with '/') or an http(s) URL"
    scheme = match.group(1)
    if scheme not in _NAVIGABLE_SCHEMES:
        return f"URL scheme must be http or https, got: {scheme!r}"
    if not url.startswith(f"{scheme}://"):
        return f"URL must be spelled {scheme}://<host>/..."
    try:
        host = urlparse(url).hostname
    except ValueError:
        return "Invalid URL"
    if not host:
        return "URL must include a host"
    return None


def decoded_readings(path: str) -> tuple[str, ...] | None:
    """*path* and each percent-decoding of it, up to the first stable one.

    Each round reads ``\\`` as ``/`` and unwraps one layer of percent-escapes;
    the readings end at the first one a further round would not change. The
    readings keep any ``\\`` a round decoded (``%5C``), so a caller that
    normalises each one sees every spelling a backend might act on.

    Returns:
        The readings in decoding order, the input first; ``None`` when the path
        is still changing after :data:`MAX_DECODE_ROUNDS` rounds.
    """
    readings: list[str] = []
    current = path
    for _round in range(MAX_DECODE_ROUNDS):
        readings.append(current)
        slashed = current.replace("\\", "/")
        decoded = unquote(slashed)
        if decoded == slashed:
            return tuple(readings)
        current = decoded
    return None


def resolved_path_segments(path: str) -> list[str] | None:
    """The segments *path* names once a server has finished reading it.

    Percent-escapes are unwrapped until the path stops changing
    (:func:`decoded_readings`), ``\\`` is read as ``/``, ``;params`` are
    dropped from every segment, empty segments (repeated slashes) are skipped
    and dot segments are resolved. Returns ``None`` for a path that is still
    changing after :data:`MAX_DECODE_ROUNDS` rounds of decoding.
    """
    readings = decoded_readings(path)
    if readings is None:
        return None
    segments: list[str] = []
    for raw in readings[-1].replace("\\", "/").split("/"):
        segment = raw.partition(";")[0]
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    return segments


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str, int] | None:
    """``(scheme, host, port)`` of an http(s) URL, with the scheme's default port filled in."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _NAVIGABLE_SCHEMES or not host:
        return None
    return scheme, host.lower(), port if port is not None else _DEFAULT_PORTS[scheme]


def _path_segments(path: str) -> list[str] | None:
    """The segments a root-relative *path* names, or ``None`` when it hides one.

    Refuses a path that is still changing after :data:`MAX_DECODE_ROUNDS`
    rounds, one whose decoded reading holds an empty segment (``//``, or a
    ``\\`` or its escape read as ``/``) or a dot segment (``.``, ``..`` and
    their escapes), and one with ``;params``: every one of those is a spelling
    a browser or a backend may read as a different place.
    """
    readings = decoded_readings(path)
    if readings is None:
        return None
    final = readings[-1].replace("\\", "/")
    if "//" in final:
        return None
    final = final.removesuffix("/")
    segments: list[str] = []
    for raw in final.split("/")[1:]:
        if raw in ("", ".", "..") or ";" in raw:
            return None
        segments.append(raw)
    return segments


def _mount_segments(mount: str) -> list[str]:
    return [segment for segment in mount.split("/") if segment]


def _canonical(segments: list[str], *, trailing_slash: bool, query: str, fragment: str) -> str:
    path = "/" + "/".join(quote(segment, safe="") for segment in segments)
    if trailing_slash and segments:
        path += "/"
    if query:
        path += "?" + query
    if fragment:
        path += "#" + fragment
    return path


def member_navigation_url(
    url: str, *, mount: str, upstream: str | None
) -> tuple[str | None, str | None]:
    """Judge a navigation URL for one panel and spell it in the panel's mount.

    *url* has passed :func:`normalise_navigation_url` and
    :func:`navigation_url_refusal`. *mount* is the panel's root-relative
    browser mount without any URL prefix (``/panel/ariel``,
    ``/panel-static/reference/``); *upstream* is the panel's configured
    backend URL, or ``None`` for a panel served from disk.

    Admitted are exactly two shapes:

    * a root-relative path whose segments, read as a server reads them
      (:func:`_path_segments`), equal the mount's or extend it;
    * an ``http(s)`` URL whose origin is the upstream's and whose path lies
      under the upstream's own base path. It is rewritten into the mount, so
      the browser receives the proxied document and never the upstream one.

    Returns:
        ``(admitted, None)`` with the canonical root-relative URL (segments
        re-quoted, query and fragment kept), or ``(None, refusal)``.
    """
    base = _mount_segments(mount)
    trailing = mount.endswith("/")
    if url.startswith("/"):
        parts = urlsplit(url)
        segments = _path_segments(parts.path)
        if segments is None or segments[: len(base)] != base:
            return None, f"a url for this panel is a path under {mount}"
        slash = parts.path.endswith("/") or (len(segments) == len(base) and trailing)
        return _canonical(
            segments, trailing_slash=slash, query=parts.query, fragment=parts.fragment
        ), None

    if upstream is None:
        return None, f"a url for this panel is a path under {mount}; it has no server of its own"
    target, own = _origin(url), _origin(upstream)
    if own is None or target != own:
        return None, (
            f"a url for this panel is a path under {mount} or an http(s) URL on its own server"
        )
    parts = urlsplit(url)
    segments = _path_segments(parts.path or "/")
    base_path = _path_segments(urlsplit(upstream).path or "/")
    if segments is None or base_path is None or segments[: len(base_path)] != base_path:
        return None, f"a url for this panel is a path under {mount} or under its server's own path"
    rest = segments[len(base_path) :]
    slash = (parts.path or "/").endswith("/") or (not rest and trailing)
    return _canonical(
        base + rest, trailing_slash=slash, query=parts.query, fragment=parts.fragment
    ), None
