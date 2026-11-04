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
* :func:`agent_hop_url_refusal` keeps a url away from a panel the agent drives
  through its own tool (CANOPY): the terminal runs it on every op and
  registration that takes a url, and the MCP tool runs it again for a friendlier
  error. :func:`canonical_host` reads a host as the browser does, so
  ``127.1``, ``0x7f.0.0.1`` and ``[::ffff:127.0.0.1]`` are the loopback they name.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Mapping
from typing import Any
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


# --- the agent-hop rule ------------------------------------------------------

#: A host as the URL parser reads it, and the origin built from it.
Origin = tuple[str, str, int]


def _ipv4_number(part: str) -> int | None:
    """One dotted part of an IPv4 host, as the WHATWG IPv4 number parser reads it."""
    if part[:2].lower() == "0x":
        digits, base = part[2:], 16
    elif len(part) > 1 and part.startswith("0"):
        digits, base = part[1:], 8
    else:
        digits, base = part, 10
    if not digits:
        return 0
    try:
        return int(digits, base) if digits.isalnum() else None
    except ValueError:
        return None


def _ipv4_host(host: str) -> str | None:
    """*host* as the dotted IPv4 address a browser reads it as, or ``None``.

    The browser reads a host whose last label is a number as an IPv4 address,
    in any of the legacy spellings: fewer than four parts (``127.1``), one
    number (``2130706433``), hex (``0x7f.0.0.1``) or octal (``0177.0.0.1``).
    ``None`` when the host is not such an address (a name, or a malformed
    number, which the browser refuses outright).
    """
    parts = host.split(".")
    if len(parts) > 1 and parts[-1] == "":
        parts.pop()
    last = parts[-1]
    if not (last.isdigit() or (last[:2].lower() == "0x" and _ipv4_number(last) is not None)):
        return None
    numbers = [_ipv4_number(part) for part in parts]
    if len(numbers) > 4 or any(number is None for number in numbers):
        return None
    head, tail = numbers[:-1], numbers[-1]
    if any(number > 255 for number in head) or tail >= 256 ** (5 - len(numbers)):  # type: ignore[operator]
        return None
    value = tail
    for index, number in enumerate(head):
        value += number * 256 ** (3 - index)  # type: ignore[operator]
    return str(ipaddress.IPv4Address(value))


def canonical_host(host: str) -> str:
    """*host* as a browser reaches it: case-folded, an IPv4 in dotted form, an IPv6 compressed.

    A name keeps its spelling but for case; ``localhost.`` and ``localhost``
    stay apart here, as they do in a browser's origin.
    """
    folded = host.strip().strip("[]").casefold()
    if ":" in folded:
        try:
            return str(ipaddress.IPv6Address(folded.partition("%")[0]))
        except ValueError:
            return folded
    return _ipv4_host(folded) or folded


def is_loopback_host(host: str) -> bool:
    """Whether *host*, in any spelling a browser or resolver reads, addresses this machine.

    ``localhost`` and every ``*.localhost`` (with or without the root's
    trailing dot), any 127/8 or ``::1`` address, the unspecified addresses, and
    an IPv4-mapped IPv6 address of any of those.
    """
    canonical = canonical_host(host)
    name = canonical.rstrip(".")
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(canonical)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback or address.is_unspecified


def canonical_origin(url: str) -> Origin | None:
    """``(scheme, host, port)`` of an http(s) URL as a browser reads it, or ``None``."""
    origin = _origin(url)
    if origin is None:
        return None
    scheme, host, port = origin
    return scheme, canonical_host(host), port


def same_place(a: Origin, b: Origin) -> bool:
    """Whether two origins reach the same listener: equal, or both loopback on one port."""
    if a == b:
        return True
    return a[0] == b[0] and a[2] == b[2] and is_loopback_host(a[1]) and is_loopback_host(b[1])


def agent_hop_panel_ids(panels: Mapping[str, Any]) -> frozenset[str]:
    """The sidecar panels whose own MCP transport the agent drives directly.

    A sidecar that declares an ``mcp_path`` and opts in with ``agent_hop``
    (:class:`~osprey.profiles.sidecar_panels.SidecarPanel`): the same rule the
    gate's panel tier and the panel proxy apply. Such a panel has a tool of its
    own for what is on its screen.
    """
    return frozenset(
        panel_id
        for panel_id, panel in panels.items()
        if getattr(panel, "agent_hop", False) and getattr(panel, "mcp_path", None)
    )


def agent_hop_url_refusal(
    member: str,
    url: str,
    *,
    hop_ids: frozenset[str],
    is_hop_origin: Callable[[Origin], bool],
) -> str | None:
    """Why *url* may not be pushed into panel *member*, or ``None``.

    A url pushed into an agent-hop panel's iframe would move its screen around
    the panel's own tool, its input checks and its attribution; one that
    reaches the panel's process directly would bypass the terminal's gate. So
    a url is refused when:

    * *member* is an agent-hop panel, whatever the url;
    * its path resolves under ``/panel/<agent-hop panel>`` (behind any
      per-user prefix, in any spelling a server reads the same way);
    * it is an http(s) URL whose origin, read as a browser reads it
      (:func:`canonical_origin`), *is_hop_origin* names. The terminal names
      the listener each agent-hop panel's configured backend is on
      (:func:`same_place`); the agent's MCP tool, which has no configuration,
      guesses from the port, for a friendlier error only.

    A url this rule passes is still judged by the navigation guard, which
    admits an http(s) URL only on the member's own configured backend.

    Args:
        member: The panel the url is for.
        url: The url, already passed through :func:`normalise_navigation_url`.
        hop_ids: The agent-hop panel ids (:func:`agent_hop_panel_ids`).
        is_hop_origin: Whether an origin may be an agent-hop panel's process.
    """
    if member in hop_ids:
        return f"panel '{member}' is driven by its own workspace tool, not by url"
    try:
        parts = urlsplit(url)
    except ValueError:
        return None  # malformed: the navigation guard words the refusal
    segments = resolved_path_segments(parts.path)
    if segments is None:
        return "the url path is encoded too many times to read"
    folded = [segment.casefold() for segment in segments]
    for index, segment in enumerate(folded[:-1]):
        if segment == "panel" and folded[index + 1] in hop_ids:
            return f"the url opens panel '{segments[index + 1]}', which has its own workspace tool"
    origin = canonical_origin(url)
    if origin is not None and is_hop_origin(origin):
        return "the url addresses a panel sidecar directly"
    return None
