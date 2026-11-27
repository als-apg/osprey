"""How the web terminal answers a panel's static files: validated, compressed, read once.

A panel's page is made of files the browser asks for on every open. Three
kinds reach it through ``/panel/<id>/``:

* **The hub's own files** (the design system's modules, the shared fonts, a
  first-party panel's module tree). :func:`serve_hub_file` reads a file once,
  keeps its bytes, their digest and their gzipped copy in memory, and reads it
  again only when its stat changes. It is answered ``no-cache`` with a strong
  ``ETag``: the browser stores it and each later use costs one ``304``.
* **A panel backend's static files** (scripts, style sheets, fonts, images).
  :func:`shape_relayed` gives each a validator of the bytes the browser
  receives, answers ``304`` to a matching ``If-None-Match``, and states
  ``no-cache`` where the backend stated nothing.
* **Everything else a backend answers** (pages, API answers). Its caching is
  left as the caller decided it; :func:`shape_relayed` only compresses it.

Text over :data:`MIN_COMPRESS_BYTES` leaves gzipped when the browser accepts
that. Compression runs in a worker thread, where zlib releases the
interpreter, so a large answer does not hold the event loop.

A validator this module computes is a *hub tag*, spelled ``"h-<digest>"``
(:func:`is_hub_tag`). The proxy keeps a request whose ``If-None-Match`` names
one from reaching the backend as a conditional request: the backend never
issued that tag, and the answer is compared here.

A backend is not trusted to say what the hub computed. An ``ETag`` of a
backend that is spelled like a hub tag is dropped before anything else reads
it, whether a tag is the hub's is recorded where the tag is made
(``Kept.ours``) and never read back from its spelling, and the gzipped copies
of relayed bodies are kept by a digest this module takes of the body it is
about to send, never by a string a backend or a browser supplied.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import Response

from osprey.interfaces._assets import CACHE_REVALIDATE, etag_matches

__all__ = [
    "MIN_COMPRESS_BYTES",
    "Kept",
    "is_hub_tag",
    "is_static_type",
    "serve_hub_file",
    "shape_relayed",
]

#: Below this size the gzip framing costs more than it saves.
MIN_COMPRESS_BYTES = 1024

#: A compressed copy is used only when it is at most this fraction of the body.
_MIN_SAVING = 0.9

#: The largest relayed body this module digests for a validator.
_MAX_DIGEST_BYTES = 16 * 1024 * 1024

#: How many bytes of bodies and compressed copies are kept, in all.
_KEPT_BYTES = 64 * 1024 * 1024

#: How a hub tag starts, quote included.
_HUB_TAG_PREFIX = '"h-'

#: Content types sent compressed, besides every ``text/`` type but the stream.
_COMPRESSIBLE_TYPES = frozenset(
    {
        "application/javascript",
        "application/json",
        "application/manifest+json",
        "application/wasm",
        "application/xhtml+xml",
        "application/xml",
        "font/ttf",
        "image/svg+xml",
    }
)

#: Content types of a page's static files, besides every font and image type.
_STATIC_TYPES = frozenset(
    {"application/javascript", "application/wasm", "text/css", "text/javascript"}
)


def is_hub_tag(if_none_match: str | None) -> bool:
    """Whether an ``If-None-Match`` header names a validator this module computed."""
    return any(_claims_hub(tag) for tag in (if_none_match or "").split(","))


def is_static_type(base_type: str) -> bool:
    """Whether *base_type* is the content type of a script, style sheet, font or image."""
    return base_type in _STATIC_TYPES or base_type.startswith(("font/", "image/"))


def _is_compressible(base_type: str) -> bool:
    if base_type == "text/event-stream":
        return False
    return base_type.startswith("text/") or base_type in _COMPRESSIBLE_TYPES


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _tag(body: bytes, digest: str | None = None) -> str:
    return f'{_HUB_TAG_PREFIX}{(digest or _digest(body))[:32]}"'


def _claims_hub(tag: str) -> bool:
    """Whether *tag*, strong or weak, is spelled as only this module's tags are."""
    spelled = tag.strip()
    if spelled[:2].upper() == "W/":
        spelled = spelled[2:].lstrip()
    return spelled.startswith(_HUB_TAG_PREFIX)


def _compressed_tag(tag: str, ours: bool) -> str:
    """The validator of the gzipped form of what *tag* validates.

    Args:
        tag: The validator of the bytes before compression.
        ours: Whether this module computed *tag*; stated by the caller that
            made the tag, not read from how the tag is spelled.
    """
    if ours:
        return f'{tag[:-1]}-gz"'
    # A backend's own tag names the bytes it sent, not their gzipped form.
    return tag if tag.startswith("W/") else f"W/{tag}"


def _pack(body: bytes) -> bytes | None:
    """*body* gzipped, or ``None`` when that saves too little."""
    # mtime=0 keeps the bytes a pure function of the body.
    packed = gzip.compress(body, compresslevel=6, mtime=0)
    return packed if len(packed) <= len(body) * _MIN_SAVING else None


def _accepts_gzip(request_headers: Mapping[str, str]) -> bool:
    accepted = {
        coding.split(";", 1)[0].strip().lower()
        for coding in request_headers.get("accept-encoding", "").split(",")
    }
    return "gzip" in accepted


@dataclass(frozen=True, slots=True)
class Kept:
    """One body held in memory.

    Attributes:
        signature: The ``(mtime_ns, size)`` of the file the body was read
            from; ``None`` for a body that came from no file.
        body: The bytes to send.
        tag: Their validator.
        packed: Their gzipped form; ``None`` when it is not worth sending, or
            not yet made (``pack_tried`` says which).
        pack_tried: Whether ``packed`` was decided.
        ours: Whether this module computed ``tag`` from ``body``. A tag a
            backend issued is relayed, never treated as the hub's.
    """

    signature: tuple[int, int] | None
    body: bytes
    tag: str
    packed: bytes | None = None
    pack_tried: bool = False
    ours: bool = False

    @property
    def size(self) -> int:
        """The bytes this entry holds."""
        return len(self.body) + len(self.packed or b"")


class _Keep:
    """Bodies by key, the least recently used dropped beyond :data:`_KEPT_BYTES`."""

    def __init__(self, limit: int = _KEPT_BYTES) -> None:
        self._limit = limit
        self._held: OrderedDict[Hashable, Kept] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Kept | None:
        with self._lock:
            kept = self._held.get(key)
            if kept is not None:
                self._held.move_to_end(key)
            return kept

    def put(self, key: Hashable, kept: Kept) -> None:
        with self._lock:
            old = self._held.pop(key, None)
            if old is not None:
                self._bytes -= old.size
            if kept.size > self._limit:
                return
            self._held[key] = kept
            self._bytes += kept.size
            while self._bytes > self._limit:
                _, dropped = self._held.popitem(last=False)
                self._bytes -= dropped.size

    def clear(self) -> None:
        with self._lock:
            self._held.clear()
            self._bytes = 0


#: The hub's own files, by ``(path, variant)``.
_FILES = _Keep()
#: The gzipped form of relayed bodies, by ``(digest, length)`` of the body
#: itself, both taken here. Gzip is a pure function of the body, so one entry
#: can only ever answer for the bytes it was made from, whatever backend,
#: content type or validator they arrived with.
_PACKED = _Keep()


def _answer(
    request_headers: Mapping[str, str],
    method: str,
    kept: Kept,
    *,
    media_type: str | None,
    headers: dict[str, str],
    status_code: int = 200,
    conditional: bool = True,
) -> Response:
    """*kept* as a response: ``304`` to a matching validator, gzipped where accepted."""
    compress = kept.packed is not None and _accepts_gzip(request_headers)
    if kept.tag:
        headers["etag"] = _compressed_tag(kept.tag, kept.ours) if compress else kept.tag
    if kept.pack_tried and kept.packed is not None:
        varies = [part.strip() for part in headers.get("vary", "").split(",") if part.strip()]
        if "accept-encoding" not in {part.lower() for part in varies} and "*" not in varies:
            headers["vary"] = ", ".join((*varies, "Accept-Encoding"))
    if (
        conditional
        and kept.tag
        and etag_matches(
            request_headers.get("if-none-match"),
            kept.tag.removeprefix("W/"),
            _compressed_tag(kept.tag, kept.ours).removeprefix("W/"),
        )
    ):
        return Response(status_code=304, headers=headers)
    if compress and kept.packed is not None:
        headers["content-encoding"] = "gzip"
        body = kept.packed
    else:
        body = kept.body
    if method == "HEAD":
        headers["content-length"] = str(len(body))
        body = b""
    return Response(content=body, status_code=status_code, headers=headers, media_type=media_type)


def _read_file(
    path: Path, signature: tuple[int, int], transform: Callable[[bytes], bytes] | None, pack: bool
) -> Kept:
    body = path.read_bytes()
    if transform is not None:
        body = transform(body)
    packed = _pack(body) if pack and len(body) >= MIN_COMPRESS_BYTES else None
    return Kept(
        signature=signature, body=body, tag=_tag(body), packed=packed, pack_tried=True, ours=True
    )


async def serve_hub_file(
    request_headers: Headers,
    method: str,
    path: Path,
    *,
    media_type: str,
    headers: Mapping[str, str],
    variant: Hashable = None,
    transform: Callable[[bytes], bytes] | None = None,
    cache_control: str = CACHE_REVALIDATE,
) -> Response:
    """Answer with one of the hub's own files, read once per content.

    Args:
        request_headers: The request's headers.
        method: The request's method, ``GET`` or ``HEAD``.
        path: The file, already resolved and contained by the caller.
        media_type: The content type to send.
        headers: Headers every answer carries besides the caching ones.
        variant: What *transform* depends on besides the file; bodies of one
            file under two variants are kept apart.
        transform: Turns the file's bytes into the body; none sends the file.
        cache_control: The ``Cache-Control`` of the answer.

    Returns:
        The body, its gzipped form, or ``304``; ``404`` when the file is gone.
    """
    try:
        stat_result = path.stat()
    except OSError:
        return Response(content="Not found", status_code=404)
    signature = (stat_result.st_mtime_ns, stat_result.st_size)
    key = (str(path), variant)
    kept = _FILES.get(key)
    if kept is None or kept.signature != signature:
        pack = _is_compressible(media_type.split(";", 1)[0].strip().lower())
        try:
            kept = await asyncio.to_thread(_read_file, path, signature, transform, pack)
        except OSError:
            return Response(content="Not found", status_code=404)
        _FILES.put(key, kept)
    return _answer(
        request_headers,
        method,
        kept,
        media_type=media_type,
        headers={**headers, "cache-control": cache_control},
    )


async def shape_relayed(
    request_headers: Headers,
    method: str,
    *,
    body: bytes,
    status_code: int,
    headers: dict[str, str],
    media_type: str | None,
    rewritten: bool,
    default_cache_control: str,
) -> Response:
    """Turn a backend's buffered answer into the response the browser gets.

    A ``GET`` answered ``200`` with a static content type (:func:`is_static_type`)
    is validated: a body the proxy rewrote, or one the backend gave no
    ``ETag``, gets a hub tag of the bytes sent, and a matching
    ``If-None-Match`` is answered ``304``. Where the backend stated no
    ``Cache-Control`` such an answer is ``no-cache``, as is a backend's own
    ``304``; every other answer gets *default_cache_control*. A ``200`` body of a compressible type is sent
    gzipped to a browser that accepts it.

    Args:
        request_headers: The browser's request headers.
        method: The browser's method.
        body: The body to send, after any rewrite.
        status_code: The backend's status.
        headers: The backend's headers the proxy relays; changed in place.
        media_type: The backend's ``Content-Type``, parameters included.
        rewritten: Whether the proxy changed the body.
        default_cache_control: The ``Cache-Control`` of an answer that is not
            a validated static file and whose backend stated none.
    """
    lowered = {name.lower(): name for name in headers}

    def drop(name: str) -> None:
        if name in lowered:
            del headers[lowered.pop(name)]

    base_type = (media_type or "").split(";", 1)[0].strip().lower()
    plain = status_code == 200 and "content-range" not in lowered
    static = plain and method == "GET" and is_static_type(base_type)

    if "etag" in lowered and _claims_hub(headers[lowered["etag"]]):
        # Only this module issues hub tags. A backend that sends one is not
        # believed: the tag is gone before anything below can read it.
        drop("etag")

    tag = ""
    ours = False
    digest: str | None = None
    if static and len(body) <= _MAX_DIGEST_BYTES:
        upstream_tag = headers[lowered["etag"]] if "etag" in lowered else ""
        if rewritten or not upstream_tag:
            digest = _digest(body)
            tag = _tag(body, digest)
            ours = True
            drop("etag")
            if rewritten:
                # The body follows the rewrite as well as the backend's file.
                drop("last-modified")
        else:
            tag = upstream_tag
            drop("etag")
    elif rewritten:
        # The backend's validators name the body before the rewrite.
        drop("etag")
        drop("last-modified")
    if "cache-control" not in lowered:
        # A 304 answers a browser that stored the file, which it did under a
        # validator; the default must not tell it to stop storing.
        validated = bool(tag) or status_code == 304
        headers["cache-control"] = CACHE_REVALIDATE if validated else default_cache_control

    packed: bytes | None = None
    pack_tried = False
    if (
        plain
        and method != "HEAD"
        and len(body) >= MIN_COMPRESS_BYTES
        and _is_compressible(base_type)
        and _accepts_gzip(request_headers)
    ):
        pack_tried = True
        # Kept only for a body digested here, under that digest.
        key = (digest, len(body)) if ours and digest is not None else None
        held = _PACKED.get(key) if key is not None else None
        if held is not None:
            packed = held.packed
        else:
            packed = await asyncio.to_thread(_pack, body)
            if key is not None and packed is not None:
                _PACKED.put(key, Kept(None, b"", "", packed, True))
    kept = Kept(None, body, tag, packed, pack_tried, ours)
    if not tag and "etag" in lowered and packed is not None:
        # A backend's tag on an answer this module does not validate still
        # names the bytes before compression.
        name = lowered["etag"]
        headers[name] = _compressed_tag(headers[name], ours=False)
    return _answer(
        request_headers,
        method,
        kept,
        media_type=media_type,
        headers=headers,
        status_code=status_code,
        conditional=bool(tag),
    )
