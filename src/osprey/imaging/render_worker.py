"""The isolated picture render worker.

Run as ``python -I -m osprey.imaging.render_worker`` with an empty environment
and ``cwd='/'``. The interpreter imports the ``osprey`` package first; this
module's own imports are only Pillow and :mod:`osprey.imaging.formats`. When run
as a program, :func:`harden` is the first thing that executes: it limits the
process (no core files, a lifetime CPU budget, on Linux an address-space cap and
death with the parent) and turns Pillow's decompression-bomb warning into an
error, before one handshake line is written and any input is read.

Wire protocol (no pickle):

* handshake -- one JSON line ``{"ready": true, "pillow": "<version>"}``;
* request -- a 4-byte big-endian length, then that many picture bytes;
* reply -- one JSON header line ``{"ok", "format", "w", "h", "mode", "mime",
  "reason"}``, then a 4-byte big-endian length, then the rendition bytes
  (empty when ``ok`` is false).

The worker exits on end of input. Every request is answered by :func:`handle`,
a pure function from request payload to reply frame; a refused picture is a
reply with ``ok: false`` and a reason from
:data:`~osprey.imaging.formats.CONTENT_SKIP_REASONS`.
"""

from __future__ import annotations

import io
import json
import math
import resource
import signal
import struct
import sys
import warnings
from collections.abc import Callable
from typing import IO, Any, cast

import PIL
from PIL import Image, ImageOps

from osprey.imaging.formats import (
    ACCEPTED,
    RENDITION_MAX_BYTES,
    RENDITION_MAX_SIDE,
    RENDITION_MODES,
    admitted_formats,
    sniff,
)

PIXEL_LIMIT = 40_000_000
"""Largest picture, in pixels, the worker decodes (after a JPEG draft)."""

PROGRESSIVE_MAX_COEF_BYTES = 360_000_000
"""Largest coefficient buffer a multi-scan JPEG may need.

libjpeg decodes a progressive JPEG, and a sequential one whose first scan
carries fewer components than the frame (non-interleaved scans), through
whole-image coefficient buffers whose size follows the source dimensions,
component count and chroma sampling whatever the draft scale, so such a
picture is bounded from the header before any decode.
"""

DRAFT_SIZE = (2048, 2048)
"""Target handed to the JPEG draft (a power-of-two DCT downscale)."""

RETRY_SIZE = 768
"""Longest side of the one smaller re-encode of an oversized rendition."""

JPEG_QUALITY = 85

PNG_TO_JPEG_BYTES = int(1.5 * 1024 * 1024)
"""A PNG rendition above this size with no alpha is re-encoded as JPEG."""

MAX_REQUEST_BYTES = 512 * 1024 * 1024
"""Largest request payload the frame reader accepts; a longer frame ends the worker."""

ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
"""Linux address-space cap of the worker process."""

CPU_SECONDS = 100 * 30 + 60
"""Lifetime CPU budget: the 100-task respawn times the 30 s per-task clock, plus margin."""

PR_SET_PDEATHSIG = 1

_ALPHA_MODES = frozenset({"RGBA", "LA"})
_JPEG_FORMATS = frozenset({"JPEG", "MPO"})
_LENGTH = struct.Struct(">I")

# Start-of-frame markers that carry a frame header (DHT 0xC4, JPG 0xC8 and
# DAC 0xCC share the range but are not frames), and the progressive ones.
_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_PROGRESSIVE_SOF_MARKERS = frozenset({0xC2, 0xC6, 0xCA, 0xCE})
_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD9)})
_SOS = 0xDA
_EOI = 0xD9

# The header reports the Pillow opener that admitted the picture: MPO has no
# opener of its own and is reached through the JPEG opener.
_REPORTED_FORMAT = {"MPO": "JPEG"}


class _Refused(Exception):
    """A picture the bytes themselves refuse; carries the content skip reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# -- process hardening ------------------------------------------------------------


def _set_pdeathsig() -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)


def _capped(limit: int, current_hard: int) -> tuple[int, int]:
    """``(limit, limit)``, lowered to the current hard limit when that is smaller."""
    if current_hard != resource.RLIM_INFINITY and current_hard < limit:
        limit = current_hard
    return (limit, limit)


def harden(
    *,
    platform: str = sys.platform,
    setrlimit: Callable[[int, tuple[int, int]], None] = resource.setrlimit,
    getrlimit: Callable[[int], tuple[int, int]] = resource.getrlimit,
    set_pdeathsig: Callable[[], None] = _set_pdeathsig,
) -> None:
    """Limit this process before any input is read.

    Sets ``RLIMIT_CORE = (0, 0)``; on Linux ``RLIMIT_AS`` to
    :data:`ADDRESS_SPACE_BYTES` and ``PR_SET_PDEATHSIG = SIGKILL``; a lifetime
    ``RLIMIT_CPU`` of :data:`CPU_SECONDS`; the decompression-bomb warning as an
    error; and ``Image.MAX_IMAGE_PIXELS`` to :data:`PIXEL_LIMIT`. The limits are
    irreversible, so this runs only in the worker process.

    Args:
        platform: The platform name (``sys.platform``).
        setrlimit: ``resource.setrlimit``.
        getrlimit: ``resource.getrlimit``.
        set_pdeathsig: Arms parent-death delivery of SIGKILL (Linux).
    """
    setrlimit(resource.RLIMIT_CORE, (0, 0))
    if platform.startswith("linux"):
        setrlimit(
            resource.RLIMIT_AS,
            _capped(ADDRESS_SPACE_BYTES, getrlimit(resource.RLIMIT_AS)[1]),
        )
        set_pdeathsig()
    setrlimit(resource.RLIMIT_CPU, _capped(CPU_SECONDS, getrlimit(resource.RLIMIT_CPU)[1]))
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    Image.MAX_IMAGE_PIXELS = PIXEL_LIMIT


def handshake() -> bytes:
    """The one line the worker writes before reading input."""
    return json.dumps({"ready": True, "pillow": PIL.__version__}).encode() + b"\n"


# -- framing ----------------------------------------------------------------------


def _read_exactly(stream: IO[bytes], size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: IO[bytes]) -> bytes | None:
    """Read one length-prefixed request payload.

    Returns:
        The payload, or ``None`` at end of input, on a truncated frame, or when
        the announced length exceeds :data:`MAX_REQUEST_BYTES` -- each of which
        ends the worker.
    """
    prefix = _read_exactly(stream, _LENGTH.size)
    if prefix is None:
        return None
    (length,) = _LENGTH.unpack(prefix)
    if length > MAX_REQUEST_BYTES:
        return None
    return _read_exactly(stream, length)


def _reply(header: dict[str, Any], body: bytes = b"") -> bytes:
    return json.dumps(header).encode() + b"\n" + _LENGTH.pack(len(body)) + body


def _refusal(reason: str) -> bytes:
    return _reply(
        {
            "ok": False,
            "format": None,
            "w": None,
            "h": None,
            "mode": None,
            "mime": None,
            "reason": reason,
        }
    )


# -- rendering --------------------------------------------------------------------


def _open(data: bytes, *, jpeg: bool) -> Image.Image:
    """``Image.open`` restricted to :data:`ACCEPTED`.

    A JPEG opens with no pixel limit (its size is bounded after the draft);
    the limit is restored right after.
    """
    if not jpeg:
        return Image.open(io.BytesIO(data), formats=ACCEPTED)
    Image.MAX_IMAGE_PIXELS = None
    try:
        return Image.open(io.BytesIO(data), formats=ACCEPTED)
    finally:
        Image.MAX_IMAGE_PIXELS = PIXEL_LIMIT


def progressive_coefficient_bytes(
    size: tuple[int, int], layers: list[tuple[Any, int, int, Any]]
) -> int:
    """Bytes of coefficient buffers a multi-scan JPEG decode allocates.

    ``Σ_c ceil(ceil(W·h_c/hmax)/8) · ceil(ceil(H·v_c/vmax)/8) · 128`` over the
    components ``(id, h, v, q)`` of the frame header.
    """
    width, height = size
    hmax = max(layer[1] for layer in layers)
    vmax = max(layer[2] for layer in layers)
    total = 0
    for _, h, v, _ in layers:
        blocks_w = math.ceil(math.ceil(width * h / hmax) / 8)
        blocks_h = math.ceil(math.ceil(height * v / vmax) / 8)
        total += blocks_w * blocks_h * 128
    return total


def _is_progressive(im: Image.Image) -> bool:
    return "progressive" in im.info or "progression" in im.info


def _next_marker(data: bytes, pos: int) -> tuple[int, int] | None:
    """The next marker at or after ``pos`` as ``(code, offset of its 0xFF)``.

    Skips bytes the way libjpeg's ``next_marker`` does: anything up to an
    0xFF, then 0xFF fill bytes, and a stuffed ``FF 00`` pair.
    """
    end = len(data)
    while True:
        pos = data.find(b"\xff", pos)
        if pos < 0:
            return None
        start = pos
        while pos < end and data[pos] == 0xFF:
            pos += 1
        if pos >= end:
            return None
        if data[pos] != 0x00:
            return data[pos], start
        pos += 1


def jpeg_has_multiple_scans(data: bytes) -> bool:
    """Whether libjpeg may decode this JPEG through whole-image coefficient buffers.

    Walks the marker stream up to the first start-of-scan: true for a
    progressive frame, or when that scan carries fewer components than the
    frame (libjpeg's ``has_multiple_scans``). The walk fails closed: a stream
    that does not reach a first scan with a known frame component count is
    reported as multi-scan, so the coefficient bound applies to it.
    """
    pos = 2
    progressive = False
    frame_components: int | None = None
    end = len(data)
    while True:
        found = _next_marker(data, pos)
        if found is None:
            return True
        marker, start = found
        pos = start
        while data[pos] == 0xFF:
            pos += 1
        pos += 1  # past the marker code
        if marker == _EOI:  # end of image before any scan
            return True
        if marker in _STANDALONE_MARKERS:
            continue
        if pos + 2 > end:
            return True
        (length,) = struct.unpack(">H", data[pos : pos + 2])
        if marker in _SOF_MARKERS:
            if pos + 8 > end:
                return True
            progressive = marker in _PROGRESSIVE_SOF_MARKERS
            frame_components = data[pos + 7]
        elif marker == _SOS:
            if pos + 3 > end or frame_components is None:
                return True
            return progressive or data[pos + 2] < frame_components
        pos += length


def _normalise_mode(im: Image.Image) -> Image.Image:
    """Bring any decoded mode to one of :data:`RENDITION_MODES`."""
    if im.mode.startswith("I;16"):
        im = im.convert("I")
    if im.mode in ("I", "F"):
        lo, hi = cast("tuple[float, float]", im.getextrema())
        span = max(hi - lo, 1)
        return im.point(lambda v: (v - lo) * 255.0 / span).convert("L")
    if im.mode in RENDITION_MODES:
        return im
    if im.has_transparency_data or "transparency" in im.info:
        return im.convert("RGBA")
    return im.convert("RGB")


def _encode(im: Image.Image, *, jpeg_source: bool) -> tuple[bytes, str]:
    """Encode a rendition: JPEG for a JPEG source in RGB or L, else PNG.

    A PNG over :data:`PNG_TO_JPEG_BYTES` with no alpha is re-encoded as JPEG.
    """
    if jpeg_source and im.mode in ("RGB", "L"):
        return _save(im, "JPEG"), "image/jpeg"
    png = _save(im, "PNG")
    if len(png) > PNG_TO_JPEG_BYTES and im.mode not in _ALPHA_MODES:
        return _save(im, "JPEG"), "image/jpeg"
    return png, "image/png"


def _save(im: Image.Image, fmt: str) -> bytes:
    out = io.BytesIO()
    if fmt == "JPEG":
        im.save(out, "JPEG", quality=JPEG_QUALITY)
    else:
        im.save(out, "PNG")
    return out.getvalue()


def _render(data: bytes) -> bytes:
    sniffed = sniff(data)
    if not sniffed.is_image:
        raise _Refused("not_an_image")
    admitted = admitted_formats(sniffed.mime)

    try:
        im = _open(data, jpeg=sniffed.mime == "image/jpeg")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise _Refused("rendition_too_large") from exc
    source_format = im.format or ""
    if source_format not in admitted:
        raise _Refused("format_mismatch")
    jpeg_source = source_format in _JPEG_FORMATS

    if jpeg_source:
        multi_scan = _is_progressive(im) or jpeg_has_multiple_scans(data)
        coefficients = progressive_coefficient_bytes(im.size, cast(Any, im).layer)
        if multi_scan and coefficients > PROGRESSIVE_MAX_COEF_BYTES:
            raise _Refused("rendition_too_large")
        im.draft("RGB", DRAFT_SIZE)
    if im.width * im.height > PIXEL_LIMIT:
        raise _Refused("rendition_too_large")

    im = _normalise_mode(im)
    im.thumbnail(
        (RENDITION_MAX_SIDE, RENDITION_MAX_SIDE), Image.Resampling.LANCZOS, reducing_gap=2.0
    )
    ImageOps.exif_transpose(im, in_place=True)

    body, mime = _encode(im, jpeg_source=jpeg_source)
    if len(body) > RENDITION_MAX_BYTES:
        im.thumbnail((RETRY_SIZE, RETRY_SIZE), Image.Resampling.LANCZOS, reducing_gap=2.0)
        body, mime = _encode(im, jpeg_source=jpeg_source)
        if len(body) > RENDITION_MAX_BYTES:
            raise _Refused("rendition_too_large")

    return _reply(
        {
            "ok": True,
            "format": _REPORTED_FORMAT.get(source_format, source_format),
            "w": im.width,
            "h": im.height,
            "mode": im.mode,
            "mime": mime,
            "reason": None,
        },
        body,
    )


def handle(frame: bytes) -> bytes:
    """Answer one request: validate, decode, encode, and build the reply frame.

    Pure apart from Pillow's process-global pixel limit and the bomb-warning
    filter, both of which it enforces for the duration of the call and restores
    afterwards.

    Args:
        frame: The request payload -- the picture bytes.

    Returns:
        The reply frame: JSON header line, 4-byte length, rendition bytes.
    """
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = PIXEL_LIMIT
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            return _render(frame)
    except _Refused as refused:
        return _refusal(refused.reason)
    except (MemoryError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        # A size failure repeats on every retry: it is a deterministic refusal.
        return _refusal("rendition_too_large")
    except Exception:
        return _refusal("decoder_failed")
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


# The frame loop runs only in the env-less -I child, where coverage never starts.
def main() -> None:  # pragma: no cover
    """Harden, hand-shake, then answer frames until end of input."""
    harden()
    out = sys.stdout.buffer
    out.write(handshake())
    out.flush()
    stdin = sys.stdin.buffer
    while (frame := read_frame(stdin)) is not None:
        out.write(handle(frame))
        out.flush()


if __name__ == "__main__":  # pragma: no cover - entry point of the -I child
    main()
