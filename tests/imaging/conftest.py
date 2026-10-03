"""Picture fixtures for the render worker, generated at test time (never committed).

Two kinds of oversized JPEG are built here:

* refusal cases -- a tiny real JPEG whose SOF0/SOF2 frame header is patched to
  claim huge dimensions. The worker must refuse them from the header alone; if
  it ever decoded one, the missing scan data would surface as
  ``decoder_failed`` instead of ``rendition_too_large``.
* decode cases -- large JPEGs that really decode. Writing them through Pillow
  would allocate the full raster, so they are assembled by hand: a one-entry
  DC Huffman table and a one-entry AC table (EOB only) make every block of a
  flat mid-grey picture two zero bits, so the scan data is a run of zero bytes
  whatever the size. They are built once per session.
"""

from __future__ import annotations

import io
import math
import struct
import zlib
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

_SOF_MARKERS = {0xC0, 0xC1, 0xC2}
_STANDALONE_MARKERS = {0x01, *range(0xD0, 0xD8)}


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def _scan_data(blocks: int, bits_per_block: int) -> bytes:
    """All-zero entropy-coded data for ``blocks`` blocks, padded with 1-bits."""
    bits = blocks * bits_per_block
    data = b"\x00" * (bits // 8)
    if bits % 8:
        data += bytes((0xFF >> (bits % 8),))
    return data


def flat_jpeg(
    width: int, height: int, *, progressive: bool = False, interleaved: bool = True
) -> bytes:
    """A flat mid-grey YCbCr JPEG of any size, with all-zero scan data.

    Every block carries a zero DC difference and an immediate end-of-block, so
    the decoded picture is uniform grey (128, 128, 128). The default is 4:2:0
    with one interleaved scan; the progressive variant has a single interleaved
    DC scan, its AC coefficients staying zero. ``interleaved=False`` gives a
    baseline 4:4:4 picture coded as three single-component scans (libjpeg's
    multi-scan sequential case).
    """
    dqt = _segment(0xDB, b"\x00" + b"\x01" * 64)
    sampling = 0x22 if interleaved else 0x11
    components = ((1, sampling), (2, 0x11), (3, 0x11))
    sof = _segment(
        0xC2 if progressive else 0xC0,
        struct.pack(">BHHB", 8, height, width, len(components))
        + b"".join(struct.pack(">BBB", cid, s, 0) for cid, s in components),
    )
    one_symbol = b"\x01" + b"\x00" * 15 + b"\x00"  # one 1-bit code for symbol 0
    dht = _segment(0xC4, b"\x00" + one_symbol + (b"" if progressive else b"\x10" + one_symbol))
    se = 0 if progressive else 63
    bits_per_block = 1 if progressive else 2

    def sos(ids) -> bytes:
        return _segment(
            0xDA,
            bytes((len(ids),)) + b"".join(bytes((cid, 0x00)) for cid in ids) + bytes((0, se, 0)),
        )

    if interleaved:
        mcus = math.ceil(width / 16) * math.ceil(height / 16)
        scans = sos([cid for cid, _ in components]) + _scan_data(mcus * 6, bits_per_block)
    else:
        blocks = math.ceil(width / 8) * math.ceil(height / 8)
        scans = b"".join(sos([cid]) + _scan_data(blocks, bits_per_block) for cid, _ in components)
    return b"\xff\xd8" + dqt + sof + dht + scans + b"\xff\xd9"


def patch_jpeg_size(data: bytes, width: int, height: int) -> bytes:
    """Rewrite the frame header of a JPEG to claim ``width`` x ``height``."""
    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ValueError("not at a marker")
        marker = data[pos + 1]
        if marker in _STANDALONE_MARKERS:
            pos += 2
            continue
        (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        if marker in _SOF_MARKERS:
            head = pos + 5  # marker, length, precision
            return data[:head] + struct.pack(">HH", height, width) + data[head + 4 :]
        pos += 2 + length
    raise ValueError("no frame header")


def patch_png_size(data: bytes, width: int, height: int) -> bytes:
    """Rewrite the IHDR of a PNG to claim ``width`` x ``height`` (CRC fixed)."""
    ihdr = data[12:29]  # type + 13-byte body
    body = struct.pack(">II", width, height) + ihdr[12:]
    chunk = b"IHDR" + body
    return data[:12] + chunk + struct.pack(">I", zlib.crc32(chunk)) + data[33:]


def encode(im, fmt: str, **params) -> bytes:
    out = io.BytesIO()
    im.save(out, fmt, **params)
    return out.getvalue()


def tiny_jpeg(*, mode: str = "RGB", progressive: bool = False, subsampling: int = 2) -> bytes:
    colour = (10, 20, 30, 40)[: len(mode)] if mode != "L" else 90
    return encode(
        Image.new(mode, (16, 16), colour),
        "JPEG",
        progressive=progressive,
        subsampling=subsampling,
    )


@pytest.fixture(scope="session")
def big_jpegs(tmp_path_factory) -> dict[str, Path]:
    """The three large JPEGs that must really decode, written once per session."""
    root = tmp_path_factory.mktemp("render_big")
    specs = {
        "baseline_12000x9000": (12000, 9000, False),
        "baseline_16000x12000": (16000, 12000, False),
        "progressive_420_12000x9000": (12000, 9000, True),
    }
    paths = {}
    for name, (width, height, progressive) in specs.items():
        path = root / f"{name}.jpg"
        path.write_bytes(flat_jpeg(width, height, progressive=progressive))
        paths[name] = path
    return paths
