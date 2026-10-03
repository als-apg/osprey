"""Tests for the isolated picture render worker.

Request handling is exercised two ways: in-process through ``handle`` (so the
decode and encode paths are measured), and through real worker processes
spawned exactly as production spawns them -- ``python -I -m
osprey.imaging.render_worker`` with an empty environment and ``cwd='/'``.
"""

from __future__ import annotations

import json
import os
import random
import resource
import signal
import struct
import subprocess
import sys
import warnings
from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest

from osprey.imaging import render_worker
from osprey.imaging.formats import ACCEPTED, CONTENT_SKIP_REASONS, sniff
from osprey.imaging.render_worker import (
    PIXEL_LIMIT,
    PROGRESSIVE_MAX_COEF_BYTES,
    handle,
    handshake,
    harden,
    jpeg_has_multiple_scans,
    progressive_coefficient_bytes,
    read_frame,
)

from .conftest import encode, flat_jpeg, patch_jpeg_size, patch_png_size, tiny_jpeg

Image = pytest.importorskip("PIL.Image")

LINUX = sys.platform.startswith("linux")
WORKER_ARGS = [sys.executable, "-I", "-m", "osprey.imaging.render_worker"]
HEADER_KEYS = {"ok", "format", "w", "h", "mode", "mime", "reason"}
# Pillow imports defusedxml (its XMP parser) when it is installed.
ALLOWED_THIRD_PARTY_ROOTS = {"PIL", "defusedxml"}
ALLOWED_OSPREY_MODULES = {
    "osprey",
    "osprey.version",
    "osprey._version",
    "osprey.imaging",
    "osprey.imaging.formats",
    "osprey.imaging.render_worker",
}


def parse_reply(reply: bytes) -> tuple[dict, bytes]:
    line, rest = reply.split(b"\n", 1)
    header = json.loads(line)
    (length,) = struct.unpack(">I", rest[:4])
    body = rest[4:]
    assert len(body) == length
    assert set(header) == HEADER_KEYS
    return header, body


def render(data: bytes) -> tuple[dict, bytes]:
    return parse_reply(handle(data))


def assert_rendition(header: dict, body: bytes, *, mime: str, mode: str | None = None) -> None:
    assert header["ok"] is True
    assert header["reason"] is None
    assert header["format"] in ACCEPTED
    assert header["mime"] == mime
    assert sniff(body).mime == mime
    assert 0 < header["w"] <= 1024 and 0 < header["h"] <= 1024
    assert header["mode"] in {"RGB", "L", "RGBA", "LA"}
    if mode is not None:
        assert header["mode"] == mode
    decoded = Image.open(BytesIO(body))
    assert decoded.size == (header["w"], header["h"])


def assert_refused(reply: tuple[dict, bytes], reason: str) -> None:
    header, body = reply
    assert reason in CONTENT_SKIP_REASONS
    assert header == {
        "ok": False,
        "format": None,
        "w": None,
        "h": None,
        "mode": None,
        "mime": None,
        "reason": reason,
    }
    assert body == b""


def frame(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


class Worker:
    """A worker process spawned the way production spawns it."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            WORKER_ARGS,
            env={},
            cwd="/",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.ready = json.loads(self.proc.stdout.readline())

    def render(self, data: bytes) -> tuple[dict, bytes]:
        self.proc.stdin.write(frame(data))
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, self.proc.stderr.read().decode()
        (length,) = struct.unpack(">I", self.proc.stdout.read(4))
        return parse_reply(line + struct.pack(">I", length) + self.proc.stdout.read(length))

    def close(self) -> int:
        self.proc.stdin.close()
        code = self.proc.wait(timeout=30)
        self.proc.stdout.close()
        self.proc.stderr.close()
        return code


@pytest.fixture
def worker():
    w = Worker()
    try:
        yield w
    finally:
        if w.proc.poll() is None:
            w.proc.kill()
            w.proc.wait()


def noise(mode: str, size: tuple[int, int], seed: int = 7) -> Image.Image:
    rng = random.Random(seed)
    bands = len(mode)
    return Image.frombytes(mode, size, rng.randbytes(size[0] * size[1] * bands))


# -- handshake and framing -------------------------------------------------------


def test_handshake_is_one_json_line_naming_pillow():
    import PIL

    line = handshake()
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert json.loads(line) == {"ready": True, "pillow": PIL.__version__}


def test_read_frame_returns_payloads_until_end_of_input():
    stream = BytesIO(frame(b"abc") + frame(b"") + frame(b"x" * 70000))
    assert read_frame(stream) == b"abc"
    assert read_frame(stream) == b""
    assert read_frame(stream) == b"x" * 70000
    assert read_frame(stream) is None


@pytest.mark.parametrize(
    "stream",
    [b"", b"\x00\x00", struct.pack(">I", 10) + b"short"],
    ids=["empty", "truncated-length", "truncated-payload"],
)
def test_read_frame_treats_a_truncated_frame_as_end_of_input(stream):
    assert read_frame(BytesIO(stream)) is None


def test_read_frame_refuses_an_oversized_length():
    stream = BytesIO(struct.pack(">I", render_worker.MAX_REQUEST_BYTES + 1) + b"x")
    assert read_frame(stream) is None


def test_reads_survive_short_reads():
    class Trickle(BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 3))

    assert read_frame(Trickle(frame(b"0123456789"))) == b"0123456789"


# -- hardening -------------------------------------------------------------------


def _record_harden(platform: str, hard: int = resource.RLIM_INFINITY):
    calls: list[tuple] = []
    with (
        mock.patch.object(render_worker.Image, "MAX_IMAGE_PIXELS", None),
        warnings.catch_warnings(),
    ):
        harden(
            platform=platform,
            setrlimit=lambda which, limits: calls.append(("setrlimit", which, limits)),
            getrlimit=lambda which: (hard, hard),
            set_pdeathsig=lambda: calls.append(("pdeathsig",)),
        )
        bomb_filter = [
            f
            for f in warnings.filters
            if f[0] == "error" and f[2] is render_worker.Image.DecompressionBombWarning
        ]
        limit = render_worker.Image.MAX_IMAGE_PIXELS
    return calls, bomb_filter, limit


def test_harden_on_linux_sets_every_limit_in_order():
    calls, bomb_filter, limit = _record_harden("linux")
    assert calls == [
        ("setrlimit", resource.RLIMIT_CORE, (0, 0)),
        ("setrlimit", resource.RLIMIT_AS, (1024**3, 1024**3)),
        ("pdeathsig",),
        ("setrlimit", resource.RLIMIT_CPU, (3060, 3060)),
    ]
    assert bomb_filter
    assert limit == PIXEL_LIMIT == 40_000_000


def test_harden_elsewhere_skips_the_linux_only_limits():
    calls, bomb_filter, limit = _record_harden("darwin")
    assert calls == [
        ("setrlimit", resource.RLIMIT_CORE, (0, 0)),
        ("setrlimit", resource.RLIMIT_CPU, (3060, 3060)),
    ]
    assert bomb_filter and limit == PIXEL_LIMIT


def test_harden_never_asks_above_an_existing_hard_limit():
    calls, _, _ = _record_harden("linux", hard=100)
    assert ("setrlimit", resource.RLIMIT_AS, (100, 100)) in calls
    assert ("setrlimit", resource.RLIMIT_CPU, (100, 100)) in calls


def test_pdeathsig_is_armed_through_prctl():
    fake_libc = mock.Mock()
    with mock.patch("ctypes.CDLL", return_value=fake_libc) as cdll:
        render_worker._set_pdeathsig()
    cdll.assert_called_once_with(None, use_errno=True)
    fake_libc.prctl.assert_called_once_with(1, int(signal.SIGKILL), 0, 0, 0)


# -- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [b"", b"hello world, plain text", b"<!doctype html><html></html>", b"%PDF-1.7\n"],
    ids=["empty", "text", "html", "pdf"],
)
def test_bytes_that_do_not_sniff_as_a_picture_are_not_an_image(data):
    assert_refused(render(data), "not_an_image")


def test_a_picture_whose_decoded_format_differs_from_its_magic_is_a_mismatch():
    png_sniff = sniff(encode(Image.new("RGB", (4, 4)), "PNG"))
    with mock.patch.object(render_worker, "sniff", return_value=png_sniff):
        assert_refused(render(tiny_jpeg()), "format_mismatch")


@pytest.mark.parametrize(
    "data",
    [
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 40,
        tiny_jpeg()[:200],
        encode(Image.new("RGB", (64, 64), "red"), "PNG")[:-30],
    ],
    ids=["png-garbage", "truncated-jpeg", "truncated-png"],
)
def test_undecodable_bytes_are_decoder_failed(data):
    assert_refused(render(data), "decoder_failed")


# -- size limits -----------------------------------------------------------------


def test_progressive_coefficient_bytes_matches_the_formula():
    layers_420 = [(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)]
    assert progressive_coefficient_bytes((12000, 9000), layers_420) == (
        1500 * 1125 * 128 + 2 * 750 * 563 * 128
    )
    layers_444 = [(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)]
    assert progressive_coefficient_bytes((12000, 9000), layers_444) == 3 * 1500 * 1125 * 128
    assert progressive_coefficient_bytes((12000, 9000), layers_420) <= PROGRESSIVE_MAX_COEF_BYTES
    assert progressive_coefficient_bytes((12000, 9000), layers_444) > PROGRESSIVE_MAX_COEF_BYTES


def insert_junk_before(data: bytes, marker: int) -> bytes:
    """Put one non-0xFF byte right before the first ``FF <marker>`` segment."""
    at = data.index(bytes([0xFF, marker]), 2)
    return data[:at] + b"\x00" + data[at:]


TOO_LARGE_JPEG_HEADERS = {
    "progressive-444-12000x9000": lambda: patch_jpeg_size(
        tiny_jpeg(progressive=True, subsampling=0), 12000, 9000
    ),
    "progressive-cmyk-12000x9000": lambda: patch_jpeg_size(
        tiny_jpeg(mode="CMYK", progressive=True), 12000, 9000
    ),
    "progressive-420-16000x12000": lambda: patch_jpeg_size(
        tiny_jpeg(progressive=True), 16000, 12000
    ),
    "baseline-60000x60000": lambda: patch_jpeg_size(tiny_jpeg(), 60000, 60000),
    # Sequential, but its first scan carries one of three components: libjpeg
    # buffers the whole image's coefficients (3 x 2000 x 1500 blocks x 128 B),
    # though the draft would bring it to 12 Mpx.
    "baseline-noninterleaved-444-16000x12000": lambda: patch_jpeg_size(
        flat_jpeg(16, 16, interleaved=False), 16000, 12000
    ),
    # libjpeg and Pillow skip a stray byte between segments; the multi-scan
    # walk must not stop there and call the stream single-scan.
    "baseline-noninterleaved-junk-before-sos-16000x12000": lambda: insert_junk_before(
        patch_jpeg_size(flat_jpeg(16, 16, interleaved=False), 16000, 12000), 0xDA
    ),
    "baseline-noninterleaved-junk-before-sof-16000x12000": lambda: insert_junk_before(
        patch_jpeg_size(flat_jpeg(16, 16, interleaved=False), 16000, 12000), 0xC0
    ),
}


def test_multi_scan_detection_follows_the_first_scan():
    assert jpeg_has_multiple_scans(tiny_jpeg()) is False
    assert jpeg_has_multiple_scans(tiny_jpeg(mode="L")) is False
    assert jpeg_has_multiple_scans(tiny_jpeg(progressive=True)) is True
    assert jpeg_has_multiple_scans(flat_jpeg(16, 16)) is False
    assert jpeg_has_multiple_scans(flat_jpeg(16, 16, progressive=True)) is True
    assert jpeg_has_multiple_scans(flat_jpeg(16, 16, interleaved=False)) is True


@pytest.mark.parametrize(
    "data",
    [
        b"\xff\xd8",
        b"\xff\xd8garbage",
        b"\xff\xd8\xff\xff\xff\xd0\xff\xdb\x00",
        b"\xff\xd8\xff\xda\x00\x08\x01",
        b"\xff\xd8\xff\xda",
        tiny_jpeg()[:30],
        b"\xff\xd8\x00\xff\xff",
        b"\xff\xd8\xff\xc0\x00\x11\x08\x00",
    ],
    ids=[
        "soi-only",
        "no-marker",
        "short-length",
        "scan-before-frame",
        "short-scan",
        "cut",
        "trailing-fill",
        "short-frame",
    ],
)
def test_multi_scan_detection_fails_closed_on_an_unwalkable_stream(data):
    # The coefficient bound then applies; a small stream stays under it.
    assert jpeg_has_multiple_scans(data) is True


def test_multi_scan_detection_fails_closed_on_end_of_image_before_a_scan():
    assert jpeg_has_multiple_scans(b"\xff\xd8\xff\xd9") is True


@pytest.mark.parametrize("marker", [0xDA, 0xC0, 0xDB], ids=["sos", "sof", "dqt"])
def test_multi_scan_detection_skips_stray_bytes_like_libjpeg(marker):
    assert jpeg_has_multiple_scans(insert_junk_before(flat_jpeg(16, 16), marker)) is False
    assert (
        jpeg_has_multiple_scans(insert_junk_before(flat_jpeg(16, 16, interleaved=False), marker))
        is True
    )


def test_multi_scan_detection_skips_stuffed_and_fill_bytes():
    data = flat_jpeg(16, 16)
    at = data.index(b"\xff\xda")
    assert jpeg_has_multiple_scans(data[:at] + b"\xff\x00\xff\xff" + data[at:]) is False


def test_a_small_jpeg_with_a_stray_byte_still_renders():
    header, body = render(insert_junk_before(flat_jpeg(64, 48, interleaved=False), 0xDA))
    assert_rendition(header, body, mime="image/jpeg", mode="RGB")


def test_a_small_non_interleaved_jpeg_renders():
    header, body = render(flat_jpeg(64, 48, interleaved=False))
    assert_rendition(header, body, mime="image/jpeg", mode="RGB")
    assert (header["w"], header["h"]) == (64, 48)


@pytest.mark.parametrize(
    "error",
    [
        MemoryError,
        lambda: Image.DecompressionBombError("bomb"),
        lambda: Image.DecompressionBombWarning("bomb"),
    ],
    ids=["memory", "bomb-error", "bomb-warning"],
)
def test_a_size_failure_during_decode_is_rendition_too_large(error):
    with mock.patch.object(render_worker, "_normalise_mode", side_effect=error()):
        assert_refused(render(encode(Image.new("RGB", (4, 4)), "PNG")), "rendition_too_large")


@pytest.mark.parametrize("name", sorted(TOO_LARGE_JPEG_HEADERS))
def test_oversized_jpegs_are_refused_from_the_header(name):
    # The scan data is a 16x16 picture's: any decode attempt would fail as
    # decoder_failed, so rendition_too_large proves the refusal came first.
    assert_refused(render(TOO_LARGE_JPEG_HEADERS[name]()), "rendition_too_large")


@pytest.mark.parametrize(
    "size", [(9000, 9000), (7000, 7000)], ids=["over-twice-limit", "over-limit"]
)
def test_oversized_non_jpeg_is_refused(size):
    png = encode(Image.new("L", (8, 8)), "PNG")
    assert_refused(render(patch_png_size(png, *size)), "rendition_too_large")


def test_the_jpeg_pixel_limit_is_lifted_for_open_only():
    seen = []
    real_open = render_worker.Image.open

    def spy(*args, **kwargs):
        seen.append(render_worker.Image.MAX_IMAGE_PIXELS)
        im = real_open(*args, **kwargs)
        return im

    with mock.patch.object(render_worker.Image, "open", side_effect=spy):
        header, _ = render(tiny_jpeg())
        assert header["ok"]
        render(encode(Image.new("RGB", (4, 4)), "PNG"))
    assert seen == [None, PIXEL_LIMIT]


def test_handle_restores_the_callers_pixel_limit():
    before = Image.MAX_IMAGE_PIXELS
    render(tiny_jpeg())
    render(b"not a picture")
    assert Image.MAX_IMAGE_PIXELS == before


@pytest.mark.xdist_group("render_big")
@pytest.mark.parametrize("name", ["baseline_12000x9000", "baseline_16000x12000"])
def test_large_baseline_jpegs_render_in_process(big_jpegs, name):
    header, body = render(big_jpegs[name].read_bytes())
    assert_rendition(header, body, mime="image/jpeg", mode="RGB")
    assert (header["w"], header["h"]) == (1024, 768)


# -- mode normalisation ----------------------------------------------------------


def _tiff(mode: str, value, **params) -> bytes:
    return encode(Image.new(mode, (40, 30), value), "TIFF", **params)


MODE_FIXTURES = {
    "cmyk-jpeg": (lambda: encode(Image.new("CMYK", (40, 30), (0, 50, 100, 0)), "JPEG"), "RGB"),
    "cmyk-tiff": (lambda: _tiff("CMYK", (0, 50, 100, 0)), "RGB"),
    # Pillow's raw YCbCr TIFF writer emits a file its reader rejects; libtiff
    # writes a valid one (photometric 6).
    "ycbcr-tiff": (lambda: _tiff("YCbCr", (100, 120, 140), compression="tiff_lzw"), "RGB"),
    "lab-tiff": (lambda: _tiff("LAB", (50, 10, 20)), "RGB"),
    "i16b-tiff": (lambda: _tiff("I;16B", 600), "L"),
    "i16-tiff": (lambda: _tiff("I;16", 600), "L"),
    "i-tiff": (lambda: _tiff("I", 70000), "L"),
    "f-tiff": (lambda: _tiff("F", 0.25), "L"),
    "bilevel-png": (lambda: encode(Image.new("1", (40, 30), 1), "PNG"), "RGB"),
    "palette-png": (lambda: encode(Image.new("P", (40, 30), 3), "PNG"), "RGB"),
    "palette-transparent-png": (
        lambda: encode(Image.new("P", (40, 30), 3), "PNG", transparency=3),
        "RGBA",
    ),
    "pa-tiff": (lambda: _tiff("PA", (3, 128)), "RGBA"),
    "la-png": (lambda: encode(Image.new("LA", (40, 30), (90, 128)), "PNG"), "LA"),
    "rgba-png": (lambda: encode(Image.new("RGBA", (40, 30), (1, 2, 3, 4)), "PNG"), "RGBA"),
    "l-jpeg": (lambda: tiny_jpeg(mode="L"), "L"),
    "gif": (lambda: encode(Image.new("P", (40, 30), 1), "GIF"), "RGB"),
    "webp": (lambda: encode(Image.new("RGB", (40, 30), "blue"), "WEBP"), "RGB"),
    "bmp": (lambda: encode(Image.new("RGB", (40, 30), "green"), "BMP"), "RGB"),
}


@pytest.mark.parametrize("name", sorted(MODE_FIXTURES))
def test_every_decoded_mode_is_normalised(name):
    build, mode = MODE_FIXTURES[name]
    data = build()
    header, body = render(data)
    source_is_jpeg = sniff(data).mime == "image/jpeg"
    mime = "image/jpeg" if source_is_jpeg and mode in ("RGB", "L") else "image/png"
    assert_rendition(header, body, mime=mime, mode=mode)


@pytest.mark.parametrize("mode", ["I", "F"])
def test_flat_high_bit_depth_pictures_render_flat(mode):
    header, body = render(_tiff(mode, 5))
    assert_rendition(header, body, mime="image/png", mode="L")
    assert Image.open(BytesIO(body)).getextrema() == (0, 0)


def test_high_bit_depth_range_is_stretched_to_eight_bits():
    im = Image.new("I;16", (2, 1))
    im.putpixel((0, 0), 1000)
    im.putpixel((1, 0), 3000)
    header, body = render(encode(im, "TIFF"))
    assert header["mode"] == "L"
    assert Image.open(BytesIO(body)).getextrema() == (0, 255)


def test_an_mpo_renders_its_first_frame_as_jpeg():
    first = Image.new("RGB", (40, 30), (200, 0, 0))
    second = Image.new("RGB", (40, 30), (0, 0, 200))
    data = encode(first, "MPO", save_all=True, append_images=[second])
    assert Image.open(BytesIO(data)).format == "MPO"
    header, body = render(data)
    assert_rendition(header, body, mime="image/jpeg", mode="RGB")
    assert header["format"] == "JPEG"
    red, _, blue = Image.open(BytesIO(body)).getpixel((20, 15))
    assert red > 150 and blue < 60


def test_an_animated_gif_renders_frame_zero():
    frames = [Image.new("P", (20, 20), i) for i in (1, 2)]
    data = encode(frames[0], "GIF", save_all=True, append_images=frames[1:])
    header, body = render(data)
    assert_rendition(header, body, mime="image/png")
    assert header["format"] == "GIF"


# -- geometry and encoding -------------------------------------------------------


def test_renditions_fit_in_1024_keeping_aspect():
    header, body = render(encode(Image.new("RGB", (3000, 1500), "red"), "PNG"))
    assert_rendition(header, body, mime="image/png")
    assert (header["w"], header["h"]) == (1024, 512)


def test_small_pictures_are_not_enlarged():
    header, _ = render(encode(Image.new("RGB", (40, 30)), "PNG"))
    assert (header["w"], header["h"]) == (40, 30)


def test_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise
    data = encode(Image.new("RGB", (200, 100), "red"), "JPEG", exif=exif)
    header, body = render(data)
    assert (header["w"], header["h"]) == (100, 200)
    assert 0x0112 not in Image.open(BytesIO(body)).getexif()


def test_a_large_opaque_png_rendition_becomes_jpeg():
    data = encode(noise("RGB", (1024, 1024)), "PNG")
    header, body = render(data)
    assert_rendition(header, body, mime="image/jpeg", mode="RGB")


def test_a_large_alpha_png_rendition_stays_png_at_768():
    data = encode(noise("RGBA", (1024, 1024)), "PNG")
    assert len(data) > render_worker.RENDITION_MAX_BYTES
    header, body = render(data)
    assert_rendition(header, body, mime="image/png", mode="RGBA")
    assert (header["w"], header["h"]) == (768, 768)
    assert len(body) <= render_worker.RENDITION_MAX_BYTES


def test_a_rendition_still_too_large_at_768_is_refused():
    data = encode(noise("RGBA", (1024, 1024)), "PNG")
    with mock.patch.object(render_worker, "RENDITION_MAX_BYTES", 1000):
        assert_refused(render(data), "rendition_too_large")


# -- the real worker process -----------------------------------------------------


def test_worker_round_trip_and_clean_exit_on_end_of_input(worker):
    import PIL

    assert worker.ready == {"ready": True, "pillow": PIL.__version__}
    header, body = worker.render(tiny_jpeg())
    assert_rendition(header, body, mime="image/jpeg")
    assert_refused(worker.render(b"plain text"), "not_an_image")
    assert_refused(worker.render(tiny_jpeg()[:200]), "decoder_failed")
    header, body = worker.render(encode(Image.new("RGBA", (30, 30)), "PNG"))
    assert_rendition(header, body, mime="image/png", mode="RGBA")
    assert worker.close() == 0


@pytest.mark.xdist_group("render_big")
def test_worker_renders_the_large_jpegs(worker, big_jpegs):
    for name in ("baseline_12000x9000", "baseline_16000x12000", "progressive_420_12000x9000"):
        header, body = worker.render(big_jpegs[name].read_bytes())
        assert_rendition(header, body, mime="image/jpeg", mode="RGB")
        assert (header["w"], header["h"]) == (1024, 768), name
    for name in sorted(TOO_LARGE_JPEG_HEADERS):
        assert_refused(worker.render(TOO_LARGE_JPEG_HEADERS[name]()), "rendition_too_large")
    assert worker.close() == 0


def test_worker_refuses_a_picture_between_the_pixel_limit_and_twice_it(worker):
    # 49 Mpx: above MAX_IMAGE_PIXELS, below Pillow's error threshold, so only
    # the bomb warning turned into an error refuses it at open.
    png = patch_png_size(encode(Image.new("L", (8, 8)), "PNG"), 7000, 7000)
    assert_refused(worker.render(png), "rendition_too_large")
    assert worker.close() == 0


_MAIN_PROBE = r"""
import json, resource, runpy, sys, warnings
runpy.run_module("osprey.imaging.render_worker", run_name="__main__")
from PIL import Image
sys.stderr.write(json.dumps({
    "core": resource.getrlimit(resource.RLIMIT_CORE),
    "cpu": resource.getrlimit(resource.RLIMIT_CPU),
    "bomb_filter": any(
        f[0] == "error" and f[2] is Image.DecompressionBombWarning for f in warnings.filters
    ),
    "max_pixels": Image.MAX_IMAGE_PIXELS,
}))
"""


def test_the_worker_entry_point_hardens_before_its_handshake():
    import PIL

    result = subprocess.run(
        [sys.executable, "-I", "-c", _MAIN_PROBE],
        input=b"",
        env={},
        cwd="/",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout) == {"ready": True, "pillow": PIL.__version__}
    state = json.loads(result.stderr)
    assert state["core"] == [0, 0]
    assert 0 < state["cpu"][0] <= 3060 and 0 < state["cpu"][1] <= 3060
    assert state["bomb_filter"] is True
    assert state["max_pixels"] == PIXEL_LIMIT


@pytest.mark.parametrize(
    ("progressive", "interleaved"), [(False, True), (True, True), (False, False)]
)
def test_flat_jpeg_fixture_decodes_to_grey(progressive, interleaved):
    im = Image.open(BytesIO(flat_jpeg(40, 24, progressive=progressive, interleaved=interleaved)))
    assert im.info.get("progressive", 0) == int(progressive)
    assert im.convert("RGB").getextrema() == ((128, 128),) * 3


_PROBE = r"""
import json, os, resource, sys, warnings
before = set(sys.modules)
from osprey.imaging import render_worker
render_worker.harden()
from PIL import Image
data = sys.stdin.buffer.read()
header = render_worker.handle(data).split(b"\n", 1)[0]
usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({
    "header": json.loads(header),
    "loaded": sorted(set(sys.modules) - before),
    "core": resource.getrlimit(resource.RLIMIT_CORE),
    "cpu": resource.getrlimit(resource.RLIMIT_CPU),
    "as": resource.getrlimit(resource.RLIMIT_AS),
    "cwd": os.getcwd(),
    "environ": dict(os.environ),
    "bomb_filter": any(
        f[0] == "error" and f[2] is Image.DecompressionBombWarning for f in warnings.filters
    ),
    "max_pixels": Image.MAX_IMAGE_PIXELS,
    "peak_rss_mb": usage / (1024 * 1024 if sys.platform == "darwin" else 1024),
    "proc_environ": (
        open("/proc/self/environ", "rb").read().decode() if sys.platform.startswith("linux") else ""
    ),
}))
"""


def _probe(data: bytes) -> dict:
    result = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE],
        input=data,
        env={},
        cwd="/",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    return json.loads(result.stdout)


def test_worker_process_is_hardened_and_imports_no_service_code():
    report = _probe(tiny_jpeg())
    assert report["header"]["ok"] is True
    assert report["core"] == [0, 0]
    assert report["cpu"][1] <= 3060
    assert report["cwd"] == "/"
    assert set(report["environ"]) <= {"__CF_USER_TEXT_ENCODING", "LC_CTYPE"}
    assert report["bomb_filter"] is True
    assert report["max_pixels"] == PIXEL_LIMIT

    loaded = report["loaded"]
    assert not [m for m in loaded if m == "osprey.services" or m.startswith("osprey.services.")]
    assert "osprey.utils.config" not in loaded
    stdlib = sys.stdlib_module_names
    outside = [
        m
        for m in loaded
        if m.split(".")[0] not in stdlib
        and m not in ALLOWED_OSPREY_MODULES
        and m.split(".")[0] not in ALLOWED_THIRD_PARTY_ROOTS
    ]
    assert outside == []


@pytest.mark.skipif(not LINUX, reason="RLIMIT_AS and /proc/self/environ are Linux-only")
def test_worker_process_on_linux_has_no_environment_and_a_capped_address_space():
    report = _probe(tiny_jpeg())
    assert report["proc_environ"] == ""
    assert report["as"] == [1024**3, 1024**3]


@pytest.mark.parametrize("name", sorted(TOO_LARGE_JPEG_HEADERS))
def test_header_refusals_allocate_nothing(name):
    report = _probe(TOO_LARGE_JPEG_HEADERS[name]())
    assert report["header"]["reason"] == "rendition_too_large"
    assert report["peak_rss_mb"] < 200


@pytest.mark.skipif(not LINUX, reason="/proc inspection of the real worker is Linux-only")
def test_real_worker_process_state_on_linux(worker):
    pid = worker.proc.pid
    assert Path(f"/proc/{pid}/environ").read_bytes() == b""
    assert os.readlink(f"/proc/{pid}/cwd") == "/"
    limits = Path(f"/proc/{pid}/limits").read_text()
    core = next(line for line in limits.splitlines() if line.startswith("Max core file size"))
    assert core.split()[4:6] == ["0", "0"]
    cpu = next(line for line in limits.splitlines() if line.startswith("Max cpu time"))
    assert cpu.split()[3:5] == ["3060", "3060"]
    assert worker.close() == 0
