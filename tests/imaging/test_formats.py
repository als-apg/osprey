"""Tests for the dependency-free picture format registry.

The registry classifies bytes by magic number alone, so every fixture here is a
hand-built byte prefix: no image library is needed to produce or classify one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from osprey.imaging import formats
from osprey.imaging.formats import (
    ACCEPTED,
    CONFIG_SKIP_REASONS,
    CONTENT_SKIP_REASONS,
    RENDITION_MAX_BYTES,
    ROW_SKIP_REASONS,
    SOURCE_SKIP_REASONS,
    SUMMARY_ONLY_REASONS,
    VIEWABLE_SQL,
    admitted_formats,
    is_markup,
    is_viewable,
    kind,
    skip_reason_text,
    sniff,
    viewable_sql,
)

_SRC = str(Path(__file__).resolve().parents[2] / "src")

# Magic-byte prefixes padded past the 32-byte sniff window with filler, so a
# classifier that read beyond its window would see bytes that are not part of
# the magic.
_PAD = b"\x00\x01\x02\x03" * 16

IMAGE_FIXTURES = {
    "png": (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + _PAD, "image/png"),
    "jpeg": (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + _PAD, "image/jpeg"),
    "jpeg-exif": (b"\xff\xd8\xff\xe1\x00\x10Exif\x00" + _PAD, "image/jpeg"),
    "gif87a": (b"GIF87a\x01\x00\x01\x00" + _PAD, "image/gif"),
    "gif89a": (b"GIF89a\x01\x00\x01\x00" + _PAD, "image/gif"),
    "webp": (b"RIFF\x24\x00\x00\x00WEBPVP8 " + _PAD, "image/webp"),
    "bmp": (
        b"BM\x3a\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00" + _PAD,
        "image/bmp",
    ),
    "tiff-le": (b"II*\x00\x08\x00\x00\x00" + _PAD, "image/tiff"),
    "tiff-be": (b"MM\x00*\x00\x00\x00\x08" + _PAD, "image/tiff"),
}

RESERVED_FIXTURES = {
    "svg": (b'<svg xmlns="http://www.w3.org/2000/svg" width="1"/>' + b" " * 20, "image/svg+xml"),
    "svg-xml-prolog": (
        b'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="x"/>',
        "image/svg+xml",
    ),
    "pdf": (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + _PAD, "application/pdf"),
    "csv": (b"timestamp,pv,value\n2026-10-01,SR:C01:BPM,1.25\n2026-10-01,x,2\n", "text/plain"),
    "mp4": (b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + _PAD, "video/mp4"),
    "zip": (b"PK\x03\x04\x14\x00\x00\x00\x08\x00" + _PAD, "application/zip"),
}


# -- sniff: accepted and reserved rows ----------------------------------------


@pytest.mark.parametrize("name", sorted(IMAGE_FIXTURES))
def test_accepted_formats_sniff_as_image(name):
    data, mime = IMAGE_FIXTURES[name]
    result = sniff(data)
    assert result.kind == "image"
    assert result.is_image
    assert result.mime == mime
    assert result.skip_reason is None
    assert not is_markup(result)


@pytest.mark.parametrize("name", sorted(RESERVED_FIXTURES))
def test_reserved_formats_are_skipped_with_reserved_format(name):
    data, mime = RESERVED_FIXTURES[name]
    result = sniff(data)
    assert result.kind == "reserved"
    assert not result.is_image
    assert result.mime == mime
    assert result.skip_reason == "reserved_format"
    assert not is_markup(result)


def test_svg_with_xml_prolog_stays_reserved_svg():
    result = sniff(RESERVED_FIXTURES["svg-xml-prolog"][0])
    assert (result.kind, result.row, result.mime) == ("reserved", "svg", "image/svg+xml")


def test_csv_is_classed_as_reserved_text():
    result = sniff(RESERVED_FIXTURES["csv"][0])
    assert (result.kind, result.row, result.mime) == ("reserved", "text", "text/plain")


def test_sniff_reads_only_the_first_32_bytes_for_magic():
    # A PNG signature after byte 32 is not a PNG.
    assert sniff(b"\x00" * 40 + b"\x89PNG\r\n\x1a\n").kind != "image"


@pytest.mark.parametrize("data", [b"", b"\x00\x00\x00\x00garbage\xff\xfe\x80", b"\x89PN"])
def test_unknown_bytes_are_not_an_image(data):
    result = sniff(data)
    assert result.kind == "unknown"
    assert result.skip_reason == "not_an_image"
    assert result.mime == "application/octet-stream"
    assert not is_markup(result)


# -- markup -------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"\xef\xbb\xbf\n<!DOCTYPE html><html><head><title>Sign in</title>",
        b"<!doctype HTML>\n<html lang=en>",
        b"  \r\n\t<HTML><body>proxy error</body></html>",
        b"<head><meta charset=utf-8></head>",
        b"\n<Body>Access denied</Body>",
    ],
)
def test_html_is_markup(data):
    result = sniff(data)
    assert is_markup(result)
    assert result.kind == "markup"
    assert result.mime == "text/html"
    assert result.skip_reason == "not_an_image"
    assert not result.is_image


def test_html_with_bom_and_newline_is_markup_never_text():
    data = b"\xef\xbb\xbf\n<!doctype html>\n<html><body>login</body></html>\n" * 3
    result = sniff(data)
    assert result.kind == "markup"
    assert result.row != "text"
    assert result.mime != "text/plain"


def test_markup_prefix_beyond_32_bytes_of_whitespace_is_not_markup():
    result = sniff(b" " * 33 + b"<html><body>x</body></html>")
    assert not is_markup(result)


def test_svg_and_xml_prolog_are_not_markup():
    assert not is_markup(sniff(RESERVED_FIXTURES["svg"][0]))
    assert not is_markup(sniff(RESERVED_FIXTURES["svg-xml-prolog"][0]))


def test_is_markup_accepts_none_and_non_results():
    assert not is_markup(None)


# -- text row -----------------------------------------------------------------


def test_text_row_accepts_utf8_cut_mid_character_at_window_edge():
    # 63 ASCII bytes then a two-byte UTF-8 character straddling the 64-byte window.
    data = b"a" * 63 + "é".encode() + b"rest"
    assert sniff(data).row == "text"


def test_text_starting_with_bmp_or_bzip_letters_is_text():
    assert sniff(b"BM readings,value\nBM1,0.5\nBM2,0.7\n").row == "text"
    assert sniff(b"BZh is not a bzip2 header here\n").row == "text"


def test_bzip2_is_reserved_archive():
    assert sniff(b"BZh91AY&SY" + _PAD).row == "archive"


def test_binary_with_control_bytes_is_not_text():
    assert sniff(b"abc\x00def" * 10).kind == "unknown"


# -- Pillow is never consulted --------------------------------------------------


def test_classification_never_calls_pillow():
    pil_image = pytest.importorskip("PIL.Image")
    fixtures = {**IMAGE_FIXTURES, **RESERVED_FIXTURES}
    with mock.patch.object(pil_image, "open", side_effect=AssertionError("Pillow called")):
        for name, (data, mime) in fixtures.items():
            assert sniff(data).mime == mime, name
        assert is_markup(sniff(b"<!doctype html><html>"))


def test_module_imports_nothing_outside_the_stdlib():
    code = (
        "import json, sys;"
        "before = set(sys.modules);"
        "import osprey.imaging.formats;"
        "allowed = {'osprey', 'osprey.version', 'osprey._version', 'osprey.docs_links', "
        "'osprey.imaging', 'osprey.imaging.formats'};"
        "delta = set(sys.modules) - before - allowed;"
        "print(json.dumps(sorted("
        "m for m in delta if m.split('.')[0] not in sys.stdlib_module_names)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=_SRC),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


# -- rows and Pillow format sets -----------------------------------------------


def test_admitted_pillow_format_sets():
    assert admitted_formats("image/jpeg") == frozenset({"JPEG", "MPO"})
    assert admitted_formats("image/png") == frozenset({"PNG"})
    assert admitted_formats("image/gif") == frozenset({"GIF"})
    assert admitted_formats("image/webp") == frozenset({"WEBP"})
    assert admitted_formats("image/bmp") == frozenset({"BMP", "DIB"})
    assert admitted_formats("image/tiff") == frozenset({"TIFF"})
    assert admitted_formats("image/svg+xml") == frozenset()
    assert admitted_formats("application/pdf") == frozenset()


def test_accepted_is_the_pillow_open_list():
    assert set(ACCEPTED) == {"PNG", "JPEG", "GIF", "WEBP", "BMP", "DIB", "TIFF"}
    # MPO is reached through the JPEG opener, never named to Image.open.
    assert "MPO" not in ACCEPTED


def test_accepted_names_are_registered_pillow_openers():
    pil_image = pytest.importorskip("PIL.Image")
    pil_image.init()
    assert set(ACCEPTED) <= set(pil_image.OPEN)


def test_reserved_rows_exist():
    reserved = {row.name for row in formats.ROWS.values() if row.status == "reserved"}
    assert {"svg", "pdf", "text", "video", "audio", "archive"} <= reserved
    accepted = {row.name for row in formats.ROWS.values() if row.status == "accepted"}
    assert accepted == {"png", "jpeg", "gif", "webp", "bmp", "tiff"}


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("image/png", "image"),
        ("IMAGE/JPEG; charset=binary", "image"),
        ("image/svg+xml", "reserved"),
        ("application/pdf", "reserved"),
        ("text/plain", "reserved"),
        ("text/html", "markup"),
        ("application/octet-stream", "unknown"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_kind_of_mime(mime, expected):
    assert kind(mime) == expected


# -- skip reasons -------------------------------------------------------------


def test_skip_reason_sets():
    assert CONTENT_SKIP_REASONS == frozenset(
        {
            "not_an_image",
            "reserved_format",
            "decoder_failed",
            "format_mismatch",
            "rendition_too_large",
            "not_a_regular_file",
        }
    )
    assert CONFIG_SKIP_REASONS == frozenset(
        {"origin_not_allowed", "size_cap", "per_entry_limit", "copy_on_ingest_mode"}
    )
    assert SOURCE_SKIP_REASONS == frozenset({"source_gone", "source_refused", "fetch_failed"})
    assert SUMMARY_ONLY_REASONS == frozenset({"no_source_url"})
    assert ROW_SKIP_REASONS == CONTENT_SKIP_REASONS | CONFIG_SKIP_REASONS | SOURCE_SKIP_REASONS


def test_skip_reason_sets_are_disjoint_and_summary_only_is_outside():
    sets = [CONTENT_SKIP_REASONS, CONFIG_SKIP_REASONS, SOURCE_SKIP_REASONS]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            assert not a & b
    assert not SUMMARY_ONLY_REASONS & ROW_SKIP_REASONS


@pytest.mark.parametrize("code", sorted(ROW_SKIP_REASONS | SUMMARY_ONLY_REASONS))
def test_every_code_has_text(code):
    text = skip_reason_text(code)
    assert isinstance(text, str) and text
    assert text != code


def test_source_refused_text():
    text = skip_reason_text("source_refused")
    assert "refused" in text
    assert "web page" in text
    assert "login" in text and "proxy" in text
    assert "osprey ariel attachments backfill" in text


@pytest.mark.parametrize(
    ("code", "key"),
    [
        ("origin_not_allowed", "ariel.attachments.allowed_origins"),
        ("size_cap", "ariel.attachments.max_file_mb"),
        ("per_entry_limit", "ariel.attachments.max_file_mb"),
        ("copy_on_ingest_mode", "ariel.attachments.copy_on_ingest"),
    ],
)
def test_config_reasons_name_their_key(code, key):
    assert key in skip_reason_text(code)


@pytest.mark.parametrize("code", sorted(SOURCE_SKIP_REASONS))
def test_source_reasons_name_backfill(code):
    assert "osprey ariel attachments backfill" in skip_reason_text(code)


def test_decoder_failed_names_retry_flag():
    assert "--retry-decoder-failed" in skip_reason_text("decoder_failed")


def test_unknown_code_text_does_not_raise():
    assert "mystery" in skip_reason_text("mystery")


def test_rendition_max_bytes():
    assert RENDITION_MAX_BYTES == int(3.5 * 1024 * 1024)


# -- viewable -----------------------------------------------------------------


def _row(**overrides):
    row = {
        "copy_status": "copied",
        "skip_reason": None,
        "mime_type": "image/png",
        "rendition_sha256": "ab" * 32,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("label", "row", "expected"),
    [
        ("copied png", _row(), True),
        ("copied png, no rendition yet", _row(rendition_sha256=None), False),
        (
            "copied + rendition_too_large",
            _row(skip_reason="rendition_too_large", rendition_sha256=None),
            False,
        ),
        (
            "copied pdf in all mode",
            _row(mime_type="application/pdf", skip_reason="not_an_image", rendition_sha256=None),
            False,
        ),
        ("pending", _row(copy_status="pending", mime_type=None, rendition_sha256=None), False),
        (
            "skipped",
            _row(copy_status="skipped", skip_reason="size_cap", rendition_sha256=None),
            False,
        ),
    ],
)
def test_is_viewable_table(label, row, expected):
    assert is_viewable(row) is expected, label
    assert is_viewable(SimpleNamespace(**row)) is expected, label


def test_copied_row_with_any_skip_reason_is_not_viewable():
    for code in sorted(ROW_SKIP_REASONS):
        assert not is_viewable(_row(skip_reason=code))


def test_viewable_sql_names_every_condition():
    assert "copy_status = 'copied'" in VIEWABLE_SQL
    assert "skip_reason IS NULL" in VIEWABLE_SQL
    assert "rendition_sha256 IS NOT NULL" in VIEWABLE_SQL
    for mime in ("image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp", "image/tiff"):
        assert f"'{mime}'" in VIEWABLE_SQL
    assert "%" not in VIEWABLE_SQL  # safe to splice into a statement with named placeholders


def test_viewable_sql_with_alias():
    sql = viewable_sql("f")
    assert "f.copy_status = 'copied'" in sql
    assert "f.rendition_sha256 IS NOT NULL" in sql
    assert viewable_sql() == VIEWABLE_SQL
    with pytest.raises(ValueError):
        viewable_sql("f; DROP TABLE x")
