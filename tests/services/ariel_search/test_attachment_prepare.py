"""``prepare_picture``: sniff first, then render only accepted pictures."""

from __future__ import annotations

import hashlib
import io

import pytest

from osprey.imaging import render
from osprey.imaging.render import RenderOutcome, RenderUnavailable, Rendition
from osprey.services.ariel_search.attachments import prepare
from osprey.services.ariel_search.attachments.prepare import PreparedPicture, prepare_picture

Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.timeout(60)

HTML = b"<!DOCTYPE html><html><head><title>Login</title></head><body>sign in</body></html>"
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"


def _png(size: tuple[int, int] = (16, 12)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(out, "PNG")
    return out.getvalue()


@pytest.fixture
async def no_worker_left():
    """Close any cached worker before and after the test."""
    await render.close_render_worker()
    yield
    await render.close_render_worker()


@pytest.fixture
def forbid_render(monkeypatch):
    """Fail the test if the render worker is ever asked to run."""

    async def _boom(*args, **kwargs):  # pragma: no cover - only runs on a regression
        raise AssertionError("render_isolated must not be called")

    monkeypatch.setattr(render, "render_isolated", _boom)


@pytest.mark.usefixtures("no_worker_left", "forbid_render")
async def test_html_declared_png_is_not_an_image_without_worker():
    # The caller's declared type plays no part: only the bytes are sniffed.
    result = await prepare_picture(HTML)

    assert result.skip_reason == "not_an_image"
    assert result.mime_type == "text/html"
    assert not result.has_rendition
    assert result.rendition_sha256 is None
    assert render.worker_pid() is None


@pytest.mark.usefixtures("no_worker_left", "forbid_render")
async def test_reserved_format_is_reserved_without_worker():
    result = await prepare_picture(PDF)

    assert result == PreparedPicture(mime_type="application/pdf", skip_reason="reserved_format")
    assert render.worker_pid() is None


@pytest.mark.usefixtures("forbid_render")
async def test_unknown_bytes_are_not_an_image():
    result = await prepare_picture(b"\x00\x01\x02\x03" * 40)

    assert result.skip_reason == "not_an_image"
    assert result.mime_type == "application/octet-stream"
    assert not result.has_rendition


@pytest.mark.usefixtures("no_worker_left")
async def test_png_returns_rendition_fields_and_sha256():
    data = _png()

    result = await prepare_picture(data)

    assert result.skip_reason is None
    assert result.mime_type == "image/png"
    assert result.rendition_mime in {"image/png", "image/jpeg"}
    assert (result.rendition_w, result.rendition_h) == (16, 12)
    assert result.rendition_bytes
    assert result.rendition_sha256 == hashlib.sha256(result.rendition_bytes).hexdigest()
    rendered = Image.open(io.BytesIO(result.rendition_bytes))
    assert rendered.size == (16, 12)


async def test_render_content_skip_keeps_sniffed_mime(monkeypatch):
    calls = []

    async def _refuse(_data, *, task_id=None):
        calls.append(task_id)
        return RenderOutcome(rendition=None, reason="decoder_failed")

    monkeypatch.setattr(render, "render_isolated", _refuse)

    result = await prepare_picture(_png(), task_id="att-1")

    assert calls == ["att-1"]
    assert result == PreparedPicture(mime_type="image/png", skip_reason="decoder_failed")


async def test_rendition_fields_come_from_the_worker_reply(monkeypatch):
    body = _png((4, 3))

    async def _ok(_data, **_kwargs):
        return RenderOutcome(
            rendition=Rendition(
                data=body, mime="image/png", format="PNG", width=4, height=3, mode="RGB"
            ),
            reason=None,
        )

    monkeypatch.setattr(render, "render_isolated", _ok)

    result = await prepare_picture(_png())

    assert result.rendition_bytes == body
    assert (result.rendition_w, result.rendition_h) == (4, 3)
    assert result.rendition_sha256 == hashlib.sha256(body).hexdigest()


async def test_render_unavailable_propagates(monkeypatch):
    async def _down(_data, **_kwargs):
        raise RenderUnavailable("render worker exited (exit code 1)", 1)

    monkeypatch.setattr(render, "render_isolated", _down)

    with pytest.raises(RenderUnavailable) as excinfo:
        await prepare_picture(_png())
    assert excinfo.value.exit_code == 1


def test_module_holds_no_database_state():
    with open(prepare.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert "enhanced_entries" not in text
    assert "ariel_search.database" not in text
    assert "psycopg" not in text
