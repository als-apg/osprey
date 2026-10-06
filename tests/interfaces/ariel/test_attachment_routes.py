"""Tests for the ARIEL attachment download routes and the shared response helper.

Every stored attachment is served through one helper that sniffs the bytes it
is about to send: only a raster picture is answered ``inline`` with its sniffed
type; everything else goes out as an ``application/octet-stream`` download.
Stored and declared types are never trusted for serving.
"""

from __future__ import annotations

import asyncio
import io
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.ariel.api import routes
from osprey.interfaces.ariel.api.attachment_response import attachment_response

CSP = "sandbox; default-src 'none'"
ATT_ID = "att-0123456789ab"
HTML = b"<!doctype html><html><body><script>alert(1)</script></body></html>"
PDF = b"%PDF-1.4\n%fake pdf body\n%%EOF\n"


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def _jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (30, 200, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _viewable_row(**overrides) -> dict:
    row = {
        "attachment_id": ATT_ID,
        "entry_id": "e-1",
        "filename": "beam.png",
        "mime_type": "image/png",
        "copy_status": "copied",
        "skip_reason": None,
        "rendition_mime": "image/jpeg",
        "rendition_sha256": "ab" * 32,
        "rendition_bytes": _jpeg(),
    }
    row.update(overrides)
    return row


@pytest.fixture
def service():
    svc = AsyncMock()
    svc.repository = AsyncMock()
    svc.repository.get_attachment_original = AsyncMock(return_value=None)
    svc.repository.get_rendition = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(routes.router)
    app.state.ariel_service = service
    app.state.config_panel_enabled = True
    return TestClient(app)


def _assert_hardened(resp) -> None:
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["content-security-policy"] == CSP


# ---------------------------------------------------------------------------
# attachment_response helper
# ---------------------------------------------------------------------------


class TestAttachmentResponse:
    def test_raster_picture_is_inline_with_sniffed_type(self) -> None:
        resp = attachment_response(_png(), "beam.png")
        assert resp.media_type == "image/png"
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["content-disposition"] == "inline; filename*=UTF-8''beam.png"
        _assert_hardened(resp)

    def test_sniffed_type_wins_over_the_filename(self) -> None:
        resp = attachment_response(_jpeg(), "beam.png")
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.headers["content-disposition"].startswith("inline;")

    @pytest.mark.parametrize(
        "data",
        [HTML, PDF, b"<svg xmlns='http://www.w3.org/2000/svg'></svg>", b"plain text", b""],
        ids=["html", "pdf", "svg", "text", "empty"],
    )
    def test_anything_else_is_an_octet_stream_download(self, data: bytes) -> None:
        resp = attachment_response(data, "page.png")
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["content-disposition"] == "attachment; filename*=UTF-8''page.png"
        _assert_hardened(resp)
        assert resp.body == data

    def test_filename_is_rfc5987_encoded(self) -> None:
        resp = attachment_response(PDF, 'résumé "x";\r\nX-Evil: 1.pdf')
        disposition = resp.headers["content-disposition"]
        assert disposition.startswith("attachment; filename*=UTF-8''")
        assert "r%C3%A9sum%C3%A9" in disposition
        for raw in ('"', "\r", "\n", ";", " "):
            assert raw not in disposition.split("''", 1)[1]
        assert "filename=" not in disposition.replace("filename*=", "")

    def test_only_the_first_sniff_window_is_read(self) -> None:
        """Bytes beyond the sniff window never change the answer."""
        png = _png()
        resp = attachment_response(png[:32] + HTML * 10, "beam.png")
        assert resp.headers["content-type"] == "image/png"


# ---------------------------------------------------------------------------
# GET /api/attachments/{id}
# ---------------------------------------------------------------------------


class TestOriginalRoute:
    def test_picture_is_served_inline(self, client, service) -> None:
        png = _png()
        service.repository.get_attachment_original.return_value = {
            "filename": "beam.png",
            "mime_type": "image/png",
            "data": png,
        }

        resp = client.get(f"/api/attachments/{ATT_ID}")

        assert resp.status_code == 200
        assert resp.content == png
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["content-disposition"] == "inline; filename*=UTF-8''beam.png"
        _assert_hardened(resp)
        service.repository.get_attachment_original.assert_awaited_once_with(ATT_ID)
        service.repository.get_rendition.assert_not_called()

    def test_html_declared_as_png_is_served_as_a_download(self, client, service) -> None:
        service.repository.get_attachment_original.return_value = {
            "filename": "beam.png",
            "mime_type": "image/png",
            "data": HTML,
        }

        resp = client.get(f"/api/attachments/{ATT_ID}")

        assert resp.status_code == 200
        assert resp.content == HTML
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["content-disposition"].startswith("attachment;")
        _assert_hardened(resp)

    def test_pdf_is_served_as_a_download(self, client, service) -> None:
        service.repository.get_attachment_original.return_value = {
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "data": memoryview(PDF),
        }

        resp = client.get(f"/api/attachments/{ATT_ID}")

        assert resp.status_code == 200
        assert resp.content == PDF
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["content-disposition"] == "attachment; filename*=UTF-8''report.pdf"

    def test_missing_filename_falls_back_to_a_name(self, client, service) -> None:
        service.repository.get_attachment_original.return_value = {
            "filename": None,
            "mime_type": None,
            "data": PDF,
        }

        resp = client.get(f"/api/attachments/{ATT_ID}")

        assert resp.status_code == 200
        assert "filename*=UTF-8''file" in resp.headers["content-disposition"]

    def test_absent_original_is_404_picture_not_available(self, client, service) -> None:
        resp = client.get(f"/api/attachments/{ATT_ID}")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "picture not available"
        assert ATT_ID not in resp.text
        service.repository.get_attachment_original.assert_awaited_once_with(ATT_ID)

    @pytest.mark.parametrize(
        "bad_id",
        ["att-XYZ<script>", "att-0123456789ab0", "att-0123456789AB", "other-id"],
    )
    def test_malformed_id_is_404_without_echo_or_lookup(self, client, service, bad_id) -> None:
        resp = client.get(f"/api/attachments/{bad_id}")

        assert resp.status_code == 404
        assert bad_id not in resp.text
        assert "XYZ" not in resp.text
        service.repository.get_attachment_original.assert_not_called()

    def test_long_form_id_is_accepted(self, client, service) -> None:
        long_id = "att-" + "0123456789ab" * 2
        service.repository.get_attachment_original.return_value = {
            "filename": "a.pdf",
            "mime_type": "application/pdf",
            "data": PDF,
        }

        resp = client.get(f"/api/attachments/{long_id}")

        assert resp.status_code == 200
        service.repository.get_attachment_original.assert_awaited_once_with(long_id)


# ---------------------------------------------------------------------------
# GET /api/attachments/{id}/rendition
# ---------------------------------------------------------------------------


class TestRenditionRoute:
    def test_viewable_rendition_is_served_inline_with_hardening(self, client, service) -> None:
        row = _viewable_row()
        service.repository.get_rendition.return_value = row

        resp = client.get(f"/api/attachments/{ATT_ID}/rendition")

        assert resp.status_code == 200
        assert resp.content == row["rendition_bytes"]
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.headers["content-disposition"] == "inline; filename*=UTF-8''beam.png"
        _assert_hardened(resp)
        service.repository.get_rendition.assert_awaited_once_with(ATT_ID)
        service.repository.get_attachment_original.assert_not_called()

    def test_rendition_bytes_are_sniffed_not_trusted(self, client, service) -> None:
        service.repository.get_rendition.return_value = _viewable_row(
            rendition_bytes=memoryview(HTML)
        )

        resp = client.get(f"/api/attachments/{ATT_ID}/rendition")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["content-disposition"].startswith("attachment;")
        _assert_hardened(resp)

    @pytest.mark.parametrize(
        "row",
        [
            None,
            _viewable_row(skip_reason="reserved_format", mime_type="application/pdf"),
            _viewable_row(copy_status="pending"),
            _viewable_row(rendition_sha256=None),
            _viewable_row(mime_type="application/pdf"),
            _viewable_row(skip_reason="too_large"),
        ],
        ids=["no-row", "pdf", "pending", "no-rendition", "non-image", "skipped"],
    )
    def test_non_viewable_is_404(self, client, service, row) -> None:
        service.repository.get_rendition.return_value = row

        resp = client.get(f"/api/attachments/{ATT_ID}/rendition")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "picture not available"
        assert ATT_ID not in resp.text

    def test_malformed_id_is_404_without_echo_or_lookup(self, client, service) -> None:
        resp = client.get("/api/attachments/att-nothex%3Cb%3E/rendition")

        assert resp.status_code == 404
        assert "nothex" not in resp.text
        service.repository.get_rendition.assert_not_called()


# ---------------------------------------------------------------------------
# Serving never renders
# ---------------------------------------------------------------------------


def test_serving_spawns_no_child_process(client, service, monkeypatch) -> None:
    """Viewers serve stored bytes only: no render worker, no picture preparation."""
    from osprey.imaging import render as render_module
    from osprey.services.ariel_search.attachments import prepare as prepare_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a viewer route must never start a child process")

    async def forbidden_async(*_args, **_kwargs):
        raise AssertionError("a viewer route must never start a child process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_async)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", forbidden_async)
    monkeypatch.setattr(render_module, "_spawn", forbidden_async)
    monkeypatch.setattr(prepare_module, "prepare_picture", forbidden_async)
    monkeypatch.setattr("subprocess.Popen", forbidden)

    service.repository.get_rendition.return_value = _viewable_row()
    service.repository.get_attachment_original.return_value = {
        "filename": "beam.png",
        "mime_type": "image/png",
        "data": _png(),
    }

    assert client.get(f"/api/attachments/{ATT_ID}/rendition").status_code == 200
    assert client.get(f"/api/attachments/{ATT_ID}").status_code == 200

    service.repository.get_rendition.return_value = None
    assert client.get(f"/api/attachments/{ATT_ID}/rendition").status_code == 404
