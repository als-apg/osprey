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
from fastapi.testclient import TestClient

from osprey.interfaces.ariel.api import routes
from osprey.interfaces.ariel.api.attachment_response import attachment_response
from tests.interfaces.ariel._attachments import ATT_ID, jpeg_bytes, viewable_row
from tests.interfaces.ariel._routes_app import routes_app

CSP = "sandbox; default-src 'none'"
HTML = b"<!doctype html><html><body><script>alert(1)</script></body></html>"
PDF = b"%PDF-1.4\n%fake pdf body\n%%EOF\n"
WEB_KEY = "web-key-1"
WEB_T = "33333333-3333-4333-8333-333333333333"
CLI_T = "44444444-4444-4444-8444-444444444444"


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def service():
    svc = AsyncMock()
    svc.repository = AsyncMock()
    svc.repository.get_attachment_original = AsyncMock(return_value=None)
    svc.repository.get_rendition = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def client(service):
    app = routes_app(service)
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
        resp = attachment_response(jpeg_bytes(), "beam.png")
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
# Tests of GET /api/attachments/{id}
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
# Tests of GET /api/attachments/{id}/rendition
# ---------------------------------------------------------------------------


class TestRenditionRoute:
    def test_viewable_rendition_is_served_inline_with_hardening(self, client, service) -> None:
        row = viewable_row()
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
        service.repository.get_rendition.return_value = viewable_row(
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
            viewable_row(skip_reason="reserved_format", mime_type="application/pdf"),
            viewable_row(copy_status="pending"),
            viewable_row(rendition_sha256=None),
            viewable_row(mime_type="application/pdf"),
            viewable_row(skip_reason="too_large"),
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
# Tests of the attachment-to-artifact route, POST /api/attachments/{id}/artifact
# ---------------------------------------------------------------------------


class TestWorkspaceArtifactRoute:
    def test_viewable_picture_is_copied_using_the_shared_helper(
        self, client, service, monkeypatch
    ) -> None:
        row = viewable_row()
        entry = {"entry_id": "e-1"}
        summary = {"attachment_id": ATT_ID, "filename": "beam.png"}
        copied = {"artifact_id": "artifact-1", "created": True, "attachment_id": ATT_ID}

        async def picture_for_artifact(actual_service, actual_id):
            assert actual_service is service
            assert actual_id == ATT_ID
            return row, entry, summary

        monkeypatch.setattr(routes, "_picture_for_workspace_artifact", picture_for_artifact)
        from osprey.services.ariel_search.attachments import artifacts

        monkeypatch.setattr(
            artifacts, "save_viewable_picture_as_artifact", lambda *args, **kwargs: copied
        )

        response = client.post(f"/api/attachments/{ATT_ID}/artifact")

        assert response.status_code == 200
        assert response.json() == copied

    @pytest.fixture
    def saved(self, tmp_path, monkeypatch):
        """A real artifact store behind the route, and a viewable picture to copy.

        The process environment names a CLI conversation with its pointer
        present and a different web key, as the ``osprey chat`` launcher's
        does; the browser route must ignore all of it.
        """
        from osprey.stores import artifact_store as artifact_store_module
        from osprey.stores.artifact_store import ArtifactStore

        root = tmp_path / "agent_data"
        store = ArtifactStore(workspace_root=root, auto_launch=False)
        monkeypatch.setattr(artifact_store_module, "_artifact_store", store)
        (root / "cli-transcripts").mkdir(parents=True)
        (root / "cli-transcripts" / "launch-1").write_text(CLI_T + "\n", encoding="utf-8")
        monkeypatch.setenv("OSPREY_TERMINAL_SESSION", "chat-launcher")
        monkeypatch.setenv("OSPREY_CLI_LAUNCH", "launch-1")
        monkeypatch.setenv("OSPREY_SESSION_ID", "launcher-env-key")
        monkeypatch.delenv("OSPREY_DISPATCH_RUN_ID", raising=False)
        monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)

        row = viewable_row()
        picture = (row, {"entry_id": "e-1"}, {"attachment_id": ATT_ID, "filename": "beam.png"})

        async def picture_for_artifact(_service, _id):
            return picture

        monkeypatch.setattr(routes, "_picture_for_workspace_artifact", picture_for_artifact)
        return store, row

    def test_a_keyed_save_belongs_to_the_keys_current_transcript(
        self, client, saved, monkeypatch
    ) -> None:
        from osprey.stores import artifact_scope
        from osprey.stores.artifact_store import owner_of
        from osprey.stores.session_ledger import EPOCH, Ledger, LedgerRow

        store, _row = saved
        ledger = Ledger(rows=(LedgerRow(WEB_KEY, WEB_T, EPOCH),))
        monkeypatch.setattr(artifact_scope, "read_ledger", lambda: (ledger, True))

        url = f"/api/attachments/{ATT_ID}/artifact"
        first = client.post(url, params={"session_key": WEB_KEY})
        second = client.post(url, params={"session_key": WEB_KEY})

        assert first.status_code == 200, first.text
        assert first.json()["created"] is True
        assert second.json()["created"] is False
        assert second.json()["artifact_id"] == first.json()["artifact_id"]
        [entry] = store.list_entries()
        assert entry.session_id == WEB_KEY
        assert owner_of(entry, ledger) == WEB_T

    def test_a_keyless_save_is_one_earlier_artifact(self, client, saved, monkeypatch) -> None:
        import hashlib

        from osprey.stores import artifact_scope
        from osprey.stores.artifact_store import owner_of
        from osprey.stores.session_ledger import Ledger

        store, row = saved
        monkeypatch.setattr(artifact_scope, "read_ledger", lambda: (Ledger(), True))
        # The launcher's CLI conversation already holds the same bytes.
        cli_copy = store.save_file(
            row["rendition_bytes"],
            "beam.jpg",
            "image",
            "beam.png (entry e-1)",
            tool_source="attachment_to_artifact",
            metadata={"sha256": hashlib.sha256(row["rendition_bytes"]).hexdigest()},
            session_id="cli:" + CLI_T,
        )

        url = f"/api/attachments/{ATT_ID}/artifact"
        first = client.post(url)
        second = client.post(url)

        assert first.status_code == 200, first.text
        assert first.json()["artifact_id"] != cli_copy.id
        assert first.json()["created"] is True
        assert second.json()["artifact_id"] == first.json()["artifact_id"]
        assert second.json()["created"] is False
        entry = store.get_entry(first.json()["artifact_id"])
        assert entry.session_id == ""
        assert "transcript_id" not in entry.metadata
        assert owner_of(entry, Ledger()) is None
        assert len(store.list_entries()) == 2

    def test_malformed_id_is_404_before_lookup(self, client, service, monkeypatch) -> None:
        async def should_not_run(*_args):
            raise AssertionError("a malformed attachment id must not be looked up")

        monkeypatch.setattr(routes, "_picture_for_workspace_artifact", should_not_run)

        response = client.post("/api/attachments/not-an-id/artifact")

        assert response.status_code == 404
        assert response.json()["detail"] == "picture not available"
        service.repository.get_rendition.assert_not_called()

    def test_unavailable_picture_is_404(self, client, monkeypatch) -> None:
        async def unavailable(*_args):
            raise routes.HTTPException(status_code=404, detail="picture not available")

        monkeypatch.setattr(routes, "_picture_for_workspace_artifact", unavailable)

        response = client.post(f"/api/attachments/{ATT_ID}/artifact")

        assert response.status_code == 404
        assert response.json()["detail"] == "picture not available"


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

    service.repository.get_rendition.return_value = viewable_row()
    service.repository.get_attachment_original.return_value = {
        "filename": "beam.png",
        "mime_type": "image/png",
        "data": _png(),
    }

    assert client.get(f"/api/attachments/{ATT_ID}/rendition").status_code == 200
    assert client.get(f"/api/attachments/{ATT_ID}").status_code == 200

    service.repository.get_rendition.return_value = None
    assert client.get(f"/api/attachments/{ATT_ID}/rendition").status_code == 404
