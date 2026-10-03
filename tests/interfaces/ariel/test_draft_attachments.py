"""Tests for draft attachment serving in ARIEL Draft API."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.interfaces.ariel.api.drafts as drafts_mod
from osprey.interfaces.ariel.api import routes
from osprey.interfaces.ariel.api.drafts import draft_router, write_draft
from osprey.services.ariel_search.database.repository import SchemaFacts

#: The 8-byte PNG signature followed by an IHDR chunk header.
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 24
PDF = b"%PDF-1.4\n%draft pdf body\n%%EOF\n"


@pytest.fixture
def draft_client(tmp_path, monkeypatch):
    """Create a test client with drafts_dir pointing to tmp_path."""
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    monkeypatch.setattr(drafts_mod, "_drafts_dir", lambda: drafts_dir)

    app = FastAPI()
    app.include_router(draft_router)
    return TestClient(app), drafts_dir


def test_get_draft_attachment_success(draft_client):
    """Serving a listed attachment returns correct content and content-type."""
    client, drafts_dir = draft_client

    # Create a fake attachment file
    attachment_dir = drafts_dir / "staging"
    attachment_dir.mkdir()
    img_file = attachment_dir / "plot.png"
    img_file.write_bytes(PNG)

    # Write draft with attachment_paths
    write_draft(
        "draft-test001",
        {
            "draft_id": "draft-test001",
            "subject": "Test",
            "details": "Details",
            "attachment_paths": [str(img_file)],
        },
    )

    resp = client.get("/api/drafts/draft-test001/attachments/plot.png")
    assert resp.status_code == 200
    assert resp.content == PNG
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["content-disposition"] == "inline; filename*=UTF-8''plot.png"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["content-security-policy"] == "sandbox; default-src 'none'"


def test_get_draft_attachment_not_in_draft(draft_client):
    """Requesting an attachment not listed in draft returns 404."""
    client, drafts_dir = draft_client

    write_draft(
        "draft-test002",
        {
            "draft_id": "draft-test002",
            "subject": "Test",
            "details": "Details",
            "attachment_paths": ["/some/other/file.png"],
        },
    )

    resp = client.get("/api/drafts/draft-test002/attachments/unlisted.png")
    assert resp.status_code == 404


def test_get_draft_attachment_draft_not_found(draft_client):
    """Missing draft returns 404."""
    client, _ = draft_client

    resp = client.get("/api/drafts/draft-missing/attachments/file.png")
    assert resp.status_code == 404


def test_draft_response_includes_attachment_paths(draft_client):
    """DraftResponse schema includes attachment_paths field."""
    client, _ = draft_client

    write_draft(
        "draft-test003",
        {
            "draft_id": "draft-test003",
            "subject": "Test",
            "details": "Details",
            "attachment_paths": ["/path/to/image.png"],
        },
    )

    resp = client.get("/api/drafts/draft-test003")
    assert resp.status_code == 200
    data = resp.json()
    assert data["attachment_paths"] == ["/path/to/image.png"]


def test_draft_response_without_attachment_paths(draft_client):
    """DraftResponse without attachment_paths returns null for the field."""
    client, _ = draft_client

    write_draft(
        "draft-test004",
        {
            "draft_id": "draft-test004",
            "subject": "Test",
            "details": "Details",
        },
    )

    resp = client.get("/api/drafts/draft-test004")
    assert resp.status_code == 200
    data = resp.json()
    assert data["attachment_paths"] is None


def test_get_draft_attachment_file_missing_on_disk(draft_client):
    """Attachment listed in draft but missing on disk returns 404."""
    client, _ = draft_client

    write_draft(
        "draft-test005",
        {
            "draft_id": "draft-test005",
            "subject": "Test",
            "details": "Details",
            "attachment_paths": ["/nonexistent/path/image.png"],
        },
    )

    resp = client.get("/api/drafts/draft-test005/attachments/image.png")
    assert resp.status_code == 404


def _write_draft_with(drafts_dir, draft_id: str, name: str, data: bytes) -> None:
    staging = drafts_dir / "staging"
    staging.mkdir(exist_ok=True)
    path = staging / name
    path.write_bytes(data)
    write_draft(
        draft_id,
        {
            "draft_id": draft_id,
            "subject": "Test",
            "details": "Details",
            "attachment_paths": [str(path)],
        },
    )


def test_get_draft_attachment_sniffs_instead_of_guessing(draft_client):
    """A web page named like a picture is downloaded, never rendered inline."""
    client, drafts_dir = draft_client
    html = b"<!doctype html><html><body>login</body></html>"
    _write_draft_with(drafts_dir, "draft-test006", "plot.png", html)

    resp = client.get("/api/drafts/draft-test006/attachments/plot.png")

    assert resp.status_code == 200
    assert resp.content == html
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["content-disposition"] == "attachment; filename*=UTF-8''plot.png"
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_get_draft_attachment_pdf_is_a_download(draft_client):
    """A draft PDF is served as an octet-stream download."""
    client, drafts_dir = draft_client
    _write_draft_with(drafts_dir, "draft-test007", "report.pdf", PDF)

    resp = client.get("/api/drafts/draft-test007/attachments/report.pdf")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["content-disposition"].startswith("attachment;")


# ---------------------------------------------------------------------------
# Draft PDF loaded by the form, then uploaded: stored as application/pdf
# ---------------------------------------------------------------------------


def _upload_service(has_copy_state: bool) -> AsyncMock:
    service = AsyncMock()
    service.create_entry = AsyncMock(side_effect=NotImplementedError("read-only"))
    service.repository = AsyncMock()
    service.repository.schema_facts = AsyncMock(
        return_value=SchemaFacts(has_copy_state, has_copy_state)
    )
    service.repository.get_entry = AsyncMock(
        return_value={
            "entry_id": "ariel-xyz",
            "source_system": "ARIEL Web",
            "timestamp": datetime.now(),
            "author": "Anonymous",
            "raw_text": "Test\n\nBody",
            "attachments": [],
            "metadata": {},
        }
    )
    return service


@pytest.mark.parametrize("has_copy_state", [True, False], ids=["copy-state", "no-copy-state"])
def test_draft_pdf_uploaded_as_octet_stream_is_stored_as_pdf(
    draft_client, has_copy_state, monkeypatch
):
    """The form re-uploads the draft blob with its served type; the name restores the type."""
    from osprey.services.ariel_search.attachments import prepare as prepare_module

    async def unavailable(*_args, **_kwargs):
        raise prepare_module.RenderUnavailable("no worker in unit tests")

    monkeypatch.setattr(prepare_module, "prepare_picture", unavailable)

    draft_http, drafts_dir = draft_client
    _write_draft_with(drafts_dir, "draft-test008", "report.pdf", PDF)
    served = draft_http.get("/api/drafts/draft-test008/attachments/report.pdf")
    blob_type = served.headers["content-type"]
    assert blob_type == "application/octet-stream"

    service = _upload_service(has_copy_state)
    app = FastAPI()
    app.include_router(routes.router)
    app.state.ariel_service = service
    app.state.config_panel_enabled = True
    upload_http = TestClient(app)

    response = upload_http.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=[("files", ("report.pdf", served.content, blob_type))],
    )

    assert response.status_code == 200, response.text
    assert response.json()["attachment_count"] == 1
    if has_copy_state:
        kwargs = service.repository.insert_native_attachment.call_args.kwargs
    else:
        kwargs = service.repository.store_attachment.call_args.kwargs
    assert kwargs["mime_type"] == "application/pdf"
    assert kwargs["data"] == PDF
    linked = service.repository.upsert_entry.call_args_list[-1].args[0]
    assert linked["attachments"][0]["type"] == "application/pdf"


def test_upload_keeps_a_declared_specific_type():
    """Only octet-stream counts as absent; a specific declared type is kept."""
    service = _upload_service(False)
    app = FastAPI()
    app.include_router(routes.router)
    app.state.ariel_service = service
    app.state.config_panel_enabled = True

    response = TestClient(app).post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=[("files", ("notes.bin", b"abc", "text/plain"))],
    )

    assert response.status_code == 200, response.text
    assert service.repository.store_attachment.call_args.kwargs["mime_type"] == "text/plain"
