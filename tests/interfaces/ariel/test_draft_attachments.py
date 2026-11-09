"""Tests for draft attachment serving in ARIEL Draft API."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.interfaces.ariel.api.drafts as drafts_mod
from osprey.interfaces.ariel.api.drafts import draft_router, write_draft
from osprey.services.ariel_search.database.repository import SchemaFacts
from tests.interfaces.ariel._routes_app import routes_app

#: The 8-byte PNG signature followed by an IHDR chunk header.
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 24
PDF = b"%PDF-1.4\n%draft pdf body\n%%EOF\n"


@pytest.fixture(autouse=True)
def _no_entry_fields():
    """The route double declares no entry fields, so no registry lookup is made."""
    with patch(
        "osprey.services.ariel_search.entry_fields.entry_field_descriptors",
        return_value=[],
    ):
        yield


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
        "draft-000000000001",
        {
            "draft_id": "draft-000000000001",
            "subject": "Test",
            "details": "Details",
            "attachment_paths": [str(img_file)],
        },
    )

    resp = client.get("/api/drafts/draft-000000000001/attachments/plot.png")
    assert resp.status_code == 200
    assert resp.content == PNG
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["content-disposition"] == "inline; filename*=UTF-8''plot.png"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["content-security-policy"] == "sandbox; default-src 'none'"


@pytest.mark.parametrize(
    ("draft_id", "attachment_path", "filename"),
    [
        # Requesting an attachment not listed in draft.
        pytest.param(
            "draft-000000000002", "/some/other/file.png", "unlisted.png", id="not-in-draft"
        ),
        # Attachment listed in draft but missing on disk.
        pytest.param(
            "draft-000000000005",
            "/nonexistent/path/image.png",
            "image.png",
            id="file-missing-on-disk",
        ),
    ],
)
def test_get_draft_attachment_not_servable_returns_404(
    draft_client, draft_id, attachment_path, filename
):
    client, drafts_dir = draft_client

    write_draft(
        draft_id,
        {
            "draft_id": draft_id,
            "subject": "Test",
            "details": "Details",
            "attachment_paths": [attachment_path],
        },
    )

    resp = client.get(f"/api/drafts/{draft_id}/attachments/{filename}")
    assert resp.status_code == 404


def test_get_draft_attachment_draft_not_found(draft_client):
    """Missing draft returns 404."""
    client, _ = draft_client

    resp = client.get("/api/drafts/draft-0000000dead0/attachments/file.png")
    assert resp.status_code == 404


def test_draft_response_includes_attachment_paths(draft_client):
    """DraftResponse schema includes attachment_paths field."""
    client, _ = draft_client

    write_draft(
        "draft-000000000003",
        {
            "draft_id": "draft-000000000003",
            "subject": "Test",
            "details": "Details",
            "attachment_paths": ["/path/to/image.png"],
        },
    )

    resp = client.get("/api/drafts/draft-000000000003")
    assert resp.status_code == 200
    data = resp.json()
    assert data["attachment_paths"] == ["/path/to/image.png"]


def test_draft_response_without_attachment_paths(draft_client):
    """DraftResponse without attachment_paths returns null for the field."""
    client, _ = draft_client

    write_draft(
        "draft-000000000004",
        {
            "draft_id": "draft-000000000004",
            "subject": "Test",
            "details": "Details",
        },
    )

    resp = client.get("/api/drafts/draft-000000000004")
    assert resp.status_code == 200
    data = resp.json()
    assert data["attachment_paths"] is None


def test_draft_response_includes_fields(draft_client):
    """GET /drafts/{id} returns the draft's declared entry-field values."""
    client, _ = draft_client

    write_draft(
        "draft-f1e1d5000001",
        {
            "draft_id": "draft-f1e1d5000001",
            "subject": "Test",
            "details": "Details",
            "fields": {"book": "physics", "beam_current": 401.5, "beam_on": False},
            "metadata": {"session_metadata": {"id": "s-1"}},
        },
    )

    resp = client.get("/api/drafts/draft-f1e1d5000001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fields"] == {"book": "physics", "beam_current": 401.5, "beam_on": False}
    assert data["metadata"] == {"session_metadata": {"id": "s-1"}}


def test_draft_response_without_fields(draft_client):
    """A draft without declared values returns null for fields."""
    client, _ = draft_client

    write_draft(
        "draft-f1e1d5000002", {"draft_id": "draft-f1e1d5000002", "subject": "T", "details": "D"}
    )

    resp = client.get("/api/drafts/draft-f1e1d5000002")
    assert resp.status_code == 200
    assert resp.json()["fields"] is None


def test_create_draft_does_not_store_fields(draft_client):
    """POST /drafts ignores a fields key; only the agent's draft writer sets it."""
    client, _ = draft_client

    resp = client.post(
        "/api/drafts",
        json={"subject": "T", "details": "D", "fields": {"book": "physics"}},
    )
    assert resp.status_code == 200
    draft_id = resp.json()["draft_id"]
    assert "fields" not in drafts_mod.read_draft(draft_id)
    assert client.get(f"/api/drafts/{draft_id}").json()["fields"] is None


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
    _write_draft_with(drafts_dir, "draft-000000000006", "plot.png", html)

    resp = client.get("/api/drafts/draft-000000000006/attachments/plot.png")

    assert resp.status_code == 200
    assert resp.content == html
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["content-disposition"] == "attachment; filename*=UTF-8''plot.png"
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_get_draft_attachment_pdf_is_a_download(draft_client):
    """A draft PDF is served as an octet-stream download."""
    client, drafts_dir = draft_client
    _write_draft_with(drafts_dir, "draft-000000000007", "report.pdf", PDF)

    resp = client.get("/api/drafts/draft-000000000007/attachments/report.pdf")

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
            "source_system": "OSPREY Logbook Web",
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
    _write_draft_with(drafts_dir, "draft-000000000008", "report.pdf", PDF)
    served = draft_http.get("/api/drafts/draft-000000000008/attachments/report.pdf")
    blob_type = served.headers["content-type"]
    assert blob_type == "application/octet-stream"

    service = _upload_service(has_copy_state)
    app = routes_app(service)
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
    app = routes_app(service)

    response = TestClient(app).post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=[("files", ("notes.bin", b"abc", "text/plain"))],
    )

    assert response.status_code == 200, response.text
    assert service.repository.store_attachment.call_args.kwargs["mime_type"] == "text/plain"
