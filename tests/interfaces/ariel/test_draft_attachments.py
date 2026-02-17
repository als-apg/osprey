"""Tests for draft attachment serving in ARIEL Draft API."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.interfaces.ariel.api.drafts as drafts_mod
from osprey.interfaces.ariel.api.drafts import draft_router, read_draft, write_draft


@pytest.fixture
def draft_client(tmp_path, monkeypatch):
    """Create a test client with drafts_dir pointing to tmp_path."""
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    monkeypatch.setattr(drafts_mod, "DRAFTS_DIR", drafts_dir)

    app = FastAPI()
    app.include_router(draft_router)
    return TestClient(app), drafts_dir


@pytest.mark.unit
def test_get_draft_attachment_success(draft_client):
    """Serving a listed attachment returns correct content and content-type."""
    client, drafts_dir = draft_client

    # Create a fake attachment file
    attachment_dir = drafts_dir / "staging"
    attachment_dir.mkdir()
    img_file = attachment_dir / "plot.png"
    img_file.write_bytes(b"\x89PNG fake image data")

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
    assert resp.content == b"\x89PNG fake image data"
    assert resp.headers["content-type"] == "image/png"


@pytest.mark.unit
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


@pytest.mark.unit
def test_get_draft_attachment_draft_not_found(draft_client):
    """Missing draft returns 404."""
    client, _ = draft_client

    resp = client.get("/api/drafts/draft-missing/attachments/file.png")
    assert resp.status_code == 404


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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
