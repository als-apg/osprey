"""Tests for ARIEL draft file store and API routes."""

import json
import os
import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from osprey.interfaces.ariel.api.drafts import (
    DRAFT_TTL_SECONDS,
    draft_router,
    read_draft,
    write_draft,
)

# ---------------------------------------------------------------------------
# Draft file store tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_and_read_draft(tmp_path, monkeypatch):
    """Write a draft file and read it back."""
    import osprey.interfaces.ariel.api.drafts as drafts_mod

    monkeypatch.setattr(drafts_mod, "DRAFTS_DIR", tmp_path / "drafts")

    draft_id = "draft-abc123"
    data = {
        "draft_id": draft_id,
        "subject": "Test subject",
        "details": "Test details",
        "author": "Alice",
        "logbook": "Operations",
        "shift": "Day",
        "tags": ["test"],
    }
    write_draft(draft_id, data)

    result = read_draft(draft_id)
    assert result is not None
    assert result["subject"] == "Test subject"
    assert result["author"] == "Alice"
    assert result["tags"] == ["test"]


@pytest.mark.unit
def test_get_nonexistent_returns_none(tmp_path, monkeypatch):
    """Reading a nonexistent draft returns None."""
    import osprey.interfaces.ariel.api.drafts as drafts_mod

    monkeypatch.setattr(drafts_mod, "DRAFTS_DIR", tmp_path / "drafts")

    result = read_draft("draft-nonexistent")
    assert result is None


@pytest.mark.unit
def test_expired_drafts_cleaned_up(tmp_path, monkeypatch):
    """Drafts older than TTL are deleted on cleanup."""
    import osprey.interfaces.ariel.api.drafts as drafts_mod

    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    monkeypatch.setattr(drafts_mod, "DRAFTS_DIR", drafts_dir)

    # Write a draft file and backdate its mtime
    draft_id = "draft-expired1"
    filepath = drafts_dir / f"{draft_id}.json"
    filepath.write_text(json.dumps({"draft_id": draft_id, "subject": "Old"}))
    old_time = time.time() - DRAFT_TTL_SECONDS - 60
    os.utime(filepath, (old_time, old_time))

    # Write a fresh draft
    fresh_id = "draft-fresh1"
    fresh_path = drafts_dir / f"{fresh_id}.json"
    fresh_path.write_text(json.dumps({"draft_id": fresh_id, "subject": "New"}))

    # Trigger cleanup
    drafts_mod._cleanup_expired_drafts()

    assert not filepath.exists(), "Expired draft should be deleted"
    assert fresh_path.exists(), "Fresh draft should survive"


# ---------------------------------------------------------------------------
# Draft API route tests
# ---------------------------------------------------------------------------


@pytest.fixture
def draft_app(tmp_path, monkeypatch):
    """Create a minimal FastAPI app with the draft router."""
    import osprey.interfaces.ariel.api.drafts as drafts_mod

    monkeypatch.setattr(drafts_mod, "DRAFTS_DIR", tmp_path / "drafts")

    app = FastAPI()
    app.include_router(draft_router)
    return app


@pytest.mark.unit
async def test_api_create_and_get_draft(draft_app):
    """POST /api/drafts then GET /api/drafts/{id} returns the draft."""
    transport = ASGITransport(app=draft_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create
        resp = await client.post(
            "/api/drafts",
            json={"subject": "Test", "details": "Details", "author": "Bob"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["draft_id"].startswith("draft-")

        # Get
        resp2 = await client.get(f"/api/drafts/{body['draft_id']}")
        assert resp2.status_code == 200
        draft = resp2.json()
        assert draft["subject"] == "Test"
        assert draft["details"] == "Details"
        assert draft["author"] == "Bob"


@pytest.mark.unit
async def test_api_get_nonexistent_returns_404(draft_app):
    """GET /api/drafts/{bad_id} returns 404."""
    transport = ASGITransport(app=draft_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/drafts/draft-doesnotexist")
        assert resp.status_code == 404
