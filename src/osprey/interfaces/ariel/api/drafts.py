"""ARIEL Draft Entry API — filesystem-based draft store.

Allows Claude Code (via MCP tool) to pre-fill the web form so a human
can review, edit, and submit manually. Drafts are stored as JSON files
in osprey-workspace/drafts/ and expire after 1 hour.
"""

from __future__ import annotations

import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

DRAFTS_DIR = Path.cwd() / "osprey-workspace" / "drafts"

draft_router = APIRouter(prefix="/api")


class DraftCreateRequest(BaseModel):
    """Request body for creating a draft."""

    subject: str
    details: str
    author: str | None = None
    logbook: str | None = None
    shift: str | None = None
    tags: list[str] | None = None


class DraftCreateResponse(BaseModel):
    """Response after creating a draft."""

    draft_id: str
    message: str


class DraftResponse(BaseModel):
    """Full draft data returned to the frontend."""

    draft_id: str
    subject: str
    details: str
    author: str | None = None
    logbook: str | None = None
    shift: str | None = None
    tags: list[str] | None = None
    attachment_paths: list[str] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRAFT_TTL_SECONDS = 3600  # 1 hour


def _ensure_drafts_dir() -> Path:
    """Ensure the drafts directory exists and return its path."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    return DRAFTS_DIR


def _cleanup_expired_drafts() -> None:
    """Delete draft files older than DRAFT_TTL_SECONDS."""
    if not DRAFTS_DIR.exists():
        return
    now = time.time()
    for path in DRAFTS_DIR.glob("draft-*.json"):
        if now - path.stat().st_mtime > DRAFT_TTL_SECONDS:
            path.unlink(missing_ok=True)


def generate_draft_id() -> str:
    """Generate a unique draft ID."""
    return f"draft-{uuid.uuid4().hex[:12]}"


def write_draft(draft_id: str, data: dict[str, Any]) -> Path:
    """Write a draft JSON file to the drafts directory."""
    drafts_dir = _ensure_drafts_dir()
    filepath = drafts_dir / f"{draft_id}.json"
    filepath.write_text(json.dumps(data, indent=2))
    return filepath


def read_draft(draft_id: str) -> dict[str, Any] | None:
    """Read a draft JSON file. Returns None if not found."""
    filepath = DRAFTS_DIR / f"{draft_id}.json"
    if not filepath.exists():
        return None
    return json.loads(filepath.read_text())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@draft_router.post("/drafts", response_model=DraftCreateResponse)
async def create_draft(req: DraftCreateRequest) -> DraftCreateResponse:
    """Create a new draft entry (pre-fill data for the web form)."""
    draft_id = generate_draft_id()
    data = {
        "draft_id": draft_id,
        "subject": req.subject,
        "details": req.details,
        "author": req.author,
        "logbook": req.logbook,
        "shift": req.shift,
        "tags": req.tags or [],
    }
    write_draft(draft_id, data)
    return DraftCreateResponse(
        draft_id=draft_id,
        message=f"Draft {draft_id} created",
    )


@draft_router.get("/drafts/{draft_id}", response_model=DraftResponse)
async def get_draft(draft_id: str) -> DraftResponse:
    """Retrieve a draft by ID. Triggers cleanup of expired drafts."""
    _cleanup_expired_drafts()

    data = read_draft(draft_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")

    return DraftResponse(**data)


@draft_router.get("/drafts/{draft_id}/attachments/{filename}")
async def get_draft_attachment(draft_id: str, filename: str) -> Response:
    """Serve a staged attachment file from a draft.

    Only files listed in the draft's ``attachment_paths`` are served
    (prevents path-traversal attacks).
    """
    data = read_draft(draft_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")

    attachment_paths = data.get("attachment_paths") or []
    # Find the path whose basename matches the requested filename
    matched_path: str | None = None
    for p in attachment_paths:
        if Path(p).name == filename:
            matched_path = p
            break

    if matched_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Attachment '{filename}' not found in draft {draft_id}",
        )

    file_path = Path(matched_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Attachment file not found on disk")

    content_type, _ = mimetypes.guess_type(str(file_path))
    return Response(
        content=file_path.read_bytes(),
        media_type=content_type or "application/octet-stream",
    )
