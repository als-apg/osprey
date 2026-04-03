"""Channel Finder Pending Review REST API.

Exposes CRUD operations on the PendingReviewStore for browsing,
approving, and dismissing agent-captured searches awaiting operator review.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

_MIME_MAP = {
    ".md": "text/markdown",
    ".html": "text/html",
    ".json": "application/json",
    ".txt": "text/plain",
    ".csv": "text/csv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pending_store(request: Request):
    """Get the pending review store from app state, or raise 404."""
    from osprey.services.channel_finder.feedback.pending_store import PendingReviewStore

    store: PendingReviewStore | None = getattr(request.app.state, "pending_review_store", None)
    if store is None:
        raise HTTPException(404, "Pending review store not available")
    return store


def _get_feedback_store(request: Request):
    """Get the feedback store from app state, or raise 404."""
    from osprey.services.channel_finder.feedback.store import FeedbackStore

    store: FeedbackStore | None = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(404, "Feedback store not available")
    return store


def _resolve_artifact(item: dict, project_cwd: str) -> dict | None:
    """Find the artifact best matching this pending review item's channels.

    Returns ``{"id": ..., "title": ..., "filename": ...}`` or ``None``.
    """
    try:
        artifacts_path = Path(project_cwd) / "_agent_data" / "artifacts" / "artifacts.json"
        if not artifacts_path.is_file():
            return None

        artifacts_data = json.loads(artifacts_path.read_text())
        entries = (
            artifacts_data
            if isinstance(artifacts_data, list)
            else artifacts_data.get("artifacts", [])
        )

        # Parse channel names from the item's tool_response
        item_channels: set[str] = set()
        try:
            raw = item.get("tool_response", "")
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            for ch in parsed.get("channels", []):
                if isinstance(ch, str):
                    item_channels.add(ch)
                elif isinstance(ch, dict):
                    item_channels.add(ch.get("name", ""))
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        if not item_channels:
            return None

        best: dict | None = None
        best_overlap = 0
        best_created = ""

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("source_agent") != "channel-finder":
                continue
            if entry.get("category") != "channel_finder":
                continue

            meta = entry.get("metadata", {})
            entry_ids = set(meta.get("entry_ids", []))
            overlap = len(item_channels & entry_ids)
            created = entry.get("created_at", "")

            if overlap > best_overlap or (overlap == best_overlap and created > best_created):
                best_overlap = overlap
                best_created = created
                best = entry

        if best is None or best_overlap == 0:
            return None

        return {
            "id": best.get("id", ""),
            "title": best.get("title", ""),
            "filename": best.get("filename", ""),
        }
    except Exception:
        return None


def _enrich_item(item: dict, project_cwd: str) -> dict:
    """Add artifact info to a pending review item (in-place)."""
    artifact = _resolve_artifact(item, project_cwd)
    if artifact:
        item["artifact"] = artifact
    return item


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ApproveRequest(BaseModel):
    """Optional overrides when approving a pending review item."""

    facility: str | None = None
    selections: dict[str, Any] | None = None
    channel_count: int | None = None


class RejectRequest(BaseModel):
    """Optional fields when recording a rejection as a failure in FeedbackStore."""

    reason: str = "operator rejected"
    partial_selections: dict[str, Any] | None = None
    facility: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/pending-reviews/status")
async def pending_reviews_status(request: Request):
    """Check pending review store availability. Always returns 200."""
    store = getattr(request.app.state, "pending_review_store", None)
    if store is None:
        return {"available": False, "item_count": 0}
    return {
        "available": True,
        "item_count": len(store.list_items()),
    }


@router.get("/pending-reviews")
async def pending_reviews_list(request: Request):
    """List all pending review items (newest first)."""
    store = _get_pending_store(request)
    project_cwd = getattr(request.app.state, "project_cwd", "")
    items = store.list_items()
    for item in items:
        _enrich_item(item, project_cwd)
    return {"items": items}


@router.get("/pending-reviews/{item_id}")
async def pending_reviews_detail(item_id: str, request: Request):
    """Get a single pending review item."""
    store = _get_pending_store(request)
    item = store.get_item(item_id)
    if item is None:
        raise HTTPException(404, f"Pending review item not found: {item_id}")
    project_cwd = getattr(request.app.state, "project_cwd", "")
    _enrich_item(item, project_cwd)
    return item


@router.post("/pending-reviews/{item_id}/approve")
async def pending_reviews_approve(
    item_id: str, request: Request, body: ApproveRequest | None = None
):
    """Approve a pending review item, promoting it to the FeedbackStore.

    The operator can optionally override facility, selections, or channel_count
    before promotion. Deletes the item from the pending store after success.
    """
    pending_store = _get_pending_store(request)
    feedback_store = _get_feedback_store(request)

    item = pending_store.get_item(item_id)
    if item is None:
        raise HTTPException(404, f"Pending review item not found: {item_id}")

    # Apply overrides from request body
    facility = item.get("facility", "") or getattr(request.app.state, "facility_name", "")
    selections = item.get("selections", {})
    channel_count = item.get("channel_count", 0)
    query = item.get("agent_task", "") or item.get("query", "")

    if body:
        if body.facility is not None:
            facility = body.facility
        if body.selections is not None:
            selections = body.selections
        if body.channel_count is not None:
            channel_count = body.channel_count

    # Promote to feedback store via add_manual_entry
    key = feedback_store.add_manual_entry(
        query=query,
        facility=facility,
        entry_type="success",
        selections=selections,
        channel_count=channel_count,
    )

    # Remove from pending store
    pending_store.delete(item_id)

    return {"ok": True, "feedback_key": key}


@router.post("/pending-reviews/{item_id}/reject")
async def pending_reviews_reject(item_id: str, request: Request, body: RejectRequest | None = None):
    """Reject a pending review item, recording it as a failure in FeedbackStore.

    This is distinct from DELETE (dismiss): it actively records the bad path
    so the agent learns to avoid it. Deletes from pending store after recording.
    """
    pending_store = _get_pending_store(request)
    feedback_store = _get_feedback_store(request)

    item = pending_store.get_item(item_id)
    if item is None:
        raise HTTPException(404, f"Pending review item not found: {item_id}")

    reason = "operator rejected"
    partial_selections = item.get("selections", {})
    facility = item.get("facility", "") or getattr(request.app.state, "facility_name", "")
    query = item.get("agent_task", "") or item.get("query", "")

    if body:
        if body.reason is not None:
            reason = body.reason
        if body.partial_selections is not None:
            partial_selections = body.partial_selections
        if body.facility is not None:
            facility = body.facility

    feedback_store.record_failure(
        query=query,
        facility=facility,
        partial_selections=partial_selections,
        reason=reason,
    )

    pending_store.delete(item_id)

    return {"ok": True}


@router.delete("/pending-reviews/{item_id}")
async def pending_reviews_dismiss(item_id: str, request: Request):
    """Dismiss (delete) a single pending review item."""
    store = _get_pending_store(request)
    if not store.delete(item_id):
        raise HTTPException(404, f"Pending review item not found: {item_id}")
    return {"ok": True}


@router.delete("/pending-reviews")
async def pending_reviews_clear(request: Request, confirm: bool = False):
    """Clear all pending review items. Requires confirm=true."""
    store = _get_pending_store(request)
    if not confirm:
        raise HTTPException(400, "Must pass confirm=true to clear all pending reviews")
    store.clear()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Artifact file serving
# ---------------------------------------------------------------------------


@router.get("/artifacts/{filename}")
async def serve_artifact(filename: str, request: Request):
    """Serve an artifact file from the workspace artifacts directory.

    Guards against path traversal by rejecting filenames containing
    ``/`` or ``..``.
    """
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    project_cwd = getattr(request.app.state, "project_cwd", "")
    artifact_path = Path(project_cwd) / "_agent_data" / "artifacts" / filename

    if not artifact_path.is_file():
        raise HTTPException(404, "Artifact not found")

    suffix = artifact_path.suffix.lower()
    media_type = _MIME_MAP.get(suffix, "application/octet-stream")

    if media_type.startswith("text/"):
        content = artifact_path.read_text(errors="replace")
        return PlainTextResponse(content, media_type=media_type)

    return FileResponse(artifact_path, media_type=media_type)
