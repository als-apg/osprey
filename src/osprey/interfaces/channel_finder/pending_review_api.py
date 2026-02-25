"""Channel Finder Pending Review REST API.

Exposes CRUD operations on the PendingReviewStore for browsing,
approving, and dismissing agent-captured searches awaiting operator review.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pending_store(request: Request):
    """Get the pending review store from app state, or raise 404."""
    from osprey.services.channel_finder.feedback.pending_store import PendingReviewStore

    store: PendingReviewStore | None = getattr(
        request.app.state, "pending_review_store", None
    )
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


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ApproveRequest(BaseModel):
    """Optional overrides when approving a pending review item."""

    facility: str | None = None
    selections: dict[str, Any] | None = None
    channel_count: int | None = None


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
    return {"items": store.list_items()}


@router.get("/pending-reviews/{item_id}")
async def pending_reviews_detail(item_id: str, request: Request):
    """Get a single pending review item."""
    store = _get_pending_store(request)
    item = store.get_item(item_id)
    if item is None:
        raise HTTPException(404, f"Pending review item not found: {item_id}")
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
    facility = item.get("facility", "")
    selections = item.get("selections", {})
    channel_count = item.get("channel_count", 0)
    query = item.get("query", "")

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
