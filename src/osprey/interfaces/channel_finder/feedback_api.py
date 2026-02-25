"""Channel Finder Feedback REST API.

Exposes CRUD operations on the FeedbackStore for browsing and managing
feedback entries (successful/failed navigation paths) used by the
hierarchical pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_store(request: Request):
    """Get the feedback store from app state, or raise 404."""
    from osprey.services.channel_finder.feedback.store import FeedbackStore

    store: FeedbackStore | None = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(404, "Feedback store not available")
    return store


def _validate_record_type(record_type: str) -> None:
    """Validate record_type path parameter."""
    if record_type not in ("successes", "failures"):
        raise HTTPException(
            400, f"Invalid record_type: {record_type}. Must be 'successes' or 'failures'"
        )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AddEntryRequest(BaseModel):
    """Request body for adding a manual feedback entry."""

    query: str
    facility: str
    entry_type: str = "success"
    selections: dict[str, Any] = {}
    channel_count: int = 0
    reason: str = ""


class EditRecordRequest(BaseModel):
    """Request body for editing a single feedback record."""

    expected_timestamp: str
    selections: dict[str, Any] | None = None
    channel_count: int | None = None
    partial_selections: dict[str, Any] | None = None
    reason: str | None = None


class DeleteRecordRequest(BaseModel):
    """Request body for deleting a single feedback record."""

    expected_timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/feedback/status")
async def feedback_status(request: Request):
    """Check feedback store availability. Always returns 200."""
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        return {"available": False, "entry_count": 0, "store_path": None}
    return {
        "available": True,
        "entry_count": len(store.list_keys()),
        "store_path": str(store._path),
    }


@router.get("/feedback/export")
async def feedback_export(request: Request):
    """Export full feedback store as JSON download."""
    store = _get_store(request)
    data = store.export_data()
    content = json.dumps(data, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=feedback_export.json"},
    )


@router.get("/feedback/{key}")
async def feedback_detail(key: str, request: Request):
    """Get full bucket for a feedback entry."""
    store = _get_store(request)
    entry = store.get_entry(key)
    if entry is None:
        raise HTTPException(404, f"Feedback entry not found: {key}")
    return entry


@router.get("/feedback")
async def feedback_list(request: Request):
    """List all feedback entries with summary metadata."""
    store = _get_store(request)
    return {"entries": store.list_keys()}


@router.post("/feedback")
async def feedback_add(request: Request, body: AddEntryRequest):
    """Add a manual feedback entry."""
    store = _get_store(request)
    if body.entry_type not in ("success", "failure"):
        raise HTTPException(
            400, f"Invalid entry_type: {body.entry_type}. Must be 'success' or 'failure'"
        )
    key = store.add_manual_entry(
        query=body.query,
        facility=body.facility,
        entry_type=body.entry_type,
        selections=body.selections,
        channel_count=body.channel_count,
        reason=body.reason,
    )
    return {"key": key}


@router.put("/feedback/{key}/{record_type}/{index}")
async def feedback_edit(
    key: str, record_type: str, index: int, request: Request, body: EditRecordRequest
):
    """Edit a single feedback record. Returns 409 on stale timestamp."""
    store = _get_store(request)
    _validate_record_type(record_type)
    try:
        fields: dict[str, Any] = {}
        if body.selections is not None:
            fields["selections"] = body.selections
        if body.channel_count is not None:
            fields["channel_count"] = body.channel_count
        if body.partial_selections is not None:
            fields["partial_selections"] = body.partial_selections
        if body.reason is not None:
            fields["reason"] = body.reason

        store.update_record(key, record_type, index, body.expected_timestamp, **fields)
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/feedback/{key}/{record_type}/{index}")
async def feedback_delete_record(
    key: str, record_type: str, index: int, request: Request, body: DeleteRecordRequest
):
    """Delete a single feedback record. Returns 409 on stale timestamp."""
    store = _get_store(request)
    _validate_record_type(record_type)
    try:
        store.delete_record(key, record_type, index, body.expected_timestamp)
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/feedback/{key}")
async def feedback_delete_entry(key: str, request: Request):
    """Delete an entire feedback entry."""
    store = _get_store(request)
    if not store.delete_entry(key):
        raise HTTPException(404, f"Feedback entry not found: {key}")
    return {"ok": True}


@router.delete("/feedback")
async def feedback_clear(request: Request, confirm: bool = False):
    """Clear all feedback data. Requires confirm=true query parameter."""
    store = _get_store(request)
    if not confirm:
        raise HTTPException(400, "Must pass confirm=true to clear all feedback data")
    store.clear()
    return {"ok": True}
