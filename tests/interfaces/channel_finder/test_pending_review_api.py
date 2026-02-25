"""Tests for Pending Review REST API endpoints."""

import pytest


# ------------------------------------------------------------------
# 1. Status endpoint
# ------------------------------------------------------------------


def test_status_returns_available(pending_review_client):
    resp = pending_review_client.get("/api/pending-reviews/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["item_count"] == 0


def test_status_unavailable(client):
    """When store is None, status still returns 200 with available=False."""
    # Force store to None (lifespan initializes it unconditionally)
    client.app.state.pending_review_store = None
    resp = client.get("/api/pending-reviews/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


# ------------------------------------------------------------------
# 2. List endpoint
# ------------------------------------------------------------------


def test_list_empty(pending_review_client):
    resp = pending_review_client.get("/api/pending-reviews")
    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_list_returns_items(pending_review_client):
    # Capture some items directly via store
    store = pending_review_client.app.state.pending_review_store
    store.capture({"query": "magnets", "facility": "ALS", "channel_count": 10})
    store.capture({"query": "bpms", "facility": "ALS", "channel_count": 5})

    resp = pending_review_client.get("/api/pending-reviews")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 2
    # Newest first
    assert items[0]["query"] == "bpms"


# ------------------------------------------------------------------
# 3. Detail endpoint
# ------------------------------------------------------------------


def test_get_existing_item(pending_review_client):
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture({"query": "magnets", "facility": "ALS"})

    resp = pending_review_client.get(f"/api/pending-reviews/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["query"] == "magnets"


def test_get_missing_item(pending_review_client):
    resp = pending_review_client.get("/api/pending-reviews/nonexistent")
    assert resp.status_code == 404


# ------------------------------------------------------------------
# 4. Approve endpoint (promote to feedback store)
# ------------------------------------------------------------------


def test_approve_promotes_to_feedback(pending_review_client):
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture({
        "query": "show me magnets",
        "facility": "ALS",
        "channel_count": 42,
        "selections": {"system": "MAG"},
    })

    resp = pending_review_client.post(f"/api/pending-reviews/{item_id}/approve")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "feedback_key" in data

    # Item should be removed from pending
    assert store.get_item(item_id) is None

    # Item should exist in feedback store
    fb_store = pending_review_client.app.state.feedback_store
    hints = fb_store.get_hints("show me magnets", "ALS")
    assert len(hints) == 1
    assert hints[0]["selections"] == {"system": "MAG"}
    assert hints[0]["channel_count"] == 42


def test_approve_with_overrides(pending_review_client):
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture({
        "query": "magnets",
        "facility": "ALS",
        "channel_count": 10,
        "selections": {},
    })

    resp = pending_review_client.post(
        f"/api/pending-reviews/{item_id}/approve",
        json={
            "facility": "LCLS",
            "selections": {"system": "QUAD"},
            "channel_count": 99,
        },
    )
    assert resp.status_code == 200

    # Check overrides were applied
    fb_store = pending_review_client.app.state.feedback_store
    hints = fb_store.get_hints("magnets", "LCLS")
    assert len(hints) == 1
    assert hints[0]["selections"] == {"system": "QUAD"}
    assert hints[0]["channel_count"] == 99


def test_approve_missing_item(pending_review_client):
    resp = pending_review_client.post("/api/pending-reviews/nonexistent/approve")
    assert resp.status_code == 404


# ------------------------------------------------------------------
# 5. Dismiss endpoint
# ------------------------------------------------------------------


def test_dismiss_deletes_item(pending_review_client):
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture({"query": "magnets", "facility": "ALS"})

    resp = pending_review_client.delete(f"/api/pending-reviews/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Verify deleted
    assert store.get_item(item_id) is None


def test_dismiss_missing_item(pending_review_client):
    resp = pending_review_client.delete("/api/pending-reviews/nonexistent")
    assert resp.status_code == 404


# ------------------------------------------------------------------
# 6. Clear endpoint
# ------------------------------------------------------------------


def test_clear_requires_confirm(pending_review_client):
    resp = pending_review_client.delete("/api/pending-reviews")
    assert resp.status_code == 400


def test_clear_removes_all(pending_review_client):
    store = pending_review_client.app.state.pending_review_store
    store.capture({"query": "q1", "facility": "ALS"})
    store.capture({"query": "q2", "facility": "ALS"})

    resp = pending_review_client.delete("/api/pending-reviews?confirm=true")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    assert store.list_items() == []
