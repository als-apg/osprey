"""Tests for POST /api/pending-reviews/{id}/reject endpoint."""

from __future__ import annotations


def test_reject_records_failure_in_feedback_store(pending_review_client):
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture(
        {
            "query": "show me magnets",
            "facility": "ALS",
            "selections": {"system": "MAG"},
            "channel_count": 10,
        }
    )

    resp = pending_review_client.post(f"/api/pending-reviews/{item_id}/reject")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Item removed from pending
    assert store.get_item(item_id) is None

    # Failure recorded in feedback store
    fb_store = pending_review_client.app.state.feedback_store
    keys = fb_store.list_keys()
    assert len(keys) == 1
    entry = fb_store.get_entry(keys[0]["key"])
    assert len(entry["failures"]) == 1
    assert entry["failures"][0]["reason"] == "operator rejected"


def test_reject_with_custom_reason(pending_review_client):
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture(
        {
            "query": "beam current",
            "facility": "ALS",
            "selections": {"system": "SR"},
        }
    )

    resp = pending_review_client.post(
        f"/api/pending-reviews/{item_id}/reject",
        json={"reason": "wrong system selected"},
    )
    assert resp.status_code == 200

    fb_store = pending_review_client.app.state.feedback_store
    keys = fb_store.list_keys()
    assert len(keys) == 1
    entry = fb_store.get_entry(keys[0]["key"])
    assert entry["failures"][0]["reason"] == "wrong system selected"


def test_reject_with_overrides(pending_review_client):
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture(
        {
            "query": "test",
            "facility": "ALS",
            "selections": {"system": "X"},
        }
    )

    resp = pending_review_client.post(
        f"/api/pending-reviews/{item_id}/reject",
        json={
            "reason": "wrong path",
            "partial_selections": {"system": "Y"},
            "facility": "BESSY",
        },
    )
    assert resp.status_code == 200

    fb_store = pending_review_client.app.state.feedback_store
    keys = fb_store.list_keys()
    # Key is based on facility="BESSY", query="test"
    assert keys[0]["facility"] == "BESSY"
    entry = fb_store.get_entry(keys[0]["key"])
    assert entry["failures"][0]["partial_selections"] == {"system": "Y"}


def test_reject_uses_agent_task_as_query(pending_review_client):
    """agent_task is the delegation prompt — it should be used as the query."""
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture(
        {
            "agent_task": "Find all corrector magnets",
            "facility": "ALS",
            "selections": {"system": "MAG"},
        }
    )

    resp = pending_review_client.post(f"/api/pending-reviews/{item_id}/reject")
    assert resp.status_code == 200

    fb_store = pending_review_client.app.state.feedback_store
    keys = fb_store.list_keys()
    assert len(keys) == 1
    assert keys[0]["query"] == "Find all corrector magnets"


def test_reject_falls_back_to_config_facility(pending_review_client):
    """When item has no facility, fall back to app.state.facility_name."""
    pending_review_client.app.state.facility_name = "BESSY"

    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture(
        {
            "agent_task": "show me BPMs",
            "selections": {"system": "DIA"},
        }
    )

    resp = pending_review_client.post(f"/api/pending-reviews/{item_id}/reject")
    assert resp.status_code == 200

    fb_store = pending_review_client.app.state.feedback_store
    keys = fb_store.list_keys()
    assert len(keys) == 1
    assert keys[0]["facility"] == "BESSY"


def test_reject_missing_item(pending_review_client):
    resp = pending_review_client.post("/api/pending-reviews/nonexistent/reject")
    assert resp.status_code == 404


def test_reject_requires_feedback_store(pending_review_client):
    pending_review_client.app.state.feedback_store = None
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture({"query": "test", "facility": "ALS"})
    resp = pending_review_client.post(f"/api/pending-reviews/{item_id}/reject")
    assert resp.status_code == 404
