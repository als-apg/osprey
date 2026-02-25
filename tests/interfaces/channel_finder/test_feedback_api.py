"""Tests for Feedback Management REST API."""

from __future__ import annotations

import pytest


# ------------------------------------------------------------------
# Status endpoint
# ------------------------------------------------------------------


class TestFeedbackStatus:
    def test_available(self, feedback_client):
        resp = feedback_client.get("/api/feedback/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["entry_count"] == 0
        assert data["store_path"] is not None

    def test_unavailable(self, client):
        """When feedback_store is None, status returns available=false."""
        resp = client.get("/api/feedback/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False


# ------------------------------------------------------------------
# List entries
# ------------------------------------------------------------------


class TestFeedbackList:
    def test_empty(self, feedback_client):
        resp = feedback_client.get("/api/feedback")
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    def test_populated(self, feedback_client):
        feedback_client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 42,
        })
        resp = feedback_client.get("/api/feedback")
        entries = resp.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["query"] == "magnets"
        assert entries[0]["success_count"] == 1


# ------------------------------------------------------------------
# Detail
# ------------------------------------------------------------------


class TestFeedbackDetail:
    def test_existing_key(self, feedback_client):
        add_resp = feedback_client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 10,
        })
        key = add_resp.json()["key"]
        resp = feedback_client.get(f"/api/feedback/{key}")
        assert resp.status_code == 200
        assert len(resp.json()["successes"]) == 1

    def test_missing_key(self, feedback_client):
        resp = feedback_client.get("/api/feedback/nonexistent")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# Add entry
# ------------------------------------------------------------------


class TestFeedbackAdd:
    def test_add_success(self, feedback_client):
        resp = feedback_client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 42,
        })
        assert resp.status_code == 200
        assert len(resp.json()["key"]) == 64

    def test_add_failure(self, feedback_client):
        resp = feedback_client.post("/api/feedback", json={
            "query": "bad", "facility": "ALS",
            "entry_type": "failure", "selections": {"system": "X"},
            "reason": "not found",
        })
        assert resp.status_code == 200

    def test_invalid_entry_type(self, feedback_client):
        resp = feedback_client.post("/api/feedback", json={
            "query": "test", "facility": "ALS",
            "entry_type": "invalid",
        })
        assert resp.status_code == 400


# ------------------------------------------------------------------
# Edit record
# ------------------------------------------------------------------


class TestFeedbackEdit:
    def _add_and_get_ts(self, client):
        client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 10,
        })
        entries = client.get("/api/feedback").json()["entries"]
        key = entries[0]["key"]
        detail = client.get(f"/api/feedback/{key}").json()
        ts = detail["successes"][0]["timestamp"]
        return key, ts

    def test_edit_success_record(self, feedback_client):
        key, ts = self._add_and_get_ts(feedback_client)
        resp = feedback_client.put(f"/api/feedback/{key}/successes/0", json={
            "expected_timestamp": ts,
            "selections": {"system": "QUAD"},
            "channel_count": 99,
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_stale_timestamp(self, feedback_client):
        key, _ = self._add_and_get_ts(feedback_client)
        resp = feedback_client.put(f"/api/feedback/{key}/successes/0", json={
            "expected_timestamp": "wrong",
            "selections": {"system": "QUAD"},
        })
        assert resp.status_code == 409

    def test_invalid_record_type(self, feedback_client):
        key, ts = self._add_and_get_ts(feedback_client)
        resp = feedback_client.put(f"/api/feedback/{key}/invalid/0", json={
            "expected_timestamp": ts,
        })
        assert resp.status_code == 400


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------


class TestFeedbackDelete:
    def test_delete_entry(self, feedback_client):
        resp = feedback_client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 10,
        })
        key = resp.json()["key"]
        del_resp = feedback_client.delete(f"/api/feedback/{key}")
        assert del_resp.status_code == 200
        assert feedback_client.get(f"/api/feedback/{key}").status_code == 404

    def test_delete_missing_entry(self, feedback_client):
        resp = feedback_client.delete("/api/feedback/nonexistent")
        assert resp.status_code == 404

    def test_delete_record_with_valid_timestamp(self, feedback_client):
        feedback_client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 10,
        })
        entries = feedback_client.get("/api/feedback").json()["entries"]
        key = entries[0]["key"]
        detail = feedback_client.get(f"/api/feedback/{key}").json()
        ts = detail["successes"][0]["timestamp"]

        resp = feedback_client.request(
            "DELETE", f"/api/feedback/{key}/successes/0",
            json={"expected_timestamp": ts},
        )
        assert resp.status_code == 200

    def test_delete_record_stale_timestamp(self, feedback_client):
        feedback_client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 10,
        })
        entries = feedback_client.get("/api/feedback").json()["entries"]
        key = entries[0]["key"]

        resp = feedback_client.request(
            "DELETE", f"/api/feedback/{key}/successes/0",
            json={"expected_timestamp": "wrong"},
        )
        assert resp.status_code == 409


# ------------------------------------------------------------------
# Clear all
# ------------------------------------------------------------------


class TestFeedbackClear:
    def test_requires_confirm(self, feedback_client):
        resp = feedback_client.delete("/api/feedback")
        assert resp.status_code == 400

    def test_clears_data(self, feedback_client):
        feedback_client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 10,
        })
        resp = feedback_client.delete("/api/feedback?confirm=true")
        assert resp.status_code == 200
        assert feedback_client.get("/api/feedback").json()["entries"] == []


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


class TestFeedbackExport:
    def test_export(self, feedback_client):
        feedback_client.post("/api/feedback", json={
            "query": "magnets", "facility": "ALS",
            "entry_type": "success", "selections": {"system": "MAG"},
            "channel_count": 10,
        })
        resp = feedback_client.get("/api/feedback/export")
        assert resp.status_code == 200
        assert "attachment" in resp.headers.get("content-disposition", "")
        data = resp.json()
        assert data["version"] == 2
        assert len(data["entries"]) == 1


# ------------------------------------------------------------------
# Unavailable store
# ------------------------------------------------------------------


class TestFeedbackUnavailable:
    """All CRUD endpoints return 404 when store is None."""

    def test_list_returns_404(self, client):
        assert client.get("/api/feedback").status_code == 404

    def test_detail_returns_404(self, client):
        assert client.get("/api/feedback/somekey").status_code == 404

    def test_add_returns_404(self, client):
        resp = client.post("/api/feedback", json={
            "query": "x", "facility": "X", "entry_type": "success",
        })
        assert resp.status_code == 404

    def test_delete_returns_404(self, client):
        assert client.delete("/api/feedback/somekey").status_code == 404

    def test_clear_returns_404(self, client):
        assert client.delete("/api/feedback?confirm=true").status_code == 404

    def test_export_returns_404(self, client):
        assert client.get("/api/feedback/export").status_code == 404
