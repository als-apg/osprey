"""Tests for Pending Review REST API endpoints."""

import json
from pathlib import Path

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
    item_id = store.capture(
        {
            "query": "show me magnets",
            "facility": "ALS",
            "channel_count": 42,
            "selections": {"system": "MAG"},
        }
    )

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
    item_id = store.capture(
        {
            "query": "magnets",
            "facility": "ALS",
            "channel_count": 10,
            "selections": {},
        }
    )

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


def test_approve_uses_agent_task_as_query(pending_review_client):
    """agent_task is the delegation prompt — it should be used as the query."""
    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture(
        {
            "agent_task": "Find all corrector magnets for the storage ring",
            "selections": {"system": "MAG"},
            "channel_count": 5,
            "facility": "ALS",
        }
    )

    resp = pending_review_client.post(f"/api/pending-reviews/{item_id}/approve")
    assert resp.status_code == 200

    fb_store = pending_review_client.app.state.feedback_store
    hints = fb_store.get_hints("Find all corrector magnets for the storage ring", "ALS")
    assert len(hints) == 1


def test_approve_falls_back_to_config_facility(pending_review_client):
    """When item has no facility, fall back to app.state.facility_name."""
    pending_review_client.app.state.facility_name = "BESSY"

    store = pending_review_client.app.state.pending_review_store
    item_id = store.capture(
        {
            "agent_task": "show me BPMs",
            "selections": {"system": "DIA"},
            "channel_count": 3,
        }
    )

    resp = pending_review_client.post(f"/api/pending-reviews/{item_id}/approve")
    assert resp.status_code == 200

    fb_store = pending_review_client.app.state.feedback_store
    hints = fb_store.get_hints("show me BPMs", "BESSY")
    assert len(hints) == 1


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


# ------------------------------------------------------------------
# 7. Artifact resolution
# ------------------------------------------------------------------


def _artifact_store_dir(monkeypatch, tmp_path):
    """Point the agent-data resolution at *tmp_path* and return the store's dir.

    The endpoint locates artifacts the way the STORE does — the deployment's
    agent-data root plus the store's ``artifacts`` subdirectory — so these tests
    have to place their fixtures where that resolution lands rather than where
    the endpoint used to look. Patching the config the resolver reads is what
    makes the real derivation run against a temporary repo.
    """
    repo = tmp_path / "repo"
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config",
        lambda: {"project_root": str(repo), "agent_data": {"base_dir": "var/agent_data"}},
    )
    from osprey.utils.workspace import resolve_shared_data_root

    artifacts_dir = resolve_shared_data_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def _write_decoy(project_cwd, filename="abc123_channel-finder.md"):
    """Populate the RETIRED location the endpoint used to read.

    A decoy, deliberately: every assertion in these tests names the artifact
    written to the store's real directory, so repointing the endpoint back at
    ``<project_cwd>/_agent_data`` fails them instead of quietly passing on a
    file that happens to be there.
    """
    retired = Path(project_cwd) / "_agent_data" / "artifacts"
    retired.mkdir(parents=True, exist_ok=True)
    (retired / "artifacts.json").write_text(
        json.dumps(
            [
                {
                    "id": "decoy",
                    "title": "Decoy In The Retired Zone",
                    "filename": filename,
                    "source_agent": "channel-finder",
                    "category": "channel_finder",
                    "created_at": "2026-02-25T10:00:00Z",
                    "metadata": {"entry_ids": ["DECOY:01"]},
                }
            ]
        )
    )
    (retired / filename).write_text("# Decoy\n\nRead from the retired zone.")
    return retired


def _setup_artifacts(artifacts_dir, channels, filename="abc123_channel-finder.md"):
    """Create a minimal artifacts.json with one channel-finder artifact."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "id": "abc123",
        "title": "Channel Finder Result",
        "filename": filename,
        "source_agent": "channel-finder",
        "category": "channel_finder",
        "created_at": "2026-02-25T10:00:00Z",
        "metadata": {"entry_ids": channels},
    }
    (artifacts_dir / "artifacts.json").write_text(json.dumps([artifact]))
    # Create the actual artifact file
    (artifacts_dir / filename).write_text("# Channel Finder Result\n\nSome content.")
    return artifact


def test_list_enriches_with_artifact(pending_review_client, tmp_path, monkeypatch):
    """List endpoint should enrich items with matching artifact info."""
    project_cwd = str(tmp_path / "proj")
    pending_review_client.app.state.project_cwd = project_cwd
    _write_decoy(project_cwd)

    channels = ["BPM:01", "BPM:02", "BPM:03"]
    _setup_artifacts(_artifact_store_dir(monkeypatch, tmp_path), channels)

    store = pending_review_client.app.state.pending_review_store
    tool_resp = json.dumps({"channels": [{"name": c} for c in channels], "total": 3})
    store.capture(
        {"query": "bpms", "facility": "ALS", "channel_count": 3, "tool_response": tool_resp}
    )

    resp = pending_review_client.get("/api/pending-reviews")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert "artifact" in items[0]
    assert items[0]["artifact"]["filename"] == "abc123_channel-finder.md"


def test_detail_enriches_with_artifact(pending_review_client, tmp_path, monkeypatch):
    """Detail endpoint should enrich item with matching artifact info."""
    project_cwd = str(tmp_path / "proj")
    pending_review_client.app.state.project_cwd = project_cwd
    _write_decoy(project_cwd)

    channels = ["MAG:Q1", "MAG:Q2"]
    _setup_artifacts(_artifact_store_dir(monkeypatch, tmp_path), channels)

    store = pending_review_client.app.state.pending_review_store
    tool_resp = json.dumps({"channels": [{"name": c} for c in channels], "total": 2})
    item_id = store.capture(
        {"query": "magnets", "facility": "ALS", "channel_count": 2, "tool_response": tool_resp}
    )

    resp = pending_review_client.get(f"/api/pending-reviews/{item_id}")
    assert resp.status_code == 200
    assert "artifact" in resp.json()
    assert resp.json()["artifact"]["id"] == "abc123"


def test_no_artifact_when_no_overlap(pending_review_client, tmp_path, monkeypatch):
    """When no artifact channels match, no artifact field should appear."""
    project_cwd = str(tmp_path / "proj")
    pending_review_client.app.state.project_cwd = project_cwd

    _setup_artifacts(_artifact_store_dir(monkeypatch, tmp_path), ["UNRELATED:01"])

    store = pending_review_client.app.state.pending_review_store
    tool_resp = json.dumps({"channels": [{"name": "OTHER:01"}], "total": 1})
    store.capture(
        {"query": "other", "facility": "ALS", "channel_count": 1, "tool_response": tool_resp}
    )

    resp = pending_review_client.get("/api/pending-reviews")
    items = resp.json()["items"]
    assert len(items) == 1
    assert "artifact" not in items[0]


def test_no_artifact_when_file_missing(pending_review_client, tmp_path, monkeypatch):
    """When artifacts.json doesn't exist, no artifact field should appear."""
    project_cwd = str(tmp_path / "proj")
    pending_review_client.app.state.project_cwd = project_cwd
    # An empty store directory — and the retired zone POPULATED, so a reading of
    # the wrong zone would surface an artifact and fail this.
    _artifact_store_dir(monkeypatch, tmp_path)
    _write_decoy(project_cwd)

    store = pending_review_client.app.state.pending_review_store
    store.capture({"query": "bpms", "facility": "ALS", "channel_count": 1})

    resp = pending_review_client.get("/api/pending-reviews")
    items = resp.json()["items"]
    assert len(items) == 1
    assert "artifact" not in items[0]


# ------------------------------------------------------------------
# 8. Artifact file serving
# ------------------------------------------------------------------


def test_serve_artifact_markdown(pending_review_client, tmp_path, monkeypatch):
    """GET /api/artifacts/{filename} should serve the artifact file."""
    project_cwd = str(tmp_path / "proj")
    pending_review_client.app.state.project_cwd = project_cwd
    _write_decoy(project_cwd)

    _setup_artifacts(_artifact_store_dir(monkeypatch, tmp_path), [])

    resp = pending_review_client.get("/api/artifacts/abc123_channel-finder.md")
    assert resp.status_code == 200
    assert "text/markdown" in resp.headers["content-type"]
    assert "Channel Finder Result" in resp.text


def test_serve_artifact_not_found(pending_review_client, tmp_path, monkeypatch):
    """GET /api/artifacts/{filename} returns 404 for missing files."""
    project_cwd = str(tmp_path / "proj")
    pending_review_client.app.state.project_cwd = project_cwd
    _artifact_store_dir(monkeypatch, tmp_path)

    resp = pending_review_client.get("/api/artifacts/nonexistent.md")
    assert resp.status_code == 404


def test_artifact_resolves_and_serves_from_the_store_root(
    pending_review_client, tmp_path, monkeypatch
):
    """End to end: the item resolves its artifact, and the bytes come from the store.

    The endpoint read ``<project_cwd>/_agent_data/artifacts`` while the store
    writes under the deployment's agent-data root, so ``_resolve_artifact``
    returned None and ``serve_artifact`` 404'd for every pending-review item —
    fail-soft, nothing logged, the feature simply dead in the UI.

    Both zones hold a file of the SAME NAME here, with different contents, so
    only the base decides which bytes come back: this fails if the endpoint is
    ever repointed at the retired zone, rather than passing on whatever happens
    to be there.
    """
    project_cwd = str(tmp_path / "proj")
    pending_review_client.app.state.project_cwd = project_cwd
    _write_decoy(project_cwd)

    channels = ["BPM:07", "BPM:08"]
    _setup_artifacts(_artifact_store_dir(monkeypatch, tmp_path), channels)

    store = pending_review_client.app.state.pending_review_store
    tool_resp = json.dumps({"channels": [{"name": c} for c in channels], "total": 2})
    item_id = store.capture(
        {"query": "bpms", "facility": "ALS", "channel_count": 2, "tool_response": tool_resp}
    )

    detail = pending_review_client.get(f"/api/pending-reviews/{item_id}")
    assert detail.status_code == 200
    artifact = detail.json()["artifact"]
    assert artifact["id"] == "abc123"  # the store's entry, not the decoy's

    served = pending_review_client.get(f"/api/artifacts/{artifact['filename']}")
    assert served.status_code == 200
    assert "Channel Finder Result" in served.text
    assert "retired zone" not in served.text


def test_serve_artifact_path_traversal(pending_review_client):
    """Path traversal attempts should be rejected."""
    resp = pending_review_client.get("/api/artifacts/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404, 422)

    resp = pending_review_client.get("/api/artifacts/foo/bar.md")
    # FastAPI might return 404 for path with slashes, or 400
    assert resp.status_code in (400, 404, 422)
