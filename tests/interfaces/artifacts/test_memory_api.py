"""Tests for the Memory REST API routes in the gallery app.

Covers:
  - GET /api/memory — list entries with type, tag, search filters
  - GET /api/memory/{id} — get single entry, 404 for missing
  - GET/POST /api/memory/focus — focus management
  - PATCH /api/memory/{id} — update entry, 404 for missing
  - DELETE /api/memory/{id} — delete entry, 404 for missing
"""

import pytest


def _save_memory(
    memory_store,
    content="test memory",
    memory_type="note",
    tags=None,
    importance="normal",
):
    """Helper to save a memory entry directly on the store."""
    return memory_store.save(
        memory_type=memory_type,
        content=content,
        tags=tags or [],
        importance=importance,
    )


# ---------------------------------------------------------------------------
# Gallery app memory routes
# ---------------------------------------------------------------------------


class TestMemoryRoutes:
    """Tests for the memory REST API routes."""

    @pytest.fixture
    def app_client(self, tmp_path):
        """Create a test client for the gallery app."""
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_list_empty(self, app_client):
        resp = app_client.get("/api/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["entries"] == []

    @pytest.mark.unit
    def test_list_with_entries(self, app_client):
        mem = app_client.app.state.memory_store
        _save_memory(mem, content="beam current note")
        _save_memory(mem, content="vacuum observation")

        resp = app_client.get("/api/memory")
        data = resp.json()
        assert data["count"] == 2
        assert len(data["entries"]) == 2

    @pytest.mark.unit
    def test_list_filter_by_type(self, app_client):
        mem = app_client.app.state.memory_store
        _save_memory(mem, memory_type="note", content="a note")
        _save_memory(mem, memory_type="pin", content="a pin")

        resp = app_client.get("/api/memory?type=note")
        data = resp.json()
        assert data["count"] == 1
        assert data["entries"][0]["memory_type"] == "note"

    @pytest.mark.unit
    def test_list_filter_by_tag(self, app_client):
        mem = app_client.app.state.memory_store
        _save_memory(mem, tags=["beam", "current"])
        _save_memory(mem, tags=["vacuum"])

        resp = app_client.get("/api/memory?tag=beam")
        data = resp.json()
        assert data["count"] == 1
        assert "beam" in data["entries"][0]["tags"]

    @pytest.mark.unit
    def test_list_search(self, app_client):
        mem = app_client.app.state.memory_store
        _save_memory(mem, content="beam current reading at 500 mA")
        _save_memory(mem, content="vacuum pressure nominal")

        resp = app_client.get("/api/memory?search=beam")
        data = resp.json()
        assert data["count"] == 1
        assert "beam" in data["entries"][0]["content"].lower()

    @pytest.mark.unit
    def test_get_memory(self, app_client):
        mem = app_client.app.state.memory_store
        entry = _save_memory(mem, content="test get memory")

        resp = app_client.get(f"/api/memory/{entry.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == entry.id
        assert data["content"] == "test get memory"

    @pytest.mark.unit
    def test_get_memory_not_found(self, app_client):
        resp = app_client.get("/api/memory/999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Memory focus endpoints
# ---------------------------------------------------------------------------


class TestMemoryFocusRoutes:
    """Tests for memory focus GET/POST routes."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_get_focus_empty(self, app_client):
        resp = app_client.get("/api/memory/focus")
        assert resp.status_code == 200
        data = resp.json()
        assert data["focused"] is False
        assert data["entry"] is None

    @pytest.mark.unit
    def test_get_focus_returns_latest(self, app_client):
        mem = app_client.app.state.memory_store
        _save_memory(mem, content="first memory")
        _save_memory(mem, content="second memory")

        resp = app_client.get("/api/memory/focus")
        data = resp.json()
        assert data["focused"] is False
        assert data["entry"]["content"] == "second memory"

    @pytest.mark.unit
    def test_set_and_get_focus(self, app_client):
        mem = app_client.app.state.memory_store
        e1 = _save_memory(mem, content="first memory")
        _save_memory(mem, content="second memory")

        # Focus on the first entry
        resp = app_client.post("/api/memory/focus", json={"memory_id": e1.id})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # GET should return the focused entry
        resp = app_client.get("/api/memory/focus")
        data = resp.json()
        assert data["focused"] is True
        assert data["entry"]["id"] == e1.id

    @pytest.mark.unit
    def test_set_focus_not_found(self, app_client):
        resp = app_client.post("/api/memory/focus", json={"memory_id": 999})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Memory update
# ---------------------------------------------------------------------------


class TestMemoryUpdate:
    """Tests for PATCH /api/memory/{id} update route."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_update_memory(self, app_client):
        mem = app_client.app.state.memory_store
        entry = _save_memory(mem, content="original content")

        resp = app_client.patch(
            f"/api/memory/{entry.id}",
            json={"content": "updated content"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "updated content"

    @pytest.mark.unit
    def test_update_not_found(self, app_client):
        resp = app_client.patch(
            "/api/memory/999",
            json={"content": "does not matter"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Memory delete
# ---------------------------------------------------------------------------


class TestMemoryDelete:
    """Tests for DELETE /api/memory/{id} delete route."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_delete_memory(self, app_client):
        mem = app_client.app.state.memory_store
        entry = _save_memory(mem, content="to be deleted")

        resp = app_client.delete(f"/api/memory/{entry.id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify it's gone
        resp = app_client.get(f"/api/memory/{entry.id}")
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_delete_not_found(self, app_client):
        resp = app_client.delete("/api/memory/999")
        assert resp.status_code == 404
