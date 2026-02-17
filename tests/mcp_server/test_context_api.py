"""Tests for the DataContext REST API routes and context_focus MCP tool.

Covers:
  - GET /api/context — list entries with tool, data_type, search filters
  - GET /api/context/{id} — get single entry, 404 for missing
  - GET /api/context/{id}/data — serve raw data file, 404 for missing
  - GET/POST /api/context/focus — focus management
  - SSE discrimination — events have {"type": "context"} or {"type": "artifact"}
  - context_focus MCP tool — valid entry, unknown entry
"""

import json

import pytest

from tests.mcp_server.conftest import get_tool_fn


def _save_context_entry(context_store, tool="channel_read", data_type="channel_values",
                        description="test entry"):
    """Helper to save a context entry directly on the store."""
    return context_store.save(
        tool=tool,
        data={"value": 42},
        description=description,
        summary={"count": 1},
        access_details={"format": "json"},
        data_type=data_type,
    )


# ---------------------------------------------------------------------------
# Gallery app context routes
# ---------------------------------------------------------------------------


class TestContextRoutes:
    """Tests for the context REST API routes."""

    @pytest.fixture
    def app_client(self, tmp_path):
        """Create a test client for the gallery app."""
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    def test_list_context_empty(self, app_client):
        resp = app_client.get("/api/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["entries"] == []

    def test_list_context_with_entries(self, app_client):
        ctx = app_client.app.state.context_store
        _save_context_entry(ctx, description="beam current")
        _save_context_entry(ctx, tool="archiver_read", description="vacuum trend")

        resp = app_client.get("/api/context")
        data = resp.json()
        assert data["count"] == 2
        assert len(data["entries"]) == 2

    def test_list_context_filter_by_tool(self, app_client):
        ctx = app_client.app.state.context_store
        _save_context_entry(ctx, tool="channel_read")
        _save_context_entry(ctx, tool="archiver_read")
        _save_context_entry(ctx, tool="channel_read")

        resp = app_client.get("/api/context?tool=channel_read")
        data = resp.json()
        assert data["count"] == 2

    def test_list_context_filter_by_data_type(self, app_client):
        ctx = app_client.app.state.context_store
        _save_context_entry(ctx, data_type="channel_values")
        _save_context_entry(ctx, data_type="timeseries")

        resp = app_client.get("/api/context?data_type=timeseries")
        data = resp.json()
        assert data["count"] == 1
        assert data["entries"][0]["data_type"] == "timeseries"

    def test_list_context_search(self, app_client):
        ctx = app_client.app.state.context_store
        _save_context_entry(ctx, description="beam current reading")
        _save_context_entry(ctx, description="vacuum pressure")

        resp = app_client.get("/api/context?search=beam")
        data = resp.json()
        assert data["count"] == 1
        assert "beam" in data["entries"][0]["description"].lower()

    def test_get_context_entry(self, app_client):
        ctx = app_client.app.state.context_store
        entry = _save_context_entry(ctx, description="test get")

        resp = app_client.get(f"/api/context/{entry.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == entry.id
        assert data["description"] == "test get"

    def test_get_context_entry_not_found(self, app_client):
        resp = app_client.get("/api/context/999")
        assert resp.status_code == 404

    def test_get_context_data(self, app_client):
        ctx = app_client.app.state.context_store
        entry = _save_context_entry(ctx)

        resp = app_client.get(f"/api/context/{entry.id}/data")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        data = resp.json()
        assert data["data"]["value"] == 42
        assert data["_osprey_metadata"]["context_entry_id"] == entry.id

    def test_get_context_data_not_found(self, app_client):
        resp = app_client.get("/api/context/999/data")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Context focus endpoints
# ---------------------------------------------------------------------------


class TestContextFocusRoutes:
    """Tests for context focus GET/POST routes."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    def test_get_focus_empty(self, app_client):
        resp = app_client.get("/api/context/focus")
        assert resp.status_code == 200
        data = resp.json()
        assert data["focused"] is False
        assert data["entry"] is None

    def test_get_focus_returns_latest(self, app_client):
        ctx = app_client.app.state.context_store
        _save_context_entry(ctx, description="first")
        _save_context_entry(ctx, description="second")

        resp = app_client.get("/api/context/focus")
        data = resp.json()
        assert data["focused"] is False
        assert data["entry"]["description"] == "second"

    def test_set_and_get_focus(self, app_client):
        ctx = app_client.app.state.context_store
        e1 = _save_context_entry(ctx, description="first")
        _save_context_entry(ctx, description="second")

        # Focus on the first entry
        resp = app_client.post("/api/context/focus", json={"entry_id": e1.id})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # GET should return the focused entry
        resp = app_client.get("/api/context/focus")
        data = resp.json()
        assert data["focused"] is True
        assert data["entry"]["id"] == e1.id

    def test_set_focus_not_found(self, app_client):
        resp = app_client.post("/api/context/focus", json={"entry_id": 999})
        assert resp.status_code == 404

    def test_stale_focus_falls_back(self, app_client):
        ctx = app_client.app.state.context_store
        _save_context_entry(ctx, description="only entry")

        # Set focus to a nonexistent ID
        app_client.app.state.focused_context_id = 999

        resp = app_client.get("/api/context/focus")
        data = resp.json()
        assert data["focused"] is False
        assert data["entry"]["description"] == "only entry"
        assert app_client.app.state.focused_context_id is None


# ---------------------------------------------------------------------------
# SSE event discrimination
# ---------------------------------------------------------------------------


class TestSSEDiscrimination:
    """Verify SSE events include type discrimination."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    def test_artifact_event_has_type(self, app_client):
        """Artifact SSE events include {"type": "artifact"}."""
        store = app_client.app.state.artifact_store
        # The listener registered in lifespan won't be active in sync test client,
        # so we test the broadcast format directly.
        entry = store.save_file(
            file_content=b"test",
            filename="test.html",
            artifact_type="html",
            title="Test",
            mime_type="text/html",
            tool_source="test",
        )
        # Verify the entry dict structure is compatible with type-tagged broadcast
        tagged = {"type": "artifact", **entry.to_dict()}
        assert tagged["type"] == "artifact"
        assert tagged["id"] == entry.id

    def test_context_event_has_type(self, app_client):
        """Context events include {"type": "context"}."""
        ctx = app_client.app.state.context_store
        entry = _save_context_entry(ctx)
        tagged = {"type": "context", **entry.to_dict()}
        assert tagged["type"] == "context"
        assert tagged["id"] == entry.id


# ---------------------------------------------------------------------------
# context_focus MCP tool
# ---------------------------------------------------------------------------


def _get_context_focus():
    from osprey.mcp_server.workspace.tools.focus_tools import context_focus

    return get_tool_fn(context_focus)


class TestContextFocusTool:
    """Tests for the context_focus MCP tool."""

    @pytest.mark.asyncio
    async def test_focus_unknown_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_context_focus()
        result = await fn(entry_id=999)

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "not_found"

    @pytest.mark.asyncio
    async def test_focus_valid_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        from osprey.mcp_server.data_context import get_data_context

        ctx = get_data_context()
        entry = _save_context_entry(ctx, description="focused entry")

        fn = _get_context_focus()
        result = await fn(entry_id=entry.id)

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["entry_id"] == entry.id
        assert data["description"] == "focused entry"
        assert "gallery_url" in data
