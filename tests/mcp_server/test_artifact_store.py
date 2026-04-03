"""Tests for the OSPREY Artifact Store and artifact-related tools.

Covers:
  - ArtifactStore: save_file, save_object, save_from_path, list/get
  - Smart serialization (_serialize_object)
  - save_artifact() injection into execute tool namespace
  - artifact_save MCP tool (file_path and content modes)
  - Artifact Gallery app routes
"""

import json
from unittest.mock import patch

import pytest

from tests.mcp_server.conftest import get_tool_fn

# ---------------------------------------------------------------------------
# ArtifactStore — core storage layer
# ---------------------------------------------------------------------------


class TestArtifactStore:
    """Unit tests for ArtifactStore."""

    def test_save_file_creates_file_and_index(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_file(
            file_content=b"<h1>Hello</h1>",
            filename="report.html",
            artifact_type="html",
            title="Test Report",
            description="A test artifact",
            mime_type="text/html",
            tool_source="test",
        )

        assert entry.title == "Test Report"
        assert entry.artifact_type == "html"
        assert entry.size_bytes == len(b"<h1>Hello</h1>")
        assert entry.tool_source == "test"

        # File should exist on disk
        filepath = store.get_file_path(entry.id)
        assert filepath is not None
        assert filepath.exists()
        assert filepath.read_bytes() == b"<h1>Hello</h1>"

        # Index file should exist
        index_file = tmp_path / "artifacts" / "artifacts.json"
        assert index_file.exists()
        index = json.loads(index_file.read_text())
        assert index["entry_count"] == 1
        assert index["entries"][0]["id"] == entry.id

    def test_save_object_string_markdown(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("# Hello World", title="Markdown Test")

        assert entry.artifact_type == "markdown"
        assert entry.mime_type == "text/markdown"

    def test_save_object_string_html(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("<h1>Title</h1><p>Body</p>", title="HTML Test")

        assert entry.artifact_type == "html"
        assert entry.mime_type == "text/html"

    def test_save_object_dict(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object({"key": "value", "n": 42}, title="JSON Data")

        assert entry.artifact_type == "json"
        filepath = store.get_file_path(entry.id)
        content = json.loads(filepath.read_text())
        assert content["key"] == "value"

    def test_save_object_list(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object([1, 2, 3], title="List Data")

        assert entry.artifact_type == "json"

    def test_save_object_bytes(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object(b"\x89PNG\r\n", title="Binary Data")

        assert entry.artifact_type == "binary"

    def test_save_from_path(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        # Create a source file
        source = tmp_path / "source_file.png"
        source.write_bytes(b"\x89PNG fake image data")

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_from_path(source, title="Test Image")

        assert entry.artifact_type == "image"
        assert entry.mime_type == "image/png"

        # File should be copied into artifact dir
        filepath = store.get_file_path(entry.id)
        assert filepath.exists()
        assert filepath.read_bytes() == b"\x89PNG fake image data"

    def test_save_from_path_file_not_found(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.save_from_path("/nonexistent/file.txt", title="Missing")

    def test_list_entries_no_filter(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_object("# A", title="First")
        store.save_object({"x": 1}, title="Second")

        entries = store.list_entries()
        assert len(entries) == 2

    def test_list_entries_type_filter(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_object("# A", title="Markdown")
        store.save_object({"x": 1}, title="JSON")

        md = store.list_entries(type_filter="markdown")
        assert len(md) == 1
        assert md[0].title == "Markdown"

    def test_list_entries_search(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        store.save_object("# A", title="Beam Current Plot")
        store.save_object("# B", title="Vacuum Trend")

        results = store.list_entries(search="beam")
        assert len(results) == 1
        assert results[0].title == "Beam Current Plot"

    def test_get_entry(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("data", title="Test")

        found = store.get_entry(entry.id)
        assert found is not None
        assert found.title == "Test"

        assert store.get_entry("nonexistent") is None

    def test_index_persistence(self, tmp_path):
        """Index survives re-instantiation."""
        from osprey.stores.artifact_store import ArtifactStore

        store1 = ArtifactStore(workspace_root=tmp_path)
        entry = store1.save_object("data", title="Persistent")

        store2 = ArtifactStore(workspace_root=tmp_path)
        assert len(store2.list_entries()) == 1
        assert store2.get_entry(entry.id) is not None

    def test_to_tool_response(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("data", title="Resp Test")

        resp = entry.to_tool_response(gallery_url="http://localhost:8086")
        assert resp["status"] == "success"
        assert resp["artifact_id"] == entry.id
        assert resp["gallery_url"] == "http://localhost:8086"

    def test_unique_ids(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        ids = set()
        for i in range(10):
            entry = store.save_object(f"data{i}", title=f"Entry {i}")
            ids.add(entry.id)
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------


class TestDeleteEntry:
    """Tests for ArtifactStore.delete_entry()."""

    def test_delete_existing(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_file(
            file_content=b"<h1>Delete me</h1>",
            filename="delete.html",
            artifact_type="html",
            title="To Delete",
            mime_type="text/html",
            tool_source="test",
        )

        filepath = store.get_file_path(entry.id)
        assert filepath.exists()

        result = store.delete_entry(entry.id)
        assert result is True
        assert store.get_entry(entry.id) is None
        assert len(store.list_entries()) == 0
        assert not filepath.exists()

    def test_delete_nonexistent(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        result = store.delete_entry("nonexistent")
        assert result is False

    def test_delete_preserves_other_entries(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        e1 = store.save_object("data1", title="Keep")
        e2 = store.save_object("data2", title="Delete")

        store.delete_entry(e2.id)
        assert len(store.list_entries()) == 1
        assert store.get_entry(e1.id) is not None


class TestUpdateEntryMetadata:
    """Tests for BaseStore.update_entry_metadata()."""

    def test_update_single_field(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("# Hello", title="Test")

        result = store.update_entry_metadata(entry.id, category="document")
        assert result is not None
        assert result.category == "document"

        # Verify persisted to disk
        store2 = ArtifactStore(workspace_root=tmp_path)
        reloaded = store2.get_entry(entry.id)
        assert reloaded.category == "document"

    def test_update_multiple_fields(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("# Hello", title="Test")

        result = store.update_entry_metadata(
            entry.id, category="document", source_agent="data-visualizer"
        )
        assert result.category == "document"
        assert result.source_agent == "data-visualizer"

    def test_invalid_entry_id_returns_none(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        result = store.update_entry_metadata("nonexistent", category="x")
        assert result is None

    def test_invalid_attribute_raises(self, tmp_path):
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_object("# Hello", title="Test")

        with pytest.raises(AttributeError, match="no attribute 'bogus_field'"):
            store.update_entry_metadata(entry.id, bogus_field="oops")


class TestCrossProcessSafety:
    """Tests for cross-process file-locking in ArtifactStore."""

    def test_save_no_orphan_files(self, tmp_path):
        """Both instances save — all artifact files are referenced in the index."""
        from osprey.stores.artifact_store import ArtifactStore

        store_a = ArtifactStore(workspace_root=tmp_path)
        store_b = ArtifactStore(workspace_root=tmp_path)

        store_a.save_file(
            file_content=b"<h1>A</h1>",
            filename="a.html",
            artifact_type="html",
            title="From A",
            mime_type="text/html",
            tool_source="test",
        )
        store_b.save_file(
            file_content=b"<h1>B</h1>",
            filename="b.html",
            artifact_type="html",
            title="From B",
            mime_type="text/html",
            tool_source="test",
        )

        index = json.loads((tmp_path / "artifacts" / "artifacts.json").read_text())
        assert index["entry_count"] == 2
        index_filenames = {e["filename"] for e in index["entries"]}
        assert len(index_filenames) == 2

    def test_concurrent_saves_both_persist(self, tmp_path):
        """Both instances save — both entries present on reload."""
        from osprey.stores.artifact_store import ArtifactStore

        store_a = ArtifactStore(workspace_root=tmp_path)
        store_b = ArtifactStore(workspace_root=tmp_path)

        store_a.save_object("# First", title="From A")
        store_b.save_object("# Second", title="From B")

        store_check = ArtifactStore(workspace_root=tmp_path)
        entries = store_check.list_entries()
        assert len(entries) == 2
        titles = {e.title for e in entries}
        assert titles == {"From A", "From B"}

    def test_delete_concurrent_with_save(self, tmp_path):
        """Delete from one instance while another saves — no data loss."""
        from osprey.stores.artifact_store import ArtifactStore

        store_a = ArtifactStore(workspace_root=tmp_path)
        e1 = store_a.save_object("data", title="To Delete")

        store_b = ArtifactStore(workspace_root=tmp_path)
        store_c = ArtifactStore(workspace_root=tmp_path)

        store_b.delete_entry(e1.id)
        e2 = store_c.save_object("data2", title="New Entry")

        store_check = ArtifactStore(workspace_root=tmp_path)
        entries = store_check.list_entries()
        assert len(entries) == 1
        assert entries[0].id == e2.id
        assert entries[0].title == "New Entry"


class TestSingleton:
    """Tests for singleton lifecycle."""

    def test_get_and_reset(self, tmp_path, monkeypatch):
        from osprey.stores.artifact_store import (
            get_artifact_store,
            reset_artifact_store,
        )

        monkeypatch.chdir(tmp_path)

        store1 = get_artifact_store()
        store2 = get_artifact_store()
        assert store1 is store2

        reset_artifact_store()
        store3 = get_artifact_store()
        assert store3 is not store1


# ---------------------------------------------------------------------------
# artifact_save MCP tool
# ---------------------------------------------------------------------------


def _get_artifact_save():
    from osprey.mcp_server.workspace.tools.artifact_save import artifact_save

    return get_tool_fn(artifact_save)


class TestArtifactSaveTool:
    """Tests for the artifact_save MCP tool."""

    @pytest.mark.asyncio
    async def test_save_inline_markdown(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_save()
        result = await fn(
            title="Test Summary",
            content="# Summary\n\nAll systems nominal.",
            content_type="markdown",
        )

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["artifact_type"] == "markdown"

    @pytest.mark.asyncio
    async def test_save_inline_html(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_save()
        result = await fn(
            title="HTML Report",
            content="<h1>Report</h1><p>Done.</p>",
            content_type="html",
        )

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["artifact_type"] == "html"

    @pytest.mark.asyncio
    async def test_save_inline_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_save()
        result = await fn(
            title="JSON Data",
            content='{"key": "value"}',
            content_type="json",
        )

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["artifact_type"] == "json"

    @pytest.mark.asyncio
    async def test_save_file_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create a file to register
        source = tmp_path / "test_data.csv"
        source.write_text("a,b,c\n1,2,3\n")

        fn = _get_artifact_save()
        result = await fn(
            title="CSV Data",
            file_path=str(source),
        )

        data = json.loads(result)
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_both_file_and_content_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_save()
        result = await fn(
            title="Bad",
            file_path="/some/file",
            content="some content",
        )

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_neither_file_nor_content_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_save()
        result = await fn(title="Empty")

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_invalid_content_type_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_save()
        result = await fn(
            title="Bad Type",
            content="data",
            content_type="xml",
        )

        data = json.loads(result)
        assert data["error"] is True
        assert "Unknown content_type" in data["error_message"]

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_save()
        result = await fn(
            title="Missing",
            file_path="/nonexistent/file.txt",
        )

        data = json.loads(result)
        assert data["error"] is True
        assert data["error_type"] == "file_not_found"


# ---------------------------------------------------------------------------
# Artifact Gallery app routes
# ---------------------------------------------------------------------------


class TestArtifactGalleryApp:
    """Tests for the Artifact Gallery FastAPI app."""

    @pytest.fixture
    def app_client(self, tmp_path):
        """Create a test client for the gallery app."""
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app), tmp_path

    def test_health(self, app_client):
        client, _ = app_client
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["artifact_count"] == 0

    def test_list_artifacts_empty(self, app_client):
        client, _ = app_client
        resp = client.get("/api/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["artifacts"] == []

    def test_list_artifacts_with_data(self, app_client):
        client, tmp_path = app_client
        client.app.state.artifact_store.save_object("# Hello", title="Test Artifact")

        resp = client.get("/api/artifacts")
        data = resp.json()
        assert data["count"] == 1

    def test_get_artifact_not_found(self, app_client):
        client, _ = app_client
        resp = client.get("/api/artifacts/nonexistent")
        assert resp.status_code == 404

    def test_serve_file(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = store.save_file(
            file_content=b"<h1>Gallery</h1>",
            filename="test.html",
            artifact_type="html",
            title="Served File",
            mime_type="text/html",
            tool_source="test",
        )

        resp = client.get(f"/files/{entry.id}/{entry.filename}")
        assert resp.status_code == 200
        assert b"<h1>Gallery</h1>" in resp.content

    def test_serve_file_not_found(self, app_client):
        client, _ = app_client
        resp = client.get("/files/fake-id/fake.txt")
        assert resp.status_code == 404

    def test_type_filter(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        store.save_object("# Markdown", title="MD")
        store.save_object({"key": "val"}, title="JSON")

        resp = client.get("/api/artifacts?type=markdown")
        data = resp.json()
        assert data["count"] == 1
        assert data["artifacts"][0]["title"] == "MD"

    def test_search_filter(self, app_client):
        client, _ = app_client
        store = client.app.state.artifact_store
        store.save_object("data", title="Beam Current Analysis")
        store.save_object("data", title="Vacuum Trend")

        resp = client.get("/api/artifacts?search=beam")
        data = resp.json()
        assert data["count"] == 1
        assert "Beam" in data["artifacts"][0]["title"]

    def test_get_focus_empty(self, app_client):
        """GET /api/focus with no artifacts returns artifact: null."""
        client, _ = app_client
        resp = client.get("/api/focus")
        assert resp.status_code == 200
        data = resp.json()
        assert data["focused"] is False
        assert data["artifact"] is None

    def test_get_focus_returns_latest(self, app_client):
        """GET /api/focus returns the most recent artifact with focused: False."""
        client, _ = app_client
        store = client.app.state.artifact_store
        store.save_object("# First", title="First")
        store.save_object("# Second", title="Second")

        resp = client.get("/api/focus")
        data = resp.json()
        assert data["focused"] is False
        assert data["artifact"]["title"] == "Second"

    def test_set_and_get_focus(self, app_client):
        """POST /api/focus sets focus, GET returns it with focused: True."""
        client, _ = app_client
        store = client.app.state.artifact_store
        entry1 = store.save_object("# First", title="First")
        store.save_object("# Second", title="Second")

        # Focus on the first artifact
        resp = client.post("/api/focus", json={"artifact_id": entry1.id})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # GET should return the focused artifact
        resp = client.get("/api/focus")
        data = resp.json()
        assert data["focused"] is True
        assert data["artifact"]["id"] == entry1.id

    def test_set_focus_not_found(self, app_client):
        """POST /api/focus with unknown ID returns 404."""
        client, _ = app_client
        resp = client.post("/api/focus", json={"artifact_id": "nonexistent"})
        assert resp.status_code == 404

    def test_stale_focus_falls_back(self, app_client):
        """Stale focus ID clears and returns latest artifact."""
        client, _ = app_client
        store = client.app.state.artifact_store
        store.save_object("# Only", title="Only Artifact")

        # Set focus to a fake ID (simulating a deleted artifact)
        client.app.state.focused_artifact_id = "deleted-id"

        resp = client.get("/api/focus")
        data = resp.json()
        assert data["focused"] is False
        assert data["artifact"]["title"] == "Only Artifact"
        # Focus should have been cleared
        assert client.app.state.focused_artifact_id is None

    def test_delete_artifact_endpoint(self, app_client):
        """DELETE /api/artifacts/{id} removes artifact and returns ok."""
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = store.save_object("# Delete me", title="To Delete")

        resp = client.delete(f"/api/artifacts/{entry.id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert store.get_entry(entry.id) is None

    def test_delete_artifact_not_found(self, app_client):
        """DELETE /api/artifacts/{id} returns 404 for unknown ID."""
        client, _ = app_client
        resp = client.delete("/api/artifacts/nonexistent")
        assert resp.status_code == 404

    def test_delete_artifact_clears_focus(self, app_client):
        """DELETE /api/artifacts/{id} clears focus if it was the focused artifact."""
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = store.save_object("# Focused", title="Focused Artifact")
        client.app.state.focused_artifact_id = entry.id

        resp = client.delete(f"/api/artifacts/{entry.id}")
        assert resp.status_code == 200
        assert client.app.state.focused_artifact_id is None


# ---------------------------------------------------------------------------
# Gallery visibility — Issue 2 tests
# ---------------------------------------------------------------------------


class TestAutoLaunchLogging:
    """Tests for auto-launch failure logging in artifact_store.py."""

    def test_auto_launch_failure_logged_as_warning(self, tmp_path, caplog):
        """Auto-launch failure produces a warning-level log (not debug)."""
        import logging

        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(workspace_root=tmp_path)

        with (
            patch(
                "osprey.infrastructure.server_launcher.ensure_artifact_server",
                side_effect=RuntimeError("server launch failed"),
            ),
            caplog.at_level(logging.WARNING, logger="osprey.stores.artifact_store"),
        ):
            store.save_file(
                file_content=b"test",
                filename="test.txt",
                artifact_type="text",
                title="Test",
                mime_type="text/plain",
                tool_source="test",
            )

        assert any("auto-launch failed" in r.message.lower() for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)


class TestServerLauncherRetry:
    """Tests for server launcher crash recovery."""

    def test_server_launcher_retries_after_crash(self):
        """_launched resets to False when the server thread crashes."""
        from osprey.infrastructure.server_launcher import ServerLauncher

        def crash_app_factory(**kwargs):
            raise RuntimeError("app factory crash")

        launcher = ServerLauncher(
            name="Test Server",
            config_reader=lambda: ("127.0.0.1", 19999),
            auto_launch_checker=lambda: True,
            app_factory=crash_app_factory,
            pass_workspace=False,
        )

        # First call: sets _launched = True, then thread crashes and resets it
        launcher.ensure_running()

        # Give the thread time to crash and reset _launched
        import time

        time.sleep(1.0)

        # After crash, _launched should be reset to False
        assert launcher._launched is False

    def test_server_launcher_health_check_warning(self, caplog):
        """Health-check failure after launch produces a warning log."""
        import logging

        from osprey.infrastructure.server_launcher import ServerLauncher

        def noop_app_factory(**kwargs):
            # Return something but don't actually start a server
            return None

        launcher = ServerLauncher(
            name="Test Server",
            config_reader=lambda: ("127.0.0.1", 19998),
            auto_launch_checker=lambda: True,
            app_factory=noop_app_factory,
            pass_workspace=False,
        )

        with caplog.at_level(logging.WARNING, logger="osprey.infrastructure.server_launcher"):
            launcher.ensure_running()

        assert any("health check failed" in r.message.lower() for r in caplog.records)
