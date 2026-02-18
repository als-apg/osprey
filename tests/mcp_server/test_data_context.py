"""Tests for the OSPREY DataContext — data management layer for MCP tool outputs.

Covers:
  - DataContext: save, list_entries, get_entry, get_file_path
  - Filtering: tool_filter, data_type_filter, last_n, search
  - Staleness detection: _refresh_if_stale
  - Listener system: register, invoke, error isolation
  - Singleton lifecycle: get_data_context, initialize_data_context, reset_data_context
  - Index persistence across instances
"""

import json
import time
from pathlib import Path

import pytest

from osprey.mcp_server.data_context import (
    DataContext,
    get_data_context,
    initialize_data_context,
    register_context_listener,
    reset_data_context,
    unregister_context_listener,
)


def _save_entry(ctx, tool="channel_read", data_type="channel_values", description="test"):
    """Helper to save a context entry with minimal boilerplate."""
    return ctx.save(
        tool=tool,
        data={"value": 42},
        description=description,
        summary={"count": 1},
        access_details={"format": "json"},
        data_type=data_type,
    )


# ---------------------------------------------------------------------------
# Core save / index
# ---------------------------------------------------------------------------


class TestDataContextSave:
    """Tests for DataContext.save()."""

    def test_save_creates_entry_and_data_file(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        entry = _save_entry(ctx)

        assert entry.id == 1
        assert entry.tool == "channel_read"
        assert entry.data_type == "channel_values"
        assert entry.size_bytes > 0

        # Data file exists on disk
        from pathlib import Path

        data_path = Path(entry.data_file)
        assert data_path.exists()
        payload = json.loads(data_path.read_text())
        assert payload["_osprey_metadata"]["context_entry_id"] == 1
        assert payload["data"]["value"] == 42

    def test_save_updates_index_file(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx)

        index_file = tmp_path / "data_context.json"
        assert index_file.exists()
        index = json.loads(index_file.read_text())
        assert index["version"] == 1
        assert index["entry_count"] == 1
        assert len(index["entries"]) == 1

    def test_save_increments_ids(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        e1 = _save_entry(ctx)
        e2 = _save_entry(ctx, tool="archiver_read")
        e3 = _save_entry(ctx, tool="python_execute")

        assert e1.id == 1
        assert e2.id == 2
        assert e3.id == 3

    def test_to_tool_response(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        entry = _save_entry(ctx)

        resp = entry.to_tool_response()
        assert resp["status"] == "success"
        assert resp["context_entry_id"] == 1
        assert "data_file" in resp
        assert "hint" in resp

    def test_to_dict_roundtrip(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        entry = _save_entry(ctx)

        d = entry.to_dict()
        assert d["id"] == entry.id
        assert d["tool"] == entry.tool
        assert d["data_type"] == entry.data_type


# ---------------------------------------------------------------------------
# list_entries
# ---------------------------------------------------------------------------


class TestListEntries:
    """Tests for DataContext.list_entries()."""

    def test_list_all(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx, tool="channel_read")
        _save_entry(ctx, tool="archiver_read")

        entries = ctx.list_entries()
        assert len(entries) == 2

    def test_filter_by_tool(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx, tool="channel_read")
        _save_entry(ctx, tool="archiver_read")
        _save_entry(ctx, tool="channel_read")

        entries = ctx.list_entries(tool_filter="channel_read")
        assert len(entries) == 2
        assert all(e.tool == "channel_read" for e in entries)

    def test_filter_by_data_type(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx, data_type="channel_values")
        _save_entry(ctx, data_type="timeseries")
        _save_entry(ctx, data_type="channel_values")

        entries = ctx.list_entries(data_type_filter="timeseries")
        assert len(entries) == 1
        assert entries[0].data_type == "timeseries"

    def test_last_n(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        for i in range(5):
            _save_entry(ctx, description=f"entry {i}")

        entries = ctx.list_entries(last_n=2)
        assert len(entries) == 2
        assert entries[0].description == "entry 3"
        assert entries[1].description == "entry 4"

    def test_search_by_description(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx, description="Beam current reading")
        _save_entry(ctx, description="Vacuum pressure trend")

        entries = ctx.list_entries(search="beam")
        assert len(entries) == 1
        assert "Beam" in entries[0].description

    def test_search_by_tool(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx, tool="archiver_read", description="some data")
        _save_entry(ctx, tool="channel_read", description="other data")

        entries = ctx.list_entries(search="archiver")
        assert len(entries) == 1
        assert entries[0].tool == "archiver_read"

    def test_search_case_insensitive(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx, description="BEAM CURRENT")

        entries = ctx.list_entries(search="beam current")
        assert len(entries) == 1

    def test_filter_by_source_agent(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        ctx.save(
            tool="submit_response",
            data={"value": 1},
            description="logbook result",
            summary={"count": 1},
            access_details={"format": "json"},
            data_type="logbook_research",
            source_agent="logbook-search",
        )
        ctx.save(
            tool="submit_response",
            data={"value": 2},
            description="wiki result",
            summary={"count": 1},
            access_details={"format": "json"},
            data_type="wiki_research",
            source_agent="wiki-search",
        )
        ctx.save(
            tool="channel_read",
            data={"value": 3},
            description="channel data",
            summary={"count": 1},
            access_details={"format": "json"},
            data_type="channel_values",
        )

        entries = ctx.list_entries(source_agent_filter="logbook-search")
        assert len(entries) == 1
        assert entries[0].source_agent == "logbook-search"
        assert entries[0].description == "logbook result"

        entries = ctx.list_entries(source_agent_filter="wiki-search")
        assert len(entries) == 1
        assert entries[0].source_agent == "wiki-search"

        # No match
        entries = ctx.list_entries(source_agent_filter="nonexistent")
        assert len(entries) == 0

    def test_combined_filters(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx, tool="channel_read", data_type="channel_values", description="beam")
        _save_entry(ctx, tool="archiver_read", data_type="timeseries", description="beam")
        _save_entry(ctx, tool="channel_read", data_type="channel_values", description="vacuum")

        entries = ctx.list_entries(tool_filter="channel_read", search="beam")
        assert len(entries) == 1
        assert entries[0].tool == "channel_read"
        assert "beam" in entries[0].description


# ---------------------------------------------------------------------------
# get_entry
# ---------------------------------------------------------------------------


class TestGetEntry:
    """Tests for DataContext.get_entry()."""

    def test_found(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        entry = _save_entry(ctx)

        found = ctx.get_entry(entry.id)
        assert found is not None
        assert found.id == entry.id
        assert found.tool == entry.tool

    def test_not_found(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        assert ctx.get_entry(999) is None


# ---------------------------------------------------------------------------
# get_file_path
# ---------------------------------------------------------------------------


class TestGetFilePath:
    """Tests for DataContext.get_file_path()."""

    def test_exists(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        entry = _save_entry(ctx)

        path = ctx.get_file_path(entry.id)
        assert path is not None
        assert path.exists()

    def test_file_deleted(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        entry = _save_entry(ctx)

        # Delete the data file
        from pathlib import Path

        Path(entry.data_file).unlink()

        path = ctx.get_file_path(entry.id)
        assert path is None

    def test_invalid_id(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        assert ctx.get_file_path(999) is None


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------


class TestDeleteEntry:
    """Tests for DataContext.delete_entry()."""

    def test_delete_existing(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        entry = _save_entry(ctx, description="to delete")

        data_path = Path(entry.data_file)
        assert data_path.exists()

        result = ctx.delete_entry(entry.id)
        assert result is True
        assert ctx.get_entry(entry.id) is None
        assert len(ctx.list_entries()) == 0
        assert not data_path.exists()

    def test_delete_nonexistent(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        result = ctx.delete_entry(999)
        assert result is False

    def test_delete_preserves_other_entries(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        e1 = _save_entry(ctx, description="keep")
        e2 = _save_entry(ctx, description="delete")

        ctx.delete_entry(e2.id)
        assert len(ctx.list_entries()) == 1
        assert ctx.get_entry(e1.id) is not None


class TestStalenessDetection:
    """Tests for _refresh_if_stale()."""

    def test_detects_external_write(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        _save_entry(ctx, description="original")

        assert len(ctx.list_entries()) == 1

        # Simulate an external process writing to the index
        # (e.g. another MCP server instance)
        ctx2 = DataContext(workspace_root=tmp_path)
        _save_entry(ctx2, description="external")

        # Ensure mtime differs (some filesystems have 1s resolution)
        time.sleep(0.05)
        index_file = tmp_path / "data_context.json"
        index_file.write_text(index_file.read_text())

        # ctx should detect the stale index on next list_entries
        entries = ctx.list_entries()
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Listener system
# ---------------------------------------------------------------------------


class TestListeners:
    """Tests for context event listeners."""

    def test_listener_called_on_save(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        received = []

        def listener(entry):
            received.append(entry)

        register_context_listener(listener)
        try:
            _save_entry(ctx)
            assert len(received) == 1
            assert received[0].id == 1
        finally:
            unregister_context_listener(listener)

    def test_multiple_listeners(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        calls_a = []
        calls_b = []

        def listener_a(entry):
            calls_a.append(entry)

        def listener_b(entry):
            calls_b.append(entry)

        register_context_listener(listener_a)
        register_context_listener(listener_b)
        try:
            _save_entry(ctx)
            assert len(calls_a) == 1
            assert len(calls_b) == 1
        finally:
            unregister_context_listener(listener_a)
            unregister_context_listener(listener_b)

    def test_listener_error_isolation(self, tmp_path):
        """A failing listener must not prevent save from succeeding."""
        ctx = DataContext(workspace_root=tmp_path)

        def bad_listener(entry):
            raise RuntimeError("boom")

        register_context_listener(bad_listener)
        try:
            entry = _save_entry(ctx)
            # Save should still succeed
            assert entry.id == 1
            assert ctx.get_entry(1) is not None
        finally:
            unregister_context_listener(bad_listener)

    def test_unregister_listener(self, tmp_path):
        ctx = DataContext(workspace_root=tmp_path)
        calls = []

        def listener(entry):
            calls.append(entry)

        register_context_listener(listener)
        _save_entry(ctx)
        assert len(calls) == 1

        unregister_context_listener(listener)
        _save_entry(ctx)
        assert len(calls) == 1  # No new calls


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for singleton management functions."""

    def test_get_data_context_lazy_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        ctx1 = get_data_context()
        ctx2 = get_data_context()
        assert ctx1 is ctx2

    def test_reset_clears_singleton(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        ctx1 = get_data_context()
        reset_data_context()
        ctx2 = get_data_context()
        assert ctx2 is not ctx1

    def test_initialize_with_workspace(self, tmp_path):
        ctx = initialize_data_context(workspace_root=tmp_path)
        assert ctx is get_data_context()


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------


class TestIndexPersistence:
    """Tests for index survival across DataContext instances."""

    def test_entries_restored_from_disk(self, tmp_path):
        ctx1 = DataContext(workspace_root=tmp_path)
        e1 = _save_entry(ctx1, tool="channel_read", description="first")
        e2 = _save_entry(ctx1, tool="archiver_read", description="second")

        # Create a fresh instance from the same directory
        ctx2 = DataContext(workspace_root=tmp_path)
        entries = ctx2.list_entries()
        assert len(entries) == 2
        assert entries[0].id == e1.id
        assert entries[1].id == e2.id
        assert entries[0].description == "first"

    def test_id_continues_after_reload(self, tmp_path):
        ctx1 = DataContext(workspace_root=tmp_path)
        _save_entry(ctx1)
        _save_entry(ctx1)

        ctx2 = DataContext(workspace_root=tmp_path)
        e3 = _save_entry(ctx2)
        assert e3.id == 3


# ---------------------------------------------------------------------------
# Cross-process write safety
# ---------------------------------------------------------------------------


class TestCrossProcessSafety:
    """Tests for cross-process file-locking in DataContext."""

    def test_save_refreshes_before_id_assignment(self, tmp_path):
        """Two DataContext instances on same workspace: second save gets correct ID."""
        ctx_a = DataContext(workspace_root=tmp_path)
        ctx_b = DataContext(workspace_root=tmp_path)

        e1 = _save_entry(ctx_a, description="from A")
        assert e1.id == 1

        # ctx_b still has _next_id=1 in memory, but the lock should reload
        e2 = _save_entry(ctx_b, description="from B")
        assert e2.id == 2

        # Both entries in the index
        index = json.loads((tmp_path / "data_context.json").read_text())
        assert index["entry_count"] == 2

    def test_concurrent_saves_no_orphan_files(self, tmp_path):
        """Both instances save — all data files are referenced in the index."""
        ctx_a = DataContext(workspace_root=tmp_path)
        ctx_b = DataContext(workspace_root=tmp_path)

        _save_entry(ctx_a, tool="channel_read", description="A")
        _save_entry(ctx_b, tool="archiver_read", description="B")

        index = json.loads((tmp_path / "data_context.json").read_text())
        index_files = {e["data_file"] for e in index["entries"]}

        data_dir = tmp_path / "data"
        disk_files = {str(f) for f in data_dir.iterdir() if f.suffix == ".json"}

        assert disk_files == index_files

    def test_delete_concurrent_with_save(self, tmp_path):
        """Delete from one instance while another saves — no data loss."""
        ctx_a = DataContext(workspace_root=tmp_path)
        e1 = _save_entry(ctx_a, description="to delete")

        ctx_b = DataContext(workspace_root=tmp_path)
        ctx_c = DataContext(workspace_root=tmp_path)

        ctx_b.delete_entry(e1.id)
        e2 = _save_entry(ctx_c, description="new entry")

        # Reload and verify
        ctx_check = DataContext(workspace_root=tmp_path)
        entries = ctx_check.list_entries()
        assert len(entries) == 1
        assert entries[0].id == e2.id
        assert entries[0].description == "new entry"
