"""Tests for the OSPREY MemoryStore — persistent session memory for MCP tools.

Covers:
  - MemoryStore: save, list_entries, get_entry, update_entry, delete_entry
  - Filtering: type_filter, tags, importance, search, last_n
  - Entry serialization: to_dict, to_tool_response
  - Backward compatibility: migration from old flat-list format
  - Staleness detection: _refresh_if_stale
  - Listener system: register, invoke, error isolation
  - Singleton lifecycle: get_memory_store, initialize_memory_store, reset_memory_store
"""

import json
import time

import pytest

from osprey.mcp_server.memory_store import (
    MemoryStore,
    get_memory_store,
    initialize_memory_store,
    register_memory_listener,
    reset_memory_store,
    unregister_memory_listener,
)


def _save_note(
    store,
    content="test",
    memory_type="note",
    tags=None,
    importance="normal",
    linked_artifact_id=None,
    linked_label=None,
):
    """Helper to save a memory entry with minimal boilerplate."""
    return store.save(
        content=content,
        memory_type=memory_type,
        tags=tags or [],
        importance=importance,
        linked_artifact_id=linked_artifact_id,
        linked_label=linked_label,
    )


# ---------------------------------------------------------------------------
# Core save / index
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryStoreSave:
    """Tests for MemoryStore.save()."""

    def test_save_note_basic(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        entry = _save_note(store, content="Remember beam current was 500 mA")

        assert entry.id == 1
        assert entry.memory_type == "note"
        assert entry.content == "Remember beam current was 500 mA"
        assert entry.tags == []
        assert entry.importance == "normal"

    def test_save_pin_with_links(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        entry = _save_note(
            store,
            content="Important plot of beam lifetime",
            memory_type="pin",
            linked_artifact_id="abc123",
            linked_label="Beam Lifetime Plot",
        )

        assert entry.memory_type == "pin"
        assert entry.linked_artifact_id == "abc123"
        assert entry.linked_label == "Beam Lifetime Plot"

    def test_save_increments_ids(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        e1 = _save_note(store, content="first")
        e2 = _save_note(store, content="second")
        e3 = _save_note(store, content="third")

        assert e1.id == 1
        assert e2.id == 2
        assert e3.id == 3

    def test_to_dict_roundtrip(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        entry = _save_note(
            store,
            content="test content",
            memory_type="note",
            tags=["beam", "current"],
            importance="high",
        )

        d = entry.to_dict()
        assert d["id"] == entry.id
        assert d["content"] == "test content"
        assert d["memory_type"] == "note"
        assert d["tags"] == ["beam", "current"]
        assert d["importance"] == "high"
        assert "timestamp" in d

    def test_to_tool_response(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        entry = _save_note(store, content="important finding")

        resp = entry.to_tool_response()
        assert resp["status"] == "success"
        assert resp["memory_id"] == 1
        assert "content" in resp or "content_preview" in resp


# ---------------------------------------------------------------------------
# list_entries
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListEntries:
    """Tests for MemoryStore.list_entries()."""

    def test_list_all(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        _save_note(store, content="first")
        _save_note(store, content="second")

        entries = store.list_entries()
        assert len(entries) == 2

    def test_filter_by_type(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        _save_note(store, content="a note", memory_type="note")
        _save_note(store, content="a pin", memory_type="pin")
        _save_note(store, content="another note", memory_type="note")

        entries = store.list_entries(memory_type="pin")
        assert len(entries) == 1
        assert entries[0].memory_type == "pin"

    def test_filter_by_tags(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        _save_note(store, content="beam data", tags=["beam", "current"])
        _save_note(store, content="vacuum data", tags=["vacuum"])
        _save_note(store, content="beam lifetime", tags=["beam", "lifetime"])

        # Any-match semantics: entries with at least one matching tag
        entries = store.list_entries(tags=["beam"])
        assert len(entries) == 2
        assert all("beam" in e.tags for e in entries)

    def test_filter_by_importance(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        _save_note(store, content="routine", importance="normal")
        _save_note(store, content="critical finding", importance="high")
        _save_note(store, content="another routine", importance="normal")

        entries = store.list_entries(importance="high")
        assert len(entries) == 1
        assert entries[0].content == "critical finding"

    def test_search_content(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        _save_note(store, content="Beam current was 500 mA")
        _save_note(store, content="Vacuum pressure nominal")

        entries = store.list_entries(search="beam")
        assert len(entries) == 1
        assert "Beam" in entries[0].content

    def test_last_n(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        for i in range(5):
            _save_note(store, content=f"entry {i}")

        entries = store.list_entries(last_n=2)
        assert len(entries) == 2
        assert entries[0].content == "entry 3"
        assert entries[1].content == "entry 4"


# ---------------------------------------------------------------------------
# get_entry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetEntry:
    """Tests for MemoryStore.get_entry()."""

    def test_found(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        entry = _save_note(store, content="findable")

        found = store.get_entry(entry.id)
        assert found is not None
        assert found.id == entry.id
        assert found.content == "findable"

    def test_not_found(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        assert store.get_entry(999) is None


# ---------------------------------------------------------------------------
# update_entry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpdateEntry:
    """Tests for MemoryStore.update_entry()."""

    def test_update_content(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        entry = _save_note(store, content="original")

        updated = store.update_entry(entry.id, content="updated content")
        assert updated is not None
        assert updated.content == "updated content"

        # Verify the update persisted
        fetched = store.get_entry(entry.id)
        assert fetched is not None
        assert fetched.content == "updated content"

    def test_update_nonexistent(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        result = store.update_entry(999, content="does not exist")
        assert result is None


# ---------------------------------------------------------------------------
# delete_entry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteEntry:
    """Tests for MemoryStore.delete_entry()."""

    def test_delete_existing(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        entry = _save_note(store, content="to delete")

        result = store.delete_entry(entry.id)
        assert result is True

        # Verify it's gone
        assert store.get_entry(entry.id) is None
        assert len(store.list_entries()) == 0

    def test_delete_nonexistent(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        result = store.delete_entry(999)
        assert result is False


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackwardCompat:
    """Tests for migration from old flat-list memory format."""

    def test_migrate_old_format(self, tmp_path):
        # Write the old flat-list format that memory.py used to produce
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True)
        old_memories = [
            {
                "id": 1,
                "content": "Beam was at 500 mA",
                "category": "finding",
                "timestamp": "2024-06-01T12:00:00+00:00",
            },
            {
                "id": 2,
                "content": "Vacuum interlock tripped",
                "category": "procedure",
                "timestamp": "2024-06-01T13:00:00+00:00",
            },
        ]
        memories_file = memory_dir / "memories.json"
        memories_file.write_text(json.dumps(old_memories, indent=2))

        # Create a MemoryStore pointing at this workspace
        store = MemoryStore(workspace_root=tmp_path)

        # Verify old entries were migrated
        entries = store.list_entries()
        assert len(entries) == 2
        assert entries[0].content == "Beam was at 500 mA"
        assert entries[1].content == "Vacuum interlock tripped"


# ---------------------------------------------------------------------------
# Listener system
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListeners:
    """Tests for memory event listeners."""

    def test_listener_called_on_save(self, tmp_path):
        store = MemoryStore(workspace_root=tmp_path)
        received = []

        def listener(entry):
            received.append(entry)

        register_memory_listener(listener)
        try:
            _save_note(store, content="listen to this")
            assert len(received) == 1
            assert received[0].id == 1
        finally:
            unregister_memory_listener(listener)

    def test_listener_error_isolation(self, tmp_path):
        """A failing listener must not prevent save from succeeding."""
        store = MemoryStore(workspace_root=tmp_path)

        def bad_listener(entry):
            raise RuntimeError("boom")

        register_memory_listener(bad_listener)
        try:
            entry = _save_note(store, content="should still save")
            # Save should still succeed
            assert entry.id == 1
            assert store.get_entry(1) is not None
        finally:
            unregister_memory_listener(bad_listener)


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStaleness:
    """Tests for _refresh_if_stale()."""

    def test_detects_external_write(self, tmp_path):
        store1 = MemoryStore(workspace_root=tmp_path)
        _save_note(store1, content="original")

        assert len(store1.list_entries()) == 1

        # Simulate an external process writing to the index
        store2 = MemoryStore(workspace_root=tmp_path)
        _save_note(store2, content="external")

        # Ensure mtime differs (some filesystems have 1s resolution)
        time.sleep(0.05)
        index_file = tmp_path / "memory" / "memories.json"
        index_file.write_text(index_file.read_text())

        # store1 should detect the stale index on next list_entries
        entries = store1.list_entries()
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleton:
    """Tests for singleton management functions."""

    def test_get_memory_store_lazy_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        reset_memory_store()

        store1 = get_memory_store()
        store2 = get_memory_store()
        assert store1 is store2

    def test_reset_clears_singleton(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        reset_memory_store()

        store1 = get_memory_store()
        reset_memory_store()
        store2 = get_memory_store()
        assert store2 is not store1


# ---------------------------------------------------------------------------
# Cross-process write safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossProcessSafety:
    """Tests for cross-process file-locking in MemoryStore."""

    def test_save_refreshes_before_id_assignment(self, tmp_path):
        """Two MemoryStore instances on same workspace: second save gets correct ID."""
        store_a = MemoryStore(workspace_root=tmp_path)
        store_b = MemoryStore(workspace_root=tmp_path)

        e1 = _save_note(store_a, content="from A")
        assert e1.id == 1

        e2 = _save_note(store_b, content="from B")
        assert e2.id == 2

        index = json.loads((tmp_path / "memory" / "memories.json").read_text())
        assert index["entry_count"] == 2

    def test_concurrent_saves_no_data_loss(self, tmp_path):
        """Both instances save — all entries present in the index."""
        store_a = MemoryStore(workspace_root=tmp_path)
        store_b = MemoryStore(workspace_root=tmp_path)

        _save_note(store_a, content="note A")
        _save_note(store_b, content="note B")

        store_check = MemoryStore(workspace_root=tmp_path)
        entries = store_check.list_entries()
        assert len(entries) == 2
        contents = {e.content for e in entries}
        assert contents == {"note A", "note B"}

    def test_delete_concurrent_with_save(self, tmp_path):
        """Delete from one instance while another saves — no data loss."""
        store_a = MemoryStore(workspace_root=tmp_path)
        e1 = _save_note(store_a, content="to delete")

        store_b = MemoryStore(workspace_root=tmp_path)
        store_c = MemoryStore(workspace_root=tmp_path)

        store_b.delete_entry(e1.id)
        e2 = _save_note(store_c, content="new note")

        store_check = MemoryStore(workspace_root=tmp_path)
        entries = store_check.list_entries()
        assert len(entries) == 1
        assert entries[0].id == e2.id
        assert entries[0].content == "new note"

    def test_update_concurrent_with_save(self, tmp_path):
        """Update from one instance while another saves — both persist."""
        store_a = MemoryStore(workspace_root=tmp_path)
        e1 = _save_note(store_a, content="original")

        store_b = MemoryStore(workspace_root=tmp_path)
        store_c = MemoryStore(workspace_root=tmp_path)

        store_b.update_entry(e1.id, content="updated")
        e2 = _save_note(store_c, content="new note")

        store_check = MemoryStore(workspace_root=tmp_path)
        entries = store_check.list_entries()
        assert len(entries) == 2
        assert entries[0].content == "updated"
        assert entries[1].id == e2.id
