"""Tests for ArtifactEntry highlighted/pinned fields.

Covers:
  - New entries default to highlighted=True, pinned=False
  - set_pinned / set_highlighted toggle correctly
  - list_entries filtering by highlighted/pinned
  - Backward compat: old index files without highlighted/pinned load fine
  - to_tool_response includes highlighted/pinned
"""

import json

import pytest

from osprey.mcp_server.artifact_store import ArtifactEntry, ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(workspace_root=tmp_path)


def _save(store, title="Test", highlighted=True):
    return store.save_file(
        file_content=b"hello",
        filename="test.txt",
        artifact_type="text",
        title=title,
        description="desc",
        mime_type="text/plain",
        tool_source="test",
        highlighted=highlighted,
    )


class TestHighlightedPinned:
    @pytest.mark.unit
    def test_new_entry_defaults(self, store):
        """New artifacts are highlighted=True, pinned=False by default."""
        entry = _save(store)
        assert entry.highlighted is True
        assert entry.pinned is False

    @pytest.mark.unit
    def test_save_with_highlighted_false(self, store):
        """Can explicitly save with highlighted=False."""
        entry = _save(store, highlighted=False)
        assert entry.highlighted is False

    @pytest.mark.unit
    def test_set_pinned(self, store):
        """set_pinned toggles the pin flag and persists."""
        entry = _save(store)
        assert entry.pinned is False

        updated = store.set_pinned(entry.id, True)
        assert updated is not None
        assert updated.pinned is True

        # Verify it persists across reload
        store2 = ArtifactStore(workspace_root=store._workspace)
        reloaded = store2.get_entry(entry.id)
        assert reloaded.pinned is True

    @pytest.mark.unit
    def test_set_highlighted(self, store):
        """set_highlighted toggles the highlight flag and persists."""
        entry = _save(store)
        assert entry.highlighted is True

        updated = store.set_highlighted(entry.id, False)
        assert updated is not None
        assert updated.highlighted is False

        store2 = ArtifactStore(workspace_root=store._workspace)
        reloaded = store2.get_entry(entry.id)
        assert reloaded.highlighted is False

    @pytest.mark.unit
    def test_set_pinned_not_found(self, store):
        """set_pinned on nonexistent ID returns None."""
        assert store.set_pinned("nonexistent", True) is None

    @pytest.mark.unit
    def test_set_highlighted_not_found(self, store):
        """set_highlighted on nonexistent ID returns None."""
        assert store.set_highlighted("nonexistent", True) is None

    @pytest.mark.unit
    def test_list_filter_highlighted(self, store):
        """list_entries(highlighted=True) returns only highlighted."""
        e1 = _save(store, title="A", highlighted=True)
        e2 = _save(store, title="B", highlighted=False)

        highlighted = store.list_entries(highlighted=True)
        assert len(highlighted) == 1
        assert highlighted[0].id == e1.id

        not_highlighted = store.list_entries(highlighted=False)
        assert len(not_highlighted) == 1
        assert not_highlighted[0].id == e2.id

    @pytest.mark.unit
    def test_list_filter_pinned(self, store):
        """list_entries(pinned=True) returns only pinned."""
        e1 = _save(store, title="A")
        e2 = _save(store, title="B")
        store.set_pinned(e1.id, True)

        pinned = store.list_entries(pinned=True)
        assert len(pinned) == 1
        assert pinned[0].id == e1.id

    @pytest.mark.unit
    def test_list_filter_combined(self, store):
        """Can combine highlighted and pinned filters."""
        e1 = _save(store, title="A", highlighted=True)
        e2 = _save(store, title="B", highlighted=False)
        store.set_pinned(e1.id, True)

        both = store.list_entries(highlighted=True, pinned=True)
        assert len(both) == 1
        assert both[0].id == e1.id

    @pytest.mark.unit
    def test_to_tool_response_includes_fields(self, store):
        """to_tool_response includes highlighted and pinned."""
        entry = _save(store)
        store.set_pinned(entry.id, True)

        refreshed = store.get_entry(entry.id)
        resp = refreshed.to_tool_response(gallery_url="http://localhost:8086")
        assert resp["highlighted"] is True
        assert resp["pinned"] is True
        assert "gallery_url" in resp

    @pytest.mark.unit
    def test_to_dict_includes_fields(self, store):
        """to_dict includes highlighted and pinned."""
        entry = _save(store)
        d = entry.to_dict()
        assert "highlighted" in d
        assert "pinned" in d
        assert d["highlighted"] is True
        assert d["pinned"] is False


class TestBackwardCompat:
    @pytest.mark.unit
    def test_old_index_without_fields(self, tmp_path):
        """Index files from before highlighted/pinned still load correctly."""
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir(parents=True)

        # Write an old-format index without highlighted/pinned
        old_entry = {
            "id": "abc123",
            "artifact_type": "text",
            "title": "Old artifact",
            "description": "from before migration",
            "filename": "abc123_old.txt",
            "mime_type": "text/plain",
            "size_bytes": 5,
            "timestamp": "2024-01-01T00:00:00",
            "tool_source": "test",
            "metadata": {},
        }
        index_data = {"version": 1, "entry_count": 1, "entries": [old_entry]}
        (art_dir / "artifacts.json").write_text(json.dumps(index_data))
        (art_dir / "abc123_old.txt").write_text("hello")

        store = ArtifactStore(workspace_root=tmp_path)
        entries = store.list_entries()
        assert len(entries) == 1
        assert entries[0].highlighted is False
        assert entries[0].pinned is False
        assert entries[0].title == "Old artifact"
