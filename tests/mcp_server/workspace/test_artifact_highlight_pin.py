"""Tests for ArtifactEntry pinned field.

Covers:
  - New entries default to pinned=False
  - set_pinned toggles correctly
  - list_entries filtering by pinned
  - Backward compat: old index files without pinned load fine
  - to_tool_response includes pinned
"""

import json

import pytest

from osprey.stores.artifact_store import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(workspace_root=tmp_path)


def _save(store, title="Test"):
    return store.save_file(
        file_content=b"hello",
        filename="test.txt",
        artifact_type="text",
        title=title,
        description="desc",
        mime_type="text/plain",
        tool_source="test",
    )


class TestPinned:
    @pytest.mark.unit
    def test_new_entry_defaults(self, store):
        """New artifacts are pinned=False by default."""
        entry = _save(store)
        assert entry.pinned is False

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
    def test_set_pinned_not_found(self, store):
        """set_pinned on nonexistent ID returns None."""
        assert store.set_pinned("nonexistent", True) is None

    @pytest.mark.unit
    def test_list_filter_pinned(self, store):
        """list_entries(pinned=True) returns only pinned."""
        e1 = _save(store, title="A")
        _save(store, title="B")
        store.set_pinned(e1.id, True)

        pinned = store.list_entries(pinned=True)
        assert len(pinned) == 1
        assert pinned[0].id == e1.id

    @pytest.mark.unit
    def test_to_tool_response_includes_pinned(self, store):
        """to_tool_response includes pinned."""
        entry = _save(store)
        store.set_pinned(entry.id, True)

        refreshed = store.get_entry(entry.id)
        resp = refreshed.to_tool_response(gallery_url="http://localhost:8086")
        assert resp["pinned"] is True
        assert "gallery_url" in resp

    @pytest.mark.unit
    def test_to_dict_includes_pinned(self, store):
        """to_dict includes pinned."""
        entry = _save(store)
        d = entry.to_dict()
        assert "pinned" in d
        assert d["pinned"] is False


class TestBackwardCompat:
    @pytest.mark.unit
    def test_old_index_without_fields(self, tmp_path):
        """Index files from before pinned still load correctly."""
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir(parents=True)

        # Write an old-format index without pinned (and with old highlighted field)
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
            "highlighted": True,  # Old field — should be stripped on load
        }
        index_data = {"version": 1, "entry_count": 1, "entries": [old_entry]}
        (art_dir / "artifacts.json").write_text(json.dumps(index_data))
        (art_dir / "abc123_old.txt").write_text("hello")

        store = ArtifactStore(workspace_root=tmp_path)
        entries = store.list_entries()
        assert len(entries) == 1
        assert entries[0].pinned is False
        assert entries[0].title == "Old artifact"
        # Verify highlighted field was stripped (not on the dataclass)
        assert not hasattr(entries[0], "highlighted") or "highlighted" not in vars(entries[0])
