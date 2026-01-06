"""Tests for Memory Storage Manager."""

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from osprey.services.memory_storage.models import MemoryContent
from osprey.services.memory_storage.storage_manager import (
    MemoryStorageManager,
    get_memory_storage_manager,
)


class TestMemoryStorageManagerInit:
    """Test MemoryStorageManager initialization."""

    def test_init_creates_directory(self, tmp_path):
        """MemoryStorageManager should create directory if it doesn't exist."""
        memory_dir = tmp_path / "new_memory_dir"
        assert not memory_dir.exists()

        manager = MemoryStorageManager(str(memory_dir))

        assert memory_dir.exists()
        assert manager.memory_dir == memory_dir

    def test_init_accepts_existing_directory(self, tmp_path):
        """MemoryStorageManager should accept existing directory."""
        memory_dir = tmp_path / "existing_dir"
        memory_dir.mkdir()

        manager = MemoryStorageManager(str(memory_dir))

        assert manager.memory_dir == memory_dir

    def test_init_creates_nested_directories(self, tmp_path):
        """MemoryStorageManager should create nested directories."""
        memory_dir = tmp_path / "level1" / "level2" / "memory"
        assert not memory_dir.exists()

        MemoryStorageManager(str(memory_dir))

        assert memory_dir.exists()


class TestGetMemoryFilePath:
    """Test _get_memory_file_path method."""

    def test_sanitizes_user_id(self, tmp_path):
        """_get_memory_file_path should sanitize user_id for filename."""
        manager = MemoryStorageManager(str(tmp_path))

        # User ID with special characters
        user_id = "user@example.com"
        path = manager._get_memory_file_path(user_id)

        # Should only contain safe characters
        assert path.name == "userexamplecom.json"

    def test_preserves_alphanumeric_and_safe_chars(self, tmp_path):
        """_get_memory_file_path should preserve alphanumeric, dash, and underscore."""
        manager = MemoryStorageManager(str(tmp_path))

        user_id = "user_123-test"
        path = manager._get_memory_file_path(user_id)

        assert path.name == "user_123-test.json"

    def test_raises_error_for_empty_user_id(self, tmp_path):
        """_get_memory_file_path should raise ValueError for empty user_id."""
        manager = MemoryStorageManager(str(tmp_path))

        with pytest.raises(ValueError, match="User ID cannot be empty"):
            manager._get_memory_file_path("")

        with pytest.raises(ValueError, match="User ID cannot be empty"):
            manager._get_memory_file_path("   ")


class TestLoadMemoryData:
    """Test _load_memory_data method."""

    def test_loads_existing_memory_file(self, tmp_path):
        """_load_memory_data should load entries from existing file."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        # Create memory file manually
        memory_file = tmp_path / "user123.json"
        data = {
            "user_id": user_id,
            "last_updated": "2024-01-15 10:00",
            "entries": [
                {"timestamp": "2024-01-15 09:00", "content": "Test memory 1"},
                {"timestamp": "2024-01-15 10:00", "content": "Test memory 2"},
            ],
        }
        memory_file.write_text(json.dumps(data))

        entries = manager._load_memory_data(user_id)

        assert len(entries) == 2
        assert entries[0]["content"] == "Test memory 1"
        assert entries[1]["content"] == "Test memory 2"

    def test_returns_empty_list_for_nonexistent_file(self, tmp_path):
        """_load_memory_data should return empty list if file doesn't exist."""
        manager = MemoryStorageManager(str(tmp_path))

        entries = manager._load_memory_data("nonexistent_user")

        assert entries == []

    def test_handles_corrupt_json_gracefully(self, tmp_path):
        """_load_memory_data should handle corrupt JSON gracefully."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        # Create corrupt JSON file
        memory_file = tmp_path / "user123.json"
        memory_file.write_text("{ invalid json }")

        entries = manager._load_memory_data(user_id)

        assert entries == []  # Returns empty list instead of crashing


class TestSaveMemoryData:
    """Test _save_memory_data method."""

    def test_saves_memory_entries_to_file(self, tmp_path):
        """_save_memory_data should save entries to JSON file."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        entries = [
            {"timestamp": "2024-01-15 09:00", "content": "Memory 1"},
            {"timestamp": "2024-01-15 10:00", "content": "Memory 2"},
        ]

        success = manager._save_memory_data(user_id, entries)

        assert success is True

        # Verify file contents
        memory_file = tmp_path / "user123.json"
        assert memory_file.exists()

        data = json.loads(memory_file.read_text())
        assert data["user_id"] == user_id
        assert len(data["entries"]) == 2
        assert data["entries"][0]["content"] == "Memory 1"

    def test_updates_last_updated_timestamp(self, tmp_path):
        """_save_memory_data should update last_updated timestamp."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        manager._save_memory_data(user_id, [])

        memory_file = tmp_path / "user123.json"
        data = json.loads(memory_file.read_text())

        assert "last_updated" in data
        # Should be recent timestamp
        assert data["last_updated"] is not None


class TestGetUserMemory:
    """Test get_user_memory method."""

    def test_returns_formatted_memory_string(self, tmp_path):
        """get_user_memory should return formatted memory string."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        # Add some entries
        entries = [
            {"timestamp": "2024-01-15 09:00", "content": "First memory"},
            {"timestamp": "2024-01-15 10:00", "content": "Second memory"},
        ]
        manager._save_memory_data(user_id, entries)

        memory_text = manager.get_user_memory(user_id)

        assert "[2024-01-15 09:00] First memory" in memory_text
        assert "[2024-01-15 10:00] Second memory" in memory_text

    def test_returns_empty_string_for_no_memory(self, tmp_path):
        """get_user_memory should return empty string if no memory exists."""
        manager = MemoryStorageManager(str(tmp_path))

        memory_text = manager.get_user_memory("nonexistent_user")

        assert memory_text == ""


class TestGetAllMemoryEntries:
    """Test get_all_memory_entries method."""

    def test_returns_memory_content_objects(self, tmp_path):
        """get_all_memory_entries should return MemoryContent objects."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        entries = [
            {"timestamp": "2024-01-15 09:00", "content": "Test memory"},
        ]
        manager._save_memory_data(user_id, entries)

        memory_contents = manager.get_all_memory_entries(user_id)

        assert len(memory_contents) == 1
        assert isinstance(memory_contents[0], MemoryContent)
        assert memory_contents[0].content == "Test memory"

    def test_parses_timestamps_correctly(self, tmp_path):
        """get_all_memory_entries should parse timestamps correctly."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        entries = [
            {"timestamp": "2024-01-15 09:30", "content": "Test"},
        ]
        manager._save_memory_data(user_id, entries)

        memory_contents = manager.get_all_memory_entries(user_id)

        assert memory_contents[0].timestamp == datetime(2024, 1, 15, 9, 30)

    def test_handles_invalid_timestamps_gracefully(self, tmp_path):
        """get_all_memory_entries should handle invalid timestamps with fallback."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        entries = [
            {"timestamp": "invalid-timestamp", "content": "Test memory"},
        ]
        manager._save_memory_data(user_id, entries)

        memory_contents = manager.get_all_memory_entries(user_id)

        # Should still return the entry with current time as fallback
        assert len(memory_contents) == 1
        assert memory_contents[0].content == "Test memory"

    def test_filters_empty_content(self, tmp_path):
        """get_all_memory_entries should filter out empty content."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        entries = [
            {"timestamp": "2024-01-15 09:00", "content": "Valid memory"},
            {"timestamp": "2024-01-15 10:00", "content": ""},
            {"timestamp": "2024-01-15 11:00", "content": "   "},
        ]
        manager._save_memory_data(user_id, entries)

        memory_contents = manager.get_all_memory_entries(user_id)

        assert len(memory_contents) == 1
        assert memory_contents[0].content == "Valid memory"


class TestAddMemoryEntry:
    """Test add_memory_entry method."""

    def test_adds_new_memory_entry(self, tmp_path):
        """add_memory_entry should add new entry to existing memories."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        new_entry = MemoryContent(timestamp=datetime(2024, 1, 15, 12, 0), content="New memory")

        success = manager.add_memory_entry(user_id, new_entry)

        assert success is True

        # Verify entry was added
        entries = manager.get_all_memory_entries(user_id)
        assert len(entries) == 1
        assert entries[0].content == "New memory"

    def test_appends_to_existing_entries(self, tmp_path):
        """add_memory_entry should append to existing entries."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        # Add first entry
        entry1 = MemoryContent(timestamp=datetime(2024, 1, 15, 9, 0), content="First")
        manager.add_memory_entry(user_id, entry1)

        # Add second entry
        entry2 = MemoryContent(timestamp=datetime(2024, 1, 15, 10, 0), content="Second")
        manager.add_memory_entry(user_id, entry2)

        # Verify both exist
        entries = manager.get_all_memory_entries(user_id)
        assert len(entries) == 2

    def test_strips_whitespace_from_content(self, tmp_path):
        """add_memory_entry should strip whitespace from content."""
        manager = MemoryStorageManager(str(tmp_path))
        user_id = "user123"

        entry = MemoryContent(
            timestamp=datetime(2024, 1, 15, 12, 0), content="  Content with spaces  "
        )

        manager.add_memory_entry(user_id, entry)

        entries = manager.get_all_memory_entries(user_id)
        assert entries[0].content == "Content with spaces"


class TestGetMemoryStorageManager:
    """Test get_memory_storage_manager global instance function."""

    @patch("osprey.services.memory_storage.storage_manager.get_agent_dir")
    def test_creates_singleton_instance(self, mock_get_agent_dir, tmp_path):
        """get_memory_storage_manager should create and cache singleton instance."""
        mock_get_agent_dir.return_value = str(tmp_path / "memory")

        # Reset global state
        import osprey.services.memory_storage.storage_manager as sm

        sm._memory_storage_manager = None

        # First call creates instance
        manager1 = get_memory_storage_manager()
        assert manager1 is not None

        # Second call returns same instance
        manager2 = get_memory_storage_manager()
        assert manager2 is manager1

    @patch("osprey.services.memory_storage.storage_manager.get_agent_dir")
    def test_uses_configured_directory(self, mock_get_agent_dir, tmp_path):
        """get_memory_storage_manager should use configured directory."""
        expected_dir = str(tmp_path / "configured_memory")
        mock_get_agent_dir.return_value = expected_dir

        # Reset global state
        import osprey.services.memory_storage.storage_manager as sm

        sm._memory_storage_manager = None

        manager = get_memory_storage_manager()

        assert str(manager.memory_dir) == expected_dir
        mock_get_agent_dir.assert_called_once_with("user_memory_dir")
