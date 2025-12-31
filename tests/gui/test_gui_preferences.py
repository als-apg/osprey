"""
Tests for GUIPreferences

Tests the GUI preferences management functionality including loading,
saving, getting, setting, and resetting preferences.
"""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from osprey.interfaces.pyqt.gui_preferences import GUIPreferences


class TestGUIPreferencesInitialization:
    """Test suite for GUIPreferences initialization."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_init_creates_paths(self, mock_exists, mock_home):
        """Test initialization creates correct paths."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()

        assert prefs.preferences_dir == Path("/mock/home") / ".osprey"
        assert prefs.preferences_path == Path("/mock/home") / ".osprey" / "gui_preferences.yml"

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_init_loads_defaults_when_no_file(self, mock_exists, mock_home):
        """Test initialization uses defaults when no preferences file exists."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()

        # Check that defaults are loaded
        assert prefs.preferences == prefs.defaults
        assert prefs.get("use_persistent_conversations") is True
        assert prefs.get("memory_monitor_enabled") is True

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch(
        "builtins.open", new_callable=mock_open, read_data="use_persistent_conversations: false\n"
    )
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_init_loads_from_file(self, mock_exists, mock_file, mock_home):
        """Test initialization loads preferences from file."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = True

        prefs = GUIPreferences()

        # Check that file was read
        mock_file.assert_called()
        # Check that loaded value overrides default
        assert prefs.get("use_persistent_conversations") is False

    def test_init_has_all_default_keys(self):
        """Test initialization includes all expected default keys."""
        with (
            patch("osprey.interfaces.pyqt.gui_preferences.Path.home") as mock_home,
            patch("osprey.interfaces.pyqt.gui_preferences.Path.exists") as mock_exists,
        ):
            mock_home.return_value = Path("/mock/home")
            mock_exists.return_value = False

            prefs = GUIPreferences()

            expected_keys = {
                "use_persistent_conversations",
                "conversation_storage_mode",
                "redirect_output_to_gui",
                "suppress_terminal_output",
                "group_system_messages",
                "enable_routing_feedback",
                "memory_monitor_enabled",
                "memory_warning_threshold_mb",
                "memory_critical_threshold_mb",
                "memory_check_interval_seconds",
            }

            assert set(prefs.defaults.keys()) == expected_keys


class TestLoadPreferences:
    """Test suite for loading preferences."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_load_returns_defaults_when_file_missing(self, mock_exists, mock_home):
        """Test loading returns defaults when file doesn't exist."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        loaded = prefs._load_preferences()

        assert loaded == prefs.defaults

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("builtins.open", new_callable=mock_open, read_data="memory_monitor_enabled: false\n")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_load_merges_with_defaults(self, mock_exists, mock_file, mock_home):
        """Test loading merges file content with defaults."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = True

        prefs = GUIPreferences()

        # Should have the loaded value
        assert prefs.preferences["memory_monitor_enabled"] is False
        # Should still have other defaults
        assert prefs.preferences["use_persistent_conversations"] is True

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("builtins.open", side_effect=OSError("Read error"))
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_load_handles_read_error(self, mock_exists, mock_file, mock_home):
        """Test loading handles file read errors gracefully."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = True

        prefs = GUIPreferences()

        # Should fall back to defaults on error
        assert prefs.preferences == prefs.defaults

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_load_handles_empty_file(self, mock_exists, mock_file, mock_home):
        """Test loading handles empty file."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = True

        prefs = GUIPreferences()

        # Should use defaults for empty file
        assert prefs.preferences == prefs.defaults

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: content:")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_load_handles_invalid_yaml(self, mock_exists, mock_file, mock_home):
        """Test loading handles invalid YAML gracefully."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = True

        # Should not raise exception, should fall back to defaults
        prefs = GUIPreferences()
        assert prefs.preferences == prefs.defaults


class TestSavePreferences:
    """Test suite for saving preferences."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_save_creates_directory(self, mock_exists, mock_file, mock_mkdir, mock_home):
        """Test saving creates preferences directory if needed."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        result = prefs.save_preferences()

        assert result is True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_save_writes_yaml(self, mock_exists, mock_file, mock_mkdir, mock_home):
        """Test saving writes preferences as YAML."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        prefs.set("test_key", "test_value")
        result = prefs.save_preferences()

        assert result is True
        # Verify file was opened for writing
        mock_file.assert_called()
        handle = mock_file()
        handle.write.assert_called()

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.mkdir")
    @patch("builtins.open", side_effect=OSError("Write error"))
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_save_handles_write_error(self, mock_exists, mock_file, mock_mkdir, mock_home):
        """Test saving handles write errors gracefully."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        result = prefs.save_preferences()

        assert result is False

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch(
        "osprey.interfaces.pyqt.gui_preferences.Path.mkdir",
        side_effect=OSError("Permission denied"),
    )
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_save_handles_mkdir_error(self, mock_exists, mock_mkdir, mock_home):
        """Test saving handles directory creation errors."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        result = prefs.save_preferences()

        assert result is False


class TestGetPreference:
    """Test suite for getting preference values."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_get_existing_key(self, mock_exists, mock_home):
        """Test getting an existing preference."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        value = prefs.get("use_persistent_conversations")

        assert value is True

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_get_nonexistent_key_with_default(self, mock_exists, mock_home):
        """Test getting a nonexistent key returns default."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        value = prefs.get("nonexistent_key", "default_value")

        assert value == "default_value"

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_get_nonexistent_key_without_default(self, mock_exists, mock_home):
        """Test getting a nonexistent key without default returns None."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        value = prefs.get("nonexistent_key")

        assert value is None

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_get_different_types(self, mock_exists, mock_home):
        """Test getting preferences of different types."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()

        # Boolean
        assert isinstance(prefs.get("use_persistent_conversations"), bool)
        # String
        assert isinstance(prefs.get("conversation_storage_mode"), str)
        # Integer
        assert isinstance(prefs.get("memory_warning_threshold_mb"), int)


class TestSetPreference:
    """Test suite for setting preference values."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_set_existing_key(self, mock_exists, mock_home):
        """Test setting an existing preference."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        prefs.set("use_persistent_conversations", False)

        assert prefs.get("use_persistent_conversations") is False

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_set_new_key(self, mock_exists, mock_home):
        """Test setting a new preference."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        prefs.set("new_preference", "new_value")

        assert prefs.get("new_preference") == "new_value"

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_set_different_types(self, mock_exists, mock_home):
        """Test setting preferences of different types."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()

        # Boolean
        prefs.set("bool_pref", True)
        assert prefs.get("bool_pref") is True

        # String
        prefs.set("string_pref", "test")
        assert prefs.get("string_pref") == "test"

        # Integer
        prefs.set("int_pref", 42)
        assert prefs.get("int_pref") == 42

        # List
        prefs.set("list_pref", [1, 2, 3])
        assert prefs.get("list_pref") == [1, 2, 3]

        # Dict
        prefs.set("dict_pref", {"key": "value"})
        assert prefs.get("dict_pref") == {"key": "value"}

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_set_overwrites_existing(self, mock_exists, mock_home):
        """Test setting overwrites existing value."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        prefs.set("test_key", "value1")
        assert prefs.get("test_key") == "value1"

        prefs.set("test_key", "value2")
        assert prefs.get("test_key") == "value2"


class TestUpdatePreferences:
    """Test suite for updating multiple preferences."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_update_multiple_preferences(self, mock_exists, mock_home):
        """Test updating multiple preferences at once."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        updates = {
            "use_persistent_conversations": False,
            "memory_monitor_enabled": False,
            "new_key": "new_value",
        }
        prefs.update(updates)

        assert prefs.get("use_persistent_conversations") is False
        assert prefs.get("memory_monitor_enabled") is False
        assert prefs.get("new_key") == "new_value"

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_update_preserves_other_preferences(self, mock_exists, mock_home):
        """Test update doesn't affect other preferences."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        original_value = prefs.get("conversation_storage_mode")

        prefs.update({"use_persistent_conversations": False})

        assert prefs.get("conversation_storage_mode") == original_value

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_update_with_empty_dict(self, mock_exists, mock_home):
        """Test update with empty dictionary."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        original_prefs = prefs.get_all()

        prefs.update({})

        assert prefs.get_all() == original_prefs


class TestGetAllPreferences:
    """Test suite for getting all preferences."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_get_all_returns_all_preferences(self, mock_exists, mock_home):
        """Test get_all returns all preferences."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        all_prefs = prefs.get_all()

        assert len(all_prefs) == len(prefs.defaults)
        assert all(key in all_prefs for key in prefs.defaults.keys())

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_get_all_returns_copy(self, mock_exists, mock_home):
        """Test get_all returns a copy, not reference."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        all_prefs = prefs.get_all()

        # Modify the returned dict
        all_prefs["use_persistent_conversations"] = False

        # Original should be unchanged
        assert prefs.get("use_persistent_conversations") is True

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_get_all_includes_custom_preferences(self, mock_exists, mock_home):
        """Test get_all includes custom preferences."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        prefs.set("custom_key", "custom_value")

        all_prefs = prefs.get_all()
        assert "custom_key" in all_prefs
        assert all_prefs["custom_key"] == "custom_value"


class TestResetToDefaults:
    """Test suite for resetting preferences to defaults."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_reset_restores_defaults(self, mock_exists, mock_home):
        """Test reset restores all default values."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()

        # Modify some preferences
        prefs.set("use_persistent_conversations", False)
        prefs.set("memory_monitor_enabled", False)
        prefs.set("custom_key", "custom_value")

        # Reset
        prefs.reset_to_defaults()

        # Check defaults are restored
        assert prefs.get("use_persistent_conversations") is True
        assert prefs.get("memory_monitor_enabled") is True
        # Custom key should be gone
        assert prefs.get("custom_key") is None

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_reset_removes_custom_preferences(self, mock_exists, mock_home):
        """Test reset removes custom preferences."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        prefs.set("custom1", "value1")
        prefs.set("custom2", "value2")

        prefs.reset_to_defaults()

        assert prefs.get_all() == prefs.defaults


class TestEdgeCases:
    """Test suite for edge cases."""

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_set_none_value(self, mock_exists, mock_home):
        """Test setting None as a value."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs = GUIPreferences()
        prefs.set("test_key", None)

        assert prefs.get("test_key") is None

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_multiple_instances_independent(self, mock_exists, mock_home):
        """Test multiple instances are independent."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = False

        prefs1 = GUIPreferences()
        prefs2 = GUIPreferences()

        prefs1.set("test_key", "value1")
        prefs2.set("test_key", "value2")

        assert prefs1.get("test_key") == "value1"
        assert prefs2.get("test_key") == "value2"

    @patch("osprey.interfaces.pyqt.gui_preferences.Path.home")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="use_persistent_conversations: false\nmemory_monitor_enabled: true\n",
    )
    @patch("osprey.interfaces.pyqt.gui_preferences.Path.exists")
    def test_load_and_modify_workflow(self, mock_exists, mock_file, mock_home):
        """Test typical workflow of loading, modifying, and saving."""
        mock_home.return_value = Path("/mock/home")
        mock_exists.return_value = True

        # Load
        prefs = GUIPreferences()
        assert prefs.get("use_persistent_conversations") is False

        # Modify
        prefs.set("use_persistent_conversations", True)
        prefs.set("new_setting", "new_value")

        # Verify changes
        assert prefs.get("use_persistent_conversations") is True
        assert prefs.get("new_setting") == "new_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
