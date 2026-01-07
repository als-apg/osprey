"""
Tests for SettingsManager

Tests the settings management functionality including initialization,
loading from config, updating settings, and retrieving settings.
"""

from unittest.mock import mock_open, patch

import pytest

from osprey.interfaces.pyqt.settings_manager import (
    AgentControlSettings,
    ApprovalSettings,
    DevelopmentSettings,
    ExecutionLimits,
    GUISettings,
    RoutingSettings,
    SettingsManager,
)


class TestDataclassDefaults:
    """Test suite for dataclass default values."""

    def test_agent_control_defaults(self):
        """Test AgentControlSettings has correct defaults."""
        settings = AgentControlSettings()

        assert settings.planning_mode_enabled is False
        assert settings.epics_writes_enabled is False
        assert settings.task_extraction_bypass_enabled is False
        assert settings.capability_selection_bypass_enabled is False
        assert settings.parallel_execution_enabled is False

    def test_approval_defaults(self):
        """Test ApprovalSettings has correct defaults."""
        settings = ApprovalSettings()

        assert settings.approval_global_mode == "selective"
        assert settings.python_execution_approval_enabled is True
        assert settings.python_execution_approval_mode == "all_code"
        assert settings.memory_approval_enabled is True

    def test_execution_limits_defaults(self):
        """Test ExecutionLimits has correct defaults."""
        settings = ExecutionLimits()

        assert settings.max_reclassifications == 1
        assert settings.max_planning_attempts == 2
        assert settings.max_step_retries == 0
        assert settings.max_execution_time_seconds == 300
        assert settings.max_concurrent_classifications == 5

    def test_gui_settings_defaults(self):
        """Test GUISettings has correct defaults."""
        settings = GUISettings()

        assert settings.use_persistent_conversations is True
        assert settings.conversation_storage_mode == "json"
        assert settings.redirect_output_to_gui is True
        assert settings.suppress_terminal_output is False
        assert settings.group_system_messages is True
        assert settings.enable_routing_feedback is True

    def test_development_settings_defaults(self):
        """Test DevelopmentSettings has correct defaults."""
        settings = DevelopmentSettings()

        assert settings.debug_mode is False
        assert settings.verbose_logging is False
        assert settings.raise_raw_errors is False
        assert settings.print_prompts is False
        assert settings.show_prompts is False
        assert settings.prompts_latest_only is True

    def test_routing_settings_defaults(self):
        """Test RoutingSettings has correct defaults."""
        settings = RoutingSettings()

        assert settings.enable_routing_cache is True
        assert settings.cache_max_size == 100
        assert settings.cache_ttl_seconds == 3600.0
        assert settings.cache_similarity_threshold == 0.85
        assert settings.enable_semantic_analysis is True


class TestSettingsManagerInitialization:
    """Test suite for SettingsManager initialization."""

    def test_init_without_config(self):
        """Test initialization without config file."""
        manager = SettingsManager()

        assert manager.config_path is None
        assert isinstance(manager.agent_control, AgentControlSettings)
        assert isinstance(manager.approval, ApprovalSettings)
        assert isinstance(manager.execution_limits, ExecutionLimits)
        assert isinstance(manager.gui, GUISettings)
        assert isinstance(manager.development, DevelopmentSettings)
        assert isinstance(manager.routing, RoutingSettings)

    def test_init_memory_monitoring_defaults(self):
        """Test initialization includes memory monitoring defaults."""
        manager = SettingsManager()

        assert manager.memory_monitor_enabled is True
        assert manager.memory_warning_threshold_mb == 500
        assert manager.memory_critical_threshold_mb == 1000
        assert manager.memory_check_interval_seconds == 5

    @patch.object(SettingsManager, "load_from_config")
    def test_init_with_config_path(self, mock_load):
        """Test initialization with config path loads config."""
        mock_load.return_value = True

        manager = SettingsManager("config.yml")

        assert manager.config_path == "config.yml"
        mock_load.assert_called_once_with("config.yml")


class TestGetAllSettings:
    """Test suite for get_all_settings method."""

    def test_get_all_settings_returns_dict(self):
        """Test get_all_settings returns dictionary."""
        manager = SettingsManager()
        settings = manager.get_all_settings()

        assert isinstance(settings, dict)
        assert len(settings) > 0

    def test_get_all_settings_includes_agent_control(self):
        """Test get_all_settings includes agent control settings."""
        manager = SettingsManager()
        settings = manager.get_all_settings()

        assert "planning_mode_enabled" in settings
        assert "epics_writes_enabled" in settings
        assert "task_extraction_bypass_enabled" in settings

    def test_get_all_settings_includes_all_categories(self):
        """Test get_all_settings includes all setting categories."""
        manager = SettingsManager()
        settings = manager.get_all_settings()

        # Agent control
        assert "planning_mode_enabled" in settings
        # Approval
        assert "approval_global_mode" in settings
        # Execution limits
        assert "max_reclassifications" in settings
        # GUI
        assert "use_persistent_conversations" in settings
        # Development
        assert "debug_mode" in settings
        # Routing
        assert "enable_routing_cache" in settings
        # Memory monitoring
        assert "memory_monitor_enabled" in settings

    def test_get_all_settings_reflects_changes(self):
        """Test get_all_settings reflects setting changes."""
        manager = SettingsManager()
        manager.agent_control.planning_mode_enabled = True

        settings = manager.get_all_settings()

        assert settings["planning_mode_enabled"] is True


class TestUpdateFromDict:
    """Test suite for update_from_dict method."""

    def test_update_agent_control_settings(self):
        """Test updating agent control settings."""
        manager = SettingsManager()
        updates = {
            "planning_mode_enabled": True,
            "epics_writes_enabled": True,
        }

        manager.update_from_dict(updates)

        assert manager.agent_control.planning_mode_enabled is True
        assert manager.agent_control.epics_writes_enabled is True

    def test_update_approval_settings(self):
        """Test updating approval settings."""
        manager = SettingsManager()
        updates = {
            "approval_global_mode": "all",
            "python_execution_approval_enabled": False,
        }

        manager.update_from_dict(updates)

        assert manager.approval.approval_global_mode == "all"
        assert manager.approval.python_execution_approval_enabled is False

    def test_update_execution_limits(self):
        """Test updating execution limits."""
        manager = SettingsManager()
        updates = {
            "max_reclassifications": 5,
            "max_execution_time_seconds": 600,
        }

        manager.update_from_dict(updates)

        assert manager.execution_limits.max_reclassifications == 5
        assert manager.execution_limits.max_execution_time_seconds == 600

    def test_update_gui_settings(self):
        """Test updating GUI settings."""
        manager = SettingsManager()
        updates = {
            "use_persistent_conversations": False,
            "conversation_storage_mode": "memory",
        }

        manager.update_from_dict(updates)

        assert manager.gui.use_persistent_conversations is False
        assert manager.gui.conversation_storage_mode == "memory"

    def test_update_development_settings(self):
        """Test updating development settings."""
        manager = SettingsManager()
        updates = {
            "debug_mode": True,
            "verbose_logging": True,
        }

        manager.update_from_dict(updates)

        assert manager.development.debug_mode is True
        assert manager.development.verbose_logging is True

    def test_update_routing_settings(self):
        """Test updating routing settings."""
        manager = SettingsManager()
        updates = {
            "enable_routing_cache": False,
            "cache_max_size": 200,
        }

        manager.update_from_dict(updates)

        assert manager.routing.enable_routing_cache is False
        assert manager.routing.cache_max_size == 200

    def test_update_memory_monitoring(self):
        """Test updating memory monitoring settings."""
        manager = SettingsManager()
        updates = {
            "memory_monitor_enabled": False,
            "memory_warning_threshold_mb": 1000,
        }

        manager.update_from_dict(updates)

        assert manager.memory_monitor_enabled is False
        assert manager.memory_warning_threshold_mb == 1000

    def test_update_partial_settings(self):
        """Test updating only some settings."""
        manager = SettingsManager()
        original_value = manager.agent_control.epics_writes_enabled

        updates = {"planning_mode_enabled": True}
        manager.update_from_dict(updates)

        assert manager.agent_control.planning_mode_enabled is True
        assert manager.agent_control.epics_writes_enabled == original_value

    def test_update_with_empty_dict(self):
        """Test updating with empty dictionary."""
        manager = SettingsManager()
        original_settings = manager.get_all_settings()

        manager.update_from_dict({})

        assert manager.get_all_settings() == original_settings


class TestLoadFromConfig:
    """Test suite for load_from_config method."""

    @patch("builtins.open", new_callable=mock_open, read_data="")
    @patch("osprey.interfaces.pyqt.settings_manager.Path.exists")
    def test_load_from_nonexistent_file(self, mock_exists, mock_file):
        """Test loading from nonexistent file returns False."""
        mock_exists.return_value = False

        manager = SettingsManager()
        result = manager.load_from_config("nonexistent.yml")

        assert result is False

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="execution_control:\n  agent_control:\n    parallel_execution_enabled: true\n",
    )
    @patch("osprey.interfaces.pyqt.settings_manager.Path.exists")
    def test_load_agent_control_settings(self, mock_exists, mock_file):
        """Test loading agent control settings from config."""
        mock_exists.return_value = True

        manager = SettingsManager()
        result = manager.load_from_config("config.yml")

        assert result is True
        assert manager.agent_control.parallel_execution_enabled is True

    @patch("builtins.open", new_callable=mock_open, read_data="approval:\n  global_mode: all\n")
    @patch("osprey.interfaces.pyqt.settings_manager.Path.exists")
    def test_load_approval_settings(self, mock_exists, mock_file):
        """Test loading approval settings from config."""
        mock_exists.return_value = True

        manager = SettingsManager()
        result = manager.load_from_config("config.yml")

        assert result is True
        assert manager.approval.approval_global_mode == "all"

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="execution_control:\n  limits:\n    max_reclassifications: 5\n",
    )
    @patch("osprey.interfaces.pyqt.settings_manager.Path.exists")
    def test_load_execution_limits(self, mock_exists, mock_file):
        """Test loading execution limits from config."""
        mock_exists.return_value = True

        manager = SettingsManager()
        result = manager.load_from_config("config.yml")

        assert result is True
        assert manager.execution_limits.max_reclassifications == 5

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="gui:\n  use_persistent_conversations: false\n",
    )
    @patch("osprey.interfaces.pyqt.settings_manager.Path.exists")
    def test_load_gui_settings(self, mock_exists, mock_file):
        """Test loading GUI settings from config."""
        mock_exists.return_value = True

        manager = SettingsManager()
        result = manager.load_from_config("config.yml")

        assert result is True
        assert manager.gui.use_persistent_conversations is False

    @patch("builtins.open", side_effect=Exception("Read error"))
    @patch("osprey.interfaces.pyqt.settings_manager.Path.exists")
    def test_load_handles_errors(self, mock_exists, mock_file):
        """Test loading handles errors gracefully."""
        mock_exists.return_value = True

        manager = SettingsManager()
        result = manager.load_from_config("config.yml")

        assert result is False


class TestGetMethod:
    """Test suite for get method."""

    def test_get_existing_setting(self):
        """Test getting an existing setting."""
        manager = SettingsManager()
        value = manager.get("planning_mode_enabled")

        assert value is False

    def test_get_nonexistent_setting_with_default(self):
        """Test getting nonexistent setting returns default."""
        manager = SettingsManager()
        value = manager.get("nonexistent_key", "default_value")

        assert value == "default_value"

    def test_get_nonexistent_setting_without_default(self):
        """Test getting nonexistent setting without default returns None."""
        manager = SettingsManager()
        value = manager.get("nonexistent_key")

        assert value is None

    def test_get_after_update(self):
        """Test get returns updated value."""
        manager = SettingsManager()
        manager.update_from_dict({"debug_mode": True})

        value = manager.get("debug_mode")

        assert value is True


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_multiple_managers_independent(self):
        """Test multiple manager instances are independent."""
        manager1 = SettingsManager()
        manager2 = SettingsManager()

        manager1.agent_control.planning_mode_enabled = True
        manager2.agent_control.planning_mode_enabled = False

        assert manager1.agent_control.planning_mode_enabled is True
        assert manager2.agent_control.planning_mode_enabled is False

    def test_update_with_invalid_types(self):
        """Test update handles invalid types gracefully."""
        manager = SettingsManager()

        # Should not raise exception
        manager.update_from_dict({"max_reclassifications": "invalid"})

        # Value should be updated (no type checking in update_from_dict)
        assert manager.execution_limits.max_reclassifications == "invalid"

    def test_get_all_settings_returns_copy(self):
        """Test get_all_settings returns independent dict."""
        manager = SettingsManager()
        settings1 = manager.get_all_settings()
        settings2 = manager.get_all_settings()

        settings1["planning_mode_enabled"] = True

        # Original should be unchanged
        assert settings2["planning_mode_enabled"] is False

    def test_complex_update_workflow(self):
        """Test complex workflow of updates and retrievals."""
        manager = SettingsManager()

        # Initial state
        assert manager.get("debug_mode") is False

        # Update via dict
        manager.update_from_dict({"debug_mode": True})
        assert manager.get("debug_mode") is True

        # Update via direct access
        manager.development.debug_mode = False
        assert manager.get("debug_mode") is False

        # Get all settings
        all_settings = manager.get_all_settings()
        assert all_settings["debug_mode"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
