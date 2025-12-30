"""
Tests for RuntimeOverrideManager

Tests the runtime override functionality including override setting,
getting, removal, and application to configuration dictionaries.
"""

import pytest

from osprey.interfaces.pyqt.runtime_overrides import RuntimeOverrideManager


class TestRuntimeOverrideManagerInitialization:
    """Test suite for RuntimeOverrideManager initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        manager = RuntimeOverrideManager()

        assert manager.overrides == {}

    def test_init_creates_empty_overrides(self):
        """Test that initialization creates empty overrides dict."""
        manager = RuntimeOverrideManager()

        assert isinstance(manager.overrides, dict)
        assert len(manager.overrides) == 0


class TestSetOverride:
    """Test suite for setting overrides."""

    def test_set_override_simple(self):
        """Test setting a simple override."""
        manager = RuntimeOverrideManager()

        manager.set_override("test_key", "test_value")

        assert manager.get_override("test_key") == "test_value"

    def test_set_override_dotted_path(self):
        """Test setting override with dotted path."""
        manager = RuntimeOverrideManager()

        manager.set_override("execution_control.limits.max_retries", 5)

        assert manager.get_override("execution_control.limits.max_retries") == 5

    def test_set_override_overwrites_existing(self):
        """Test that setting override overwrites existing value."""
        manager = RuntimeOverrideManager()

        manager.set_override("test_key", "value1")
        manager.set_override("test_key", "value2")

        assert manager.get_override("test_key") == "value2"

    def test_set_override_different_types(self):
        """Test setting overrides with different value types."""
        manager = RuntimeOverrideManager()

        manager.set_override("string_key", "string_value")
        manager.set_override("int_key", 42)
        manager.set_override("bool_key", True)
        manager.set_override("list_key", [1, 2, 3])
        manager.set_override("dict_key", {"nested": "value"})

        assert manager.get_override("string_key") == "string_value"
        assert manager.get_override("int_key") == 42
        assert manager.get_override("bool_key") is True
        assert manager.get_override("list_key") == [1, 2, 3]
        assert manager.get_override("dict_key") == {"nested": "value"}


class TestSetOverrides:
    """Test suite for setting multiple overrides."""

    def test_set_overrides_multiple(self):
        """Test setting multiple overrides at once."""
        manager = RuntimeOverrideManager()

        overrides = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

        manager.set_overrides(overrides)

        assert manager.get_override("key1") == "value1"
        assert manager.get_override("key2") == "value2"
        assert manager.get_override("key3") == "value3"

    def test_set_overrides_merges_with_existing(self):
        """Test that set_overrides merges with existing overrides."""
        manager = RuntimeOverrideManager()

        manager.set_override("existing_key", "existing_value")

        new_overrides = {
            "new_key1": "new_value1",
            "new_key2": "new_value2",
        }

        manager.set_overrides(new_overrides)

        assert manager.get_override("existing_key") == "existing_value"
        assert manager.get_override("new_key1") == "new_value1"
        assert manager.get_override("new_key2") == "new_value2"

    def test_set_overrides_empty_dict(self):
        """Test setting empty overrides dict."""
        manager = RuntimeOverrideManager()

        manager.set_override("existing_key", "existing_value")
        manager.set_overrides({})

        # Should not change existing overrides
        assert manager.get_override("existing_key") == "existing_value"


class TestGetOverride:
    """Test suite for getting overrides."""

    def test_get_override_existing(self):
        """Test getting an existing override."""
        manager = RuntimeOverrideManager()

        manager.set_override("test_key", "test_value")

        assert manager.get_override("test_key") == "test_value"

    def test_get_override_nonexistent_with_default(self):
        """Test getting non-existent override with default."""
        manager = RuntimeOverrideManager()

        value = manager.get_override("nonexistent_key", "default_value")

        assert value == "default_value"

    def test_get_override_nonexistent_without_default(self):
        """Test getting non-existent override without default."""
        manager = RuntimeOverrideManager()

        value = manager.get_override("nonexistent_key")

        assert value is None


class TestRemoveOverride:
    """Test suite for removing overrides."""

    def test_remove_override_existing(self):
        """Test removing an existing override."""
        manager = RuntimeOverrideManager()

        manager.set_override("test_key", "test_value")
        manager.remove_override("test_key")

        assert manager.get_override("test_key") is None

    def test_remove_override_nonexistent(self):
        """Test removing non-existent override."""
        manager = RuntimeOverrideManager()

        # Should not raise exception
        manager.remove_override("nonexistent_key")

    def test_remove_override_doesnt_affect_others(self):
        """Test that removing one override doesn't affect others."""
        manager = RuntimeOverrideManager()

        manager.set_override("key1", "value1")
        manager.set_override("key2", "value2")

        manager.remove_override("key1")

        assert manager.get_override("key1") is None
        assert manager.get_override("key2") == "value2"


class TestClearOverrides:
    """Test suite for clearing overrides."""

    def test_clear_overrides(self):
        """Test clearing all overrides."""
        manager = RuntimeOverrideManager()

        manager.set_override("key1", "value1")
        manager.set_override("key2", "value2")
        manager.set_override("key3", "value3")

        manager.clear_overrides()

        assert manager.get_override("key1") is None
        assert manager.get_override("key2") is None
        assert manager.get_override("key3") is None
        assert len(manager.overrides) == 0

    def test_clear_overrides_empty(self):
        """Test clearing when no overrides exist."""
        manager = RuntimeOverrideManager()

        # Should not raise exception
        manager.clear_overrides()

        assert len(manager.overrides) == 0


class TestGetAllOverrides:
    """Test suite for getting all overrides."""

    def test_get_all_overrides(self):
        """Test getting all overrides."""
        manager = RuntimeOverrideManager()

        manager.set_override("key1", "value1")
        manager.set_override("key2", "value2")

        all_overrides = manager.get_all_overrides()

        assert all_overrides == {"key1": "value1", "key2": "value2"}

    def test_get_all_overrides_returns_copy(self):
        """Test that get_all_overrides returns a copy."""
        manager = RuntimeOverrideManager()

        manager.set_override("key1", "value1")

        all_overrides = manager.get_all_overrides()
        all_overrides["key2"] = "value2"

        # Original should not be modified
        assert manager.get_override("key2") is None

    def test_get_all_overrides_empty(self):
        """Test getting all overrides when none exist."""
        manager = RuntimeOverrideManager()

        all_overrides = manager.get_all_overrides()

        assert all_overrides == {}


class TestApplyToConfig:
    """Test suite for applying overrides to config."""

    def test_apply_to_config_simple(self):
        """Test applying simple overrides to config."""
        manager = RuntimeOverrideManager()

        manager.set_override("key1", "override_value")

        config = {"key1": "original_value", "key2": "value2"}
        result = manager.apply_to_config(config)

        assert result["key1"] == "override_value"
        assert result["key2"] == "value2"

    def test_apply_to_config_nested(self):
        """Test applying nested overrides to config."""
        manager = RuntimeOverrideManager()

        manager.set_override("execution_control.limits.max_retries", 10)

        config = {
            "execution_control": {
                "limits": {
                    "max_retries": 3,
                    "max_time": 300,
                }
            }
        }

        result = manager.apply_to_config(config)

        assert result["execution_control"]["limits"]["max_retries"] == 10
        assert result["execution_control"]["limits"]["max_time"] == 300

    def test_apply_to_config_creates_nested_structure(self):
        """Test that apply creates nested structure if it doesn't exist."""
        manager = RuntimeOverrideManager()

        manager.set_override("new.nested.key", "value")

        config = {}
        result = manager.apply_to_config(config)

        assert result["new"]["nested"]["key"] == "value"

    def test_apply_to_config_doesnt_modify_original(self):
        """Test that apply doesn't modify original config."""
        manager = RuntimeOverrideManager()

        manager.set_override("key1", "override_value")

        config = {"key1": "original_value"}
        result = manager.apply_to_config(config)

        # Original should be unchanged
        assert config["key1"] == "original_value"
        # Result should have override
        assert result["key1"] == "override_value"

    def test_apply_to_config_empty_overrides(self):
        """Test applying when no overrides exist."""
        manager = RuntimeOverrideManager()

        config = {"key1": "value1", "key2": "value2"}
        result = manager.apply_to_config(config)

        assert result == config
        assert result is not config  # Should be a copy


class TestCreateAgentControlDefaults:
    """Test suite for creating agent control defaults."""

    def test_create_agent_control_defaults_with_defaults(self):
        """Test creating agent control defaults with no overrides."""
        manager = RuntimeOverrideManager()

        defaults = manager.create_agent_control_defaults()

        assert isinstance(defaults, dict)
        assert "planning_mode_enabled" in defaults
        assert "epics_writes_enabled" in defaults
        assert "max_reclassifications" in defaults
        assert defaults["planning_mode_enabled"] is False
        assert defaults["epics_writes_enabled"] is False

    def test_create_agent_control_defaults_with_overrides(self):
        """Test creating agent control defaults with overrides."""
        manager = RuntimeOverrideManager()

        manager.set_override("planning_mode_enabled", True)
        manager.set_override("max_reclassifications", 5)

        defaults = manager.create_agent_control_defaults()

        assert defaults["planning_mode_enabled"] is True
        assert defaults["max_reclassifications"] == 5

    def test_create_agent_control_defaults_includes_all_settings(self):
        """Test that agent control defaults includes all expected settings."""
        manager = RuntimeOverrideManager()

        defaults = manager.create_agent_control_defaults()

        # Agent Control
        assert "planning_mode_enabled" in defaults
        assert "epics_writes_enabled" in defaults
        assert "task_extraction_bypass_enabled" in defaults
        assert "capability_selection_bypass_enabled" in defaults
        assert "parallel_execution_enabled" in defaults

        # Approval
        assert "approval_global_mode" in defaults
        assert "python_execution_approval_enabled" in defaults

        # Execution Limits
        assert "max_reclassifications" in defaults
        assert "max_planning_attempts" in defaults
        assert "max_step_retries" in defaults

        # Development
        assert "debug_mode" in defaults
        assert "verbose_logging" in defaults


class TestNestedValueSetting:
    """Test suite for nested value setting."""

    def test_set_nested_value_simple(self):
        """Test setting simple nested value."""
        manager = RuntimeOverrideManager()

        d = {}
        manager._set_nested_value(d, "key", "value")

        assert d["key"] == "value"

    def test_set_nested_value_two_levels(self):
        """Test setting two-level nested value."""
        manager = RuntimeOverrideManager()

        d = {}
        manager._set_nested_value(d, "level1.level2", "value")

        assert d["level1"]["level2"] == "value"

    def test_set_nested_value_three_levels(self):
        """Test setting three-level nested value."""
        manager = RuntimeOverrideManager()

        d = {}
        manager._set_nested_value(d, "level1.level2.level3", "value")

        assert d["level1"]["level2"]["level3"] == "value"

    def test_set_nested_value_preserves_existing(self):
        """Test that setting nested value preserves existing keys."""
        manager = RuntimeOverrideManager()

        d = {"level1": {"existing": "value"}}
        manager._set_nested_value(d, "level1.new_key", "new_value")

        assert d["level1"]["existing"] == "value"
        assert d["level1"]["new_key"] == "new_value"


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_set_override_none_value(self):
        """Test setting None as override value."""
        manager = RuntimeOverrideManager()

        manager.set_override("test_key", None)

        assert manager.get_override("test_key") is None

    def test_multiple_managers_independent(self):
        """Test that multiple managers are independent."""
        manager1 = RuntimeOverrideManager()
        manager2 = RuntimeOverrideManager()

        manager1.set_override("key1", "value1")
        manager2.set_override("key2", "value2")

        assert manager1.get_override("key1") == "value1"
        assert manager1.get_override("key2") is None
        assert manager2.get_override("key1") is None
        assert manager2.get_override("key2") == "value2"

    def test_apply_to_config_with_complex_structure(self):
        """Test applying overrides to complex config structure."""
        manager = RuntimeOverrideManager()

        manager.set_override("a.b.c", 1)
        manager.set_override("a.b.d", 2)
        manager.set_override("x.y", 3)

        config = {
            "a": {
                "b": {
                    "c": 0,
                    "e": 5,
                }
            }
        }

        result = manager.apply_to_config(config)

        assert result["a"]["b"]["c"] == 1
        assert result["a"]["b"]["d"] == 2
        assert result["a"]["b"]["e"] == 5
        assert result["x"]["y"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
