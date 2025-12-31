"""Tests for parallel execution configuration.

This module tests the parallel_execution_enabled configuration setting
and its integration with state management.
"""

from unittest.mock import patch

from osprey.state import StateManager
from osprey.state.state_manager import get_agent_control_defaults


def test_agent_control_defaults_include_parallel_execution():
    """Test that agent control defaults include parallel_execution_enabled."""
    defaults = get_agent_control_defaults()

    assert "parallel_execution_enabled" in defaults
    assert isinstance(defaults["parallel_execution_enabled"], bool)


def test_parallel_execution_disabled_by_default():
    """Test that parallel execution is disabled by default."""
    defaults = get_agent_control_defaults()

    # Should be False by default for safety
    assert defaults["parallel_execution_enabled"] is False


def test_fresh_state_includes_parallel_execution_setting():
    """Test that fresh state includes parallel execution setting in agent_control."""
    state = StateManager.create_fresh_state("Test query")

    assert "agent_control" in state
    assert "parallel_execution_enabled" in state["agent_control"]


def test_fresh_state_parallel_execution_default():
    """Test that fresh state has parallel execution disabled by default."""
    state = StateManager.create_fresh_state("Test query")

    assert state["agent_control"]["parallel_execution_enabled"] is False


def test_parallel_execution_can_be_enabled():
    """Test that parallel execution can be enabled in state."""
    state = StateManager.create_fresh_state("Test query")

    # Enable parallel execution
    state["agent_control"]["parallel_execution_enabled"] = True

    assert state["agent_control"]["parallel_execution_enabled"] is True


def test_parallel_execution_preserved_across_state_updates():
    """Test that parallel execution setting is preserved in state updates."""
    state = StateManager.create_fresh_state("Test query")
    state["agent_control"]["parallel_execution_enabled"] = True

    # Simulate state update
    updated_state = {**state}

    assert updated_state["agent_control"]["parallel_execution_enabled"] is True


@patch("osprey.state.state_manager._get_agent_control_defaults")
def test_config_loading_error_uses_safe_defaults(mock_get_defaults):
    """Test that config loading errors fall back to safe defaults."""
    # Simulate config loading error
    mock_get_defaults.side_effect = Exception("Config error")

    defaults = get_agent_control_defaults()

    # Should still return defaults with parallel_execution_enabled
    assert "parallel_execution_enabled" in defaults
    assert defaults["parallel_execution_enabled"] is False


def test_agent_control_defaults_structure():
    """Test that agent control defaults have expected structure."""
    defaults = get_agent_control_defaults()

    # Should include all expected fields
    expected_fields = [
        "planning_mode_enabled",
        "epics_writes_enabled",
        "parallel_execution_enabled",
        "max_reclassifications",
        "max_planning_attempts",
    ]

    for field in expected_fields:
        assert field in defaults, f"Missing field: {field}"


def test_parallel_execution_type_validation():
    """Test that parallel_execution_enabled is always a boolean."""
    defaults = get_agent_control_defaults()

    assert isinstance(defaults["parallel_execution_enabled"], bool)
    assert defaults["parallel_execution_enabled"] in [True, False]


def test_state_manager_preserves_agent_control():
    """Test that StateManager preserves agent_control settings."""
    # Create initial state with parallel execution enabled
    state1 = StateManager.create_fresh_state("Query 1")
    state1["agent_control"]["parallel_execution_enabled"] = True

    # Create new state (should reset agent_control to defaults)
    state2 = StateManager.create_fresh_state("Query 2", current_state=state1)

    # agent_control should be reset to defaults (not preserved)
    assert state2["agent_control"]["parallel_execution_enabled"] is False


def test_parallel_execution_independent_of_other_settings():
    """Test that parallel execution setting is independent of other agent control settings."""
    state = StateManager.create_fresh_state("Test query")

    # Modify other settings
    state["agent_control"]["planning_mode_enabled"] = True
    state["agent_control"]["epics_writes_enabled"] = True

    # Parallel execution should still be at default
    assert state["agent_control"]["parallel_execution_enabled"] is False

    # Enable parallel execution
    state["agent_control"]["parallel_execution_enabled"] = True

    # Other settings should be unchanged
    assert state["agent_control"]["planning_mode_enabled"] is True
    assert state["agent_control"]["epics_writes_enabled"] is True


def test_parallel_execution_config_in_safe_defaults():
    """Test that safe defaults include parallel_execution_enabled."""
    # This tests the fallback path in get_agent_control_defaults
    with patch("osprey.state.state_manager._get_agent_control_defaults") as mock_get:
        mock_get.side_effect = Exception("Config error")

        defaults = get_agent_control_defaults()

        # Safe defaults should include parallel_execution_enabled = False
        assert defaults["parallel_execution_enabled"] is False


def test_multiple_state_creations_have_consistent_defaults():
    """Test that multiple state creations have consistent parallel execution defaults."""
    state1 = StateManager.create_fresh_state("Query 1")
    state2 = StateManager.create_fresh_state("Query 2")
    state3 = StateManager.create_fresh_state("Query 3")

    # All should have same default
    assert (
        state1["agent_control"]["parallel_execution_enabled"]
        == state2["agent_control"]["parallel_execution_enabled"]
        == state3["agent_control"]["parallel_execution_enabled"]
    )
