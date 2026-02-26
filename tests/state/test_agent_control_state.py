"""
Tests for AgentControlState and apply_slash_commands_to_agent_control_state.

Ensures that ALL slash command handlers in categories.py produce fields
that are recognized by AgentControlState and correctly applied by
apply_slash_commands_to_agent_control_state().

This test module was added to catch the bug where /task:off, /caps:off,
and /approval:off silently dropped their state changes because the fields
returned by command handlers were not present in AgentControlState.__annotations__.
"""

from unittest.mock import patch

import pytest

from osprey.state.control import AgentControlState, apply_slash_commands_to_agent_control_state

# Mock execution limits so tests don't require config.yml
MOCK_EXECUTION_LIMITS = {
    "max_reclassifications": 1,
    "max_planning_attempts": 2,
    "max_step_retries": 0,
    "max_execution_time_seconds": 300,
    "max_concurrent_classifications": 5,
}


@pytest.fixture(autouse=True)
def mock_execution_limits():
    """Mock get_execution_limits to avoid requiring config.yml in unit tests."""
    with patch("osprey.state.control.get_execution_limits", return_value=MOCK_EXECUTION_LIMITS):
        yield


class TestAgentControlStateAnnotations:
    """Verify AgentControlState declares all fields used by slash command handlers."""

    def test_task_extraction_bypass_enabled_in_annotations(self):
        """task_extraction_bypass_enabled must be declared in AgentControlState."""
        assert "task_extraction_bypass_enabled" in AgentControlState.__annotations__, (
            "AgentControlState is missing 'task_extraction_bypass_enabled' field. "
            "The /task:off command returns this field but it will be silently dropped."
        )

    def test_capability_selection_bypass_enabled_in_annotations(self):
        """capability_selection_bypass_enabled must be declared in AgentControlState."""
        assert "capability_selection_bypass_enabled" in AgentControlState.__annotations__, (
            "AgentControlState is missing 'capability_selection_bypass_enabled' field. "
            "The /caps:off command returns this field but it will be silently dropped."
        )

    def test_approval_mode_in_annotations(self):
        """approval_mode must be declared in AgentControlState.

        The /approval handler returns 'approval_mode' but the TypedDict only
        has 'approval_global_mode'. The field name must match what the handler returns.
        """
        assert "approval_mode" in AgentControlState.__annotations__, (
            "AgentControlState is missing 'approval_mode' field. "
            "The /approval command returns this field but it will be silently dropped."
        )


class TestApplySlashCommandsTaskBypass:
    """Tests for applying /task:off and /task:on command changes."""

    def test_task_off_applies_bypass_enabled(self):
        """'/task:off' sets task_extraction_bypass_enabled=True in control state."""
        current_state = AgentControlState(planning_mode_enabled=False)
        # This is what task_handler returns for /task:off
        changes = {"task_extraction_bypass_enabled": True}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("task_extraction_bypass_enabled") is True

    def test_task_on_applies_bypass_disabled(self):
        """'/task:on' sets task_extraction_bypass_enabled=False in control state."""
        current_state = AgentControlState(planning_mode_enabled=False)
        changes = {"task_extraction_bypass_enabled": False}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("task_extraction_bypass_enabled") is False

    def test_task_bypass_preserved_across_updates(self):
        """Task bypass setting is preserved when other fields are updated."""
        current_state = AgentControlState(
            planning_mode_enabled=False,
            task_extraction_bypass_enabled=True,
        )
        # Apply a different change - task bypass should be preserved
        changes = {"planning_mode_enabled": True}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("task_extraction_bypass_enabled") is True
        assert new_state.get("planning_mode_enabled") is True


class TestApplySlashCommandsCapsBypass:
    """Tests for applying /caps:off and /caps:on command changes."""

    def test_caps_off_applies_bypass_enabled(self):
        """'/caps:off' sets capability_selection_bypass_enabled=True in control state."""
        current_state = AgentControlState(planning_mode_enabled=False)
        changes = {"capability_selection_bypass_enabled": True}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("capability_selection_bypass_enabled") is True

    def test_caps_on_applies_bypass_disabled(self):
        """'/caps:on' sets capability_selection_bypass_enabled=False in control state."""
        current_state = AgentControlState(planning_mode_enabled=False)
        changes = {"capability_selection_bypass_enabled": False}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("capability_selection_bypass_enabled") is False

    def test_caps_bypass_preserved_across_updates(self):
        """Caps bypass setting is preserved when other fields are updated."""
        current_state = AgentControlState(
            planning_mode_enabled=False,
            capability_selection_bypass_enabled=True,
        )
        changes = {"planning_mode_enabled": True}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("capability_selection_bypass_enabled") is True
        assert new_state.get("planning_mode_enabled") is True


class TestApplySlashCommandsApprovalMode:
    """Tests for applying /approval command changes."""

    def test_approval_off_applies_disabled(self):
        """'/approval:off' sets approval_mode='disabled' in control state."""
        current_state = AgentControlState(planning_mode_enabled=False)
        # This is what approval_handler returns for /approval:off
        changes = {"approval_mode": "disabled"}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("approval_mode") == "disabled"

    def test_approval_on_applies_enabled(self):
        """'/approval:on' sets approval_mode='enabled' in control state."""
        current_state = AgentControlState(planning_mode_enabled=False)
        changes = {"approval_mode": "enabled"}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("approval_mode") == "enabled"

    def test_approval_selective_applies_selective(self):
        """'/approval:selective' sets approval_mode='selective' in control state."""
        current_state = AgentControlState(planning_mode_enabled=False)
        changes = {"approval_mode": "selective"}

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("approval_mode") == "selective"


class TestApplySlashCommandsCombined:
    """Tests for applying multiple command changes at once."""

    def test_multiple_bypass_commands_together(self):
        """Multiple bypass commands applied simultaneously all take effect."""
        current_state = AgentControlState(planning_mode_enabled=False)
        changes = {
            "task_extraction_bypass_enabled": True,
            "capability_selection_bypass_enabled": True,
        }

        new_state = apply_slash_commands_to_agent_control_state(current_state, changes)

        assert new_state.get("task_extraction_bypass_enabled") is True
        assert new_state.get("capability_selection_bypass_enabled") is True

    def test_all_agent_control_handlers_produce_recognized_fields(self):
        """Every field returned by agent control command handlers exists in AgentControlState.

        This is a meta-test that validates the contract between command handlers
        and the state type system. If a new command is added but its field is not
        added to AgentControlState, this test will catch it.
        """
        # Fields returned by command handlers in categories.py:
        # planning_handler -> planning_mode_enabled
        # approval_handler -> approval_mode
        # task_handler -> task_extraction_bypass_enabled
        # caps_handler -> capability_selection_bypass_enabled
        handler_fields = [
            "planning_mode_enabled",
            "approval_mode",
            "task_extraction_bypass_enabled",
            "capability_selection_bypass_enabled",
        ]

        annotations = AgentControlState.__annotations__
        for field in handler_fields:
            assert field in annotations, (
                f"AgentControlState is missing '{field}' which is returned by a "
                f"slash command handler. This will cause the command to silently fail."
            )
