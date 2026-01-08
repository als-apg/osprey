"""Tests for session state persistence in AgentState.

This module tests the session_state field and its reducer function
which enables preferences like direct chat mode to persist across
conversation turns.
"""

from osprey.state import AgentState
from osprey.state.state import merge_session_state


class TestSessionStateReducer:
    """Tests for the merge_session_state reducer function."""

    def test_merge_with_none_existing(self):
        """Merge should return new state when existing is None."""
        result = merge_session_state(None, {"direct_chat_capability": "weather_mcp"})
        assert result == {"direct_chat_capability": "weather_mcp"}

    def test_merge_with_empty_new(self):
        """Merge should return existing when new is empty."""
        existing = {"direct_chat_capability": "weather_mcp"}
        result = merge_session_state(existing, {})
        assert result == existing

    def test_merge_both_none(self):
        """Merge should return empty dict when both are None/empty."""
        result = merge_session_state(None, None)
        assert result == {}

    def test_merge_updates_existing(self):
        """Merge should update existing with new values."""
        existing = {"direct_chat_capability": "weather_mcp", "session_id": "123"}
        new = {"direct_chat_capability": None, "new_key": "value"}
        result = merge_session_state(existing, new)

        assert result["direct_chat_capability"] is None
        assert result["session_id"] == "123"
        assert result["new_key"] == "value"


class TestAgentStateWithSessionState:
    """Tests for AgentState with session_state field."""

    def test_agent_state_has_session_state_field(self):
        """AgentState should have session_state as a valid field."""
        # AgentState is a TypedDict, check it has the field
        assert "session_state" in AgentState.__annotations__

    def test_create_state_with_session_state(self):
        """Should be able to create AgentState with session_state."""
        state = {
            "messages": [],
            "capability_context_data": {},
            "session_state": {"direct_chat_capability": "test_cap"},
            "agent_control": {},
            "status_updates": [],
            "task_current_task": "",
            "task_depends_on_chat_history": False,
            "task_depends_on_user_memory": False,
            "task_custom_message": None,
            "planning_active_capabilities": [],
            "planning_execution_plan": None,
            "planning_current_step_index": 0,
            "execution_step_results": [],
            "execution_last_result": None,
            "execution_pending_approvals": [],
            "execution_start_time": None,
            "execution_total_time": None,
            "control_reclassification_reason": None,
            "control_retry_count": 0,
            "control_has_error": False,
            "control_error_info": None,
            "control_is_killed": False,
            "control_kill_reason": None,
            "control_routing_timestamp": None,
            "control_routing_count": 0,
            "control_plans_created_count": 0,
            "control_reclassification_count": 0,
            "ui_captured_notebooks": [],
            "ui_captured_figures": [],
            "ui_launchable_commands": [],
            "ui_agent_context": None,
            "runtime_checkpoint_metadata": {},
            "runtime_info": {},
        }

        assert state["session_state"]["direct_chat_capability"] == "test_cap"


class TestStateManagerPreservation:
    """Tests for StateManager preserving session_state."""

    def test_preserves_session_state_on_fresh_state(self):
        """StateManager.create_fresh_state should preserve session_state."""
        from osprey.state import StateManager

        # Create previous state with session_state
        prev_state = {
            "session_state": {
                "direct_chat_capability": "weather_mcp",
                "session_id": "test-session-123",
            },
            "capability_context_data": {},
        }

        # Create fresh state from previous
        fresh_state = StateManager.create_fresh_state("Hello", current_state=prev_state)

        # Session state should be preserved
        assert "session_state" in fresh_state
        assert fresh_state["session_state"]["direct_chat_capability"] == "weather_mcp"
        assert fresh_state["session_state"]["session_id"] == "test-session-123"
