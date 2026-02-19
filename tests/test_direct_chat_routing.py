"""Tests for direct chat mode routing in router and gateway.

This module tests the infrastructure handling of direct chat mode,
including router conditional edge logic and gateway message handling.
"""

import pytest

pytestmark = pytest.mark.langgraph


class TestRouterDirectChatMode:
    """Tests for router handling direct chat mode."""

    def test_router_routes_direct_chat(self, monkeypatch):
        """Router should route directly to capability in direct chat mode."""
        from unittest.mock import MagicMock

        from osprey.infrastructure.router_node import router_conditional_edge

        # Mock registry
        mock_reg_instance = MagicMock()

        # Setup capability instance with direct_chat_enabled
        mock_cap_instance = MagicMock()
        mock_cap_instance.direct_chat_enabled = True
        mock_reg_instance.get_capability = MagicMock(return_value=mock_cap_instance)

        monkeypatch.setattr(
            "osprey.infrastructure.router_node.get_registry", lambda: mock_reg_instance
        )

        # Create state with direct chat mode enabled
        state = {
            "session_state": {"direct_chat_capability": "weather_mcp"},
            "control_has_error": False,
        }

        result = router_conditional_edge(state)

        assert result == "weather_mcp"

    def test_router_clears_invalid_capability(self, monkeypatch):
        """Router should clear session_state when capability doesn't support direct chat."""
        from unittest.mock import MagicMock

        from osprey.infrastructure.router_node import router_conditional_edge

        # Mock registry
        mock_reg_instance = MagicMock()

        # Setup capability instance WITHOUT direct_chat_enabled
        mock_cap_instance = MagicMock()
        mock_cap_instance.direct_chat_enabled = False
        mock_reg_instance.get_capability = MagicMock(return_value=mock_cap_instance)
        mock_reg_instance.get_node = MagicMock(return_value=None)

        monkeypatch.setattr(
            "osprey.infrastructure.router_node.get_registry", lambda: mock_reg_instance
        )

        # Create state with direct chat mode pointing to non-direct-chat capability
        state = {
            "session_state": {"direct_chat_capability": "normal_cap"},
            "control_has_error": False,
            "task_current_task": None,  # Will trigger task_extraction
        }

        result = router_conditional_edge(state)

        # Should have cleared the invalid direct chat and routed to task_extraction
        assert state["session_state"]["direct_chat_capability"] is None
        assert result == "task_extraction"


class TestGatewayDirectChatMode:
    """Tests for Gateway handling direct chat mode."""

    @pytest.mark.asyncio
    async def test_gateway_detects_direct_chat_mode(self):
        """Gateway should detect direct chat mode and preserve message history."""
        from unittest.mock import MagicMock

        from osprey.infrastructure.gateway import Gateway

        gateway = Gateway()

        # Mock graph with state in direct chat mode
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "session_state": {"direct_chat_capability": "weather_mcp"},
            "messages": [],
        }
        mock_graph.get_state.return_value = mock_state

        config = {"configurable": {"thread_id": "test"}}

        result = await gateway._handle_new_message_flow("Hello", mock_graph, config)

        # In direct chat mode, should have messages (appended) and session_state
        assert result.agent_state is not None
        assert "messages" in result.agent_state
        assert "session_state" in result.agent_state

    @pytest.mark.asyncio
    async def test_gateway_clears_execution_last_result_for_new_turn(self):
        """Gateway should clear execution_last_result to signal new turn to router.

        This prevents the bug where router would immediately return END on second
        message because execution_last_result still had previous turn's result.
        """
        from unittest.mock import MagicMock

        from osprey.infrastructure.gateway import Gateway

        gateway = Gateway()

        # Mock graph with state in direct chat mode WITH previous execution result
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "session_state": {"direct_chat_capability": "weather_mcp"},
            "messages": [],
            # This is the previous turn's result that was causing the bug
            "execution_last_result": {"capability": "weather_mcp", "content": "Previous response"},
        }
        mock_graph.get_state.return_value = mock_state

        config = {"configurable": {"thread_id": "test"}}

        result = await gateway._handle_new_message_flow("Second message", mock_graph, config)

        # execution_last_result should be cleared to None so router knows it's a new turn
        assert result.agent_state is not None
        assert "execution_last_result" in result.agent_state
        assert result.agent_state["execution_last_result"] is None


class TestRouterMultiTurnDirectChat:
    """Tests for router handling multi-turn direct chat correctly."""

    def test_router_routes_to_capability_on_new_turn(self, monkeypatch):
        """Router should route to capability when execution_last_result is cleared (new turn)."""
        from unittest.mock import MagicMock

        from osprey.infrastructure.router_node import router_conditional_edge

        # Mock registry
        mock_reg_instance = MagicMock()
        mock_cap_instance = MagicMock()
        mock_cap_instance.direct_chat_enabled = True
        mock_reg_instance.get_capability = MagicMock(return_value=mock_cap_instance)

        monkeypatch.setattr(
            "osprey.infrastructure.router_node.get_registry", lambda: mock_reg_instance
        )

        # State with direct chat mode but execution_last_result cleared (new turn)
        state = {
            "session_state": {"direct_chat_capability": "weather_mcp"},
            "control_has_error": False,
            "execution_last_result": None,  # Cleared by gateway for new turn
        }

        result = router_conditional_edge(state)

        # Should route to capability, not END
        assert result == "weather_mcp"

    def test_router_returns_end_after_capability_executes(self, monkeypatch):
        """Router should return END when execution_last_result matches capability (turn complete)."""
        from unittest.mock import MagicMock

        from osprey.infrastructure.router_node import router_conditional_edge

        # Mock registry
        mock_reg_instance = MagicMock()
        mock_cap_instance = MagicMock()
        mock_cap_instance.direct_chat_enabled = True
        mock_reg_instance.get_capability = MagicMock(return_value=mock_cap_instance)

        monkeypatch.setattr(
            "osprey.infrastructure.router_node.get_registry", lambda: mock_reg_instance
        )

        # State after capability executed this turn
        state = {
            "session_state": {"direct_chat_capability": "weather_mcp"},
            "control_has_error": False,
            "execution_last_result": {"capability": "weather_mcp", "content": "Response"},
        }

        result = router_conditional_edge(state)

        # Should return END because capability already ran this turn
        assert result == "END"
