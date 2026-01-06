"""Test RouterNode instance method pattern."""

import inspect

import pytest

from osprey.infrastructure.router_node import RouterNode


class TestRouterNodeMigration:
    """Test RouterNode uses instance method pattern correctly."""

    def test_execute_is_instance_method(self):
        """Test execute() is an instance method, not static."""
        execute_method = inspect.getattr_static(RouterNode, "execute")
        assert not isinstance(execute_method, staticmethod), (
            "RouterNode.execute() should be instance method (not @staticmethod)"
        )

    @pytest.mark.asyncio
    async def test_execute_accesses_state_via_instance(self, base_state):
        """Test execute() can access state via self._state pattern."""

        # Create instance
        node = RouterNode()
        node._state = base_state

        # Execute
        result = await node.execute()

        # Verify result
        assert "control_routing_timestamp" in result
        assert "control_routing_count" in result
        assert result["control_routing_count"] == 1  # 0 + 1

    @pytest.mark.asyncio
    async def test_increments_routing_count(self, base_state):
        """Test router increments routing count correctly."""

        base_state["control_routing_count"] = 5

        node = RouterNode()
        node._state = base_state
        result = await node.execute()

        assert result["control_routing_count"] == 6  # 5 + 1

    @pytest.mark.asyncio
    async def test_handles_missing_routing_count(self, base_state):
        """Test router handles missing routing_count gracefully."""

        # Remove routing_count if present
        base_state.pop("control_routing_count", None)

        node = RouterNode()
        node._state = base_state
        result = await node.execute()

        assert result["control_routing_count"] == 1  # 0 + 1 (default)

    @pytest.mark.asyncio
    async def test_via_decorator_wrapper(self, base_state):
        """Test execute() works when called via decorator wrapper."""

        # Get the LangGraph node function created by decorator
        node_func = RouterNode.langgraph_node

        # Execute via decorator
        result = await node_func(base_state)

        # Verify result
        assert "control_routing_timestamp" in result
        assert "control_routing_count" in result
        assert isinstance(result["control_routing_timestamp"], float)
