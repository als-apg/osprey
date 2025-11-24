"""Tests for MemoryOperationsCapability instance method pattern migration."""

import inspect

import pytest

from osprey.capabilities.memory import MemoryOperationsCapability


class TestMemoryCapabilityMigration:
    """Test MemoryOperationsCapability successfully migrated to instance method pattern."""

    def test_uses_instance_method_not_static(self):
        """Verify execute() migrated from @staticmethod to instance method."""
        execute_method = inspect.getattr_static(MemoryOperationsCapability, "execute")
        assert not isinstance(execute_method, staticmethod)

        sig = inspect.signature(MemoryOperationsCapability.execute)
        params = list(sig.parameters.keys())
        assert params == ["self"]

    def test_state_can_be_injected(self, mock_state, mock_step):
        """Verify capability instance can receive _state and _step injection."""
        capability = MemoryOperationsCapability()
        capability._state = mock_state
        capability._step = mock_step

        assert capability._state == mock_state
        assert capability._step == mock_step

    def test_has_langgraph_node_decorator(self):
        """Verify @capability_node decorator created langgraph_node attribute."""
        assert hasattr(MemoryOperationsCapability, "langgraph_node")
        assert callable(MemoryOperationsCapability.langgraph_node)


class TestMemoryCapabilityApprovalPath:
    """Test specific execution path that's critical for migration validation."""

    @pytest.mark.asyncio
    async def test_state_injection_in_approval_path(self, mock_state, mock_step, monkeypatch):
        """Test approved operation path validates state injection works."""
        from unittest.mock import AsyncMock, MagicMock

        # Mock get_session_info
        monkeypatch.setattr(
            "osprey.capabilities.memory.get_session_info",
            MagicMock(return_value={"user_id": "test_user_123"}),
        )

        # Approved payload (bypasses complex internal logic)
        approved_payload = {"content": "Test memory content", "user_id": "test_user_123"}
        monkeypatch.setattr(
            "osprey.capabilities.memory.get_approval_resume_data",
            MagicMock(return_value=(True, approved_payload)),
        )

        monkeypatch.setattr(
            "osprey.capabilities.memory.get_streamer", MagicMock(return_value=MagicMock())
        )

        mock_sm = MagicMock()
        mock_sm.store_context.return_value = {"context_data": {}}
        monkeypatch.setattr("osprey.capabilities.memory.StateManager", mock_sm)

        # Mock the save operation
        async def mock_save(content, user_id, logger):
            return MagicMock(success=True, memory_id="mem_123")

        monkeypatch.setattr("osprey.capabilities.memory._perform_memory_save_operation", mock_save)

        # Create instance and inject state/step
        capability = MemoryOperationsCapability()
        capability._state = mock_state
        capability._step = mock_step

        # Execute - validates self._state and self._step are accessible
        result = await capability.execute()

        assert isinstance(result, dict)
        assert "context_data" in result
