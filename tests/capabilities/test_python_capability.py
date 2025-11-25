"""Tests for PythonCapability instance method pattern migration."""

import inspect

import pytest

from osprey.capabilities.python import PythonCapability


class TestPythonCapabilityMigration:
    """Test PythonCapability successfully migrated to instance method pattern."""

    def test_uses_instance_method_not_static(self):
        """Verify execute() migrated from @staticmethod to instance method."""
        execute_method = inspect.getattr_static(PythonCapability, "execute")
        assert not isinstance(execute_method, staticmethod)

        sig = inspect.signature(PythonCapability.execute)
        params = list(sig.parameters.keys())
        assert params == ["self"]

    def test_state_can_be_injected(self, mock_state, mock_step):
        """Verify capability instance can receive _state and _step injection."""
        capability = PythonCapability()
        capability._state = mock_state
        capability._step = mock_step

        assert capability._state == mock_state
        assert capability._step == mock_step

    def test_has_langgraph_node_decorator(self):
        """Verify @capability_node decorator created langgraph_node attribute."""
        assert hasattr(PythonCapability, "langgraph_node")
        assert callable(PythonCapability.langgraph_node)
