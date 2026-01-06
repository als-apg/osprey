"""Test infrastructure_node decorator with instance method pattern."""

import inspect
from typing import Any

import pytest

from osprey.base.decorators import infrastructure_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.nodes import BaseInfrastructureNode
from osprey.state import AgentState


class TestDecoratorMethodDetection:
    """Test that decorator correctly detects static vs instance methods."""

    def test_detects_static_method_legacy_pattern(self):
        """Test decorator recognizes @staticmethod (legacy pattern)."""

        @infrastructure_node
        class LegacyNode(BaseInfrastructureNode):
            name = "legacy_test"
            description = "Legacy static method test"

            @staticmethod
            async def execute(state: AgentState, **kwargs) -> dict[str, Any]:
                return {"test": "legacy"}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL, user_message="Test error"
                )

        # Verify it's still static
        execute_method = inspect.getattr_static(LegacyNode, "execute")
        assert isinstance(execute_method, staticmethod), "Legacy execute() should remain static"

    def test_detects_instance_method_new_pattern(self):
        """Test decorator recognizes instance methods (new pattern)."""

        @infrastructure_node
        class NewNode(BaseInfrastructureNode):
            name = "new_test"
            description = "New instance method test"

            async def execute(self) -> dict[str, Any]:
                return {"test": "new"}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL, user_message="Test error"
                )

        # Verify it's an instance method (not static)
        execute_method = inspect.getattr_static(NewNode, "execute")
        assert not isinstance(execute_method, staticmethod), (
            "New execute() should be instance method"
        )
        assert callable(execute_method), "execute() must be callable"


class TestDecoratorStateInjection:
    """Test that decorator injects _state correctly."""

    @pytest.mark.asyncio
    async def test_injects_state_for_instance_method(self, base_state):
        """Test decorator injects self._state for instance methods."""

        state_captured = None

        @infrastructure_node
        class TestNode(BaseInfrastructureNode):
            name = "state_test"
            description = "State injection test"

            async def execute(self) -> dict[str, Any]:
                nonlocal state_captured
                # Verify we have access to self._state
                assert hasattr(self, "_state"), "Instance should have _state attribute"
                state_captured = self._state
                return {"test": "passed"}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL, user_message="Test error"
                )

        # Execute the node
        node_func = TestNode.langgraph_node
        result = await node_func(base_state)

        # Verify state was injected
        assert state_captured is not None, "State should have been captured"
        assert state_captured == base_state, "Injected state should match input state"
        assert result["test"] == "passed"


class TestDecoratorStepInjection:
    """Test that decorator injects _step only for specific nodes."""

    @pytest.mark.asyncio
    async def test_injects_step_for_clarify_node(self, state_with_plan):
        """Test decorator injects self._step for clarify node."""

        step_captured = None

        @infrastructure_node
        class ClarifyTestNode(BaseInfrastructureNode):
            name = "clarify"  # Must match NODES_NEEDING_STEP
            description = "Clarify step injection test"

            async def execute(self) -> dict[str, Any]:
                nonlocal step_captured
                # Verify we have access to self._step
                assert hasattr(self, "_step"), "Clarify node should have _step attribute"
                step_captured = self._step
                return {"test": "passed"}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL, user_message="Test error"
                )

        # Execute the node
        node_func = ClarifyTestNode.langgraph_node
        result = await node_func(state_with_plan)

        # Verify step was injected
        assert step_captured is not None, "Step should have been captured"
        assert step_captured.get("step_index") == 0
        assert result["test"] == "passed"

    @pytest.mark.asyncio
    async def test_injects_step_for_respond_node(self, state_with_plan):
        """Test decorator injects self._step for respond node."""

        step_captured = None

        @infrastructure_node
        class RespondTestNode(BaseInfrastructureNode):
            name = "respond"  # Must match NODES_NEEDING_STEP
            description = "Respond step injection test"

            async def execute(self) -> dict[str, Any]:
                nonlocal step_captured
                # Verify we have access to self._step
                assert hasattr(self, "_step"), "Respond node should have _step attribute"
                step_captured = self._step
                return {"test": "passed"}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL, user_message="Test error"
                )

        # Execute the node
        node_func = RespondTestNode.langgraph_node
        result = await node_func(state_with_plan)

        # Verify step was injected
        assert step_captured is not None, "Step should have been captured"
        assert result["test"] == "passed"

    @pytest.mark.asyncio
    async def test_no_step_injection_for_router_node(self, base_state):
        """Test decorator does NOT inject _step for router node."""

        @infrastructure_node
        class RouterTestNode(BaseInfrastructureNode):
            name = "router"  # NOT in NODES_NEEDING_STEP
            description = "Router no-step test"

            async def execute(self) -> dict[str, Any]:
                # Verify we do NOT have _step
                assert not hasattr(self, "_step"), "Router node should NOT have _step attribute"
                return {"test": "passed"}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL, user_message="Test error"
                )

        # Execute the node
        node_func = RouterTestNode.langgraph_node
        result = await node_func(base_state)

        assert result["test"] == "passed"

    @pytest.mark.asyncio
    async def test_no_step_injection_for_error_node(self, base_state):
        """Test decorator does NOT inject _step for error node."""

        @infrastructure_node
        class ErrorTestNode(BaseInfrastructureNode):
            name = "error"  # NOT in NODES_NEEDING_STEP
            description = "Error no-step test"

            async def execute(self) -> dict[str, Any]:
                # Verify we do NOT have _step
                assert not hasattr(self, "_step"), "Error node should NOT have _step attribute"
                # Error node uses StateManager.get_current_step_index() instead
                return {"test": "passed"}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL, user_message="Test error"
                )

        # Execute the node
        node_func = ErrorTestNode.langgraph_node
        result = await node_func(base_state)

        assert result["test"] == "passed"


class TestDecoratorBackwardCompatibility:
    """Test decorator maintains backward compatibility with static methods."""

    @pytest.mark.asyncio
    async def test_static_method_still_works(self, base_state):
        """Test that static methods (legacy pattern) still work."""

        @infrastructure_node
        class LegacyStaticNode(BaseInfrastructureNode):
            name = "legacy_compat"
            description = "Legacy compatibility test"

            @staticmethod
            async def execute(state: AgentState, **kwargs) -> dict[str, Any]:
                # Old pattern - receives state as parameter
                assert state is not None
                assert "messages" in state
                return {"legacy": "works", "count": state.get("control_routing_count", 0)}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL, user_message="Test error"
                )

        # Execute the node
        node_func = LegacyStaticNode.langgraph_node
        result = await node_func(base_state)

        assert result["legacy"] == "works"
        assert result["count"] == 0


class TestDecoratorValidation:
    """Test decorator validates method types correctly."""

    def test_rejects_classmethod(self):
        """Test decorator rejects @classmethod on execute()."""

        with pytest.raises(ValueError, match="cannot be a classmethod"):

            @infrastructure_node
            class BadNode(BaseInfrastructureNode):
                name = "bad_node"
                description = "Invalid classmethod test"

                @classmethod
                async def execute(cls, state: AgentState) -> dict[str, Any]:
                    return {}

                @staticmethod
                def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                    return ErrorClassification(
                        severity=ErrorSeverity.CRITICAL, user_message="Test error"
                    )

    def test_rejects_property(self):
        """Test decorator rejects @property on execute()."""

        with pytest.raises(ValueError, match="cannot be a property"):

            @infrastructure_node
            class BadNode(BaseInfrastructureNode):
                name = "bad_node"
                description = "Invalid property test"

                @property
                def execute(self) -> dict[str, Any]:
                    return {}

                @staticmethod
                def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                    return ErrorClassification(
                        severity=ErrorSeverity.CRITICAL, user_message="Test error"
                    )
