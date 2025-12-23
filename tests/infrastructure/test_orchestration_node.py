"""Tests for orchestration node - execution planning and capability orchestration."""

import inspect
from unittest.mock import Mock, patch

import pytest

from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.infrastructure.orchestration_node import (
    OrchestrationNode,
    _validate_and_fix_execution_plan,
)


# =============================================================================
# Test Plan Validation (Helper Function)
# =============================================================================


class TestPlanValidation:
    """Test execution plan validation and fixing logic."""

    def test_empty_plan_gets_default_respond_step(self):
        """Test that empty plan gets a default respond step."""
        empty_plan: ExecutionPlan = {"steps": []}
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_registry.return_value = mock_reg

            result = _validate_and_fix_execution_plan(empty_plan, "test task", logger)

            assert len(result["steps"]) == 1
            assert result["steps"][0]["capability"] == "respond"
            assert logger.warning.called

    def test_plan_with_valid_capabilities(self):
        """Test plan with all valid capabilities passes through."""
        valid_plan: ExecutionPlan = {
            "steps": [
                PlannedStep(
                    context_key="step1",
                    capability="python",  # Built-in capability
                    task_objective="Run code",
                    expected_output="result",
                    success_criteria="Code runs",
                    inputs=[],
                ),
                PlannedStep(
                    context_key="step2",
                    capability="respond",
                    task_objective="Respond",
                    expected_output="response",
                    success_criteria="Response given",
                    inputs=[],
                ),
            ]
        }
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()  # All capabilities exist
            mock_registry.return_value = mock_reg

            result = _validate_and_fix_execution_plan(valid_plan, "test task", logger)

            # Should keep both steps
            assert len(result["steps"]) == 2
            assert result["steps"][1]["capability"] == "respond"

    def test_plan_without_respond_gets_respond_appended(self):
        """Test plan without respond/clarify step gets respond appended."""
        plan_without_respond: ExecutionPlan = {
            "steps": [
                PlannedStep(
                    context_key="step1",
                    capability="python",
                    task_objective="Run code",
                    expected_output="result",
                    success_criteria="Code runs",
                    inputs=[],
                ),
            ]
        }
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()
            mock_registry.return_value = mock_reg

            result = _validate_and_fix_execution_plan(
                plan_without_respond, "test task", logger
            )

            # Should have original step plus respond
            assert len(result["steps"]) == 2
            assert result["steps"][0]["capability"] == "python"
            assert result["steps"][1]["capability"] == "respond"

    def test_plan_with_hallucinated_capability_raises_error(self):
        """Test plan with non-existent capability raises ValueError."""
        bad_plan: ExecutionPlan = {
            "steps": [
                PlannedStep(
                    context_key="step1",
                    capability="nonexistent_capability",
                    task_objective="Do something",
                    expected_output="result",
                    success_criteria="Success",
                    inputs=[],
                ),
            ]
        }
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = None  # Capability doesn't exist
            mock_reg.get_stats.return_value = {"capability_names": ["python", "respond"]}
            mock_registry.return_value = mock_reg

            with pytest.raises(ValueError) as exc_info:
                _validate_and_fix_execution_plan(bad_plan, "test task", logger)

            assert "hallucinated" in str(exc_info.value).lower()

    def test_plan_ending_with_clarify_not_modified(self):
        """Test plan ending with clarify step is not modified."""
        plan_with_clarify: ExecutionPlan = {
            "steps": [
                PlannedStep(
                    context_key="step1",
                    capability="clarify",
                    task_objective="Ask for clarification",
                    expected_output="clarification",
                    success_criteria="Question asked",
                    inputs=[],
                ),
            ]
        }
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()
            mock_registry.return_value = mock_reg

            result = _validate_and_fix_execution_plan(
                plan_with_clarify, "test task", logger
            )

            # Should not append respond since it ends with clarify
            assert len(result["steps"]) == 1
            assert result["steps"][0]["capability"] == "clarify"


# =============================================================================
# Test OrchestrationNode Class
# =============================================================================


class TestOrchestrationNode:
    """Test OrchestrationNode infrastructure node."""

    def test_node_exists_and_is_callable(self):
        """Verify OrchestrationNode can be instantiated."""
        node = OrchestrationNode()
        assert node is not None
        assert hasattr(node, "execute")

    def test_execute_is_instance_method(self):
        """Test execute() is an instance method, not static."""
        execute_method = inspect.getattr_static(OrchestrationNode, "execute")
        assert not isinstance(execute_method, staticmethod), (
            "OrchestrationNode.execute() should be instance method"
        )

    def test_has_langgraph_node_attribute(self):
        """Test that OrchestrationNode has langgraph_node from decorator."""
        assert hasattr(OrchestrationNode, "langgraph_node")
        assert callable(OrchestrationNode.langgraph_node)

    def test_classify_error_method_exists(self):
        """Test that classify_error static method exists."""
        assert hasattr(OrchestrationNode, "classify_error")
        assert callable(OrchestrationNode.classify_error)



# =============================================================================
# Test Error Classification
# =============================================================================


class TestOrchestrationErrorClassification:
    """Test error classification for orchestration operations."""

    def test_classify_timeout_error(self):
        """Test timeout errors are classified as retriable."""
        exc = TimeoutError("LLM request timeout")
        context = {"operation": "planning"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "retriable"
        assert "retry" in classification.user_message.lower() or "timeout" in classification.user_message.lower()

    def test_classify_value_error(self):
        """Test ValueError is classified as critical."""
        exc = ValueError("Invalid plan format")
        context = {"operation": "validation"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value in ["critical", "moderate"]

    def test_classify_connection_error(self):
        """Test connection errors are classified as retriable."""
        exc = ConnectionError("Network error")
        context = {"operation": "llm_call"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "retriable"



