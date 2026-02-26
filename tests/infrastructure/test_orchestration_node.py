"""Tests for orchestration node - execution planning and capability orchestration."""

import inspect
from unittest.mock import Mock, patch

import pytest

from osprey.base.errors import InvalidContextKeyError
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
        state = {"capability_context_data": {}}  # Empty state
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_registry.return_value = mock_reg

            result = _validate_and_fix_execution_plan(empty_plan, "test task", state, logger)

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
                    inputs=[{"RESULT": "step1"}],  # Valid reference to earlier step
                ),
            ]
        }
        state = {"capability_context_data": {}}  # Empty state
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()  # All capabilities exist
            mock_registry.return_value = mock_reg

            result = _validate_and_fix_execution_plan(valid_plan, "test task", state, logger)

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
        state = {"capability_context_data": {}}  # Empty state
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()
            mock_registry.return_value = mock_reg

            result = _validate_and_fix_execution_plan(
                plan_without_respond, "test task", state, logger
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
        state = {"capability_context_data": {}}  # Empty state
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = None  # Capability doesn't exist
            mock_reg.get_stats.return_value = {"capability_names": ["python", "respond"]}
            mock_registry.return_value = mock_reg

            with pytest.raises(ValueError) as exc_info:
                _validate_and_fix_execution_plan(bad_plan, "test task", state, logger)

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
        state = {"capability_context_data": {}}  # Empty state
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()
            mock_registry.return_value = mock_reg

            result = _validate_and_fix_execution_plan(plan_with_clarify, "test task", state, logger)

            # Should not append respond since it ends with clarify
            assert len(result["steps"]) == 1
            assert result["steps"][0]["capability"] == "clarify"

    def test_plan_with_invalid_context_key_raises_error(self):
        """Test plan with invalid context key reference raises InvalidContextKeyError."""
        bad_plan: ExecutionPlan = {
            "steps": [
                PlannedStep(
                    context_key="step1",
                    capability="python",
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
                    inputs=[{"DATA": "nonexistent_key"}],  # Invalid reference
                ),
            ]
        }
        state = {"capability_context_data": {}}  # Empty state
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()  # All capabilities exist
            mock_registry.return_value = mock_reg

            with pytest.raises(InvalidContextKeyError) as exc_info:
                _validate_and_fix_execution_plan(bad_plan, "test task", state, logger)

            # Error message should mention the invalid key and available keys
            assert "nonexistent_key" in str(exc_info.value)
            assert "step1" in str(exc_info.value)  # Available key should be listed

    def test_plan_with_forward_reference_raises_error(self):
        """Test plan where step references a key from a later step raises error."""
        bad_plan: ExecutionPlan = {
            "steps": [
                PlannedStep(
                    context_key="step1",
                    capability="python",
                    task_objective="Run code",
                    expected_output="result",
                    success_criteria="Code runs",
                    inputs=[{"DATA": "step2"}],  # Forward reference to later step
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
        state = {"capability_context_data": {}}  # Empty state
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()  # All capabilities exist
            mock_registry.return_value = mock_reg

            with pytest.raises(InvalidContextKeyError) as exc_info:
                _validate_and_fix_execution_plan(bad_plan, "test task", state, logger)

            # Error message should explain the ordering issue
            assert "later" in str(exc_info.value).lower()

    def test_plan_with_existing_context_key_passes(self):
        """Test plan that references existing context from state passes validation."""
        plan: ExecutionPlan = {
            "steps": [
                PlannedStep(
                    context_key="step1",
                    capability="python",
                    task_objective="Run code",
                    expected_output="result",
                    success_criteria="Code runs",
                    inputs=[{"DATA": "existing_data"}],  # Reference to existing context
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
        # State has existing context that the plan references
        state = {"capability_context_data": {"DATA": {"existing_data": {"some": "value"}}}}
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_node.return_value = Mock()  # All capabilities exist
            mock_registry.return_value = mock_reg

            # Should pass validation without error
            result = _validate_and_fix_execution_plan(plan, "test task", state, logger)
            assert len(result["steps"]) == 2


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
        assert (
            "retry" in classification.user_message.lower()
            or "timeout" in classification.user_message.lower()
        )

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

    def test_classify_pydantic_validation_error(self):
        """Test Pydantic validation errors are retriable."""
        exc = ValueError("validation error for ExecutionPlan")
        context = {"operation": "llm_call"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "retriable"
        assert "valid execution plan" in classification.user_message.lower()

    def test_classify_json_parsing_error(self):
        """Test JSON parsing errors are retriable."""
        exc = ValueError("JSON parsing failed")
        context = {"operation": "llm_call"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "retriable"

    def test_classify_import_error(self):
        """Test ImportError is classified as critical."""
        exc = ImportError("No module named 'missing_module'")
        context = {"operation": "initialization"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "critical"
        assert "infrastructure dependency" in classification.user_message.lower()

    def test_classify_module_not_found_error(self):
        """Test ModuleNotFoundError is classified as critical."""
        exc = ModuleNotFoundError("No module named 'xyz'")
        context = {"operation": "initialization"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "critical"

    def test_classify_reclassification_required_error(self):
        """Test ReclassificationRequiredError is handled correctly."""
        from osprey.base.errors import ReclassificationRequiredError

        exc = ReclassificationRequiredError("Need to reclassify task")
        context = {"operation": "planning"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "reclassification"
        assert "reclassification" in classification.user_message.lower()

    def test_classify_invalid_context_key_error(self):
        """Test InvalidContextKeyError triggers replanning (not reclassification)."""
        exc = InvalidContextKeyError("Invalid key 'bad_key' not found")
        context = {"operation": "validation"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "replanning"
        assert "context key" in classification.user_message.lower()

    def test_classify_unknown_error(self):
        """Test unknown errors default to critical."""
        exc = RuntimeError("Something unexpected happened")
        context = {"operation": "unknown"}

        classification = OrchestrationNode.classify_error(exc, context)

        assert classification.severity.value == "critical"
        assert "unknown" in classification.user_message.lower()


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions used by orchestration node."""

    def test_is_planning_mode_enabled_true(self):
        """Test planning mode detection when enabled."""
        from osprey.infrastructure.orchestration_node import _is_planning_mode_enabled

        state = {"agent_control": {"planning_mode_enabled": True}}

        assert _is_planning_mode_enabled(state) is True

    def test_is_planning_mode_enabled_false(self):
        """Test planning mode detection when disabled."""
        from osprey.infrastructure.orchestration_node import _is_planning_mode_enabled

        state = {"agent_control": {"planning_mode_enabled": False}}

        assert _is_planning_mode_enabled(state) is False

    def test_is_planning_mode_enabled_missing_key(self):
        """Test planning mode detection with missing keys."""
        from osprey.infrastructure.orchestration_node import _is_planning_mode_enabled

        state = {}

        assert _is_planning_mode_enabled(state) is False

    def test_clear_error_state(self):
        """Test error state clearing."""
        from osprey.infrastructure.orchestration_node import _clear_error_state

        result = _clear_error_state()

        assert result["control_has_error"] is False
        assert result["control_error_info"] is None
        assert result["control_last_error"] is None
        assert result["control_retry_count"] == 0
        assert result["control_current_step_retry_count"] == 0

    def test_log_execution_plan(self):
        """Test execution plan logging."""
        from osprey.infrastructure.orchestration_node import _log_execution_plan

        plan = {
            "steps": [
                {
                    "context_key": "step1",
                    "capability": "python",
                    "task_objective": "Run code",
                    "inputs": ["input1"],
                }
            ]
        }
        logger = Mock()

        _log_execution_plan(plan, logger)

        # Verify logger.key_info was called (execution plan logs use key_info())
        assert logger.key_info.called

    def test_save_execution_plan_to_file_success(self, tmp_path):
        """Test saving execution plan to file."""
        from osprey.infrastructure.orchestration_node import _save_execution_plan_to_file

        plan = {"steps": [{"capability": "python", "task_objective": "test"}]}
        current_task = "Test task"
        state = {"messages": [{"role": "user", "content": "Original query"}]}
        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_agent_dir") as mock_get_dir:
            mock_get_dir.return_value = str(tmp_path)

            result = _save_execution_plan_to_file(plan, current_task, state, logger)

            assert result["success"] is True
            assert "file_path" in result
            assert "pending_plans_dir" in result

    def test_load_execution_plan_from_file_not_found(self):
        """Test loading execution plan when file doesn't exist."""
        from osprey.infrastructure.orchestration_node import _load_execution_plan_from_file

        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_agent_dir") as mock_get_dir:
            mock_get_dir.return_value = "/nonexistent/path"

            result = _load_execution_plan_from_file(logger)

            assert result["success"] is False
            assert "error" in result

    def test_cleanup_processed_plan_files(self, tmp_path):
        """Test cleanup of processed plan files."""
        from osprey.infrastructure.orchestration_node import _cleanup_processed_plan_files

        # Create dummy plan files
        pending_dir = tmp_path / "pending_plans"
        pending_dir.mkdir()
        pending_file = pending_dir / "pending_execution_plan.json"
        pending_file.write_text("{}")
        modified_file = pending_dir / "modified_execution_plan.json"
        modified_file.write_text("{}")

        logger = Mock()

        with patch("osprey.infrastructure.orchestration_node.get_agent_dir") as mock_get_dir:
            mock_get_dir.return_value = str(tmp_path)

            _cleanup_processed_plan_files(logger)

            # Files should be removed
            assert not pending_file.exists()
            assert not modified_file.exists()

    def test_create_state_updates(self):
        """Test state updates creation."""
        from osprey.infrastructure.orchestration_node import _create_state_updates

        state = {"control_plans_created_count": 5}
        plan = {"steps": [{"capability": "python"}]}
        approach = "llm_based"

        updates = _create_state_updates(state, plan, approach)

        assert updates["planning_execution_plan"] == plan
        assert updates["planning_current_step_index"] == 0
        assert updates["control_plans_created_count"] == 6
        assert updates["control_has_error"] is False


# =============================================================================
# Test Retry Policy
# =============================================================================


class TestRetryPolicy:
    """Test custom retry policy for orchestration."""

    def test_get_retry_policy(self):
        """Test retry policy returns correct values."""
        policy = OrchestrationNode.get_retry_policy()

        assert "max_attempts" in policy
        assert "delay_seconds" in policy
        assert "backoff_factor" in policy
        assert policy["max_attempts"] >= 3
        assert policy["delay_seconds"] > 0
        assert policy["backoff_factor"] > 1
