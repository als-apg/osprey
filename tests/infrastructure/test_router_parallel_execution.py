"""Tests for router parallel execution support."""

from unittest.mock import Mock, patch

import pytest

from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.infrastructure.router_node import _get_next_executable_batch, router_conditional_edge
from osprey.state import StateManager

try:
    from langgraph.types import Send
except ImportError:
    Send = None


def test_get_next_executable_batch_no_dependencies():
    """Test finding executable steps when all steps are independent."""
    steps = [
        PlannedStep(
            context_key="step_0",
            capability="pv_reader",
            task_objective="Get PV1",
            success_criteria="PV1 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_1",
            capability="pv_reader",
            task_objective="Get PV2",
            success_criteria="PV2 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_2",
            capability="pv_reader",
            task_objective="Get PV3",
            success_criteria="PV3 data retrieved",
            inputs=[],
        ),
    ]

    # No steps completed yet
    executable = _get_next_executable_batch(steps, set())

    # All three steps should be executable in parallel
    assert len(executable) == 3
    assert set(executable) == {0, 1, 2}


def test_get_next_executable_batch_with_dependencies():
    """Test finding executable steps with dependencies."""
    steps = [
        PlannedStep(
            context_key="step_0",
            capability="pv_reader",
            task_objective="Get PV1",
            success_criteria="PV1 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_1",
            capability="pv_reader",
            task_objective="Get PV2",
            success_criteria="PV2 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_2",
            capability="calculator",
            task_objective="Calculate average",
            success_criteria="Average calculated",
            inputs=[{"PV_DATA": "step_0"}, {"PV_DATA": "step_1"}],
        ),
    ]

    # No steps completed yet
    executable = _get_next_executable_batch(steps, set())

    # Only steps 0 and 1 should be executable (step 2 depends on them)
    assert len(executable) == 2
    assert set(executable) == {0, 1}

    # After completing steps 0 and 1
    executable = _get_next_executable_batch(steps, {0, 1})

    # Now step 2 should be executable
    assert executable == [2]


def test_get_next_executable_batch_sequential():
    """Test finding executable steps in sequential plan."""
    steps = [
        PlannedStep(
            context_key="step_0",
            capability="data_loader",
            task_objective="Load data",
            success_criteria="Data loaded",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_1",
            capability="data_processor",
            task_objective="Process data",
            success_criteria="Data processed",
            inputs=[{"RAW_DATA": "step_0"}],
        ),
        PlannedStep(
            context_key="step_2",
            capability="data_analyzer",
            task_objective="Analyze data",
            success_criteria="Analysis complete",
            inputs=[{"PROCESSED_DATA": "step_1"}],
        ),
    ]

    # No steps completed
    executable = _get_next_executable_batch(steps, set())
    assert executable == [0]

    # After step 0
    executable = _get_next_executable_batch(steps, {0})
    assert executable == [1]

    # After steps 0 and 1
    executable = _get_next_executable_batch(steps, {0, 1})
    assert executable == [2]


def test_get_next_executable_batch_all_completed():
    """Test when all steps are completed."""
    steps = [
        PlannedStep(
            context_key="step_0",
            capability="pv_reader",
            task_objective="Get PV1",
            success_criteria="PV1 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_1",
            capability="pv_reader",
            task_objective="Get PV2",
            success_criteria="PV2 data retrieved",
            inputs=[],
        ),
    ]

    # All steps completed
    executable = _get_next_executable_batch(steps, {0, 1})

    # No steps should be executable
    assert executable == []


def test_get_next_executable_batch_mixed_dependencies():
    """Test with mix of independent and dependent steps."""
    steps = [
        PlannedStep(
            context_key="step_0",
            capability="pv_reader",
            task_objective="Get PV1",
            success_criteria="PV1 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_1",
            capability="pv_reader",
            task_objective="Get PV2",
            success_criteria="PV2 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_2",
            capability="pv_reader",
            task_objective="Get PV3",
            success_criteria="PV3 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_3",
            capability="calculator",
            task_objective="Calculate sum",
            success_criteria="Sum calculated",
            inputs=[{"PV_DATA": "step_0"}, {"PV_DATA": "step_1"}, {"PV_DATA": "step_2"}],
        ),
    ]

    # No steps completed
    executable = _get_next_executable_batch(steps, set())

    # Steps 0, 1, 2 should be executable in parallel
    assert len(executable) == 3
    assert set(executable) == {0, 1, 2}

    # After completing all independent steps
    executable = _get_next_executable_batch(steps, {0, 1, 2})

    # Step 3 should now be executable
    assert executable == [3]


def test_get_next_executable_batch_partial_completion():
    """Test with some steps completed."""
    steps = [
        PlannedStep(
            context_key="step_0",
            capability="pv_reader",
            task_objective="Get PV1",
            success_criteria="PV1 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_1",
            capability="pv_reader",
            task_objective="Get PV2",
            success_criteria="PV2 data retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_2",
            capability="pv_reader",
            task_objective="Get PV3",
            success_criteria="PV3 data retrieved",
            inputs=[],
        ),
    ]

    # Step 0 already completed
    executable = _get_next_executable_batch(steps, {0})

    # Steps 1 and 2 should be executable
    assert len(executable) == 2
    assert set(executable) == {1, 2}


# ===== SEND() COMMAND GENERATION TESTS =====


@pytest.fixture
def mock_registry():
    """Mock registry that returns True for all capability lookups."""
    registry = Mock()
    registry.get_node = Mock(return_value=True)
    return registry


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
@patch("osprey.infrastructure.router_node.get_registry")
def test_router_generates_send_commands_for_parallel_steps(mock_get_registry, mock_registry):
    """Test that router generates Send() commands with correct step indices."""
    mock_get_registry.return_value = mock_registry

    # Create plan with 3 independent steps
    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="pv_reader",
                task_objective="Get PV1",
                success_criteria="PV1 retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_1",
                capability="pv_reader",
                task_objective="Get PV2",
                success_criteria="PV2 retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_2",
                capability="pv_reader",
                task_objective="Get PV3",
                success_criteria="PV3 retrieved",
                inputs=[],
            ),
        ]
    )

    # Create state with parallel execution enabled
    state = StateManager.create_fresh_state("Get PV1, PV2, PV3")
    state["task_current_task"] = "Get PV1, PV2, PV3"
    state["planning_active_capabilities"] = ["pv_reader"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True

    result = router_conditional_edge(state)

    # Should return list of Send commands
    assert isinstance(result, list)
    assert len(result) == 3

    # Each should be a Send command
    for send_cmd in result:
        assert isinstance(send_cmd, Send)
        assert send_cmd.node == "pv_reader"


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
@patch("osprey.infrastructure.router_node.get_registry")
def test_send_commands_include_correct_step_indices(mock_get_registry, mock_registry):
    """Test that Send commands include the correct step index in state."""
    mock_get_registry.return_value = mock_registry

    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="pv_reader",
                task_objective="Get PV1",
                success_criteria="PV1 retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_1",
                capability="pv_reader",
                task_objective="Get PV2",
                success_criteria="PV2 retrieved",
                inputs=[],
            ),
        ]
    )

    state = StateManager.create_fresh_state("Get PV1, PV2")
    state["task_current_task"] = "Get PV1, PV2"
    state["planning_active_capabilities"] = ["pv_reader"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True

    result = router_conditional_edge(state)

    # Extract step indices from Send commands
    step_indices = [send_cmd.arg["planning_current_step_index"] for send_cmd in result]

    # Should have step indices 0 and 1
    assert set(step_indices) == {0, 1}


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
@patch("osprey.infrastructure.router_node.get_registry")
def test_router_filters_respond_clarify_from_parallel_batch(mock_get_registry, mock_registry):
    """Test that respond/clarify steps are not included in parallel batches."""
    mock_get_registry.return_value = mock_registry

    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="pv_reader",
                task_objective="Get PV1",
                success_criteria="PV1 retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_1",
                capability="pv_reader",
                task_objective="Get PV2",
                success_criteria="PV2 retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_2",
                capability="respond",
                task_objective="Respond to user",
                success_criteria="Response sent",
                inputs=[],
            ),
        ]
    )

    state = StateManager.create_fresh_state("Get PV1, PV2")
    state["task_current_task"] = "Get PV1, PV2"
    state["planning_active_capabilities"] = ["pv_reader", "respond"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True

    result = router_conditional_edge(state)

    # Should only return 2 Send commands (not including respond)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(send_cmd.node == "pv_reader" for send_cmd in result)


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
@patch("osprey.infrastructure.router_node.get_registry")
def test_router_uses_send_for_single_final_step(mock_get_registry, mock_registry):
    """Test that router uses Send() for single final step after parallel batch."""
    mock_get_registry.return_value = mock_registry

    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="pv_reader",
                task_objective="Get PV1",
                success_criteria="PV1 retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_1",
                capability="respond",
                task_objective="Respond to user",
                success_criteria="Response sent",
                inputs=[{"PV_DATA": "step_0"}],
            ),
        ]
    )

    # Step 0 completed, only step 1 (respond) remaining
    state = StateManager.create_fresh_state("Get PV1")
    state["task_current_task"] = "Get PV1"
    state["planning_active_capabilities"] = ["pv_reader", "respond"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {"step_0": {"step_index": 0, "success": True}}
    state["agent_control"]["parallel_execution_enabled"] = True

    result = router_conditional_edge(state)

    # Should return list with single Send command
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].node == "respond"
    assert result[0].arg["planning_current_step_index"] == 1


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
@patch("osprey.infrastructure.router_node.get_registry")
def test_router_validates_capabilities_before_send(mock_get_registry):
    """Test that router validates capabilities exist before creating Send commands."""
    # Mock registry that returns None for unknown capabilities
    registry = Mock()
    registry.get_node = Mock(side_effect=lambda name: True if name == "pv_reader" else None)
    mock_get_registry.return_value = registry

    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="pv_reader",
                task_objective="Get PV1",
                success_criteria="PV1 retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_1",
                capability="unknown_capability",
                task_objective="Unknown step",
                success_criteria="Unknown",
                inputs=[],
            ),
        ]
    )

    state = StateManager.create_fresh_state("Test")
    state["task_current_task"] = "Test"
    state["planning_active_capabilities"] = ["pv_reader", "unknown_capability"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True

    result = router_conditional_edge(state)

    # Should only create Send for valid capability
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].node == "pv_reader"


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
@patch("osprey.infrastructure.router_node.get_registry")
def test_router_fallback_when_no_valid_send_commands(mock_get_registry):
    """Test router falls back to sequential when no valid Send commands."""
    # Mock registry that returns None for all capabilities
    registry = Mock()
    registry.get_node = Mock(return_value=None)
    mock_get_registry.return_value = registry

    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="unknown_capability",
                task_objective="Unknown",
                success_criteria="Unknown",
                inputs=[],
            ),
        ]
    )

    state = StateManager.create_fresh_state("Test")
    state["task_current_task"] = "Test"
    state["planning_active_capabilities"] = ["unknown_capability"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True

    result = router_conditional_edge(state)

    # Should fall back to error routing
    assert result == "error"
