"""Integration tests for parallel execution functionality.

These tests verify that the parallel execution system works correctly end-to-end,
including state management, dependency analysis, and LangGraph integration.
"""

from unittest.mock import Mock, patch

import pytest

from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.infrastructure.router_node import router_conditional_edge
from osprey.state import StateManager


@pytest.fixture
def mock_langgraph_context():
    """Mock LangGraph context to avoid 'Called get_config outside of a runnable context' errors."""
    with (
        patch("osprey.base.decorators.get_stream_writer", return_value=None),
        patch("osprey.base.decorators.get_config", return_value={}),
    ):
        yield


@pytest.fixture
def mock_registry():
    """Mock registry that returns True for all capability lookups."""
    registry = Mock()
    registry.get_node = Mock(return_value=True)
    return registry


@patch("osprey.infrastructure.router_node.get_registry")
def test_router_returns_parallel_steps_for_independent_tasks(mock_get_registry, mock_registry):
    """Test that router returns list of capabilities for independent steps."""
    mock_get_registry.return_value = mock_registry

    # Create execution plan with 3 independent steps
    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="pv_reader",
                task_objective="Get PV1 current value",
                success_criteria="PV1 value retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_1",
                capability="pv_reader",
                task_objective="Get PV2 current value",
                success_criteria="PV2 value retrieved",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_2",
                capability="pv_reader",
                task_objective="Get PV3 current value",
                success_criteria="PV3 value retrieved",
                inputs=[],
            ),
        ]
    )

    # Create state with plan
    state = StateManager.create_fresh_state("Get PV1, PV2, PV3")
    state["task_current_task"] = "Get PV1, PV2, PV3"
    state["planning_active_capabilities"] = ["pv_reader"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True  # Enable parallel execution

    # Router should return list of 3 capabilities for parallel execution
    result = router_conditional_edge(state)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(cap == "pv_reader" for cap in result)


@patch("osprey.infrastructure.router_node.get_registry")
def test_router_returns_single_step_after_parallel_completion(mock_get_registry, mock_registry):
    """Test that router returns single capability after parallel steps complete."""
    mock_get_registry.return_value = mock_registry

    # Create execution plan: 2 parallel reads, then 1 calculation
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
                capability="calculator",
                task_objective="Calculate average",
                success_criteria="Average calculated",
                inputs=[{"PV_DATA": "step_0"}, {"PV_DATA": "step_1"}],
            ),
        ]
    )

    # Create state with completed parallel steps
    state = StateManager.create_fresh_state("Get PV1, PV2, calculate average")
    state["task_current_task"] = "Get PV1, PV2, calculate average"
    state["planning_active_capabilities"] = ["pv_reader", "calculator"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {
        "step_0": {"step_index": 0, "success": True},
        "step_1": {"step_index": 1, "success": True},
    }
    state["agent_control"]["parallel_execution_enabled"] = True  # Enable parallel execution

    # Router should return single capability for dependent step
    result = router_conditional_edge(state)

    assert isinstance(result, str)
    assert result == "calculator"


def test_state_reducer_merges_parallel_results():
    """Test that execution_step_results reducer merges parallel execution results."""
    from osprey.state import merge_execution_step_results

    # Simulate parallel execution of 3 steps
    existing = {}

    # First parallel step completes
    step_0_result = {"step_0": {"step_index": 0, "success": True, "result": "PV1 data"}}
    state_after_0 = merge_execution_step_results(existing, step_0_result)

    # Second parallel step completes
    step_1_result = {"step_1": {"step_index": 1, "success": True, "result": "PV2 data"}}
    state_after_1 = merge_execution_step_results(state_after_0, step_1_result)

    # Third parallel step completes
    step_2_result = {"step_2": {"step_index": 2, "success": True, "result": "PV3 data"}}
    final_state = merge_execution_step_results(state_after_1, step_2_result)

    # All three results should be present
    assert len(final_state) == 3
    assert "step_0" in final_state
    assert "step_1" in final_state
    assert "step_2" in final_state
    assert final_state["step_0"]["result"] == "PV1 data"
    assert final_state["step_1"]["result"] == "PV2 data"
    assert final_state["step_2"]["result"] == "PV3 data"


@patch("osprey.infrastructure.router_node.get_registry")
def test_router_handles_sequential_execution(mock_get_registry, mock_registry):
    """Test that router still handles sequential execution correctly."""
    mock_get_registry.return_value = mock_registry

    # Create execution plan with sequential dependencies
    plan = ExecutionPlan(
        steps=[
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
    )

    # Test step 0 (no dependencies)
    state = StateManager.create_fresh_state("Load, process, analyze data")
    state["task_current_task"] = "Load, process, analyze data"
    state["planning_active_capabilities"] = ["data_loader", "data_processor", "data_analyzer"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True  # Enable parallel execution

    result = router_conditional_edge(state)
    assert result == "data_loader"

    # Test step 1 (depends on step 0)
    state["execution_step_results"] = {"step_0": {"step_index": 0, "success": True}}
    result = router_conditional_edge(state)
    assert result == "data_processor"

    # Test step 2 (depends on step 1)
    state["execution_step_results"] = {
        "step_0": {"step_index": 0, "success": True},
        "step_1": {"step_index": 1, "success": True},
    }
    result = router_conditional_edge(state)
    assert result == "data_analyzer"


@patch("osprey.infrastructure.router_node.get_registry")
def test_router_handles_mixed_parallel_and_sequential(mock_get_registry, mock_registry):
    """Test router with mix of parallel and sequential steps."""
    mock_get_registry.return_value = mock_registry

    # Plan: 3 parallel reads, then 1 calculation
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
            PlannedStep(
                context_key="step_3",
                capability="calculator",
                task_objective="Calculate sum",
                success_criteria="Sum calculated",
                inputs=[{"PV_DATA": "step_0"}, {"PV_DATA": "step_1"}, {"PV_DATA": "step_2"}],
            ),
        ]
    )

    # Initial state - should return 3 parallel steps
    state = StateManager.create_fresh_state("Get PV1, PV2, PV3, calculate sum")
    state["task_current_task"] = "Get PV1, PV2, PV3, calculate sum"
    state["planning_active_capabilities"] = ["pv_reader", "calculator"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True  # Enable parallel execution

    result = router_conditional_edge(state)
    assert isinstance(result, list)
    assert len(result) == 3

    # After parallel completion - should return single calculator step
    state["execution_step_results"] = {
        "step_0": {"step_index": 0, "success": True},
        "step_1": {"step_index": 1, "success": True},
        "step_2": {"step_index": 2, "success": True},
    }

    result = router_conditional_edge(state)
    assert isinstance(result, str)
    assert result == "calculator"


@patch("osprey.infrastructure.router_node.get_registry")
def test_router_skips_completed_steps(mock_get_registry, mock_registry):
    """Test that router doesn't return already completed steps."""
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
                capability="pv_reader",
                task_objective="Get PV3",
                success_criteria="PV3 retrieved",
                inputs=[],
            ),
        ]
    )

    # Step 0 already completed
    state = StateManager.create_fresh_state("Get PV1, PV2, PV3")
    state["task_current_task"] = "Get PV1, PV2, PV3"
    state["planning_active_capabilities"] = ["pv_reader"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {"step_0": {"step_index": 0, "success": True}}
    state["agent_control"]["parallel_execution_enabled"] = True  # Enable parallel execution

    result = router_conditional_edge(state)

    # Should return only steps 1 and 2
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(cap == "pv_reader" for cap in result)


# ===== END-TO-END PARALLEL EXECUTION TESTS =====


@pytest.mark.asyncio
async def test_parallel_execution_with_actual_capabilities(mock_langgraph_context, mock_registry):
    """Test end-to-end parallel execution with actual capability execution."""
    from osprey.base.capability import BaseCapability
    from osprey.base.decorators import capability_node

    @capability_node
    class TestParallelCapability(BaseCapability):
        name = "test_parallel"
        description = "Test capability for parallel execution"

        async def execute(self):
            step_key = self._step.get("context_key", "test")
            return {
                "capability_context_data": {
                    "TEST_RESULTS": {step_key: {"data": f"result_{step_key}"}}
                }
            }

    # Create plan with 2 parallel steps
    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="test_parallel",
                task_objective="Execute step 0",
                success_criteria="Step 0 complete",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_1",
                capability="test_parallel",
                task_objective="Execute step 1",
                success_criteria="Step 1 complete",
                inputs=[],
            ),
        ]
    )

    # Create state with parallel execution enabled
    state = StateManager.create_fresh_state("Test parallel execution")
    state["task_current_task"] = "Test parallel execution"
    state["planning_active_capabilities"] = ["test_parallel"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True

    # Execute step 0
    state["planning_current_step_index"] = 0
    result_0 = await TestParallelCapability.langgraph_node(state)

    # Execute step 1
    state["planning_current_step_index"] = 1
    result_1 = await TestParallelCapability.langgraph_node(state)

    # Both should have execution_step_results
    assert "execution_step_results" in result_0
    assert "execution_step_results" in result_1

    # Both should have their respective step keys
    assert "step_0" in result_0["execution_step_results"]
    assert "step_1" in result_1["execution_step_results"]

    # Both should be successful
    assert result_0["execution_step_results"]["step_0"]["success"] is True
    assert result_1["execution_step_results"]["step_1"]["success"] is True


@pytest.mark.asyncio
async def test_error_handling_during_parallel_execution(mock_langgraph_context):
    """Test that errors during parallel execution are handled correctly."""
    from osprey.base.capability import BaseCapability
    from osprey.base.decorators import capability_node

    @capability_node
    class FailingParallelCapability(BaseCapability):
        name = "failing_parallel"
        description = "Capability that fails during parallel execution"

        async def execute(self):
            step_index = self._state.get("planning_current_step_index", 0)
            if step_index == 1:
                raise ValueError("Simulated error in step 1")
            return {
                "capability_context_data": {
                    "TEST_RESULTS": {self._step.get("context_key", "test"): {"data": "success"}}
                }
            }

    plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key="step_0",
                capability="failing_parallel",
                task_objective="Execute step 0",
                success_criteria="Step 1 complete",
                inputs=[],
            ),
            PlannedStep(
                context_key="step_1",
                capability="failing_parallel",
                task_objective="Execute step 1",
                success_criteria="Step 1 complete",
                inputs=[],
            ),
        ]
    )

    state = StateManager.create_fresh_state("Test error handling")
    state["task_current_task"] = "Test error handling"
    state["planning_active_capabilities"] = ["failing_parallel"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True

    # Execute step 0 (should succeed)
    state["planning_current_step_index"] = 0
    result_0 = await FailingParallelCapability.langgraph_node(state)
    assert result_0["execution_step_results"]["step_0"]["success"] is True

    # Execute step 1 (should fail)
    state["planning_current_step_index"] = 1
    result_1 = await FailingParallelCapability.langgraph_node(state)
    assert result_1["control_has_error"] is True
    assert result_1["execution_step_results"]["step_1"]["success"] is False


@pytest.mark.asyncio
async def test_step_result_accumulation_across_parallel_batches():
    """Test that step results accumulate correctly across multiple parallel batches."""
    from osprey.state import merge_execution_step_results

    # Simulate first parallel batch (steps 0, 1, 2)
    batch_1_results = {
        "step_0": {"step_index": 0, "success": True, "capability": "reader"},
        "step_1": {"step_index": 1, "success": True, "capability": "reader"},
        "step_2": {"step_index": 2, "success": True, "capability": "reader"},
    }

    # Merge first batch
    state_after_batch_1 = merge_execution_step_results({}, batch_1_results)

    # Simulate second parallel batch (steps 3, 4)
    batch_2_results = {
        "step_3": {"step_index": 3, "success": True, "capability": "processor"},
        "step_4": {"step_index": 4, "success": True, "capability": "processor"},
    }

    # Merge second batch
    final_state = merge_execution_step_results(state_after_batch_1, batch_2_results)

    # All 5 results should be present
    assert len(final_state) == 5
    assert all(f"step_{i}" in final_state for i in range(5))
    assert all(final_state[f"step_{i}"]["success"] for i in range(5))


@patch("osprey.infrastructure.router_node.get_registry")
def test_parallel_execution_disabled_uses_sequential_mode(mock_get_registry, mock_registry):
    """Test that disabling parallel execution falls back to sequential mode."""
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
    state["agent_control"]["parallel_execution_enabled"] = False  # Disabled

    result = router_conditional_edge(state)

    # Should return single capability name (sequential mode)
    assert isinstance(result, str)
    assert result == "pv_reader"


@patch("osprey.infrastructure.router_node.get_registry")
def test_parallel_execution_with_partial_failures(mock_get_registry, mock_registry):
    """Test parallel execution when some steps fail."""
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
                capability="calculator",
                task_objective="Calculate average",
                success_criteria="Average calculated",
                inputs=[{"PV_DATA": "step_0"}, {"PV_DATA": "step_1"}],
            ),
        ]
    )

    state = StateManager.create_fresh_state("Get PV1, PV2, calculate")
    state["task_current_task"] = "Get PV1, PV2, calculate"
    state["planning_active_capabilities"] = ["pv_reader", "calculator"]
    state["planning_execution_plan"] = plan
    state["agent_control"]["parallel_execution_enabled"] = True

    # Step 0 succeeded, step 1 failed
    state["execution_step_results"] = {
        "step_0": {"step_index": 0, "success": True},
        "step_1": {"step_index": 1, "success": False},  # Failed
    }

    result = router_conditional_edge(state)

    # Should still try to execute step 2 (calculator)
    # The calculator will need to handle missing data from step 1
    assert result == "calculator"


def test_parallel_execution_preserves_step_ordering():
    """Test that parallel execution preserves step ordering in results."""
    from osprey.state import merge_execution_step_results

    # Execute steps out of order
    results = {}

    # Step 2 completes first
    results = merge_execution_step_results(results, {"step_2": {"step_index": 2, "success": True}})

    # Step 0 completes second
    results = merge_execution_step_results(results, {"step_0": {"step_index": 0, "success": True}})

    # Step 1 completes last
    results = merge_execution_step_results(results, {"step_1": {"step_index": 1, "success": True}})

    # All results should be present
    assert len(results) == 3

    # Step indices should be preserved
    assert results["step_0"]["step_index"] == 0
    assert results["step_1"]["step_index"] == 1
    assert results["step_2"]["step_index"] == 2
