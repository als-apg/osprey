"""Tests for decorator parallel execution behavior.

This module tests the parallel execution specific logic in the capability_node
decorator, ensuring that state updates and control flow differ correctly between
parallel and sequential execution modes.
"""

from unittest.mock import patch

import pytest

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.state import StateManager


@capability_node
class MockParallelCapability(BaseCapability):
    """Mock capability for testing parallel execution."""

    name = "mock_parallel"
    description = "Mock capability for parallel testing"

    async def execute(self):
        """Execute mock capability."""
        return {
            "capability_context_data": {
                "TEST_DATA": {self._step.get("context_key", "test"): {"result": "test_data"}}
            }
        }


@pytest.fixture
def mock_langgraph_context():
    """Mock LangGraph context to avoid 'Called get_config outside of a runnable context' errors."""
    with (
        patch("osprey.base.decorators.get_stream_writer", return_value=None),
        patch("osprey.base.decorators.get_config", return_value={}),
    ):
        yield


@pytest.fixture
def base_state():
    """Create base state for testing."""
    state = StateManager.create_fresh_state("Test query")
    state["task_current_task"] = "Test task"
    state["planning_active_capabilities"] = ["mock_parallel"]
    state["planning_execution_plan"] = {
        "steps": [
            {
                "context_key": "step_0",
                "capability": "mock_parallel",
                "task_objective": "Test step 0",
                "success_criteria": "Success",
                "inputs": [],
            },
            {
                "context_key": "step_1",
                "capability": "mock_parallel",
                "task_objective": "Test step 1",
                "success_criteria": "Success",
                "inputs": [],
            },
        ]
    }
    state["planning_current_step_index"] = 0
    state["execution_step_results"] = {}
    return state


@pytest.mark.asyncio
async def test_sequential_mode_increments_step_index(mock_langgraph_context, base_state):
    """Test that sequential mode increments planning_current_step_index."""
    # Sequential mode (parallel_execution_enabled = False)
    base_state["agent_control"]["parallel_execution_enabled"] = False

    # Execute capability
    result = await MockParallelCapability.langgraph_node(base_state)

    # Should increment step index in sequential mode
    assert result["planning_current_step_index"] == 1


@pytest.mark.asyncio
async def test_parallel_mode_does_not_increment_step_index(mock_langgraph_context, base_state):
    """Test that parallel mode does NOT increment planning_current_step_index."""
    # Parallel mode (parallel_execution_enabled = True)
    base_state["agent_control"]["parallel_execution_enabled"] = True

    # Execute capability
    result = await MockParallelCapability.langgraph_node(base_state)

    # Should NOT increment step index in parallel mode
    assert "planning_current_step_index" not in result


@pytest.mark.asyncio
async def test_sequential_mode_updates_control_flow_fields(mock_langgraph_context, base_state):
    """Test that sequential mode updates control flow fields."""
    base_state["agent_control"]["parallel_execution_enabled"] = False

    result = await MockParallelCapability.langgraph_node(base_state)

    # Should update control flow fields in sequential mode
    assert result["control_current_step_retry_count"] == 0
    assert result["control_has_error"] is False
    assert result["control_retry_count"] == 0
    assert result["control_error_info"] is None


@pytest.mark.asyncio
async def test_parallel_mode_skips_control_flow_updates(mock_langgraph_context, base_state):
    """Test that parallel mode skips control flow field updates."""
    base_state["agent_control"]["parallel_execution_enabled"] = True

    result = await MockParallelCapability.langgraph_node(base_state)

    # Should NOT update control flow fields in parallel mode
    assert "control_current_step_retry_count" not in result
    assert "control_has_error" not in result
    assert "control_retry_count" not in result
    assert "control_error_info" not in result


@pytest.mark.asyncio
async def test_sequential_mode_updates_execution_last_result(mock_langgraph_context, base_state):
    """Test that sequential mode updates execution_last_result."""
    base_state["agent_control"]["parallel_execution_enabled"] = False

    result = await MockParallelCapability.langgraph_node(base_state)

    # Should update execution_last_result in sequential mode
    assert "execution_last_result" in result
    assert result["execution_last_result"]["capability"] == "mock_parallel"
    assert result["execution_last_result"]["success"] is True


@pytest.mark.asyncio
async def test_parallel_mode_skips_execution_last_result(mock_langgraph_context, base_state):
    """Test that parallel mode skips execution_last_result update."""
    base_state["agent_control"]["parallel_execution_enabled"] = True

    result = await MockParallelCapability.langgraph_node(base_state)

    # Should NOT update execution_last_result in parallel mode
    assert "execution_last_result" not in result


@pytest.mark.asyncio
async def test_both_modes_update_execution_step_results(mock_langgraph_context, base_state):
    """Test that both modes update execution_step_results."""
    # Test sequential mode
    base_state["agent_control"]["parallel_execution_enabled"] = False
    result_seq = await MockParallelCapability.langgraph_node(base_state)

    assert "execution_step_results" in result_seq
    assert "step_0" in result_seq["execution_step_results"]
    assert result_seq["execution_step_results"]["step_0"]["success"] is True

    # Test parallel mode
    base_state["agent_control"]["parallel_execution_enabled"] = True
    result_par = await MockParallelCapability.langgraph_node(base_state)

    assert "execution_step_results" in result_par
    assert "step_0" in result_par["execution_step_results"]
    assert result_par["execution_step_results"]["step_0"]["success"] is True


@pytest.mark.asyncio
async def test_parallel_mode_uses_correct_step_index(mock_langgraph_context, base_state):
    """Test that parallel mode uses the step index from state correctly."""
    base_state["agent_control"]["parallel_execution_enabled"] = True

    # Simulate parallel execution of step 1 (not step 0)
    base_state["planning_current_step_index"] = 1

    result = await MockParallelCapability.langgraph_node(base_state)

    # Should use step 1's information
    assert "step_1" in result["execution_step_results"]
    assert result["execution_step_results"]["step_1"]["step_index"] == 1
    assert result["execution_step_results"]["step_1"]["task_objective"] == "Test step 1"


@pytest.mark.asyncio
async def test_parallel_mode_logging(mock_langgraph_context, base_state, caplog):
    """Test that parallel mode logs execution with step index and key."""
    base_state["agent_control"]["parallel_execution_enabled"] = True

    with caplog.at_level("INFO"):
        await MockParallelCapability.langgraph_node(base_state)

    # Should log with step index and context key
    log_messages = [record.message for record in caplog.records]
    assert any("step 0" in msg and "key=step_0" in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_sequential_mode_logging(mock_langgraph_context, base_state, caplog):
    """Test that sequential mode logs without step details."""
    base_state["agent_control"]["parallel_execution_enabled"] = False

    with caplog.at_level("INFO"):
        await MockParallelCapability.langgraph_node(base_state)

    # Should log simple execution message
    log_messages = [record.message for record in caplog.records]
    assert any("Executing capability: mock_parallel" in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_step_result_storage_with_correct_key(mock_langgraph_context, base_state):
    """Test that step results are stored with correct context_key."""
    base_state["agent_control"]["parallel_execution_enabled"] = True

    # Execute step 0
    base_state["planning_current_step_index"] = 0
    result = await MockParallelCapability.langgraph_node(base_state)

    # Should use context_key from step
    assert "step_0" in result["execution_step_results"]
    assert result["execution_step_results"]["step_0"]["capability"] == "mock_parallel"


@pytest.mark.asyncio
async def test_parallel_execution_preserves_capability_context(mock_langgraph_context, base_state):
    """Test that capability context data is preserved in both modes."""
    # Test sequential mode
    base_state["agent_control"]["parallel_execution_enabled"] = False
    result_seq = await MockParallelCapability.langgraph_node(base_state)

    assert "capability_context_data" in result_seq

    # Test parallel mode
    base_state["agent_control"]["parallel_execution_enabled"] = True
    result_par = await MockParallelCapability.langgraph_node(base_state)

    assert "capability_context_data" in result_par


@pytest.mark.asyncio
async def test_error_handling_in_parallel_mode(mock_langgraph_context, base_state):
    """Test that errors are handled correctly in parallel mode."""

    @capability_node
    class FailingCapability(BaseCapability):
        name = "failing_capability"
        description = "Capability that fails"

        async def execute(self):
            raise ValueError("Test error")

    base_state["agent_control"]["parallel_execution_enabled"] = True

    result = await FailingCapability.langgraph_node(base_state)

    # Should set error state
    assert result["control_has_error"] is True
    assert "control_error_info" in result

    # Should still record step result
    assert "execution_step_results" in result
    step_key = list(result["execution_step_results"].keys())[0]
    assert result["execution_step_results"][step_key]["success"] is False


@pytest.mark.asyncio
async def test_multiple_parallel_steps_accumulate_results(mock_langgraph_context, base_state):
    """Test that multiple parallel steps accumulate in execution_step_results."""
    base_state["agent_control"]["parallel_execution_enabled"] = True

    # Execute step 0
    base_state["planning_current_step_index"] = 0
    result_0 = await MockParallelCapability.langgraph_node(base_state)

    # Simulate state merge
    base_state["execution_step_results"].update(result_0["execution_step_results"])

    # Execute step 1
    base_state["planning_current_step_index"] = 1
    result_1 = await MockParallelCapability.langgraph_node(base_state)

    # Both results should be present
    assert "step_0" in result_0["execution_step_results"]
    assert "step_1" in result_1["execution_step_results"]
