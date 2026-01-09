"""Performance and stress tests for parallel execution.

This module contains performance benchmarks, stress tests, and concurrency tests
for the parallel execution system to ensure it scales properly and maintains
state consistency under load.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.infrastructure.router_node import _get_next_executable_batch, router_conditional_edge
from osprey.state import StateManager

try:
    from langgraph.types import Send
except ImportError:
    Send = None


# ===== PERFORMANCE BENCHMARKS =====


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
@patch("osprey.infrastructure.router_node.get_registry")
def test_parallel_execution_speedup(mock_get_registry):
    """Measure speedup from parallel execution vs sequential.

    This test compares the time to find executable batches in parallel mode
    versus sequential mode to verify that parallel execution provides
    performance benefits.
    """
    # Mock registry
    registry = Mock()
    registry.get_node = Mock(return_value=True)
    mock_get_registry.return_value = registry

    # Create plan with 10 independent steps
    num_steps = 10
    steps = [
        PlannedStep(
            context_key=f"step_{i}",
            capability="pv_reader",
            task_objective=f"Get PV{i}",
            success_criteria=f"PV{i} retrieved",
            inputs=[],
        )
        for i in range(num_steps)
    ]

    plan = ExecutionPlan(steps=steps)

    # Measure parallel execution time
    state_parallel = StateManager.create_fresh_state("Get all PVs")
    state_parallel["task_current_task"] = "Get all PVs"
    state_parallel["planning_active_capabilities"] = ["pv_reader"]
    state_parallel["planning_execution_plan"] = plan
    state_parallel["execution_step_results"] = {}
    state_parallel["agent_control"]["parallel_execution_enabled"] = True

    start_parallel = time.perf_counter()
    result_parallel = router_conditional_edge(state_parallel)
    time_parallel = time.perf_counter() - start_parallel

    # Measure sequential execution time (simulate by processing one at a time)
    state_sequential = StateManager.create_fresh_state("Get all PVs")
    state_sequential["task_current_task"] = "Get all PVs"
    state_sequential["planning_active_capabilities"] = ["pv_reader"]
    state_sequential["planning_execution_plan"] = plan
    state_sequential["execution_step_results"] = {}
    state_sequential["agent_control"]["parallel_execution_enabled"] = False

    start_sequential = time.perf_counter()
    for i in range(num_steps):
        state_sequential["planning_current_step_index"] = i
        router_conditional_edge(state_sequential)
    time_sequential = time.perf_counter() - start_sequential

    # Verify parallel returns all steps at once
    assert isinstance(result_parallel, list)
    assert len(result_parallel) == num_steps

    # Parallel should be faster (or at least not significantly slower)
    # Note: In practice, parallel execution overhead might make small batches slower,
    # but the routing logic itself should be comparable or faster
    print(f"\nParallel routing time: {time_parallel * 1000:.2f}ms")
    print(f"Sequential routing time: {time_sequential * 1000:.2f}ms")
    print(f"Speedup factor: {time_sequential / time_parallel:.2f}x")

    # Assert that parallel execution doesn't add significant overhead
    # (allow up to 2x overhead for small batches due to Send() object creation)
    assert time_parallel < time_sequential * 2.0, (
        f"Parallel execution too slow: {time_parallel:.4f}s vs {time_sequential:.4f}s"
    )


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
def test_dependency_analysis_performance():
    """Test performance of dependency analysis with complex plans.

    Verifies that _get_next_executable_batch() scales well with
    increasing plan complexity.
    """
    # Create plan with 50 steps in a diamond pattern:
    # - 10 independent steps at start
    # - 30 steps that depend on various combinations
    # - 10 final aggregation steps

    steps = []

    # Layer 1: 10 independent steps
    for i in range(10):
        steps.append(
            PlannedStep(
                context_key=f"step_{i}",
                capability="pv_reader",
                task_objective=f"Get PV{i}",
                success_criteria=f"PV{i} retrieved",
                inputs=[],
            )
        )

    # Layer 2: 30 steps with dependencies on layer 1
    for i in range(10, 40):
        # Each step depends on 2-3 steps from layer 1
        dep_indices = [(i - 10) % 10, (i - 9) % 10]
        inputs = [{"PV_DATA": f"step_{idx}"} for idx in dep_indices]

        steps.append(
            PlannedStep(
                context_key=f"step_{i}",
                capability="calculator",
                task_objective=f"Calculate {i}",
                success_criteria=f"Calculation {i} complete",
                inputs=inputs,
            )
        )

    # Layer 3: 10 final aggregation steps
    for i in range(40, 50):
        # Each depends on 3 steps from layer 2
        dep_indices = [10 + (i - 40) * 3 + j for j in range(3)]
        inputs = [{"CALC_DATA": f"step_{idx}"} for idx in dep_indices]

        steps.append(
            PlannedStep(
                context_key=f"step_{i}",
                capability="aggregator",
                task_objective=f"Aggregate {i}",
                success_criteria=f"Aggregation {i} complete",
                inputs=inputs,
            )
        )

    # Measure time to find executable batches
    start = time.perf_counter()

    # Layer 1: All 10 should be executable
    batch1 = _get_next_executable_batch(steps, set())
    assert len(batch1) == 10

    # Layer 2: After layer 1 completes, all 30 should be executable
    batch2 = _get_next_executable_batch(steps, set(range(10)))
    assert len(batch2) == 30

    # Layer 3: After layers 1 & 2, all 10 should be executable
    batch3 = _get_next_executable_batch(steps, set(range(40)))
    assert len(batch3) == 10

    elapsed = time.perf_counter() - start

    print(f"\nDependency analysis for 50-step plan: {elapsed * 1000:.2f}ms")

    # Should complete in reasonable time (< 100ms for 50 steps)
    assert elapsed < 0.1, f"Dependency analysis too slow: {elapsed:.4f}s"


# ===== STRESS TESTS =====


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
@patch("osprey.infrastructure.router_node.get_registry")
def test_large_parallel_batch(mock_get_registry):
    """Test with 50+ independent steps to verify no performance degradation.

    This stress test ensures the parallel execution system can handle
    large batches without memory issues or significant slowdown.
    """
    # Mock registry
    registry = Mock()
    registry.get_node = Mock(return_value=True)
    mock_get_registry.return_value = registry

    # Create plan with 100 independent steps
    num_steps = 100
    steps = [
        PlannedStep(
            context_key=f"step_{i}",
            capability="pv_reader",
            task_objective=f"Get PV{i}",
            success_criteria=f"PV{i} retrieved",
            inputs=[],
        )
        for i in range(num_steps)
    ]

    plan = ExecutionPlan(steps=steps)

    state = StateManager.create_fresh_state("Get 100 PVs")
    state["task_current_task"] = "Get 100 PVs"
    state["planning_active_capabilities"] = ["pv_reader"]
    state["planning_execution_plan"] = plan
    state["execution_step_results"] = {}
    state["agent_control"]["parallel_execution_enabled"] = True

    # Measure execution time
    start = time.perf_counter()
    result = router_conditional_edge(state)
    elapsed = time.perf_counter() - start

    # Verify all steps dispatched
    assert isinstance(result, list)
    assert len(result) == num_steps

    # Verify all are Send commands
    for send_cmd in result:
        assert isinstance(send_cmd, Send)
        assert send_cmd.node == "pv_reader"

    print(f"\nRouting 100 parallel steps: {elapsed * 1000:.2f}ms")

    # Should complete in reasonable time (< 500ms for 100 steps)
    assert elapsed < 0.5, f"Large batch routing too slow: {elapsed:.4f}s"


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
def test_deep_dependency_chain():
    """Test with deep sequential dependency chain (50 steps).

    Verifies that dependency analysis handles deep chains efficiently
    without stack overflow or excessive recursion.
    """
    # Create 50-step sequential chain
    steps = []
    for i in range(50):
        if i == 0:
            inputs = []
        else:
            inputs = [{"DATA": f"step_{i - 1}"}]

        steps.append(
            PlannedStep(
                context_key=f"step_{i}",
                capability="processor",
                task_objective=f"Process step {i}",
                success_criteria=f"Step {i} complete",
                inputs=inputs,
            )
        )

    # Verify each step becomes executable one at a time
    completed = set()

    start = time.perf_counter()
    for i in range(50):
        executable = _get_next_executable_batch(steps, completed)
        assert executable == [i], f"Expected step {i}, got {executable}"
        completed.add(i)

    elapsed = time.perf_counter() - start

    print(f"\nProcessing 50-step dependency chain: {elapsed * 1000:.2f}ms")

    # Should complete in reasonable time
    assert elapsed < 0.1, f"Deep chain analysis too slow: {elapsed:.4f}s"


@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
def test_wide_dependency_fan_out():
    """Test with wide fan-out pattern (1 step → 50 dependent steps).

    Verifies that the system handles wide dependency patterns efficiently.
    """
    steps = []

    # Single root step
    steps.append(
        PlannedStep(
            context_key="step_0",
            capability="data_loader",
            task_objective="Load data",
            success_criteria="Data loaded",
            inputs=[],
        )
    )

    # 50 steps that all depend on step 0
    for i in range(1, 51):
        steps.append(
            PlannedStep(
                context_key=f"step_{i}",
                capability="processor",
                task_objective=f"Process {i}",
                success_criteria=f"Processing {i} complete",
                inputs=[{"DATA": "step_0"}],
            )
        )

    # Initially, only step 0 should be executable
    batch1 = _get_next_executable_batch(steps, set())
    assert batch1 == [0]

    # After step 0, all 50 should be executable in parallel
    start = time.perf_counter()
    batch2 = _get_next_executable_batch(steps, {0})
    elapsed = time.perf_counter() - start

    assert len(batch2) == 50
    assert set(batch2) == set(range(1, 51))

    print(f"\nFan-out analysis (1→50): {elapsed * 1000:.2f}ms")

    # Should complete quickly
    assert elapsed < 0.05, f"Fan-out analysis too slow: {elapsed:.4f}s"


# ===== CONCURRENCY TESTS =====


@pytest.mark.asyncio
@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
async def test_concurrent_state_merging():
    """Test state merging under concurrent updates.

    Simulates multiple parallel steps completing simultaneously and
    verifies that execution_step_results merges correctly without
    data loss or corruption.
    """
    # Create initial state
    state = StateManager.create_fresh_state("Concurrent test")
    state["execution_step_results"] = {}

    # Simulate 10 parallel steps completing concurrently
    async def complete_step(step_idx: int, delay: float):
        """Simulate a step completing after a delay."""
        await asyncio.sleep(delay)
        return {
            f"step_{step_idx}": {
                "step_index": step_idx,
                "capability": "pv_reader",
                "task_objective": f"Get PV{step_idx}",
                "success": True,
                "execution_time": delay,
            }
        }

    # Run 10 steps concurrently with varying delays
    tasks = [
        complete_step(i, 0.01 * (i % 3))  # Varying delays to create race conditions
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)

    # Merge all results into state
    for result in results:
        state["execution_step_results"].update(result)

    # Verify all 10 steps are in results
    assert len(state["execution_step_results"]) == 10

    # Verify each step has correct data
    for i in range(10):
        step_key = f"step_{i}"
        assert step_key in state["execution_step_results"]
        assert state["execution_step_results"][step_key]["step_index"] == i
        assert state["execution_step_results"][step_key]["success"] is True


@pytest.mark.asyncio
@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
async def test_concurrent_batch_execution():
    """Test concurrent execution of multiple batches.

    Simulates the scenario where multiple batches of steps execute
    concurrently and verifies correct state progression.
    """
    # Create plan with 3 batches:
    # Batch 1: steps 0, 1, 2 (independent)
    # Batch 2: steps 3, 4 (depend on batch 1)
    # Batch 3: step 5 (depends on batch 2)

    steps = [
        # Batch 1
        PlannedStep(
            context_key="step_0",
            capability="pv_reader",
            task_objective="Get PV0",
            success_criteria="PV0 retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_1",
            capability="pv_reader",
            task_objective="Get PV1",
            success_criteria="PV1 retrieved",
            inputs=[],
        ),
        PlannedStep(
            context_key="step_2",
            capability="pv_reader",
            task_objective="Get PV2",
            success_criteria="PV2 retrieved",
            inputs=[],
        ),
        # Batch 2
        PlannedStep(
            context_key="step_3",
            capability="calculator",
            task_objective="Calc 1",
            success_criteria="Calc 1 complete",
            inputs=[{"DATA": "step_0"}, {"DATA": "step_1"}],
        ),
        PlannedStep(
            context_key="step_4",
            capability="calculator",
            task_objective="Calc 2",
            success_criteria="Calc 2 complete",
            inputs=[{"DATA": "step_1"}, {"DATA": "step_2"}],
        ),
        # Batch 3
        PlannedStep(
            context_key="step_5",
            capability="aggregator",
            task_objective="Aggregate",
            success_criteria="Aggregation complete",
            inputs=[{"DATA": "step_3"}, {"DATA": "step_4"}],
        ),
    ]

    # Batch 1: All 3 should be executable
    batch1 = _get_next_executable_batch(steps, set())
    assert set(batch1) == {0, 1, 2}

    # Simulate batch 1 completing concurrently
    completed = set()

    async def complete_batch(indices: list[int]):
        """Simulate batch completion."""
        await asyncio.sleep(0.01)  # Simulate work
        return set(indices)

    batch1_completed = await complete_batch(batch1)
    completed.update(batch1_completed)

    # Batch 2: Steps 3 and 4 should be executable
    batch2 = _get_next_executable_batch(steps, completed)
    assert set(batch2) == {3, 4}

    batch2_completed = await complete_batch(batch2)
    completed.update(batch2_completed)

    # Batch 3: Step 5 should be executable
    batch3 = _get_next_executable_batch(steps, completed)
    assert batch3 == [5]

    batch3_completed = await complete_batch(batch3)
    completed.update(batch3_completed)

    # All steps should be completed
    assert len(completed) == 6
    assert completed == {0, 1, 2, 3, 4, 5}


@pytest.mark.asyncio
@pytest.mark.skipif(Send is None, reason="LangGraph Send not available")
async def test_state_consistency_under_load():
    """Test state consistency with rapid concurrent updates.

    Simulates high-frequency state updates to verify no race conditions
    or data corruption occurs.
    """
    state = {"execution_step_results": {}}

    # Simulate 100 rapid concurrent updates
    async def rapid_update(step_idx: int):
        """Rapidly update state."""
        for i in range(10):
            await asyncio.sleep(0.001)  # Very short delay
            state["execution_step_results"][f"step_{step_idx}_{i}"] = {
                "step_index": step_idx * 10 + i,
                "success": True,
            }

    # Run 10 concurrent updaters
    tasks = [rapid_update(i) for i in range(10)]
    await asyncio.gather(*tasks)

    # Verify all 100 updates are present
    assert len(state["execution_step_results"]) == 100

    # Verify no data corruption
    for i in range(10):
        for j in range(10):
            key = f"step_{i}_{j}"
            assert key in state["execution_step_results"]
            assert state["execution_step_results"][key]["step_index"] == i * 10 + j
