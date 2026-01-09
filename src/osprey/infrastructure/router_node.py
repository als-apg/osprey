"""
Osprey Agentic Framework - Dynamic Router for LangGraph

This module contains the router node and conditional edge function for routing decisions.
The router is the central decision-making authority that determines what happens next.

Architecture:
- RouterNode: Minimal node that handles routing metadata and decisions
- router_conditional_edge: Pure conditional edge function for actual routing
- All business logic nodes route back to router for next decisions
- Supports parallel execution of independent steps via LangGraph Send API
"""

from __future__ import annotations

import time
from typing import Any

from osprey.base.decorators import infrastructure_node
from osprey.base.errors import ErrorSeverity
from osprey.base.nodes import BaseInfrastructureNode
from osprey.base.planning import PlannedStep
from osprey.registry import get_registry

# Fixed import to use new TypedDict state
from osprey.state import AgentState, StateManager
from osprey.utils.config import get_execution_limits
from osprey.utils.logger import get_logger

# Import Send for parallel execution
try:
    from langgraph.types import Send
except ImportError:
    Send = None


@infrastructure_node(quiet=True)
class RouterNode(BaseInfrastructureNode):
    """Central routing decision node for the Osprey Agent Framework.

    This node serves as the single decision-making authority that determines
    what should happen next based on the current agent state. It does no business
    logic - only routing decisions and metadata management.

    The actual routing is handled by the router_conditional_edge function.
    """

    name = "router"
    description = "Central routing decision authority"

    async def execute(self) -> dict[str, Any]:
        """Router node execution - updates routing metadata only.

        This node serves as the entry point and routing hub, but does no routing logic itself.
        The actual routing decision is made by the conditional edge function.
        This keeps the logic DRY and avoids duplication.

        :return: Dictionary of state updates for routing metadata
        :rtype: Dict[str, Any]
        """
        state = self._state

        # Update routing metadata only - no routing logic to avoid duplication
        return {
            "control_routing_timestamp": time.time(),
            "control_routing_count": state.get("control_routing_count", 0) + 1,
        }


def _get_next_executable_batch(steps: list[PlannedStep], completed_indices: set[int]) -> list[int]:
    """
    Determine which steps can be executed in parallel based on dependencies.

    This function analyzes the execution plan to find all steps that:
    1. Haven't been completed yet
    2. Have all their dependencies satisfied

    Args:
        steps: List of planned steps from the execution plan
        completed_indices: Set of step indices that have been completed

    Returns:
        List of step indices that can be executed in parallel

    Example:
        >>> steps = [
        ...     PlannedStep(context_key="step_0", capability="pv_reader", inputs=[]),
        ...     PlannedStep(context_key="step_1", capability="pv_reader", inputs=[]),
        ...     PlannedStep(context_key="step_2", capability="calculator",
        ...                 inputs=[{"PV_DATA": "step_0"}, {"PV_DATA": "step_1"}])
        ... ]
        >>> _get_next_executable_batch(steps, set())
        [0, 1]  # Steps 0 and 1 can run in parallel
        >>> _get_next_executable_batch(steps, {0, 1})
        [2]  # Step 2 can run after 0 and 1 complete
    """
    executable = []

    for idx, step in enumerate(steps):
        # Skip if already completed
        if idx in completed_indices:
            continue

        # Check if all dependencies are satisfied
        dependencies_satisfied = True
        inputs = step.get("inputs", [])

        for input_spec in inputs:
            # input_spec is a dict like {"PV_DATA": "step_0"}
            # Extract the context_key from the value
            for context_key in input_spec.values():
                # Find the step index that produces this context_key
                dep_idx = None
                for dep_step_idx, dep_step in enumerate(steps):
                    if dep_step.get("context_key") == context_key:
                        dep_idx = dep_step_idx
                        break

                # If dependency not completed, this step can't execute yet
                if dep_idx is not None and dep_idx not in completed_indices:
                    dependencies_satisfied = False
                    break

            if not dependencies_satisfied:
                break

        if dependencies_satisfied:
            executable.append(idx)

    return executable


def router_conditional_edge(state: AgentState) -> str | list[Send]:
    """LangGraph conditional edge function for dynamic routing.

    This is the main export of this module - a pure conditional edge function
    that determines which node should execute next based on agent state.

    Follows LangGraph native patterns where conditional edge functions take only
    the state parameter and handle logging internally.



    Manual retry handling:
    - Checks for errors and retry count first
    - Routes retriable errors back to same capability if retries available
    - Routes to error node when retries exhausted
    - Routes critical/replanning errors immediately

    :param state: Current agent state containing all execution context
    :type state: AgentState
    :return: Name of next node to execute or "END" to terminate
    :rtype: str
    """
    # Get logger internally - LangGraph native pattern
    logger = get_logger("router")

    # Get registry for node lookup
    registry = get_registry()

    # ==== MANUAL RETRY HANDLING - Check first before normal routing ====
    if state.get("control_has_error", False):
        error_info = state.get("control_error_info", {})
        error_classification = error_info.get("classification")
        capability_name = error_info.get("capability_name") or error_info.get("node_name")
        retry_policy = error_info.get("retry_policy", {})

        if error_classification and capability_name:
            retry_count = state.get("control_retry_count", 0)

            # Use node-specific retry policy, with fallback defaults
            max_retries = retry_policy.get("max_attempts", 3)
            delay_seconds = retry_policy.get("delay_seconds", 0.5)
            backoff_factor = retry_policy.get("backoff_factor", 1.5)

            if error_classification.severity == ErrorSeverity.RETRIABLE:
                if retry_count < max_retries:
                    # Calculate delay with backoff for this retry attempt
                    actual_delay = (
                        delay_seconds * (backoff_factor ** (retry_count - 1))
                        if retry_count > 0
                        else 0
                    )

                    # Apply delay if this is a retry (not the first attempt)
                    if retry_count > 0 and actual_delay > 0:
                        logger.error(
                            f"Applying {actual_delay:.2f}s delay before retry {retry_count + 1}"
                        )
                        time.sleep(actual_delay)  # Simple sleep for now, could be async

                    # CRITICAL FIX: Increment retry count in state before routing back
                    new_retry_count = retry_count + 1
                    state["control_retry_count"] = new_retry_count

                    # Retry available - route back to same capability
                    logger.error(
                        f"Router: Retrying {capability_name} (attempt {new_retry_count}/{max_retries})"
                    )
                    return capability_name
                else:
                    # Retries exhausted - route to error node
                    logger.error(
                        f"Retries exhausted for {capability_name} ({retry_count}/{max_retries}), routing to error node"
                    )
                    return "error"

            elif error_classification.severity == ErrorSeverity.REPLANNING:
                # Check how many plans have been created by orchestrator
                current_plans_created = state.get("control_plans_created_count", 0)

                # Get max planning attempts from execution limits config
                limits = get_execution_limits()
                max_planning_attempts = limits.get("max_planning_attempts", 2)

                if current_plans_created < max_planning_attempts:
                    # Orchestrator will increment counter when it creates new plan
                    logger.error(
                        f"Router: Replanning error in {capability_name}, routing to orchestrator "
                        f"(plan #{current_plans_created + 1}/{max_planning_attempts})"
                    )
                    return "orchestrator"
                else:
                    # Planning attempts exhausted - route to error node
                    logger.error(
                        f"Router: Planning attempts exhausted for {capability_name} "
                        f"({current_plans_created}/{max_planning_attempts} plans created), routing to error node"
                    )
                    return "error"

            elif error_classification.severity == ErrorSeverity.RECLASSIFICATION:
                # Check how many reclassifications have been performed
                current_reclassifications = state.get("control_reclassification_count", 0)

                # Get max reclassification attempts from config
                limits = get_execution_limits()
                max_reclassifications = limits.get("max_reclassifications", 1)

                if current_reclassifications < max_reclassifications:
                    # Route to classifier for reclassification (state will be updated by classifier)
                    logger.error(
                        f"Router: Reclassification error in {capability_name}, routing to classifier "
                        f"(attempt #{current_reclassifications + 1}/{max_reclassifications})"
                    )
                    return "classifier"
                else:
                    # Reclassification attempts exhausted - route to error node
                    logger.error(
                        f"Router: Reclassification attempts exhausted for {capability_name} "
                        f"({current_reclassifications}/{max_reclassifications} attempts), routing to error node"
                    )
                    return "error"

            elif error_classification.severity == ErrorSeverity.CRITICAL:
                # Route to error node immediately
                logger.error(f"Critical error in {capability_name}, routing to error node")
                return "error"

        # Fallback for unknown error types - route to error node
        logger.warning("Unknown error type, routing to error node")
        return "error"

    # ==== NORMAL ROUTING LOGIC ====

    # Reset retry count when no error (clean state for next operation)
    if "control_retry_count" in state:
        state["control_retry_count"] = 0

    # Check if killed
    if state.get("control_is_killed", False):
        kill_reason = state.get("control_kill_reason", "Unknown reason")
        logger.key_info(f"Execution terminated: {kill_reason}")
        return "error"

    # Check if task extraction is needed first
    current_task = StateManager.get_current_task(state)
    if not current_task:
        logger.key_info("No current task extracted, routing to task extraction")
        return "task_extraction"

    # Check if has active capabilities from prefixed state structure
    active_capabilities = state.get("planning_active_capabilities")
    if not active_capabilities:
        logger.key_info("No active capabilities, routing to classifier")
        return "classifier"

    # Check if has execution plan using StateManager utility
    execution_plan = StateManager.get_execution_plan(state)
    if not execution_plan:
        logger.key_info("No execution plan, routing to orchestrator")
        return "orchestrator"

    # Check if more steps to execute using StateManager utility
    current_index = StateManager.get_current_step_index(state)

    # Type validation already done by StateManager.get_execution_plan()
    plan_steps = execution_plan.get("steps", [])
    if current_index >= len(plan_steps):
        # This should NEVER happen - orchestrator guarantees plans end with respond/clarify
        # If it does happen, it indicates a serious bug in the orchestrator validation
        raise RuntimeError(
            f"CRITICAL BUG: current_step_index {current_index} >= plan_steps length {len(plan_steps)}. "
            f"Orchestrator validation failed - all execution plans must end with respond/clarify steps. "
            f"This indicates a bug in _validate_and_fix_execution_plan()."
        )

    # Check if parallel execution is enabled
    agent_control = state.get("agent_control", {})
    parallel_enabled = agent_control.get("parallel_execution_enabled", False)

    if parallel_enabled and Send is not None:
        # Determine which steps have been completed
        step_results = state.get("execution_step_results", {})
        completed_indices = set()

        for _step_key, result in step_results.items():
            if result.get("success", False):
                step_idx = result.get("step_index")
                if step_idx is not None:
                    completed_indices.add(step_idx)

        # Get next executable batch
        executable_indices = _get_next_executable_batch(plan_steps, completed_indices)

        if len(executable_indices) == 0:
            # Check if all steps are actually completed
            if len(completed_indices) >= len(plan_steps):
                # All steps completed - execution should end
                logger.key_info(
                    f"✅ All {len(plan_steps)} steps completed in parallel execution mode"
                )
                return "END"
            else:
                # No executable steps but plan not complete - possible dependency deadlock
                logger.error(
                    f"⚠️ Parallel execution deadlock: {len(completed_indices)}/{len(plan_steps)} "
                    f"steps completed but no executable steps found. Check for circular dependencies."
                )
                return "error"
        elif len(executable_indices) > 1:
            # Multiple steps can execute in parallel
            # BUT: Never include respond/clarify in parallel batch - they must wait for all data
            parallel_indices = []

            for idx in executable_indices:
                step = plan_steps[idx]
                step_capability = step.get("capability", "respond")

                # Check if this is a final response step
                if step_capability.lower() in ["respond", "clarify"]:
                    # Skip final response steps - they must wait for all data
                    pass
                else:
                    parallel_indices.append(idx)

            # If we have multiple non-response steps, execute them in parallel
            if len(parallel_indices) > 1:
                logger.key_info(
                    f"🚀 Parallel execution: {len(parallel_indices)} steps will run concurrently"
                )

                # Create Send commands for each parallel step
                send_commands = []
                for idx in parallel_indices:
                    step = plan_steps[idx]
                    step_capability = step.get("capability", "respond")

                    # Validate capability exists
                    if not registry.get_node(step_capability):
                        logger.error(f"Capability '{step_capability}' not registered - skipping")
                        continue

                    # Create Send command WITH step index in state
                    parallel_state = {**state, "planning_current_step_index": idx}
                    send_commands.append(Send(step_capability, parallel_state))
                    logger.key_info(f"  → Dispatching step {idx + 1}/{len(plan_steps)}: {step_capability}")

                if send_commands:
                    logger.key_info(
                        f"✨ Dispatched {len(send_commands)} parallel tasks to LangGraph execution engine"
                    )
                    return send_commands
                # Fall through to sequential if no valid commands
            # If only 1 parallel step or only final step, use sequential execution below
        else:
            # Exactly 1 executable step - use Send() to pass correct step index
            # This handles the final step (respond/clarify) after parallel batch
            current_index = executable_indices[0]
            current_step = plan_steps[current_index]
            step_capability = current_step.get("capability", "respond")

            logger.key_info(
                f"Executing step {current_index + 1}/{len(plan_steps)} - capability: {step_capability}"
            )

            # Validate capability exists
            if not registry.get_node(step_capability):
                logger.error(f"Capability '{step_capability}' not registered - skipping")
                return "error"

            # Use Send() to pass the correct step index in state
            # This ensures the capability gets step 2's information, not step 0's
            sequential_state = {**state, "planning_current_step_index": current_index}
            return [Send(step_capability, sequential_state)]

    # Sequential execution (default or fallback)
    current_step = plan_steps[current_index]

    # PlannedStep is a TypedDict, so access it as a dictionary
    step_capability = current_step.get("capability", "respond")

    logger.key_info(
        f"Executing step {current_index + 1}/{len(plan_steps)} - capability: {step_capability}"
    )

    # Validate that the capability exists as a registered node
    if not registry.get_node(step_capability):
        logger.error(
            f"Capability '{step_capability}' not registered - orchestrator may have hallucinated non-existent capability"
        )
        return "error"

    # Return the capability name - this must match the node name in LangGraph
    return step_capability
