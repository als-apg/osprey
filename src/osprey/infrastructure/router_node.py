"""
Osprey Agentic Framework - Dynamic Router for LangGraph

This module contains the router node and conditional edge function for routing decisions.
The router is the central decision-making authority that determines what happens next.

Architecture:
- RouterNode: Minimal node that handles routing metadata and decisions
- router_conditional_edge: Pure conditional edge function for actual routing
- All business logic nodes route back to router for next decisions
"""

from __future__ import annotations

import time
from typing import Any

from osprey.base.decorators import infrastructure_node
from osprey.base.errors import ErrorSeverity
from osprey.base.nodes import BaseInfrastructureNode
from osprey.registry import get_registry

# Fixed import to use new TypedDict state
from osprey.state import AgentState, StateManager
from osprey.utils.config import get_execution_limits
from osprey.utils.logger import get_logger


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


def router_conditional_edge(state: AgentState) -> str:
    """LangGraph conditional edge function for dynamic routing.

    This is the main export of this module - a pure conditional edge function
    that determines which node should execute next based on agent state.

    Follows LangGraph native patterns where conditional edge functions take only
    the state parameter and handle logging internally.

    Routing priority:
    1. Direct chat mode - routes directly to capability, bypassing pipeline
    2. Manual retry handling - checks errors and retry count
    3. Normal routing - task extraction → classification → orchestration → execution

    :param state: Current agent state containing all execution context
    :type state: AgentState
    :return: Name of next node to execute or "END" to terminate
    :rtype: str
    """
    # Get logger internally - LangGraph native pattern
    logger = get_logger("router")

    # Get registry for node lookup
    registry = get_registry()

    # Check if this is an active execution (vs state-only evaluation)
    # State-only updates (mode switches) set execution_start_time to None
    is_active_execution = state.get("execution_start_time") is not None

    # ==== HIGHEST PRIORITY: DIRECT CHAT MODE ====
    session_state = state.get("session_state", {})
    direct_chat_capability = session_state.get("direct_chat_capability")

    if direct_chat_capability:
        # Check if capability already executed this turn (prevents infinite loop)
        # Use `or {}` because gateway explicitly sets to None for new turns
        last_result = state.get("execution_last_result") or {}
        if last_result.get("capability") == direct_chat_capability:
            # Direct chat turn complete - end execution
            logger.status(f"🎯 Direct chat turn complete for {direct_chat_capability}")
            return "END"

        # Validate capability exists and supports direct chat
        cap_instance = registry.get_capability(direct_chat_capability)
        if cap_instance is None:
            logger.error(f"Direct chat capability '{direct_chat_capability}' not found in registry")
            # Clear invalid state and fall through to normal routing
            state["session_state"] = {
                **session_state,
                "direct_chat_capability": None,
                "last_direct_chat_result": None,
            }
        else:
            if not getattr(cap_instance, "direct_chat_enabled", False):
                logger.error(
                    f"Capability '{direct_chat_capability}' doesn't support direct chat mode"
                )
                # Clear invalid state and fall through to normal routing
                state["session_state"] = {
                    **session_state,
                    "direct_chat_capability": None,
                    "last_direct_chat_result": None,
                }
            else:
                # Valid direct chat mode - route directly to capability
                logger.status(f"🎯 Direct chat mode: routing to {direct_chat_capability}")
                return direct_chat_capability

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
        logger.status(f"Execution terminated: {kill_reason}")
        return "error"

    # Check if task extraction is needed first
    current_task = StateManager.get_current_task(state)
    if not current_task:
        if is_active_execution:
            logger.status("No current task extracted, routing to task extraction")
        return "task_extraction"

    # Check if has active capabilities from prefixed state structure
    active_capabilities = state.get("planning_active_capabilities")
    if not active_capabilities:
        if is_active_execution:
            logger.status("No active capabilities, routing to classifier")
        return "classifier"

    # Check if has execution plan using StateManager utility
    execution_plan = StateManager.get_execution_plan(state)
    if not execution_plan:
        if is_active_execution:
            logger.status("No execution plan, routing to orchestrator")
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

    # Execute next step
    current_step = plan_steps[current_index]

    # PlannedStep is a TypedDict, so access it as a dictionary
    step_capability = current_step.get("capability", "respond")

    if is_active_execution:
        logger.status(
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
