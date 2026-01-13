"""Approval Node for XOpt Optimizer Service.

This node handles human approval for XOpt configurations using the standard
Osprey LangGraph interrupt pattern. The approval interrupt data is pre-created
by the yaml_generation node, following the pattern from Python executor's
analyzer node.
"""

from typing import Any

from langgraph.types import interrupt

from osprey.utils.logger import get_logger

from ..models import XOptExecutionState

logger = get_logger("xopt_optimizer")


def create_approval_node():
    """Create a pure approval node function for LangGraph integration.

    This factory function creates a specialized approval node that serves as a
    clean interrupt handler. The node is designed with single responsibility:
    processing LangGraph interrupts for user approval.

    The approval interrupt data is pre-created by the yaml_generation node,
    following the pattern from Python executor's analyzer node.

    Returns:
        Async function that takes XOptExecutionState and returns state updates
    """

    async def approval_node(state: XOptExecutionState) -> dict[str, Any]:
        """Process approval interrupt and return user response for workflow routing."""

        # Get logger with streaming support
        node_logger = get_logger("xopt_optimizer", state=state)
        node_logger.status("Requesting human approval...")

        # Get the pre-created interrupt data from yaml_generation node
        interrupt_data = state.get("approval_interrupt_data")
        if not interrupt_data:
            raise RuntimeError(
                "No approval interrupt data found in state. "
                "The yaml_generation node should create this data."
            )

        node_logger.info("Requesting human approval for XOpt configuration")

        # This is the ONLY critical line - everything else is routing
        human_response = interrupt(interrupt_data)

        # Simple approval processing for routing
        approved = human_response.get("approved", False)
        node_logger.info(f"Approval result: {approved}")

        return {
            "approval_result": human_response,
            "approved": approved,
        }

    return approval_node
