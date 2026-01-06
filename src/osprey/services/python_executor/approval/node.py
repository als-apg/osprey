"""Approval Node for Python Executor Service - LangGraph Integration.

This module implements the approval workflow node within the Python executor service's
LangGraph-based architecture. It provides a clean, focused implementation that handles
user approval interrupts for Python code execution without duplicating approval logic.

The approval node follows a specialized design pattern where it serves as a pure
interrupt handler, delegating all approval decision logic to the analyzer node to
maintain clean separation of concerns and avoid execution duplication.

Key Design Principles:
    - **Single Responsibility**: Only handles LangGraph interrupt processing
    - **Clean Separation**: Approval logic resides in analyzer node, not here
    - **Interrupt-Focused**: Leverages LangGraph's native interrupt system
    - **Minimal State**: Processes pre-created interrupt data without modification
    - **Simple Routing**: Provides basic approval result processing for workflow routing

The node integrates seamlessly with the broader approval system while maintaining
the clean architecture principles of the Python executor service.

.. note::
   This node expects approval interrupt data to be pre-created by the analyzer node.
   It does not perform approval decision logic or create interrupt data.

.. warning::
   This node will raise a RuntimeError if called without proper interrupt data
   being set up by the analyzer node.

.. seealso::
   :class:`osprey.services.python_executor.analysis.node` : Creates approval interrupt data
   :class:`osprey.approval.ApprovalManager` : Framework-level approval management
   :func:`langgraph.types.interrupt` : LangGraph interrupt mechanism

Examples:
    The approval node is typically used within the service graph::

        >>> # This is handled automatically by the service
        >>> workflow.add_node("python_approval_node", create_approval_node())
        >>> workflow.add_conditional_edges(
        ...     "python_code_analyzer",
        ...     analyzer_conditional_edge,
        ...     {"approve": "python_approval_node", ...}
        ... )

    The node processes interrupt data created by the analyzer::

        >>> # Analyzer creates interrupt data
        >>> interrupt_data = create_code_approval_interrupt(
        ...     code=generated_code,
        ...     analysis_details=analysis_result,
        ...     safety_concerns=["File system access detected"]
        ... )
        >>> # Approval node processes the interrupt
        >>> human_response = interrupt(interrupt_data)  # LangGraph handles this
"""

from typing import Any

from langgraph.types import interrupt

from osprey.utils.logger import get_logger

from ..models import PythonExecutionState

logger = get_logger("python")


def create_approval_node():
    """Create a pure approval node function for LangGraph integration.

    This factory function creates a specialized approval node that serves as a
    clean interrupt handler within the Python executor service's workflow. The
    node is designed with a single responsibility: processing LangGraph interrupts
    for user approval without duplicating approval decision logic.

    The created node follows a minimalist design pattern where all approval
    logic and interrupt data creation is handled by the analyzer node, ensuring
    clean separation of concerns and avoiding execution duplication that could
    occur if approval logic was spread across multiple nodes.

    :return: Async function implementing the approval node logic
    :rtype: Callable[[PythonExecutionState], Awaitable[Dict[str, Any]]]

    .. note::
       The returned function expects the state to contain pre-created interrupt
       data from the analyzer node. It will raise RuntimeError if this data
       is missing.

    .. warning::
       This is a factory function that creates the actual node function. The
       returned function should be used as a LangGraph node, not this factory.

    .. seealso::
       :func:`osprey.services.python_executor.analysis.node.create_analyzer_node` : Creates interrupt data
       :class:`PythonExecutionState` : State structure containing interrupt data

    Examples:
        Creating and using the approval node in a LangGraph workflow::

            >>> approval_node_func = create_approval_node()
            >>> workflow.add_node("python_approval_node", approval_node_func)
            >>>
            >>> # The node processes states with interrupt data
            >>> state = PythonExecutionState(
            ...     approval_interrupt_data={"code": "...", "concerns": [...]}
            ... )
            >>> result = await approval_node_func(state)
            >>> print(f"Approved: {result['approved']}")
    """

    async def approval_node(state: PythonExecutionState) -> dict[str, Any]:
        """Process approval interrupt and return user response for workflow routing.

        This function implements the core approval node logic, serving as a pure
        interrupt processor that handles user approval requests through LangGraph's
        native interrupt system. It processes pre-created interrupt data and
        returns the user's approval decision for workflow routing.

        The function maintains minimal state and focuses solely on interrupt
        processing, delegating all approval logic to the analyzer node that
        created the interrupt data. This design ensures clean separation of
        concerns and prevents execution duplication.

        :param state: Current execution state containing pre-created interrupt data
        :type state: PythonExecutionState
        :return: State updates containing approval result and routing information
        :rtype: Dict[str, Any]
        :raises RuntimeError: If approval_interrupt_data is missing from state

        .. note::
           The function uses LangGraph's interrupt mechanism to pause execution
           and wait for user input, then processes the response for routing.

        Examples:
            State processing with approval interrupt::

                >>> state = PythonExecutionState(
                ...     approval_interrupt_data={
                ...         "code": "import os; os.listdir('/')",
                ...         "concerns": ["File system access detected"],
                ...         "execution_mode": "read_only"
                ...     }
                ... )
                >>> result = await approval_node(state)
                >>> # Result contains approval decision for routing
                >>> print(f"User approved: {result['approved']}")
        """

        # Get logger with streaming support (shadows module-level logger)
        node_logger = get_logger("python", state=state)
        node_logger.status("Requesting human approval...")

        # Get the pre-created interrupt data from analyzer
        interrupt_data = state.get("approval_interrupt_data")
        if not interrupt_data:
            raise RuntimeError("No approval interrupt data found in state")

        node_logger.info("Requesting human approval for Python code execution")

        # This is the ONLY critical line - everything else is routing
        human_response = interrupt(interrupt_data)

        # Simple approval processing for routing
        approved = human_response.get("approved", False)
        node_logger.info(f"Approval result: {approved}")

        return {"approval_result": human_response, "approved": approved}

    return approval_node
