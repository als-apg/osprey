"""Framework State - Execution State Management.

This module provides comprehensive execution state management for the Osprey
Agent Framework, including approval workflows, planning state, execution tracking,
and control flow management. The execution state system supports the complete
execution lifecycle from planning through completion.

**Core Components:**

- :class:`ApprovalRequest`: Approval workflow management for sensitive operations
- :class:`PlanningState`: Planning and orchestration state management
- :class:`ExecutionState`: Execution progress and result tracking
- :class:`ControlFlowState`: Control flow and retry logic management
- :class:`ClassificationResult`: Task classification and graph building results

**Execution Lifecycle:**

The execution state system manages the complete execution lifecycle:

1. **Classification**: Task analysis and capability selection
2. **Planning**: Execution plan creation and step definition
3. **Execution**: Step-by-step execution with result tracking
4. **Approval**: Human-in-the-loop approval for sensitive operations
5. **Control Flow**: Retry logic, error handling, and execution limits

**State Architecture:**

All state classes use TypedDict with total=False to support LangGraph's partial
state update patterns. The execution state integrates with the main AgentState
to provide comprehensive execution tracking and control.

**Approval System:**

The approval system provides human-in-the-loop control for sensitive operations:
- Python code execution approval
- Memory operation approval
- EPICS write operation approval
- Custom capability-specific approvals

.. note::
   All execution state classes support partial updates for LangGraph compatibility.
   Default values are documented in each class and applied by StateManager.

.. seealso::
   :class:`osprey.state.AgentState` : Main state structure using execution state
   :mod:`osprey.base.results` : Execution result and record structures
   :mod:`osprey.approval` : Approval system implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing_extensions import TypedDict

from osprey.base.planning import ExecutionPlan
from osprey.base.results import ExecutionRecord, ExecutionResult

# Context data is now stored as pure dictionaries - no import needed
from osprey.utils.config import get_config_value


@dataclass
class ApprovalRequest:
    """Comprehensive approval request data structure for human-in-the-loop workflows.

    This dataclass represents a complete approval request that tracks all necessary
    context for human decision-making in sensitive operations. The approval system
    uses these requests to manage pending approvals, track approval decisions, and
    provide audit trails for security-sensitive operations.

    **Approval Workflow:**

    1. **Request Creation**: Created when a capability encounters an approval requirement
    2. **Context Capture**: Records step context, capability, and operation details
    3. **Human Review**: Presented to users through UI for approval decision
    4. **Decision Tracking**: Records approval/rejection with timestamp
    5. **Execution Control**: Controls whether operation proceeds or is blocked

    **Security Integration:**

    ApprovalRequest integrates with the framework security model to provide
    controlled access to sensitive operations including:

    - Python code execution with potential system access
    - Memory operations that could store sensitive information
    - EPICS write operations that could affect accelerator systems
    - Custom capability-specific sensitive operations

    :param step_objective: Human-readable description of the step requiring approval
    :type step_objective: str
    :param capability: Name of the capability requesting approval for identification
    :type capability: str
    :param approval_type: Type of approval required for categorization and handling
    :type approval_type: str
    :param timestamp: Unix timestamp when the approval request was created
    :type timestamp: float
    :param approved: Current approval status of the request
    :type approved: bool
    :param approval_data: Original approval exception or additional context data
    :type approval_data: Optional[Any]

    .. note::
       The approval_data field can contain the original approval exception or
       additional context needed for the approval decision. This enables
       capability-specific approval handling and rich context presentation.

    .. warning::
       ApprovalRequest instances should be treated as immutable once created,
       except for the approved field which is updated when decisions are made.
       This ensures audit trail integrity.

    Examples:
        Python code execution approval::

            >>> import time
            >>> request = ApprovalRequest(
            ...     step_objective="Execute data analysis script",
            ...     capability="python_executor",
            ...     approval_type="python_code_execution",
            ...     timestamp=time.time(),
            ...     approved=False
            ... )
            >>> request.is_pending
            True

        Memory operation approval::

            >>> request = ApprovalRequest(
            ...     step_objective="Save user preferences",
            ...     capability="memory_manager",
            ...     approval_type="memory_save",
            ...     timestamp=time.time(),
            ...     approved=False,
            ...     approval_data={"memory_type": "user_preferences"}
            ... )

        Approval decision handling::

            >>> request.approve()
            >>> request.is_approved
            True
            >>> request.is_pending
            False

    .. seealso::
       :mod:`osprey.approval` : Approval system implementation
       :class:`osprey.state.AgentState` : State containing pending approvals
       :mod:`osprey.services.python_executor` : Python execution approval integration
    """

    step_objective: str
    capability: str
    approval_type: str  # e.g., "python_code_execution", "memory_save"
    timestamp: float
    approved: bool = False
    approval_data: Any | None = (
        None  # Store original approval exception for capability-specific access
    )

    @property
    def is_approved(self) -> bool:
        """Check if this approval request has been approved.

        :return: True if the request has been approved
        :rtype: bool
        """
        return self.approved

    @property
    def is_pending(self) -> bool:
        """Check if this approval request is still pending.

        :return: True if the request is still awaiting approval
        :rtype: bool
        """
        return not self.approved

    def approve(self) -> None:
        """Mark this approval request as approved.

        Sets the approved flag to True, indicating that the request has been
        granted approval and execution can proceed.
        """
        self.approved = True


class PlanningState(TypedDict, total=False):
    """Planning and orchestration state with execution plan management.

    Manages the planning phase of execution including capability selection,
    execution plan creation, and step tracking. The orchestrator agent is
    created fresh when needed rather than stored in state.

    All fields are optional to support partial updates in LangGraph.

    Default values (when not provided):
    - active_capabilities: []
    - execution_plan: None
    - current_step_index: 0
    """

    active_capabilities: list[str]  # List of capability names available for the current task
    execution_plan: ExecutionPlan | None  # Current execution plan with ordered steps
    current_step_index: int  # Index of the currently executing step


class ExecutionState(TypedDict, total=False):
    """Tracks execution progress, results, and accumulated context.

    Maintains the state of execution including step results, execution history,
    accumulated context, and pending approvals. This class provides the core
    execution tracking functionality for the agent framework.

    All fields are optional to support partial updates in LangGraph.

    Default values (when not provided):
    - step_results: {}
    - execution_history: []
    - capability_context_data: {}
    - last_result: None
    - pending_approvals: {}
    """

    step_results: dict[str, Any]  # Dictionary of results keyed by step context
    execution_history: list[ExecutionRecord]  # List of completed execution records
    capability_context_data: dict[
        str, dict[str, dict[str, Any]]
    ]  # Raw capability context data for LangGraph compatibility
    last_result: ExecutionResult | None  # Most recent execution result
    pending_approvals: dict[
        str, ApprovalRequest
    ]  # Dictionary of pending approval requests (format: {step_context_key: ApprovalRequest})


class ControlFlowState(TypedDict, total=False):
    """Manages reclassification logic and execution flow control.

    Controls the execution flow including reclassification logic, retry behavior,
    execution limits, and validation workflows. Provides safety mechanisms to
    prevent infinite loops and excessive resource consumption.

    All fields are optional to support partial updates in LangGraph.

    Default values (when not provided):
    - reclassification_reason: None
    - max_reclassifications: 1
    - reclassification_count: 0
    - current_step_retry_count: 0
    - max_step_retries: 0

    - execution_start_time: None
    - total_execution_time: None
    - max_execution_time_seconds: 300
    - is_killed: False
    - kill_reason: None
    - is_awaiting_validation: False
    - validation_context: None
    - validation_timestamp: None
    """

    # Reclassification control
    reclassification_reason: str | None  # Reason for requesting reclassification
    max_reclassifications: int  # Maximum number of reclassifications allowed
    reclassification_count: int  # Current number of reclassifications performed

    # Step retry tracking to prevent infinite loops
    current_step_retry_count: int  # Number of retries for the current step
    max_step_retries: int  # Maximum retries allowed per step

    # Execution safety and loop prevention
    execution_start_time: float | None  # Unix timestamp when execution started
    total_execution_time: float | None  # Total execution time in seconds
    max_execution_time_seconds: int  # Maximum execution time allowed
    is_killed: bool  # Whether execution has been killed
    kill_reason: str | None  # Reason for killing execution

    # Human-in-the-loop validation state
    is_awaiting_validation: bool  # Whether execution is awaiting human validation
    validation_context: dict[str, Any] | None  # Store context about what needs validation
    validation_timestamp: float | None  # When validation was requested


def create_control_flow_state_from_config() -> ControlFlowState:
    """Create ControlFlowState with limits from configuration.

    Initializes a ControlFlowState instance with execution limits extracted
    from the configuration. Provides safe defaults if configuration
    values are not available.

    :return: ControlFlowState instance with configured limits
    :rtype: ControlFlowState
    """
    # Get execution limits from config, with fallback defaults
    limits = get_config_value("execution_control.limits", {})

    return ControlFlowState(
        max_reclassifications=limits.get("max_reclassifications", 1),
        max_step_retries=limits.get("max_step_retries", 0),
        max_execution_time_seconds=limits.get("max_execution_time_seconds", 300),
    )


@dataclass
class ClassificationResult:
    """Result of task classification and graph building phase.

    Contains the output of task classification including identified requirements,
    active capabilities, and the dynamically built execution graph.

    :param requirements: Dictionary of boolean requirements identified for the task
    :type requirements: dict[str, bool]
    :param active_capabilities: List of capability names selected for task execution
    :type active_capabilities: list[str]
    :param dynamic_graph: Execution graph built for the classified task
    :type dynamic_graph: Any
    """

    requirements: dict[str, bool]
    active_capabilities: list[str]  # Store capability names instead of instances
    dynamic_graph: Any  # Graph type to avoid circular import
