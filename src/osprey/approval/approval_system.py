"""Osprey Approval System for Production-Ready Approval Workflows.

This module provides the core LangGraph-native approval system that integrates
with the production Gateway architecture for clean, secure approval handling.
The system enables structured interrupts for human approval of operations
requiring oversight, with comprehensive validation and error handling.

Key Components:
    - Dynamic approval type generation for flexible capability integration
    - Structured interrupt data creation for LangGraph compatibility
    - State management utilities for approval workflow handling
    - Service integration helpers for consistent interrupt propagation

The approval system supports multiple operation types including execution plans,
code execution, and memory operations, with extensible architecture for adding
new approval types without framework modifications.

Examples:
    Create approval for code execution::

        >>> interrupt_data = create_code_approval_interrupt(
        ...     code="print('Hello, World!')",
        ...     analysis_details={'safety_level': 'low'},
        ...     execution_mode='readonly',
        ...     safety_concerns=[]
        ... )
        >>> # Use interrupt_data with LangGraph interrupt() function

    Handle approval resume::

        >>> has_resume, payload = get_approval_resume_data(state, "python_executor")
        >>> if has_resume and payload:
        ...     code = payload['code']
        ...     # Execute approved code

.. note::
   This module is designed for security-critical operations. All approval
   functions include comprehensive validation and error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.state import AgentState

from langgraph.types import interrupt

from osprey.base.planning import ExecutionPlan
from osprey.events import ErrorEvent, EventEmitter, StatusEvent
from osprey.utils.logger import get_logger

logger = get_logger("approval_system")


# =============================================================================
# LANGGRAPH-NATIVE APPROVAL SYSTEM
# =============================================================================


def create_approval_type(capability_name: str, operation_type: str = None) -> str:
    """Generate dynamic approval type identifier from capability and operation.

    Creates unique approval type identifiers that replace the hard-coded ApprovalType
    enum with a flexible system. This enables any capability to request approval
    without requiring framework modifications, while maintaining clear identification
    and supporting operation-level granularity within capabilities.

    The generated identifiers follow a consistent naming pattern that ensures
    uniqueness and readability for logging, debugging, and user interfaces.

    :param capability_name: Name of the capability requesting approval
    :type capability_name: str
    :param operation_type: Optional specific operation type for granular control
    :type operation_type: str, optional
    :return: Unique string identifier for the approval type
    :rtype: str

    Examples:
        Basic capability approval::

            >>> approval_type = create_approval_type("python")
            >>> print(approval_type)
            "python"

        Operation-specific approval::

            >>> approval_type = create_approval_type("memory", "save")
            >>> print(approval_type)
            "memory_save"

        Complex capability with operation::

            >>> approval_type = create_approval_type("data_analysis", "execute_query")
            >>> print(approval_type)
            "data_analysis_execute_query"

    .. seealso::
       :func:`create_code_approval_interrupt` : Uses this function for approval type creation
       :func:`create_memory_approval_interrupt` : Uses this function for approval type creation
       :func:`create_plan_approval_interrupt` : Uses this function for approval type creation
       :func:`get_approval_resume_data` : Uses approval types for state management
    """
    if operation_type:
        return f"{capability_name}_{operation_type}"
    return capability_name


def create_plan_approval_interrupt(
    execution_plan: ExecutionPlan, plan_file_path: str = None, pending_plans_dir: str = None
) -> dict[str, Any]:
    """Create structured interrupt data for execution plan approval with file-based storage support.

    Generates LangGraph-compatible interrupt data that presents execution plans
    to users for approval. The interrupt includes formatted step details, clear
    approval instructions, and structured payload data for seamless resume
    operations after user approval.

    The function supports file-based execution plan storage for enhanced
    human-in-the-loop workflows, particularly for Open WebUI integration.

    The generated user message provides a comprehensive view of planned operations
    with step-by-step breakdown, making it easy for users to understand and
    evaluate the proposed execution plan.

    :param execution_plan: Execution plan object containing steps and configuration
    :type execution_plan: ExecutionPlan
    :param plan_file_path: Optional file path where the execution plan was saved
    :type plan_file_path: str, optional
    :param pending_plans_dir: Optional directory path for pending plan files
    :type pending_plans_dir: str, optional
    :return: Dictionary containing user_message and resume_payload for LangGraph
    :rtype: Dict[str, Any]

    Examples:
        Basic plan approval::

            >>> from osprey.base.planning import ExecutionPlan
            >>> plan = ExecutionPlan(steps=[
            ...     {'task_objective': 'Load data', 'capability': 'data_loader'},
            ...     {'task_objective': 'Analyze trends', 'capability': 'data_analysis'}
            ... ])
            >>> interrupt_data = create_plan_approval_interrupt(plan)
            >>> print(interrupt_data['user_message'])  # Contains formatted approval request

    .. note::
       The interrupt data follows LangGraph's standard structure with user_message
       for display and resume_payload for execution continuation.

    .. seealso::
       :class:`osprey.base.planning.ExecutionPlan` : Input structure for this function
       :func:`create_approval_type` : Approval type generation used by this function
       :func:`get_approval_resume_data` : Function that processes the resume payload
       :func:`clear_approval_state` : State cleanup after approval processing
    """
    # Extract plan structure for user presentation
    steps = execution_plan.get("steps", [])
    estimated_steps = len(steps)

    # Build human-readable approval prompt with step details
    steps_text = ""
    for i, step in enumerate(steps, 1):
        steps_text += f"**Step {i}:** {step.get('task_objective', 'unknown')} ({step.get('capability', 'unknown')})\n"

    # Add file information if available
    file_info = ""
    if plan_file_path:
        file_info = f"\n**Plan File:** `{plan_file_path}`"
        if pending_plans_dir:
            file_info += f"\n**Plans Directory:** `{pending_plans_dir}`"
        file_info += "\n"

    user_message = f"""
⚠️ **HUMAN APPROVAL REQUIRED** ⚠️

**Planned Steps ({estimated_steps} total):**
{steps_text}{file_info}

**To proceed, respond with:**
- **`yes`** to approve and execute the plan
- **`no`** to cancel this operation
""".strip()

    # Create enhanced resume payload with file information
    resume_payload = {
        "approval_type": create_approval_type("orchestrator", "plan"),
        "execution_plan": execution_plan,  # Keep for backward compatibility
    }

    # Add file-based parameters if provided
    if plan_file_path:
        resume_payload["plan_file_path"] = plan_file_path
    if pending_plans_dir:
        resume_payload["pending_plans_dir"] = pending_plans_dir

    return {"user_message": user_message, "resume_payload": resume_payload}


def create_step_approval_interrupt(
    step: dict[str, Any],
    step_number: int,
    execution_plan: dict[str, Any],
) -> dict[str, Any]:
    """Create structured interrupt data for reactive orchestrator step approval.

    Generates LangGraph-compatible interrupt data that presents a single
    reactive step to the user for approval before execution.  Used when
    ``planning_mode_enabled`` is ``True`` in reactive orchestration mode.

    The resume payload includes the execution plan and reactive loop state
    (``react_messages``, ``react_step_count``) so that the orchestrator can
    restore its state on approval without re-calling the LLM.

    :param step: The planned step dict (capability, task_objective, …)
    :type step: dict[str, Any]
    :param step_number: Human-readable step number (1-based)
    :type step_number: int
    :param execution_plan: Single-step execution plan containing the step
    :type execution_plan: dict[str, Any]
    :return: Dictionary containing user_message and resume_payload for LangGraph
    :rtype: Dict[str, Any]
    """
    capability = step.get("capability", "unknown")
    objective = step.get("task_objective", "unknown")
    expected_output = step.get("expected_output", "N/A")

    user_message = f"""
⚠️ **HUMAN APPROVAL REQUIRED** ⚠️

**Reactive Step {step_number}:**
- **Capability:** {capability}
- **Objective:** {objective}
- **Expected Output:** {expected_output}

**To proceed, respond with:**
- **`yes`** to approve and execute this step
- **`no`** to reject and choose a different approach
""".strip()

    return {
        "user_message": user_message,
        "resume_payload": {
            "approval_type": create_approval_type("reactive_orchestrator", "step"),
            "execution_plan": execution_plan,
            "step": step,
            "step_number": step_number,
        },
    }


def create_memory_approval_interrupt(
    content: str,
    operation_type: str,
    user_id: str,
    existing_memory: str = "",
    step_objective: str = "Save content to memory",
) -> dict[str, Any]:
    """Create structured interrupt data for memory operation approval.

    Generates LangGraph-compatible interrupt data for memory operations that
    require human approval. The interrupt presents the memory content clearly
    formatted for user review, along with operation context and clear approval
    instructions.

    This function supports all memory operations (create, update, delete) and
    provides appropriate context for each operation type. The structured payload
    enables seamless resume after user approval.

    :param content: Memory content to be saved, updated, or referenced for deletion
    :type content: str
    :param operation_type: Type of memory operation being requested
    :type operation_type: str
    :param user_id: Unique identifier for the user requesting the operation
    :type user_id: str
    :param existing_memory: Current memory content when updating existing memories
    :type existing_memory: str, optional
    :param step_objective: High-level objective description for user context
    :type step_objective: str
    :return: Dictionary containing user_message and resume_payload for LangGraph
    :rtype: Dict[str, Any]

    Examples:
        Create new memory::

            >>> interrupt_data = create_memory_approval_interrupt(
            ...     content="User prefers morning meetings",
            ...     operation_type="create",
            ...     user_id="user123",
            ...     step_objective="Save user preference"
            ... )
            >>> print('yes' in interrupt_data['user_message'])  # Shows approval options

        Update existing memory::

            >>> interrupt_data = create_memory_approval_interrupt(
            ...     content="Updated preference: afternoon meetings",
            ...     operation_type="update",
            ...     user_id="user123",
            ...     existing_memory="User prefers morning meetings",
            ...     step_objective="Update user preference"
            ... )

    .. note::
       The content is displayed in a code block format for clear readability
       and to preserve formatting of structured data.
    """
    user_message = f"""
⚠️ **HUMAN APPROVAL REQUIRED** ⚠️

**Task:** {step_objective}

Memory save operation requires human approval

**Content to {operation_type}:**
```
{content}
```

**To proceed, respond with:**
- **`yes`** to approve and save to memory
- **`no`** to cancel this operation
""".strip()

    return {
        "user_message": user_message,
        "resume_payload": {
            "approval_type": create_approval_type("memory", operation_type),
            "step_objective": step_objective,
            "content": content,
            "operation_type": operation_type,
            "user_id": user_id,
            "existing_memory": existing_memory,
        },
    }


def create_channel_write_approval_interrupt(
    operations: list,
    analysis_details: dict[str, Any],
    safety_concerns: list[str] | None = None,
    step_objective: str = "Write values to control system channels",
) -> dict[str, Any]:
    """Create structured interrupt data for channel write operation approval.

    Generates LangGraph-compatible interrupt data for channel write operations that
    require human approval before execution. The interrupt provides comprehensive
    context including channel addresses, target values, verification levels, and
    clear approval instructions.

    :param operations: List of WriteOperation objects to be approved
    :type operations: list
    :param analysis_details: Analysis details including operation count and channels
    :type analysis_details: Dict[str, Any]
    :param safety_concerns: List of identified safety concerns or risks
    :type safety_concerns: List[str], optional
    :param step_objective: High-level objective description for user context
    :type step_objective: str
    :return: Dictionary containing user_message and resume_payload for LangGraph
    :rtype: Dict[str, Any]
    """
    if safety_concerns is None:
        safety_concerns = []

    # Build operation summary
    operations_text = ""
    for i, (channel, value) in enumerate(analysis_details.get("values", []), 1):
        operations_text += f"**{i}.** {channel} = {value}\n"

    # Build safety concerns section
    safety_section = ""
    if safety_concerns:
        safety_section = "\n**⚠️  Safety Concerns:**\n"
        for concern in safety_concerns:
            safety_section += f"- {concern}\n"

    user_message = f"""
⚠️ **HUMAN APPROVAL REQUIRED** ⚠️

**Task:** {step_objective}

Channel write operation requires human approval

**Channels to write ({analysis_details.get("operation_count", 0)} total):**
{operations_text}{safety_section}

**To proceed, respond with:**
- **`yes`** to approve and execute the write operations
- **`no`** to cancel this operation
""".strip()

    return {
        "user_message": user_message,
        "resume_payload": {
            "approval_type": create_approval_type("channel_write"),
            "step_objective": step_objective,
            "operations": [
                {
                    "channel_address": op.channel_address,
                    "value": op.value,
                    "units": op.units,
                    "notes": op.notes,
                }
                for op in operations
            ],
            "analysis_details": analysis_details,
            "safety_concerns": safety_concerns,
        },
    }


def create_code_approval_interrupt(
    code: str,
    analysis_details: dict[str, Any],
    execution_mode: str,
    safety_concerns: list[str],
    notebook_path: Path | None = None,
    notebook_link: str | None = None,
    execution_request: Any | None = None,
    expected_results: dict[str, Any] | None = None,
    execution_folder_path: Path | None = None,
    step_objective: str = "Execute Python code",
) -> dict[str, Any]:
    """Create structured interrupt data for Python code execution approval.

    Generates LangGraph-compatible interrupt data for Python code that requires
    human approval before execution. The interrupt provides comprehensive context
    including code analysis, safety assessment, execution environment details,
    and clear approval instructions.

    The function supports multiple execution modes and integrates with Jupyter
    notebooks for code review. Safety concerns and analysis details are presented
    to help users make informed approval decisions.

    :param code: Python code requiring approval before execution
    :type code: str
    :param analysis_details: Results from code analysis including safety assessment
    :type analysis_details: Dict[str, Any]
    :param execution_mode: Mode of execution (readonly, simulation, write)
    :type execution_mode: str
    :param safety_concerns: List of identified safety concerns or risks
    :type safety_concerns: List[str]
    :param notebook_path: File system path to Jupyter notebook for review
    :type notebook_path: Path, optional
    :param notebook_link: Web link to notebook interface for code review
    :type notebook_link: str, optional
    :param execution_request: Complete execution request data for context
    :type execution_request: Any, optional
    :param expected_results: Anticipated results or outputs from code execution
    :type expected_results: Dict[str, Any], optional
    :param execution_folder_path: Directory path where code will be executed
    :type execution_folder_path: Path, optional
    :param step_objective: High-level objective description for user context
    :type step_objective: str
    :return: Dictionary containing user_message and resume_payload for LangGraph
    :rtype: Dict[str, Any]

    Examples:
        Basic code approval::

            >>> interrupt_data = create_code_approval_interrupt(
            ...     code="import pandas as pd\ndf = pd.read_csv('data.csv')",
            ...     analysis_details={'safety_level': 'low', 'file_operations': ['read']},
            ...     execution_mode='readonly',
            ...     safety_concerns=[],
            ...     step_objective="Load and analyze data"
            ... )
            >>> 'yes' in interrupt_data['user_message']
            True

        Code with safety concerns::

            >>> interrupt_data = create_code_approval_interrupt(
            ...     code="os.system('rm -rf /')",
            ...     analysis_details={'safety_level': 'critical'},
            ...     execution_mode='write',
            ...     safety_concerns=['System command execution', 'File deletion'],
            ...     notebook_link="http://localhost:8888/notebooks/review.ipynb"
            ... )

    .. warning::
       This function is used for security-critical approval decisions. Ensure
       analysis_details and safety_concerns are thoroughly populated.
    """
    # Create notebook review section
    if notebook_link:
        notebook_section = f"**📓 Review Code:** [Open Jupyter Notebook]({notebook_link})"
    else:
        notebook_section = "**Code is available for review in the execution environment.**"

    reasoning = analysis_details.get(
        "approval_reasoning", f"Python code requires human approval for {execution_mode} mode"
    )

    user_message = f"""
⚠️ **HUMAN APPROVAL REQUIRED** ⚠️

**Task:** {step_objective}

{reasoning}

{notebook_section}

**To proceed, respond with:**
- **`yes`** to approve and execute the code
- **`no`** to cancel this operation
""".strip()

    return {
        "user_message": user_message,
        "resume_payload": {
            "approval_type": create_approval_type("python_executor"),
            "step_objective": step_objective,
            "code": code,
            "analysis_details": analysis_details,
            "execution_mode": execution_mode,
            "safety_concerns": safety_concerns,
            "notebook_path": str(notebook_path) if notebook_path else None,
            "notebook_link": notebook_link,
            "execution_request": execution_request,
            "expected_results": expected_results,
            "execution_folder_path": str(execution_folder_path) if execution_folder_path else None,
        },
    }


# =============================================================================
# STREAMLINED APPROVAL HELPERS
# =============================================================================


def get_approval_resume_data(
    state: AgentState, expected_approval_type: str
) -> tuple[bool, dict[str, Any] | None]:
    """Extract and validate approval resume data from agent state.

    Provides standardized, type-safe access to approval state with comprehensive
    validation. This function serves as the single source of truth for checking
    approval resume state, ensuring all capabilities handle approval consistently.

    The function performs extensive validation to detect invalid or inconsistent
    approval states, preventing security vulnerabilities from malformed approval
    data. It distinguishes between normal execution, approved resumes, and
    rejected operations.

    :param state: Current agent state containing approval information
    :type state: AgentState
    :param expected_approval_type: Expected approval type for validation
    :type expected_approval_type: str
    :return: Tuple containing resume status and payload data
        - has_approval_resume: True if this is resuming from approval
        - approved_payload: Payload data if approved, None if rejected/normal
    :rtype: tuple[bool, Optional[Dict[str, Any]]]
    :raises ValueError: If approval state structure is invalid or inconsistent

    Examples:
        Normal execution (no approval state)::

            >>> has_resume, payload = get_approval_resume_data(state, "python")
            >>> print(f"Resume: {has_resume}, Payload: {payload}")
            Resume: False, Payload: None

        Approved resume::

            >>> # After user approves code execution
            >>> has_resume, payload = get_approval_resume_data(state, "python_executor")
            >>> if has_resume and payload:
            ...     print(f"Executing approved code: {payload['code'][:50]}...")

        Rejected operation::

            >>> # After user rejects approval
            >>> has_resume, payload = get_approval_resume_data(state, "python_executor")
            >>> if has_resume and not payload:
            ...     print("Operation was rejected by user")

    .. note::
       All capabilities should use this function instead of direct state access
       to ensure consistent approval handling and proper validation.
    """
    approval_approved = state.get("approval_approved")
    approved_payload = state.get("approved_payload")

    # No approval state = normal execution
    if approval_approved is None:
        return False, None

    # Validate approval state structure when present
    if approval_approved and not approved_payload:
        raise ValueError("approval_approved=True but no approved_payload found")

    if approved_payload:
        if not isinstance(approved_payload, dict):
            raise ValueError(f"approved_payload must be dict, got {type(approved_payload)}")

        if "approval_type" not in approved_payload:
            raise ValueError("approved_payload missing required 'approval_type' field")

        # Validate approval_type is a non-empty string
        approval_type = approved_payload["approval_type"]
        if not isinstance(approval_type, str) or not approval_type.strip():
            raise ValueError(
                f"approval_type must be a non-empty string, got: {repr(approval_type)}"
            )

    # Has approval state = this is a resume
    if approval_approved:
        # Extract and validate payload for the expected type
        payload = get_approved_payload_from_state(state, expected_approval_type)
        if not payload:
            raise ValueError(
                f"Approval was approved but no valid payload found for type {expected_approval_type}"
            )
        return True, payload
    else:
        # Explicitly rejected
        return True, None


def get_approved_payload_from_state(
    state: AgentState, expected_approval_type: str
) -> dict[str, Any] | None:
    """Extract approved payload directly from agent state for specific approval type.

    Provides direct access to approved payload data from agent state with type
    validation. This is a lower-level function used internally by the approval
    system for payload extraction. Most capabilities should use
    get_approval_resume_data() instead for full validation.

    :param state: Current agent state containing approval information
    :type state: AgentState
    :param expected_approval_type: Approval type identifier to match against
    :type expected_approval_type: str
    :return: Approved payload data if available and matches type, None otherwise
    :rtype: Optional[Dict[str, Any]]

    Examples:
        Extract specific payload::

            >>> payload = get_approved_payload_from_state(state, "memory_save")
            >>> if payload:
            ...     content = payload.get('content')
            ...     user_id = payload.get('user_id')

    .. note::
       This function performs minimal validation. Use get_approval_resume_data()
       for comprehensive validation and error handling.
    """
    if state.get("approval_approved") and (payload := state.get("approved_payload")):
        if payload.get("approval_type") == expected_approval_type:
            return payload
    return None


def clear_approval_state() -> dict[str, Any]:
    """Clear approval state to prevent contamination between operations.

    Provides centralized cleanup of approval state fields to maintain clean
    state hygiene between operations. This prevents approval data from previous
    interrupts from interfering with subsequent operations, ensuring each
    approval request is handled independently.

    This function is typically called after processing approval results or
    when initializing new operations that should not inherit approval state.

    :return: Dictionary containing approval state fields reset to None
    :rtype: Dict[str, Any]

    Examples:
        Clean state after processing approval::

            >>> # After handling approved operation
            >>> state_updates = clear_approval_state()
            >>> # Apply to current state
            >>> current_state.update(state_updates)

        Initialize clean operation::

            >>> # Before starting new capability that might need approval
            >>> clean_state = clear_approval_state()
            >>> new_state = {**current_state, **clean_state}

    .. note::
       This function only returns the state updates - callers must apply
       them to the actual state object.
    """
    return {"approval_approved": None, "approved_payload": None}


async def handle_service_with_interrupts(
    service: Any,
    request: Any,
    config: dict[str, Any],
    logger,
    capability_name: str = "parent_capability",
) -> Any:
    """Handle service calls with consistent interrupt propagation.

    Provides standardized handling for service calls that may generate
    GraphInterrupts, ensuring consistent interrupt propagation from subgraphs
    to the main graph. This eliminates duplicate interrupt handling code
    across capabilities while maintaining proper error handling.

    The function catches GraphInterrupts from subgraph services, extracts
    the interrupt data, and re-raises them in the main graph context.
    Non-interrupt exceptions are re-raised unchanged for normal error handling.

    :param service: Service instance to invoke (must support ainvoke method)
    :type service: Any
    :param request: Request object to send to the service
    :type request: Any
    :param config: Configuration dictionary for service invocation
    :type config: Dict[str, Any]
    :param logger: Logger instance for operation tracking and debugging
    :type logger: logging.Logger
    :param capability_name: Name of calling capability for logging context
    :type capability_name: str
    :return: Service result if execution completes normally
    :rtype: Any
    :raises RuntimeError: If interrupt handling fails or interrupt mechanism fails
    :raises Exception: Re-raises any non-GraphInterrupt exceptions unchanged

    Examples:
        Handle service with potential interrupts::

            >>> import logging
            >>> logger = logging.getLogger(__name__)
            >>> try:
            ...     result = await handle_service_with_interrupts(
            ...         service=python_executor_service,
            ...         request={'code': 'print("hello")', 'mode': 'readonly'},
            ...         config={'timeout': 30},
            ...         logger=logger,
            ...         capability_name='data_analysis'
            ...     )
            ...     print(f"Service completed: {result}")
            ... except RuntimeError as e:
            ...     print(f"Service handling failed: {e}")

    .. warning::
       This function expects GraphInterrupts to follow LangGraph's standard
       structure. Malformed interrupts will cause RuntimeError to be raised.

    .. note::
       The interrupt() call should pause execution - if it returns normally,
       a RuntimeError is raised as this indicates a system malfunction.
    """
    # Create event emitter for typed events
    emitter = EventEmitter(capability_name)

    try:
        # Call the service - may return normally or raise GraphInterrupt
        service_result = await service.ainvoke(request, config)

        # Emit completion status
        emitter.emit(
            StatusEvent(
                component=capability_name,
                message="Service completed normally",
                level="info",
            )
        )
        return service_result

    except Exception as e:
        # Import here to avoid circular imports
        from langgraph.errors import GraphInterrupt

        # Check if this is a GraphInterrupt from the subgraph
        if isinstance(e, GraphInterrupt):
            # Emit interrupt status
            emitter.emit(
                StatusEvent(
                    component=capability_name,
                    message="Service interrupted - waiting for approval",
                    level="info",
                )
            )

            try:
                # Extract interrupt data from GraphInterrupt using standard structure
                # GraphInterrupt structure: e.args[0][0].value contains the interrupt data
                interrupt_data = e.args[0][0].value
                emitter.emit(
                    StatusEvent(
                        component=capability_name,
                        message=f"Extracted interrupt data with keys: {list(interrupt_data.keys())}",
                        level="debug",
                    )
                )

                # Create new interrupt in main graph context using the extracted data
                # Note: Interrupt event is already emitted via ApprovalRequiredEvent
                interrupt(interrupt_data)

                # This line should never be reached - interrupt() should pause execution
                emitter.emit(
                    ErrorEvent(
                        component=capability_name,
                        error_type="SystemError",
                        error_message="UNEXPECTED: interrupt() returned instead of pausing execution",
                        recoverable=False,
                    )
                )
                raise RuntimeError(f"Interrupt mechanism failed in {capability_name}")

            except (IndexError, KeyError, AttributeError) as extract_error:
                # Emit error event for user visibility
                emitter.emit(
                    ErrorEvent(
                        component=capability_name,
                        error_type="InterruptError",
                        error_message=f"Failed to extract interrupt data: {extract_error}",
                        recoverable=False,
                        stack_trace=str(extract_error),
                    )
                )
                emitter.emit(
                    StatusEvent(
                        component=capability_name,
                        message=f"GraphInterrupt args structure: {e.args}",
                        level="debug",
                    )
                )
                raise RuntimeError(
                    f"{capability_name}: Failed to handle service interrupt: {extract_error}"
                ) from extract_error
        else:
            # Handle all other exceptions as actual errors - re-raise as-is
            emitter.emit(
                ErrorEvent(
                    component=capability_name,
                    error_type="ServiceError",
                    error_message=f"Service failed: {str(e)}",
                    recoverable=False,
                )
            )
            raise
