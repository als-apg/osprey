"""Core Models and State Management for XOpt Optimizer Service.

This module provides the foundational data structures, state management classes,
and configuration utilities that support the XOpt optimizer service's
LangGraph-based workflow.

The module is organized into several key areas:

**Type Definitions**: Core data structures for execution requests, results, and
metadata tracking. These provide type-safe interfaces for service communication
and ensure consistent data handling across the optimization pipeline.

**State Management**: LangGraph-compatible state classes that track execution
progress, approval workflows, and intermediate results throughout the service
execution lifecycle.

**Enumerations**: Machine state and strategy enums that drive workflow decisions
and routing logic.

Key Design Principles:
    - **Type Safety**: All public interfaces use Pydantic models or dataclasses
      with comprehensive type annotations
    - **LangGraph Integration**: State classes implement TypedDict patterns for
      seamless integration with LangGraph's state management and checkpointing
    - **Placeholder-First**: Machine-affecting components use placeholders until
      domain requirements are defined by operators
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from osprey.utils.logger import get_logger

logger = get_logger("xopt_optimizer")


# =============================================================================
# CUSTOM REDUCERS FOR STATE MANAGEMENT
# =============================================================================


def preserve_once_set(existing: Any | None, new: Any | None) -> Any | None:
    """Preserve field value once set - never allow it to be replaced or lost.

    This reducer ensures that critical fields like 'request' are never lost during
    LangGraph state updates, including checkpoint resumption with Command objects.

    Args:
        existing: Current value of the field (may be None)
        new: New value being applied to the field (may be None)

    Returns:
        The existing value if it's set, otherwise the new value
    """
    if existing is not None:
        return existing
    return new


# =============================================================================
# ENUMERATIONS
# =============================================================================


class MachineState(str, Enum):
    """Machine states for optimization readiness.

    NOTE: These are placeholders. Actual states will be determined
    based on facility requirements and operator feedback.

    DO NOT add accelerator-specific states without operator input.
    """

    READY = "ready"  # Machine ready for optimization
    NOT_READY = "not_ready"  # Cannot proceed (reason in details)
    UNKNOWN = "unknown"  # Assessment inconclusive

    # Domain-specific states to be added based on facility requirements, e.g.:
    # NO_CHARGE = "no_charge"
    # NO_BEAM = "no_beam"
    # INTERLOCK_ACTIVE = "interlock_active"


class XOptStrategy(str, Enum):
    """Optimization strategy to execute."""

    EXPLORATION = "exploration"  # Explore parameter space
    OPTIMIZATION = "optimization"  # Optimize toward goal
    ABORT = "abort"  # Cannot proceed


# =============================================================================
# ERROR TRACKING
# =============================================================================


@dataclass
class XOptError:
    """Structured error information for debugging and iteration refinement.

    Captures error context to help subsequent nodes understand what failed
    and potentially adjust their approach.

    :param error_type: Category of error (state_assessment, yaml_generation, execution, analysis)
    :param error_message: Human-readable error message
    :param stage: Pipeline stage where error occurred
    :param attempt_number: Which attempt this error occurred in
    :param details: Additional error details for debugging
    """

    error_type: str
    error_message: str
    stage: str
    attempt_number: int = 1
    details: dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        """Format error for inclusion in agent prompts."""
        parts = [f"**Attempt {self.attempt_number} - {self.stage.upper()} FAILED**"]
        parts.append(f"\n**Error Type:** {self.error_type}")
        parts.append(f"**Error:** {self.error_message}")
        if self.details:
            parts.append(f"\n**Details:** {self.details}")
        return "\n".join(parts)


# =============================================================================
# REQUEST MODEL
# =============================================================================


class XOptExecutionRequest(BaseModel):
    """Request model for XOpt optimization service.

    Serializable request that captures all information needed to run
    an optimization workflow. Matches the pattern from PythonExecutionRequest.

    :param user_query: User's optimization request
    :param optimization_objective: What to optimize
    :param max_iterations: Maximum optimization iterations
    :param retries: Maximum YAML generation retries per iteration
    :param reference_files_path: Path to reference documentation
    :param historical_yamls_path: Path to historical YAML configurations
    :param capability_context_data: Capability context from main graph state
    :param require_approval: Whether human approval is required
    :param session_context: Session info including chat_id, user_id
    """

    user_query: str = Field(..., description="User's optimization request")
    optimization_objective: str = Field(..., description="What to optimize")

    max_iterations: int = Field(default=3, description="Maximum optimization iterations")
    retries: int = Field(default=3, description="Maximum YAML generation retries per iteration")

    # Paths to reference data (configured per deployment)
    reference_files_path: str | None = None
    historical_yamls_path: str | None = None

    # Capability context (for cross-capability data access)
    capability_context_data: dict[str, Any] | None = Field(
        None, description="Capability context data from main graph state"
    )

    # Standard Osprey fields
    require_approval: bool = Field(default=True)
    session_context: dict[str, Any] | None = Field(
        None, description="Session info including chat_id, user_id"
    )


# =============================================================================
# SERVICE RESULT
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class XOptServiceResult:
    """Structured, type-safe result from XOpt optimizer service.

    This eliminates the need for validation and error checking in capabilities.
    The service guarantees this structure is always returned on success.
    On failure, the service raises appropriate exceptions.

    :param run_artifact: Optimization run output data
    :param generated_yaml: XOpt YAML configuration used
    :param strategy: Strategy used (exploration/optimization)
    :param total_iterations: Number of iterations completed
    :param analysis_summary: Summary of optimization analysis
    :param recommendations: List of recommendations from analysis
    """

    run_artifact: dict[str, Any]
    generated_yaml: str
    strategy: XOptStrategy
    total_iterations: int
    analysis_summary: dict[str, Any]
    recommendations: tuple[str, ...]  # Use tuple for frozen dataclass

    def __post_init__(self):
        """Validate immutable structure."""
        # Frozen dataclass handles immutability


# =============================================================================
# STATE MANAGEMENT
# =============================================================================


class XOptExecutionState(TypedDict):
    """LangGraph state for XOpt optimizer service.

    This state is used internally by the service and includes both the
    original request and execution tracking fields.

    CRITICAL: The 'request' field uses the preserve_once_set reducer to ensure
    it's never lost during state updates or checkpoint resumption (e.g., approval workflows).

    NOTE: capability_context_data is extracted to top level for ContextManager compatibility.
    """

    # Original request (preserved via reducer) - NEVER lost once set
    request: Annotated[XOptExecutionRequest, preserve_once_set]

    # Capability context data (for cross-capability integration)
    capability_context_data: dict[str, dict[str, dict[str, Any]]] | None

    # Error tracking (matches Python executor pattern)
    error_chain: list[XOptError]
    yaml_generation_attempt: int  # For YAML regeneration retries

    # Machine state assessment
    machine_state: MachineState | None
    machine_state_details: dict[str, Any] | None  # Readings, reasoning, etc.

    # Decision
    selected_strategy: XOptStrategy | None
    decision_reasoning: str | None

    # YAML configuration
    generated_yaml: str | None
    yaml_generation_failed: bool | None

    # Approval state (standard Osprey pattern)
    requires_approval: bool | None
    approval_interrupt_data: dict[str, Any] | None  # LangGraph interrupt data
    approval_result: dict[str, Any] | None  # Response from interrupt
    approved: bool | None  # Final approval status

    # Execution
    run_artifact: Any | None  # In-memory result from XOpt
    execution_error: str | None
    execution_failed: bool | None

    # Analysis
    analysis_result: dict[str, Any] | None
    recommendations: list[str] | None

    # Loop control
    iteration_count: int
    max_iterations: int
    should_continue: bool

    # Control flags
    is_successful: bool
    is_failed: bool
    failure_reason: str | None
    current_stage: str  # "state_id", "decision", "yaml_gen", "approval", "execution", "analysis", "complete", "failed"
