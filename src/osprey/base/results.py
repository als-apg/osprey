"""Result Type Hierarchy - Execution Tracking and Data Management

This module provides the complete type hierarchy for tracking execution results,
records, and data throughout the Osprey Framework. The result system enables
comprehensive execution monitoring, error tracking, and state management with
type safety and consistent structure across all framework components.

The result type hierarchy serves multiple critical functions:
1. **Execution Tracking**: Complete records of capability and infrastructure execution
2. **Error Management**: Structured error information with recovery strategies
3. **Data Flow**: Type-safe data passing between capabilities and infrastructure
4. **Performance Monitoring**: Timing and execution metrics for system optimization
5. **State Management**: Consistent result storage and retrieval patterns

Type Hierarchy Structure:
    - **ExecutionResult**: Individual execution outcomes with error handling
    - **ExecutionRecord**: Historical execution records with timing
    - **CapabilityMatch**: Task classification results for capability selection

The result system integrates seamlessly with LangGraph's state management and
checkpointing systems through pure dictionary operations and standard Python
data structures. All result types support JSON serialization for persistence
and inter-process communication.

.. note::
   All result types are designed for LangGraph compatibility with proper
   serialization support and type safety through Pydantic models and dataclasses.

.. seealso::
   :mod:`osprey.state` : State management and agent state structure
   :mod:`osprey.base.errors` : Error classification and handling system
   :mod:`osprey.base.planning` : Execution planning and step management
"""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .errors import ExecutionError

if TYPE_CHECKING:
    from osprey.base.planning import PlannedStep


@dataclass
class ExecutionResult:
    """Comprehensive result container for capability and infrastructure node executions.

    This dataclass provides a complete record of execution outcomes including
    success status, result data, error information, and comprehensive timing
    details. It serves as the primary result container throughout the framework
    for execution tracking, error handling, and performance monitoring.

    ExecutionResult enables comprehensive execution analysis by capturing:
    1. **Outcome Status**: Clear success/failure indication
    2. **Result Data**: Actual output from successful executions
    3. **Error Information**: Structured error details for failures
    4. **Timing Metrics**: Performance data for optimization
    5. **Execution Context**: Temporal information for debugging

    The result structure supports both synchronous analysis and asynchronous
    processing patterns while maintaining type safety and serialization
    compatibility for persistence and inter-process communication.

    :param success: Whether the execution completed successfully without errors
    :type success: bool
    :param data: Result data from successful execution, None for failures
    :type data: Optional[Any]
    :param error: Structured error information for failed executions, None for success
    :type error: Optional[ExecutionError]
    :param execution_time: Total execution duration in seconds for performance tracking
    :type execution_time: Optional[float]
    :param start_time: UTC timestamp when execution began
    :type start_time: Optional[datetime]
    :param end_time: UTC timestamp when execution completed
    :type end_time: Optional[datetime]

    .. note::
       The success field determines which additional fields are meaningful:
       - Success=True: data should contain results, error should be None
       - Success=False: error should contain details, data should be None
       Timing fields are optional but highly recommended for monitoring.

    .. warning::
       Avoid setting both data and error fields simultaneously as this creates
       ambiguous result states. Use success field to determine the authoritative
       execution outcome.

    Examples:
        Successful execution result::

            result = ExecutionResult(
                success=True,
                data={"weather_data": weather_info, "location": "San Francisco"},
                execution_time=1.23,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )

        Failed execution result::

            result = ExecutionResult(
                success=False,
                error=ExecutionError(
                    severity=ErrorSeverity.RETRIABLE,
                    message="Connection timeout",
                    metadata={"technical_details": "HTTP 408 Request Timeout"}
                ),
                execution_time=5.0,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )

    .. seealso::
       :class:`ExecutionError` : Structured error information for failures
       :class:`ExecutionRecord` : Historical execution records with steps
    """

    success: bool
    data: Any | None = None
    error: ExecutionError | None = None
    execution_time: float | None = None  # Duration in seconds
    start_time: datetime | None = None  # When execution started
    end_time: datetime | None = None  # When execution completed


@dataclass
class ExecutionRecord:
    """Comprehensive historical record of completed execution steps.

    This dataclass maintains complete records of execution step history including
    the original planned step, comprehensive timing information, and detailed
    execution results. It serves as the primary mechanism for execution history
    tracking, performance analysis, debugging, and audit trails throughout the
    framework.

    ExecutionRecord enables comprehensive execution analysis by preserving:
    1. **Step Context**: Original planned step with objectives and requirements
    2. **Timing Data**: Detailed execution timing for performance analysis
    3. **Result Information**: Complete outcome data including success/failure details
    4. **Historical Tracking**: Sequential execution records for audit trails
    5. **Debug Information**: Context needed for troubleshooting execution issues

    The record structure supports both real-time monitoring and historical
    analysis while maintaining referential integrity between planned steps
    and their execution outcomes. Records are designed for efficient storage
    and retrieval in execution history systems.

    :param step: The planned step that was executed, containing objectives and configuration
    :type step: PlannedStep
    :param start_time: UTC timestamp when step execution began
    :type start_time: datetime
    :param result: Complete execution result with outcome data and error information
    :type result: ExecutionResult
    :param end_time: UTC timestamp when step execution completed (optional)
    :type end_time: Optional[datetime]

    .. note::
       The end_time field may be None if timing information can be derived from
       result.end_time. This provides flexibility in record construction while
       maintaining timing accuracy. Prefer result.end_time when available for
       consistency.

    .. warning::
       ExecutionRecord instances should be treated as immutable once created
       to maintain execution history integrity. Create new records rather than
       modifying existing ones.

    Examples:
        Successful step execution record::

            record = ExecutionRecord(
                step=PlannedStep(
                    context_key="weather_data",
                    capability="weather_retrieval",
                    task_objective="Get current weather for San Francisco",
                    success_criteria="Weather data retrieved with temperature"
                ),
                start_time=datetime.utcnow(),
                result=ExecutionResult(
                    success=True,
                    data={"temperature": 72, "conditions": "sunny"},
                    execution_time=1.2
                )
            )

        Failed step execution record::

            record = ExecutionRecord(
                step=planned_step,
                start_time=start_timestamp,
                result=ExecutionResult(
                    success=False,
                    error=ExecutionError(
                        severity=ErrorSeverity.RETRIABLE,
                        message="API rate limit exceeded"
                    ),
                    execution_time=0.5
                ),
                end_time=end_timestamp
            )

    .. seealso::
       :class:`PlannedStep` : Execution step planning and configuration
       :class:`ExecutionResult` : Individual execution outcome data
       :mod:`osprey.base.planning` : Execution planning system
    """

    step: "PlannedStep"  # Import handled at runtime
    start_time: datetime
    result: ExecutionResult
    end_time: datetime | None = None


class CapabilityMatch(BaseModel):
    """Task classification result for capability matching and selection.

    This Pydantic model represents the outcome of task classification analysis
    to determine whether a user's request should be handled by a specific
    capability. It serves as the primary data structure used by the classification
    system to route requests to appropriate capabilities based on sophisticated
    task analysis and capability matching algorithms.

    CapabilityMatch enables intelligent capability selection by providing:
    1. **Binary Classification**: Clear match/no-match decision for routing
    2. **Type Safety**: Pydantic validation ensures data integrity
    3. **Serialization**: JSON-compatible for inter-process communication
    4. **Integration**: Seamless integration with classification pipelines
    5. **Consistency**: Standardized format across all capability matchers

    The model is designed for use in classification workflows where multiple
    capabilities are evaluated against a user request, and the classification
    system needs to make routing decisions based on the match results.

    :param is_match: Boolean indicating whether the user's request matches this capability
    :type is_match: bool

    .. note::
       This uses Pydantic BaseModel to ensure type safety, validation, and
       JSON serialization support. The model automatically validates that
       is_match is a proper boolean value.

    .. warning::
       The classification system relies on the accuracy of this match result
       for proper capability routing. Ensure classification logic is thoroughly
       tested to avoid routing errors.

    Examples:
        Positive capability match::

            match = CapabilityMatch(is_match=True)
            # Indicates the capability should handle this request

        Negative capability match::

            match = CapabilityMatch(is_match=False)
            # Indicates the capability should not handle this request

        Usage in classification workflow::

            matches = []
            for capability in available_capabilities:
                classifier_result = classify_request(user_request, capability)
                match = CapabilityMatch(is_match=classifier_result)
                matches.append((capability, match))

            # Select capabilities with positive matches
            selected_capabilities = [
                cap for cap, match in matches if match.is_match
            ]

    .. seealso::
       :mod:`osprey.infrastructure.classifier` : Task classification system
       :class:`TaskClassifierGuide` : Classification guidance for capabilities
       :class:`ClassifierExample` : Few-shot examples for classification
    """

    is_match: bool = Field(
        description="A boolean (true or false) indicating if the user's request matches the capability."
    )
