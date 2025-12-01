"""Error Response Generation Infrastructure

This module provides centralized, intelligent error handling for the Osprey
Agent Framework. The ErrorNode generates comprehensive error responses by combining
structured error reports with LLM-generated explanations and recovery suggestions.

The error handling system follows a two-phase approach:
1. Structured Error Report: Automatically generated factual information including
   error classification, execution statistics, and step-by-step execution summary
2. LLM Analysis: Intelligent explanation of what went wrong and why, with context-aware
   recovery suggestions based on available system capabilities

Key Components:
    ErrorNode: Primary infrastructure node for error response generation
    ErrorContext: Data structure containing error details and execution state

Integration:
    - Receives error information from capability decorators via agent state
    - Uses ErrorClassification.format_for_llm() for consistent metadata formatting
    - Uses unified logger system for status updates and streaming
    - Returns responses as AIMessage objects for direct user presentation

The error node is designed to never fail - if LLM generation fails, it provides
a structured fallback response to ensure users always receive meaningful error information.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage

from osprey.base.decorators import infrastructure_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.nodes import BaseInfrastructureNode
from osprey.models import get_chat_completion
from osprey.prompts.loader import get_framework_prompts
from osprey.registry import get_registry
from osprey.state import AgentState, StateManager
from osprey.utils.config import get_model_config
from osprey.utils.logger import get_logger

logger = get_logger("error")


@dataclass
class ErrorContext:
    """Comprehensive error context for generating detailed error responses.

    This data structure encapsulates all information required to generate meaningful
    error reports including original error classification, execution statistics, and
    step-by-step execution history. The class serves as the primary data container
    for the error response generation pipeline, ensuring consistent access to error
    details and execution context across all error handling components.

    The ErrorContext integrates with the ErrorClassification system to maintain
    authoritative error information while adding execution-specific details such as
    timing, retry attempts, and step-by-step progress tracking.

    :param error_classification: Complete error classification with severity, messages, and metadata
    :type error_classification: ErrorClassification
    :param current_task: Human-readable description of the high-level task being executed
    :type current_task: str
    :param failed_operation: Specific operation or capability name that encountered the error
    :type failed_operation: str
    :param total_operations: Total number of operations attempted in current execution cycle
    :type total_operations: int
    :param execution_time: Duration in seconds from start to error occurrence
    :type execution_time: float, optional
    :param retry_count: Number of retry attempts made before final failure
    :type retry_count: int, optional
    :param successful_steps: Chronological list of execution steps that completed successfully
    :type successful_steps: list[str], optional
    :param failed_steps: Chronological list of execution steps that failed during execution
    :type failed_steps: list[str], optional

    .. note::
       The class automatically initializes list fields to empty lists if None is provided,
       ensuring safe iteration over step results in error report generation.

    .. warning::
       This class is designed for read-only access during error response generation.
       Modifying fields after creation may lead to inconsistent error reports.

    Examples:
        Basic error context creation::

            >>> from osprey.base.errors import ErrorClassification, ErrorSeverity
            >>> classification = ErrorClassification(
            ...     severity=ErrorSeverity.CRITICAL,
            ...     user_message="Database connection failed",
            ...     metadata={"host": "db.example.com"}
            ... )
            >>> context = ErrorContext(
            ...     error_classification=classification,
            ...     current_task="Fetch user data",
            ...     failed_operation="database_query",
            ...     execution_time=2.5
            ... )
            >>> print(f"Severity: {context.error_severity.value}")
            critical

        Context with execution history::

            >>> context = ErrorContext(
            ...     error_classification=classification,
            ...     current_task="Process user request",
            ...     failed_operation="user_authentication",
            ...     total_operations=3,
            ...     successful_steps=["Step 1: Validate input", "Step 2: Load config"],
            ...     failed_steps=["Step 3: Authenticate user - Failed"]
            ... )
            >>> print(f"Progress: {len(context.successful_steps)}/{context.total_operations}")
            Progress: 2/3

    .. seealso::
       :class:`osprey.base.errors.ErrorClassification` : Error classification system
       :class:`ErrorNode` : Primary consumer of ErrorContext instances
    """

    error_classification: ErrorClassification
    current_task: str
    failed_operation: str
    total_operations: int = 0
    execution_time: float | None = None
    retry_count: int | None = None
    successful_steps: list[str] = None
    failed_steps: list[str] = None

    def __post_init__(self):
        """Initialize list fields to empty lists if None."""
        if self.successful_steps is None:
            self.successful_steps = []
        if self.failed_steps is None:
            self.failed_steps = []

    @property
    def error_severity(self) -> ErrorSeverity:
        """Extract error severity level from the underlying error classification.

        :return: Severity level indicating error impact and recovery strategy
        :rtype: ErrorSeverity

        .. note::
           Severity levels guide error handling strategy: RETRIABLE errors may be
           retried, REPLANNING errors require task modification, CRITICAL errors
           need user intervention, and FATAL errors terminate execution.
        """
        return self.error_classification.severity

    @property
    def error_message(self) -> str:
        """Extract user-friendly error message from the error classification.

        :return: Human-readable error message suitable for user presentation,
                with automatic fallback to generic message if none provided
        :rtype: str

        .. note::
           This property ensures that error responses always contain a meaningful
           message even when the original error classification lacks user-facing text.
        """
        return self.error_classification.user_message or "Unknown error occurred"

    @property
    def capability_name(self) -> str | None:
        """Extract the name of the specific capability that encountered the error.

        :return: Name of the failing capability if available in context, None otherwise
        :rtype: str, optional

        .. note::
           This property accesses a dynamically set attribute (_capability_name)
           that is populated during error context creation from agent state.
        """
        return getattr(self, "_capability_name", None)


@infrastructure_node
class ErrorNode(BaseInfrastructureNode):
    """Generate comprehensive, user-friendly error responses with intelligent analysis.

    The ErrorNode serves as the centralized error response generation system for the
    Osprey Agent Framework. It transforms technical error information into
    comprehensive user responses by combining structured factual reports with
    context-aware LLM analysis and recovery suggestions.

    This infrastructure node operates as the final destination in the error handling
    pipeline, ensuring that all system failures result in meaningful, actionable
    information for users. The node implements a robust two-phase approach to error
    response generation with multiple fallback mechanisms to guarantee response
    delivery even under adverse conditions.

    Architecture Overview:
        The error response generation follows a structured two-phase approach:

        1. **Structured Report Generation**:
           - Extracts error details from agent state control_error_info
           - Formats using ErrorClassification.format_for_llm() for consistency
           - Adds execution statistics, timing data, and retry information
           - Generates step-by-step execution summaries with success/failure tracking

        2. **LLM Analysis Phase**:
           - Provides error context and available system capabilities to LLM
           - Generates intelligent explanations of failure causes
           - Produces context-aware recovery suggestions and next steps
           - Integrates with framework prompt system for consistent analysis quality

    Error Recovery Strategy:
        The node implements multiple layers of error handling to ensure reliability:
        - Comprehensive fallback response if LLM generation fails
        - Self-classification of internal errors as FATAL to prevent infinite loops
        - Structured logging of all error generation attempts for monitoring
        - Guaranteed response delivery through robust exception handling

    Integration Points:
        - **Input**: Pre-classified errors from capability decorators via agent state
        - **Logging**: Unified logger system for status updates and streaming
        - **Output**: AIMessage objects formatted for direct user presentation
        - **Monitoring**: Comprehensive logging integration for operational visibility

    .. warning::
       The ErrorNode must never raise unhandled exceptions as it serves as the
       final error handling mechanism. All internal errors are caught and result
       in structured fallback responses.

    .. note::
       Error classification within this node always uses FATAL severity to prevent
       recursive error handling that could lead to infinite loops or system instability.

    Examples:
        The ErrorNode is typically invoked automatically by the framework, but can
        be tested with manual state construction::

            >>> from osprey.state import AgentState
            >>> from osprey.base.errors import ErrorClassification, ErrorSeverity
            >>>
            >>> # Construct agent state with error information
            >>> state = AgentState()
            >>> state['control_error_info'] = {
            ...     'classification': ErrorClassification(
            ...         severity=ErrorSeverity.CRITICAL,
            ...         user_message="Database connection timeout",
            ...         metadata={"timeout": 30, "host": "db.example.com"}
            ...     ),
            ...     'capability_name': 'database_query',
            ...     'execution_time': 31.5
            ... }
            >>> state['task_current_task'] = "Retrieve user profile data"
            >>>
            >>> # Execute error response generation
            >>> result = await ErrorNode.execute(state)
            >>> print(f"Response type: {type(result['messages'][0])}")
            <class 'langchain_core.messages.ai.AIMessage'>

        Framework integration through error decorator::

            >>> @capability("database_operations")
            ... async def query_user_data(user_id: int, state: AgentState):
            ...     # This will automatically route to ErrorNode on failure
            ...     connection = await get_db_connection()
            ...     return await connection.fetch_user(user_id)

    .. seealso::
       :class:`ErrorContext` : Data structure for error response generation
       :class:`osprey.base.errors.ErrorClassification` : Error classification system
       :func:`osprey.base.decorators.capability` : Capability decorator with error handling
       :class:`osprey.state.AgentState` : Agent state management system
    """

    name = "error"
    description = "Error Response Generation"

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify internal ErrorNode failures with FATAL severity to prevent infinite loops.

        This method handles the critical responsibility of classifying errors that occur
        within the error response generation system itself. All such errors are automatically
        classified as FATAL to ensure clean termination and prevent recursive error handling
        scenarios that could destabilize the entire system.

        The FATAL classification ensures that if the error response generation mechanism
        fails, execution terminates immediately rather than attempting additional error
        recovery operations that could compound the original problem or create infinite
        error handling loops.

        :param exc: Exception that occurred during error response generation process
        :type exc: Exception
        :param context: Execution context containing node information, timing data, and state
        :type context: dict
        :return: Error classification with FATAL severity and diagnostic metadata
        :rtype: ErrorClassification

        .. warning::
           This method should only be called by the framework's error handling system.
           Manual invocation could disrupt the error classification hierarchy.

        .. note::
           The FATAL severity ensures immediate execution termination without further
           error recovery attempts, preventing system instability.

        Examples:
            Framework automatic error classification::

                >>> try:
                ...     # ErrorNode internal operation fails
                ...     await ErrorNode.execute(state)
                ... except Exception as e:
                ...     classification = ErrorNode.classify_error(e, context)
                ...     print(f"Severity: {classification.severity.value}")
                fatal

            Error classification structure::

                >>> context = {"node_name": "error", "execution_time": 1.2}
                >>> exc = RuntimeError("LLM generation failed")
                >>> classification = ErrorNode.classify_error(exc, context)
                >>> print(classification.metadata["technical_details"])
                Error node failure: LLM generation failed
        """
        return ErrorClassification(
            severity=ErrorSeverity.FATAL,
            user_message="Error node failed during error handling",
            metadata={"technical_details": f"Error node failure: {str(exc)}"},
        )

    async def execute(self) -> dict[str, Any]:
        """Generate comprehensive error response with structured analysis and LLM insights.

        This method orchestrates the complete error response generation pipeline,
        transforming technical error information into user-friendly responses with
        actionable recovery suggestions. The process combines factual error reporting
        with intelligent analysis to provide maximum value to users encountering issues.

        The execution follows a carefully designed two-phase approach that ensures
        robust error handling even when components of the error generation system
        itself experience failures.

        Processing Pipeline:
            1. **Context Extraction**: Reads error details from agent state including
               error classification, execution statistics, and step-by-step history
            2. **Context Population**: Enriches error context with execution timeline,
               successful operations, and failure categorization
            3. **Structured Report Generation**: Creates factual error report using
               standardized formatting and execution statistics
            4. **LLM Analysis**: Generates intelligent explanations and recovery
               suggestions based on error context and available capabilities
            5. **Response Assembly**: Combines structured report with LLM analysis
               into coherent user response

        Error Handling Strategy:
            - Comprehensive exception handling prevents method failure
            - Automatic fallback to structured response if LLM generation fails
            - All failures logged for operational monitoring and debugging

        :param state: Agent state containing error information in control_error_info field
        :type state: AgentState
        :param kwargs: Additional LangGraph execution parameters including config and streaming
        :type kwargs: dict
        :return: Dictionary containing AIMessage with formatted error response for user presentation
        :rtype: dict[str, list[AIMessage]]

        .. note::
           This method is designed to never raise exceptions. All internal errors
           result in structured fallback responses to ensure users receive meaningful
           information regardless of system state.

        .. warning::
           The method expects error information to be present in state['control_error_info'].
           Missing error information will result in fallback responses with generic messaging.

        Examples:
            Standard error response generation::

                >>> from osprey.state import AgentState
                >>> from osprey.base.errors import ErrorClassification, ErrorSeverity
                >>>
                >>> # Prepare agent state with error information
                >>> state = AgentState()
                >>> state['control_error_info'] = {
                ...     'classification': ErrorClassification(
                ...         severity=ErrorSeverity.REPLANNING,
                ...         user_message="API rate limit exceeded",
                ...         metadata={"retry_after": 300}
                ...     ),
                ...     'capability_name': 'external_api_call',
                ...     'execution_time': 5.2
                ... }
                >>> state['task_current_task'] = "Fetch weather data"
                >>>
                >>> # Generate error response
                >>> result = await ErrorNode.execute(state)
                >>> message = result['messages'][0]
                >>> print(f"Response length: {len(message.content)} characters")
                Response length: 847 characters

            Error response with execution history::

                >>> state['execution_step_results'] = {
                ...     'step_0': {
                ...         'step_index': 0,
                ...         'capability': 'input_validation',
                ...         'task_objective': 'Validate API parameters',
                ...         'success': True
                ...     },
                ...     'step_1': {
                ...         'step_index': 1,
                ...         'capability': 'external_api_call',
                ...         'task_objective': 'Fetch weather data',
                ...         'success': False
                ...     }
                ... }
                >>> result = await ErrorNode.execute(state)
                >>> # Response includes execution summary with successful/failed steps

        .. seealso::
           :func:`_create_error_context_from_state` : Error context extraction
           :func:`_generate_error_response` : Response generation pipeline
           :class:`AIMessage` : Response message format
        """
        state = self._state
        logger = self.get_logger()

        try:
            error_context = _create_error_context_from_state(state)
            _populate_error_context(error_context, state)

            response = await _generate_error_response(error_context)

            return {"messages": [AIMessage(content=response)]}

        except Exception as e:
            logger.error(f"Error response generation failed: {e}")
            return {"messages": [AIMessage(content=_create_fallback_response(state, e))]}


def _create_fallback_response(state: AgentState, generation_error: Exception) -> str:
    """Generate structured fallback response when LLM error analysis fails.

    This function provides a critical safety mechanism for error response generation
    by creating meaningful error messages even when the intelligent LLM analysis
    component fails. The fallback response combines original error information
    with diagnostic details about the analysis failure, ensuring users receive
    actionable information regardless of system state.

    The fallback mechanism maintains the dual-error reporting pattern where both
    the original operational error and the secondary error generation failure are
    clearly communicated to users with appropriate context and severity indication.

    :param state: Agent state containing original error details in control_error_info
    :type state: AgentState
    :param generation_error: Exception that occurred during LLM analysis generation
    :type generation_error: Exception
    :return: Formatted fallback error message combining original and generation errors
    :rtype: str

    .. warning::
       This function is called only when the primary error response generation
       fails. It should be robust and never raise additional exceptions.

    .. note::
       The fallback response format clearly distinguishes between the original
       operational error and the secondary error generation failure.

    Examples:
        Fallback response for LLM generation failure::

            >>> from osprey.state import AgentState
            >>> state = AgentState()
            >>> state['control_error_info'] = {
            ...     'original_error': 'Database connection timeout',
            ...     'capability_name': 'user_data_fetch'
            ... }
            >>> generation_error = RuntimeError("OpenAI API unavailable")
            >>> response = _create_fallback_response(state, generation_error)
            >>> print("⚠️" in response and "Original Issue" in response)
            True

        Fallback with minimal error information::

            >>> state = AgentState()  # Empty state
            >>> generation_error = ConnectionError("Network timeout")
            >>> response = _create_fallback_response(state, generation_error)
            >>> print("unknown operation" in response)
            True

    .. seealso::
       :func:`ErrorNode.execute` : Primary error response generation
       :func:`_generate_error_response` : LLM-based error analysis
    """
    error_info = state.get("control_error_info", {})
    original_error = error_info.get("original_error", "Unknown error occurred")
    capability_name = error_info.get("capability_name", "unknown operation")

    return f"""⚠️ **System Error During Error Handling**

**Original Issue:** The '{capability_name}' operation failed with: {original_error}

**Secondary Issue:** The error response generation system encountered an internal error: {str(generation_error)}

This appears to be a system-level issue. The original operation failed (which may be expected), but the error handling system also experienced problems.
"""


def _create_error_context_from_state(state: AgentState) -> ErrorContext:
    """Extract and structure comprehensive error information from agent execution state.

    This function serves as the primary interface between the agent state management
    system and the error response generation pipeline. It reads error details from
    the control_error_info field (populated by capability decorators) and constructs
    a complete ErrorContext object containing all information needed for comprehensive
    error response generation.

    The function implements robust error information extraction with fallback
    mechanisms to ensure error response generation can proceed even when state
    information is incomplete or malformed. ErrorClassification serves as the
    authoritative source for error severity and metadata.

    Error Information Sources:
        - **control_error_info**: Primary error details from capability decorators
        - **task_current_task**: High-level task description for context
        - **control_retry_count**: Retry attempt information for execution statistics
        - **execution_step_results**: Step-by-step execution history for progress tracking

    :param state: Agent state containing error information and complete execution history
    :type state: AgentState
    :return: Comprehensive error context ready for response generation pipeline
    :rtype: ErrorContext

    .. note::
       If no error classification is found in state, the function creates a fallback
       CRITICAL classification to ensure error response generation can proceed safely.

    .. warning::
       This function assumes state follows the standard agent state structure.
       Malformed state may result in incomplete error context with fallback values.

    Examples:
        Extract context from complete error state::

            >>> from osprey.state import AgentState
            >>> from osprey.base.errors import ErrorClassification, ErrorSeverity
            >>>
            >>> state = AgentState()
            >>> state['control_error_info'] = {
            ...     'classification': ErrorClassification(
            ...         severity=ErrorSeverity.RETRIABLE,
            ...         user_message="Temporary network error",
            ...         metadata={"retry_delay": 5}
            ...     ),
            ...     'capability_name': 'api_client',
            ...     'execution_time': 2.8
            ... }
            >>> state['task_current_task'] = "Fetch user preferences"
            >>> state['control_retry_count'] = 2
            >>>
            >>> context = _create_error_context_from_state(state)
            >>> print(f"Severity: {context.error_severity.value}, Retries: {context.retry_count}")
            Severity: retriable, Retries: 2

        Fallback context creation with minimal state::

            >>> state = AgentState()
            >>> state['control_error_info'] = {
            ...     'original_error': 'Unknown system error'
            ... }
            >>> context = _create_error_context_from_state(state)
            >>> print(f"Fallback severity: {context.error_severity.value}")
            Fallback severity: critical

    .. seealso::
       :class:`ErrorContext` : Target data structure for error information
       :class:`osprey.base.errors.ErrorClassification` : Error classification system
       :func:`_populate_error_context` : Additional context population
    """
    current_task = state.get("task_current_task", "Unknown task")
    error_info = state.get("control_error_info", {})

    # Extract or create error classification
    error_classification = error_info.get("classification")
    if not error_classification:
        original_error = error_info.get("original_error", "Unknown error occurred")
        error_classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=original_error,
            metadata={"technical_details": original_error},
        )

    # Extract execution details
    capability_name = error_info.get("capability_name") or error_info.get("node_name")
    execution_time = error_info.get("execution_time", 0.0)
    retry_count = state.get("control_retry_count", 0)
    failed_operation = capability_name or "Unknown operation"

    # Construct error context
    context = ErrorContext(
        error_classification=error_classification,
        current_task=current_task,
        failed_operation=failed_operation,
        execution_time=execution_time,
        retry_count=retry_count,
        total_operations=StateManager.get_current_step_index(state) + 1,
    )

    # Store capability name for property access
    context._capability_name = capability_name
    return context


def _populate_error_context(error_context: ErrorContext, state: AgentState) -> None:
    """Enrich error context with chronological execution history and step categorization.

    This function enhances the error context with detailed execution history by
    extracting step-by-step results from agent state and organizing them into
    successful and failed operation categories. This provides users with valuable
    context about system behavior leading up to the failure, enabling better
    understanding of the error's impact and potential recovery strategies.

    The function processes execution_step_results from agent state, maintaining
    chronological order through step_index sorting to present a coherent narrative
    of system execution. Each step is categorized based on its success status and
    formatted with descriptive information for user comprehension.

    Processing Logic:
        1. Extract execution_step_results from agent state
        2. Sort results by step_index to maintain chronological order
        3. Format each step with index, capability name, and task objective
        4. Categorize steps into successful_steps or failed_steps based on status
        5. Populate error_context lists in-place for response generation

    :param error_context: Error context object to enhance with execution history
    :type error_context: ErrorContext
    :param state: Agent state containing execution_step_results with detailed step information
    :type state: AgentState

    .. note::
       This function modifies the error_context object in-place, populating the
       successful_steps and failed_steps lists with formatted step descriptions.

    .. warning::
       The function assumes execution_step_results follows the standard format
       with step_index, capability, task_objective, and success fields.

    Examples:
        Populate context with execution history::

            >>> from osprey.state import AgentState
            >>> context = ErrorContext(
            ...     error_classification=classification,
            ...     current_task="Process user request",
            ...     failed_operation="database_query"
            ... )
            >>> state = AgentState()
            >>> state['execution_step_results'] = {
            ...     'step_0': {
            ...         'step_index': 0,
            ...         'capability': 'input_validation',
            ...         'task_objective': 'Validate user input',
            ...         'success': True
            ...     },
            ...     'step_1': {
            ...         'step_index': 1,
            ...         'capability': 'database_query',
            ...         'task_objective': 'Fetch user data',
            ...         'success': False
            ...     }
            ... }
            >>> _populate_error_context(context, state)
            >>> print(f"Successful: {len(context.successful_steps)}, Failed: {len(context.failed_steps)}")
            Successful: 1, Failed: 1

        Handle missing execution results gracefully::

            >>> state = AgentState()  # No execution_step_results
            >>> _populate_error_context(context, state)
            >>> print(f"Steps populated: {len(context.successful_steps + context.failed_steps)}")
            Steps populated: 0

    .. seealso::
       :class:`ErrorContext` : Error context data structure
       :class:`osprey.state.AgentState` : Agent state management
       :func:`_create_error_context_from_state` : Initial context creation
    """
    step_results = state.get("execution_step_results", {})
    if not step_results:
        return

    # Sort by step_index to maintain chronological order
    ordered_results = sorted(step_results.items(), key=lambda x: x[1].get("step_index", 0))

    for _, result in ordered_results:
        step_index = result.get("step_index", 0)
        capability_name = result.get("capability", "unknown")
        task_objective = result.get("task_objective", capability_name)
        step_description = f"Step {step_index + 1}: {task_objective}"

        if result.get("success", False):
            error_context.successful_steps.append(step_description)
        else:
            error_context.failed_steps.append(f"{step_description} - Failed")


async def _generate_error_response(error_context: ErrorContext) -> str:
    """Orchestrate complete error response generation with structured reporting and LLM analysis.

    This function coordinates the comprehensive two-phase error response generation
    process that transforms technical error information into user-friendly responses.
    The approach combines factual structured reporting with intelligent LLM analysis
    to provide maximum value and actionability for users encountering system errors.

    The function implements asynchronous processing to maintain system responsiveness
    during LLM generation while ensuring that streaming progress updates continue
    to provide real-time feedback to users. The structured report provides immediate
    factual information while the LLM analysis adds contextual understanding and
    recovery guidance.

    Response Generation Pipeline:
        1. **Structured Report Building**: Creates factual error report using standardized
           formatting with execution statistics, timing information, and step summaries
        2. **Asynchronous LLM Analysis**: Generates intelligent explanations and recovery
           suggestions using context-aware prompting and capability information
        3. **Response Assembly**: Combines structured report with LLM insights into
           coherent, comprehensive user response

    :param error_context: Complete error context containing classification and execution data
    :type error_context: ErrorContext
    :return: Formatted error response combining structured facts with intelligent analysis
    :rtype: str

    .. note::
       Uses asyncio.to_thread for LLM generation to prevent blocking of streaming
       updates and maintain system responsiveness during response generation.

    .. warning::
       This function assumes error_context is fully populated with all required
       fields. Incomplete context may result in reduced response quality.

    Examples:
        Generate complete error response::

            >>> from osprey.base.errors import ErrorClassification, ErrorSeverity
            >>> context = ErrorContext(
            ...     error_classification=ErrorClassification(
            ...         severity=ErrorSeverity.REPLANNING,
            ...         user_message="API authentication failed",
            ...         metadata={"endpoint": "/api/v1/users"}
            ...     ),
            ...     current_task="Fetch user profile",
            ...     failed_operation="api_authentication",
            ...     execution_time=1.2,
            ...     successful_steps=["Step 1: Validate input"],
            ...     failed_steps=["Step 2: Authenticate API - Failed"]
            ... )
            >>> response = await _generate_error_response(context)
            >>> print("ERROR REPORT" in response and "Analysis:" in response)
            True

        Response structure verification::

            >>> response_lines = response.split('\n')
            >>> structured_section = any("ERROR REPORT" in line for line in response_lines)
            >>> analysis_section = any("Analysis:" in line for line in response_lines)
            >>> print(f"Structured: {structured_section}, Analysis: {analysis_section}")
            Structured: True, Analysis: True

    .. seealso::
       :func:`_build_structured_error_report` : Structured report generation
       :func:`_generate_llm_explanation` : LLM analysis generation
       :class:`ErrorContext` : Error context data structure
    """
    error_report_sections = _build_structured_error_report(error_context)
    llm_explanation = await asyncio.to_thread(_generate_llm_explanation, error_context)
    return f"{error_report_sections}\n\n{llm_explanation}"


def _build_structured_error_report(error_context: ErrorContext) -> str:
    """Build comprehensive structured error report with standardized formatting.

    This function creates the factual foundation of error responses by generating
    standardized reports that combine error classification details with execution
    statistics and chronological progress summaries. The structured report provides
    immediate, actionable information to users while maintaining consistent formatting
    across all error types and severity levels.

    The report generation uses ErrorClassification.format_for_llm() for consistent
    error detail formatting and integrates execution-specific information including
    timing data, retry statistics, and step-by-step progress tracking to provide
    comprehensive context for error understanding and recovery planning.

    Report Structure:
        - **Header**: Timestamp and severity indication
        - **Context**: Task description and failed operation identification
        - **Error Details**: Standardized error classification formatting
        - **Execution Statistics**: Timing, operations count, and retry information
        - **Progress Summary**: Chronological breakdown of successful and failed steps

    :param error_context: Complete error context with classification and execution data
    :type error_context: ErrorContext
    :return: Formatted structured report ready for integration with LLM analysis
    :rtype: str

    .. note::
       The function uses ErrorClassification.format_for_llm() when available,
       falling back to basic error message formatting for compatibility.

    .. warning::
       This function assumes error_context contains valid error_classification.
       Missing classification will result in fallback formatting.

    Examples:
        Generate structured report with full context::

            >>> from osprey.base.errors import ErrorClassification, ErrorSeverity
            >>> context = ErrorContext(
            ...     error_classification=ErrorClassification(
            ...         severity=ErrorSeverity.CRITICAL,
            ...         user_message="Database connection failed",
            ...         metadata={"host": "db.example.com", "port": 5432}
            ...     ),
            ...     current_task="Update user profile",
            ...     failed_operation="database_connection",
            ...     total_operations=3,
            ...     execution_time=15.7,
            ...     retry_count=2,
            ...     successful_steps=["Step 1: Validate input", "Step 2: Prepare query"],
            ...     failed_steps=["Step 3: Connect to database - Failed"]
            ... )
            >>> report = _build_structured_error_report(context)
            >>> print("ERROR REPORT" in report and "CRITICAL" in report)
            True

        Report with minimal execution context::

            >>> minimal_context = ErrorContext(
            ...     error_classification=ErrorClassification(
            ...         severity=ErrorSeverity.RETRIABLE,
            ...         user_message="Temporary network error"
            ...     ),
            ...     current_task="Fetch data",
            ...     failed_operation="network_request"
            ... )
            >>> report = _build_structured_error_report(minimal_context)
            >>> print("Execution Stats" not in report)  # No stats to include
            True

    .. seealso::
       :class:`ErrorContext` : Error context data structure
       :meth:`ErrorClassification.format_for_llm` : Standardized error formatting
       :func:`_generate_error_response` : Complete response generation
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_sections = [
        f"⚠️  **ERROR REPORT** - {timestamp}",
        f"**Error Severity:** {error_context.error_severity.value.upper()}",
        f"**Task:** {error_context.current_task}",
        f"**Failed Operation:** {error_context.failed_operation}",
    ]

    if error_context.capability_name:
        report_sections.append(f"**Capability:** {error_context.capability_name}")

    # Use standardized error formatting
    if hasattr(error_context.error_classification, "format_for_llm"):
        error_details = error_context.error_classification.format_for_llm()
        report_sections.append(error_details)
    else:
        report_sections.append(f"**Error Message:** {error_context.error_message}")

    # Add execution statistics
    stats_parts = []
    if error_context.total_operations > 0:
        stats_parts.append(f"Total operations: {error_context.total_operations}")
    if error_context.execution_time is not None:
        stats_parts.append(f"Execution time: {error_context.execution_time:.1f}s")
    if error_context.retry_count and error_context.retry_count > 0:
        stats_parts.append(f"Retry attempts: {error_context.retry_count}")

    if stats_parts:
        report_sections.append(f"**Execution Stats:** {', '.join(stats_parts)}")

    # Add execution summary if steps were tracked
    if error_context.successful_steps or error_context.failed_steps:
        summary_lines = ["**Execution Summary:**"]
        if error_context.successful_steps:
            summary_lines.append("✅ **Completed successfully:**")
            summary_lines.extend(f"   • {step}" for step in error_context.successful_steps)
        if error_context.failed_steps:
            summary_lines.append("❌ **Failed steps:**")
            summary_lines.extend(f"   • {step}" for step in error_context.failed_steps)
        report_sections.extend(summary_lines)

    return "\n".join(report_sections)


def _generate_llm_explanation(error_context: ErrorContext) -> str:
    """Generate intelligent error analysis and recovery suggestions using LLM reasoning.

    This function leverages large language model capabilities to create context-aware
    error explanations and actionable recovery suggestions. The LLM analysis goes
    beyond basic error reporting to provide intelligent insights about failure causes,
    system context, and practical next steps for error resolution.

    The function integrates error context with system capability information to
    generate contextually appropriate suggestions that align with available system
    functionality. Robust error handling ensures that LLM generation failures never
    prevent error response delivery, with graceful fallback to structured messaging.

    Analysis Generation Process:
        1. **Context Assembly**: Combines error details with system capabilities overview
        2. **Prompt Construction**: Uses framework prompt system for consistent analysis quality
        3. **LLM Generation**: Invokes language model with optimized parameters for concise analysis
        4. **Response Validation**: Ensures generated content is meaningful and properly formatted
        5. **Fallback Handling**: Provides structured fallback if LLM generation fails

    :param error_context: Complete error context with classification and execution details
    :type error_context: ErrorContext
    :return: Formatted LLM analysis with explanations and recovery suggestions,
             or structured fallback message if generation fails
    :rtype: str

    .. note::
       The function uses framework model configuration for "response" type to ensure
       appropriate token limits and generation parameters for error analysis.

    .. warning::
       This function handles all LLM generation errors internally. External callers
       should not expect exceptions from this function.

    Examples:
        Generate LLM analysis for API error::

            >>> from osprey.base.errors import ErrorClassification, ErrorSeverity
            >>> context = ErrorContext(
            ...     error_classification=ErrorClassification(
            ...         severity=ErrorSeverity.REPLANNING,
            ...         user_message="Rate limit exceeded",
            ...         metadata={"retry_after": 300, "limit": 1000}
            ...     ),
            ...     current_task="Fetch weather data",
            ...     failed_operation="weather_api_call",
            ...     execution_time=0.8
            ... )
            >>> analysis = _generate_llm_explanation(context)
            >>> print(analysis.startswith("**Analysis:**"))
            True

        Fallback behavior on LLM failure::

            >>> # Simulate LLM unavailability
            >>> import unittest.mock
            >>> with unittest.mock.patch('osprey.models.get_chat_completion', side_effect=Exception("API down")):
            ...     analysis = _generate_llm_explanation(context)
            ...     print("structured report" in analysis.lower())
            True

    .. seealso::
       :func:`osprey.models.get_chat_completion` : LLM generation interface
       :func:`osprey.prompts.loader.get_framework_prompts` : Prompt system
       :func:`osprey.registry.get_registry` : System capabilities registry
    """
    try:
        capabilities_overview = get_registry().get_capabilities_overview()
        prompt_provider = get_framework_prompts()
        error_builder = prompt_provider.get_error_analysis_prompt_builder()

        prompt = error_builder.get_system_instructions(
            capabilities_overview=capabilities_overview, error_context=error_context
        )

        explanation = get_chat_completion(
            model_config=get_model_config("response"), message=prompt, max_tokens=500
        )

        if isinstance(explanation, str) and explanation.strip():
            return f"**Analysis:** {explanation.strip()}"
        else:
            return "**Analysis:** The error occurred during system operation. Please review the recovery options above."

    except Exception as e:
        logger.error(f"Error generating LLM explanation: {e}")
        return "**Analysis:** Error details are provided in the structured report above."
