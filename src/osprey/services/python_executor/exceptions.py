"""Comprehensive Exception Hierarchy for Python Executor Service.

This module defines a clean, categorized exception hierarchy that provides precise
error classification for all failure modes in the Python executor service. The
exceptions are designed to support intelligent retry logic, user-friendly error
reporting, and comprehensive debugging information.

The exception system follows a principled design based on error categories that
determine appropriate recovery strategies:

**Infrastructure Errors**: Issues with container connectivity, configuration, or
external service dependencies. These errors typically warrant execution retries
without code regeneration.

**Code-Related Errors**: Problems with generated or user-provided code including
syntax errors, runtime failures, and logical issues. These errors trigger code
regeneration and analysis retry cycles.

**Workflow Errors**: Service workflow control issues including timeouts, approval
requirements, and maximum retry limits. These errors require special handling
or user intervention.

**Configuration Errors**: Invalid configuration settings or missing required
configuration data. These errors typically indicate deployment or setup issues.

Key Design Principles:
    - **Categorized Recovery**: Each exception includes category information that
      determines the appropriate recovery strategy (retry execution, regenerate code,
      or abort with user notification)
    - **Rich Context**: Exceptions capture technical details, file paths, and
      metadata to support comprehensive debugging and error reporting
    - **User-Friendly Messages**: Infrastructure errors provide user-friendly
      messages that abstract technical details while preserving debugging information
    - **Audit Trail**: Exceptions include error chains and attempt tracking to
      provide complete visibility into failure sequences

The exception hierarchy integrates with the service's retry logic and error
handling strategies to provide robust failure recovery while maintaining
clear separation between different types of errors.

.. note::
   All exceptions inherit from PythonExecutorException, which provides common
   functionality for error categorization and context management.

.. seealso::
   :class:`PythonExecutorService` : Service that raises and handles these exceptions
   :class:`ErrorCategory` : Enumeration of error categories for recovery logic
   :func:`osprey.services.python_executor.analysis.node` : Analysis node error handling

Examples:
    Catching and handling different error categories::

        >>> try:
        ...     result = await executor.execute_code(code)
        ... except ContainerConnectivityError as e:
        ...     logger.warning(f"Container unreachable: {e.get_user_message()}")
        ...     # Retry with different container or fallback to local execution
        ... except CodeRuntimeError as e:
        ...     logger.info(f"Code failed, regenerating: {e.message}")
        ...     # Trigger code regeneration with error feedback
        ... except ExecutionTimeoutError as e:
        ...     logger.error(f"Execution timeout: {e.timeout_seconds}s")
        ...     # Notify user and potentially increase timeout

    Error category-based retry logic::

        >>> if exception.should_retry_execution():
        ...     # Infrastructure error - retry same code
        ...     await retry_execution(code)
        ... elif exception.should_retry_code_generation():
        ...     # Code error - regenerate and retry
        ...     new_code = await regenerate_code(error_feedback=str(exception))
        ...     await execute_code(new_code)
"""

from enum import Enum
from pathlib import Path
from typing import Any


class ErrorCategory(Enum):
    """High-level error categories that determine appropriate recovery strategies.

    This enumeration classifies all Python executor errors into categories that
    directly correspond to different recovery and retry strategies. The categorization
    enables intelligent error handling that can automatically determine whether to
    retry execution, regenerate code, or require user intervention.

    :cvar INFRASTRUCTURE: Container connectivity, network, or external service issues
    :cvar CODE_RELATED: Syntax errors, runtime failures, or logical issues in generated code
    :cvar WORKFLOW: Service workflow control issues like timeouts or retry limits
    :cvar CONFIGURATION: Invalid or missing configuration settings

    .. note::
       Error categories are used by the service's retry logic to determine the
       appropriate recovery strategy without requiring explicit error type checking.

    .. seealso::
       :class:`PythonExecutorException` : Base exception class using these categories
       :meth:`PythonExecutorException.should_retry_execution` : Infrastructure retry logic
       :meth:`PythonExecutorException.should_retry_code_generation` : Code regeneration logic
    """
    INFRASTRUCTURE = "infrastructure"  # Container/connectivity issues
    CODE_RELATED = "code_related"      # Syntax/runtime/logic errors
    WORKFLOW = "workflow"              # Approval, timeout, etc.
    CONFIGURATION = "configuration"   # Config/setup issues


class PythonExecutorException(Exception):
    """Base exception class for all Python executor service operations.

    This abstract base class provides common functionality for all Python executor
    exceptions, including error categorization, context management, and retry logic
    determination. It serves as the foundation for the entire exception hierarchy
    and enables consistent error handling across the service.

    The class implements a category-based approach to error handling that allows
    the service to automatically determine appropriate recovery strategies without
    requiring explicit exception type checking in the retry logic.

    :param message: Human-readable error description
    :type message: str
    :param category: Error category that determines recovery strategy
    :type category: ErrorCategory
    :param technical_details: Additional technical information for debugging
    :type technical_details: Dict[str, Any], optional
    :param folder_path: Path to execution folder if available for debugging
    :type folder_path: Path, optional

    .. note::
       This base class should not be raised directly. Use specific exception
       subclasses that provide more detailed error information.

    .. seealso::
       :class:`ErrorCategory` : Error categorization for recovery strategies
       :class:`ContainerConnectivityError` : Infrastructure error example
       :class:`CodeRuntimeError` : Code-related error example
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        technical_details: dict[str, Any] | None = None,
        folder_path: Path | None = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.technical_details = technical_details or {}
        self.folder_path = folder_path

    def is_infrastructure_error(self) -> bool:
        """Check if this is an infrastructure or connectivity error.

        Infrastructure errors indicate problems with external dependencies like
        container connectivity, network issues, or service availability. These
        errors typically warrant retrying the same operation after a delay.

        :return: True if this is an infrastructure error
        :rtype: bool

        Examples:
            Checking error type for retry logic::

                >>> try:
                ...     await execute_code(code)
                ... except PythonExecutorException as e:
                ...     if e.is_infrastructure_error():
                ...         await asyncio.sleep(1)  # Brief delay
                ...         await execute_code(code)  # Retry same code
        """
        return self.category == ErrorCategory.INFRASTRUCTURE

    def is_code_error(self) -> bool:
        """Check if this is a code-related error requiring code regeneration.

        Code errors indicate problems with the generated or provided Python code,
        including syntax errors, runtime failures, or logical issues. These errors
        typically require regenerating the code with error feedback.

        :return: True if this is a code-related error
        :rtype: bool

        Examples:
            Handling code errors with regeneration::

                >>> try:
                ...     await execute_code(code)
                ... except PythonExecutorException as e:
                ...     if e.is_code_error():
                ...         new_code = await regenerate_code(error_feedback=str(e))
                ...         await execute_code(new_code)
        """
        return self.category == ErrorCategory.CODE_RELATED

    def is_workflow_error(self) -> bool:
        """Check if this is a workflow control error requiring special handling.

        Workflow errors indicate issues with the service's execution workflow,
        such as timeouts, maximum retry limits, or approval requirements. These
        errors typically require user intervention or service configuration changes.

        :return: True if this is a workflow control error
        :rtype: bool

        Examples:
            Handling workflow errors with user notification::

                >>> try:
                ...     await execute_code(code)
                ... except PythonExecutorException as e:
                ...     if e.is_workflow_error():
                ...         await notify_user(f"Execution failed: {e.message}")
        """
        return self.category == ErrorCategory.WORKFLOW

    def should_retry_execution(self) -> bool:
        """Determine if the same code execution should be retried.

        Returns True for infrastructure errors where the code itself is likely
        correct but external dependencies (containers, network) caused the failure.
        This enables automatic retry of the same code without regeneration.

        :return: True if execution should be retried with the same code
        :rtype: bool

        Examples:
            Automatic retry logic based on error category::

                >>> if exception.should_retry_execution():
                ...     logger.info("Infrastructure issue, retrying execution...")
                ...     await retry_execution_with_backoff(code)
        """
        return self.category == ErrorCategory.INFRASTRUCTURE

    def should_retry_code_generation(self) -> bool:
        """Determine if code should be regenerated and execution retried.

        Returns True for code-related errors where the generated code has issues
        that require regeneration with error feedback. This enables automatic
        code improvement through iterative generation.

        :return: True if code should be regenerated and execution retried
        :rtype: bool

        Examples:
            Code regeneration retry logic::

                >>> if exception.should_retry_code_generation():
                ...     logger.info("Code issue, regenerating with feedback...")
                ...     improved_code = await regenerate_with_feedback(str(exception))
                ...     await execute_code(improved_code)
        """
        return self.category == ErrorCategory.CODE_RELATED


# =============================================================================
# INFRASTRUCTURE ERRORS (Container/Connectivity Issues)
# =============================================================================

class ContainerConnectivityError(PythonExecutorException):
    """Exception raised when Jupyter container is unreachable or connection fails.

    This infrastructure error indicates that the Python executor service cannot
    establish communication with the configured Jupyter container endpoint. This
    typically occurs due to network issues, container startup problems, or
    configuration mismatches.

    The error provides both technical details for debugging and user-friendly
    messages that abstract the underlying infrastructure complexity while
    preserving essential information for troubleshooting.

    :param message: Technical error description for debugging
    :type message: str
    :param host: Container host address that failed to connect
    :type host: str
    :param port: Container port that failed to connect
    :type port: int
    :param technical_details: Additional technical information for debugging
    :type technical_details: Dict[str, Any], optional

    .. note::
       This error triggers automatic retry logic since the code itself is likely
       correct and the issue is with external infrastructure.

    .. seealso::
       :class:`ContainerConfigurationError` : Configuration-related container issues
       :class:`PythonExecutorException.should_retry_execution` : Retry logic for infrastructure errors

    Examples:
        Handling container connectivity issues::

            >>> try:
            ...     result = await container_executor.execute_code(code)
            ... except ContainerConnectivityError as e:
            ...     logger.warning(f"Container issue: {e.get_user_message()}")
            ...     # Automatic retry or fallback to local execution
            ...     result = await local_executor.execute_code(code)
    """

    def __init__(
        self,
        message: str,
        host: str,
        port: int,
        technical_details: dict[str, Any] | None = None
    ):
        super().__init__(message, ErrorCategory.INFRASTRUCTURE, technical_details)
        self.host = host
        self.port = port

    def get_user_message(self) -> str:
        """Get user-friendly error message abstracting technical details.

        Provides a clear, non-technical explanation of the connectivity issue
        that users can understand without needing to know about container
        infrastructure details.

        :return: User-friendly error description
        :rtype: str

        Examples:
            Displaying user-friendly error messages::

                >>> error = ContainerConnectivityError(
                ...     "Connection refused", "localhost", 8888
                ... )
                >>> print(error.get_user_message())
                Python execution environment is not reachable at localhost:8888
        """
        return f"Python execution environment is not reachable at {self.host}:{self.port}"


class ContainerConfigurationError(PythonExecutorException):
    """Container configuration is invalid"""

    def __init__(self, message: str, technical_details: dict[str, Any] | None = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, technical_details)


# =============================================================================
# CODE-RELATED ERRORS (Require Code Regeneration)
# =============================================================================

class CodeGenerationError(PythonExecutorException):
    """LLM failed to generate valid code"""

    def __init__(
        self,
        message: str,
        generation_attempt: int,
        error_chain: list[str],
        technical_details: dict[str, Any] | None = None
    ):
        super().__init__(message, ErrorCategory.CODE_RELATED, technical_details)
        self.generation_attempt = generation_attempt
        self.error_chain = error_chain


class CodeSyntaxError(PythonExecutorException):
    """Generated code has syntax errors"""

    def __init__(
        self,
        message: str,
        syntax_issues: list[str],
        technical_details: dict[str, Any] | None = None
    ):
        super().__init__(message, ErrorCategory.CODE_RELATED, technical_details)
        self.syntax_issues = syntax_issues


class CodeRuntimeError(PythonExecutorException):
    """Code failed during execution due to runtime errors"""

    def __init__(
        self,
        message: str,
        traceback_info: str,
        execution_attempt: int,
        technical_details: dict[str, Any] | None = None,
        folder_path: Path | None = None
    ):
        super().__init__(message, ErrorCategory.CODE_RELATED, technical_details, folder_path)
        self.traceback_info = traceback_info
        self.execution_attempt = execution_attempt


class ChannelLimitsViolationError(PythonExecutorException):
    """Raised when a channel write violates configured limits.

    This code-related error indicates that generated or user code attempted to
    write a value to a channel that violates safety limits defined in the
    limits database. This includes min/max limit violations, read-only channel
    writes, excessive step sizes, or writes to unlisted channels.

    The error provides comprehensive details about the violation including the
    channel address, attempted value, current value (for step violations), and the
    configured limits to help users understand why the write was blocked.

    :param channel_address: Channel address that was accessed
    :type channel_address: str
    :param value: The value that was attempted to be written
    :type value: Any
    :param violation_type: Type of violation (MIN_EXCEEDED, MAX_EXCEEDED, READ_ONLY_CHANNEL,
                          UNLISTED_CHANNEL, MAX_STEP_EXCEEDED, STEP_CHECK_FAILED)
    :type violation_type: str
    :param violation_reason: Human-readable explanation of the violation
    :type violation_reason: str
    :param min_value: Configured minimum value for the channel
    :type min_value: float, optional
    :param max_value: Configured maximum value for the channel
    :type max_value: float, optional
    :param max_step: Configured maximum step size for the channel
    :type max_step: float, optional
    :param current_value: Current channel value (for step violations)
    :type current_value: Any, optional

    .. note::
       This error is raised during code execution when runtime channel limits
       checking is enabled and detects a safety violation.

    .. seealso::
       :class:`LimitsValidator` : Validation engine that raises this exception
       :class:`ChannelLimitsConfig` : Configuration for channel limits

    Examples:
        Handling channel limits violations::

            >>> try:
            ...     await executor.execute_code(code)
            ... except ChannelLimitsViolationError as e:
            ...     logger.error(f"Safety violation: {e.violation_reason}")
            ...     logger.error(f"Attempted to write {e.attempted_value} to {e.channel_address}")
            ...     # Code should be regenerated with safer values
    """

    def __init__(
        self,
        channel_address: str,
        value: Any,
        violation_type: str,
        violation_reason: str,
        min_value: float | None = None,
        max_value: float | None = None,
        max_step: float | None = None,
        current_value: Any | None = None
    ):
        self.channel_address = channel_address
        self.attempted_value = value
        self.violation_type = violation_type
        self.violation_reason = violation_reason
        self.min_value = min_value
        self.max_value = max_value
        self.max_step = max_step
        self.current_value = current_value

        message = self._format_violation_message()

        super().__init__(
            message=message,
            category=ErrorCategory.CODE_RELATED
        )

    def _format_violation_message(self) -> str:
        """Format a user-friendly violation message with all relevant details."""
        msg = [
            "\n" + "="*70,
            "CHANNEL LIMITS VIOLATION DETECTED",
            "="*70,
            f"Channel Address: {self.channel_address}",
            f"Attempted Value: {self.attempted_value}",
        ]

        # Include current value for step violations
        if self.current_value is not None:
            msg.append(f"Current Value: {self.current_value}")

        msg.append(f"Violation: {self.violation_reason}")

        # Show allowed range if available
        if self.min_value is not None or self.max_value is not None:
            msg.append(f"Allowed Range: [{self.min_value}, {self.max_value}]")

        # Show max step if available
        if self.max_step is not None:
            msg.append(f"Maximum Step Size: {self.max_step}")

        msg.extend([
            "="*70,
            "⚠️  Write operation BLOCKED for safety",
            "="*70,
        ])

        return "\n".join(msg)


# Backward compatibility aliases
PVBoundaryViolationError = ChannelLimitsViolationError
ChannelBoundaryViolationError = ChannelLimitsViolationError


# =============================================================================
# WORKFLOW ERRORS (Special Flow Control)
# =============================================================================



class ExecutionTimeoutError(PythonExecutorException):
    """Code execution exceeded timeout"""

    def __init__(
        self,
        timeout_seconds: int,
        technical_details: dict[str, Any] | None = None,
        folder_path: Path | None = None
    ):
        message = f"Python code execution timeout after {timeout_seconds} seconds"
        super().__init__(message, ErrorCategory.WORKFLOW, technical_details, folder_path)
        self.timeout_seconds = timeout_seconds


class MaxAttemptsExceededError(PythonExecutorException):
    """Maximum execution attempts exceeded"""

    def __init__(
        self,
        operation_type: str,  # "code_generation", "execution", "connectivity"
        max_attempts: int,
        error_chain: list[str],
        technical_details: dict[str, Any] | None = None,
        folder_path: Path | None = None
    ):
        message = f"Maximum {operation_type} attempts ({max_attempts}) exceeded"
        super().__init__(message, ErrorCategory.WORKFLOW, technical_details, folder_path)
        self.operation_type = operation_type
        self.max_attempts = max_attempts
        self.error_chain = error_chain


class WorkflowError(PythonExecutorException):
    """Unexpected workflow error (bugs in our code, not user code)"""

    def __init__(
        self,
        message: str,
        stage: str,  # "code_generation", "static_analysis", "execution", "orchestration"
        original_exception: Exception | None = None,
        technical_details: dict[str, Any] | None = None,
        folder_path: Path | None = None
    ):
        super().__init__(message, ErrorCategory.WORKFLOW, technical_details, folder_path)
        self.stage = stage
        self.original_exception = original_exception

    def get_user_message(self) -> str:
        return f"An unexpected error occurred in the Python executor during {self.stage}"
