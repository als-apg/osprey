"""Exception hierarchy for the Python executor service.

Exceptions are categorized by ErrorCategory to drive retry logic:
infrastructure errors retry execution, code errors trigger regeneration.
"""

from enum import Enum
from pathlib import Path
from typing import Any


class ErrorCategory(Enum):
    """Error categories that drive retry strategy selection."""

    INFRASTRUCTURE = "infrastructure"  # Container/connectivity issues
    CODE_RELATED = "code_related"  # Syntax/runtime/logic errors
    WORKFLOW = "workflow"  # Approval, timeout, etc.
    CONFIGURATION = "configuration"  # Config/setup issues


class PythonExecutorException(Exception):
    """Base exception for all Python executor failures.

    Subclasses set ``category`` to drive retry logic via
    ``should_retry_execution()`` and ``should_retry_code_generation()``.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        technical_details: dict[str, Any] | None = None,
        folder_path: Path | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.technical_details = technical_details or {}
        self.folder_path = folder_path

    def is_infrastructure_error(self) -> bool:
        """True if this is an infrastructure or connectivity error."""
        return self.category == ErrorCategory.INFRASTRUCTURE

    def is_code_error(self) -> bool:
        """True if this is a code-related error."""
        return self.category == ErrorCategory.CODE_RELATED

    def is_workflow_error(self) -> bool:
        """True if this is a workflow control error."""
        return self.category == ErrorCategory.WORKFLOW

    def should_retry_execution(self) -> bool:
        """True if the same code execution should be retried (infrastructure errors)."""
        return self.category == ErrorCategory.INFRASTRUCTURE

    def should_retry_code_generation(self) -> bool:
        """True if code should be regenerated and execution retried (code errors)."""
        return self.category == ErrorCategory.CODE_RELATED


# =============================================================================
# INFRASTRUCTURE ERRORS (Container/Connectivity Issues)
# =============================================================================


class ContainerConnectivityError(PythonExecutorException):
    """Raised when the Jupyter container is unreachable.

    Categorized as INFRASTRUCTURE so the same code is retried
    rather than regenerated.
    """

    def __init__(
        self, message: str, host: str, port: int, technical_details: dict[str, Any] | None = None
    ):
        super().__init__(message, ErrorCategory.INFRASTRUCTURE, technical_details)
        self.host = host
        self.port = port

    def get_user_message(self) -> str:
        """Return a user-friendly error message."""
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
        technical_details: dict[str, Any] | None = None,
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
        technical_details: dict[str, Any] | None = None,
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
        folder_path: Path | None = None,
    ):
        super().__init__(message, ErrorCategory.CODE_RELATED, technical_details, folder_path)
        self.traceback_info = traceback_info
        self.execution_attempt = execution_attempt


class ChannelLimitsViolationError(PythonExecutorException):
    """Raised when a channel write violates configured safety limits.

    Covers min/max range violations, read-only channel writes,
    excessive step sizes, and writes to unlisted channels.
    Categorized as CODE_RELATED so the code is regenerated with safer values.
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
        current_value: Any | None = None,
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

        super().__init__(message=message, category=ErrorCategory.CODE_RELATED)

    def _format_violation_message(self) -> str:
        """Format a user-friendly violation message with all relevant details."""
        msg = [
            "\n" + "=" * 70,
            "CHANNEL LIMITS VIOLATION DETECTED",
            "=" * 70,
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

        msg.extend(
            [
                "=" * 70,
                "⚠️  Write operation BLOCKED for safety",
                "=" * 70,
            ]
        )

        return "\n".join(msg)


# =============================================================================
# WORKFLOW ERRORS (Special Flow Control)
# =============================================================================


class ExecutionTimeoutError(PythonExecutorException):
    """Code execution exceeded timeout"""

    def __init__(
        self,
        timeout_seconds: int,
        technical_details: dict[str, Any] | None = None,
        folder_path: Path | None = None,
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
        folder_path: Path | None = None,
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
        folder_path: Path | None = None,
    ):
        super().__init__(message, ErrorCategory.WORKFLOW, technical_details, folder_path)
        self.stage = stage
        self.original_exception = original_exception

    def get_user_message(self) -> str:
        return f"An unexpected error occurred in the Python executor during {self.stage}"
