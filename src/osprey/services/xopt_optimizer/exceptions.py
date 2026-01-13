"""Exception Hierarchy for XOpt Optimizer Service.

This module defines a clean, categorized exception hierarchy that provides precise
error classification for all failure modes in the XOpt optimizer service. The
exceptions are designed to support intelligent retry logic, user-friendly error
reporting, and comprehensive debugging information.

Error Categories:
    - MACHINE_STATE: Machine not ready - may retry after delay
    - YAML_GENERATION: Code generation issues - retry with feedback
    - EXECUTION: XOpt runtime errors
    - CONFIGURATION: Invalid configuration
    - WORKFLOW: Service-level workflow issues
"""

from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    """Categorization of errors for retry logic."""

    MACHINE_STATE = "machine_state"  # Machine not ready - may retry after delay
    YAML_GENERATION = "yaml_generation"  # Code generation issues - retry with feedback
    EXECUTION = "execution"  # XOpt runtime errors
    CONFIGURATION = "configuration"  # Invalid configuration
    WORKFLOW = "workflow"  # Service-level workflow issues


class XOptExecutorException(Exception):
    """Base exception for all XOpt optimizer service errors.

    Provides categorization and structured error information for
    proper error handling and retry logic.

    :param message: Human-readable error description
    :param category: Error category for recovery strategy
    :param technical_details: Additional technical information for debugging
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.WORKFLOW,
        technical_details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.technical_details = technical_details or {}

    def is_retriable(self) -> bool:
        """Check if this error type typically warrants a retry."""
        return self.category in (ErrorCategory.MACHINE_STATE, ErrorCategory.YAML_GENERATION)

    def should_retry_yaml_generation(self) -> bool:
        """Check if YAML should be regenerated."""
        return self.category == ErrorCategory.YAML_GENERATION


class MachineStateAssessmentError(XOptExecutorException):
    """Failed to assess machine state.

    Raised when the state identification agent cannot determine
    machine readiness. May be retryable after addressing machine issues.

    :param message: Error description
    :param assessment_details: Details from the assessment attempt
    """

    def __init__(
        self,
        message: str,
        assessment_details: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.MACHINE_STATE, **kwargs)
        self.assessment_details = assessment_details or {}


class YamlGenerationError(XOptExecutorException):
    """Failed to generate valid XOpt YAML configuration.

    Raised when the YAML generation agent produces invalid configuration.
    Usually retryable with error feedback.

    :param message: Error description
    :param generated_yaml: The invalid YAML that was generated
    :param validation_errors: List of validation errors found
    """

    def __init__(
        self,
        message: str,
        generated_yaml: str | None = None,
        validation_errors: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.YAML_GENERATION, **kwargs)
        self.generated_yaml = generated_yaml
        self.validation_errors = validation_errors or []


class XOptExecutionError(XOptExecutorException):
    """XOpt execution failed at runtime.

    Raised when XOpt itself fails during execution.

    :param message: Error description
    :param yaml_used: The YAML configuration that was used
    :param xopt_error: The original XOpt error message
    """

    def __init__(
        self,
        message: str,
        yaml_used: str | None = None,
        xopt_error: str | None = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.EXECUTION, **kwargs)
        self.yaml_used = yaml_used
        self.xopt_error = xopt_error


class MaxIterationsExceededError(XOptExecutorException):
    """Maximum optimization iterations exceeded without convergence.

    :param message: Error description
    :param iterations_completed: Number of iterations that were completed
    """

    def __init__(self, message: str, iterations_completed: int = 0, **kwargs):
        super().__init__(message, category=ErrorCategory.WORKFLOW, **kwargs)
        self.iterations_completed = iterations_completed


class ConfigurationError(XOptExecutorException):
    """Invalid service configuration.

    :param message: Error description
    :param config_key: The configuration key that is invalid
    """

    def __init__(self, message: str, config_key: str | None = None, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)
        self.config_key = config_key
