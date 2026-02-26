"""Error classification and handling for the Osprey Framework.

Defines severity levels (ErrorSeverity), structured classification results
(ErrorClassification), execution error containers (ExecutionError), and
the framework exception hierarchy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Error severity levels that determine recovery strategy.

    - CRITICAL: End execution immediately.
    - RETRIABLE: Retry the current step with the same parameters.
    - REPLANNING: Discard the plan and create a new one.
    - RECLASSIFICATION: Re-select capabilities for the task.
    - FATAL: System-level failure; raise immediately to prevent corruption.
    """

    CRITICAL = "critical"  # End execution
    RETRIABLE = "retriable"  # Retry execution step
    REPLANNING = "replanning"  # Replan the execution plan
    RECLASSIFICATION = "reclassification"  # Reclassify task capabilities
    FATAL = "fatal"  # System-level failure - raise exception immediately


@dataclass
class ErrorClassification:
    """Structured error classification result pairing a severity with context.

    The severity determines recovery strategy; user_message and metadata
    carry information for logging, debugging, and user-facing display.

    Example::

        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Invalid configuration detected",
            metadata={
                "technical_details": "Missing required parameter 'api_key'",
                "suggestions": ["Check configuration file", "Verify credentials"],
            }
        )
    """

    severity: ErrorSeverity
    user_message: str | None = None
    metadata: dict[str, Any] | None = None

    def format_for_llm(self) -> str:
        """Format this classification as a markdown snippet for LLM prompts.

        Returns:
            Structured markdown string with user message and metadata fields.
        """
        import json

        sections = [
            "**Previous Execution Error:**",
            f"- **User Message:** {self.user_message or 'No error message available'}",
        ]

        if self.metadata:
            for key, value in self.metadata.items():
                display_key = key.replace("_", " ").title()

                if isinstance(value, (list, tuple)):
                    formatted_value = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    try:
                        formatted_value = json.dumps(value, indent=2)
                    except (TypeError, ValueError):
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)

                sections.append(f"- **{display_key}:** {formatted_value}")

        return "\n".join(sections)


@dataclass
class ExecutionError:
    """Execution error container carrying severity, message, and optional metadata.

    Created by error-classification methods in capabilities and infrastructure
    nodes. The framework uses severity to select recovery strategy (retry,
    replan, terminate, etc.).

    Example::

        error = ExecutionError(
            severity=ErrorSeverity.RETRIABLE,
            message="Database connection failed",
            capability_name="database_query",
            metadata={"technical_details": "PostgreSQL timeout after 30s"}
        )
    """

    severity: ErrorSeverity
    message: str
    capability_name: str | None = None  # Which capability generated this error

    metadata: dict[str, Any] | None = None  # Structured error context and debugging information


# Framework-specific exception classes
class FrameworkError(Exception):
    """Base exception for all framework-related errors.

    This is the root exception class for all custom exceptions within the
    Osprey Framework. It provides a common base for framework-specific
    error handling and categorization.
    """

    pass


class RegistryError(FrameworkError):
    """Exception for registry-related errors.

    Raised when issues occur with component registration, lookup, or
    management within the framework's registry system.
    """

    pass


class ConfigurationError(FrameworkError):
    """Exception for configuration-related errors.

    Raised when configuration files are invalid, missing required settings,
    or contain incompatible values that prevent proper system operation.
    """

    pass


class ReclassificationRequiredError(FrameworkError):
    """Exception for cases where task reclassification is needed.

    Raised when the current capability selection is insufficient for the task
    and requires reclassification to select different or additional capabilities.
    This typically occurs when:
    - Orchestrator validation fails due to hallucinated capabilities
    - No active capabilities are found for the task
    - Task extraction fails to identify proper task requirements
    """

    pass


class InvalidContextKeyError(FrameworkError):
    """Exception for invalid context key references in execution plans.

    Raised when the orchestrator creates an execution plan where a step references
    a context key that doesn't exist (neither in existing context nor created by
    an earlier step). This triggers replanning (not reclassification) because the
    capability selection was correct - only the key references need fixing.

    This typically occurs when:
    - Orchestrator uses inconsistent naming between context_key and input references
    - A typo or permutation in the key name (e.g., 'horizontal_bpm' vs 'bpm_horizontal')
    - Step references a key that was never created

    The error message should include:
    - Which step has the invalid reference
    - What key was referenced
    - What keys are actually available
    """

    pass
