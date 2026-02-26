"""Error classification and handling for the Osprey Framework.

Defines severity levels (ErrorSeverity), structured classification results
(ErrorClassification), and the framework exception hierarchy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Error severity levels that determine recovery strategy.

    - CRITICAL: End execution immediately.
    - RETRIABLE: Retry the current step with the same parameters.
    - FATAL: System-level failure; raise immediately to prevent corruption.
    """

    CRITICAL = "critical"
    RETRIABLE = "retriable"
    FATAL = "fatal"


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
