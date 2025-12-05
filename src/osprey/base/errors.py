"""Error Classification Framework - Comprehensive Error Handling System

This module provides the complete error classification and handling infrastructure
for the Osprey Framework. It implements a sophisticated error management system
that enables intelligent error classification, recovery strategy selection, and
comprehensive error tracking throughout the entire framework ecosystem.

The error classification system serves as the foundation for robust error handling
by providing structured error analysis, severity classification, and recovery
strategy coordination. It integrates seamlessly with both capability execution
and infrastructure operations to ensure consistent error handling patterns.

Key Error Management Components:
    1. **ErrorSeverity Enum**: Defines classification levels and recovery strategies
    2. **ErrorClassification**: Structured error analysis with recovery information
    3. **ExecutionError**: Comprehensive error data for execution failures
    4. **Framework Exceptions**: Custom exception hierarchy for system errors

Error Classification Levels:
    - **CRITICAL**: End execution immediately - unrecoverable errors
    - **RETRIABLE**: Retry execution with same parameters - transient failures
    - **REPLANNING**: Create new execution plan - strategy failures
    - **RECLASSIFICATION**: Reclassify task capabilities
    - **FATAL**: System-level failure - immediate termination required

The error system integrates with LangGraph's execution model while providing
manual retry coordination through the router system. This ensures consistent
error handling behavior across all framework components while maintaining
compatibility with LangGraph's checkpoint and streaming systems.

.. note::
   The framework uses manual retry handling rather than LangGraph's native
   retry policies to ensure consistent behavior and sophisticated error
   classification across all components.

.. warning::
   FATAL errors immediately terminate execution to prevent system corruption.
   Use FATAL severity only for errors that indicate serious system issues.

.. seealso::
   :mod:`osprey.base.decorators` : Decorator integration with error handling
   :mod:`osprey.base.results` : Result types and execution tracking
   :class:`BaseCapability` : Capability error classification methods
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Enumeration of error severity levels with comprehensive recovery strategies.

    This enum defines the complete spectrum of error severity classifications
    and their corresponding recovery strategies used throughout the Osprey
    Framework. Each severity level triggers specific recovery behavior designed
    to maintain robust system operation while enabling intelligent error handling
    and graceful degradation.

    The severity levels form a hierarchy of recovery strategies from simple
    retries to complete execution termination. The framework's error handling
    system uses these classifications to coordinate recovery efforts between
    capabilities, infrastructure nodes, and the overall execution system.

    Recovery Strategy Hierarchy:
    1. **Automatic Recovery**: RETRIABLE errors with retry mechanisms
    2. **Strategy Adjustment**: REPLANNING for execution plan adaptation
    3. **Capability Adjustment**: RECLASSIFICATION for capability selection adaptation
    4. **Execution Control**: CRITICAL for graceful termination
    5. **System Protection**: FATAL for immediate termination

    :param CRITICAL: End execution immediately - unrecoverable errors requiring termination
    :type CRITICAL: str
    :param RETRIABLE: Retry current execution step with same parameters - transient failures
    :type RETRIABLE: str
    :param REPLANNING: Create new execution plan with different strategy - approach failures
    :type REPLANNING: str
    :param RECLASSIFICATION: Reclassify task to select different capabilities - selection failures
    :type RECLASSIFICATION: str
    :param FATAL: System-level failure requiring immediate termination - corruption prevention
    :type FATAL: str

    .. note::
       The framework uses manual retry coordination rather than automatic retries
       to ensure consistent behavior and sophisticated error analysis across all
       components.

    .. warning::
       FATAL errors immediately raise exceptions to terminate execution and prevent
       system corruption. Use FATAL only for errors that indicate serious system
       issues that could compromise framework integrity.

    Examples:
        Network error classification::

            if isinstance(exc, YourCustomConnectionError):
                return ErrorClassification(severity=ErrorSeverity.RETRIABLE, ...)
            elif isinstance(exc, YourCustomAuthenticationError):
                return ErrorClassification(severity=ErrorSeverity.CRITICAL, ...)

        Data validation error handling (example exception classes)::

            if isinstance(exc, ValidationError):
                return ErrorClassification(severity=ErrorSeverity.REPLANNING, ...)
            elif isinstance(exc, YourCustomCapabilityMismatchError):
                return ErrorClassification(severity=ErrorSeverity.RECLASSIFICATION, ...)
            elif isinstance(exc, YourCustomCorruptionError):
                return ErrorClassification(severity=ErrorSeverity.FATAL, ...)

        .. note::
           The exception classes in these examples (YourCustomCapabilityMismatchError,
           YourCustomCorruptionError) are not provided by the framework - they are
           examples of domain-specific exceptions you might implement in your capabilities.

    .. seealso::
       :class:`ErrorClassification` : Structured error analysis with severity
       :class:`ExecutionError` : Comprehensive error information container
    """

    CRITICAL = "critical"  # End execution
    RETRIABLE = "retriable"  # Retry execution step
    REPLANNING = "replanning"  # Replan the execution plan
    RECLASSIFICATION = "reclassification"  # Reclassify task capabilities
    FATAL = "fatal"  # System-level failure - raise exception immediately


@dataclass
class ErrorClassification:
    """Comprehensive error classification result with recovery strategy coordination.

    This dataclass provides sophisticated error classification results that enable
    intelligent recovery strategy selection and coordination across the framework.
    It serves as the primary interface between error analysis and recovery systems,
    supporting both automated recovery mechanisms and human-guided error resolution.

    ErrorClassification enables comprehensive error handling by providing:
    1. **Severity Assessment**: Clear classification of error impact and recovery strategy
    2. **User Communication**: Human-readable error descriptions for interfaces
    3. **Technical Context**: Detailed debugging information for developers
    4. **Extensible Metadata**: Additional context for capability-specific error handling

    The classification system supports multiple recovery approaches including
    automatic retries, execution replanning, and graceful degradation patterns.
    The severity field determines the recovery strategy while user_message and
    metadata provide contextual information for logging, debugging, and recovery guidance.

    :param severity: Error severity level determining recovery strategy
    :type severity: ErrorSeverity
    :param user_message: Human-readable error description for user interfaces and logs
    :type user_message: Optional[str]
    :param metadata: Structured error context including technical details and recovery hints
    :type metadata: Optional[Dict[str, Any]]

    .. note::
       The framework uses this classification to coordinate recovery strategies
       across multiple system components. Different severity levels trigger
       different recovery workflows through the router system.

    .. warning::
       Ensure severity levels are chosen carefully as they directly impact
       system behavior and recovery strategies. Inappropriate classifications
       can lead to ineffective error handling.

    Examples:
        Network timeout classification::

            classification = ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Network connection timeout, retrying...",
                metadata={"technical_details": "HTTP request timeout after 30 seconds"}
            )

        Missing step input requiring replanning::

            classification = ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message="Required data not available, need different approach",
                metadata={
                    "technical_details": "Step expected 'SENSOR_DATA' context but found None",
                    "replanning_reason": "Missing required input data"
                }
            )

        Wrong capability selected requiring reclassification::

            classification = ErrorClassification(
                severity=ErrorSeverity.RECLASSIFICATION,
                user_message="This capability cannot handle this type of request",
                metadata={
                    "technical_details": "Weather capability received machine operation request",
                    "reclassification_reason": "Capability mismatch detected"
                }
            )

        Comprehensive error with rich metadata::

            classification = ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Invalid configuration detected",
                metadata={
                    "technical_details": "Missing required parameter 'api_key' in capability config",
                    "safety_abort_reason": "Security validation failed",
                    "suggestions": ["Check configuration file", "Verify credentials"],
                    "error_code": "CONFIG_MISSING_KEY",
                    "retry_after": 30
                }
            )



    .. seealso::
       :class:`ErrorSeverity` : Severity levels and recovery strategies
       :class:`ExecutionError` : Complete error information container
    """

    severity: ErrorSeverity
    user_message: str | None = None
    metadata: dict[str, Any] | None = None

    def format_for_llm(self) -> str:
        """Format this error classification for LLM consumption during replanning.

        Converts the error classification into a structured, human-readable format
        optimized for LLM understanding and processing. Follows the framework's
        established format_for_llm() pattern for consistent formatting.

        :return: Formatted string optimized for LLM prompt inclusion
        :rtype: str

        Examples:
            Basic error formatting::

                classification = ErrorClassification(
                    severity=ErrorSeverity.REPLANNING,
                    user_message="Data not available",
                    metadata={"technical_details": "Missing sensor data"}
                )
                formatted = classification.format_for_llm()
                # Returns:
                # **Previous Execution Error:**
                # - **Failed Operation:** unknown operation
                # - **User Message:** Data not available
                # - **Technical Details:** Missing sensor data

        .. note::
           This method formats error classification data independently of the
           error_info dictionary structure, making it suitable for direct
           error classification formatting.
        """
        import json

        # Build basic error context sections
        sections = [
            "**Previous Execution Error:**",
            f"- **User Message:** {self.user_message or 'No error message available'}",
        ]

        # Add metadata if available
        if self.metadata:
            # Process all metadata keys generically
            for key, value in self.metadata.items():
                # Format key name for display (convert snake_case to Title Case)
                display_key = key.replace("_", " ").title()

                # Handle different value types appropriately
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
    """Comprehensive execution error container with recovery coordination support.

    This dataclass provides a complete representation of execution errors including
    severity classification, recovery suggestions, technical debugging information,
    and context for coordinating recovery strategies. It serves as the primary
    error data structure used throughout the framework for error handling,
    logging, and recovery coordination.

    ExecutionError enables sophisticated error management by providing:
    1. **Error Classification**: Severity-based recovery strategy determination
    2. **User Communication**: Clear, actionable error messages for interfaces
    3. **Developer Support**: Technical details and debugging context

    5. **System Integration**: Context for automated recovery systems

    The error structure supports both automated error handling workflows and
    human-guided error resolution processes. It integrates seamlessly with
    the framework's classification system and retry mechanisms to provide
    comprehensive error management.

    :param severity: Error severity classification for recovery strategy selection
    :type severity: ErrorSeverity
    :param message: Clear, human-readable description of the error condition
    :type message: str
    :param capability_name: Name of the capability or component that generated this error
    :type capability_name: Optional[str]

    :param metadata: Structured error context including technical details and debugging information
    :type metadata: Optional[Dict[str, Any]]


    .. note::
       ExecutionError instances are typically created by error classification
       methods in capabilities and infrastructure nodes. The framework's
       decorators automatically handle the creation and routing of these errors.

    .. warning::
       The severity field directly impacts system behavior through recovery
       strategy selection. Ensure appropriate severity classification to avoid
       ineffective error handling or unnecessary system termination.

    Examples:
        Database connection error::

            error = ExecutionError(
                severity=ErrorSeverity.RETRIABLE,
                message="Database connection failed",
                capability_name="database_query",

                metadata={"technical_details": "PostgreSQL connection timeout after 30 seconds"}
            )



        Data corruption requiring immediate attention::

            error = ExecutionError(
                severity=ErrorSeverity.FATAL,
                message="Critical data corruption detected",
                capability_name="data_processor",
                metadata={
                    "technical_details": "Checksum validation failed on primary data store",
                    "safety_abort_reason": "Data integrity compromised"
                },
                suggestions=[
                    "Initiate emergency backup procedures",
                    "Contact system administrator immediately",
                    "Do not proceed with further operations"
                ]
            )

    .. seealso::
       :class:`ErrorSeverity` : Severity levels and recovery strategies
       :class:`ErrorClassification` : Error analysis and classification system
       :class:`ExecutionResult` : Result containers with error integration
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
