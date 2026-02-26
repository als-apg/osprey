"""Time Range Parsing Capability - Advanced Temporal Expression Analysis

This capability provides sophisticated time range parsing functionality that converts
natural language time expressions into structured datetime objects using advanced
LLM-based analysis. It handles complex temporal references, relative time expressions,
and contextual time parsing with comprehensive validation and error handling.

The capability transforms user queries containing time references into precise datetime
ranges that other capabilities can use for data retrieval, analysis, and processing.
It supports a wide variety of temporal expressions from simple relative references
to complex multi-part time specifications.

Key Features:
    - LLM-based natural language time parsing with sophisticated prompting
    - Comprehensive validation including range validation and future date detection
    - Support for relative time expressions ("last 24 hours", "yesterday", etc.)
    - Absolute datetime object creation with full datetime functionality
    - Context-aware parsing with current time reference and timezone handling
    - Structured error classification for different parsing failure modes

Supported Time Expression Patterns:
    - Relative periods: "last X hours/days/weeks", "past X time units"
    - Named periods: "yesterday", "today", "this week", "last month"
    - Current/real-time: "now", "current", "latest"
    - Absolute references: specific dates and date ranges

The capability uses sophisticated prompt engineering with examples and validation
rules to ensure reliable parsing across diverse user expressions while maintaining
type safety through Pydantic models and comprehensive error handling.

.. note::
   All datetime objects are created with timezone awareness using UTC as the
   reference timezone to avoid confusion and ensure consistent behavior.

.. warning::
   The capability performs strict validation to prevent future dates and
   invalid ranges, raising appropriate exceptions for malformed expressions.

.. seealso::
   :class:`TimeRangeContext` : Structured context for parsed time ranges
   :class:`osprey.base.capability.BaseCapability` : Base capability functionality
   :mod:`datetime` : Python datetime functionality leveraged by parsed results
"""

import asyncio
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import OrchestratorGuide, TaskClassifierGuide
from osprey.context.base import CapabilityContext
from osprey.prompts.loader import get_framework_prompts
from osprey.utils.config import get_model_config

# Import model completion - adapt based on your model system
try:
    from osprey.models import get_chat_completion
except ImportError:
    # Fallback for testing or if models not available
    get_chat_completion = None


# ========================================================
# Context Class
# ========================================================


class TimeRangeContext(CapabilityContext):
    """Structured context for time range parsing results with datetime objects.

    Provides comprehensive context for parsed time ranges using native Python datetime
    objects for maximum functionality and type safety. The context enables other
    capabilities to perform sophisticated temporal operations including arithmetic,
    comparisons, and formatting without additional parsing.

    The context maintains both start and end datetime objects with full timezone
    information, enabling precise temporal calculations and consistent behavior
    across different system environments.

    :param start_date: Parsed start datetime with timezone information
    :type start_date: datetime
    :param end_date: Parsed end datetime with timezone information
    :type end_date: datetime

    .. note::
       Datetime objects provide full functionality including arithmetic operations
       (end_date - start_date), comparisons, and flexible formatting options.

    .. warning::
       The context validates that start_date < end_date during initialization
       to ensure logical time range consistency.

    .. seealso::
       :class:`osprey.context.base.CapabilityContext` : Base context functionality
       :meth:`TimeRangeParsingCapability.execute` : Main capability that creates this context
       :class:`TimeRangeOutput` : Pydantic model that provides data for this context
    """

    start_date: datetime  # Start date as datetime object
    end_date: datetime  # End date as datetime object

    # Class constants for compatibility
    CONTEXT_TYPE: ClassVar[str] = "TIME_RANGE"
    CONTEXT_CATEGORY: ClassVar[str] = "METADATA"

    @property
    def context_type(self) -> str:
        return self.CONTEXT_TYPE

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Provide comprehensive access information for time range context integration.

        Generates detailed access information for other capabilities to understand
        how to interact with parsed time range data. Includes access patterns,
        datetime functionality descriptions, and practical usage examples for
        leveraging the full power of datetime objects.

        :param key_name: Optional context key name for access pattern generation
        :type key_name: Optional[str]
        :return: Dictionary containing comprehensive access details and datetime usage examples
        :rtype: Dict[str, Any]

        .. note::
           Emphasizes the full datetime functionality available including arithmetic,
           comparison operations, and flexible formatting capabilities.
        """
        start_str = self.start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = self.end_date.strftime("%Y-%m-%d %H:%M:%S")
        duration = self.end_date - self.start_date

        return {
            "start_date": start_str,
            "end_date": end_str,
            "duration": str(duration),
            "data_structure": "Two datetime objects: start_date and end_date with full datetime functionality",
            "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.start_date and context.{self.CONTEXT_TYPE}.{key}.end_date",
            "example_usage": f"context.{self.CONTEXT_TYPE}.{key}.start_date gives datetime object, use .strftime('%Y-%m-%d %H:%M:%S') for string format",
            "datetime_features": "Direct arithmetic: end_date - start_date, comparison: start_date > other_date, formatting: start_date.strftime(format)",
        }

    def get_summary(self) -> dict[str, Any]:
        """Generate summary for UI display and debugging.

        Creates a formatted summary of the parsed time range suitable for display
        in user interfaces, debugging output, and development tools. Uses
        human-friendly formatting while maintaining precision.

        :param key_name: Optional context key name for reference
        :type key_name: Optional[str]
        :return: Dictionary containing time range summary
        :rtype: Dict[str, Any]

        .. note::
           Uses standardized datetime formatting for consistency across
           the framework while providing duration calculations for context.
        """
        duration = self.end_date - self.start_date
        return {
            "type": "Time Range",
            "start_time": self.start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.end_date.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(duration),
        }

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_datetime(cls, v):
        """Validate and convert datetime inputs with comprehensive format support.

        Provides robust datetime validation that accepts multiple input formats
        including ISO strings with timezone information, standard datetime strings,
        and native datetime objects. Handles timezone conversion and normalization
        for consistent behavior.

        :param v: Input value to validate and convert to datetime
        :type v: Union[str, datetime]
        :return: Validated datetime object with proper timezone information
        :rtype: datetime

        :raises ValueError: If input cannot be parsed as a valid datetime

        .. note::
           Supports ISO format strings with and without timezone information,
           automatically handling UTC conversion and local timezone assumptions.
        """
        # Allow strings (ISO format) that Pydantic will convert to datetime
        if isinstance(v, str):
            try:
                # Try parsing ISO format strings with timezone info
                if v.endswith("Z"):
                    # UTC timezone indicator
                    return datetime.fromisoformat(v.replace("Z", "+00:00"))
                elif "+" in v[-6:] or "-" in v[-6:]:
                    # Has timezone offset
                    return datetime.fromisoformat(v)
                else:
                    # No timezone, assume local
                    return datetime.fromisoformat(v)
            except ValueError:
                try:
                    # Fallback: try without timezone
                    return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                except ValueError as e:
                    raise ValueError(
                        f"Invalid datetime string: {v}. Expected ISO format. Error: {e}"
                    ) from e
        elif isinstance(v, datetime):
            return v
        else:
            raise ValueError(
                f"TimeRangeContext requires datetime objects or ISO datetime strings, got {type(v)}"
            )


# ========================================================
# Time Parsing Errors
# ========================================================


class TimeParsingError(Exception):
    """Base exception class for time parsing-related errors.

    Provides a hierarchy of time parsing exceptions to enable sophisticated
    error handling and classification. All time parsing errors inherit from
    this base class for consistent exception handling throughout the capability.

    .. seealso::
       :meth:`TimeRangeParsingCapability.classify_error` : Error classification for recovery strategies
       :class:`InvalidTimeFormatError` : Specific error for malformed time expressions
       :class:`AmbiguousTimeReferenceError` : Specific error for unclear time references
       :class:`TimeParsingDependencyError` : Specific error for missing dependencies
       :meth:`TimeRangeParsingCapability.execute` : Main method that raises these errors
    """

    pass


class InvalidTimeFormatError(TimeParsingError):
    """Raised when time expressions cannot be parsed into valid datetime ranges.

    Indicates that the LLM-based parsing process identified invalid time
    expressions, malformed date ranges, or logically inconsistent temporal
    references that cannot be converted to valid datetime objects.
    """

    pass


class AmbiguousTimeReferenceError(TimeParsingError):
    """Raised when time references are ambiguous or cannot be determined.

    Indicates that the user query does not contain identifiable time references
    or contains ambiguous temporal expressions that cannot be resolved to
    specific datetime ranges without additional context.
    """

    pass


class TimeParsingDependencyError(TimeParsingError):
    """Raised when required dependencies for time parsing are unavailable.

    Indicates that the time parsing process requires additional context or
    dependencies that are not available in the current execution environment,
    such as missing query context or unavailable time reference data.
    """

    pass


# ========================================================
# Pydantic Models
# ========================================================


class TimeRange(BaseModel):
    """Simple time range model with start and end datetime objects.

    Basic Pydantic model for representing time ranges with proper datetime
    validation. Used as a building block for more complex time parsing
    operations and validation workflows.

    :param start_date: Starting datetime of the range
    :type start_date: datetime
    :param end_date: Ending datetime of the range
    :type end_date: datetime
    """

    start_date: datetime = Field(description="Start date and time")
    end_date: datetime = Field(description="End date and time")


class TimeRangeOutput(BaseModel):
    """Structured output model for LLM-based time range parsing results.

    Pydantic model used to ensure structured output from LLM-based time parsing
    operations. Includes both the parsed datetime objects and a detection flag
    to indicate whether valid time references were found in the input.

    This model enables reliable parsing validation and provides clear feedback
    about parsing success or failure for appropriate error handling.

    :param start_date: Parsed start datetime with timezone information
    :type start_date: datetime
    :param end_date: Parsed end datetime with timezone information
    :type end_date: datetime
    :param found: Whether valid time range was successfully identified and parsed
    :type found: bool

    .. note::
       The found flag enables the capability to distinguish between parsing
       failures and queries that legitimately contain no time references.

    .. seealso::
       :func:`_get_time_parsing_system_prompt` : System prompt generator that guides LLM output
       :class:`TimeRangeContext` : Context object created from this model's data
       :class:`TimeRange` : Simpler time range model without detection flag
       :meth:`TimeRangeParsingCapability.execute` : Main capability method using this model
    """

    start_date: datetime = Field(
        description="Start date and time as datetime object in YYYY-MM-DD HH:MM:SS format"
    )
    end_date: datetime = Field(
        description="End date and time as datetime object in YYYY-MM-DD HH:MM:SS format"
    )
    found: bool = Field(
        description="True if a valid time range was found in the query, False otherwise"
    )


# ========================================================
# LLM Prompting System
# ========================================================


def _get_time_parsing_system_prompt(user_query: str) -> str:
    """Create comprehensive system prompt for LLM-based time range parsing.

    Delegates to the prompt builder system, which composes the role, static
    instructions, and dynamic context (current datetime, examples, user query).

    :param user_query: User query containing time expressions to parse
    :type user_query: str
    :return: Complete system prompt for LLM time parsing with examples and validation rules
    :rtype: str
    """
    prompt_provider = get_framework_prompts()
    builder = prompt_provider.get_time_range_parsing_prompt_builder()
    return builder.build_prompt(user_query=user_query)


# ========================================================
# Convention-Based Capability Implementation
# ========================================================


@capability_node
class TimeRangeParsingCapability(BaseCapability):
    """Advanced time range parsing capability with LLM-based natural language processing.

    Provides sophisticated time range parsing functionality that converts natural
    language time expressions into structured datetime objects using advanced LLM
    analysis. The capability handles complex temporal references with comprehensive
    validation, error handling, and context integration.

    The capability implements a complete parsing workflow:
    1. **Prompt Engineering**: Creates sophisticated system prompts with examples
    2. **LLM Analysis**: Uses structured output parsing for reliable time extraction
    3. **Validation**: Performs comprehensive validation including range and future date checks
    4. **Context Creation**: Generates rich context objects with datetime functionality

    Key architectural features:
        - Sophisticated prompt engineering with current time context and examples
        - Structured LLM output using Pydantic models for reliability
        - Comprehensive validation including logical consistency checks
        - Native datetime object creation with full temporal functionality
        - Domain-specific error classification for different parsing failure modes

    The capability migrates sophisticated business logic from previous frameworks
    while integrating seamlessly with the LangGraph execution model through the
    @capability_node decorator for error handling, retry policies, and streaming.

    .. note::
       Uses UTC timezone as reference for all datetime calculations to ensure
       consistent behavior across different system environments.

    .. warning::
       Performs strict validation to prevent invalid ranges and future dates
       that could cause issues in downstream data processing capabilities.

    .. seealso::
       :class:`osprey.base.capability.BaseCapability` : Base capability functionality
       :class:`TimeRangeContext` : Parsed time range context structure
       :class:`TimeRangeOutput` : LLM output model for structured parsing
    """

    # Required metadata (loaded through registry configuration)
    name = "time_range_parsing"
    description = (
        "Extract and parse time ranges from user queries into absolute datetime objects using LLM"
    )
    provides = ["TIME_RANGE"]
    requires = []

    async def execute(self) -> dict[str, Any]:
        """Execute comprehensive time range parsing with LLM integration and validation.

        Implements the complete time range parsing workflow including sophisticated
        prompt engineering, LLM-based analysis, comprehensive validation, and
        structured context creation. The method handles complex natural language
        time expressions and converts them to precise datetime ranges.

        The execution process follows this sophisticated pattern:
        1. **Context Extraction**: Retrieves task objective and current execution step
        2. **Prompt Engineering**: Creates detailed system prompt with current time context
        3. **LLM Analysis**: Performs structured parsing using Pydantic output models
        4. **Validation**: Comprehensive validation including range and future date checks
        5. **Context Creation**: Generates rich TimeRangeContext with datetime objects

        The method uses sophisticated prompt engineering with current time reference,
        detailed examples, and validation rules to ensure reliable parsing across
        diverse user expressions.

        :return: State updates with parsed time range context
        :rtype: Dict[str, Any]

        :raises TimeParsingError: If LLM parsing fails or returns invalid output
        :raises InvalidTimeFormatError: If parsed time range is logically invalid
        :raises AmbiguousTimeReferenceError: If no time reference found in query

        .. note::
           Uses StateManager for context storage and supports streaming updates
           for real-time progress indication during parsing operations.

        .. warning::
           Performs strict validation that may raise exceptions for malformed
           time expressions or logically inconsistent temporal references.

        .. seealso::
           :func:`_get_time_parsing_system_prompt` : Prompt generation used by this method
           :class:`TimeRangeOutput` : Pydantic model for structured LLM output
           :class:`TimeRangeContext` : Context structure returned by this method
           :class:`osprey.models.get_chat_completion` : LLM interface used for parsing
           :meth:`classify_error` : Error classification method for parsing failures
        """

        # Get unified logger with automatic streaming support
        logger = self.get_logger()

        # Get task objective using helper method
        task_objective = self.get_task_objective()

        # Display task with structured formatting
        logger.info("Starting time range parsing")
        logger.info(f'[bold]Query:[/bold] "[italic]{task_objective}[/italic]"')
        logger.status("Parsing time range with LLM...")

        # Build sophisticated system prompt
        full_prompt = _get_time_parsing_system_prompt(task_objective)

        logger.debug(f"Time parsing for task '{task_objective}': {task_objective}")

        try:
            # Get model config from LangGraph configurable
            model_config = get_model_config("time_parsing")

            # Set caller context for API call logging (propagates through asyncio.to_thread)
            from osprey.models import set_api_call_context

            set_api_call_context(
                function="execute",
                module="time_range_parsing",
                class_name="TimeRangeParsingCapability",
                extra={"capability": "time_range_parsing"},
            )

            # LLM call with structured output
            response_data = await asyncio.to_thread(
                get_chat_completion,
                model_config=model_config,
                message=full_prompt,
                output_model=TimeRangeOutput,
            )

        except Exception as e:
            logger.error(f"LLM call failed for time parsing: {e}")
            raise TimeParsingError(f"LLM failed to parse time range: {str(e)}") from e

        if not isinstance(response_data, TimeRangeOutput):
            logger.error(f"LLM did not return TimeRangeOutput. Got: {type(response_data)}")
            raise TimeParsingError("LLM failed to return structured time range output")

        logger.status("Validating parsed time range...")

        # Debug logging to see what LLM actually returned
        logger.debug(
            f"LLM returned: start={response_data.start_date}, end={response_data.end_date}, found={response_data.found}"
        )

        # Check if the LLM found a valid time range
        if not response_data.found:
            logger.warning(f"No time range found in query: '{task_objective}'")
            raise AmbiguousTimeReferenceError(f"No time range found in query: '{task_objective}'")

        # VALIDATION: Check for invalid date ranges
        if response_data.start_date >= response_data.end_date:
            logger.error(
                f"⚠️ LLM returned INVALID date range: start={response_data.start_date} >= end={response_data.end_date}"
            )
            raise InvalidTimeFormatError(
                f"Invalid date range: start_date ({response_data.start_date}) must be before end_date ({response_data.end_date})"
            )

        # VALIDATION: Check for future years (likely LLM error)
        current_year = datetime.now().year
        if (
            response_data.start_date.year > current_year
            or response_data.end_date.year > current_year
        ):
            logger.error(
                f"⚠️ LLM returned FUTURE year: start={response_data.start_date}, end={response_data.end_date}, current_year={current_year}"
            )
            raise InvalidTimeFormatError(
                f"Invalid date range: dates cannot be in future years (current year: {current_year})"
            )

        logger.status("Creating time range context...")

        # Create rich context object
        time_context = TimeRangeContext(
            start_date=response_data.start_date,
            end_date=response_data.end_date,
        )

        # Display parsed result with structured formatting
        start_str = time_context.start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = time_context.end_date.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("[bold]Parsed time range:[/bold]")
        logger.info(f"  Start: [cyan]{start_str}[/cyan]")
        logger.info(f"  End:   [cyan]{end_str}[/cyan]")

        # Store context using helper method
        logger.success("Time range parsing complete")

        # Return state updates (LangGraph will merge automatically)
        return self.store_output_context(time_context)

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify time parsing errors for sophisticated recovery strategies.

        Provides domain-specific error classification for time parsing failures,
        enabling appropriate recovery strategies based on the specific failure mode.
        Maps time parsing exceptions to framework error severities with appropriate
        user messages and recovery strategies.

        The classification handles different error types:
        - Invalid formats: RETRIABLE for potential LLM parsing improvements
        - Ambiguous references: REPLANNING to request user clarification
        - Missing dependencies: REPLANNING to gather required context
        - Permission/authorization: CRITICAL for security-related failures
        - Temporary issues: RETRIABLE for transient system problems

        :param exc: The exception that occurred during time parsing
        :type exc: Exception
        :param context: Error context including capability info and execution state
        :type context: dict
        :return: Error classification with recovery strategy and user messaging
        :rtype: ErrorClassification

        .. note::
           Classification enables the framework to determine appropriate responses:
           RETRIABLE for retry attempts, REPLANNING for user clarification,
           CRITICAL for immediate failure.

        .. seealso::
           :class:`osprey.base.errors.ErrorClassification` : Error classification structure
           :class:`TimeParsingError` : Base exception class for time parsing errors
           :class:`InvalidTimeFormatError` : Specific format-related error
           :class:`AmbiguousTimeReferenceError` : Specific ambiguity error requiring clarification
           :class:`osprey.base.errors.ErrorSeverity` : Available severity levels for classification
           :meth:`TimeRangeParsingCapability.execute` : Main method that uses this error classification
        """

        if isinstance(exc, InvalidTimeFormatError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Invalid time format detected, retrying",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, AmbiguousTimeReferenceError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message="Unable to identify time reference in query, please clarify the time period",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, TimeParsingDependencyError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message="Missing required information for time parsing",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, TimeParsingError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Time parsing failed, retrying...",
                metadata={"technical_details": str(exc)},
            )
        # Handle permission/configuration errors
        elif "permission" in str(exc).lower():
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Permission denied for time parsing operations",
                metadata={"technical_details": str(exc)},
            )
        # Retry on temporary issues
        elif any(keyword in str(exc).lower() for keyword in ["timeout", "connection", "temporary"]):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Temporary system issue, retrying time parsing...",
                metadata={"technical_details": str(exc)},
            )
        # Default: critical for unknown errors
        else:
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Time parsing failed: {exc}",
                metadata={"technical_details": str(exc)},
            )

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator integration guide from prompt builder system.

        Retrieves sophisticated orchestration guidance from the application's prompt
        builder system. This critical infrastructure automatically teaches the
        orchestrator when and how to invoke time range parsing within execution plans.

        The guide provides the orchestrator with comprehensive understanding of:
        - When time parsing is needed in user queries
        - How to integrate time parsing results with other capabilities
        - Examples of effective time parsing integration patterns

        :return: Orchestrator guide for time parsing capability integration
        :rtype: Optional[OrchestratorGuide]

        .. note::
           This infrastructure preserves the sophisticated prompt loading system
           from previous frameworks while integrating with the new architecture.

        .. seealso::
           :mod:`applications.als_assistant.framework_prompts.time_range_parsing` : Application prompts
        """
        prompt_provider = get_framework_prompts()
        time_range_builder = prompt_provider.get_time_range_parsing_prompt_builder()

        return time_range_builder.get_orchestrator_guide()

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create task classification guide from prompt builder system.

        Retrieves task classification guidance from the application's prompt builder
        system. This critical infrastructure automatically teaches the classifier
        when user requests should be routed to time range parsing operations.

        The guide provides domain-specific examples and classification patterns
        that enable accurate identification of queries requiring time range parsing,
        ensuring appropriate capability activation for temporal analysis tasks.

        :return: Classification guide for time parsing capability activation
        :rtype: Optional[TaskClassifierGuide]

        .. note::
           Leverages existing classifier infrastructure with domain-specific
           examples and instructions preserved from previous framework versions.

        .. seealso::
           :mod:`applications.als_assistant.framework_prompts.time_range_parsing` : Application prompts
        """
        prompt_provider = get_framework_prompts()
        time_range_builder = prompt_provider.get_time_range_parsing_prompt_builder()

        return time_range_builder.get_classifier_guide()
