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
   All datetime objects are created with timezone awareness. The system supports
   both UTC mode (default) and local timezone mode (configurable via system.time_parsing_local).
   Internal storage uses UTC for consistency, with local timezone conversion for display.

.. warning::
   The capability performs strict validation to prevent future dates and
   invalid ranges, raising appropriate exceptions for malformed expressions.

.. seealso::
   :class:`TimeRangeContext` : Structured context for parsed time ranges
   :class:`osprey.base.capability.BaseCapability` : Base capability functionality
   :mod:`datetime` : Python datetime functionality leveraged by parsed results
"""

import asyncio
import re
import textwrap
import threading
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import OrchestratorGuide, TaskClassifierGuide
from osprey.context.base import CapabilityContext
from osprey.prompts.loader import get_framework_prompts
from osprey.utils.config import get_config_value, get_model_config

# ========================================================
# Module Constants
# ========================================================

# Clock skew buffer for future date validation (in minutes)
CLOCK_SKEW_BUFFER_MINUTES: int = 5

# Maximum length for user queries to prevent token exhaustion
MAX_USER_QUERY_LENGTH: int = 1000

# Timezone offset suffix length for detection
TIMEZONE_OFFSET_SUFFIX_LENGTH: int = 6

# ========================================================
# Public API
# ========================================================

__all__ = [
    "TimeRangeParsingCapability",
    "TimeRangeContext",
    "TimeRangeOutput",
    "TimeRange",
    "TimeParsingError",
    "InvalidTimeFormatError",
    "AmbiguousTimeReferenceError",
    "TimeParsingDependencyError",
    "TimezoneConfig",
]

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

    @model_validator(mode="after")
    def validate_date_order(self) -> "TimeRangeContext":
        """Validate that start_date is before end_date.

        :raises ValueError: If start_date is not before end_date
        :return: The validated context instance
        :rtype: TimeRangeContext
        """
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
            )
        return self

    @property
    def context_type(self) -> str:
        return self.CONTEXT_TYPE

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Provide comprehensive access information for time range context integration.

        Generates detailed access information for other capabilities to understand
        how to interact with parsed time range data. Includes access patterns,
        datetime functionality descriptions, and practical usage examples for
        leveraging the full power of datetime objects.

        :param key: Context key name for access pattern generation
        :type key: str
        :return: Dictionary containing comprehensive access details and datetime usage examples
        :rtype: dict[str, Any]

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

        :return: Dictionary containing time range summary with type, start_time,
            end_time, and duration fields
        :rtype: dict[str, Any]

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
                elif (
                    "+" in v[-TIMEZONE_OFFSET_SUFFIX_LENGTH:]
                    or "-" in v[-TIMEZONE_OFFSET_SUFFIX_LENGTH:]
                ):
                    # Has timezone offset
                    return datetime.fromisoformat(v)
                else:
                    # No timezone, assume UTC for consistency
                    dt = datetime.fromisoformat(v)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    return dt
            except ValueError:
                try:
                    # Fallback: try without timezone, assume UTC
                    dt = datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                    return dt.replace(tzinfo=UTC)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid datetime string: {v}. Expected ISO format. Error: {e}"
                    ) from e
        elif isinstance(v, datetime):
            # Ensure datetime has timezone info
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
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
# Helper Functions
# ========================================================


class TimezoneConfig:
    """Configuration holder for timezone settings.

    Caches timezone configuration to avoid repeated environment variable
    and config file reads during a single execution. Thread-safe singleton
    implementation using double-checked locking pattern.
    """

    _instance: "TimezoneConfig | None" = None
    _lock: threading.Lock = threading.Lock()
    _local_tz: ZoneInfo | timezone | None = None
    _local_tz_name: str | None = None
    _time_parsing_local: bool | None = None

    @classmethod
    def get_instance(cls) -> "TimezoneConfig":
        """Get or create the singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._initialize()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None
            cls._local_tz = None
            cls._local_tz_name = None
            cls._time_parsing_local = None

    def _initialize(self) -> None:
        """Initialize timezone configuration from config file."""
        # Get local time parsing preference from config (single source of truth)
        self._time_parsing_local = get_config_value("system.time_parsing_local", False)

        if self._time_parsing_local:
            # Get timezone from config.yml
            tz_name = get_config_value("system.timezone", None)

            if not tz_name:
                # If timezone not configured, use system's local timezone
                now_local = datetime.now().astimezone()
                self._local_tz = now_local.tzinfo
                self._local_tz_name = now_local.tzname() or str(self._local_tz)
            else:
                # Use configured timezone from config.yml with error handling
                try:
                    self._local_tz = ZoneInfo(tz_name)
                    now_local = datetime.now(self._local_tz)
                    self._local_tz_name = now_local.tzname() or tz_name
                except KeyError as e:
                    raise ValueError(
                        f"Invalid timezone '{tz_name}' in config. "
                        f"Use a valid IANA timezone name (e.g., 'America/Chicago'). "
                        f"Error: {e}"
                    ) from e
        else:
            # Use UTC timezone (original behavior)
            self._local_tz = UTC
            self._local_tz_name = "UTC"

    @property
    def local_tz(self) -> ZoneInfo | timezone:
        """Get the configured local timezone."""
        return self._local_tz

    @property
    def local_tz_name(self) -> str:
        """Get the configured local timezone name."""
        return self._local_tz_name

    @property
    def time_parsing_local(self) -> bool:
        """Check if local time parsing is enabled."""
        return self._time_parsing_local

    def get_current_time(self) -> datetime:
        """Get current time in the configured timezone."""
        if self._time_parsing_local:
            return datetime.now(self._local_tz)
        return datetime.now(UTC)


def _sanitize_user_query(query: str) -> str:
    """Sanitize user query to prevent prompt injection attacks.

    Removes potentially dangerous patterns and limits query length to prevent
    token exhaustion and prompt manipulation.

    :param query: Raw user query string
    :type query: str
    :return: Sanitized query string
    :rtype: str

    .. note::
       This is a defense-in-depth measure. The LLM should also be configured
       with appropriate safety measures.
    """
    # Remove potential prompt manipulation patterns
    sanitized = query.replace("```", "")
    sanitized = sanitized.replace("---", "")
    # Remove potential instruction injection patterns
    sanitized = re.sub(r"(?i)(ignore|forget|disregard)\s+(previous|above|all)", "", sanitized)
    # Limit length to prevent token exhaustion
    return sanitized[:MAX_USER_QUERY_LENGTH]


def _get_timezone_info() -> tuple[ZoneInfo | timezone, str, datetime, bool]:
    """Get timezone information for time parsing.

    Returns a tuple of (local_tz, local_tz_name, now_local, time_parsing_local)
    based on configuration. Uses ZoneInfo for proper DST handling when
    time_parsing_local=true.

    :return: Tuple of (local_tz, local_tz_name, now_local, time_parsing_local)
    :rtype: tuple[ZoneInfo | timezone, str, datetime, bool]

    :raises ValueError: If configured timezone name is invalid
    """
    config = TimezoneConfig.get_instance()
    now_local = config.get_current_time()
    return config.local_tz, config.local_tz_name, now_local, config.time_parsing_local


# ========================================================
# LLM Prompting System
# ========================================================


def _get_time_parsing_system_prompt(user_query: str) -> str:
    """Create comprehensive system prompt for LLM-based time range parsing.

    Constructs a sophisticated system prompt that provides the LLM with complete
    context for accurate time parsing including current time reference, timezone
    information, common patterns, validation rules, and detailed examples.

    The prompt includes critical validation rules to prevent common parsing errors
    such as future dates, invalid ranges, and incorrect relative time calculations.
    It provides step-by-step calculation examples to ensure consistent parsing behavior.

    :param user_query: User query containing time expressions to parse
    :type user_query: str
    :return: Complete system prompt for LLM time parsing with examples and validation rules
    :rtype: str

    .. note::
       Uses configured timezone (UTC or local) as reference and includes extensive
       examples with current time calculations to ensure accurate relative time parsing.
       The timezone is determined by the system.time_parsing_local configuration setting.

    .. warning::
       Includes critical validation rules to prevent future dates and
       invalid ranges that could cause downstream processing errors.

    .. seealso::
       :class:`TimeRangeOutput` : Pydantic model that structures the LLM response
       :meth:`TimeRangeParsingCapability.execute` : Main capability method using this prompt
       :mod:`datetime` : Python module providing time calculation functionality
       :class:`TimeRangeContext` : Final context object created from parsed results
    """
    # Sanitize user query to prevent prompt injection
    safe_query = _sanitize_user_query(user_query)

    # Get timezone information using helper function
    local_tz, local_tz_name, now_local, time_parsing_local = _get_timezone_info()

    current_time_str = now_local.strftime("%Y-%m-%d %H:%M:%S")
    current_weekday = now_local.strftime("%A")

    # Calculate example dates for the prompt (in local time)
    two_hours_ago = (now_local - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    yesterday_start = (now_local - timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
    yesterday_end = (now_local - timedelta(days=1)).strftime("%Y-%m-%d") + " 23:59:59"
    twenty_four_hours_ago = (now_local - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    two_weeks_ago = (now_local - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")

    prompt = textwrap.dedent(
        f"""
        You are an expert time range parser. Your task is to extract time ranges from user queries and convert them to absolute datetime values.

        Current time context:
        - Current datetime (local): {current_time_str}
        - Current weekday: {current_weekday}
        - Local timezone: {local_tz_name}

        CRITICAL REQUIREMENTS FOR DATE INTERPRETATION:
        - **IMPORTANT**: When users specify dates like "2025-10-10 to 2025-10-12", interpret these as LOCAL TIME ({local_tz_name}), NOT UTC
        - **IMPORTANT**: Specific dates without times should use 00:00:00 as the start time and 23:59:59 as the end time IN LOCAL TIME
        - start_date and end_date must be valid datetime values in ISO format
        - Use format 'YYYY-MM-DD HH:MM:SS' (e.g., "{current_time_str}")
        - Return as datetime objects, not strings with extra text or descriptions
        - **CRITICAL**: start_date MUST be BEFORE end_date (start < end)
        - **CRITICAL**: Dates must be in the current year ({now_local.year}) or earlier - NO future years beyond {now_local.year}
        - **CRITICAL**: Current time is {current_time_str} ({local_tz_name}) - use this as reference
        - For historical data requests, end_date should typically be close to current time

        Instructions:
        1. Parse the user query to identify time range references
        2. Convert relative time references to absolute datetime values IN LOCAL TIME ({local_tz_name})
        3. For specific dates (e.g., "2025-10-10 to 2025-10-12"), interpret as LOCAL TIME dates
        4. Set found=true if you can identify a time range, found=false if no time reference exists
        5. If found=false, use current time for both start_date and end_date as placeholders

        Common patterns and their conversions (all in LOCAL TIME {local_tz_name}):
        - "last X hours/minutes/days" → X time units BEFORE current time to NOW
        - "past X hours/minutes/days" → X time units BEFORE current time to NOW
        - "yesterday" → previous day from 00:00:00 to 23:59:59 (LOCAL TIME)
        - "today" → current day from 00:00:00 to current time (LOCAL TIME)
        - "this week" → from start of current week to now (LOCAL TIME)
        - "last week" → previous week (Monday to Sunday, LOCAL TIME)
        - Specific date ranges like "2025-10-10 to 2025-10-12" → interpret as LOCAL TIME dates
        - Current/real-time requests → very recent time (last few minutes)

        CRITICAL CALCULATION RULES FOR RELATIVE TIMES:
        **STEP-BY-STEP for "past/last X hours":**
        1. start_date = current_time MINUS X hours (earlier time)
        2. end_date = current_time (later time)
        3. Verify: start_date < end_date

        **STEP-BY-STEP for "past/last X days":**
        1. start_date = current_time MINUS X days (earlier time)
        2. end_date = current_time (later time)
        3. Verify: start_date < end_date

        **EXAMPLE CALCULATION for "past 24 hours" when current time is {current_time_str}:**
        1. start_date = {current_time_str} - 24 hours = {twenty_four_hours_ago}
        2. end_date = {current_time_str}
        3. Check: {twenty_four_hours_ago} < {current_time_str} ✓

        EXAMPLES with exact format expected:
        - "last 2 hours" → start_date: "{two_hours_ago}", end_date: "{current_time_str}"
        - "yesterday" → start_date: "{yesterday_start}", end_date: "{yesterday_end}"
        - "last 24 hours" → start_date: "{twenty_four_hours_ago}", end_date: "{current_time_str}"
        - "past 24 hours" → start_date: "{twenty_four_hours_ago}", end_date: "{current_time_str}"
        - "past 2 weeks" → start_date: "{two_weeks_ago}", end_date: "{current_time_str}"

        TIMEZONE HANDLING:
        - Return all dates in LOCAL TIME ({local_tz_name}) using format 'YYYY-MM-DD HH:MM:SS'
        - The system will handle conversion to UTC internally for archiver queries
        - DO NOT convert to UTC yourself - return dates in the user's local timezone

        Respond with a JSON object containing start_date, end_date, and found.
        The start_date and end_date fields should be datetime values in YYYY-MM-DD HH:MM:SS format
        in LOCAL TIME ({local_tz_name}) that will be automatically converted to Python datetime objects.

        User query to parse: {safe_query}"""
    )

    return prompt


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
       Supports both UTC mode (default) and local timezone mode (configurable).
       When local timezone mode is enabled, the LLM parses times in the configured
       local timezone, which are then converted to UTC for internal storage and
       archiver compatibility. This ensures consistent behavior while supporting
       user-friendly local time references.

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

        # Check if LLM interface is available
        if get_chat_completion is None:
            raise TimeParsingDependencyError(
                "LLM model interface not available. Ensure osprey.models is installed."
            )

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

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Network error during LLM call for time parsing: {e}")
            raise TimeParsingError(f"Network error during time parsing: {str(e)}") from e
        except ValueError as e:
            logger.error(f"Invalid response from LLM for time parsing: {e}")
            raise TimeParsingError(f"Invalid LLM response: {str(e)}") from e
        except Exception as e:
            # Log unexpected errors with full traceback for debugging
            logger.exception(f"Unexpected error in LLM call for time parsing: {e}")
            raise TimeParsingError(f"LLM failed to parse time range: {str(e)}") from e

        if not isinstance(response_data, TimeRangeOutput):
            logger.error(f"LLM did not return TimeRangeOutput. Got: {type(response_data)}")
            raise TimeParsingError("LLM failed to return structured time range output")

        logger.status("Validating parsed time range...")

        # Get timezone information using helper function (includes time_parsing_local flag)
        local_tz, local_tz_name, now_local, time_parsing_local = _get_timezone_info()

        # Debug logging to see what LLM actually returned
        logger.debug(
            f"LLM returned (local time): start={response_data.start_date}, end={response_data.end_date}, found={response_data.found}"
        )

        # Check if the LLM found a valid time range
        if not response_data.found:
            logger.warning(f"No time range found in query: '{task_objective}'")
            raise AmbiguousTimeReferenceError(f"No time range found in query: '{task_objective}'")

        # Convert LLM response to timezone-aware datetimes and then to UTC
        start_local, end_local, start_utc, end_utc = self._convert_to_utc(
            response_data.start_date,
            response_data.end_date,
            local_tz,
            time_parsing_local,
        )

        logger.debug(f"Converted to UTC: start={start_utc}, end={end_utc}")

        # Get current time in UTC for validation (use consistent reference)
        now_utc = now_local.astimezone(UTC)

        # VALIDATION: Check for invalid date ranges
        time_diff = (end_utc - start_utc).total_seconds()

        if time_diff < 0:
            # Start is after end - invalid range
            logger.error(
                f"⚠️ LLM returned INVALID date range: start={start_utc} is "
                f"{abs(time_diff) / 3600:.1f} hours after end={end_utc}"
            )
            raise InvalidTimeFormatError(
                f"Invalid date range: start_date ({start_local}) must be before "
                f"end_date ({end_local})"
            )
        elif time_diff == 0:
            # Exact same time - this is invalid for a time range
            logger.error(
                f"⚠️ Start and end times are identical: {start_utc}. "
                "A valid time range requires different start and end times."
            )
            raise InvalidTimeFormatError(
                f"Invalid date range: start_date and end_date cannot be identical ({start_local})"
            )

        # VALIDATION: Check for future years (likely LLM error)
        # Use the same timezone reference (now_local) for consistency
        current_year = now_local.year
        if start_local.year > current_year or end_local.year > current_year:
            logger.error(
                f"⚠️ LLM returned FUTURE year: start={start_local}, end={end_local}, "
                f"current_year={current_year}"
            )
            raise InvalidTimeFormatError(
                f"Invalid date range: dates cannot be in future years "
                f"(current year: {current_year})"
            )

        # VALIDATION: Check for end date in the future
        # Allow a small buffer for clock skew and processing time
        future_buffer = timedelta(minutes=CLOCK_SKEW_BUFFER_MINUTES)
        if end_utc > now_utc + future_buffer:
            logger.error(f"⚠️ LLM returned FUTURE end date: end={end_utc}, now={now_utc}")
            raise InvalidTimeFormatError(
                f"Invalid date range: end_date ({end_local}) cannot be in the future"
            )

        logger.status("Creating time range context...")

        # Create rich context object with UTC times (for archiver compatibility)
        time_context = TimeRangeContext(
            start_date=start_utc,
            end_date=end_utc,
        )

        # Log and return the result
        self._log_parsed_result(
            time_context, start_local, end_local, local_tz_name, time_parsing_local, logger
        )

        logger.success("Time range parsing complete")

        # Return state updates (LangGraph will merge automatically)
        return self.store_output_context(time_context)

    def _log_parsed_result(
        self,
        time_context: TimeRangeContext,
        start_local: datetime,
        end_local: datetime,
        local_tz_name: str,
        time_parsing_local: bool,
        logger,
    ) -> None:
        """Log the parsed time range result with structured formatting.

        :param time_context: The parsed time range context
        :type time_context: TimeRangeContext
        :param start_local: Start datetime in local timezone
        :type start_local: datetime
        :param end_local: End datetime in local timezone
        :type end_local: datetime
        :param local_tz_name: Name of the local timezone
        :type local_tz_name: str
        :param time_parsing_local: Whether local time parsing is enabled
        :type time_parsing_local: bool
        :param logger: Logger instance for output
        """
        if time_parsing_local:
            # Show both local and UTC when local parsing is enabled
            # Get timezone name from the actual parsed dates (important for DST handling)
            tz_name = start_local.tzname() or local_tz_name

            start_str_local = start_local.strftime("%Y-%m-%d %H:%M:%S")
            end_str_local = end_local.strftime("%Y-%m-%d %H:%M:%S")
            start_str_utc = time_context.start_date.strftime("%Y-%m-%d %H:%M:%S")
            end_str_utc = time_context.end_date.strftime("%Y-%m-%d %H:%M:%S")

            logger.info("[bold]Parsed time range:[/bold]")
            logger.info(f"  Start (local {tz_name}): [cyan]{start_str_local}[/cyan]")
            logger.info(f"  End (local {tz_name}):   [cyan]{end_str_local}[/cyan]")
            logger.info(f"  Start (UTC): [dim]{start_str_utc}[/dim]")
            logger.info(f"  End (UTC):   [dim]{end_str_utc}[/dim]")
        else:
            # Show only UTC when local parsing is disabled (original behavior)
            start_str_utc = time_context.start_date.strftime("%Y-%m-%d %H:%M:%S")
            end_str_utc = time_context.end_date.strftime("%Y-%m-%d %H:%M:%S")

            logger.info("[bold]Parsed time range:[/bold]")
            logger.info(f"  Start: [cyan]{start_str_utc}[/cyan]")
            logger.info(f"  End:   [cyan]{end_str_utc}[/cyan]")

    @staticmethod
    def _localize_naive_datetime(naive_dt: datetime, tz: ZoneInfo | timezone) -> datetime:
        """Convert a naive datetime to a timezone-aware datetime with proper DST handling.

        For ZoneInfo timezones, this method properly handles DST transitions by
        using the fold attribute to disambiguate times that occur twice during
        fall-back transitions. For times that don't exist during spring-forward,
        the behavior follows Python's standard: the time is treated as if DST
        is not in effect.

        :param naive_dt: Naive datetime object without timezone info
        :type naive_dt: datetime
        :param tz: Timezone to apply (ZoneInfo or UTC)
        :type tz: ZoneInfo | type[UTC]
        :return: Timezone-aware datetime object with correct UTC offset
        :rtype: datetime

        .. note::
           For ambiguous times during DST fall-back (e.g., 1:30 AM occurs twice),
           this method defaults to the first occurrence (fold=0, DST in effect).
           For non-existent times during DST spring-forward (e.g., 2:30 AM doesn't
           exist), the time is interpreted as if DST is not yet in effect.

        .. warning::
           Using datetime(..., tzinfo=tz) directly does NOT properly handle DST
           for ZoneInfo. However, for UTC (which has no DST), it works correctly.
        """
        if tz is UTC:
            # UTC has no DST, so simple replacement is fine
            return naive_dt.replace(tzinfo=UTC)

        # For ZoneInfo timezones, we need to handle DST properly
        # Create a naive datetime and use fold=0 (first occurrence for ambiguous times)
        # Then convert to the target timezone
        aware_dt = naive_dt.replace(tzinfo=tz, fold=0)

        # Verify the conversion is valid by round-tripping through UTC
        # This handles the case where the local time doesn't exist (spring forward)
        utc_dt = aware_dt.astimezone(UTC)
        back_to_local = utc_dt.astimezone(tz)

        # If the hour changed, we hit a non-existent time (spring forward gap)
        # In this case, we adjust to the valid time after the gap
        if back_to_local.hour != naive_dt.hour:
            # The time doesn't exist, use the post-gap time
            return back_to_local

        return aware_dt

    def _convert_to_utc(
        self,
        start_date: datetime,
        end_date: datetime,
        local_tz,
        time_parsing_local: bool,
    ) -> tuple[datetime, datetime, datetime, datetime]:
        """Convert parsed datetime objects to UTC with proper timezone handling.

        Handles the conversion of LLM-returned datetime objects to both local
        timezone-aware and UTC representations. Properly handles DST transitions
        by using the _localize_naive_datetime() method for proper fold handling.

        When time_parsing_local is True, the LLM returns naive datetime objects
        representing local time, which are then localized to the configured timezone
        and converted to UTC. When False, the LLM returns UTC times directly.

        :param start_date: Start datetime from LLM (may be naive or aware)
        :type start_date: datetime
        :param end_date: End datetime from LLM (may be naive or aware)
        :type end_date: datetime
        :param local_tz: Local timezone object (ZoneInfo or UTC)
        :type local_tz: ZoneInfo | timezone
        :param time_parsing_local: Whether local time parsing is enabled
        :type time_parsing_local: bool
        :return: Tuple of (start_local, end_local, start_utc, end_utc) where
            start_local and end_local are timezone-aware in the local timezone,
            and start_utc and end_utc are timezone-aware in UTC
        :rtype: tuple[datetime, datetime, datetime, datetime]

        **Usage Examples:**

        Example 1: Local time parsing enabled (America/Chicago)
            >>> from datetime import datetime
            >>> from zoneinfo import ZoneInfo
            >>> chicago_tz = ZoneInfo("America/Chicago")
            >>> # LLM returns naive datetime representing local time
            >>> start_naive = datetime(2025, 1, 15, 14, 0, 0)  # 2 PM Chicago
            >>> end_naive = datetime(2025, 1, 15, 16, 0, 0)    # 4 PM Chicago
            >>> start_local, end_local, start_utc, end_utc = self._convert_to_utc(
            ...     start_naive, end_naive, chicago_tz, time_parsing_local=True
            ... )
            >>> # start_local: 2025-01-15 14:00:00-06:00 (CST)
            >>> # start_utc:   2025-01-15 20:00:00+00:00 (UTC)

        Example 2: UTC mode (original behavior)
            >>> from datetime import UTC
            >>> # LLM returns naive datetime representing UTC
            >>> start_naive = datetime(2025, 1, 15, 20, 0, 0)
            >>> end_naive = datetime(2025, 1, 15, 22, 0, 0)
            >>> start_local, end_local, start_utc, end_utc = self._convert_to_utc(
            ...     start_naive, end_naive, UTC, time_parsing_local=False
            ... )
            >>> # All times are UTC: start_local == start_utc

        Example 3: Handling DST transition (spring forward)
            >>> chicago_tz = ZoneInfo("America/Chicago")
            >>> # March 10, 2024 at 2:30 AM doesn't exist (spring forward)
            >>> start_naive = datetime(2024, 3, 10, 2, 30, 0)
            >>> end_naive = datetime(2024, 3, 10, 4, 0, 0)
            >>> start_local, end_local, start_utc, end_utc = self._convert_to_utc(
            ...     start_naive, end_naive, chicago_tz, time_parsing_local=True
            ... )
            >>> # start_local is adjusted to valid time after gap

        .. note::
            This method uses _localize_naive_datetime() which properly handles
            DST transitions including non-existent times (spring forward) and
            ambiguous times (fall back).

        .. seealso::
            :meth:`_localize_naive_datetime` : Handles DST transitions properly
        """
        if time_parsing_local:
            # The LLM returns naive datetime objects representing local time
            # Properly localize naive datetimes to handle DST transitions correctly
            start_local = self._ensure_timezone_aware(start_date, local_tz)
            end_local = self._ensure_timezone_aware(end_date, local_tz)

            # Convert to UTC for internal use
            start_utc = start_local.astimezone(UTC)
            end_utc = end_local.astimezone(UTC)
        else:
            # LLM returns UTC times directly (original behavior)
            start_utc = start_date if start_date.tzinfo else start_date.replace(tzinfo=UTC)
            end_utc = end_date if end_date.tzinfo else end_date.replace(tzinfo=UTC)
            start_local = start_utc
            end_local = end_utc

        return start_local, end_local, start_utc, end_utc

    def _ensure_timezone_aware(self, dt: datetime, tz: ZoneInfo | timezone) -> datetime:
        """Ensure a datetime is timezone-aware, localizing if necessary.

        :param dt: Datetime object (may be naive or aware)
        :type dt: datetime
        :param tz: Timezone to apply if datetime is naive
        :type tz: ZoneInfo | timezone
        :return: Timezone-aware datetime
        :rtype: datetime
        """
        if dt.tzinfo is None:
            return self._localize_naive_datetime(dt, tz)
        return dt

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
        :param context: Error context including capability info and execution state (unused but required by interface)
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
