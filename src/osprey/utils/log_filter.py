r"""Flexible logging filter utilities for dynamic log suppression and control.

This module provides utilities for fine-grained control over Python's logging output,
allowing selective suppression or filtering of log messages based on logger name,
log level, message content patterns, or any combination thereof.

The primary use case is temporarily suppressing verbose logging during operations
like health checks, tests, or other scenarios where specific log output would be
noise rather than signal, while still preserving error and warning messages.

Key Features:
    - Filter by logger name (e.g., suppress only 'REGISTRY' logger)
    - Filter by log level (e.g., suppress only INFO, keep WARNING+)
    - Filter by message content patterns (regex-based)
    - Invert logic (show only matches instead of suppressing)
    - Combine multiple criteria
    - Context manager for temporary filtering
    - Thread-safe filter management

Examples:
    Suppress all REGISTRY logger output::

        >>> filter = LoggerFilter(logger_names=['REGISTRY'])
        >>> logging.getLogger('REGISTRY').addFilter(filter)

    Suppress specific message patterns::

        >>> filter = LoggerFilter(
        ...     logger_names=['REGISTRY'],
        ...     message_patterns=[r'Configured.*registry', r'Added.*sys\.path']
        ... )
        >>> logging.getLogger('REGISTRY').addFilter(filter)

    Temporarily suppress with context manager::

        >>> with suppress_logger('REGISTRY'):
        ...     # REGISTRY logs suppressed here
        ...     initialize_registry()

    Suppress only INFO-level messages::

        >>> with suppress_logger_level('REGISTRY', logging.INFO):
        ...     # Only INFO suppressed, WARNING/ERROR still shown
        ...     do_something()

    Show only errors from specific logger::

        >>> filter = LoggerFilter(
        ...     logger_names=['REGISTRY'],
        ...     levels=[logging.ERROR],
        ...     invert=True
        ... )
        >>> logging.getLogger('REGISTRY').addFilter(filter)

.. seealso::
   :class:`LoggerFilter` : Main filtering class
   :func:`suppress_logger` : Context manager for temporary suppression
   :func:`suppress_logger_level` : Context manager for level-based suppression
"""

import logging
import re
from contextlib import contextmanager
from re import Pattern


class LoggerFilter(logging.Filter):
    """Flexible filter for selective log suppression based on multiple criteria.

    This filter can suppress log messages based on logger name, log level,
    message content patterns, or any combination. It supports both positive
    (suppress matches) and negative (show only matches) filtering logic.

    The filter can be attached to individual loggers or to handlers to affect
    multiple loggers at once. Filters attached to loggers only affect that
    logger's messages, while filters attached to handlers affect all messages
    passing through that handler.

    Attributes:
        logger_names: Set of logger names this filter applies to
        message_patterns: Compiled regex patterns to match against messages
        levels: Set of log levels this filter applies to
        invert: If True, inverts the filter logic (show only matches)

    .. note::
       When multiple criteria are specified, they are combined with AND logic.
       A message must match all criteria to be affected by the filter.

    .. warning::
       Be careful with invert=True as it can suppress important error messages
       if not configured carefully. Always test your filter configuration.
    """

    def __init__(
        self,
        logger_names: list[str] | None = None,
        message_patterns: list[str] | None = None,
        levels: list[int] | None = None,
        invert: bool = False,
        name: str = "",
    ):
        r"""Initialize the log filter with filtering criteria.

        Args:
            logger_names: List of logger names to filter. If None or empty,
                applies to all loggers. Example: ['REGISTRY', 'DATABASE']
            message_patterns: List of regex patterns to match in log messages.
                Patterns are matched against the formatted message string.
                Example: [r'Configured.*registry', r'Added.*sys\.path']
            levels: List of log levels to filter (e.g., [logging.INFO, logging.DEBUG]).
                If None or empty, applies to all levels. Use logging constants:
                DEBUG (10), INFO (20), WARNING (30), ERROR (40), CRITICAL (50)
            invert: If True, inverts the logic - suppresses everything EXCEPT matches.
                Useful for "show only errors" type filters.
            name: Optional name for the filter (passed to parent Filter class)

        Examples:
            Suppress all messages from specific loggers::

                >>> filter = LoggerFilter(logger_names=['REGISTRY', 'DATABASE'])

            Suppress specific message patterns::

                >>> filter = LoggerFilter(
                ...     logger_names=['REGISTRY'],
                ...     message_patterns=[r'Configured.*', r'Loaded.*']
                ... )

            Suppress only INFO-level messages::

                >>> filter = LoggerFilter(
                ...     logger_names=['REGISTRY'],
                ...     levels=[logging.INFO]
                ... )

            Show ONLY errors (suppress everything else)::

                >>> filter = LoggerFilter(
                ...     logger_names=['REGISTRY'],
                ...     levels=[logging.ERROR, logging.CRITICAL],
                ...     invert=True
                ... )
        """
        super().__init__(name=name)

        # Convert logger_names to set for O(1) lookup
        self.logger_names: set[str] = set(logger_names or [])

        # Compile message patterns for efficient matching
        self.message_patterns: list[Pattern] = [
            re.compile(pattern) for pattern in (message_patterns or [])
        ]

        # Convert levels to set for O(1) lookup
        self.levels: set[int] = set(levels or [])

        # Invert flag for "show only" logic
        self.invert = invert

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine whether to allow or suppress a log record.

        This method is called by Python's logging system for each log record.
        Return True to allow the record through, False to suppress it.

        Filtering Logic:
            1. If logger_names specified and record not in list → allow (don't filter other loggers)
            2. If levels specified and record level not in list → allow (don't filter other levels)
            3. If message_patterns specified, check if message matches any pattern
            4. Apply invert logic if enabled
            5. If no patterns specified, apply filter to all matching logger/level combinations

        Args:
            record: The LogRecord to filter

        Returns:
            True to allow the record, False to suppress it

        .. note::
           This method is called for every log message, so it should be efficient.
           Regex patterns are pre-compiled in __init__ for performance.
        """
        # Step 1: Check if this filter applies to this logger
        if self.logger_names and record.name not in self.logger_names:
            return True  # Don't filter messages from other loggers

        # Step 2: Check if this filter applies to this level
        if self.levels and record.levelno not in self.levels:
            return True  # Don't filter messages at other levels

        # Step 3: Check message patterns if specified
        if self.message_patterns:
            message = record.getMessage()
            matches = any(pattern.search(message) for pattern in self.message_patterns)

            if self.invert:
                # Inverted: show only matches, suppress non-matches
                return matches
            else:
                # Normal: suppress matches, show non-matches
                return not matches

        # Step 4: No patterns - filter all matching logger/level combinations
        if self.invert:
            # Inverted with no patterns: show everything (shouldn't get here)
            return True
        else:
            # Normal with no patterns: suppress everything matching logger/level
            return False

    def __repr__(self) -> str:
        """String representation for debugging."""
        parts = []
        if self.logger_names:
            parts.append(f"loggers={list(self.logger_names)}")
        if self.levels:
            level_names = [logging.getLevelName(lvl) for lvl in self.levels]
            parts.append(f"levels={level_names}")
        if self.message_patterns:
            patterns = [p.pattern for p in self.message_patterns]
            parts.append(f"patterns={patterns}")
        if self.invert:
            parts.append("inverted=True")

        criteria = ", ".join(parts) if parts else "no criteria"
        return f"LoggerFilter({criteria})"


@contextmanager
def suppress_logger(
    logger_name: str | list[str],
    levels: list[int] | None = None,
    message_patterns: list[str] | None = None,
):
    """Context manager to temporarily suppress logger output.

    Installs a LoggerFilter on the specified logger(s) for the duration of
    the context, then removes it automatically. This is useful for suppressing
    verbose logging during specific operations without permanently affecting
    the logger configuration.

    Args:
        logger_name: Name of logger(s) to suppress. Can be a single string
            or list of strings for multiple loggers.
        levels: Optional list of specific levels to suppress. If None,
            suppresses all levels. Example: [logging.INFO, logging.DEBUG]
        message_patterns: Optional list of regex patterns. If provided,
            only messages matching these patterns are suppressed.

    Yields:
        The LoggerFilter instance, allowing inspection if needed

    Examples:
        Suppress all REGISTRY output::

            >>> with suppress_logger('REGISTRY'):
            ...     initialize_registry()  # No REGISTRY logs shown

        Suppress only INFO-level messages::

            >>> with suppress_logger('REGISTRY', levels=[logging.INFO]):
            ...     initialize_registry()  # WARNING/ERROR still shown

        Suppress specific message patterns::

            >>> with suppress_logger(
            ...     'REGISTRY',
            ...     message_patterns=[r'Configured.*', r'Loaded.*']
            ... ):
            ...     initialize_registry()

        Suppress multiple loggers::

            >>> with suppress_logger(['REGISTRY', 'DATABASE']):
            ...     do_something()

    .. warning::
       The filter is automatically removed when the context exits, even if
       an exception is raised. However, if the logger is reconfigured during
       the context, the filter may not be properly removed.
    """
    # Normalize logger_name to list
    logger_names = [logger_name] if isinstance(logger_name, str) else logger_name

    # Create filter
    log_filter = LoggerFilter(
        logger_names=logger_names, levels=levels, message_patterns=message_patterns
    )

    # Get logger objects and add filter to each
    loggers = [logging.getLogger(name) for name in logger_names]
    for logger in loggers:
        logger.addFilter(log_filter)

    try:
        yield log_filter
    finally:
        # Remove filter from all loggers
        for logger in loggers:
            logger.removeFilter(log_filter)


@contextmanager
def suppress_logger_level(logger_name: str | list[str], level: int):
    """Context manager to temporarily raise logger level to suppress messages.

    This is a simpler alternative to suppress_logger() that works by temporarily
    changing the logger's level rather than using a filter. This approach is
    more efficient but less flexible - it suppresses all messages below the
    specified level, not just specific patterns.

    Args:
        logger_name: Name of logger(s) to modify. Can be a single string
            or list of strings for multiple loggers.
        level: The temporary log level to set. Messages below this level
            will be suppressed. Use logging constants:
            - logging.DEBUG (10) - show everything
            - logging.INFO (20) - suppress DEBUG
            - logging.WARNING (30) - suppress DEBUG, INFO
            - logging.ERROR (40) - suppress DEBUG, INFO, WARNING
            - logging.CRITICAL (50) - show only CRITICAL

    Yields:
        Dictionary mapping logger names to their original levels

    Examples:
        Suppress INFO and below, show WARNING+::

            >>> with suppress_logger_level('REGISTRY', logging.WARNING):
            ...     initialize_registry()  # Only WARNING+ shown

        Suppress everything except CRITICAL::

            >>> with suppress_logger_level('REGISTRY', logging.CRITICAL):
            ...     do_something()

        Suppress multiple loggers::

            >>> with suppress_logger_level(['REGISTRY', 'DATABASE'], logging.WARNING):
            ...     do_something()

    .. note::
       This is more efficient than LoggerFilter but less flexible. Use this
       when you want simple level-based filtering. Use suppress_logger() or
       LoggerFilter directly for pattern-based filtering.
    """
    # Normalize logger_name to list
    logger_names = [logger_name] if isinstance(logger_name, str) else logger_name

    # Get loggers and store original levels
    loggers = [logging.getLogger(name) for name in logger_names]
    original_levels = {name: logger.level for name, logger in zip(logger_names, loggers)}

    # Set new level on all loggers
    for logger in loggers:
        logger.setLevel(level)

    try:
        yield original_levels
    finally:
        # Restore original levels
        for name, logger in zip(logger_names, loggers):
            logger.setLevel(original_levels[name])


@contextmanager
def quiet_logger(logger_name: str | list[str]):
    """Context manager to temporarily suppress INFO-level messages from logger(s).

    This is a convenience wrapper around suppress_logger_level() that specifically
    suppresses INFO and DEBUG messages while preserving WARNING, ERROR, and CRITICAL.
    This is the most common use case for "quiet" operations.

    Args:
        logger_name: Name of logger(s) to quiet. Can be a single string
            or list of strings for multiple loggers.

    Yields:
        Dictionary mapping logger names to their original levels

    Examples:
        Quiet a single logger::

            >>> with quiet_logger('REGISTRY'):
            ...     initialize_registry()  # No INFO messages shown

        Quiet multiple loggers::

            >>> with quiet_logger(['REGISTRY', 'DATABASE']):
            ...     do_something()

    .. note::
       This is equivalent to suppress_logger_level(logger_name, logging.WARNING)
       but with a more intuitive name for the common "quiet mode" use case.
    """
    with suppress_logger_level(logger_name, logging.WARNING) as levels:
        yield levels


# Export public API
__all__ = [
    "LoggerFilter",
    "suppress_logger",
    "suppress_logger_level",
    "quiet_logger",
]
