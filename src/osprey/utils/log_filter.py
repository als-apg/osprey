"""Log filtering utilities for selective suppression by logger name, level, or message pattern.

Provides LoggerFilter for fine-grained suppression and context managers
(suppress_logger, suppress_logger_level, quiet_logger) for temporary filtering.
"""

import logging
import re
from contextlib import contextmanager
from re import Pattern


class LoggerFilter(logging.Filter):
    """Selective log suppression based on logger name, level, and message patterns.

    When multiple criteria are specified, they are combined with AND logic:
    a message must match all criteria to be affected by the filter.
    """

    def __init__(
        self,
        logger_names: list[str] | None = None,
        message_patterns: list[str] | None = None,
        levels: list[int] | None = None,
        invert: bool = False,
        name: str = "",
    ):
        """Initialize the log filter with filtering criteria.

        Args:
            logger_names: Logger names to filter. None/empty applies to all loggers.
            message_patterns: Regex patterns to match against formatted log messages.
            levels: Log levels to filter. None/empty applies to all levels.
            invert: If True, suppresses everything EXCEPT matches.
            name: Optional name for the filter (passed to parent Filter class).
        """
        super().__init__(name=name)

        self.logger_names: set[str] = set(logger_names or [])
        self.message_patterns: list[Pattern] = [
            re.compile(pattern) for pattern in (message_patterns or [])
        ]
        self.levels: set[int] = set(levels or [])
        self.invert = invert

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine whether to allow or suppress a log record.

        Args:
            record: The LogRecord to filter

        Returns:
            True to allow the record, False to suppress it
        """
        if self.logger_names and record.name not in self.logger_names:
            return True

        if self.levels and record.levelno not in self.levels:
            return True

        if self.message_patterns:
            message = record.getMessage()
            matches = any(pattern.search(message) for pattern in self.message_patterns)

            if self.invert:
                return matches
            else:
                return not matches

        if self.invert:
            return True
        else:
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

    Args:
        logger_name: Name of logger(s) to suppress (string or list of strings).
        levels: Specific levels to suppress. None suppresses all levels.
        message_patterns: Regex patterns; only matching messages are suppressed.

    Yields:
        The LoggerFilter instance.
    """
    logger_names = [logger_name] if isinstance(logger_name, str) else logger_name

    log_filter = LoggerFilter(
        logger_names=logger_names, levels=levels, message_patterns=message_patterns
    )

    loggers = [logging.getLogger(name) for name in logger_names]
    for logger in loggers:
        logger.addFilter(log_filter)

    try:
        yield log_filter
    finally:
        for logger in loggers:
            logger.removeFilter(log_filter)


@contextmanager
def suppress_logger_level(logger_name: str | list[str], level: int):
    """Context manager to temporarily raise logger level to suppress messages.

    Args:
        logger_name: Name of logger(s) to modify (string or list of strings).
        level: Temporary log level; messages below this level are suppressed.

    Yields:
        Dictionary mapping logger names to their original levels.
    """
    logger_names = [logger_name] if isinstance(logger_name, str) else logger_name

    loggers = [logging.getLogger(name) for name in logger_names]
    original_levels = {
        name: logger.level for name, logger in zip(logger_names, loggers, strict=False)
    }

    for logger in loggers:
        logger.setLevel(level)

    try:
        yield original_levels
    finally:
        for name, logger in zip(logger_names, loggers, strict=False):
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

            >>> with quiet_logger('registry'):
            ...     initialize_registry()  # No INFO messages shown

        Quiet multiple loggers::

            >>> with quiet_logger(['registry', 'DATABASE']):
            ...     do_something()

    .. note::
       This is equivalent to suppress_logger_level(logger_name, logging.WARNING)
       but with a more intuitive name for the common "quiet mode" use case.
    """
    with suppress_logger_level(logger_name, logging.WARNING) as levels:
        yield levels


__all__ = [
    "LoggerFilter",
    "suppress_logger",
    "suppress_logger_level",
    "quiet_logger",
]
