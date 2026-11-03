"""Temporary suppression of a logger's low-level messages.

:func:`quiet_logger` raises one or more loggers to WARNING for the duration of
a ``with`` block and restores their levels afterwards.
"""

import logging
from contextlib import contextmanager


@contextmanager
def _suppress_logger_level(logger_name: str | list[str], level: int):
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

    This is a convenience wrapper around _suppress_logger_level() that specifically
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
       This raises the level to ``logging.WARNING`` for the block.
    """
    with _suppress_logger_level(logger_name, logging.WARNING) as levels:
        yield levels


__all__ = [
    "quiet_logger",
]
