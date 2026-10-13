"""Logging configuration for ALS injector optimization."""

import logging
import sys


def configure_library_loggers(level: int = logging.INFO) -> None:
    """Configure log levels for specific libraries.

    Args:
        level: Base logging level
    """
    logging.getLogger("osprey.services.optimizer").setLevel(level)

    # Suppress xopt info logs (like "Running Xopt step") unless in debug mode
    logging.getLogger("xopt").setLevel(logging.WARNING if level > logging.DEBUG else level)

    logging.getLogger("urllib3").setLevel(logging.WARNING)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application (CLI/Standalone).

    Args:
        level: Logging level (default: INFO)

    Sets up a console handler with a clear format showing
    timestamp, logger name, level, and message.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    configure_library_loggers(level)
