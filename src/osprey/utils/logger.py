"""
Component Logger Framework

Provides colored logging for Osprey and application components with:
- Unified API for all components (capabilities, infrastructure, pipelines)
- Rich terminal output with component-specific colors
- Graceful fallbacks when configuration is unavailable
- Simple, clear interface

Usage:
    # Module-level
    logger = get_logger("orchestrator")
    logger.key_info("Starting orchestration")

    logger = get_logger("data_processor")
    logger.info("Processing data")
    logger.debug("Detailed trace")
    logger.success("Operation completed")
    logger.warning("Something to note")
    logger.error("Something went wrong")
    logger.timing("Execution took 2.5 seconds")
    logger.approval("Waiting for user approval")

    # Custom loggers with explicit parameters
    logger = get_logger(name="custom_component", color="blue")
"""

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from osprey.utils.config import get_config_value


class ComponentLogger:
    """
    Rich-formatted logger for Osprey and application components with color coding and message hierarchy.

    Message Types:
    - status: High-level status updates
    - key_info: Important operational information
    - info: Normal operational messages
    - debug: Detailed tracing information
    - warning: Warning messages
    - error: Error messages
    - success: Success messages
    - timing: Timing information
    - approval: Approval messages
    - resume: Resume messages
    """

    def __init__(
        self,
        base_logger: logging.Logger,
        component_name: str,
        color: str = "white",
        state: Any = None,
    ):
        """
        Initialize component logger.

        Args:
            base_logger: Underlying Python logger
            component_name: Name of the component (e.g., 'data_analysis', 'router', 'mongo')
            color: Rich color name for this component
            state: Optional state for context (unused, kept for API compat)
        """
        self.base_logger = base_logger
        self.component_name = component_name
        self.color = color
        self._state = state

    def _log(self, level: int, message: str, *args, **kwargs) -> None:
        """Core logging method that delegates to the stdlib logger."""
        # Strip event-system kwargs that callers may still pass
        kwargs.pop("error", None)
        kwargs.pop("error_type", None)
        kwargs.pop("recoverable", None)
        kwargs.pop("stack_trace", None)
        kwargs.pop("warning", None)
        self.base_logger.log(level, message, *args, **kwargs)

    def status(self, message: str, *args, **kwargs) -> None:
        """Status update — high-level progress messages."""
        self._log(logging.INFO, message, *args, **kwargs)

    def key_info(self, message: str, *args, **kwargs) -> None:
        """Important operational information."""
        self._log(logging.INFO, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Normal operational messages."""
        self._log(logging.INFO, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Debug-level messages."""
        self._log(logging.DEBUG, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Warning messages."""
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Error messages."""
        self._log(logging.ERROR, message, *args, exc_info=exc_info, **kwargs)

    def success(self, message: str, *args, **kwargs) -> None:
        """Success messages."""
        self._log(logging.INFO, message, *args, **kwargs)

    def timing(self, message: str, *args, **kwargs) -> None:
        """Timing information."""
        self._log(logging.INFO, message, *args, **kwargs)

    def approval(self, message: str, *args, **kwargs) -> None:
        """Approval messages."""
        self._log(logging.INFO, message, *args, **kwargs)

    def resume(self, message: str, *args, **kwargs) -> None:
        """Resume messages."""
        self._log(logging.INFO, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Critical error messages."""
        self._log(logging.CRITICAL, message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """Exception with traceback."""
        self._log(logging.ERROR, message, *args, exc_info=True, **kwargs)

    # Delegate stdlib Logger interface so callers can treat ComponentLogger as a Logger.
    @property
    def level(self) -> int:
        return self.base_logger.level

    @property
    def name(self) -> str:
        return self.base_logger.name

    def setLevel(self, level: int) -> None:
        self.base_logger.setLevel(level)

    def isEnabledFor(self, level: int) -> bool:
        return self.base_logger.isEnabledFor(level)


def _setup_rich_logging(level: int = logging.INFO) -> None:
    """Configure Rich logging for the root logger (called once)."""
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, RichHandler):
            return

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(level)

    # Load user-configurable display preferences from config
    try:
        # Security-conscious defaults: hide locals to prevent sensitive data exposure
        rich_tracebacks = get_config_value("logging.rich_tracebacks", True)
        show_traceback_locals = get_config_value("logging.show_traceback_locals", False)
        show_full_paths = get_config_value("logging.show_full_paths", False)

    except Exception:
        # Config system unavailable; use secure defaults.
        # Cannot log here: logging infrastructure is mid-configuration
        # (handlers cleared but RichHandler not yet installed).
        rich_tracebacks = True
        show_traceback_locals = False
        show_full_paths = False

    # Optimize console for containerized and CI/CD environments
    console = Console(
        force_terminal=True,
        width=120,
        color_system="truecolor",
    )

    handler = RichHandler(
        console=console,
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_path=show_full_paths,
        show_time=True,
        show_level=True,
        tracebacks_show_locals=show_traceback_locals,
    )

    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    for lib in ["httpx", "httpcore", "requests", "urllib3", "LiteLLM"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(
    component_name: str = None,
    level: int = logging.INFO,
    *,
    state: Any = None,
    name: str = None,
    color: str = None,
) -> ComponentLogger:
    """
    Get a unified logger for CLI logging.

    Primary API (recommended):
        component_name: Component name (e.g., 'orchestrator', 'data_analysis')
        state: Optional state for context
        level: Logging level

    Explicit API (for custom loggers or module-level usage):
        name: Direct logger name (keyword-only)
        color: Direct color specification (keyword-only)
        level: Logging level

    Returns:
        ComponentLogger instance

    Examples:
        # Module-level
        logger = get_logger("orchestrator")
        logger.info("Planning started")

        # Custom logger
        logger = get_logger(name="test_logger", color="blue")
    """
    _setup_rich_logging(level)

    if name is not None:
        base_logger = logging.getLogger(name)
        actual_color = color or "white"
        return ComponentLogger(base_logger, name, actual_color, state=state)

    # Validate that component_name is provided
    if component_name is None:
        raise ValueError(
            "Component name is required. Usage: get_logger('component_name') or "
            "get_logger(name='custom_name', color='blue')"
        )

    base_logger = logging.getLogger(component_name)

    try:
        config_path = f"logging.logging_colors.{component_name}"
        color = get_config_value(config_path)

        if not color:
            color = "white"

    except Exception as e:
        color = "white"
        # Only show warning in debug mode to reduce noise
        import os

        if os.getenv("DEBUG_LOGGING"):
            print(
                f"WARNING: Failed to load color config for {component_name}: {e}. Using white as fallback."
            )

    return ComponentLogger(base_logger, component_name, color, state=state)
