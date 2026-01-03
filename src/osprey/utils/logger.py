"""
Component Logger Framework

Provides colored logging for Osprey and application components with:
- Unified API for all components (capabilities, infrastructure, pipelines)
- Rich terminal output with component-specific colors
- Graceful fallbacks when configuration is unavailable
- Simple, clear interface
- Optional LangGraph streaming support via lazy initialization

Usage:
    # Components with streaming (via BaseCapability.get_logger())
    logger = self.get_logger()
    logger.status("Creating execution plan...")  # Logs + streams
    logger.info("Active capabilities: [...]")   # Logs only

    # Module-level (no streaming)
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

    Now includes optional LangGraph streaming support via lazy initialization.

    Message Types:
    - status: High-level status updates (logs + streams automatically)
    - key_info: Important operational information
    - info: Normal operational messages
    - debug: Detailed tracing information
    - warning: Warning messages
    - error: Error messages (logs + streams automatically)
    - success: Success messages (logs + streams by default)
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
            state: Optional AgentState for streaming context
        """
        self.base_logger = base_logger
        self.component_name = component_name
        self.color = color
        self._state = state

        # Lazy initialization - only when first needed
        self._stream_writer = None
        self._stream_writer_attempted = False
        self._step_info = None

    def _get_stream_writer(self):
        """Lazy initialization of stream writer (only when first needed)."""
        if not self._stream_writer_attempted:
            self._stream_writer_attempted = True
            try:
                from langgraph.config import get_stream_writer

                self._stream_writer = get_stream_writer()
                # Also extract step info when we get the writer
                self._step_info = self._extract_step_info(self._state)
            except (RuntimeError, ImportError):
                # Not in LangGraph context - that's fine
                self._stream_writer = None
                self._step_info = {}
        return self._stream_writer

    def _extract_step_info(self, state):
        """Extract step context from AgentState for streaming metadata.

        Reuses existing logic from StreamWriter._get_step_info():
        - Task preparation phases use hard-coded step mapping
        - Execution phases extract from execution plan in state
        - Falls back to basic component info
        """
        # Import the step mapping from streaming module
        try:
            from osprey.utils.streaming import TASK_PREPARATION_STEPS
        except ImportError:
            TASK_PREPARATION_STEPS = {}

        # Check if this is a task preparation component
        if self.component_name in TASK_PREPARATION_STEPS:
            return TASK_PREPARATION_STEPS[self.component_name]

        # For execution phase, extract from state
        if state and hasattr(state, "get"):
            try:
                from osprey.state.state_manager import StateManager

                execution_plan = state.get("planning_execution_plan")
                if execution_plan and execution_plan.get("steps"):
                    current_step_index = StateManager.get_current_step_index(state)
                    total_steps = len(execution_plan.get("steps", []))

                    if total_steps > 0:
                        return {
                            "step": current_step_index + 1,
                            "total_steps": total_steps,
                            "phase": "Execution",
                        }
            except Exception:
                pass  # Graceful degradation

        # Default: no step info
        return {
            "step": None,
            "total_steps": None,
            "phase": self.component_name.replace("_", " ").title(),
        }

    def _emit_stream_event(self, message: str, event_type: str = "status", **kwargs):
        """Emit streaming event if writer available."""
        writer = self._get_stream_writer()  # Lazy init
        if not writer:
            return  # Silently skip if no stream available

        try:
            import time

            event = {
                "event_type": event_type,
                "message": message,
                "component": self.component_name,
                "timestamp": time.time(),
                **(self._step_info or {}),
                **kwargs,
            }

            # Clean up None values
            event = {k: v for k, v in event.items() if v is not None}

            writer(event)
        except Exception as e:
            # Don't crash logging just because streaming failed
            self.debug(f"Failed to emit stream event: {e}")

    def _format_message(self, message: str, style: str, emoji: str = "") -> str:
        """Format message with Rich markup and emoji prefix."""
        try:
            prefix = f"{emoji}{self.component_name.title()}: "
            if style:
                return f"[{style}]{prefix}{message}[/{style}]"
            else:
                return f"{prefix}{message}"
        except Exception:
            # Graceful degradation for environments where Rich markup fails
            return f"{emoji}{self.component_name.title()}: {message}"

    def status(self, message: str, **kwargs) -> None:
        """Status update - logs and streams automatically.

        Use for high-level progress updates that users should see in both
        CLI and web interfaces.

        Args:
            message: Status message
            **kwargs: Additional metadata for streaming event

        Example:
            logger.status("Creating execution plan...")
            logger.status("Processing batch 2/5", batch=2, total=5)
        """
        self.key_info(message)  # Log to CLI with bold formatting
        self._emit_stream_event(message, "status", **kwargs)

    def key_info(self, message: str, stream: bool = False, **kwargs) -> None:
        """Important operational information - logs and optionally streams.

        Args:
            message: Info message
            stream: Whether to also stream this message
            **kwargs: Additional metadata for streaming event
        """
        style = f"bold {self.color}" if self.color != "white" else "bold white"
        formatted = self._format_message(message, style, "")
        self.base_logger.info(formatted)

        if stream:
            self._emit_stream_event(message, "key_info", **kwargs)

    def info(self, message: str, stream: bool = False, **kwargs) -> None:
        """Info message - logs always, streams optionally.

        By default, info messages only go to CLI logs. Use stream=True
        to also send to web interface.

        Args:
            message: Info message
            stream: Whether to also stream this message
            **kwargs: Additional metadata for streaming event

        Example:
            logger.info("Active capabilities: [...]")  # CLI only
            logger.info("Step completed", stream=True)  # CLI + stream
        """
        formatted = self._format_message(message, self.color, "")
        self.base_logger.info(formatted)

        if stream:
            self._emit_stream_event(message, "info", **kwargs)

    def debug(self, message: str, stream: bool = False, **kwargs) -> None:
        """Debug message - logs only (never streams by default).

        Debug messages are detailed technical info not meant for web UI.

        Args:
            message: Debug message
            stream: Whether to stream (default: False)
            **kwargs: Additional metadata for streaming event
        """
        style = f"dim {self.color}" if self.color != "white" else "dim white"
        formatted = self._format_message(message, style, "🔍 ")
        self.base_logger.debug(formatted)

        if stream:
            self._emit_stream_event(message, "debug", **kwargs)

    def warning(self, message: str, stream: bool = True, **kwargs) -> None:
        """Warning message - logs and optionally streams.

        Warnings stream by default since they're important for users to see.

        Args:
            message: Warning message
            stream: Whether to stream (default: True)
            **kwargs: Additional metadata for streaming event
        """
        formatted = self._format_message(message, "bold yellow", "⚠️  ")
        self.base_logger.warning(formatted)

        if stream:
            self._emit_stream_event(message, "warning", warning=True, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Error message - always logs and streams.

        Errors are important and should always be visible in both interfaces.

        Args:
            message: Error message
            exc_info: Whether to include exception traceback
            **kwargs: Additional error metadata for streaming event
        """
        formatted = self._format_message(message, "bold red", "❌ ")
        self.base_logger.error(formatted, exc_info=exc_info)
        self._emit_stream_event(message, "error", error=True, **kwargs)

    def success(self, message: str, stream: bool = True, **kwargs) -> None:
        """Success message - logs and optionally streams.

        Success messages stream by default to give users feedback.

        Args:
            message: Success message
            stream: Whether to stream (default: True)
            **kwargs: Additional metadata for streaming event
        """
        formatted = self._format_message(message, "bold green", "✅ ")
        self.base_logger.info(formatted)

        if stream:
            self._emit_stream_event(message, "success", **kwargs)

    def timing(self, message: str, stream: bool = False, **kwargs) -> None:
        """Timing information - logs and optionally streams.

        Args:
            message: Timing message
            stream: Whether to stream (default: False)
            **kwargs: Additional metadata for streaming event
        """
        formatted = self._format_message(message, "bold white", "🕒 ")
        self.base_logger.info(formatted)

        if stream:
            self._emit_stream_event(message, "timing", **kwargs)

    def approval(self, message: str, stream: bool = True, **kwargs) -> None:
        """Approval messages - logs and optionally streams.

        Approval requests stream by default so users see them in web UI.

        Args:
            message: Approval message
            stream: Whether to stream (default: True)
            **kwargs: Additional metadata for streaming event
        """
        formatted = self._format_message(message, "bold yellow", "🔍⚠️ ")
        self.base_logger.info(formatted)

        if stream:
            self._emit_stream_event(message, "approval", **kwargs)

    def resume(self, message: str, stream: bool = True, **kwargs) -> None:
        """Resume messages - logs and optionally streams.

        Resume messages stream by default to provide feedback.

        Args:
            message: Resume message
            stream: Whether to stream (default: True)
            **kwargs: Additional metadata for streaming event
        """
        formatted = self._format_message(message, "bold green", "🔄 ")
        self.base_logger.info(formatted)

        if stream:
            self._emit_stream_event(message, "resume", **kwargs)

    # Compatibility methods - delegate to base logger
    def critical(self, message: str, *args, **kwargs) -> None:
        formatted = self._format_message(message, "bold red", "❌ ")
        self.base_logger.critical(formatted, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        formatted = self._format_message(message, "bold red", "❌ ")
        self.base_logger.exception(formatted, *args, **kwargs)

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        self.base_logger.log(level, message, *args, **kwargs)

    # Properties for compatibility
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

    # Prevent duplicate handler registration for consistent logging behavior
    for handler in root_logger.handlers:
        if isinstance(handler, RichHandler):
            return

    # Ensure clean handler state to prevent duplicate log messages
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
        # Secure defaults when configuration system is unavailable
        rich_tracebacks = True
        show_traceback_locals = False
        show_full_paths = False

    # Optimize console for containerized and CI/CD environments
    console = Console(
        force_terminal=True,  # Ensure color output in Docker containers and CI systems
        width=120,  # Prevent line wrapping in standard terminal sizes
        color_system="truecolor",  # Enable full color spectrum for component identification
    )

    handler = RichHandler(
        console=console,  # Use our custom console
        rich_tracebacks=rich_tracebacks,  # Configurable rich tracebacks
        markup=True,  # Enable [bold], [green], etc. in log messages
        show_path=show_full_paths,  # Configurable path display
        show_time=True,  # Show timestamp
        show_level=True,  # Show log level
        tracebacks_show_locals=show_traceback_locals,  # Configurable local variables
    )

    root_logger.addHandler(handler)

    # Reduce third-party library noise to focus on application-specific issues
    for lib in ["httpx", "httpcore", "requests", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(
    component_name: str = None,
    level: int = logging.INFO,
    *,
    state: Any = None,
    name: str = None,
    color: str = None,
    # Deprecated parameters - kept for backward compatibility
    source: str = None,
) -> ComponentLogger:
    """
    Get a unified logger that handles both CLI logging and LangGraph streaming.

    Primary API (recommended - use via BaseCapability.get_logger()):
        component_name: Component name (e.g., 'orchestrator', 'data_analysis')
        state: Optional AgentState for streaming context and step tracking
        level: Logging level

    Explicit API (for custom loggers or module-level usage):
        name: Direct logger name (keyword-only)
        color: Direct color specification (keyword-only)
        level: Logging level

    Returns:
        ComponentLogger instance that logs to CLI and optionally streams

    Examples:
        # Recommended: Use via BaseCapability
        class MyCapability(BaseCapability):
            async def execute(self):
                logger = self.get_logger()  # Auto-streams!
                logger.status("Working...")

        # Module-level (no streaming)
        logger = get_logger("orchestrator")
        logger.info("Planning started")

        # With streaming (when you have state)
        logger = get_logger("orchestrator", state=state)
        logger.status("Creating execution plan...")  # Logs + streams
        logger.info("Active capabilities: [...]")   # Logs only
        logger.error("Failed!")                      # Logs + streams

        # Custom logger
        logger = get_logger(name="test_logger", color="blue")

    .. deprecated::
        The two-parameter API get_logger(source, component_name) is deprecated.
        Use get_logger(component_name) instead. The flat configuration structure
        (logging.logging_colors.{component_name}) replaces the old nested structure.
    """
    import warnings

    # Initialize logging infrastructure with Rich formatting support
    _setup_rich_logging(level)

    # Handle explicit API for custom logger creation (tests, utilities)
    if name is not None:
        # Direct logger creation bypasses convention-based color assignment
        base_logger = logging.getLogger(name)
        actual_color = color or "white"
        return ComponentLogger(base_logger, name, actual_color, state=state)

    # Handle deprecated two-parameter API: get_logger("framework", "component")
    # This maintains backward compatibility while warning users to migrate
    if source is not None:
        warnings.warn(
            f"The two-parameter API get_logger('{source}', '{component_name}') is deprecated. "
            f"Use get_logger('{component_name}') instead. The 'source' parameter is no longer needed "
            f"as the configuration uses a flat structure: logging.logging_colors.{component_name}",
            DeprecationWarning,
            stacklevel=2,
        )
        # For backward compatibility, still accept the old format but ignore source
        # component_name is already set from the second positional argument

    # Validate that component_name is provided
    if component_name is None:
        raise ValueError(
            "Component name is required. Usage: get_logger('component_name') or "
            "get_logger(name='custom_name', color='blue')"
        )

    # Use component name as logger identifier for hierarchical organization
    base_logger = logging.getLogger(component_name)

    # Retrieve component-specific color from flat configuration structure
    try:
        # New flat structure: logging.logging_colors.{component_name}
        config_path = f"logging.logging_colors.{component_name}"
        color = get_config_value(config_path)

        if not color:
            color = "white"

    except Exception as e:
        # Graceful degradation ensures logging continues even with config issues
        color = "white"
        # Only show warning in debug mode to reduce noise
        import os

        if os.getenv("DEBUG_LOGGING"):
            print(
                f"⚠️  WARNING: Failed to load color config for {component_name}: {e}. Using white as fallback."
            )

    # Pass state to enable streaming
    return ComponentLogger(base_logger, component_name, color, state=state)
