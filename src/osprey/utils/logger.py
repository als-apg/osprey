"""
Component Logger Framework

Provides colored logging for Osprey and application components with:
- Unified API for all components (capabilities, infrastructure, pipelines)
- Rich terminal output with component-specific colors
- Graceful fallbacks when configuration is unavailable
- Simple, clear interface
- Typed event emission for structured streaming (OspreyEvent types)

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

    # Emit typed events directly (for infrastructure nodes)
    from osprey.events import PhaseStartEvent
    logger.emit_event(PhaseStartEvent(phase="task_extraction"))
"""

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from osprey.events import ErrorEvent, EventEmitter, OspreyEvent, StatusEvent
from osprey.utils.config import get_config_value

# Hard-coded step mapping for task preparation phases
# (Moved from deprecated streaming.py module)
TASK_PREPARATION_STEPS = {
    "task_extraction": {"step": 1, "total_steps": 1, "phase": "Task Preparation"},
}


class ComponentLogger:
    """
    Rich-formatted logger for Osprey and application components with color coding and message hierarchy.

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
        self._step_info = None

        # Typed event emitter for the new event streaming system
        self._event_emitter = EventEmitter(component_name)

    def _extract_step_info(self, state):
        """Extract step context for streaming metadata."""
        if self.component_name in TASK_PREPARATION_STEPS:
            return TASK_PREPARATION_STEPS[self.component_name]
        return {
            "step": None,
            "total_steps": None,
            "phase": self.component_name.replace("_", " ").title(),
        }

    def _emit_stream_event(self, message: str, *args, event_type: str = "status", **kwargs):
        """Emit streaming event as typed OspreyEvent.

        Uses the EventEmitter to emit typed StatusEvent or ErrorEvent instances.
        The emitter handles event streaming and fallback handlers automatically.

        Supports stdlib-style format args: ``logger.info("msg %s", val)``
        """
        # Apply %-style formatting if positional args are provided (stdlib compat)
        if args:
            try:
                message = message % args
            except (TypeError, ValueError):
                pass

        # Extract step info for the event (lazy init if needed)
        if self._step_info is None:
            self._step_info = self._extract_step_info(self._state)

        step_info = self._step_info or {}

        try:
            # Create typed event based on event_type
            if event_type == "error" or kwargs.get("error"):
                event = ErrorEvent(
                    component=self.component_name,
                    error_type=kwargs.get("error_type", "ExecutionError"),
                    error_message=message,
                    recoverable=kwargs.get("recoverable", False),
                    stack_trace=kwargs.get("stack_trace"),
                )
            else:
                _valid_levels = frozenset(
                    {
                        "status",
                        "info",
                        "debug",
                        "warning",
                        "success",
                        "key_info",
                        "timing",
                        "approval",
                        "resume",
                    }
                )
                level = event_type if event_type in _valid_levels else "info"

                event = StatusEvent(
                    component=self.component_name,
                    message=message,
                    level=level,
                    phase=step_info.get("phase"),
                    step=step_info.get("step"),
                    total_steps=step_info.get("total_steps"),
                )

            # Emit via the typed event system
            self._event_emitter.emit(event)

        except Exception:
            # Don't crash logging just because streaming failed
            # Avoid recursive debug() call that could cause infinite loop
            pass

    def emit_event(self, event: OspreyEvent) -> None:
        """Emit a typed OspreyEvent directly.

        Use this for structured events like PhaseStartEvent, CapabilityStartEvent, etc.
        that don't fit the standard logging pattern.

        Args:
            event: The typed event to emit

        Example:
            from osprey.events import PhaseStartEvent
            logger.emit_event(PhaseStartEvent(
                phase="task_extraction",
                description="Extracting task from query"
            ))
        """
        # Ensure component is set if not already
        if not event.component:
            event.component = self.component_name

        self._event_emitter.emit(event)

    def emit_llm_request(
        self, prompt: str, key: str = "", model: str = "", provider: str = ""
    ) -> None:
        """Emit LLMRequestEvent with full prompt for TUI display.

        Args:
            prompt: The complete LLM prompt text
            key: Optional key for accumulating multiple prompts (e.g., capability name)
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            provider: Provider name (e.g., "openai", "anthropic")
        """
        from osprey.events import LLMRequestEvent

        event = LLMRequestEvent(
            component=self.component_name,
            prompt_preview=prompt[:200] + "..." if len(prompt) > 200 else prompt,
            prompt_length=len(prompt),
            model=model,
            provider=provider,
            full_prompt=prompt,
            key=key,
        )
        self._event_emitter.emit(event)

    def emit_llm_response(
        self,
        response: str,
        key: str = "",
        duration_ms: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Emit LLMResponseEvent with full response for TUI display.

        Args:
            response: The complete LLM response text
            key: Optional key for accumulating multiple responses (e.g., capability name)
            duration_ms: How long the request took in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        from osprey.events import LLMResponseEvent

        event = LLMResponseEvent(
            component=self.component_name,
            response_preview=response[:200] + "..." if len(response) > 200 else response,
            response_length=len(response),
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            full_response=response,
            key=key,
        )
        self._event_emitter.emit(event)

    def status(self, message: str, *args, **kwargs) -> None:
        """Status update - emits StatusEvent.

        User-facing output. Transport is automatic:
        - During execution: event streaming
        - Outside execution: fallback transport via TypedEventHandler

        Args:
            message: Status message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event

        Example:
            logger.status("Creating execution plan...")
            logger.status("Processing batch 2/5", batch=2, total=5)
        """
        self._emit_stream_event(message, *args, event_type="status", **kwargs)

    def key_info(self, message: str, *args, **kwargs) -> None:
        """Important operational information - emits StatusEvent with info level.

        User-facing output. Transport is automatic.

        Args:
            message: Info message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event
        """
        self._emit_stream_event(message, *args, event_type="key_info", **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Info message - emits StatusEvent with info level.

        User-facing output. Transport is automatic.

        Args:
            message: Info message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event

        Example:
            logger.info("Active capabilities: [...]")
            logger.info("Step completed")
        """
        self._emit_stream_event(message, *args, event_type="info", **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Debug message - emits StatusEvent with debug level.

        User-facing output (filtered by client if not needed).
        Transport is automatic.

        Args:
            message: Debug message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event
        """
        self._emit_stream_event(message, *args, event_type="debug", **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Warning message - emits StatusEvent with warning level.

        User-facing output. Transport is automatic.

        Args:
            message: Warning message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event
        """
        self._emit_stream_event(message, *args, event_type="warning", warning=True, **kwargs)

    def error(self, message: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Error message - emits ErrorEvent.

        User-facing output. Transport is automatic.

        Args:
            message: Error message
            *args: Optional %-style format args (stdlib compat)
            exc_info: Whether to include exception traceback in ErrorEvent
            **kwargs: Additional error metadata for streaming event
        """
        # Include stack trace in ErrorEvent if exc_info=True
        if exc_info and "stack_trace" not in kwargs:
            import traceback

            kwargs["stack_trace"] = traceback.format_exc()

        self._emit_stream_event(message, *args, event_type="error", error=True, **kwargs)

    def success(self, message: str, *args, **kwargs) -> None:
        """Success message - emits StatusEvent with success level.

        User-facing output. Transport is automatic.

        Args:
            message: Success message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event
        """
        self._emit_stream_event(message, *args, event_type="success", **kwargs)

    def timing(self, message: str, *args, **kwargs) -> None:
        """Timing information - emits StatusEvent with timing level.

        User-facing output. Transport is automatic.

        Args:
            message: Timing message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event
        """
        self._emit_stream_event(message, *args, event_type="timing", **kwargs)

    def approval(self, message: str, *args, **kwargs) -> None:
        """Approval message - emits StatusEvent with approval level.

        User-facing output. Transport is automatic.

        Args:
            message: Approval message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event
        """
        self._emit_stream_event(message, *args, event_type="approval", **kwargs)

    def resume(self, message: str, *args, **kwargs) -> None:
        """Resume message - emits StatusEvent with resume level.

        User-facing output. Transport is automatic.

        Args:
            message: Resume message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional metadata for streaming event
        """
        self._emit_stream_event(message, *args, event_type="resume", **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Critical error - emits ErrorEvent.

        User-facing output. Transport is automatic.

        Args:
            message: Critical error message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional error metadata for streaming event
        """
        self._emit_stream_event(
            message, *args, event_type="error", error=True, error_type="CriticalError", **kwargs
        )

    def exception(self, message: str, *args, **kwargs) -> None:
        """Exception with traceback - emits ErrorEvent with stack trace.

        User-facing output. Transport is automatic.

        Args:
            message: Exception message
            *args: Optional %-style format args (stdlib compat)
            **kwargs: Additional error metadata for streaming event
        """
        import traceback

        if "stack_trace" not in kwargs:
            kwargs["stack_trace"] = traceback.format_exc()
        self._emit_stream_event(message, *args, event_type="error", error=True, **kwargs)

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
    Get a unified logger that handles both CLI logging and event streaming.

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
                f"⚠️  WARNING: Failed to load color config for {component_name}: {e}. Using white as fallback."
            )

    return ComponentLogger(base_logger, component_name, color, state=state)
