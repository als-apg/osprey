"""
Component Logger Framework

Provides colored logging for Osprey and application components with:
- Unified API for all components (capabilities, infrastructure, pipelines)
- Rich terminal output on stderr with component-specific colors
- Graceful fallbacks when configuration is unavailable
- Simple, clear interface

Handler setup is explicit and separate from logger lookup. Entry points — the
CLI, service startups, the dispatch worker, MCP servers — call
``configure_logging()`` once; everything else just calls ``get_logger()``, which
has no global side effects.

Usage:
    # Entry point, once at startup
    configure_logging()

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

from osprey_connectors.config import get_config_value


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

    Every emitted record carries the name of the method that produced it as the
    ``osprey_intent`` attribute, so handlers and filters can tell intents apart
    where the level cannot (``info`` and ``key_info`` both log at INFO). It is
    metadata only: no level is remapped and no formatter reads it.
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

    def _log(self, level: int, message: str, *args, intent: str, **kwargs) -> None:
        """Core logging method that delegates to the stdlib logger.

        Args:
            level: Standard library level for the record.
            message: Log message, possibly a %-style format string.
            *args: %-style format arguments for ``message``.
            intent: Name of the public intent method that produced this record.
                Rides along as the ``osprey_intent`` record attribute so
                handlers can tell intents apart without guessing from the level
                (``info`` and ``key_info`` both log at INFO). Metadata only —
                it changes no level and nothing about rendering.
            **kwargs: Forwarded to :meth:`logging.Logger.log`. A caller-supplied
                ``extra`` mapping is merged, not replaced; its dict is left
                unmutated.
        """
        # Strip event-system kwargs that callers may still pass
        kwargs.pop("error", None)
        kwargs.pop("error_type", None)
        kwargs.pop("recoverable", None)
        kwargs.pop("stack_trace", None)
        kwargs.pop("warning", None)
        caller_extra = kwargs.pop("extra", None)
        extra = {**(caller_extra or {}), "osprey_intent": intent}
        self.base_logger.log(level, message, *args, extra=extra, **kwargs)

    def status(self, message: str, *args, **kwargs) -> None:
        """Status update — high-level progress messages."""
        self._log(logging.INFO, message, *args, intent="status", **kwargs)

    def key_info(self, message: str, *args, **kwargs) -> None:
        """Important operational information."""
        self._log(logging.INFO, message, *args, intent="key_info", **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Normal operational messages."""
        self._log(logging.INFO, message, *args, intent="info", **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Debug-level messages."""
        self._log(logging.DEBUG, message, *args, intent="debug", **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Warning messages."""
        self._log(logging.WARNING, message, *args, intent="warning", **kwargs)

    def error(self, message: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Error messages."""
        self._log(logging.ERROR, message, *args, intent="error", exc_info=exc_info, **kwargs)

    def success(self, message: str, *args, **kwargs) -> None:
        """Success messages."""
        self._log(logging.INFO, message, *args, intent="success", **kwargs)

    def timing(self, message: str, *args, **kwargs) -> None:
        """Timing information."""
        self._log(logging.INFO, message, *args, intent="timing", **kwargs)

    def approval(self, message: str, *args, **kwargs) -> None:
        """Approval messages."""
        self._log(logging.INFO, message, *args, intent="approval", **kwargs)

    def resume(self, message: str, *args, **kwargs) -> None:
        """Resume messages."""
        self._log(logging.INFO, message, *args, intent="resume", **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Critical error messages."""
        self._log(logging.CRITICAL, message, *args, intent="critical", **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """Exception with traceback."""
        self._log(logging.ERROR, message, *args, intent="exception", exc_info=True, **kwargs)

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


#: Third-party loggers that ``configure_logging()`` raises to WARNING. They are
#: chatty at INFO and their output is noise in an operator's terminal.
QUIET_THIRD_PARTY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "requests",
    "urllib3",
    "LiteLLM",
    "claude_agent_sdk",
)


def _build_rich_handler(console: Console | None = None) -> RichHandler:
    """Build the Osprey ``RichHandler``, writing to **stderr**.

    stdout is reserved for program output — MCP stdio JSON-RPC frames and
    ``--json`` CLI payloads — so log records must never land there.

    Args:
        console: Console to render records through. ``None`` — every non-CLI
            entry point, and every external consumer of this package — builds
            the fixed-width stderr console this module has always built, so
            their output is unchanged. The CLI passes the terminal-sized
            stderr console it already owns.
    """
    try:
        # Security-conscious defaults: hide locals to prevent sensitive data exposure
        rich_tracebacks = get_config_value("logging.rich_tracebacks", True)
        show_traceback_locals = get_config_value("logging.show_traceback_locals", False)
        show_full_paths = get_config_value("logging.show_full_paths", False)
    except Exception:
        # Config system unavailable; use secure defaults. Cannot log here:
        # the logging infrastructure is mid-configuration.
        rich_tracebacks = True
        show_traceback_locals = False
        show_full_paths = False

    if console is None:
        # force_terminal keeps colors in containers and CI, where stderr is a pipe.
        console = Console(
            stderr=True,
            force_terminal=True,
            width=120,
            color_system="truecolor",
        )

    return RichHandler(
        console=console,
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_path=show_full_paths,
        show_time=True,
        show_level=True,
        tracebacks_show_locals=show_traceback_locals,
    )


def set_handler_console(console: Console) -> RichHandler | None:
    """Render root-logger records through ``console`` from now on.

    For entry points that already own a console of their own — the CLI owns the
    themed, terminal-sized stderr console every command prints through — so
    that logging and printed output share one renderer instead of interleaving
    two consoles with different widths on the same stream.

    Deliberately separate from ``configure_logging()``, which stays untouched:
    the 15+ non-CLI entry points and external consumers of this package call
    that and must keep the console it has always built for them.

    Repointing an existing handler rather than rebuilding it keeps this safe to
    call any number of times in one process — CLI invocations repeat inside a
    single test process — with no handler ever stacking on the root logger.

    Args:
        console: Console the Rich handler should render through.

    Returns:
        The handler that was repointed, so a caller can go on to configure that
        same instance; ``None`` if the root logger has no ``RichHandler`` (a
        caller that has not called ``configure_logging()`` yet).
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichHandler):
            handler.console = console
            return handler
    return None


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for an Osprey entry point.

    Call this once, explicitly, from every entry point (CLI ``main``, service
    startups, the dispatch worker, MCP server ``__main__``). Library and import
    paths must never call it — importing Osprey leaves logging untouched.

    The configuration is **strictly additive**: it installs the Osprey
    ``RichHandler`` only if the root logger has no ``RichHandler`` yet, and it
    never removes or clears handlers it did not install. Handlers owned by
    someone else — ``caplog``'s capture handler, a host application's own
    handlers — survive. Calling it repeatedly is a no-op beyond re-applying the
    level.

    Args:
        level: Root logger level.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(isinstance(handler, RichHandler) for handler in root_logger.handlers):
        root_logger.addHandler(_build_rich_handler())

    for lib in QUIET_THIRD_PARTY_LOGGERS:
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

    This is a pure lookup: it mutates no global logging state. Handler and level
    configuration belongs to :func:`configure_logging`, which entry points call
    once at startup. A module that only imports Osprey therefore leaves the
    host application's logging exactly as it found it.

    Primary API (recommended):
        component_name: Component name (e.g., 'orchestrator', 'data_analysis')
        state: Optional state for context
        level: Accepted for backwards compatibility and ignored — set levels
            through :func:`configure_logging` or on the returned logger.

    Explicit API (for custom loggers or module-level usage):
        name: Direct logger name (keyword-only)
        color: Direct color specification (keyword-only)

    Returns:
        ComponentLogger instance

    Examples:
        # Module-level
        logger = get_logger("orchestrator")
        logger.info("Planning started")

        # Custom logger
        logger = get_logger(name="test_logger", color="blue")
    """
    del level  # configures nothing here; see configure_logging()

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

    # No config lookup here, deliberately. Resolving
    # ``logging.logging_colors.<component>`` would buy nothing —
    # ComponentLogger.color is write-only and _log() delegates
    # straight to the stdlib logger — while dragging
    # the whole config machinery — and its ``.env`` load — into all ~70
    # module-level ``logger = get_logger(...)`` sites, making a bare
    # ``import osprey.<anything>`` rewrite os.environ. See the module
    # docstring in osprey_connectors.config on where .env loading belongs.
    return ComponentLogger(base_logger, component_name, "white", state=state)
