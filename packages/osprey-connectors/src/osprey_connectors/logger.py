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
import os
from collections.abc import Mapping
from types import TracebackType
from typing import NamedTuple, TypeAlias

from rich.console import Console
from rich.logging import RichHandler

from osprey_connectors.config import config_flag, get_config_value

__all__ = [
    "ComponentLogger",
    "configure_logging",
    "get_logger",
    "set_handler_console",
]

#: What :meth:`logging.Logger.log` takes as ``exc_info``.
_ExcInfo: TypeAlias = (
    bool
    | BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
    | None
)

#: Keywords of the retired event system that callers may still pass; dropped.
_EVENT_KWARGS = frozenset({"error", "error_type", "recoverable", "stack_trace", "warning"})


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
        state: object = None,
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

    def _log(
        self,
        level: int,
        message: str,
        *args: object,
        intent: str,
        exc_info: _ExcInfo,
        stack_info: bool,
        stacklevel: int,
        extra: Mapping[str, object] | None,
        other: Mapping[str, object],
    ) -> None:
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
            exc_info: Forwarded to :meth:`logging.Logger.log`.
            stack_info: Forwarded to :meth:`logging.Logger.log`.
            stacklevel: Forwarded to :meth:`logging.Logger.log`.
            extra: Forwarded to :meth:`logging.Logger.log`, merged, not
                replaced; the caller's dict is left unmutated.
            other: Every other keyword the caller passed. The retired
                event-system keywords are dropped; any other is refused, as
                :meth:`logging.Logger.log` refuses it: only when the level is
                enabled.
        """
        unknown = sorted(set(other) - _EVENT_KWARGS)
        if unknown and self.base_logger.isEnabledFor(level):
            raise TypeError(f"{intent}() got an unexpected keyword argument {unknown[0]!r}")
        merged = {**(extra or {}), "osprey_intent": intent}
        self.base_logger.log(
            level,
            message,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=merged,
        )

    def status(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Status update — high-level progress messages."""
        self._log(
            logging.INFO,
            message,
            *args,
            intent="status",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def key_info(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Important operational information."""
        self._log(
            logging.INFO,
            message,
            *args,
            intent="key_info",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def info(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Normal operational messages."""
        self._log(
            logging.INFO,
            message,
            *args,
            intent="info",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def debug(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Debug-level messages."""
        self._log(
            logging.DEBUG,
            message,
            *args,
            intent="debug",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def warning(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Warning messages."""
        self._log(
            logging.WARNING,
            message,
            *args,
            intent="warning",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def error(
        self,
        message: str,
        *args: object,
        exc_info: bool = False,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Error messages."""
        self._log(
            logging.ERROR,
            message,
            *args,
            intent="error",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def success(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Success messages."""
        self._log(
            logging.INFO,
            message,
            *args,
            intent="success",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def timing(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Timing information."""
        self._log(
            logging.INFO,
            message,
            *args,
            intent="timing",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def approval(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Approval messages."""
        self._log(
            logging.INFO,
            message,
            *args,
            intent="approval",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def resume(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Resume messages."""
        self._log(
            logging.INFO,
            message,
            *args,
            intent="resume",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def critical(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Critical error messages."""
        self._log(
            logging.CRITICAL,
            message,
            *args,
            intent="critical",
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

    def exception(
        self,
        message: str,
        *args: object,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **other: object,
    ) -> None:
        """Exception with traceback."""
        self._log(
            logging.ERROR,
            message,
            *args,
            intent="exception",
            exc_info=True,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            other=other,
        )

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
_QUIET_THIRD_PARTY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "requests",
    "urllib3",
    "LiteLLM",
    "claude_agent_sdk",
)


class _HandlerSettings(NamedTuple):
    """The ``logging.*`` settings the Osprey ``RichHandler`` renders by."""

    rich_tracebacks: bool
    show_traceback_locals: bool
    show_full_paths: bool


#: What a handler renders by when the configuration cannot say. Locals stay
#: hidden so a traceback never exposes sensitive data.
_DEFAULT_HANDLER_SETTINGS = _HandlerSettings(
    rich_tracebacks=True, show_traceback_locals=False, show_full_paths=False
)

#: The handler :func:`configure_logging` installed, which is the only one a
#: later call re-applies settings to.
_installed_handler: RichHandler | None = None

#: Whether this process has already said that it could not read the settings.
_settings_warning_given = False


def _read_handler_settings() -> tuple[_HandlerSettings, Exception | None]:
    """Read the handler's settings from the configuration this process resolves.

    Each switch is read through :func:`config_flag`, so a ``"false"`` that
    reached the config as text (quoted, or a resolved ``${VAR:-false}``) is off.
    A switch spelling neither on nor off is a problem like an unavailable
    configuration: every setting falls back to the defaults, which hide locals
    and shorten paths, rather than failing the process's start-up over a
    logging key.

    Returns:
        The settings, and what kept them from being read: the defaults and the
        exception when the configuration is unavailable, else ``None``.
    """

    def switch(name: str, default: bool) -> bool:
        key = f"logging.{name}"
        return config_flag(get_config_value(key, default), key=key)

    try:
        return (
            _HandlerSettings(
                rich_tracebacks=switch("rich_tracebacks", True),
                show_traceback_locals=switch("show_traceback_locals", False),
                show_full_paths=switch("show_full_paths", False),
            ),
            None,
        )
    except Exception as exc:
        return _DEFAULT_HANDLER_SETTINGS, exc


def _apply_handler_settings(handler: RichHandler, settings: _HandlerSettings) -> None:
    """Make ``handler`` render by ``settings`` from its next record on."""
    handler.rich_tracebacks = settings.rich_tracebacks
    handler.tracebacks_show_locals = settings.show_traceback_locals
    # Rich keeps the path column on its renderer and offers no public setter.
    handler._log_render.show_path = settings.show_full_paths


def _warn_settings_unavailable(problem: Exception) -> None:
    """Say, once per process, that the handler renders by the defaults.

    A process that names no config and stands in a directory without one — a
    command run outside any deployment — has no settings to miss, so that case
    is not reported.
    """
    global _settings_warning_given
    names_a_config = os.environ.get("OSPREY_CONFIG") or os.environ.get("CONFIG_FILE")
    if _settings_warning_given or (isinstance(problem, FileNotFoundError) and not names_a_config):
        return
    _settings_warning_given = True
    summary = str(problem).strip().splitlines()[0] if str(problem).strip() else ""
    logging.getLogger(__name__).warning(
        "The logging.* settings could not be read from the configuration (%s: %s); "
        "log output uses the defaults: Rich tracebacks on, locals hidden, short paths.",
        type(problem).__name__,
        summary,
        # The exception's text is not ours: brackets in it are not markup.
        extra={"markup": False},
    )


def _build_rich_handler(
    console: Console | None = None, settings: _HandlerSettings | None = None
) -> RichHandler:
    """Build the Osprey ``RichHandler``, writing to **stderr**.

    stdout is reserved for program output — MCP stdio JSON-RPC frames and
    ``--json`` CLI payloads — so log records must never land there.

    Args:
        console: Console to render records through. ``None`` — every non-CLI
            entry point, and every external consumer of this package — builds
            the fixed-width stderr console this module has always built, so
            their output is unchanged. The CLI passes the terminal-sized
            stderr console it already owns.
        settings: What the handler renders by. ``None`` reads them from the
            configuration, falling back to the defaults when it is unavailable.
    """
    if settings is None:
        settings, _ = _read_handler_settings()

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
        rich_tracebacks=settings.rich_tracebacks,
        markup=True,
        show_path=settings.show_full_paths,
        show_time=True,
        show_level=True,
        tracebacks_show_locals=settings.show_traceback_locals,
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
    level and the ``logging.*`` settings.

    Those settings are read from the configuration on every call and applied to
    the handler this function installed. An entry point that configures logging
    before it knows which deployment it serves calls again once it does, and
    the handler then renders by that deployment's settings. Settings that
    cannot be read fall back to the defaults, with one warning per process.

    Args:
        level: Root logger level.
    """
    global _installed_handler
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    settings, problem = _read_handler_settings()
    if not any(isinstance(handler, RichHandler) for handler in root_logger.handlers):
        _installed_handler = _build_rich_handler(settings=settings)
        root_logger.addHandler(_installed_handler)
    elif _installed_handler in root_logger.handlers:
        # Only the handler installed here: a RichHandler someone else put on
        # the root logger is theirs.
        _apply_handler_settings(_installed_handler, settings)
    if problem is not None:
        _warn_settings_unavailable(problem)

    for lib in _QUIET_THIRD_PARTY_LOGGERS:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(
    component_name: str | None = None,
    level: int = logging.INFO,
    *,
    state: object = None,
    name: str | None = None,
    color: str | None = None,
) -> ComponentLogger:
    """
    Get a unified logger for CLI logging.

    This is a pure lookup: it mutates no global logging state. Handler and level
    configuration belongs to :func:`configure_logging`, which entry points call
    once at startup. A module that only imports Osprey therefore leaves the
    host application's logging exactly as it found it.

    Name the logger by its component (the recommended form), or give ``name``
    and ``color`` directly for a custom or module-level logger.

    Args:
        component_name: Component name (e.g., 'orchestrator', 'data_analysis')
        level: Accepted for backwards compatibility and ignored — set levels
            through :func:`configure_logging` or on the returned logger.
        state: Optional state for context
        name: Direct logger name (keyword-only)
        color: Direct color specification (keyword-only)

    Returns:
        ComponentLogger instance

    Examples:
        .. code-block:: python

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
