"""CLI Event Handler with Pattern Matching.

This module provides the CLIEventHandler class that processes typed Osprey
events using Python's pattern matching (match/case) for clean console output.

The handler provides focused output for CLI usage - showing status messages,
success confirmations, warnings, errors, and final results. Component colors
are loaded from the logging.logging_colors config for visual consistency.

Output format uses fixed-width columns: Role      Message
Uses Rich Table for proper text wrapping alignment.

Usage:
    from osprey.events import parse_event
    from osprey.interfaces.cli.event_handler import CLIEventHandler

    handler = CLIEventHandler(console)

    async for chunk in graph.astream(..., stream_mode="custom"):
        event = parse_event(chunk)
        if event:
            await handler.handle(event)
"""

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.table import Table
from rich.text import Text

from osprey.events import (
    ErrorEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    OspreyEvent,
    StatusEvent,
)
from osprey.utils.config import get_config_value


class CLIEventHandler:
    """Handles typed Osprey events for CLI output using pattern matching.

    This handler provides focused console output suitable for CLI usage.
    It shows status updates, success confirmations, warnings, errors,
    and final results with component-specific colors from config.

    Attributes:
        console: Rich Console for output
        verbose: Whether to show debug-level messages
    """

    ROLE_COLUMN_WIDTH = 18  # Fixed width for role column alignment

    def __init__(
        self,
        console: Console | None = None,
        verbose: bool = False,
        show_timing: bool = True,
    ):
        """Initialize the CLI event handler.

        Args:
            console: Rich Console for output (creates new one if not provided)
            verbose: Whether to show debug-level status messages
            show_timing: Kept for API compatibility (not used currently)
        """
        self.console = console or Console()
        self.verbose = verbose
        self.show_timing = show_timing
        self._color_cache: dict[str, str] = {}
        self._highlighter = ReprHighlighter()  # For data type highlighting
        self._suppress_respond_events = False  # Suppress respond events after streaming

    def start_response_streaming(self) -> None:
        """Call when LLM streaming begins to suppress post-streaming respond events.

        After streaming starts, completion events from the respond component
        (like "Response generated") would appear after the full response,
        which is jarring UX. This method enables suppression of those events.
        """
        self._suppress_respond_events = True

    def reset_suppression(self) -> None:
        """Reset respond event suppression for a new query."""
        self._suppress_respond_events = False

    def _format_component_name(self, component: str) -> str:
        """Format component name to title case.

        Examples:
            "registry" -> "Registry"
            "task_extraction" -> "Task_Extraction"
            "current_weather" -> "Current_Weather"

        Args:
            component: Raw component name

        Returns:
            Title case formatted name
        """
        parts = component.split("_")
        return "_".join(part.capitalize() for part in parts)

    def _format_role_name(self, component: str) -> str:
        """Format role name with truncation (no brackets).

        Creates a role name for display, truncating long names with "...".

        Args:
            component: Component name to format

        Returns:
            Formatted role name (without brackets)
        """
        name = self._format_component_name(component)
        max_len = self.ROLE_COLUMN_WIDTH - 1  # Leave room for at least 1 space
        if len(name) > max_len:
            name = name[: max_len - 3] + "..."
        return name

    def _get_component_color(self, component: str) -> str:
        """Get component color from config with caching.

        Looks up the color from logging.logging_colors.{component} config.
        Falls back to 'white' if not found or on error.

        Args:
            component: Component name (e.g., 'classifier', 'orchestrator')

        Returns:
            Rich color name (e.g., 'cyan', 'light_salmon1')
        """
        if component not in self._color_cache:
            try:
                color = get_config_value(f"logging.logging_colors.{component}")
                self._color_cache[component] = color or "white"
            except Exception:
                self._color_cache[component] = "white"
        return self._color_cache[component]

    def _print_aligned(
        self,
        role: str,
        message: str,
        role_style: str,
        msg_style: str,
        prefix: str = "",
    ) -> None:
        """Print message with proper column alignment using Rich Table.

        Uses a borderless table to ensure:
        - Role column has fixed width
        - Message column wraps properly
        - Wrapped lines align with message start, not role start

        Args:
            role: Component name (without brackets)
            message: Message content
            role_style: Style for role column
            msg_style: Style for message column
            prefix: Optional prefix like "✓ " or "⚠ "
        """
        table = Table(
            show_header=False,
            show_edge=False,
            show_footer=False,
            padding=(0, 0),
            collapse_padding=True,
            pad_edge=False,
            box=None,
        )

        # Role column: fixed width, no wrap
        table.add_column(width=self.ROLE_COLUMN_WIDTH, no_wrap=True, style=role_style)

        # Message column: flexible, wraps
        table.add_column(overflow="fold", style=msg_style)

        # Apply data highlighting to message
        msg_text = Text(message)
        self._highlighter.highlight(msg_text)

        role_text = f"{prefix}{role}" if prefix else role
        table.add_row(role_text, msg_text)
        self.console.print(table)

    async def handle(self, event: OspreyEvent) -> None:
        """Process a typed event using pattern matching.

        Routes the event to appropriate output based on event type and level.
        StatusEvents are filtered by level (status, info, success, warning,
        error shown; debug only in verbose mode; timing/approval/resume hidden).

        Args:
            event: The typed OspreyEvent to process
        """
        # Suppress respond events after streaming starts (better UX)
        if self._suppress_respond_events and hasattr(event, "component"):
            if event.component == "respond":
                return

        match event:
            # StatusEvent - status level (most common - high-level progress)
            case StatusEvent(message=msg, level="status", component=comp):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                self._print_aligned(role, msg, f"bold {color}", f"bold {color}")

            # StatusEvent - key_info level (important info with bold styling)
            case StatusEvent(message=msg, level="key_info", component=comp):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                self._print_aligned(role, msg, f"bold {color}", f"bold {color}")

            # StatusEvent - info level (operational info)
            case StatusEvent(message=msg, level="info", component=comp):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                self._print_aligned(role, msg, color, color)

            # StatusEvent - success level (completion confirmations)
            case StatusEvent(message=msg, level="success", component=comp):
                role = self._format_role_name(comp)
                self._print_aligned(role, f"✓ {msg}", "bold green", "bold green")

            # StatusEvent - warning level (warnings)
            case StatusEvent(message=msg, level="warning", component=comp):
                role = self._format_role_name(comp)
                self._print_aligned(role, f"⚠ {msg}", "yellow", "yellow")

            # StatusEvent - error level (errors)
            case StatusEvent(message=msg, level="error", component=comp):
                role = self._format_role_name(comp)
                self._print_aligned(role, f"✗ {msg}", "red", "red")

            # StatusEvent - debug level (only in verbose mode)
            case StatusEvent(message=msg, level="debug", component=comp) if self.verbose:
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                self._print_aligned(role, f"🔍 {msg}", f"dim {color}", f"dim {color}")

            # LLMRequestEvent - "LLM prompt built"
            case LLMRequestEvent(component=comp, prompt_length=length, key=key):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                if key:
                    msg = f"LLM prompt built for {key} ({length} chars)"
                else:
                    msg = f"LLM prompt built ({length} chars)"
                self._print_aligned(role, msg, color, color)

            # LLMResponseEvent - "LLM response received"
            case LLMResponseEvent(component=comp, response_length=length, duration_ms=dur, key=key):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                dur_sec = dur / 1000 if dur else 0
                if key:
                    msg = f"LLM response for {key} ({length} chars, {dur_sec:.2f}s)"
                else:
                    msg = f"LLM response received ({length} chars, {dur_sec:.2f}s)"
                self._print_aligned(role, msg, color, color)

            # ErrorEvent - execution errors
            case ErrorEvent(error_type=err_type, error_message=msg, component=comp):
                role = self._format_role_name(comp)
                self._print_aligned(role, f"✗ {err_type}", "red", "red")
                # Error detail on next line, aligned with message column
                self._print_aligned("", msg, "", "red")

            case _:
                # All other events silently ignored (timing, approval, resume,
                # PhaseStart/Complete, CapabilityStart/Complete, etc.)
                pass

    def handle_sync(self, event: OspreyEvent) -> None:
        """Synchronous version of handle for non-async contexts.

        Args:
            event: The typed OspreyEvent to process
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._handle_sync_impl(event)
            else:
                loop.run_until_complete(self.handle(event))
        except RuntimeError:
            self._handle_sync_impl(event)

    def _handle_sync_impl(self, event: OspreyEvent) -> None:
        """Direct synchronous implementation of event handling.

        Args:
            event: The typed OspreyEvent to process
        """
        # Suppress respond events after streaming starts (better UX)
        if self._suppress_respond_events and hasattr(event, "component"):
            if event.component == "respond":
                return

        match event:
            # StatusEvent - status level
            case StatusEvent(message=msg, level="status", component=comp):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                self._print_aligned(role, msg, f"bold {color}", f"bold {color}")

            # StatusEvent - key_info level (important info with bold styling)
            case StatusEvent(message=msg, level="key_info", component=comp):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                self._print_aligned(role, msg, f"bold {color}", f"bold {color}")

            # StatusEvent - info level
            case StatusEvent(message=msg, level="info", component=comp):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                self._print_aligned(role, msg, color, color)

            # StatusEvent - success level
            case StatusEvent(message=msg, level="success", component=comp):
                role = self._format_role_name(comp)
                self._print_aligned(role, f"✓ {msg}", "bold green", "bold green")

            # StatusEvent - warning level
            case StatusEvent(message=msg, level="warning", component=comp):
                role = self._format_role_name(comp)
                self._print_aligned(role, f"⚠ {msg}", "yellow", "yellow")

            # StatusEvent - error level
            case StatusEvent(message=msg, level="error", component=comp):
                role = self._format_role_name(comp)
                self._print_aligned(role, f"✗ {msg}", "red", "red")

            # StatusEvent - debug level (verbose only)
            case StatusEvent(message=msg, level="debug", component=comp) if self.verbose:
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                self._print_aligned(role, f"🔍 {msg}", f"dim {color}", f"dim {color}")

            # LLMRequestEvent - "LLM prompt built"
            case LLMRequestEvent(component=comp, prompt_length=length, key=key):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                if key:
                    msg = f"LLM prompt built for {key} ({length} chars)"
                else:
                    msg = f"LLM prompt built ({length} chars)"
                self._print_aligned(role, msg, color, color)

            # LLMResponseEvent - "LLM response received"
            case LLMResponseEvent(component=comp, response_length=length, duration_ms=dur, key=key):
                color = self._get_component_color(comp)
                role = self._format_role_name(comp)
                dur_sec = dur / 1000 if dur else 0
                if key:
                    msg = f"LLM response for {key} ({length} chars, {dur_sec:.2f}s)"
                else:
                    msg = f"LLM response received ({length} chars, {dur_sec:.2f}s)"
                self._print_aligned(role, msg, color, color)

            # ErrorEvent
            case ErrorEvent(error_type=err_type, error_message=msg, component=comp):
                role = self._format_role_name(comp)
                self._print_aligned(role, f"✗ {err_type}", "red", "red")
                # Error detail on next line, aligned with message column
                self._print_aligned("", msg, "", "red")

            case _:
                pass
