"""Log Viewer modal for displaying step logs with live updates."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Static

if TYPE_CHECKING:
    from osprey.interfaces.tui.widgets.blocks import ProcessingStep


class LogViewer(ModalScreen[None]):
    """Modal screen for viewing step logs with live updates.

    Displays formatted log messages in a scrollable container.
    Supports live updates when given a reference to a ProcessingStep.
    """

    BINDINGS = [
        ("escape", "dismiss_viewer", "Close"),
        ("enter", "dismiss_viewer", "Close"),
    ]

    def __init__(
        self,
        title: str,
        log_source: list[tuple[str, str]] | ProcessingStep,
    ):
        """Initialize the log viewer.

        Args:
            title: The title to display (e.g., "Task Extraction - Logs").
            log_source: Either a list of (status, message) tuples (static),
                       or a ProcessingStep reference (live updates).
        """
        super().__init__()
        self.log_title = title
        self._log_source = log_source
        self._refresh_timer: Timer | None = None

    def _get_logs(self) -> list[tuple[str, str]]:
        """Get current logs from the source.

        Returns:
            List of (status, message) tuples.
        """
        if isinstance(self._log_source, list):
            return self._log_source
        # It's a ProcessingStep - read from its _log_messages
        return getattr(self._log_source, "_log_messages", [])

    def _is_live_source(self) -> bool:
        """Check if log source supports live updates."""
        return not isinstance(self._log_source, list)

    def compose(self) -> ComposeResult:
        """Compose the log viewer layout."""
        with Container(id="log-viewer-container"):
            with Horizontal(id="log-viewer-header"):
                yield Static(self.log_title, id="log-viewer-title")
                yield Static("enter/esc", id="log-viewer-dismiss-hint")
            with ScrollableContainer(id="log-viewer-content"):
                yield Static(self._format_logs(), id="log-viewer-logs")

    def on_mount(self) -> None:
        """Start refresh timer for live updates."""
        if self._is_live_source():
            self._refresh_timer = self.set_interval(0.5, self._refresh_logs)

    def on_unmount(self) -> None:
        """Stop refresh timer."""
        if self._refresh_timer:
            self._refresh_timer.stop()
            self._refresh_timer = None

    def _refresh_logs(self) -> None:
        """Refresh log display with latest logs."""
        try:
            logs_widget = self.query_one("#log-viewer-logs", Static)
            logs_widget.update(self._format_logs())
        except Exception:
            pass  # Widget may not exist during transitions

    def _format_logs(self) -> str:
        """Format log messages with status indicators and colors.

        Returns:
            Formatted string with Rich markup for colors.
        """
        logs = self._get_logs()
        if not logs:
            return "[dim]No logs available[/dim]"

        # Same indicators as ProcessingStep/ProcessingBlock
        INDICATOR_PENDING = "·"
        INDICATOR_SUCCESS = "✓"
        INDICATOR_ERROR = "✗"
        INDICATOR_WARNING = "⚠"

        lines = []
        for status, msg in logs:
            # Map status to color and indicator
            color_map = {
                "error": "$error",
                "warning": "$warning",
                "success": "$success",
                "key_info": "$text",
                "info": "",
                "debug": "$text-muted",
                "timing": "$accent",
                "status": "",
                "approval": "$warning",
                "resume": "$success",
            }
            indicator_map = {
                "error": INDICATOR_ERROR,
                "warning": INDICATOR_WARNING,
                "success": INDICATOR_SUCCESS,
                "key_info": INDICATOR_SUCCESS,
                "approval": INDICATOR_WARNING,
            }

            color = color_map.get(status, "")
            indicator = indicator_map.get(status, INDICATOR_PENDING)

            # Wrap long messages
            wrapped = textwrap.fill(
                msg,
                width=100,
                initial_indent="",
                subsequent_indent="  ",
            )

            if color:
                lines.append(f"[{color}]{indicator} {wrapped}[/{color}]")
            else:
                lines.append(f"{indicator} {wrapped}")

        return "\n".join(lines)

    def action_dismiss_viewer(self) -> None:
        """Dismiss the log viewer."""
        self.dismiss(None)
