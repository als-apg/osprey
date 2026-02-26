"""Processing block widgets for the TUI."""

import textwrap
from datetime import datetime
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Collapsible, Static

if TYPE_CHECKING:
    from textual.events import Click


class ProcessingBlock(Static):
    """Base class for processing blocks with indicator, input, and collapsible output."""

    # Text-based indicators (professional look)
    INDICATOR_PENDING = "·"
    INDICATOR_ACTIVE = "*"
    INDICATOR_SUCCESS = "✓"
    INDICATOR_ERROR = "✗"
    INDICATOR_WARNING = "⚠"

    # Breathing animation frames - asterisk style
    BREATHING_FRAMES = ["*", "✱", "✳", "✱"]

    # Default expanded header text (subclasses override)
    EXPANDED_HEADER = "Output"

    def __init__(self, title: str, **kwargs):
        """Initialize a processing block.

        Args:
            title: The title displayed in the block header.
        """
        super().__init__(**kwargs)
        self.title = title
        self._status = "pending"
        self._mounted = False
        # Pending state to apply after mount
        self._pending_input: str | None = None
        self._pending_output: tuple[str, str] | None = None  # (text, status)
        # Breathing animation state
        self._breathing_timer = None
        self._breathing_index = 0
        # Output preview for collapsible toggle
        self._output_preview: str = ""
        # Input preview for collapsible toggle
        self._input_preview: str = ""
        # LOG section - streaming messages for debugging
        self._log_messages: list[tuple[str, str, datetime | None]] = []
        # Track if IN was populated from streaming (vs placeholder)
        self._input_set: bool = False
        # Data dict for extracted information (task, capabilities, steps, etc.)
        self._data: dict[str, Any] = {}
        # Track last error message for OUT section on block close
        self._last_error_msg: str = ""

    def compose(self) -> ComposeResult:
        """Compose the block with header, input, separator, OUT, and LOG sections."""
        header_text = f"{self.INDICATOR_PENDING} {self.title}"
        yield Static(header_text, classes="block-header", id="block-header")
        # IN section - collapsible like OUT/LOG
        yield Collapsible(
            Static("", id="block-input-content"),
            title="",
            collapsed=True,
            collapsed_symbol="",
            expanded_symbol="",
            id="block-input",
        )
        # OUT section - final outcome only (hide built-in arrows)
        yield Collapsible(
            Static("", id="block-output-content"),
            title="",
            collapsed=True,
            collapsed_symbol="",
            expanded_symbol="",
            id="block-output",
        )
        # Full-width separator (will be truncated by container)
        yield Static("─" * 120, classes="block-separator", id="block-separator")
        # LOG section - all streaming messages (collapsed by default)
        yield Collapsible(
            Static("", id="block-log-content"),
            title="",
            collapsed=True,
            collapsed_symbol="",
            expanded_symbol="",
            id="block-log",
        )

    def on_mount(self) -> None:
        """Apply pending state after widget is mounted."""
        self._mounted = True
        # Hide IN section initially
        inputs = self.query("#block-input")
        if inputs:
            inputs.first().display = False
        # Hide separator initially
        separator = self.query_one("#block-separator", Static)
        separator.display = False
        # Hide OUT section initially
        outputs = self.query("#block-output")
        if outputs:
            outputs.first().display = False
        # Hide LOG section initially
        logs = self.query("#block-log")
        if logs:
            logs.first().display = False
        # Apply pending state
        if self._status == "active":
            self._apply_active()
        if self._pending_input is not None:
            self._apply_input(self._pending_input)
        if self._pending_output is not None:
            if self._status == "active":
                # Block still active - use partial output (keeps breathing)
                self._apply_partial_output(*self._pending_output)
            else:
                # Block complete - use full output (stops breathing)
                self._apply_output(*self._pending_output)
        # Show LOG section if logs were added before on_mount()
        if self._log_messages:
            self._update_log_display()

    def _start_breathing(self) -> None:
        """Start the breathing animation timer."""
        if self._breathing_timer is None:
            self._breathing_timer = self.set_interval(0.4, self._breathing_tick)

    def _breathing_tick(self) -> None:
        """Update breathing animation frame."""
        if self._status != "active":
            self._stop_breathing()
            return

        self._breathing_index = (self._breathing_index + 1) % len(self.BREATHING_FRAMES)
        indicator = self.BREATHING_FRAMES[self._breathing_index]
        header = self.query_one("#block-header", Static)
        header.update(f"[$accent]{indicator}[/] {self.title}")

    def _stop_breathing(self) -> None:
        """Stop the breathing animation."""
        if self._breathing_timer:
            self._breathing_timer.stop()
            self._breathing_timer = None

    def _apply_active(self) -> None:
        """Internal: apply active state to header and start breathing."""
        header = self.query_one("#block-header", Static)
        header.update(f"{self.INDICATOR_ACTIVE} {self.title}")
        self.add_class("block-active")
        # Start breathing animation
        self._start_breathing()

    def _apply_input(self, text: str) -> None:
        """Internal: apply input text to collapsible section."""
        input_section = self.query_one("#block-input", Collapsible)
        input_section.display = True

        # Get preview for collapsed state
        self._input_preview = self._get_preview(text, max_len=60)
        input_section.title = f"[bold]IN[/bold]    ▸ {self._input_preview}"

        # Update full content inside collapsible
        content = self.query_one("#block-input-content", Static)
        content.update(text)

    def _get_preview(self, text: str, max_len: int = 60) -> str:
        """Get one-line preview of output."""
        first_line = text.split("\n")[0] if text else ""
        if len(first_line) > max_len:
            return first_line[:max_len] + "..."
        elif "\n" in text:
            return first_line + " ..."
        return first_line

    def _apply_output(self, text: str, status: str) -> None:
        """Internal: apply output with collapsible preview.

        Args:
            text: The output text to display.
            status: The completion status ('success' or 'error').
        """
        # Stop breathing animation
        self._stop_breathing()

        # Update header indicator based on status (only success/error as terminal states)
        indicators = {
            "success": self.INDICATOR_SUCCESS,
            "error": self.INDICATOR_ERROR,
        }
        indicator = indicators.get(status, self.INDICATOR_SUCCESS)
        header = self.query_one("#block-header", Static)
        header.update(f"{indicator} {self.title}")

        # Show separator
        separator = self.query_one("#block-separator", Static)
        separator.display = True

        # Store preview for collapsible toggle events
        self._output_preview = self._get_preview(text)

        # Show and update collapsible output (official Collapsible)
        output = self.query_one("#block-output", Collapsible)
        output.display = True
        # Format: "[bold]OUT[/bold]   ▸ {preview}" - arrow before preview
        output.title = f"[bold]OUT[/bold]   ▸ {self._output_preview}"
        # Update the content inside the collapsible
        content = self.query_one("#block-output-content", Static)
        content.update(text)

        # Update CSS classes for status-specific styling
        self.remove_class("block-active")
        self.add_class(f"block-{status}")

        # Add status class to output collapsible for color styling
        output.add_class(f"output-{status}")

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        """Show descriptive header when expanded."""
        if event.collapsible.id == "block-input":
            event.collapsible.title = "[bold]IN[/bold]    ▾ Full input"
        elif event.collapsible.id == "block-output":
            event.collapsible.title = f"[bold]OUT[/bold]   ▾ {self.EXPANDED_HEADER}"
        elif event.collapsible.id == "block-log":
            event.collapsible.title = "[bold]LOG[/bold]   ▾ Streaming logs"

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        """Show collapsed arrow with preview."""
        if event.collapsible.id == "block-input":
            event.collapsible.title = f"[bold]IN[/bold]    ▸ {self._input_preview}"
        elif event.collapsible.id == "block-output":
            event.collapsible.title = f"[bold]OUT[/bold]   ▸ {self._output_preview}"
        elif event.collapsible.id == "block-log":
            count = len(self._log_messages)
            event.collapsible.title = f"[bold]LOG[/bold]   ▸ {count} messages"

    def set_active(self) -> None:
        """Mark the block as actively processing."""
        self._status = "active"
        if self._mounted:
            self._apply_active()

    def set_input(self, text: str, mark_set: bool = True) -> None:
        """Set the input section text.

        Args:
            text: The input text to display.
            mark_set: If True, mark _input_set flag (use False for placeholders).
        """
        self._pending_input = text
        if mark_set:
            self._input_set = True
        if self._mounted:
            self._apply_input(text)

    def set_output(self, text: str, status: str = "success") -> None:
        """Set the output section and mark complete.

        Args:
            text: The output text to display.
            status: The completion status ('success' or 'error').
        """
        self._status = status
        self._pending_output = (text, status)
        if self._mounted:
            self._apply_output(text, status)

    def set_partial_output(self, text: str, status: str = "pending") -> None:
        """Set partial output while block is still active (keeps breathing).

        Unlike set_output(), this doesn't mark the block as complete.
        Used for real-time status/error updates during streaming.

        Args:
            text: The output text to display.
            status: The status for styling ('pending', 'error', etc.).
        """
        self._pending_output = (text, status)
        self._output_preview = self._get_preview(text)
        if self._mounted:
            self._apply_partial_output(text, status)

    def _apply_partial_output(self, text: str, status: str) -> None:
        """Show separator and OUT section without stopping block.

        Args:
            text: The output text to display.
            status: The status for styling.
        """
        # Show separator
        separator = self.query_one("#block-separator", Static)
        separator.display = True

        # Show and update OUT section
        output = self.query_one("#block-output", Collapsible)
        output.display = True
        output.title = f"[bold]OUT[/bold]   ▸ {self._output_preview}"
        content = self.query_one("#block-output-content", Static)
        content.update(text)

        # DON'T stop breathing or change header indicator
        # Block remains "active" with breathing animation

    def add_log(
        self, message: str, status: str = "status", timestamp: datetime | None = None
    ) -> None:
        """Add a message to the LOG section.

        Args:
            message: The message text.
            status: The message status ('status', 'success', 'error', 'warning').
            timestamp: Optional timestamp for the log entry.
        """
        if message:
            self._log_messages.append((status, message, timestamp))
            self._update_log_display()
            # Track last error message for OUT section on block close
            if status == "error":
                self._last_error_msg = message

    def _format_log_messages(self) -> str:
        """Format all log messages with status symbols and hanging indent.

        Uses Textual CSS theme variables for colors that adapt with theme changes.
        See: https://textual.textualize.io/guide/content/
        """
        if not self._log_messages:
            return ""
        lines = []
        for entry in self._log_messages:
            msg_status, msg = entry[0], entry[1]
            # Map log_type to Textual CSS theme variables
            # These adapt automatically when theme changes
            color_map = {
                "error": "$error",
                "warning": "$warning",
                "success": "$success",
                "key_info": "$text",
                "info": "",  # No color, uses inherited
                "debug": "$text-muted",
                "timing": "$accent",
                "status": "",
                "approval": "$warning",
                "resume": "$success",
            }
            color = color_map.get(msg_status, "")
            prefix = {
                "error": self.INDICATOR_ERROR,
                "warning": self.INDICATOR_WARNING,
                "success": self.INDICATOR_SUCCESS,
                "key_info": self.INDICATOR_SUCCESS,
                "status": self.INDICATOR_PENDING,
                "approval": self.INDICATOR_WARNING,
            }.get(msg_status, self.INDICATOR_PENDING)

            # Wrap message with hanging indent (prefix is 2 chars: "· ")
            wrapped = textwrap.fill(
                msg,
                width=78,
                initial_indent="",
                subsequent_indent="  ",  # 2 spaces to align with text after symbol
            )

            if color:
                lines.append(f"[{color}]{prefix} {wrapped}[/{color}]")
            else:
                lines.append(f"{prefix} {wrapped}")
        return "\n".join(lines)

    def _update_log_display(self) -> None:
        """Update the LOG section display with current messages."""
        if not self._mounted:
            return

        # Show LOG section
        log_section = self.query_one("#block-log", Collapsible)
        log_section.display = True
        log_section.title = f"[bold]LOG[/bold]   ▸ {len(self._log_messages)} messages"

        # Update content
        content = self.query_one("#block-log-content", Static)
        content.update(self._format_log_messages())


class LogsLink(Static):
    """Focusable logs link that opens log viewer modal on Enter or click."""

    can_focus = True

    BINDINGS = [
        Binding("enter", "activate", "Open logs", show=False),
    ]

    def action_activate(self) -> None:
        """Handle Enter key press - open logs modal."""
        # Find parent ProcessingStep and call _show_logs
        for ancestor in self.ancestors_with_self:
            if isinstance(ancestor, ProcessingStep):
                ancestor._show_logs()
                break


class PromptLink(Static):
    """Focusable prompt link that opens content viewer modal on Enter or click."""

    can_focus = True

    BINDINGS = [
        Binding("enter", "activate", "Open prompt", show=False),
    ]

    def action_activate(self) -> None:
        """Handle Enter key press - open prompt modal."""
        # Find parent ProcessingStep and call _show_prompt
        for ancestor in self.ancestors_with_self:
            if isinstance(ancestor, ProcessingStep):
                ancestor._show_prompt()
                break


class ResponseLink(Static):
    """Focusable response link that opens content viewer modal on Enter or click."""

    can_focus = True

    BINDINGS = [
        Binding("enter", "activate", "Open response", show=False),
    ]

    def action_activate(self) -> None:
        """Handle Enter key press - open response modal."""
        # Find parent ProcessingStep and call _show_response
        for ancestor in self.ancestors_with_self:
            if isinstance(ancestor, ProcessingStep):
                ancestor._show_response()
                break


class WrappedStatic(Static):
    """Static widget that wraps text with proper indentation at render time.

    Unlike regular Static, this widget re-wraps content when the widget
    is resized, ensuring text always fits within the available width.
    """

    def __init__(
        self,
        content: str = "",
        initial_indent: str = "",
        subsequent_indent: str = "",
        **kwargs,
    ):
        """Initialize wrapped static widget.

        Args:
            content: Initial text content.
            initial_indent: Prefix for first line (e.g., "  ╰ ").
            subsequent_indent: Prefix for wrapped lines (e.g., "    ").
        """
        super().__init__(content, **kwargs)
        self._raw_content = content
        self._initial_indent = initial_indent
        self._subsequent_indent = subsequent_indent

    def set_content(self, content: str) -> None:
        """Set raw content to be wrapped at render time.

        Args:
            content: The raw text content (without indentation).
        """
        self._raw_content = content
        self._wrap_and_update()

    def on_resize(self) -> None:
        """Re-wrap content when widget is resized."""
        self._wrap_and_update()

    def _wrap_and_update(self) -> None:
        """Wrap content to fit current width and update display."""
        if not self._raw_content:
            self.update("")
            return

        # Use content_size.width if available, else size.width, else fallback
        try:
            width = self.content_size.width
            if width <= 0:
                width = self.size.width
            if width <= 0:
                width = 80
        except Exception:
            width = 80

        # Wrap each line separately (preserve hard line breaks)
        wrapped_lines = []
        for i, line in enumerate(self._raw_content.split("\n")):
            indent = self._initial_indent if i == 0 else self._subsequent_indent
            wrapped = textwrap.fill(
                line,
                width=width,
                initial_indent=indent,
                subsequent_indent=self._subsequent_indent,
            )
            wrapped_lines.append(wrapped)

        self.update("\n".join(wrapped_lines))


class ProcessingStep(Static):
    """Base class for minimal processing steps with indicator and title only.

    Unlike ProcessingBlock, this displays just a single line with an indicator
    and title, plus an optional output line showing the result.
    Includes a "logs" link to view full log history in a modal.
    """

    class LogAdded(Message):
        """Posted when a new log is added to this step."""

        def __init__(self, status: str, message: str, timestamp: datetime | None) -> None:
            super().__init__()
            self.status = status
            self.message = message
            self.timestamp = timestamp

    # Same indicators as ProcessingBlock for consistency
    INDICATOR_PENDING = "·"
    INDICATOR_ACTIVE = "*"
    INDICATOR_SUCCESS = "●"  # Filled circle (color differentiates)
    INDICATOR_ERROR = "●"  # Filled circle (color differentiates)

    # Visual guide for output line
    OUTPUT_GUIDE = "╰"

    # Breathing animation frames - same as ProcessingBlock
    BREATHING_FRAMES = ["*", "✱", "✳", "✱"]

    def __init__(self, title: str, **kwargs):
        """Initialize a processing step.

        Args:
            title: The title displayed on the step line.
        """
        super().__init__(**kwargs)
        self.title = title
        self._status = "pending"
        self._mounted = False
        # Breathing animation state
        self._breathing_timer = None
        self._breathing_index = 0
        # Internal log storage (displayed via LogViewer modal)
        self._log_messages: list[tuple[str, str, datetime | None]] = []
        # Data dict for extracted information
        self._data: dict[str, Any] = {}
        # Track if input was set (for compatibility)
        self._input_set: bool = False
        # Output message for display
        self._output_message: str = ""
        # Track last error message (for compatibility with ProcessingBlock)
        self._last_error_msg: str = ""
        # LLM prompt and response storage (dict for multi-capability support)
        self._llm_prompts: dict[str, str] = {}
        self._llm_responses: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        """Compose the step with title line, links, and output line."""
        with Horizontal(id="step-header"):
            yield Static(
                f"[bold]{self.INDICATOR_PENDING} {self.title}[/bold]",
                id="step-title",
            )
            yield LogsLink("logs", id="step-logs-link")
            yield PromptLink("prompt", id="step-prompt-link")
            yield ResponseLink("response", id="step-response-link")
        yield WrappedStatic(
            "",
            initial_indent=f"  {self.OUTPUT_GUIDE} ",
            subsequent_indent="    ",
            id="step-output",
        )

    def on_mount(self) -> None:
        """Apply pending state after widget is mounted."""
        self._mounted = True
        # Hide output line initially (if present - subclasses may use different output widgets)
        try:
            output = self.query_one("#step-output", WrappedStatic)
            output.display = False
        except Exception:
            pass  # Subclass uses different output widget (e.g., TodoList)
        # Hide logs link initially (shown when logs available)
        try:
            logs_link = self.query_one("#step-logs-link", LogsLink)
            logs_link.display = False
            # Show logs link if data was set before mounting (race condition fix)
            if self._log_messages:
                logs_link.display = True
        except Exception:
            pass  # Subclass doesn't have logs link
        # Hide prompt/response links initially (shown when data available)
        try:
            prompt_link = self.query_one("#step-prompt-link", PromptLink)
            prompt_link.display = False
            if self._llm_prompts:
                prompt_link.display = True
        except Exception:
            pass  # Subclass doesn't have prompt link
        try:
            response_link = self.query_one("#step-response-link", ResponseLink)
            response_link.display = False
            if self._llm_responses:
                response_link.display = True
        except Exception:
            pass  # Subclass doesn't have response link
        # Apply pending state
        if self._status == "active":
            self._apply_active()

    def on_click(self, event: "Click") -> None:
        """Handle click events on the links."""
        try:
            logs_link = self.query_one("#step-logs-link", LogsLink)
            if logs_link in event.widget.ancestors_with_self:
                self._show_logs()
                return
        except Exception:
            pass  # Subclass doesn't have logs link

        try:
            prompt_link = self.query_one("#step-prompt-link", PromptLink)
            if prompt_link in event.widget.ancestors_with_self:
                self._show_prompt()
                return
        except Exception:
            pass  # Subclass doesn't have prompt link

        try:
            response_link = self.query_one("#step-response-link", ResponseLink)
            if response_link in event.widget.ancestors_with_self:
                self._show_response()
        except Exception:
            pass  # Subclass doesn't have response link

    def _show_logs(self) -> None:
        """Open the log viewer modal with live updates."""
        from osprey.interfaces.tui.widgets.log_viewer import LogViewer

        viewer = LogViewer(f"{self.title} - Logs", self)
        self.app.push_screen(viewer)

    def _show_prompt(self) -> None:
        """Open the prompt viewer modal."""
        from osprey.interfaces.tui.widgets.content_viewer import ContentViewer

        # Single prompt with empty key -> pass as string; else pass dict for tabs
        if len(self._llm_prompts) == 1 and "" in self._llm_prompts:
            content: str | dict[str, str] = self._llm_prompts[""]
        else:
            content = self._llm_prompts

        viewer = ContentViewer(f"{self.title} - Prompt", content, language="markdown")
        self.app.push_screen(viewer)

    def _show_response(self) -> None:
        """Open the response viewer modal with smart language detection."""
        import json as json_module

        from osprey.interfaces.tui.widgets.content_viewer import ContentViewer

        # Single response with empty key -> pass as string; else pass dict for tabs
        if len(self._llm_responses) == 1 and "" in self._llm_responses:
            content: str | dict[str, str] = self._llm_responses[""]
        else:
            content = self._llm_responses

        # Smart language detection: JSON for structured data, markdown for text
        language = "json"  # default for T/C/O structured responses
        if isinstance(content, str):
            try:
                json_module.loads(content)
            except (json_module.JSONDecodeError, ValueError):
                # Not valid JSON, probably markdown (e.g., respond capability)
                language = "markdown"

        viewer = ContentViewer(f"{self.title} - Response", content, language=language)
        self.app.push_screen(viewer)

    def set_llm_prompt(self, prompt: str | dict[str, str]) -> None:
        """Set the LLM prompt(s) and show the prompt link.

        Args:
            prompt: The LLM prompt text (str) or dict of {capability_name: prompt_text}.
        """
        if isinstance(prompt, str):
            self._llm_prompts[""] = prompt
        else:
            self._llm_prompts.update(prompt)
        if self._mounted:
            prompt_link = self.query_one("#step-prompt-link", PromptLink)
            prompt_link.display = True

    def set_llm_response(self, response: str | dict[str, str]) -> None:
        """Set the LLM response(s) and show the response link.

        Args:
            response: The LLM response text (str) or dict of {capability_name: response_json}.
        """
        if isinstance(response, str):
            self._llm_responses[""] = response
        else:
            self._llm_responses.update(response)
        if self._mounted:
            response_link = self.query_one("#step-response-link", ResponseLink)
            response_link.display = True

    def _start_breathing(self) -> None:
        """Start the breathing animation timer."""
        if self._breathing_timer is None:
            self._breathing_timer = self.set_interval(0.4, self._breathing_tick)

    def _breathing_tick(self) -> None:
        """Update breathing animation frame."""
        if self._status != "active":
            self._stop_breathing()
            return

        frames = self.BREATHING_FRAMES
        self._breathing_index = (self._breathing_index + 1) % len(frames)
        indicator = frames[self._breathing_index]
        title_line = self.query_one("#step-title", Static)
        title_line.update(f"[bold][$accent]{indicator}[/] {self.title}[/bold]")

    def _stop_breathing(self) -> None:
        """Stop the breathing animation."""
        if self._breathing_timer:
            self._breathing_timer.stop()
            self._breathing_timer = None

    def _apply_active(self) -> None:
        """Internal: apply active state and start breathing."""
        title_line = self.query_one("#step-title", Static)
        title_line.update(f"[bold]{self.INDICATOR_ACTIVE} {self.title}[/bold]")
        self.add_class("step-active")
        self._start_breathing()

    def set_active(self) -> None:
        """Mark the step as actively processing."""
        self._status = "active"
        if self._mounted:
            self._apply_active()

    def set_complete(self, status: str = "success", output_msg: str = "") -> None:
        """Mark the step as complete.

        Args:
            status: The completion status ('success' or 'error').
            output_msg: Output message to display on the second line.
        """
        self._status = status
        self._stop_breathing()

        indicator = self.INDICATOR_SUCCESS if status == "success" else self.INDICATOR_ERROR

        # Only color the indicator, keep title bold
        # Framework steps: dim indicator on success, error indicator on failure
        if status == "success":
            title_markup = f"{indicator} [bold]{self.title}[/bold]"
        else:
            title_markup = f"[$error]{indicator}[/$error] [bold]{self.title}[/bold]"

        title_line = self.query_one("#step-title", Static)
        title_line.update(title_markup)

        self.remove_class("step-active")
        self.add_class(f"step-{status}")

        # Show logs link now that step is complete
        if self._mounted:
            logs_link = self.query_one("#step-logs-link", LogsLink)
            logs_link.display = True

        # Show output message on second line (for both success and error)
        if output_msg and self._mounted:
            self._output_message = output_msg
            output = self.query_one("#step-output", WrappedStatic)
            output.set_content(output_msg)
            output.display = True

    def add_log(
        self, message: str, status: str = "status", timestamp: datetime | None = None
    ) -> None:
        """Store log message and show logs link on first log.

        Args:
            message: The message text.
            status: The message status.
            timestamp: Optional timestamp for the log entry.
        """
        if message:
            self._log_messages.append((status, message, timestamp))
            # Show logs link as soon as first log arrives
            if self._mounted and len(self._log_messages) == 1:
                logs_link = self.query_one("#step-logs-link", LogsLink)
                logs_link.display = True
            # Track last error for potential display
            if status == "error":
                self._output_message = message
                self._last_error_msg = message
            # Direct call to registered LogViewer (if watching this step)
            viewer = getattr(self.app, "_active_log_viewer", None)
            if viewer is not None and viewer._log_source is self:
                viewer.receive_log(status, message, timestamp)

    def _get_output_width(self) -> int:
        """Get the actual width available for output text wrapping.

        Uses the step widget's own size (which is available even when
        output is hidden) and accounts for padding and prefix.
        Falls back to 80 if widget dimensions aren't available yet.
        """
        try:
            # Use step's own width (available even when output is hidden)
            width = self.size.width
            if width <= 0:
                width = self.content_size.width
            if width <= 0:
                width = 80
        except Exception:
            width = 80

        # Account for:
        # - padding: 0 2 (4 chars horizontal)
        # - "╰ " prefix (2 chars)
        return max(width - 4 - 2, 40)

    # Compatibility methods for app.py interface
    def set_input(self, text: str, mark_set: bool = True) -> None:
        """No-op for step (no IN section).

        Args:
            text: The input text (ignored).
            mark_set: Whether to mark as set (ignored).
        """
        if mark_set:
            self._input_set = True

    def set_output(self, text: str, status: str = "success") -> None:
        """Mark complete with output message.

        Args:
            text: The output text to display on second line.
            status: The completion status ('success' or 'error').
        """
        self.set_complete(status, text)

    def set_partial_output(self, text: str, status: str = "pending") -> None:
        """No-op for step (no streaming output display).

        Args:
            text: The output text (ignored).
            status: The status (ignored).
        """
        pass


class TaskExtractionStep(ProcessingStep):
    """Step widget for task extraction phase - minimal UI."""

    def __init__(self, **kwargs):
        """Initialize task extraction step."""
        super().__init__("Task Extraction", **kwargs)


class ClassificationStep(ProcessingStep):
    """Step widget for capability classification phase - minimal UI.

    Shows selected capabilities as a bulleted vertical list on completion.
    """

    def __init__(self, **kwargs):
        """Initialize classification step."""
        super().__init__("Classification", **kwargs)
        self._all_capabilities: list[str] = []
        self._selected_capabilities: list[str] = []

    def set_capabilities(self, all_caps: list[str], selected: list[str]) -> None:
        """Show selected capabilities as bulleted list.

        Args:
            all_caps: All available capabilities (stored but not displayed).
            selected: The selected/active capabilities to display.
        """
        self._all_capabilities = all_caps
        self._selected_capabilities = selected

        if selected:
            output_text = "\n".join(f"✓ {cap}" for cap in selected)
        else:
            output_text = "No capabilities selected"

        self.set_complete("success", output_text)


class WrappedLabel(Static):
    """A label that wraps text dynamically and re-wraps on resize.

    Unlike WrappedStatic, this widget is simpler and doesn't use
    initial_indent/subsequent_indent - it just wraps the raw text
    to fit the available width. Used inside TodoItem where the bullet
    is in a separate widget.
    """

    def __init__(self, text: str = "", **kwargs):
        """Initialize the wrapped label.

        Args:
            text: The text content to display and wrap.
            **kwargs: Additional arguments passed to Static.
        """
        super().__init__(**kwargs)
        self._raw_text = text
        self._mounted = False

    def on_mount(self) -> None:
        """Handle mount by scheduling initial wrap after layout."""
        self._mounted = True
        self.call_after_refresh(self._wrap_and_update)

    def on_resize(self) -> None:
        """Re-wrap text when widget is resized."""
        if self._mounted:
            self._wrap_and_update()

    def _wrap_and_update(self) -> None:
        """Wrap text to current width and update display."""
        width = self.size.width or self.content_size.width or 40
        if width > 0 and self._raw_text:
            wrapped = textwrap.fill(self._raw_text, width=width)
            self.update(wrapped)
        elif not self._raw_text:
            self.update("")

    def set_text(self, text: str) -> None:
        """Update the text content.

        Args:
            text: New text to display.
        """
        self._raw_text = text
        if self._mounted:
            self._wrap_and_update()


class TodoItem(Horizontal):
    """A single todo item with bullet and wrapping text.

    Uses CSS classes for state-based styling:
    - .todo-pending: Normal styling
    - .todo-current: Accent-colored bullet
    - .todo-done: Strikethrough + dimmed opacity

    The bullet is in a fixed-width Static, and the text is in a
    flexible WrappedLabel that handles its own wrapping. This ensures
    continuation lines align with the first line text, not the bullet.
    """

    BULLETS = {
        "pending": "☐",  # U+2610 BALLOT BOX
        "current": "▣",  # U+25A3 WHITE SQUARE CONTAINING BLACK SMALL SQUARE
        "done": "☑",  # U+2611 BALLOT BOX WITH CHECK
    }

    def __init__(self, text: str, state: str = "pending", **kwargs):
        """Initialize the todo item.

        Args:
            text: The todo item text.
            state: One of "pending", "current", or "done".
            **kwargs: Additional arguments passed to Horizontal.
        """
        super().__init__(**kwargs)
        self._text = text
        self._state = state
        self.add_class(f"todo-{state}")

    def compose(self) -> ComposeResult:
        """Compose the todo item with bullet and text."""
        yield Static(self.BULLETS[self._state], classes="todo-bullet")
        yield WrappedLabel(self._text, classes="todo-text")

    def set_state(self, state: str) -> None:
        """Update the todo item state.

        Args:
            state: New state - one of "pending", "current", or "done".
        """
        self.remove_class(f"todo-{self._state}")
        self._state = state
        self.add_class(f"todo-{state}")
        bullet = self.query_one(".todo-bullet", Static)
        bullet.update(self.BULLETS[state])


class TodoList(Vertical):
    """A list of todo items with CSS-based state styling.

    Manages a collection of TodoItem widgets, replacing them when
    set_todos() is called with new data.
    """

    def __init__(self, **kwargs):
        """Initialize the todo list.

        Args:
            **kwargs: Additional arguments passed to Vertical.
        """
        super().__init__(**kwargs)
        self._items: list[TodoItem] = []
        self._todos: list[dict] = []
        self._states: list[str] = []

    def set_todos(self, todos: list[dict], states: list[str]) -> None:
        """Set all todos, replacing existing items.

        Args:
            todos: List of todo dicts with 'task_objective' key.
            states: List of states ("pending", "current", "done").
        """
        # Store for later reference
        self._todos = todos
        self._states = states.copy()

        # Clear existing items
        for item in self._items:
            item.remove()
        self._items.clear()

        # Batch all mounts to avoid multiple layout passes
        # This prevents the "expand then shrink" visual jump
        with self.app.batch_update():
            for todo, state in zip(todos, states, strict=True):
                text = todo.get("task_objective", "")
                item = TodoItem(text, state)
                self._items.append(item)
                self.mount(item)
            # Show the list within the batch
            self.display = True

    def update_states(self, states: list[str]) -> None:
        """Update states in-place without rebuilding widgets.

        Only updates items whose state has changed. More efficient than
        calling set_todos() when the structure hasn't changed.

        Args:
            states: New list of states ("pending", "current", "done").
        """
        if len(states) != len(self._items):
            # Mismatch - can't update in place
            return

        with self.app.batch_update():
            for item, old_state, new_state in zip(self._items, self._states, states, strict=True):
                if old_state != new_state:
                    item.set_state(new_state)
        self._states = states.copy()


class OrchestrationStep(ProcessingStep):
    """Step widget for orchestration/planning phase - minimal UI.

    Shows "Planning" title with breathing indicator while active,
    and the todo list (initial plan) on completion.

    Uses TodoList widget for proper CSS-based styling of todo items.
    """

    def __init__(self, **kwargs):
        """Initialize orchestration step."""
        super().__init__("Planning", **kwargs)
        self._plan_steps: list[dict] = []

    def compose(self) -> ComposeResult:
        """Compose the step with title line, links, output, and todo list."""
        with Horizontal(id="step-header"):
            yield Static(
                f"[bold]{self.INDICATOR_PENDING} {self.title}[/bold]",
                id="step-title",
            )
            yield LogsLink("logs", id="step-logs-link")
            yield PromptLink("prompt", id="step-prompt-link")
            yield ResponseLink("response", id="step-response-link")
        # Summary line with "╰" prefix (like other steps)
        yield WrappedStatic(
            "",
            initial_indent=f"  {self.OUTPUT_GUIDE} ",
            subsequent_indent="    ",
            id="step-output",
        )
        # TodoList with indent to align with text after "╰ "
        yield TodoList(id="todo-list")

    def on_mount(self) -> None:
        """Apply pending state after widget is mounted."""
        self._mounted = True
        # Hide output line initially
        output = self.query_one("#step-output", WrappedStatic)
        output.display = False
        # Hide logs link initially (shown when logs available)
        logs_link = self.query_one("#step-logs-link", LogsLink)
        logs_link.display = False
        # Hide prompt/response links initially (shown when data available)
        prompt_link = self.query_one("#step-prompt-link", PromptLink)
        prompt_link.display = False
        response_link = self.query_one("#step-response-link", ResponseLink)
        response_link.display = False
        # Show links if data was set before mounting (race condition fix)
        if self._log_messages:
            logs_link.display = True
        if self._llm_prompts:
            prompt_link.display = True
        if self._llm_responses:
            response_link.display = True
        # Start breathing animation
        self._start_breathing()

    def set_plan(self, steps: list[dict]) -> None:
        """Store execution plan steps and show initial todo list.

        Args:
            steps: List of execution plan step dicts with
                   'task_objective' and 'capability' keys.
        """
        self._plan_steps = steps
        count = len(steps)
        plural = "s" if count != 1 else ""

        # Stop breathing animation
        self._stop_breathing()

        # Update title - keep as "Planning", just change indicator
        title_markup = f"{self.INDICATOR_SUCCESS} [bold]{self.title}[/bold]"
        title_line = self.query_one("#step-title", Static)
        title_line.update(title_markup)

        # Show summary in output line with "╰" prefix
        output = self.query_one("#step-output", WrappedStatic)
        output.set_content(f"{count} step{plural} planned")
        output.display = True

        # Update todo list with all pending
        todo_list = self.query_one("#todo-list", TodoList)
        todo_list.set_todos(steps, ["pending"] * count)

        # Update styling
        self.remove_class("step-active")
        self.add_class("step-success")

        # Show logs link if we have logs
        if self._mounted and self._log_messages:
            logs_link = self.query_one("#step-logs-link", LogsLink)
            logs_link.display = True


class ExecutionStep(ProcessingStep):
    """Step widget for execution phase - same style as T/C/O steps.

    Displays capability execution as a single-line step with:
    - Title: Just the capability name (no "Step N:" prefix)
    - Logs link: Opens modal with execution logs
    - Prompt/response links: Only shown if step is LLM-driven
    - Result: Shown as single line with "╰" prefix

    Think of it as a "tool call" widget: tool name, input (from todo),
    and output (result).
    """

    def __init__(self, capability: str, **kwargs):
        """Initialize execution step.

        Args:
            capability: The capability name being executed.
        """
        super().__init__(capability, **kwargs)
        self.capability = capability

    def set_complete(self, status: str = "success", output_msg: str = "") -> None:
        """Override with conditional indicator coloring.

        Smart steps (respond/clarify): plain indicator, CSS handles muted color like T/C/O.
        Dumb steps (capabilities): green indicator, stays bright.
        """
        self._status = status
        self._stop_breathing()

        indicator = self.INDICATOR_SUCCESS if status == "success" else self.INDICATOR_ERROR

        # Smart steps: plain indicator (CSS handles muted color like T/C/O)
        # Dumb steps: green indicator (stays bright)
        if self.has_class("smart-step"):
            # Match T/C/O style - plain indicator, CSS applies muted color
            if status == "success":
                title_markup = f"{indicator} [bold]{self.title}[/bold]"
            else:
                title_markup = f"[$error]{indicator}[/$error] [bold]{self.title}[/bold]"
        else:
            # Dumb capability steps - green indicator
            if status == "success":
                title_markup = f"[$success]{indicator}[/$success] [bold]{self.title}[/bold]"
            else:
                title_markup = f"[$error]{indicator}[/$error] [bold]{self.title}[/bold]"

        title_line = self.query_one("#step-title", Static)
        title_line.update(title_markup)

        if output_msg:
            output = self.query_one("#step-output", WrappedStatic)
            output.set_content(output_msg)
            output.display = True

        self.remove_class("step-active")
        self.add_class(f"step-{status}")

    def set_partial_output(self, text: str, status: str = "pending") -> None:
        """Show partial output while block is still active (keeps breathing).

        Unlike set_output()/set_complete(), this doesn't mark the block as complete.
        Used for real-time status updates during streaming (e.g., "Response streaming...").

        Args:
            text: The output text to display.
            status: The status for styling (ignored, block stays active).
        """
        if self._mounted and text:
            output = self.query_one("#step-output", WrappedStatic)
            output.set_content(text)
            output.display = True

    def set_llm_prompt(self, prompt: str | dict[str, str]) -> None:
        """Override to mark as smart/infrastructure step."""
        super().set_llm_prompt(prompt)
        self.add_class("smart-step")  # Smart steps get grayed out styling


class TaskExtractionBlock(ProcessingBlock):
    """Block for task extraction phase (deprecated, use TaskExtractionStep)."""

    # Expanded header text (overrides base class)
    EXPANDED_HEADER = "Extracted task"

    def __init__(self, **kwargs):
        """Initialize task extraction block."""
        super().__init__("Task Extraction", **kwargs)


class ClassificationBlock(ProcessingBlock):
    """Block for capability classification phase with simple text output.

    .. deprecated::
        Use ClassificationStep instead for minimal UI.
    """

    # Expanded header text
    EXPANDED_HEADER = "Activated capabilities"

    def __init__(self, **kwargs):
        """Initialize classification block."""
        super().__init__("Classification", **kwargs)
        self._all_capabilities: list[str] = []
        self._selected_capabilities: list[str] = []

    def set_capabilities(self, all_caps: list[str], selected: list[str]) -> None:
        """Show capabilities as simple text with checkmarks (fast rendering).

        Args:
            all_caps: All available capabilities.
            selected: The selected/active capabilities.
        """
        self._all_capabilities = all_caps
        self._selected_capabilities = selected

        # Format full list with checkmarks - gray out unselected
        lines = []
        for cap in all_caps:
            if cap in selected:
                lines.append(f"✓ {cap}")
            else:
                # Gray out unselected capabilities using Rich dim markup
                lines.append(f"[dim]· {cap}[/dim]")

        output_text = "\n".join(lines) if lines else "No capabilities"

        # Call parent's set_output first
        self.set_output(output_text)

        # Override with custom preview (after set_output overwrites it)
        if selected:
            preview = f"Activated: {', '.join(selected)}"
        else:
            preview = "No capabilities activated"
        self._output_preview = self._get_preview(preview)

        # Update the collapsible title with our custom preview
        if self._mounted:
            output = self.query_one("#block-output", Collapsible)
            output.title = f"[bold]OUT[/bold]   ▸ {self._output_preview}"


class OrchestrationBlock(ProcessingBlock):
    """Block for orchestration/planning phase."""

    # Expanded header text (overrides base class)
    EXPANDED_HEADER = "Planned steps"

    def __init__(self, **kwargs):
        """Initialize orchestration block."""
        super().__init__("Orchestration", **kwargs)

    def set_plan(self, steps: list[dict]) -> None:
        """Show execution plan steps.

        Uses simple bullet-style formatting (like LOG section) - numbers act as
        bullets and terminal handles soft-wrapping naturally. No textwrap needed.

        Args:
            steps: List of execution plan step dicts.
        """
        lines = []
        for i, step in enumerate(steps, 1):
            objective = step.get("task_objective", "")
            capability = step.get("capability", "")
            # Simple format: "1. objective [capability]" - no textwrap
            lines.append(f"{i}. {objective} [{capability}]")

        self.set_output("\n".join(lines) if lines else "No steps")
