"""Osprey TUI Application.

A Terminal User Interface for the Osprey Agent Framework built with Textual.
"""

import asyncio
import logging
import re
import textwrap
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from textual import work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.events import Key
from textual.message import Message
from textual.widgets import Collapsible, Footer, Header, Static, TextArea

from osprey.graph import create_graph
from osprey.infrastructure.gateway import Gateway
from osprey.registry import get_registry, initialize_registry
from osprey.utils.config import get_config_value, get_full_configuration

# Pattern to detect router execution step messages
EXEC_STEP_PATTERN = re.compile(r"Executing step (\d+)/(\d+) - capability: (\w+)")

# Components that create Task Preparation blocks (allowlist)
TASK_PREP_COMPONENTS = {"task_extraction", "classifier", "orchestrator"}


class QueueLogHandler(logging.Handler):
    """Routes Python log records to TUI event queue.

    Extracts ALL metadata from ComponentLogger's extra dict:
    - raw_message, log_type (for LOG section)
    - task, capabilities, steps, phase (for block lifecycle)
    """

    # Fields to extract from LogRecord extra dict
    EXTRA_FIELDS = [
        "task",
        "capabilities",
        "capability_names",
        "steps",
        "phase",
        "step_num",
        "step_name",
    ]

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """Initialize the handler.

        Args:
            queue: The asyncio queue to send events to.
            loop: The event loop for thread-safe queue operations.
        """
        super().__init__()
        self.queue = queue
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the TUI event queue.

        Args:
            record: The log record to emit.
        """
        # Extract raw message from extra (set by ComponentLogger)
        raw_msg = getattr(record, "raw_message", None)
        log_type = getattr(record, "log_type", record.levelname.lower())

        # Skip if no raw message (not from ComponentLogger)
        if raw_msg is None:
            return

        event = {
            "event_type": "log",  # ALL TUI events are "log" type
            "level": log_type,
            "message": raw_msg,
            "component": record.name,
        }

        # Extract ALL streaming data fields
        for key in self.EXTRA_FIELDS:
            val = getattr(record, key, None)
            if val is not None:
                event[key] = val

        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:
            pass  # Event loop closed


class ChatMessage(Static):
    """A single chat message widget styled as a card/block."""

    def __init__(self, content: str, role: str = "user", **kwargs):
        """Initialize a chat message.

        Args:
            content: The message content.
            role: The role (user or assistant).
        """
        super().__init__(**kwargs)
        self.message_content = content
        self.role = role
        self.add_class(f"message-{role}")

    def compose(self) -> ComposeResult:
        """Compose the message with content and role label."""
        yield Static(self.message_content, classes="message-content")
        yield Static(self.role, classes="role-label")


class StreamingMessage(Static):
    """A message that updates in real-time during streaming."""

    def __init__(self, **kwargs):
        """Initialize streaming message."""
        super().__init__(**kwargs)
        self.add_class("message-assistant")
        self._mounted = False
        self._pending_status: str | None = None
        self._pending_content: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the streaming message with placeholders."""
        yield Static("", classes="message-content")
        yield Static("", classes="streaming-status")
        yield Static("assistant", classes="role-label")

    def on_mount(self) -> None:
        """Apply pending state after widget is mounted."""
        self._mounted = True
        if self._pending_status is not None:
            self._apply_status(self._pending_status)
        if self._pending_content is not None:
            self._apply_content(self._pending_content)

    def _apply_status(self, status: str) -> None:
        """Internal: apply status to widget."""
        status_widget = self.query_one(".streaming-status", Static)
        status_widget.update(f"🔄 {status}")

    def _apply_content(self, content: str) -> None:
        """Internal: apply content and clear status."""
        content_widget = self.query_one(".message-content", Static)
        content_widget.update(content)
        status_widget = self.query_one(".streaming-status", Static)
        status_widget.update("")

    def update_status(self, status: str) -> None:
        """Update the status line during streaming."""
        self._pending_status = status
        if self._mounted:
            self._apply_status(status)

    def finalize(self, content: str) -> None:
        """Finalize with the actual response content."""
        self._pending_content = content
        if self._mounted:
            self._apply_content(content)


class DebugBlock(Static):
    """Debug block to show streaming events for debugging."""

    def __init__(self, **kwargs):
        """Initialize debug block."""
        super().__init__(**kwargs)
        self.add_class("debug-block")
        self._events: list[str] = []
        self._mounted = False

    def compose(self) -> ComposeResult:
        """Compose the debug block with header and scrollable content."""
        yield Static("📊 Debug Events:", classes="debug-header")
        yield ScrollableContainer(
            Static("", classes="debug-content", id="debug-content"),
            id="debug-scroll",
            classes="debug-scroll",
        )

    def on_mount(self) -> None:
        """Apply pending state after widget is mounted."""
        self._mounted = True
        self._update_display()

    def add_event(self, chunk: dict) -> None:
        """Add an event to the debug display."""
        component = chunk.get("component", "?")
        phase = chunk.get("phase", "?")
        event_type = chunk.get("event_type", "?")
        complete = chunk.get("complete", False)
        msg = chunk.get("message", "")[:40]

        # success event or explicit complete flag = DONE
        is_done = (event_type == "success") or complete
        status = "✅ DONE" if is_done else "▶️ START"
        line = f"[{component}] {status} | {phase} | {msg}"
        self._events.append(line)

        if self._mounted:
            self._update_display()

    def _update_display(self) -> None:
        """Update the display with ALL events (scrollable)."""
        content = self.query_one("#debug-content", Static)
        content.update("\n".join(self._events))  # Show ALL events
        # Auto-scroll to bottom
        scroll = self.query_one("#debug-scroll", ScrollableContainer)
        scroll.scroll_end(animate=False)

    def clear(self) -> None:
        """Clear all events."""
        self._events = []
        if self._mounted:
            self._update_display()


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
        # LOG section - streaming messages for debugging
        self._log_messages: list[tuple[str, str]] = []  # [(status, message), ...]
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
        yield Static("", classes="block-input", id="block-input")
        # Full-width separator (will be truncated by container)
        yield Static("─" * 120, classes="block-separator", id="block-separator")
        # OUT section - final outcome only (hide built-in arrows)
        yield Collapsible(
            Static("", id="block-output-content"),
            title="",
            collapsed=True,
            collapsed_symbol="",
            expanded_symbol="",
            id="block-output",
        )
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
            self._apply_output(*self._pending_output)

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
        header.update(f"{indicator} {self.title}")

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
        """Internal: apply input text with hanging indent for wrapped lines."""
        input_widget = self.query_one("#block-input", Static)

        # Prefix: "  IN    " = 8 visible chars (2 spaces + IN + 4 spaces)
        prefix = "  [bold]IN[/bold]    "
        indent = "          "  # 10 spaces for continuation (aligns with text after IN)

        # Wrap text with hanging indent
        wrapped = textwrap.fill(
            text,
            width=80,
            initial_indent="",
            subsequent_indent=indent,
        )

        input_widget.update(f"{prefix}{wrapped}")

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
        if event.collapsible.id == "block-output":
            event.collapsible.title = f"[bold]OUT[/bold]   ▾ {self.EXPANDED_HEADER}"
        elif event.collapsible.id == "block-log":
            event.collapsible.title = f"[bold]LOG[/bold]   ▾ Streaming logs"

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        """Show collapsed arrow with preview."""
        if event.collapsible.id == "block-output":
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

    def add_log(self, message: str, status: str = "status") -> None:
        """Add a message to the LOG section.

        Args:
            message: The message text.
            status: The message status ('status', 'success', 'error', 'warning').
        """
        if message:
            self._log_messages.append((status, message))
            self._update_log_display()
            # Track last error message for OUT section on block close
            if status == "error":
                self._last_error_msg = message

    def _format_log_messages(self) -> str:
        """Format all log messages with status symbols.

        Uses Textual CSS theme variables for colors that adapt with theme changes.
        See: https://textual.textualize.io/guide/content/
        """
        if not self._log_messages:
            return ""
        lines = []
        for msg_status, msg in self._log_messages:
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
            if color:
                lines.append(f"[{color}]{prefix} {msg}[/{color}]")
            else:
                lines.append(f"{prefix} {msg}")
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


class TaskExtractionBlock(ProcessingBlock):
    """Block for task extraction phase."""

    # Expanded header text (overrides base class)
    EXPANDED_HEADER = "Extracted task"

    def __init__(self, **kwargs):
        """Initialize task extraction block."""
        super().__init__("Task Extraction", **kwargs)


class ClassificationBlock(ProcessingBlock):
    """Block for capability classification phase with simple text output."""

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
        """Show execution plan steps with proper wrapping alignment.

        Args:
            steps: List of execution plan step dicts.
        """
        lines = []
        # Prefix width: " 1. " = 4 chars, use 5 for cleaner indent
        indent = "     "  # 5 spaces for continuation lines
        width = 90  # Reasonable wrap width

        for i, step in enumerate(steps, 1):
            objective = step.get("task_objective", "")
            capability = step.get("capability", "")

            prefix = f"{i:2}. "
            content = f"{objective} [{capability}]"

            # Wrap with hanging indent for continuation lines
            wrapped = textwrap.fill(
                content, width=width, initial_indent=prefix, subsequent_indent=indent
            )
            lines.append(wrapped)

        self.set_output("\n".join(lines) if lines else "No steps")


class ExecutionStepBlock(ProcessingBlock):
    """Block for a single execution step in the plan."""

    # Expanded header text (overrides base class)
    EXPANDED_HEADER = "Execution result"

    def __init__(self, step_number: int, capability: str, objective: str, **kwargs):
        """Initialize execution step block.

        Args:
            step_number: The 1-based step number.
            capability: The capability being executed.
            objective: The task objective for this step.
        """
        super().__init__(f"Step {step_number}: {capability}", **kwargs)
        self.step_number = step_number
        self.capability = capability
        self.objective = objective

    def on_mount(self) -> None:
        """Apply pending state and show objective as input."""
        super().on_mount()
        # Show objective as input when mounted
        if self.objective:
            self.set_input(self.objective)


class ChatDisplay(ScrollableContainer):
    """Scrollable container for chat messages and processing blocks."""

    def __init__(self, **kwargs):
        """Initialize chat display with block tracking."""
        super().__init__(**kwargs)
        self._current_blocks: dict[str, ProcessingBlock] = {}
        # Track which START events we've seen (for deferred block creation)
        self._seen_start_events: set[str] = set()
        # Track attempt index per component (for retry/reclassification)
        self._component_attempt_index: dict[str, int] = {}
        # Track components that need retry (set by WARNING events)
        self._retry_triggered: set[str] = set()
        # Queue for messages that arrive before block is created
        # Format: {component: [(event_type, message, chunk), ...]}
        self._pending_messages: dict[str, list[tuple[str, str, dict]]] = {}
        # Debug block for showing events (enabled for debugging)
        self._debug_enabled = False
        self._debug_block: DebugBlock | None = None
        # Event queue for decoupling streaming from rendering
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def start_new_query(self, user_query: str) -> None:
        """Reset blocks for a new query and add user message.

        Args:
            user_query: The user's input message.
        """
        self._current_blocks = {}
        self._seen_start_events = set()
        self._component_attempt_index = {}
        self._retry_triggered = set()
        self._pending_messages = {}
        if self._debug_block:
            self._debug_block.clear()
        self.add_message(user_query, "user")

    def get_or_create_debug_block(self) -> DebugBlock | None:
        """Get or create the debug block for event visualization.

        Returns None if debug is disabled.
        """
        if not self._debug_enabled:
            return None
        if not self._debug_block:
            self._debug_block = DebugBlock()
            self.mount(self._debug_block)
            self.scroll_end(animate=False)
        return self._debug_block

    def get_or_create_block(self, block_type: str, **kwargs) -> ProcessingBlock | None:
        """Get existing block or create new one.

        Args:
            block_type: Type of block (task_extraction, classifier, etc.)
            **kwargs: Additional arguments for block creation.

        Returns:
            The processing block, or None if invalid type.
        """
        if block_type not in self._current_blocks:
            block_class = {
                "task_extraction": TaskExtractionBlock,
                "classifier": ClassificationBlock,
                "orchestrator": OrchestrationBlock,
            }.get(block_type)

            if block_class:
                block = block_class()
            else:
                return None

            self._current_blocks[block_type] = block
            self.mount(block)
            self.scroll_end(animate=False)

        return self._current_blocks[block_type]

    def add_message(self, content: str, role: str = "user") -> None:
        """Add a message to the chat display.

        Args:
            content: The message content.
            role: The role (user or assistant).
        """
        message = ChatMessage(content, role)
        self.mount(message)
        self.scroll_end(animate=False)

    def add_streaming_message(self) -> StreamingMessage:
        """Add a streaming message placeholder that updates in real-time."""
        message = StreamingMessage()
        self.mount(message)
        self.scroll_end(animate=False)
        return message


class ChatInput(TextArea):
    """Multi-line text input for chat messages.

    Press Enter to send, Option+Enter (Alt+Enter) for new line.
    """

    class Submitted(Message):
        """Event posted when user submits input."""

        def __init__(self, value: str):
            super().__init__()
            self.value = value

    def __init__(self, **kwargs):
        """Initialize the chat input."""
        super().__init__(**kwargs)
        self.show_line_numbers = False

    def _on_key(self, event: Key) -> None:
        """Handle key events - Enter submits, Option+Enter for newline."""
        if event.key == "enter":
            # Enter = submit
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
            return
        elif event.key == "alt+enter":
            # Option+Enter (Alt+Enter) = newline
            event.prevent_default()
            event.stop()
            self.insert("\n")
            self.scroll_cursor_visible()
            return
        # Word movement (Alt+Arrow)
        elif event.key == "alt+left":
            event.prevent_default()
            event.stop()
            self.action_cursor_word_left()
            return
        elif event.key == "alt+right":
            event.prevent_default()
            event.stop()
            self.action_cursor_word_right()
            return
        # Line start/end (Cmd+Arrow → ctrl in terminal)
        elif event.key == "ctrl+left":
            event.prevent_default()
            event.stop()
            self.action_cursor_line_start()
            return
        elif event.key == "ctrl+right":
            event.prevent_default()
            event.stop()
            self.action_cursor_line_end()
            return
        # Document start/end (Cmd+Up/Down)
        elif event.key == "ctrl+up":
            event.prevent_default()
            event.stop()
            self.move_cursor((0, 0))
            return
        elif event.key == "ctrl+down":
            event.prevent_default()
            event.stop()
            self.move_cursor(self.document.end)
            return
        # Let parent handle all other keys
        super()._on_key(event)


class OspreyTUI(App):
    """Osprey Terminal User Interface.

    A TUI for interacting with the Osprey Agent Framework.
    """

    TITLE = "Osprey TUI"
    SUB_TITLE = "AI Agent Framework"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, config_path: str = "config.yml"):
        """Initialize the TUI.

        Args:
            config_path: Path to the configuration file.
        """
        super().__init__()
        self.config_path = config_path

        # Generate unique thread ID for this session
        self.thread_id = f"tui_session_{uuid.uuid4().hex[:8]}"

        # Will be initialized in on_mount
        self.graph = None
        self.gateway = None
        self.base_config = None
        self.current_state = None
        # Shared data for passing info between blocks (T→C→O)
        self._shared_data: dict[str, Any] = {}  # {task, capability_names, ...}

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Vertical(
            ChatDisplay(id="chat-display"), ChatInput(id="chat-input"), id="main-content"
        )
        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount event - initialize agent components."""
        # Initialize registry
        initialize_registry(config_path=self.config_path)
        registry = get_registry()

        # Cache all capability names for classification block (avoids delay later)
        self.all_capability_names = registry.get_stats()["capability_names"]

        # Create checkpointer and graph
        checkpointer = MemorySaver()
        self.graph = create_graph(registry, checkpointer=checkpointer)
        self.gateway = Gateway()

        # Build base config
        configurable = get_full_configuration(config_path=self.config_path).copy()
        configurable.update(
            {
                "user_id": "tui_user",
                "thread_id": self.thread_id,
                "chat_id": "tui_chat",
                "session_id": self.thread_id,
                "interface_context": "tui",
            }
        )

        recursion_limit = get_config_value("execution_control.limits.graph_recursion_limit")
        self.base_config = {
            "configurable": configurable,
            "recursion_limit": recursion_limit,
        }

        # Focus the input field when app starts
        self.query_one("#chat-input", ChatInput).focus()

        # Add welcome message
        chat_display = self.query_one("#chat-display", ChatDisplay)
        chat_display.add_message(
            "Welcome to Osprey TUI! Enter to send, Option+Enter for newline.",
            "assistant",
        )

        # Set up log handler to capture Python logs via single-channel architecture
        # All logs from ComponentLogger include raw_message and log_type in extra dict
        loop = asyncio.get_event_loop()
        self._log_handler = QueueLogHandler(chat_display._event_queue, loop)
        self._log_handler.setLevel(logging.DEBUG)  # Capture all levels

        # Attach to root logger - captures ALL logs from any component
        # QueueLogHandler.emit() filters to only ComponentLogger logs (via raw_message check)
        logging.getLogger().addHandler(self._log_handler)

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle input submission.

        Args:
            event: The input submitted event.
        """
        user_input = event.value.strip()

        if not user_input:
            return

        # Handle quit commands locally
        if user_input.lower() in ("bye", "end", "quit", "exit"):
            self.exit()
            return

        # Process with agent (async worker) - start_new_query called inside
        self.process_with_agent(user_input)

    async def _consume_events(self, user_query: str, chat_display: ChatDisplay) -> None:
        """Event consumer - single gateway for all TUI block updates.

        SINGLE-CHANNEL ARCHITECTURE: Only processes log events (event_type == "log").
        Stream events from Gateway are ignored - all data comes from Python logs.

        Block lifecycle: Open on first log, close when DIFFERENT component arrives.
        Only one Task Preparation block is active at any time.

        Args:
            user_query: The original user query.
            chat_display: The chat display widget.
        """
        # Track single active block (for Task Preparation phase)
        current_component: str | None = None
        current_block: ProcessingBlock | None = None
        # Track execution state for routing capability logs to correct step block
        execution_started = False
        current_capability: str | None = None
        current_execution_step = 1

        while True:
            try:
                chunk = await chat_display._event_queue.get()

                event_type = chunk.get("event_type", "")

                # SINGLE-CHANNEL: Only process log events - ignore stream events
                if event_type != "log":
                    chat_display._event_queue.task_done()
                    continue

                component = chunk.get("component", "")
                level = chunk.get("level", "info")
                msg = chunk.get("message", "")
                phase = chunk.get("phase", "")

                # Skip if no component
                if not component:
                    chat_display._event_queue.task_done()
                    continue

                # DEBUG: Log events to debug block (if enabled)
                debug_block = chat_display.get_or_create_debug_block()
                if debug_block:
                    debug_block.add_event(chunk)

                # Determine phase using allowlist approach
                # Only known components create blocks; skip infrastructure logs
                if not phase:
                    if component in TASK_PREP_COMPONENTS:
                        phase = "Task Preparation"
                    elif component == "router":
                        # Router needs special handling below (EXEC_STEP_PATTERN)
                        pass
                    elif execution_started and component == current_capability:
                        # Capability logs for active execution step
                        phase = "Execution"
                    else:
                        # Skip infrastructure/unknown components (gateway, cli, etc.)
                        chat_display._event_queue.task_done()
                        continue

                # Handle router messages - only process execution step patterns
                if component == "router":
                    if msg:
                        match = EXEC_STEP_PATTERN.search(msg)
                        if match:
                            phase = "Execution"
                            # Parse step info for _handle_execution_event
                            step_num = int(match.group(1))
                            chunk["step"] = step_num
                            chunk["total_steps"] = int(match.group(2))
                            component = match.group(3)  # Use capability as component
                            # Track execution state for subsequent capability logs
                            execution_started = True
                            current_capability = component  # e.g., "current_weather"
                            current_execution_step = step_num
                        else:
                            # Skip router messages that don't match execution step pattern
                            chat_display._event_queue.task_done()
                            continue
                    else:
                        # Skip router messages with no message
                        chat_display._event_queue.task_done()
                        continue

                # Route by phase
                if phase == "Task Preparation":
                    current_component, current_block = self._consume_task_prep_log(
                        chunk,
                        component,
                        level,
                        msg,
                        user_query,
                        chat_display,
                        current_component,
                        current_block,
                    )
                elif phase == "Execution":
                    # Close last Task Prep block when Execution starts
                    if current_block and current_component in (
                        "task_extraction",
                        "classifier",
                        "orchestrator",
                    ):
                        self._close_task_prep_block(current_block, current_component)
                        current_component, current_block = None, None
                    # Inject current step for capability logs that don't have step field
                    if "step" not in chunk:
                        chunk["step"] = current_execution_step
                    # Execution phase uses different approach (multiple step blocks)
                    is_complete = (level == "success") or chunk.get("complete", False)
                    self._handle_execution_event(chunk, component, is_complete, level, chat_display)

                chat_display._event_queue.task_done()
            except asyncio.CancelledError:
                # Close last active block when stream ends
                if current_block:
                    self._close_task_prep_block(current_block, current_component)
                break

    def _consume_task_prep_log(
        self,
        chunk: dict,
        component: str,
        level: str,
        msg: str,
        user_query: str,
        display: ChatDisplay,
        current_component: str | None,
        current_block: ProcessingBlock | None,
    ) -> tuple[str | None, ProcessingBlock | None]:
        """Handle Task Preparation log events in the consumer.

        SINGLE-CHANNEL: All data comes from log events (via QueueLogHandler).
        Block lifecycle:
        - Open: First log for a component
        - Close: When log from DIFFERENT component arrives

        Args:
            chunk: The log event data (from QueueLogHandler).
            component: The component name.
            level: The log level/type (status, success, error, info, etc.).
            msg: The message text.
            user_query: The original user query.
            display: The chat display widget.
            current_component: Currently active component (or None).
            current_block: Currently active block (or None).

        Returns:
            Tuple of (new_current_component, new_current_block).
        """
        # 1. Component transition: close current block when different component arrives
        # (Same component logs all go to the same block - no retry detection needed)
        if component != current_component and current_block:
            self._close_task_prep_block(current_block, current_component)
            current_component, current_block = None, None

        # 3. Create new block if needed (on first log from this component)
        if current_block is None and component:
            block = self._create_task_prep_block(component, user_query, display)
            if block:
                current_component = component
                current_block = block
                block.set_active()
                # Set initial IN for task_extraction
                if component == "task_extraction":
                    block.set_input(user_query)
                    block._data["user_query"] = user_query

        if not current_block:
            return current_component, current_block

        # 4. Extract data into block._data dict (from log event's extra fields)
        for key in ["task", "capabilities", "capability_names", "steps", "user_query"]:
            if key in chunk:
                current_block._data[key] = chunk[key]

        # 5. Add to LOG section
        if msg:
            current_block.add_log(msg, status=level)

        # 6. Real-time IN update (when data becomes available)
        self._update_input_from_data(current_block, component)

        # 7. Real-time OUT update on success/error (but DON'T close block)
        if level == "success":
            self._update_output_from_data(current_block, component, chunk)
        elif level == "error":
            error_msg = msg if msg else f"{component} error"
            current_block.set_output(error_msg, status="error")

        return current_component, current_block

    def _close_task_prep_block(self, block: ProcessingBlock, component: str | None) -> None:
        """Close a Task Preparation block by setting its final output.

        Args:
            block: The block to close.
            component: The component name.
        """
        # If block doesn't have output set yet, set a default
        if block._status == "active":
            # Use data from _data dict if available
            data = block._data
            if component == "task_extraction":
                task = data.get("task", "")
                block.set_output(task if task else "Task extracted")
            elif component == "classifier":
                caps = data.get("capability_names", [])
                if caps and isinstance(block, ClassificationBlock):
                    block.set_capabilities(self.all_capability_names, caps)
                else:
                    block.set_output("Classification complete")
            elif component == "orchestrator":
                steps = data.get("steps", [])
                if steps and isinstance(block, OrchestrationBlock):
                    block.set_plan(steps)
                elif block._last_error_msg:
                    # Use last error message for failed orchestration
                    block.set_output(block._last_error_msg, status="error")
                else:
                    block.set_output("Planning complete")
            else:
                block.set_output("Complete")

    def _create_task_prep_block(
        self, component: str, user_query: str, display: ChatDisplay
    ) -> ProcessingBlock | None:
        """Create and mount a Task Preparation block.

        Handles retry numbering by checking existing blocks in display.

        Args:
            component: The component name.
            user_query: The original user query.
            display: The chat display widget.

        Returns:
            The created block, or None if invalid component.
        """
        # Get current attempt index for this component
        attempt_idx = display._component_attempt_index.get(component, 0)

        # Check if we need to increment (existing block is complete)
        block_key = f"{component}_{attempt_idx}"
        existing = display._current_blocks.get(block_key)
        if existing and existing._status in ("success", "error"):
            # Previous block exists and is complete - increment for retry
            attempt_idx += 1
            display._component_attempt_index[component] = attempt_idx
            block_key = f"{component}_{attempt_idx}"

        # Determine block class and title
        block_classes = {
            "task_extraction": (TaskExtractionBlock, "Task Extraction"),
            "classifier": (ClassificationBlock, "Classification"),
            "orchestrator": (OrchestrationBlock, "Orchestration"),
        }

        if component not in block_classes:
            return None

        block_class, base_title = block_classes[component]

        # Add retry number to title if not first attempt
        title = base_title if attempt_idx == 0 else f"{base_title} (retry #{attempt_idx})"

        # Create and mount the block
        block = block_class()
        block.title = title
        display._current_blocks[block_key] = block
        display.mount(block)
        display.scroll_end(animate=False)

        # Initialize block._data from shared_data (for IN sections)
        if component == "classifier":
            if "task" in self._shared_data:
                block._data["task"] = self._shared_data["task"]
        elif component == "orchestrator":
            if "task" in self._shared_data:
                block._data["task"] = self._shared_data["task"]
            if "capability_names" in self._shared_data:
                block._data["capabilities"] = self._shared_data["capability_names"]

        return block

    def _update_input_from_data(self, block: ProcessingBlock, component: str) -> None:
        """Update block IN section from _data dict when data is available.

        Args:
            block: The processing block.
            component: The component name.
        """
        # Skip if already set from streaming
        if block._input_set:
            return

        data = block._data

        if component == "task_extraction":
            # T:IN = user_query (already set on creation)
            pass
        elif component == "classifier":
            # C:IN = task
            task = data.get("task", "")
            if task:
                block.set_input(task)
        elif component == "orchestrator":
            # O:IN = task → [caps]
            task = data.get("task", "")
            caps = data.get("capabilities", [])
            if task and caps:
                block.set_input(f"{task} → [{', '.join(caps)}]")
            elif task:
                # Task available but no caps yet - show partial
                block.set_input(task, mark_set=False)

    def _update_output_from_data(self, block: ProcessingBlock, component: str, chunk: dict) -> None:
        """Update block OUT section from _data dict on completion.

        Also updates _shared_data for passing info to subsequent blocks.

        Args:
            block: The processing block.
            component: The component name.
            chunk: The completion event chunk.
        """
        data = block._data
        msg = chunk.get("message", "")

        if component == "task_extraction":
            # T:OUT = task
            task = data.get("task", "")
            block.set_output(task if task else (msg if msg else "Task extracted"))
            # Save task to shared_data for C and O blocks
            if task:
                self._shared_data["task"] = task

        elif component == "classifier":
            # C:OUT = capability checklist
            selected_caps = data.get("capability_names", [])
            if selected_caps and isinstance(block, ClassificationBlock):
                block.set_capabilities(self.all_capability_names, selected_caps)
                # Save capability_names to shared_data for O block
                self._shared_data["capability_names"] = selected_caps
            else:
                block.set_output(msg if msg else "Classification complete")

        elif component == "orchestrator":
            # O:OUT = planned steps
            steps = data.get("steps", [])
            if steps and isinstance(block, OrchestrationBlock):
                block.set_plan(steps)
            else:
                block.set_output(msg if msg else "Planning complete")

    @work(exclusive=True)
    async def process_with_agent(self, user_input: str) -> None:
        """Process user input through Gateway and stream response."""
        chat_display = self.query_one("#chat-display", ChatDisplay)

        # Start new query - resets blocks and adds user message
        chat_display.start_new_query(user_input)

        # Clear shared data and cached plan from previous query
        self._shared_data = {}
        self._cached_plan = None
        for attr in list(vars(self)):
            if attr.startswith("_step_") and attr.endswith("_msg"):
                delattr(self, attr)

        try:
            # Process through Gateway
            result = await self.gateway.process_message(
                user_input,
                self.graph,
                self.base_config,
            )

            if result.error:
                chat_display.add_message(f"Error: {result.error}", "assistant")
                return

            # Determine input for streaming
            input_data = result.resume_command if result.resume_command else result.agent_state

            if input_data is None:
                return

            # Start event consumer before streaming
            consumer_task = asyncio.create_task(self._consume_events(user_input, chat_display))

            try:
                # Stream events to queue (consumer processes them)
                async for chunk in self.graph.astream(
                    input_data,
                    config=self.base_config,
                    stream_mode="custom",
                ):
                    await chat_display._event_queue.put(chunk)

                # Wait for queue to be fully processed
                await chat_display._event_queue.join()
            finally:
                # Cancel consumer when done
                consumer_task.cancel()
                try:
                    await consumer_task
                except asyncio.CancelledError:
                    pass

            # Get final state
            state = self.graph.get_state(config=self.base_config)
            self.current_state = state.values

            # Finalize blocks with state data
            self._finalize_blocks(state.values, chat_display)

            # NOW create response block (after all processing blocks)
            streaming_msg = chat_display.add_streaming_message()

            # Check for interrupts (approval needed)
            if state.interrupts:
                interrupt = state.interrupts[0]
                user_msg = interrupt.value.get("user_message", "Approval required")
                streaming_msg.finalize(f"⚠️ {user_msg}\n\nRespond with 'yes'/'no' or feedback.")
                return

            # Show final response
            self._show_final_response(state.values, streaming_msg)

        except Exception as e:
            chat_display.add_message(f"Error: {e}", "assistant")

    def _get_current_block(self, component: str, display: ChatDisplay) -> ProcessingBlock | None:
        """Get the current active block for a component."""
        attempt_idx = display._component_attempt_index.get(component, 0)
        block_key = f"{component}_{attempt_idx}"
        return display._current_blocks.get(block_key)

    def _handle_execution_event(
        self,
        chunk: dict,
        component: str,
        is_complete: bool,
        event_type: str,
        display: ChatDisplay,
    ) -> None:
        """Handle Execution phase events for capability steps.

        Captures status messages for both IN (first message) and OUT (last message).

        Args:
            chunk: The streaming event data.
            component: The capability name being executed.
            is_complete: Whether this is a completion event.
            event_type: The event type (status, success, error, warning).
            display: The chat display widget.
        """
        # Get step info from streaming event
        step_num = chunk.get("step", 1)  # 1-based step number from streaming
        step_index = step_num - 1  # Convert to 0-based for internal use
        message = chunk.get("message", "")

        block_key = f"execution_step_{step_index}"
        prev_block_key = f"execution_step_{step_index - 1}" if step_index > 0 else None

        # Handle ERROR/WARNING events - finalize current block with error/warning status
        if event_type in ("error", "warning"):
            block = display._current_blocks.get(block_key)
            if block and block._status == "active":
                block.set_output(message, status=event_type)
            return

        # When new step starts, mark previous step as complete
        if prev_block_key and prev_block_key in display._current_blocks:
            prev_block = display._current_blocks[prev_block_key]
            if prev_block._status == "active":
                # Use last captured message as output
                last_msg = getattr(self, f"_step_{step_index - 1}_last_msg", "Completed")
                prev_block.set_output(last_msg)

        # Create block if not exists (first event for this step)
        if block_key not in display._current_blocks:
            # Get objective from cached plan if available, otherwise use message
            objective = ""
            if hasattr(self, "_cached_plan") and self._cached_plan:
                steps = self._cached_plan.get("steps", [])
                if 0 <= step_index < len(steps):
                    objective = steps[step_index].get("task_objective", "")

            block = ExecutionStepBlock(
                step_number=step_num,
                capability=component,
                objective=objective if objective else message,
            )
            display._current_blocks[block_key] = block
            display.mount(block)
            block.set_active()
            display.scroll_end(animate=True)

            # Store first message as potential input (if not using plan objective)
            if not objective and message:
                setattr(self, f"_step_{step_index}_first_msg", message)

        # Get block for LOG updates
        block = display._current_blocks.get(block_key)

        # Add message to LOG section (like T/C/O blocks)
        if block and message:
            block.add_log(message, status=event_type)

        # Always capture message for potential output (last message wins)
        if message:
            setattr(self, f"_step_{step_index}_last_msg", message)

    def _finalize_blocks(self, state: dict, display: ChatDisplay) -> None:
        """Update blocks with final state data.

        For retry scenarios, only updates the LATEST block of each type
        (the one with the highest attempt index).

        Args:
            state: The final agent state.
            display: The chat display widget.
        """
        # Get latest attempt index for each component
        te_idx = display._component_attempt_index.get("task_extraction", 0)
        cl_idx = display._component_attempt_index.get("classifier", 0)
        or_idx = display._component_attempt_index.get("orchestrator", 0)

        # Task extraction output - no truncation
        te_key = f"task_extraction_{te_idx}"
        if te_key in display._current_blocks:
            task = state.get("task_current_task", "")
            display._current_blocks[te_key].set_output(task if task else "No task extracted")

        # Classification: update input (if not already set) and capabilities
        cl_key = f"classifier_{cl_idx}"
        if cl_key in display._current_blocks:
            cl_block = display._current_blocks[cl_key]
            # Only set IN if not already populated from streaming
            if not cl_block._input_set:
                task = state.get("task_current_task", "")
                if task:
                    cl_block.set_input(task)
            # Use cached all_capabilities (fast) + selected from state
            selected_caps = state.get("planning_active_capabilities", [])
            cl_block.set_capabilities(self.all_capability_names, selected_caps)

        # Orchestration: update input (if not set) and output
        or_key = f"orchestrator_{or_idx}"
        if or_key in display._current_blocks:
            or_block = display._current_blocks[or_key]
            # Only set IN if not already populated from streaming
            if not or_block._input_set:
                task = state.get("task_current_task", "")
                caps = state.get("planning_active_capabilities", [])
                or_block.set_input(f"{task} → [{', '.join(caps)}]")

            # Only update output if block hasn't already been set to error
            # (error state from streaming should be preserved)
            if or_block._status != "error":
                plan = state.get("planning_execution_plan", {})
                steps = plan.get("steps", []) if plan else []
                if steps:
                    or_block.set_plan(steps)
                else:
                    # No plan and not already error - show generic message
                    or_block.set_output("No execution plan")

            # Cache execution plan for step block creation during execution phase
            self._cached_plan = state.get("planning_execution_plan", {})

        # Finalize execution step blocks using captured status messages
        plan = state.get("planning_execution_plan", {})
        steps = plan.get("steps", []) if plan else []
        step_results = state.get("execution_step_results", {})

        for i, step in enumerate(steps):
            block_key = f"execution_step_{i}"
            block = display._current_blocks.get(block_key)
            if block and isinstance(block, ExecutionStepBlock):
                # Only finalize if still active (not already completed)
                if block._status == "active":
                    # Use captured last message as output
                    last_msg = getattr(self, f"_step_{i}_last_msg", None)
                    if last_msg:
                        block.set_output(last_msg)
                    else:
                        # Fallback to step_results metadata
                        context_key = step.get("context_key", "")
                        result = step_results.get(context_key, {})
                        success = result.get("success", True)
                        block.set_output("Completed", status="success" if success else "error")

    def _show_final_response(self, state: dict, streaming_msg: StreamingMessage) -> None:
        """Show final AI response.

        Args:
            state: The final agent state.
            streaming_msg: The streaming message widget to finalize.
        """
        messages = state.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    if not hasattr(msg, "type") or msg.type != "human":
                        streaming_msg.finalize(msg.content)
                        return
        streaming_msg.finalize("(No response)")


async def run_tui(config_path: str = "config.yml") -> None:
    """Run the Osprey TUI application.

    Args:
        config_path: Path to the configuration file.
    """
    app = OspreyTUI(config_path=config_path)
    await app.run_async()
