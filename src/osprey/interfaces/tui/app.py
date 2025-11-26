"""Osprey TUI Application.

A Terminal User Interface for the Osprey Agent Framework built with Textual.
"""

import textwrap
import uuid

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

    def _format_log_messages(self) -> str:
        """Format all log messages with status symbols."""
        if not self._log_messages:
            return ""
        lines = []
        for msg_status, msg in self._log_messages:
            prefix = {
                "error": self.INDICATOR_ERROR,
                "warning": self.INDICATOR_WARNING,
                "success": self.INDICATOR_SUCCESS,
                "status": self.INDICATOR_PENDING,
            }.get(msg_status, self.INDICATOR_PENDING)
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
                content,
                width=width,
                initial_indent=prefix,
                subsequent_indent=indent
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
        # Debug block for showing events (disabled by default)
        self._debug_enabled = False
        self._debug_block: DebugBlock | None = None

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

    def get_or_create_block(
        self, block_type: str, **kwargs
    ) -> ProcessingBlock | None:
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

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Vertical(
            ChatDisplay(id="chat-display"),
            ChatInput(id="chat-input"),
            id="main-content"
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
        configurable.update({
            "user_id": "tui_user",
            "thread_id": self.thread_id,
            "chat_id": "tui_chat",
            "session_id": self.thread_id,
            "interface_context": "tui",
        })

        recursion_limit = get_config_value(
            "execution_control.limits.graph_recursion_limit"
        )
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

    @work(exclusive=True)
    async def process_with_agent(self, user_input: str) -> None:
        """Process user input through Gateway and stream response."""
        chat_display = self.query_one("#chat-display", ChatDisplay)

        # Start new query - resets blocks and adds user message
        chat_display.start_new_query(user_input)

        # Clear cached plan and step messages from previous query
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
            input_data = (
                result.resume_command if result.resume_command else result.agent_state
            )

            if input_data is None:
                return

            # Stream with block updates (no StreamingMessage yet)
            async for chunk in self.graph.astream(
                input_data,
                config=self.base_config,
                stream_mode="custom",
            ):
                self._handle_block_update(chunk, user_input, chat_display)

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
                streaming_msg.finalize(
                    f"⚠️ {user_msg}\n\nRespond with 'yes'/'no' or feedback."
                )
                return

            # Show final response
            self._show_final_response(state.values, streaming_msg)

        except Exception as e:
            chat_display.add_message(f"Error: {e}", "assistant")

    def _get_next_component(self, component: str) -> str | None:
        """Get the next component in the task preparation sequence."""
        sequence = {"task_extraction": "classifier", "classifier": "orchestrator"}
        return sequence.get(component)

    def _get_prev_component(self, component: str) -> str | None:
        """Get the previous component in the task preparation sequence."""
        sequence = {"classifier": "task_extraction", "orchestrator": "classifier"}
        return sequence.get(component)

    def _close_and_create_retry(
        self,
        close_component: str,
        create_component: str,
        user_query: str,
        display: ChatDisplay,
    ) -> None:
        """Close a block as error and create a retry block.

        Args:
            close_component: Component whose block to close.
            create_component: Component to create retry block for.
            user_query: Original user query.
            display: Chat display widget.
        """
        # Close the current block for close_component
        block = self._get_current_block(close_component, display)
        if block and block._status == "active":
            block.set_output("Retry required", status="error")

        # Mark retry and create new block
        display._retry_triggered.add(create_component)
        display._seen_start_events.discard(create_component)
        self._create_and_activate_block(create_component, user_query, display)

    def _create_and_activate_block(
        self, component: str, user_query: str, display: ChatDisplay
    ) -> ProcessingBlock | None:
        """Create and activate a processing block with automatic retry detection.

        Retry detection is handled here: if a completed block exists for this
        component, we increment the attempt index and create a new block.

        Args:
            component: The component name (task_extraction, classifier, orchestrator).
            user_query: The original user query.
            display: The chat display widget.

        Returns:
            The created or existing block, or None if invalid component.
        """
        # Get current attempt index for this component
        attempt_idx = display._component_attempt_index.get(component, 0)
        block_key = f"{component}_{attempt_idx}"

        # Check existing block state
        existing = display._current_blocks.get(block_key)

        # If block exists and is still active, return it (no new block needed)
        if existing and existing._status == "active":
            return existing

        # If block exists and is COMPLETED, check if retry was triggered
        if existing and existing._status in ("success", "error"):
            # Only create new block if retry was explicitly triggered
            if component in display._retry_triggered:
                attempt_idx += 1
                display._component_attempt_index[component] = attempt_idx
                display._retry_triggered.discard(component)
                block_key = f"{component}_{attempt_idx}"
            else:
                # Block complete, no retry - return existing (don't create new)
                return existing

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
        block.title = title  # Override the default title
        display._current_blocks[block_key] = block
        display.mount(block)
        display.scroll_end(animate=False)

        block.set_active()

        # Apply any pending messages that arrived before block existed
        self._apply_pending_messages(block, component, display)

        # Set appropriate input text (only if not already set from pending messages)
        if not block._input_set:
            if component == "task_extraction":
                block.set_input(user_query)
            elif component == "classifier":
                input_text = "Reclassifying..." if attempt_idx > 0 else "Analyzing task..."
                block.set_input(input_text, mark_set=False)
            elif component == "orchestrator":
                input_text = "Re-planning..." if attempt_idx > 0 else "Creating plan..."
                block.set_input(input_text, mark_set=False)

        return block

    def _handle_block_update(
        self, chunk: dict, user_query: str, display: ChatDisplay
    ) -> None:
        """Route streaming event to appropriate block.

        Handles deferred block creation: START events are recorded, and
        blocks are created when the previous block completes.

        Args:
            chunk: The streaming event data.
            user_query: The original user query.
            display: The chat display widget.
        """
        event_type = chunk.get("event_type", "")

        # DEBUG: Log events to debug block (if enabled)
        if event_type in ("status", "success", "error", "warning"):
            debug_block = display.get_or_create_debug_block()
            if debug_block:
                debug_block.add_event(chunk)

        # Accept status, success, error, and warning events
        if event_type not in ("status", "success", "error", "warning"):
            return

        component = chunk.get("component", "")
        phase = chunk.get("phase", "")
        # success event = completion, or explicit complete flag
        is_complete = (event_type == "success") or chunk.get("complete", False)

        # Dispatch by phase
        if phase == "Task Preparation":
            self._handle_task_preparation_event(
                chunk, component, is_complete, event_type, user_query, display
            )
        elif phase == "Execution":
            self._handle_execution_event(chunk, component, is_complete, event_type, display)

    def _get_current_block(
        self, component: str, display: ChatDisplay
    ) -> ProcessingBlock | None:
        """Get the current active block for a component."""
        attempt_idx = display._component_attempt_index.get(component, 0)
        block_key = f"{component}_{attempt_idx}"
        return display._current_blocks.get(block_key)

    def _queue_pending_message(
        self,
        display: ChatDisplay,
        component: str,
        event_type: str,
        msg: str,
        chunk: dict,
    ) -> None:
        """Queue a message for a component whose block doesn't exist yet.

        Args:
            display: The chat display widget.
            component: The component name.
            event_type: The event type (status, success, error, warning).
            msg: The message text.
            chunk: The full streaming event data.
        """
        if component not in display._pending_messages:
            display._pending_messages[component] = []
        display._pending_messages[component].append((event_type, msg, chunk))

    def _apply_pending_messages(
        self,
        block: ProcessingBlock,
        component: str,
        display: ChatDisplay,
    ) -> None:
        """Apply queued messages to a newly created block.

        Args:
            block: The processing block to apply messages to.
            component: The component name.
            display: The chat display widget.
        """
        if component not in display._pending_messages:
            return
        for event_type, msg, chunk in display._pending_messages[component]:
            if msg:
                block.add_log(msg, status=event_type)
                # Also try to update IN from these messages
                if not block._input_set:
                    self._update_block_input_from_streaming(block, component, msg, chunk)
        # Clear the queue
        del display._pending_messages[component]

    def _update_block_input_from_streaming(
        self,
        block: ProcessingBlock,
        component: str,
        msg: str,
        chunk: dict,
    ) -> None:
        """Update block IN from streaming message when info is available.

        Args:
            block: The processing block to update.
            component: The component name.
            msg: The streaming message.
            chunk: The full streaming event data.
        """
        if component == "classifier":
            # Look for "Classifying task:" in message
            if "classifying task:" in msg.lower():
                # Extract task from message (after the colon)
                idx = msg.lower().find("classifying task:")
                task = msg[idx + len("classifying task:"):].strip()
                if task:
                    block.set_input(task)
        elif component == "orchestrator":
            # Look for task and capabilities info
            task = chunk.get("task", "")
            caps = chunk.get("capabilities", [])
            if task and caps:
                block.set_input(f"{task} → [{', '.join(caps)}]")
            elif "planning for task:" in msg.lower():
                # Extract task from message
                idx = msg.lower().find("planning for task:")
                task = msg[idx + len("planning for task:"):].strip()
                if task:
                    # Set with placeholder - will be updated when caps arrive
                    block.set_input(task, mark_set=False)

    def _handle_task_preparation_event(
        self,
        chunk: dict,
        component: str,
        is_complete: bool,
        event_type: str,
        user_query: str,
        display: ChatDisplay,
    ) -> None:
        """Handle Task Preparation phase events with LOG/OUT separation.

        Retry detection is handled in _create_and_activate_block(), not here.
        Status messages go to LOG section, final output goes to OUT section.

        Args:
            chunk: The streaming event data.
            component: The component name.
            is_complete: Whether this is a completion event.
            event_type: The event type (status, success, error, warning).
            user_query: The original user query.
            display: The chat display widget.
        """
        msg = chunk.get("message", "")

        # Handle ERROR events - log and finalize block
        if event_type == "error":
            block = self._get_current_block(component, display)
            if block and block._status == "active":
                if msg:
                    block.add_log(msg, status="error")
                block.set_output(msg if msg else f"{component} error", status="error")
            return

        # Handle WARNING events - just log, don't change block status
        # Block closure is handled by retry indicators ("Reclassifying...", "Re-planning...")
        if event_type == "warning":
            block = self._get_current_block(component, display)
            if block and block._status == "active" and msg:
                block.add_log(msg, status="warning")
            return

        # Handle COMPLETION - log and set final output
        if is_complete:
            block = self._get_current_block(component, display)
            if block:
                if msg:
                    block.add_log(msg, status="success")

                # Special handling for classifier - populate capability list
                if component == "classifier" and isinstance(block, ClassificationBlock):
                    selected_caps = chunk.get("capability_names", [])
                    if selected_caps:
                        block.set_capabilities(self.all_capability_names, selected_caps)
                    else:
                        block.set_output(msg if msg else "Classification complete")
                else:
                    block.set_output(msg if msg else f"{component} complete")

            # Check if next block's START was already seen, create it now
            next_component = self._get_next_component(component)
            if next_component and next_component in display._seen_start_events:
                self._create_and_activate_block(next_component, user_query, display)

            # When classifier completes, check if orchestrator needs new attempt
            if component == "classifier":
                next_component = "orchestrator"
                next_block = self._get_current_block(next_component, display)
                if next_block and next_block._status == "error":
                    # Orchestrator failed earlier, classifier just retried, create new O
                    display._retry_triggered.add(next_component)
                    self._create_and_activate_block(next_component, user_query, display)
            return

        # Check for retry indicators - close old block and create retry immediately
        if msg:
            msg_lower = msg.lower()
            if "reclassifying" in msg_lower:
                # Orchestrator failed, need new capabilities
                # Close orchestrator, create classifier retry
                self._close_and_create_retry("orchestrator", "classifier", user_query, display)
            elif "re-planning" in msg_lower:
                # Orchestrator needs to retry planning
                # Close orchestrator, create new orchestrator
                self._close_and_create_retry("orchestrator", "orchestrator", user_query, display)

        # Only run block creation on FIRST event for this component
        is_first_event = component not in display._seen_start_events
        if is_first_event:
            display._seen_start_events.add(component)

            if component == "task_extraction":
                # First block - always create immediately
                self._create_and_activate_block(component, user_query, display)
            elif component in ("classifier", "orchestrator"):
                # Check if previous is complete - if so, create now
                prev_component = self._get_prev_component(component)
                prev_block = self._get_current_block(prev_component, display)
                if prev_block and prev_block._status in ("success", "error"):
                    self._create_and_activate_block(component, user_query, display)
                # Otherwise: deferred, will be created when prev completes

        # Handle STATUS events - log message to existing block OR queue it
        block = self._get_current_block(component, display)
        if block and msg:
            block.add_log(msg, status="status")
            # Update IN from streaming when info is available
            if not block._input_set:
                self._update_block_input_from_streaming(block, component, msg, chunk)
        elif msg:
            # Block doesn't exist yet - queue the message for later
            self._queue_pending_message(display, component, "status", msg, chunk)

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
            display._current_blocks[te_key].set_output(
                task if task else "No task extracted"
            )

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
            # Set output from execution plan
            plan = state.get("planning_execution_plan", {})
            steps = plan.get("steps", []) if plan else []
            or_block.set_plan(steps)

            # Cache execution plan for step block creation during execution phase
            self._cached_plan = plan

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

    def _show_final_response(
        self, state: dict, streaming_msg: StreamingMessage
    ) -> None:
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
