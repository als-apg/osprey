"""Osprey TUI Application.

A Terminal User Interface for the Osprey Agent Framework built with Textual.
"""

import uuid

from langgraph.checkpoint.memory import MemorySaver
from textual import work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.events import Key
from textual.message import Message
from textual.widgets import Collapsible, Footer, Header, SelectionList, Static, TextArea
from textual.widgets.selection_list import Selection

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

    # Breathing animation frames - asterisk style
    BREATHING_FRAMES = ["*", "✱", "✳", "✱"]

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
        self._pending_output: tuple[str, bool] | None = None
        # Breathing animation state
        self._breathing_timer = None
        self._breathing_index = 0
        # Output preview for collapsible toggle
        self._output_preview: str = ""

    def compose(self) -> ComposeResult:
        """Compose the block with header, input, separator, and collapsible output."""
        header_text = f"{self.INDICATOR_PENDING} {self.title}"
        yield Static(header_text, classes="block-header", id="block-header")
        yield Static("", classes="block-input", id="block-input")
        # Full-width separator (will be truncated by container)
        yield Static("─" * 120, classes="block-separator", id="block-separator")
        # Official Collapsible widget for OUT section (hide built-in arrows)
        yield Collapsible(
            Static("", id="block-output-content"),
            title="",
            collapsed=True,
            collapsed_symbol="",
            expanded_symbol="",
            id="block-output",
        )

    def on_mount(self) -> None:
        """Apply pending state after widget is mounted."""
        self._mounted = True
        # Hide separator initially
        separator = self.query_one("#block-separator", Static)
        separator.display = False
        # Hide output initially (if it exists - subclasses may override)
        outputs = self.query("#block-output")
        if outputs:
            outputs.first().display = False
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
        """Internal: apply input text with bold label."""
        input_widget = self.query_one("#block-input", Static)
        # Bold label with fixed width (4 chars: "IN  " or "OUT ")
        input_widget.update(f"[bold]IN[/bold]    {text}")

    def _get_preview(self, text: str, max_len: int = 60) -> str:
        """Get one-line preview of output."""
        first_line = text.split("\n")[0] if text else ""
        if len(first_line) > max_len:
            return first_line[:max_len] + "..."
        elif "\n" in text:
            return first_line + " ..."
        return first_line

    def _apply_output(self, text: str, success: bool) -> None:
        """Internal: apply output with collapsible preview."""
        # Stop breathing animation
        self._stop_breathing()

        # Update header indicator
        indicator = self.INDICATOR_SUCCESS if success else self.INDICATOR_ERROR
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

        self.remove_class("block-active")

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        """Show expanded arrow with spaces to maintain clickable area."""
        if event.collapsible.id == "block-output":
            # Use spaces to maintain clickable width
            spaces = " " * len(self._output_preview) if self._output_preview else "   "
            event.collapsible.title = f"[bold]OUT[/bold]   ▾ {spaces}"

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        """Show collapsed arrow with preview."""
        if event.collapsible.id == "block-output":
            event.collapsible.title = f"[bold]OUT[/bold]   ▸ {self._output_preview}"

    def set_active(self) -> None:
        """Mark the block as actively processing."""
        self._status = "active"
        if self._mounted:
            self._apply_active()

    def set_input(self, text: str) -> None:
        """Set the input section text."""
        self._pending_input = text
        if self._mounted:
            self._apply_input(text)

    def set_output(self, text: str, success: bool = True) -> None:
        """Set the output section and mark complete.

        Args:
            text: The output text to display.
            success: Whether the operation succeeded.
        """
        self._status = "success" if success else "error"
        self._pending_output = (text, success)
        if self._mounted:
            self._apply_output(text, success)


class TaskExtractionBlock(ProcessingBlock):
    """Block for task extraction phase."""

    def __init__(self, **kwargs):
        """Initialize task extraction block."""
        super().__init__("Task Extraction", **kwargs)


class ClassificationBlock(ProcessingBlock):
    """Block for capability classification phase with checkbox list."""

    def __init__(self, **kwargs):
        """Initialize classification block."""
        super().__init__("Classification", **kwargs)
        self._all_capabilities: list[str] = []
        self._selected_capabilities: list[str] = []

    def compose(self) -> ComposeResult:
        """Override to add SelectionList inside Collapsible."""
        header_text = f"{self.INDICATOR_PENDING} {self.title}"
        yield Static(header_text, classes="block-header", id="block-header")
        yield Static("", classes="block-input", id="block-input")
        yield Static("─" * 120, classes="block-separator", id="block-separator")
        # Collapsible with SelectionList inside (hide built-in arrows)
        yield Collapsible(
            SelectionList[str](id="capability-list"),
            title="",
            collapsed=True,
            collapsed_symbol="",
            expanded_symbol="",
            id="block-output",
        )

    def on_mount(self) -> None:
        """Apply pending state after widget is mounted."""
        self._mounted = True
        separator = self.query_one("#block-separator", Static)
        separator.display = False
        output = self.query_one("#block-output", Collapsible)
        output.display = False
        # Apply pending state
        if self._status == "active":
            self._apply_active()
        if self._pending_input is not None:
            self._apply_input(self._pending_input)
        if self._all_capabilities:
            self._apply_capability_output()

    def set_output(self, text: str, success: bool = True) -> None:
        """Override to prevent base class from querying missing widgets.

        ClassificationBlock uses set_capabilities() instead of set_output().
        This is called during streaming completion but the actual data comes
        from _finalize_blocks() which calls set_capabilities().
        """
        # Store the status but don't try to render - set_capabilities handles
        self._status = "success" if success else "error"

    def set_capabilities(
        self, all_caps: list[str], selected: list[str]
    ) -> None:
        """Show capabilities as checklist.

        Args:
            all_caps: All available capabilities.
            selected: The selected/active capabilities.
        """
        self._all_capabilities = all_caps
        self._selected_capabilities = selected
        self._status = "success"
        if self._mounted:
            self._apply_capability_output()

    def _apply_capability_output(self) -> None:
        """Apply capability list with collapsible preview."""
        self._stop_breathing()

        # Update header indicator
        header = self.query_one("#block-header", Static)
        header.update(f"{self.INDICATOR_SUCCESS} {self.title}")

        # Show separator
        separator = self.query_one("#block-separator", Static)
        separator.display = True

        # Create preview: comma-separated selected capabilities
        preview = ", ".join(self._selected_capabilities) if self._selected_capabilities else "None"
        self._output_preview = self._get_preview(preview)

        # Show and update collapsible
        output = self.query_one("#block-output", Collapsible)
        output.display = True
        output.title = f"[bold]OUT[/bold]   ▸ {self._output_preview}"

        # Populate SelectionList inside collapsible
        capability_list = self.query_one("#capability-list", SelectionList)
        capability_list.clear_options()
        for cap in self._all_capabilities:
            is_selected = cap in self._selected_capabilities
            capability_list.add_option(
                Selection(cap, cap, initial_state=is_selected)
            )
        capability_list.disabled = True

        self.remove_class("block-active")

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        """Show expanded arrow with spaces to maintain clickable area."""
        if event.collapsible.id == "block-output":
            spaces = " " * len(self._output_preview) if self._output_preview else "   "
            event.collapsible.title = f"[bold]OUT[/bold]   ▾ {spaces}"

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        """Show collapsed arrow with preview."""
        if event.collapsible.id == "block-output":
            event.collapsible.title = f"[bold]OUT[/bold]   ▸ {self._output_preview}"


class OrchestrationBlock(ProcessingBlock):
    """Block for orchestration/planning phase."""

    def __init__(self, **kwargs):
        """Initialize orchestration block."""
        super().__init__("Orchestration", **kwargs)

    def set_plan(self, steps: list[dict]) -> None:
        """Show execution plan steps with aligned numbers.

        Args:
            steps: List of execution plan step dicts.
        """
        lines = []
        for i, step in enumerate(steps, 1):
            objective = step.get("task_objective", "")
            capability = step.get("capability", "")
            # Fixed-width number (3 chars: " 1." to "99.")
            num = f"{i:2}."
            lines.append(f"{num} {objective} [{capability}]")
        self.set_output("\n".join(lines) if lines else "No steps")


class ChatDisplay(ScrollableContainer):
    """Scrollable container for chat messages and processing blocks."""

    def __init__(self, **kwargs):
        """Initialize chat display with block tracking."""
        super().__init__(**kwargs)
        self._current_blocks: dict[str, ProcessingBlock] = {}
        # Track which START events we've seen (for deferred block creation)
        self._seen_start_events: set[str] = set()
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

    def _create_and_activate_block(
        self, component: str, user_query: str, display: ChatDisplay
    ) -> None:
        """Create and activate a processing block."""
        block = display.get_or_create_block(component)
        if not block:
            return

        block.set_active()

        # Set appropriate input text
        if component == "task_extraction":
            block.set_input(user_query)
        elif component == "classifier":
            block.set_input("Analyzing extracted task...")
        elif component == "orchestrator":
            block.set_input("Creating execution plan...")

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

        # DEBUG: Log status and success events to debug block (if enabled)
        if event_type in ("status", "success"):
            debug_block = display.get_or_create_debug_block()
            if debug_block:
                debug_block.add_event(chunk)

        # Accept both status (start/progress) and success (completion) events
        if event_type not in ("status", "success"):
            return

        component = chunk.get("component", "")
        phase = chunk.get("phase", "")
        # success event = completion, or explicit complete flag
        is_complete = (event_type == "success") or chunk.get("complete", False)

        # Task Preparation blocks only (Phase 1)
        if phase != "Task Preparation":
            return

        # Handle COMPLETION - also triggers deferred next block creation
        if is_complete:
            block = display._current_blocks.get(component)
            if block:
                msg = chunk.get("message", f"{component} complete")
                block.set_output(msg)

            # Check if next block's START was already seen, create it now
            next_component = self._get_next_component(component)
            if next_component and next_component in display._seen_start_events:
                self._create_and_activate_block(
                    next_component, user_query, display
                )
            return

        # Handle START - always record it
        display._seen_start_events.add(component)

        if component == "task_extraction":
            # First block - always create immediately
            self._create_and_activate_block(component, user_query, display)
        elif component in ("classifier", "orchestrator"):
            # Check if previous is complete - if so, create now
            prev_component = self._get_prev_component(component)
            prev = display._current_blocks.get(prev_component)
            if prev and prev._status in ("success", "error"):
                self._create_and_activate_block(component, user_query, display)
            # Otherwise: already recorded, will be created when prev completes

    def _finalize_blocks(self, state: dict, display: ChatDisplay) -> None:
        """Update blocks with final state data.

        Args:
            state: The final agent state.
            display: The chat display widget.
        """
        # Task extraction output - no truncation
        if "task_extraction" in display._current_blocks:
            task = state.get("task_current_task", "")
            display._current_blocks["task_extraction"].set_output(
                task if task else "No task extracted"
            )

        # Classification output - use set_capabilities for checkbox list
        if "classifier" in display._current_blocks:
            # Set input from task_current_task
            task = state.get("task_current_task", "")
            if task:
                display._current_blocks["classifier"].set_input(task)
            # Get all capabilities from registry and selected ones from state
            registry = get_registry()
            all_caps = registry.get_stats()["capability_names"]
            selected_caps = state.get("planning_active_capabilities", [])
            display._current_blocks["classifier"].set_capabilities(
                all_caps, selected_caps
            )

        # Orchestration output - no truncation
        if "orchestrator" in display._current_blocks:
            # Set input (task → capabilities)
            task = state.get("task_current_task", "")
            caps = state.get("planning_active_capabilities", [])
            display._current_blocks["orchestrator"].set_input(
                f"{task} → [{', '.join(caps)}]"
            )
            # Set output from execution plan
            plan = state.get("planning_execution_plan", {})
            steps = plan.get("steps", []) if plan else []
            display._current_blocks["orchestrator"].set_plan(steps)

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
