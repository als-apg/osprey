"""Osprey TUI Application.

A Terminal User Interface for the Osprey Agent Framework built with Textual.
"""

import asyncio
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import TextArea

from osprey.events import parse_event
from osprey.graph import create_graph
from osprey.infrastructure.gateway import Gateway
from osprey.interfaces.tui.event_handler import TUIEventHandler
from osprey.interfaces.tui.widgets import (
    ChatDisplay,
    ChatInput,
    ClassificationBlock,
    ClassificationStep,
    CommandDropdown,
    CommandPalette,
    ExecutionStep,
    OrchestrationStep,
    PlanProgressBar,
    ProcessingBlock,
    ProcessingStep,
    StatusPanel,
    TaskExtractionStep,
    ThemePicker,
    WelcomeScreen,
)
from osprey.registry import get_registry, initialize_registry
from osprey.utils.config import get_config_value, get_full_configuration


class OspreyTUI(App):
    """Osprey Terminal User Interface.

    A TUI for interacting with the Osprey Agent Framework.
    """

    TITLE = "Osprey TUI"
    SUB_TITLE = "AI Agent Framework"

    CSS_PATH = "styles.tcss"

    # Prevent inheriting ctrl+q from parent App class
    _inherit_bindings = False

    BINDINGS = [
        # Override default ctrl+q quit (built-in has priority=True, so we must too)
        Binding("ctrl+q", "noop", "", show=False, priority=True),
        ("ctrl+c", "smart_ctrl_c", "Copy/Quit"),
        # Command palette
        ("ctrl+p", "show_command_palette", "Command palette"),
        Binding("ctrl+shift+p", "command_palette", "Debug palette", show=False),
        # Focus input
        ("ctrl+l", "focus_input", "Focus input"),
        # Theme picker
        ("ctrl+t", "switch_theme", "Switch theme"),
        # Help - toggle keys panel
        ("ctrl+h", "toggle_help_panel", "Toggle help"),
        # Toggle plan progress bar
        ("ctrl+o", "toggle_plan_progress", "Toggle plan"),
        # Chat body scrolling (when focus not on input)
        Binding("space", "scroll_down", "Scroll down", show=False),
        Binding("b", "scroll_up", "Scroll up", show=False),
        Binding("g", "scroll_home", "Go to top", show=False),
        Binding("G", "scroll_end_chat", "Go to bottom", show=False),
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

        # Double Ctrl+C quit state
        self._quit_pending: bool = False
        self._quit_timer: asyncio.TimerHandle | None = None

        # Welcome screen mode - starts True, becomes False on first user input
        self._welcome_mode: bool = True

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        # Welcome screen (shown initially)
        yield WelcomeScreen(version=self._get_version(), id="welcome-screen")
        # Chat screen (hidden initially)
        yield Vertical(
            ChatDisplay(id="chat-display"),
            CommandDropdown(id="command-dropdown"),
            PlanProgressBar(id="plan-progress"),  # Floating todo progress bar
            ChatInput(id="chat-input", placeholder="Type your message here..."),
            StatusPanel(id="status-panel"),
            id="main-content",
        )

    def action_request_quit(self) -> None:
        """Handle Ctrl+C - requires double press within 1.0s to quit."""
        if self._quit_pending:
            # Second Ctrl+C within timeout - actually quit
            self._cancel_quit_timer()
            self.exit()
        else:
            # First Ctrl+C - show confirmation message
            self._quit_pending = True
            self._show_quit_hint()
            # Set timer to reset after 1.0s
            self._quit_timer = asyncio.get_event_loop().call_later(1.0, self._reset_quit_state)

    def _show_quit_hint(self) -> None:
        """Show quit confirmation hint in status panel."""
        panel_id = "#welcome-status" if self._welcome_mode else "#status-panel"
        try:
            status = self.query_one(panel_id, StatusPanel)
            status.set_message(
                [
                    ("Press ", "desc"),
                    ("Ctrl-C", "cmd"),
                    (" again to exit", "desc"),
                ]
            )
        except Exception:
            pass

    def _reset_quit_state(self) -> None:
        """Reset quit state and restore status panel."""
        self._quit_pending = False
        self._quit_timer = None
        panel_id = "#welcome-status" if self._welcome_mode else "#status-panel"
        try:
            status = self.query_one(panel_id, StatusPanel)
            status.set_tips(
                [
                    ("/", "for commands"),
                    ("option + ⏎", "for newline"),
                    ("↑↓", "for history"),
                ]
            )
        except Exception:
            pass

    def _cancel_quit_timer(self) -> None:
        """Cancel the quit timer if active."""
        if self._quit_timer:
            self._quit_timer.cancel()
            self._quit_timer = None

    def action_smart_ctrl_c(self) -> None:
        """Smart Ctrl+C: copy if text selected, otherwise trigger quit."""
        # Try to find focused TextArea with selection
        try:
            from textual.widgets import TextArea

            focused = self.focused
            if isinstance(focused, TextArea) and focused.selected_text:
                # Copy selected text to clipboard
                self._copy_to_clipboard(focused.selected_text)
                return
        except Exception:
            pass
        # No selection - trigger quit behavior
        self.action_request_quit()

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard using pyperclip."""
        try:
            import pyperclip

            pyperclip.copy(text)
        except Exception:
            pass  # Clipboard not available - fail silently

    def action_noop(self) -> None:
        """Do nothing - used to disable default bindings."""
        pass

    def action_show_command_palette(self) -> None:
        """Show the command palette modal."""

        def handle_result(result: str | None) -> None:
            if result:
                # Execute the command action
                action = getattr(self, f"action_{result}", None)
                if action:
                    action()

        self.push_screen(CommandPalette(), handle_result)

    def action_focus_input(self) -> None:
        """Focus the chat input bar."""
        input_id = "#welcome-input" if self._welcome_mode else "#chat-input"
        try:
            self.query_one(input_id, ChatInput).focus()
        except Exception:
            pass

    def action_switch_theme(self) -> None:
        """Show the theme picker modal."""
        self.push_screen(ThemePicker())

    def action_toggle_help_panel(self) -> None:
        """Toggle the keys panel (HelpPanel) on the right side."""
        from textual.css.query import NoMatches
        from textual.widgets import HelpPanel

        try:
            help_panel = self.screen.query_one(HelpPanel)
            help_panel.remove()
        except NoMatches:
            self.screen.mount(HelpPanel())

    def action_toggle_plan_progress(self) -> None:
        """Toggle the plan progress bar visibility."""
        progress_bar = self.query_one("#plan-progress", PlanProgressBar)
        new_display = not progress_bar.display

        # If showing progress bar, hide command dropdown (mutual exclusivity)
        if new_display:
            try:
                dropdown = self.query_one("#command-dropdown", CommandDropdown)
                if dropdown.is_visible:
                    dropdown.hide()
            except Exception:
                pass

        progress_bar.display = new_display
        progress_bar.refresh()  # Force immediate UI update

    def action_exit_app(self) -> None:
        """Exit the application."""
        self.exit()

    def action_scroll_down(self) -> None:
        """Scroll chat down by one page (when not in input)."""
        if not isinstance(self.focused, TextArea):
            chat = self.query_one("#chat-display", ChatDisplay)
            chat.scroll_page_down(animate=False)

    def action_scroll_up(self) -> None:
        """Scroll chat up by one page (when not in input)."""
        if not isinstance(self.focused, TextArea):
            chat = self.query_one("#chat-display", ChatDisplay)
            chat.scroll_page_up(animate=False)

    def action_scroll_home(self) -> None:
        """Scroll to top of chat (when not in input)."""
        if not isinstance(self.focused, TextArea):
            chat = self.query_one("#chat-display", ChatDisplay)
            chat.scroll_home(animate=False)

    def action_scroll_end_chat(self) -> None:
        """Scroll to bottom of chat (when not in input)."""
        if not isinstance(self.focused, TextArea):
            chat = self.query_one("#chat-display", ChatDisplay)
            chat.scroll_end(animate=False)

    def _get_version(self) -> str:
        """Get the framework version."""
        try:
            from osprey import __version__

            return __version__
        except ImportError:
            return ""

    def _exit_welcome_mode(self) -> None:
        """Exit welcome mode by switching from welcome screen to chat UI."""
        if not self._welcome_mode:
            return

        self._welcome_mode = False

        # Hide welcome screen, show main content
        try:
            welcome_screen = self.query_one("#welcome-screen", WelcomeScreen)
            welcome_screen.display = False
        except Exception:
            pass

        try:
            main_content = self.query_one("#main-content")
            main_content.display = True
            # Focus the chat input
            self.query_one("#chat-input", ChatInput).focus()
        except Exception:
            pass

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

        # Hide main content initially (welcome screen is shown)
        self.query_one("#main-content").display = False

        # Focus the welcome input field
        self.query_one("#welcome-input", ChatInput).focus()

        # Note: QueueLogHandler removed as part of Phase 4 migration.
        # TUI now relies solely on typed events via LangGraph streaming.
        # See EVENT_STREAMING.md for architecture details.

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle input submission.

        Args:
            event: The input submitted event.
        """
        user_input = event.value.strip()

        if not user_input:
            return

        # Exit welcome mode on any input (message or slash command)
        if self._welcome_mode:
            self._exit_welcome_mode()

        # Handle slash commands
        if event.is_command:
            self._execute_slash_command(user_input)
            return

        # Process with agent (async worker) - start_new_query called inside
        self.process_with_agent(user_input)

    def _execute_slash_command(self, command_line: str) -> None:
        """Execute a slash command and show result.

        Args:
            command_line: The full command line (e.g., "/planning:on").
        """
        chat_display = self.query_one("#chat-display", ChatDisplay)

        # Parse command: /command or /command:option
        cmd_name, option = self._parse_command(command_line)

        if cmd_name == "clear":
            self._cmd_clear()
        elif cmd_name == "help":
            self._cmd_help(option)
        elif cmd_name == "config":
            self._cmd_config()
        elif cmd_name == "status":
            self._cmd_status()
        elif cmd_name in ("planning", "approval", "task", "caps"):
            self._cmd_agent_control(cmd_name, option)
        elif cmd_name == "exit":
            self.exit()
        else:
            chat_display.add_message(
                f"Unknown command: `{command_line}`\n\nType `/help` to see available commands.",
                "assistant",
                message_type="instant",
            )

    def _parse_command(self, command_line: str) -> tuple[str, str | None]:
        """Parse a command line into command name and option.

        Args:
            command_line: The full command line (e.g., "/planning:on").

        Returns:
            Tuple of (command_name, option or None).
        """
        # Remove leading slash
        line = command_line[1:] if command_line.startswith("/") else command_line

        # Check for colon syntax: /command:option
        if ":" in line:
            parts = line.split(":", 1)
            return parts[0].lower(), parts[1]

        return line.lower(), None

    def _cmd_clear(self) -> None:
        """Clear all conversation blocks."""
        chat_display = self.query_one("#chat-display", ChatDisplay)
        # Remove all children from chat display
        for child in list(chat_display.children):
            child.remove()
        # Reset block tracking
        chat_display._current_blocks = {}
        chat_display._seen_start_events = set()
        chat_display._component_attempt_index = {}
        chat_display._retry_triggered = set()
        chat_display._pending_messages = {}

    def _cmd_help(self, option: str | None) -> None:
        """Show help for commands.

        Args:
            option: Optional specific command to get help for.
        """
        chat_display = self.query_one("#chat-display", ChatDisplay)

        if option:
            # Show help for specific command
            help_text = self._get_command_help(option)
        else:
            # Show all commands
            help_text = self._format_all_commands_help()

        chat_display.add_message(help_text, "assistant", message_type="instant")

    def _get_command_help(self, cmd_name: str) -> str:
        """Get help text for a specific command.

        Args:
            cmd_name: The command name.

        Returns:
            Help text for the command.
        """
        commands = {
            "help": ("Show available commands", "Usage: `/help` or `/help:command`"),
            "clear": ("Clear conversation history", "Usage: `/clear`"),
            "config": ("Show current configuration", "Usage: `/config`"),
            "status": ("Show system status", "Usage: `/status`"),
            "planning": ("Control planning mode", "Usage: `/planning:on` or `/planning:off`"),
            "approval": (
                "Control approval workflow",
                "Usage: `/approval:on`, `/approval:off`, or `/approval:selective`",
            ),
            "task": ("Control task extraction bypass", "Usage: `/task:on` or `/task:off`"),
            "caps": ("Control capability selection bypass", "Usage: `/caps:on` or `/caps:off`"),
            "exit": ("Exit the application", "Usage: `/exit`"),
        }

        if cmd_name in commands:
            desc, usage = commands[cmd_name]
            return f"**/{cmd_name}** - {desc}\n\n{usage}"

        return f"Unknown command: /{cmd_name}"

    def _format_all_commands_help(self) -> str:
        """Format help text for all available commands.

        Returns:
            Formatted help text.
        """
        help_lines = [
            "## Available Commands\n",
            "| Command | Description |",
            "|---------|-------------|",
            "| `/help` | Show available commands |",
            "| `/clear` | Clear conversation |",
            "| `/config` | Show configuration |",
            "| `/status` | System health check |",
            "| `/planning:on/off` | Enable/disable planning mode |",
            "| `/approval:on/off/selective` | Control approval workflow |",
            "| `/task:on/off` | Enable/disable task extraction |",
            "| `/caps:on/off` | Enable/disable capability selection |",
            "| `/exit` | Exit the application |",
            "",
            "Type `/help:command` for detailed help on a specific command.",
        ]
        return "\n".join(help_lines)

    def _cmd_config(self) -> None:
        """Show current configuration."""
        chat_display = self.query_one("#chat-display", ChatDisplay)

        config = self.base_config.get("configurable", {})
        lines = ["## Current Configuration\n"]

        # Key config values to show
        keys_to_show = [
            ("planning_mode_enabled", "Planning Mode"),
            ("task_extraction_bypass_enabled", "Task Extraction Bypass"),
            ("capability_selection_bypass_enabled", "Capability Selection Bypass"),
            ("approval_global_mode", "Approval Mode"),
            ("interface_context", "Interface"),
            ("thread_id", "Session ID"),
        ]

        for key, label in keys_to_show:
            value = config.get(key, "N/A")
            lines.append(f"- **{label}**: {value}")

        chat_display.add_message("\n".join(lines), "assistant", message_type="instant")

    def _cmd_status(self) -> None:
        """Show system status."""
        chat_display = self.query_one("#chat-display", ChatDisplay)

        # Get registry stats
        registry = get_registry()
        stats = registry.get_stats()

        lines = [
            "## System Status\n",
            f"- **Capabilities**: {stats.get('capability_count', 0)} registered",
            f"- **Session**: {self.thread_id}",
            f"- **Graph**: {'Ready' if self.graph else 'Not initialized'}",
            f"- **Gateway**: {'Ready' if self.gateway else 'Not initialized'}",
        ]

        chat_display.add_message("\n".join(lines), "assistant", message_type="instant")

    def _cmd_agent_control(self, cmd: str, value: str | None) -> None:
        """Handle agent control commands (planning, approval, task, caps).

        Args:
            cmd: The command name.
            value: The option value (on/off/selective).
        """
        chat_display = self.query_one("#chat-display", ChatDisplay)

        if value is None:
            chat_display.add_message(
                f"Usage: /{cmd}:on or /{cmd}:off", "assistant", message_type="instant"
            )
            return

        value = value.lower()

        # Map command to config key
        config_keys = {
            "planning": "planning_mode_enabled",
            "task": "task_extraction_bypass_enabled",
            "caps": "capability_selection_bypass_enabled",
        }

        if cmd == "approval":
            # Special handling for approval modes
            mode_map = {"on": "all_capabilities", "off": "disabled", "selective": "selective"}
            if value in mode_map:
                self.base_config["configurable"]["approval_global_mode"] = mode_map[value]
                chat_display.add_message(
                    f"Approval mode: **{value}**", "assistant", message_type="instant"
                )
            else:
                chat_display.add_message(
                    "Invalid option. Use: `/approval:on`, `/approval:off`, or `/approval:selective`",
                    "assistant",
                    message_type="instant",
                )
        elif cmd in config_keys:
            enabled = value in ("on", "true", "enabled", "1")
            self.base_config["configurable"][config_keys[cmd]] = enabled
            status = "enabled" if enabled else "disabled"
            cmd_display = {
                "planning": "Planning mode",
                "task": "Task extraction",
                "caps": "Capability selection",
            }
            chat_display.add_message(
                f"{cmd_display[cmd]}: **{status}**", "assistant", message_type="instant"
            )
        else:
            chat_display.add_message(
                f"Unknown command: /{cmd}", "assistant", message_type="instant"
            )

    async def _consume_events(self, user_query: str, chat_display: ChatDisplay) -> None:
        """Event consumer - processes typed OspreyEvents from LangGraph streaming.

        ARCHITECTURE: Phase 4 migration complete - uses typed events exclusively.

        Typed events (with event_class field) are parsed and routed via TUIEventHandler.
        Block lifecycle is managed entirely through typed events:
        - PhaseStartEvent → Create block
        - StatusEvent → Add logs to block
        - Data events (TaskExtractedEvent, etc.) → Set block output
        - PhaseCompleteEvent → Finalize block
        - CapabilityStartEvent/CapabilityCompleteEvent → Execution blocks

        Args:
            user_query: The original user query.
            chat_display: The chat display widget.

        See EVENT_STREAMING.md for architecture details.
        """
        # Initialize typed event handler
        typed_handler = TUIEventHandler(chat_display, self._shared_data)

        while True:
            try:
                chunk = await chat_display._event_queue.get()

                # Parse as typed OspreyEvent
                typed_event = parse_event(chunk) if isinstance(chunk, dict) else None
                if typed_event:
                    # Route typed events through the handler
                    await typed_handler.handle(typed_event)
                    # Extract shared data for cross-block communication
                    typed_handler.extract_shared_data(typed_event)

                # DEBUG: Log raw events to debug block (if enabled)
                if isinstance(chunk, dict):
                    debug_block = chat_display.get_or_create_debug_block()
                    if debug_block:
                        debug_block.add_event(chunk)

                chat_display._event_queue.task_done()
            except asyncio.CancelledError:
                break

    @work(exclusive=True)
    async def process_with_agent(self, user_input: str) -> None:
        """Process user input through Gateway and stream response."""
        chat_display = self.query_one("#chat-display", ChatDisplay)

        # Start new query - resets blocks and adds user message
        chat_display.start_new_query(user_input)

        # Hide and reset plan progress bar from previous query
        progress_bar = self.query_one("#plan-progress", PlanProgressBar)
        progress_bar.clear()

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
                chat_display.add_message(
                    f"Error: {result.error}", "assistant", message_type="agent"
                )
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

            # Check for interrupts (approval needed)
            if state.interrupts:
                interrupt = state.interrupts[0]
                user_msg = interrupt.value.get("user_message", "Approval required")
                chat_display.add_message(
                    f"⚠️ {user_msg}\n\nRespond with 'yes'/'no' or feedback.",
                    "assistant",
                    message_type="agent",
                )
                return

            # Show final response
            self._show_final_response(state.values, chat_display)

        except Exception as e:
            chat_display.add_message(f"Error: {e}", "assistant", message_type="agent")

    def _get_current_block(self, component: str, display: ChatDisplay) -> ProcessingBlock | None:
        """Get the current active block for a component."""
        attempt_idx = display._component_attempt_index.get(component, 0)
        block_key = f"{component}_{attempt_idx}"
        return display._current_blocks.get(block_key)

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
                # Only set plan if not already set (avoid re-render jump)
                if steps and not or_block._plan_steps:
                    or_block.set_plan(steps)
                elif not steps and not or_block._plan_steps:
                    # No plan and not already set - show generic message
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
            if block and isinstance(block, ExecutionStep):
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

    def _show_final_response(self, state: dict, chat_display: ChatDisplay) -> None:
        """Show final AI response.

        Args:
            state: The final agent state.
            chat_display: The chat display to add the message to.
        """
        # Mark plan as complete and hide (keeps data for later viewing via Ctrl+O)
        progress_bar = self.query_one("#plan-progress", PlanProgressBar)
        progress_bar.mark_complete()

        content = "(No response)"
        messages = state.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    if not hasattr(msg, "type") or msg.type != "human":
                        content = msg.content
                        break

        chat_display.add_message(content, "assistant", message_type="agent")


async def run_tui(config_path: str = "config.yml") -> None:
    """Run the Osprey TUI application.

    Args:
        config_path: Path to the configuration file.
    """
    app = OspreyTUI(config_path=config_path)
    await app.run_async()
