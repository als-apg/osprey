"""Osprey TUI Application.

A Terminal User Interface for the Osprey Agent Framework built with Textual.
"""

import asyncio
import logging
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import TextArea

from osprey.graph import create_graph
from osprey.infrastructure.gateway import Gateway
from osprey.interfaces.tui.constants import EXEC_STEP_PATTERN, TASK_PREP_COMPONENTS
from osprey.interfaces.tui.handlers import QueueLogHandler
from osprey.interfaces.tui.widgets import (
    ArtifactItem,
    ArtifactViewer,
    ChatDisplay,
    ChatInput,
    ClassificationBlock,
    ClassificationStep,
    CommandDropdown,
    CommandPalette,
    ExecutionStep,
    OrchestrationStep,
    ProcessingBlock,
    ProcessingStep,
    StatusPanel,
    TaskExtractionStep,
    ThemePicker,
    TodoUpdateStep,
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
        # Chat body scrolling (when focus not on input)
        Binding("space", "scroll_down", "Scroll down", show=False),
        Binding("b", "scroll_up", "Scroll up", show=False),
        Binding("g", "scroll_home", "Go to top", show=False),
        Binding("G", "scroll_end_chat", "Go to bottom", show=False),
        # Artifact gallery (priority=True to override TextArea's select-all)
        Binding("ctrl+a", "focus_artifacts", "Artifacts", priority=True),
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

    def on_artifact_item_selected(self, event: ArtifactItem.Selected) -> None:
        """Handle artifact selection - open the artifact viewer modal.

        Args:
            event: The artifact selection event containing the artifact data.
        """
        self.push_screen(ArtifactViewer(event.artifact))

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

    def action_focus_artifacts(self) -> None:
        """Focus the artifact gallery for keyboard navigation."""
        try:
            chat_display = self.query_one("#chat-display", ChatDisplay)
            gallery = chat_display.get_artifact_gallery()
            if gallery and gallery.display:
                gallery.focus()
                # Scroll to make gallery visible
                chat_display.scroll_to_widget(gallery)
            else:
                self.notify("No artifacts available", severity="information")
        except Exception:
            self.notify("No artifacts available", severity="information")

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

        # Set up log handler to capture Python logs via single-channel architecture
        chat_display = self.query_one("#chat-display", ChatDisplay)
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
        # Reset artifact gallery
        chat_display._artifact_gallery = None
        chat_display.clear_artifact_history()

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
        current_block: ProcessingBlock | ProcessingStep | None,
    ) -> tuple[str | None, ProcessingBlock | ProcessingStep | None]:
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

        # 2. Create new block if needed (on first log from this component)
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

        # 3. Extract data into block._data dict (from log event's extra fields)
        for key in ["task", "capabilities", "capability_names", "steps", "user_query"]:
            if key in chunk:
                current_block._data[key] = chunk[key]

        # 3a. Extract LLM prompt/response if available (for ProcessingStep widgets)
        if "llm_prompt" in chunk and hasattr(current_block, "set_llm_prompt"):
            current_block.set_llm_prompt(chunk["llm_prompt"])
        if "llm_response" in chunk and hasattr(current_block, "set_llm_response"):
            current_block.set_llm_response(chunk["llm_response"])

        # 4. Add to LOG section and update OUT with latest message
        if msg:
            current_block.add_log(msg, status=level)
            # Update OUT section with every log message (real-time feedback)
            current_block.set_partial_output(msg, status=level)

        # 5. Real-time IN update (when data becomes available)
        self._update_input_from_data(current_block, component)

        # 6. Update shared_data for downstream blocks (populates task/caps for C/O blocks)
        self._update_output_from_data(current_block, component, chunk)

        return current_component, current_block

    def _close_task_prep_block(
        self, block: ProcessingBlock | ProcessingStep, component: str | None
    ) -> None:
        """Close a Task Preparation block/step by setting its final output.

        Args:
            block: The block or step to close.
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
                if caps and isinstance(block, (ClassificationBlock, ClassificationStep)):
                    block.set_capabilities(self.all_capability_names, caps)
                else:
                    block.set_output("Classification complete")
            elif component == "orchestrator":
                steps = data.get("steps", [])
                if steps and isinstance(block, OrchestrationStep):
                    block.set_plan(steps)  # Shows todo list in step output
                    # Initialize plan state for TodoUpdateStep during execution
                    display = self.query_one("#chat-display", ChatDisplay)
                    display._plan_steps = steps
                    display._plan_step_states = ["pending"] * len(steps)
                elif block._last_error_msg:
                    # Use last error message for failed orchestration
                    block.set_output(block._last_error_msg, status="error")
                else:
                    block.set_output("Planning complete")
            else:
                block.set_output("Complete")

    def _create_task_prep_block(
        self, component: str, user_query: str, display: ChatDisplay
    ) -> ProcessingBlock | ProcessingStep | None:
        """Create and mount a Task Preparation block or step.

        Handles retry numbering by checking existing blocks in display.
        Task extraction uses minimal step widget, others use full blocks.

        Args:
            component: The component name.
            user_query: The original user query.
            display: The chat display widget.

        Returns:
            The created block/step, or None if invalid component.
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

        # Determine block/step class and title
        # Task extraction, classification, and orchestration use minimal step widgets
        block_classes = {
            "task_extraction": (TaskExtractionStep, "Task Extraction"),
            "classifier": (ClassificationStep, "Classification"),
            "orchestrator": (OrchestrationStep, "Planning"),
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

    def _update_input_from_data(
        self, block: ProcessingBlock | ProcessingStep, component: str
    ) -> None:
        """Update block IN section from _data dict when data is available.

        For ProcessingStep, set_input() is a no-op (steps have no IN section).

        Args:
            block: The processing block or step.
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
            # C:IN = task (check both block._data and _shared_data)
            task = data.get("task") or self._shared_data.get("task", "")
            if task:
                block.set_input(task)
                block._data["task"] = task  # Store for future reference
        elif component == "orchestrator":
            # O:IN = task → [caps] (check both block._data and _shared_data)
            task = data.get("task") or self._shared_data.get("task", "")
            caps = data.get("capabilities") or self._shared_data.get("capability_names", [])
            if task and caps:
                block.set_input(f"{task} → [{', '.join(caps)}]")
            elif task:
                # Task available but no caps yet - show partial
                block.set_input(task, mark_set=False)

    def _update_output_from_data(
        self, block: ProcessingBlock | ProcessingStep, component: str, chunk: dict
    ) -> None:
        """Update block OUT section from _data dict during streaming.

        Uses set_partial_output() for real-time updates (keeps block active).
        Also updates _shared_data for passing info to subsequent blocks.
        For ProcessingStep, set_partial_output() is a no-op.

        Args:
            block: The processing block or step.
            component: The component name.
            chunk: The completion event chunk.
        """
        data = block._data

        if component == "task_extraction":
            # T:OUT = task (partial - full will be set on close)
            task = data.get("task", "")
            if task:
                block.set_partial_output(task)
                # Save task to shared_data for C and O blocks
                self._shared_data["task"] = task

        elif component == "classifier":
            # C:OUT = selected capabilities (partial preview)
            selected_caps = data.get("capability_names", [])
            if selected_caps:
                block.set_partial_output(f"Selected: {', '.join(selected_caps)}")
                # Save capability_names to shared_data for O block
                self._shared_data["capability_names"] = selected_caps

        elif component == "orchestrator":
            # O:OUT = planned steps (partial preview)
            steps = data.get("steps", [])
            if steps:
                block.set_partial_output(f"{len(steps)} steps planned")

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

            # Show artifacts AFTER the response (so they appear below)
            artifacts = state.values.get("ui_artifacts", [])
            if artifacts:
                chat_display.update_artifacts(artifacts)

        except Exception as e:
            chat_display.add_message(f"Error: {e}", "assistant", message_type="agent")

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
            # Mount TodoUpdateStep FIRST - show progress before execution
            if display._plan_steps:
                # Update states: mark previous as done, current as active
                for i in range(step_index):
                    if display._plan_step_states[i] != "done":
                        display._plan_step_states[i] = "done"
                if step_index < len(display._plan_step_states):
                    display._plan_step_states[step_index] = "current"
                update_step = TodoUpdateStep()
                display.mount(update_step)
                update_step.set_todos(display._plan_steps, display._plan_step_states)

            # Mount ExecutionStep SECOND - then execute capability
            block = ExecutionStep(capability=component)
            display._current_blocks[block_key] = block
            display.mount(block)
            block.set_active()
            display.scroll_end(animate=True)

        # Get block for LOG updates
        block = display._current_blocks.get(block_key)

        # Add message to LOG section (like T/C/O blocks)
        if block and message:
            block.add_log(message, status=event_type)

        # Extract LLM prompt/response if available (for smart capabilities like respond/clarify)
        if block:
            if "llm_prompt" in chunk:
                block.set_llm_prompt(chunk["llm_prompt"])
            if "llm_response" in chunk:
                block.set_llm_response(chunk["llm_response"])

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
