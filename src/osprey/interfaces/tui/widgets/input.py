"""Input widgets for the TUI."""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.events import Blur, Focus, Key
from textual.message import Message
from textual.style import Style
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    pass


class ChatInput(TextArea):
    """Multi-line text input for chat messages.

    Press Enter to send, Option+Enter (Alt+Enter) for new line.
    """

    class Submitted(Message):
        """Event posted when user submits input."""

        def __init__(self, value: str, is_command: bool = False):
            super().__init__()
            self.value = value
            self.is_command = is_command

    def __init__(
        self,
        placeholder: str = "",
        dropdown_id: str = "#command-dropdown",
        status_id: str = "#status-panel",
        **kwargs,
    ):
        """Initialize the chat input.

        Args:
            placeholder: Placeholder text shown when input is empty.
            dropdown_id: CSS selector for the command dropdown widget.
            status_id: CSS selector for the status panel widget.
        """
        super().__init__(**kwargs)
        self.show_line_numbers = False
        self.cursor_blink = False
        self.highlight_line = False
        self._dropdown_id = dropdown_id
        self._status_id = status_id
        # Set placeholder if provided (TextArea supports this attribute)
        if placeholder:
            self.placeholder = placeholder
        # History support
        self._history: list[str] = []
        self._history_index: int = -1  # -1 means current input (not in history)
        self._current_input: str = ""  # Save current input when navigating history
        self._history_file = os.path.expanduser("~/.osprey_cli_history")
        self._expected_text: str = ""  # Track expected text after programmatic changes
        self._load_history()

    # Tips shown when input is focused
    INPUT_TIPS = [
        ("/", "commands"),
        ("option + ⏎", "newline"),
        ("\u2191\u2193", "history"),
    ]

    # Tips shown when focus is elsewhere (chat navigation)
    CHAT_TIPS = [
        ("Ctrl+L", "focus input"),
        ("tab", "cycle links"),
        ("j/k", "scroll"),
        ("g/G", "top/bottom"),
    ]

    def _on_focus(self, event: Focus) -> None:
        """Show input-specific tips when focused."""
        status = self._get_status_panel()
        if status:
            status.set_tips(self.INPUT_TIPS)

    # Tips shown on welcome screen when focus is lost (no chat to navigate)
    WELCOME_BLUR_TIPS = [
        ("Ctrl+L", "focus input"),
    ]

    def _on_blur(self, event: Blur) -> None:
        """Show navigation tips when focus leaves input."""
        status = self._get_status_panel()
        if status:
            # On welcome screen, show simplified refocus hint
            if self.id == "welcome-input":
                status.set_tips(self.WELCOME_BLUR_TIPS)
            else:
                status.set_tips(self.CHAT_TIPS)

    def _load_history(self) -> None:
        """Load history from file (prompt_toolkit FileHistory format)."""
        try:
            if os.path.exists(self._history_file):
                with open(self._history_file) as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and timestamp comments
                        if not line or line.startswith("#"):
                            continue
                        # Lines starting with + are queries
                        if line.startswith("+"):
                            self._history.append(line[1:])  # Strip the + prefix
        except Exception:
            self._history = []

    def _save_to_history(self, query: str) -> None:
        """Append query to history file (prompt_toolkit FileHistory format)."""
        if not query.strip():
            return
        query = query.strip()
        # Add to in-memory history
        self._history.append(query)
        # Append to file in prompt_toolkit format
        try:
            with open(self._history_file, "a") as f:
                # Write timestamp comment
                f.write(f"\n# {datetime.now()}\n")
                # Write query with + prefix
                f.write(f"+{query}\n")
        except Exception:
            pass  # Non-critical: history file write failed

    def _history_up(self) -> None:
        """Navigate to previous history entry."""
        if not self._history:
            return

        # Save current input if starting to navigate
        if self._history_index == -1:
            self._current_input = self.text

        # Move up in history (toward older entries)
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            # History is stored oldest-first, so we navigate from end
            history_entry = self._history[-(self._history_index + 1)]
            self._expected_text = history_entry
            self.text = history_entry
            # Move cursor to first line (so up arrow can continue navigating history)
            self.move_cursor((0, 0))

    def _history_down(self) -> None:
        """Navigate to next history entry (toward current input)."""
        # If not navigating history, do nothing (prevents losing current input)
        if self._history_index == -1:
            return

        if self._history_index == 0:
            # Return to current input
            self._history_index = -1
            self._expected_text = self._current_input
            self.text = self._current_input
            self.move_cursor(self.document.end)
            return

        # Move down in history (toward newer entries)
        self._history_index -= 1
        history_entry = self._history[-(self._history_index + 1)]
        self._expected_text = history_entry
        self.text = history_entry
        self.move_cursor(self.document.end)

    def _on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Reset history state when user edits text (not programmatic changes)."""
        if self._history_index > -1 and self.text != self._expected_text:
            # User edited a history entry - treat as new input
            self._current_input = self.text
            self._history_index = -1
            self._expected_text = ""

        # Update command dropdown
        self._update_command_dropdown()

    def _get_dropdown(self) -> CommandDropdown | None:
        """Get the command dropdown widget."""
        try:
            return self.app.query_one(self._dropdown_id, CommandDropdown)
        except Exception:
            return None

    def _update_command_dropdown(self) -> None:
        """Show/hide command dropdown based on current text."""
        dropdown = self._get_dropdown()
        if not dropdown:
            return

        text = self.text.strip()
        if text.startswith("/") and "\n" not in text:
            # Show dropdown with matching commands
            dropdown.show_matches(text)
        else:
            # Hide dropdown
            dropdown.hide()
            # Reset status panel when dropdown hides
            self._reset_status_panel()

    def _get_status_panel(self) -> StatusPanel | None:
        """Get the status panel widget."""
        try:
            return self.app.query_one(self._status_id, StatusPanel)
        except Exception:
            return None

    def _show_options_hint(self, options: list[str]) -> None:
        """Show options hint in the status panel.

        Args:
            options: List of available options.
        """
        status = self._get_status_panel()
        if status:
            options_str = ", ".join(options)
            status.set_tips([("Options:", options_str)])

    def _reset_status_panel(self) -> None:
        """Reset status panel to input tips (called when input has focus)."""
        status = self._get_status_panel()
        if status:
            status.set_tips(self.INPUT_TIPS)

    def _on_key(self, event: Key) -> None:
        """Handle key events - Enter submits, Option+Enter for newline."""
        dropdown = self._get_dropdown()
        dropdown_visible = dropdown and dropdown.is_visible

        # Handle Escape - hide dropdown
        if event.key == "escape" and dropdown_visible:
            event.prevent_default()
            event.stop()
            dropdown.hide()
            self._reset_status_panel()
            return

        # Handle Tab - select from dropdown if visible
        if event.key == "tab" and dropdown_visible:
            event.prevent_default()
            event.stop()
            command, is_complete = dropdown.select_highlighted()
            if command:
                if is_complete:
                    # Complete command - just insert it
                    self.text = command
                    self.move_cursor(self.document.end)
                else:
                    # Incomplete command - show with colon for options
                    self.text = command + ":"
                    self.move_cursor(self.document.end)
                    # Update status panel to show options hint
                    self._show_options_hint(dropdown._pending_options)
            return

        if event.key == "enter":
            # If dropdown visible, handle selection
            if dropdown_visible:
                event.prevent_default()
                event.stop()
                command, is_complete = dropdown.select_highlighted()
                if command:
                    if is_complete:
                        # Complete command - execute immediately
                        self._history_index = -1
                        self._current_input = ""
                        self.post_message(self.Submitted(command, is_command=True))
                        self.clear()
                    else:
                        # Incomplete command - show with colon for options
                        self.text = command + ":"
                        self.move_cursor(self.document.end)
                        # Update status panel to show options hint
                        self._show_options_hint(dropdown._pending_options)
                return

            # Enter = submit
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                is_command = text.startswith("/")
                if not is_command:
                    self._save_to_history(text)  # Only save non-commands to history
                self._history_index = -1  # Reset history position
                self._current_input = ""  # Clear saved input
                self.post_message(self.Submitted(text, is_command=is_command))
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
        # History/dropdown navigation (Up/Down arrows)
        elif event.key == "up":
            # If dropdown visible, navigate dropdown
            if dropdown_visible:
                event.prevent_default()
                event.stop()
                dropdown.move_highlight_up()
                return

            row, _ = self.cursor_location
            if row == 0:
                # On first line - navigate history
                event.prevent_default()
                event.stop()
                self._history_up()
                return
            # Otherwise, let parent handle normal line navigation
            super()._on_key(event)
            return
        elif event.key == "down":
            # If dropdown visible, navigate dropdown
            if dropdown_visible:
                event.prevent_default()
                event.stop()
                dropdown.move_highlight_down()
                return

            row, _ = self.cursor_location
            last_row = self.document.line_count - 1
            if row >= last_row:
                # On last line - navigate history
                event.prevent_default()
                event.stop()
                self._history_down()
                return
            # Otherwise, let parent handle normal line navigation
            super()._on_key(event)
            return
        # Let parent handle all other keys
        super()._on_key(event)


class StatusPanel(Static):
    """Status panel showing tips and shortcuts below the input area."""

    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "status-panel--command",
        "status-panel--description",
    }

    def __init__(self, **kwargs):
        """Initialize the status panel with default tips."""
        super().__init__(**kwargs)
        # Defer styled tips until mounted (CSS not available in __init__)
        self._default_tips = ChatInput.INPUT_TIPS

    def on_mount(self) -> None:
        """Set styled tips after mount when CSS is available."""
        self.set_tips(self._default_tips)

    def set_message(self, parts: list[tuple[str, str]]) -> None:
        """Set styled message with explicit style for each part.

        Args:
            parts: List of (text, style) tuples where style is "cmd" or "desc".
        """
        cmd_style = Style.from_styles(self.get_component_styles("status-panel--command"))
        desc_style = Style.from_styles(self.get_component_styles("status-panel--description"))

        styled_parts = []
        for text, style in parts:
            s = cmd_style if style == "cmd" else desc_style
            styled_parts.append((text, s))

        self.update(Content.assemble(*styled_parts))

    def set_tips(self, tips: list[tuple[str, str]]) -> None:
        """Set tips with (command, description) pairs separated by ·.

        Args:
            tips: List of (command, description) tuples.
        """
        parts = []
        for i, (cmd, desc) in enumerate(tips):
            if i > 0:
                parts.append((" · ", "desc"))
            parts.append((cmd, "cmd"))
            parts.append((f" {desc}", "desc"))
        self.set_message(parts)


class CommandDropdown(OptionList):
    """Dropdown showing matching slash commands with descriptions."""

    can_focus = False  # Controlled via ChatInput, not directly focusable

    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "command-dropdown--command",
        "command-dropdown--description",
    }

    # Available commands with metadata (description and options)
    COMMANDS = {
        # Simple commands (no options)
        "/help": {"desc": "Show available commands", "options": None},
        "/clear": {"desc": "Clear conversation", "options": None},
        "/config": {"desc": "Show configuration", "options": None},
        "/status": {"desc": "System health check", "options": None},
        "/exit": {"desc": "Exit the application", "options": None},
        # Toggle commands (with options)
        "/planning": {"desc": "Toggle planning mode", "options": ["on", "off"]},
        "/approval": {
            "desc": "Control approval workflow",
            "options": ["on", "off", "selective"],
        },
        "/task": {"desc": "Toggle task extraction", "options": ["on", "off"]},
        "/caps": {"desc": "Toggle capability selection", "options": ["on", "off"]},
    }

    # Max command length for alignment (computed once)
    _MAX_CMD_LEN = max(len(cmd) for cmd in COMMANDS.keys())

    class CommandSelected(Message):
        """Event posted when a command is selected from dropdown."""

        def __init__(self, command: str):
            super().__init__()
            self.command = command

    def __init__(self, **kwargs):
        """Initialize the command dropdown."""
        super().__init__(**kwargs)
        self._visible = False
        self._mode = "commands"  # "commands" or "options"
        self._pending_command: str | None = None  # e.g., "/planning" when showing options
        self._pending_options: list[str] = []  # ["on", "off"] for the pending command

    def on_mount(self) -> None:
        """Hide dropdown initially."""
        self.display = False

    def show_matches(self, prefix: str) -> None:
        """Show dropdown with commands or options matching the prefix.

        Args:
            prefix: The current input text (should start with /).
        """
        # Hide plan progress bar (mutual exclusivity with overlays)
        try:
            from osprey.interfaces.tui.widgets.plan_progress import PlanProgressBar

            progress_bar = self.app.query_one("#plan-progress", PlanProgressBar)
            if progress_bar.display:
                progress_bar.display = False
                progress_bar.refresh()
        except Exception:
            pass  # Progress bar may not exist

        self.clear_options()
        prefix_lower = prefix.lower()

        # Check if text contains ":" - might be editing options for a toggle command
        if ":" in prefix_lower:
            base_cmd, partial_opt = prefix_lower.rsplit(":", 1)

            # Look up the base command to get its options
            meta = self.COMMANDS.get(base_cmd)
            if meta and meta["options"]:
                # This is a toggle command - show matching options
                self._mode = "options"
                self._pending_command = base_cmd
                self._pending_options = meta["options"]

                for opt in self._pending_options:
                    if opt.startswith(partial_opt):
                        self.add_option(Option(opt, id=opt))

                if self.option_count > 0:
                    self.display = True
                    self._visible = True
                    self.highlighted = 0
                    self._position_upward()
                else:
                    self.hide()
                return

        # Normal command matching mode
        self._mode = "commands"
        self._pending_command = None
        self._pending_options = []

        matches = []
        for cmd, meta in self.COMMANDS.items():
            if cmd.startswith(prefix_lower):
                matches.append((cmd, meta["desc"]))

        if matches:
            # Get styles from CSS
            cmd_style = Style.from_styles(self.get_component_styles("command-dropdown--command"))
            desc_style = Style.from_styles(
                self.get_component_styles("command-dropdown--description")
            )

            for cmd, desc in matches:
                # Pad command to align descriptions (table-like)
                padded_cmd = cmd.ljust(self._MAX_CMD_LEN + 8)
                # Create styled content: command + description
                prompt = Content.assemble(
                    (padded_cmd, cmd_style),  # Command with CSS style
                    (desc, desc_style),  # Description with CSS style
                )
                self.add_option(Option(prompt, id=cmd))
            self.display = True
            self._visible = True
            # Highlight first option
            self.highlighted = 0
            # Position dropdown upward
            self._position_upward()
        else:
            self.hide()

    def hide(self) -> None:
        """Hide the dropdown and reset state."""
        self.display = False
        self._visible = False
        self._mode = "commands"
        self._pending_command = None
        self._pending_options = []
        # Reset offset when hiding
        self.styles.offset = (0, 0)

    def _position_upward(self) -> None:
        """Position dropdown to open upward (bottom aligns with input top)."""
        # Call after next refresh to get accurate height
        self.call_after_refresh(self._apply_upward_offset)

    def _apply_upward_offset(self) -> None:
        """Apply negative offset to position dropdown above its DOM position."""
        if self.display and self.option_count > 0:
            # Each option is ~1 row, plus border (tall border = 2 rows total)
            height = self.option_count + 2
            self.styles.offset = (0, -height)

    @property
    def is_visible(self) -> bool:
        """Check if dropdown is currently visible."""
        return self._visible

    def select_highlighted(self) -> tuple[str | None, bool]:
        """Select the currently highlighted option.

        Returns:
            Tuple of (command_string, is_complete).
            is_complete=False means we need to show options dropdown.
        """
        if self.highlighted is None or self.option_count == 0:
            return (None, False)

        option = self.get_option_at_index(self.highlighted)
        if not option or not option.id:
            return (None, False)

        selected_id = str(option.id)

        if self._mode == "commands":
            # Selecting a command
            meta = self.COMMANDS.get(selected_id)

            if meta and meta["options"]:
                # Has options - switch to options mode
                self._pending_command = selected_id
                self._pending_options = meta["options"]
                self._show_options_dropdown()
                return (selected_id, False)  # Incomplete - need option
            else:
                # Simple command - complete
                self.hide()
                return (selected_id, True)

        elif self._mode == "options":
            # Selecting an option
            full_command = f"{self._pending_command}:{selected_id}"
            self.hide()
            return (full_command, True)

        return (None, False)

    def _show_options_dropdown(self) -> None:
        """Show options dropdown for a toggle command."""
        self.clear_options()
        self._mode = "options"

        for opt in self._pending_options:
            self.add_option(Option(opt, id=opt))

        self.display = True
        self._visible = True
        self.highlighted = 0
        # Position dropdown upward
        self._position_upward()

    def move_highlight_up(self) -> None:
        """Move highlight to previous option, cycling to last if at first."""
        if self.highlighted is not None and self.option_count > 0:
            if self.highlighted == 0:
                self.highlighted = self.option_count - 1  # Cycle to last
            else:
                self.highlighted = self.highlighted - 1

    def move_highlight_down(self) -> None:
        """Move highlight to next option, cycling to first if at last."""
        if self.highlighted is not None and self.option_count > 0:
            if self.highlighted >= self.option_count - 1:
                self.highlighted = 0  # Cycle to first
            else:
                self.highlighted = self.highlighted + 1


class InfoBar(Horizontal):
    """Bottom info bar showing environment, mode, and tips.

    Always visible at the bottom of the window (1 line height).
    SSH mode uses $warning background via CSS class.
    Split layout: mode on left, shortcuts on right.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "info-bar--label",
        "info-bar--value",
    }

    def __init__(self, is_ssh: bool = False, **kwargs):
        """Initialize the info bar.

        Args:
            is_ssh: Whether the session is running over SSH.
        """
        super().__init__(**kwargs)
        self._is_ssh = is_ssh
        self._selection_mode = False
        self._restore_timer = None

    def compose(self) -> ComposeResult:
        """Compose InfoBar with left (mode) and right (shortcuts) sections."""
        yield Static("", id="info-bar-left")
        yield Static("", id="info-bar-right")

    def on_mount(self) -> None:
        """Apply SSH class and render initial content."""
        if self._is_ssh:
            self.add_class("info-bar--ssh")
        self._update_content()

    def set_selection_mode(self, enabled: bool) -> None:
        """Update mode display (no notification, just update content)."""
        self._selection_mode = enabled
        if enabled:
            self.add_class("info-bar--select-mode")
        else:
            self.remove_class("info-bar--select-mode")
        self._update_content()

    def show_temporary(self, msg: str, duration: float = 2.0) -> None:
        """Show temporary message, then restore previous content."""
        label_style = Style.from_styles(self.get_component_styles("info-bar--label"))
        # Show message in right section
        try:
            right = self.query_one("#info-bar-right", Static)
            right.update(Content.assemble((msg, label_style)))
        except Exception:
            pass
        # Cancel any pending restore
        if self._restore_timer is not None:
            self._restore_timer.stop()
        self._restore_timer = self.set_timer(duration, self._restore_content)

    def _restore_content(self) -> None:
        """Restore normal content after temporary message."""
        self._restore_timer = None
        self._update_content()

    def _update_content(self) -> None:
        """Render the info bar content based on current state."""
        label_style = Style.from_styles(self.get_component_styles("info-bar--label"))
        value_style = Style.from_styles(self.get_component_styles("info-bar--value"))

        # Build mode label (left) and shortcuts (right) separately
        mode_parts: list[tuple[str, Style]] = []
        shortcut_parts: list[tuple[str, Style]] = []
        sep = (" \u00b7 ", value_style)  # " · "

        if self._selection_mode:
            # Left: mode label
            mode_parts.append(("SELECT MODE", label_style))
            # Right: shortcuts
            shortcut_parts.append(("drag", label_style))
            shortcut_parts.append((" to copy", value_style))
            shortcut_parts.append(sep)
            shortcut_parts.append(("Ctrl+S", label_style))
            shortcut_parts.append((" exit", value_style))
            shortcut_parts.append(sep)
            shortcut_parts.append(("Ctrl+Y", label_style))
            shortcut_parts.append((" copy response", value_style))
        elif self._is_ssh:
            # Left: mode label
            mode_parts.append(("SSH", label_style))
            # Right: shortcuts
            shortcut_parts.append(("Shift+drag", label_style))
            shortcut_parts.append((" to select", value_style))
            shortcut_parts.append((" (", value_style))
            shortcut_parts.append(("Option+drag", label_style))
            shortcut_parts.append((" in iTerm)", value_style))
            shortcut_parts.append(sep)
            shortcut_parts.append(("Ctrl+S", label_style))
            shortcut_parts.append((" select mode", value_style))
            shortcut_parts.append(sep)
            shortcut_parts.append(("Ctrl+Y", label_style))
            shortcut_parts.append((" copy response", value_style))
        else:
            # Left: empty (no mode label)
            # Right: shortcuts only
            shortcut_parts.append(("Ctrl+S", label_style))
            shortcut_parts.append((" select mode", value_style))
            shortcut_parts.append(sep)
            shortcut_parts.append(("Ctrl+Y", label_style))
            shortcut_parts.append((" copy response", value_style))

        # Update left and right sections
        try:
            left = self.query_one("#info-bar-left", Static)
            right = self.query_one("#info-bar-right", Static)
            left.update(Content.assemble(*mode_parts) if mode_parts else "")
            right.update(Content.assemble(*shortcut_parts))
        except Exception:
            pass
