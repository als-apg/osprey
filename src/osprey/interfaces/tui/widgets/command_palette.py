"""Command palette widget for the TUI."""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.content import Content
from textual.events import Key, Resize
from textual.screen import ModalScreen
from textual.style import Style
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


class CommandPalette(ModalScreen[str | None]):
    """Modal command palette with search and categorized commands."""

    BINDINGS = [
        ("escape", "dismiss_palette", "Close"),
        Binding("tab", "noop", "", show=False),  # Block tab from moving focus
        Binding("shift+tab", "noop", "", show=False),  # Block shift+tab too
    ]

    # Command registry: {id: {label, shortcut, category}}
    # Ordered by category for display
    COMMANDS: ClassVar[dict[str, dict[str, str]]] = {
        "focus_input": {
            "label": "Focus input",
            "shortcut": "ctrl + l",
            "category": "Session",
        },
        "switch_theme": {
            "label": "Switch theme",
            "shortcut": "ctrl + t",
            "category": "System",
        },
    }

    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "palette--label",
        "palette--shortcut",
        "palette--category",
    }

    # Container width 60, padding 4*2=8, leaves 52 usable
    OPTION_WIDTH: ClassVar[int] = 52

    def compose(self) -> ComposeResult:
        """Compose the command palette layout."""
        with Container(id="palette-container"):
            with Horizontal(id="palette-header"):
                yield Static("Commands", id="palette-title")
                yield Static("esc", id="palette-dismiss-hint")
            yield Input(placeholder="Search", id="palette-search")
            yield OptionList(id="palette-options")

    def on_mount(self) -> None:
        """Initialize the palette on mount."""
        # Set initial margin based on app size (before first render)
        self._update_position(self.app.size.height)
        self._populate_options()
        # Disable focus on option list - keep focus on search bar
        self.query_one("#palette-options", OptionList).can_focus = False
        search_input = self.query_one("#palette-search", Input)
        search_input.cursor_blink = False
        search_input.focus()

    def on_resize(self, event: Resize) -> None:
        """Update palette position when terminal is resized."""
        self._update_position(event.size.height)

    def _update_position(self, screen_height: int) -> None:
        """Set palette margin and option list height based on screen size."""
        container = self.query_one("#palette-container", Container)
        options_list = self.query_one("#palette-options", OptionList)

        # Calculate natural content height
        content_height = self._calculate_content_height()

        # Max palette height is 50% of screen
        max_height = screen_height // 2

        # Palette height is min of content and max
        palette_height = min(content_height, max_height)

        # Center the palette vertically
        margin_top = (screen_height - palette_height) // 2
        container.styles.margin = (margin_top, 0, 0, 0)

        # Calculate and set OptionList max-height
        # Overhead: container padding (2) + header+margin (2) + search+margin (2) = 6
        overhead = 6
        options_max_height = max(1, max_height - overhead)
        options_list.styles.max_height = options_max_height
        options_list.styles.overflow_y = "auto"  # Enable scrolling

        # Debug: uncomment to verify values
        # self.log(f"screen={screen_height}, max_height={max_height}, options_max={options_max_height}")

    def _calculate_content_height(self) -> int:
        """Calculate the natural height of the palette content."""
        height = 2  # Container padding (1 top + 1 bottom)
        height += 2  # Header (1) + margin-bottom (1)
        height += 2  # Search (1) + margin-bottom (1)

        # Count categories and commands
        num_categories = len({cmd["category"] for cmd in self.COMMANDS.values()})
        num_commands = len(self.COMMANDS)

        # First category: just header (1), no spacer
        # Additional categories: spacer (1) + header (1) each
        # Each command: 1 row
        if num_categories > 0:
            height += num_categories  # Category headers
            height += max(0, num_categories - 1)  # Spacers (skip first)
        height += num_commands

        return height

    def _populate_options(self, filter_text: str = "") -> None:
        """Populate the options list with commands grouped by category.

        Args:
            filter_text: Text to filter commands by.
        """
        options_list = self.query_one("#palette-options", OptionList)
        options_list.clear_options()

        # Get styles for Content.assemble
        label_style = Style.from_styles(self.get_component_styles("palette--label"))
        shortcut_style = Style.from_styles(self.get_component_styles("palette--shortcut"))
        category_style = Style.from_styles(self.get_component_styles("palette--category"))

        # Group commands by category
        categories: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
        filter_lower = filter_text.lower()

        for cmd_id, cmd_data in self.COMMANDS.items():
            # Filter by label or shortcut
            if filter_lower:
                label_match = filter_lower in cmd_data["label"].lower()
                shortcut_match = filter_lower in cmd_data["shortcut"].lower()
                if not label_match and not shortcut_match:
                    continue
            categories[cmd_data["category"]].append((cmd_id, cmd_data))

        # Add options grouped by category
        first_category = True
        for category, cmds in categories.items():
            if not cmds:
                continue

            # Add spacing before category (except first)
            if not first_category:
                options_list.add_option(Option("", disabled=True, id=f"spacer_{category}"))
            first_category = False

            # Add category header with styled content (non-selectable)
            category_content = Content.assemble((category, category_style))
            options_list.add_option(Option(category_content, disabled=True, id=f"cat_{category}"))

            # Add commands in this category
            for cmd_id, cmd_data in cmds:
                label = cmd_data["label"]
                shortcut = cmd_data["shortcut"]
                # Calculate padding to push shortcut to right edge
                pad_len = self.OPTION_WIDTH - len(label) - len(shortcut)
                padding = " " * max(pad_len, 2)

                prompt = Content.assemble(
                    (label, label_style),
                    (padding, label_style),
                    (shortcut, shortcut_style),
                )
                options_list.add_option(Option(prompt, id=cmd_id))

        # Highlight first selectable option
        if options_list.option_count > 0:
            # Skip category headers to find first selectable
            for i in range(options_list.option_count):
                opt = options_list.get_option_at_index(i)
                if opt and not opt.disabled:
                    options_list.highlighted = i
                    break

        # Refresh option list to recalculate scroll
        options_list.refresh(layout=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter options when search input changes."""
        if event.input.id == "palette-search":
            self._populate_options(event.value)

    def on_key(self, event: Key) -> None:
        """Handle keyboard navigation while keeping focus on search."""
        options = self.query_one("#palette-options", OptionList)

        if event.key == "down":
            options.action_cursor_down()
            options.scroll_to_highlight()
            event.prevent_default()
        elif event.key == "up":
            options.action_cursor_up()
            options.scroll_to_highlight()
            event.prevent_default()
        elif event.key == "enter":
            if options.highlighted is not None:
                opt = options.get_option_at_index(options.highlighted)
                if opt and opt.id and not str(opt.id).startswith("cat_"):
                    self.dismiss(str(opt.id))
            event.prevent_default()

    def action_dismiss_palette(self) -> None:
        """Dismiss the palette without selecting."""
        self.dismiss(None)

    def action_noop(self) -> None:
        """Do nothing - blocks default tab behavior."""
        pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        if event.option.id and not str(event.option.id).startswith("cat_"):
            self.dismiss(str(event.option.id))
