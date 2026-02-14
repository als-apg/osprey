"""Theme picker widget for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.content import Content
from textual.events import Key, Resize
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from osprey.interfaces.tui.widgets.command_palette import CommandPalette


class ThemePicker(ModalScreen[str | None]):
    """Modal for selecting application theme with live preview."""

    BINDINGS = [
        ("escape", "dismiss_picker", "Close"),
        Binding("tab", "noop", "", show=False),
        Binding("shift+tab", "noop", "", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._original_theme: str = ""

    def compose(self) -> ComposeResult:
        with Container(id="theme-container"):
            with Horizontal(id="theme-header"):
                yield Static("Themes", id="theme-title")
                yield Static("esc", id="theme-dismiss-hint")
            yield Input(placeholder="Search", id="theme-search")
            yield OptionList(id="theme-options")

    def on_mount(self) -> None:
        # Store original theme for revert
        self._original_theme = self.app.theme
        # Set initial position based on app size (before first render)
        self._update_position(self.app.size.height)
        self._populate_options()
        # Focus search input and disable cursor blink
        search_input = self.query_one("#theme-search", Input)
        search_input.cursor_blink = False
        search_input.focus()

    def on_resize(self, event: Resize) -> None:
        """Update picker position and option list height based on screen size."""
        self._update_position(event.size.height)

    def _update_position(self, screen_height: int) -> None:
        """Set picker margin and option list height based on screen size."""
        container = self.query_one("#theme-container", Container)
        options_list = self.query_one("#theme-options", OptionList)

        # Calculate natural content height
        content_height = self._calculate_content_height()

        # Max picker height is 50% of screen
        max_height = screen_height // 2

        # Picker height is min of content and max
        picker_height = min(content_height, max_height)

        # Center the picker vertically
        margin_top = (screen_height - picker_height) // 2
        container.styles.margin = (margin_top, 0, 0, 0)

        # Calculate and set OptionList max-height
        # Overhead: container padding (2) + header+margin (2) + search+margin (2) = 6
        overhead = 6
        options_max_height = max(1, max_height - overhead)
        options_list.styles.max_height = options_max_height
        options_list.styles.overflow_y = "auto"

    def _calculate_content_height(self) -> int:
        """Calculate content height using CommandPalette's COMMANDS for consistency."""
        height = 2  # Container padding (1 top + 1 bottom)
        height += 2  # Header (1) + margin-bottom (1)
        height += 2  # Search (1) + margin-bottom (1)

        # Use CommandPalette's COMMANDS for consistent height
        num_categories = len({cmd["category"] for cmd in CommandPalette.COMMANDS.values()})
        num_commands = len(CommandPalette.COMMANDS)

        if num_categories > 0:
            height += num_categories  # Category headers
            height += max(0, num_categories - 1)  # Spacers (skip first)
        height += num_commands

        return height

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter options when search input changes."""
        if event.input.id == "theme-search":
            self._populate_options(event.value)

    def _populate_options(self, filter_text: str = "") -> None:
        options_list = self.query_one("#theme-options", OptionList)
        options_list.can_focus = False
        options_list.clear_options()

        # Group themes by dark/light, apply filter (exclude textual-ansi)
        themes = self.app.available_themes
        filter_lower = filter_text.lower()
        dark_themes = [
            (name, t)
            for name, t in themes.items()
            if t.dark and filter_lower in name.lower() and name != "textual-ansi"
        ]
        light_themes = [
            (name, t)
            for name, t in themes.items()
            if not t.dark and filter_lower in name.lower() and name != "textual-ansi"
        ]

        current_theme = self._original_theme
        first_category = True
        first_selectable_idx = None
        current_theme_idx = None
        current_idx = 0

        for category, theme_list in [("Dark", dark_themes), ("Light", light_themes)]:
            if not theme_list:
                continue

            if not first_category:
                options_list.add_option(Option("", disabled=True, id=f"spacer_{category}"))
                current_idx += 1
            first_category = False

            # Category header - use from_markup for CSS variable resolution
            cat_content = Content.from_markup(f"[$primary bold]  {category}[/]")
            options_list.add_option(Option(cat_content, disabled=True, id=f"cat_{category}"))
            current_idx += 1

            # Theme options - use from_markup for CSS variable resolution
            for name, _theme in sorted(theme_list, key=lambda x: x[0]):
                if name == current_theme:
                    content = Content.from_markup(f"[$accent]● [/]{name}")
                else:
                    content = Content.from_markup(f"  {name}")
                options_list.add_option(Option(content, id=name))

                # Track indices
                if first_selectable_idx is None:
                    first_selectable_idx = current_idx
                if name == current_theme:
                    current_theme_idx = current_idx
                current_idx += 1

        # Highlight current theme, or first selectable if not found
        if current_theme_idx is not None:
            options_list.highlighted = current_theme_idx
        elif first_selectable_idx is not None:
            options_list.highlighted = first_selectable_idx

        # Refresh option list to recalculate scroll
        options_list.refresh(layout=True)

    def on_key(self, event: Key) -> None:
        """Handle keyboard navigation while keeping focus on search."""
        options = self.query_one("#theme-options", OptionList)

        if event.key == "down":
            # Predict if we'll cycle to first selectable
            if options.highlighted is not None:
                has_next = any(
                    options.get_option_at_index(i) and not options.get_option_at_index(i).disabled
                    for i in range(options.highlighted + 1, options.option_count)
                )
                if not has_next:
                    # Find first selectable
                    for i in range(options.option_count):
                        opt = options.get_option_at_index(i)
                        if opt and not opt.disabled:
                            options.highlighted = i
                            options.scroll_to(y=0, animate=False)
                            self._preview_theme(opt.id)
                            event.prevent_default()
                            return
            options.action_cursor_down()
            self._preview_highlighted_theme()
            event.prevent_default()

        elif event.key == "up":
            if options.highlighted is not None:
                # Find previous selectable
                prev_idx = next(
                    (
                        i
                        for i in range(options.highlighted - 1, -1, -1)
                        if options.get_option_at_index(i)
                        and not options.get_option_at_index(i).disabled
                    ),
                    None,
                )
                # Find first selectable index
                first_selectable = next(
                    (
                        i
                        for i in range(options.option_count)
                        if options.get_option_at_index(i)
                        and not options.get_option_at_index(i).disabled
                    ),
                    None,
                )
                if prev_idx == first_selectable:
                    options.highlighted = first_selectable
                    options.scroll_to(y=0, animate=False)
                    opt = options.get_option_at_index(first_selectable)
                    if opt:
                        self._preview_theme(opt.id)
                    event.prevent_default()
                    return
            options.action_cursor_up()
            self._preview_highlighted_theme()
            event.prevent_default()

        elif event.key == "enter":
            if options.highlighted is not None:
                opt = options.get_option_at_index(options.highlighted)
                if opt and opt.id and not str(opt.id).startswith("cat_"):
                    self.dismiss(str(opt.id))
            event.prevent_default()

    def _preview_highlighted_theme(self) -> None:
        """Apply the currently highlighted theme for preview."""
        options = self.query_one("#theme-options", OptionList)
        if options.highlighted is not None:
            opt = options.get_option_at_index(options.highlighted)
            if opt and opt.id:
                self._preview_theme(opt.id)

    def _preview_theme(self, theme_id: str | object) -> None:
        """Apply theme for preview."""
        theme_name = str(theme_id)
        if not theme_name.startswith("cat_") and not theme_name.startswith("spacer_"):
            self.app.theme = theme_name

    def notify_style_update(self) -> None:
        """Called when CSS changes (e.g., theme change). Refresh options."""
        super().notify_style_update()
        # Force OptionList to re-render with new theme colors
        try:
            self.query_one("#theme-options", OptionList).refresh()
        except Exception:
            pass  # Widget may not be mounted yet

    def action_dismiss_picker(self) -> None:
        """Revert theme and dismiss."""
        self.app.theme = self._original_theme
        self.dismiss(None)

    def action_noop(self) -> None:
        """Do nothing - blocks default tab behavior."""
        pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle mouse selection."""
        if event.option.id and not str(event.option.id).startswith("cat_"):
            self.dismiss(str(event.option.id))
