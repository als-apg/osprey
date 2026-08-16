"""Centralized color and style management for Osprey CLI.

This module provides a unified color scheme and styling utilities for all CLI
components, ensuring consistent visual appearance across the framework.

Design Philosophy:
- Semantic color names (success, error, warning) rather than direct colors
- Theme-based approach allowing easy theme switching
- Rich console markup helpers for inline styling
- Questionary style integration for interactive prompts
- Clear separation between fixed UI standards and customizable theme colors
- One data-table look and one panel look, built by the factories at the bottom of
  this module, so box/padding/border tokens are spelled here rather than at the
  call sites the structure guard walks
"""

import sys
from dataclasses import dataclass
from typing import Any

from rich import box
from rich.box import Box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from osprey.utils.logger import get_logger

logger = get_logger("base")

try:
    from questionary import Style as QuestionaryStyle

    QUESTIONARY_AVAILABLE = True
except ImportError:
    QuestionaryStyle = None
    QUESTIONARY_AVAILABLE = False


@dataclass
class ColorTheme:
    """Defines a complete color theme for the CLI.

    Separates colors into three categories:
    1. Fixed standard colors (error, warning, success) - UI conventions
    2. Configurable theme colors (primary, accent, etc.) - customizable identity
    3. Neutral infrastructure colors - contrast and structure

    Derived colors (lighter/darker variations) are automatically calculated.
    """

    # === FIXED STANDARD COLORS (UI Conventions) ===
    # These remain consistent across themes for familiarity
    error: str = "#ff0000"  # Red - universal error indicator
    warning: str = "#ffaa00"  # Orange - universal warning indicator

    # === CONFIGURABLE THEME COLORS ===
    # These define your visual identity and can be customized
    primary: str = "#C75F71"  # Main brand color
    success: str = "#9988A1"  # Success indicator
    accent: str = "#F0B8B8"  # Interactive elements & highlights
    command: str = "#9988A1"  # Shell commands & actions
    path: str = "#A2AE9D"  # File paths & locations
    info: str = "#9988A1"  # Informational messages

    # === NEUTRAL COLORS (Dark Theme Infrastructure) ===
    # These provide contrast and structure - less important to customize
    text_primary: str = "#ffffff"
    text_secondary: str = "#888888"
    text_dim: str = "#666666"
    text_disabled: str = "#444444"
    bg_highlight: str = "#2d2d2d"
    bg_selected: str = "#1a1a1a"
    border_default: str = "#555555"
    border_dim: str = "#444444"

    def __post_init__(self):
        """Calculate derived colors from theme colors."""
        # Derive darker/lighter variations of primary
        self.primary_dark = self._adjust_brightness(self.primary, 0.85)
        self.primary_light = self._adjust_brightness(self.primary, 1.15)

        # Derive dimmed variations of standard colors
        self.success_dim = self._adjust_brightness(self.success, 0.8)
        self.error_dim = self._adjust_brightness(self.error, 0.8)
        self.warning_dim = self._adjust_brightness(self.warning, 0.8)
        self.info_dim = self._adjust_brightness(self.info, 0.8)

        # Structural colors derive from primary
        self.header = self.primary
        self.subheader = self.primary_dark

        # Accent border uses info color
        self.border_accent = self.info

    @staticmethod
    def _adjust_brightness(hex_color: str, factor: float) -> str:
        """Adjust brightness of a hex color by a factor.

        Args:
            hex_color: Hex color string (e.g., "#ff0000")
            factor: Brightness multiplier (0.0-1.0 darkens, >1.0 lightens)

        Returns:
            Adjusted hex color string
        """
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f"#{r:02x}{g:02x}{b:02x}"


# Default Osprey theme
OSPREY_THEME = ColorTheme()

# Back-compat alias: "vulcan" was never a distinct theme, only another name for
# the default. Kept so existing configs keep loading; not offered as a choice.
VULCAN_THEME = OSPREY_THEME

# Theme registry for easy lookup
THEME_REGISTRY = {
    "default": VULCAN_THEME,
    "vulcan": VULCAN_THEME,
}


_active_theme = OSPREY_THEME


def get_active_theme() -> ColorTheme:
    """Get the currently active color theme."""
    return _active_theme


def set_theme(theme: ColorTheme):
    """Set a new active theme, re-theming the existing consoles in place.

    The module-level consoles keep their identity for the process lifetime, so
    ``from .styles import console`` aliases taken at import time stay correct
    after a theme switch.

    Args:
        theme: The ColorTheme to activate
    """
    global _active_theme, custom_style
    _active_theme = theme
    _retheme_consoles(theme)
    custom_style = _build_questionary_style(theme)


def load_theme_from_config(config_path: str | None = None) -> ColorTheme:
    """Load and apply theme from configuration file.

    Args:
        config_path: Optional path to config file (uses default if None)

    Returns:
        The loaded ColorTheme instance

    Examples:
        >>> # Load theme from default config
        >>> theme = load_theme_from_config()

        >>> # Load theme from specific config
        >>> theme = load_theme_from_config("/path/to/config.yml")
    """
    from osprey.utils.config import get_config_value

    # Get theme name from config (default to "default")
    theme_name = get_config_value("cli.theme", "default", config_path)

    # Handle custom theme
    if theme_name == "custom":
        custom_colors = get_config_value("cli.custom_theme", {}, config_path)
        if custom_colors:
            try:
                # Validate hex colors before creating theme
                for key, value in custom_colors.items():
                    if not isinstance(value, str) or not value.startswith("#"):
                        logger.warning(f"Invalid color format for {key}: {value}, using default")
                        return VULCAN_THEME

                # Create custom theme from config
                theme = ColorTheme(**custom_colors)
                return theme
            except Exception as e:
                logger.warning(f"Failed to create custom theme: {e}, using default")
                return VULCAN_THEME
        else:
            logger.warning("Custom theme selected but no custom_theme config found, using default")
            theme_name = "default"

    # Load predefined theme
    theme = THEME_REGISTRY.get(theme_name)
    if theme is None:
        logger.warning(f"Unknown theme '{theme_name}', using default")
        theme = VULCAN_THEME

    return theme


def initialize_theme_from_config(config_path: str | None = None):
    """Initialize and apply theme from configuration.

    This should be called at CLI startup to apply the configured theme.

    Args:
        config_path: Optional path to config file
    """
    try:
        theme = load_theme_from_config(config_path)
        set_theme(theme)
        logger.debug("Applied theme from configuration")
    except Exception as e:
        logger.debug(f"Failed to load theme from config: {e}, using default")
        set_theme(VULCAN_THEME)


def _build_rich_theme(theme: ColorTheme) -> Theme:
    """Build a Rich Theme from a ColorTheme.

    Args:
        theme: The ColorTheme to convert

    Returns:
        Rich Theme object
    """
    return Theme(
        {
            # Status styles
            "success": f"bold {theme.success}",
            "error": f"bold {theme.error}",
            "warning": f"bold {theme.warning}",
            "info": f"bold {theme.info}",
            # Text styles
            "primary": f"bold {theme.primary}",
            "secondary": theme.text_secondary,
            "dim": theme.text_dim,
            "disabled": theme.text_disabled,
            # Emphasis styles
            "bold": "bold",
            "italic": "italic",
            "bold_primary": f"bold {theme.primary}",
            "bold_success": f"bold {theme.success}",
            "bold_error": f"bold {theme.error}",
            # Component-specific styles
            "banner": theme.primary,
            "banner_alt": "bold cyan",
            "header": f"bold {theme.header}",
            "subheader": f"bold {theme.subheader}",
            "label": "bold",
            "value": theme.success,
            "path": theme.path,
            "command": theme.command,
            "accent": theme.accent,
            # Borders and UI elements
            "border": theme.border_default,
            "border_accent": theme.border_accent,
            "border_dim": theme.border_dim,
            # System/framework messages (matches old Styles.INFO)
            "system": f"bold {theme.info}",
        }
    )


def _build_questionary_style(theme: ColorTheme) -> "QuestionaryStyle | None":
    """Build a Questionary style from a ColorTheme.

    Args:
        theme: The ColorTheme to convert

    Returns:
        QuestionaryStyle object or None if questionary not installed
    """
    if QuestionaryStyle is None:
        return None

    return QuestionaryStyle(
        [
            ("qmark", f"fg:{theme.accent} bold"),  # Question mark (accent)
            ("question", "bold"),  # Question text
            ("answer", f"fg:{theme.primary} bold"),  # User's answer (primary)
            ("pointer", f"fg:{theme.primary} bold"),  # Selection pointer (primary)
            ("highlighted", f"fg:{theme.primary} bold"),  # Highlighted item (primary)
            ("selected", f"fg:{theme.accent}"),  # Selected item (accent)
            ("separator", f"fg:{theme.text_dim}"),  # Separators
            ("instruction", f"fg:{theme.text_dim} italic"),  # Instructions
            ("text", f"fg:{theme.text_secondary}"),  # Regular text
            ("disabled", f"fg:{theme.text_dim}"),  # Disabled items
            ("default", f"fg:{theme.text_primary}"),  # Default items
        ]
    )


# Build the initial theme objects from the active theme
osprey_theme = _build_rich_theme(_active_theme)
custom_style = _build_questionary_style(_active_theme)


def _make_console(*, stderr: bool = False) -> Console:
    """Construct a themed console with the platform-constant terminal settings.

    The Windows special case (treat the stream as a terminal, and use the modern
    console renderer rather than the legacy one, so Unicode characters such as
    ✓, ✗ and ⚠️ render) is platform-constant, so it is applied once here at
    construction and never revisited when the theme changes.

    Args:
        stderr: If True, the console writes to stderr instead of stdout

    Returns:
        A themed Console instance
    """
    if sys.platform == "win32":
        return Console(theme=osprey_theme, force_terminal=True, legacy_windows=False, stderr=stderr)
    return Console(theme=osprey_theme, stderr=stderr)


# Singleton console instances with theme: one for stdout, one for stderr.
# Both are identity-stable for the process lifetime - set_theme() re-themes them
# in place rather than rebuilding them.
console = _make_console()
err_console = _make_console(stderr=True)

# Whether a theme has been pushed on top of the consoles' base theme. Guards the
# pop half of the pop-before-push so the first set_theme() call is safe.
_theme_pushed = False


def _retheme_consoles(theme: ColorTheme) -> None:
    """Re-theme the module consoles in place, keeping the theme stack flat.

    Pops the previously pushed theme (if any) before pushing the new one, so any
    number of theme switches leaves the consoles' theme stack at constant depth.

    Args:
        theme: The ColorTheme to apply to both consoles
    """
    global _theme_pushed

    rich_theme = _build_rich_theme(theme)
    for target in (console, err_console):
        if _theme_pushed:
            target.pop_theme()
        target.push_theme(rich_theme, inherit=False)
    _theme_pushed = True


def get_questionary_style() -> "QuestionaryStyle | None":
    """Get the Questionary style for interactive prompts.

    Returns the current active theme's questionary style.

    Returns:
        QuestionaryStyle object or None if questionary not installed
    """
    return custom_style


class Styles:
    """Collection of reusable style strings for Rich markup.

    These are string constants that reference styles defined in the Rich theme.
    Use these for consistent styling across the CLI.
    """

    # Status indicators
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    # Text styles
    BOLD = "bold"
    DIM = "dim"
    ITALIC = "italic"
    PRIMARY = "primary"
    SECONDARY = "secondary"

    # Combined styles
    BOLD_PRIMARY = "bold_primary"
    BOLD_SUCCESS = "bold_success"
    BOLD_ERROR = "bold_error"

    # Component styles
    BANNER = "banner"
    HEADER = "header"
    SUBHEADER = "subheader"
    LABEL = "label"
    VALUE = "value"
    PATH = "path"
    COMMAND = "command"
    ACCENT = "accent"

    # Borders
    BORDER = "border"
    BORDER_ACCENT = "border_accent"
    BORDER_DIM = "border_dim"


class Messages:
    """Pre-formatted message helpers for common patterns.

    These provide consistent formatting for status messages and UI elements.
    """

    @staticmethod
    def success(text: str) -> str:
        """Format a success message with checkmark."""
        return f"[success]✓ {text}[/success]"

    @staticmethod
    def error(text: str) -> str:
        """Format an error message with X mark."""
        return f"[error]✗ {text}[/error]"

    @staticmethod
    def warning(text: str) -> str:
        """Format a warning message with warning symbol."""
        return f"[warning]! {text}[/warning]"

    @staticmethod
    def info(text: str) -> str:
        """Format an info message with info symbol."""
        return f"[info]{text}[/info]"

    @staticmethod
    def header(text: str) -> str:
        """Format a section header."""
        return f"[header]{text}[/header]"

    @staticmethod
    def label_value(label: str, value: str) -> str:
        """Format a label-value pair."""
        return f"[label]{label}:[/label] [value]{value}[/value]"

    @staticmethod
    def command(text: str) -> str:
        """Format a command string."""
        return f"[command]{text}[/command]"

    @staticmethod
    def path(text: str) -> str:
        """Format a file path."""
        return f"[path]{text}[/path]"


class ThemeConfig:
    """Configuration utilities for theme customization.

    Provides helper methods for accessing theme settings consistently. Loading a
    theme from config, custom palettes included, is
    :func:`load_theme_from_config`.
    """

    @staticmethod
    def get_spinner_style() -> str:
        """Get the spinner style for loading indicators."""
        return "info"

    @staticmethod
    def get_border_style(dim: bool = False) -> str:
        """Get border style for panels and tables.

        Args:
            dim: If True, return dimmed border style

        Returns:
            Style name for borders
        """
        return Styles.BORDER_DIM if dim else Styles.BORDER

    @staticmethod
    def get_banner_style() -> str:
        """Get the style for ASCII art banners."""
        return Styles.BANNER


# ---------------------------------------------------------------------------
# Table and panel factories
#
# Every data table and panel under ``cli/``, ``deployment/`` and
# ``health/render.py`` is built here, which is the reach
# ``tests/cli/test_output_structure_guard.py`` enforces. A call site picks the
# *kind* of surface it wants and passes its own columns and content, never its
# own borders, so retheming reaches all of them at once and a look that needs to
# change changes here. Outside that reach the tables under
# ``services/channel_finder/tools/`` still spell their own box and padding.
#
# The defaults are the dominant shape of what the CLI already prints — a
# bordered table with a themed header row, and a rounded panel — so adopting a
# factory is a no-op for most call sites.
# ---------------------------------------------------------------------------

#: Box drawing for a data table: Rich's default heavy-ruled header, which a
#: table that names no box of its own already inherits.
DATA_TABLE_BOX: Box = box.HEAVY_HEAD

#: Cell padding for a data table, as ``(top, right, bottom, left)`` collapsed to
#: the vertical/horizontal pair Rich accepts.
DATA_TABLE_PADDING: tuple[int, int] = (0, 1)

#: Style of the header row. ``header`` and ``bold_primary`` resolve to the same
#: rendered style, so this is the single spelling of what call sites variously
#: reached for.
TABLE_HEADER_STYLE: str = Styles.HEADER

#: Style of every rule and frame the factories draw -- table rules and panel
#: borders alike, so the two never disagree about how a border looks.
BORDER_STYLE: str = Styles.BORDER_DIM

#: Blank columns between two adjacent columns of a layout table. A layout table
#: has no rules, so the gutter is the only thing separating its columns.
LAYOUT_TABLE_GUTTER: int = 3

#: Cell padding for a layout table: the gutter on the left of every cell and
#: nothing anywhere else.
LAYOUT_TABLE_PADDING: tuple[int, int, int, int] = (0, 0, 0, LAYOUT_TABLE_GUTTER)

#: Box drawing for a panel: Rich's rounded frame, which the CLI's panels already
#: inherit.
PANEL_BOX: Box = box.ROUNDED

#: Padding between a panel's frame and its contents.
PANEL_PADDING: tuple[int, int] = (0, 1)


def data_table(**overrides: Any) -> Table:
    """Build the CLI's data table: bordered, with a themed header row.

    This is the one look for a table that *presents data* — a list of services,
    a set of audit findings, a status matrix. Callers add columns and rows; they
    do not choose borders.

    Args:
        **overrides: Any :class:`~rich.table.Table` keyword, passed straight
            through and winning over the defaults. Use it for the table's
            content-shaped options (``title``, ``show_lines``, ``expand``), not
            to reopen the look.

    Returns:
        An empty :class:`~rich.table.Table` ready for ``add_column``/``add_row``.

    Examples:
        >>> table = data_table(title="Services")
        >>> table.add_column("Name", style=Styles.ACCENT)
    """
    params: dict[str, Any] = {
        "box": DATA_TABLE_BOX,
        "show_header": True,
        "header_style": TABLE_HEADER_STYLE,
        "border_style": BORDER_STYLE,
        "padding": DATA_TABLE_PADDING,
    }
    params.update(overrides)
    return Table(**params)


def layout_table(**overrides: Any) -> Table:
    """Build a layout table: no box, no header, columns separated by a gutter.

    Deliberately not a data table. A layout table draws nothing of its own — it
    exists to align columns of text that would otherwise be spaced by hand, so
    the reader sees aligned text rather than a table. The live build region is
    the standing example.

    Args:
        **overrides: Any :class:`~rich.table.Table` keyword, passed straight
            through and winning over the defaults.

    Returns:
        An empty :class:`~rich.table.Table` ready for ``add_column``/``add_row``.
    """
    params: dict[str, Any] = {
        "box": None,
        "show_header": False,
        "padding": LAYOUT_TABLE_PADDING,
        "pad_edge": False,
    }
    params.update(overrides)
    return Table(**params)


def panel(renderable: RenderableType, **overrides: Any) -> Panel:
    """Build the CLI's panel: a rounded, themed frame around one renderable.

    A panel sets its contents apart from the surrounding output — a summary, a
    heading block. It is the one panel look; call sites choose the ``title`` and
    the contents, not the frame.

    Args:
        renderable: What goes inside the frame — a string (Rich markup is
            honoured), a :class:`~rich.text.Text`, or any renderable.
        **overrides: Any :class:`~rich.panel.Panel` keyword, passed straight
            through and winning over the defaults (``title``, ``expand``, …).

    Returns:
        The :class:`~rich.panel.Panel`, ready to print.

    Examples:
        >>> console.print(panel("All checks passed", title="Summary"))
    """
    params: dict[str, Any] = {
        "box": PANEL_BOX,
        "border_style": BORDER_STYLE,
        "padding": PANEL_PADDING,
    }
    params.update(overrides)
    return Panel(renderable, **params)


__all__ = [
    # Theme management
    "ColorTheme",
    "OSPREY_THEME",
    "VULCAN_THEME",
    "THEME_REGISTRY",
    "get_active_theme",
    "set_theme",
    "load_theme_from_config",
    "initialize_theme_from_config",
    "osprey_theme",
    # Console and styles
    "console",
    "err_console",
    "get_questionary_style",
    "custom_style",
    # Style helpers
    "Styles",
    "Messages",
    "ThemeConfig",
    # Table and panel factories
    "data_table",
    "layout_table",
    "panel",
    "DATA_TABLE_BOX",
    "DATA_TABLE_PADDING",
    "TABLE_HEADER_STYLE",
    "BORDER_STYLE",
    "LAYOUT_TABLE_GUTTER",
    "LAYOUT_TABLE_PADDING",
    "PANEL_BOX",
    "PANEL_PADDING",
]
