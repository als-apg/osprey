"""Centralized color and style management for Osprey CLI.

This module provides a unified color scheme and styling utilities for all CLI
components, ensuring consistent visual appearance across the framework.

Design Philosophy:
- Semantic color names (success, error, warning) rather than direct colors
- Theme-based approach allowing easy theme switching
- Rich console markup helpers for inline styling
- Questionary style integration for interactive prompts
- Clear separation between fixed UI standards and customizable theme colors
"""

import sys
from dataclasses import dataclass

from rich.console import Console
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
    primary: str = "#C75F71"  # Main brand color (medium purple)
    success: str = "#9988A1"  # Success indicator
    accent: str = "#F0B8B8"  # Interactive elements & highlights (teal)
    command: str = "#9988A1"  # Shell commands & actions (orange)
    path: str = "#A2AE9D"  # File paths & locations (gray)
    info: str = "#9988A1"  # Informational messages (cyan)

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


# Default Osprey theme (purple-teal)
OSPREY_THEME = ColorTheme()

# Vulcan theme (default Osprey theme - branded name)
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
    """Set a new active theme and rebuild console/styles.

    Args:
        theme: The ColorTheme to activate
    """
    global _active_theme, console, custom_style
    _active_theme = theme
    # Rebuild theme-dependent objects
    # On Windows, force UTF-8 encoding to support Unicode characters
    if sys.platform == "win32":
        console = Console(theme=_build_rich_theme(theme), force_terminal=True, legacy_windows=False)
    else:
        console = Console(theme=_build_rich_theme(theme))
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


def _build_questionary_style(theme: ColorTheme) -> QuestionaryStyle | None:
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


# Singleton console instance with theme
# On Windows, force UTF-8 encoding to support Unicode characters (✓, ✗, ⚠️, etc.)
if sys.platform == "win32":
    console = Console(theme=osprey_theme, force_terminal=True, legacy_windows=False)
else:
    console = Console(theme=osprey_theme)


def get_questionary_style() -> QuestionaryStyle | None:
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
        return f"[warning]⚠️  {text}[/warning]"

    @staticmethod
    def info(text: str) -> str:
        """Format an info message with info symbol."""
        return f"[info]ℹ️  {text}[/info]"

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

    Provides helper methods for accessing theme settings consistently.
    Future enhancement: Load custom themes from config file.
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
    "get_questionary_style",
    "custom_style",
    # Style helpers
    "Styles",
    "Messages",
    "ThemeConfig",
]
