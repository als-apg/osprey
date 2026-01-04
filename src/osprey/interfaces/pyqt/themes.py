"""Theme management for Osprey Framework GUI.

This module provides a centralized theme system that allows for easy switching
between different color schemes. Currently supports dark theme with the ability
to add more themes in the future.

The theme system is designed to work consistently across Windows, Linux, and macOS.
"""

from PyQt5.QtGui import QColor, QPalette


class Theme:
    """Base class for GUI themes."""

    def __init__(self, name: str):
        self.name = name

    def get_palette(self) -> QPalette:
        """Get the QPalette for this theme."""
        raise NotImplementedError

    def get_stylesheet(self) -> str:
        """Get the global stylesheet for this theme."""
        raise NotImplementedError


class DarkTheme(Theme):
    """Dark theme for the Osprey Framework GUI.
    
    This theme provides a dark color scheme optimized for reduced eye strain
    and professional appearance. It's designed to work consistently across
    Windows, Linux, and macOS platforms.
    """

    def __init__(self):
        super().__init__("Dark")

    def get_palette(self) -> QPalette:
        """Create a dark color palette for the GUI.

        Returns:
            QPalette configured with dark theme colors
        """
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 48))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(60, 60, 60))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        return palette

    def get_stylesheet(self) -> str:
        """Get comprehensive dark theme stylesheet.
        
        This stylesheet ensures consistent dark theme appearance across all
        widgets, dialogs, and windows on Windows, Linux, and macOS.
        
        Returns:
            Complete CSS stylesheet for dark theme
        """
        return """
            /* Base widget styling */
            QWidget {
                background-color: #2D2D30;
                color: #FFFFFF;
            }
            
            /* Main window */
            QMainWindow {
                background-color: #2D2D30;
            }
            
            /* Dialogs */
            QDialog {
                background-color: #2D2D30;
                color: #FFFFFF;
            }
            
            /* Menu bar */
            QMenuBar {
                background-color: #2D2D30;
                color: #FFFFFF;
                border-bottom: 1px solid #3F3F46;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #0078D4;
            }
            
            /* Menus */
            QMenu {
                background-color: #2D2D30;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
            }
            QMenu::item {
                padding: 4px 20px;
            }
            QMenu::item:selected {
                background-color: #0078D4;
            }
            
            /* Tab widgets */
            QTabWidget::pane {
                border: 1px solid #3F3F46;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background-color: #2D2D30;
                color: #FFFFFF;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #3F3F46;
            }
            QTabBar::tab:selected {
                background-color: #0078D4;
                color: #FFFFFF;
            }
            QTabBar::tab:hover {
                background-color: #3F3F46;
            }
            
            /* Buttons */
            QPushButton {
                background-color: #4A5568;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5A6578;
            }
            QPushButton:pressed {
                background-color: #3A4558;
            }
            QPushButton:disabled {
                background-color: #2D2D30;
                color: #808080;
            }
            
            /* Status bar */
            QStatusBar {
                background-color: #2D2D30;
                color: #FFFFFF;
                border-top: 1px solid #3F3F46;
            }
            
            /* Group boxes */
            QGroupBox {
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #FFFFFF;
            }
            
            /* Combo boxes */
            QComboBox {
                background-color: #2D2D30;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #FFFFFF;
            }
            QComboBox QAbstractItemView {
                background-color: #2D2D30;
                color: #FFFFFF;
                selection-background-color: #0078D4;
                selection-color: #FFFFFF;
                border: 1px solid #3F3F46;
            }
            
            /* Spin boxes */
            QSpinBox, QDoubleSpinBox {
                background-color: #2D2D30;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                padding: 3px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                background-color: #3F3F46;
                border: none;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #3F3F46;
                border: none;
            }
            
            /* Check boxes */
            QCheckBox {
                color: #FFFFFF;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #3F3F46;
                border-radius: 3px;
                background-color: #2D2D30;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #0078D4;
                background-color: #3F3F46;
            }
            QCheckBox::indicator:checked {
                background-color: #0078D4;
                border: 1px solid #0078D4;
                image: none;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #1084E4;
            }
            QCheckBox::indicator:disabled {
                background-color: #1E1E1E;
                border: 1px solid #2D2D30;
            }
            
            /* Labels */
            QLabel {
                color: #FFFFFF;
                background-color: transparent;
            }
            
            /* Scroll bars */
            QScrollBar:vertical {
                background-color: #2D2D30;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #4A5568;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #5A6578;
            }
            QScrollBar:horizontal {
                background-color: #2D2D30;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background-color: #4A5568;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #5A6578;
            }
        """


class ThemeManager:
    """Manages GUI themes for the application."""

    def __init__(self):
        self._themes = {
            "dark": DarkTheme(),
            # Future themes can be added here:
            # "light": LightTheme(),
            # "high_contrast": HighContrastTheme(),
        }
        self._current_theme = "dark"

    def get_theme(self, theme_name: str = None) -> Theme:
        """Get a theme by name.

        Args:
            theme_name: Name of the theme to get. If None, returns current theme.

        Returns:
            Theme object

        Raises:
            KeyError: If theme_name doesn't exist
        """
        if theme_name is None:
            theme_name = self._current_theme
        return self._themes[theme_name]

    def set_current_theme(self, theme_name: str):
        """Set the current theme.

        Args:
            theme_name: Name of the theme to set as current

        Raises:
            KeyError: If theme_name doesn't exist
        """
        if theme_name not in self._themes:
            raise KeyError(f"Theme '{theme_name}' not found")
        self._current_theme = theme_name

    def get_available_themes(self) -> list:
        """Get list of available theme names.

        Returns:
            List of theme names
        """
        return list(self._themes.keys())

    def apply_theme(self, app, theme_name: str = None):
        """Apply a theme to the QApplication.

        Args:
            app: QApplication instance
            theme_name: Name of theme to apply. If None, applies current theme.
        """
        theme = self.get_theme(theme_name)
        app.setPalette(theme.get_palette())
        app.setStyleSheet(theme.get_stylesheet())


# Global theme manager instance
_theme_manager = None


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance.

    Returns:
        ThemeManager singleton instance
    """
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager