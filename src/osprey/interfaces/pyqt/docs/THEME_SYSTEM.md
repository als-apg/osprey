# GUI Theme System

## Overview

The Osprey Framework GUI uses a centralized theme system that ensures consistent appearance across Windows, Linux, and macOS platforms. The theme system is designed to be easily extensible, allowing new themes to be added in the future.

## Architecture

The theme system consists of three main components:

1. **Theme Base Class** (`Theme`): Abstract base class that defines the interface for all themes
2. **Concrete Theme Classes** (e.g., `DarkTheme`): Implement specific color schemes
3. **Theme Manager** (`ThemeManager`): Manages available themes and applies them to the application

## Current Themes

### Dark Theme (Default)
- **Name**: "dark"
- **Description**: Professional dark color scheme optimized for reduced eye strain
- **Colors**:
  - Background: `#2D2D30` (dark gray)
  - Text: `#FFFFFF` (white)
  - Accent: `#0078D4` (blue)
  - Borders: `#3F3F46` (medium gray)

## Adding a New Theme

To add a new theme to the system:

### 1. Create a Theme Class

Create a new class in `osprey/src/osprey/interfaces/pyqt/themes.py`:

```python
class LightTheme(Theme):
    """Light theme for the Osprey Framework GUI."""

    def __init__(self):
        super().__init__("Light")

    def get_palette(self) -> QPalette:
        """Create a light color palette."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        # ... configure other palette colors
        return palette

    def get_stylesheet(self) -> str:
        """Get light theme stylesheet."""
        return """
            QWidget {
                background-color: #F0F0F0;
                color: #000000;
            }
            /* ... other widget styles */
        """
```

### 2. Register the Theme

Add your theme to the `ThemeManager.__init__()` method:

```python
def __init__(self):
    self._themes = {
        "dark": DarkTheme(),
        "light": LightTheme(),  # Add your new theme here
    }
    self._current_theme = "dark"
```

### 3. Apply the Theme

The theme can be applied in two ways:

**At Application Startup** (in `gui.py`):
```python
theme_manager = get_theme_manager()
theme_manager.apply_theme(app, "light")  # Apply specific theme
```

**Dynamically** (for theme switching):
```python
from osprey.interfaces.pyqt.themes import get_theme_manager

theme_manager = get_theme_manager()
theme_manager.set_current_theme("light")
theme_manager.apply_theme(app)
```

## Theme Components

### QPalette
The palette defines system-level colors for Qt widgets. Key palette roles:

- `Window`: Main window background
- `WindowText`: Text on windows
- `Base`: Background for text entry widgets
- `Text`: Text in entry widgets
- `Button`: Button background
- `ButtonText`: Button text
- `Highlight`: Selection background
- `HighlightedText`: Selected text

### Stylesheet
The stylesheet provides fine-grained control over widget appearance using CSS-like syntax. It's essential for:

- **Windows Compatibility**: Windows doesn't always respect QPalette settings
- **Consistent Appearance**: Ensures all widgets look the same across platforms
- **Custom Styling**: Allows detailed customization beyond palette colors

## Platform Considerations

### Windows
- **Issue**: Qt's default styling may not apply palette colors to all widgets
- **Solution**: Comprehensive stylesheet ensures all widgets are styled correctly
- **Testing**: Always test themes on Windows to verify appearance

### Linux
- **Issue**: Different desktop environments (GNOME, KDE, etc.) may have different defaults
- **Solution**: Stylesheet overrides system defaults for consistency
- **Testing**: Test on multiple desktop environments if possible

### macOS
- **Issue**: macOS has its own native styling that may conflict
- **Solution**: Stylesheet provides consistent cross-platform appearance
- **Testing**: Verify that native macOS elements (menus, dialogs) are styled correctly

## Best Practices

1. **Test on All Platforms**: Always test new themes on Windows, Linux, and macOS
2. **Use Hex Colors**: Use hex color codes (e.g., `#2D2D30`) for consistency
3. **Maintain Contrast**: Ensure sufficient contrast between text and background (WCAG AA: 4.5:1 minimum)
4. **Document Colors**: Add comments explaining color choices and their purpose
5. **Consistent Naming**: Use descriptive names for theme classes (e.g., `HighContrastTheme`)

## Future Enhancements

Potential improvements to the theme system:

1. **User Theme Selection**: Add UI for users to switch themes at runtime
2. **Theme Persistence**: Save user's theme preference to configuration file
3. **Custom Themes**: Allow users to create and load custom theme files
4. **Theme Preview**: Show theme preview before applying
5. **Accessibility Themes**: Add high-contrast themes for accessibility

## Troubleshooting

### Theme Not Applied on Windows
- Verify stylesheet is being set on QApplication, not just QMainWindow
- Check that all widget types are included in stylesheet
- Ensure `app.setStyleSheet()` is called before creating windows

### Colors Look Different on Different Platforms
- Use explicit hex colors in stylesheet instead of relying on palette
- Test on actual hardware, not just virtual machines
- Consider platform-specific stylesheet sections if needed

### Performance Issues
- Keep stylesheets concise - avoid overly complex selectors
- Use palette for simple color changes, stylesheet for complex styling
- Profile application startup time if themes cause delays

## References

- [Qt Style Sheets Reference](https://doc.qt.io/qt-5/stylesheet-reference.html)
- [QPalette Documentation](https://doc.qt.io/qt-5/qpalette.html)
- [Qt Style Sheets Examples](https://doc.qt.io/qt-5/stylesheet-examples.html)