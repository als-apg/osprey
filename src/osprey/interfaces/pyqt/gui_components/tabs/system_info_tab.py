"""System Information Tab for Osprey GUI."""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QFont, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from osprey.interfaces.pyqt.collapsible_widget import MessageGroupWidget


class SystemInfoTab(QWidget):
    """Tab for displaying system information and logs."""

    def __init__(self, parent=None):
        """Initialize the System Information tab.

        Args:
            parent: Parent OspreyGUI instance
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup the system information tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Header with label and toggle button
        header_layout = QHBoxLayout()
        label = QLabel("System Information:")
        label.setStyleSheet("color: #1E90FF; font-weight: bold;")
        header_layout.addWidget(label)

        header_layout.addStretch()

        # Toggle button to switch between grouped and ungrouped view
        self.group_messages_toggle = QPushButton("📋 Grouped View")
        self.group_messages_toggle.setCheckable(True)
        self.group_messages_toggle.setChecked(
            self.parent_gui.settings_manager.get("group_system_messages", True)
        )
        self.group_messages_toggle.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")
        self.group_messages_toggle.clicked.connect(self.toggle_message_grouping)
        header_layout.addWidget(self.group_messages_toggle)

        layout.addLayout(header_layout)

        # Create scroll area for message groups
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #1E1E1E; border: 1px solid #3F3F46;")

        # Create message group widget
        self.message_group_widget = MessageGroupWidget()
        scroll_area.setWidget(self.message_group_widget)

        # Create traditional text edit (hidden by default if grouping is enabled)
        self.session_info = QTextEdit()
        self.session_info.setReadOnly(True)
        self.session_info.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
        self.session_info.setAcceptRichText(False)  # Use plain text for better performance
        self.session_info.setFont(QFont("Monospace", 9))
        self.session_info.setPlainText(
            "Initializing Osprey Framework...\n\nPlease wait while the system initializes."
        )

        # Add both widgets to layout (we'll show/hide based on setting)
        layout.addWidget(scroll_area)
        layout.addWidget(self.session_info)

        # Show/hide based on initial setting
        if self.parent_gui.settings_manager.get("group_system_messages", True):
            scroll_area.setVisible(True)
            self.session_info.setVisible(False)
            self.message_group_widget.add_message("Initializing Osprey Framework...", "INFO")
            self.message_group_widget.add_message(
                "Please wait while the system initializes.", "INFO"
            )
        else:
            scroll_area.setVisible(False)
            self.session_info.setVisible(True)

        # Store reference to scroll area for toggling
        self.message_group_scroll = scroll_area

        # Add clear button
        clear_btn = QPushButton("Clear System Info")
        clear_btn.clicked.connect(self.clear_system_info)
        layout.addWidget(clear_btn)

    def toggle_message_grouping(self):
        """Toggle between grouped and ungrouped message view."""
        is_grouped = self.group_messages_toggle.isChecked()
        # Update via update_from_dict to ensure proper handling
        self.parent_gui.settings_manager.update_from_dict({"group_system_messages": is_grouped})
        self.parent_gui.settings = (
            self.parent_gui.settings_manager.get_all_settings()
        )  # Update backward compat dict

        # Update button text
        if is_grouped:
            self.group_messages_toggle.setText("📋 Grouped View")
        else:
            self.group_messages_toggle.setText("📄 List View")

        # Show/hide appropriate widget
        self.message_group_scroll.setVisible(is_grouped)
        self.session_info.setVisible(not is_grouped)

    @pyqtSlot(str)
    def append_to_system_info(self, message: str):
        """
        Append a message to the System Information tab with Rich markup parsing.

        This method is called by the logging system to redirect output to the GUI.
        Thread-safe for use from background threads via Qt's signal/slot mechanism.
        Parses Rich markup tags (e.g., [white], [bold green]) and applies colors.

        Args:
            message: The message to append (may contain Rich markup)
        """
        # Skip empty messages
        if not message or not message.strip():
            return

        # Determine message type from the message content
        message_type = self._extract_message_type(message)

        # If grouping is enabled, add to grouped widget WITH color formatting
        if self.parent_gui.settings_manager.get("group_system_messages", True):
            # Pass the original message with Rich markup for color formatting
            self.message_group_widget.add_message(message, message_type, rich_markup=message)

        # Always add to traditional text edit (for when user switches view)
        cursor = self.session_info.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Parse and apply Rich markup
        self._insert_rich_text(cursor, message)

        # Add newline
        cursor.insertText("\n")
        self.session_info.setTextCursor(cursor)
        self.session_info.ensureCursorVisible()

    def _extract_message_type(self, message: str) -> str:
        """
        Extract the message type (log level) from a message.

        Args:
            message: The message text

        Returns:
            str: Message type (INFO, WARNING, ERROR, DEBUG, etc.)
        """
        import re

        # Extract log level from the formatted message
        # Format: [MM/DD/YYYY HH:MM:SS AM/PM] LEVEL     component: message
        # We need to match the LEVEL part specifically, not just any occurrence

        # First, try to extract from the standard logging format
        # Match pattern: [timestamp] LEVEL (with padding) component:
        level_match = re.search(r"\]\s+(ERROR|CRITICAL|WARNING|WARN|INFO|DEBUG)\s+", message)

        if level_match:
            level = level_match.group(1).upper()
            if level in ("ERROR", "CRITICAL"):
                return "ERROR"
            elif level in ("WARNING", "WARN"):
                return "WARNING"
            elif level == "DEBUG":
                return "DEBUG"
            elif level == "INFO":
                return "INFO"

        # Fallback: Check for error/warning indicators in the message content
        # Only check AFTER the component name to avoid false positives
        message.upper()

        # Check for error emoji (strong indicator)
        if "❌" in message:
            return "ERROR"

        # Check for warning emoji (strong indicator)
        if "⚠️" in message:
            return "WARNING"

        # Check for success emoji (treat as INFO)
        if "✅" in message:
            return "INFO"

        # Default to INFO for any unrecognized format
        return "INFO"

    def _insert_rich_text(self, cursor, text: str):
        """
        Parse Rich markup and insert colored text.

        Supports Rich color tags like [white], [bold green], [sky_blue2], etc.
        Preserves timestamps in format [MM/DD/YYYY HH:MM:SS AM/PM]

        Args:
            cursor: QTextCursor to insert text at
            text: Text with Rich markup tags
        """
        import re

        # Rich color mapping to Qt colors
        rich_color_map = {
            "white": "#FFFFFF",
            "black": "#000000",
            "red": "#FF0000",
            "green": "#00FF00",
            "yellow": "#FFFF00",
            "blue": "#0000FF",
            "magenta": "#FF00FF",
            "cyan": "#00FFFF",
            "bright_white": "#FFFFFF",
            "bright_black": "#808080",
            "bright_red": "#FF6B6B",
            "bright_green": "#00FF00",
            "bright_yellow": "#FFFF00",
            "bright_blue": "#6B9EFF",
            "bright_magenta": "#FF6BFF",
            "bright_cyan": "#00FFFF",
            "sky_blue2": "#87CEEB",
            "bold": None,  # Style modifier, not a color
        }

        # Pattern to match Rich markup: [color] or [style color]
        # But NOT timestamps like [12/03/2025 09:28:08 AM]
        # Timestamps contain digits, slashes, colons, and spaces
        pattern = r"\[([^\]]+)\](.*?)(?=\[|$)"

        pos = 0
        for match in re.finditer(pattern, text):
            style_spec = match.group(1)

            # Check if this looks like a timestamp (contains digits and slashes/colons)
            # Timestamp pattern: contains digits, slashes, colons, spaces, AM/PM
            is_timestamp = bool(re.search(r"\d+[/:]", style_spec))

            if is_timestamp:
                # This is a timestamp, not a Rich markup tag - keep it as-is
                if match.start() > pos:
                    cursor.insertText(text[pos : match.start()])
                # Insert the timestamp with brackets
                cursor.insertText(f"[{style_spec}]")
                pos = match.start() + len(f"[{style_spec}]")
                continue

            # Insert any text before the match
            if match.start() > pos:
                cursor.insertText(text[pos : match.start()])

            content = match.group(2)

            # Parse style specification (e.g., "bold green", "white", "sky_blue2")
            parts = style_spec.split()
            color = "#FFFFFF"  # Default white
            bold = False

            for part in parts:
                if part == "bold":
                    bold = True
                elif part in rich_color_map:
                    if rich_color_map[part] is not None:
                        color = rich_color_map[part]

            # Create text format with color and style
            text_format = QTextCharFormat()
            text_format.setForeground(QBrush(QColor(color)))
            if bold:
                text_format.setFontWeight(QFont.Bold)

            # Insert formatted text
            cursor.insertText(content, text_format)

            pos = match.end()

        # Insert any remaining text
        if pos < len(text):
            cursor.insertText(text[pos:])

    def clear_system_info(self):
        """Clear the System Information tab."""
        self.session_info.clear()
        self.message_group_widget.clear()
        if hasattr(self.parent_gui, "update_session_info"):
            self.parent_gui.update_session_info()
