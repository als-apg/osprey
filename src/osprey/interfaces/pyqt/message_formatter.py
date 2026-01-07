"""Message Formatter for Osprey Framework GUI.

This module provides utilities for formatting and displaying messages in the GUI,
including Rich markup parsing, color formatting, and message type extraction.
"""

import re

from PyQt5.QtGui import QBrush, QColor, QFont, QTextCharFormat, QTextCursor


class MessageFormatter:
    """Handles message formatting and Rich markup parsing for GUI display."""

    # Rich color mapping to Qt colors
    RICH_COLOR_MAP = {
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

    @staticmethod
    def extract_message_type(message: str) -> str:
        """
        Extract the message type (log level) from a message.

        Args:
            message: The message text

        Returns:
            str: Message type (INFO, WARNING, ERROR, DEBUG, etc.)
        """
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

    @staticmethod
    def strip_rich_markup(text: str) -> str:
        """
        Strip Rich markup tags from text.

        Args:
            text: Text with Rich markup

        Returns:
            str: Text without markup tags
        """
        # Remove Rich markup tags like [white], [bold green], etc.
        return re.sub(r"\[([^\]]+)\]", "", text)

    @staticmethod
    def insert_rich_text(cursor: QTextCursor, text: str):
        """
        Parse Rich markup and insert colored text.

        Supports Rich color tags like [white], [bold green], [sky_blue2], etc.
        Preserves timestamps in format [MM/DD/YYYY HH:MM:SS AM/PM]

        Args:
            cursor: QTextCursor to insert text at
            text: Text with Rich markup tags
        """
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
                elif part in MessageFormatter.RICH_COLOR_MAP:
                    if MessageFormatter.RICH_COLOR_MAP[part] is not None:
                        color = MessageFormatter.RICH_COLOR_MAP[part]

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

    @staticmethod
    def append_formatted_text(
        text: str, color: str, widget, prefix: str = "\n", suffix: str = "\n"
    ):
        """
        Unified method to append formatted text to a text widget.

        Args:
            text: Text to append
            color: Color hex code (e.g., "#FFFFFF")
            widget: QTextEdit widget to append to
            prefix: Text to prepend (default: newline)
            suffix: Text to append (default: newline)
        """
        cursor = widget.textCursor()
        cursor.movePosition(QTextCursor.End)

        text_format = QTextCharFormat()
        text_format.setForeground(QBrush(QColor(color)))

        cursor.insertText(f"{prefix}{text}{suffix}", text_format)

        widget.setTextCursor(cursor)
        widget.ensureCursorVisible()
