#!/usr/bin/env python3
"""
Collapsible widget for grouping similar messages in the GUI.

This widget allows messages to be grouped and collapsed/expanded by the user,
improving readability when there are many similar messages.
"""

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QPropertyAnimation, QParallelAnimationGroup
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QToolButton, QFrame, QTextEdit
)
from PyQt5.QtGui import QFont, QTextCursor, QTextCharFormat, QBrush, QColor


class CollapsibleBox(QWidget):
    """
    A collapsible box widget that can contain other widgets.

    Features:
    - Animated expand/collapse
    - Customizable title
    - Can contain any Qt widget as content
    - Visual indicator (arrow) showing collapsed/expanded state
    """

    def __init__(self, title="", parent=None, start_collapsed=True):
        """
        Initialize the collapsible box.

        Args:
            title: Title text to display in the header
            parent: Parent widget
            start_collapsed: Whether to start in collapsed state (default: True)
        """
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(
            text=title, checkable=True, checked=not start_collapsed
        )
        self.toggle_button.setStyleSheet("""
            QToolButton {
                border: none;
                background-color: #2D2D30;
                color: #FFFFFF;
                padding: 5px;
                text-align: left;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: #3F3F46;
            }
        """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if not start_collapsed else Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QParallelAnimationGroup(self)

        self.content_area = QScrollArea(
            maximumHeight=0 if start_collapsed else 16777215,
            minimumHeight=0
        )
        self.content_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setStyleSheet("""
            QScrollArea {
                background-color: #1E1E1E;
                border: 1px solid #3F3F46;
            }
        """)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QPropertyAnimation(self.content_area, b"maximumHeight")
        )

        # Set initial state
        if not start_collapsed:
            # If starting expanded, we need to set the content layout first
            # before the animation can work properly
            pass

    @QtCore.pyqtSlot()
    def on_pressed(self):
        """Handle toggle button press to expand/collapse the content."""
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            Qt.DownArrow if not checked else Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QParallelAnimationGroup.Forward
            if not checked
            else QParallelAnimationGroup.Backward
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        """
        Set the content layout for the collapsible area.

        Args:
            layout: QLayout to use for the content area
        """
        # Safely remove existing layout if present
        existing_layout = self.content_area.layout()
        if existing_layout is not None:
            QtWidgets.QWidget().setLayout(existing_layout)

        self.content_area.setLayout(layout)

        # Get the toggle button height to use as collapsed height
        toggle_button_height = self.toggle_button.sizeHint().height()

        # Get content height from layout
        content_height = layout.sizeHint().height()

        # Ensure content_height is reasonable (not too large or zero)
        if content_height <= 0:
            content_height = 100  # Default minimum height
        elif content_height > 500:
            content_height = 500  # Cap maximum height for performance

        # Calculate collapsed height (just the toggle button)
        collapsed_height = toggle_button_height

        # Ensure collapsed_height is never negative
        if collapsed_height < 0:
            collapsed_height = 30  # Fallback to reasonable default

        # Configure animations with safe values
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(300)  # Faster animation (300ms instead of 500ms)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        # Configure content area animation
        if self.toggle_animation.animationCount() > 0:
            content_animation = self.toggle_animation.animationAt(
                self.toggle_animation.animationCount() - 1
            )
            content_animation.setDuration(300)
            content_animation.setStartValue(0)
            content_animation.setEndValue(content_height)

    def setTitle(self, title):
        """
        Update the title text.

        Args:
            title: New title text
        """
        self.toggle_button.setText(title)

    def isExpanded(self):
        """
        Check if the box is currently expanded.

        Returns:
            bool: True if expanded, False if collapsed
        """
        return self.toggle_button.isChecked()

    def setExpanded(self, expanded):
        """
        Programmatically expand or collapse the box.

        Args:
            expanded: True to expand, False to collapse
        """
        if self.isExpanded() != expanded:
            self.toggle_button.setChecked(expanded)
            self.on_pressed()


class MessageGroupWidget(QWidget):
    """
    Widget for displaying grouped messages with collapsible sections.

    Messages are grouped by their log level (INFO, WARNING, ERROR, etc.)
    and can be expanded/collapsed individually.
    """

    def __init__(self, parent=None):
        """
        Initialize the message group widget.

        Args:
            parent: Parent widget
        """
        super(MessageGroupWidget, self).__init__(parent)

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(2)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Dictionary to store collapsible boxes by message type
        self.message_groups = {}

        # Dictionary to store text edits for each group
        self.group_text_edits = {}

        # Add stretch at the end to push groups to the top
        self.layout.addStretch()

    def add_message(self, message: str, message_type: str = "INFO", rich_markup: str = None):
        """
        Add a message to the appropriate group.

        Args:
            message: The message text to add (plain text or with Rich markup)
            message_type: Type of message (INFO, WARNING, ERROR, DEBUG, etc.)
            rich_markup: Optional original message with Rich markup for color formatting
        """
        # Skip empty messages
        if not message or not message.strip():
            return

        # Create group if it doesn't exist
        if message_type not in self.message_groups:
            self._create_group(message_type)

        # Add message to the group's text edit
        text_edit = self.group_text_edits[message_type]

        # Use append() for thread-safe text insertion
        # This is more reliable than manual cursor manipulation
        if rich_markup:
            # For rich markup, we still need cursor manipulation
            cursor = text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._insert_rich_text(cursor, rich_markup)
            cursor.insertText('\n')
            text_edit.setTextCursor(cursor)
        else:
            # For plain text, use append which is thread-safe
            text_edit.append(message)

        # Ensure the latest message is visible
        text_edit.ensureCursorVisible()

        # Update the group title with message count
        self._update_group_title(message_type)

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
            'white': '#FFFFFF',
            'black': '#000000',
            'red': '#FF0000',
            'green': '#00FF00',
            'yellow': '#FFFF00',
            'blue': '#0000FF',
            'magenta': '#FF00FF',
            'cyan': '#00FFFF',
            'bright_white': '#FFFFFF',
            'bright_black': '#808080',
            'bright_red': '#FF6B6B',
            'bright_green': '#00FF00',
            'bright_yellow': '#FFFF00',
            'bright_blue': '#6B9EFF',
            'bright_magenta': '#FF6BFF',
            'bright_cyan': '#00FFFF',
            'sky_blue2': '#87CEEB',
            'bold': None,  # Style modifier, not a color
        }

        # Pattern to match Rich markup: [color] or [style color]
        # But NOT timestamps like [12/03/2025 09:28:08 AM]
        pattern = r'\[([^\]]+)\](.*?)(?=\[|$)'

        pos = 0
        for match in re.finditer(pattern, text):
            style_spec = match.group(1)

            # Check if this looks like a timestamp (contains digits and slashes/colons)
            # Timestamp pattern: contains digits, slashes, colons, spaces, AM/PM
            is_timestamp = bool(re.search(r'\d+[/:]', style_spec))

            if is_timestamp:
                # This is a timestamp, not a Rich markup tag - keep it as-is
                if match.start() > pos:
                    cursor.insertText(text[pos:match.start()])
                # Insert the timestamp with brackets
                cursor.insertText(f"[{style_spec}]")
                pos = match.start() + len(f"[{style_spec}]")
                continue

            # Insert any text before the match
            if match.start() > pos:
                cursor.insertText(text[pos:match.start()])

            content = match.group(2)

            # Parse style specification (e.g., "bold green", "white", "sky_blue2")
            parts = style_spec.split()
            color = '#FFFFFF'  # Default white
            bold = False

            for part in parts:
                if part == 'bold':
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

    def _create_group(self, message_type: str):
        """
        Create a new collapsible group for a message type.

        Args:
            message_type: Type of message (INFO, WARNING, ERROR, etc.)
        """
        # Determine if group should start collapsed based on type
        # ERROR and WARNING start expanded, others start collapsed
        start_collapsed = message_type not in ["ERROR", "WARNING"]

        # Create collapsible box
        box = CollapsibleBox(
            title=f"{message_type} (0 messages)",
            start_collapsed=start_collapsed
        )

        # Create text edit for messages
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Monospace", 9))
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: none;
            }
        """)
        text_edit.setMaximumHeight(200)  # Limit height of each group

        # Create layout for the content
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.addWidget(text_edit)

        # Set the content layout
        box.setContentLayout(content_layout)

        # Insert before the stretch at the end
        self.layout.insertWidget(self.layout.count() - 1, box)

        # Store references
        self.message_groups[message_type] = box
        self.group_text_edits[message_type] = text_edit

    def _update_group_title(self, message_type: str):
        """
        Update the title of a group to show message count.

        Args:
            message_type: Type of message group to update
        """
        if message_type in self.message_groups:
            text_edit = self.group_text_edits[message_type]
            # Count lines in the text edit
            message_count = text_edit.document().lineCount() - 1  # -1 for trailing newline

            # Color code the title based on message type
            color_map = {
                'ERROR': '🔴',
                'WARNING': '🟡',
                'INFO': '🔵',
                'DEBUG': '⚪',
            }
            icon = color_map.get(message_type, '⚪')

            self.message_groups[message_type].setTitle(
                f"{icon} {message_type} ({message_count} messages)"
            )

    def clear(self):
        """Clear all message groups."""
        for text_edit in self.group_text_edits.values():
            text_edit.clear()

        # Update all titles
        for message_type in self.message_groups:
            self._update_group_title(message_type)

    def get_message_count(self, message_type: str = None):
        """
        Get the count of messages.

        Args:
            message_type: If specified, get count for that type only.
                         If None, get total count across all types.

        Returns:
            int: Message count
        """
        if message_type:
            if message_type in self.group_text_edits:
                return self.group_text_edits[message_type].document().lineCount() - 1
            return 0
        else:
            total = 0
            for text_edit in self.group_text_edits.values():
                total += text_edit.document().lineCount() - 1
            return total
