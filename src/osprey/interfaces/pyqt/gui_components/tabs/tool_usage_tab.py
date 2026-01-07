"""Tool Usage Tab for Osprey GUI."""

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class ToolUsageTab(QWidget):
    """Tab for displaying LLM tool usage and reasoning."""

    def __init__(self, parent=None):
        """Initialize the Tool Usage tab.

        Args:
            parent: Parent OspreyGUI instance
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup the tool usage tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        label = QLabel("LLM Tool Usage and Reasoning:")
        label.setStyleSheet("color: #FF69B4; font-weight: bold;")
        layout.addWidget(label)

        self.tool_usage_display = QTextEdit()
        self.tool_usage_display.setReadOnly(True)
        self.tool_usage_display.setFont(QFont("Monospace", 9))
        self.tool_usage_display.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
        layout.addWidget(self.tool_usage_display)

        clear_btn = QPushButton("Clear Tool Usage")
        clear_btn.clicked.connect(lambda: self.tool_usage_display.clear())
        layout.addWidget(clear_btn)
