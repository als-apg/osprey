"""LLM Details Tab for Osprey GUI."""

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class LLMDetailsTab(QWidget):
    """Tab for displaying LLM conversation details."""

    def __init__(self, parent=None):
        """Initialize the LLM Details tab.

        Args:
            parent: Parent OspreyGUI instance
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup the LLM details tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        label = QLabel("LLM Conversation Details:")
        label.setStyleSheet("color: #FFD700; font-weight: bold;")
        layout.addWidget(label)

        self.llm_details_display = QTextEdit()
        self.llm_details_display.setReadOnly(True)
        self.llm_details_display.setFont(QFont("Monospace", 9))
        self.llm_details_display.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
        layout.addWidget(self.llm_details_display)

        clear_btn = QPushButton("Clear Details")
        clear_btn.clicked.connect(lambda: self.llm_details_display.clear())
        layout.addWidget(clear_btn)
