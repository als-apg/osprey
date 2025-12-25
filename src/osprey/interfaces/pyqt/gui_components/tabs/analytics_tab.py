"""Analytics Dashboard Tab for Osprey GUI."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


class AnalyticsTab(QWidget):
    """Tab for displaying analytics dashboard."""

    def __init__(self, parent=None):
        """Initialize the Analytics tab.

        Args:
            parent: Parent OspreyGUI instance
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.analytics_dashboard = None
        self.placeholder_label = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the analytics tab UI with placeholder."""
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Show placeholder initially (router not initialized yet)
        self.placeholder_label = QLabel(
            "Analytics Dashboard\n\n"
            "Initializing router...\n"
            "Analytics will appear once the system is ready."
        )
        self.placeholder_label.setStyleSheet("color: #00FFFF; font-size: 14px;")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.placeholder_label)

    def initialize_analytics(self):
        """Initialize analytics dashboard after router is ready.

        This should be called after the router is initialized in the main GUI.
        """
        # Check if router is initialized and analytics is enabled
        analytics = self.parent_gui.router.get_analytics() if self.parent_gui.router else None

        # Remove placeholder if it exists
        if self.placeholder_label:
            self.layout.removeWidget(self.placeholder_label)
            self.placeholder_label.deleteLater()
            self.placeholder_label = None

        if analytics:
            from osprey.interfaces.pyqt.analytics_dashboard import AnalyticsDashboard
            self.analytics_dashboard = AnalyticsDashboard(analytics, self.parent_gui)
            self.layout.addWidget(self.analytics_dashboard)
        else:
            # Analytics disabled message
            label = QLabel("Analytics is currently disabled.\n\nEnable analytics in router configuration to view metrics.")
            label.setStyleSheet("color: #FFA500; font-size: 14px;")
            label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(label)
