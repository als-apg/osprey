"""
Analytics Dashboard Widget for Routing Metrics

This module provides a PyQt widget for displaying routing analytics
in a comprehensive dashboard with visualizations and statistics.

Key Features:
- Project usage visualization (bar charts)
- Performance metrics display
- Query pattern analysis
- Time-series graphs
- Real-time updates
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton, QGroupBox, QScrollArea,
    QHeaderView, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.interfaces.pyqt.routing_analytics import RoutingAnalytics

from osprey.utils.logger import get_logger

logger = get_logger("analytics_dashboard")


class AnalyticsDashboard(QWidget):
    """
    Dashboard widget for displaying routing analytics.

    Displays:
    - Project usage statistics
    - Performance metrics
    - Query patterns
    - Cache performance
    - Time-series data
    """

    def __init__(self, analytics: 'RoutingAnalytics', parent=None):
        """Initialize analytics dashboard.

        Args:
            analytics: RoutingAnalytics instance.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.analytics = analytics
        self.logger = logger

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.auto_refresh_enabled = False

        self.setup_ui()
        self.refresh_data()

    def setup_ui(self):
        """Setup the dashboard UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("📊 Routing Analytics Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #00FFFF;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Time range selector
        time_range_label = QLabel("Time Range:")
        time_range_label.setStyleSheet("color: #FFFFFF;")
        header_layout.addWidget(time_range_label)

        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems([
            "Last Hour",
            "Last 6 Hours",
            "Last 24 Hours",
            "Last 7 Days",
            "All Time"
        ])
        self.time_range_combo.setCurrentIndex(2)  # Default: Last 24 Hours
        self.time_range_combo.currentIndexChanged.connect(self.refresh_data)
        self.time_range_combo.setStyleSheet("""
            QComboBox {
                background-color: #2D2D30;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #2D2D30;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #FFFFFF;
                width: 0;
                height: 0;
            }
            QComboBox QAbstractItemView {
                background-color: #2D2D30;
                color: #FFFFFF;
                selection-background-color: #0078D4;
                selection-color: #FFFFFF;
                border: 1px solid #3F3F46;
            }
        """)
        header_layout.addWidget(self.time_range_combo)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        refresh_btn.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")
        header_layout.addWidget(refresh_btn)

        # Auto-refresh toggle
        self.auto_refresh_btn = QPushButton("⏸️ Auto-Refresh: OFF")
        self.auto_refresh_btn.setCheckable(True)
        self.auto_refresh_btn.clicked.connect(self.toggle_auto_refresh)
        self.auto_refresh_btn.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")
        header_layout.addWidget(self.auto_refresh_btn)

        layout.addLayout(header_layout)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #1E1E1E; border: none;")

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)

        # Overview metrics
        overview_group = self._create_overview_section()
        content_layout.addWidget(overview_group)

        # Project usage section
        project_usage_group = self._create_project_usage_section()
        content_layout.addWidget(project_usage_group)

        # Performance metrics section
        performance_group = self._create_performance_section()
        content_layout.addWidget(performance_group)

        # Query patterns section
        patterns_group = self._create_query_patterns_section()
        content_layout.addWidget(patterns_group)

        content_layout.addStretch()

        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # Export button at bottom
        export_layout = QHBoxLayout()
        export_layout.addStretch()

        export_btn = QPushButton("💾 Export Metrics")
        export_btn.clicked.connect(self.export_metrics)
        export_btn.setStyleSheet("background-color: #0078D4; color: #FFFFFF;")
        export_layout.addWidget(export_btn)

        clear_btn = QPushButton("🗑️ Clear Metrics")
        clear_btn.clicked.connect(self.clear_metrics)
        clear_btn.setStyleSheet("background-color: #D13438; color: #FFFFFF;")
        export_layout.addWidget(clear_btn)

        layout.addLayout(export_layout)

    def _create_overview_section(self) -> QGroupBox:
        """Create overview metrics section."""
        group = QGroupBox("Overview")
        group.setStyleSheet("""
            QGroupBox {
                color: #FFD700;
                font-weight: bold;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QHBoxLayout()
        group.setLayout(layout)

        # Create metric cards
        self.total_queries_label = self._create_metric_card("Total Queries", "0", "#00FFFF")
        self.unique_queries_label = self._create_metric_card("Unique Queries", "0", "#00FF00")
        self.avg_confidence_label = self._create_metric_card("Avg Confidence", "0%", "#FFD700")
        self.cache_hit_rate_label = self._create_metric_card("Cache Hit Rate", "0%", "#FF69B4")

        layout.addWidget(self.total_queries_label)
        layout.addWidget(self.unique_queries_label)
        layout.addWidget(self.avg_confidence_label)
        layout.addWidget(self.cache_hit_rate_label)

        return group

    def _create_metric_card(self, title: str, value: str, color: str) -> QWidget:
        """Create a metric card widget."""
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{
                background-color: #2D2D30;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                padding: 10px;
            }}
        """)

        layout = QVBoxLayout()
        card.setLayout(layout)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #FFFFFF; font-size: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setObjectName(f"{title}_value")
        layout.addWidget(value_label)

        return card

    def _create_project_usage_section(self) -> QGroupBox:
        """Create project usage section."""
        group = QGroupBox("Project Usage")
        group.setStyleSheet("""
            QGroupBox {
                color: #FFD700;
                font-weight: bold;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QVBoxLayout()
        group.setLayout(layout)

        # Table for project statistics
        self.project_table = QTableWidget()
        self.project_table.setColumnCount(6)
        self.project_table.setHorizontalHeaderLabels([
            'Project', 'Queries', 'Percentage', 'Avg Confidence', 'Cache Hit Rate', 'Failures'
        ])
        self.project_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.project_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                gridline-color: #3F3F46;
            }
            QTableWidget::item {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #2D2D30;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3F3F46;
                font-weight: bold;
            }
        """)
        self.project_table.setAlternatingRowColors(False)
        self.project_table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout.addWidget(self.project_table)

        return group

    def _create_performance_section(self) -> QGroupBox:
        """Create performance metrics section."""
        group = QGroupBox("Performance Metrics")
        group.setStyleSheet("""
            QGroupBox {
                color: #FFD700;
                font-weight: bold;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QVBoxLayout()
        group.setLayout(layout)

        # Performance stats
        stats_layout = QHBoxLayout()

        self.avg_routing_time_label = QLabel("Avg Routing Time: 0ms")
        self.avg_routing_time_label.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        stats_layout.addWidget(self.avg_routing_time_label)

        self.failed_routings_label = QLabel("Failed Routings: 0")
        self.failed_routings_label.setStyleSheet("color: #FF6B6B; font-size: 12px;")
        stats_layout.addWidget(self.failed_routings_label)

        self.manual_vs_auto_label = QLabel("Manual: 0 | Automatic: 0")
        self.manual_vs_auto_label.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        stats_layout.addWidget(self.manual_vs_auto_label)

        layout.addLayout(stats_layout)

        return group

    def _create_query_patterns_section(self) -> QGroupBox:
        """Create query patterns section."""
        group = QGroupBox("Top Query Patterns")
        group.setStyleSheet("""
            QGroupBox {
                color: #FFD700;
                font-weight: bold;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QVBoxLayout()
        group.setLayout(layout)

        # Table for query patterns
        self.patterns_table = QTableWidget()
        self.patterns_table.setColumnCount(4)
        self.patterns_table.setHorizontalHeaderLabels([
            'Pattern', 'Count', 'Most Common Project', 'Avg Confidence'
        ])
        self.patterns_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.patterns_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                gridline-color: #3F3F46;
            }
            QTableWidget::item {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #2D2D30;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3F3F46;
                font-weight: bold;
            }
        """)
        self.patterns_table.setAlternatingRowColors(False)
        self.patterns_table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout.addWidget(self.patterns_table)

        return group

    def refresh_data(self):
        """Refresh dashboard data."""
        try:
            # Get time range
            time_range_hours = self._get_time_range_hours()

            # Get summary
            summary = self.analytics.get_summary(time_range_hours)

            # Update overview metrics
            self._update_overview_metrics(summary)

            # Update project usage table
            self._update_project_usage_table(summary)

            # Update performance metrics
            self._update_performance_metrics(summary)

            # Update query patterns
            self._update_query_patterns()

            self.logger.debug("Dashboard data refreshed")

        except Exception as e:
            self.logger.error(f"Failed to refresh dashboard: {e}")

    def _get_time_range_hours(self) -> Optional[float]:
        """Get selected time range in hours."""
        index = self.time_range_combo.currentIndex()

        time_ranges = {
            0: 1.0,      # Last Hour
            1: 6.0,      # Last 6 Hours
            2: 24.0,     # Last 24 Hours
            3: 168.0,    # Last 7 Days
            4: None      # All Time
        }

        return time_ranges.get(index)

    def _update_overview_metrics(self, summary):
        """Update overview metric cards."""
        # Find and update value labels
        for card in self.findChildren(QWidget):
            value_label = card.findChild(QLabel, "Total Queries_value")
            if value_label:
                value_label.setText(str(summary.total_queries))

            value_label = card.findChild(QLabel, "Unique Queries_value")
            if value_label:
                value_label.setText(str(summary.unique_queries))

            value_label = card.findChild(QLabel, "Avg Confidence_value")
            if value_label:
                value_label.setText(f"{summary.avg_confidence * 100:.1f}%")

            value_label = card.findChild(QLabel, "Cache Hit Rate_value")
            if value_label:
                value_label.setText(f"{summary.cache_hit_rate * 100:.1f}%")

    def _update_project_usage_table(self, summary):
        """Update project usage table."""
        self.project_table.setRowCount(0)

        if not summary.project_usage:
            return

        total_queries = summary.total_queries

        # Sort by usage (descending)
        sorted_projects = sorted(
            summary.project_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for row, (project, count) in enumerate(sorted_projects):
            self.project_table.insertRow(row)

            # Get project stats
            stats = self.analytics.get_project_stats(project)

            # Project name
            item = QTableWidgetItem(project)
            item.setForeground(QColor("#00FFFF"))
            self.project_table.setItem(row, 0, item)

            # Query count
            item = QTableWidgetItem(str(count))
            item.setForeground(QColor("#FFFFFF"))
            self.project_table.setItem(row, 1, item)

            # Percentage
            percentage = (count / total_queries * 100) if total_queries > 0 else 0
            item = QTableWidgetItem(f"{percentage:.1f}%")
            item.setForeground(QColor("#00FF00"))
            self.project_table.setItem(row, 2, item)

            # Avg confidence
            item = QTableWidgetItem(f"{stats['avg_confidence'] * 100:.1f}%")
            item.setForeground(QColor("#FFD700"))
            self.project_table.setItem(row, 3, item)

            # Cache hit rate
            item = QTableWidgetItem(f"{stats['cache_hit_rate'] * 100:.1f}%")
            item.setForeground(QColor("#FF69B4"))
            self.project_table.setItem(row, 4, item)

            # Failures
            failures = int(stats['failure_rate'] * count)
            item = QTableWidgetItem(str(failures))
            item.setForeground(QColor("#FF6B6B") if failures > 0 else QColor("#00FF00"))
            self.project_table.setItem(row, 5, item)

    def _update_performance_metrics(self, summary):
        """Update performance metrics."""
        self.avg_routing_time_label.setText(
            f"Avg Routing Time: {summary.avg_routing_time_ms:.0f}ms"
        )

        self.failed_routings_label.setText(
            f"Failed Routings: {summary.failed_routings}"
        )

        manual = summary.manual_vs_automatic.get('manual', 0)
        automatic = summary.manual_vs_automatic.get('automatic', 0)
        self.manual_vs_auto_label.setText(
            f"Manual: {manual} | Automatic: {automatic}"
        )

    def _update_query_patterns(self):
        """Update query patterns table."""
        self.patterns_table.setRowCount(0)

        patterns = self.analytics.get_query_patterns(limit=10)

        for row, (pattern, count, project, confidence) in enumerate(patterns):
            self.patterns_table.insertRow(row)

            # Pattern
            item = QTableWidgetItem(pattern)
            item.setForeground(QColor("#FFFFFF"))
            self.patterns_table.setItem(row, 0, item)

            # Count
            item = QTableWidgetItem(str(count))
            item.setForeground(QColor("#00FF00"))
            self.patterns_table.setItem(row, 1, item)

            # Most common project
            item = QTableWidgetItem(project)
            item.setForeground(QColor("#00FFFF"))
            self.patterns_table.setItem(row, 2, item)

            # Avg confidence
            item = QTableWidgetItem(f"{confidence * 100:.1f}%")
            item.setForeground(QColor("#FFD700"))
            self.patterns_table.setItem(row, 3, item)

    def toggle_auto_refresh(self):
        """Toggle auto-refresh."""
        self.auto_refresh_enabled = self.auto_refresh_btn.isChecked()

        if self.auto_refresh_enabled:
            self.refresh_timer.start(5000)  # Refresh every 5 seconds
            self.auto_refresh_btn.setText("▶️ Auto-Refresh: ON")
            self.auto_refresh_btn.setStyleSheet("background-color: #107C10; color: #FFFFFF;")
        else:
            self.refresh_timer.stop()
            self.auto_refresh_btn.setText("⏸️ Auto-Refresh: OFF")
            self.auto_refresh_btn.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")

    def export_metrics(self):
        """Export metrics to file."""
        from PyQt5.QtWidgets import QFileDialog
        from pathlib import Path

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metrics",
            str(Path.home() / "routing_analytics.json"),
            "JSON Files (*.json)"
        )

        if filepath:
            success = self.analytics.export_metrics(Path(filepath))
            if success:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Metrics exported to:\n{filepath}"
                )

    def clear_metrics(self):
        """Clear all metrics."""
        from PyQt5.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Clear Metrics",
            "Are you sure you want to clear all routing metrics?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.analytics.clear_metrics()
            self.refresh_data()
            QMessageBox.information(
                self,
                "Metrics Cleared",
                "All routing metrics have been cleared."
            )
