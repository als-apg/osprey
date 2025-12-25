"""
Project Control Panel Manager for Osprey PyQt GUI

This module handles all project control panel operations including:
- Project selector UI management
- Routing mode switching (automatic/manual)
- Cache management and statistics
- Conversation context management
- Project enable/disable operations
"""

from PyQt5.QtWidgets import QMessageBox
from osprey.utils.logger import get_logger

logger = get_logger("project_control")


class ProjectControlManager:
    """Manages project control panel operations for the GUI."""

    def __init__(self, gui):
        """
        Initialize the project control manager.

        Args:
            gui: Reference to the main OspreyGUI instance
        """
        self.gui = gui

    def update_project_selector(self):
        """Update the project selector dropdown with current projects."""
        # Save current selection
        current_data = self.gui.project_selector.currentData()

        # Clear all items except "Automatic Routing"
        while self.gui.project_selector.count() > 1:
            self.gui.project_selector.removeItem(1)

        # Add enabled projects
        enabled_projects = self.gui.project_manager.get_enabled_projects()
        for project_context in enabled_projects:
            project_name = project_context.metadata.name
            self.gui.project_selector.addItem(f"📁 {project_name}", project_name)

        # Restore selection if possible
        if current_data:
            index = self.gui.project_selector.findData(current_data)
            if index >= 0:
                self.gui.project_selector.setCurrentIndex(index)

    def on_project_selected(self, index):
        """
        Handle project selection from dropdown.

        Args:
            index: Index of selected item in dropdown
        """
        project_data = self.gui.project_selector.currentData()

        if project_data == "auto":
            # Switch to automatic mode
            self.gui.router.set_automatic_mode()
            self.gui.routing_mode_label.setText("Mode: Automatic")
            self.gui.routing_mode_label.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")
            self.gui.routing_explanation_label.setText("Queries will be automatically routed to the best project")
            self.gui.routing_explanation_label.setVisible(True)
            self.gui.add_status("Switched to automatic routing mode", "base")
        else:
            # Switch to manual mode
            self.gui.router.set_manual_mode(project_data)
            self.gui.routing_mode_label.setText(f"Mode: Manual")
            self.gui.routing_mode_label.setStyleSheet("color: #FFD700; font-size: 10px; font-weight: bold;")
            self.gui.routing_explanation_label.setText(f"All queries will use: {project_data}")
            self.gui.routing_explanation_label.setVisible(True)
            self.gui.add_status(f"Switched to manual mode: {project_data}", "base")

    def clear_routing_cache(self):
        """Clear the routing cache."""
        try:
            self.gui.router.clear_cache()
            self.gui.add_status("Routing cache cleared", "base")
            self.update_cache_statistics()
            QMessageBox.information(
                self.gui,
                "Cache Cleared",
                "Routing cache has been cleared.\nNext queries will use fresh routing decisions."
            )
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            self.gui.add_status(f"❌ Failed to clear cache: {e}", "error")

    def toggle_cache_stats(self):
        """Toggle cache statistics display."""
        is_visible = self.gui.show_cache_stats_button.isChecked()
        self.gui.cache_stats_label.setVisible(is_visible)

        if is_visible:
            self.update_cache_statistics()
            self.gui.show_cache_stats_button.setText("📊 Hide Stats")
        else:
            self.gui.show_cache_stats_button.setText("📊 Stats")

    def update_cache_statistics(self):
        """Update cache statistics display."""
        try:
            stats = self.gui.router.get_cache_statistics()

            if not stats:
                self.gui.cache_stats_label.setText("Cache: Disabled")
                return

            # Format statistics
            hit_rate_pct = stats.hit_rate * 100
            miss_rate_pct = stats.miss_rate * 100

            stats_text = (
                f"Cache Stats:\n"
                f"  Queries: {stats.total_queries} | "
                f"Hits: {stats.cache_hits} ({hit_rate_pct:.1f}%) | "
                f"Misses: {stats.cache_misses} ({miss_rate_pct:.1f}%)\n"
                f"  Entries: {stats.total_entries} | "
                f"Evictions: {stats.evictions}"
            )

            self.gui.cache_stats_label.setText(stats_text)

        except Exception as e:
            logger.error(f"Failed to update cache statistics: {e}")
            self.gui.cache_stats_label.setText(f"Cache Stats: Error - {e}")

    def clear_conversation_context(self):
        """Clear the conversation context."""
        try:
            self.gui.router.clear_conversation_context()
            self.gui.add_status("Conversation context cleared", "base")
            self.update_context_display()
            QMessageBox.information(
                self.gui,
                "Context Cleared",
                "Conversation context has been cleared.\nTopic detection will start fresh."
            )
        except Exception as e:
            logger.error(f"Failed to clear conversation context: {e}")
            self.gui.add_status(f"❌ Failed to clear context: {e}", "error")

    def toggle_context_display(self):
        """Toggle conversation context display."""
        is_visible = self.gui.show_context_button.isChecked()
        self.gui.context_summary_label.setVisible(is_visible)

        if is_visible:
            self.update_context_display()
            self.gui.show_context_button.setText("💬 Hide Context")
        else:
            self.gui.show_context_button.setText("💬 Context")

    def update_context_display(self):
        """Update conversation context display."""
        try:
            summary = self.gui.router.get_conversation_context_summary()

            if summary and summary != "Conversation context disabled":
                self.gui.context_summary_label.setText(f"Context: {summary}")
            else:
                self.gui.context_summary_label.setText("Context: No conversation history")

        except Exception as e:
            logger.error(f"Failed to update context display: {e}")
            self.gui.context_summary_label.setText(f"Context: Error - {e}")

    def toggle_project_enabled(self, project_name, enabled):
        """
        Enable or disable a project.

        Args:
            project_name: Name of the project to toggle
            enabled: True to enable, False to disable
        """
        try:
            if enabled:
                self.gui.project_manager.enable_project(project_name)
                self.gui.add_status(f"✓ Enabled: {project_name}", "base")
            else:
                self.gui.project_manager.disable_project(project_name)
                self.gui.add_status(f"✗ Disabled: {project_name}", "base")

            # Update project selector
            self.update_project_selector()

            # If we disabled the currently selected project in manual mode, switch to auto
            if not enabled and self.gui.project_selector.currentData() == project_name:
                self.gui.project_selector.setCurrentIndex(0)  # Switch to automatic

        except Exception as e:
            logger.error(f"Failed to toggle project {project_name}: {e}")
            self.gui.add_status(f"❌ Failed to toggle {project_name}: {e}", "error")
