"""
Model Preferences Manager for Osprey PyQt GUI

This module handles model preference configuration for projects:
- Configuring project models via dialog
- Applying model preferences to runtime configuration
- Managing model overrides for infrastructure steps
"""

from PyQt5.QtWidgets import QMessageBox, QDialog
from osprey.utils.logger import get_logger

logger = get_logger("model_preferences_manager")


class ModelPreferencesUIHandler:
    """Manages model preferences and configuration for projects."""

    def __init__(self, gui):
        """
        Initialize the model preferences manager.

        Args:
            gui: Reference to the main OspreyGUI instance
        """
        self.gui = gui

    def configure_project_models(self, project_info):
        """
        Open dialog to configure models for a project.

        Args:
            project_info: Dictionary with project information
        """
        from osprey.interfaces.pyqt.model_config_dialog import ModelConfigDialog

        dialog = ModelConfigDialog(project_info, self.gui.model_preferences, self.gui)
        if dialog.exec_() == QDialog.DialogCode.Accepted:
            # Refresh the projects table to show updated configuration
            self.gui.refresh_projects_display()

            pref_count = self.gui.model_preferences.get_preference_count(project_info['name'])
            if pref_count > 0:
                QMessageBox.information(
                    self.gui,
                    "Configuration Saved",
                    f"Model configuration for {project_info['name']} has been saved.\n"
                    f"{pref_count} step(s) configured."
                )
            else:
                QMessageBox.information(
                    self.gui,
                    "Configuration Cleared",
                    f"Model configuration for {project_info['name']} has been cleared.\n"
                    f"All steps will use default models from config."
                )

    def apply_model_preferences_to_config(self, project_name: str):
        """
        Apply model preferences for a project to the runtime configuration.

        This should be called when loading/switching to a project to ensure
        the configured models are used for infrastructure steps.

        Args:
            project_name: Name of the project to apply preferences for
        """
        from osprey.utils.config import set_runtime_model_override

        # Apply preferences for this project (don't clear - we want to merge all projects)
        preferences = self.gui.model_preferences.get_all_preferences(project_name)
        if preferences:
            for step, model_id in preferences.items():
                set_runtime_model_override(step, model_id)
                self.gui.add_status(f"✓ {project_name}: {step} → {model_id}", "base")

    def apply_all_model_preferences(self):
        """
        Apply model preferences for all discovered projects.

        This implements the hybrid approach where:
        - Infrastructure steps use GUI-configured models (runtime overrides)
        - Each project's capabilities can have their own models
        - Later projects' preferences override earlier ones for infrastructure
        """
        from osprey.utils.config import clear_runtime_model_overrides, get_runtime_model_overrides

        # Clear existing overrides first
        clear_runtime_model_overrides()

        if not self.gui.discovered_projects:
            return

        self.gui.add_status("Applying model preferences for multi-project setup...", "base")

        # Apply preferences for each project
        # Later projects override earlier ones for infrastructure steps
        for project in self.gui.discovered_projects:
            project_name = project['name']
            self.apply_model_preferences_to_config(project_name)

        # Log summary
        overrides = get_runtime_model_overrides()
        if overrides:
            self.gui.add_status(f"Applied {len(overrides)} model override(s) for infrastructure steps", "base")
        else:
            self.gui.add_status("No model preferences configured - using defaults from config", "base")
