"""Projects Tab for Osprey GUI."""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from osprey.interfaces.pyqt.model_config_dialog import ModelConfigDialog
from osprey.utils.logger import get_logger

logger = get_logger("pyqt_gui.projects_tab")


class ProjectsTab(QWidget):
    """Tab for displaying and managing discovered projects."""

    def __init__(self, parent=None):
        """Initialize the Projects tab.

        Args:
            parent: Parent OspreyGUI instance
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup the projects tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Header with refresh button
        header_layout = QHBoxLayout()
        label = QLabel("Discovered Projects:")
        label.setStyleSheet("color: #00FF00; font-weight: bold;")
        header_layout.addWidget(label)

        header_layout.addStretch()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_projects_display)
        refresh_btn.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Projects table
        self.projects_table = QTableWidget()
        self.projects_table.setColumnCount(7)
        self.projects_table.setHorizontalHeaderLabels(
            [
                "Status",
                "Project Name",
                "Capabilities",
                "Models",
                "Path",
                "Config File",
                "Model Config",
            ]
        )
        self.projects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.projects_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                gridline-color: #3F3F46;
            }
            QTableWidget::item {
                padding: 2px;
                color: #FFFFFF;
                background-color: #1E1E1E;
            }
            QTableWidget::item:alternate {
                background-color: #2D2D30;
            }
            QTableWidget::item:selected {
                background-color: #0078D4;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #2D2D30;
                color: #FFFFFF;
                padding: 3px;
                border: 1px solid #3F3F46;
                font-weight: bold;
            }
        """)
        self.projects_table.setAlternatingRowColors(True)
        self.projects_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.projects_table.setSelectionBehavior(QTableWidget.SelectRows)
        # Enable row resizing - users can drag row borders to resize
        self.projects_table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        # Set a reasonable default row height
        self.projects_table.verticalHeader().setDefaultSectionSize(80)
        # Show vertical header so users can see and drag row borders
        self.projects_table.verticalHeader().setVisible(True)
        layout.addWidget(self.projects_table)

        # Info label
        self.projects_info_label = QLabel(
            "No projects discovered yet. Click Refresh to scan for projects."
        )
        self.projects_info_label.setStyleSheet("color: #808080; font-style: italic; padding: 10px;")
        layout.addWidget(self.projects_info_label)

    def refresh_projects_display(self, force_refresh: bool = False):
        """
        Refresh the discovered projects display.

        Args:
            force_refresh: If True, bypass cache and rediscover projects
        """
        try:
            self.parent_gui.add_status("Refreshing project list...", "base")

            # Get loaded projects from ProjectManager
            loaded_projects = self.parent_gui.project_manager.list_loaded_projects()

            # Build display data from ProjectManager
            display_projects = []
            for project_name in loaded_projects:
                context = self.parent_gui.project_manager.get_project(project_name)
                if not context:
                    continue

                # Get capabilities from ProjectManager
                capabilities = self.parent_gui.project_manager.get_project_capabilities(
                    project_name
                )
                capability_names = list(capabilities.keys())

                # Get models from project config
                models = context.config.raw_config.get("models", {})

                display_projects.append(
                    {
                        "name": context.metadata.name,
                        "path": str(context.metadata.path),
                        "config_path": str(context.metadata.config_path),
                        "description": context.metadata.description,
                        "version": context.metadata.version,
                        "capabilities": capability_names,
                        "models": models,
                    }
                )

            # Update the cached list for backward compatibility
            self.parent_gui.discovered_projects = display_projects
            self.parent_gui._projects_cache_valid = True

            # Update table
            self.projects_table.setRowCount(len(display_projects))

            # Enable word wrap and adjust row heights
            self.projects_table.setWordWrap(True)

            for row, project in enumerate(display_projects):
                # Status column with enable/disable checkbox
                status_widget = QWidget()
                status_layout = QHBoxLayout(status_widget)
                status_layout.setContentsMargins(2, 1, 2, 1)

                enabled_checkbox = QCheckBox()
                # Check if project is enabled in ProjectManager
                is_enabled = self.parent_gui.project_manager.is_project_enabled(project["name"])
                enabled_checkbox.setChecked(is_enabled)
                enabled_checkbox.stateChanged.connect(
                    lambda state, p=project["name"]: self.parent_gui.toggle_project_enabled(
                        p, state == Qt.Checked
                    )
                )
                status_layout.addWidget(enabled_checkbox)

                status_label = QLabel("Enabled" if is_enabled else "Disabled")
                status_label.setStyleSheet("color: #00FF00;" if is_enabled else "color: #808080;")
                status_layout.addWidget(status_label)

                self.projects_table.setCellWidget(row, 0, status_widget)

                # Project name
                name_item = QTableWidgetItem(project["name"])
                name_item.setForeground(QColor("#00FFFF"))
                self.projects_table.setItem(row, 1, name_item)

                # Capabilities column - show count and list
                capabilities = project.get("capabilities", [])
                cap_count = len(capabilities)
                if cap_count > 0:
                    cap_text = f"{cap_count} capabilities:\n" + "\n".join(
                        f"  • {cap}" for cap in capabilities[:5]
                    )
                    if cap_count > 5:
                        cap_text += f"\n  ... and {cap_count - 5} more"
                else:
                    cap_text = "No capabilities"
                cap_item = QTableWidgetItem(cap_text)
                cap_item.setForeground(QColor("#00FF00") if cap_count > 0 else QColor("#808080"))
                cap_item.setToolTip(
                    "\n".join(capabilities) if capabilities else "No capabilities found"
                )
                self.projects_table.setItem(row, 2, cap_item)

                # Models column - show configured models
                models = project.get("models", {})
                model_count = len(models)
                if model_count > 0:
                    model_text = f"{model_count} models:\n" + "\n".join(
                        f"  • {step}: {model}" for step, model in list(models.items())[:5]
                    )
                    if model_count > 5:
                        model_text += f"\n  ... and {model_count - 5} more"
                else:
                    model_text = "No models"
                model_item = QTableWidgetItem(model_text)
                model_item.setForeground(
                    QColor("#FFD700") if model_count > 0 else QColor("#808080")
                )
                model_item.setToolTip(
                    "\n".join(f"{step}: {model}" for step, model in models.items())
                    if models
                    else "No models configured"
                )
                self.projects_table.setItem(row, 3, model_item)

                # Project path
                path_item = QTableWidgetItem(project["path"])
                path_item.setForeground(QColor("#FFFFFF"))
                self.projects_table.setItem(row, 4, path_item)

                # Config path
                config_path = Path(project["config_path"]).name
                config_item = QTableWidgetItem(config_path)
                config_item.setForeground(QColor("#00FF00"))
                self.projects_table.setItem(row, 5, config_item)

                # Model configuration button
                models_widget = QWidget()
                models_layout = QHBoxLayout(models_widget)
                models_layout.setContentsMargins(2, 1, 2, 1)

                config_btn = QPushButton("Configure")
                config_btn.setToolTip("Configure runtime model overrides for infrastructure steps")
                config_btn.clicked.connect(
                    lambda checked, p=project: self.configure_project_models(p)
                )
                models_layout.addWidget(config_btn)

                # Show indicator if runtime overrides are configured
                pref_count = self.parent_gui.model_preferences.get_preference_count(project["name"])
                if pref_count > 0:
                    indicator = QLabel(f"✓ ({pref_count})")
                    indicator.setToolTip(f"{pref_count} runtime override(s) configured")
                    indicator.setStyleSheet("color: #00FF00;")
                    models_layout.addWidget(indicator)

                self.projects_table.setCellWidget(row, 6, models_widget)

            # Resize rows to fit content initially, then users can manually adjust
            self.projects_table.resizeRowsToContents()

            # Update info label
            if display_projects:
                enabled_count = len(
                    [
                        p
                        for p in display_projects
                        if self.parent_gui.project_manager.is_project_enabled(p["name"])
                    ]
                )
                self.projects_info_label.setText(
                    f"Found {len(display_projects)} project(s) • "
                    f"{enabled_count} enabled • "
                    f"Use checkboxes to enable/disable projects for routing"
                )
                self.projects_info_label.setStyleSheet("color: #00FF00; padding: 10px;")
            else:
                self.projects_info_label.setText(
                    "No projects found. Projects must have a config.yml file in their root directory."
                )
                self.projects_info_label.setStyleSheet("color: #FFA500; padding: 10px;")

            self.parent_gui.add_status(f"Found {len(display_projects)} project(s)", "base")

        except Exception as e:
            logger.exception(f"Error refreshing projects: {e}")
            self.parent_gui.add_status(f"❌ Failed to refresh projects: {e}", "error")
            QMessageBox.warning(self, "Error", f"Failed to refresh projects:\n{e}")

    def configure_project_models(self, project_info):
        """Open dialog to configure models for a project."""
        dialog = ModelConfigDialog(project_info, self.parent_gui.model_preferences, self)
        if dialog.exec_() == QDialog.DialogCode.Accepted:
            # Refresh the projects table to show updated configuration
            self.refresh_projects_display()

            pref_count = self.parent_gui.model_preferences.get_preference_count(
                project_info["name"]
            )
            if pref_count > 0:
                QMessageBox.information(
                    self,
                    "Configuration Saved",
                    f"Model configuration for {project_info['name']} has been saved.\n"
                    f"{pref_count} step(s) configured.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Configuration Cleared",
                    f"Model configuration for {project_info['name']} has been cleared.\n"
                    f"All steps will use default models from config.",
                )
