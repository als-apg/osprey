"""
Model Configuration Dialog for PyQt GUI

Provides a dialog for configuring per-step LLM models for discovered projects.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFormLayout, QMessageBox
)

from osprey.interfaces.pyqt.model_preferences import ModelPreferencesStore


class ModelConfigDialog(QDialog):
    """Dialog for configuring per-step models for a project."""
    
    def __init__(self, project_info: dict, preferences_manager: ModelPreferencesStore, parent=None):
        """
        Initialize the model configuration dialog.
        
        Args:
            project_info: Project information dictionary from discover_projects()
            preferences_manager: ModelPreferencesManager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.project_info = project_info
        self.preferences_manager = preferences_manager
        self.step_model_combos = {}
        
        self.setWindowTitle(f"Configure Models - {project_info['name']}")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel(f"<h3>Model Configuration for {self.project_info['name']}</h3>")
        layout.addWidget(header)
        
        # Get provider from config
        config_path = self.project_info.get('config_path')
        provider = None
        if config_path:
            provider = self.preferences_manager.get_provider_from_config(config_path)
        
        # Provider info
        provider_label = QLabel(
            f"<b>LLM Provider:</b> {provider or 'Not configured'}"
        )
        layout.addWidget(provider_label)
        
        layout.addWidget(QLabel(
            "<p>Select which model to use for each infrastructure step:</p>"
        ))
        
        # Get available models for this provider (with dynamic discovery)
        available_models = []
        if provider:
            available_models = self.preferences_manager.get_available_models(
                provider,
                config_path=config_path,
                use_dynamic=True
            )
        
        if not available_models:
            layout.addWidget(QLabel(
                f"<i>No models available for provider '{provider or 'unknown'}'</i>"
            ))
        else:
            # Create form for each step
            form_layout = QFormLayout()
            
            # Get current preferences
            current_prefs = self.preferences_manager.get_all_preferences(
                self.project_info['name']
            )
            
            for step in ModelPreferencesStore.INFRASTRUCTURE_STEPS:
                combo = QComboBox()
                combo.addItem("(Use default from config)", "")
                
                for model in available_models:
                    combo.addItem(model, model)
                
                # Set current selection if exists
                current_model = current_prefs.get(step)
                if current_model:
                    index = combo.findData(current_model)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                
                self.step_model_combos[step] = combo
                
                # Format step name for display
                step_display = step.replace('_', ' ').title()
                form_layout.addRow(f"{step_display}:", combo)
            
            layout.addLayout(form_layout)
        
        # Info text
        info_label = QLabel(
            "<p style='color: #808080; font-size: 10px;'>"
            "<i>Note: These preferences are stored in memory and will be lost when the application closes.</i>"
            "</p>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_configuration)
        save_btn.setDefault(True)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(clear_btn)
        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def clear_all(self):
        """Clear all model selections."""
        reply = QMessageBox.question(
            self,
            "Clear All",
            "Reset all model selections to default?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for combo in self.step_model_combos.values():
                combo.setCurrentIndex(0)  # Set to "(Use default)"
    
    def save_configuration(self):
        """Save the model configuration."""
        step_models = {}
        
        for step, combo in self.step_model_combos.items():
            model = combo.currentData()
            if model:  # Only save if not default
                step_models[step] = model
        
        # Update preferences manager
        if step_models:
            self.preferences_manager.set_all_preferences(
                self.project_info['name'],
                step_models
            )
        else:
            # Clear preferences if all are default
            self.preferences_manager.clear_preferences(
                self.project_info['name']
            )
        
        self.accept()