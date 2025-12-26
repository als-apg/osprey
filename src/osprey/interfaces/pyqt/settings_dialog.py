"""Settings Dialog for Osprey Framework GUI.

This module provides the settings dialog for configuring framework settings
including agent control, approval modes, execution limits, GUI settings,
development/debug options, and advanced routing configuration.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFormLayout,
    QCheckBox, QSpinBox, QComboBox, QTabWidget, QWidget, QLabel,
    QGroupBox, QDoubleSpinBox
)

from osprey.interfaces.pyqt.gui_utils import create_dark_palette


class SettingsDialog(QDialog):
    """Dialog for configuring framework settings."""

    def __init__(self, parent, title, current_settings):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)  # Make dialog modeless (non-blocking)

        # Set a comfortable default size that fits all tabs without scrolling
        # Made wider and about twice the height for better visibility
        self.resize(750, 600)
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        # Apply dark theme to settings dialog
        self.setPalette(create_dark_palette())

        self.current_settings = current_settings.copy()
        self.setup_ui()

    def setup_ui(self):
        """Setup the settings dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget for organized settings
        tab_widget = QTabWidget()

        # ===== AGENT CONTROL TAB =====
        agent_control_tab = QWidget()
        agent_control_layout = QVBoxLayout()
        agent_control_tab.setLayout(agent_control_layout)

        agent_form = QFormLayout()

        # Planning Mode
        self.planning_mode_checkbox = QCheckBox()
        self.planning_mode_checkbox.setChecked(self.current_settings.get('planning_mode_enabled', False))
        self.planning_mode_checkbox.setToolTip("Enable multi-step planning and orchestration")
        agent_form.addRow("Planning Mode:", self.planning_mode_checkbox)

        # EPICS Writes
        self.epics_writes_checkbox = QCheckBox()
        self.epics_writes_checkbox.setChecked(self.current_settings.get('epics_writes_enabled', False))
        self.epics_writes_checkbox.setToolTip("Allow EPICS write operations (requires approval)")
        agent_form.addRow("EPICS Writes:", self.epics_writes_checkbox)

        # Task Extraction Bypass
        self.task_extraction_bypass_checkbox = QCheckBox()
        self.task_extraction_bypass_checkbox.setChecked(
            self.current_settings.get('task_extraction_bypass_enabled', False)
        )
        self.task_extraction_bypass_checkbox.setToolTip(
            "Skip task extraction step for performance (use full context)"
        )
        agent_form.addRow("Task Extraction Bypass:", self.task_extraction_bypass_checkbox)

        # Capability Selection Bypass
        self.capability_selection_bypass_checkbox = QCheckBox()
        self.capability_selection_bypass_checkbox.setChecked(
            self.current_settings.get('capability_selection_bypass_enabled', False)
        )
        self.capability_selection_bypass_checkbox.setToolTip(
            "Skip capability selection (activate all capabilities)"
        )
        agent_form.addRow("Capability Selection Bypass:", self.capability_selection_bypass_checkbox)

        agent_control_layout.addLayout(agent_form)
        agent_control_layout.addStretch()
        tab_widget.addTab(agent_control_tab, "Agent Control")

        # ===== APPROVAL TAB =====
        approval_tab = QWidget()
        approval_layout = QVBoxLayout()
        approval_tab.setLayout(approval_layout)

        approval_form = QFormLayout()

        # Global Approval Mode
        self.approval_global_mode_combo = QComboBox()
        self.approval_global_mode_combo.addItems(['disabled', 'selective', 'all_capabilities'])
        current_global_mode = self.current_settings.get('approval_global_mode', 'selective')
        index = self.approval_global_mode_combo.findText(current_global_mode)
        if index >= 0:
            self.approval_global_mode_combo.setCurrentIndex(index)
        self.approval_global_mode_combo.setToolTip(
            "Global approval mode:\n"
            "• disabled: No approvals required\n"
            "• selective: Use capability-specific settings\n"
            "• all_capabilities: All operations require approval"
        )
        approval_form.addRow("Global Approval Mode:", self.approval_global_mode_combo)

        # Python Execution Approval
        self.python_execution_approval_checkbox = QCheckBox()
        self.python_execution_approval_checkbox.setChecked(
            self.current_settings.get('python_execution_approval_enabled', True)
        )
        self.python_execution_approval_checkbox.setToolTip("Require approval for Python code execution")
        approval_form.addRow("Python Execution Approval:", self.python_execution_approval_checkbox)

        # Python Execution Approval Mode
        self.python_execution_approval_mode_combo = QComboBox()
        self.python_execution_approval_mode_combo.addItems(['disabled', 'epics_writes', 'all_code'])
        current_py_mode = self.current_settings.get('python_execution_approval_mode', 'all_code')
        index = self.python_execution_approval_mode_combo.findText(current_py_mode)
        if index >= 0:
            self.python_execution_approval_mode_combo.setCurrentIndex(index)
        self.python_execution_approval_mode_combo.setToolTip(
            "Python approval granularity:\n"
            "• disabled: No approval needed\n"
            "• epics_writes: Approve only EPICS write operations\n"
            "• all_code: Approve all code execution"
        )
        approval_form.addRow("Python Approval Mode:", self.python_execution_approval_mode_combo)

        # Memory Approval
        self.memory_approval_checkbox = QCheckBox()
        self.memory_approval_checkbox.setChecked(
            self.current_settings.get('memory_approval_enabled', True)
        )
        self.memory_approval_checkbox.setToolTip("Require approval for memory operations")
        approval_form.addRow("Memory Approval:", self.memory_approval_checkbox)

        approval_layout.addLayout(approval_form)
        approval_layout.addStretch()
        tab_widget.addTab(approval_tab, "Approval")

        # ===== EXECUTION LIMITS TAB =====
        limits_tab = QWidget()
        limits_layout = QVBoxLayout()
        limits_tab.setLayout(limits_layout)

        limits_form = QFormLayout()

        # Max Reclassifications
        self.max_reclassifications_spin = QSpinBox()
        self.max_reclassifications_spin.setRange(0, 10)
        self.max_reclassifications_spin.setValue(
            self.current_settings.get('max_reclassifications', 1)
        )
        self.max_reclassifications_spin.setToolTip("Maximum task reclassification attempts")
        limits_form.addRow("Max Reclassifications:", self.max_reclassifications_spin)

        # Max Planning Attempts
        self.max_planning_attempts_spin = QSpinBox()
        self.max_planning_attempts_spin.setRange(1, 10)
        self.max_planning_attempts_spin.setValue(
            self.current_settings.get('max_planning_attempts', 2)
        )
        self.max_planning_attempts_spin.setToolTip("Maximum planning attempts before giving up")
        limits_form.addRow("Max Planning Attempts:", self.max_planning_attempts_spin)

        # Max Step Retries
        self.max_step_retries_spin = QSpinBox()
        self.max_step_retries_spin.setRange(0, 10)
        self.max_step_retries_spin.setValue(
            self.current_settings.get('max_step_retries', 0)
        )
        self.max_step_retries_spin.setToolTip("Maximum retries per execution step")
        limits_form.addRow("Max Step Retries:", self.max_step_retries_spin)

        # Max Execution Time
        self.max_execution_time_spin = QSpinBox()
        self.max_execution_time_spin.setRange(10, 7200)
        self.max_execution_time_spin.setValue(
            self.current_settings.get('max_execution_time_seconds', 300)
        )
        self.max_execution_time_spin.setSuffix(" seconds")
        self.max_execution_time_spin.setToolTip("Maximum total execution time")
        limits_form.addRow("Max Execution Time:", self.max_execution_time_spin)

        # Max Concurrent Classifications
        self.max_concurrent_classifications_spin = QSpinBox()
        self.max_concurrent_classifications_spin.setRange(1, 20)
        self.max_concurrent_classifications_spin.setValue(
            self.current_settings.get('max_concurrent_classifications', 5)
        )
        self.max_concurrent_classifications_spin.setToolTip("Maximum parallel LLM classification requests")
        limits_form.addRow("Max Concurrent Classifications:", self.max_concurrent_classifications_spin)

        limits_layout.addLayout(limits_form)
        limits_layout.addStretch()
        tab_widget.addTab(limits_tab, "Execution Limits")

        # ===== GUI SETTINGS TAB =====
        gui_tab = QWidget()
        gui_layout = QVBoxLayout()
        gui_tab.setLayout(gui_layout)

        gui_form = QFormLayout()

        # Conversation Persistence
        self.use_persistent_conversations_checkbox = QCheckBox()
        self.use_persistent_conversations_checkbox.setChecked(
            self.current_settings.get('use_persistent_conversations', True)
        )
        self.use_persistent_conversations_checkbox.setToolTip("Save conversation history to database")
        gui_form.addRow("Save Conversation History:", self.use_persistent_conversations_checkbox)

        # Conversation Storage Mode
        self.conversation_storage_mode_combo = QComboBox()
        self.conversation_storage_mode_combo.addItems(['json', 'postgresql'])
        current_storage_mode = self.current_settings.get('conversation_storage_mode', 'json')
        index = self.conversation_storage_mode_combo.findText(current_storage_mode)
        if index >= 0:
            self.conversation_storage_mode_combo.setCurrentIndex(index)
        self.conversation_storage_mode_combo.setToolTip(
            "Conversation message storage:\n"
            "• json: Store messages in JSON file (simple, portable)\n"
            "• postgresql: Store messages in PostgreSQL database (requires setup)"
        )
        gui_form.addRow("Message Storage Mode:", self.conversation_storage_mode_combo)

        # GUI Output Redirection
        self.redirect_output_to_gui_checkbox = QCheckBox()
        self.redirect_output_to_gui_checkbox.setChecked(
            self.current_settings.get('redirect_output_to_gui', True)
        )
        self.redirect_output_to_gui_checkbox.setToolTip(
            "Redirect terminal output to System Information tab (requires restart)"
        )
        gui_form.addRow("Redirect Output to GUI:", self.redirect_output_to_gui_checkbox)

        # Group System Messages
        self.group_system_messages_checkbox = QCheckBox()
        self.group_system_messages_checkbox.setChecked(
            self.current_settings.get('group_system_messages', True)
        )
        self.group_system_messages_checkbox.setToolTip(
            "Group system messages by type in collapsible sections"
        )
        gui_form.addRow("Group System Messages:", self.group_system_messages_checkbox)

        # Suppress Terminal Output
        self.suppress_terminal_output_checkbox = QCheckBox()
        self.suppress_terminal_output_checkbox.setChecked(
            self.current_settings.get('suppress_terminal_output', False)
        )
        self.suppress_terminal_output_checkbox.setToolTip(
            "Suppress terminal output (show only in GUI)\n"
            "When unchecked: messages appear in both terminal and GUI\n"
            "When checked: messages appear only in GUI System Information tab"
        )
        gui_form.addRow("Suppress Terminal Output:", self.suppress_terminal_output_checkbox)

        # Enable Routing Feedback
        self.enable_routing_feedback_checkbox = QCheckBox()
        self.enable_routing_feedback_checkbox.setChecked(
            self.current_settings.get('enable_routing_feedback', True)
        )
        self.enable_routing_feedback_checkbox.setToolTip(
            "Enable user feedback collection for routing decisions\n"
            "When enabled: prompts for feedback after each routing decision\n"
            "When disabled: no feedback prompts shown\n"
            "Feedback helps the system learn and improve routing accuracy"
        )
        gui_form.addRow("Enable Routing Feedback:", self.enable_routing_feedback_checkbox)

        gui_layout.addLayout(gui_form)
        gui_layout.addStretch()
        tab_widget.addTab(gui_tab, "GUI Settings")

        # ===== DEVELOPMENT/DEBUG TAB =====
        dev_tab = QWidget()
        dev_layout = QVBoxLayout()
        dev_tab.setLayout(dev_layout)

        dev_form = QFormLayout()

        # Debug Mode
        self.debug_mode_checkbox = QCheckBox()
        self.debug_mode_checkbox.setChecked(
            self.current_settings.get('debug_mode', False)
        )
        self.debug_mode_checkbox.setToolTip(
            "Enable DEBUG logging level (shows all framework debug messages)\n"
            "When unchecked: INFO level (shows only important messages)\n"
            "Changes apply immediately when you click Save"
        )
        dev_form.addRow("Debug Mode:", self.debug_mode_checkbox)

        # Verbose Logging
        self.verbose_logging_checkbox = QCheckBox()
        self.verbose_logging_checkbox.setChecked(
            self.current_settings.get('verbose_logging', False)
        )
        self.verbose_logging_checkbox.setToolTip(
            "Enable verbose logging output (development.verbose_logging)\n"
            "Note: Terminal logging level is set in logger.py, not here"
        )
        dev_form.addRow("Verbose Logging:", self.verbose_logging_checkbox)

        # Raise Raw Errors
        self.raise_raw_errors_checkbox = QCheckBox()
        self.raise_raw_errors_checkbox.setChecked(
            self.current_settings.get('raise_raw_errors', False)
        )
        self.raise_raw_errors_checkbox.setToolTip(
            "Show full error stack traces (development.raise_raw_errors)\n"
            "Re-raises original exceptions instead of wrapped framework errors"
        )
        dev_form.addRow("Raise Raw Errors:", self.raise_raw_errors_checkbox)

        # Print Prompts to Files
        self.print_prompts_checkbox = QCheckBox()
        self.print_prompts_checkbox.setChecked(
            self.current_settings.get('print_prompts', False)
        )
        self.print_prompts_checkbox.setToolTip(
            "Save all prompts to files (development.prompts.print_all)\n"
            "Prompts are saved to prompts/ directory for inspection"
        )
        dev_form.addRow("Save Prompts to Files:", self.print_prompts_checkbox)

        # Show Prompts in Console
        self.show_prompts_checkbox = QCheckBox()
        self.show_prompts_checkbox.setChecked(
            self.current_settings.get('show_prompts', False)
        )
        self.show_prompts_checkbox.setToolTip(
            "Display prompts in console (development.prompts.show_all)\n"
            "Shows prompts in System Information tab with detailed formatting"
        )
        dev_form.addRow("Show Prompts in Console:", self.show_prompts_checkbox)

        # Latest Only (for prompt files)
        self.prompts_latest_only_checkbox = QCheckBox()
        self.prompts_latest_only_checkbox.setChecked(
            self.current_settings.get('prompts_latest_only', True)
        )
        self.prompts_latest_only_checkbox.setToolTip(
            "Use latest.md filename instead of timestamped files for prompts"
        )
        dev_form.addRow("Prompts: Latest Only:", self.prompts_latest_only_checkbox)

        dev_layout.addLayout(dev_form)

        # Memory Monitoring Section
        memory_group = QGroupBox("Memory Monitoring")
        memory_layout = QFormLayout()

        self.memory_monitor_enabled_checkbox = QCheckBox()
        self.memory_monitor_enabled_checkbox.setChecked(
            self.current_settings.get('memory_monitor_enabled', True)
        )
        self.memory_monitor_enabled_checkbox.setToolTip(
            "Enable automatic memory monitoring in Memory tab"
        )
        memory_layout.addRow("Enable Memory Monitor:", self.memory_monitor_enabled_checkbox)

        self.memory_warning_threshold_spin = QSpinBox()
        self.memory_warning_threshold_spin.setRange(100, 10000)
        self.memory_warning_threshold_spin.setValue(
            self.current_settings.get('memory_warning_threshold_mb', 500)
        )
        self.memory_warning_threshold_spin.setSuffix(" MB")
        self.memory_warning_threshold_spin.setToolTip("Warning threshold for framework memory usage")
        memory_layout.addRow("Warning Threshold:", self.memory_warning_threshold_spin)

        self.memory_critical_threshold_spin = QSpinBox()
        self.memory_critical_threshold_spin.setRange(200, 20000)
        self.memory_critical_threshold_spin.setValue(
            self.current_settings.get('memory_critical_threshold_mb', 1000)
        )
        self.memory_critical_threshold_spin.setSuffix(" MB")
        self.memory_critical_threshold_spin.setToolTip("Critical threshold for framework memory usage")
        memory_layout.addRow("Critical Threshold:", self.memory_critical_threshold_spin)

        self.memory_check_interval_spin = QSpinBox()
        self.memory_check_interval_spin.setRange(1, 60)
        self.memory_check_interval_spin.setValue(
            self.current_settings.get('memory_check_interval_seconds', 5)
        )
        self.memory_check_interval_spin.setSuffix(" seconds")
        self.memory_check_interval_spin.setToolTip("How often to update memory statistics")
        memory_layout.addRow("Check Interval:", self.memory_check_interval_spin)

        memory_group.setLayout(memory_layout)
        dev_layout.addWidget(memory_group)

        # Add warning message
        warning_label = QLabel(
            "⚠️ Warning: Debug settings may impact performance and generate large log files.\n"
            "Recommended for development and troubleshooting only."
        )
        warning_label.setStyleSheet("color: #FFA500; font-style: italic; padding: 10px;")
        warning_label.setWordWrap(True)
        dev_layout.addWidget(warning_label)

        dev_layout.addStretch()
        tab_widget.addTab(dev_tab, "Development/Debug")

        # ===== ADVANCED ROUTING TAB (Phase 2.4) =====
        routing_tab = QWidget()
        routing_layout = QVBoxLayout()
        routing_tab.setLayout(routing_layout)

        routing_form = QFormLayout()

        # === Cache Configuration ===
        cache_group = QGroupBox("Cache Configuration")
        cache_layout = QFormLayout()

        self.enable_routing_cache_checkbox = QCheckBox()
        self.enable_routing_cache_checkbox.setChecked(
            self.current_settings.get('enable_routing_cache', True)
        )
        self.enable_routing_cache_checkbox.setToolTip("Enable routing decision caching")
        cache_layout.addRow("Enable Routing Cache:", self.enable_routing_cache_checkbox)

        self.cache_max_size_spin = QSpinBox()
        self.cache_max_size_spin.setRange(10, 1000)
        self.cache_max_size_spin.setValue(
            self.current_settings.get('cache_max_size', 100)
        )
        self.cache_max_size_spin.setToolTip("Maximum number of cached routing decisions")
        cache_layout.addRow("Cache Size:", self.cache_max_size_spin)

        self.cache_ttl_spin = QSpinBox()
        self.cache_ttl_spin.setRange(60, 86400)
        self.cache_ttl_spin.setValue(
            int(self.current_settings.get('cache_ttl_seconds', 3600))
        )
        self.cache_ttl_spin.setSuffix(" seconds")
        self.cache_ttl_spin.setToolTip("Time-to-live for cache entries (1 hour = 3600s)")
        cache_layout.addRow("Cache TTL:", self.cache_ttl_spin)

        self.cache_similarity_spin = QDoubleSpinBox()
        self.cache_similarity_spin.setRange(0.5, 1.0)
        self.cache_similarity_spin.setSingleStep(0.05)
        self.cache_similarity_spin.setValue(
            self.current_settings.get('cache_similarity_threshold', 0.85)
        )
        self.cache_similarity_spin.setToolTip("Minimum similarity score for cache hit (0.5-1.0)")
        cache_layout.addRow("Similarity Threshold:", self.cache_similarity_spin)

        cache_group.setLayout(cache_layout)
        routing_layout.addWidget(cache_group)

        # === Advanced Cache Invalidation ===
        invalidation_group = QGroupBox("Advanced Cache Invalidation")
        invalidation_layout = QFormLayout()

        self.enable_advanced_invalidation_checkbox = QCheckBox()
        self.enable_advanced_invalidation_checkbox.setChecked(
            self.current_settings.get('enable_advanced_invalidation', True)
        )
        self.enable_advanced_invalidation_checkbox.setToolTip(
            "Enable intelligent cache invalidation strategies"
        )
        invalidation_layout.addRow("Enable Advanced Invalidation:", self.enable_advanced_invalidation_checkbox)

        self.enable_adaptive_ttl_checkbox = QCheckBox()
        self.enable_adaptive_ttl_checkbox.setChecked(
            self.current_settings.get('enable_adaptive_ttl', True)
        )
        self.enable_adaptive_ttl_checkbox.setToolTip(
            "Hot entries cached longer, cold entries expire faster"
        )
        invalidation_layout.addRow("  Adaptive TTL:", self.enable_adaptive_ttl_checkbox)

        self.enable_probabilistic_expiration_checkbox = QCheckBox()
        self.enable_probabilistic_expiration_checkbox.setChecked(
            self.current_settings.get('enable_probabilistic_expiration', True)
        )
        self.enable_probabilistic_expiration_checkbox.setToolTip(
            "XFetch algorithm prevents cache stampede"
        )
        invalidation_layout.addRow("  Probabilistic Expiration:", self.enable_probabilistic_expiration_checkbox)

        self.enable_event_driven_invalidation_checkbox = QCheckBox()
        self.enable_event_driven_invalidation_checkbox.setChecked(
            self.current_settings.get('enable_event_driven_invalidation', True)
        )
        self.enable_event_driven_invalidation_checkbox.setToolTip(
            "Auto-invalidate on config/capability changes"
        )
        invalidation_layout.addRow("  Event-Driven Invalidation:", self.enable_event_driven_invalidation_checkbox)

        invalidation_group.setLayout(invalidation_layout)
        routing_layout.addWidget(invalidation_group)

        # === Semantic Analysis ===
        semantic_group = QGroupBox("Semantic Context Analysis")
        semantic_layout = QFormLayout()

        self.enable_semantic_analysis_checkbox = QCheckBox()
        self.enable_semantic_analysis_checkbox.setChecked(
            self.current_settings.get('enable_semantic_analysis', True)
        )
        self.enable_semantic_analysis_checkbox.setToolTip(
            "Enable semantic understanding of conversation context"
        )
        semantic_layout.addRow("Enable Semantic Analysis:", self.enable_semantic_analysis_checkbox)

        self.semantic_similarity_spin = QDoubleSpinBox()
        self.semantic_similarity_spin.setRange(0.3, 0.9)
        self.semantic_similarity_spin.setSingleStep(0.05)
        self.semantic_similarity_spin.setValue(
            self.current_settings.get('semantic_similarity_threshold', 0.5)
        )
        self.semantic_similarity_spin.setToolTip("Minimum similarity for context relevance")
        semantic_layout.addRow("  Similarity Threshold:", self.semantic_similarity_spin)

        self.topic_similarity_spin = QDoubleSpinBox()
        self.topic_similarity_spin.setRange(0.3, 0.9)
        self.topic_similarity_spin.setSingleStep(0.05)
        self.topic_similarity_spin.setValue(
            self.current_settings.get('topic_similarity_threshold', 0.6)
        )
        self.topic_similarity_spin.setToolTip("Minimum similarity for same topic detection")
        semantic_layout.addRow("  Topic Similarity:", self.topic_similarity_spin)

        self.max_context_history_spin = QSpinBox()
        self.max_context_history_spin.setRange(5, 100)
        self.max_context_history_spin.setValue(
            self.current_settings.get('max_context_history', 20)
        )
        self.max_context_history_spin.setToolTip("Maximum conversation queries to track")
        semantic_layout.addRow("  Max Context History:", self.max_context_history_spin)

        semantic_group.setLayout(semantic_layout)
        routing_layout.addWidget(semantic_group)

        # === Orchestration & Analytics ===
        other_group = QGroupBox("Orchestration & Analytics")
        other_layout = QFormLayout()

        self.orchestration_max_parallel_spin = QSpinBox()
        self.orchestration_max_parallel_spin.setRange(1, 10)
        self.orchestration_max_parallel_spin.setValue(
            self.current_settings.get('orchestration_max_parallel', 3)
        )
        self.orchestration_max_parallel_spin.setToolTip("Maximum parallel sub-queries for orchestration")
        other_layout.addRow("Max Parallel Queries:", self.orchestration_max_parallel_spin)

        self.analytics_max_history_spin = QSpinBox()
        self.analytics_max_history_spin.setRange(100, 10000)
        self.analytics_max_history_spin.setSingleStep(100)
        self.analytics_max_history_spin.setValue(
            self.current_settings.get('analytics_max_history', 1000)
        )
        self.analytics_max_history_spin.setToolTip("Maximum analytics records to keep")
        other_layout.addRow("Analytics Max History:", self.analytics_max_history_spin)

        other_group.setLayout(other_layout)
        routing_layout.addWidget(other_group)

        routing_layout.addStretch()
        tab_widget.addTab(routing_tab, "Advanced Routing")

        layout.addWidget(tab_widget)

        # Buttons
        button_layout = QHBoxLayout()

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        button_layout.addWidget(save_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def get_settings(self):
        """Get the current settings from the dialog."""
        return {
            # Agent Control
            'planning_mode_enabled': self.planning_mode_checkbox.isChecked(),
            'epics_writes_enabled': self.epics_writes_checkbox.isChecked(),
            'task_extraction_bypass_enabled': self.task_extraction_bypass_checkbox.isChecked(),
            'capability_selection_bypass_enabled': self.capability_selection_bypass_checkbox.isChecked(),

            # Approval Settings
            'approval_global_mode': self.approval_global_mode_combo.currentText(),
            'python_execution_approval_enabled': self.python_execution_approval_checkbox.isChecked(),
            'python_execution_approval_mode': self.python_execution_approval_mode_combo.currentText(),
            'memory_approval_enabled': self.memory_approval_checkbox.isChecked(),

            # Execution Limits
            'max_reclassifications': self.max_reclassifications_spin.value(),
            'max_planning_attempts': self.max_planning_attempts_spin.value(),
            'max_step_retries': self.max_step_retries_spin.value(),
            'max_execution_time_seconds': self.max_execution_time_spin.value(),
            'max_concurrent_classifications': self.max_concurrent_classifications_spin.value(),

            # GUI Settings
            'use_persistent_conversations': self.use_persistent_conversations_checkbox.isChecked(),
            'conversation_storage_mode': self.conversation_storage_mode_combo.currentText(),
            'redirect_output_to_gui': self.redirect_output_to_gui_checkbox.isChecked(),
            'group_system_messages': self.group_system_messages_checkbox.isChecked(),
            'suppress_terminal_output': self.suppress_terminal_output_checkbox.isChecked(),
            'enable_routing_feedback': self.enable_routing_feedback_checkbox.isChecked(),

            # Development/Debug Settings
            'debug_mode': self.debug_mode_checkbox.isChecked(),
            'verbose_logging': self.verbose_logging_checkbox.isChecked(),
            'raise_raw_errors': self.raise_raw_errors_checkbox.isChecked(),
            'print_prompts': self.print_prompts_checkbox.isChecked(),
            'show_prompts': self.show_prompts_checkbox.isChecked(),
            'prompts_latest_only': self.prompts_latest_only_checkbox.isChecked(),

            # Memory Monitoring Settings
            'memory_monitor_enabled': self.memory_monitor_enabled_checkbox.isChecked(),
            'memory_warning_threshold_mb': self.memory_warning_threshold_spin.value(),
            'memory_critical_threshold_mb': self.memory_critical_threshold_spin.value(),
            'memory_check_interval_seconds': self.memory_check_interval_spin.value(),

            # Advanced Routing Settings (Phase 2.4)
            'enable_routing_cache': self.enable_routing_cache_checkbox.isChecked(),
            'cache_max_size': self.cache_max_size_spin.value(),
            'cache_ttl_seconds': float(self.cache_ttl_spin.value()),
            'cache_similarity_threshold': self.cache_similarity_spin.value(),
            'enable_advanced_invalidation': self.enable_advanced_invalidation_checkbox.isChecked(),
            'enable_adaptive_ttl': self.enable_adaptive_ttl_checkbox.isChecked(),
            'enable_probabilistic_expiration': self.enable_probabilistic_expiration_checkbox.isChecked(),
            'enable_event_driven_invalidation': self.enable_event_driven_invalidation_checkbox.isChecked(),
            'enable_semantic_analysis': self.enable_semantic_analysis_checkbox.isChecked(),
            'semantic_similarity_threshold': self.semantic_similarity_spin.value(),
            'topic_similarity_threshold': self.topic_similarity_spin.value(),
            'max_context_history': self.max_context_history_spin.value(),
            'orchestration_max_parallel': self.orchestration_max_parallel_spin.value(),
            'analytics_max_history': self.analytics_max_history_spin.value(),
        }
