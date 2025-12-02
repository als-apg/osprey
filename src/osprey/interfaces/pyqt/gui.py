#!/usr/bin/env python3
"""
Osprey Framework PyQt GUI

This GUI provides a graphical interface for the Osprey Framework, integrated
with the framework's Gateway, graph architecture, and configuration system.

Features:
- Framework-integrated conversation interface
- Real-time status updates during agent processing
- Conversation history management
- LLM interaction details and tool usage tracking
- System information display
- Settings management
"""

# GUI Version
__version__ = "0.0.1"

import asyncio
import sys
import os
import json
import uuid
from typing import Optional, Any, Dict
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSplitter, QStatusBar,
    QMenuBar, QAction, QMessageBox, QDialog, QFormLayout, QCheckBox,
    QSpinBox, QComboBox, QListWidget, QTabWidget, QListWidgetItem,
    QInputDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette, QTextOption, QTextCharFormat, QBrush

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver

from osprey.registry import initialize_registry, get_registry
from osprey.graph import create_graph, create_async_postgres_checkpointer
from osprey.infrastructure.gateway import Gateway
from osprey.utils.config import get_full_configuration, get_config_value
from osprey.utils.logger import get_logger
from osprey.interfaces.pyqt.project_discovery import (
    discover_projects,
    create_unified_config,
    create_unified_registry
)
from osprey.interfaces.pyqt.model_preferences import ModelPreferencesManager
from osprey.interfaces.pyqt.model_config_dialog import ModelConfigDialog
from osprey.interfaces.pyqt.help_dialog import show_help_dialog

logger = get_logger("pyqt_gui")


class AgentWorker(QThread):
    """Background worker thread for agent processing."""
    
    message_received = pyqtSignal(str)
    status_update = pyqtSignal(str, str)  # (message, component_type)
    error_occurred = pyqtSignal(str)
    processing_complete = pyqtSignal()
    llm_detail = pyqtSignal(str, str)  # (detail, event_type)
    tool_usage = pyqtSignal(str, str)  # (tool_name, reasoning)
    
    def __init__(self, gateway, graph, config, user_message):
        super().__init__()
        self.gateway = gateway
        self.graph = graph
        self.config = config
        self.user_message = user_message
        self._loop = None
    
    def run(self):
        """Execute agent processing in background thread."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            self.status_update.emit("Processing message...", "base")
            result = self._loop.run_until_complete(
                self.gateway.process_message(
                    self.user_message,
                    self.graph,
                    self.config
                )
            )
            
            if result.error:
                self.error_occurred.emit(f"Error: {result.error}")
                return
            
            if result.resume_command:
                self.status_update.emit("Resuming from interrupt...", "orchestrator")
                self._execute_graph(result.resume_command)
            elif result.agent_state:
                self.status_update.emit("Starting conversation...", "orchestrator")
                self._execute_graph(result.agent_state)
            else:
                self.message_received.emit("⚠️ No action required")
            
            self.processing_complete.emit()
            
        except Exception as e:
            logger.exception(f"Error in agent worker: {e}")
            self.error_occurred.emit(str(e))
        finally:
            if self._loop:
                self._loop.close()
    
    def _execute_graph(self, input_data):
        """Execute graph with streaming updates."""
        try:
            async def stream_execution():
                async for chunk in self.graph.astream(
                    input_data,
                    config=self.config,
                    stream_mode="custom"
                ):
                    event_type = chunk.get("event_type", "")
                    
                    if event_type == "status":
                        message = chunk.get("message", "")
                        component = chunk.get("component", "base")
                        
                        self.status_update.emit(message, component)
                        self.llm_detail.emit(message, "status")
            
            self._loop.run_until_complete(stream_execution())
            
            # Get final state and extract response
            state = self.graph.get_state(config=self.config)
            
            # Extract and emit execution step results for tool usage display
            self._extract_and_emit_execution_info(state.values)
            
            if state.interrupts:
                interrupt = state.interrupts[0]
                user_msg = interrupt.value.get('user_message', 'Input required')
                self.message_received.emit(f"\n⚠️ {user_msg}\n")
            else:
                messages = state.values.get("messages", [])
                if messages:
                    for msg in reversed(messages):
                        if hasattr(msg, 'content') and msg.content:
                            if not hasattr(msg, 'type') or msg.type != 'human':
                                self.message_received.emit(f"\n🤖 {msg.content}\n")
                                break
                else:
                    self.message_received.emit("\n✅ Execution completed\n")
        
        except Exception as e:
            logger.exception(f"Error executing graph: {e}")
            self.error_occurred.emit(str(e))
    
    def _extract_and_emit_execution_info(self, state_values):
        """Extract execution step results and emit as tool usage events."""
        try:
            execution_step_results = state_values.get("execution_step_results", {})
            
            if not execution_step_results:
                return
            
            # Sort by step_index to maintain execution order
            ordered_results = sorted(
                execution_step_results.items(),
                key=lambda x: x[1].get('step_index', 0)
            )
            
            # Emit tool usage for each executed step
            for step_key, step_data in ordered_results:
                capability = step_data.get('capability', 'unknown')
                task_objective = step_data.get('task_objective', 'No objective specified')
                success = step_data.get('success', False)
                execution_time = step_data.get('execution_time', 0)
                
                # Build detailed information
                info_parts = []
                
                # Status and objective
                status_icon = "✅" if success else "❌"
                info_parts.append(f"{status_icon} {task_objective}")
                
                # Execution time
                info_parts.append(f"⏱️  Execution time: {execution_time:.2f}s")
                
                # Combine all information
                detailed_info = "\n".join(info_parts)
                
                # Emit tool usage event with detailed information
                self.tool_usage.emit(capability, detailed_info)
                
        except Exception as e:
            logger.warning(f"Failed to extract execution info: {e}")


class OspreyGUI(QMainWindow):
    """Main Qt GUI window for Osprey Framework."""
    
    def __init__(self, config_path=None):
        super().__init__()
        self.config_path = config_path  # None means use framework defaults
        self.graph = None
        self.gateway = None
        self.thread_id = None
        self.base_config = None
        self.worker = None
        self._initialized = False
        self.discovered_projects = []  # Store discovered projects
        self.model_preferences = ModelPreferencesManager()  # Model preferences manager
        
        # Conversation history management
        self.conversations = {}
        self.current_conversation_id = None
        self.conversation_lock_file = None  # For multi-instance locking
        
        # Settings
        self.settings = {
            'planning_mode_enabled': False,
            'epics_writes_enabled': False,
            'approval_mode': 'disabled',
            'max_execution_time': 300,
            'use_persistent_conversations': True,  # Use SQLite checkpointer for persistence
        }
        
        # Color mapping for components
        self.component_colors = {
            'base': '#FFFFFF',
            'context': '#AFD7FF',
            'router': '#FF00FF',
            'orchestrator': '#00FFFF',
            'monitor': '#CD8500',
            'classifier': '#FFA07A',
            'task_extraction': '#D8BFD8',
            'error': '#FF0000',
            'gateway': '#FFA07A',
            'approval': '#FFA07A',
            'time_range_parsing': '#1E90FF',
            'memory': '#FFA07A',
            'python': '#FFA07A',
            'respond': '#D8BFD8',
            'clarify': '#D8BFD8',
        }
        
        try:
            self.setup_ui()
            logger.info("UI setup complete")
            QTimer.singleShot(100, self.initialize_framework)
            logger.info("Framework initialization scheduled")
        except Exception as e:
            logger.exception(f"Error during GUI initialization: {e}")
            raise
    
    def eventFilter(self, obj, event):
        """Event filter to handle Enter/Shift+Enter in input field."""
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent
        
        if obj == self.input_field and event.type() == QEvent.KeyPress:
            key_event = event
            # Check if Enter/Return was pressed without Shift
            if key_event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if key_event.modifiers() == Qt.NoModifier:
                    # Enter without Shift - send message
                    self.send_message()
                    return True  # Event handled
                elif key_event.modifiers() == Qt.ShiftModifier:
                    # Shift+Enter - insert newline (default behavior)
                    return False  # Let default handler insert newline
        
        # For all other events, use default handling
        return super().eventFilter(obj, event)
    
    def setup_ui(self):
        """Setup the main window UI."""
        self.setWindowTitle("Osprey Framework")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application-wide color scheme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 48))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(60, 60, 60))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        self.create_menu_bar()
        
        # Create tab widget for different views
        tab_widget = QTabWidget()
        tab_widget.setMovable(True)
        
        # Main conversation tab
        conversation_tab = self.create_conversation_tab()
        tab_widget.addTab(conversation_tab, "Conversation")
        
        # LLM Conversation Details tab
        llm_details_tab = self.create_llm_details_tab()
        tab_widget.addTab(llm_details_tab, "LLM Details")
        
        # LLM Tool Usage tab
        tool_usage_tab = self.create_tool_usage_tab()
        tab_widget.addTab(tool_usage_tab, "Tool Usage")
        
        # Discovered Projects tab
        projects_tab = self.create_projects_tab()
        tab_widget.addTab(projects_tab, "Discovered Projects")
        
        # System Information tab
        system_info_tab = self.create_system_info_tab()
        tab_widget.addTab(system_info_tab, "System Information")
        
        main_layout.addWidget(tab_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing...")
    
    def create_conversation_tab(self):
        """Create the main conversation interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Create splitter for conversation history, conversation, and info panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Conversation History
        history_widget = QWidget()
        history_layout = QVBoxLayout()
        history_widget.setLayout(history_layout)
        
        history_label = QLabel("Conversation History:")
        history_label.setStyleSheet("color: #FFD700; font-weight: bold;")
        history_layout.addWidget(history_label)
        
        self.conversation_list = QListWidget()
        self.conversation_list.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;")
        self.conversation_list.itemClicked.connect(self.switch_conversation)
        history_layout.addWidget(self.conversation_list)
        
        # Buttons for conversation management
        history_button_layout = QHBoxLayout()
        
        new_conv_btn = QPushButton("+")
        new_conv_btn.setMaximumWidth(40)
        new_conv_btn.clicked.connect(self.create_new_conversation)
        history_button_layout.addWidget(new_conv_btn)
        
        delete_conv_btn = QPushButton("🗑")
        delete_conv_btn.setMaximumWidth(40)
        delete_conv_btn.clicked.connect(self.delete_selected_conversation)
        history_button_layout.addWidget(delete_conv_btn)
        
        rename_conv_btn = QPushButton("✏")
        rename_conv_btn.setMaximumWidth(40)
        rename_conv_btn.clicked.connect(self.rename_selected_conversation)
        history_button_layout.addWidget(rename_conv_btn)
        
        history_layout.addLayout(history_button_layout)
        splitter.addWidget(history_widget)
        
        # Middle panel - Conversation
        conversation_widget = QWidget()
        conversation_layout = QVBoxLayout()
        conversation_widget.setLayout(conversation_layout)
        
        label = QLabel("Conversation:")
        label.setStyleSheet("color: #00FFFF; font-weight: bold;")
        conversation_layout.addWidget(label)
        
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setFont(QFont("Monospace", 10))
        self.conversation_display.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;")
        self.conversation_display.setHtml('<span style="color: #00FFFF;">Welcome to Osprey Framework</span><br><span style="color: #FFFFFF;">Initializing system...</span>')
        conversation_layout.addWidget(self.conversation_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Ask anything... (Press Enter to send, Shift+Enter for new line)")
        self.input_field.setWordWrapMode(QTextOption.WordWrap)
        self.input_field.setAcceptRichText(False)
        
        # Double the height - 4 lines instead of 2
        font_metrics = self.input_field.fontMetrics()
        line_height = font_metrics.lineSpacing()
        self.input_field.setFixedHeight(line_height * 4 + 10)
        self.input_field.setMaximumWidth(800)
        
        # Install event filter to handle Enter/Shift+Enter
        self.input_field.installEventFilter(self)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        self.new_conversation_button = QPushButton("New Conversation")
        self.new_conversation_button.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")
        self.new_conversation_button.clicked.connect(self.start_new_conversation)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.new_conversation_button)
        conversation_layout.addLayout(input_layout)
        
        splitter.addWidget(conversation_widget)
        
        # Right panel - Status log
        info_widget = QWidget()
        info_layout = QVBoxLayout()
        info_widget.setLayout(info_layout)
        
        label = QLabel("Status Log:")
        label.setStyleSheet("color: #00FF00; font-weight: bold;")
        info_layout.addWidget(label)
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setFont(QFont("Monospace", 9))
        self.status_log.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;")
        self.status_log.setHtml('<span style="color: #00FF00;">System starting...</span>')
        info_layout.addWidget(self.status_log)
        
        splitter.addWidget(info_widget)
        splitter.setSizes([250, 650, 400])
        
        layout.addWidget(splitter)
        return widget
    
    def create_system_info_tab(self):
        """Create the system information tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        label = QLabel("System Information:")
        label.setStyleSheet("color: #1E90FF; font-weight: bold;")
        layout.addWidget(label)
        self.session_info = QTextEdit()
        self.session_info.setReadOnly(True)
        self.session_info.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;")
        self.session_info.setAcceptRichText(True)
        self.session_info.setHtml('<pre style="color: #FFFFFF; font-family: monospace;"><span style="color: #00FFFF;">Initializing Osprey Framework...</span>\n\nPlease wait while the system initializes.</pre>')
        layout.addWidget(self.session_info)
        
        return widget
    
    def create_llm_details_tab(self):
        """Create the LLM conversation details tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        label = QLabel("LLM Conversation Details:")
        label.setStyleSheet("color: #FFD700; font-weight: bold;")
        layout.addWidget(label)
        self.llm_details_display = QTextEdit()
        self.llm_details_display.setReadOnly(True)
        self.llm_details_display.setFont(QFont("Monospace", 9))
        self.llm_details_display.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;")
        layout.addWidget(self.llm_details_display)
        
        clear_btn = QPushButton("Clear Details")
        clear_btn.clicked.connect(lambda: self.llm_details_display.clear())
        layout.addWidget(clear_btn)
        
        return widget
    
    def create_tool_usage_tab(self):
        """Create the LLM tool usage tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        label = QLabel("LLM Tool Usage and Reasoning:")
        label.setStyleSheet("color: #FF69B4; font-weight: bold;")
        layout.addWidget(label)
        self.tool_usage_display = QTextEdit()
        self.tool_usage_display.setReadOnly(True)
        self.tool_usage_display.setFont(QFont("Monospace", 9))
        self.tool_usage_display.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;")
        layout.addWidget(self.tool_usage_display)
        
        clear_btn = QPushButton("Clear Tool Usage")
        clear_btn.clicked.connect(lambda: self.tool_usage_display.clear())
        layout.addWidget(clear_btn)
        
        return widget
    
    def create_projects_tab(self):
        """Create the discovered projects tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
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
        self.projects_table.setColumnCount(5)
        self.projects_table.setHorizontalHeaderLabels(['Project Name', 'Path', 'Config File', 'Registry File', 'Models'])
        self.projects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.projects_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                gridline-color: #3F3F46;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2D2D30;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3F3F46;
                font-weight: bold;
            }
        """)
        self.projects_table.setAlternatingRowColors(True)
        self.projects_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.projects_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.projects_table)
        
        # Info label
        self.projects_info_label = QLabel("No projects discovered yet. Click Refresh to scan for projects.")
        self.projects_info_label.setStyleSheet("color: #808080; font-style: italic; padding: 10px;")
        layout.addWidget(self.projects_info_label)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        generate_btn = QPushButton("Generate Unified Config")
        generate_btn.clicked.connect(self.generate_unified_files)
        generate_btn.setStyleSheet("background-color: #0078D4; color: #FFFFFF;")
        button_layout.addWidget(generate_btn)
        
        load_btn = QPushButton("Load Unified Config")
        load_btn.clicked.connect(self.load_unified_config)
        load_btn.setStyleSheet("background-color: #107C10; color: #FFFFFF;")
        button_layout.addWidget(load_btn)
        
        layout.addLayout(button_layout)
        
        return widget
    
    def refresh_projects_display(self):
        """Refresh the discovered projects display."""
        try:
            self.add_status("Refreshing project list...", "base")
            
            # Discover projects
            base_dir = Path.cwd()
            self.discovered_projects = discover_projects(base_dir)
            
            # Update table
            self.projects_table.setRowCount(len(self.discovered_projects))
            
            for row, project in enumerate(self.discovered_projects):
                # Project name
                name_item = QTableWidgetItem(project['name'])
                name_item.setForeground(QColor("#00FFFF"))
                self.projects_table.setItem(row, 0, name_item)
                
                # Project path
                path_item = QTableWidgetItem(project['path'])
                path_item.setForeground(QColor("#FFFFFF"))
                self.projects_table.setItem(row, 1, path_item)
                
                # Config path
                config_path = Path(project['config_path']).name
                config_item = QTableWidgetItem(config_path)
                config_item.setForeground(QColor("#00FF00"))
                self.projects_table.setItem(row, 2, config_item)
                
                # Registry path
                registry_path = project.get('registry_path', 'N/A')
                if registry_path != 'N/A':
                    registry_path = Path(registry_path).name
                registry_item = QTableWidgetItem(registry_path)
                registry_item.setForeground(QColor("#FFD700") if registry_path != 'N/A' else QColor("#808080"))
                self.projects_table.setItem(row, 3, registry_item)
                
                # Models configuration column
                models_widget = QWidget()
                models_layout = QHBoxLayout(models_widget)
                models_layout.setContentsMargins(4, 4, 4, 4)
                
                config_btn = QPushButton("Configure")
                config_btn.setToolTip("Configure models for each processing step")
                config_btn.clicked.connect(lambda checked, p=project: self.configure_project_models(p))
                models_layout.addWidget(config_btn)
                
                # Show indicator if models are configured
                pref_count = self.model_preferences.get_preference_count(project['name'])
                if pref_count > 0:
                    indicator = QLabel(f"✓ ({pref_count})")
                    indicator.setToolTip(f"{pref_count} step(s) configured")
                    indicator.setStyleSheet("color: #00FF00;")
                    models_layout.addWidget(indicator)
                
                self.projects_table.setCellWidget(row, 4, models_widget)
            
            # Update info label
            if self.discovered_projects:
                self.projects_info_label.setText(
                    f"Found {len(self.discovered_projects)} project(s). "
                    f"Use 'Generate Unified Config' to combine them."
                )
                self.projects_info_label.setStyleSheet("color: #00FF00; padding: 10px;")
            else:
                self.projects_info_label.setText(
                    "No projects found. Projects must have a config.yml file in their root directory."
                )
                self.projects_info_label.setStyleSheet("color: #FFA500; padding: 10px;")
            
            self.add_status(f"Found {len(self.discovered_projects)} project(s)", "base")
            
        except Exception as e:
            logger.exception(f"Error refreshing projects: {e}")
            self.add_status(f"❌ Failed to refresh projects: {e}", "error")
            QMessageBox.warning(self, "Error", f"Failed to refresh projects:\n{e}")
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_conversation_action = QAction("New Conversation", self)
        new_conversation_action.triggered.connect(self.start_new_conversation)
        file_menu.addAction(new_conversation_action)
        
        clear_action = QAction("Clear Conversation", self)
        clear_action.triggered.connect(self.clear_conversation)
        file_menu.addAction(clear_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        settings_action = QAction("Framework Settings", self)
        settings_action.triggered.connect(self.show_settings)
        settings_menu.addAction(settings_action)
        
        # Multi-Project menu
        multi_project_menu = menubar.addMenu("Multi-Project")
        
        discover_action = QAction("Discover Projects", self)
        discover_action.triggered.connect(self.discover_projects)
        multi_project_menu.addAction(discover_action)
        
        generate_unified_action = QAction("Generate Unified Config", self)
        generate_unified_action.triggered.connect(self.generate_unified_files)
        multi_project_menu.addAction(generate_unified_action)
        
        load_unified_action = QAction("Load Unified Config", self)
        load_unified_action.triggered.connect(self.load_unified_config)
        multi_project_menu.addAction(load_unified_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        help_action = QAction("Help Documentation", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(help_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def configure_project_models(self, project_info):
        """Open dialog to configure models for a project."""
        dialog = ModelConfigDialog(project_info, self.model_preferences, self)
        if dialog.exec_() == QDialog.DialogCode.Accepted:
            # Refresh the projects table to show updated configuration
            self.refresh_projects_display()
            
            pref_count = self.model_preferences.get_preference_count(project_info['name'])
            if pref_count > 0:
                QMessageBox.information(
                    self,
                    "Configuration Saved",
                    f"Model configuration for {project_info['name']} has been saved.\n"
                    f"{pref_count} step(s) configured."
                )
            else:
                QMessageBox.information(
                    self,
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
        from osprey.utils.config import set_runtime_model_override, clear_runtime_model_overrides
        
        # Clear any existing overrides
        clear_runtime_model_overrides()
        
        # Apply preferences for this project
        preferences = self.model_preferences.get_all_preferences(project_name)
        for step, model_id in preferences.items():
            set_runtime_model_override(step, model_id)
            self.add_status(f"Using {model_id} for {step}", "base")
    
    def initialize_framework(self):
        """Initialize the Osprey framework components."""
        try:
            # ALWAYS discover projects and create unified config automatically
            self.add_status("Auto-discovering projects...", "base")
            
            base_dir = Path.cwd()
            self.discovered_projects = discover_projects(base_dir)
            
            # Update projects tab display
            QTimer.singleShot(200, self.refresh_projects_display)
            
            if self.discovered_projects:
                self.add_status(f"Found {len(self.discovered_projects)} project(s)", "base")
                
                # Automatically create unified files
                try:
                    config_path = create_unified_config(self.discovered_projects)
                    registry_path = create_unified_registry(self.discovered_projects)
                    
                    self.add_status(f"Created {Path(config_path).name}", "base")
                    self.add_status(f"Created {Path(registry_path).name}", "base")
                    
                    # Use the unified config
                    self.config_path = config_path
                    
                except Exception as e:
                    logger.error(f"Failed to create unified files: {e}")
                    self.add_status(f"⚠️ Failed to create unified files: {e}", "error")
            else:
                self.add_status("No projects found for auto-discovery", "base")
            
            self.add_status("Initializing Osprey framework...", "base")
            
            # Load conversation history from file first
            self.load_conversation_history()
            
            # Create initial conversation if none exist
            if not self.conversations:
                self.thread_id = f"gui_session_{uuid.uuid4().hex[:8]}"
                self.current_conversation_id = self.thread_id
                self.conversations[self.thread_id] = {
                    'name': 'Initial Conversation',
                    'messages': [],
                    'timestamp': datetime.now(),
                    'thread_id': self.thread_id
                }
            else:
                sorted_convs = sorted(
                    self.conversations.items(),
                    key=lambda x: x[1]['timestamp'],
                    reverse=True
                )
                self.current_conversation_id = sorted_convs[0][0]
                self.thread_id = self.current_conversation_id
            
            self.update_conversation_list()
            
            # Get configuration
            configurable = get_full_configuration(config_path=self.config_path).copy()
            configurable.update({
                "user_id": "gui_user",
                "thread_id": self.thread_id,
                "chat_id": "gui_chat",
                "session_id": self.thread_id,
                "interface_context": "pyqt_gui"
            })
            
            # Apply settings
            agent_control_defaults = configurable.get("agent_control_defaults", {})
            agent_control_defaults.update(self.settings)
            configurable["agent_control_defaults"] = agent_control_defaults
            
            recursion_limit = get_config_value("execution_limits.graph_recursion_limit")
            
            self.base_config = {
                "configurable": configurable,
                "recursion_limit": recursion_limit
            }
            
            # Initialize framework
            self.add_status(f"Initializing registry with config: {self.config_path}", "base")
            initialize_registry(config_path=self.config_path)
            registry = get_registry(config_path=self.config_path)
            
            # Create checkpointer based on settings
            checkpointer = self._create_checkpointer()
            
            self.graph = create_graph(registry, checkpointer=checkpointer)
            self.gateway = Gateway()
            
            # Load conversation history after graph is created
            if self.settings['use_persistent_conversations']:
                self._load_conversation_list()
            
            self.add_status("✅ Framework initialized successfully", "base")
            self.update_session_info()
            self.status_bar.showMessage("Osprey Framework ready")
            self._initialized = True
            
            # Load current conversation display
            QTimer.singleShot(100, self.load_current_conversation_display)
            
        except Exception as e:
            logger.exception(f"Failed to initialize framework: {e}")
            self.add_status(f"❌ Framework initialization failed: {e}", "error")
            QMessageBox.critical(self, "Initialization Error",
                               f"Failed to initialize framework:\n{e}")
    
    def update_session_info(self):
        """Update the session information display."""
        registry = get_registry()
        capabilities = registry.get_all_capabilities() if registry else []
        
        html_parts = ['<pre style="color: #FFFFFF; font-family: monospace;">']
        
        html_parts.extend([
            '<span style="color: #808080;">' + "=" * 80 + '</span>',
            '<span style="color: #00FFFF; font-weight: bold;">OSPREY FRAMEWORK SESSION</span>',
            '<span style="color: #808080;">' + "=" * 80 + '</span>',
            f'Thread ID: {self.thread_id}',
            f'Config Path: {self.config_path}',
            f'Capabilities: {len(capabilities)}',
            ''
        ])
        
        html_parts.append('</pre>')
        
        self.session_info.setHtml('\n'.join(html_parts))
    
    def add_status(self, message, component="base"):
        """Add a status message to the status log with color coding."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        color = self.component_colors.get(component, self.component_colors['base'])
        
        cursor = self.status_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        timestamp_format = QTextCharFormat()
        timestamp_format.setForeground(QBrush(QColor("#808080")))
        cursor.insertText(f"[{timestamp}] ", timestamp_format)
        
        message_format = QTextCharFormat()
        message_format.setForeground(QBrush(QColor(color)))
        cursor.insertText(f"{message}\n", message_format)
        
        self.status_log.setTextCursor(cursor)
        self.status_log.ensureCursorVisible()
    
    def send_message(self):
        """Send user message to the framework."""
        user_message = self.input_field.toPlainText().strip()
        if not user_message:
            return
        
        self._append_colored_message(f"👤 You: {user_message}", "#D8BFD8")
        
        if self.current_conversation_id and self.current_conversation_id in self.conversations:
            self.conversations[self.current_conversation_id]['messages'].append({
                'type': 'user',
                'content': user_message,
                'timestamp': datetime.now()
            })
            self.conversations[self.current_conversation_id]['timestamp'] = datetime.now()
            self.update_conversation_list()
            self.save_conversation_history()
        
        self.input_field.clear()
        
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status_bar.showMessage("Processing...")
        
        self.worker = AgentWorker(
            self.gateway,
            self.graph,
            self.base_config,
            user_message
        )
        self.worker.message_received.connect(self.on_message_received)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.processing_complete.connect(self.on_processing_complete)
        self.worker.llm_detail.connect(self.on_llm_detail)
        self.worker.tool_usage.connect(self.on_tool_usage)
        self.worker.start()
    
    def on_message_received(self, message):
        """Handle message received from agent."""
        if self.current_conversation_id and self.current_conversation_id in self.conversations:
            self.conversations[self.current_conversation_id]['messages'].append({
                'type': 'agent',
                'content': message,
                'timestamp': datetime.now()
            })
            self.conversations[self.current_conversation_id]['timestamp'] = datetime.now()
            self.update_conversation_list()
            self.save_conversation_history()
        
        if "✅" in message or "completed" in message.lower():
            self._append_colored_message(message, "#00FF00")
        else:
            self._append_colored_message(message, "#FFFFFF")
    
    def _append_colored_message(self, message, color):
        """Append a colored message to the conversation display."""
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        text_format = QTextCharFormat()
        text_format.setForeground(QBrush(QColor(color)))
        
        cursor.insertText(f"\n{message}\n", text_format)
        
        self.conversation_display.setTextCursor(cursor)
        self.conversation_display.ensureCursorVisible()
    
    def on_status_update(self, status, component="base"):
        """Handle status update from agent."""
        self.add_status(status, component)
        self.status_bar.showMessage(status)
    
    def on_error(self, error):
        """Handle error from agent."""
        self.conversation_display.append(f"\n❌ Error: {error}\n")
        self.add_status(f"Error: {error}", "error")
        QMessageBox.warning(self, "Processing Error", f"An error occurred:\n{error}")
    
    def on_llm_detail(self, detail, event_type="base"):
        """Handle LLM conversation detail with color coding."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        event_colors = {
            'llm_start': '#00FFFF',
            'llm_end': '#00FF00',
            'llm_stream': '#FFFF00',
            'classification': '#FFD700',
            'base': '#FFFFFF'
        }
        
        color = event_colors.get(event_type, event_colors['base'])
        
        cursor = self.llm_details_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        timestamp_format = QTextCharFormat()
        timestamp_format.setForeground(QBrush(QColor("#808080")))
        cursor.insertText(f"[{timestamp}] ", timestamp_format)
        
        tag_format = QTextCharFormat()
        tag_format.setForeground(QBrush(QColor(color)))
        cursor.insertText(f"[{event_type.upper()}] ", tag_format)
        
        detail_format = QTextCharFormat()
        detail_format.setForeground(QBrush(QColor("#FFFFFF")))
        cursor.insertText(f"{detail}\n", detail_format)
        
        self.llm_details_display.setTextCursor(cursor)
        self.llm_details_display.ensureCursorVisible()
    
    def on_tool_usage(self, tool_name, reasoning):
        """Handle tool usage information."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        cursor = self.tool_usage_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        cursor.insertText("\n")
        
        timestamp_format = QTextCharFormat()
        timestamp_format.setForeground(QBrush(QColor("#808080")))
        cursor.insertText(f"[{timestamp}] ", timestamp_format)
        
        label_format = QTextCharFormat()
        label_format.setForeground(QBrush(QColor("#FFA500")))
        cursor.insertText("Capability: ", label_format)
        
        tool_format = QTextCharFormat()
        tool_format.setForeground(QBrush(QColor("#00FFFF")))
        cursor.insertText(f"{tool_name}\n", tool_format)
        
        lines = reasoning.split('\n')
        for line in lines:
            if not line.strip():
                continue
            
            if line.startswith('✅'):
                line_format = QTextCharFormat()
                line_format.setForeground(QBrush(QColor("#00FF00")))
                cursor.insertText(f"{line}\n", line_format)
            elif line.startswith('❌'):
                line_format = QTextCharFormat()
                line_format.setForeground(QBrush(QColor("#FF6B6B")))
                cursor.insertText(f"{line}\n", line_format)
            elif line.startswith('⏱️'):
                line_format = QTextCharFormat()
                line_format.setForeground(QBrush(QColor("#FFD700")))
                cursor.insertText(f"{line}\n", line_format)
            else:
                line_format = QTextCharFormat()
                line_format.setForeground(QBrush(QColor("#FFFFFF")))
                cursor.insertText(f"{line}\n", line_format)
        
        separator_format = QTextCharFormat()
        separator_format.setForeground(QBrush(QColor("#404040")))
        cursor.insertText("=" * 80 + "\n", separator_format)
        
        self.tool_usage_display.setTextCursor(cursor)
        self.tool_usage_display.ensureCursorVisible()
    
    def on_processing_complete(self):
        """Handle completion of agent processing."""
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()
        self.status_bar.showMessage("Ready")
    
    def clear_conversation(self):
        """Clear the conversation display and history."""
        reply = QMessageBox.question(
            self,
            "Clear Conversation",
            "Are you sure you want to clear the conversation history?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.conversation_display.clear()
            
            if self.current_conversation_id and self.current_conversation_id in self.conversations:
                self.conversations[self.current_conversation_id]['messages'] = []
                self.conversations[self.current_conversation_id]['timestamp'] = datetime.now()
                self.save_conversation_history()
                self.update_conversation_list()
            
            self.add_status("Conversation history cleared", "base")
    
    def start_new_conversation(self):
        """Start a new conversation."""
        self.create_new_conversation()
    
    def create_new_conversation(self):
        """Create a new conversation."""
        try:
            old_thread_id = self.thread_id
            self.thread_id = f"gui_session_{uuid.uuid4().hex[:8]}"
            
            conv_number = len(self.conversations) + 1
            self.current_conversation_id = self.thread_id
            self.conversations[self.thread_id] = {
                'name': f'Conversation {conv_number}',
                'messages': [],
                'timestamp': datetime.now(),
                'thread_id': self.thread_id
            }
            
            self.save_conversation_history()
            
            if self.base_config:
                self.base_config["configurable"]["thread_id"] = self.thread_id
                self.base_config["configurable"]["session_id"] = self.thread_id
            
            self.conversation_display.clear()
            
            self._append_colored_message(
                "=" * 80 + "\n" +
                "🔄 NEW CONVERSATION STARTED\n" +
                "=" * 80 + "\n",
                "#00FFFF"
            )
            
            self.add_status(f"New conversation started (Thread: {self.thread_id})", "base")
            self.update_conversation_list()
            self.update_session_info()
            self.load_current_conversation_display()
            self.input_field.setFocus()
            
        except Exception as e:
            logger.exception(f"Error starting new conversation: {e}")
            self.add_status(f"❌ Failed to start new conversation: {e}", "error")
            QMessageBox.warning(self, "Error", f"Failed to start new conversation:\n{e}")
    
    def update_conversation_list(self):
        """Update the conversation history list."""
        self.conversation_list.clear()
        
        sorted_convs = sorted(
            self.conversations.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        for thread_id, conv_data in sorted_convs:
            name = conv_data['name']
            timestamp = conv_data['timestamp'].strftime("%Y-%m-%d %H:%M")
            msg_count = len(conv_data['messages'])
            
            is_current = (thread_id == self.current_conversation_id)
            
            prefix = "▶ " if is_current else "  "
            item_text = f"{prefix}{name}\n   {timestamp} • {msg_count} messages"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, thread_id)
            
            if is_current:
                item.setForeground(QColor("#00FF00"))
            else:
                item.setForeground(QColor("#FFD700"))
            
            self.conversation_list.addItem(item)
    
    def switch_conversation(self, item):
        """Switch to a different conversation."""
        thread_id = item.data(Qt.UserRole)
        if thread_id not in self.conversations:
            return
        
        try:
            # Update thread ID and config FIRST before loading messages
            self.current_conversation_id = thread_id
            self.thread_id = thread_id
            
            if self.base_config:
                self.base_config["configurable"]["thread_id"] = self.thread_id
                self.base_config["configurable"]["session_id"] = self.thread_id
            
            # Clear display
            self.conversation_display.clear()
            
            conv_data = self.conversations[thread_id]
            
            self._append_colored_message(
                "=" * 80 + "\n" +
                f"📂 SWITCHED TO: {conv_data['name']}\n" +
                "=" * 80 + "\n",
                "#00FFFF"
            )
            
            # Load conversation history from checkpointer or in-memory storage
            if self.settings['use_persistent_conversations'] and self.graph:
                # Load from SQLite checkpointer - this is the source of truth
                self._load_conversation_messages(thread_id)
            else:
                # Display from in-memory storage (fallback)
                for msg in conv_data.get('messages', []):
                    if msg['type'] == 'user':
                        self._append_colored_message(f"👤 You: {msg['content']}", "#D8BFD8")
                    else:
                        self._append_colored_message(msg['content'], "#FFFFFF")
            
            self.update_conversation_list()
            self.update_session_info()
            
            self.add_status(f"Switched to conversation: {conv_data['name']}", "base")
            
        except Exception as e:
            logger.exception(f"Error switching conversation: {e}")
            self.add_status(f"❌ Failed to switch conversation: {e}", "error")
            QMessageBox.warning(self, "Error", f"Failed to switch conversation:\n{e}")
    
    def delete_selected_conversation(self):
        """Delete the currently selected conversation."""
        current_item = self.conversation_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select a conversation to delete.")
            return
        
        thread_id = current_item.data(Qt.UserRole)
        if thread_id not in self.conversations:
            return
        
        conv_name = self.conversations[thread_id]['name']
        
        reply = QMessageBox.question(
            self,
            "Delete Conversation",
            f"Are you sure you want to delete '{conv_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if thread_id == self.current_conversation_id and len(self.conversations) == 1:
                QMessageBox.warning(self, "Cannot Delete", "Cannot delete the only conversation.")
                return
            
            if thread_id == self.current_conversation_id:
                for other_id in self.conversations:
                    if other_id != thread_id:
                        for i in range(self.conversation_list.count()):
                            item = self.conversation_list.item(i)
                            if item.data(Qt.UserRole) == other_id:
                                self.switch_conversation(item)
                                break
                        break
            
            del self.conversations[thread_id]
            self.update_conversation_list()
            self.save_conversation_history()
            self.add_status(f"Deleted conversation: {conv_name}", "base")
    
    def rename_selected_conversation(self):
        """Rename the currently selected conversation."""
        current_item = self.conversation_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select a conversation to rename.")
            return
        
        thread_id = current_item.data(Qt.UserRole)
        if thread_id not in self.conversations:
            return
        
        old_name = self.conversations[thread_id]['name']
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Conversation",
            "Enter new name:",
            text=old_name
        )
        
        if ok and new_name.strip():
            self.conversations[thread_id]['name'] = new_name.strip()
            self.update_conversation_list()
            self.save_conversation_history()
            self.add_status(f"Renamed conversation: '{old_name}' → '{new_name}'", "base")
    
    def _create_checkpointer(self):
        """Create checkpointer based on settings."""
        if self.settings['use_persistent_conversations']:
            # Check if PostgreSQL URI is configured
            postgres_uri = os.getenv('POSTGRESQL_URI')
            
            if postgres_uri:
                try:
                    # Use PostgreSQL checkpointer if URI is configured
                    checkpointer = create_async_postgres_checkpointer(postgres_uri)
                    logger.info(f"✅ Using PostgreSQL checkpointer for persistent conversations")
                    return checkpointer
                except Exception as e:
                    logger.warning(f"⚠️  Failed to create PostgreSQL checkpointer: {e}")
                    logger.info("📝 Falling back to in-memory checkpointer")
                    return MemorySaver()
            else:
                # Check if local PostgreSQL is running before attempting connection
                if self._is_postgres_running():
                    try:
                        # Attempt to connect to local PostgreSQL
                        local_uri = "postgresql://postgres:postgres@localhost:5432/osprey"
                        checkpointer = create_async_postgres_checkpointer(local_uri)
                        logger.info(f"✅ Using local PostgreSQL checkpointer for persistent conversations")
                        logger.info(f"💡 Database: {local_uri}")
                        return checkpointer
                    except Exception as e:
                        # Fall back to in-memory if connection fails
                        logger.warning(f"⚠️  PostgreSQL connection failed: {e}")
                        logger.info("📝 Using in-memory checkpointer (conversations will not persist between sessions)")
                        return MemorySaver()
                else:
                    # PostgreSQL not running - use in-memory
                    logger.info("📝 Using in-memory checkpointer (conversations will not persist between sessions)")
                    logger.info("💡 To enable persistent conversations:")
                    logger.info("   1. Install and start PostgreSQL")
                    logger.info("   2. Create database: createdb osprey")
                    logger.info("   3. Or set POSTGRESQL_URI environment variable")
                    return MemorySaver()
        else:
            logger.info("📝 Using in-memory checkpointer (persistence disabled in settings)")
            return MemorySaver()
    
    def _is_postgres_running(self, host='localhost', port=5432, timeout=1):
        """Check if PostgreSQL is running by attempting a socket connection."""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _acquire_conversation_lock(self, db_path):
        """Acquire a lock file to prevent conflicts with other GUI instances."""
        try:
            import fcntl
        except ImportError:
            # Windows doesn't have fcntl, skip locking
            logger.debug("File locking not available on this platform")
            return
        
        lock_file = db_path.parent / f".{db_path.name}.lock"
        try:
            self.conversation_lock_file = open(lock_file, 'w')
            fcntl.flock(self.conversation_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.conversation_lock_file.write(f"{os.getpid()}\n")
            self.conversation_lock_file.flush()
            logger.debug(f"Acquired conversation lock: {lock_file}")
        except (IOError, OSError) as e:
            logger.warning(f"Could not acquire exclusive lock (another GUI instance may be running): {e}")
            # Continue anyway - PostgreSQL handles concurrent access
            if self.conversation_lock_file:
                self.conversation_lock_file.close()
                self.conversation_lock_file = None
    
    def _release_conversation_lock(self):
        """Release the conversation lock file."""
        if self.conversation_lock_file:
            try:
                import fcntl
                fcntl.flock(self.conversation_lock_file.fileno(), fcntl.LOCK_UN)
                self.conversation_lock_file.close()
                logger.debug("Released conversation lock")
            except Exception as e:
                logger.warning(f"Error releasing conversation lock: {e}")
            finally:
                self.conversation_lock_file = None
    
    def _load_conversation_list(self):
        """Load list of conversations from checkpointer."""
        try:
            if not self.graph or not hasattr(self.graph, 'checkpointer'):
                logger.debug("No checkpointer available for loading conversations")
                return
            
            checkpointer = self.graph.checkpointer
            
            # Try to get all thread IDs from the checkpointer
            # Different checkpointer types have different methods
            thread_ids = set()
            
            # For MemorySaver checkpointer
            if hasattr(checkpointer, 'storage') and isinstance(checkpointer.storage, dict):
                # MemorySaver stores data as {(thread_id, checkpoint_ns): checkpoint}
                for key in checkpointer.storage.keys():
                    if isinstance(key, tuple) and len(key) >= 1:
                        thread_ids.add(key[0])
            
            # For PostgreSQL checkpointer (AsyncPostgresSaver)
            elif hasattr(checkpointer, 'conn'):
                # PostgreSQL checkpointer - we'd need to query the database
                # This is more complex and would require async operations
                logger.info("PostgreSQL checkpointer detected - loading conversations from database")
                # For now, we'll skip this and rely on on-demand loading
                return
            
            # Load conversations for each thread ID found
            loaded_count = 0
            for thread_id in thread_ids:
                try:
                    # Create config for this thread
                    config = {
                        "configurable": {
                            **self.base_config["configurable"],
                            "thread_id": thread_id,
                            "session_id": thread_id
                        },
                        "recursion_limit": self.base_config.get("recursion_limit", 100)
                    }
                    
                    # Get state from checkpointer
                    state = self.graph.get_state(config=config)
                    
                    if state and state.values:
                        messages = state.values.get('messages', [])
                        
                        if messages and len(messages) > 0:
                            # Create conversation entry
                            # Try to extract a meaningful name from the first user message
                            first_user_msg = None
                            for msg in messages:
                                if hasattr(msg, 'type') and msg.type == 'human':
                                    first_user_msg = msg.content[:50] if hasattr(msg, 'content') else None
                                    break
                            
                            conv_name = first_user_msg if first_user_msg else f"Conversation {len(self.conversations) + 1}"
                            
                            # Get timestamp from state metadata if available
                            timestamp = datetime.now()
                            if hasattr(state, 'created_at') and state.created_at:
                                try:
                                    timestamp = datetime.fromisoformat(state.created_at)
                                except:
                                    pass
                            
                            # Add to conversations dict
                            self.conversations[thread_id] = {
                                'name': conv_name,
                                'messages': [],  # We'll load these on-demand
                                'timestamp': timestamp,
                                'thread_id': thread_id
                            }
                            loaded_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to load conversation {thread_id}: {e}")
                    continue
            
            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} conversation(s) from checkpointer")
                self.update_conversation_list()
            else:
                logger.info("No existing conversations found in checkpointer")
            
        except Exception as e:
            logger.error(f"Failed to load conversation list: {e}")
    
    def _load_conversation_messages(self, thread_id):
        """Load and display messages for a conversation from checkpointer."""
        try:
            # Create a config with the specific thread_id to load from checkpointer
            config = {
                "configurable": {
                    **self.base_config["configurable"],
                    "thread_id": thread_id,
                    "session_id": thread_id
                },
                "recursion_limit": self.base_config.get("recursion_limit", 100)
            }
            
            # Get state from checkpointer for this specific thread
            state = self.graph.get_state(config=config)
            
            if state and state.values:
                messages = state.values.get('messages', [])
                
                if messages:
                    # Display messages
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            if hasattr(msg, 'type'):
                                if msg.type == 'human':
                                    self._append_colored_message(f"👤 You: {msg.content}", "#D8BFD8")
                                else:
                                    self._append_colored_message(f"🤖 {msg.content}", "#FFFFFF")
                            else:
                                # AIMessage or other
                                self._append_colored_message(f"🤖 {msg.content}", "#FFFFFF")
                    
                    logger.info(f"Loaded {len(messages)} messages from checkpointer for thread {thread_id}")
                else:
                    logger.debug(f"No messages in checkpointer for thread {thread_id}")
                    self._append_colored_message("No messages in this conversation yet.", "#808080")
            else:
                logger.debug(f"No state found in checkpointer for thread {thread_id}")
                self._append_colored_message("No messages in this conversation yet.", "#808080")
                
        except Exception as e:
            logger.error(f"Failed to load conversation messages: {e}")
            self._append_colored_message(f"⚠️ Could not load conversation history: {e}", "#FFA500")
    
    def save_conversation_history(self):
        """Save conversation metadata to persistent storage."""
        try:
            # Save conversation metadata (names, timestamps) to JSON
            # The actual messages are saved in the checkpointer
            conversations_file = Path.cwd() / '_agent_data' / 'conversations.json'
            conversations_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert datetime objects to strings for JSON serialization
            serializable_convs = {}
            for thread_id, conv_data in self.conversations.items():
                serializable_convs[thread_id] = {
                    'name': conv_data['name'],
                    'thread_id': conv_data['thread_id'],
                    'timestamp': conv_data['timestamp'].isoformat(),
                }
            
            with open(conversations_file, 'w') as f:
                json.dump(serializable_convs, f, indent=2)
            
            logger.debug(f"Saved conversation metadata to {conversations_file}")
        except Exception as e:
            logger.warning(f"Failed to save conversation metadata: {e}")
    
    def load_conversation_history(self):
        """Load conversation metadata from persistent storage."""
        try:
            conversations_file = Path.cwd() / '_agent_data' / 'conversations.json'
            
            if not conversations_file.exists():
                logger.debug("No conversation history file found")
                return
            
            with open(conversations_file, 'r') as f:
                serializable_convs = json.load(f)
            
            # Convert ISO format strings back to datetime objects
            for thread_id, conv_data in serializable_convs.items():
                self.conversations[thread_id] = {
                    'name': conv_data['name'],
                    'thread_id': conv_data['thread_id'],
                    'timestamp': datetime.fromisoformat(conv_data['timestamp']),
                    'messages': []  # Messages loaded from checkpointer on demand
                }
            
            logger.info(f"Loaded {len(self.conversations)} conversation(s) from history file")
            
        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
    
    def load_current_conversation_display(self):
        """Load the current conversation messages into the display."""
        if not self.current_conversation_id or self.current_conversation_id not in self.conversations:
            return
        
        try:
            conv_data = self.conversations[self.current_conversation_id]
            
            self.conversation_display.clear()
            
            self._append_colored_message(
                "=" * 80 + "\n" +
                f"📂 {conv_data['name']}\n" +
                "=" * 80 + "\n",
                "#00FFFF"
            )
            
            # Load from checkpointer if persistent conversations are enabled
            if self.settings['use_persistent_conversations'] and self.graph:
                # Load from SQLite checkpointer - this is the source of truth
                self._load_conversation_messages(self.current_conversation_id)
            elif conv_data['messages']:
                # Load from in-memory storage (fallback)
                for msg in conv_data['messages']:
                    if msg['type'] == 'user':
                        self._append_colored_message(f"👤 You: {msg['content']}", "#D8BFD8")
                    else:
                        self._append_colored_message(msg['content'], "#FFFFFF")
                
                self.add_status(f"Loaded conversation: {conv_data['name']} ({len(conv_data['messages'])} messages)", "base")
            else:
                self._append_colored_message(
                    "Welcome! Start a conversation by typing a message below.",
                    "#00FFFF"
                )
        except Exception as e:
            logger.error(f"Failed to load conversation display: {e}")
            self.add_status(f"⚠️ Failed to load conversation display: {e}", "error")
    
    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self, "Framework Settings", self.settings)
        if dialog.exec_() == QDialog.Accepted:
            self.settings = dialog.get_settings()
            
            # Update base config with new settings
            if self.base_config:
                agent_control_defaults = self.base_config["configurable"].get("agent_control_defaults", {})
                agent_control_defaults.update(self.settings)
                self.base_config["configurable"]["agent_control_defaults"] = agent_control_defaults
            
            self.update_session_info()
            self.add_status("Settings updated", "base")
    
    def discover_projects(self):
        """Discover osprey projects in subdirectories."""
        try:
            from pathlib import Path
            
            self.add_status("Discovering projects...", "base")
            
            # Discover projects in current directory
            base_dir = Path.cwd()
            projects = discover_projects(base_dir)
            
            if not projects:
                QMessageBox.information(
                    self,
                    "No Projects Found",
                    "No osprey projects found in subdirectories.\n\n"
                    "Projects must have a config.yml file."
                )
                self.add_status("No projects found", "base")
                return
            
            # Show discovered projects
            project_list = "\n".join([f"  • {p['name']}" for p in projects])
            QMessageBox.information(
                self,
                "Projects Discovered",
                f"Found {len(projects)} project(s):\n\n{project_list}\n\n"
                f"Use 'Generate Unified Config' to create unified configuration."
            )
            
            self.add_status(f"Discovered {len(projects)} projects", "base")
            
        except Exception as e:
            logger.exception(f"Error discovering projects: {e}")
            QMessageBox.critical(
                self,
                "Discovery Error",
                f"Failed to discover projects:\n{e}"
            )
    
    def generate_unified_files(self):
        """Generate unified config and registry files."""
        try:
            from pathlib import Path
            
            self.add_status("Generating unified configuration...", "base")
            
            # Discover projects
            base_dir = Path.cwd()
            projects = discover_projects(base_dir)
            
            if not projects:
                QMessageBox.warning(
                    self,
                    "No Projects Found",
                    "No osprey projects found in subdirectories.\n\n"
                    "Cannot generate unified configuration."
                )
                return
            
            # Confirm generation
            project_list = "\n".join([f"  • {p['name']}" for p in projects])
            reply = QMessageBox.question(
                self,
                "Generate Unified Configuration",
                f"Generate unified configuration from {len(projects)} project(s)?\n\n"
                f"{project_list}\n\n"
                f"This will create:\n"
                f"  • unified_config.yml\n"
                f"  • unified_registry.py\n\n"
                f"in the project root directory",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Generate unified config
            config_path = create_unified_config(projects)
            self.add_status(f"Created unified config: {config_path}", "base")
            
            # Generate unified registry
            registry_path = create_unified_registry(projects)
            self.add_status(f"Created unified registry: {registry_path}", "base")
            
            # Success message
            QMessageBox.information(
                self,
                "Success",
                f"Unified configuration generated successfully!\n\n"
                f"Config: {Path(config_path).name}\n"
                f"Registry: {Path(registry_path).name}\n\n"
                f"Use 'Load Unified Config' to initialize with these files."
            )
            
        except Exception as e:
            logger.exception(f"Error generating unified files: {e}")
            QMessageBox.critical(
                self,
                "Generation Error",
                f"Failed to generate unified files:\n{e}"
            )
    
    def load_unified_config(self):
        """Load the unified configuration and reinitialize."""
        try:
            from pathlib import Path
            
            # Check if unified files exist in the root directory
            root_dir = Path.cwd()
            unified_config = root_dir / "unified_config.yml"
            unified_registry = root_dir / "unified_registry.py"
            
            if not unified_config.exists():
                QMessageBox.warning(
                    self,
                    "Unified Config Not Found",
                    "Unified configuration file not found in project root.\n\n"
                    "Use 'Generate Unified Config' first."
                )
                return
            
            if not unified_registry.exists():
                QMessageBox.warning(
                    self,
                    "Unified Registry Not Found",
                    "Unified registry file not found in project root.\n\n"
                    "Use 'Generate Unified Config' first."
                )
                return
            
            # Confirm reload
            reply = QMessageBox.question(
                self,
                "Load Unified Configuration",
                "Load unified configuration and reinitialize?\n\n"
                "This will:\n"
                "  • Reset the current session\n"
                "  • Load all discovered projects\n"
                "  • Combine all capabilities\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                return
            
            self.add_status("Loading unified configuration...", "base")
            
            # Update config path and reinitialize
            self.config_path = str(unified_config)
            
            # Reset registry to clear old state
            from osprey.registry import reset_registry
            reset_registry()
            
            # Reinitialize framework with unified config
            self.initialize_framework()
            
            QMessageBox.information(
                self,
                "Success",
                "Unified configuration loaded successfully!\n\n"
                "All project capabilities are now available."
            )
            
        except Exception as e:
            logger.exception(f"Error loading unified config: {e}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load unified configuration:\n{e}"
            )
    
    def show_help_dialog(self):
        """Show the help dialog."""
        show_help_dialog(self)
    
    def show_about_dialog(self):
        """Show the about dialog as a non-modal window."""
        import platform
        from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
        from PyQt5.QtWidgets import QTextBrowser
        from osprey.interfaces.pyqt.version_info import get_all_versions
        from osprey.interfaces.pyqt import gui
        
        # Get GUI version
        gui_version = gui.__version__
        
        # Get comprehensive version information
        versions = get_all_versions()
        osprey_version = versions['osprey']
        python_version = versions['python']
        
        qt_version = QT_VERSION_STR
        pyqt_version = PYQT_VERSION_STR
        os_info = f"{platform.system()} {platform.release()}"
        
        # Build core dependencies HTML
        core_deps_html = ""
        for pkg, ver in versions['core'].items():
            core_deps_html += f"<li>{pkg}: {ver}</li>\n"
        
        # Build optional dependencies HTML (only installed ones)
        optional_deps_html = ""
        installed_optional = {pkg: ver for pkg, ver in versions['optional'].items()
                            if ver != "Not installed"}
        if installed_optional:
            for pkg, ver in installed_optional.items():
                optional_deps_html += f"<li>{pkg}: {ver}</li>\n"
        
        # Create a non-modal dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("About Osprey Framework")
        dialog.setModal(False)  # Non-modal - can be moved and interact with main window
        dialog.resize(600, 600)
        
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        
        # Use QTextBrowser for rich text display
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(f"""
            <p><b>Osprey Framework Version:</b> {osprey_version}</p>
            <p><b>PyQt GUI Interface Version:</b> {gui_version}</p>
            <hr>
            <p><b>System Information:</b></p>
            <ul>
            <li>Python: {python_version}</li>
            <li>Qt: {qt_version}</li>
            <li>PyQt: {pyqt_version}</li>
            <li>OS: {os_info}</li>
            </ul>
            <hr>
            <p><b>Core Dependencies:</b></p>
            <ul>
            {core_deps_html}
            </ul>
            {f'<hr><p><b>Optional Dependencies (Installed):</b></p><ul>{optional_deps_html}</ul>' if optional_deps_html else ''}
        """)
        layout.addWidget(text_browser)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)
        
        # Show non-modal dialog
        dialog.show()


class SettingsDialog(QDialog):
    """Dialog for configuring framework settings."""
    
    def __init__(self, parent, title, current_settings):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.current_settings = current_settings.copy()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the settings dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        form_layout = QFormLayout()
        
        # Planning Mode
        self.planning_mode_checkbox = QCheckBox()
        self.planning_mode_checkbox.setChecked(self.current_settings.get('planning_mode_enabled', False))
        form_layout.addRow("Planning Mode:", self.planning_mode_checkbox)
        
        # EPICS Writes
        self.epics_writes_checkbox = QCheckBox()
        self.epics_writes_checkbox.setChecked(self.current_settings.get('epics_writes_enabled', False))
        form_layout.addRow("EPICS Writes:", self.epics_writes_checkbox)
        
        # Approval Mode
        self.approval_mode_combo = QComboBox()
        self.approval_mode_combo.addItems(['disabled', 'selective', 'all_capabilities'])
        current_approval = self.current_settings.get('approval_mode', 'disabled')
        index = self.approval_mode_combo.findText(current_approval)
        if index >= 0:
            self.approval_mode_combo.setCurrentIndex(index)
        form_layout.addRow("Approval Mode:", self.approval_mode_combo)
        
        # Max Execution Time
        self.max_execution_time_spin = QSpinBox()
        self.max_execution_time_spin.setRange(10, 3600)
        self.max_execution_time_spin.setValue(self.current_settings.get('max_execution_time', 300))
        self.max_execution_time_spin.setSuffix(" seconds")
        form_layout.addRow("Max Execution Time:", self.max_execution_time_spin)
        
        # Conversation Persistence
        self.use_persistent_conversations_checkbox = QCheckBox()
        self.use_persistent_conversations_checkbox.setChecked(
            self.current_settings.get('use_persistent_conversations', True)
        )
        form_layout.addRow("Save Conversation History:", self.use_persistent_conversations_checkbox)
        
        layout.addLayout(form_layout)
        
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
            'planning_mode_enabled': self.planning_mode_checkbox.isChecked(),
            'epics_writes_enabled': self.epics_writes_checkbox.isChecked(),
            'approval_mode': self.approval_mode_combo.currentText(),
            'max_execution_time': self.max_execution_time_spin.value(),
            'use_persistent_conversations': self.use_persistent_conversations_checkbox.isChecked(),
        }


def main(config_path=None):
    """Main entry point for the PyQt GUI application.
    
    Args:
        config_path: Path to config file. If None, framework will search for config.yml
                    in current directory or use defaults.
    """
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Osprey Framework")
    
    window = OspreyGUI(config_path=config_path)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()