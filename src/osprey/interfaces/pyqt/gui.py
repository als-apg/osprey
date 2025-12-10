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
__version__ = "0.1.0"

import asyncio
import sys
import os
import json
import uuid
from typing import Optional, Any, Dict, List
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSplitter, QStatusBar,
    QMenuBar, QAction, QMessageBox, QDialog, QFormLayout, QCheckBox,
    QSpinBox, QComboBox, QListWidget, QTabWidget, QListWidgetItem,
    QInputDialog, QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QGroupBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot, Q_ARG
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
from osprey.interfaces.pyqt.gui_utils import create_dark_palette, load_config_safe
from osprey.interfaces.pyqt.collapsible_widget import MessageGroupWidget
from osprey.interfaces.pyqt.project_manager import ProjectManager
from osprey.interfaces.pyqt.capability_registry import CapabilityRegistry
from osprey.interfaces.pyqt.multi_project_router import MultiProjectRouter

logger = get_logger("pyqt_gui")


class GUIOutputSignal(QThread):
    """Helper class to emit GUI output signals from any thread."""
    output_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        # Don't start the thread - we just use it as a signal container
        
    def emit_output(self, message: str):
        """Thread-safe method to emit output signal."""
        self.output_signal.emit(message)


class AgentWorker(QThread):
    """Background worker thread for agent processing."""
    
    message_received = pyqtSignal(str)
    status_update = pyqtSignal(str, str, dict)  # (message, component_type, model_info)
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
            
            self.status_update.emit("Processing message...", "base", {})
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
                self.status_update.emit("Resuming from interrupt...", "orchestrator", {})
                self._execute_graph(result.resume_command)
            elif result.agent_state:
                self.status_update.emit("Starting conversation...", "orchestrator", {})
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
                        
                        # Extract model info if available
                        model_info = {}
                        if "model_provider" in chunk:
                            model_info["model_provider"] = chunk.get("model_provider")
                        if "model_id" in chunk:
                            model_info["model_id"] = chunk.get("model_id")
                        
                        self.status_update.emit(message, component, model_info)
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
        # If no config path provided, use the GUI-specific config file
        if config_path is None:
            import os
            gui_config = Path(__file__).parent / "gui_config.yml"
            if gui_config.exists():
                config_path = str(gui_config)
        self.config_path = config_path
        self.graph = None
        self.gateway = None
        self.thread_id = None
        self.base_config = None
        self.worker = None
        self._initialized = False
        self.discovered_projects = []  # Store discovered projects (cached)
        self._projects_cache_valid = False  # Track if cache is valid
        self.model_preferences = ModelPreferencesManager()  # Model preferences manager
        
        # Phase 1 Components - Multi-Project Support
        self.project_manager = ProjectManager()
        self.capability_registry = CapabilityRegistry()
        # Initialize router - will be configured with settings after UI setup
        self.router = None
        
        # Track current routing decision for feedback
        self.current_routing_decision = None
        self.current_query = None
        self._waiting_for_correction = False
        self._correction_options = []
        self._agent_processing = False  # Track if agent is currently processing
        self._queued_message = None  # Store one queued message to process after completion
        
        # Create signal emitter for thread-safe GUI output
        self.gui_output_signal = GUIOutputSignal()
        self.gui_output_signal.output_signal.connect(self.append_to_system_info)
        
        # CRITICAL: Set up GUI output redirection IMMEDIATELY before ANY other initialization
        # This must happen BEFORE setup_ui() to capture all logging from the start
        # Note: suppress_terminal setting will be applied after settings are loaded
        from osprey.utils.logger import set_gui_output_callback
        # Use the signal emitter's method for thread-safe GUI updates
        # Start with suppress_terminal=False to show messages in both places initially
        set_gui_output_callback(self.gui_output_signal.emit_output, suppress_terminal=False)
        
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
            'conversation_storage_mode': 'json',  # 'json' or 'postgresql' - where to store conversation messages
            'redirect_output_to_gui': True,  # Redirect terminal output to GUI System Information tab
            'suppress_terminal_output': False,  # Suppress terminal output when GUI is active
            'group_system_messages': True,  # Group system messages by type in collapsible sections
            'enable_routing_feedback': True,  # Enable user feedback collection for routing decisions
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
            # GUI output redirection is now set up in __init__ BEFORE this point
            # to capture all early logging messages
            
            self.setup_ui()
            logger.info("UI setup complete")
            logger.info("GUI output redirection enabled")
            
            # Router will be initialized after framework initialization
            # (moved to initialize_framework method)
            
            QTimer.singleShot(100, self.initialize_framework)
            logger.info("Framework initialization scheduled")
        except Exception as e:
            logger.exception(f"Error during GUI initialization: {e}")
            raise
    
    def closeEvent(self, event):
        """Handle window close event - cleanup GUI output redirection."""
        try:
            # Disable GUI output redirection when closing and re-enable terminal output
            from osprey.utils.logger import set_gui_output_callback
            set_gui_output_callback(None)  # This will re-enable terminal output
            logger.info("GUI output redirection disabled")
        except Exception as e:
            logger.warning(f"Error disabling GUI output redirection: {e}")
        
        # Call parent close event
        super().closeEvent(event)
    def _initialize_router(self):
        """Initialize or reinitialize router with current settings."""
        try:
            logger.info("Initializing MultiProjectRouter with current settings")
            
            self.router = MultiProjectRouter(
                self.capability_registry,
                # Cache settings
                enable_cache=self.settings.get('enable_routing_cache', True),
                cache_max_size=self.settings.get('cache_max_size', 100),
                cache_ttl_seconds=self.settings.get('cache_ttl_seconds', 3600.0),
                cache_similarity_threshold=self.settings.get('cache_similarity_threshold', 0.85),
                # Advanced invalidation settings
                enable_advanced_invalidation=self.settings.get('enable_advanced_invalidation', True),
                enable_adaptive_ttl=self.settings.get('enable_adaptive_ttl', True),
                enable_probabilistic_expiration=self.settings.get('enable_probabilistic_expiration', True),
                enable_event_driven_invalidation=self.settings.get('enable_event_driven_invalidation', True),
                # Conversation context settings
                enable_conversation_context=True,
                context_max_history=self.settings.get('max_context_history', 20),
                # Orchestration settings
                enable_orchestration=True,
                orchestration_max_parallel=self.settings.get('orchestration_max_parallel', 3),
                # Analytics settings
                enable_analytics=True,
                analytics_max_history=self.settings.get('analytics_max_history', 1000),
                # Feedback settings
                enable_feedback=self.settings.get('enable_routing_feedback', True),
                feedback_max_history=1000
            )
            
            logger.info("MultiProjectRouter initialized successfully with all Phase 2.4 settings")
            
        except Exception as e:
            logger.exception(f"Failed to initialize router: {e}")
            # Create a basic router as fallback
            self.router = MultiProjectRouter(self.capability_registry)
    
    
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
        self.setWindowTitle("Osprey-Framework Multi-Agent Interface")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application-wide color scheme using shared utility
        self.setPalette(create_dark_palette())
        
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
        
        # Analytics Dashboard tab
        analytics_tab = self.create_analytics_tab()
        tab_widget.addTab(analytics_tab, "📊 Analytics")
        
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
        
        # Left panel - Conversation History + Project Control
        history_widget = QWidget()
        history_layout = QVBoxLayout()
        history_widget.setLayout(history_layout)
        
        # Project Control Panel (NEW)
        project_control_panel = self._build_project_control_panel()
        history_layout.addWidget(project_control_panel)
        
        # Separator
        separator = QLabel()
        separator.setStyleSheet("border-bottom: 1px solid #3F3F46; margin: 5px 0;")
        history_layout.addWidget(separator)
        
        history_label = QLabel("Conversation History:")
        history_label.setStyleSheet("color: #FFD700; font-weight: bold;")
        history_layout.addWidget(history_label)
        
        self.conversation_list = QListWidget()
        self.conversation_list.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;")
        self.conversation_list.setSelectionMode(QListWidget.ExtendedSelection)  # Enable multi-selection
        self.conversation_list.itemClicked.connect(self.switch_conversation)
        history_layout.addWidget(self.conversation_list)
        
        # Buttons for conversation management
        history_button_layout = QHBoxLayout()
        
        new_conv_btn = QPushButton("Add")
        new_conv_btn.setMaximumWidth(50)
        new_conv_btn.setToolTip("Create new conversation")
        new_conv_btn.clicked.connect(self.create_new_conversation)
        history_button_layout.addWidget(new_conv_btn)
        
        delete_conv_btn = QPushButton("Del")
        delete_conv_btn.setMaximumWidth(50)
        delete_conv_btn.setToolTip("Delete conversation")
        delete_conv_btn.clicked.connect(self.delete_selected_conversation)
        history_button_layout.addWidget(delete_conv_btn)
        
        rename_conv_btn = QPushButton("Edit")
        rename_conv_btn.setMaximumWidth(50)
        rename_conv_btn.setToolTip("Rename conversation")
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
        
        # Complete the tab setup with status log panel
        self._complete_conversation_tab_setup(layout, splitter)
        
        return widget
    
    def _build_project_control_panel(self):
        """Build UI panel for project selection and control."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title_label = QLabel("Project Control:")
        title_label.setStyleSheet("color: #00FFFF; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Project selector dropdown
        selector_label = QLabel("Active Project:")
        selector_label.setStyleSheet("color: #FFFFFF; font-size: 10px;")
        layout.addWidget(selector_label)
        
        self.project_selector = QComboBox()
        self.project_selector.setStyleSheet("""
            QComboBox {
                background-color: #2D2D30;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #FFFFFF;
            }
            QComboBox QAbstractItemView {
                background-color: #2D2D30;
                color: #FFFFFF;
                selection-background-color: #0078D4;
                selection-color: #FFFFFF;
                border: 1px solid #3F3F46;
            }
        """)
        
        # Add "Automatic Routing" as first option
        self.project_selector.addItem("🤖 Automatic Routing", "auto")
        
        # Will be populated with projects after initialization
        self.project_selector.currentIndexChanged.connect(self.on_project_selected)
        layout.addWidget(self.project_selector)
        
        # Routing mode indicator
        self.routing_mode_label = QLabel("Mode: Automatic")
        self.routing_mode_label.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")
        layout.addWidget(self.routing_mode_label)
        
        # Routing explanation (initially hidden)
        self.routing_explanation_label = QLabel("")
        self.routing_explanation_label.setStyleSheet("color: #808080; font-size: 9px; font-style: italic;")
        self.routing_explanation_label.setWordWrap(True)
        self.routing_explanation_label.setVisible(False)
        layout.addWidget(self.routing_explanation_label)
        
        # Cache statistics (initially hidden)
        self.cache_stats_label = QLabel("")
        self.cache_stats_label.setStyleSheet("color: #00FF00; font-size: 9px; font-family: monospace;")
        self.cache_stats_label.setWordWrap(True)
        self.cache_stats_label.setVisible(False)
        layout.addWidget(self.cache_stats_label)
        
        # Conversation context summary (initially hidden)
        self.context_summary_label = QLabel("")
        self.context_summary_label.setStyleSheet("color: #87CEEB; font-size: 9px; font-family: monospace;")
        self.context_summary_label.setWordWrap(True)
        self.context_summary_label.setVisible(False)
        layout.addWidget(self.context_summary_label)
        
        # Cache control buttons
        cache_button_layout = QHBoxLayout()
        
        self.clear_cache_button = QPushButton("🗑️ Clear Cache")
        self.clear_cache_button.setStyleSheet("background-color: #4A5568; color: #FFFFFF; font-size: 9px;")
        self.clear_cache_button.setMaximumHeight(25)
        self.clear_cache_button.clicked.connect(self.clear_routing_cache)
        self.clear_cache_button.setToolTip("Clear routing cache to force fresh routing decisions")
        cache_button_layout.addWidget(self.clear_cache_button)
        
        self.show_cache_stats_button = QPushButton("📊 Stats")
        self.show_cache_stats_button.setStyleSheet("background-color: #4A5568; color: #FFFFFF; font-size: 9px;")
        self.show_cache_stats_button.setMaximumHeight(25)
        self.show_cache_stats_button.setCheckable(True)
        self.show_cache_stats_button.clicked.connect(self.toggle_cache_stats)
        self.show_cache_stats_button.setToolTip("Show/hide cache statistics")
        cache_button_layout.addWidget(self.show_cache_stats_button)
        
        layout.addLayout(cache_button_layout)
        
        # Conversation context control buttons
        context_button_layout = QHBoxLayout()
        
        self.clear_context_button = QPushButton("🔄 Clear Context")
        self.clear_context_button.setStyleSheet("background-color: #4A5568; color: #FFFFFF; font-size: 9px;")
        self.clear_context_button.setMaximumHeight(25)
        self.clear_context_button.clicked.connect(self.clear_conversation_context)
        self.clear_context_button.setToolTip("Clear conversation context to start fresh topic detection")
        context_button_layout.addWidget(self.clear_context_button)
        
        self.show_context_button = QPushButton("💬 Context")
        self.show_context_button.setStyleSheet("background-color: #4A5568; color: #FFFFFF; font-size: 9px;")
        self.show_context_button.setMaximumHeight(25)
        self.show_context_button.setCheckable(True)
        self.show_context_button.clicked.connect(self.toggle_context_display)
        self.show_context_button.setToolTip("Show/hide conversation context")
        context_button_layout.addWidget(self.show_context_button)
        
        layout.addLayout(context_button_layout)
        
        return panel
    
    def _update_project_selector(self):
        """Update the project selector dropdown with current projects."""
        # Save current selection
        current_data = self.project_selector.currentData()
        
        # Clear all items except "Automatic Routing"
        while self.project_selector.count() > 1:
            self.project_selector.removeItem(1)
        
        # Add enabled projects
        enabled_projects = self.project_manager.get_enabled_projects()
        for project_context in enabled_projects:
            project_name = project_context.metadata.name
            self.project_selector.addItem(f"📁 {project_name}", project_name)
        
        # Restore selection if possible
        if current_data:
            index = self.project_selector.findData(current_data)
            if index >= 0:
                self.project_selector.setCurrentIndex(index)
    
    def on_project_selected(self, index):
        """Handle project selection from dropdown."""
        project_data = self.project_selector.currentData()
        
        if project_data == "auto":
            # Switch to automatic mode
            self.router.set_automatic_mode()
            self.routing_mode_label.setText("Mode: Automatic")
            self.routing_mode_label.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")
            self.routing_explanation_label.setText("Queries will be automatically routed to the best project")
            self.routing_explanation_label.setVisible(True)
            self.add_status("Switched to automatic routing mode", "base")
        else:
            # Switch to manual mode
            self.router.set_manual_mode(project_data)
            self.routing_mode_label.setText(f"Mode: Manual")
            self.routing_mode_label.setStyleSheet("color: #FFD700; font-size: 10px; font-weight: bold;")
            self.routing_explanation_label.setText(f"All queries will use: {project_data}")
            self.routing_explanation_label.setVisible(True)
            self.add_status(f"Switched to manual mode: {project_data}", "base")
    
    def clear_routing_cache(self):
        """Clear the routing cache."""
        try:
            self.router.clear_cache()
            self.add_status("Routing cache cleared", "base")
            self._update_cache_statistics()
            QMessageBox.information(
                self,
                "Cache Cleared",
                "Routing cache has been cleared.\nNext queries will use fresh routing decisions."
            )
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            self.add_status(f"❌ Failed to clear cache: {e}", "error")
    
    def toggle_cache_stats(self):
        """Toggle cache statistics display."""
        is_visible = self.show_cache_stats_button.isChecked()
        self.cache_stats_label.setVisible(is_visible)
        
        if is_visible:
            self._update_cache_statistics()
            self.show_cache_stats_button.setText("📊 Hide Stats")
        else:
            self.show_cache_stats_button.setText("📊 Stats")
    
    def _update_cache_statistics(self):
        """Update cache statistics display."""
        try:
            stats = self.router.get_cache_statistics()
            
            if not stats:
                self.cache_stats_label.setText("Cache: Disabled")
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
            
            self.cache_stats_label.setText(stats_text)
            
        except Exception as e:
            logger.error(f"Failed to update cache statistics: {e}")
            self.cache_stats_label.setText(f"Cache Stats: Error - {e}")
    
    def clear_conversation_context(self):
        """Clear the conversation context."""
        try:
            self.router.clear_conversation_context()
            self.add_status("Conversation context cleared", "base")
            self._update_context_display()
            QMessageBox.information(
                self,
                "Context Cleared",
                "Conversation context has been cleared.\nTopic detection will start fresh."
            )
        except Exception as e:
            logger.error(f"Failed to clear conversation context: {e}")
            self.add_status(f"❌ Failed to clear context: {e}", "error")
    
    def toggle_context_display(self):
        """Toggle conversation context display."""
        is_visible = self.show_context_button.isChecked()
        self.context_summary_label.setVisible(is_visible)
        
        if is_visible:
            self._update_context_display()
            self.show_context_button.setText("💬 Hide Context")
        else:
            self.show_context_button.setText("💬 Context")
    
    def _update_context_display(self):
        """Update conversation context display."""
        try:
            summary = self.router.get_conversation_context_summary()
            
            if summary and summary != "Conversation context disabled":
                self.context_summary_label.setText(f"Context: {summary}")
            else:
                self.context_summary_label.setText("Context: No conversation history")
                
        except Exception as e:
            logger.error(f"Failed to update context display: {e}")
            self.context_summary_label.setText(f"Context: Error - {e}")
    
    def toggle_project_enabled(self, project_name, enabled):
        """Enable or disable a project."""
        try:
            if enabled:
                self.project_manager.enable_project(project_name)
                self.add_status(f"✓ Enabled: {project_name}", "base")
            else:
                self.project_manager.disable_project(project_name)
                self.add_status(f"✗ Disabled: {project_name}", "base")
            
            # Update project selector
            self._update_project_selector()
            
            # If we disabled the currently selected project in manual mode, switch to auto
            if not enabled and self.project_selector.currentData() == project_name:
                self.project_selector.setCurrentIndex(0)  # Switch to automatic
                
        except Exception as e:
            logger.error(f"Failed to toggle project {project_name}: {e}")
            self.add_status(f"❌ Failed to toggle {project_name}: {e}", "error")
    
    def _complete_conversation_tab_setup(self, layout, splitter):
        """Complete the conversation tab setup (helper to fix broken method)."""
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
    
    def create_system_info_tab(self):
        """Create the system information tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Header with label and toggle button
        header_layout = QHBoxLayout()
        label = QLabel("System Information:")
        label.setStyleSheet("color: #1E90FF; font-weight: bold;")
        header_layout.addWidget(label)
        
        header_layout.addStretch()
        
        # Toggle button to switch between grouped and ungrouped view
        self.group_messages_toggle = QPushButton("📋 Grouped View")
        self.group_messages_toggle.setCheckable(True)
        self.group_messages_toggle.setChecked(self.settings.get('group_system_messages', True))
        self.group_messages_toggle.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")
        self.group_messages_toggle.clicked.connect(self.toggle_message_grouping)
        header_layout.addWidget(self.group_messages_toggle)
        
        layout.addLayout(header_layout)
        
        # Create scroll area for message groups
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #1E1E1E; border: 1px solid #3F3F46;")
        
        # Create message group widget
        self.message_group_widget = MessageGroupWidget()
        scroll_area.setWidget(self.message_group_widget)
        
        # Create traditional text edit (hidden by default if grouping is enabled)
        self.session_info = QTextEdit()
        self.session_info.setReadOnly(True)
        self.session_info.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;")
        self.session_info.setAcceptRichText(False)  # Use plain text for better performance
        self.session_info.setFont(QFont("Monospace", 9))
        self.session_info.setPlainText('Initializing Osprey Framework...\n\nPlease wait while the system initializes.')
        
        # Add both widgets to layout (we'll show/hide based on setting)
        layout.addWidget(scroll_area)
        layout.addWidget(self.session_info)
        
        # Show/hide based on initial setting
        if self.settings.get('group_system_messages', True):
            scroll_area.setVisible(True)
            self.session_info.setVisible(False)
            self.message_group_widget.add_message('Initializing Osprey Framework...', 'INFO')
            self.message_group_widget.add_message('Please wait while the system initializes.', 'INFO')
        else:
            scroll_area.setVisible(False)
            self.session_info.setVisible(True)
        
        # Store reference to scroll area for toggling
        self.message_group_scroll = scroll_area
        
        # Add clear button
        clear_btn = QPushButton("Clear System Info")
        clear_btn.clicked.connect(self.clear_system_info)
        layout.addWidget(clear_btn)
        
        return widget
    
    def create_analytics_tab(self):
        """Create the analytics dashboard tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Check if router is initialized and analytics is enabled
        analytics = self.router.get_analytics() if self.router else None
        
        if analytics:
            from osprey.interfaces.pyqt.analytics_dashboard import AnalyticsDashboard
            self.analytics_dashboard = AnalyticsDashboard(analytics, self)
            layout.addWidget(self.analytics_dashboard)
        else:
            # Analytics disabled message
            label = QLabel("Analytics is currently disabled.\n\nEnable analytics in router configuration to view metrics.")
            label.setStyleSheet("color: #FFA500; font-size: 14px;")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
        
        return widget
    
    @pyqtSlot(str)
    def append_to_system_info(self, message: str):
        """
        Append a message to the System Information tab with Rich markup parsing.
        
        This method is called by the logging system to redirect output to the GUI.
        Thread-safe for use from background threads via Qt's signal/slot mechanism.
        Parses Rich markup tags (e.g., [white], [bold green]) and applies colors.
        
        Args:
            message: The message to append (may contain Rich markup)
        """
        # Skip empty messages
        if not message or not message.strip():
            return
        
        # Determine message type from the message content
        message_type = self._extract_message_type(message)
        
        # If grouping is enabled, add to grouped widget WITH color formatting
        if self.settings.get('group_system_messages', True):
            # Pass the original message with Rich markup for color formatting
            self.message_group_widget.add_message(message, message_type, rich_markup=message)
        
        # Always add to traditional text edit (for when user switches view)
        cursor = self.session_info.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Parse and apply Rich markup
        self._insert_rich_text(cursor, message)
        
        # Add newline
        cursor.insertText('\n')
        self.session_info.setTextCursor(cursor)
        self.session_info.ensureCursorVisible()
    
    def _extract_message_type(self, message: str) -> str:
        """
        Extract the message type (log level) from a message.
        
        Args:
            message: The message text
        
        Returns:
            str: Message type (INFO, WARNING, ERROR, DEBUG, etc.)
        """
        import re
        
        # Extract log level from the formatted message
        # Format: [MM/DD/YYYY HH:MM:SS AM/PM] LEVEL     component: message
        # We need to match the LEVEL part specifically, not just any occurrence
        
        # First, try to extract from the standard logging format
        # Match pattern: [timestamp] LEVEL (with padding) component:
        level_match = re.search(r'\]\s+(ERROR|CRITICAL|WARNING|WARN|INFO|DEBUG)\s+', message)
        
        if level_match:
            level = level_match.group(1).upper()
            if level in ('ERROR', 'CRITICAL'):
                return 'ERROR'
            elif level in ('WARNING', 'WARN'):
                return 'WARNING'
            elif level == 'DEBUG':
                return 'DEBUG'
            elif level == 'INFO':
                return 'INFO'
        
        # Fallback: Check for error/warning indicators in the message content
        # Only check AFTER the component name to avoid false positives
        message_upper = message.upper()
        
        # Check for error emoji (strong indicator)
        if '❌' in message:
            return 'ERROR'
        
        # Check for warning emoji (strong indicator)
        if '⚠️' in message:
            return 'WARNING'
        
        # Check for success emoji (treat as INFO)
        if '✅' in message:
            return 'INFO'
        
        # Default to INFO for any unrecognized format
        return 'INFO'
    
    def _strip_rich_markup(self, text: str) -> str:
        """
        Strip Rich markup tags from text.
        
        Args:
            text: Text with Rich markup
        
        Returns:
            str: Text without markup tags
        """
        import re
        # Remove Rich markup tags like [white], [bold green], etc.
        return re.sub(r'\[([^\]]+)\]', '', text)
    
    def toggle_message_grouping(self):
        """Toggle between grouped and ungrouped message view."""
        is_grouped = self.group_messages_toggle.isChecked()
        self.settings['group_system_messages'] = is_grouped
        
        # Update button text
        if is_grouped:
            self.group_messages_toggle.setText("📋 Grouped View")
        else:
            self.group_messages_toggle.setText("📄 List View")
        
        # Show/hide appropriate widget
        self.message_group_scroll.setVisible(is_grouped)
        self.session_info.setVisible(not is_grouped)
    
    def _insert_rich_text(self, cursor, text: str):
        """
        Parse Rich markup and insert colored text.
        
        Supports Rich color tags like [white], [bold green], [sky_blue2], etc.
        Preserves timestamps in format [MM/DD/YYYY HH:MM:SS AM/PM]
        
        Args:
            cursor: QTextCursor to insert text at
            text: Text with Rich markup tags
        """
        import re
        
        # Rich color mapping to Qt colors
        rich_color_map = {
            'white': '#FFFFFF',
            'black': '#000000',
            'red': '#FF0000',
            'green': '#00FF00',
            'yellow': '#FFFF00',
            'blue': '#0000FF',
            'magenta': '#FF00FF',
            'cyan': '#00FFFF',
            'bright_white': '#FFFFFF',
            'bright_black': '#808080',
            'bright_red': '#FF6B6B',
            'bright_green': '#00FF00',
            'bright_yellow': '#FFFF00',
            'bright_blue': '#6B9EFF',
            'bright_magenta': '#FF6BFF',
            'bright_cyan': '#00FFFF',
            'sky_blue2': '#87CEEB',
            'bold': None,  # Style modifier, not a color
        }
        
        # Pattern to match Rich markup: [color] or [style color]
        # But NOT timestamps like [12/03/2025 09:28:08 AM]
        # Timestamps contain digits, slashes, colons, and spaces
        pattern = r'\[([^\]]+)\](.*?)(?=\[|$)'
        
        pos = 0
        for match in re.finditer(pattern, text):
            style_spec = match.group(1)
            
            # Check if this looks like a timestamp (contains digits and slashes/colons)
            # Timestamp pattern: contains digits, slashes, colons, spaces, AM/PM
            is_timestamp = bool(re.search(r'\d+[/:]', style_spec))
            
            if is_timestamp:
                # This is a timestamp, not a Rich markup tag - keep it as-is
                if match.start() > pos:
                    cursor.insertText(text[pos:match.start()])
                # Insert the timestamp with brackets
                cursor.insertText(f"[{style_spec}]")
                pos = match.start() + len(f"[{style_spec}]")
                continue
            
            # Insert any text before the match
            if match.start() > pos:
                cursor.insertText(text[pos:match.start()])
            
            content = match.group(2)
            
            # Parse style specification (e.g., "bold green", "white", "sky_blue2")
            parts = style_spec.split()
            color = '#FFFFFF'  # Default white
            bold = False
            
            for part in parts:
                if part == 'bold':
                    bold = True
                elif part in rich_color_map:
                    if rich_color_map[part] is not None:
                        color = rich_color_map[part]
            
            # Create text format with color and style
            text_format = QTextCharFormat()
            text_format.setForeground(QBrush(QColor(color)))
            if bold:
                text_format.setFontWeight(QFont.Bold)
            
            # Insert formatted text
            cursor.insertText(content, text_format)
            
            pos = match.end()
        
        # Insert any remaining text
        if pos < len(text):
            cursor.insertText(text[pos:])
    
    def clear_system_info(self):
        """Clear the System Information tab."""
        self.session_info.clear()
        self.message_group_widget.clear()
        self.update_session_info()
    
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
        self.projects_table.setColumnCount(7)
        self.projects_table.setHorizontalHeaderLabels(['Status', 'Project Name', 'Capabilities', 'Models', 'Path', 'Config File', 'Model Config'])
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
        self.projects_info_label = QLabel("No projects discovered yet. Click Refresh to scan for projects.")
        self.projects_info_label.setStyleSheet("color: #808080; font-style: italic; padding: 10px;")
        layout.addWidget(self.projects_info_label)
        
        return widget
    
    def refresh_projects_display(self, force_refresh: bool = False):
        """
        Refresh the discovered projects display.
        
        Args:
            force_refresh: If True, bypass cache and rediscover projects
        """
        try:
            self.add_status("Refreshing project list...", "base")
            
            # Get loaded projects from ProjectManager
            loaded_projects = self.project_manager.list_loaded_projects()
            
            # Build display data from ProjectManager
            display_projects = []
            for project_name in loaded_projects:
                context = self.project_manager.get_project(project_name)
                if not context:
                    continue
                
                # Get capabilities from ProjectManager
                capabilities = self.project_manager.get_project_capabilities(project_name)
                capability_names = list(capabilities.keys())
                
                # Get models from project config
                models = context.config.raw_config.get('models', {})
                
                display_projects.append({
                    'name': context.metadata.name,
                    'path': str(context.metadata.path),
                    'config_path': str(context.metadata.config_path),
                    'description': context.metadata.description,
                    'version': context.metadata.version,
                    'capabilities': capability_names,
                    'models': models
                })
            
            # Update the cached list for backward compatibility
            self.discovered_projects = display_projects
            self._projects_cache_valid = True
            
            # Update table
            self.projects_table.setRowCount(len(display_projects))
            
            # Enable word wrap and adjust row heights
            self.projects_table.setWordWrap(True)
            
            for row, project in enumerate(self.discovered_projects):
                # Status column with enable/disable checkbox (NEW - Phase 2.2)
                status_widget = QWidget()
                status_layout = QHBoxLayout(status_widget)
                status_layout.setContentsMargins(2, 1, 2, 1)
                
                enabled_checkbox = QCheckBox()
                # Check if project is enabled in ProjectManager
                is_enabled = self.project_manager.is_project_enabled(project['name'])
                enabled_checkbox.setChecked(is_enabled)
                enabled_checkbox.stateChanged.connect(
                    lambda state, p=project['name']:
                        self.toggle_project_enabled(p, state == Qt.Checked)
                )
                status_layout.addWidget(enabled_checkbox)
                
                status_label = QLabel("Enabled" if is_enabled else "Disabled")
                status_label.setStyleSheet(
                    "color: #00FF00;" if is_enabled else "color: #808080;"
                )
                status_layout.addWidget(status_label)
                
                self.projects_table.setCellWidget(row, 0, status_widget)
                
                # Project name
                name_item = QTableWidgetItem(project['name'])
                name_item.setForeground(QColor("#00FFFF"))
                self.projects_table.setItem(row, 1, name_item)
                
                # Capabilities column - show count and list
                capabilities = project.get('capabilities', [])
                cap_count = len(capabilities)
                if cap_count > 0:
                    cap_text = f"{cap_count} capabilities:\n" + "\n".join(f"  • {cap}" for cap in capabilities[:5])
                    if cap_count > 5:
                        cap_text += f"\n  ... and {cap_count - 5} more"
                else:
                    cap_text = "No capabilities"
                cap_item = QTableWidgetItem(cap_text)
                cap_item.setForeground(QColor("#00FF00") if cap_count > 0 else QColor("#808080"))
                cap_item.setToolTip("\n".join(capabilities) if capabilities else "No capabilities found")
                self.projects_table.setItem(row, 2, cap_item)
                
                # Models column - show configured models
                models = project.get('models', {})
                model_count = len(models)
                if model_count > 0:
                    model_text = f"{model_count} models:\n" + "\n".join(f"  • {step}: {model}" for step, model in list(models.items())[:5])
                    if model_count > 5:
                        model_text += f"\n  ... and {model_count - 5} more"
                else:
                    model_text = "No models"
                model_item = QTableWidgetItem(model_text)
                model_item.setForeground(QColor("#FFD700") if model_count > 0 else QColor("#808080"))
                model_item.setToolTip("\n".join(f"{step}: {model}" for step, model in models.items()) if models else "No models configured")
                self.projects_table.setItem(row, 3, model_item)
                
                # Project path
                path_item = QTableWidgetItem(project['path'])
                path_item.setForeground(QColor("#FFFFFF"))
                self.projects_table.setItem(row, 4, path_item)
                
                # Config path
                config_path = Path(project['config_path']).name
                config_item = QTableWidgetItem(config_path)
                config_item.setForeground(QColor("#00FF00"))
                self.projects_table.setItem(row, 5, config_item)
                
                # Model configuration button
                models_widget = QWidget()
                models_layout = QHBoxLayout(models_widget)
                models_layout.setContentsMargins(2, 1, 2, 1)
                
                config_btn = QPushButton("Configure")
                config_btn.setToolTip("Configure runtime model overrides for infrastructure steps")
                config_btn.clicked.connect(lambda checked, p=project: self.configure_project_models(p))
                models_layout.addWidget(config_btn)
                
                # Show indicator if runtime overrides are configured
                pref_count = self.model_preferences.get_preference_count(project['name'])
                if pref_count > 0:
                    indicator = QLabel(f"✓ ({pref_count})")
                    indicator.setToolTip(f"{pref_count} runtime override(s) configured")
                    indicator.setStyleSheet("color: #00FF00;")
                    models_layout.addWidget(indicator)
                
                self.projects_table.setCellWidget(row, 6, models_widget)
                
                # Let row height adjust based on content
                # Users can manually resize rows by dragging the row borders in the vertical header
                # Don't set explicit height - let Interactive mode handle it
            
            # Resize rows to fit content initially, then users can manually adjust
            self.projects_table.resizeRowsToContents()
            
            # Update info label
            if self.discovered_projects:
                enabled_count = len([p for p in self.discovered_projects
                                    if self.project_manager.is_project_enabled(p['name'])])
                self.projects_info_label.setText(
                    f"Found {len(self.discovered_projects)} project(s) • "
                    f"{enabled_count} enabled • "
                    f"Use checkboxes to enable/disable projects for routing"
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
        from osprey.utils.config import set_runtime_model_override
        
        # Apply preferences for this project (don't clear - we want to merge all projects)
        preferences = self.model_preferences.get_all_preferences(project_name)
        if preferences:
            for step, model_id in preferences.items():
                set_runtime_model_override(step, model_id)
                self.add_status(f"✓ {project_name}: {step} → {model_id}", "base")
    
    def _apply_all_model_preferences(self):
        """
        Apply model preferences for all discovered projects.
        
        This implements the hybrid approach where:
        - Infrastructure steps use GUI-configured models (runtime overrides)
        - Each project's capabilities can have their own models
        - Later projects' preferences override earlier ones for infrastructure
        """
        from osprey.utils.config import clear_runtime_model_overrides
        
        # Clear existing overrides first
        clear_runtime_model_overrides()
        
        if not self.discovered_projects:
            return
        
        self.add_status("Applying model preferences for multi-project setup...", "base")
        
        # Apply preferences for each project
        # Later projects override earlier ones for infrastructure steps
        for project in self.discovered_projects:
            project_name = project['name']
            self.apply_model_preferences_to_config(project_name)
        
        # Log summary
        from osprey.utils.config import get_runtime_model_overrides
        overrides = get_runtime_model_overrides()
        if overrides:
            self.add_status(f"Applied {len(overrides)} model override(s) for infrastructure steps", "base")
        else:
            self.add_status("No model preferences configured - using defaults from config", "base")
    
    def initialize_framework(self):
        """Initialize the Osprey framework components."""
        try:
            # GUI output redirection is already set up in __init__
            # Phase 2.1: Use ProjectManager for discovery and loading
            self.add_status("Discovering projects with ProjectManager...", "base")
            
            # 1. Discover projects using ProjectManager
            discovered = self.project_manager.discover_projects()
            
            # Update old discovered_projects for backward compatibility with projects tab
            self.discovered_projects = [
                {
                    'name': metadata.name,
                    'path': str(metadata.path),
                    'config_path': str(metadata.config_path),
                    'description': metadata.description,
                    'version': metadata.version,
                    'capabilities': [],  # Will be populated after loading
                    'models': {}  # Will be populated after loading
                }
                for metadata in discovered
            ]
            self._projects_cache_valid = True
            
            # Update projects tab display
            QTimer.singleShot(200, lambda: self.refresh_projects_display(force_refresh=False))
            
            if discovered:
                self.add_status(f"Found {len(discovered)} project(s)", "base")
                
                # 2. Load each project with isolated context
                for metadata in discovered:
                    try:
                        self.add_status(f"Loading project: {metadata.name}", "base")
                        context = self.project_manager.load_project(metadata.name)
                        
                        # 3. Register capabilities in CapabilityRegistry
                        capabilities = self.project_manager.get_project_capabilities(metadata.name)
                        self.capability_registry.register_project_capabilities(
                            metadata.name,
                            capabilities
                        )
                        
                        self.add_status(f"✓ Loaded: {metadata.name}", "base")
                        
                    except Exception as e:
                        logger.error(f"Failed to load project {metadata.name}: {e}")
                        self.add_status(f"⚠️ Failed to load {metadata.name}: {e}", "error")
                
                # 4. Use first enabled project's gateway for backward compatibility
                enabled_projects = self.project_manager.get_enabled_projects()
                if enabled_projects:
                    first_project = enabled_projects[0]
                    self.gateway = first_project.gateway
                    # Note: graph will be created below with checkpointer
                    # Only set config_path if it wasn't already set (e.g., from gui_config.yml)
                    if not self.config_path:
                        self.config_path = str(first_project.metadata.config_path)
                    self.add_status(f"Using {first_project.metadata.name} as primary project", "base")
                    
                    # Update project selector with loaded projects (Phase 2.2)
                    QTimer.singleShot(300, self._update_project_selector)
                else:
                    self.add_status("⚠️ No projects enabled", "error")
                    
            else:
                self.add_status("No projects found for auto-discovery", "base")
            
            self.add_status("Initializing Osprey framework...", "base")
            
            # Get configuration (use first enabled project's config if available)
            # This also sets the default config for the config system
            configurable = get_full_configuration(config_path=self.config_path).copy()
            
            # Load conversation history AFTER config is initialized
            # This ensures get_agent_dir() can find the config
            self.load_conversation_history()
            
            # Update conversation list UI after loading from JSON
            if self.conversations:
                self.update_conversation_list()
            
            # Create initial conversation if none exist (after loading history)
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
            
            configurable.update({
                "user_id": "gui_user",
                "thread_id": self.thread_id,
                "chat_id": "gui_chat",
                "session_id": self.thread_id,
                "interface_context": "pyqt_gui"
            })
            
            # Load current agent control settings from config to initialize GUI settings
            agent_control_defaults = configurable.get("agent_control_defaults", {})
            development_config = configurable.get("development", {})
            prompts_config = development_config.get("prompts", {})
            
            # Read debug mode from config and apply it to logging BEFORE detecting level
            import logging
            config_debug_mode = development_config.get('debug', False)
            
            # Apply the config debug setting to the logger immediately
            root_logger = logging.getLogger()
            desired_level = logging.DEBUG if config_debug_mode else logging.INFO
            root_logger.setLevel(desired_level)
            
            # Also update all existing loggers
            for logger_name in logging.Logger.manager.loggerDict:
                logger_obj = logging.getLogger(logger_name)
                if isinstance(logger_obj, logging.Logger):
                    logger_obj.setLevel(desired_level)
            
            # Update GUI handler level if it exists
            for handler in root_logger.handlers:
                from osprey.utils.logger import GUIHandler
                if isinstance(handler, GUIHandler):
                    handler.setLevel(desired_level)
            
            # Now detect the level (should match config)
            current_debug_mode = root_logger.level <= logging.DEBUG
            
            # Update self.settings with values from config (so GUI shows current state)
            self.settings.update({
                'planning_mode_enabled': agent_control_defaults.get('planning_mode_enabled', False),
                'epics_writes_enabled': agent_control_defaults.get('epics_writes_enabled', False),
                'task_extraction_bypass_enabled': agent_control_defaults.get('task_extraction_bypass_enabled', False),
                'capability_selection_bypass_enabled': agent_control_defaults.get('capability_selection_bypass_enabled', False),
                'approval_global_mode': agent_control_defaults.get('approval_global_mode', 'selective'),
                'python_execution_approval_enabled': agent_control_defaults.get('python_execution_approval_enabled', True),
                'python_execution_approval_mode': agent_control_defaults.get('python_execution_approval_mode', 'all_code'),
                'memory_approval_enabled': agent_control_defaults.get('memory_approval_enabled', True),
                'max_reclassifications': agent_control_defaults.get('max_reclassifications', 1),
                'max_planning_attempts': agent_control_defaults.get('max_planning_attempts', 2),
                'max_step_retries': agent_control_defaults.get('max_step_retries', 0),
                'max_execution_time_seconds': agent_control_defaults.get('max_execution_time_seconds', 300),
                'max_concurrent_classifications': agent_control_defaults.get('max_concurrent_classifications', 5),
                # Development/Debug settings - use actual logging level for debug_mode
                'debug_mode': current_debug_mode,
                'verbose_logging': development_config.get('verbose_logging', False),
                'raise_raw_errors': development_config.get('raise_raw_errors', False),
                'print_prompts': prompts_config.get('print_all', False),
                'show_prompts': prompts_config.get('show_all', False),
                'prompts_latest_only': prompts_config.get('latest_only', True),
            })
            
            # Apply settings back to config (in case any were missing)
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
            
            # Initialize router AFTER framework is initialized (so config is available)
            if not self.router:
                self._initialize_router()
            
            # Load current conversation display
            QTimer.singleShot(100, self.load_current_conversation_display)
            
        except Exception as e:
            logger.exception(f"Failed to initialize framework: {e}")
            self.add_status(f"❌ Framework initialization failed: {e}", "error")
            QMessageBox.critical(self, "Initialization Error",
                               f"Failed to initialize framework:\n{e}")
    
    def update_session_info(self):
        """Update the session information display by appending session header."""
        registry = get_registry()
        capabilities = registry.get_all_capabilities() if registry else []
        
        # Build session info text
        session_text = []
        session_text.append("=" * 80)
        session_text.append("OSPREY FRAMEWORK SESSION")
        session_text.append("=" * 80)
        session_text.append(f"Thread ID: {self.thread_id}")
        session_text.append(f"Config Path: {self.config_path}")
        session_text.append(f"Capabilities: {len(capabilities)}")
        session_text.append("")
        
        # Append to system info (don't overwrite)
        cursor = self.session_info.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText('\n'.join(session_text) + '\n')
        self.session_info.setTextCursor(cursor)
        self.session_info.ensureCursorVisible()
    
    def add_status(self, message, component="base", model_info=None):
        """Add a status message to the status log with color coding and optional model info.
        
        Args:
            message: Status message to display
            component: Component type for color coding
            model_info: Optional dict with 'model_provider' and 'model_id' keys
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.component_colors.get(component, self.component_colors['base'])
        
        # Add timestamp
        self._append_formatted_text(f"[{timestamp}] ", "#808080", self.status_log, prefix="", suffix="")
        
        # Add model info if available
        if model_info and isinstance(model_info, dict):
            provider = model_info.get('model_provider', '')
            model_id = model_info.get('model_id', '')
            if provider and model_id:
                self._append_formatted_text(f"[{provider}/{model_id}] ", "#FFD700", self.status_log, prefix="", suffix="")
        
        # Add message
        self._append_formatted_text(message, color, self.status_log, prefix="", suffix="\n")
    
    def send_message(self):
        """Send user message to the framework with multi-project routing."""
        user_message = self.input_field.toPlainText().strip()
        if not user_message:
            return
        
        # Check if waiting for correction input (only if feedback enabled)
        if self.settings.get('enable_routing_feedback', True) and self._waiting_for_correction:
            self._handle_correction_input(user_message)
            self.input_field.clear()
            return
        
        # Check if this is feedback for previous routing (y/n) (only if feedback enabled)
        if (self.settings.get('enable_routing_feedback', True) and
            self.current_routing_decision and
            user_message.lower() in ['y', 'n', 'yes', 'no']):
            self._handle_routing_feedback(user_message.lower())
            self.input_field.clear()
            # Don't process queued message here - user answered feedback separately
            return
        
        # If agent is processing, queue this message (only allow 1 queued message)
        if self._agent_processing:
            if self._queued_message is None:
                self._queued_message = user_message
                self.input_field.clear()
                
                # Clear routing feedback state since user is moving on to a new question
                # This prevents confusion if they later type y/n for unrelated reasons
                if self.current_routing_decision:
                    self.current_routing_decision = None
                    self.current_query = None
                
                self._append_colored_message(
                    f"📝 Queued: {user_message}\n"
                    "   (Will be processed after current query completes)",
                    "#FFD700"
                )
            else:
                self._append_colored_message(
                    "⚠️ A message is already queued. Please wait for it to be processed.",
                    "#FFA500"
                )
            return
        
        # IMMEDIATELY display user message and clear input for better UX
        self._append_colored_message(f"👤 You: {user_message}", "#D8BFD8")
        self.input_field.clear()
        
        # Mark agent as processing
        self._agent_processing = True
        
        # Keep input enabled so user can provide feedback or queue next message
        self.status_bar.showMessage("Processing...")
        
        # Force GUI update to show message immediately
        QApplication.processEvents()
        
        # Update conversation history (fast operation)
        if self.current_conversation_id and self.current_conversation_id in self.conversations:
            self.conversations[self.current_conversation_id]['messages'].append({
                'type': 'user',
                'content': user_message,
                'timestamp': datetime.now()
            })
            self.conversations[self.current_conversation_id]['timestamp'] = datetime.now()
            self.update_conversation_list()
            self.save_conversation_history()
        
        # Phase 2.3: Multi-Project Routing with Orchestration
        enabled_projects = self.project_manager.get_enabled_projects()
        
        if not enabled_projects:
            self._append_colored_message(
                "⚠️ No projects enabled. Please enable at least one project in the Projects tab.",
                "#FFA500"
            )
            self.add_status("No enabled projects available", "error")
            return
        
        # Check if query requires orchestration
        try:
            orchestration_plan = self.router.analyze_for_orchestration(
                user_message,
                enabled_projects
            )
            
            if orchestration_plan.is_multi_project:
                # Handle multi-project orchestration
                self._handle_orchestrated_query(
                    user_message,
                    orchestration_plan,
                    enabled_projects
                )
                return
            
        except Exception as e:
            logger.warning(f"Orchestration analysis failed: {e}, falling back to single routing")
        
        # Single-project routing
        try:
            routing_decision = self.router.route_query(user_message, enabled_projects)
            
            # Store for feedback collection
            self.current_routing_decision = routing_decision
            self.current_query = user_message
            
            # Display routing decision
            self._display_routing_decision(routing_decision)
            
            # Get the selected project
            selected_project = self.project_manager.get_project(routing_decision.project_name)
            
            if not selected_project:
                raise Exception(f"Selected project '{routing_decision.project_name}' not found")
            
            # Use selected project's gateway AND graph
            self.gateway = selected_project.gateway
            
            # CRITICAL FIX: Use the project's own graph, not the main graph
            # Each project has its own graph with its own state/execution tracking
            project_graph = selected_project.graph
            if not project_graph:
                logger.warning(f"Project {routing_decision.project_name} has no graph, using main graph")
                project_graph = self.graph
            
            # Update config with selected project's thread_id
            if self.base_config:
                self.base_config["configurable"]["thread_id"] = self.thread_id
                self.base_config["configurable"]["session_id"] = self.thread_id
            
            self.add_status(f"Using project: {routing_decision.project_name}", "base")
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            self._append_colored_message(
                f"⚠️ Routing error: {e}\nUsing fallback project.",
                "#FFA500"
            )
            self.add_status(f"Routing error: {e}", "error")
            
            # Fallback to first enabled project
            if enabled_projects:
                fallback_project = enabled_projects[0]
                self.gateway = fallback_project.gateway
                # Note: graph is stored in self.graph, not in gateway
                self.add_status(f"Fallback to: {fallback_project.metadata.name}", "base")
        
        # Use project-specific graph if we routed to a specific project
        # Otherwise use the main graph
        graph_to_use = project_graph if 'project_graph' in locals() else self.graph
        
        self.worker = AgentWorker(
            self.gateway,
            graph_to_use,  # Use project's graph, not main graph
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
    
    def _append_formatted_text(self, text: str, color: str, widget=None, prefix: str = "\n", suffix: str = "\n"):
        """
        Unified method to append formatted text to a text widget.
        
        Args:
            text: Text to append
            color: Color hex code (e.g., "#FFFFFF")
            widget: QTextEdit widget (defaults to conversation_display)
            prefix: Text to prepend (default: newline)
            suffix: Text to append (default: newline)
        """
        if widget is None:
            widget = self.conversation_display
        
        cursor = widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        text_format = QTextCharFormat()
        text_format.setForeground(QBrush(QColor(color)))
        
        cursor.insertText(f"{prefix}{text}{suffix}", text_format)
        
        widget.setTextCursor(cursor)
        widget.ensureCursorVisible()
    
    def _append_colored_message(self, message, color):
        """Append a colored message to the conversation display."""
        self._append_formatted_text(message, color)
    
    def _display_routing_decision(self, decision):
        """Display routing decision to user with visual feedback.
        
        Args:
            decision: RoutingDecision object with routing information.
        """
        # Determine color based on confidence
        if decision.confidence >= 0.8:
            confidence_color = "#00FF00"  # Green for high confidence
            confidence_icon = "✅"
        elif decision.confidence >= 0.5:
            confidence_color = "#FFD700"  # Gold for medium confidence
            confidence_icon = "⚠️"
        else:
            confidence_color = "#FFA500"  # Orange for low confidence
            confidence_icon = "⚠️"
        
        # Display routing mode
        if self.router.is_automatic_mode():
            mode_text = "🎯 Automatic Routing"
            mode_color = "#00FFFF"
        else:
            mode_text = "📌 Manual Selection"
            mode_color = "#FFD700"
        
        # Build routing message
        routing_msg = f"\n{mode_text} → {decision.project_name}"
        self._append_colored_message(routing_msg, mode_color)
        
        # Display confidence
        confidence_msg = f"{confidence_icon} Confidence: {decision.confidence:.0%}"
        self._append_colored_message(confidence_msg, confidence_color)
        
        # Display reasoning if available
        if decision.reasoning:
            reasoning_msg = f"   Reason: {decision.reasoning}"
            self._append_colored_message(reasoning_msg, "#808080")
        
        # Display alternatives if available
        if decision.alternative_projects:
            alt_msg = f"   Alternatives: {', '.join(decision.alternative_projects)}"
            self._append_colored_message(alt_msg, "#606060")
        
        # Display feedback prompt (only in automatic mode and if feedback enabled)
        if self.router.is_automatic_mode() and self.settings.get('enable_routing_feedback', True):
            feedback_msg = (
                "   Was this routing correct?\n"
                "   Type 'y' (yes/correct) or 'n' (no/incorrect) to provide feedback\n"
                "   Or type your next query (one query can be queued while processing)"
            )
            self._append_colored_message(feedback_msg, "#87CEEB")
        
        # Add separator
        self._append_colored_message("─" * 60, "#404040")
        
        # Update cache statistics if visible
        if self.show_cache_stats_button.isChecked():
            self._update_cache_statistics()
        
        # Update conversation context if visible
        if self.show_context_button.isChecked():
            self._update_context_display()
    
    def _handle_routing_feedback(self, feedback: str):
        """Handle user feedback on routing decision.
        
        Args:
            feedback: User feedback ('y', 'n', 'yes', 'no')
        """
        # Check if feedback is enabled
        if not self.settings.get('enable_routing_feedback', True):
            return
        
        if not self.current_routing_decision or not self.current_query:
            self._append_colored_message(
                "⚠️ No routing decision to provide feedback for.",
                "#FFA500"
            )
            return
        
        # Determine if feedback is positive or negative
        is_correct = feedback in ['y', 'yes']
        
        if is_correct:
            # Positive feedback
            self._append_colored_message(
                "✅ Thank you! Routing feedback recorded as correct.",
                "#00FF00"
            )
            
            # Record positive feedback
            self.router.record_routing_feedback(
                query=self.current_query,
                selected_project=self.current_routing_decision.project_name,
                confidence=self.current_routing_decision.confidence,
                user_feedback="correct",
                reasoning=self.current_routing_decision.reasoning
            )
            
            # Clear current routing decision (user answered feedback)
            self.current_routing_decision = None
            self.current_query = None
            
        else:
            # Negative feedback - ask for correct project
            self._append_colored_message(
                "👎 Routing was incorrect. Which project should have been used?",
                "#FFA500"
            )
            
            # Get enabled projects for selection
            enabled_projects = self.project_manager.get_enabled_projects()
            project_names = [p.metadata.name for p in enabled_projects]
            
            # Display options
            options_msg = "Available projects:\n" + "\n".join(
                f"  {i+1}. {name}" for i, name in enumerate(project_names)
            )
            self._append_colored_message(options_msg, "#FFFFFF")
            self._append_colored_message(
                "Enter the number or name of the correct project:",
                "#87CEEB"
            )
            
            # Set state to wait for correction
            self._waiting_for_correction = True
            self._correction_options = project_names
            return
        
        # Clear current routing decision
        self.current_routing_decision = None
        self.current_query = None
    
    def _handle_correction_input(self, user_input: str):
        """Handle user input for routing correction.
        
        Args:
            user_input: User's correction input (project name or number)
        """
        # Check if feedback is enabled
        if not self.settings.get('enable_routing_feedback', True):
            self._waiting_for_correction = False
            return
        
        if not self._correction_options:
            self._append_colored_message(
                "⚠️ No correction options available.",
                "#FFA500"
            )
            self._waiting_for_correction = False
            return
        
        # Try to parse as number
        correct_project = None
        try:
            index = int(user_input) - 1
            if 0 <= index < len(self._correction_options):
                correct_project = self._correction_options[index]
        except ValueError:
            # Not a number, try as project name
            if user_input in self._correction_options:
                correct_project = user_input
        
        if not correct_project:
            self._append_colored_message(
                f"⚠️ Invalid selection: '{user_input}'. Please try again.",
                "#FFA500"
            )
            return
        
        # Record negative feedback with correction
        self.router.record_routing_feedback(
            query=self.current_query,
            selected_project=self.current_routing_decision.project_name,
            confidence=self.current_routing_decision.confidence,
            user_feedback="incorrect",
            correct_project=correct_project,
            reasoning=self.current_routing_decision.reasoning
        )
        
        self._append_colored_message(
            f"✅ Thank you! Feedback recorded. Correct project: {correct_project}",
            "#00FF00"
        )
        self._append_colored_message(
            "   The system will learn from this correction.",
            "#87CEEB"
        )
        
        # Clear state
        self.current_routing_decision = None
        self.current_query = None
        self._waiting_for_correction = False
        self._correction_options = []
    
    def _handle_orchestrated_query(
        self,
        query: str,
        plan,
        enabled_projects: List
    ):
        """Handle a multi-project orchestrated query.
        
        Args:
            query: Original user query.
            plan: OrchestrationPlan from analysis.
            enabled_projects: List of enabled projects.
        """
        try:
            # Display orchestration plan
            self._display_orchestration_plan(plan)
            
            # Create project contexts dictionary
            project_contexts = {
                p.metadata.name: p for p in enabled_projects
            }
            
            # Execute orchestration plan using the orchestrator
            self._append_colored_message(
                "🔄 Executing multi-project orchestration...",
                "#00FFFF"
            )
            
            # Execute each sub-query sequentially (for now)
            # TODO: Implement parallel execution based on dependencies
            results = {}
            for idx, sub_query in enumerate(plan.sub_queries):
                self._append_colored_message(
                    f"\n  {idx + 1}. [{sub_query.project_name}] {sub_query.query}",
                    "#FFD700"
                )
                
                # Get the project context
                project = project_contexts.get(sub_query.project_name)
                if not project:
                    error_msg = f"Project not found: {sub_query.project_name}"
                    self._append_colored_message(f"     ❌ {error_msg}", "#FF0000")
                    results[idx] = f"Error: {error_msg}"
                    continue
                
                # Execute the sub-query using the project's OWN gateway and graph
                try:
                    self._append_colored_message(f"     ⏳ Processing...", "#808080")
                    
                    # CRITICAL: Use the project's own graph, not the GUI's unified graph
                    # Each project has its own capabilities loaded in its own graph
                    project_graph = project.graph
                    if not project_graph:
                        error_msg = f"Project {sub_query.project_name} has no graph loaded"
                        self._append_colored_message(f"     ❌ {error_msg}", "#FF0000")
                        results[idx] = f"Error: {error_msg}"
                        continue
                    
                    # Execute synchronously (blocking)
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Create project-specific config
                    project_config = {
                        "configurable": {
                            **self.base_config["configurable"],
                            "thread_id": f"{sub_query.project_name}_{idx}",
                            "session_id": f"{sub_query.project_name}_{idx}"
                        },
                        "recursion_limit": self.base_config.get("recursion_limit", 100)
                    }
                    
                    # Process the message through project's gateway
                    result = loop.run_until_complete(
                        project.gateway.process_message(
                            sub_query.query,
                            project_graph,  # Use PROJECT's graph, not GUI's graph!
                            project_config
                        )
                    )
                    
                    # If we have agent_state, execute the PROJECT's graph to get the final response
                    if result.agent_state and not result.error:
                        # Execute PROJECT's graph to completion
                        async def execute_graph():
                            final_state = None
                            async for chunk in project_graph.astream(
                                result.agent_state,
                                config=project_config,
                                stream_mode="custom"
                            ):
                                # Just consume the stream
                                pass
                            
                            # Get final state from PROJECT's graph
                            final_state = project_graph.get_state(config=project_config)
                            return final_state
                        
                        final_state = loop.run_until_complete(execute_graph())
                        
                        # Extract response from final state
                        if final_state and final_state.values:
                            messages = final_state.values.get("messages", [])
                            if messages:
                                # Get the last AI message
                                for msg in reversed(messages):
                                    if hasattr(msg, 'content') and msg.content:
                                        if not hasattr(msg, 'type') or msg.type != 'human':
                                            response = msg.content
                                            break
                                else:
                                    response = "No response generated"
                            else:
                                response = "No response generated"
                        else:
                            response = "No response generated"
                        
                        self._append_colored_message(f"     ✅ Complete", "#00FF00")
                    elif result.error:
                        response = f"Error: {result.error}"
                        self._append_colored_message(f"     ❌ {response}", "#FF0000")
                    else:
                        response = "No response generated"
                        self._append_colored_message(f"     ⚠️ {response}", "#FFA500")
                    
                    loop.close()
                    
                    results[idx] = response
                    
                except Exception as e:
                    error_msg = f"Execution failed: {e}"
                    logger.error(f"Sub-query {idx} failed: {e}")
                    self._append_colored_message(f"     ❌ {error_msg}", "#FF0000")
                    results[idx] = f"Error: {error_msg}"
            
            # Synthesize results
            self._append_colored_message(
                "\n🔗 Synthesizing results...",
                "#00FFFF"
            )
            
            # Use orchestrator to combine results
            combined_result = self.router.orchestrator._combine_results(plan, results)
            
            # Display final answer
            self._append_colored_message(
                "\n" + "=" * 60,
                "#404040"
            )
            self._append_colored_message(
                "\n🤖 Combined Answer:",
                "#00FF00"
            )
            self._append_colored_message(
                combined_result,
                "#FFFFFF"
            )
            self._append_colored_message(
                "\n" + "=" * 60,
                "#404040"
            )
            
            # Mark agent as no longer processing
            self._agent_processing = False
            
            # Re-enable input
            self.input_field.setEnabled(True)
            self.send_button.setEnabled(True)
            self.input_field.setFocus()
            self.status_bar.showMessage("Ready")
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            self._append_colored_message(
                f"⚠️ Orchestration error: {e}",
                "#FF0000"
            )
            self.add_status(f"Orchestration error: {e}", "error")
            
            # Mark agent as no longer processing
            self._agent_processing = False
            
            # Re-enable input
            self.input_field.setEnabled(True)
            self.send_button.setEnabled(True)
    
    def _display_orchestration_plan(self, plan):
        """Display orchestration plan to user.
        
        Args:
            plan: OrchestrationPlan to display.
        """
        # Display header
        self._append_colored_message(
            "\n🎯 Multi-Project Query Detected",
            "#00FFFF"
        )
        
        # Display reasoning
        if plan.reasoning:
            self._append_colored_message(
                f"   Reason: {plan.reasoning}",
                "#808080"
            )
        
        # Display sub-queries
        self._append_colored_message(
            f"   Decomposed into {len(plan.sub_queries)} sub-queries:",
            "#FFFFFF"
        )
        
        # Add separator
        self._append_colored_message("─" * 60, "#404040")
    
    def on_status_update(self, status, component="base", model_info=None):
        """Handle status update from agent.
        
        Args:
            status: Status message
            component: Component type for color coding
            model_info: Optional dict with model_provider and model_id
        """
        self.add_status(status, component, model_info=model_info)
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
        
        # Add timestamp
        self._append_formatted_text(f"[{timestamp}] ", "#808080", self.llm_details_display, prefix="", suffix="")
        # Add event type tag
        self._append_formatted_text(f"[{event_type.upper()}] ", color, self.llm_details_display, prefix="", suffix="")
        # Add detail
        self._append_formatted_text(detail, "#FFFFFF", self.llm_details_display, prefix="", suffix="\n")
    
    def on_tool_usage(self, tool_name, reasoning):
        """Handle tool usage information."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add newline separator
        self._append_formatted_text("", "#FFFFFF", self.tool_usage_display, prefix="\n", suffix="")
        
        # Add timestamp
        self._append_formatted_text(f"[{timestamp}] ", "#808080", self.tool_usage_display, prefix="", suffix="")
        
        # Add capability label and name
        self._append_formatted_text("Capability: ", "#FFA500", self.tool_usage_display, prefix="", suffix="")
        self._append_formatted_text(tool_name, "#00FFFF", self.tool_usage_display, prefix="", suffix="\n")
        
        # Add reasoning lines with appropriate colors
        line_colors = {
            '✅': '#00FF00',  # Success - green
            '❌': '#FF6B6B',  # Failure - red
            '⏱️': '#FFD700',  # Timing - gold
        }
        
        for line in reasoning.split('\n'):
            if not line.strip():
                continue
            
            # Determine color based on line prefix
            color = '#FFFFFF'  # Default white
            for prefix, prefix_color in line_colors.items():
                if line.startswith(prefix):
                    color = prefix_color
                    break
            
            self._append_formatted_text(line, color, self.tool_usage_display, prefix="", suffix="\n")
        
        # Add separator
        self._append_formatted_text("=" * 80, "#404040", self.tool_usage_display, prefix="", suffix="\n")
    
    def on_processing_complete(self):
        """Handle completion of agent processing."""
        # Mark agent as no longer processing
        self._agent_processing = False
        
        # Ensure input is enabled
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        
        # Check if there's a queued message to process
        if self._queued_message is not None:
            queued = self._queued_message
            self._queued_message = None
            
            self._append_colored_message(
                f"▶️ Processing queued message...",
                "#00FFFF"
            )
            
            # Set the queued message in the input field and send it
            self.input_field.setPlainText(queued)
            # Use QTimer to allow GUI to update before processing
            QTimer.singleShot(100, self.send_message)
        else:
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
        """Switch to a different conversation and reload all messages."""
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
                f"📂 LOADED CONVERSATION: {conv_data['name']}\n" +
                f"   Created: {conv_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n" +
                "=" * 80 + "\n",
                "#00FFFF"
            )
            
            # Load and display all messages from this conversation
            storage_mode = self.settings.get('conversation_storage_mode', 'json')
            
            if storage_mode == 'json' and conv_data.get('messages'):
                # Load from JSON storage (in-memory)
                message_count = len(conv_data['messages'])
                self.add_status(f"Loading {message_count} messages from JSON storage...", "base")
                
                for msg in conv_data['messages']:
                    if msg['type'] == 'user':
                        self._append_colored_message(f"👤 You: {msg['content']}", "#D8BFD8")
                    else:
                        self._append_colored_message(f"🤖 {msg['content']}", "#FFFFFF")
                
                self.add_status(f"✅ Loaded {message_count} messages", "base")
                
            elif storage_mode == 'postgresql' and self.settings['use_persistent_conversations'] and self.graph:
                # Load from PostgreSQL checkpointer
                self.add_status("Loading messages from PostgreSQL...", "base")
                self._load_from_checkpointer(thread_id)
                
            else:
                # No messages to load
                self._append_colored_message(
                    "No messages in this conversation yet. Start chatting below!",
                    "#808080"
                )
            
            self.update_conversation_list()
            self.update_session_info()
            
            self.add_status(f"Switched to conversation: {conv_data['name']}", "base")
            
        except Exception as e:
            logger.exception(f"Error switching conversation: {e}")
            self.add_status(f"❌ Failed to switch conversation: {e}", "error")
            QMessageBox.warning(self, "Error", f"Failed to switch conversation:\n{e}")
    
    def delete_selected_conversation(self):
        """Delete the currently selected conversation(s)."""
        selected_items = self.conversation_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select one or more conversations to delete.")
            return
        
        # Get thread IDs and names of selected conversations
        selected_convs = []
        for item in selected_items:
            thread_id = item.data(Qt.UserRole)
            if thread_id in self.conversations:
                selected_convs.append({
                    'thread_id': thread_id,
                    'name': self.conversations[thread_id]['name']
                })
        
        if not selected_convs:
            return
        
        # Check if trying to delete all conversations
        if len(selected_convs) == len(self.conversations):
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete all conversations. At least one must remain.")
            return
        
        # Build confirmation message
        if len(selected_convs) == 1:
            message = f"Are you sure you want to delete '{selected_convs[0]['name']}'?"
        else:
            conv_names = "\n  • ".join([conv['name'] for conv in selected_convs])
            message = f"Are you sure you want to delete {len(selected_convs)} conversations?\n\n  • {conv_names}"
        
        reply = QMessageBox.question(
            self,
            "Delete Conversation(s)",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Check if current conversation is being deleted
            current_being_deleted = self.current_conversation_id in [conv['thread_id'] for conv in selected_convs]
            
            if current_being_deleted:
                # Switch to a conversation that's not being deleted
                for other_id in self.conversations:
                    if other_id not in [conv['thread_id'] for conv in selected_convs]:
                        for i in range(self.conversation_list.count()):
                            item = self.conversation_list.item(i)
                            if item.data(Qt.UserRole) == other_id:
                                self.switch_conversation(item)
                                break
                        break
            
            # Delete all selected conversations
            deleted_names = []
            for conv in selected_convs:
                thread_id = conv['thread_id']
                
                # Delete from in-memory dict
                if thread_id in self.conversations:
                    del self.conversations[thread_id]
                    deleted_names.append(conv['name'])
                
                # Delete from persistent storage (database or JSON)
                self._delete_conversation_from_storage(thread_id)
            
            self.update_conversation_list()
            
            # Log deletion
            if len(deleted_names) == 1:
                self.add_status(f"Deleted conversation: {deleted_names[0]}", "base")
            else:
                self.add_status(f"Deleted {len(deleted_names)} conversations", "base")
    
    def _delete_conversation_from_storage(self, thread_id: str):
        """Delete a conversation from persistent storage.
        
        Args:
            thread_id: Thread ID of the conversation to delete
        """
        storage_mode = self.settings.get('conversation_storage_mode', 'json')
        
        try:
            if storage_mode == 'json':
                # For JSON storage, just save the updated conversations dict
                # (the conversation was already removed from self.conversations)
                self.save_conversation_history()
                logger.debug(f"Deleted conversation {thread_id} from JSON storage")
            elif storage_mode == 'postgresql':
                # For PostgreSQL storage, delete from the database checkpointer
                if self.graph and hasattr(self.graph, 'checkpointer'):
                    checkpointer = self.graph.checkpointer
                    
                    # Check if checkpointer has a delete method
                    if hasattr(checkpointer, 'delete'):
                        # Use the checkpointer's delete method if available
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # Create config for this thread
                            config = {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "session_id": thread_id
                                }
                            }
                            # Delete the checkpoint
                            loop.run_until_complete(checkpointer.delete(config))
                            logger.info(f"Deleted conversation {thread_id} from PostgreSQL database")
                        finally:
                            loop.close()
                    else:
                        # Checkpointer doesn't have delete method - manual cleanup needed
                        logger.warning(
                            f"PostgreSQL checkpointer doesn't support deletion. "
                            f"Conversation {thread_id} removed from GUI but may remain in database. "
                            f"Manual cleanup may be required."
                        )
                else:
                    logger.warning(f"No checkpointer available for PostgreSQL deletion of {thread_id}")
        except Exception as e:
            logger.error(f"Failed to delete conversation from storage: {e}")
            # Don't raise - the conversation is already removed from memory
            # Show warning to user
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Deletion Warning",
                f"Conversation removed from GUI but may not be fully deleted from database:\n{e}\n\n"
                f"The conversation will not appear in the GUI, but database cleanup may be needed."
            )
    
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
        storage_mode = self.settings.get('conversation_storage_mode', 'json')
        
        # If using JSON storage mode, use in-memory checkpointer (messages saved to JSON)
        if storage_mode == 'json':
            logger.info("📝 Using JSON file storage for conversations (in-memory checkpointer)")
            logger.info("💡 Conversation messages will be saved to conversations.json")
            return MemorySaver()
        
        # If using PostgreSQL storage mode
        if storage_mode == 'postgresql' and self.settings['use_persistent_conversations']:
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
                    self._show_postgresql_setup_guidance()
                    logger.info("📝 Falling back to JSON storage mode")
                    self.settings['conversation_storage_mode'] = 'json'
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
                        # Fall back to JSON storage if connection fails
                        logger.warning(f"⚠️  PostgreSQL connection failed: {e}")
                        self._show_postgresql_setup_guidance()
                        logger.info("📝 Falling back to JSON storage mode")
                        self.settings['conversation_storage_mode'] = 'json'
                        return MemorySaver()
                else:
                    # PostgreSQL not running - show guidance and fall back to JSON
                    logger.warning("⚠️  PostgreSQL is not running")
                    self._show_postgresql_setup_guidance()
                    logger.info("📝 Falling back to JSON storage mode")
                    self.settings['conversation_storage_mode'] = 'json'
                    return MemorySaver()
        else:
            logger.info("📝 Using in-memory checkpointer (persistence disabled in settings)")
            return MemorySaver()
    
    def _show_postgresql_setup_guidance(self):
        """Show guidance for setting up PostgreSQL for conversation storage."""
        logger.info("=" * 80)
        logger.info("PostgreSQL Setup Required")
        logger.info("=" * 80)
        logger.info("To use PostgreSQL for conversation storage, you need to:")
        logger.info("")
        logger.info("1. Install PostgreSQL:")
        logger.info("   • Ubuntu/Debian: sudo apt-get install postgresql")
        logger.info("   • macOS: brew install postgresql")
        logger.info("   • Windows: Download from https://www.postgresql.org/download/")
        logger.info("")
        logger.info("2. Start PostgreSQL service:")
        logger.info("   • Ubuntu/Debian: sudo systemctl start postgresql")
        logger.info("   • macOS: brew services start postgresql")
        logger.info("   • Windows: Start via Services or pg_ctl")
        logger.info("")
        logger.info("3. Create the 'osprey' database:")
        logger.info("   createdb osprey")
        logger.info("")
        logger.info("4. (Optional) Set custom connection via environment variable:")
        logger.info("   export POSTGRESQL_URI='postgresql://user:pass@host:port/dbname'")
        logger.info("")
        logger.info("5. Restart the GUI to use PostgreSQL storage")
        logger.info("=" * 80)
        
        # Also show a GUI dialog
        QTimer.singleShot(1000, self._show_postgresql_setup_dialog)
    
    def _show_postgresql_setup_dialog(self):
        """Show a GUI dialog with PostgreSQL setup instructions."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("PostgreSQL Setup Required")
        msg.setText("PostgreSQL is not available for conversation storage.")
        msg.setInformativeText(
            "To use PostgreSQL for storing conversation messages:\n\n"
            "1. Install PostgreSQL on your system\n"
            "2. Start the PostgreSQL service\n"
            "3. Create the 'osprey' database: createdb osprey\n"
            "4. Restart the GUI\n\n"
            "For now, conversations will be saved to JSON files.\n"
            "See the System Information tab for detailed instructions."
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
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
    
    def _load_and_display_conversation(self, thread_id: str, show_header: bool = True, clear_display: bool = False):
        """
        Unified method to load and display conversation messages.
        
        Args:
            thread_id: Thread ID of the conversation to load
            show_header: Whether to show conversation header
            clear_display: Whether to clear display before loading
        """
        try:
            if clear_display:
                self.conversation_display.clear()
            
            conv_data = self.conversations.get(thread_id)
            if not conv_data:
                logger.warning(f"Conversation {thread_id} not found")
                return
            
            # Show header if requested
            if show_header:
                self._append_colored_message(
                    "=" * 80 + "\n" +
                    f"📂 {conv_data['name']}\n" +
                    "=" * 80 + "\n",
                    "#00FFFF"
                )
            
            # Load from checkpointer if persistent conversations are enabled
            if self.settings['use_persistent_conversations'] and self.graph:
                self._load_from_checkpointer(thread_id)
            elif conv_data.get('messages'):
                # Load from in-memory storage (fallback)
                self._load_from_memory(conv_data['messages'])
            else:
                self._append_colored_message(
                    "Welcome! Start a conversation by typing a message below.",
                    "#00FFFF"
                )
                
        except Exception as e:
            logger.error(f"Failed to load conversation display: {e}")
            self._append_colored_message(f"⚠️ Failed to load conversation: {e}", "#FFA500")
    
    def _load_from_checkpointer(self, thread_id: str):
        """Load messages from checkpointer and display them."""
        try:
            # Create a config with the specific thread_id
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
                
                if messages:
                    message_count = 0
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            if hasattr(msg, 'type') and msg.type == 'human':
                                self._append_colored_message(f"👤 You: {msg.content}", "#D8BFD8")
                                message_count += 1
                            else:
                                self._append_colored_message(f"🤖 {msg.content}", "#FFFFFF")
                                message_count += 1
                    
                    logger.info(f"Loaded {message_count} messages from checkpointer")
                    self.add_status(f"✅ Loaded {message_count} messages from database", "base")
                else:
                    self._append_colored_message("No messages in this conversation yet.", "#808080")
            else:
                self._append_colored_message("No messages in this conversation yet.", "#808080")
                
        except Exception as e:
            logger.error(f"Failed to load from checkpointer: {e}")
            self._append_colored_message(f"⚠️ Could not load conversation history: {e}", "#FFA500")
            self.add_status(f"❌ Failed to load from database: {e}", "error")
    
    def _load_from_memory(self, messages: list):
        """Load messages from in-memory storage and display them."""
        try:
            for msg in messages:
                if msg['type'] == 'user':
                    self._append_colored_message(f"👤 You: {msg['content']}", "#D8BFD8")
                else:
                    self._append_colored_message(msg['content'], "#FFFFFF")
            
            logger.debug(f"Loaded {len(messages)} messages from memory")
        except Exception as e:
            logger.error(f"Failed to load from memory: {e}")
            self._append_colored_message(f"⚠️ Could not load messages: {e}", "#FFA500")
    
    def save_conversation_history(self):
        """Save conversation metadata and optionally messages to persistent storage."""
        storage_mode = self.settings.get('conversation_storage_mode', 'json')
        
        # Only save to JSON file if using JSON storage mode
        if storage_mode != 'json':
            logger.debug(f"Skipping JSON save - using {storage_mode} storage mode (messages stored in database)")
            return
        
        try:
            # Save conversation data to JSON file
            from osprey.utils.config import get_agent_dir
            agent_data_dir = Path(get_agent_dir('conversations'))
            conversations_file = agent_data_dir.parent / 'conversations.json'
            conversations_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert datetime objects to strings for JSON serialization
            serializable_convs = {}
            
            for thread_id, conv_data in self.conversations.items():
                conv_entry = {
                    'name': conv_data['name'],
                    'thread_id': conv_data['thread_id'],
                    'timestamp': conv_data['timestamp'].isoformat(),
                }
                
                # Serialize messages
                messages = []
                for msg in conv_data.get('messages', []):
                    msg_copy = msg.copy()
                    # Convert timestamp to ISO format if present
                    if 'timestamp' in msg_copy and isinstance(msg_copy['timestamp'], datetime):
                        msg_copy['timestamp'] = msg_copy['timestamp'].isoformat()
                    messages.append(msg_copy)
                conv_entry['messages'] = messages
                
                serializable_convs[thread_id] = conv_entry
            
            with open(conversations_file, 'w') as f:
                json.dump(serializable_convs, f, indent=2)
            
            logger.debug(f"Saved conversation data with messages to {conversations_file}")
        except Exception as e:
            logger.warning(f"Failed to save conversation data: {e}")
    
    def load_conversation_history(self):
        """Load conversation metadata and optionally messages from persistent storage."""
        storage_mode = self.settings.get('conversation_storage_mode', 'json')
        
        # Only load from JSON file if using JSON storage mode
        if storage_mode != 'json':
            logger.debug(f"Skipping JSON load - using {storage_mode} storage mode (messages stored in database)")
            return
        
        try:
            from osprey.utils.config import get_agent_dir
            agent_data_dir = Path(get_agent_dir('conversations'))
            conversations_file = agent_data_dir.parent / 'conversations.json'
            
            if not conversations_file.exists():
                logger.debug("No conversation history file found")
                return
            
            with open(conversations_file, 'r') as f:
                serializable_convs = json.load(f)
            
            # Convert ISO format strings back to datetime objects
            for thread_id, conv_data in serializable_convs.items():
                messages = []
                
                # Load messages from JSON
                if 'messages' in conv_data:
                    for msg in conv_data['messages']:
                        msg_copy = msg.copy()
                        # Convert timestamp back to datetime if present
                        if 'timestamp' in msg_copy and isinstance(msg_copy['timestamp'], str):
                            try:
                                msg_copy['timestamp'] = datetime.fromisoformat(msg_copy['timestamp'])
                            except:
                                msg_copy['timestamp'] = datetime.now()
                        messages.append(msg_copy)
                
                self.conversations[thread_id] = {
                    'name': conv_data['name'],
                    'thread_id': conv_data['thread_id'],
                    'timestamp': datetime.fromisoformat(conv_data['timestamp']),
                    'messages': messages
                }
            
            logger.info(f"Loaded {len(self.conversations)} conversation(s) with messages from JSON file")
            
        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
    
    def load_current_conversation_display(self):
        """Load the current conversation messages into the display."""
        if not self.current_conversation_id or self.current_conversation_id not in self.conversations:
            return
        
        # Use unified loading method
        self._load_and_display_conversation(
            self.current_conversation_id,
            show_header=True,
            clear_display=True
        )
    
    def show_settings(self):
        """Show settings dialog (modeless)."""
        dialog = SettingsDialog(self, "Framework Settings", self.settings)
        
        # Connect the accepted signal to handle settings update
        def on_settings_accepted():
            old_routing_settings = {
                k: v for k, v in self.settings.items()
                if k.startswith(('enable_routing', 'cache_', 'enable_advanced', 'enable_adaptive',
                                'enable_probabilistic', 'enable_event', 'enable_semantic',
                                'semantic_', 'topic_', 'max_context', 'orchestration_', 'analytics_'))
            }
            
            self.settings = dialog.get_settings()
            
            # Check if routing settings changed
            new_routing_settings = {
                k: v for k, v in self.settings.items()
                if k.startswith(('enable_routing', 'cache_', 'enable_advanced', 'enable_adaptive',
                                'enable_probabilistic', 'enable_event', 'enable_semantic',
                                'semantic_', 'topic_', 'max_context', 'orchestration_', 'analytics_'))
            }
            
            routing_settings_changed = old_routing_settings != new_routing_settings
            
            # Update base config with new settings
            if self.base_config:
                agent_control_defaults = self.base_config["configurable"].get("agent_control_defaults", {})
                agent_control_defaults.update(self.settings)
                self.base_config["configurable"]["agent_control_defaults"] = agent_control_defaults
                
                # Apply development/debug settings to the configuration
                development_config = self.base_config["configurable"].get("development", {})
                development_config["debug"] = self.settings.get('debug_mode', False)
                development_config["raise_raw_errors"] = self.settings.get('raise_raw_errors', False)
                
                # Apply prompt settings
                prompts_config = development_config.get("prompts", {})
                prompts_config["print_all"] = self.settings.get('print_prompts', False)
                prompts_config["show_all"] = self.settings.get('show_prompts', False)
                prompts_config["latest_only"] = self.settings.get('prompts_latest_only', True)
                development_config["prompts"] = prompts_config
                
                self.base_config["configurable"]["development"] = development_config
            
            # Apply logging level changes immediately
            import logging
            debug_mode = self.settings.get('debug_mode', False)
            new_level = logging.DEBUG if debug_mode else logging.INFO
            
            # Update root logger level
            root_logger = logging.getLogger()
            root_logger.setLevel(new_level)
            
            # Update all existing loggers
            for logger_name in logging.Logger.manager.loggerDict:
                logger_obj = logging.getLogger(logger_name)
                if isinstance(logger_obj, logging.Logger):
                    logger_obj.setLevel(new_level)
            
            # Apply terminal suppression setting and update GUI handler level
            suppress_terminal = self.settings.get('suppress_terminal_output', False)
            from osprey.utils.logger import set_gui_output_callback
            set_gui_output_callback(self.gui_output_signal.emit_output, suppress_terminal=suppress_terminal)
            
            # Update GUI handler level to match the new logging level
            for handler in root_logger.handlers:
                from osprey.utils.logger import GUIHandler
                if isinstance(handler, GUIHandler):
                    handler.setLevel(new_level)
                    logger.debug(f"Updated GUI handler level to {logging.getLevelName(new_level)}")
            
            # Save settings to config.yml file for persistence
            self._save_settings_to_config()
            
            self.update_session_info()
            level_name = "DEBUG" if debug_mode else "INFO"
            self.add_status(f"Settings updated and saved - Logging level: {level_name}", "base")
            
            # Reinitialize router if routing settings changed
            if routing_settings_changed:
                self.add_status("Routing settings changed - reinitializing router...", "base")
                self._initialize_router()
                self.add_status("✅ Router reinitialized with new settings", "base")
            
            # Inform user about settings application
            message = (
                f"Settings have been updated and saved to config file.\n\n"
                f"Logging level: {level_name}\n\n"
                "Note: Settings will persist across GUI restarts.\n\n"
                "Agent control settings will apply to ALL future messages in ALL conversations "
                "(both new and existing)."
            )
            
            if routing_settings_changed:
                message += "\n\n⚠️ Routing settings changed - router has been reinitialized."
            
            QMessageBox.information(
                self,
                "Settings Updated",
                message
            )
        
        dialog.accepted.connect(on_settings_accepted)
        dialog.show()  # Show modeless dialog
    
    def _save_settings_to_config(self):
        """Save current settings back to the config.yml file for persistence."""
        try:
            if not self.config_path:
                logger.warning("No config path available, settings not saved to file")
                return
            
            import yaml
            from pathlib import Path
            
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_file}")
                return
            
            # Read current config
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Update development section
            if 'development' not in config_data:
                config_data['development'] = {}
            
            debug_mode = self.settings.get('debug_mode', False)
            config_data['development']['debug'] = debug_mode
            config_data['development']['raise_raw_errors'] = self.settings.get('raise_raw_errors', False)
            
            if 'prompts' not in config_data['development']:
                config_data['development']['prompts'] = {}
            
            config_data['development']['prompts']['print_all'] = self.settings.get('print_prompts', False)
            config_data['development']['prompts']['show_all'] = self.settings.get('show_prompts', False)
            config_data['development']['prompts']['latest_only'] = self.settings.get('prompts_latest_only', True)
            
            # Update execution_control section for agent control settings
            if 'execution_control' not in config_data:
                config_data['execution_control'] = {}
            
            if 'agent_control' not in config_data['execution_control']:
                config_data['execution_control']['agent_control'] = {}
            
            config_data['execution_control']['agent_control']['task_extraction_bypass_enabled'] = self.settings.get('task_extraction_bypass_enabled', False)
            config_data['execution_control']['agent_control']['capability_selection_bypass_enabled'] = self.settings.get('capability_selection_bypass_enabled', False)
            
            if 'epics' not in config_data['execution_control']:
                config_data['execution_control']['epics'] = {}
            
            config_data['execution_control']['epics']['writes_enabled'] = self.settings.get('epics_writes_enabled', False)
            
            # Update approval section
            if 'approval' not in config_data:
                config_data['approval'] = {}
            
            config_data['approval']['global_mode'] = self.settings.get('approval_global_mode', 'selective')
            
            if 'capabilities' not in config_data['approval']:
                config_data['approval']['capabilities'] = {}
            
            if 'python_execution' not in config_data['approval']['capabilities']:
                config_data['approval']['capabilities']['python_execution'] = {}
            
            config_data['approval']['capabilities']['python_execution']['enabled'] = self.settings.get('python_execution_approval_enabled', True)
            config_data['approval']['capabilities']['python_execution']['mode'] = self.settings.get('python_execution_approval_mode', 'all_code')
            
            if 'memory' not in config_data['approval']['capabilities']:
                config_data['approval']['capabilities']['memory'] = {}
            
            config_data['approval']['capabilities']['memory']['enabled'] = self.settings.get('memory_approval_enabled', True)
            
            # Update execution limits
            if 'limits' not in config_data['execution_control']:
                config_data['execution_control']['limits'] = {}
            
            config_data['execution_control']['limits']['max_reclassifications'] = self.settings.get('max_reclassifications', 1)
            config_data['execution_control']['limits']['max_planning_attempts'] = self.settings.get('max_planning_attempts', 2)
            config_data['execution_control']['limits']['max_step_retries'] = self.settings.get('max_step_retries', 0)
            config_data['execution_control']['limits']['max_execution_time_seconds'] = self.settings.get('max_execution_time_seconds', 300)
            config_data['execution_control']['limits']['max_concurrent_classifications'] = self.settings.get('max_concurrent_classifications', 5)
            
            # Update routing section for Phase 2.4 settings
            if 'routing' not in config_data:
                config_data['routing'] = {}
            
            # Cache settings
            if 'cache' not in config_data['routing']:
                config_data['routing']['cache'] = {}
            
            config_data['routing']['cache']['enabled'] = self.settings.get('enable_routing_cache', True)
            config_data['routing']['cache']['max_size'] = self.settings.get('cache_max_size', 100)
            config_data['routing']['cache']['ttl_seconds'] = self.settings.get('cache_ttl_seconds', 3600.0)
            config_data['routing']['cache']['similarity_threshold'] = self.settings.get('cache_similarity_threshold', 0.85)
            
            # Advanced invalidation settings
            if 'advanced_invalidation' not in config_data['routing']:
                config_data['routing']['advanced_invalidation'] = {}
            
            config_data['routing']['advanced_invalidation']['enabled'] = self.settings.get('enable_advanced_invalidation', True)
            config_data['routing']['advanced_invalidation']['adaptive_ttl'] = self.settings.get('enable_adaptive_ttl', True)
            config_data['routing']['advanced_invalidation']['probabilistic_expiration'] = self.settings.get('enable_probabilistic_expiration', True)
            config_data['routing']['advanced_invalidation']['event_driven'] = self.settings.get('enable_event_driven_invalidation', True)
            
            # Semantic analysis settings
            if 'semantic_analysis' not in config_data['routing']:
                config_data['routing']['semantic_analysis'] = {}
            
            config_data['routing']['semantic_analysis']['enabled'] = self.settings.get('enable_semantic_analysis', True)
            config_data['routing']['semantic_analysis']['similarity_threshold'] = self.settings.get('semantic_similarity_threshold', 0.5)
            config_data['routing']['semantic_analysis']['topic_similarity_threshold'] = self.settings.get('topic_similarity_threshold', 0.6)
            config_data['routing']['semantic_analysis']['max_context_history'] = self.settings.get('max_context_history', 20)
            
            # Orchestration settings
            if 'orchestration' not in config_data['routing']:
                config_data['routing']['orchestration'] = {}
            
            config_data['routing']['orchestration']['max_parallel'] = self.settings.get('orchestration_max_parallel', 3)
            
            # Analytics settings
            if 'analytics' not in config_data['routing']:
                config_data['routing']['analytics'] = {}
            
            config_data['routing']['analytics']['max_history'] = self.settings.get('analytics_max_history', 1000)
            
            # Feedback settings
            if 'feedback' not in config_data['routing']:
                config_data['routing']['feedback'] = {}
            
            config_data['routing']['feedback']['enabled'] = self.settings.get('enable_routing_feedback', True)
            
            # GUI-specific settings
            if 'gui' not in config_data:
                config_data['gui'] = {}
            
            config_data['gui']['use_persistent_conversations'] = self.settings.get('use_persistent_conversations', True)
            config_data['gui']['conversation_storage_mode'] = self.settings.get('conversation_storage_mode', 'json')
            config_data['gui']['redirect_output_to_gui'] = self.settings.get('redirect_output_to_gui', True)
            config_data['gui']['group_system_messages'] = self.settings.get('group_system_messages', True)
            config_data['gui']['suppress_terminal_output'] = self.settings.get('suppress_terminal_output', False)
            
            # Write back to unified config file
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)
            
            logger.info(f"Settings saved to {config_file}")
            
            # ALSO update individual project config files if we're using unified config
            if 'unified_config' in str(config_file) and self.discovered_projects:
                logger.info("Updating individual project config files...")
                for project in self.discovered_projects:
                    try:
                        project_config_path = Path(project['config_path'])
                        if project_config_path.exists():
                            with open(project_config_path, 'r') as f:
                                project_config = yaml.safe_load(f) or {}
                            
                            # Update development section in project config
                            if 'development' not in project_config:
                                project_config['development'] = {}
                            
                            project_config['development']['debug'] = debug_mode
                            project_config['development']['raise_raw_errors'] = self.settings.get('raise_raw_errors', False)
                            
                            # Update prompts section
                            if 'prompts' not in project_config['development']:
                                project_config['development']['prompts'] = {}
                            
                            project_config['development']['prompts']['print_all'] = self.settings.get('print_prompts', False)
                            project_config['development']['prompts']['show_all'] = self.settings.get('show_prompts', False)
                            project_config['development']['prompts']['latest_only'] = self.settings.get('prompts_latest_only', True)
                            
                            # Write back to project config
                            with open(project_config_path, 'w') as f:
                                yaml.dump(project_config, f, default_flow_style=False, sort_keys=False, indent=2)
                            
                            logger.info(f"Updated {project['name']} config: {project_config_path}")
                    except Exception as e:
                        logger.warning(f"Failed to update {project['name']} config: {e}")
            
        except Exception as e:
            logger.error(f"Failed to save settings to config file: {e}")
            QMessageBox.warning(
                self,
                "Save Warning",
                f"Settings applied but could not be saved to config file:\n{e}\n\n"
                "Settings will be lost when GUI is restarted."
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
        
        # Get GUI version directly from module-level variable
        gui_version = __version__
        
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