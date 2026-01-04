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

import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from PyQt5.QtCore import QEvent, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QTextCursor, QTextOption
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from osprey.interfaces.pyqt.about_dialog import show_about_dialog
from osprey.interfaces.pyqt.capability_registry import CapabilityRegistry
from osprey.interfaces.pyqt.collapsible_widget import MessageGroupWidget
from osprey.interfaces.pyqt.conversation_display import ConversationDisplayManager
from osprey.interfaces.pyqt.conversation_history import ConversationHistory
from osprey.interfaces.pyqt.conversation_management import ConversationManagement
from osprey.interfaces.pyqt.conversation_manager import ConversationManager
from osprey.interfaces.pyqt.enums import Colors, EventTypes, LLMEventType
from osprey.interfaces.pyqt.event_bus import EventBus
from osprey.interfaces.pyqt.gui_components.tabs import (
    AnalyticsTab,
    LLMDetailsTab,
    MemoryTab,
    ProjectsTab,
    SystemInfoTab,
    ToolUsageTab,
)
from osprey.interfaces.pyqt.gui_preferences import GUIPreferences
from osprey.interfaces.pyqt.gui_utils import create_dark_palette
from osprey.interfaces.pyqt.help_dialog import show_help_dialog
from osprey.interfaces.pyqt.message_formatter import MessageFormatter
from osprey.interfaces.pyqt.message_handlers import MessageHandlers
from osprey.interfaces.pyqt.model_preferences import ModelPreferencesStore
from osprey.interfaces.pyqt.model_preferences_manager import (
    ModelPreferencesUIHandler as ModelPrefMgr,
)
from osprey.interfaces.pyqt.multi_project_router import MultiProjectRouter
from osprey.interfaces.pyqt.orchestration_ui import OrchestrationUIHandler
from osprey.interfaces.pyqt.project_control import ProjectControlManager
from osprey.interfaces.pyqt.project_manager import ProjectManager
from osprey.interfaces.pyqt.routing_ui import RoutingUIHandler
from osprey.interfaces.pyqt.runtime_overrides import RuntimeOverrideManager
from osprey.interfaces.pyqt.settings_dialog import SettingsDialog
from osprey.interfaces.pyqt.settings_manager import SettingsManager
from osprey.interfaces.pyqt.worker_thread import AgentWorker
from osprey.registry.manager import get_registry
from osprey.utils.config import get_config_value, get_full_configuration
from osprey.utils.logger import get_logger

# Load environment variables after imports
load_dotenv()

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


class OspreyGUI(QMainWindow):
    """Main Qt GUI window for Osprey Framework."""

    def __init__(self, config_path=None):
        super().__init__()

        # Check for gui_config.yml and provide helpful error if missing
        if config_path is None:
            gui_config = Path(__file__).parent / "gui_config.yml"
            if not gui_config.exists():
                # Show error dialog before initializing GUI
                self._show_missing_config_error(gui_config)
                raise FileNotFoundError(f"Required configuration file not found: {gui_config}")
            config_path = str(gui_config)

        self.config_path = config_path
        self.graph = None
        self.gateway = None
        self.thread_id = None
        self.current_conversation_id = None  # Initialize early to avoid AttributeError
        self.base_config = None
        self.worker = None
        self._initialized = False
        self.discovered_projects = []  # Store discovered projects (cached)
        self._projects_cache_valid = False  # Track if cache is valid
        self.model_preferences = ModelPreferencesStore()  # Model preferences manager

        # Phase 1 Components - Multi-Project Support
        # Initialize ProjectManager with project_root from config if available
        project_search_paths = None
        if config_path:
            try:
                import yaml

                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                    if "project_root" in config_data:
                        project_root = Path(config_data["project_root"])
                        if project_root.exists():
                            project_search_paths = [project_root]
                            logger.info(f"Using project_root from config: {project_root}")
            except Exception as e:
                logger.warning(f"Could not read project_root from config: {e}")

        self.project_manager = ProjectManager(project_search_paths=project_search_paths)
        self.capability_registry = CapabilityRegistry()
        # Initialize router - will be configured with settings after UI setup
        self.router = None

        # Initialize routing UI handler
        self.routing_ui = None  # Will be initialized after UI setup

        # Initialize orchestration UI handler
        self.orchestration_ui = None  # Will be initialized after UI setup

        # Initialize conversation display manager
        self.conversation_display_mgr = None  # Will be initialized after UI setup

        # Initialize project control manager
        self.project_control_mgr = None  # Will be initialized after UI setup

        # Initialize message handlers
        self.message_handlers = None  # Will be initialized after UI setup

        # Initialize conversation management
        self.conversation_mgmt = None  # Will be initialized after UI setup

        # Initialize conversation history
        self.conversation_history = None  # Will be initialized after UI setup

        # Initialize model preferences manager
        self.model_pref_mgr = None  # Will be initialized after UI setup

        self._agent_processing = False  # Track if agent is currently processing
        self._queued_message = None  # Store one queued message to process after completion

        # NEW: Separate GUI preferences and runtime overrides
        self.gui_preferences = (
            GUIPreferences()
        )  # User UI settings (saved to ~/.osprey/gui_preferences.yml)
        self.runtime_overrides = RuntimeOverrideManager()  # Execution behavior (in-memory only)

        # Settings Manager (for backward compatibility during transition)
        # Will load from config file if available
        gui_config_path = (
            Path(__file__).parent / "gui_config.yml"
            if not self.config_path
            else Path(self.config_path)
        )
        self.settings_manager = SettingsManager(
            config_path=gui_config_path if gui_config_path.exists() else None
        )

        # Initialize runtime overrides from settings manager
        self.runtime_overrides.set_overrides(self.settings_manager.get_all_settings())

        # Backward compatibility: provide dict-like interface
        self.settings = self.settings_manager.get_all_settings()

        # Create signal emitter for thread-safe GUI output
        self.gui_output_signal = GUIOutputSignal()
        self.gui_output_signal.output_signal.connect(self.append_to_system_info)

        # CRITICAL: Set up GUI output redirection AFTER settings are loaded
        # This must happen BEFORE setup_ui() to capture all logging from the start
        from osprey.utils.logger import set_gui_output_callback

        # Use the signal emitter's method for thread-safe GUI updates
        # Start with suppress_terminal=False to show messages in both places initially
        set_gui_output_callback(self.gui_output_signal.emit_output, suppress_terminal=False)

        # Conversation history management using ConversationManager
        # Initialize AFTER settings are defined
        self.conversation_manager = ConversationManager(
            storage_mode=self.settings_manager.get("conversation_storage_mode", "json")
        )
        self.conversation_lock_file = None  # For multi-instance locking

        # Color mapping for components
        self.component_colors = {
            "base": "#FFFFFF",
            "context": "#AFD7FF",
            "router": "#FF00FF",
            "orchestrator": "#00FFFF",
            "monitor": "#CD8500",
            "classifier": "#FFA07A",
            "task_extraction": "#D8BFD8",
            "error": "#FF0000",
            "gateway": "#FFA07A",
            "approval": "#FFA07A",
            "time_range_parsing": "#1E90FF",
            "memory": "#FFA07A",
            "python": "#FFA07A",
            "respond": "#D8BFD8",
            "clarify": "#D8BFD8",
        }

        try:
            # GUI output redirection is now set up in __init__ BEFORE this point
            # to capture all early logging messages

            self.setup_ui()
            logger.info("UI setup complete")
            logger.info("GUI output redirection enabled")

            # Initialize routing UI handler after UI setup
            self.routing_ui = RoutingUIHandler(self)

            # Initialize orchestration UI handler after UI setup
            self.orchestration_ui = OrchestrationUIHandler(self)

            # Initialize conversation display manager after UI setup
            self.conversation_display_mgr = ConversationDisplayManager(self)

            # Initialize project control manager after UI setup
            self.project_control_mgr = ProjectControlManager(self)

            # Initialize event bus for refactored architecture
            self.event_bus = EventBus()

            # Initialize message handlers with event bus after UI setup
            self.message_handlers = MessageHandlers(
                event_bus=self.event_bus,
                conversation_id_provider=lambda: self.current_conversation_id,
            )

            # Setup event subscriptions
            self._setup_event_subscriptions()

            # Initialize conversation management after UI setup
            self.conversation_mgmt = ConversationManagement(self)

            # Initialize conversation history after UI setup
            self.conversation_history = ConversationHistory(self)

            # Initialize model preferences manager after UI setup
            self.model_pref_mgr = ModelPrefMgr(self)

            # Router will be initialized after framework initialization
            # (moved to initialize_framework method)

            QTimer.singleShot(100, self.initialize_framework)
            logger.info("Framework initialization scheduled")
        except Exception as e:
            logger.exception(f"Error during GUI initialization: {e}")
            raise

    def _show_missing_config_error(self, gui_config_path: Path):
        """Show user-friendly error dialog for missing gui_config.yml.

        Args:
            gui_config_path: Path where gui_config.yml should be located
        """
        from PyQt5.QtWidgets import QMessageBox

        # Create a minimal QApplication if one doesn't exist yet
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        error_msg = (
            f"<h3>❌ Missing Configuration File</h3>"
            f"<p>The GUI requires <b>gui_config.yml</b> to start, but it was not found.</p>"
            f"<p><b>Expected location:</b><br><code>{gui_config_path}</code></p>"
            f"<hr>"
            f"<p><b>To fix this issue:</b></p>"
            f"<ol>"
            f"<li>Copy the example configuration file:<br>"
            f"<code>cp gui_config.yml.example gui_config.yml</code></li>"
            f"<li>Edit <code>gui_config.yml</code> to configure your LLM provider</li>"
            f"<li>Set the required environment variable (e.g., <code>ARGO_API_KEY</code>)</li>"
            f"<li>Restart the GUI</li>"
            f"</ol>"
            f"<hr>"
            f"<p><b>What to configure:</b></p>"
            f"<ul>"
            f"<li><b>models.classifier</b> - LLM model for routing queries between projects</li>"
            f"<li><b>api.providers</b> - API credentials for your LLM provider</li>"
            f"</ul>"
            f"<p>See <code>gui_config.yml.example</code> for a complete template with all available options.</p>"
        )

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Osprey GUI - Configuration Required")
        msg_box.setText(error_msg)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _setup_event_subscriptions(self):
        """Subscribe to event bus events."""
        # Message events
        self.event_bus.subscribe(EventTypes.MESSAGE_RECEIVED, self._handle_message_received)
        self.event_bus.subscribe(EventTypes.STATUS_UPDATE, self._handle_status_update)
        self.event_bus.subscribe(EventTypes.ERROR_OCCURRED, self._handle_error)
        self.event_bus.subscribe(EventTypes.PROCESSING_COMPLETE, self._handle_processing_complete)

        # LLM events
        self.event_bus.subscribe(EventTypes.LLM_DETAIL, self._handle_llm_detail)
        self.event_bus.subscribe(EventTypes.TOOL_USAGE, self._handle_tool_usage)

        # Conversation events
        self.event_bus.subscribe(EventTypes.CONVERSATION_UPDATED, self._handle_conversation_updated)

        # Custom display events
        self.event_bus.subscribe("display_message", self._handle_display_message)
        self.event_bus.subscribe("display_error", self._handle_display_error)
        self.event_bus.subscribe("update_status_bar", self._handle_update_status_bar)
        self.event_bus.subscribe(
            "save_conversation_history", self._handle_save_conversation_history
        )

    def _handle_message_received(self, data: dict):
        """Handle message received event from event bus."""
        conversation_id = data["conversation_id"]
        message_type = data["message_type"]
        content = data["content"]

        self.conversation_manager.add_message(conversation_id, message_type, content)

    def _handle_status_update(self, data: dict):
        """Handle status update event from event bus."""
        self.add_status(data["status"], data["component"], model_info=data.get("model_info", {}))

    def _handle_error(self, data: dict):
        """Handle error event from event bus."""
        error = data["error"]
        self.conversation_display.append(f"\n❌ Error: {error}\n")
        self.add_status(f"Error: {error}", "error")
        QMessageBox.warning(self, "Processing Error", f"An error occurred:\n{error}")

    def _handle_processing_complete(self, data: dict):
        """Handle processing complete event from event bus."""
        self._agent_processing = False
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)

        if self._queued_message is not None:
            queued = self._queued_message
            self._queued_message = None

            self._append_colored_message("▶️ Processing queued message...", "#00FFFF")

            self.input_field.setPlainText(queued)
            QTimer.singleShot(100, self.send_message)
        else:
            self.input_field.setFocus()
            self.status_bar.showMessage("✅ Ready - Type your message")

    def _handle_llm_detail(self, data: dict):
        """Handle LLM detail event from event bus."""
        detail = data["detail"]
        event_type = data["event_type"]
        timestamp = data["timestamp"]

        # Get color for event type
        try:
            event_enum = LLMEventType[event_type.upper()]
            color = event_enum.color
        except (KeyError, AttributeError):
            color = LLMEventType.BASE.color

        # Add timestamp
        self._append_formatted_text(
            f"[{timestamp}] ",
            "#808080",
            self.llm_details_tab.llm_details_display,
            prefix="",
            suffix="",
        )
        # Add event type tag
        self._append_formatted_text(
            f"[{event_type.upper()}] ",
            color,
            self.llm_details_tab.llm_details_display,
            prefix="",
            suffix="",
        )
        # Add detail
        self._append_formatted_text(
            detail, "#FFFFFF", self.llm_details_tab.llm_details_display, prefix="", suffix="\n"
        )

    def _handle_tool_usage(self, data: dict):
        """Handle tool usage event from event bus."""
        tool_name = data["tool_name"]
        reasoning = data["reasoning"]
        timestamp = data["timestamp"]

        # Add newline separator
        self._append_formatted_text(
            "", "#FFFFFF", self.tool_usage_tab.tool_usage_display, prefix="\n", suffix=""
        )

        # Add timestamp
        self._append_formatted_text(
            f"[{timestamp}] ",
            Colors.TIMESTAMP,
            self.tool_usage_tab.tool_usage_display,
            prefix="",
            suffix="",
        )

        # Add capability label and name
        self._append_formatted_text(
            "Capability: ",
            Colors.TOOL_LABEL,
            self.tool_usage_tab.tool_usage_display,
            prefix="",
            suffix="",
        )
        self._append_formatted_text(
            tool_name,
            Colors.TOOL_CAPABILITY,
            self.tool_usage_tab.tool_usage_display,
            prefix="",
            suffix="\n",
        )

        # Add reasoning lines with appropriate colors
        line_colors = {
            "✅": Colors.TOOL_SUCCESS,
            "❌": Colors.TOOL_FAILURE,
            "⏱️": Colors.TOOL_TIMING,
        }

        for line in reasoning.split("\n"):
            if not line.strip():
                continue

            # Determine color based on line prefix
            color = "#FFFFFF"  # Default white
            for prefix, prefix_color in line_colors.items():
                if line.startswith(prefix):
                    color = prefix_color
                    break

            self._append_formatted_text(
                line, color, self.tool_usage_tab.tool_usage_display, prefix="", suffix="\n"
            )

        # Add separator
        self._append_formatted_text(
            "=" * 80,
            Colors.SEPARATOR,
            self.tool_usage_tab.tool_usage_display,
            prefix="",
            suffix="\n",
        )

    def _handle_conversation_updated(self, data: dict):
        """Handle conversation updated event from event bus."""
        self.conversation_display_mgr.update_conversation_list()

    def _handle_display_message(self, data: dict):
        """Handle display message event from event bus."""
        # Auto-open plots for new agent messages (not historical ones)
        auto_open = data.get("auto_open_plots", True)  # Default to True for new messages
        self._append_colored_message(data["message"], data["color"], auto_open_plots=auto_open)

    def _handle_display_error(self, data: dict):
        """Handle display error event from event bus."""
        error = data["error"]
        self.conversation_display.append(f"\n❌ Error: {error}\n")
        QMessageBox.warning(self, "Processing Error", f"An error occurred:\n{error}")

    def _handle_update_status_bar(self, data: dict):
        """Handle update status bar event from event bus."""
        self.status_bar.showMessage(data["message"])

    def _handle_save_conversation_history(self, data: dict):
        """Handle save conversation history event from event bus."""
        self.save_conversation_history()

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
                enable_cache=self.settings_manager.get("enable_routing_cache", True),
                cache_max_size=self.settings_manager.get("cache_max_size", 100),
                cache_ttl_seconds=self.settings_manager.get("cache_ttl_seconds", 3600.0),
                cache_similarity_threshold=self.settings_manager.get(
                    "cache_similarity_threshold", 0.85
                ),
                # Advanced invalidation settings
                enable_advanced_invalidation=self.settings_manager.get(
                    "enable_advanced_invalidation", True
                ),
                enable_adaptive_ttl=self.settings_manager.get("enable_adaptive_ttl", True),
                enable_probabilistic_expiration=self.settings_manager.get(
                    "enable_probabilistic_expiration", True
                ),
                enable_event_driven_invalidation=self.settings_manager.get(
                    "enable_event_driven_invalidation", True
                ),
                # Conversation context settings
                enable_conversation_context=True,
                enable_semantic_context=self.settings_manager.get("enable_semantic_context", False),
                context_max_history=self.settings_manager.get("max_context_history", 20),
                semantic_similarity_threshold=self.settings_manager.get(
                    "semantic_similarity_threshold", 0.5
                ),
                semantic_topic_threshold=self.settings_manager.get("semantic_topic_threshold", 0.6),
                # Orchestration settings
                enable_orchestration=True,
                orchestration_max_parallel=self.settings_manager.get(
                    "orchestration_max_parallel", 3
                ),
                # Analytics settings
                enable_analytics=self.settings_manager.get("enable_analytics", True),
                analytics_max_history=self.settings_manager.get("analytics_max_history", 1000),
                # Feedback settings
                enable_feedback=self.settings_manager.get("enable_routing_feedback", True),
                feedback_max_history=1000,
            )

            logger.info("MultiProjectRouter initialized successfully with all Phase 2.4 settings")

        except Exception as e:
            logger.exception(f"Failed to initialize router: {e}")
            # Create a basic router as fallback
            self.router = MultiProjectRouter(self.capability_registry)

    def eventFilter(self, obj, event):
        """Event filter to handle Enter/Shift+Enter in input field."""

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

        # LLM Conversation Details tab - using extracted tab class
        self.llm_details_tab = LLMDetailsTab(self)
        tab_widget.addTab(self.llm_details_tab, "LLM Details")

        # LLM Tool Usage tab - using extracted tab class
        self.tool_usage_tab = ToolUsageTab(self)
        tab_widget.addTab(self.tool_usage_tab, "Tool Usage")

        # Discovered Projects tab - using extracted tab class
        self.projects_tab = ProjectsTab(self)
        tab_widget.addTab(self.projects_tab, "Discovered Projects")

        # System Information tab - using extracted tab class
        self.system_info_tab = SystemInfoTab(self)
        tab_widget.addTab(self.system_info_tab, "System Information")

        # Memory Monitoring tab - using extracted tab class
        self.memory_tab = MemoryTab(self)
        tab_widget.addTab(self.memory_tab, "💾 Memory")

        # Analytics Dashboard tab - using extracted tab class
        self.analytics_tab = AnalyticsTab(self)
        tab_widget.addTab(self.analytics_tab, "📊 Analytics")

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
        self.conversation_list.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
        self.conversation_list.setSelectionMode(
            QListWidget.ExtendedSelection
        )  # Enable multi-selection
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
        self.conversation_display.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
        self.conversation_display.setHtml(
            '<span style="color: #00FFFF;">Welcome to Osprey Framework</span><br><span style="color: #FFFFFF;">Initializing system...</span>'
        )
        conversation_layout.addWidget(self.conversation_display)

        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText(
            "Ask anything... (Press Enter to send, Shift+Enter for new line)"
        )
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
        self.routing_explanation_label.setStyleSheet(
            "color: #808080; font-size: 9px; font-style: italic;"
        )
        self.routing_explanation_label.setWordWrap(True)
        self.routing_explanation_label.setVisible(False)
        layout.addWidget(self.routing_explanation_label)

        # Cache statistics (initially hidden)
        self.cache_stats_label = QLabel("")
        self.cache_stats_label.setStyleSheet(
            "color: #00FF00; font-size: 9px; font-family: monospace;"
        )
        self.cache_stats_label.setWordWrap(True)
        self.cache_stats_label.setVisible(False)
        layout.addWidget(self.cache_stats_label)

        # Conversation context summary (initially hidden)
        self.context_summary_label = QLabel("")
        self.context_summary_label.setStyleSheet(
            "color: #87CEEB; font-size: 9px; font-family: monospace;"
        )
        self.context_summary_label.setWordWrap(True)
        self.context_summary_label.setVisible(False)
        layout.addWidget(self.context_summary_label)

        # Cache control buttons
        cache_button_layout = QHBoxLayout()

        self.clear_cache_button = QPushButton("🗑️ Clear Cache")
        self.clear_cache_button.setStyleSheet(
            "background-color: #4A5568; color: #FFFFFF; font-size: 9px;"
        )
        self.clear_cache_button.setMaximumHeight(25)
        self.clear_cache_button.clicked.connect(self.clear_routing_cache)
        self.clear_cache_button.setToolTip("Clear routing cache to force fresh routing decisions")
        cache_button_layout.addWidget(self.clear_cache_button)

        self.show_cache_stats_button = QPushButton("📊 Stats")
        self.show_cache_stats_button.setStyleSheet(
            "background-color: #4A5568; color: #FFFFFF; font-size: 9px;"
        )
        self.show_cache_stats_button.setMaximumHeight(25)
        self.show_cache_stats_button.setCheckable(True)
        self.show_cache_stats_button.clicked.connect(self.toggle_cache_stats)
        self.show_cache_stats_button.setToolTip("Show/hide cache statistics")
        cache_button_layout.addWidget(self.show_cache_stats_button)

        layout.addLayout(cache_button_layout)

        # Conversation context control buttons
        context_button_layout = QHBoxLayout()

        self.clear_context_button = QPushButton("🔄 Clear Context")
        self.clear_context_button.setStyleSheet(
            "background-color: #4A5568; color: #FFFFFF; font-size: 9px;"
        )
        self.clear_context_button.setMaximumHeight(25)
        self.clear_context_button.clicked.connect(self.clear_conversation_context)
        self.clear_context_button.setToolTip(
            "Clear conversation context to start fresh topic detection"
        )
        context_button_layout.addWidget(self.clear_context_button)

        self.show_context_button = QPushButton("💬 Context")
        self.show_context_button.setStyleSheet(
            "background-color: #4A5568; color: #FFFFFF; font-size: 9px;"
        )
        self.show_context_button.setMaximumHeight(25)
        self.show_context_button.setCheckable(True)
        self.show_context_button.clicked.connect(self.toggle_context_display)
        self.show_context_button.setToolTip("Show/hide conversation context")
        context_button_layout.addWidget(self.show_context_button)

        layout.addLayout(context_button_layout)

        return panel

    def _update_project_selector(self):
        """Update the project selector (delegated to ProjectControlManager)."""
        self.project_control_mgr.update_project_selector()

    def on_project_selected(self, index):
        """Handle project selection (delegated to ProjectControlManager)."""
        self.project_control_mgr.on_project_selected(index)

    def clear_routing_cache(self):
        """Clear the routing cache (delegated to ProjectControlManager)."""
        self.project_control_mgr.clear_routing_cache()

    def toggle_cache_stats(self):
        """Toggle cache statistics (delegated to ProjectControlManager)."""
        self.project_control_mgr.toggle_cache_stats()

    def _update_cache_statistics(self):
        """Update cache statistics (delegated to ProjectControlManager)."""
        self.project_control_mgr.update_cache_statistics()

    def clear_conversation_context(self):
        """Clear conversation context (delegated to ProjectControlManager)."""
        self.project_control_mgr.clear_conversation_context()

    def toggle_context_display(self):
        """Toggle context display (delegated to ProjectControlManager)."""
        self.project_control_mgr.toggle_context_display()

    def _update_context_display(self):
        """Update context display (delegated to ProjectControlManager)."""
        self.project_control_mgr.update_context_display()

    def toggle_project_enabled(self, project_name, enabled):
        """Enable or disable a project (delegated to ProjectControlManager)."""
        self.project_control_mgr.toggle_project_enabled(project_name, enabled)

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
        self.status_log.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
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
        self.group_messages_toggle.setChecked(
            self.settings_manager.get("group_system_messages", True)
        )
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
        self.session_info.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
        self.session_info.setAcceptRichText(False)  # Use plain text for better performance
        self.session_info.setFont(QFont("Monospace", 9))
        self.session_info.setPlainText(
            "Initializing Osprey Framework...\n\nPlease wait while the system initializes."
        )

        # Add both widgets to layout (we'll show/hide based on setting)
        layout.addWidget(scroll_area)
        layout.addWidget(self.session_info)

        # Show/hide based on initial setting
        if self.settings_manager.get("group_system_messages", True):
            scroll_area.setVisible(True)
            self.session_info.setVisible(False)
            self.message_group_widget.add_message("Initializing Osprey Framework...", "INFO")
            self.message_group_widget.add_message(
                "Please wait while the system initializes.", "INFO"
            )
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
            label = QLabel(
                "Analytics is currently disabled.\n\nEnable analytics in router configuration to view metrics."
            )
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
        message_type = MessageFormatter.extract_message_type(message)

        # If grouping is enabled, add to grouped widget WITH color formatting
        if self.settings_manager.get("group_system_messages", True):
            # Pass the original message with Rich markup for color formatting
            self.system_info_tab.message_group_widget.add_message(
                message, message_type, rich_markup=message
            )

        # Always add to traditional text edit (for when user switches view)
        cursor = self.system_info_tab.session_info.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Parse and apply Rich markup
        MessageFormatter.insert_rich_text(cursor, message)

        # Add newline
        cursor.insertText("\n")
        self.system_info_tab.session_info.setTextCursor(cursor)
        self.system_info_tab.session_info.ensureCursorVisible()

    def toggle_message_grouping(self):
        """Toggle between grouped and ungrouped message view."""
        is_grouped = self.system_info_tab.group_messages_toggle.isChecked()
        # Update via update_from_dict to ensure proper handling
        self.settings_manager.update_from_dict({"group_system_messages": is_grouped})
        self.settings = self.settings_manager.get_all_settings()  # Update backward compat dict

        # Update button text
        if is_grouped:
            self.system_info_tab.group_messages_toggle.setText("📋 Grouped View")
        else:
            self.system_info_tab.group_messages_toggle.setText("📄 List View")

        # Show/hide appropriate widget
        self.system_info_tab.message_group_scroll.setVisible(is_grouped)
        self.system_info_tab.session_info.setVisible(not is_grouped)

    def clear_system_info(self):
        """Clear the System Information tab."""
        self.system_info_tab.session_info.clear()
        self.system_info_tab.message_group_widget.clear()
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
        self.llm_details_display.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
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
        self.tool_usage_display.setStyleSheet(
            "background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #3F3F46;"
        )
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
            self.discovered_projects = display_projects
            self._projects_cache_valid = True

            # Update table
            self.projects_tab.projects_table.setRowCount(len(display_projects))

            # Enable word wrap and adjust row heights
            self.projects_tab.projects_table.setWordWrap(True)

            for row, project in enumerate(self.discovered_projects):
                # Status column with enable/disable checkbox (NEW - Phase 2.2)
                status_widget = QWidget()
                status_layout = QHBoxLayout(status_widget)
                status_layout.setContentsMargins(2, 1, 2, 1)

                enabled_checkbox = QCheckBox()
                # Check if project is enabled in ProjectManager
                is_enabled = self.project_manager.is_project_enabled(project["name"])
                enabled_checkbox.setChecked(is_enabled)
                enabled_checkbox.stateChanged.connect(
                    lambda state, p=project["name"]: self.toggle_project_enabled(
                        p, state == Qt.Checked
                    )
                )
                status_layout.addWidget(enabled_checkbox)

                status_label = QLabel("Enabled" if is_enabled else "Disabled")
                status_label.setStyleSheet("color: #00FF00;" if is_enabled else "color: #808080;")
                status_layout.addWidget(status_label)

                self.projects_tab.projects_table.setCellWidget(row, 0, status_widget)

                # Project name
                name_item = QTableWidgetItem(project["name"])
                name_item.setForeground(QColor("#00FFFF"))
                self.projects_tab.projects_table.setItem(row, 1, name_item)

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
                self.projects_tab.projects_table.setItem(row, 2, cap_item)

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
                self.projects_tab.projects_table.setItem(row, 3, model_item)

                # Project path
                path_item = QTableWidgetItem(project["path"])
                path_item.setForeground(QColor("#FFFFFF"))
                self.projects_tab.projects_table.setItem(row, 4, path_item)

                # Config path
                config_path = Path(project["config_path"]).name
                config_item = QTableWidgetItem(config_path)
                config_item.setForeground(QColor("#00FF00"))
                self.projects_tab.projects_table.setItem(row, 5, config_item)

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
                pref_count = self.model_preferences.get_preference_count(project["name"])
                if pref_count > 0:
                    indicator = QLabel(f"✓ ({pref_count})")
                    indicator.setToolTip(f"{pref_count} runtime override(s) configured")
                    indicator.setStyleSheet("color: #00FF00;")
                    models_layout.addWidget(indicator)

                self.projects_tab.projects_table.setCellWidget(row, 6, models_widget)

                # Let row height adjust based on content
                # Users can manually resize rows by dragging the row borders in the vertical header
                # Don't set explicit height - let Interactive mode handle it

            # Resize rows to fit content initially, then users can manually adjust
            self.projects_tab.projects_table.resizeRowsToContents()

            # Update info label
            if self.discovered_projects:
                enabled_count = len(
                    [
                        p
                        for p in self.discovered_projects
                        if self.project_manager.is_project_enabled(p["name"])
                    ]
                )
                self.projects_tab.projects_info_label.setText(
                    f"Found {len(self.discovered_projects)} project(s) • "
                    f"{enabled_count} enabled • "
                    f"Use checkboxes to enable/disable projects for routing"
                )
                self.projects_tab.projects_info_label.setStyleSheet(
                    "color: #00FF00; padding: 10px;"
                )
            else:
                self.projects_tab.projects_info_label.setText(
                    "No projects found. Projects must have a config.yml file in their root directory."
                )
                self.projects_tab.projects_info_label.setStyleSheet(
                    "color: #FFA500; padding: 10px;"
                )

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
        """Open dialog to configure models for a project (delegated to ModelPreferencesManager)."""
        self.model_pref_mgr.configure_project_models(project_info)

    def apply_model_preferences_to_config(self, project_name: str):
        """Apply model preferences for a project (delegated to ModelPreferencesManager)."""
        self.model_pref_mgr.apply_model_preferences_to_config(project_name)

    def _apply_all_model_preferences(self):
        """Apply model preferences for all projects (delegated to ModelPreferencesManager)."""
        self.model_pref_mgr.apply_all_model_preferences()

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
                    "name": metadata.name,
                    "path": str(metadata.path),
                    "config_path": str(metadata.config_path),
                    "description": metadata.description,
                    "version": metadata.version,
                    "capabilities": [],  # Will be populated after loading
                    "models": {},  # Will be populated after loading
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
                        self.project_manager.load_project(metadata.name)

                        # 3. Register capabilities in CapabilityRegistry
                        capabilities = self.project_manager.get_project_capabilities(metadata.name)
                        self.capability_registry.register_project_capabilities(
                            metadata.name, capabilities
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
                    self.add_status(
                        f"Using {first_project.metadata.name} as primary project", "base"
                    )

                    # Update project selector with loaded projects (Phase 2.2)
                    QTimer.singleShot(300, self._update_project_selector)

                    # CRITICAL: Initialize router AFTER first project loads (so registry is available)
                    # The router needs the global registry singleton to be initialized for LLM calls
                    if not self.router:
                        self._initialize_router()

                        # Initialize analytics tab now that router is ready
                        if hasattr(self, "analytics_tab"):
                            self.analytics_tab.initialize_analytics()
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
            if self.conversation_manager.conversations:
                self.conversation_display_mgr.update_conversation_list()

            # Create initial conversation if none exist (after loading history)
            if not self.conversation_manager.conversations:
                self.thread_id = self.conversation_manager.create_conversation(
                    "Initial Conversation"
                )
                self.current_conversation_id = self.thread_id
            else:
                # Use the conversation manager's current conversation
                self.current_conversation_id = self.conversation_manager.current_conversation_id
                self.thread_id = self.current_conversation_id

            configurable.update(
                {
                    "user_id": "gui_user",
                    "thread_id": self.thread_id,
                    "chat_id": "gui_chat",
                    "session_id": self.thread_id,
                    "interface_context": "pyqt_gui",
                }
            )

            # Load settings from config file into SettingsManager
            if self.config_path:
                self.settings_manager.load_from_config(Path(self.config_path))

            # Update backward compatibility dict
            self.settings = self.settings_manager.get_all_settings()

            # Apply debug mode from settings to logging
            import logging

            debug_mode = self.settings_manager.get("debug_mode", False)

            root_logger = logging.getLogger()
            desired_level = logging.DEBUG if debug_mode else logging.INFO
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

            # Apply runtime overrides to config for framework use
            agent_control_defaults = self.runtime_overrides.create_agent_control_defaults()
            configurable["agent_control_defaults"] = agent_control_defaults

            recursion_limit = get_config_value("execution_limits.graph_recursion_limit")

            self.base_config = {"configurable": configurable, "recursion_limit": recursion_limit}

            # NOTE: We do NOT create a unified graph here anymore!
            # Each project has its own isolated graph with its own capabilities.
            # The GUI will use the project-specific graph when routing queries.

            # Set self.graph to None - it will be set to the project's graph when routing
            self.graph = None
            self.gateway = None

            # Load conversation history (doesn't require graph)
            # Conversation loading will work with project-specific graphs
            # if self.settings['use_persistent_conversations']:
            #     self._load_conversation_list()

            self.add_status("✅ Framework initialized successfully", "base")
            self.update_session_info()
            self.status_bar.showMessage("Osprey Framework ready")
            self._initialized = True

            # Router initialization moved earlier (after first project loads)
            # to ensure global registry singleton is available for LLM calls

            # Load current conversation display
            QTimer.singleShot(100, self.load_current_conversation_display)

        except Exception as e:
            logger.exception(f"Failed to initialize framework: {e}")
            self.add_status(f"❌ Framework initialization failed: {e}", "error")
            QMessageBox.critical(
                self, "Initialization Error", f"Failed to initialize framework:\n{e}"
            )

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
        cursor = self.system_info_tab.session_info.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n".join(session_text) + "\n")
        self.system_info_tab.session_info.setTextCursor(cursor)
        self.system_info_tab.session_info.ensureCursorVisible()

    def add_status(self, message, component="base", model_info=None):
        """Add a status message to the status log with color coding and optional model info.

        Args:
            message: Status message to display
            component: Component type for color coding
            model_info: Optional dict with 'model_provider' and 'model_id' keys
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.component_colors.get(component, self.component_colors["base"])

        # Add timestamp
        self._append_formatted_text(
            f"[{timestamp}] ", "#808080", self.status_log, prefix="", suffix=""
        )

        # Add model info if available
        if model_info and isinstance(model_info, dict):
            provider = model_info.get("model_provider", "")
            model_id = model_info.get("model_id", "")
            if provider and model_id:
                self._append_formatted_text(
                    f"[{provider}/{model_id}] ", "#FFD700", self.status_log, prefix="", suffix=""
                )

        # Add message
        self._append_formatted_text(message, color, self.status_log, prefix="", suffix="\n")

    def send_message(self):
        """Send user message to the framework with multi-project routing."""
        user_message = self.input_field.toPlainText().strip()
        if not user_message:
            return

        # Preprocess message to resolve conversation history references (GUI-only feature)
        from osprey.interfaces.pyqt.conversation_preprocessor import ConversationPreprocessor
        
        preprocessed_message, explanation = ConversationPreprocessor.preprocess_message(
            user_message, self.conversation_manager, self.current_conversation_id
        )
        
        # If preprocessing occurred, show explanation to user
        if explanation:
            self._append_colored_message(explanation, "#FFD700")
            # Use preprocessed message for framework
            user_message = preprocessed_message

        # Check if waiting for correction input (only if feedback enabled)
        if (
            self.settings_manager.get("enable_routing_feedback", True)
            and self.routing_ui.is_waiting_for_correction()
        ):
            self.routing_ui.handle_correction_input(user_message)
            self.input_field.clear()
            return

        # Check if this is feedback for previous routing (y/n) (only if feedback enabled)
        if (
            self.settings_manager.get("enable_routing_feedback", True)
            and self.routing_ui.has_pending_feedback()
            and user_message.lower() in ["y", "n", "yes", "no"]
        ):
            self.routing_ui.handle_routing_feedback(
                user_message.lower(), self.routing_ui.current_query
            )
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
                if self.routing_ui.has_pending_feedback():
                    self.routing_ui.clear_feedback_state()

                self._append_colored_message(
                    f"📝 Queued: {user_message}\n"
                    "   (Will be processed after current query completes)",
                    "#FFD700",
                )
            else:
                self._append_colored_message(
                    "⚠️ A message is already queued. Please wait for it to be processed.", "#FFA500"
                )
            return

        # IMMEDIATELY display user message and clear input for better UX
        self._append_colored_message(f"👤 You: {user_message}", "#D8BFD8")
        self.input_field.clear()

        # Add visual processing indicator
        self._append_colored_message("⏳ Processing...", "#808080")

        # Mark agent as processing
        self._agent_processing = True

        # Keep input enabled so user can provide feedback or queue next message
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)

        # Update status bar with processing indicator
        self.status_bar.showMessage("⏳ Processing your message...")

        # Force GUI update to show message immediately and ensure responsiveness
        # Process events to update the display before starting worker thread
        QApplication.processEvents()

        # Update conversation history (fast operation)
        if self.current_conversation_id:
            # Use ConversationManager to add message
            self.conversation_manager.add_message(
                self.current_conversation_id, "user", user_message
            )
            self.conversation_display_mgr.update_conversation_list()
            self.save_conversation_history()

        # Process events again to ensure GUI is fully updated
        QApplication.processEvents()

        # Phase 2.3: Multi-Project Routing with Orchestration
        enabled_projects = self.project_manager.get_enabled_projects()

        if not enabled_projects:
            self._append_colored_message(
                "⚠️ No projects enabled. Please enable at least one project in the Projects tab.",
                "#FFA500",
            )
            self.add_status("No enabled projects available", "error")
            return

        # Check if query requires orchestration
        try:
            orchestration_plan = self.router.analyze_for_orchestration(
                user_message, enabled_projects
            )

            if orchestration_plan.is_multi_project:
                # Handle multi-project orchestration
                self.orchestration_ui.handle_orchestrated_query(
                    user_message, orchestration_plan, enabled_projects
                )
                return

        except Exception as e:
            logger.warning(f"Orchestration analysis failed: {e}, falling back to single routing")

        # Single-project routing
        try:
            routing_decision = self.router.route_query(user_message, enabled_projects)

            # Store query in routing UI for feedback
            self.routing_ui.current_query = user_message

            # Display routing decision
            self.routing_ui.display_routing_decision(routing_decision)

            # Get the selected project
            selected_project = self.project_manager.get_project(routing_decision.project_name)

            if not selected_project:
                raise Exception(f"Selected project '{routing_decision.project_name}' not found")

            # CRITICAL: Initialize global registry singleton with this project's registry
            # This ensures CLI operations work by making the global registry point to
            # the currently active project's registry
            selected_project.initialize_global_registry()

            # Use selected project's gateway AND graph
            self.gateway = selected_project.gateway

            # CRITICAL FIX: Use the project's own graph, not the main graph
            # Each project has its own graph with its own state/execution tracking
            project_graph = selected_project.graph
            if not project_graph:
                logger.warning(
                    f"Project {routing_decision.project_name} has no graph, using main graph"
                )
                project_graph = self.graph

            # Update config with selected project's thread_id
            if self.base_config:
                self.base_config["configurable"]["thread_id"] = self.thread_id
                self.base_config["configurable"]["session_id"] = self.thread_id

            self.add_status(f"Using project: {routing_decision.project_name}", "base")

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            self._append_colored_message(
                f"⚠️ Routing error: {e}\nUsing fallback project.", "#FFA500"
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
        graph_to_use = project_graph if "project_graph" in locals() else self.graph

        self.worker = AgentWorker(
            self.gateway,
            graph_to_use,  # Use project's graph, not main graph
            self.base_config,
            user_message,
        )
        self.worker.message_received.connect(self.on_message_received)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.processing_complete.connect(self.on_processing_complete)
        self.worker.llm_detail.connect(self.on_llm_detail)
        self.worker.tool_usage.connect(self.on_tool_usage)

        # Start worker thread - this runs asynchronously and won't block the GUI
        self.worker.start()

        # Process events one more time to ensure GUI remains responsive
        QApplication.processEvents()

    def on_message_received(self, message):
        """Handle message received (delegated to MessageHandlers)."""
        self.message_handlers.on_message_received(message)

    def _append_formatted_text(
        self, text: str, color: str, widget=None, prefix: str = "\n", suffix: str = "\n"
    ):
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

        MessageFormatter.append_formatted_text(text, color, widget, prefix, suffix)

    def _append_colored_message(self, message, color, auto_open_plots=False):
        """Append a colored message to the conversation display with image detection.

        Args:
            message: The message text to display
            color: Color hex code for the message
            auto_open_plots: If True, automatically open plot viewer for new plots (default: False)
        """
        from osprey.interfaces.pyqt.image_display import ImageDisplayHandler, ImageViewerDialog

        # First, append the text message
        self._append_formatted_text(message, color)

        # Check if message contains image references
        if ImageDisplayHandler.should_display_inline(message):
            # Extract image paths from the message
            image_paths = ImageDisplayHandler.extract_image_paths(message)

            if image_paths:
                try:
                    logger.info(f"Found {len(image_paths)} image(s) to display")

                    # Add a separator before the plot viewer
                    self._append_formatted_text("\n" + "─" * 80, "#3F3F46", prefix="", suffix="\n")

                    # Create and insert plot viewer widget
                    # Since QTextEdit doesn't support embedded widgets, we'll add
                    # a visual indicator and provide file paths

                    for idx, image_path in enumerate(image_paths, 1):
                        self._append_formatted_text(
                            f"\n📊 Plot {idx}: {image_path.name}", "#00FFFF", prefix="", suffix="\n"
                        )
                        self._append_formatted_text(
                            f"   📍 Location: {image_path}", "#808080", prefix="", suffix="\n"
                        )

                        # Check if file exists
                        if image_path.exists():
                            # Get image dimensions
                            try:
                                from PyQt5.QtGui import QPixmap

                                pixmap = QPixmap(str(image_path))
                                if not pixmap.isNull():
                                    self._append_formatted_text(
                                        f"   📐 Size: {pixmap.width()}×{pixmap.height()}px",
                                        "#808080",
                                        prefix="",
                                        suffix="\n",
                                    )
                            except Exception:
                                pass

                            self._append_formatted_text(
                                "   ✅ File exists - Double-click path above to open",
                                "#00FF00",
                                prefix="",
                                suffix="\n",
                            )
                        else:
                            self._append_formatted_text(
                                "   ⚠️  File not found", "#FFA500", prefix="", suffix="\n"
                            )

                    # Add separator after plots
                    self._append_formatted_text("─" * 80 + "\n", "#3F3F46", prefix="", suffix="\n")

                    logger.info(f"Displayed {len(image_paths)} plot reference(s)")

                    # AUTOMATICALLY OPEN PLOT VIEWER WINDOW for the first valid image
                    # BUT ONLY if auto_open_plots is True (i.e., this is a NEW message from the agent)
                    if auto_open_plots:
                        valid_images = [p for p in image_paths if p.exists()]
                        if valid_images:
                            logger.info(f"Auto-opening plot viewer for {valid_images[0]}")
                            # Open the first image in a modeless dialog
                            dialog = ImageViewerDialog(valid_images[0], self)
                            dialog.show()

                            # If there are multiple images, inform the user
                            if len(valid_images) > 1:
                                self._append_formatted_text(
                                    f"\n💡 Tip: {len(valid_images)} plot(s) generated. First plot opened automatically. "
                                    f"Click paths above to view others.\n",
                                    "#FFD700",
                                    prefix="",
                                    suffix="\n",
                                )

                except Exception as e:
                    logger.error(f"Error displaying images: {e}")

    def on_status_update(self, status, component="base", model_info=None):
        """Handle status update (delegated to MessageHandlers)."""
        self.message_handlers.on_status_update(status, component, model_info)

    def on_error(self, error):
        """Handle error (delegated to MessageHandlers)."""
        self.message_handlers.on_error(error)

    def on_llm_detail(self, detail, event_type="base"):
        """Handle LLM detail (delegated to MessageHandlers)."""
        self.message_handlers.on_llm_detail(detail, event_type)

    def on_tool_usage(self, tool_name, reasoning):
        """Handle tool usage (delegated to MessageHandlers)."""
        self.message_handlers.on_tool_usage(tool_name, reasoning)

    def on_processing_complete(self):
        """Handle processing complete (delegated to MessageHandlers)."""
        self.message_handlers.on_processing_complete()

    def clear_conversation(self):
        """Clear conversation (delegated to ConversationManagement)."""
        self.conversation_mgmt.clear_conversation()

    def start_new_conversation(self):
        """Start new conversation (delegated to ConversationManagement)."""
        self.conversation_mgmt.start_new_conversation()

    def create_new_conversation(self):
        """Create new conversation (delegated to ConversationManagement)."""
        self.conversation_mgmt.create_new_conversation()

    def update_conversation_list(self):
        """Update the conversation history list (delegated to ConversationDisplayManager)."""
        self.conversation_display_mgr.update_conversation_list()

    def switch_conversation(self, item):
        """Switch to a different conversation (delegated to ConversationDisplayManager)."""
        self.conversation_display_mgr.switch_conversation(item)

    def delete_selected_conversation(self):
        """Delete selected conversation (delegated to ConversationManagement)."""
        self.conversation_mgmt.delete_selected_conversation()

    def rename_selected_conversation(self):
        """Rename selected conversation (delegated to ConversationManagement)."""
        self.conversation_mgmt.rename_selected_conversation()

    def _load_conversation_list(self):
        """Load conversation list (delegated to ConversationHistory)."""
        self.conversation_history.load_conversation_list()

    def save_conversation_history(self):
        """Save conversation history (delegated to ConversationHistory)."""
        self.conversation_history.save_conversation_history()

    def load_conversation_history(self):
        """Load conversation history (delegated to ConversationHistory)."""
        self.conversation_history.load_conversation_history()

    def load_current_conversation_display(self):
        """Load the current conversation messages (delegated to ConversationDisplayManager)."""
        self.conversation_display_mgr.load_current_conversation_display()

    def show_settings(self):
        """Show settings dialog (modeless)."""
        dialog = SettingsDialog(self, "Framework Settings", self.settings)

        # Connect the accepted signal to handle settings update
        def on_settings_accepted():
            # Get old routing settings for comparison
            old_routing_settings = self.settings_manager.routing.__dict__.copy()

            # Update settings from dialog
            new_settings_dict = dialog.get_settings()
            self.settings_manager.update_from_dict(new_settings_dict)
            self.settings = self.settings_manager.get_all_settings()  # Update backward compat dict

            # Check if routing settings changed
            new_routing_settings = self.settings_manager.routing.__dict__.copy()
            routing_settings_changed = old_routing_settings != new_routing_settings

            # Update base config with new runtime overrides
            if self.base_config:
                agent_control_defaults = self.runtime_overrides.create_agent_control_defaults()
                self.base_config["configurable"]["agent_control_defaults"] = agent_control_defaults

                # Apply development/debug settings to the configuration
                development_config = self.base_config["configurable"].get("development", {})
                development_config["debug"] = self.settings_manager.get("debug_mode", False)
                development_config["raise_raw_errors"] = self.settings_manager.get(
                    "raise_raw_errors", False
                )

                # Apply prompt settings
                prompts_config = development_config.get("prompts", {})
                prompts_config["print_all"] = self.settings_manager.get("print_prompts", False)
                prompts_config["show_all"] = self.settings_manager.get("show_prompts", False)
                prompts_config["latest_only"] = self.settings_manager.get(
                    "prompts_latest_only", True
                )
                development_config["prompts"] = prompts_config

                self.base_config["configurable"]["development"] = development_config

            # Apply logging level changes immediately
            import logging

            debug_mode = self.settings_manager.get("debug_mode", False)
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
            suppress_terminal = self.settings_manager.get("suppress_terminal_output", False)
            from osprey.utils.logger import set_gui_output_callback

            set_gui_output_callback(
                self.gui_output_signal.emit_output, suppress_terminal=suppress_terminal
            )

            # Update GUI handler level to match the new logging level
            for handler in root_logger.handlers:
                from osprey.utils.logger import GUIHandler

                if isinstance(handler, GUIHandler):
                    handler.setLevel(new_level)
                    logger.debug(f"Updated GUI handler level to {logging.getLevelName(new_level)}")

            # Save GUI preferences to user's home directory
            self._save_gui_preferences()

            # Update runtime overrides (in-memory only, NOT written to disk)
            self.runtime_overrides.set_overrides(self.settings)

            self.update_session_info()
            level_name = "DEBUG" if debug_mode else "INFO"
            self.add_status(f"Settings updated and saved - Logging level: {level_name}", "base")

            # Update Memory tab threshold display if it exists
            if hasattr(self, "memory_tab") and self.memory_tab:
                self.memory_tab.update_threshold_display()
                self.add_status("Memory monitoring thresholds updated", "base")

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

            QMessageBox.information(self, "Settings Updated", message)

        dialog.accepted.connect(on_settings_accepted)
        dialog.show()  # Show modeless dialog

    def _save_gui_preferences(self):
        """Save GUI preferences to user's home directory.

        This saves ONLY user interface preferences (NOT execution behavior).
        Execution behavior is handled by runtime overrides (in-memory only).
        """
        try:
            # Extract GUI-specific preferences
            gui_prefs = {
                "use_persistent_conversations": self.settings_manager.gui.use_persistent_conversations,
                "conversation_storage_mode": self.settings_manager.gui.conversation_storage_mode,
                "redirect_output_to_gui": self.settings_manager.gui.redirect_output_to_gui,
                "suppress_terminal_output": self.settings_manager.gui.suppress_terminal_output,
                "group_system_messages": self.settings_manager.gui.group_system_messages,
                "enable_routing_feedback": self.settings_manager.gui.enable_routing_feedback,
                "memory_monitor_enabled": self.settings_manager.memory_monitor_enabled,
                "memory_warning_threshold_mb": self.settings_manager.memory_warning_threshold_mb,
                "memory_critical_threshold_mb": self.settings_manager.memory_critical_threshold_mb,
                "memory_check_interval_seconds": self.settings_manager.memory_check_interval_seconds,
            }

            # Update and save GUI preferences
            self.gui_preferences.update(gui_prefs)
            if self.gui_preferences.save_preferences():
                logger.info(f"GUI preferences saved to {self.gui_preferences.preferences_path}")
            else:
                logger.warning("Failed to save GUI preferences")

        except Exception as e:
            logger.error(f"Failed to save GUI preferences: {e}")
            QMessageBox.warning(
                self,
                "Save Warning",
                f"GUI preferences could not be saved:\n{e}\n\n"
                "Preferences will be lost when GUI is restarted.",
            )

    def show_help_dialog(self):
        """Show the help dialog."""
        show_help_dialog(self)

    def show_about_dialog(self):
        """Show the about dialog (delegated to about_dialog module)."""
        show_about_dialog(self, __version__)


# SettingsDialog has been extracted to settings_dialog.py


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

    # Apply theme using the theme manager
    # This ensures consistent appearance across Windows, Linux, and macOS
    from osprey.interfaces.pyqt.themes import get_theme_manager
    
    theme_manager = get_theme_manager()
    theme_manager.apply_theme(app)  # Applies dark theme by default

    window = OspreyGUI(config_path=config_path)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
