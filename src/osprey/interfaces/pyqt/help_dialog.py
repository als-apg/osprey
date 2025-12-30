#!/usr/bin/env python3
"""
Help Dialog for Osprey Framework PyQt GUI

Provides comprehensive help information across multiple tabs.
"""

from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from osprey.interfaces.pyqt.gui_utils import HELP_DIALOG_CSS, create_dark_palette


class HelpDialog(QDialog):
    """Non-modal help dialog with tabbed help content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Osprey Framework - Help")
        self.setModal(False)  # Non-modal - can interact with main window
        self.resize(800, 600)

        # Apply dark theme matching main window
        self.apply_dark_theme()

        self.setup_ui()

    def apply_dark_theme(self):
        """Apply dark theme to match the main GUI."""
        self.setPalette(create_dark_palette())

    def setup_ui(self):
        """Setup the help dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget with dark theme styling
        tab_widget = QTabWidget()
        tab_widget.setMovable(False)
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3F3F46;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background-color: #2D2D30;
                color: #FFFFFF;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #3F3F46;
            }
            QTabBar::tab:selected {
                background-color: #0078D4;
                color: #FFFFFF;
            }
            QTabBar::tab:hover {
                background-color: #3F3F46;
            }
        """)

        # Add help tabs
        tab_widget.addTab(self.create_getting_started_tab(), "Getting Started")
        tab_widget.addTab(self.create_conversation_tab(), "Conversations")
        tab_widget.addTab(self.create_features_tab(), "Features")
        tab_widget.addTab(self.create_multi_project_tab(), "Multi-Project")
        tab_widget.addTab(self.create_settings_tab(), "Settings")
        tab_widget.addTab(self.create_troubleshooting_tab(), "Troubleshooting")
        tab_widget.addTab(self.create_keyboard_shortcuts_tab(), "Shortcuts")

        layout.addWidget(tab_widget)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.setMinimumWidth(100)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

    def create_styled_text_browser(self):
        """Create a text browser with dark theme styling."""
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setStyleSheet("""
            QTextBrowser {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                padding: 10px;
            }
        """)
        return text_browser

    def create_getting_started_tab(self):
        """Create the Getting Started help tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        text_browser = self.create_styled_text_browser()
        text_browser.setHtml(f"""{HELP_DIALOG_CSS}
            <h2>Getting Started with Osprey Framework</h2>

            <h3>What is Osprey Framework?</h3>
            <p>Osprey Framework is an AI-powered assistant system designed for scientific facilities
            and control systems. It provides intelligent conversation capabilities, task automation,
            and integration with various data sources and control systems.</p>

            <h3>Quick Start Guide</h3>
            <ol>
                <li><b>Start a Conversation:</b> Type your question or request in the input field
                at the bottom of the Conversation tab and click "Send" or press Enter.</li>

                <li><b>View Responses:</b> The AI assistant's responses will appear in the main
                conversation area with color-coded messages.</li>

                <li><b>Monitor Processing:</b> Watch the Status Log panel on the right to see
                real-time updates about what the system is doing.</li>

                <li><b>Track Details:</b> Switch to the "LLM Details" and "Tool Usage" tabs to
                see detailed information about the AI's reasoning and actions.</li>
            </ol>

            <h3>Main Interface Components</h3>
            <ul>
                <li><b>Conversation History (Left Panel):</b> Lists all your conversations.
                Click to switch between them.</li>

                <li><b>Conversation Display (Center):</b> Shows the current conversation with
                color-coded messages (purple for you, white for assistant).</li>

                <li><b>Status Log (Right Panel):</b> Real-time status updates with color-coded
                components showing what's happening behind the scenes.</li>

                <li><b>Input Field (Bottom):</b> Type your messages here. Use Shift+Enter for
                new lines.</li>
            </ul>

            <h3>Understanding Color Codes</h3>
            <ul>
                <li><span style="color: #D8BFD8;">Purple:</span> Your messages</li>
                <li><span style="color: #FFFFFF;">White:</span> Assistant responses</li>
                <li><span style="color: #00FF00;">Green:</span> Success messages and status</li>
                <li><span style="color: #00FFFF;">Cyan:</span> System information</li>
                <li><span style="color: #FFD700;">Gold:</span> Important notifications</li>
                <li><span style="color: #FF0000;">Red:</span> Errors</li>
            </ul>

            <h3>Tips for Best Results</h3>
            <ul>
                <li>Be specific in your requests - the more detail you provide, the better
                the assistant can help.</li>
                <li>Use the conversation history to maintain context across multiple questions.</li>
                <li>Check the Tool Usage tab to understand what capabilities were used.</li>
                <li>Review the Status Log to troubleshoot any issues.</li>
            </ul>
        """)
        layout.addWidget(text_browser)

        return widget

    def create_conversation_tab(self):
        """Create the Conversations help tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        text_browser = self.create_styled_text_browser()
        text_browser.setHtml(f"""{HELP_DIALOG_CSS}
            <h2>Managing Conversations</h2>

            <h3>Creating Conversations</h3>
            <p>Conversations help you organize different topics or work sessions:</p>
            <ul>
                <li><b>New Conversation Button:</b> Click the "New Conversation" button in the
                input area or use File → New Conversation.</li>
                <li><b>Quick Create:</b> Click the "+" button in the conversation history panel.</li>
                <li><b>Auto-naming:</b> Conversations are automatically numbered, but you can
                rename them.</li>
            </ul>

            <h3>Switching Between Conversations</h3>
            <ul>
                <li>Click any conversation in the left panel to switch to it.</li>
                <li>The active conversation is highlighted in green with a "▶" marker.</li>
                <li>All conversation history is preserved when switching.</li>
            </ul>

            <h3>Organizing Conversations</h3>
            <ul>
                <li><b>Rename:</b> Click the "✏" (pencil) button or right-click a conversation
                to rename it with a meaningful title.</li>
                <li><b>Delete:</b> Click the "🗑" (trash) button to delete conversations you
                no longer need.</li>
                <li><b>Sorting:</b> Conversations are automatically sorted by most recent activity.</li>
            </ul>

            <h3>Conversation Persistence</h3>
            <p>Your conversations are automatically saved:</p>
            <ul>
                <li><b>Automatic Saving:</b> Every message is saved immediately.</li>
                <li><b>PostgreSQL Storage:</b> If PostgreSQL is configured, conversations persist
                across sessions.</li>
                <li><b>In-Memory Fallback:</b> Without PostgreSQL, conversations are saved to
                local JSON files.</li>
                <li><b>Session Recovery:</b> Restart the application and your conversations
                will be restored.</li>
            </ul>

            <h3>Conversation Information</h3>
            <p>Each conversation displays:</p>
            <ul>
                <li>Conversation name</li>
                <li>Last activity timestamp</li>
                <li>Number of messages</li>
                <li>Thread ID (for technical reference)</li>
            </ul>

            <h3>Best Practices</h3>
            <ul>
                <li>Create separate conversations for different topics or projects.</li>
                <li>Rename conversations with descriptive titles for easy identification.</li>
                <li>Delete old conversations to keep your workspace organized.</li>
                <li>Use the conversation context to build on previous discussions.</li>
            </ul>
        """)
        layout.addWidget(text_browser)

        return widget

    def create_features_tab(self):
        """Create the Features help tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        text_browser = self.create_styled_text_browser()
        text_browser.setHtml(f"""{HELP_DIALOG_CSS}
            <h2>GUI Features and Tabs</h2>

            <h3>Conversation Tab</h3>
            <p>The main interface for interacting with the AI assistant:</p>
            <ul>
                <li><b>Three-Panel Layout:</b> History, conversation, and status log.</li>
                <li><b>Real-time Updates:</b> See responses as they're generated.</li>
                <li><b>Color-Coded Messages:</b> Easy visual distinction between user and assistant.</li>
                <li><b>Status Monitoring:</b> Track processing in real-time.</li>
            </ul>

            <h3>LLM Details Tab</h3>
            <p>View detailed information about AI processing:</p>
            <ul>
                <li><b>Event Tracking:</b> See when the AI starts and completes operations.</li>
                <li><b>Timestamped Logs:</b> Precise timing information for each event.</li>
                <li><b>Color-Coded Events:</b> Different colors for different event types.</li>
                <li><b>Clear Button:</b> Reset the view when needed.</li>
            </ul>

            <h3>Tool Usage Tab</h3>
            <p>Understand what capabilities the AI is using:</p>
            <ul>
                <li><b>Capability Tracking:</b> See which tools/capabilities were invoked.</li>
                <li><b>Task Objectives:</b> Understand what each capability was trying to accomplish.</li>
                <li><b>Success Indicators:</b> Green checkmarks for success, red X for failures.</li>
                <li><b>Execution Time:</b> Performance metrics for each capability.</li>
                <li><b>Detailed Reasoning:</b> See why the AI chose specific tools.</li>
            </ul>

            <h3>Discovered Projects Tab</h3>
            <p>Manage multiple Osprey projects:</p>
            <ul>
                <li><b>Auto-Discovery:</b> Automatically finds Osprey projects in subdirectories.</li>
                <li><b>Project Table:</b> View all discovered projects with their configurations.</li>
                <li><b>Model Configuration:</b> Configure which AI models to use for each project.</li>
                <li><b>Unified Config:</b> Combine multiple projects into one configuration.</li>
                <li><b>Refresh:</b> Re-scan for new or updated projects.</li>
            </ul>

            <h3>System Information Tab</h3>
            <p>View technical details about your session:</p>
            <ul>
                <li><b>Thread ID:</b> Unique identifier for the current conversation.</li>
                <li><b>Config Path:</b> Location of the active configuration file.</li>
                <li><b>Capabilities:</b> Number of available capabilities.</li>
                <li><b>Session Details:</b> Technical information for troubleshooting.</li>
                <li><b>View Modes:</b> Toggle between Grouped and List view.</li>
            </ul>

            <h3>Memory Tab (💾)</h3>
            <p>Monitor memory usage of framework processes:</p>
            <ul>
                <li><b>System Memory:</b> Overall system memory usage with progress bar.</li>
                <li><b>Framework Total:</b> Combined memory of all framework processes.</li>
                <li><b>Memory Trend:</b> Real-time trend analysis (increasing/stable/decreasing).</li>
                <li><b>Process Details:</b> Breakdown by process type (GUI, containers, etc.).</li>
                <li><b>Threshold Alerts:</b> Visual warnings when memory exceeds limits.</li>
                <li><b>Controls:</b> Pause/Resume monitoring and manual refresh.</li>
            </ul>

            <h3>Analytics Tab (📊)</h3>
            <p>View usage statistics and performance metrics:</p>
            <ul>
                <li><b>Query Routing:</b> Statistics on project/capability selection.</li>
                <li><b>Cache Performance:</b> Hit/miss rates for routing cache.</li>
                <li><b>Response Times:</b> Performance metrics over time.</li>
                <li><b>Capability Usage:</b> Frequency of capability invocations.</li>
            </ul>

            <h3>Menu Bar Features</h3>
            <p><b>File Menu:</b></p>
            <ul>
                <li>New Conversation - Start fresh conversation</li>
                <li>Clear Conversation - Remove all messages from current conversation</li>
                <li>Exit - Close the application</li>
            </ul>

            <p><b>Settings Menu:</b></p>
            <ul>
                <li>Framework Settings - Configure behavior, features, and performance</li>
            </ul>

            <p><b>Help Menu:</b></p>
            <ul>
                <li>Help Documentation (F1) - Open this comprehensive help guide</li>
                <li>About - Version and system information</li>
            </ul>
        """)
        layout.addWidget(text_browser)

        return widget

    def create_multi_project_tab(self):
        """Create the Multi-Project help tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        text_browser = self.create_styled_text_browser()
        text_browser.setHtml(f"""{HELP_DIALOG_CSS}
            <h2>Multi-Project Management</h2>

            <h3>What is Multi-Project Support?</h3>
            <p>Osprey Framework can discover and combine multiple project configurations,
            allowing you to use capabilities from different projects simultaneously.</p>

            <h3>Project Discovery</h3>
            <p>The GUI automatically discovers Osprey projects:</p>
            <ul>
                <li><b>Automatic Scan:</b> On startup, the GUI scans subdirectories for projects.</li>
                <li><b>Project Requirements:</b> Each project must have a <code>config.yml</code> file.</li>
                <li><b>Manual Refresh:</b> Click "🔄 Refresh" in the Discovered Projects tab.</li>
                <li><b>Project Information:</b> View paths, config files, and registry files.</li>
            </ul>

            <h3>Generating Unified Configuration</h3>
            <p>Combine multiple projects into one configuration:</p>
            <ol>
                <li>Go to the "Discovered Projects" tab.</li>
                <li>Click "Generate Unified Config" button.</li>
                <li>Review the list of projects to be combined.</li>
                <li>Confirm to create <code>unified_config.yml</code> and
                <code>unified_registry.py</code>.</li>
            </ol>

            <h3>Loading Unified Configuration</h3>
            <p>Use the combined configuration:</p>
            <ol>
                <li>Click "Load Unified Config" button.</li>
                <li>Confirm the reload (this resets the current session).</li>
                <li>All capabilities from all projects are now available.</li>
            </ol>

            <h3>Model Configuration Per Project</h3>
            <p>Configure which AI models to use for each project:</p>
            <ul>
                <li><b>Configure Button:</b> Click "Configure" in the Models column.</li>
                <li><b>Step Selection:</b> Choose models for different processing steps
                (classification, orchestration, etc.).</li>
                <li><b>Model Options:</b> Select from available models in your configuration.</li>
                <li><b>Save Preferences:</b> Settings are saved and applied automatically.</li>
                <li><b>Indicators:</b> Green checkmark shows configured steps.</li>
            </ul>

            <h3>Project Structure</h3>
            <p>Each Osprey project should have:</p>
            <ul>
                <li><code>config.yml</code> - Main configuration file (required)</li>
                <li><code>registry.py</code> - Capability registration (optional)</li>
                <li>Capability modules - Python files with capability implementations</li>
                <li>Data sources - Connectors to external systems</li>
            </ul>

            <h3>Use Cases</h3>
            <ul>
                <li><b>Facility Integration:</b> Combine capabilities from different beamlines
                or systems.</li>
                <li><b>Specialized Assistants:</b> Use domain-specific projects together.</li>
                <li><b>Development:</b> Test new capabilities alongside production ones.</li>
                <li><b>Modular Design:</b> Organize capabilities by function or department.</li>
            </ul>

            <h3>Best Practices</h3>
            <ul>
                <li>Keep projects focused on specific domains or capabilities.</li>
                <li>Use clear, descriptive project names.</li>
                <li>Document project dependencies and requirements.</li>
                <li>Test individual projects before combining them.</li>
                <li>Regenerate unified config after updating individual projects.</li>
            </ul>
        """)
        layout.addWidget(text_browser)

        return widget

    def create_settings_tab(self):
        """Create the Settings help tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        text_browser = self.create_styled_text_browser()
        text_browser.setHtml(f"""{HELP_DIALOG_CSS}
            <h2>Framework Settings</h2>

            <h3>Accessing Settings</h3>
            <p>Open settings via: <b>Settings → Framework Settings</b></p>
            <p>Settings are organized into multiple tabs for easy navigation.</p>

            <h3>Agent Control Settings</h3>

            <h4>Planning Mode</h4>
            <ul>
                <li><b>Purpose:</b> Enable multi-step planning before execution.</li>
                <li><b>When Enabled:</b> The AI creates a plan before taking actions.</li>
                <li><b>When Disabled:</b> Direct execution without explicit planning.</li>
                <li><b>Use Case:</b> Enable for complex, multi-step tasks.</li>
            </ul>

            <h4>EPICS Writes</h4>
            <ul>
                <li><b>Purpose:</b> Control whether the system can write to EPICS control system.</li>
                <li><b>When Enabled:</b> AI can modify EPICS process variables.</li>
                <li><b>When Disabled:</b> Read-only access to EPICS (safer for production).</li>
                <li><b>Security:</b> Keep disabled unless you need write access.</li>
            </ul>

            <h4>Task Extraction Bypass</h4>
            <ul>
                <li><b>Purpose:</b> Skip LLM-based task extraction for faster processing.</li>
                <li><b>When Enabled:</b> Uses full conversation context directly.</li>
                <li><b>Use Case:</b> Enable for simple queries or performance optimization.</li>
            </ul>

            <h4>Capability Selection Bypass</h4>
            <ul>
                <li><b>Purpose:</b> Skip capability classification and activate all capabilities.</li>
                <li><b>When Enabled:</b> All registered capabilities are available.</li>
                <li><b>Use Case:</b> Enable when you want all capabilities active.</li>
            </ul>

            <h3>Approval Settings</h3>

            <h4>Global Approval Mode</h4>
            <p>Control when human approval is required:</p>
            <ul>
                <li><b>Disabled:</b> No approval required (fastest, use with caution).</li>
                <li><b>Selective:</b> Approval required for specific capabilities
                (balanced approach).</li>
                <li><b>All Capabilities:</b> Approval required for every action
                (safest, slowest).</li>
            </ul>

            <h4>Python Execution Approval</h4>
            <ul>
                <li><b>Purpose:</b> Control approval for Python code execution.</li>
                <li><b>Modes:</b> Disabled, EPICS Writes Only, or All Code.</li>
            </ul>

            <h4>Memory Approval</h4>
            <ul>
                <li><b>Purpose:</b> Require approval before storing information in memory.</li>
            </ul>

            <h3>Execution Limits</h3>

            <h4>Max Execution Time</h4>
            <ul>
                <li><b>Purpose:</b> Timeout for capability execution.</li>
                <li><b>Range:</b> 10 to 3600 seconds.</li>
                <li><b>Default:</b> 300 seconds (5 minutes).</li>
                <li><b>Recommendation:</b> Increase for long-running operations,
                decrease for quick tasks.</li>
            </ul>

            <h4>Max Reclassifications</h4>
            <ul>
                <li><b>Purpose:</b> Limit retry attempts when capability selection fails.</li>
                <li><b>Default:</b> 3 attempts.</li>
            </ul>

            <h4>Max Planning Attempts</h4>
            <ul>
                <li><b>Purpose:</b> Limit retry attempts for orchestration planning.</li>
                <li><b>Default:</b> 3 attempts.</li>
            </ul>

            <h4>Max Step Retries</h4>
            <ul>
                <li><b>Purpose:</b> Limit retry attempts for individual execution steps.</li>
                <li><b>Default:</b> 2 attempts.</li>
            </ul>

            <h4>Max Concurrent Classifications</h4>
            <ul>
                <li><b>Purpose:</b> Control parallel capability classification.</li>
                <li><b>Default:</b> 5 concurrent classifications.</li>
            </ul>

            <h3>GUI Settings</h3>

            <h4>Save Conversation History</h4>
            <ul>
                <li><b>Purpose:</b> Enable persistent conversation storage.</li>
                <li><b>When Enabled:</b> Conversations saved to PostgreSQL or local files.</li>
                <li><b>When Disabled:</b> Conversations lost when application closes.</li>
                <li><b>Storage:</b> Uses PostgreSQL if available, otherwise JSON files.</li>
            </ul>

            <h4>Message Storage Mode</h4>
            <ul>
                <li><b>Options:</b> JSON (local files) or PostgreSQL (database).</li>
                <li><b>Default:</b> JSON for simplicity.</li>
            </ul>

            <h4>Redirect Output to GUI</h4>
            <ul>
                <li><b>Purpose:</b> Show execution output in the GUI interface.</li>
                <li><b>Recommended:</b> Keep enabled for better visibility.</li>
            </ul>

            <h4>Group System Messages</h4>
            <ul>
                <li><b>Purpose:</b> Organize system messages by category in System Information tab.</li>
            </ul>

            <h4>Suppress Terminal Output</h4>
            <ul>
                <li><b>Purpose:</b> Reduce console logging when running GUI.</li>
            </ul>

            <h4>Enable Routing Feedback</h4>
            <ul>
                <li><b>Purpose:</b> Show which project/capability was selected for queries.</li>
            </ul>

            <h3>Development/Debug Settings</h3>

            <h4>Debug Mode</h4>
            <ul>
                <li><b>Purpose:</b> Enable DEBUG level logging for troubleshooting.</li>
                <li><b>When Enabled:</b> More detailed logs in Status Log and terminal.</li>
            </ul>

            <h4>Verbose Logging</h4>
            <ul>
                <li><b>Purpose:</b> Additional detailed logging output.</li>
            </ul>

            <h4>Save Prompts to Files</h4>
            <ul>
                <li><b>Purpose:</b> Save LLM prompts to files for inspection.</li>
                <li><b>Location:</b> _agent_data/prompts/ directory.</li>
            </ul>

            <h4>Show Prompts in Console</h4>
            <ul>
                <li><b>Purpose:</b> Display prompts in terminal output.</li>
            </ul>

            <h4>Memory Monitoring</h4>
            <ul>
                <li><b>Enable Memory Monitor:</b> Track memory usage of framework processes.</li>
                <li><b>Warning Threshold:</b> Alert when memory exceeds limit (default: 500 MB).</li>
                <li><b>Critical Threshold:</b> Critical alert level (default: 1000 MB).</li>
                <li><b>Check Interval:</b> How often to check memory (default: 5 seconds).</li>
            </ul>

            <h3>Advanced Routing Settings</h3>

            <h4>Routing Cache</h4>
            <ul>
                <li><b>Enable Routing Cache:</b> Cache routing decisions for similar queries.</li>
                <li><b>Cache Size:</b> Maximum number of cached entries.</li>
                <li><b>Cache TTL:</b> Time-to-live for cache entries in seconds.</li>
                <li><b>Similarity Threshold:</b> How similar queries must be to use cache.</li>
            </ul>

            <h4>Semantic Context Analysis</h4>
            <ul>
                <li><b>Enable Semantic Analysis:</b> Use semantic similarity for routing.</li>
                <li><b>Topic Similarity:</b> Threshold for topic matching.</li>
                <li><b>Max Context History:</b> Number of previous messages to analyze.</li>
            </ul>

            <h3>Applying Settings</h3>
            <ul>
                <li>Click "Save" to apply changes immediately.</li>
                <li>Settings affect new operations, not ongoing ones.</li>
                <li>Some settings may require restarting conversations.</li>
                <li>Settings are preserved across application restarts.</li>
            </ul>

            <h3>Recommended Configurations</h3>

            <p><b>Development/Testing:</b></p>
            <ul>
                <li>Planning Mode: Enabled</li>
                <li>EPICS Writes: Disabled</li>
                <li>Approval Mode: Selective</li>
                <li>Max Execution Time: 300 seconds</li>
            </ul>

            <p><b>Production (Safe):</b></p>
            <ul>
                <li>Planning Mode: Enabled</li>
                <li>EPICS Writes: Disabled</li>
                <li>Approval Mode: All Capabilities</li>
                <li>Max Execution Time: 300 seconds</li>
            </ul>

            <p><b>Production (Automated):</b></p>
            <ul>
                <li>Planning Mode: Enabled</li>
                <li>EPICS Writes: Enabled (if needed)</li>
                <li>Approval Mode: Selective</li>
                <li>Max Execution Time: 600 seconds</li>
            </ul>
        """)
        layout.addWidget(text_browser)

        return widget

    def create_troubleshooting_tab(self):
        """Create the Troubleshooting help tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        text_browser = self.create_styled_text_browser()
        text_browser.setHtml(f"""{HELP_DIALOG_CSS}
            <h2>Troubleshooting Guide</h2>

            <h3>Common Issues and Solutions</h3>

            <h4>GUI Won't Start</h4>
            <ul>
                <li><b>Check Python Version:</b> Requires Python 3.8 or higher.</li>
                <li><b>Install Dependencies:</b> Run <code>pip install -r requirements.txt</code></li>
                <li><b>PyQt5 Issues:</b> Ensure PyQt5 is properly installed.</li>
                <li><b>Check Logs:</b> Look for error messages in the terminal.</li>
            </ul>

            <h4>No Response from AI</h4>
            <ul>
                <li><b>Check API Keys:</b> Ensure LLM API keys are configured in environment.</li>
                <li><b>Network Connection:</b> Verify internet connectivity for cloud LLMs.</li>
                <li><b>Status Log:</b> Check for error messages in the status log panel.</li>
                <li><b>Model Configuration:</b> Verify models are properly configured in config.yml.</li>
            </ul>

            <h4>Conversations Not Saving</h4>
            <ul>
                <li><b>Check Setting:</b> Ensure "Save Conversation History" is enabled.</li>
                <li><b>PostgreSQL:</b> If using PostgreSQL, verify it's running and accessible.</li>
                <li><b>File Permissions:</b> Check write permissions for _agent_data directory.</li>
                <li><b>Disk Space:</b> Ensure sufficient disk space available.</li>
            </ul>

            <h4>Project Discovery Fails</h4>
            <ul>
                <li><b>Config Files:</b> Ensure each project has a config.yml file.</li>
                <li><b>File Permissions:</b> Check read permissions for project directories.</li>
                <li><b>YAML Syntax:</b> Validate YAML syntax in config files.</li>
                <li><b>Refresh:</b> Try clicking the Refresh button in Discovered Projects tab.</li>
            </ul>

            <h4>Slow Performance</h4>
            <ul>
                <li><b>Model Selection:</b> Smaller/faster models may improve response time.</li>
                <li><b>Network Latency:</b> Cloud LLMs depend on internet speed.</li>
                <li><b>System Resources:</b> Check CPU and memory usage.</li>
                <li><b>Reduce Timeout:</b> Lower max execution time for faster failures.</li>
            </ul>

            <h3>Diagnostic Information</h3>

            <h4>Where to Find Information</h4>
            <ul>
                <li><b>Status Log:</b> Real-time processing updates and errors.</li>
                <li><b>LLM Details Tab:</b> Detailed AI processing information.</li>
                <li><b>Tool Usage Tab:</b> Capability execution details and timing.</li>
                <li><b>System Information Tab:</b> Session and configuration details.</li>
                <li><b>Terminal Output:</b> Detailed logs if running from command line.</li>
            </ul>

            <h4>Collecting Debug Information</h4>
            <p>When reporting issues, include:</p>
            <ul>
                <li>Osprey Framework version (Help → About)</li>
                <li>Python version and OS information</li>
                <li>Error messages from Status Log</li>
                <li>Steps to reproduce the issue</li>
                <li>Configuration file (sanitize sensitive data)</li>
                <li>Terminal output if available</li>
            </ul>

            <h3>Getting Help</h3>
            <ul>
                <li><b>Documentation:</b> Check the Osprey Framework documentation.</li>
                <li><b>GitHub Issues:</b> Search for similar issues or create a new one.</li>
                <li><b>Logs:</b> Enable debug logging for more detailed information.</li>
                <li><b>Community:</b> Reach out to the Osprey Framework community.</li>
            </ul>

            <h3>Reset Options</h3>
            <ul>
                <li><b>Clear Conversation:</b> File → Clear Conversation</li>
                <li><b>New Conversation:</b> Start fresh with File → New Conversation</li>
                <li><b>Restart Application:</b> Close and reopen the GUI</li>
                <li><b>Reset Settings:</b> Delete settings and restart (advanced)</li>
            </ul>
        """)
        layout.addWidget(text_browser)

        return widget

    def create_keyboard_shortcuts_tab(self):
        """Create the Keyboard Shortcuts help tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        text_browser = self.create_styled_text_browser()
        # Keyboard shortcuts tab needs additional table styling
        shortcuts_css = HELP_DIALOG_CSS.replace(
            "</style>",
            """
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th { background-color: #2D2D30; color: #00FFFF; padding: 8px; text-align: left; border: 1px solid #3F3F46; }
                td { padding: 8px; border: 1px solid #3F3F46; color: #FFFFFF; }
                tr:nth-child(even) { background-color: #252526; }
            </style>""",
        )
        text_browser.setHtml(f"""{shortcuts_css}
            <h2>Keyboard Shortcuts</h2>

            <h3>Application</h3>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
                <tr style="background-color: #2D2D30;">
                    <th>Shortcut</th>
                    <th>Action</th>
                </tr>
                <tr>
                    <td><b>F1</b></td>
                    <td>Open Help Documentation</td>
                </tr>
            </table>

            <h3>Message Input</h3>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
                <tr style="background-color: #2D2D30;">
                    <th>Shortcut</th>
                    <th>Action</th>
                </tr>
                <tr>
                    <td><b>Enter</b></td>
                    <td>Send message (when input field is focused)</td>
                </tr>
                <tr>
                    <td><b>Shift + Enter</b></td>
                    <td>Insert new line in message</td>
                </tr>
            </table>

            <h3>Navigation</h3>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
                <tr style="background-color: #2D2D30;">
                    <th>Shortcut</th>
                    <th>Action</th>
                </tr>
                <tr>
                    <td><b>Tab</b></td>
                    <td>Navigate between tabs</td>
                </tr>
                <tr>
                    <td><b>Ctrl + Tab</b></td>
                    <td>Switch to next tab</td>
                </tr>
                <tr>
                    <td><b>Ctrl + Shift + Tab</b></td>
                    <td>Switch to previous tab</td>
                </tr>
            </table>

            <h3>Window Management</h3>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
                <tr style="background-color: #2D2D30;">
                    <th>Shortcut</th>
                    <th>Action</th>
                </tr>
                <tr>
                    <td><b>Alt + F4</b> (Windows/Linux)</td>
                    <td>Close application</td>
                </tr>
                <tr>
                    <td><b>Cmd + Q</b> (macOS)</td>
                    <td>Quit application</td>
                </tr>
            </table>

            <h3>Text Editing</h3>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
                <tr style="background-color: #2D2D30;">
                    <th>Shortcut</th>
                    <th>Action</th>
                </tr>
                <tr>
                    <td><b>Ctrl + A</b> (Cmd + A on macOS)</td>
                    <td>Select all text</td>
                </tr>
                <tr>
                    <td><b>Ctrl + C</b> (Cmd + C on macOS)</td>
                    <td>Copy selected text</td>
                </tr>
                <tr>
                    <td><b>Ctrl + V</b> (Cmd + V on macOS)</td>
                    <td>Paste text</td>
                </tr>
                <tr>
                    <td><b>Ctrl + X</b> (Cmd + X on macOS)</td>
                    <td>Cut selected text</td>
                </tr>
                <tr>
                    <td><b>Ctrl + Z</b> (Cmd + Z on macOS)</td>
                    <td>Undo</td>
                </tr>
            </table>

            <h3>Tips for Efficient Use</h3>
            <ul>
                <li>Use <b>Enter</b> to quickly send messages without clicking the Send button.</li>
                <li>Use <b>Shift + Enter</b> to compose multi-line messages.</li>
                <li>Navigate tabs with <b>Ctrl + Tab</b> to quickly check different views.</li>
                <li>Copy text from conversation or logs using standard copy shortcuts.</li>
                <li>Use mouse to click conversation items in the history panel.</li>
            </ul>

            <h3>Mouse Actions</h3>
            <ul>
                <li><b>Click conversation:</b> Switch to that conversation</li>
                <li><b>Click + button:</b> Create new conversation</li>
                <li><b>Click ✏ button:</b> Rename selected conversation</li>
                <li><b>Click 🗑 button:</b> Delete selected conversation</li>
                <li><b>Drag splitters:</b> Resize panels to your preference</li>
                <li><b>Scroll:</b> Navigate through long conversations or logs</li>
            </ul>
        """)
        layout.addWidget(text_browser)

        return widget


def show_help_dialog(parent=None):
    """
    Show the help dialog.

    Args:
        parent: Parent widget (typically the main window)

    Returns:
        HelpDialog instance
    """
    dialog = HelpDialog(parent)
    dialog.show()
    return dialog
