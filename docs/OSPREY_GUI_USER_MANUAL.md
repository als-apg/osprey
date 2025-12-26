# Osprey Framework GUI User Manual

**Version 0.9.7+**
**For End Users and Operators**

> **📝 Recent Updates:** The GUI has undergone significant internal improvements including enhanced error handling, improved state management, and better performance. All existing functionality remains unchanged while providing a more stable and responsive experience.

---

## Table of Contents

1. [Overview](#1-overview)
   - [Important: Understanding Configuration Files](#11-important-understanding-configuration-files)
   - [What is Osprey Framework GUI?](#12-what-is-osprey-framework-gui)
   - [Key Features](#13-key-features)
   - [Who Should Use This Manual?](#14-who-should-use-this-manual)
2. [Installation and Setup](#2-installation-and-setup)
   - [System Requirements](#21-system-requirements)
   - [Python Environment Setup](#22-python-environment-setup)
   - [Installation Steps](#23-installation-steps)
3. [Configuration](#3-configuration)
   - [GUI Framework Configuration](#31-gui-framework-configuration)
   - [Project Configuration (Optional)](#32-project-configuration-optional)
   - [Environment Variables (.env file)](#33-environment-variables-env-file)
   - [Detailed GUI Configuration](#34-detailed-gui-configuration-gui_configyml)
   - [Project-Specific Configuration (Advanced)](#35-project-specific-configuration-advanced)
   - [Configuration Validation](#36-configuration-validation)
   - [Configuration Quick Reference](#37-configuration-quick-reference)
4. [Getting Started](#4-getting-started)
   - [Quick Quick Start (Already Configured)](#41-quick-quick-start-already-configured)
   - [First-Time Quick Start Guide](#42-first-time-quick-start-guide)
   - [Example First Interactions](#43-example-first-interactions)
5. [Starting the GUI](#5-starting-the-gui)
6. [User Interface Overview](#6-user-interface-overview)
7. [Working with Conversations](#7-working-with-conversations)
8. [Understanding the Tabs](#8-understanding-the-tabs)
9. [Multi-User and Multi-Host Usage](#9-multi-user-and-multi-host-usage)
10. [Using with EPICS PV Gateway](#10-using-with-epics-pv-gateway)
11. [Menu Reference](#11-menu-reference)
12. [Keyboard Shortcuts](#12-keyboard-shortcuts)
13. [Troubleshooting](#13-troubleshooting)
14. [Best Practices](#14-best-practices)
15. [Glossary](#15-glossary)

---

## 1. Overview

### 1.1 Important: Understanding Configuration Files

The Osprey GUI uses **two types of configuration**:

#### 1.1.1 GUI Configuration (Framework-Level)

**Location:** `osprey/src/osprey/interfaces/pyqt/gui_config.yml`

This file is part of the Osprey framework installation and configures the GUI itself:
- AI models used for routing and orchestration
- GUI behavior settings
- API provider connections
- Default execution limits

**You need to configure:**
- `.env` file with API keys (see [Section 3.3](#33-environment-variables-env-file))
- `gui_config.yml` model settings (if using different AI providers)

**Example location:**
```
/path/to/python-env/lib/python3.x/site-packages/osprey/interfaces/pyqt/
└── gui_config.yml    # ← GUI configuration file
```

#### 1.1.2 Project Configuration (Project-Level)

**Location:** `<project-root>/config.yml` (in each discovered project)

Each Osprey project (like `its-control-assistant/`) has its own `config.yml` that configures:
- Project-specific capabilities
- EPICS gateway settings
- Archiver configuration
- Project-specific AI model preferences

**Example project structure:**
```
/path/to/its-control-assistant/    # ← A PROJECT directory
├── .env                            # Project API keys (optional)
├── config.yml                      # Project configuration
├── src/                            # Project code
└── _agent_data/                    # Project runtime data
```

> **💡 Key Distinction:** 
> - **`gui_config.yml`** = GUI framework settings (one file for the GUI)
> - **`config.yml`** = Individual project settings (one per project)
> - The GUI discovers and loads multiple projects, each with their own `config.yml`

### 1.2 What is Osprey Framework GUI?

The Osprey Framework GUI is a graphical user interface for the Osprey Framework - an AI-powered assistant system designed for scientific facilities and control systems. The GUI features a robust, well-architected codebase with comprehensive error handling and state management to ensure reliable operation.

It provides an intuitive way to interact with AI agents that can help you with tasks such as:

- 📊 Querying and analyzing control system data
- 🔍 Finding and reading EPICS Process Variables (PVs)
- 📈 Retrieving and plotting archiver data
- 🐍 Executing Python code for data analysis
- 💾 Managing conversation history across sessions
- 🔧 Coordinating multiple specialized projects

### 1.3 Key Features

✅ **Conversational Interface** - Natural language interaction with AI assistants
✅ **Multi-Project Support** - Work with multiple specialized assistants simultaneously
✅ **Real-Time Monitoring** - See what the AI is doing as it processes your requests
✅ **Conversation History** - All conversations are automatically saved and can be resumed
✅ **Safety Controls** - Built-in approval workflows for critical operations
✅ **EPICS Integration** - Direct connection to EPICS control systems via PV Gateway
✅ **Multi-User Ready** - Multiple users can run the GUI on the same or different hosts
✅ **Robust Error Handling** - Comprehensive error detection and recovery mechanisms
✅ **Optimized Performance** - Efficient state management and resource utilization

### 1.4 Who Should Use This Manual?

This manual is designed for:
- Facility operators and scientists
- Control room personnel
- Beamline users
- Anyone who needs to interact with control systems using AI assistance

> **📝 Note:** This is a **user manual**, not a developer guide. For information about developing capabilities or extending the framework, please refer to the developer documentation.

---

## 2. Installation and Setup

### 2.1 System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 4 GB RAM
- 1 GB free disk space
- Network connection (for cloud-based AI models)

**Recommended:**
- Python 3.10 or higher
- 8 GB RAM
- 5 GB free disk space
- Stable network connection

**Operating Systems:**
- ✅ Linux (tested on RHEL, Ubuntu, Debian)
- ✅ macOS 10.14 or higher
- ✅ Windows 10 or higher

### 2.2 Python Environment Setup

**Understanding the Python Environment:**

The Osprey Framework is a Python package that installs command-line tools into your Python environment. When you install it, the `osprey-gui` command becomes available in your shell.

**Installation Methods:**

**Option 1: System-Wide Installation (Simple)**
```bash
pip install osprey-framework
```

**Option 2: Virtual Environment (Recommended)**
```bash
# Create a virtual environment
python -m venv osprey-env

# Activate it
source osprey-env/bin/activate  # Linux/macOS
# OR
osprey-env\Scripts\activate     # Windows

# Install Osprey
pip install osprey-framework
```

**Option 3: Development Installation**
```bash
# Clone the repository
git clone https://github.com/als-apg/osprey.git
cd osprey

# Install in editable mode
pip install -e .
```

**How `osprey-gui` Gets Into Your PATH:**

When you install the Osprey Framework with `pip`, it automatically installs executable scripts into your Python environment's `bin/` (or `Scripts/` on Windows) directory:

- **Virtual environment:** `osprey-env/bin/osprey-gui`
- **System Python:** `/usr/local/bin/osprey-gui` or `~/.local/bin/osprey-gui`
- **Conda environment:** `~/anaconda3/envs/myenv/bin/osprey-gui`

These directories are automatically added to your system's `PATH` when you activate the environment, making `osprey-gui` available as a command.

**Verifying Installation:**

```bash
# Check if osprey-gui is available
which osprey-gui          # Linux/macOS
where osprey-gui          # Windows

# Check Osprey version
osprey --version

# List all Osprey commands
osprey --help
```

**Troubleshooting PATH Issues:**

If `osprey-gui` is not found:

1. **Ensure your Python environment is activated:**
   ```bash
   source osprey-env/bin/activate  # If using virtual environment
   ```

2. **Check if the script was installed:**
   ```bash
   pip show osprey-framework
   ```

3. **Manually add to PATH (if needed):**
   ```bash
   # Find where pip installed scripts
   python -m site --user-base
   
   # Add to PATH (Linux/macOS - add to ~/.bashrc)
   export PATH="$HOME/.local/bin:$PATH"
   ```

4. **Reinstall if necessary:**
   ```bash
   pip uninstall osprey-framework
   pip install osprey-framework
   ```

### 2.3 Installation Steps

1. **Set Up Python Environment** (choose one method from above)
   ```bash
   # Example: Using virtual environment
   python -m venv osprey-env
   source osprey-env/bin/activate
   ```

2. **Install Osprey Framework**
   ```bash
   pip install osprey-framework
   ```

3. **Verify Installation**
   ```bash
   osprey --version
   which osprey-gui  # Should show the path to the command
   ```

4. **Create Your Project** (if starting from scratch)
   ```bash
   osprey init my-assistant --template control_assistant
   cd my-assistant
   ```

> **📝 Note:** After installation, you need to configure the GUI before first use. Continue to [Section 3: Configuration](#3-configuration) for setup instructions.

---

## 3. Configuration

**⚠️ IMPORTANT: Complete this configuration before using the GUI for the first time!**

The Osprey GUI requires configuration at the framework level. Project-level configuration is optional and only needed if you're working with specific Osprey projects.

### 3.1 GUI Framework Configuration

**Location:** `osprey/src/osprey/interfaces/pyqt/gui_config.yml`

This file is installed with the Osprey framework and configures the GUI application itself.

**To find this file:**
```bash
# Find your Python site-packages directory
python -c "import osprey; import os; print(os.path.dirname(osprey.__file__))"

# The gui_config.yml is in: <result>/interfaces/pyqt/gui_config.yml
```

**What to Configure:**

1. **API Provider Settings** - Your facility's AI provider
2. **Model Configuration** - Which AI models to use
3. **GUI Behavior Settings** - How the GUI operates

> **📝 Note:** Most users can use the default `gui_config.yml` and only need to set environment variables for API keys (see [Section 3.3](#33-environment-variables-env-file)).

### 3.2 Project Configuration (Optional)

**Location:** `<project-directory>/config.yml` (for each project)

If you're working with specific Osprey projects (like `its-control-assistant`), each project has its own configuration file.

**Project Directory Structure:**
```
/path/to/its-control-assistant/    # ← A discovered project
├── config.yml                      # Project-specific configuration
├── .env                            # Project API keys (optional)
├── src/                            # Project source code
│   └── its_control_assistant/
│       └── registry.py
└── _agent_data/                    # Project runtime data
    ├── conversations/
    ├── plots/
    └── checkpoints/
```

**Project Configuration Includes:**
- EPICS gateway settings
- Archiver configuration  
- Project-specific capabilities
- Control system settings

> **💡 Key Point:** The GUI automatically discovers projects in subdirectories. You only need to configure projects if you're using them. For basic GUI usage, only `gui_config.yml` configuration is needed.

### 3.3 Environment Variables (.env file)

**Location:** Can be in multiple places (checked in this order):

1. **Osprey framework directory:** `osprey/.env` (recommended for GUI)
2. **Project directories:** `<project-directory>/.env` (for project-specific keys)
3. **System environment:** Set in your shell

The `.env` file stores sensitive information like API keys. This file should **not** be committed to version control.

**Required Variables:**

```bash
# API Keys - At least ONE provider is required
ANTHROPIC_API_KEY=your-anthropic-key      # Recommended: Claude models
CBORG_API_KEY=your-cborg-key              # LBNL institutional provider
STANFORD_API_KEY=your-stanford-key        # Stanford AI Playground
OPENAI_API_KEY=your-openai-key            # OpenAI GPT models
GOOGLE_API_KEY=your-google-key            # Google Gemini models
ARGO_API_KEY=your-argo-key                # Argo Bridge (ANL)

# Optional: Debugging
DEBUG=1                                    # Enable debug output
```

**Setting Environment Variables:**

**Option 1: Using .env file (Recommended)**
```bash
# Create .env file in osprey framework directory
cd <path-to-osprey-installation>
nano .env            # Edit with your values
```

**Option 2: Shell environment**
```bash
# Temporary (current session only)
export ARGO_API_KEY="your-key"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export ARGO_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

### 3.4 Detailed GUI Configuration (gui_config.yml)

**Location:** `osprey/src/osprey/interfaces/pyqt/gui_config.yml`

This section provides details on configuring the GUI framework settings.

#### 3.4.1 AI Model Configuration

Configure which AI models to use for different GUI functions:

```yaml
# In gui_config.yml
models:
  classifier:
    provider: argo              # Provider name
    model_id: gpt4o            # Model identifier
  orchestrator:
    provider: argo
    model_id: gpt4o
    max_tokens: 4096
  response:
    provider: argo
    model_id: gpt4o
```

**Available Providers:** `argo`, `cborg`, `openai`, `anthropic`, `google`, `ollama`, `stanford`

#### 3.4.2 API Provider Configuration

Define connection details for AI providers:

```yaml
# In gui_config.yml
api:
  providers:
    argo:
      api_key: ${ARGO_API_KEY}              # References .env variable
      base_url: https://argo-bridge.cels.anl.gov
    openai:
      api_key: ${OPENAI_API_KEY}
      base_url: https://api.openai.com/v1
```

> **🔒 Security:** Always use `${VARIABLE_NAME}` syntax to reference environment variables. Never hardcode API keys.

#### 3.4.3 GUI Behavior Settings

Configure how the GUI behaves:

```yaml
# In gui_config.yml
gui:
  use_persistent_conversations: true    # Save conversation history
  conversation_storage_mode: json       # Storage format
  redirect_output_to_gui: true          # Show output in GUI
  suppress_terminal_output: false       # Also show in terminal
```

### 3.5 Project-Specific Configuration (Advanced)

If you're working with specific projects that the GUI discovers, each project may have its own `config.yml` with settings like:

- **EPICS Gateway Configuration** - Connection to control system
- **Archiver Settings** - Historical data retrieval
- **Safety Controls** - Write permissions and approval workflows
- **Project Capabilities** - Available functions

**Example project config.yml sections:**

```yaml
# In <project-directory>/config.yml
control_system:
  type: epics
  writes_enabled: false
  connector:
    epics:
      gateways:
        read_only:
          address: pvgatemain1.aps4.anl.gov
          port: 5064

archiver:
  type: epics_archiver
  epics_archiver:
    url: https://pvarchiver.aps.anl.gov
```

> **📝 Note:** Project configuration is only needed if you're using discovered projects. For basic GUI usage without projects, only `gui_config.yml` is required.

### 3.6 Configuration Validation

**Validate your configuration:**

```bash
# The GUI will validate configuration on startup
osprey-gui

# Check for errors in the System Information tab
```

**Common Configuration Issues:**

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing API key | "API key not found" error | Set in `.env` file or environment |
| Wrong provider | "Provider not configured" | Verify `api.providers` in `gui_config.yml` |
| Invalid YAML | GUI won't start | Check indentation in config files |

### 3.7 Configuration Quick Reference

**Essential GUI Configuration Checklist:**

- [ ] Set API key environment variable (e.g., `ARGO_API_KEY`)
- [ ] Verify `gui_config.yml` has correct provider configured
- [ ] Check `gui_config.yml` model settings match your provider
- [ ] Optionally configure project `config.yml` files if using projects
- [ ] Test with `osprey-gui` to verify configuration loads

**For Project Configuration (if using projects):**

- [ ] Configure EPICS gateway in project `config.yml`
- [ ] Set archiver URL if using historical data
- [ ] Configure safety settings (`writes_enabled: false`)
- [ ] Set project timezone

---

## 4. Getting Started

### 4.1 Quick Quick Start (Already Configured)

> **⚡ For Users Who Have Already Completed Installation and Configuration**
>
> If you have already installed Osprey and configured your environment variables with API keys, you can start immediately:

**[📸 Screenshot Placeholder: Main GUI Window]**

```bash
# 1. Activate your Python environment (if using one)
source osprey-env/bin/activate

# 2. Launch the GUI
osprey-gui

# 3. Wait for "✅ Framework initialized successfully"
# 4. Start chatting!
```

**First message to try:**
```
What capabilities are available?
```

> **📝 Note:** If you haven't installed or configured Osprey yet, go back to [Section 2: Installation and Setup](#2-installation-and-setup) and [Section 3: Configuration](#3-configuration).

---

### 4.2 First-Time Quick Start Guide

**[📸 Screenshot Placeholder: Main GUI Window]**

Now that you've completed the configuration, follow these steps to start using the Osprey GUI:

1. **Launch the Application**
   ```bash
   osprey-gui
   ```

2. **Wait for Initialization**
   - The GUI will discover available projects
   - Status messages appear in the System Information tab
   - Wait for "✅ Framework initialized successfully"

3. **Type Your First Message**
   - Click in the input field at the bottom
   - Type a question or request
   - Press **Enter** to send (or click **Send** button)

4. **View the Response**
   - The AI's response appears in the conversation area
   - Status updates show in the right panel
   - Processing details appear in the LLM Details tab

### 4.3 Example First Interactions

**Simple Query:**
```
What capabilities are available?
```

**Data Request (if using EPICS projects):**
```
Show me the current value of SR:C01-BI:G02A<BPM:X>Pos-I
```

**Analysis Task (if using archiver):**
```
Plot the beam current for the last hour
```

---

## 5. Starting the GUI

### 5.1 Basic Launch

**From Command Line:**
```bash
# Make sure your Python environment is activated
source osprey-env/bin/activate  # If using virtual environment

# Launch the GUI
osprey-gui
```

**With Specific Configuration:**
```bash
osprey-gui --config /path/to/custom/gui_config.yml
```

> **📝 Note:** If you get "command not found", ensure your Python environment is activated and Osprey is installed. See [Section 2.2: Python Environment Setup](#22-python-environment-setup).

### 5.2 Launch Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config PATH` | Use specific config file | `osprey-gui --config custom.yml` |
| `--help` | Show help message | `osprey-gui --help` |

### 5.3 What Happens at Startup

**[📸 Screenshot Placeholder: Startup Sequence]**

When the GUI starts, it:

1. ✅ Loads `gui_config.yml` configuration
2. ✅ Discovers available projects in subdirectories
3. ✅ Initializes AI model connections
4. ✅ Loads conversation history
5. ✅ Connects to EPICS gateway (if projects configured)
6. ✅ Displays "Ready" status

**Typical startup time:** 5-15 seconds depending on number of projects

---

## 6. User Interface Overview

### 6.1 Main Window Layout

**[📸 Screenshot Placeholder: Main Window with Labels]**

The GUI is organized into several key areas:

```
┌─────────────────────────────────────────────────────────────┐
│  Menu Bar: File | Settings | Help                           │
├──────────┬────────────────────────────┬─────────────────────┤
│          │                            │                     │
│ Project  │   Conversation Display     │   Status Log        │
│ Control  │   (Center Panel)           │   (Right Panel)     │
│ &        │                            │                     │
│ History  │   Your messages appear     │   Real-time status  │
│ (Left)   │   in purple                │   updates           │
│          │   AI responses in white    │                     │
│          │                            │                     │
├──────────┴────────────────────────────┴─────────────────────┤
│  Input Field                    [Send] [New Conversation]   │
└─────────────────────────────────────────────────────────────┘
│  Status Bar: Ready - Type your message                      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Panel Descriptions

#### Left Panel: Project Control & Conversation History

**Project Control Section:**
- **Active Project Dropdown** - Select which project to use
- **Routing Mode Indicator** - Shows automatic or manual routing
- **Cache Controls** - Manage routing cache
- **Context Controls** - Manage conversation context

**Conversation History Section:**
- **Conversation List** - All your saved conversations
- **Add Button** - Create new conversation
- **Delete Button** - Remove selected conversation
- **Edit Button** - Rename conversation

#### Center Panel: Conversation Display

- **Message Area** - Shows conversation history
- **Color Coding:**
  - 🟣 Purple: Your messages
  - ⚪ White: AI responses
  - 🟢 Green: Success notifications
  - 🔴 Red: Errors
  - 🟡 Gold: Important information

#### Right Panel: Status Log

- **Real-time Updates** - See what's happening
- **Component Colors:**
  - Cyan: System messages
  - Orange: Monitoring events
  - Pink: Gateway operations
  - Blue: Time parsing
  - Green: Success

#### Bottom: Input Area

- **Text Input Field** - Type your messages here
- **Send Button** - Submit your message
- **New Conversation Button** - Start fresh conversation

### 6.3 Tab Navigation

**[📸 Screenshot Placeholder: Tab Bar]**

The GUI has multiple tabs for different views:

| Tab | Icon | Purpose |
|-----|------|---------|
| **Conversation** | 💬 | Main chat interface |
| **LLM Details** | 🤖 | AI processing details |
| **Tool Usage** | 🔧 | Capability execution logs |
| **Discovered Projects** | 📁 | Project management |
| **System Information** | ℹ️ | Technical details |
| **💾 Memory** | 💾 | Memory usage monitoring |
| **📊 Analytics** | 📊 | Usage statistics |

---

## 7. Working with Conversations

### 7.1 Creating a New Conversation

**[📸 Screenshot Placeholder: New Conversation Dialog]**

**Method 1: Using the Button**
1. Click **"New Conversation"** button at bottom
2. Conversation is created with auto-generated name
3. Start typing your first message

**Method 2: Using the Menu**
1. Go to **File → New Conversation**
2. New conversation appears in history list
3. Automatically becomes active

**Method 3: Quick Add**
1. Click **"+"** button in conversation history panel
2. Instant new conversation

### 7.2 Switching Between Conversations

**[📸 Screenshot Placeholder: Conversation List]**

1. Look at the conversation history list (left panel)
2. Click on any conversation to switch to it
3. Active conversation shows **▶** marker and green highlight
4. All messages are preserved when switching

> **💡 Tip:** Conversations are sorted by most recent activity

### 7.3 Renaming Conversations

**[📸 Screenshot Placeholder: Rename Dialog]**

1. Select a conversation in the history list
2. Click the **"Edit"** (✏️) button
3. Enter a descriptive name
4. Click **OK**

**Good naming examples:**
- "Beam Current Analysis - Dec 18"
- "Magnet Troubleshooting"
- "Daily Operations Check"

### 7.4 Deleting Conversations

**[📸 Screenshot Placeholder: Delete Confirmation]**

1. Select conversation(s) to delete
2. Click the **"Del"** (🗑️) button
3. Confirm deletion in dialog
4. Conversation is permanently removed

> **⚠️ Warning:** Deleted conversations cannot be recovered!

### 7.5 Conversation Persistence

**How Conversations are Saved:**

- ✅ **Automatic Saving** - Every message is saved immediately
- ✅ **Storage Options:**
  - PostgreSQL database (if configured)
  - Local JSON files (default)
- ✅ **Session Recovery** - Conversations persist across GUI restarts
- ✅ **Multi-User** - Each user has their own conversation history

**Storage Location:**
```
<osprey-installation>/
  └── _gui_data/
      └── conversations/
          ├── conversation_001.json
          ├── conversation_002.json
          └── ...
```

---

## 8. Understanding the Tabs

### 8.1 Conversation Tab

**[📸 Screenshot Placeholder: Conversation Tab]**

**Purpose:** Main interface for chatting with the AI assistant

**Features:**
- Three-panel layout (history, conversation, status)
- Real-time message display
- Color-coded messages for easy reading
- Status updates during processing

**How to Use:**
1. Type your message in the input field
2. Press **Enter** or click **Send**
3. Watch the status log for processing updates
4. Read the AI's response in the conversation area

**Input Tips:**
- Press **Enter** to send
- Press **Shift+Enter** for new line
- Input field supports multi-line messages

### 8.2 LLM Details Tab

**[📸 Screenshot Placeholder: LLM Details Tab]**

**Purpose:** View detailed AI processing information

**What You'll See:**
- Timestamped events
- AI model being used
- Processing stages
- Token usage (if available)

**Event Types:**
- 🟢 **START** - AI begins processing
- 🔵 **THINKING** - AI is analyzing
- 🟡 **TOOL_CALL** - AI is using a capability
- ✅ **COMPLETE** - Processing finished

**Example Output:**
```
[14:23:45] [START] Processing user query
[14:23:46] [THINKING] Analyzing request for PV data
[14:23:47] [TOOL_CALL] Using channel_read capability
[14:23:49] [COMPLETE] Response generated
```

### 8.3 Tool Usage Tab

**[📸 Screenshot Placeholder: Tool Usage Tab]**

**Purpose:** See which capabilities the AI used and why

**Information Displayed:**
- Capability name
- Task objective
- Success/failure status
- Execution time
- Detailed reasoning

**Example Entry:**
```
[14:23:47] Capability: channel_read
✅ Successfully read PV value
⏱️ Execution time: 1.2 seconds
Reasoning: User requested current value of beam position PV
```

**Status Indicators:**
- ✅ Green checkmark = Success
- ❌ Red X = Failed
- ⏱️ Clock = Timing information

### 8.4 Discovered Projects Tab

**[📸 Screenshot Placeholder: Projects Tab]**

**Purpose:** Manage multiple Osprey projects

**Table Columns:**
| Column | Description |
|--------|-------------|
| **Status** | Enable/disable checkbox |
| **Project Name** | Name of the project |
| **Capabilities** | Number and list of capabilities |
| **Models** | Configured AI models |
| **Path** | Project directory location |
| **Config File** | Configuration filename |
| **Model Config** | Button to configure models |

**Actions:**
- **🔄 Refresh** - Re-scan for projects
- **Enable/Disable** - Toggle project availability
- **Configure** - Set model preferences

**Project Status:**
- 🟢 **Enabled** - Available for routing
- ⚪ **Disabled** - Not used

### 8.5 System Information Tab

**[📸 Screenshot Placeholder: System Info Tab]**

**Purpose:** View technical session details

**Display Modes:**
- **📋 Grouped View** - Messages organized by type
- **📄 List View** - Chronological list

**Information Shown:**
- Thread ID
- Configuration file path
- Number of capabilities
- System messages and logs
- Initialization status

**Toggle Button:**
- Click **"📋 Grouped View"** to switch modes
- Click **"📄 List View"** for chronological

### 8.6 Analytics Tab

**[📸 Screenshot Placeholder: Analytics Tab]**

**Purpose:** View usage statistics and performance metrics

**Metrics Displayed:**
- Query routing statistics
- Cache hit/miss rates
- Response times
- Capability usage frequency
- Project selection patterns

### 8.7 Memory Tab

**[📸 Screenshot Placeholder: Memory Tab]**

**Purpose:** Monitor memory usage of GUI and framework processes

**Features:**
- **System Memory Summary** - Overall system memory usage with progress bar
- **Framework Total** - Combined memory usage of all framework processes
- **Process Details Table** - Detailed breakdown by process type
- **Memory Trend Analysis** - Real-time trend detection (increasing/decreasing/stable)
- **Automatic Monitoring** - Configurable update intervals
- **Threshold Alerts** - Warning and critical memory thresholds

**Display Sections:**

1. **System Memory Bar**
   - Shows total system memory usage
   - Color-coded: Green (normal), Orange (75%+), Red (90%+)
   - Displays used/total MB and percentage

2. **Framework Total Bar**
   - Shows combined memory of all framework processes
   - Color-coded based on configured thresholds
   - Displays total MB and percentage of system memory

3. **Memory Trend Indicator**
   - 📈 Increasing - Memory usage growing over time
   - ⚠️📈 Rapidly Increasing - Memory growing >10 MB/min
   - 📉 Decreasing - Memory usage declining
   - ➡️ Stable - Memory usage steady
   - Shows rate of change in MB/minute

4. **Process Details Table**
   - **Type** - Process category (GUI, Child, Docker, Podman)
   - **PID/ID** - Process ID or container ID
   - **Name** - Process or container name
   - **Memory (MB)** - Current memory usage
   - **CPU %** - CPU utilization
   - **Status** - Process/container status

**Controls:**
- **⏸️ Pause / ▶️ Resume** - Toggle automatic monitoring
- **🔄 Refresh** - Manual update of statistics

**Monitored Processes:**
- GUI process itself
- Child processes spawned by framework
- Docker containers started by framework
- Podman containers started by framework

**Configuration:**
Settings → Development/Debug → Memory Monitoring section:
- Enable/disable automatic monitoring
- Warning threshold (default: 500 MB)
- Critical threshold (default: 1000 MB)
- Check interval (default: 5 seconds)

**Trend Analysis:**
- Warmup period: 2 minutes after startup
- Requires 5+ measurements for trend calculation
- Tracks last 20 measurements
- Threshold: ±0.5 MB/min to filter noise

**Status Indicators:**
- ✅ Normal - Below warning threshold
- ⚠️ WARNING - Above warning threshold
- 🔴 CRITICAL - Above critical threshold

> **📝 Note:** Memory monitoring starts automatically if enabled in settings. Trend analysis becomes available after 2-minute warmup period.

---

## 9. Multi-User and Multi-Host Usage

### 9.1 Multiple Users on Same Host

**[📸 Screenshot Placeholder: Multi-User Setup Diagram]**

**Scenario:** Several users running the GUI on the same computer

**How It Works:**
- Each user has their own conversation history
- Conversations are stored per user account
- Shared configuration files (read-only)
- Independent sessions

**Setup:**
```bash
# User 1
user1@host$ osprey-gui

# User 2 (different terminal/session)
user2@host$ osprey-gui
```

**Isolation:**
- ✅ Separate conversation histories
- ✅ Independent settings
- ✅ No interference between users
- ⚠️ Shared EPICS gateway connection (if using projects)

### 9.2 Multiple Users on Different Hosts

**[📸 Screenshot Placeholder: Multi-Host Diagram]**

**Scenario:** Users on different computers accessing the same facility

**Requirements:**
- Network access to EPICS PV Gateway (if using projects)
- Individual API keys (recommended)

**Setup on Each Host:**
```bash
# Host 1
user@host1$ osprey-gui

# Host 2
user@host2$ osprey-gui
```

**Considerations:**
- Each host needs network access to:
  - AI model provider (Argo, OpenAI, etc.)
  - EPICS PV Gateway (if using projects)
  - Archiver service (if using projects)
- Conversation histories are local to each host
- Settings can be shared or independent

### 9.3 Shared Configuration

**Recommended Directory Structure:**
```
/shared/osprey/
  ├── gui_config.yml              # Shared GUI configuration
  └── users/
      ├── user1/
      │   └── _gui_data/          # User 1's data
      └── user2/
          └── _gui_data/          # User 2's data
```

### 9.4 Best Practices for Multi-User

✅ **Do:**
- Use separate conversation directories per user
- Share read-only configuration files
- Document facility-specific settings
- Use descriptive conversation names
- Coordinate EPICS write access (if using projects)

❌ **Don't:**
- Share conversation history between users
- Modify shared configs without coordination
- Enable EPICS writes without approval (if using projects)
- Use the same API keys for all users

---

## 10. Using with EPICS PV Gateway

> **📝 Note:** This section applies only if you're using Osprey projects that have EPICS capabilities configured.

### 10.1 What is EPICS PV Gateway?

**[📸 Screenshot Placeholder: EPICS Architecture Diagram]**

The EPICS PV Gateway is a network gateway that:
- Provides controlled access to EPICS Process Variables
- Separates read-only and write-access connections
- Implements security and access control
- Reduces network load

**Osprey GUI Connection:**
```
┌─────────────┐         ┌──────────────┐         ┌─────────┐
│ Osprey GUI  │ ──────> │ PV Gateway   │ ──────> │  IOCs   │
└─────────────┘         └──────────────┘         └─────────┘
                        Port 5064 (read)
                        Port 5065 (write)
```

### 10.2 Gateway Configuration

**Location:** `<project-directory>/config.yml` (in each project)

**Section:** `control_system.connector.epics.gateways` AND `execution.epics.gateways`

```yaml
# In <project-directory>/config.yml

# Gateway configuration for control system operations
control_system:
  type: epics
  connector:
    epics:
      timeout: 5.0
      gateways:
        read_only:
          address: pvgatemain1.aps4.anl.gov
          port: 5064
          use_name_server: false
        write_access:
          address: pvgatemain1.aps4.anl.gov
          port: 5084
          use_name_server: false

# Gateway configuration for Python execution environment
execution:
  epics:
    timeout: 5.0
    gateways:
      read_only:
        address: pvgatemain1.aps4.anl.gov
        port: 5064
      write_access:
        address: pvgatemain1.aps4.anl.gov
        port: 5084
```

> **⚠️ Important:** Configure gateways in **both** `control_system.connector.epics` and `execution.epics` sections.

**Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `address` | Gateway hostname or IP | `pvgatemain1.aps4.anl.gov` |
| `port` | Gateway port number | `5064` (read), `5065` (write) |
| `timeout` | Connection timeout (seconds) | `5.0` |
| `use_name_server` | Use EPICS name server | `false` |

### 10.3 Read-Only vs Write Access

**Read-Only Gateway (Port 5064):**
- ✅ Safe for all users
- ✅ No risk of changing PV values
- ✅ Used for monitoring and queries
- ✅ Default for most operations

**Write-Access Gateway (Port 5065):**
- ⚠️ Can modify PV values
- ⚠️ Requires approval workflow
- ⚠️ Should be restricted
- ⚠️ Used only when necessary

**Enabling Write Access:**
```yaml
# In <project-directory>/config.yml
control_system:
  writes_enabled: true  # Set to false for read-only (RECOMMENDED)
```

### 10.4 Testing Gateway Connection

**From GUI:**
1. Start the GUI
2. Check System Information tab for connection status
3. Try reading a PV:
   ```
   What is the current value of SR:C01-BI:G02A<BPM:X>Pos-I?
   ```

**From Command Line:**
```bash
# Test read access
caget -S pvgatemain1.aps4.anl.gov:5064 YOUR:PV:NAME

# Test write access (if enabled)
caput -S pvgatemain1.aps4.anl.gov:5065 YOUR:PV:NAME value
```

### 10.5 Troubleshooting Gateway Issues

**Problem: Cannot connect to gateway**

**Solutions:**
1. Check network connectivity:
   ```bash
   ping pvgatemain1.aps4.anl.gov
   ```

2. Verify gateway is running:
   ```bash
   telnet pvgatemain1.aps4.anl.gov 5064
   ```

3. Check firewall rules
4. Verify gateway address in project `config.yml`

**Problem: PV not found**

**Solutions:**
1. Verify PV name spelling
2. Check if PV exists on gateway
3. Ensure gateway has access to that IOC
4. Try from command line: `caget -S gateway:port PV_NAME`

---

## 11. Menu Reference

### 11.1 File Menu

**[📸 Screenshot Placeholder: File Menu]**

| Menu Item | Shortcut | Description |
|-----------|----------|-------------|
| **New Conversation** | - | Create a new conversation |
| **Clear Conversation** | - | Remove all messages from current conversation |
| **Exit** | Alt+F4 | Close the application |

### 11.2 Settings Menu

**[📸 Screenshot Placeholder: Settings Menu]**

| Menu Item | Description |
|-----------|-------------|
| **Framework Settings** | Open settings dialog to configure behavior |

**Settings Dialog Tabs:**

**Agent Control:**
- Planning Mode
- EPICS Writes Enable/Disable
- Task Extraction Bypass
- Capability Selection Bypass

**Approval:**
- Global Approval Mode (disabled/selective/all_capabilities)
- Python Execution Approval
- Python Approval Mode (disabled/epics_writes/all_code)
- Memory Approval

**Execution Limits:**
- Max Reclassifications
- Max Planning Attempts
- Max Step Retries
- Max Execution Time
- Max Concurrent Classifications

**GUI Settings:**
- Save Conversation History
- Message Storage Mode (json/postgresql)
- Redirect Output to GUI
- Group System Messages
- Suppress Terminal Output
- Enable Routing Feedback

**Development/Debug:**
- Debug Mode (enables DEBUG logging level)
- Verbose Logging
- Raise Raw Errors
- Save Prompts to Files
- Show Prompts in Console
- Prompts: Latest Only
- **Memory Monitoring:**
  - Enable Memory Monitor
  - Warning Threshold (MB)
  - Critical Threshold (MB)
  - Check Interval (seconds)

**Advanced Routing:**
- **Cache Configuration:**
  - Enable Routing Cache
  - Cache Size
  - Cache TTL (seconds)
  - Similarity Threshold
- **Advanced Cache Invalidation:**
  - Enable Advanced Invalidation
  - Adaptive TTL
  - Probabilistic Expiration
  - Event-Driven Invalidation
- **Semantic Context Analysis:**
  - Enable Semantic Analysis
  - Similarity Threshold
  - Topic Similarity
  - Max Context History
- **Orchestration & Analytics:**
  - Max Parallel Queries
  - Analytics Max History

### 11.3 Help Menu

**[📸 Screenshot Placeholder: Help Menu]**

| Menu Item | Shortcut | Description |
|-----------|----------|-------------|
| **Help Documentation** | F1 | Open comprehensive help dialog |
| **About** | - | Show version and system information |

---

## 12. Keyboard Shortcuts

### 12.1 Quick Reference

**[📸 Screenshot Placeholder: Keyboard Shortcuts Diagram]**

#### Application

| Shortcut | Action |
|----------|--------|
| **F1** | Open Help Documentation |
| **Alt+F4** (Win/Linux) | Close Application |
| **Cmd+Q** (macOS) | Quit Application |

#### Message Input

| Shortcut | Action |
|----------|--------|
| **Enter** | Send message |
| **Shift+Enter** | Insert new line |

#### Navigation

| Shortcut | Action |
|----------|--------|
| **Tab** | Navigate between UI elements |
| **Ctrl+Tab** | Switch to next tab |
| **Ctrl+Shift+Tab** | Switch to previous tab |

#### Text Editing

| Shortcut | Action |
|----------|--------|
| **Ctrl+A** (Cmd+A) | Select all text |
| **Ctrl+C** (Cmd+C) | Copy selected text |
| **Ctrl+V** (Cmd+V) | Paste text |
| **Ctrl+X** (Cmd+X) | Cut selected text |
| **Ctrl+Z** (Cmd+Z) | Undo |

### 12.2 Mouse Actions

| Action | Result |
|--------|--------|
| **Click conversation** | Switch to that conversation |
| **Click + button** | Create new conversation |
| **Click ✏️ button** | Rename selected conversation |
| **Click 🗑️ button** | Delete selected conversation |
| **Drag splitters** | Resize panels |
| **Scroll wheel** | Navigate through messages |

---

## 13. Troubleshooting

### 13.1 Common Issues

#### Issue: GUI Won't Start

**[📸 Screenshot Placeholder: Error Message]**

**Symptoms:**
- Application crashes on startup
- Error messages in terminal
- Window doesn't appear
- "command not found: osprey-gui"

**Solutions:**

> **💡 Note:** The GUI includes enhanced error handling that provides detailed error messages to help diagnose startup issues. Check the terminal output for specific error information.

1. **Check Python Environment:**
   ```bash
   # Verify Python version
   python --version  # Should be 3.8 or higher
   
   # Check if environment is activated
   which python      # Should point to your virtual environment
   ```

2. **Verify Installation:**
   ```bash
   # Check if Osprey is installed
   pip list | grep osprey
   
   # Check if osprey-gui command exists
   which osprey-gui  # Should show path to command
   ```

3. **Activate Python Environment:**
   ```bash
   # If using virtual environment
   source osprey-env/bin/activate  # Linux/macOS
   osprey-env\Scripts\activate     # Windows
   ```

4. **Reinstall if Necessary:**
   ```bash
   pip install --upgrade osprey-framework
   ```

5. **Look for Error Messages:**
   ```bash
   osprey-gui 2>&1 | tee gui_error.log
   ```

6. **Check PATH:**
   ```bash
   # Ensure Python scripts directory is in PATH
   echo $PATH | grep -o "[^:]*bin"
   ```

#### Issue: No Response from AI

**Symptoms:**
- Message sent but no response
- Status shows "Processing..." indefinitely
- Error in status log

**Solutions:**

1. **Check API Keys:**
   ```bash
   echo $ARGO_API_KEY  # Should show your key
   ```

2. **Verify Network:**
   ```bash
   ping argo-bridge.cels.anl.gov
   ```

3. **Check Model Configuration:**
   - Open Settings → Framework Settings
   - Verify model provider is configured
   - Check `gui_config.yml` for correct model IDs

4. **Review Status Log:**
   - Switch to System Information tab
   - Look for error messages
   - Check for timeout errors

#### Issue: Conversations Not Saving

**Symptoms:**
- Conversations disappear after restart
- "Save failed" errors
- Empty conversation list

**Solutions:**

1. **Check Setting:**
   - Settings → Framework Settings
   - Ensure "Save Conversation History" is enabled

2. **Verify Permissions:**
   ```bash
   ls -la <osprey-installation>/_gui_data/conversations/
   # Should show write permissions
   ```

3. **Check Disk Space:**
   ```bash
   df -h .
   ```

4. **Check gui_config.yml:**
   ```yaml
   # In gui_config.yml
   gui:
     use_persistent_conversations: true
     conversation_storage_mode: json
   ```

#### Issue: EPICS Connection Failed

> **📝 Note:** This only applies if you're using projects with EPICS capabilities.

**Symptoms:**
- "Gateway timeout" errors
- "PV not found" messages
- Cannot read PV values

**Solutions:**

1. **Test Gateway:**
   ```bash
   caget -S gateway-host:5064 YOUR:PV:NAME
   ```

2. **Check Configuration:**
   ```yaml
   # Verify in project config.yml
   control_system:
     connector:
       epics:
         gateways:
           read_only:
             address: correct-gateway-host
             port: 5064
   ```

3. **Verify Network:**
   ```bash
   ping gateway-host
   telnet gateway-host 5064
   ```

4. **Check Firewall:**
   - Ensure ports 5064-5065 are open
   - Contact facility IT if needed

### 13.2 Diagnostic Tools

**Enable Debug Mode:**

1. Settings → Framework Settings → Development/Debug
2. Enable "Debug Mode" (enables DEBUG logging level)
3. Optional: Enable "Show Prompts in Console"
4. Optional: Enable "Save Prompts to Files"
5. Click Save

> **💡 Tip:** The GUI's improved error handling system provides more detailed diagnostic information in debug mode, making it easier to identify and resolve issues. Debug mode changes apply immediately when you click Save.

**Memory Monitoring:**

1. Settings → Framework Settings → Development/Debug → Memory Monitoring
2. Verify "Enable Memory Monitor" is checked
3. Adjust thresholds if needed:
   - Warning Threshold: 500 MB (default)
   - Critical Threshold: 1000 MB (default)
4. Set Check Interval: 5 seconds (default)
5. Click Save
6. View memory statistics in the 💾 Memory tab

**Collect Diagnostic Information:**

```bash
# System information
osprey --version
python --version
uname -a

# Find gui_config.yml location
python -c "import osprey; import os; print(os.path.dirname(osprey.__file__))"
```

### 13.3 Getting Help

**Before Asking for Help:**

✅ Check this manual  
✅ Review error messages in Status Log  
✅ Check System Information tab  
✅ Try restarting the GUI  
✅ Verify configuration files

**When Reporting Issues:**

Include:
1. Osprey version (`osprey --version`)
2. Python version
3. Operating system
4. Error messages (copy from Status Log)
5. Steps to reproduce
6. Configuration file (remove sensitive data)

**Where to Get Help:**
- 📧 Facility support team
- 🐛 GitHub Issues: https://github.com/als-apg/osprey
- 📚 Documentation: https://als-apg.github.io/osprey

---

## 14. Best Practices

### 14.1 Daily Usage

✅ **Do:**
- Start with a new conversation for each topic
- Use descriptive conversation names
- Review status log for errors
- Save important conversations
- Check Tool Usage tab to understand AI actions

❌ **Don't:**
- Mix unrelated topics in one conversation
- Ignore error messages
- Enable EPICS writes without approval (if using projects)
- Delete conversations without backing up
- Run multiple GUI instances with same config

### 14.2 Safety Guidelines

**For Control System Operations (if using EPICS projects):**

1. **Always verify before writing:**
   - Check PV name carefully
   - Verify value is within safe range
   - Confirm with approval dialog

2. **Use read-only mode by default:**
   ```yaml
   # In project config.yml
   control_system:
     writes_enabled: false
   ```

3. **Enable approval for critical operations:**
   ```yaml
   # In project config.yml
   approval:
     global_mode: selective
     capabilities:
       python_execution:
         enabled: true
         mode: epics_writes
   ```

4. **Test in safe environment first:**
   - Use test PVs
   - Verify behavior
   - Then apply to production

### 14.3 Performance Tips

**For Faster Responses:**

1. **Use smaller models for simple tasks:**
   ```yaml
   # In gui_config.yml
   models:
     classifier:
       model_id: gpt-3.5-turbo  # Faster than GPT-4
   ```

2. **Enable routing cache:**
   ```yaml
   # In gui_config.yml
   routing:
     cache:
       enabled: true
       max_size: 200
   ```

3. **Reduce timeout for quick failures:**
   ```yaml
   # In gui_config.yml
   execution_control:
     limits:
       max_execution_time_seconds: 60
   ```

4. **Monitor memory usage:**
   - Check 💾 Memory tab regularly
   - Watch for increasing trends
   - Investigate if memory exceeds thresholds
   - Restart GUI if memory grows excessively

5. **Close unused tabs:**
   - Reduces memory usage
   - Improves responsiveness

### 14.4 Organization Tips

**Managing Multiple Projects:**

1. **Use clear project names:**
   - `beamline-12-assistant`
   - `magnet-control-assistant`
   - `diagnostics-assistant`

2. **Organize conversations:**
   - Rename with dates: "Beam Tuning - 2024-12-18"
   - Use prefixes: "[URGENT] Magnet Issue"
   - Delete old conversations regularly

3. **Document custom configurations:**
   - Keep notes on facility-specific settings
   - Share configuration templates
   - Version control config files

---

## 15. Glossary

**AI/LLM Terms:**

- **LLM** - Large Language Model; the AI that powers the assistant
- **Token** - Unit of text processed by the AI
- **Prompt** - Instructions sent to the AI
- **Capability** - A specific function the AI can perform
- **Orchestration** - Coordinating multiple capabilities for complex tasks

**EPICS Terms:**

- **PV** - Process Variable; a named data point in EPICS
- **IOC** - Input/Output Controller; EPICS server
- **Gateway** - Network gateway for EPICS access
- **Channel Access** - EPICS network protocol
- **Archiver** - Service that stores historical PV data

**Osprey Terms:**

- **Project** - A collection of capabilities and configuration
- **Registry** - List of available capabilities
- **Thread ID** - Unique identifier for a conversation
- **Routing** - Selecting which project handles a query
- **Checkpointer** - System for saving conversation state
- **gui_config.yml** - GUI framework configuration file
- **config.yml** - Project-specific configuration file

**GUI Terms:**

- **Tab** - Different view in the interface
- **Panel** - Section of the window
- **Status Log** - Real-time activity display
- **Conversation History** - List of saved conversations
- **Memory Monitoring** - Tracking resource usage of framework processes
- **Trend Analysis** - Detection of memory usage patterns over time

---

## Appendix A: Icons and Symbols

**Status Indicators:**

| Symbol | Meaning |
|--------|---------|
| ✅ | Success / Completed |
| ❌ | Error / Failed |
| ⚠️ | Warning |
| ℹ️ | Information |
| 🔄 | Refresh / Reload |
| ⏳ | Processing / Loading |
| 💬 | Conversation |
| 🔧 | Tool / Capability |
| 📊 | Analytics / Statistics |
| 🗑️ | Delete |
| ✏️ | Edit / Rename |
| ▶️ | Active / Selected |

**Color Codes:**

| Color | Meaning |
|-------|---------|
| 🟣 Purple | User messages |
| ⚪ White | AI responses |
| 🟢 Green | Success / Enabled |
| 🔴 Red | Error / Critical |
| 🟡 Gold | Important / Warning |
| 🔵 Blue | Information |
| 🟠 Orange | Monitoring |

---

## Appendix B: Quick Start Checklist

**First-Time Setup:**

- [ ] Install Osprey Framework
- [ ] Set API key environment variable (e.g., `ARGO_API_KEY`)
- [ ] Verify `gui_config.yml` exists and has correct provider
- [ ] Test with `osprey-gui` to verify it starts
- [ ] Create first conversation
- [ ] Test simple query
- [ ] Review all tabs
- [ ] Configure settings as needed

**Daily Startup:**

- [ ] Activate Python environment (if using one)
- [ ] Launch GUI with `osprey-gui`
- [ ] Wait for "Ready" status
- [ ] Check System Information tab
- [ ] Create new conversation or resume existing
- [ ] Begin work

---

## Appendix C: Configuration File Locations

### GUI Configuration

**File:** `gui_config.yml`

**Location:** `<python-env>/lib/python3.x/site-packages/osprey/interfaces/pyqt/gui_config.yml`

**To find:**
```bash
python -c "import osprey; import os; print(os.path.join(os.path.dirname(osprey.__file__), 'interfaces/pyqt/gui_config.yml'))"
```

### Environment Variables

**File:** `.env`

**Locations (checked in order):**
1. Osprey framework directory: `osprey/.env`
2. Project directories: `<project-dir>/.env`
3. System environment variables

### Project Configuration

**File:** `config.yml`

**Location:** `<project-directory>/config.yml` (for each discovered project)

**Example:**
```
/path/to/its-control-assistant/config.yml
```

---

## Appendix D: Recent Improvements

### GUI Architecture Enhancements (Version 0.9.7+)

The Osprey GUI has undergone significant internal refactoring to improve code quality, maintainability, and reliability. While these changes are primarily internal, users benefit from:

**Enhanced Stability:**
- Improved state management eliminates edge cases that could cause unexpected behavior
- Comprehensive error handling provides better recovery from failures
- Validated state transitions prevent invalid operations

**Better Performance:**
- Optimized initialization reduces startup time
- Efficient resource management improves responsiveness
- Streamlined message processing for faster interactions

**Improved Reliability:**
- Centralized configuration management reduces configuration errors
- Standardized error handling provides consistent error messages
- Better separation of concerns makes the system more maintainable

**Technical Improvements:**
- 40-50% reduction in code complexity
- 90%+ elimination of duplicate code patterns
- 100% backward compatibility maintained
- Comprehensive validation of all critical functionality

These improvements ensure a more stable, responsive, and reliable user experience while maintaining full compatibility with existing workflows and configurations.

### Memory Monitoring Features (Version 0.9.7+)

The GUI now includes comprehensive memory monitoring capabilities:

**Real-Time Monitoring:**
- Automatic tracking of GUI process memory usage
- Detection of child processes spawned by framework
- Monitoring of Docker and Podman containers
- Configurable update intervals (default: 5 seconds)

**Intelligent Trend Analysis:**
- 2-minute warmup period for baseline establishment
- Tracks last 20 measurements for trend calculation
- Detects increasing, decreasing, or stable memory patterns
- Calculates rate of change in MB/minute
- Visual indicators with color coding

**Threshold Management:**
- Configurable warning threshold (default: 500 MB)
- Configurable critical threshold (default: 1000 MB)
- Visual and status bar alerts when thresholds exceeded
- Color-coded progress bars for quick assessment

**Process Details:**
- Detailed breakdown by process type
- Memory usage per process in MB
- CPU utilization percentage
- Process status monitoring
- Container tracking (Docker/Podman)

**User Controls:**
- Pause/Resume monitoring
- Manual refresh capability
- Configurable settings in Development/Debug tab
- Persistent settings across sessions

This feature helps users identify memory leaks, monitor resource usage, and ensure optimal performance of the framework.

---

**Document Version:** 2.1
**Last Updated:** December 2024
**Osprey Framework Version:** 0.9.7+

**For the latest documentation, visit:** https://als-apg.github.io/osprey

---

**Copyright Notice**

Osprey Framework Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.