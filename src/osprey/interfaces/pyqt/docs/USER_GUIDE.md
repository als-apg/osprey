# Osprey Framework PyQt GUI - Comprehensive User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [User Interface Overview](#user-interface-overview)
6. [Features and Functionality](#features-and-functionality)
7. [Configuration and Settings](#configuration-and-settings)
8. [Conversation Management](#conversation-management)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [FAQ](#faq)

---

## Introduction

The Osprey Framework PyQt GUI provides a graphical interface for interacting with the Osprey Agent Framework. It offers an intuitive, user-friendly way to:

- Have conversations with AI agents
- Manage multiple conversation threads
- Monitor agent processing in real-time
- Configure framework settings
- View detailed LLM interactions and tool usage

This guide covers everything you need to know to effectively use the GUI application.

---

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.11 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 500MB for application and dependencies
- **Display**: 1280x800 minimum resolution

### Required Software

- Python 3.11+
- PyQt5
- Osprey Framework
- SQLite (included with Python)

### Optional Requirements

- **For EPICS Integration**: EPICS base installation
- **For Multi-Project Support**: Multiple Osprey project directories
- **For X11 Forwarding** (Linux/SSH): X11 server and SSH with `-X` flag

---

## Installation

### Step 1: Install Osprey Framework

If you haven't already installed the Osprey Framework:

```bash
# Clone the repository
git clone https://github.com/als-apg/osprey.git
cd osprey

# Install in development mode
pip install -e .
```

### Step 2: Install GUI Dependencies

```bash
# Install PyQt5 and GUI-specific dependencies
pip install -r src/osprey/interfaces/pyqt/requirements-gui.txt
```

Or install manually:

```bash
pip install PyQt5>=5.15.0 python-dotenv
```

### Step 3: Verify Installation

```bash
# Test PyQt5 installation
python -c "import PyQt5; print('PyQt5 OK')"

# Test Osprey installation
python -c "import osprey; print('Osprey OK')"

# Test GUI can be imported
python -c "from osprey.interfaces.pyqt.gui import main; print('GUI OK')"
```

### Step 4: Set Up Environment (Optional)

Create a `.env` file in your project directory with API keys:

```bash
# Example .env file
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
CBORG_API_KEY=your_key_here
```

---

## Getting Started

### Launching the GUI

There are three ways to launch the GUI:

#### Method 1: Using the Launcher (Recommended)

```bash
# From your project directory
python -m osprey.interfaces.pyqt.launcher

# With a custom config file
python -m osprey.interfaces.pyqt.launcher path/to/config.yml
```

#### Method 2: Direct Python Import

```python
from osprey.interfaces.pyqt.gui import main

# Launch with default config
main()

# Launch with custom config
main(config_path="path/to/config.yml")
```

#### Method 3: Direct Script Execution

```bash
# From the osprey root directory
python src/osprey/interfaces/pyqt/launcher.py

# With custom config
python src/osprey/interfaces/pyqt/launcher.py path/to/config.yml
```

### First Launch

When you first launch the GUI:

1. **Initialization**: The framework initializes (may take 5-10 seconds)
2. **Default Conversation**: A new conversation is automatically created
3. **Ready State**: The input field becomes active when ready
4. **Status Bar**: Shows "Osprey Framework ready"

### Your First Conversation

1. Type a message in the input field at the bottom
2. Click **Send** or press **Enter**
3. Watch the status log (right panel) for real-time updates
4. The agent's response appears in the conversation display

---

## User Interface Overview

### Main Window Layout

The GUI is organized into several key areas:

```
┌─────────────────────────────────────────────────────────────┐
│ Menu Bar: File | Settings | Multi-Project | Help            │
├──────────┬────────────────────────────┬─────────────────────┤
│          │                            │                     │
│ Conver-  │   Active Conversation      │   Status Log        │
│ sation   │   Display                  │   (Real-time        │
│ History  │                            │    updates)         │
│          │                            │                     │
│ [+][🗑][✏]│                            │                     │
├──────────┴────────────────────────────┴─────────────────────┤
│ Input Field                    [Send] [New Conversation]    │
├─────────────────────────────────────────────────────────────┤
│ Status Bar: Framework status and messages                   │
└─────────────────────────────────────────────────────────────┘
```

### Tab Structure

The GUI has five main tabs:

1. **Conversation**: Main chat interface
2. **LLM Details**: Detailed LLM interaction logs
3. **Tool Usage**: Capability execution tracking
4. **Discovered Projects**: Multi-project management
5. **System Information**: Framework configuration details

---

## Features and Functionality

### 1. Conversation Tab

**Left Panel: Conversation History**
- Lists all your conversations
- Shows conversation name, timestamp, and message count
- Current conversation highlighted in green
- Inactive conversations in gold

**Center Panel: Active Conversation**
- Displays messages from current conversation
- User messages in purple
- Agent responses in white
- Success messages in green

**Right Panel: Status Log**
- Real-time processing updates
- Color-coded by component:
  - White: Base operations
  - Cyan: Context/Orchestrator
  - Orange: Monitor/Approval
  - Red: Errors
  - Purple: Response generation

**Bottom: Input Area**
- Text input field (supports multi-line with Shift+Enter)
- **Send** button: Submit message
- **New Conversation** button: Start fresh conversation

### 2. LLM Details Tab

Provides detailed view of LLM interactions:

- **Timestamps**: Precise timing of each event
- **Event Types**: 
  - `LLM_START`: LLM call initiated
  - `LLM_END`: LLM call completed
  - `STATUS`: Processing status updates
- **Color Coding**:
  - Cyan: LLM start events
  - Green: LLM completion
  - Yellow: Streaming events
  - Gold: Classification events

**Use Cases**:
- Debugging LLM behavior
- Understanding agent decision-making
- Performance monitoring
- Troubleshooting issues

### 3. Tool Usage Tab

Tracks capability execution:

- **Capability Name**: Which tool was used
- **Task Objective**: What the tool was trying to accomplish
- **Execution Time**: How long it took
- **Status Indicators**:
  - ✅ Success
  - ❌ Failure
  - ⏱️ Execution time

**Use Cases**:
- Monitor what the agent is doing
- Identify slow operations
- Debug capability failures
- Understand agent workflow

### 4. Discovered Projects Tab

Multi-project management interface:

- **Project List**: Shows all discovered Osprey projects
- **Project Details**: Name, path, config file, registry file
- **Actions**:
  - **Refresh**: Scan for projects
  - **Generate Unified Config**: Combine multiple projects
  - **Load Unified Config**: Use combined configuration

**Use Cases**:
- Work with multiple Osprey applications
- Combine capabilities from different projects
- Switch between project configurations

### 5. System Information Tab

Displays framework configuration:

- **Thread ID**: Current conversation identifier
- **Config Path**: Active configuration file
- **Capabilities**: Number of registered capabilities
- **Session Details**: Runtime information

**Use Cases**:
- Verify correct configuration loaded
- Check which capabilities are available
- Debug configuration issues
- Share session information for support

---

## Configuration and Settings

### Accessing Settings

**Menu Path**: Settings → Framework Settings

### Available Settings

#### 1. Planning Mode
- **Purpose**: Enable multi-step planning for complex tasks
- **Default**: Disabled
- **When to Enable**: For tasks requiring multiple steps or coordination
- **Impact**: Agent creates execution plans before acting

#### 2. EPICS Writes
- **Purpose**: Allow agent to write to EPICS control system
- **Default**: Disabled
- **When to Enable**: Only in controlled environments with proper safety measures
- **Impact**: Agent can modify hardware setpoints
- **⚠️ Warning**: Only enable if you understand the implications

#### 3. Approval Mode
- **Options**:
  - `disabled`: No approval required (fastest)
  - `selective`: Approval for specific operations
  - `all_capabilities`: Approval for every capability execution
- **Default**: Disabled
- **When to Use**:
  - `selective`: Production environments
  - `all_capabilities`: High-stakes or learning scenarios

#### 4. Max Execution Time
- **Purpose**: Timeout for capability execution
- **Range**: 10-3600 seconds
- **Default**: 300 seconds (5 minutes)
- **Recommendation**: 
  - Short tasks: 60-120 seconds
  - Data analysis: 300-600 seconds
  - Long computations: 600+ seconds

#### 5. Save Conversation History
- **Purpose**: Persist conversations across sessions
- **Default**: Enabled
- **Storage**: `_agent_data/checkpoints/gui_conversations.db`
- **Impact**: 
  - Enabled: Conversations saved automatically
  - Disabled: Conversations lost on exit
- **⚠️ Note**: Requires GUI restart to take effect

### Applying Settings

1. Open Settings dialog
2. Modify desired settings
3. Click **Save**
4. Settings apply immediately (except conversation persistence)
5. For persistence changes, restart GUI

---

## Conversation Management

### Creating Conversations

**Three Ways to Create**:

1. **New Conversation Button**: Bottom of main window
2. **Menu**: File → New Conversation
3. **History Panel**: Click the **+** button

**What Happens**:
- New conversation created with unique thread ID
- Previous conversation remains accessible
- Conversation list updates
- Display clears for new conversation

### Switching Conversations

**To Switch**:
1. Click any conversation in the history panel
2. Conversation loads automatically
3. Full message history displayed
4. LLM context restored

**Visual Indicators**:
- Current conversation: Green text with ▶ prefix
- Other conversations: Gold text
- Timestamp and message count shown

### Renaming Conversations

**Steps**:
1. Select conversation in history panel
2. Click **✏** (edit) button
3. Enter new name in dialog
4. Click OK

**Tips**:
- Use descriptive names (e.g., "Beam Analysis 2024-01-27")
- Names help organize related conversations
- Renaming doesn't affect conversation content

### Deleting Conversations

**Steps**:
1. Select conversation in history panel
2. Click **🗑** (delete) button
3. Confirm deletion

**Restrictions**:
- Cannot delete the only conversation
- Deletion is permanent (if persistence enabled)
- Active conversation switches to another if deleted

### Conversation Persistence

#### How It Works

**When Enabled** (default):
- All messages saved automatically to SQLite database
- Location: `_agent_data/checkpoints/gui_conversations.db`
- Conversations persist across GUI restarts
- Shared across all users accessing the project

**When Disabled**:
- Messages stored in memory only
- Lost when GUI closes
- Useful for temporary/private sessions

#### Storage Details

**Database Location**:
```
<project_root>/
└── _agent_data/
    └── checkpoints/
        ├── gui_conversations.db          # Conversation database
        └── .gui_conversations.db.lock    # Lock file (multi-instance)
```

**What's Stored**:
- All conversation messages (user and agent)
- Conversation metadata (names, timestamps)
- Complete LangGraph state
- Full LLM context

**Benefits**:
- Automatic saving (no manual action needed)
- Full context preservation
- Shared access (not user-specific)
- Production-ready (SQLite is reliable)

#### Multi-Instance Safety

**File Locking**:
- First GUI instance acquires exclusive lock
- Lock file: `.gui_conversations.db.lock`
- Additional instances can still run
- PostgreSQL handles concurrent database access
- Lock released automatically on GUI close

**Platform Notes**:
- **Linux/Mac**: Full file locking support
- **Windows**: Locking not available, but database handles concurrency

#### Backup and Recovery

**Backup Conversations**:
```bash
# Copy the entire checkpoints directory
cp -r _agent_data/checkpoints _agent_data/checkpoints.backup
```

**Restore Conversations**:
```bash
# Replace with backup
cp -r _agent_data/checkpoints.backup/* _agent_data/checkpoints/
```

**Reset (Delete All)**:
```bash
# Delete database to start fresh
rm _agent_data/checkpoints/gui_conversations.db
```

---

## Advanced Features

### Multi-Project Support

#### Discovering Projects

1. Navigate to **Discovered Projects** tab
2. Click **🔄 Refresh** button
3. GUI scans current directory for Osprey projects
4. Projects displayed in table

#### Project Requirements

For a directory to be recognized as a project:
- Must contain `config.yml` file
- Optionally contains `registry.py` file
- Located in subdirectories of current location

#### Generating Unified Configuration

**Purpose**: Combine capabilities from multiple projects

**Steps**:
1. Discover projects (Refresh button)
2. Click **Generate Unified Config**
3. Confirm project list
4. Unified files created:
   - `unified_config.yml`
   - `unified_registry.py`

**Use Cases**:
- Access capabilities from multiple applications
- Create super-agent with combined features
- Test cross-project interactions

#### Loading Unified Configuration

**Steps**:
1. Generate unified config first
2. Click **Load Unified Config**
3. Confirm reload
4. GUI reinitializes with combined capabilities

**Impact**:
- All capabilities from all projects available
- Current session resets
- New thread ID created

### Real-Time Monitoring

#### Status Log

**Purpose**: Monitor agent processing in real-time

**Information Shown**:
- Current operation
- Component being executed
- Progress indicators
- Error messages
- Completion status

**Color Coding**:
- **White**: General operations
- **Cyan**: Orchestrator/Context
- **Magenta**: Router decisions
- **Orange**: Monitor/Approval
- **Purple**: Response generation
- **Red**: Errors

#### LLM Details

**Purpose**: Deep dive into LLM behavior

**Use For**:
- Understanding agent reasoning
- Debugging unexpected behavior
- Performance analysis
- Learning how the framework works

**Clear Button**: Remove old logs for clarity

#### Tool Usage

**Purpose**: Track capability execution

**Information**:
- Which capabilities were used
- What they were trying to accomplish
- How long they took
- Whether they succeeded

**Clear Button**: Remove old entries

---

## Troubleshooting

### Common Issues and Solutions

#### GUI Won't Start

**Symptom**: Error when launching GUI

**Solutions**:

1. **Check PyQt5 Installation**:
   ```bash
   python -c "import PyQt5; print('OK')"
   ```
   If error: `pip install PyQt5`

2. **Check DISPLAY Variable** (Linux/SSH):
   ```bash
   echo $DISPLAY
   # If empty:
   export DISPLAY=:0
   # Or use SSH with X forwarding:
   ssh -X user@host
   ```

3. **Check Framework Installation**:
   ```bash
   python -c "import osprey; print('OK')"
   ```
   If error: `pip install -e .` from osprey root

#### Conversations Not Persisting

**Symptom**: Conversations lost after restart

**Solutions**:

1. **Check Setting**:
   - Settings → Framework Settings
   - Verify "Save Conversation History" is checked
   - Restart GUI if you just enabled it

2. **Check Database**:
   ```bash
   ls -la _agent_data/checkpoints/gui_conversations.db
   ```
   If missing, check write permissions

3. **Check Logs**:
   - Look for errors in status log
   - Check terminal output for database errors

#### Multiple GUI Instances Issues

**Symptom**: Warning about lock file

**Solutions**:

1. **Normal Behavior**: Warning is informational
   - Both instances can run
   - Database handles concurrent access
   - No action needed

2. **Stale Lock File**:
   ```bash
   # If GUI crashed, remove stale lock
   rm _agent_data/checkpoints/.gui_conversations.db.lock
   ```

#### Slow Performance

**Symptom**: GUI feels sluggish

**Solutions**:

1. **Check Database Size**:
   ```bash
   ls -lh _agent_data/checkpoints/gui_conversations.db
   ```
   If very large (>100MB), consider archiving old conversations

2. **Reduce Max Execution Time**:
   - Settings → Max Execution Time
   - Lower value for faster timeouts

3. **Disable Persistence** (temporary):
   - Settings → Uncheck "Save Conversation History"
   - Restart GUI
   - Use for testing/debugging only

#### Configuration Errors

**Symptom**: Framework fails to initialize

**Solutions**:

1. **Verify config.yml**:
   ```bash
   # Check syntax
   python -c "import yaml; yaml.safe_load(open('config.yml'))"
   ```

2. **Check Config Path**:
   - Verify config file exists
   - Check file permissions
   - Try absolute path

3. **Use Default Config**:
   - Launch without config parameter
   - Framework uses built-in defaults

---

## Best Practices

### Conversation Organization

1. **Use Descriptive Names**:
   - ✅ "Beam Analysis 2024-01-27"
   - ❌ "Conversation 1"

2. **Delete Old Conversations**:
   - Keep database size manageable
   - Archive important conversations first

3. **Create New Conversations**:
   - For different topics
   - When context becomes too long
   - For different projects/tasks

### Settings Configuration

1. **Start Conservative**:
   - Keep approval mode enabled initially
   - Disable EPICS writes until needed
   - Use default execution time

2. **Adjust Based on Use**:
   - Enable planning mode for complex tasks
   - Increase timeout for long operations
   - Disable approvals when confident

3. **Document Changes**:
   - Note why you changed settings
   - Keep track of what works
   - Share configurations with team

### Performance Optimization

1. **Regular Maintenance**:
   - Clear old conversations periodically
   - Archive important conversations
   - Monitor database size

2. **Efficient Usage**:
   - Close unused GUI instances
   - Use appropriate execution timeouts
   - Clear LLM Details/Tool Usage logs

3. **Resource Management**:
   - Don't run too many instances
   - Monitor system resources
   - Close when not in use

### Data Management

1. **Backup Important Conversations**:
   ```bash
   # Regular backups
   cp -r _agent_data/checkpoints backups/checkpoints-$(date +%Y%m%d)
   ```

2. **Version Control**:
   - Don't commit `_agent_data/` to git
   - Keep `.gitignore` updated
   - Backup separately

3. **Privacy Considerations**:
   - Conversations are shared (not user-specific)
   - Sensitive data persists in database
   - Use disabled persistence for private sessions

---

## FAQ

### General Questions

**Q: Where are conversations stored?**
A: In `_agent_data/checkpoints/gui_conversations.db` (SQLite database)

**Q: Are conversations private?**
A: No, they're shared across all users accessing the same project directory.

**Q: Can I run multiple GUI instances?**
A: Yes, file locking prevents conflicts. Both instances can access conversations.

**Q: How do I backup conversations?**
A: Copy the `_agent_data/checkpoints/` directory.

**Q: Can I export conversations?**
A: Currently no built-in export. The database is SQLite format (can be queried directly).

### Technical Questions

**Q: What checkpointing system is used?**
A: LangGraph's PostgreSQL checkpointer with SQLite backend.

**Q: Why SQLite instead of PostgreSQL?**
A: Simpler setup, no server required, perfect for single-machine use.

**Q: Does the LLM see conversation history?**
A: Yes, full context is preserved via the checkpointer.

**Q: What happens if the database is deleted?**
A: All conversations are lost. A new database is created on next launch.

**Q: Can I use a different database location?**
A: Not currently configurable. It's hardcoded to `_agent_data/checkpoints/`.

### Troubleshooting Questions

**Q: GUI won't start on SSH?**
A: Use `ssh -X` for X11 forwarding, or set `DISPLAY` variable.

**Q: Conversations not saving?**
A: Check "Save Conversation History" setting and restart GUI.

**Q: Lock file error?**
A: Another instance is running, or stale lock file. Safe to ignore or delete lock file.

**Q: Database corruption?**
A: Backup and delete `gui_conversations.db`. New database will be created.

**Q: Slow performance?**
A: Check database size, reduce execution timeout, or disable persistence temporarily.

---

## Support and Resources

### Getting Help

- **GitHub Issues**: https://github.com/als-apg/osprey/issues
- **Documentation**: https://als-apg.github.io/osprey
- **Paper**: https://arxiv.org/abs/2508.15066

### Additional Documentation

- **README.md**: Quick start and basic usage
- **INSTALLATION_GUIDE.md**: Detailed installation instructions
- **MULTI_PROJECT_GUIDE.md**: Multi-project setup and usage

### Contributing

Found a bug or have a feature request? Please open an issue on GitHub!

---

## Appendix

### Keyboard Shortcuts

- **Enter**: Send message
- **Shift+Enter**: New line in input field
- **Ctrl+L**: Clear screen (in some terminals)

### File Locations

```
project_root/
├── config.yml                          # Main configuration
├── _agent_data/                        # Runtime data
│   └── checkpoints/                    # Conversation storage
│       ├── gui_conversations.db        # SQLite database
│       └── .gui_conversations.db.lock  # Lock file
└── src/osprey/interfaces/pyqt/         # GUI source code
    ├── gui.py                          # Main application
    ├── launcher.py                     # Entry point
    └── README.md                       # Quick reference
```

### Version Information

- **GUI Version**: 0.9.2
- **Framework Version**: 0.9.2
- **Last Updated**: 2024-01-27

---

*This guide covers the Osprey Framework PyQt GUI as of version 0.9.2. For the latest updates, please refer to the official documentation.*