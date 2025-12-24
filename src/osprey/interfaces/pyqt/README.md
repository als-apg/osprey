# Osprey Framework PyQt GUI

A graphical user interface for the Osprey Framework built with PyQt5, providing an integrated and user-friendly way to interact with the framework.

## Features

- **Framework Integration**: Fully integrated with Osprey's Gateway, graph architecture, and configuration system
- **Conversation Management**: Create, switch, rename, and delete conversation threads
- **Real-time Status Updates**: Monitor agent processing with live status logs
- **LLM Interaction Details**: View detailed LLM conversation flow and tool usage
- **System Information**: Display framework configuration and session details
- **Settings Management**: Configure framework settings including planning mode, EPICS writes, and approval modes
- **Persistent Conversations**: Automatic conversation persistence using LangGraph's checkpointing system
- **Multi-Instance Safe**: File locking prevents conflicts when running multiple GUI instances

## Installation

### Prerequisites

- Python 3.11 or higher
- Osprey Framework installed
- PyQt5

### Install GUI Dependencies

From the project root directory:

```bash
# Install PyQt5 and other GUI dependencies
pip install -r src/osprey/interfaces/pyqt/requirements-gui.txt
```

Or install PyQt5 directly:

```bash
pip install PyQt5 python-dotenv
```

### Verify Installation

```bash
python -c "import PyQt5; print('PyQt5 installed successfully')"
```

## Usage

### Method 1: Using the Launcher Script (Recommended)

From the project root directory:

```bash
# Using default config.yml
python -m osprey.interfaces.pyqt.launcher

# Using a custom config file
python -m osprey.interfaces.pyqt.launcher path/to/your/config.yml
```

### Method 2: Direct Python Import

```python
from osprey.interfaces.pyqt.gui import main

# Launch with default config
main()

# Launch with custom config
main(config_path="path/to/your/config.yml")
```

### Method 3: Using the GUI Module Directly

```bash
# From project root
python src/osprey/interfaces/pyqt/launcher.py

# With custom config
python src/osprey/interfaces/pyqt/launcher.py path/to/config.yml
```

## GUI Components

### Main Window Tabs

1. **Conversation Tab**
   - Left panel: Conversation history with management buttons
   - Center panel: Active conversation display
   - Right panel: Real-time status log
   - Bottom: Input field and action buttons

2. **LLM Details Tab**
   - Detailed view of LLM interactions
   - Event-based logging with timestamps
   - Color-coded event types

3. **Tool Usage Tab**
   - Capability execution tracking
   - Execution time monitoring
   - Success/failure indicators

4. **System Information Tab**
   - Session details
   - Thread ID and configuration path
   - Registered capabilities count

### Menu Bar

- **File Menu**
  - New Conversation
  - Clear Conversation
  - Exit

- **Settings Menu**
  - Framework Settings (planning mode, EPICS writes, approval mode, execution time)

- **Help Menu**
  - About dialog with version and system information

## Configuration

The GUI uses the same configuration system as the rest of the Osprey Framework. By default, it looks for `config.yml` in the current directory.

### Framework Settings

Access via **Settings → Framework Settings**:

- **Planning Mode**: Enable/disable planning mode for complex tasks
- **EPICS Writes**: Enable/disable EPICS control system writes
- **Approval Mode**:
  - `disabled`: No approval required
  - `selective`: Approval for specific operations
  - `all_capabilities`: Approval for all capability executions
- **Max Execution Time**: Maximum time (in seconds) for capability execution
- **Save Conversation History**: Enable/disable persistent conversation storage (requires restart)

## Conversation Management

### Creating Conversations

- Click the **New Conversation** button or use **File → New Conversation**
- Each conversation has a unique thread ID for session continuity

### Switching Conversations

- Click on any conversation in the history panel
- The conversation display will update with the selected conversation's messages

### Managing Conversations

- **Rename**: Click the ✏ button or right-click a conversation
- **Delete**: Click the 🗑 button (cannot delete the only conversation)
- **New**: Click the + button to create a new conversation

### Conversation Persistence

**Storage Location**: `_agent_data/checkpoints/gui_conversations.db`

Conversations are automatically persisted using LangGraph's checkpointing system with a SQLite backend. This provides:

- **Automatic Saving**: All messages are saved automatically as you chat
- **Shared Access**: Conversations are shared across all users (stored in project directory, not user home)
- **Full Context**: Complete conversation history including LLM context
- **Framework Integration**: Uses Osprey's native checkpointing infrastructure

**Enabling/Disabling Persistence**:

1. Go to **Settings → Framework Settings**
2. Toggle **Save Conversation History** checkbox
3. Restart the GUI for changes to take effect

**When Enabled** (default):
- Conversations persist across GUI restarts
- Stored in `_agent_data/checkpoints/gui_conversations.db`
- Shared across all users accessing the same project

**When Disabled**:
- Conversations stored in memory only
- Lost when GUI closes
- Useful for temporary/private sessions

**Multi-Instance Safety**:

The GUI uses file locking to prevent conflicts when multiple instances run simultaneously:
- Lock file: `_agent_data/checkpoints/.gui_conversations.db.lock`
- First instance acquires exclusive lock
- Additional instances can still run (PostgreSQL handles concurrent access)
- Lock automatically released when GUI closes

**Note**: On Windows, file locking is not available, but PostgreSQL's built-in concurrency handling ensures data integrity.

## Keyboard Shortcuts

- **Enter**: Send message (in input field)
- **Shift+Enter**: New line (in input field)

## Troubleshooting

### GUI Won't Start

1. **Check PyQt5 Installation**:
   ```bash
   python -c "import PyQt5; print('OK')"
   ```

2. **Check DISPLAY Variable** (Linux/SSH):
   ```bash
   echo $DISPLAY
   # If empty, set it:
   export DISPLAY=:0
   # Or use SSH with X forwarding:
   ssh -X user@host
   ```

3. **Check Framework Installation**:
   ```bash
   python -c "import osprey; print('OK')"
   ```

### Missing Dependencies

If you see import errors, install the required packages:

```bash
pip install PyQt5 python-dotenv
pip install -e .  # Install osprey-framework in development mode
```

### Configuration Errors

Ensure your `config.yml` file is properly formatted and contains all required sections. See the main Osprey Framework documentation for configuration details.

## Environment Variables

The GUI respects the same environment variables as the CLI:

- `OSPREY_CONFIG_PATH`: Override default config file location
- Model API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- EPICS-related variables (if using EPICS connectors)

## Data Storage

### Conversation Database

**Location**: `<project_root>/_agent_data/checkpoints/gui_conversations.db`

The GUI uses a SQLite database (via LangGraph's PostgreSQL checkpointer) to store conversation history. This database contains:

- All conversation messages (user and agent)
- Conversation metadata (names, timestamps)
- Complete LangGraph state for each conversation
- Full LLM context for seamless conversation resumption

**Benefits**:
- Automatic persistence (no manual save needed)
- Shared across all users
- Production-ready (SQLite is reliable and fast)
- Framework-aligned (uses Osprey's checkpointing system)

**Backup**: Simply copy the `_agent_data/checkpoints/` directory to backup all conversations.

**Reset**: Delete `gui_conversations.db` to start fresh (conversations will be lost).

## Differences from osprey-aps GUI

This GUI is more integrated into the framework compared to the osprey-aps version:

1. **Framework Integration**: Uses Osprey's Gateway and graph architecture directly
2. **Simplified Architecture**: No multi-agent discovery needed (single framework instance)
3. **Configuration System**: Uses the framework's native configuration system
4. **Session Management**: Integrated with LangGraph's checkpointing system for automatic persistence
5. **Consistent Interface**: Follows the same patterns as the CLI interface
6. **Persistent Storage**: Uses framework's `_agent_data/checkpoints/` directory for conversation history

## Development

### Running in Development Mode

```bash
# From project root
python -m osprey.interfaces.pyqt.launcher
```

### Adding New Features

The GUI is structured with clear separation of concerns:

- `gui.py`: Main GUI application and window management
- `launcher.py`: Entry point with dependency checking
- `__init__.py`: Package exports

## Support

For issues, questions, or contributions:

- GitHub Issues: https://github.com/als-apg/osprey/issues
- Documentation: https://als-apg.github.io/osprey
- Paper: https://arxiv.org/abs/2508.15066

## License

BSD-3-Clause (same as Osprey Framework)