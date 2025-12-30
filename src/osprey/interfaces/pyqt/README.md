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

### Install with GUI Support

The recommended way to install the GUI is using the optional `[gui]` extra when installing the Osprey Framework:

```bash
# Install Osprey Framework with GUI support
pip install osprey-framework[gui]
```

This will automatically install all required GUI dependencies including PyQt5 and psutil.

### Alternative: Install GUI Dependencies Separately

If you already have the Osprey Framework installed and want to add GUI support:

```bash
# From the project root directory
pip install -r src/osprey/interfaces/pyqt/requirements-gui.txt
```

Or install the dependencies directly:

```bash
pip install PyQt5>=5.15.0 psutil>=5.9.0
```

### Verify Installation

```bash
python -c "import PyQt5; print('PyQt5 installed successfully')"
```

## Usage

### Method 1: Using the Command Line Tool (Recommended)

After installing with `pip install osprey-framework[gui]`, you can launch the GUI using the `osprey-gui` command:

```bash
# Using default config discovery (searches for config.yml in current directory)
osprey-gui

# Using a specific Osprey project config file
osprey-gui path/to/your/project/config.yml
```

**Note**: The config file should be your Osprey Framework project's main configuration file (typically `config.yml`), which contains your project settings, model configurations, and capabilities. The GUI will also check for an optional `gui_config.yml` file in the PyQt directory for GUI-specific settings. See the [Configuration](#configuration) section below for important details about GUI-specific requirements.

### Method 2: Using the Launcher Module

From the project root directory:

```bash
# Using default config.yml
python -m osprey.interfaces.pyqt.launcher

# Using a custom config file
python -m osprey.interfaces.pyqt.launcher path/to/your/config.yml
```

### Method 3: Direct Python Import

```python
from osprey.interfaces.pyqt.gui import main

# Launch with default config
main()

# Launch with custom config
main(config_path="path/to/your/config.yml")
```

### Method 4: Using the GUI Module Directly

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

The GUI uses the same configuration system as the rest of the Osprey Framework, but requires additional GUI-specific settings.

### Configuration File Priority

1. **Explicit config path**: If you provide a path via command line argument
2. **GUI-specific config**: `src/osprey/interfaces/pyqt/gui_config.yml` (if it exists)
3. **Default discovery**: Searches for `config.yml` in the current working directory

### Important: GUI Configuration Requirements

⚠️ **The GUI requires additional configuration sections** beyond a standard Osprey project config:

**Required GUI-specific sections:**
- `routing` - Multi-project routing, caching, and semantic analysis settings
- `gui` - GUI behavior settings (conversation persistence, output redirection)
- `memory_monitoring` - Memory threshold and monitoring settings

**Recommended approach:**
1. **Use the provided `gui_config.yml`** in the PyQt directory (already configured with all required sections)
2. **Or copy `gui_config.yml.example`** to create your own GUI config with all necessary sections
3. **Or add the GUI sections** to your existing project config (see `gui_config.yml.example` for reference)

If you use a standard project config file that's missing these sections, the GUI will use default values, but some features (like routing analytics, conversation persistence settings, and memory monitoring) may not work as expected.

**For complete configuration documentation**, see:
- [Configuration System API Reference](https://als-apg.github.io/osprey/api_reference/01_core_framework/04_configuration_system.html) - Complete reference for all configuration sections
- [Configuration Architecture Guide](https://als-apg.github.io/osprey/developer-guides/03_core-framework-systems/06_configuration-architecture.html) - Understanding the configuration system design

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