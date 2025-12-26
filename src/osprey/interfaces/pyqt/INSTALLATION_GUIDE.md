# PyQt GUI Installation and Testing Guide

This guide will help you install and test the Osprey Framework PyQt GUI.

## Quick Installation

### Step 1: Install PyQt5 Dependencies

Make sure you're in the `osprey-091` environment (or your preferred Python 3.11+ environment):

```bash
# Activate your environment
conda activate osprey-091  # or your environment name

# Install PyQt5
pip install PyQt5 python-dotenv
```

Alternatively, use the requirements file:

```bash
pip install -r src/osprey/interfaces/pyqt/requirements-gui.txt
```

### Step 2: Install Osprey Framework (if not already installed)

From the project root directory:

```bash
pip install -e .
```

### Step 3: Verify Installation

```bash
# Test PyQt5 installation
python -c "import PyQt5; print('✅ PyQt5 installed successfully')"

# Test Osprey installation
python -c "import osprey; print('✅ Osprey Framework installed successfully')"
```

## Running the GUI

**IMPORTANT**: The GUI requires a `config.yml` file. You must either:
1. Run the command from a directory containing `config.yml`, OR
2. Specify the path to your config file, OR
3. Set the `CONFIG_FILE` environment variable

### Method 1: Using the Console Script (After pip install)

**From a directory with config.yml:**
```bash
cd /path/to/your/project  # Directory containing config.yml
osprey-gui
```

**With a custom config path:**
```bash
osprey-gui /path/to/your/config.yml
```

**Using environment variable:**
```bash
export CONFIG_FILE=/path/to/your/config.yml
osprey-gui
```

### Method 2: Using Python Module

```bash
# From project root
python -m osprey.interfaces.pyqt.launcher

# With custom config
python -m osprey.interfaces.pyqt.launcher config.yml
```

### Method 3: Direct Script Execution

```bash
# From project root
python src/osprey/interfaces/pyqt/launcher.py

# With custom config
python src/osprey/interfaces/pyqt/launcher.py config.yml
```

## Testing the GUI

### Basic Functionality Test

1. **Launch the GUI**:
   ```bash
   osprey-gui
   ```

2. **Verify Initialization**:
   - The GUI window should open
   - Status log should show "Framework initialized successfully"
   - System Information tab should display session details

3. **Test Conversation**:
   - Type a simple message in the input field (e.g., "Hello")
   - Click "Send" or press Enter
   - Verify the message appears in the conversation display
   - Check that the agent responds

4. **Test Conversation Management**:
   - Click "New Conversation" button
   - Verify a new conversation is created in the history panel
   - Switch between conversations by clicking them in the history
   - Try renaming a conversation (✏ button)

5. **Test Settings**:
   - Go to Settings → Framework Settings
   - Toggle some settings
   - Click Save
   - Verify settings are applied

6. **Test Tabs**:
   - Switch to "LLM Details" tab - should show interaction details
   - Switch to "Tool Usage" tab - should show capability executions
   - Switch to "System Information" tab - should show session info

### Troubleshooting

#### GUI Won't Start

**Error: "No module named 'PyQt5'"**
```bash
pip install PyQt5
```

**Error: "No module named 'osprey'"**
```bash
# From project root
pip install -e .
```

**Error: "cannot connect to X server" or "could not connect to display" (Linux/SSH)**

This error occurs when running the GUI over SSH without a display. You have several options:

**Option 1: Use SSH with X11 Forwarding (Recommended for Remote Access)**
```bash
# Disconnect and reconnect with X11 forwarding
ssh -X user@host
# or for trusted X11 forwarding
ssh -Y user@host

# Then run the GUI
osprey-gui
```

**Option 2: Use VNC or Remote Desktop**
- Set up a VNC server on the remote machine
- Connect with a VNC client from your local machine
- Run the GUI in the VNC session

**Option 3: Run Locally**
- If you have access to the machine physically or via remote desktop
- Run the GUI directly on the machine's display

**Option 4: Use the CLI Instead (Recommended for SSH)**
If you're working remotely and don't need the GUI, use the CLI interface instead:
```bash
# Use the CLI interface (no display required)
osprey chat
# or
python -m osprey.interfaces.cli.direct_conversation
```

**Verify X11 Forwarding is Working:**
```bash
# Check DISPLAY variable is set
echo $DISPLAY
# Should show something like "localhost:10.0"

# Test with a simple X application
xeyes
# If this works, the GUI should work too
```

#### GUI Starts but Framework Fails to Initialize

**Check config.yml exists**:
```bash
ls -la config.yml
```

**Check environment variables**:
```bash
# Make sure API keys are set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
```

**Check logs**:
The GUI logs to the console. Look for error messages in the terminal where you launched the GUI.

#### Conversation Not Working

1. Check the Status Log tab for error messages
2. Verify your config.yml has proper model configuration
3. Check that API keys are set in your environment or .env file

## Environment Setup for Testing

### Using the osprey-091 Environment

If you're using the `osprey-091` conda environment:

```bash
# Activate environment
conda activate osprey-091

# Install GUI dependencies
pip install PyQt5 python-dotenv

# Verify installation
python -c "import PyQt5; print('OK')"

# Run GUI
osprey-gui
```

### Creating a New Test Environment

```bash
# Create new environment
conda create -n osprey-gui-test python=3.11

# Activate it
conda activate osprey-gui-test

# Install osprey framework (from the osprey project root)
cd /path/to/osprey
pip install -e .

# Install GUI dependencies
pip install PyQt5 python-dotenv

# Run GUI
osprey-gui
```

## Configuration

The GUI uses the same configuration system as the CLI. Make sure you have a valid `config.yml` file in your working directory or specify a custom path:

```bash
osprey-gui /path/to/your/config.yml
```

### Minimal config.yml for Testing

If you don't have a config.yml, create one with minimal settings:

```yaml
project_name: test-gui

models:
  default:
    provider: anthropic
    model_id: claude-3-5-sonnet-20241022

api:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
```

## Next Steps

After successful installation and testing:

1. Explore the different tabs and features
2. Try creating multiple conversations
3. Test the settings dialog
4. Review the LLM Details and Tool Usage tabs during agent execution
5. Check conversation persistence by closing and reopening the GUI

## Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Review the Status Log tab in the GUI
3. Verify all dependencies are installed
4. Check that your config.yml is valid
5. Ensure API keys are properly set

For more information, see the main README.md in this directory.