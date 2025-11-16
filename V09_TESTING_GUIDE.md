# V0.9 Testing Guide

**Requirements**: Python 3.11+

## Quick Setup

```bash
# 1. Clone and checkout the branch
git clone https://github.com/als-apg/osprey.git
cd osprey
git checkout feature/v0.9-control-assistant-connectors

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Osprey in development mode with docs dependencies
pip install -e ".[docs]"

# 4. Launch documentation locally
cd docs
python launch_docs.py
# Visit: http://localhost:8082
```

## What to Test

1. **Documentation**: Browse the new control assistant tutorial series
   - Getting Started → Control Assistant Tutorial (4 parts)

2. **Template**: Try creating a control assistant project
   ```bash
   # Option 1: Interactive menu (recommended)
   osprey
   # Select "Initialize new project" → choose "control_assistant" template

   # Option 2: Direct command
   osprey init my-control-assistant --template control_assistant
   cd my-control-assistant
   ```

3. **Features**: Check out the new capabilities
   - Control system connectors (mock and EPICS)
   - Channel Finder service
   - Dual-mode operation (mock/production)
