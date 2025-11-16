# V0.9.1 Testing Guide

**Requirements**: Python 3.11+

## Quick Setup

```bash
# 1. Clone and checkout the branch
git clone https://github.com/als-apg/osprey.git
cd osprey
git checkout feature/v0.9.1-weather-mcp-onboarding

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

### MCP Capability Generator Tutorial

Follow the complete tutorial in the documentation:

**Developer Guides → Quick Start Patterns → MCP Capability Generation**

Direct link: http://localhost:8082/developer-guides/02_quick-start-patterns/04_mcp-capability-generation.html

The tutorial walks through:
- Quick start with simulated weather tools
- Understanding generated code structure
- Creating and connecting to a live MCP server
- Integrating generated capabilities into your application

## Reporting Issues

Found a bug or have suggestions? https://github.com/als-apg/osprey/issues

