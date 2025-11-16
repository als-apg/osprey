# Osprey Framework - Latest Release (v0.9.1)

🎉 **MCP Integration Release** - Auto-generate Capabilities from Model Context Protocol Servers

## What's New in v0.9.1

### 🚀 Major New Features

**MCP Capability Generator (Prototype):**
- **Auto-generate Osprey capabilities** from Model Context Protocol (MCP) servers
- **`osprey generate capability`** - Create complete capabilities from running MCP servers
- **`osprey generate mcp-server`** - Generate demo MCP servers for testing and development
- **Automatic ReAct agent integration** - Built-in LangGraph ReAct agent pattern
- **LLM-powered guide generation** - Automatically creates classifier and orchestrator examples
- **Interactive registry integration** - Confirms and updates registry and config automatically
- **Complete tutorial** - End-to-end MCP integration workflow in Quick Start Patterns
- **Dependencies**: Requires `langchain-mcp-adapters`, `langgraph`, and provider-specific LangChain packages

**Capability Removal Command:**
- **`osprey remove capability`** - Safe, automated removal of generated capabilities
- **Comprehensive cleanup** - Removes registry entries, config models, and capability files
- **Automatic backups** - Creates `.bak` files before any modifications
- **Interactive confirmation** - Preview changes before applying them
- **Force mode** - Optional `--force` flag to skip confirmation prompts

### 🔧 Improvements

**Core Dependencies:**
- Added `matplotlib>=3.10.3` to core dependencies
- Python capability visualization now works out of the box
- Tutorial examples (plotting, visualization) work immediately after installation

## Installation

```bash
pip install osprey-framework==0.9.1
```

## Quick Start - MCP Integration

```bash
# 1. Generate a demo MCP server
osprey generate mcp-server --name weather_demo

# 2. Run the server (in another terminal)
python weather_demo_server.py

# 3. Generate capability from MCP server
osprey generate capability --from-mcp http://localhost:3001 --name weather_demo

# 4. Test your capability
osprey chat
```

## Documentation

- **Tutorial**: [End-to-End MCP Integration](https://osprey-framework.readthedocs.io/en/latest/developer-guides/02_quick-start-patterns/04_mcp-capability-generation.html)
- **Full Changelog**: See CHANGELOG.md
- **Project Homepage**: https://github.com/als-apg/osprey

## Requirements

- Python >=3.11
- For MCP integration: `pip install langchain-mcp-adapters langgraph langchain-anthropic` (or `langchain-openai`)
- Recommended: Claude Haiku 4.5 for best capability generation results

## Migration Notes

This is a prototype release for MCP integration. The API and generated code structure may evolve in future releases. Generated capabilities use the ReAct agent pattern with LangGraph for autonomous tool selection and execution.

## What's Next

- Enhanced MCP server templates and presets
- Improved context class customization
- Additional LLM provider support for capability generation
- Production-ready MCP integration patterns

---

**Previous Release**: [v0.9.0 Release Notes](https://github.com/als-apg/osprey/releases/tag/v0.9.0)
