# OSPREY Integration Guide — Quick Reference

## What This Guide Is

A practical, recipe-style guide for porting external tools (e.g., MATLAB applications) into the OSPREY ecosystem. Each recipe covers one integration layer with the pattern, a concrete example from the codebase, and the checklist of files and conventions to follow.

## Recipes

| # | Recipe | When You Need It |
|---|--------|-----------------|
| 1 | [Adding an MCP Server](01-mcp-server.md) | Expose tool capabilities to Claude Code via FastMCP |
| 2 | [Adding a Web Interface](02-web-interface.md) | Build a browser-based UI (FastAPI + vanilla JS) |
| 3 | [Adding a CLI Command](03-cli-command.md) | Register `osprey <your-command>` in the CLI |
| 4 | [Connecting to Config](04-config-system.md) | Read settings from `config.yml`, integrate with templates |
| 5 | [Frontend Conventions](05-frontend-conventions.md) | Vanilla JS modules, CSS design system, component patterns |
| 6 | [Data Storage & Workspace](06-data-storage.md) | File-backed stores, workspace directories, SSE broadcasting |
| 7 | [Adding Tests](07-testing.md) | Unit test patterns, fixtures, mocking, registry reset |
| 8 | [Universal Conventions](08-conventions.md) | Naming, error handling, safety rules, code style |
| 9 | [Custom IOCs for Testing](09-custom-iocs-for-testing.md) | caproto soft IOCs, simulation backends, recorded data replay |

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| MCP servers | [FastMCP](https://github.com/jlowin/fastmcp) `>=2.0.0` | One independent server per domain |
| Web backends | FastAPI `>=0.109.0` + Uvicorn | App factory pattern, lifespan management |
| Web frontends | Vanilla JavaScript (ES6 modules) | No frameworks. No build step. |
| Styling | CSS Custom Properties | "Precision Instrumentation" dark theme |
| Database | PostgreSQL + psycopg3 `>=3.1.0` | AsyncConnectionPool, dict_row factory |
| Schemas | Pydantic v2 | Request/response models for API routes |
| CLI | Click `>=8.1.0` | LazyGroup for fast startup |
| Templates | Jinja2 `>=3.1.6` | Project scaffolding and conditional generation |
| Config | YAML (`PyYAML` + `ruamel.yaml`) | Hierarchical `config.yml` with env var substitution |
| Testing | pytest + pytest-asyncio | Marker-based: `unit`, `integration`, `e2e` |
| Python | 3.11+ | Type hints, `match` statements, `TypedDict` |

## Directory Layout

```
src/osprey/
├── mcp_server/                    # MCP servers (workspace, control-system, python-executor)
│   ├── common.py                  # Shared: load_osprey_config(), make_error(), redirect_logging
│   ├── workspace/                 # Memory, artifacts, data context, screenshots
│   ├── control_system/            # Channel read/write, archiver, limits
│   └── python_executor/           # Code execution with save_artifact injection
│
├── interfaces/                    # Web interfaces (each is a self-contained FastAPI app)
│   ├── ariel/                     # ARIEL logbook search ← REFERENCE IMPLEMENTATION
│   │   ├── app.py                 # FastAPI app factory
│   │   ├── api/                   # Routes, schemas, drafts
│   │   ├── mcp/                   # Co-located MCP server (independent from web app)
│   │   └── static/               # HTML, JS, CSS
│   ├── artifacts/                 # Artifact gallery (plots, tables, notebooks)
│   └── web_terminal/             # PTY/operator terminal + file browser
│
├── services/                      # Domain services (business logic, no HTTP/MCP coupling)
│   ├── ariel_search/              # Search service, database, models, migrations
│   └── channel_finder/            # Channel resolution pipelines + MCP
│
├── cli/                           # Click commands
│   ├── main.py                    # LazyGroup entry point
│   ├── init_cmd.py                # `osprey init`
│   ├── artifacts_cmd.py           # `osprey artifacts`
│   └── web_cmd.py                 # `osprey web`
│
└── templates/                     # Jinja2 project templates
    ├── project/                   # Default config, env, pyproject
    ├── apps/                      # App-specific templates (minimal, control_assistant)
    └── claude_code/               # .mcp.json, CLAUDE.md, settings.json, agents
```

## Existing Components

### MCP Servers

| Server | Package | Tools | Purpose |
|--------|---------|-------|---------|
| `osprey-workspace` | `mcp_server.workspace` | memory_save, memory_recall, artifact_save, artifact_focus, data_context_list, context_focus, screen_capture | Workspace data management |
| `osprey-control-system` | `mcp_server.control_system` | channel_read, channel_write, channel_find, archiver_read | Hardware I/O (EPICS, etc.) |
| `osprey-python-executor` | `mcp_server.python_executor` | python_execute | Sandboxed code execution |
| `ariel` | `interfaces.ariel.mcp` | ariel_search, ariel_browse, ariel_filter_options, ariel_entry_get, ariel_entry_create, ariel_capabilities, ariel_status | Logbook search |
| `channel-finder` | `services.channel_finder.mcp` | cf_* (varies by pipeline) | Channel name resolution |

### Web Interfaces

| Interface | Package | Port | Purpose |
|-----------|---------|------|---------|
| ARIEL Web | `interfaces.ariel` | 8085 | Logbook search, entry creation, config management |
| Artifact Gallery | `interfaces.artifacts` | 8086 | Plot/table/notebook viewer with SSE updates |
| Web Terminal | `interfaces.web_terminal` | 8087 | PTY terminal + file browser + panel tabs |

### The Golden Rule

**ARIEL is the reference implementation.** It demonstrates the complete "MCP server + web interface" pattern. When in doubt about how to structure something, look at how ARIEL does it:

- MCP server: `src/osprey/interfaces/ariel/mcp/`
- Web interface: `src/osprey/interfaces/ariel/app.py` + `static/`
- Service layer: `src/osprey/services/ariel_search/`
- Tests: `tests/interfaces/ariel/mcp/`
