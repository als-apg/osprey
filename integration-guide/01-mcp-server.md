# Recipe 1: Adding an MCP Server

## When You Need This

You want Claude Code to be able to call your tool's functions — acquire data, run analysis, return results. Every tool that Claude can invoke is exposed through an MCP (Model Context Protocol) server.

## The Pattern

Every MCP server in OSPREY follows a 4-file structure:

```
src/osprey/interfaces/{name}/mcp/       # co-located with web interface
# — OR —
src/osprey/mcp_server/{name}/           # standalone server

├── __init__.py
├── __main__.py       # Entry point: dotenv, logging redirect, create & run
├── server.py         # Module-level FastMCP instance + create_server() factory
├── registry.py       # Singleton: config, service lifecycle, lazy creation
└── tools/
    ├── __init__.py
    └── {domain}.py   # @mcp.tool() async functions, one file per concern
```

## File-by-File Breakdown

### `server.py` — The FastMCP Instance

```python
"""MCP server for {your domain}."""

from fastmcp import FastMCP

from .registry import initialize_my_registry

mcp = FastMCP("{server-name}")


def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    initialize_my_registry()

    # Import tool modules — this triggers @mcp.tool() registration
    from .tools import analysis  # noqa: F401
    from .tools import data      # noqa: F401

    return mcp
```

**Key points:**
- `mcp` is a module-level global — tool modules import it
- `create_server()` is a factory, not a constructor
- Tool imports are inside the factory to control registration timing
- The `# noqa: F401` comments suppress unused-import warnings (the imports have side effects)

### `__main__.py` — The Entry Point

```python
"""Entry point for {server-name} MCP server."""

import sys

from dotenv import load_dotenv

from osprey.mcp_server.common import redirect_logging_to_stderr


def main() -> None:
    load_dotenv()
    redirect_logging_to_stderr()

    from .server import create_server

    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
```

**Critical rule:** `redirect_logging_to_stderr()` MUST be called before anything else. Stdout is the JSON-RPC transport — any stray print or log message on stdout will corrupt the protocol and crash the connection.

### `registry.py` — Singleton Config & Service

```python
"""Singleton registry for {your domain} MCP server."""

import logging

from osprey.mcp_server.common import load_osprey_config

logger = logging.getLogger(__name__)

_registry: "MyRegistry | None" = None


class MyRegistry:
    """Manages config and service lifecycle."""

    def __init__(self) -> None:
        self._config: dict | None = None
        self._service: "MyService | None" = None

    def initialize(self) -> None:
        """Load config from config.yml."""
        raw = load_osprey_config()
        self._config = raw.get("my_section", {})
        if not self._config:
            raise RuntimeError("Missing 'my_section' in config.yml")

    @property
    def config(self) -> dict:
        if self._config is None:
            raise RuntimeError("Registry not initialized")
        return self._config

    async def service(self) -> "MyService":
        """Lazy-create and cache the service instance."""
        if self._service is None:
            from .service import create_my_service
            self._service = await create_my_service(self.config)
        return self._service

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._service is not None:
            await self._service.close()
            self._service = None


def initialize_my_registry() -> None:
    global _registry
    _registry = MyRegistry()
    _registry.initialize()


def get_my_registry() -> MyRegistry:
    if _registry is None:
        raise RuntimeError("Registry not initialized — call initialize_my_registry() first")
    return _registry


def reset_my_registry() -> None:
    """Reset singleton (for testing only)."""
    global _registry
    _registry = None
```

**Key points:**
- Three functions: `initialize_*`, `get_*`, `reset_*` — always this trio
- Service is created lazily on first tool call, then cached
- `reset_*` exists solely for tests to prevent state leaks between test cases
- Config loading uses `load_osprey_config()` from `osprey.mcp_server.common`

### `tools/{domain}.py` — Tool Definitions

```python
"""Tools for {domain concern}."""

import json
import logging

from osprey.mcp_server.common import make_error

from ..registry import get_my_registry
from ..server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
async def my_tool(
    param1: str,
    param2: int = 10,
    param3: str = "default",
) -> str:
    """Short description Claude sees when deciding to use this tool.

    Detailed usage guidance:
    - When to use this tool vs. alternatives
    - What the parameters mean in operational context
    - What the returned JSON structure contains

    Args:
        param1: Human-readable description for Claude
        param2: What this controls (default: 10)
        param3: Operational meaning (default: "default")
    """
    try:
        registry = get_my_registry()
        service = await registry.service()

        result = await service.do_something(param1, param2)

        return json.dumps(
            {
                "param1": param1,
                "results_found": len(result.items),
                "items": [item.to_dict() for item in result.items],
            },
            default=str,
        )
    except ValueError as exc:
        return json.dumps(make_error("validation_error", str(exc)))
    except Exception as exc:
        logger.exception("my_tool failed")
        return json.dumps(make_error("internal_error", str(exc)))
```

**Rules for tools:**
1. **Always return `str`** — `json.dumps(...)`, never raw dicts
2. **Use `default=str`** in `json.dumps` — handles datetimes, Paths, UUIDs automatically
3. **Docstring is user-facing** — Claude reads it to decide when/how to use the tool. Write it for an accelerator operator, not a developer
4. **Error envelope** — always use `make_error(error_type, message, suggestions=[])`
5. **Three error types**: `"validation_error"` (bad input), `"not_found"` (missing data), `"internal_error"` (unexpected failure)
6. **One file per concern** — group related tools together (e.g., `search.py`, `browse.py`, `entry.py`)

## The Error Envelope

Every tool uses the same error format via `make_error()` from `osprey.mcp_server.common`:

```python
def make_error(
    error_type: str,
    error_message: str,
    suggestions: list[str] | None = None,
) -> dict:
    return {
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "suggestions": suggestions or [],
    }
```

Claude checks `data.get("error", False)` to detect failures and reads `suggestions` for recovery hints. Example:

```python
return json.dumps(make_error(
    "validation_error",
    f"Unknown mode '{mode}'. Valid modes: keyword, semantic, rag, agent",
    suggestions=["Try mode='keyword' for simple text matching"],
))
```

## Registration: Connecting to Claude Code

### Step 1: `.mcp.json` — Server Launch Config

Add your server to the project's `.mcp.json`:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "/path/to/python",
      "args": ["-m", "osprey.interfaces.my_tool.mcp"],
      "env": {
        "OSPREY_CONFIG": "/path/to/config.yml"
      }
    }
  }
}
```

For template-generated projects, add to `templates/claude_code/mcp.json.j2`:

```json
"my-server": {
  "command": "{{ current_python_env }}",
  "args": ["-m", "osprey.interfaces.my_tool.mcp"],
  "env": {
    "OSPREY_CONFIG": "{{ project_root }}/config.yml"
  }
}
```

### Step 2: `.claude/settings.json` — Permissions

Add permission rules so Claude knows what's allowed:

```json
{
  "permissions": {
    "allow": [
      "mcp__my-server__my_read_tool",
      "mcp__my-server__my_status_tool"
    ],
    "ask": [
      "mcp__my-server__my_write_tool"
    ]
  }
}
```

Convention:
- **allow**: Read-only, safe, idempotent tools
- **ask**: Tools that modify state, write data, or have side effects
- **deny**: Tools that should never be called (rare for MCP tools)

### Step 3: Hook Scripts (if writes are involved)

If your tool writes to hardware or has safety implications, add hooks in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__my-server__my_write_tool",
        "hooks": [
          { "command": ".claude/hooks/my_validation.py" }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "mcp__my-server__*",
        "hooks": [
          { "command": ".claude/hooks/osprey_audit.py" }
        ]
      }
    ]
  }
}
```

## Workspace Output

If your tool produces output files (search results, analysis data), save them to the workspace:

```python
from osprey.mcp_server.common import save_to_workspace

filepath = save_to_workspace(
    category="txt_analysis",          # Creates osprey-workspace/txt_analysis/
    data={"results": [...], ...},
    description="Turn-by-turn analysis results",
    tool_name="txt_analyze",
)

return json.dumps({
    "results": [...],
    "workspace_file": str(filepath),  # Include path in response
}, default=str)
```

## Concrete Reference

**ARIEL MCP server** (`src/osprey/interfaces/ariel/mcp/`):
- `server.py` — 15 lines, `create_server()` imports 5 tool modules
- `registry.py` — `ARIELMCPRegistry` with lazy `ARIELSearchService` creation
- `tools/search.py` — `ariel_search` tool, 160 lines, full error handling
- `tools/browse.py` — `ariel_browse` + `ariel_filter_options`, two tools in one file
- `tools/entry.py` — `ariel_entry_get` + `ariel_entry_create`, draft vs. direct modes
- `tools/capabilities.py` — `ariel_capabilities`, reports enabled modules
- `tools/status.py` — `ariel_status`, health check

## Checklist

- [ ] `server.py` with module-level `mcp = FastMCP("name")` and `create_server()` factory
- [ ] `__main__.py` with `load_dotenv()`, `redirect_logging_to_stderr()`, `create_server().run()`
- [ ] `registry.py` with `initialize_*`, `get_*`, `reset_*` trio and lazy service creation
- [ ] Tool files in `tools/` with `@mcp.tool()` decorators returning JSON strings
- [ ] All errors wrapped in `make_error()` envelope
- [ ] Tool docstrings written for operators, not developers
- [ ] Server registered in `.mcp.json`
- [ ] Permissions set in `.claude/settings.json`
- [ ] Hooks added for any write/modify operations
