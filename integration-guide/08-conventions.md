# Recipe 8: Universal Conventions

## Code Style

| Rule | Value |
|------|-------|
| Line length | 100 characters |
| Python version | 3.11+ (`match`, `TypedDict`, `NotRequired`, `\|` union syntax) |
| Formatter/linter | Ruff (both formatting and linting) |
| Type checker | mypy (gradual typing) |
| Docstrings | Google-style |
| Import order | stdlib → third-party → local (Ruff enforces this) |

```bash
ruff check src/ tests/           # Lint
ruff format src/ tests/          # Format
mypy src/                        # Type check
```

## Naming Conventions

### Python

| Thing | Convention | Example |
|-------|-----------|---------|
| Modules | snake_case | `txt_analyzer.py` |
| Classes | PascalCase | `TxTAnalyzer`, `MyRegistry` |
| Functions/methods | snake_case | `analyze_turns()` |
| Constants | UPPER_SNAKE | `MAX_TURNS = 8192` |
| Private attributes | leading underscore | `self._service` |
| EPICS-related classes | ALL-CAPS prefix | `EPICSConnector`, `EPICSArchiverConnector` |
| Type aliases | PascalCase | `EntryDict = dict[str, Any]` |

### MCP Tools

| Thing | Convention | Example |
|-------|-----------|---------|
| Tool names | snake_case, prefixed by domain | `txt_analyze`, `txt_get_tunes`, `txt_status` |
| Server names | kebab-case | `my-tool`, `osprey-workspace` |
| Permission paths | `mcp__{server}__{tool}` | `mcp__my-tool__txt_analyze` |

### Web API

| Thing | Convention | Example |
|-------|-----------|---------|
| Route paths | `/api/{resource}` | `/api/items`, `/api/items/{id}` |
| Query params | snake_case | `?page_size=20&sort_by=timestamp` |
| JSON keys | snake_case | `{"entry_id": "...", "raw_text": "..."}` |
| HTTP methods | REST standard | GET=read, POST=create/action, PUT=update |
| Error response | `{"detail": "message"}` | FastAPI HTTPException format |

### Files & Directories

| Thing | Convention | Example |
|-------|-----------|---------|
| Interface dirs | snake_case | `src/osprey/interfaces/my_tool/` |
| MCP server dirs | snake_case | `src/osprey/interfaces/my_tool/mcp/` |
| Test dirs | mirror source structure | `tests/interfaces/my_tool/mcp/` |
| Config keys | snake_case | `my_tool:` section in config.yml |
| CLI commands | kebab-case | `osprey my-tool web` |
| Workspace categories | snake_case | `osprey-workspace/txt_analysis/` |

## Error Handling

### MCP Tools: Return Errors, Never Raise

```python
@mcp.tool()
async def my_tool(param: str) -> str:
    try:
        # ... business logic ...
        return json.dumps({"result": ...})
    except ValueError as exc:
        # Known/expected error — validation failure
        return json.dumps(make_error("validation_error", str(exc)))
    except FileNotFoundError as exc:
        # Known/expected error — missing data
        return json.dumps(make_error("not_found", str(exc)))
    except Exception as exc:
        # Unexpected error — log full traceback
        logger.exception("my_tool failed")
        return json.dumps(make_error("internal_error", str(exc)))
```

Tools NEVER raise exceptions — they always return a JSON error envelope. This prevents Claude Code from getting raw tracebacks.

### Web API: Use HTTPException

```python
@router.post("/analyze")
async def analyze(request: Request, body: AnalyzeRequest):
    try:
        result = await service.analyze(body.query)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
```

### Repository: Wrap in Domain Exceptions

```python
async def get_entry(self, entry_id: str) -> dict | None:
    try:
        # ... SQL query ...
    except Exception as e:
        raise DatabaseQueryError(
            f"Failed to get entry {entry_id}: {e}",
            query=f"SELECT entry_id={entry_id}",
        ) from e
```

## Async Patterns

### Always Use `async def` for Tools and Routes

MCP tools and FastAPI routes are async. Even if your business logic is synchronous, keep the interface async for consistency:

```python
@mcp.tool()
async def my_tool(param: str) -> str:   # Always async
    result = compute_something(param)     # Sync is fine inside
    return json.dumps(result)
```

### Connection Pools, Not Per-Request Connections

```python
# GOOD — pool managed by registry, shared across requests
service = await registry.service()  # Returns cached service with pool

# BAD — new connection per request
conn = await psycopg.connect(uri)   # Don't do this
```

### Lazy Service Creation

```python
async def service(self) -> MyService:
    if self._service is None:
        self._service = await create_my_service(self.config)
    return self._service
```

## Safety Rules

These apply to any tool that interacts with hardware or has side effects:

1. **Master kill switch** — if your tool can write to hardware, respect `control_system.writes_enabled`
2. **Human approval** — write operations must go through the hook chain or approval system
3. **Audit logging** — all operations should be logged (PostToolUse hook)
4. **Input validation** — validate at the tool boundary, not deep in the service layer
5. **Path traversal protection** — any file-serving endpoint must validate paths with `is_relative_to()`
6. **No secrets in config.yml** — use `${ENV_VAR}` syntax for API keys, credentials

## Dependency Management

### Adding Dependencies

Add to `pyproject.toml` under the appropriate section:

```toml
[project]
dependencies = [
    # ... existing deps ...
    "your-new-dep>=1.0.0",
]

[project.optional-dependencies]
dev = [
    # ... test/dev deps ...
]
```

### Version Pinning

- **Minimum versions** for libraries: `"fastmcp>=2.0.0"`
- **Range pins** for critical dependencies: `"psycopg[binary,pool]>=3.1.0,<4.0.0"`
- **Exact pins** only for known-breaking changes: `"claude-agent-sdk==0.1.26"`

## Git Workflow

- Feature branches only — never commit directly to `main`
- Conventional commits: `type(scope): short description`
  - `feat(txt)`: New feature
  - `fix(txt)`: Bug fix
  - `refactor(txt)`: Code restructuring
  - `test(txt)`: Test additions/changes
  - `docs(txt)`: Documentation only
- Pull requests for all changes

## Package Data

If your interface has static files (HTML, CSS, JS), register them in `pyproject.toml`:

```toml
[tool.setuptools.package-data]
osprey = [
    # ... existing entries ...
    "interfaces/my_tool/static/**/*",
]
```

This ensures static files are included when the package is installed.

## Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Processing entry %s", entry_id)      # Verbose, development only
logger.info("Analysis complete: %d results", count) # Normal operation
logger.warning("Falling back to default mode")      # Unexpected but handled
logger.error("Failed to connect to database")       # Error, operation failed
logger.exception("Unexpected failure in my_tool")   # Error + full traceback
```

**Critical**: In MCP servers, all logging goes to stderr (via `redirect_logging_to_stderr()`). Never use `print()` in MCP server code — it writes to stdout and corrupts the JSON-RPC transport.

## Checklist

- [ ] Code passes `ruff check` and `ruff format` (100-char line length)
- [ ] Type hints on all public functions
- [ ] Google-style docstrings on all public classes and functions
- [ ] MCP tools return JSON error envelopes (never raise)
- [ ] Web routes use HTTPException for errors
- [ ] All SQL wrapped in domain exceptions
- [ ] No `print()` in MCP server code
- [ ] Secrets use `${ENV_VAR}` syntax in config
- [ ] Static files registered in `pyproject.toml` package-data
- [ ] Conventional commit messages
