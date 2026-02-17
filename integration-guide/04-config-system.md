# Recipe 4: Connecting to the Config System

## When You Need This

Your tool needs runtime configuration — database URIs, default parameters, feature flags, API endpoints. OSPREY uses a single `config.yml` file with hierarchical sections.

## How Config Loading Works

### The Loading Chain

```
config.yml location resolution:
  1. Explicit path argument (if provided)
  2. OSPREY_CONFIG environment variable
  3. ./config.yml in current working directory
  4. {} (empty dict fallback)
```

The shared loader lives in `osprey.mcp_server.common`:

```python
from osprey.mcp_server.common import load_osprey_config

raw_config = load_osprey_config()           # Uses env/cwd resolution
raw_config = load_osprey_config("/path")    # Explicit path
```

### Your Config Section

Add a top-level key to `config.yml`:

```yaml
# Existing sections (don't touch these)
model_configuration: ...
control_system: ...
ariel: ...

# Your new section
my_tool:
  database:
    uri: postgresql://localhost:5432/my_tool_db
  analysis:
    default_mode: fft
    max_turns: 1024
    window_function: hann
  web:
    host: 127.0.0.1
    port: 8088
    auto_launch: true
```

### Accessing Config in Your Code

**In MCP server (via registry):**

```python
# registry.py
def initialize(self) -> None:
    raw = load_osprey_config()
    self._config = raw.get("my_tool", {})
```

**In web interface (via app.state):**

```python
# app.py
def create_app(config_path=None):
    app = FastAPI(...)
    raw = load_osprey_config(config_path)
    app.state.config = raw.get("my_tool", {})
    ...
```

**In CLI commands:**

```python
# my_tool_cmd.py
@my_tool.command("web")
@click.option("--port", default=8088)
def web(port):
    config = load_osprey_config()
    section = config.get("my_tool", {}).get("web", {})
    port = section.get("port", port)  # Config overrides default, CLI flag overrides config
```

## Typed Config (Optional but Recommended)

For complex configuration, use dataclasses with a `.from_dict()` factory:

```python
# config.py
from dataclasses import dataclass, field


@dataclass
class DatabaseConfig:
    uri: str

    @classmethod
    def from_dict(cls, data: dict) -> "DatabaseConfig":
        # Handle key name mapping if needed
        uri = data.get("uri") or data.get("connection_string", "")
        return cls(uri=uri)


@dataclass
class AnalysisConfig:
    default_mode: str = "fft"
    max_turns: int = 1024
    window_function: str = "hann"

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MyToolConfig:
    database: DatabaseConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "MyToolConfig":
        return cls(
            database=DatabaseConfig.from_dict(data.get("database", {})),
            analysis=AnalysisConfig.from_dict(data.get("analysis", {})),
        )

    def is_feature_enabled(self, name: str) -> bool:
        """Check if an optional feature/module is enabled."""
        modules = self.analysis.__dict__
        return modules.get(name, {}).get("enabled", False) if isinstance(modules.get(name), dict) else False
```

ARIEL uses this pattern extensively — see `src/osprey/services/ariel_search/config.py` for the full example with `ARIELConfig`, `SearchModuleConfig`, `PipelineModuleConfig`, etc.

## Template Integration

If your tool should be available in newly-scaffolded projects, add its config to the Jinja2 templates:

### `templates/apps/control_assistant/config.yml.j2`

```yaml
# --- {Your Tool} Configuration ---
my_tool:
  database:
    uri: postgresql://localhost:5432/{{ project_name }}_my_tool
  analysis:
    default_mode: fft
    max_turns: 1024
  web:
    host: 127.0.0.1
    port: 8088
    auto_launch: true
```

### Conditional Rendering

If your tool is only relevant for certain templates:

```jinja2
{% if my_tool_enabled is defined and my_tool_enabled %}
my_tool:
  database:
    uri: postgresql://localhost:5432/{{ project_name }}_my_tool
{% endif %}
```

## Environment Variable Substitution

Use `${ENV_VAR}` syntax in config values for secrets:

```yaml
my_tool:
  database:
    uri: ${MY_TOOL_DATABASE_URL}
  api_key: ${MY_TOOL_API_KEY}
```

These are resolved at load time by the config loader, with `.env` file support via `python-dotenv`.

## Config Editing via Web UI

ARIEL's web interface includes a config editor that reads/writes `config.yml` via REST endpoints. If you want the same capability:

```python
@router.get("/config")
async def get_config():
    config_path = Path.cwd() / "config.yml"
    raw_yaml = config_path.read_text()
    parsed = yaml.safe_load(raw_yaml)
    return {"raw": raw_yaml, "parsed": parsed}

@router.put("/config")
async def update_config(body: ConfigUpdateRequest):
    config_path = Path.cwd() / "config.yml"
    # Validate YAML
    yaml.safe_load(body.content)
    # Backup
    backup_path = config_path.with_suffix(".yml.bak")
    shutil.copy2(config_path, backup_path)
    # Write with fsync
    with open(config_path, "w") as f:
        f.write(body.content)
        f.flush()
        os.fsync(f.fileno())
    return {"status": "saved", "backup": str(backup_path)}
```

**Always create a `.yml.bak` backup before writing and use `fsync` to ensure data reaches disk.**

## Concrete Reference

- `src/osprey/services/ariel_search/config.py` — Full typed config with `ARIELConfig`, module/pipeline configs, `.from_dict()` factories
- `src/osprey/interfaces/ariel/app.py:load_ariel_config()` — Config loading with Docker override support
- `src/osprey/interfaces/ariel/api/routes.py` — Config read/write endpoints with backup + fsync
- `src/osprey/templates/apps/control_assistant/config.yml.j2` — Full production config template

## Checklist

- [ ] Top-level key in `config.yml` for your tool section
- [ ] Loaded via `load_osprey_config()` from `osprey.mcp_server.common`
- [ ] Config section stored on registry (MCP) or `app.state` (web)
- [ ] Typed config dataclasses with `.from_dict()` if config is complex
- [ ] Added to Jinja2 template if tool should appear in new projects
- [ ] Secrets use `${ENV_VAR}` syntax, not hardcoded values
