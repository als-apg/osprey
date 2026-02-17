# Recipe 3: Adding a CLI Command

## When You Need This

You want users to be able to launch your interface or run operations via `osprey {your-command}`.

## The Pattern

OSPREY's CLI uses a **LazyGroup** — commands are only imported when invoked, keeping `osprey --help` fast even as the project grows.

### Step 1: Create the Command Module

```python
# src/osprey/cli/my_tool_cmd.py
"""CLI commands for {your tool}."""

import click


@click.group("my-tool")
def my_tool():
    """Commands for {your tool}."""


@my_tool.command("web")
@click.option("--port", default=8088, help="Server port")
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev)")
def web(port: int, host: str, reload: bool):
    """Launch the {your tool} web interface."""
    # Read config for overrides
    from osprey.mcp_server.common import load_osprey_config

    config = load_osprey_config()
    section = config.get("my_tool", {}).get("web", {})
    host = section.get("host", host)
    port = section.get("port", port)

    click.echo(f"Starting {your tool} at http://{host}:{port}")

    if reload:
        import uvicorn
        uvicorn.run(
            "osprey.interfaces.my_tool.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
    else:
        from osprey.interfaces.my_tool import run_server
        run_server(host=host, port=port)


@my_tool.command("status")
def status():
    """Check {your tool} service status."""
    import requests
    from osprey.mcp_server.common import load_osprey_config

    config = load_osprey_config()
    section = config.get("my_tool", {}).get("web", {})
    host = section.get("host", "127.0.0.1")
    port = section.get("port", 8088)

    try:
        resp = requests.get(f"http://{host}:{port}/health", timeout=3)
        if resp.status_code == 200:
            click.echo(f"✓ Running at http://{host}:{port}")
        else:
            click.echo(f"✗ Unhealthy (status {resp.status_code})")
    except requests.ConnectionError:
        click.echo(f"✗ Not running (tried http://{host}:{port})")
```

### Step 2: Register in `main.py`

Add your command to the `LazyGroup.get_command()` mapping in `src/osprey/cli/main.py`:

```python
class LazyGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        commands = {
            # ... existing commands ...
            "my-tool": "osprey.cli.my_tool_cmd",
        }
        if cmd_name not in commands:
            return None
        import importlib
        mod = importlib.import_module(commands[cmd_name])
        return getattr(mod, cmd_name.replace("-", "_"))
```

Also add the command name to the `list_commands()` method so it appears in `osprey --help`.

## Conventions

- **Command names**: Kebab-case (`my-tool`, not `my_tool` or `myTool`)
- **Subcommands**: Use `@click.group()` for commands with subcommands (like `osprey artifacts web`)
- **Config precedence**: CLI flags > `config.yml` values > hardcoded defaults
- **Imports inside functions**: All heavy imports (`uvicorn`, `fastapi`, service modules) happen inside the command function, not at module level. This keeps CLI startup fast.
- **`--reload` flag**: Always offer it for web server commands (developer convenience)

## Config Section

Add a section to `config.yml` for your tool:

```yaml
my_tool:
  web:
    host: 127.0.0.1
    port: 8088
    auto_launch: true
```

See [Recipe 4](04-config-system.md) for full config integration.

## Concrete Reference

- `src/osprey/cli/artifacts_cmd.py` — Group with `web` subcommand, config-driven defaults
- `src/osprey/cli/web_cmd.py` — Single command (not a group), launches web terminal
- `src/osprey/cli/main.py` — `LazyGroup` with command-to-module mapping

## Checklist

- [ ] Command module in `src/osprey/cli/{name}_cmd.py`
- [ ] Registered in `LazyGroup.get_command()` mapping in `main.py`
- [ ] Added to `list_commands()` in `main.py`
- [ ] Heavy imports inside command functions (not at module level)
- [ ] Config values used as defaults, CLI flags as overrides
- [ ] `--reload` flag for web server commands
