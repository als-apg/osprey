"""CLI commands for the OSPREY Artifact Gallery.

Provides `osprey artifacts web` to launch the gallery server manually.
"""

import click


def get_config_value(key: str, default=None):
    """Read a top-level config value from config.yml."""
    from osprey.mcp_server.common import load_osprey_config

    return load_osprey_config().get(key, default)


@click.group("artifacts")
def artifacts():
    """Artifact Gallery commands.

    Manage the OSPREY Artifact Gallery — a local web gallery that displays
    interactive plots, tables, and other outputs produced by Claude during
    analysis sessions.
    """


@artifacts.command("web")
@click.option("--port", "-p", type=int, default=None, help="Port to run on (default: from config or 8086)")
@click.option("--host", "-h", default=None, help="Host to bind to (default: from config or 127.0.0.1)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def web(port: int | None, host: str | None, reload: bool) -> None:
    """Launch the Artifact Gallery web interface.

    Starts a FastAPI server that serves the artifact gallery UI.
    Artifacts are created by Claude via save_artifact() in execute
    or the artifact_save MCP tool.

    Example:

    \b
        osprey artifacts web                    # Start on localhost:8086
        osprey artifacts web --port 9000        # Custom port
        osprey artifacts web --host 0.0.0.0     # Bind to all interfaces
        osprey artifacts web --reload           # Development mode
    """
    art_config = get_config_value("artifact_server", {})
    host = host or art_config.get("host", "127.0.0.1")
    port = port or art_config.get("port", 8086)

    click.echo(f"Starting OSPREY Artifact Gallery on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop\n")

    try:
        if reload:
            import uvicorn

            uvicorn.run(
                "osprey.interfaces.artifacts.app:create_app",
                factory=True,
                host=host,
                port=port,
                reload=reload,
                log_level="info",
            )
        else:
            from osprey.interfaces.artifacts import run_server

            run_server(host=host, port=port)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
