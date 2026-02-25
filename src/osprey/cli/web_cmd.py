"""CLI command for the OSPREY Web Terminal.

Provides `osprey web` to launch a browser-based split-pane interface
with a real terminal (PTY) and live workspace file viewer.
"""

import click


def get_config_value(key: str, default=None):
    """Read a top-level config value from config.yml."""
    from osprey.mcp_server.common import load_osprey_config

    return load_osprey_config().get(key, default)


@click.command("web")
@click.option(
    "--port", "-p", type=int, default=None, help="Port to run on (default: from config or 8087)"
)
@click.option("--host", default=None, help="Host to bind to (default: from config or 127.0.0.1)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--shell", default=None, help="Shell command to run (default: claude)")
@click.option(
    "--project",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="OSPREY project directory (default: current directory)",
)
def web(
    port: int | None, host: str | None, reload: bool, shell: str | None, project: str | None
) -> None:
    """Launch the OSPREY Web Terminal interface.

    Starts a FastAPI server with a split-pane UI: a real terminal (PTY) on the
    left and a live workspace file viewer on the right.

    Example:

    \b
        osprey web                         # Start on localhost:8087
        osprey web --port 9000             # Custom port
        osprey web --host 0.0.0.0          # Bind to all interfaces
        osprey web --shell zsh             # Use zsh instead of claude
        osprey web --reload                # Development mode
    """
    wt_config = get_config_value("web_terminal", {})
    host = host or wt_config.get("host", "127.0.0.1")
    port = port or wt_config.get("port", 8087)
    shell = shell or wt_config.get("shell") or "claude"

    if host == "0.0.0.0":
        click.echo("WARNING: Binding to 0.0.0.0 exposes the terminal to the network.")
        click.echo("This is a single-user tool — add authentication before external exposure.\n")

    click.echo(f"Starting OSPREY Web Terminal on http://{host}:{port}")
    click.echo(f"Shell: {shell}")
    click.echo("Press Ctrl+C to stop\n")

    # Load .env so secrets (e.g. CONFLUENCE_ACCESS_TOKEN) are available
    from osprey.mcp_env import load_dotenv_from_project

    load_dotenv_from_project()

    try:
        if reload:
            import uvicorn

            from osprey.interfaces.web_terminal.app import _open_browser_when_ready

            _open_browser_when_ready(f"http://{host}:{port}")

            uvicorn.run(
                "osprey.interfaces.web_terminal.app:create_app",
                factory=True,
                host=host,
                port=port,
                reload=reload,
                log_level="info",
            )
        else:
            from osprey.interfaces.web_terminal import run_web

            run_web(host=host, port=port, shell_command=shell, project_dir=project)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
