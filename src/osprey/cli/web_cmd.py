"""CLI command for the OSPREY Web Terminal.

Provides `osprey web` to launch a browser-based split-pane interface
with a real terminal (PTY) and live workspace file viewer.

Supports `--detach` for background operation and `osprey web stop`
to shut down a detached instance.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import click

PID_FILE = ".osprey-web.pid"
LOG_FILE = ".osprey-web.log"


def get_config_value(key: str, default=None):
    """Read a top-level config value from config.yml."""
    from osprey.utils.workspace import load_osprey_config

    return load_osprey_config().get(key, default)


# -- helpers ---------------------------------------------------------------


def _read_pid(project_dir: Path) -> int | None:
    """Read PID file and return the PID if the process is alive.

    Removes stale PID files automatically.
    """
    pid_path = project_dir / PID_FILE
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        pid_path.unlink(missing_ok=True)
        return None
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no signal sent
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return None
    except PermissionError:
        return pid  # process exists but owned by another user
    return pid


def _write_pid(project_dir: Path, pid: int) -> None:
    """Write PID to file."""
    (project_dir / PID_FILE).write_text(str(pid))


def _wait_for_server(
    host: str, port: int, proc: subprocess.Popen, timeout: float = 10.0
) -> bool:
    """Poll server port until connection succeeds or timeout.

    Also checks proc.poll() each iteration to detect early crashes
    (e.g. port already in use).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False  # process exited early
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


# -- CLI -------------------------------------------------------------------


@click.group("web", invoke_without_command=True)
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
@click.option("--detach", is_flag=True, help="Run in background, write PID file")
@click.pass_context
def web(
    ctx: click.Context,
    port: int | None,
    host: str | None,
    reload: bool,
    shell: str | None,
    project: str | None,
    detach: bool,
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
        osprey web --detach                # Start in background
        osprey web stop                    # Stop background server
    """
    if ctx.invoked_subcommand is not None:
        return

    wt_config = get_config_value("web_terminal", {})
    host = host or wt_config.get("host", "127.0.0.1")
    port = port or wt_config.get("port", 8087)
    from osprey.utils.shell_resolver import resolve_shell_command

    shell_raw = shell or wt_config.get("shell") or "claude"
    try:
        shell = resolve_shell_command(shell_raw)
    except FileNotFoundError as e:
        click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    if detach:
        _start_detached(host, port, shell, project)
        return

    # -- foreground (original behavior) ------------------------------------

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


def _start_detached(
    host: str, port: int, shell: str | None, project: str | None
) -> None:
    """Spawn the web server as a background process."""
    project_dir = Path(project).resolve() if project else Path.cwd()

    # Idempotent: if already running, just report
    existing = _read_pid(project_dir)
    if existing is not None:
        click.echo(f"Web terminal already running (PID {existing}).")
        click.echo("  Stop with: osprey web stop")
        return

    # Build the child command (no --detach to avoid recursion)
    cmd = [sys.executable, "-m", "osprey.cli.main", "web", "--host", host, "--port", str(port)]
    if shell:
        cmd += ["--shell", shell]
    if project:
        cmd += ["--project", str(Path(project).resolve())]

    log_path = project_dir / LOG_FILE
    log_fh = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_fh.close()  # child retains its own fd copy

    _write_pid(project_dir, proc.pid)

    if _wait_for_server(host, port, proc):
        click.echo(f"Web terminal started (PID {proc.pid}).")
        click.echo(f"  URL:  http://{host}:{port}")
        click.echo(f"  Log:  {log_path}")
        click.echo("  Stop: osprey web stop")
    else:
        exit_code = proc.poll()
        if exit_code is not None:
            click.echo(f"Server exited immediately (code {exit_code}). Check {log_path}")
            (project_dir / PID_FILE).unlink(missing_ok=True)
        else:
            click.echo(f"Server started (PID {proc.pid}) but not yet responding on port {port}.")
            click.echo(f"  Log: {log_path}")
            click.echo("  Stop: osprey web stop")


@web.command("stop")
@click.option(
    "--project",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="OSPREY project directory (default: current directory)",
)
def web_stop(project: str | None) -> None:
    """Stop a background web terminal server."""
    project_dir = Path(project).resolve() if project else Path.cwd()
    pid_path = project_dir / PID_FILE
    log_path = project_dir / LOG_FILE

    if not pid_path.exists():
        click.echo("No running web terminal found (no PID file).")
        return

    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        click.echo("Corrupt PID file — removing.")
        pid_path.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Stopped web terminal (PID {pid}).")
    except ProcessLookupError:
        click.echo(f"Process {pid} not found (already stopped). Cleaning up.")
    except PermissionError:
        click.echo(f"Permission denied killing PID {pid}.")
        return

    pid_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
