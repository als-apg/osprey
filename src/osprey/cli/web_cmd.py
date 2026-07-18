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
from collections.abc import Mapping
from pathlib import Path

import click

PID_FILE = ".osprey-web.pid"
LOG_FILE = ".osprey-web.log"
DECLARED_BIND_ENV = "OSPREY_TERMINAL_BIND_HOST"
DECLARED_WEB_PORT_ENV = "OSPREY_TERMINAL_WEB_PORT"


def resolve_bind_host(
    cli_host: str | None, config_host: str | None, env: Mapping[str, str] = os.environ
) -> str:
    """Single source of the address ``osprey web`` binds to. Enforces criterion C3.

    SECURITY INVARIANT: when a deployment DECLARES a bind host via
    ``OSPREY_TERMINAL_BIND_HOST`` (the multi-user compose sets it on every
    per-user container so nginx is the ONLY off-host path), that declaration is
    AUTHORITATIVE over ``--host`` and config. A stale or hostile image CMD
    passing ``--host 0.0.0.0`` must NOT punch through the reverse-proxy
    chokepoint. Single-user ``osprey web`` sets no such env, so ``--host`` is
    honored verbatim (``0.0.0.0`` stays supported).

    Do NOT collapse this into ``@click.option("--host", envvar=...)``: Click env
    defaults LOSE to an explicit flag, which would silently re-open the
    container to the network. This inversion is load-bearing and is pinned red
    by ``test_multiuser_env_pins_loopback_reaches_run_web``.
    """
    declared = env.get(DECLARED_BIND_ENV)
    if declared:
        return declared
    return cli_host or config_host or "127.0.0.1"


def resolve_web_port(
    cli_port: int | None, config_port: int | None, env: Mapping[str, str] = os.environ
) -> int:
    """Single source of the port ``osprey web`` binds to. Enforces criterion C3 for ports.

    DECLARATION-ONLY INVARIANT: when a deployment DECLARES a port via
    ``OSPREY_TERMINAL_WEB_PORT`` (the multi-user compose overlay sets it on
    every per-user container so nginx's per-user upstream mapping always
    matches the container's actual listener), that declaration is
    AUTHORITATIVE over ``--port`` and config. A stale or hostile image CMD
    passing a mismatched ``--port`` must NOT desync the container from the
    reverse-proxy's routing table. Single-user ``osprey web`` sets no such
    env, so ``--port`` (or the ``OSPREY_WEB_PORT`` click envvar fallback, or
    config, or the 8087 default) is honored verbatim.

    ``OSPREY_TERMINAL_WEB_PORT`` is a DECLARATION set by the compose overlay
    for THIS container only — it is never re-exported to children, unlike
    the child-facing ``OSPREY_WEB_PORT`` publication at the bottom of
    ``web()``. Do NOT collapse this into ``@click.option("--port",
    envvar=...)``: Click env defaults LOSE to an explicit flag, which is the
    opposite of "declared wins" this function exists to provide — the same
    reasoning that keeps ``resolve_bind_host`` a plain function rather than
    a click envvar.
    """
    declared = env.get(DECLARED_WEB_PORT_ENV)
    if declared:
        return int(declared)
    return cli_port or config_port or 8087


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


def _preflight_vendor_check() -> None:
    """In offline mode, fail fast if ``static/vendor/`` assets are missing.

    Only relevant when ``OSPREY_OFFLINE=1`` (or ``offline: true`` in
    ``config.yml``). In default CDN mode there's nothing local to verify —
    the browser loads assets straight from jsDelivr / cdn.plot.ly.
    """
    from osprey.interfaces.vendor import is_offline, verify_all

    if not is_offline():
        return

    _, problems = verify_all()
    if not problems:
        return

    click.echo("ERROR: offline mode is on but vendor assets are missing or corrupt:", err=True)
    for p in problems[:5]:
        click.echo(f"  {p}", err=True)
    if len(problems) > 5:
        click.echo(f"  ... and {len(problems) - 5} more", err=True)
    click.echo("\nFix:  uv run osprey vendor fetch", err=True)
    raise SystemExit(1)


# FRAMEWORK_WEB_SERVERS keys don't line up 1:1 with the panel ids
# _load_panel_config() reports: channel_finder/lattice_dashboard use
# underscores while profiles.web_panels.BUILTIN_PANELS uses the hyphenated/
# short ids the frontend and web.panels config actually key on
# ("channel-finder", "lattice"). "artifact" is intentionally absent — it's a
# UNIVERSAL_PANELS entry the lifespan launches unconditionally (see
# _create_lifespan in web_terminal/app.py), so it is never gated on
# web.panels membership.
_PANEL_ID_FOR_REGISTRY_KEY: dict[str, str] = {
    "ariel": "ariel",
    "channel_finder": "channel-finder",
    "lattice_dashboard": "lattice",
    "okf": "okf",
}


def _probe_companion_ports() -> list[str]:
    """Probe 1: TCP-connect-probe every companion panel port the lifespan will bind.

    Resolves the panel set the same way ``_create_lifespan`` does: enabled via
    ``web.panels`` (or ``artifact``, which is always launched) AND actually
    launchable per ``auto_launch``/``require_section``. A panel that is
    enabled but not launched (e.g. ``channel_finder`` with an unmet
    ``require_section``) is excluded — its port is never probed.

    A listener already bound to a companion port before we start ours is
    foreign: at best it steals the panel's tab, at worst it silently
    reverse-proxies another project's data into this UI. Zero network I/O
    beyond the local TCP connect probe itself — no server starts, no
    registry init, no LLM calls.
    """
    from osprey.infrastructure.server_launcher import (
        _launchers,
        _make_auto_launch_checker,
        _make_config_reader,
    )
    from osprey.interfaces.web_terminal.app import _load_panel_config
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    enabled_panels, _custom_panels, _default_panel = _load_panel_config()

    failures: list[str] = []
    for key, defn in FRAMEWORK_WEB_SERVERS.items():
        if key != "artifact" and _PANEL_ID_FOR_REGISTRY_KEY.get(key) not in enabled_panels:
            continue  # panel disabled in web.panels — the lifespan never calls its launcher
        if not _make_auto_launch_checker(defn)():
            continue  # auto_launch off, or require_section unmet
        host, port = _make_config_reader(defn)()
        if _launchers[key]._port_has_listener(host, port):
            failures.append(
                f"Companion panel '{key}' ({defn.name}) port {port} is already in use "
                "by another process.\n"
                f"  Find the process:  lsof -i :{port}"
            )
    return failures


def _resolve_project_config_path(project_dir: Path) -> Path:
    """Resolve config.yml the same way ``resolve_config_path()`` does, for *project_dir*.

    Mirrors ``osprey.utils.workspace.resolve_config_path()`` (``OSPREY_CONFIG``
    env var first, else ``<dir>/config.yml``) but keyed off the pre-flight's
    already-resolved ``project_dir`` instead of ``Path.cwd()`` — the two agree
    whenever ``--project`` isn't passed, since ``project_dir`` defaults to cwd.
    """
    return Path(
        os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(project_dir / "config.yml")))
    )


def _probe_auth_secret(project_dir: Path | None) -> tuple[list[str], list[str]]:
    """Probe 2: the resolved provider's auth secret must be resolvable before launch.

    A proxy provider (als-apg, cborg, a custom ``api.providers`` entry, ...)
    that can't authenticate upstream is a hard failure — the terminal would
    launch straight into an auth error. Direct Anthropic (subscription/OAuth)
    has no such requirement, so a missing ``ANTHROPIC_API_KEY`` there is only
    a warning, not an abort.

    Checks both ``os.environ`` and the project's ``.env`` (via
    ``dotenv_values``, which reads without mutating ``os.environ``): the real
    launch's ``load_dotenv_from_project()`` only runs after pre-flight on the
    foreground path (see ``web()``), so a secret that lives only in ``.env``
    must still count as present here — otherwise a healthy proxy launch would
    false-fail.

    Zero network: ``load_provider_spec`` is a pure config read. A missing or
    malformed config.yml, or an unknown provider name, is left for Probe 3 (or
    the launch itself) to report — this probe just skips quietly rather than
    duplicating that diagnosis.
    """
    if project_dir is None:
        return [], []

    from osprey.cli.claude_code_resolver import load_provider_spec

    try:
        spec = load_provider_spec(project_dir)
    except (OSError, ValueError):
        return [], []
    if spec is None or not spec.auth_secret_env:
        return [], []

    secret_present = bool(os.environ.get(spec.auth_secret_env))
    if not secret_present:
        env_file = project_dir / ".env"
        if env_file.is_file():
            from dotenv import dotenv_values

            secret_present = bool(dotenv_values(env_file).get(spec.auth_secret_env))
    if secret_present:
        return [], []

    preamble = f"auth secret ${spec.auth_secret_env} not found in environment or .env "
    if spec.needs_proxy:
        return [f"{preamble}(provider {spec.provider} requires it)"], []
    return (
        [],
        [f"{preamble}(provider {spec.provider}); falling back to subscription/OAuth login"],
    )


def _probe_config_validity(project_dir: Path | None) -> list[str]:
    """Probe 3: config.yml and .claude/settings.json must at least parse.

    ``load_osprey_config()`` swallows every exception and returns ``{}`` on
    malformed YAML (see ``osprey.utils.workspace.load_osprey_config``), which
    would otherwise let the launch silently proceed on defaults instead of the
    project's actual configuration. This probe does its own dedicated parse of
    each file so a syntax error surfaces as a pre-flight failure instead of an
    inexplicable defaults-instead-of-config bug after launch. Deliberately
    does not call ``validate_agent_tools_against_permissions()`` — agent-tool
    / permission drift is a build-time concern, not a launch gate.

    Both files are optional — a project without one is not a failure, just
    nothing to validate.
    """
    if project_dir is None:
        return []

    failures: list[str] = []

    settings_path = project_dir / ".claude" / "settings.json"
    if settings_path.exists():
        import json

        try:
            json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            failures.append(f"{settings_path}: invalid JSON ({e})")

    config_path = _resolve_project_config_path(project_dir)
    if config_path.exists():
        import yaml

        try:
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            failures.append(f"{config_path}: invalid YAML ({e})")

    return failures


def _preflight(
    config: dict, project_dir: Path | None, host: str, port: int
) -> tuple[list[str], list[str]]:
    """Run fast, synchronous, zero-network pre-flight probes before the server binds.

    Each probe appends its findings to one shared failures/warnings pair so
    later probes bolt on without reworking this orchestrator. Returns
    ``([], [])`` on a clean pass. Failures abort the launch; warnings are
    printed but don't (e.g. a direct-Anthropic provider with no
    ``ANTHROPIC_API_KEY`` in env — subscription/OAuth login is still
    launchable).

    ``config``/``host``/``port`` are threaded through for probes that need
    them; none currently do. Probe 1 (companion port collisions) reads its own
    panel/port config directly; Probes 2-3 use ``project_dir``.
    """
    failures: list[str] = []
    warnings: list[str] = []
    failures.extend(_probe_companion_ports())
    auth_failures, auth_warnings = _probe_auth_secret(project_dir)
    failures.extend(auth_failures)
    warnings.extend(auth_warnings)
    failures.extend(_probe_config_validity(project_dir))
    return failures, warnings


def _wait_for_server(host: str, port: int, proc: subprocess.Popen, timeout: float = 10.0) -> bool:
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


def _notice_declared_override(env_var: str, flag_name: str, flag_value: object, what: str) -> None:
    """Print the NOTICE when a declared env var overrides a conflicting CLI flag.

    Only the operator-facing message lives here — the declaration-wins
    precedence itself is enforced by ``resolve_bind_host``/``resolve_web_port``
    (C3), which run regardless of whether this notice fires.
    """
    declared = os.environ.get(env_var)
    if declared and flag_value is not None and str(flag_value) != declared:
        click.echo(
            f"NOTICE: {env_var}={declared} is authoritative for the "
            f"multi-user reverse-proxy {what}; ignoring {flag_name} {flag_value}.",
            err=True,
        )


def _resolve_web_shell_command(
    cc_config: dict, shell_override: str | None, wt_config: dict
) -> list[str]:
    """Resolve the argv the Web Terminal spawns in each PTY.

    Precedence (highest first):
      1. ``--shell`` CLI flag (user-explicit; defeats the pin)
      2. ``web_terminal.shell`` config field (also defeats the pin)
      3. ``claude_code.cli_version`` pin via ``build_claude_launch_argv()``
      4. bare ``claude`` (current default)

    For the default (bare ``claude``) case, ``claude`` is resolved to an
    absolute path so a stripped PATH (systemd unit / container entrypoint) still
    finds it, while the launcher's appended flags — notably
    ``--setting-sources project`` — are preserved. A pinned ``npx …`` prefix is
    left to PATH lookup unchanged. Always returns ``list[str]`` so downstream
    consumers can unpack safely.
    """
    from osprey.utils.claude_launcher import build_claude_launch_argv
    from osprey.utils.shell_resolver import resolve_shell_command

    if shell_override:
        return [resolve_shell_command(shell_override)]
    if wt_config.get("shell"):
        return [resolve_shell_command(wt_config["shell"])]
    argv = build_claude_launch_argv(cc_config)
    if argv[0] == "claude":
        return [resolve_shell_command(argv[0]), *argv[1:]]
    return argv  # pinned ["npx", "-y", ...] — leave to PATH lookup


# -- CLI -------------------------------------------------------------------


@click.group("web", invoke_without_command=True)
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    envvar="OSPREY_WEB_PORT",
    help="Port to run on (default: from config or 8087)",
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
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip pre-flight checks (companion port collisions, etc.) and launch directly.",
)
@click.pass_context
def web(
    ctx: click.Context,
    port: int | None,
    host: str | None,
    reload: bool,
    shell: str | None,
    project: str | None,
    detach: bool,
    skip_preflight: bool,
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

    _preflight_vendor_check()

    wt_config = get_config_value("web_terminal", {})
    cc_config = get_config_value("claude_code", {})
    _notice_declared_override(DECLARED_BIND_ENV, "--host", host, "chokepoint")
    host = resolve_bind_host(host, wt_config.get("host"))
    _notice_declared_override(DECLARED_WEB_PORT_ENV, "--port", port, "port mapping")
    # An explicitly chosen port must never be silently reassigned: a DECLARED
    # port (multi-user compose — MUST match nginx's per-user upstream) or an
    # explicit --port / OSPREY_WEB_PORT is authoritative. Only an unspecified
    # port (config default or the 8087 fallback) may auto-move off a busy port.
    port_pinned = os.environ.get(DECLARED_WEB_PORT_ENV) is not None or port is not None
    port = resolve_web_port(port, wt_config.get("port"))

    user_shell_override = shell  # keep raw click value for the detached re-spawn
    try:
        shell_command = _resolve_web_shell_command(cc_config, user_shell_override, wt_config)
    except FileNotFoundError as e:
        click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1) from e

    if not skip_preflight:
        from osprey.utils.workspace import load_osprey_config

        project_dir = Path(project).resolve() if project else Path.cwd()
        failures, warnings = _preflight(load_osprey_config(), project_dir, host, port)
        for warning in warnings:
            click.echo(f"WARNING: {warning}", err=True)
        if failures:
            click.echo("ERROR: pre-flight checks failed:", err=True)
            for finding in failures:
                click.echo(f"  - {finding}", err=True)
            click.echo("\nRun with --skip-preflight to bypass (not recommended).", err=True)
            raise SystemExit(1)

    if detach:
        _start_detached(host, port, user_shell_override, project)
        return

    # -- foreground (original behavior) ------------------------------------

    if host == "0.0.0.0":
        click.echo("WARNING: Binding to 0.0.0.0 exposes the terminal to the network.")
        click.echo("This is a single-user tool — add authentication before external exposure.\n")

    # Pre-flight: check if port is already in use. SO_REUSEADDR matches
    # uvicorn's own bind semantics — without it, TIME_WAIT sockets from a
    # just-killed server fail this check for ~60s even though uvicorn
    # itself would bind fine.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError as exc:
            if port_pinned:
                click.echo(f"ERROR: Port {port} is already in use.", err=True)
                click.echo(f"  Find the process:  lsof -i :{port}", err=True)
                click.echo(f"  Or use another:    osprey web --port {port + 1}", err=True)
                raise SystemExit(1) from exc
            # Port was left unspecified and the default is taken — let the OS
            # assign a free one instead of hard-failing (single-user QoL). A
            # pinned port never reaches here, so nginx's per-user routing table
            # cannot be desynced by a silent move.
            busy_port = port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as free_sock:
                free_sock.bind((host, 0))
                port = free_sock.getsockname()[1]
            click.echo(f"Port {busy_port} in use — using :{port} instead.")

    # Publish the ACTUAL port to every child process (PTY shells, their MCP
    # servers): web_terminal_url() resolves OSPREY_WEB_PORT first, and
    # without this, panel tools (switch_panel etc.) fire-and-forget their
    # focus POSTs at the config default (8087) whenever --port differs —
    # reporting success while the real terminal never hears the event.
    os.environ["OSPREY_WEB_PORT"] = str(port)

    click.echo(f"Starting OSPREY Web Terminal on http://{host}:{port}")
    click.echo(f"Shell: {' '.join(shell_command)}")
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

            run_web(host=host, port=port, shell_command=shell_command, project_dir=project)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")


def _start_detached(host: str, port: int, shell: str | None, project: str | None) -> None:
    """Spawn the web server as a background process.

    ``shell`` is the raw user ``--shell`` flag (or None), NOT the resolved argv.
    The child re-derives the shell-command precedence so any ``claude_code.cli_version``
    pin remains honored from config; if we forwarded a resolved/pinned argv here,
    it would re-enter ``resolve_shell_command()`` in the child and fail for
    multi-word forms like ``npx -y @anthropic-ai/claude-code@<v>``.
    """
    project_dir = Path(project).resolve() if project else Path.cwd()

    # Idempotent: if already running, just report
    existing = _read_pid(project_dir)
    if existing is not None:
        click.echo(f"Web terminal already running (PID {existing}).")
        click.echo("  Stop with: osprey web stop")
        return

    # Build the child command (no --detach to avoid recursion). --skip-preflight
    # is always appended: the parent already ran the pre-flight in the foreground
    # process above, and a child-side failure would only reach the log file, not
    # the terminal.
    cmd = [
        sys.executable,
        "-m",
        "osprey.cli.main",
        "web",
        "--host",
        host,
        "--port",
        str(port),
        "--skip-preflight",
    ]
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
