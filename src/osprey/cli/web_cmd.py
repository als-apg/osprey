"""CLI command for the OSPREY Web Terminal.

Provides `osprey web` to launch a browser-based split-pane interface
with a real terminal (PTY) and live workspace file viewer.

Supports `--detach` for background operation and `osprey web stop`
to shut down a detached instance.

What it serves is ``build/`` — the render, which is where ``config.yml``,
``.mcp.json`` and the ``.claude/`` tree live — found by the one discovery rule
every repo-scoped verb uses: walk up to the nearest ``profile.yml``. ``stop``
runs the *same* walk as the start, so the two can never disagree about which
server is being stopped.

``OSPREY_CONFIG`` is written here only as the publication this process makes
for its own children (PTY shells, their MCP servers, the ``--reload`` worker).
It is never read as a way of *finding* the deployment: an ambient export from
whichever project the operator last worked in must not decide what this
command serves.
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

from osprey.utils.workspace import STATE_DIR_NAME

from . import output
from .repo_resolver import find_repo_root, repo_option

#: The detached server's PID and log files, relative to the repo root and in
#: the STATE zone. Three properties of ``var/`` are all required here and no
#: other zone has them: it is git-ignored (a running server must not dirty
#: ``git status``), it is host-local (a PID means nothing on another machine),
#: and it survives ``osprey build``, which wipes ``build/`` — a rebuild
#: underneath a running server would otherwise take the only handle on it.
#: Unhidden, unlike the old repo-root spelling: hiding a file inside a
#: directory that is already private buys nothing and costs discoverability.
PID_FILE = f"{STATE_DIR_NAME}/osprey-web.pid"
LOG_FILE = f"{STATE_DIR_NAME}/osprey-web.log"
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


def _read_pid(repo_root: Path) -> int | None:
    """Read PID file and return the PID if the process is alive.

    Removes stale PID files automatically.
    """
    pid_path = repo_root / PID_FILE
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


def _write_pid(repo_root: Path, pid: int) -> None:
    """Write PID to file, creating the STATE zone if this repo has none yet."""
    pid_path = repo_root / PID_FILE
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid))


def _resolve_render(repo: Path | None) -> tuple[Path, Path, Path]:
    """Find the deployment repo, and make its render authoritative.

    One decision, made once and up front, that everything downstream derives
    from: the panel set, the companion servers, the provider wiring, the
    ``.claude/`` tree the PTY's agent reads, and the ``watch_dir`` the file
    viewer follows. The rule is the shared one — walk up from cwd (or from
    ``--repo``) to the nearest ``profile.yml`` — and the served directory is
    that repo's ``build/``, because the render *is* the project as far as every
    runtime consumer is concerned.

    The render becomes the working directory, so cwd-relative resolvers, the
    PTY shells, and the ``--reload`` worker all agree without being told twice.
    ``OSPREY_CONFIG`` is then published for this process and every child — a
    publication, not a lookup: nothing above reads it to decide *which*
    deployment to serve. The repo ``.env`` is loaded off that same publication
    (:func:`osprey.mcp_env.load_dotenv_from_project` walks up from the render
    to the repo root), so the ``get_config_value`` reads that follow can expand
    ``${VAR}`` placeholders against it.

    Returns:
        ``(repo_root, build_dir, config_path)``, all absolute.

    Raises:
        RepoNotFoundError: When no ``profile.yml`` encloses the search start.
        click.ClickException: When the repo has no render to serve.
    """
    from osprey.utils.workspace import BUILD_DIR_NAME, rendered_config_path

    repo_root = find_repo_root(repo)
    build_dir = repo_root / BUILD_DIR_NAME
    config_path = rendered_config_path(repo_root)
    if not config_path.is_file():
        raise click.ClickException(
            f"no build found at {build_dir} — run `osprey build` first.\n\n"
            "The web terminal serves the rendered deployment, and there is "
            "nothing rendered yet. Without it the terminal would come up with "
            "no panels, no companion servers and no provider."
        )

    os.chdir(build_dir)
    os.environ["OSPREY_CONFIG"] = str(config_path)

    from osprey.mcp_env import load_dotenv_from_project

    load_dotenv_from_project()
    return repo_root, build_dir, config_path


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

    listed = [str(p) for p in problems[:5]]
    if len(problems) > 5:
        listed.append(f"and {len(problems) - 5} more")
    output.fail(
        "Offline mode is on but vendor assets are missing or corrupt",
        "\n".join(listed),
        "fetch them with: uv run osprey vendor fetch",
    )
    raise SystemExit(1)


def _probe_companion_ports() -> list[str]:
    """Probe 1: TCP-connect-probe every companion panel port the lifespan will bind.

    Resolves the panel set the same way ``_create_lifespan`` does: enabled via
    ``web.panels`` (or a UNIVERSAL panel, which is always launched) AND actually
    launchable per ``auto_launch``/``require_section``. A panel that is
    enabled but not launched (e.g. ``channel_finder`` with an unmet
    ``require_section``) is excluded — its port is never probed.

    Panel ids come from each registry entry's own ``panel_id`` — the registry
    keys are a different namespace from the ids ``web.panels`` and the frontend
    use (``artifact``/``artifacts``, ``channel_finder``/``channel-finder``), and
    a local translation table here drifted from the health category's copy.

    A listener already bound to a companion port before we start ours is
    foreign: at best it steals the panel's tab, at worst it silently
    reverse-proxies another project's data into this UI. Zero network I/O
    beyond the local TCP connect probe itself — no server starts, no
    registry init, no LLM calls.
    """
    from osprey.infrastructure.server_launcher import (
        _launchers,
        _make_auto_launch_checker,
    )
    from osprey.interfaces.web_terminal.app import _load_panel_config
    from osprey.profiles.web_panels import UNIVERSAL_PANELS
    from osprey.registry.web import (
        FRAMEWORK_WEB_SERVERS,
        WebServerConfigDepthError,
        resolve_web_server_address,
    )

    enabled_panels, _custom_panels, _default_panel = _load_panel_config()

    failures: list[str] = []
    for key, defn in FRAMEWORK_WEB_SERVERS.items():
        if defn.panel_id not in UNIVERSAL_PANELS and defn.panel_id not in enabled_panels:
            continue  # panel disabled in web.panels — the lifespan never calls its launcher
        try:
            if not _make_auto_launch_checker(defn)():
                continue  # auto_launch off, or require_section unmet
            host, port = resolve_web_server_address(key)
        except WebServerConfigDepthError as exc:
            # A misplaced host/port/auto_launch key is a config defect, not a
            # port clash — report it here rather than letting it traceback out
            # of pre-flight, so `osprey web` names the key and the fix.
            failures.append(str(exc))
            continue
        if _launchers[key]._port_has_listener(host, port):
            failures.append(
                f"Companion panel '{key}' ({defn.name}) port {port} is already in use "
                "by another process.\n"
                f"  Find the process:  lsof -i :{port}"
            )
    return failures


def _probe_auth_secret(build_dir: Path, repo_root: Path) -> tuple[list[str], list[str]]:
    """Probe 2: the resolved provider's auth secret must be resolvable before launch.

    A proxy provider (als-apg, cborg, a custom ``api.providers`` entry, ...)
    that can't authenticate upstream is a hard failure — the terminal would
    launch straight into an auth error. Direct Anthropic (subscription/OAuth)
    has no such requirement, so a missing ``ANTHROPIC_API_KEY`` there is only
    a warning, not an abort.

    The two directories are genuinely different files: the provider is declared
    in the render (``build/config.yml``), while the secret it needs lives in
    the repo's ``.env`` — the SECRETS zone, deliberately outside the render so
    it survives ``osprey build`` wiping ``build/``.

    Checks both ``os.environ`` and that ``.env`` (via ``dotenv_values``, which
    reads without mutating ``os.environ``). ``_resolve_render()`` does load
    ``.env`` before pre-flight runs, but this probe must not DEPEND on that
    side effect: reading ``.env`` directly keeps it correct on its own, so a
    secret that lives only in ``.env`` counts as present regardless of load
    ordering — otherwise a healthy proxy launch could false-fail.

    Zero network: ``load_provider_spec`` is a pure config read. A missing or
    malformed config.yml, or an unknown provider name, is left for Probe 3 (or
    the launch itself) to report — this probe just skips quietly rather than
    duplicating that diagnosis.

    The one exception is
    :class:`~osprey.build.claude_code_telemetry.ObservabilityCredentialError`:
    nothing else in pre-flight resolves telemetry credentials, so a quiet skip
    there leaves the operator with an empty report and no reason for it. That
    case returns a warning naming what stopped the read, which keeps the launch
    going (telemetry credentials are orthogonal to whether the terminal can
    authenticate) while saying why the auth check never ran.
    """
    from osprey.build.claude_code_resolver import load_provider_spec
    from osprey.build.claude_code_telemetry import ObservabilityCredentialError

    try:
        # The provider is declared in the render and its ``${VAR}`` references
        # resolve from the repo's ``.env`` — the two directories this probe
        # already holds apart. Without ``env_dir`` the expansion sees only
        # ``os.environ``, so a custom provider whose ``base_url``/``api_key``
        # lives in ``.env`` resolves to a literal placeholder and the probe
        # reports on a provider the launch will not use.
        spec = load_provider_spec(build_dir, env_dir=repo_root)
    except ObservabilityCredentialError as exc:
        # Must come BEFORE the ValueError arm: this type subclasses
        # TelemetryConfigError, which subclasses ValueError, so the broad arm
        # would swallow it and the operator would see an empty pre-flight with
        # no reason for it. The provider was never read, so there is nothing to
        # say about auth; report what stopped the read instead of nothing. Only
        # names reach the message, never a credential's value.
        return [], [
            "provider auth check skipped: the telemetry credentials in this "
            "deployment could not be resolved, so the provider was never read.\n"
            f"  {exc}"
        ]
    except (OSError, ValueError):
        return [], []
    if spec is None or not spec.auth_secret_env:
        return [], []

    secret_present = bool(os.environ.get(spec.auth_secret_env))
    if not secret_present:
        env_file = repo_root / ".env"
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


def _probe_config_validity(build_dir: Path, config_path: Path) -> list[str]:
    """Probe 3: config.yml and .claude/settings.json must at least parse.

    ``load_osprey_config()`` swallows every exception and returns ``{}`` on
    malformed YAML (see ``osprey.utils.workspace.load_osprey_config``), which
    would otherwise let the launch silently proceed on defaults instead of the
    project's actual configuration. This probe does its own dedicated parse of
    each file so a syntax error surfaces as a pre-flight failure instead of an
    inexplicable defaults-instead-of-config bug after launch. Deliberately
    does not call ``validate_agent_tools_against_permissions()`` — agent-tool
    / permission drift is a build-time concern, not a launch gate.

    ``config_path`` is the path ``_resolve_render()`` already settled on —
    threaded through rather than re-derived, so pre-flight can never parse a
    different file than the one the server will actually load.
    ``settings.json`` is optional; a render without one is not a failure, just
    nothing to validate.
    """
    failures: list[str] = []

    settings_path = build_dir / ".claude" / "settings.json"
    if settings_path.exists():
        import json

        try:
            json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            failures.append(f"{settings_path}: invalid JSON ({e})")

    if config_path.exists():
        import yaml

        try:
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            failures.append(f"{config_path}: invalid YAML ({e})")

    return failures


def _preflight(
    config: dict, repo_root: Path, build_dir: Path, config_path: Path, host: str, port: int
) -> tuple[list[str], list[str]]:
    """Run fast, synchronous, zero-network pre-flight probes before the server binds.

    Each probe appends its findings to one shared failures/warnings pair so
    later probes bolt on without reworking this orchestrator. Returns
    ``([], [])`` on a clean pass. Failures abort the launch; warnings are
    printed but don't (e.g. a direct-Anthropic provider with no
    ``ANTHROPIC_API_KEY`` in env — subscription/OAuth login is still
    launchable).

    ``repo_root``/``build_dir``/``config_path`` are what ``_resolve_render()``
    settled on — every probe sees the SAME deployment the server will serve.
    ``config``/``host``/``port`` are threaded through for probes that need
    them; none currently do. Probe 1 (companion port collisions) reads its own
    panel/port config directly; Probes 2-3 use the resolved paths.
    """
    failures: list[str] = []
    warnings: list[str] = []
    failures.extend(_probe_companion_ports())
    auth_failures, auth_warnings = _probe_auth_secret(build_dir, repo_root)
    failures.extend(auth_failures)
    warnings.extend(auth_warnings)
    failures.extend(_probe_config_validity(build_dir, config_path))
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
        output.warn(
            f"{env_var}={declared} is authoritative for the multi-user reverse-proxy {what}",
            f"Ignoring {flag_name} {flag_value}.",
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
@repo_option
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
    repo: Path | None,
    detach: bool,
    skip_preflight: bool,
) -> None:
    """Launch the OSPREY Web Terminal interface.

    Starts a FastAPI server with a split-pane UI: a real terminal (PTY) on the
    left and a live workspace file viewer on the right.

    Serves the deployment enclosing the current directory — the nearest
    profile.yml above it, or --repo — and what it serves is that repo's build/
    as it was last rendered. A repo with no build/ is refused rather than
    served as an empty terminal.

    Example:

    \b
        osprey web                         # Start on localhost:8087
        osprey web --port 9000             # Custom port
        osprey web --host 0.0.0.0          # Bind to all interfaces
        osprey web --shell zsh             # Use zsh instead of claude
        osprey web --reload                # Development mode
        osprey web --detach                # Start in background
        osprey web --repo ~/als-assistant  # Serve another deployment
        osprey web stop                    # Stop background server
    """
    if ctx.invoked_subcommand is not None:
        return

    # Resolve the deployment FIRST and fail loudly if there is none. Everything
    # below — the vendor check's offline flag, host/port/shell resolution,
    # pre-flight's probes, and the server itself — reads the render's config,
    # so an unresolvable one must abort here rather than degrade into a mystery
    # terminal serving another directory's defaults.
    repo_root, build_dir, project_config = _resolve_render(repo)
    output.section("", {"Repo": repo_root, "Build": build_dir})

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
        output.fail("Cannot resolve the shell the terminal should run", str(e))
        raise SystemExit(1) from e

    if not skip_preflight:
        from osprey.utils.workspace import load_osprey_config

        failures, warnings = _preflight(
            load_osprey_config(), repo_root, build_dir, project_config, host, port
        )
        for warning in warnings:
            output.warn(warning)
        if failures:
            output.fail(
                "Pre-flight checks failed",
                "\n".join(f"- {finding}" for finding in failures),
                "fix the findings above, or pass --skip-preflight to start anyway",
            )
            raise SystemExit(1)

    if detach:
        _start_detached(host, port, user_shell_override, repo_root)
        return

    # -- foreground (original behavior) ------------------------------------

    if host == "0.0.0.0":
        output.warn(
            "Binding to 0.0.0.0 exposes the terminal to the network",
            "This is a single-user tool. Add authentication before you expose it.",
        )

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
                output.fail(
                    f"Port {port} is already in use",
                    f"Find the process holding it with: lsof -i :{port}",
                    f"or start on another port with: osprey web --port {port + 1}",
                )
                raise SystemExit(1) from exc
            # Port was left unspecified and the default is taken — let the OS
            # assign a free one instead of hard-failing (single-user QoL). A
            # pinned port never reaches here, so nginx's per-user routing table
            # cannot be desynced by a silent move.
            busy_port = port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as free_sock:
                free_sock.bind((host, 0))
                port = free_sock.getsockname()[1]
            output.warn(f"Port {busy_port} is in use, so this is on port {port} instead")

    # Publish the ACTUAL port to every child process (PTY shells, their MCP
    # servers): web_terminal_url() resolves OSPREY_WEB_PORT first, and
    # without this, panel tools (open_panel etc.) fire-and-forget their
    # focus POSTs at the config default (8087) whenever --port differs —
    # reporting success while the real terminal never hears the event.
    os.environ["OSPREY_WEB_PORT"] = str(port)

    output.report(f"Starting OSPREY Web Terminal on http://{host}:{port}")
    output.note(f"Shell: {' '.join(shell_command)}")
    output.note("Press Ctrl+C to stop")
    output.report("")

    try:
        if reload:
            import uvicorn

            from osprey.interfaces.web_terminal.app import _open_browser_when_ready

            _open_browser_when_ready(f"http://{host}:{port}")

            # The reload worker re-imports create_app() with no arguments; it
            # finds the render via the OSPREY_CONFIG publication made in
            # _resolve_render().
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

            run_web(
                host=host,
                port=port,
                shell_command=shell_command,
                config_path=str(project_config),
                project_dir=str(build_dir),
            )
    except KeyboardInterrupt:
        output.report("")
        output.report("Shutting down...")


def _start_detached(host: str, port: int, shell: str | None, repo_root: Path) -> None:
    """Spawn the web server as a background process.

    ``shell`` is the raw user ``--shell`` flag (or None), NOT the resolved argv.
    The child re-derives the shell-command precedence so any ``claude_code.cli_version``
    pin remains honored from config; if we forwarded a resolved/pinned argv here,
    it would re-enter ``resolve_shell_command()`` in the child and fail for
    multi-word forms like ``npx -y @anthropic-ai/claude-code@<v>``.

    ``repo_root`` is the repo ``_resolve_render()`` settled on — always present,
    never re-derived from cwd here, which matters more than ever now that the
    parent has already chdir'ed into the render.
    """
    # Idempotent: if already running, just report
    existing = _read_pid(repo_root)
    if existing is not None:
        output.report(f"Web terminal already running (PID {existing}).")
        output.note("Stop it with: osprey web stop")
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
    # Always name the repo in the child argv: it's what `ps` shows and what
    # humans and agents copy to restart the server. A restart from another
    # directory must not silently lose the deployment identity — and the child
    # would otherwise inherit the parent's cwd, which is the render, not a repo.
    cmd += ["--repo", str(repo_root)]

    log_path = repo_root / LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_fh.close()  # child retains its own fd copy

    _write_pid(repo_root, proc.pid)

    if _wait_for_server(host, port, proc):
        output.report(f"Web terminal started (PID {proc.pid}).")
        output.section(
            "",
            {"URL": f"http://{host}:{port}", "Log": log_path, "Stop": "osprey web stop"},
        )
    else:
        exit_code = proc.poll()
        if exit_code is not None:
            output.fail(
                f"The web terminal exited immediately with code {exit_code}",
                None,
                f"read what it wrote to {log_path}",
            )
            (repo_root / PID_FILE).unlink(missing_ok=True)
        else:
            output.warn(
                f"The web terminal started with PID {proc.pid} "
                f"but is not answering on port {port} yet",
                f"Log:  {log_path}\nStop: osprey web stop",
            )


@web.command("stop")
@repo_option
@click.pass_context
def web_stop(ctx: click.Context, repo: Path | None) -> None:
    """Stop a background web terminal server.

    Finds the deployment exactly the way starting one does — the nearest
    profile.yml at or above the current directory, or --repo — so the server
    this stops is always the server that directory's `osprey web --detach`
    started.
    """
    # `osprey web --repo X stop` and `osprey web stop --repo X` are the same
    # request. The group parses --repo before the subcommand name, so without
    # this fallback the first form would silently stop whatever server the
    # current directory happens to enclose.
    if repo is None and ctx.parent is not None:
        repo = ctx.parent.params.get("repo")

    repo_root = find_repo_root(repo)
    pid_path = repo_root / PID_FILE
    log_path = repo_root / LOG_FILE

    if not pid_path.exists():
        output.report("No running web terminal found. There is no PID file.")
        return

    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        output.warn("Removing a corrupt PID file", str(pid_path))
        pid_path.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
        output.report(f"Stopped web terminal (PID {pid}).")
    except ProcessLookupError:
        output.report(f"Process {pid} not found, so it had already stopped. Cleaning up.")
    except PermissionError:
        output.fail(f"Permission denied stopping the web terminal (PID {pid})")
        return

    pid_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
