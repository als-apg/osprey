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

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click

from osprey.port_layout import default_port, resolve_port_base
from osprey.utils.workspace import STATE_DIR_NAME, agent_data_base_dir, anchored_path

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
#: The carrier for the terminal session lifetime. Spelled literally rather than
#: imported at module scope because this module keeps its ``osprey`` imports
#: function-local (see the lazy-loading CLI convention); it MUST equal
#: :data:`osprey.interfaces.web_auth.SESSION_LIFETIME_ENV`, which is the name
#: the serving process reads. ``test_web_session_lifetime.py`` pins the two
#: spellings equal so this copy cannot drift.
DECLARED_SESSION_LIFETIME_ENV = "OSPREY_TERMINAL_SESSION_LIFETIME"
#: The carrier for the DIRECTORY holding the terminal's browser-session store.
#: Spelled literally for the same reason :data:`DECLARED_SESSION_LIFETIME_ENV`
#: is — this module keeps its ``osprey`` imports function-local — and it MUST
#: equal :data:`osprey.interfaces.web_auth.SESSION_STORE_DIR_ENV`, which is the
#: name the serving process reads. ``test_web_session_lifetime.py`` pins the two
#: spellings equal so this copy cannot drift.
DECLARED_SESSION_STORE_DIR_ENV = "OSPREY_TERMINAL_SESSION_STORE_DIR"
#: The carrier for THIS user's audit directory. The multi-user compose sets it
#: on every per-user web container (``docker-compose.web.yml.j2``) alongside the
#: ``var/audit/<identity>`` bind mount it names, so the value is a real,
#: host-visible, group-writable directory inside the container. Spelled
#: literally for the same reason the carriers above are — this module keeps its
#: ``osprey`` imports function-local — and deliberately read, never written,
#: here: ``osprey web`` is a consumer of the deployment's declaration.
AUDIT_DIR_ENV = "OSPREY_AUDIT_DIR"
#: File name of the pre-flight refusal marker inside :data:`AUDIT_DIR_ENV`.
#: Exported so ``osprey status`` reads the same name this writes instead of
#: duplicating the literal; the format is deliberately trivial (line 1 = an
#: ISO-8601 UTC timestamp, the rest = the refusal findings verbatim) so the
#: reader needs no parser and no schema version.
PREFLIGHT_REFUSED_MARKER = "preflight-refused"

_LOGGER = logging.getLogger(__name__)


def _preflight_marker_path() -> Path | None:
    """Where this container's pre-flight refusal marker lives, or ``None``.

    ``None`` means "nothing to record": no :data:`AUDIT_DIR_ENV` is declared,
    which is the bare-laptop launch — no container, no supervisor restarting
    the process, and so nobody downstream to explain a refusal to. Under
    ``restart: unless-stopped`` the variable is always set, because the compose
    that supervises the container is the same file that declares it.
    """
    declared = os.environ.get(AUDIT_DIR_ENV)
    if not declared:
        return None
    return Path(declared) / PREFLIGHT_REFUSED_MARKER


def _refusal_body(failures: list[str]) -> str:
    """The refusal findings as the one bullet list both the report and the marker carry."""
    return "\n".join(f"- {finding}" for finding in failures)


def _write_preflight_marker(body: str) -> None:
    """Record a pre-flight refusal for ``osprey status`` to render.

    A supervised container that refuses pre-flight exits 1 and is restarted, so
    the refusal text scrolls past in a log nobody is tailing and the only
    outward sign is a service flapping. This marker is the durable half of that
    story: ``osprey status`` turns it into a "restarting (pre-flight: ...)" row.

    Rewritten in full on every attempt rather than appended to — each restart
    re-runs pre-flight, and the operator needs to know what is failing *now*,
    not the archaeology of every attempt since the deploy.

    Strictly advisory. An audit directory that is absent, root-owned or on a
    read-only mount must not turn one honest failure into a second, more
    confusing one, so every filesystem error here is logged and swallowed; the
    caller goes on to report the refusal and exit as it would have anyway.
    """
    marker = _preflight_marker_path()
    if marker is None:
        _LOGGER.debug("no %s declared; skipping pre-flight refusal marker", AUDIT_DIR_ENV)
        return

    from datetime import UTC, datetime

    # `osprey status` reads this file from the host while a supervised container
    # rewrites it on every restart attempt; swap it in whole so a concurrent
    # reader never sees a truncated timestamp line.
    staging = marker.with_name(marker.name + ".tmp")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(f"{datetime.now(UTC).isoformat()}\n{body}\n", encoding="utf-8")
        os.replace(staging, marker)
    except OSError as e:
        _LOGGER.debug("could not write pre-flight refusal marker %s: %s", marker, e)


def _clear_preflight_marker() -> None:
    """Remove a marker left by an earlier refusal, now that pre-flight passed.

    Called ONLY on a real pass. ``--skip-preflight`` deliberately leaves any
    marker standing: it forces past checks it never re-ran, so clearing there
    would erase the record of the very refusal the operator is overriding while
    the underlying fault is still present. The staleness that costs — a marker
    outliving the refusal it describes — is bounded on the reading side, where
    ``osprey status`` ignores markers older than the container's ``StartedAt``.

    Swallows filesystem errors for the same reason the writer does.
    """
    marker = _preflight_marker_path()
    if marker is None:
        return
    try:
        marker.unlink(missing_ok=True)
    except OSError as e:
        _LOGGER.debug("could not clear pre-flight refusal marker %s: %s", marker, e)


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
    cli_port: int | None,
    config_port: int | None,
    *,
    base: int,
    env: Mapping[str, str] = os.environ,
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
    config, or the layout's ``web`` slot) is honored verbatim.

    ``OSPREY_TERMINAL_WEB_PORT`` is a DECLARATION set by the compose overlay
    for THIS container only — it is never re-exported to children, unlike
    the child-facing ``OSPREY_WEB_PORT`` publication at the bottom of
    ``web()``. Do NOT collapse this into ``@click.option("--port",
    envvar=...)``: Click env defaults LOSE to an explicit flag, which is the
    opposite of "declared wins" this function exists to provide — the same
    reasoning that keeps ``resolve_bind_host`` a plain function rather than
    a click envvar.

    Args:
        cli_port: The port ``--port`` (or its ``OSPREY_WEB_PORT`` envvar
            fallback) asked for, or ``None`` when unspecified.
        config_port: ``web_terminal.port`` from the rendered config, or
            ``None`` when the deployment sets none.
        base: The port base this deployment resolved, from
            :func:`~osprey.port_layout.resolve_port_base`. Keyword-only and
            without a default on purpose: the terminal fallback is the layout's
            ``web`` slot at *this* deployment's base, so a caller that can
            reach the config must hand the resolved base down rather than let
            the layout fall back to its own default.
        env: Environment to read the declaration from. Defaults to the real
            one; tests pass a mapping.

    Returns:
        The declared port when one is declared, else the first of ``cli_port``,
        ``config_port`` and the layout's ``web`` port at ``base``.
    """
    declared = env.get(DECLARED_WEB_PORT_ENV)
    if declared:
        return int(declared)
    return cli_port or config_port or default_port("web", 0, base=base)


def _refuse_session_lifetime(value: object, source: str) -> click.ClickException:
    """The one refusal message for a session lifetime that cannot be honoured.

    Names the offending SOURCE as well as the value: the same key reaches this
    launcher from two places — a deploy-time environment declaration and the
    render's own ``config.yml`` — and an operator staring at one of them cannot
    fix a message that only quotes the other.
    """
    return click.ClickException(
        f"modules.web_terminals.auth.session_lifetime must be a whole number of "
        f"seconds greater than zero — {source} says {value!r}.\n\n"
        "That value is the Max-Age stamped on every terminal session cookie, so "
        "there is no honest way to read it as a duration. Starting anyway would "
        "silently fall back to the 12-hour default while the deployment believed "
        "it had set its own lifetime, which is exactly the mistake worth failing "
        "on. Set it to a positive number of seconds and start again."
    )


def _session_seconds_from_env(text: str, source: str) -> int:
    """Coerce the environment carrier's text to a positive int, or refuse.

    Kept apart from :func:`_session_seconds_from_config` because the two sources
    are read differently on purpose: an environment variable is text by nature,
    so it is stripped and read as base 10, while the config value has to be a
    real YAML int.
    """
    try:
        seconds = int(text.strip(), 10)
    except ValueError:
        raise _refuse_session_lifetime(text, source) from None
    if seconds <= 0:
        raise _refuse_session_lifetime(text, source)
    return seconds


def _session_seconds_from_config(value: object, source: str) -> int:
    """Coerce a configured lifetime to a positive int, or refuse.

    Unlike the environment carrier, the config value must be a REAL YAML int:
    the multi-user render lint (``_check_auth_session_lifetime``) reports any
    other type as an ERROR, so accepting a quoted ``"3600"`` here would let
    ``osprey web`` start happily on a config that ``osprey build`` refuses — and
    the two deployment shapes have to agree about which configs are valid.

    ``bool`` is excluded even though ``True`` is an ``int`` in Python: it is
    never a duration.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _refuse_session_lifetime(value, source)
    return value


def resolve_session_lifetime(config: Mapping[str, Any], env: Mapping[str, str] = os.environ) -> int:
    """Single source of the terminal session lifetime, in seconds.

    Precedence is env > config > default, and the environment wins for the same
    reason it wins in :func:`resolve_bind_host` and :func:`resolve_web_port`: a
    per-user terminal container bakes its ``config.yml`` into the image, so the
    only spelling that can track a deploy-time edit is the
    ``OSPREY_TERMINAL_SESSION_LIFETIME`` the multi-user compose sets on every
    ``web-*`` service. Single-user ``osprey web`` sets no such env and reads
    ``modules.web_terminals.auth.session_lifetime`` out of its own render.

    Deliberately a plain function rather than a ``@click.option(envvar=...)``,
    for the reason spelled out in :func:`resolve_bind_host`: a Click env default
    LOSES to an explicit flag, which is the opposite of the declaration-wins
    precedence this exists to provide.

    A value that is PRESENT but not a positive whole number refuses the launch —
    and for the config source that means a real YAML ``int``, matching what the
    multi-user render lint accepts, so a config cannot be one the launcher
    starts on and the build refuses.
    :func:`osprey.interfaces.web_auth._session_ttl_from_env` refuses too, but it
    runs at credential population inside the server — far too late for a message
    an operator can act on. Validating here means the launcher publishes a value
    the server can only agree with.

    Args:
        config: The rendered deployment config (top-level mapping). Every level
            down to ``session_lifetime`` is read defensively; absent or ``None``
            anywhere along the path means nothing configured a lifetime.
        env: The environment to consult, for tests.

    Returns:
        The session lifetime in seconds.

    Raises:
        click.ClickException: When either source holds an unusable value.
    """
    from osprey.interfaces.web_auth import DEFAULT_SESSION_LIFETIME

    declared = env.get(DECLARED_SESSION_LIFETIME_ENV, "")
    if declared.strip():
        return _session_seconds_from_env(declared, DECLARED_SESSION_LIFETIME_ENV)

    section: Any = config
    for key in ("modules", "web_terminals", "auth"):
        section = section.get(key) if isinstance(section, Mapping) else None
    configured = section.get("session_lifetime") if isinstance(section, Mapping) else None
    if configured is None:
        return DEFAULT_SESSION_LIFETIME
    return _session_seconds_from_config(configured, "build/config.yml")


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


def _companion_roster_failure(
    key: str,
    name: str,
    family: str,
    port: int,
    base: int,
    section_key: str,
) -> str:
    """Word a companion-port clash this deployment's own multi-user roster caused.

    Single-user ``osprey web`` and roster user 0 both take index 0 of every
    port family, so a repo whose ``modules.web_terminals.enabled`` is true
    cannot run the two side by side at one base. That is a different diagnosis
    from a foreign listener — the port is not stolen, it is spoken for — and it
    has different remedies, so it gets its own wording rather than sending the
    operator to ``lsof`` to rediscover their own deployment.

    Args:
        key: Registry key of the companion server, e.g. ``"artifact"``.
        name: The server's display name.
        family: Port family the clash sits in — the registry key, or the
            definition's ``port_family`` when it names a different one.
        port: The port a listener was found on.
        base: The port base this deployment resolved.
        section_key: Dotted config key that overrides this server's port.

    Returns:
        The failure line, naming both escapes there are: the per-section
        ``port:`` override and ``osprey web --port``. Deliberately no second
        base knob — one deployment gets one block.
    """
    return (
        f"Companion panel '{key}' ({name}) port {port} is already in use: "
        "this deployment's multi-user roster (user 0) owns this port.\n"
        f"  modules.web_terminals.enabled is true in this repo, and {port} is the "
        f"'{family}' family's index-0 slot at port base {base} — single-user "
        "`osprey web` and roster user 0 share index 0, so the two cannot run "
        "side by side at one base.\n"
        f"  Move this panel:    set {section_key} in config.yml\n"
        "  Move the terminal:  osprey web --port <port> "
        "(its own index-0 slot clashes the same way)"
    )


def _probe_companion_ports() -> list[str]:
    """Probe 1: bind-probe every companion panel port the lifespan will bind.

    Resolves the panel set the same way ``_create_lifespan`` does: enabled via
    ``web.panels`` (or a UNIVERSAL panel, which is always launched) AND actually
    launchable per ``auto_launch``/``require_section``. A panel that is
    enabled but not launched (e.g. ``channel_finder`` with an unmet
    ``require_section``) is excluded — its port is never probed.

    Panel ids come from each registry entry's own ``panel_id`` — the registry
    keys are a different namespace from the ids ``web.panels`` and the frontend
    use (``artifact``/``artifacts``, ``channel_finder``/``channel-finder``), and
    a local translation table here drifted from the health category's copy.

    The verdict is bindability, asked by attempting the very bind the lifespan
    will attempt: a port we can bind is free, whatever else may answer there.
    Something already holding a companion port so that our bind fails is
    usually foreign: at best it steals the panel's tab, at worst it silently
    reverse-proxies another project's data into this UI. The one case it is
    NOT foreign is this deployment's own multi-user roster — see
    :func:`_companion_roster_failure` — which is why the probe resolves the
    base and reads ``modules.web_terminals.enabled`` before it words a
    failure. Zero network I/O beyond the local bind and connect probes
    themselves — no server starts, no registry init, no LLM calls.
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
        framework_web_port_default,
        resolve_web_server_address,
    )
    from osprey.utils.workspace import load_osprey_config

    enabled_panels, _custom_panels, _default_panel = _load_panel_config()

    # The render `_resolve_render()` settled on, read once and handed down:
    # both the base every index-0 slot is derived from and the roster flag the
    # attribution turns on come from THIS repo's config, never from an ambient
    # default. Passing it on also spares each server a reload — except when
    # nothing loaded, where the resolver's own no-config warning is worth more
    # than the saved read.
    config = load_osprey_config() or {}
    base = resolve_port_base(config)
    roster_enabled = bool(((config.get("modules") or {}).get("web_terminals") or {}).get("enabled"))

    failures: list[str] = []
    for key, defn in FRAMEWORK_WEB_SERVERS.items():
        if defn.panel_id not in UNIVERSAL_PANELS and defn.panel_id not in enabled_panels:
            continue  # panel disabled in web.panels — the lifespan never calls its launcher
        try:
            if not _make_auto_launch_checker(defn)():
                continue  # auto_launch off, or require_section unmet
            host, port = resolve_web_server_address(key, config or None)
        except WebServerConfigDepthError as exc:
            # A misplaced host/port/auto_launch key is a config defect, not a
            # port clash — report it here rather than letting it traceback out
            # of pre-flight, so `osprey web` names the key and the fix.
            failures.append(str(exc))
            continue
        launcher = _launchers[key]
        if launcher._port_is_bindable(host, port):
            # Bindable == free: the lifespan's own bind will succeed, so there
            # is no clash to report. Something may still answer a connect there
            # without contending for the bind (a Docker Desktop host-loopback
            # pass-through, where a Mac-side listener is visible to a connect
            # but does not block the container's bind). Name it so the operator
            # is not left wondering why `curl` answers and pre-flight is clean.
            if launcher._port_answers_connect(host, port):
                output.note(
                    f"Companion panel '{key}' port {port} answers a TCP connect but does not "
                    "block the bind. That is a foreign host-side listener, not an owner of "
                    "this port."
                )
            continue
        if roster_enabled and port == framework_web_port_default(key, base=base):
            failures.append(
                _companion_roster_failure(
                    key,
                    defn.name,
                    defn.port_family or key,
                    port,
                    base,
                    ".".join(
                        part for part in (defn.config_key, defn.config_web_subkey, "port") if part
                    ),
                )
            )
        else:
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
    a warning, not an abort. Nor is a key required by every proxy provider:
    when the models adapter registry knows the provider and its adapter
    declares ``requires_api_key = False`` (ollama, vllm, ds4 — local servers
    with no auth), a missing secret is likewise only a warning, worded with
    the adapter's own ``api_key_note``. Providers the registry does not know
    keep the strict behavior — an unknown custom proxy without a secret is
    still an abort.

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
        # Imported only on this failure path: resolving an adapter class pulls
        # in the LiteLLM stack, which the healthy launch never needs to load.
        from osprey.models.provider_registry import get_provider_registry

        adapter = get_provider_registry().get_provider(spec.provider)
        if adapter is not None and adapter.requires_api_key is False:
            note = f": {adapter.api_key_note}" if adapter.api_key_note else ""
            return [], [f"{preamble}(provider {spec.provider} does not require one{note})"]
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


def _mint_operator_url(host: str, port: int) -> tuple[str, bool]:
    """Settle this launcher's operator secret and return its login URL.

    Runs in the CLI PARENT — the process that is about to *become* the server
    (foreground), spawn the ``--reload`` worker, or spawn the ``--detach``
    child — never inside an already-serving process. Calling
    :func:`osprey.interfaces.web_auth.mint_and_announce` here mints the operator
    secret into the process-wide holder and, crucially, re-publishes it in
    ``os.environ[OSPREY_TERMINAL_SECRET]`` so whatever this launcher spawns or
    becomes inherits the SAME value. The reload worker re-imports the app
    factory in a fresh process, so the secret must be in the environment BEFORE
    that spawn — minting in the parent, not in the factory, is what makes the
    worker inherit it.

    The carrier this opens is closed again at app construction, wherever that
    construction happens: in a spawned worker or detached child by that
    process's own ``_populate`` pop, and on the direct-serve path — where this
    launcher becomes the server and no second population runs — by
    :func:`osprey.interfaces.web_auth.close_env_carriers`. Without that second
    mechanism the default ``osprey web`` would serve for its whole life with the
    operator secret sitting in the environment every SDK-spawned agent
    inherits.

    Returns:
        ``(login_url, announce)``. ``login_url`` carries the secret as its
        ``?token=`` and is the operator's only way in. ``announce`` is False
        when the secret was ALREADY in the environment before this call —
        supplied by an ancestor launcher (a ``--detach`` parent) or by a
        multi-user deployment — meaning the URL was already printed once
        upstream and this process must stay silent, so the token never lands in,
        e.g., the detached server's log file. It is True only when this process
        minted the secret itself and therefore owns announcing it.

    The environment read that decides ``announce`` happens BEFORE
    ``mint_and_announce`` re-sets the carrier, so a freshly minted secret reads
    as absent-then-present (announce) while an inherited one reads as
    already-present (stay silent).

    Raises:
        click.ClickException: when the credential holder refuses to settle —
            the container shape, where a bind host is declared but the
            deployment supplied no secret. The holder raises ``RuntimeError``
            there rather than minting a value nginx would never forward; this
            call site is outside ``web()``'s ``try`` (which handles only
            ``KeyboardInterrupt``), so without the translation the operator
            gets a traceback instead of the message that names the fix.
    """
    from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, mint_and_announce

    announce = not (os.environ.get(OPERATOR_SECRET_ENV) or "").strip()
    try:
        login_url = mint_and_announce(host, port)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    return login_url, announce


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
    help="Port to run on (default: from config, else the layout's web slot)",
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

    On launch it mints a single-user operator secret and prints an ``Open:``
    login URL carrying it as a ``?token=``; that URL is the only way into the
    terminal, which now refuses an unauthenticated request. The secret is minted
    in this parent process on every launch shape (foreground, ``--reload``,
    ``--detach``) so the serving worker or child inherits it through the
    environment — it is never written to disk, and under ``--detach`` the parent
    prints the URL once while the child stays silent so no token reaches the log.

    Example:

    \b
        osprey web                         # Start on the web slot (localhost:10100 at the default base)
        osprey web --port 9000             # Custom port
        osprey web --host 0.0.0.0          # Bind to all interfaces
        osprey web --shell zsh             # Use zsh instead of claude
        osprey web --reload                # Development mode
        osprey web --detach                # Start in background
        osprey web --repo ~/als-assistant  # Serve another deployment
        osprey web stop                    # Stop background server
        osprey web sessions clear          # Drop the persisted browser sessions
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

    # Resolve the session lifetime here, beside the other config reads and
    # before anything else is printed, then publish the ANSWER — the same
    # publication `OSPREY_WEB_PORT` gets at the bottom of this function, and for
    # the same reason. `_start_detached`'s child inherits `os.environ` (no `env=`
    # kwarg), so does the `--reload` worker, and the serving process reads this
    # carrier in `web_auth._session_ttl_from_env()`. That reader refuses a bad
    # value too, but it runs at credential population inside the server, which
    # is too late for a message the operator can act on; validating in the
    # launcher means every downstream reader sees a value already known good.
    os.environ[DECLARED_SESSION_LIFETIME_ENV] = str(
        resolve_session_lifetime({"modules": get_config_value("modules", {})})
    )

    # Publish the DIRECTORY the session store lives in, from the same config
    # read and for the same downstream readers. Deliberately the directory
    # only: the store file is named for the settled `OSPREY_WEB_PORT` (two
    # terminals on one host must not share a store), and the port is NOT
    # settled here — the busy-port auto-move happens further down in this
    # function. So the file name is resolved at credential population, in the
    # process that has already bound, from the port publication made below.
    #
    # `get_config_value` is the read the rest of `web()` uses (it is
    # `load_osprey_config()` behind a section lookup), and the anchor is the
    # `repo_root` `_resolve_render()` settled on rather than
    # `resolve_project_root(config)`: this command has already chdir'ed into
    # the render, and a `--repo` launch serves a deployment the ambient
    # project root does not name. Same reasoning as the file watcher's anchor
    # in `osprey.interfaces.web_terminal.app`.
    store_dir = (
        anchored_path(
            agent_data_base_dir({"agent_data": get_config_value("agent_data", {})}), repo_root
        )
        / "web_terminal"
    )
    os.environ[DECLARED_SESSION_STORE_DIR_ENV] = str(store_dir)

    _notice_declared_override(DECLARED_BIND_ENV, "--host", host, "chokepoint")
    host = resolve_bind_host(host, wt_config.get("host"))
    _notice_declared_override(DECLARED_WEB_PORT_ENV, "--port", port, "port mapping")
    # An explicitly chosen port must never be silently reassigned: a DECLARED
    # port (multi-user compose — MUST match nginx's per-user upstream) or an
    # explicit --port / OSPREY_WEB_PORT is authoritative. Only an unspecified
    # port (config default or the layout's web slot) may auto-move off a busy port.
    port_pinned = os.environ.get(DECLARED_WEB_PORT_ENV) is not None or port is not None
    # The base comes from the render this command just resolved, never from the
    # layout's own default: two deployments on one host differ only by their
    # ``deployment.port_base``, and a terminal that fell back to the module
    # default would land in the other deployment's block.
    port_base = resolve_port_base({"deployment": get_config_value("deployment", {})})
    port = resolve_web_port(port, wt_config.get("port"), base=port_base)

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
            # Record BEFORE the exit, not after the report: under container
            # supervision this process is about to be killed and restarted, and
            # the marker is the only thing that survives to tell `osprey status`
            # why the service is flapping.
            body = _refusal_body(failures)
            _write_preflight_marker(body)
            output.fail(
                "Pre-flight checks failed",
                body,
                "fix the findings above, or pass --skip-preflight to start anyway",
            )
            raise SystemExit(1)
        _clear_preflight_marker()

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
    # focus POSTs at the deployment's own default (the web slot) whenever
    # --port differs — reporting success while the real terminal never hears
    # the event.
    os.environ["OSPREY_WEB_PORT"] = str(port)

    # Mint the operator secret in THIS (parent) process, after the port has
    # settled so the announced URL names the socket that will actually listen.
    # In direct-serve this same process's create_app closes the carrier again
    # (configure_interface_app -> web_auth.close_env_carriers), so the secret
    # this line publishes does not survive into the environment the serving
    # process hands its agents; under --reload the ChangeReload worker uvicorn
    # spawns re-imports the factory and pops the value it inherited through the
    # environment set here — which is why the mint must precede the spawn and
    # live in the launcher, not the factory.
    login_url, announce = _mint_operator_url(host, port)

    output.report(f"Starting OSPREY Web Terminal on http://{host}:{port}")
    if announce:
        # The one line that carries the secret. Printed only when this process
        # minted it; a secret inherited from upstream was already announced
        # there, so re-printing would be the only place the token could leak.
        output.report(f"Open: {login_url}")
    elif not os.environ.get(DECLARED_BIND_ENV):
        # Silence needs an explanation on the host side. The gate above is a
        # bare presence check on OSPREY_TERMINAL_SECRET, so an operator who
        # exported that variable in their own shell gets no login URL and, until
        # this line, no hint why. Never echo the value — say where the URL is
        # and how to get a fresh one. Suppressed when a bind host is declared:
        # that is the multi-user container, where nginx owns the way in and
        # there is no login URL to be missing.
        from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV

        output.note(
            f"Using the {OPERATOR_SECRET_ENV} already set in this environment, "
            "so no login URL is printed here."
        )
        output.note(
            "Open the URL printed where that secret was minted, or unset "
            f"{OPERATOR_SECRET_ENV} and start again to have this launch mint "
            "and print its own."
        )
    output.note(f"Shell: {' '.join(shell_command)}")
    output.note("Press Ctrl+C to stop")
    output.report("")

    try:
        if reload:
            import uvicorn

            from osprey.interfaces.web_terminal.app import _open_browser_when_ready

            # Open the token URL, not the bare one: the bare URL sets no cookie,
            # so an auto-opened tab would land on the login-required page. The
            # ?token= exchange sets the session cookie and 303s to the clean URL.
            _open_browser_when_ready(login_url)

            # The reload worker re-imports create_app() with no arguments; it
            # finds the render via the OSPREY_CONFIG publication made in
            # _resolve_render(), and pops OSPREY_TERMINAL_SECRET — set by the
            # mint above, before this spawn — to recognise that same token. That
            # worker is a fresh process, so its own _populate does the pop;
            # close_env_carriers then keeps the carrier shut on every later app
            # the worker builds.
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

            # Hand run_web the TOKEN url so its auto-open exchanges the token for
            # a session cookie, matching the --reload branch. Without this it
            # opens the bare URL and the tab lands on the login-required page.
            run_web(
                host=host,
                port=port,
                shell_command=shell_command,
                config_path=str(project_config),
                project_dir=str(build_dir),
                browser_url=login_url,
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

    # Mint the operator secret in THIS (parent) process, BEFORE spawning the
    # child, so the child inherits OSPREY_TERMINAL_SECRET through the
    # environment (subprocess.Popen inherits os.environ) and pops it at its own
    # create_app. The token rides in the environment and this parent's memory
    # only — never in the child argv and never on disk. The child re-enters the
    # foreground path and finds the secret already set, so it stays silent and
    # nothing carrying the token is written to the server's log; the parent here
    # is the one that prints the login URL, exactly once.
    #
    # Blind THIS process to the session store while it settles its credentials.
    # Minting populates the parent's credential holder, and a holder built with
    # the store directory in the environment would open the deployment's real
    # store, restore its sessions, and hold them in a process that never serves
    # a request — where a later save would rewrite the file underneath the
    # child that does. Binding no store (``store is None``) makes the parent's
    # holder a pure secret carrier: no restore, no write. The real value is put
    # back before ``Popen``, which inherits ``os.environ``, so the child — the
    # process that actually serves — gets the directory.
    declared_store_dir = os.environ.get(DECLARED_SESSION_STORE_DIR_ENV)
    os.environ[DECLARED_SESSION_STORE_DIR_ENV] = ""
    try:
        login_url, announce = _mint_operator_url(host, port)
    finally:
        if declared_store_dir is None:
            os.environ.pop(DECLARED_SESSION_STORE_DIR_ENV, None)
        else:
            os.environ[DECLARED_SESSION_STORE_DIR_ENV] = declared_store_dir

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
        if announce:
            # The operator's way in. The clean URL above answers only with a
            # cookie the browser doesn't have yet; this one carries the token
            # that mints the session. Printed here in the parent — never by the
            # detached child, whose stdout is the log file.
            output.report(f"Open: {login_url}")
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


@web.group("sessions")
def web_sessions() -> None:
    """Manage the browser sessions the terminal has persisted to disk."""


def _inherited_repo(ctx: click.Context) -> Path | None:
    """The nearest ``--repo`` an enclosing group parsed, if any.

    ``osprey web --repo X sessions clear`` and
    ``osprey web sessions clear --repo X`` are the same request, and the group
    parses its own ``--repo`` before the subcommand name. ``web stop`` reads
    ``ctx.parent`` for this; a verb one level deeper has to walk, because its
    parent is the ``sessions`` group, which carries no repo of its own.
    """
    node = ctx.parent
    while node is not None:
        inherited = node.params.get("repo")
        if inherited is not None:
            return Path(inherited)
        node = node.parent
    return None


def _port_is_answering(host: str, port: int) -> bool:
    """Whether something is already listening on *host*:*port*.

    The PID file only exists for a ``--detach`` launch; a foreground
    ``osprey web`` writes none. So the socket is the probe that catches the
    shape the PID file cannot see, and the two are read together.

    A server bound to a wildcard is reached on that family's own loopback
    address -- ``0.0.0.0`` on ``127.0.0.1``, ``::`` on ``::1`` -- because the
    wildcard is a bind target, not a destination, and connecting to it is
    undefined on some platforms.

    Only the CONFIGURED port is probed. A foreground launch that found that port
    busy and auto-moved off it therefore answers somewhere this cannot see;
    that server has to be stopped by hand.
    """
    target = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    try:
        with socket.create_connection((target, port), timeout=0.5):
            return True
    except OSError:
        return False


def _persisted_session_count(path: Path) -> int:
    """How many session digests *path* holds — 0 for anything unreadable.

    Mirrors :class:`osprey.interfaces.web_auth.SessionStore`'s own posture on
    reading: a store that is truncated, hand-edited or owned by another user is
    worth no traceback here either. The count is a courtesy in the report, not
    a precondition for the delete, so a file that cannot be counted is still a
    file that gets removed.

    What it counts is the digests ON DISK, expired-but-not-yet-pruned entries
    included -- the store is rewritten on login and logout, not on a timer, so
    a deadline that has passed can sit in it for as long as the file does. The
    number is therefore what was dropped from the file, not how many sessions
    were still usable.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(payload, Mapping):
        return 0
    sessions = payload.get("sessions")
    return len(sessions) if isinstance(sessions, Mapping) else 0


@web_sessions.command("clear")
@repo_option
@click.option(
    "--force",
    is_flag=True,
    help="Clear the store even while a server is running (it keeps its live sessions).",
)
@click.pass_context
def web_sessions_clear(ctx: click.Context, repo: Path | None, force: bool) -> None:
    """Delete the terminal's persisted browser sessions.

    Sessions survive a restart because the terminal writes them to a small file
    under the deployment's agent-data directory. That is a convenience right up
    until it is not: an operator who suspects a cookie leaked, or who has just
    shortened session_lifetime and does not want yesterday's twelve-hour
    sessions riding through the change, wants every persisted session gone --
    and should not have to know the file's name or which port it is keyed to in
    order to get that.

    Refuses while a detached server's PID file is present, or while anything
    answers on the CONFIGURED host and port, because clearing then would not do
    what it looks like it does. The in-memory session map is what admits a
    request; the file is only its backup. A running server would go on honoring
    every cookie it already holds and would rewrite the file from that memory on
    the next login or logout, so the delete would be undone within minutes and
    would never have logged anyone out in the first place. Stop the server, and
    the same command means exactly what it says.

    One server escapes both checks and has to be stopped by hand: a FOREGROUND
    launch that found the configured port busy and auto-moved off it writes no
    PID file and no longer answers where this verb looks, so stop it before
    clearing.
    """
    if repo is None:
        repo = _inherited_repo(ctx)

    # `_resolve_render` chdirs into the render so `get_config_value` reads the
    # served deployment's config -- the same sequence `web()` runs, which is
    # what makes this verb look at the store the launcher would bind rather
    # than at one guessed from the current directory. Unlike `web()`, this
    # command does not go on to serve from there, so the cwd is handed back.
    # Only the cwd: `OSPREY_CONFIG` and the repo `.env` `_resolve_render` loads
    # stay published for the life of this process, as they do for every other
    # verb that resolves a render. Nothing after this reads them, and the
    # process is about to exit -- the tests restore the environment themselves.
    cwd = Path.cwd()
    try:
        repo_root, _build_dir, _config_path = _resolve_render(repo)
        wt_config = get_config_value("web_terminal", {})
        port_base = resolve_port_base({"deployment": get_config_value("deployment", {})})
        store_dir = (
            anchored_path(
                agent_data_base_dir({"agent_data": get_config_value("agent_data", {})}), repo_root
            )
            / "web_terminal"
        )
    finally:
        os.chdir(cwd)

    host = resolve_bind_host(None, wt_config.get("host"))
    port = resolve_web_port(None, wt_config.get("port"), base=port_base)

    output.section("", {"Repo": repo_root, "Store": store_dir})

    pid = _read_pid(repo_root)
    live = pid is not None or _port_is_answering(host, port)
    if live and not force:
        output.fail(
            "The web terminal is still running, so its sessions are still live",
            "The running server holds the session map in memory and rewrites "
            "the store from it on the next login or logout, so clearing the "
            "files now would neither log anyone out nor stay cleared.",
            "stop the server first: osprey web stop",
        )
        raise SystemExit(1)
    if live:
        output.warn(
            "Clearing the store while a server is running",
            "That server keeps its live sessions and warm terminals in memory "
            "until it exits. Clearing the files only stops them surviving the "
            "NEXT restart -- nobody is logged out right now.",
        )

    dropped = 0
    if not store_dir.is_dir():
        output.note(f"No session store directory at {store_dir} -- nothing has been persisted.")
    else:
        stores = sorted(p for p in store_dir.glob("sessions-*.json") if p.is_file())
        bare = store_dir / "sessions.json"
        if bare.is_file():
            stores.append(bare)
        for store in stores:
            dropped += _persisted_session_count(store)
            store.unlink(missing_ok=True)

    output.report(f"Dropped {dropped} persisted session(s).")
