"""OSPREY Web Terminal — FastAPI Application.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.vendor import vendor_url
from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster, WorkspaceWatcher
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes import router
from osprey.interfaces.web_terminal.url_prefix import apply_url_prefix, compute_url_prefix
from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from jinja2.runtime import Context

    from osprey.interfaces.design_system.generator.emit_js import ThemeManifestEntry

STATIC_DIR = Path(__file__).parent / "static"


@pass_context
def _prefixed(ctx: Context, path: str) -> str:
    """Jinja global: apply THE prefix contract to an HTML-parser-resolved URL.

    Import maps (see the ``importmap`` block injected into every served HTML
    document) only retarget module *specifiers* resolved inside already-
    loaded module code — they do NOT touch ``<link href>``, a classic
    ``<script src>``, or a module entrypoint's own ``src`` attribute. Those
    are ordinary browser URL resolutions, so the per-user prefix must be
    baked in explicitly at render time instead. Reads ``url_prefix`` off the
    template context (see ``compute_url_prefix()``); absolute URLs (e.g. a
    CDN URL from ``vendor_url()`` when not in offline mode) pass through
    unchanged, and an empty prefix is a byte-identical no-op.
    """
    return apply_url_prefix(ctx.get("url_prefix", ""), path)


templates = Jinja2Templates(directory=str(STATIC_DIR))
templates.env.globals["vendor_url"] = vendor_url
templates.env.globals["prefixed"] = _prefixed

logger = __import__("logging").getLogger(__name__)


def _launch_artifact_server(app: FastAPI) -> None:
    """Auto-launch the artifact gallery server if configured."""
    try:
        import os

        from osprey.infrastructure.server_launcher import ensure_artifact_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        art_config = config.get("artifact_server", {})
        host = art_config.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_ARTIFACT_SERVER_PORT", art_config.get("port", 8086)))

        app.state.artifact_server_url = f"http://{host}:{port}"
        ensure_artifact_server()
        logger.info("Artifact server available at %s", app.state.artifact_server_url)
    except Exception:
        logger.warning("Could not auto-launch artifact server", exc_info=True)
        app.state.artifact_server_url = "http://127.0.0.1:8086"


def _launch_ariel_server(app: FastAPI) -> None:
    """Auto-launch the ARIEL logbook server if configured."""
    try:
        from osprey.infrastructure.server_launcher import ensure_ariel_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        ariel_web = config.get("ariel", {}).get("web", {})
        host = ariel_web.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_ARIEL_PORT", ariel_web.get("port", 8085)))

        app.state.ariel_server_url = f"http://{host}:{port}"
        ensure_ariel_server()
        logger.info("ARIEL server available at %s", app.state.ariel_server_url)
    except Exception:
        logger.warning("Could not auto-launch ARIEL server", exc_info=True)
        app.state.ariel_server_url = None


def _launch_channel_finder_server(app: FastAPI) -> None:
    """Auto-launch the Channel Finder web server if configured."""
    try:
        from osprey.infrastructure.server_launcher import ensure_channel_finder_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        cf = config.get("channel_finder", {})
        if not cf:
            return
        cf_web = cf.get("web", {})
        host = cf_web.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_CHANNEL_FINDER_PORT", cf_web.get("port", 8092)))

        app.state.channel_finder_server_url = f"http://{host}:{port}"
        ensure_channel_finder_server()
        logger.info("Channel Finder server available at %s", app.state.channel_finder_server_url)
    except Exception:
        logger.warning("Could not auto-launch Channel Finder server", exc_info=True)
        app.state.channel_finder_server_url = None


def _launch_lattice_dashboard_server(app: FastAPI) -> None:
    """Auto-launch the lattice dashboard server if configured."""
    try:
        from osprey.infrastructure.server_launcher import ensure_lattice_dashboard_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        ld = config.get("lattice_dashboard", {})
        if not ld:
            return
        host = ld.get("host", "127.0.0.1")
        port = int(os.environ.get("OSPREY_LATTICE_DASHBOARD_PORT", ld.get("port", 8097)))

        app.state.lattice_dashboard_server_url = f"http://{host}:{port}"
        ensure_lattice_dashboard_server()
        logger.info("Lattice dashboard available at %s", app.state.lattice_dashboard_server_url)
    except Exception:
        logger.warning("Could not auto-launch lattice dashboard", exc_info=True)
        app.state.lattice_dashboard_server_url = None


def _launch_okf_server(app: FastAPI) -> None:
    """Auto-launch the OKF knowledge panel server if configured.

    The (host, port) computed here MUST match what ``ServerLauncher`` resolves
    for the ``okf`` definition (``registry/web.py``): host/port read directly
    from the ``facility_knowledge`` section (no ``web`` subkey), with the
    ``OSPREY_FACILITY_KNOWLEDGE_PORT`` env override. Otherwise the proxied URL
    stored here would point at a different port than the one uvicorn binds.
    """
    try:
        from osprey.infrastructure.server_launcher import ensure_okf_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        fk = config.get("facility_knowledge", {})
        if not fk:
            return
        host = fk.get("host", "127.0.0.1")
        # Guard a set-but-empty env override (e.g. compose `OSPREY_FACILITY_KNOWLEDGE_PORT=`):
        # int("") would raise and kill this launch (silent dead tab). This mirrors
        # server_launcher._make_config_reader's `if env_val:` guard so both sides
        # resolve the SAME port — the launcher would otherwise bind 8093 while we'd die.
        env_port = os.environ.get("OSPREY_FACILITY_KNOWLEDGE_PORT")
        port = int(env_port) if env_port else int(fk.get("port", 8093))

        app.state.okf_server_url = f"http://{host}:{port}"
        ensure_okf_server()
        logger.info("OKF knowledge panel available at %s", app.state.okf_server_url)
    except Exception:
        logger.warning("Could not auto-launch OKF knowledge panel", exc_info=True)
        app.state.okf_server_url = None


def _load_theme_registry() -> tuple[list[ThemeManifestEntry], dict[str, dict[str, str]]]:
    """Load the baked theme manifest + per-family defaults for SSR resolution.

    Reads the same ``tokens/`` source tree the design-system generator
    builds from (``generator/build.py::DEFAULT_TOKENS_DIR``) rather than
    parsing the generated ``tokens.js`` — one source, no risk of drifting
    from a stale generated artifact. This intentionally skips
    ``validate.assert_valid``: the checked-in tree is validated by
    ``build --check`` in CI, and full WCAG/completeness validation isn't
    needed just to read theme identity for SSR.

    Returns:
        ``(entries, defaults)`` as produced by
        :func:`~osprey.interfaces.design_system.generator.emit_js.build_theme_manifest`
        and :func:`~osprey.interfaces.design_system.generator.emit_js.build_theme_defaults`.
    """
    from osprey.interfaces.design_system.generator.build import DEFAULT_TOKENS_DIR
    from osprey.interfaces.design_system.generator.emit_js import (
        build_theme_defaults,
        build_theme_manifest,
    )
    from osprey.interfaces.design_system.generator.model import load_token_tree

    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    entries = build_theme_manifest(tree)
    defaults = build_theme_defaults(entries)
    return entries, defaults


def resolve_web_theme_id(
    configured: str,
    entries: Sequence[ThemeManifestEntry],
    defaults: dict[str, dict[str, str]],
) -> str:
    """Resolve the ``web.theme`` config value into a concrete baked theme id.

    ``configured`` may be:

    - A concrete theme id (e.g. ``"high-contrast-light"``) — used as-is.
      This is how an operator pins a specific mode instead of the
      family's dark default.
    - A theme *family* name (e.g. ``"osprey"``, ``"high-contrast"``) —
      resolved to that family's **dark** id, the canonical SSR default.
    - Anything else (unknown/misspelled) — logged as a warning and
      resolved to the ``osprey`` family's dark id.

    Mirrors the warn+fallback shape of
    :func:`osprey.cli.styles.load_theme_from_config`: never raises.

    Args:
        configured: The raw ``web.theme`` config value.
        entries: The theme manifest (see
            :func:`~osprey.interfaces.design_system.generator.emit_js.build_theme_manifest`).
        defaults: The per-family ``{family: {mode: id}}`` map (see
            :func:`~osprey.interfaces.design_system.generator.emit_js.build_theme_defaults`).

    Returns:
        A concrete theme id present in ``entries`` — the pre-paint
        ``theme-boot.js`` rung (Task 1.8) only honors a server-rendered
        ``data-theme`` that is a real baked id, never a family name or
        ``"auto"``.
    """
    valid_ids = {entry.id for entry in entries}
    if configured in valid_ids:
        return configured
    if configured in defaults:
        return defaults[configured]["dark"]

    logger.warning(
        "Unknown web.theme %r (not a theme id or family); falling back to "
        "osprey's dark theme. Valid ids: %s; valid families: %s",
        configured,
        sorted(valid_ids),
        sorted(defaults),
    )
    osprey_dark = defaults.get("osprey", {}).get("dark")
    if osprey_dark is not None:
        return osprey_dark
    # Degenerate case (no built-in ``osprey`` family): still return a real
    # baked dark id — ``build_theme_defaults`` guarantees each family has a
    # dark member — rather than an unverified literal, so Task 1.8's boot
    # rung honors it instead of silently dropping to auto (FOUC).
    for family_modes in defaults.values():
        if "dark" in family_modes:
            return family_modes["dark"]
    return next(iter(sorted(valid_ids)), "dark")


def _load_panel_config() -> tuple[set[str], list[dict], str | None]:
    """Read web.panels and web.default_panel from config.yml.

    Returns:
        (enabled_builtin_ids, custom_panel_defs, default_panel_id_or_None)

        The default panel id is returned as declared by the profile/config;
        it is **not** validated here — the frontend treats an unknown id as
        a request to fall back to DEFAULT_PANEL_FALLBACK so a typo doesn't
        leave the user staring at a blank tabset.
    """
    try:
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
    except Exception:
        return set(UNIVERSAL_PANELS), [], None

    web_config = config.get("web", {})
    panels_config = web_config.get("panels", {})
    default_panel = web_config.get("default_panel")

    enabled = set(UNIVERSAL_PANELS)  # Always on
    custom = []

    for panel_id, spec in panels_config.items():
        if panel_id in BUILTIN_PANELS:
            if spec is True or (isinstance(spec, dict) and spec.get("enabled", True)):
                enabled.add(panel_id)
        else:
            custom.append(
                {
                    "id": panel_id,
                    "label": spec.get("label", panel_id.upper()),
                    "url": spec.get("url", ""),
                    "healthEndpoint": spec.get("health_endpoint"),
                    "path": spec.get("path", "/"),
                    # Trust marker: this panel was declared in config (a trusted
                    # input), not registered at runtime via POST /api/panels/register.
                    # Only the config loader stamps it, so credential injection
                    # (routes/proxy.py) and id reservation (routes/panels.py) can key
                    # off panel *origin* rather than the forgeable id string. Set
                    # explicitly here — GET /api/panels spreads this dict to the
                    # browser, so only deliberately-placed fields are exposed.
                    "configDefined": True,
                    # Path suffixes whose JSON responses get the proxy's
                    # root-absolute-literal rewrite (see routes/proxy.py) —
                    # for backends whose SPA bootstraps its API base from a
                    # JSON config endpoint.
                    "rewriteJsonPaths": spec.get("rewrite_json_paths") or [],
                }
            )

    return enabled, custom, default_panel


class _PanelRuntimeConfig(NamedTuple):
    """Runtime-panel settings derived from config, plus the computed visible list."""

    allow_runtime_panels: bool
    runtime_panel_allowlist: list[str] | None
    visible_panels: list[str]


def _load_panel_runtime_config(
    enabled_panels: set[str], custom_panels: list[dict]
) -> _PanelRuntimeConfig:
    """Read runtime-panel settings and compute the visible-panel list.

    Honors per-panel ``hidden: true`` flags and the ``web.allow_runtime_panels`` /
    ``web.runtime_panel_allowlist`` knobs.  The raw config is re-read here rather
    than threaded through ``_load_panel_config``'s 3-tuple contract (which is
    relied on elsewhere, including tests).  Built-in panel specs are not retained
    by ``_load_panel_config`` — only the id lands in ``enabled`` — so hidden
    built-ins are tracked in a parallel set.

    ``visible_panels`` is the flat list of ids shown in the UI: enabled built-ins
    (minus hidden ones) followed by custom panels (minus hidden ones).  With no
    ``hidden`` flags it equals all enabled panels — backward compatible.

    Fails open: any config-read error yields the permissive defaults (nothing
    hidden, runtime registration off).
    """
    hidden_builtins: set[str] = set()
    hidden_custom_ids: set[str] = set()
    allow_runtime_panels = False
    runtime_panel_allowlist: list[str] | None = None
    try:
        from osprey.utils.workspace import load_osprey_config

        web_cfg = load_osprey_config().get("web", {})
        allow_runtime_panels = bool(web_cfg.get("allow_runtime_panels", False))
        allowlist_raw = web_cfg.get("runtime_panel_allowlist")
        if isinstance(allowlist_raw, list):
            # Lowercase at parse time so matching in _validate_panel_url is case-insensitive.
            runtime_panel_allowlist = [str(e).lower() for e in allowlist_raw]
        for pid, spec in web_cfg.get("panels", {}).items():
            if isinstance(spec, dict) and spec.get("hidden", False):
                if pid in BUILTIN_PANELS:
                    hidden_builtins.add(pid)
                else:
                    hidden_custom_ids.add(pid)
    except Exception:
        pass

    visible_panels = [p for p in enabled_panels if p not in hidden_builtins] + [
        cp["id"] for cp in custom_panels if cp["id"] not in hidden_custom_ids
    ]
    return _PanelRuntimeConfig(
        allow_runtime_panels=allow_runtime_panels,
        runtime_panel_allowlist=runtime_panel_allowlist,
        visible_panels=visible_panels,
    )


def _load_panel_presets(enabled_panels: set[str], custom_panels: list[dict]) -> list[dict]:
    """Read ``web.presets`` and resolve each named layout against the live panel set.

    A preset is a facility-curated, named list of panel ids a human applies in one
    click (the "Layouts" section of the "+" popover). This resolves the raw config
    into the shape the frontend consumes, mirroring how :func:`_load_panel_config`
    turns ``web.panels`` into concrete ids.

    Each preset's members are intersected with the set of known ids (enabled
    built-ins plus custom panel ids); unknown members are dropped with a warning
    (a typo or a disabled panel must not strand the user), and a preset that
    resolves to no known members is dropped entirely. Config insertion order is
    preserved (pyyaml keeps mapping order on 3.7+), so config order == menu order.

    Args:
        enabled_panels: Enabled built-in panel ids (from :func:`_load_panel_config`).
        custom_panels: Custom panel dicts (from :func:`_load_panel_config`).

    Returns:
        A list of ``{"name": str, "panels": [id, ...]}`` dicts in config order — a
        list (not a dict) so JSON serialization preserves ordering in the
        ``GET /api/panels`` payload. Fails open to ``[]`` on any config-read error.
    """
    known: set[str] = set(enabled_panels) | {cp["id"] for cp in custom_panels}
    presets: list[dict] = []
    try:
        from osprey.utils.workspace import load_osprey_config

        raw_presets = load_osprey_config().get("web", {}).get("presets", {})
    except Exception:
        return []

    if not isinstance(raw_presets, dict):
        return []

    for name, members in raw_presets.items():
        if not isinstance(members, list):
            logger.warning("web.presets[%r] is not a list of panel ids; skipping.", name)
            continue
        resolved: list[str] = []
        for member in members:
            if member in known:
                resolved.append(member)
            else:
                logger.warning(
                    "web.presets[%r] references unknown panel id %r; dropping it.", name, member
                )
        if resolved:
            presets.append({"name": str(name), "panels": resolved})
        else:
            logger.warning("web.presets[%r] has no known panel members; dropping the preset.", name)
    return presets


def _load_web_config(config_path: str | Path | None = None) -> dict:
    """Load web_terminal config section from config.yml."""
    config_paths = [
        Path(config_path) if config_path else None,
        Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
        Path("config.yml"),
    ]

    for path in config_paths:
        if path and path.exists() and path.is_file():
            with open(path) as f:
                config = yaml.safe_load(f) or {}
            return config.get("web_terminal", {})

    return {}


def _load_claude_code_config(config_path: str | Path | None = None) -> dict:
    """Load claude_code config section from config.yml.

    Mirrors :func:`_load_web_config` so the lifespan can derive the Claude
    Code launch argv (honoring ``claude_code.cli_version`` pins) even when
    no explicit ``shell_command`` was passed — e.g. under ``uvicorn --reload``
    where ``create_app`` is called with no arguments.
    """
    config_paths = [
        Path(config_path) if config_path else None,
        Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
        Path("config.yml"),
    ]

    for path in config_paths:
        if path and path.exists() and path.is_file():
            with open(path) as f:
                config = yaml.safe_load(f) or {}
            return config.get("claude_code", {})

    return {}


def _create_lifespan(
    config_path: str | Path | None = None,
    shell_command: list[str] | None = None,
    project_dir: str | Path | None = None,
):
    """Create a lifespan context manager for the app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from osprey.utils.claude_launcher import build_claude_launch_argv

        config = _load_web_config(config_path)

        import uuid

        app.state.server_session_id = uuid.uuid4().hex[:12]
        # Shell-command precedence — always normalized to list[str] so every
        # downstream consumer (websocket initial spawn + switch_session) can
        # safely unpack with [*base, ...]. The pin lookup lets --reload mode
        # honor claude_code.cli_version even though uvicorn's factory bypass
        # never lets web_cmd.py inject the argv.
        if shell_command:
            app.state.shell_command = list(shell_command)
        elif config.get("shell"):
            app.state.shell_command = [str(config["shell"])]
        else:
            app.state.shell_command = build_claude_launch_argv(
                _load_claude_code_config(config_path)
            )
        max_bg = int(config.get("max_background_sessions", 5))
        app.state.pty_registry = PtyRegistry(max_background=max_bg)
        app.state.operator_registry = OperatorRegistry()
        app.state.project_cwd = str(
            Path(project_dir).resolve() if project_dir else Path.cwd().resolve()
        )
        app.state.broadcaster = FileEventBroadcaster()
        app.state.active_panel = None
        # Optional human-readable deployment name shown in the header so
        # otherwise-identical web terminals are distinguishable. The
        # ``OSPREY_WEB_APP_NAME`` environment variable takes precedence over
        # ``web.app_name`` in config.yml, so several containers that share one
        # baked config image can each be named individually via the environment.
        # Empty/absent ⇒ no label is rendered.
        app.state.app_name = (
            os.environ.get("OSPREY_WEB_APP_NAME", "").strip()
            or str((config.get("web") or {}).get("app_name") or "").strip()
        )
        # Per-user deployment identity for multi-user compose stacks. No
        # config key exists for either of these today, so the config-side
        # fallback is always empty and ``OSPREY_TERMINAL_USER`` /
        # ``OSPREY_TERMINAL_LANDING_URL`` are the sole source. Empty ⇒ no
        # user badge / logout control is rendered.
        app.state.terminal_user = os.environ.get("OSPREY_TERMINAL_USER", "").strip()
        app.state.landing_url = os.environ.get("OSPREY_TERMINAL_LANDING_URL", "").strip()

        # Ensure OSPREY_CONFIG is set before any load_osprey_config() call
        if "OSPREY_CONFIG" not in os.environ:
            candidate = Path(app.state.project_cwd) / "config.yml"
            if candidate.exists():
                os.environ["OSPREY_CONFIG"] = str(candidate)
                logger.debug("Auto-set OSPREY_CONFIG=%s", candidate)

        # Clear any stale config cache (e.g. from web_cmd.py pre-lifespan call)
        from osprey.utils.workspace import reset_config_cache

        reset_config_cache()

        # Resolve and store config_path for the settings API
        resolved_config_path = None
        for candidate in [
            Path(config_path) if config_path else None,
            Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
            Path("config.yml"),
        ]:
            if candidate and candidate.exists() and candidate.is_file():
                resolved_config_path = candidate.resolve()
                break
        app.state.config_path = resolved_config_path

        # ── Web theme (SSR no-FOUC attribute, Task 1.10) ──
        # Resolved once at startup and server-rendered onto <html data-theme>
        # so the generated theme-boot.js first-paints with no flash (Task
        # 1.8). Fails open on any load error — a missing/broken theme
        # registry must never block server startup.
        try:
            from osprey.utils.config import get_config_value

            configured_web_theme = get_config_value("web.theme", "osprey")
            theme_entries, theme_defaults = _load_theme_registry()
            app.state.web_theme_id = resolve_web_theme_id(
                configured_web_theme, theme_entries, theme_defaults
            )
        except Exception:  # noqa: BLE001 — never let config/theme-registry load block startup
            logger.warning(
                "Could not resolve web.theme (config or theme-registry load failed); "
                "server-rendering fallback theme 'dark'",
                exc_info=True,
            )
            app.state.web_theme_id = "dark"

        # ── Regenerate stale Claude Code artifacts on launch ──
        # config.yml is a build-time input: safety-critical fields (e.g. the
        # writes_enabled kill-switch baked into settings.json's permissions.deny)
        # only take effect once the artifacts are re-rendered. Regenerating here
        # — mirroring `osprey claude chat` — means an edited config.yml is honored
        # on the next server start. Fail open so a regen error never blocks launch.
        try:
            from osprey.cli.templates.manager import TemplateManager

            project_dir_for_regen = Path(app.state.project_cwd)
            changed = TemplateManager().regen_if_drift(project_dir_for_regen)
            if changed:
                logger.info(
                    "Regenerated %d stale Claude Code artifact(s): %s",
                    len(changed),
                    ", ".join(changed),
                )
        except Exception:  # noqa: BLE001 — never let regen block server startup
            logger.warning("Claude Code artifact regen on launch failed", exc_info=True)

        # ── Provider env injection ──
        from osprey.cli.claude_code_resolver import (
            detect_managed_policy_conflicts,
            format_managed_policy_conflicts,
            inject_provider_env,
            load_provider_spec,
        )

        # Managed (enterprise) policy settings outrank the process environment
        # and --setting-sources project alike, so a policy `env` block setting a
        # provider variable would silently redirect the operator-facing terminal
        # to a backend the project did not configure. Refuse to start.
        _policy_conflicts = detect_managed_policy_conflicts()
        if _policy_conflicts:
            raise RuntimeError(
                "Refusing to start the Web Terminal.\n"
                + format_managed_policy_conflicts(_policy_conflicts)
            )

        if app.state.config_path:
            _project_dir = Path(app.state.config_path).parent
            # load_provider_spec expands ${VAR} in provider config before resolving.
            _spec = load_provider_spec(_project_dir)
            if _spec:
                inject_provider_env(os.environ, _spec, project_dir=_project_dir)

                # Start translation proxy for OpenAI-compatible providers
                if _spec.needs_proxy and _spec.upstream_base_url:
                    from osprey.infrastructure.proxy.lifecycle import start_proxy

                    proxy_port = start_proxy(
                        _spec.upstream_base_url,
                        os.environ.get(_spec.auth_env_var),
                    )
                    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"
                    logger.info(
                        "Translation proxy on :%d → %s",
                        proxy_port,
                        _spec.upstream_base_url,
                    )

        workspace_dir = Path(config.get("watch_dir") or "./_agent_data").resolve()
        app.state.workspace_dir = workspace_dir  # base path (file watcher watches all sessions)
        app.state.workspace_base = workspace_dir  # alias for clarity
        app.state.watcher = WorkspaceWatcher(workspace_dir, app.state.broadcaster)
        app.state.watcher.start()

        # Load panel config and conditionally launch servers
        enabled_panels, custom_panels, default_panel = _load_panel_config()
        app.state.enabled_panels = enabled_panels
        app.state.custom_panels = custom_panels
        app.state.default_panel = default_panel

        # Runtime-panel settings + visibility (honors hidden: true,
        # allow_runtime_panels, runtime_panel_allowlist).
        panel_runtime = _load_panel_runtime_config(enabled_panels, custom_panels)
        app.state.allow_runtime_panels = panel_runtime.allow_runtime_panels
        app.state.runtime_panel_allowlist = panel_runtime.runtime_panel_allowlist
        app.state.visible_panels = panel_runtime.visible_panels

        # Config-defined panel presets ("Layouts"): named sets of panel ids a
        # human applies in one click. Immutable config-derived state — the only
        # new server state this feature adds. Empty (the default) → the "+" menu
        # renders exactly as before.
        app.state.panel_presets = _load_panel_presets(enabled_panels, custom_panels)

        if panel_runtime.allow_runtime_panels and not panel_runtime.runtime_panel_allowlist:
            logger.warning(
                "web.allow_runtime_panels is enabled without a runtime_panel_allowlist — "
                "any http/https host on the internal network can be registered as a panel proxy."
            )

        # Discover local static panel bundles under <project>/panels/ and wire
        # them into the hub. Gated on web.allow_runtime_panels (the human opt-in);
        # fail-closed on any malformed/non-compliant bundle. See panel_discovery.
        # Wrapped so panel discovery can never block server startup (matching the
        # other config loaders in this lifespan).
        app.state.discovered_panel_dirs = {}
        try:
            from osprey.interfaces.web_terminal.panel_discovery import (
                apply_discovered_panels,
            )

            apply_discovered_panels(app)
        except Exception:
            logger.warning("Local panel discovery failed; continuing.", exc_info=True)

        # Universal servers — always launched
        _launch_artifact_server(app)

        # Domain servers — template-controlled
        if "ariel" in enabled_panels:
            _launch_ariel_server(app)
        if "channel-finder" in enabled_panels:
            _launch_channel_finder_server(app)
        if "lattice" in enabled_panels:
            _launch_lattice_dashboard_server(app)
        if "okf" in enabled_panels:
            _launch_okf_server(app)

        # Hook env placeholder — hooks read config.yml directly for
        # hot-reloadable settings (no env var propagation needed).
        app.state.hooks_env = {}

        # Shared httpx client for the panel reverse proxy.
        # trust_env=False prevents routing through the corporate HTTP proxy
        # (e.g. Squid) — all panel backends are container-local or on the
        # Docker network and must be reached directly.
        app.state.proxy_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            follow_redirects=True,
            trust_env=False,
        )

        yield

        await app.state.proxy_client.aclose()

        # Stop translation proxy if it was started
        from osprey.infrastructure.proxy.lifecycle import stop_proxy

        stop_proxy()

        app.state.watcher.stop()
        app.state.pty_registry.cleanup_all()
        await app.state.operator_registry.cleanup_all()

    return lifespan


def create_app(
    config_path: str | Path | None = None,
    shell_command: list[str] | None = None,
    project_dir: str | Path | None = None,
) -> FastAPI:
    """Create the Web Terminal FastAPI application.

    Args:
        config_path: Optional path to config.yml.
        shell_command: Shell command to spawn in the PTY.
        project_dir: Optional OSPREY project directory. When set, used as
            ``project_cwd`` instead of the current working directory.

    Returns:
        Configured FastAPI application.
    """
    url_prefix = compute_url_prefix()

    # root_path is deliberately NOT set to url_prefix. nginx strips the
    # /u/<user> prefix before proxying (see nginx.conf.j2 / docker-compose.web),
    # so this app always receives BARE paths (/static/…, /design-system/…,
    # /api/…, /ws/…). A non-empty FastAPI(root_path=…) forces
    # scope["root_path"] on every request, which makes Starlette's StaticFiles
    # Mounts recompute their child scope as root_path + mount_path and expect
    # the prefix to be present in the path — so every asset 404s on the bare
    # path nginx actually forwards, silently loading the multi-user UI with no
    # CSS/JS/fonts. The prefix is plumbed where it is genuinely needed instead:
    # the window global + import map injected into each HTML document below,
    # and routes/panels.py + routes/proxy.py (which read compute_url_prefix()
    # directly). Guarded by test_prefix_injection.py's bare-path static assert
    # and the tests/e2e/web_terminals/test_prefix_routing.py master e2e.
    app = FastAPI(
        title="OSPREY Web Terminal",
        description="Browser-based terminal with live workspace viewer",
        version="1.0.0",
        lifespan=_create_lifespan(config_path, shell_command, project_dir),
    )
    app.state.url_prefix = url_prefix

    app.include_router(router)

    @app.get("/")
    async def root(request: Request):
        app_name = getattr(request.app.state, "app_name", "")
        web_theme_id = getattr(request.app.state, "web_theme_id", "dark")
        terminal_user = getattr(request.app.state, "terminal_user", "")
        landing_url = getattr(request.app.state, "landing_url", "")
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "app_name": app_name,
                "web_theme_id": web_theme_id,
                "terminal_user": terminal_user,
                "landing_url": landing_url,
                "url_prefix": url_prefix,
            },
        )

    # session.html/safety.html are otherwise plain static files under
    # STATIC_DIR (served verbatim by the /static mount below); these two
    # routes shadow that mount for exactly those two paths so they, too, get
    # the Jinja-rendered prefix injection. Must be registered before
    # configure_interface_app() mounts /static (Starlette matches routes in
    # registration order, so an explicit route ahead of a Mount wins).
    @app.get("/static/session.html")
    async def session_page(request: Request):
        return templates.TemplateResponse(request, "session.html", {"url_prefix": url_prefix})

    @app.get("/static/safety.html")
    async def safety_page(request: Request):
        return templates.TemplateResponse(request, "safety.html", {"url_prefix": url_prefix})

    configure_interface_app(app, static_dir=STATIC_DIR)

    return app


def _open_browser_when_ready(url: str, timeout: float = 15.0) -> None:
    """Wait for the server to accept connections, then open the browser."""
    import socket
    import threading
    import time
    import webbrowser
    from urllib.parse import urlparse

    def _wait_and_open():
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8087
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)
        else:
            return  # Server didn't start in time; skip browser open
        webbrowser.open(url)

    t = threading.Thread(target=_wait_and_open, daemon=True)
    t.start()


def run_web(
    host: str = "127.0.0.1",
    port: int = 8087,
    shell_command: list[str] | None = None,
    config_path: str | None = None,
    project_dir: str | None = None,
) -> None:
    """Run the web terminal server.

    Args:
        host: Host to bind to.
        port: Port to run on.
        shell_command: Shell command to spawn in the PTY.
        config_path: Optional path to config file.
        project_dir: Optional OSPREY project directory.
    """
    import uvicorn

    url = f"http://{host}:{port}"
    _open_browser_when_ready(url)

    app = create_app(config_path=config_path, shell_command=shell_command, project_dir=project_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
