"""OSPREY Web Terminal — FastAPI Application.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context

from osprey.cli.scaffold_cmd import ScaffoldClaimError
from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.vendor import vendor_url
from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster, WorkspaceWatcher
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry
from osprey.interfaces.web_terminal.ownership import OwnershipStoreError
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
    """Auto-launch the artifact gallery server if configured.

    The URL published here is the *only* input to panel availability
    (``/api/artifact-server``), so it must never outrun the server itself.
    ``artifact_server.auto_launch: false`` makes ``ensure_artifact_server()`` a
    no-op; publishing a URL anyway left the operator an enabled WORKSPACE tab
    whose iframe returned a bare 502. Leaving it unset disables the tab instead,
    which is what a suppressed server should look like.
    """
    try:
        from osprey.infrastructure.server_launcher import (
            ensure_artifact_server,
            is_auto_launch_enabled,
        )
        from osprey.registry.web import resolve_web_server_base_url

        if not is_auto_launch_enabled("artifact"):
            logger.info("Artifact server auto-launch is disabled; WORKSPACE tab unavailable")
            return

        # Same resolver the launcher itself uses, so the URL published here and
        # the port uvicorn binds cannot drift — including the set-but-empty env
        # override (compose `OSPREY_ARTIFACT_SERVER_PORT=`), which the previous
        # inline int() turned into a ValueError and a dead tab.
        app.state.artifact_server_url = resolve_web_server_base_url("artifact")
        ensure_artifact_server()
        logger.info("Artifact server available at %s", app.state.artifact_server_url)
    except Exception:
        logger.warning("Could not auto-launch artifact server", exc_info=True)
        # No fallback URL: a launch we could not complete must not be advertised
        # as an available panel. Retract any URL assigned before the failure.
        app.state.artifact_server_url = None


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
        # registry.web.resolve_web_server_address's empty-value guard so both sides
        # resolve the SAME port — the launcher would otherwise bind 8093 while we'd die.
        env_port = os.environ.get("OSPREY_FACILITY_KNOWLEDGE_PORT")
        port = int(env_port) if env_port else int(fk.get("port", 8093))

        app.state.okf_server_url = f"http://{host}:{port}"
        ensure_okf_server()
        logger.info("OKF knowledge panel available at %s", app.state.okf_server_url)
    except Exception:
        logger.warning("Could not auto-launch OKF knowledge panel", exc_info=True)
        app.state.okf_server_url = None


def _launch_system_health_server(app: FastAPI) -> None:
    """Auto-launch the System Health dashboard server if enabled.

    The (host, port) computed here MUST match what ``ServerLauncher`` resolves
    for the ``system_health`` definition (``registry/web.py``): host/port read
    from the nested ``health.web`` subkey, with the ``OSPREY_HEALTH_PORT`` env
    override. ``require_section`` is False, so unlike the okf/channel-finder
    launchers there is no early return on a missing section — the health
    framework ships a usable default, so the panel is always launchable.
    """
    try:
        from osprey.infrastructure.server_launcher import ensure_system_health_server
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        health_web = config.get("health", {}).get("web", {})
        host = health_web.get("host", "127.0.0.1")
        # Guard a set-but-empty env override (e.g. compose `OSPREY_HEALTH_PORT=`):
        # int("") would raise and kill this launch (silent dead tab). This mirrors
        # registry.web.resolve_web_server_address's empty-value guard so both sides
        # resolve the SAME port — the launcher would otherwise bind 8094 while we'd die.
        env_port = os.environ.get("OSPREY_HEALTH_PORT")
        port = int(env_port) if env_port else int(health_web.get("port", 8094))

        app.state.system_health_server_url = f"http://{host}:{port}"
        ensure_system_health_server()
        logger.info("System Health dashboard available at %s", app.state.system_health_server_url)
    except Exception:
        logger.warning("Could not auto-launch System Health dashboard", exc_info=True)
        app.state.system_health_server_url = None


def _load_theme_registry() -> tuple[list[ThemeManifestEntry], dict[str, dict[str, str]]]:
    """Load the baked theme manifest + per-family defaults for SSR resolution.

    Thin alias for
    :func:`osprey.interfaces.design_system.theme_config.load_theme_registry`,
    kept because this module's lifespan and its tests both reach for it by this
    name. The multi-user landing-page renderer resolves the same config value
    through that same module, so the two surfaces cannot disagree about what
    ``web.theme`` means.

    Returns:
        ``(entries, defaults)`` as produced by
        :func:`~osprey.interfaces.design_system.generator.emit_js.build_theme_manifest`
        and :func:`~osprey.interfaces.design_system.generator.emit_js.build_theme_defaults`.
    """
    from osprey.interfaces.design_system.theme_config import load_theme_registry

    return load_theme_registry()


def resolve_web_theme_id(
    configured: str,
    entries: Sequence[ThemeManifestEntry],
    defaults: dict[str, dict[str, str]],
) -> str:
    """Resolve the ``web.theme`` config value into a concrete baked theme id.

    ``web.theme``-named alias for
    :func:`osprey.interfaces.design_system.theme_config.resolve_theme_id`,
    which documents the full contract (family vs concrete id, the warn +
    fallback on an unknown value, and the guarantee that the result is always a
    real baked id — what the pre-paint ``theme-boot.js`` rung requires). The
    multi-user landing-page renderer calls that same function, so the two
    surfaces cannot disagree about what ``web.theme`` means.

    The returned id alone does NOT say whether the deployment pinned a mode —
    see :func:`resolve_web_theme_pinned_mode` for that half.

    Args:
        configured: The raw ``web.theme`` config value.
        entries: The theme manifest.
        defaults: The per-family ``{family: {mode: id}}`` map.

    Returns:
        A concrete theme id present in ``entries``.
    """
    from osprey.interfaces.design_system.theme_config import resolve_theme_id

    return resolve_theme_id(configured, entries, defaults, config_key="web.theme")


def resolve_web_theme_pinned_mode(
    configured: str,
    entries: Sequence[ThemeManifestEntry],
) -> str | None:
    """The light/dark mode a ``web.theme`` value *pins*, if any.

    ``web.theme``-named alias for
    :func:`osprey.interfaces.design_system.theme_config.resolve_pinned_mode`,
    which documents why the pin has to be carried separately from the resolved
    id at all.

    The lifespan server-renders the result as ``<html data-theme-mode>``, and
    the browser needs it: without that attribute ``theme-manager.js``'s hub
    assumes ``mode: 'auto'`` on a first visit and immediately re-resolves the
    mode from the OS, silently discarding a configured pin one frame after
    paint.

    Args:
        configured: The raw ``web.theme`` config value.
        entries: The theme manifest.

    Returns:
        ``"dark"`` or ``"light"`` when ``configured`` names a concrete theme id;
        ``None`` when it names a family or is unknown.
    """
    from osprey.interfaces.design_system.theme_config import resolve_pinned_mode

    return resolve_pinned_mode(configured, entries)


#: The two supported web UI modes. ``expert`` is the full split-pane terminal
#: workspace; ``simple`` is the pared-down operator layout. ``expert`` is the
#: default so an absent/misconfigured ``web.ui_mode`` never strands a deployment
#: in the reduced surface.
UI_MODES = ("expert", "simple")
DEFAULT_UI_MODE = "expert"


def resolve_ui_mode(configured: str) -> str:
    """Resolve the ``web.ui_mode`` config value into a concrete UI mode.

    ``configured`` must be one of :data:`UI_MODES` (``"expert"`` or
    ``"simple"``). Anything else — a typo, ``None``, an empty string — is
    logged as a warning and resolved to :data:`DEFAULT_UI_MODE`.

    Mirrors the warn+fallback shape of :func:`resolve_web_theme_id`: it never
    raises, so a bad value degrades to the safe default instead of blocking
    server startup.

    Args:
        configured: The raw ``web.ui_mode`` config value.

    Returns:
        A concrete mode string in :data:`UI_MODES` — the value stamped onto
        ``<html data-ui-mode>`` for the pre-paint mode-boot rung, which only
        honors a real mode.
    """
    if configured in UI_MODES:
        return configured

    logger.warning(
        "Unknown web.ui_mode %r (expected one of %s); falling back to %r.",
        configured,
        list(UI_MODES),
        DEFAULT_UI_MODE,
    )
    return DEFAULT_UI_MODE


#: The two supported rail positions. ``left`` is the redesign's icon-rail
#: column; ``top`` renders the same rail as a horizontal strip under the
#: header — the arrangement operators know from the pre-redesign tab bar.
RAIL_POSITIONS = ("left", "top")
DEFAULT_RAIL_POSITION = "left"

#: Per-theme-family rail defaults, applied only when ``web.rail_position``
#: is absent from config. The ``retro`` family restores the pre-redesign
#: look, and the horizontal tab strip under the header is part of that look
#: — picking Retro without also moving the rail would hand back the old
#: colors inside the new layout. Any family not listed here defaults to
#: :data:`DEFAULT_RAIL_POSITION`.
#:
#: This map is the single source of truth for the coupling: ``GET
#: /api/panels`` echoes it to the browser so ``rail-position.js`` can follow
#: a live family switch without carrying its own copy.
FAMILY_RAIL_DEFAULTS: dict[str, str] = {"retro": "top"}


def family_rail_default(family: str | None) -> str:
    """The rail position a theme family implies, absent explicit config.

    Args:
        family: A theme ``$extensions.family`` value, or ``None`` when the
            family could not be resolved.

    Returns:
        The family's entry in :data:`FAMILY_RAIL_DEFAULTS`, else
        :data:`DEFAULT_RAIL_POSITION`.
    """
    return FAMILY_RAIL_DEFAULTS.get(family or "", DEFAULT_RAIL_POSITION)


def resolve_rail_position(configured: str | None, theme_family: str | None = None) -> str:
    """Resolve the ``web.rail_position`` config value into a concrete position.

    An explicit ``configured`` in :data:`RAIL_POSITIONS` (``"left"`` or
    ``"top"``) always wins — a deployment that states a rail position keeps
    it in every theme. ``None`` means the key is absent from config, in
    which case the position comes from the active theme family via
    :func:`family_rail_default`. Anything else — a typo, an empty string —
    is logged as a warning and treated as absent.

    Mirrors the warn+fallback shape of :func:`resolve_ui_mode`: it never
    raises, so a bad value degrades to the safe default instead of blocking
    server startup.

    Args:
        configured: The raw ``web.rail_position`` config value, or ``None``
            when the key is absent.
        theme_family: The resolved theme's ``$extensions.family``, used only
            when ``configured`` gives no answer.

    Returns:
        A concrete position string in :data:`RAIL_POSITIONS` — the value
        stamped onto ``<html data-rail-position>`` for the pre-paint
        rail-boot rung, which only honors a real position.
    """
    if configured in RAIL_POSITIONS:
        return configured

    if configured is not None:
        logger.warning(
            "Unknown web.rail_position %r (expected one of %s); falling back to "
            "the position the active theme family implies.",
            configured,
            list(RAIL_POSITIONS),
        )
    return family_rail_default(theme_family)


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

    if not config:
        # The CLI refuses to launch without a resolvable config, but this app
        # can also be created directly (tests, uvicorn factory) — never let
        # that degrade silently into a rail with most of its panels missing.
        logger.warning(
            "No OSPREY config resolved (OSPREY_CONFIG=%s, cwd=%s) — "
            "serving universal panels only: %s",
            os.environ.get("OSPREY_CONFIG", "<unset>"),
            Path.cwd(),
            sorted(UNIVERSAL_PANELS),
        )

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


def _load_config_section(section: str, config_path: str | Path | None = None) -> dict:
    """Load one top-level section from config.yml."""
    config_paths = [
        Path(config_path) if config_path else None,
        Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
        Path("config.yml"),
    ]

    for path in config_paths:
        if path and path.exists() and path.is_file():
            with open(path) as f:
                config = yaml.safe_load(f) or {}
            return config.get(section, {})

    return {}


def _load_web_config(config_path: str | Path | None = None) -> dict:
    """Load web_terminal config section from config.yml."""
    return _load_config_section("web_terminal", config_path)


def _load_web_ui_config(config_path: str | Path | None = None) -> dict:
    """Load the top-level ``web`` section (UI settings: app_name, theme, presets)."""
    return _load_config_section("web", config_path)


def _load_claude_code_config(config_path: str | Path | None = None) -> dict:
    """Load claude_code config section from config.yml.

    Mirrors :func:`_load_web_config` so the lifespan can derive the Claude
    Code launch argv (honoring ``claude_code.cli_version`` pins) even when
    no explicit ``shell_command`` was passed — e.g. under ``uvicorn --reload``
    where ``create_app`` is called with no arguments.
    """
    return _load_config_section("claude_code", config_path)


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

        # ── Simple-mode chat pool bounds (Task 1.7) ──
        # Three knobs bound the operator-chat pool; each fails open to its
        # default so a missing/broken config never blocks startup. Read from
        # the top-level `web` section (same section as web.theme/web.ui_mode).
        # The route handlers re-read the two timeouts off app.state via getattr
        # with these same defaults, so the attribute names are load-bearing.
        try:
            from osprey.utils.config import get_config_value

            chat_turn_timeout_s = float(get_config_value("web.chat_turn_timeout_s", 600))
            chat_idle_timeout_s = float(get_config_value("web.chat_idle_timeout_s", 1800))
            chat_max_sessions = int(get_config_value("web.chat_max_sessions", 5))
        except Exception:  # noqa: BLE001 — never let config load block startup
            logger.warning(
                "Could not resolve web.chat_* config keys; using defaults "
                "(turn=600s, idle=1800s, max=5)",
                exc_info=True,
            )
            chat_turn_timeout_s, chat_idle_timeout_s, chat_max_sessions = 600.0, 1800.0, 5
        app.state.chat_turn_timeout_s = chat_turn_timeout_s
        app.state.chat_idle_timeout_s = chat_idle_timeout_s
        app.state.chat_max_sessions = chat_max_sessions

        app.state.operator_registry = OperatorRegistry(
            chat_max_sessions=chat_max_sessions,
            chat_idle_seconds=chat_idle_timeout_s,
        )
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
            or str(_load_web_ui_config(config_path).get("app_name") or "").strip()
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

        # Put volume-owned artifact bodies back into the project tree before
        # anything reads it. In a deployed container the tree comes back
        # image-fresh on every recreation, while the operator's claimed
        # versions live on the claude-config volume; without this the agent
        # would run the framework's originals while the gallery showed the
        # operator's, and nothing would report the divergence. No-op elsewhere.
        from osprey.interfaces.web_terminal.scaffold_gallery_service import (
            restore_scaffold_bodies,
        )

        try:
            restore_scaffold_bodies(Path(app.state.project_cwd))
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            logger.warning("Could not restore user-owned artifacts from the volume: %s", exc)

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

            # ``OSPREY_WEB_THEME`` takes precedence over ``web.theme``, so
            # several containers sharing one baked config image can each be
            # themed individually via the environment — the same shape
            # ``OSPREY_WEB_APP_NAME`` uses above. Multi-user deployments set it
            # per user from the roster's ``theme:`` key.
            configured_web_theme = os.environ.get(
                "OSPREY_WEB_THEME", ""
            ).strip() or get_config_value("web.theme", "main")
            theme_entries, theme_defaults = _load_theme_registry()
            app.state.web_theme_id = resolve_web_theme_id(
                configured_web_theme, theme_entries, theme_defaults
            )
            # Whether the configured value pinned a mode (a concrete id) or only
            # a palette (a family). Server-rendered alongside data-theme so the
            # browser hub can honor a pin instead of assuming 'auto' — see
            # resolve_web_theme_pinned_mode().
            app.state.web_theme_mode = resolve_web_theme_pinned_mode(
                configured_web_theme, theme_entries
            )
            # The resolved theme's family, kept for the rail-position block
            # below (an unconfigured rail follows the family — see
            # FAMILY_RAIL_DEFAULTS).
            app.state.web_theme_family = next(
                (entry.family for entry in theme_entries if entry.id == app.state.web_theme_id),
                None,
            )
        except Exception:  # noqa: BLE001 — never let config/theme-registry load block startup
            logger.warning(
                "Could not resolve web.theme (config or theme-registry load failed); "
                "server-rendering fallback theme 'dark'",
                exc_info=True,
            )
            app.state.web_theme_id = "dark"
            app.state.web_theme_mode = None
            app.state.web_theme_family = None

        # ── Web UI mode (SSR no-flash attribute, Task 5.1) ──
        # Resolved once at startup and server-rendered onto <html data-ui-mode>
        # so the pre-paint mode-boot script (Task 5.2) first-paints in the right
        # mode. GET /api/panels also carries ui_mode, but first paint must never
        # depend on that API field — this server-rendered attribute is the
        # authoritative first-paint rung. Read via load_osprey_config (the same
        # top-level `web` section the panel loaders in this file use). Fails open
        # to the default mode on any config-read error.
        try:
            from osprey.utils.workspace import load_osprey_config

            configured_ui_mode = load_osprey_config().get("web", {}).get("ui_mode", DEFAULT_UI_MODE)
            app.state.web_ui_mode = resolve_ui_mode(configured_ui_mode)
        except Exception:  # noqa: BLE001 — never let config load block startup
            logger.warning(
                "Could not resolve web.ui_mode (config load failed); "
                "server-rendering fallback mode %r",
                DEFAULT_UI_MODE,
                exc_info=True,
            )
            app.state.web_ui_mode = DEFAULT_UI_MODE

        # ── Rail position (SSR no-flash attribute) ──
        # Same shape as web.ui_mode above: resolved once at startup and
        # server-rendered onto <html data-rail-position> so the pre-paint
        # rail-boot script first-paints the right rail orientation. GET
        # /api/panels also carries rail_position, but first paint must never
        # depend on that API field — this attribute is the authoritative rung.
        # Fails open to the default position on any config-read error.
        try:
            from osprey.utils.workspace import load_osprey_config

            configured_rail = load_osprey_config().get("web", {}).get("rail_position")
            app.state.web_rail_position = resolve_rail_position(
                configured_rail, getattr(app.state, "web_theme_family", None)
            )
            # Whether the deployment stated a position of its own. The browser
            # needs this to know if a live theme-family switch may move the
            # rail: an explicit config value outranks the family default.
            app.state.web_rail_position_configured = configured_rail in RAIL_POSITIONS
        except Exception:  # noqa: BLE001 — never let config load block startup
            logger.warning(
                "Could not resolve web.rail_position (config load failed); "
                "server-rendering fallback position %r",
                DEFAULT_RAIL_POSITION,
                exc_info=True,
            )
            app.state.web_rail_position = DEFAULT_RAIL_POSITION
            app.state.web_rail_position_configured = False

        # ── Regenerate stale Claude Code artifacts on launch ──
        # config.yml is a build-time input: safety-critical fields (e.g. the
        # writes_enabled kill-switch baked into settings.json's permissions.deny)
        # only take effect once the artifacts are re-rendered. Regenerating here
        # — mirroring `osprey chat` — means an edited config.yml is honored
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
        from osprey.build.claude_code_resolver import (
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
            from osprey.utils.workspace import repo_root_for_config

            _project_dir = Path(app.state.config_path).parent
            # load_provider_spec expands ${VAR} in provider config before
            # resolving, and it does so against the REPO root's .env — the
            # deployment's secret store, which deliberately does not live in
            # the render `osprey build` re-creates from scratch. Reading the
            # spec from <repo>/build/config.yml while expanding from <repo>/.env
            # is the same split the dispatch worker uses; without it a custom
            # provider's `base_url: ${ARGO_PROD_URL}` starts the terminal
            # pointed at a literal placeholder. In a container the two coincide
            # (the project dir IS the render), which repo_root_for_config
            # answers with the same call.
            _env_dir = repo_root_for_config(app.state.config_path)
            _spec = load_provider_spec(_project_dir, env_dir=_env_dir)
            if _spec:
                # The SPEC is read from the render; the ENV is read from the
                # repo. inject_provider_env's bulk `.env` passthrough is given
                # the repo root for the same reason load_provider_spec is given
                # it above: `<repo>/build/.env` is a file no build ever writes,
                # so pointing the injection at the render made the whole layer
                # a no-op — every key it was supposed to publish silently
                # absent. Every other launch path (chat, the dispatch worker)
                # already reads the repo's `.env`; this is web joining them.
                # Inside a container the two directories coincide, so nothing
                # about the deployed case changes.
                inject_provider_env(os.environ, _spec, project_dir=_env_dir)

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

        # The watcher's default follows the deployment's CONFIGURED agent-data
        # root, anchored on the repo. The literal it replaced — `./_agent_data`
        # — was wrong twice over: it named a directory the three-zone layout
        # retired (state lives under `var/`), and it anchored on the working
        # directory, so a terminal started from anywhere but the repo root
        # watched a path that did not exist and reported no sessions at all.
        #
        # Deliberately NOT resolve_agent_data_root(): that applies
        # OSPREY_SESSION_ID isolation, and this watcher's whole job is to see
        # EVERY session's directory rather than one.
        if config.get("watch_dir"):
            workspace_dir = Path(config["watch_dir"]).resolve()
        else:
            from osprey.utils.workspace import (
                agent_data_base_dir,
                anchored_path,
                load_osprey_config,
                resolve_project_root,
            )

            _osprey_config = load_osprey_config()
            workspace_dir = anchored_path(
                agent_data_base_dir(_osprey_config), resolve_project_root(_osprey_config)
            ).resolve()
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
        if "system-health" in enabled_panels:
            _launch_system_health_server(app)

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

        # ── Idle chat-session reaper (Task 1.7) ──
        # Periodically evicts idle chat sessions (per the registry's idle
        # predicate, which also collects zombie-busy sessions) so an abandoned
        # Simple-mode tab does not pin a pool slot indefinitely. Fail-open at
        # every level: a per-cycle exception is swallowed+logged, and the whole
        # task is wrapped so a reaper crash can never take the app down. Interval
        # is idle_timeout/4, clamped to [30s, 300s].
        reap_interval = max(30.0, min(chat_idle_timeout_s / 4.0, 300.0))
        registry = app.state.operator_registry

        async def _reap_idle_chats() -> None:
            while True:
                await asyncio.sleep(reap_interval)
                try:
                    reaped = await registry.reap_idle_chat_sessions()
                    if reaped:
                        logger.info("Idle chat reaper evicted %d session(s)", reaped)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — one bad cycle must not kill the reaper
                    logger.warning("Idle chat reaper cycle failed", exc_info=True)

        reaper_task = asyncio.create_task(_reap_idle_chats())

        yield

        reaper_task.cancel()
        with suppress(asyncio.CancelledError):
            await reaper_task

        await app.state.proxy_client.aclose()

        # Stop translation proxy if it was started
        from osprey.infrastructure.proxy.lifecycle import stop_proxy

        stop_proxy()

        app.state.watcher.stop()
        app.state.pty_registry.cleanup_all()
        await app.state.operator_registry.cleanup_all()

    return lifespan


async def _scaffold_claim_conflict(request: Request, exc: Exception) -> JSONResponse:
    """Render a refused claim as a 409 carrying the refusal verbatim.

    A refused claim is a conflict with the state of the project, and the
    message names what to do about it — so it is surfaced as written rather
    than being let through to become a bare 500 with the message stripped.
    Saving over a generated file raises the same refusal in the same words,
    naming the channel that actually owns the file, and reads the same way.

    Args:
        request: The request whose route raised. Unused; part of the Starlette
            handler signature.
        exc: The :class:`~osprey.cli.scaffold_cmd.ScaffoldClaimError` raised.

    Returns:
        A 409 whose ``detail`` is the exception's message.
    """
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def _ownership_store_conflict(request: Request, exc: Exception) -> JSONResponse:
    """Render a store that would not take the write as a 409.

    Nothing was recorded, so this must not read as success. Surfacing the
    reason beats the bare 500 an uncaught store error would otherwise give.

    Args:
        request: The request whose route raised. Unused; part of the Starlette
            handler signature.
        exc: The
            :class:`~osprey.interfaces.web_terminal.ownership.OwnershipStoreError`
            raised.

    Returns:
        A 409 whose ``detail`` is the exception's message.
    """
    return JSONResponse(status_code=409, content={"detail": str(exc)})


def register_scaffold_conflict_handlers(app: FastAPI) -> None:
    """Translate the scaffold family's two refusal types into 409s app-wide.

    Both exceptions mean one thing to the browser — the write was refused and
    the message says why — and both can come out of most of the gallery's
    write routes. Handling them once here keeps each route to the translations
    that genuinely differ per endpoint. Only the scaffold routes reach the code
    that raises either, so registering them on the app narrows to that family
    in practice.

    Args:
        app: The application to register the handlers on.
    """
    app.add_exception_handler(ScaffoldClaimError, _scaffold_claim_conflict)
    app.add_exception_handler(OwnershipStoreError, _ownership_store_conflict)


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
    register_scaffold_conflict_handlers(app)

    @app.get("/")
    async def root(request: Request):
        app_name = getattr(request.app.state, "app_name", "")
        web_theme_id = getattr(request.app.state, "web_theme_id", "dark")
        web_theme_mode = getattr(request.app.state, "web_theme_mode", None)
        web_ui_mode = getattr(request.app.state, "web_ui_mode", DEFAULT_UI_MODE)
        web_rail_position = getattr(request.app.state, "web_rail_position", DEFAULT_RAIL_POSITION)
        terminal_user = getattr(request.app.state, "terminal_user", "")
        landing_url = getattr(request.app.state, "landing_url", "")
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "app_name": app_name,
                "web_theme_id": web_theme_id,
                "web_theme_mode": web_theme_mode or "",
                "web_ui_mode": web_ui_mode,
                "web_rail_position": web_rail_position,
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
