"""OSPREY Web Terminal — FastAPI Application.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
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
from osprey.interfaces.common_middleware import apply_url_prefix, compute_url_prefix
from osprey.interfaces.vendor import vendor_url
from osprey.interfaces.web_terminal.file_watcher import (
    FileEventBroadcaster,
    WorkspaceWatcher,
    resolve_store_rel,
)
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry
from osprey.interfaces.web_terminal.ownership import OwnershipStoreError
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes import router
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX
from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS
from osprey.registry.web import PANEL_ID_TO_REGISTRY_KEY, panel_url_state_attr

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


def _launch_enabled_panel_servers(app: FastAPI, enabled_panels: set[str]) -> None:
    """Launch the companion server behind every enabled built-in panel.

    The lifespan used to gate each panel on its own ``if "<id>" in
    enabled_panels`` line, which made registering a companion server a two-place
    edit — and left the universal/domain split spelled a second time here, out of
    sync with :func:`_load_panel_config`, which already folds
    ``UNIVERSAL_PANELS`` into ``enabled_panels`` unconditionally.

    Args:
        app: The web-terminal application whose ``state`` the URLs are published on.
        enabled_panels: Enabled panel ids from :func:`_load_panel_config`. Ids with
            no companion server behind them (URL-backed custom panels such as
            ``events``) are skipped — the framework serves nothing for those.
    """
    for panel_id in sorted(enabled_panels & BUILTIN_PANELS):
        _launch_panel_server(app, PANEL_ID_TO_REGISTRY_KEY[panel_id])


def _launch_panel_server(app: FastAPI, key: str) -> None:
    """Auto-launch the companion web server *key* and publish its panel URL.

    One implementation for all six panels. The URL published here is the *only*
    input to panel availability (``/api/<panel>-server`` and the ``/panel/<id>/``
    proxy), so it must never outrun the server itself: ``auto_launch: false``
    makes the launch a no-op, and publishing a URL anyway left the operator an
    enabled tab whose iframe returned a bare 502. Leaving it unset disables the
    tab instead, which is what a suppressed server should look like.

    Both the gate and the address come from the shared registry derivations —
    :func:`~osprey.infrastructure.server_launcher.is_auto_launch_enabled` and
    :func:`~osprey.registry.web.resolve_web_server_base_url`, the same ones
    ``ServerLauncher`` itself uses. That is what keeps the port advertised here
    identical to the port uvicorn binds, including the set-but-empty env
    override (compose ``OSPREY_<CONFIG_KEY>_PORT=``) that an inline ``int()``
    turns into a ValueError and a silently dead tab.

    Args:
        app: The web-terminal application whose ``state`` the URL is published on.
        key: Key into ``registry.web.FRAMEWORK_WEB_SERVERS``, e.g. ``"artifact"``.
    """
    attr = panel_url_state_attr(key)
    try:
        from osprey.infrastructure.server_launcher import (
            ensure_web_server,
            is_auto_launch_enabled,
        )
        from osprey.registry.web import FRAMEWORK_WEB_SERVERS, resolve_web_server_base_url

        definition = FRAMEWORK_WEB_SERVERS[key]

        # Covers require_section too: a server whose config section is absent
        # reports auto-launch off, so a panel with nothing behind it stays dark.
        if not is_auto_launch_enabled(key):
            logger.info(
                "%s auto-launch is disabled; the %s panel is unavailable",
                definition.name,
                definition.panel_id,
            )
            setattr(app.state, attr, None)
            return

        setattr(app.state, attr, resolve_web_server_base_url(key))
        ensure_web_server(key)
        logger.info("%s available at %s", definition.name, getattr(app.state, attr))
    except Exception:
        logger.warning("Could not auto-launch companion server %r", key, exc_info=True)
        # No fallback URL: a launch we could not complete must not be advertised
        # as an available panel. Retract any URL assigned before the failure.
        setattr(app.state, attr, None)


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


#: The two supported rail positions. ``left`` is the icon-rail column;
#: ``top`` renders the same rail as a horizontal strip under the header.
RAIL_POSITIONS = ("left", "top")
DEFAULT_RAIL_POSITION = "left"

#: Per-theme-family rail defaults, applied only when ``web.rail_position``
#: is absent from config. The ``retro`` family carries a horizontal tab strip
#: under the header as part of its look — picking Retro without also moving
#: the rail would give its colors a layout they were never drawn for. Any
#: family not listed here defaults to :data:`DEFAULT_RAIL_POSITION`.
#:
#: This map is the single source of truth for the coupling: ``GET
#: /api/panels`` echoes it to the browser so ``rail-position.js`` can follow
#: a live family switch without carrying its own copy.
FAMILY_RAIL_DEFAULTS: dict[str, str] = {"retro": "top"}

#: Where the Documentation control in the rail's utility cluster points when
#: ``web.docs_url`` is absent. A facility hosting its own copy of the docs
#: overrides the key; the default is the published site.
DEFAULT_DOCS_URL = "https://als-apg.github.io/osprey"

#: ``owner/repo`` used to build the prefilled new-issue URL when
#: ``web.feedback.github_repo`` is absent.
DEFAULT_FEEDBACK_GITHUB_REPO = "als-apg/osprey"

#: Recipient of the prefilled ``mailto:`` draft when ``web.feedback.email``
#: is absent — the OSPREY maintainers.
DEFAULT_FEEDBACK_EMAIL = "thellert@lbl.gov"

#: Ceiling (bytes) on the on-disk feedback store before the oldest saved
#: contexts are pruned; 256 MB unless ``web.feedback.max_store_bytes`` says
#: otherwise. Submission headers are never pruned, only their contexts.
DEFAULT_FEEDBACK_MAX_STORE_BYTES = 256 * 1024 * 1024


def coerce_config_str(key: str, value: object, default: str) -> str:
    """Return a configured string value, falling back to *default*.

    An **absent** key means "use the default": the caller reads it with
    *default* already in hand, so a facility with no ``config.yml`` still gets
    working Documentation and Feedback controls.

    An **explicitly blank** value (``docs_url: ""``) means "this deployment has
    no such target", and is returned as ``""``. That posture is what the rail
    anchor, the status-bar link and the dialog's channel guard are built on: an
    air-gapped control room blanks ``web.docs_url`` and gets no documentation
    link rather than one that opens a dead tab, and blanking
    ``web.feedback.github_repo`` retires the GitHub channel instead of aiming it
    at the upstream maintainers' tracker. Folding blank back into the default
    would make that whole posture unreachable while the UI kept claiming it.

    A YAML key written with no value at all (``docs_url:``, i.e. ``None``) reads
    as absent, not as blank — "I have not decided" rather than "there is none".

    A value of some other type (a nested mapping from a mis-indented
    ``config.yml``, say) is reported and discarded rather than repr'd into an
    ``href`` or a ``mailto:``, which would render a control that silently goes
    nowhere.

    Args:
        key: Dotted config key, used only for the warning message.
        value: Whatever the config reader returned.
        default: The shipped default for *key*.

    Returns:
        The stripped configured string (possibly ``""``), or *default*.
    """
    if isinstance(value, str):
        return value.strip()
    if value is not None:
        logger.warning("%s is %r, not a string; using %r instead", key, value, default)
    return default


def coerce_store_ceiling(value: object, default: int = DEFAULT_FEEDBACK_MAX_STORE_BYTES) -> int:
    """Return ``web.feedback.max_store_bytes`` as a positive byte count.

    Guarded rather than trusted: the pruner deletes stored contexts until the
    store fits under this number, so a ``0``, a negative, or a ``True`` that
    ``int()`` would happily turn into ``1`` would empty the store on the next
    submission while looking like ordinary pruning. A human-written ``256MB``
    is rejected the same way — this key is a plain byte count.

    Args:
        value: Whatever the config reader returned.
        default: The shipped ceiling to fall back to.

    Returns:
        A positive integer byte ceiling.
    """
    if value is None:
        return default
    if not isinstance(value, bool):
        try:
            ceiling = int(value)
        except (TypeError, ValueError, OverflowError):
            # OverflowError is not hypothetical: YAML parses `.inf` and any
            # overflowing exponent (1.0e+400) to float("inf"), which int()
            # refuses. This helper is called outside the lifespan's try, so an
            # escaping exception would abort startup outright.
            ceiling = 0
        if ceiling > 0:
            return ceiling
    logger.warning(
        "web.feedback.max_store_bytes is %r, not a positive byte count; using %d",
        value,
        default,
    )
    return default


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

        # ── Simple-mode chat pool bounds ──
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
        # Bounded history of agent-activity events. The SSE stream only reaches
        # browsers that are already connected, so the ring is what a browser
        # opened (or reloaded) mid-session reads to catch up on recent actions.
        app.state.agent_activity_ring = deque(maxlen=ACTIVITY_RING_MAX)
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
        # config key exists for either of these, so the config-side
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

        # ── Web theme (SSR no-FOUC attribute) ──
        # Resolved once at startup and server-rendered onto <html data-theme>
        # so the generated theme-boot.js first-paints with no
        # flash. Fails open on any load error — a missing/broken theme
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

        # ── Web UI mode (SSR no-flash attribute) ──
        # Resolved once at startup and server-rendered onto <html data-ui-mode>
        # so the pre-paint mode-boot script first-paints in the right
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
        from osprey.build.claude_code_telemetry import (
            ObservabilityCredentialError,
            telemetry_creds_are_store_issued,
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
            try:
                _spec = load_provider_spec(_project_dir, env_dir=_env_dir)
            except ObservabilityCredentialError as exc:
                # Keep this arm ahead of any broader one added later: it
                # subclasses ValueError.
                #
                # Resolving the provider resolves the telemetry block with it,
                # so a store-issued credential no deploy has minted yet arrives
                # here as a failure to read the provider — and the server exits
                # at startup over it. A terminal nobody can open is the worse
                # outcome: serve without telemetry and say so. Anything else —
                # a credential an operator has to set, or one that is simply
                # blank — keeps raising and still refuses the start.
                if not telemetry_creds_are_store_issued(exc):
                    raise
                logger.warning(
                    "Telemetry is off for this server — `osprey up` issues %s when it starts "
                    "the telemetry store and passes them into each terminal container, so "
                    "this server is either running outside a deployment or against a store "
                    "that was never started",
                    ", ".join(exc.unresolved_vars),
                )
                _spec = load_provider_spec(_project_dir, env_dir=_env_dir, include_telemetry=False)
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
        # root, anchored on the repo. Never a cwd-relative literal: state lives
        # under `var/`, and a path anchored on the working directory means a
        # terminal started from anywhere but the repo root watches a directory
        # that does not exist and reports no sessions at all.
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

        # ── Documentation + feedback controls ──
        # Read once here (the web.ui_mode / web.rail_position pattern) and
        # echoed to the browser by GET /api/panels. Every read fails open: a
        # facility with no config.yml still gets working Documentation and
        # Feedback controls pointed at the project defaults.
        #
        # The four raw values are read together — a failure here means the
        # config is unreadable, which really does concern all four — but each
        # is validated SEPARATELY below. A single unusable value (a hand-written
        # "256MB" ceiling, say) must not drag the others back to project
        # defaults: silently redirecting a facility's feedback address to the
        # upstream maintainer is exactly the failure a fail-open path must not
        # produce.
        try:
            from osprey.utils.config import get_config_value

            raw_docs_url = get_config_value("web.docs_url", DEFAULT_DOCS_URL)
            raw_github_repo = get_config_value(
                "web.feedback.github_repo", DEFAULT_FEEDBACK_GITHUB_REPO
            )
            raw_email = get_config_value("web.feedback.email", DEFAULT_FEEDBACK_EMAIL)
            raw_max_store_bytes = get_config_value(
                "web.feedback.max_store_bytes", DEFAULT_FEEDBACK_MAX_STORE_BYTES
            )
        except Exception:  # noqa: BLE001 — never let config load block startup
            logger.warning(
                "Could not read web.docs_url / web.feedback.* config keys; using defaults",
                exc_info=True,
            )
            raw_docs_url = raw_github_repo = raw_email = raw_max_store_bytes = None
        app.state.docs_url = coerce_config_str("web.docs_url", raw_docs_url, DEFAULT_DOCS_URL)
        app.state.feedback_github_repo = coerce_config_str(
            "web.feedback.github_repo", raw_github_repo, DEFAULT_FEEDBACK_GITHUB_REPO
        )
        app.state.feedback_email = coerce_config_str(
            "web.feedback.email", raw_email, DEFAULT_FEEDBACK_EMAIL
        )
        app.state.feedback_max_store_bytes = coerce_store_ceiling(raw_max_store_bytes)

        # The feedback store is sited on the CONFIGURED agent-data root and
        # deliberately NOT on workspace_dir: with web_terminal.watch_dir set,
        # the watched tree is somewhere else entirely and records written
        # there would land outside the {user}-agent-data volume, where
        # `osprey feedback` on the host cannot reach them. Never
        # resolve_agent_data_root() either — that appends sessions/<id>, and
        # the store spans sessions.
        try:
            from osprey.utils.workspace import resolve_shared_data_root

            feedback_dir = resolve_shared_data_root() / "feedback"
        except Exception:  # noqa: BLE001 — never let config load block startup
            feedback_dir = workspace_dir / "feedback"
            logger.warning(
                "Could not resolve the shared data root for the feedback store; falling back to %s",
                feedback_dir,
                exc_info=True,
            )
        app.state.feedback_dir = feedback_dir
        # Workspace-relative form of the store: the *file watcher's* form, used
        # below to drop change events for feedback writes. The file browser is
        # not a consumer — routes/files.py derives its own predicate from
        # ``feedback_dir``, because it must also handle symlink aliases and
        # session-scoped roots that a single relative path cannot express.
        # ``None`` when the store lies outside the watched tree (the watch_dir
        # case above) — nothing to conceal there, and a bare relative_to()
        # would raise and abort startup. The derivation is case-folded on a
        # case-insensitive filesystem, where the store root and watch_dir can
        # spell one directory two ways: an exact comparison would yield
        # ``None`` there and silently disable the watcher's concealment.
        # The derivation probes the filesystem, so the directory has to exist
        # first. ``WorkspaceWatcher.start()`` creates it a few lines below, but
        # that is too late: on a first-ever startup the probe would read an
        # absent directory, yield ``None``, and leave concealment off for the
        # life of the process. Suppressed because the watcher's own mkdir stays
        # the authority on a genuine failure here.
        with suppress(OSError):
            workspace_dir.mkdir(parents=True, exist_ok=True)
        app.state.feedback_rel = resolve_store_rel(feedback_dir, workspace_dir)

        app.state.watcher = WorkspaceWatcher(
            workspace_dir, app.state.broadcaster, feedback_rel=app.state.feedback_rel
        )
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
        # human applies in one click. Immutable config-derived state. Empty
        # (the default) → the "+" menu renders no presets section.
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

        _launch_enabled_panel_servers(app, enabled_panels)

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

        # ── Idle chat-session reaper ──
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

    # session.html is otherwise a plain static file under STATIC_DIR (served
    # verbatim by the /static mount below); this route shadows that mount for
    # exactly that path so it, too, gets the Jinja-rendered prefix injection.
    # Must be registered before configure_interface_app() mounts /static
    # (Starlette matches routes in registration order, so an explicit route
    # ahead of a Mount wins).
    #
    # A ``?token=`` arriving here is answered by WebAuthMiddleware before this
    # route runs (it mints the session cookie and redirects to the clean URL),
    # so the handler only ever renders the page.
    @app.get("/static/session.html")
    async def session_page(request: Request):
        return templates.TemplateResponse(request, "session.html", {"url_prefix": url_prefix})

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
    *,
    browser_url: str | None = None,
) -> None:
    """Run the web terminal server.

    Args:
        host: Host to bind to.
        port: Port to run on.
        shell_command: Shell command to spawn in the PTY.
        config_path: Optional path to config file.
        project_dir: Optional OSPREY project directory.
        browser_url: The URL to auto-open once the server answers. Defaults to
            the bare ``http://<host>:<port>``. The single-user launcher passes
            the operator's one-time ``?token=`` login URL here instead: the bare
            URL sets no session cookie, so an auto-opened tab would land on the
            login-required page, whereas the token URL exchanges for a cookie and
            redirects to the clean URL. Keyword-only and defaulted so every other
            caller keeps the bare-URL behavior unchanged.
    """
    import uvicorn

    _open_browser_when_ready(browser_url or f"http://{host}:{port}")

    app = create_app(config_path=config_path, shell_command=shell_command, project_dir=project_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
