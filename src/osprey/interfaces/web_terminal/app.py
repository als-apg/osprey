"""OSPREY Web Terminal — FastAPI Application.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import shlex
import sys
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context

from osprey.apps.manifest import launch_token_env
from osprey.cli.scaffold_cmd import ScaffoldClaimError
from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.common_middleware import (
    UNSAFE_FORWARDED_VALUE,
    apply_url_prefix,
    compute_url_prefix,
    forwarded_identity,
    resolve_storage_scope,
)
from osprey.interfaces.vendor import vendor_url
from osprey.interfaces.web_auth import set_panel_agent_routes
from osprey.interfaces.web_terminal.control_context_owner import start_control_context_owner
from osprey.interfaces.web_terminal.feedback_destination import (
    DEFAULT_DOCS_URL,
    osprey_version,
    resolve_deployment_identity,
    resolve_feedback_destination,
    upstream_escalation_url,
)
from osprey.interfaces.web_terminal.file_watcher import (
    FileEventBroadcaster,
    WorkspaceWatcher,
    resolve_store_rel,
)
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry
from osprey.interfaces.web_terminal.ownership import OwnershipStoreError
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes import router
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_BUFFER_MAX
from osprey.interfaces.web_terminal.routes.chat_attachments import sweep_uploads_at_startup
from osprey.interfaces.web_terminal.routes.proxy import configured_agent_routes
from osprey.interfaces.web_terminal.routes.watchdog import make_notifier
from osprey.interfaces.web_terminal.routes.watchdog import router as watchdog_router
from osprey.interfaces.web_terminal.sidecar_status import (
    PANEL_STATUS_DIRNAME,
    SidecarStatus,
    clear_status,
    failure_reason,
    write_status,
)
from osprey.interfaces.web_terminal.transcript_map import load as load_transcript_map
from osprey.port_layout import default_port
from osprey.profiles.web_panels import (
    BUILTIN_PANEL_LABELS,
    BUILTIN_PANELS,
    SIDECAR_PANELS,
    UNIVERSAL_PANELS,
    panel_id_refusal,
    panel_spec_enabled,
)
from osprey.registry.web import PANEL_ID_TO_REGISTRY_KEY, panel_url_state_attr
from osprey.services.watchdog.definitions import load_watchdogs
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from osprey.utils.seconds import positive_seconds
from osprey.utils.workspace import session_hold_dir
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.session_hold import HOLD_DIR_ENV_VAR
from osprey_connectors.types import WRITES_KEY
from osprey_connectors.workspace import repo_root_for_config

if TYPE_CHECKING:
    import socket
    from collections.abc import AsyncIterator

    from jinja2.runtime import Context

STATIC_DIR = Path(__file__).parent / "static"

#: The tab icon, in the design system's static tree.
_FAVICON = Path(__file__).parent.parent / "design_system" / "static" / "img" / "osprey-mark.svg"


def _favicon_url() -> str:
    """The tab icon's root-relative URL, versioned by the icon's content.

    Firefox keeps a page's tab icon in its own store, keyed by the icon's URL,
    for days, and does not ask the server again while that entry lasts, whatever
    the response's cache headers said. An edited mark therefore reaches a
    browser that has seen the old one only at a new address: the hash makes
    every edit one.
    """
    digest = hashlib.sha256(_FAVICON.read_bytes()).hexdigest()[:12]
    return f"/design-system/img/{_FAVICON.name}?v={digest}"


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
templates.env.globals["favicon_url"] = _favicon_url()

logger = __import__("logging").getLogger(__name__)


def _launch_enabled_panel_servers(app: FastAPI, enabled_panels: set[str]) -> None:
    """Launch the companion server behind every enabled built-in panel.

    The lifespan used to gate each panel on its own ``if "<id>" in
    enabled_panels`` line, which made registering a companion server a two-place
    edit — and left the universal/domain split spelled a second time here, out of
    sync with :func:`_load_panel_config`, which already folds
    ``UNIVERSAL_PANELS`` into ``enabled_panels`` unconditionally.

    Sidecar panel ids are diverted before the registry lookup: a sidecar is in
    no companion registry and has no key to look up, so indexing
    ``PANEL_ID_TO_REGISTRY_KEY`` with one raises. :func:`_launch_enabled_sidecars`
    starts those, from the same ``enabled_panels`` set.

    Args:
        app: The web-terminal application whose ``state`` the URLs are published on.
        enabled_panels: Enabled panel ids from :func:`_load_panel_config`. Ids with
            no companion server behind them (URL-backed custom panels such as
            ``events``) are skipped — the framework serves nothing for those.
    """
    for panel_id in sorted(enabled_panels & (BUILTIN_PANELS - set(SIDECAR_PANELS))):
        _launch_panel_server(app, PANEL_ID_TO_REGISTRY_KEY[panel_id])


#: The config key that sets how long a sidecar gets to answer its own status
#: endpoint after it is spawned.
SIDECAR_READY_TIMEOUT_KEY = "web.sidecar_ready_timeout_s"

#: The wait in seconds when the key is unset or unusable. Generous: the first
#: launch on a cold deployment builds the panel's assets.
DEFAULT_SIDECAR_READY_TIMEOUT_S = 60.0


def resolve_sidecar_ready_timeout(value: object) -> float:
    """Return the sidecar startup wait in seconds that *value* configures.

    ``None`` (the key unset or set to null) gives the default. A usable value is
    a finite number greater than zero, or a string that parses as one, since
    ``${VAR}`` interpolation hands the reader strings. A ``bool`` is refused
    although ``float(True)`` is ``1.0``: a one-second wait is never what
    ``true`` meant. Anything unusable logs one warning naming the key and the
    value, and the default applies; the reader never raises, because one tab's
    wait must not block the terminal's start.

    Args:
        value: The raw value of ``web.sidecar_ready_timeout_s``.

    Returns:
        The wait in seconds.
    """
    if value is None:
        return DEFAULT_SIDECAR_READY_TIMEOUT_S
    seconds = positive_seconds(value)
    if seconds is not None:
        return seconds
    logger.warning(
        "%s is %r, not a positive number of seconds; using %s s",
        SIDECAR_READY_TIMEOUT_KEY,
        value,
        DEFAULT_SIDECAR_READY_TIMEOUT_S,
    )
    return DEFAULT_SIDECAR_READY_TIMEOUT_S


async def _launch_enabled_sidecars(app: FastAPI, enabled_panels: set[str]) -> None:
    """Start every enabled sidecar panel and publish what the proxy needs.

    The async peer of :func:`_launch_enabled_panel_servers`. Split from it
    rather than folded in because a sidecar launch blocks — an interpreter
    probe, then a server that has to come up — and the lifespan's loop is
    already running by the time this is reached.

    Args:
        app: The web-terminal application whose ``state`` the URLs are published on.
        enabled_panels: Enabled panel ids from :func:`_load_panel_config`.
    """
    for panel_id in sorted(enabled_panels & set(SIDECAR_PANELS)):
        await _launch_sidecar(app, panel_id)


async def _launch_sidecar(app: FastAPI, panel_id: str) -> None:
    """Start the sidecar behind *panel_id* and publish its URL and credential.

    The class comes from :data:`~osprey.profiles.web_panels.SIDECAR_PANELS` as a
    dotted ``module:attribute`` path, resolved here the way
    :func:`_launch_panel_server` resolves a registry key — so adding a sidecar
    is an entry in that registry and nothing in this module.

    Publishing is all-or-nothing, and it happens only once the sidecar answers:
    the panel URL (``app.state.<id>_server_url``) is the sole input to panel
    availability, and the credential (``app.state.panel_auth_headers[<id>]``) is
    what the proxy re-issues on the operator's behalf. A launch that failed
    anywhere leaves the URL unset, which is what a switched-off panel looks like.

    The blocking halves run in a worker thread. ``spawn()`` blocks only for the
    fork, so it stays on the loop. The readiness wait is
    ``app.state.sidecar_ready_timeout_s``, which the lifespan resolves from
    ``web.sidecar_ready_timeout_s``; every launch of a sidecar reads it there.

    Every outcome is recorded (:func:`_record_sidecar_status`): ``starting``
    before the sidecar is built, then ``running`` or ``failed`` with a one-line
    reason. The object is registered in ``app.state.sidecars`` as soon as it is
    built, so shutdown reaches a sidecar that is still coming up; a failed
    launch removes it again.

    A sidecar that dies later is retracted the same way it was published: its
    ``on_exit`` hook, which the sidecar fires from its own watcher thread, hands
    the loop a callback that clears the URL, drops the credential and records
    the sidecar as failed, so the availability route answers unavailable at
    once. The sidecar logs the exit itself, with its stderr tail. Nothing
    restarts it on its own: :func:`request_sidecar_start` starts a fresh one
    when the operator opens the panel, and until then the dead object stays in
    ``app.state.sidecars`` so shutdown still removes its per-launch tempdir.

    Args:
        app: The web-terminal application whose ``state`` the URL is published on.
        panel_id: The sidecar's panel id, a key of ``SIDECAR_PANELS``.
    """
    attr = panel_url_state_attr(panel_id)
    setattr(app.state, attr, None)
    _record_sidecar_status(app, panel_id, SidecarStatus.starting())
    sidecar = None
    try:
        from osprey.interfaces.web_terminal.operator_session import resolve_agent_data_root

        module_path, _, attribute = SIDECAR_PANELS[panel_id].factory_path.partition(":")
        factory = getattr(importlib.import_module(module_path), attribute)
        sidecar = factory(
            Path(resolve_agent_data_root(app)),
            compute_url_prefix(),
            getattr(app.state, "web_theme_mode", None),
        )
        app.state.sidecars[panel_id] = sidecar
        await asyncio.to_thread(sidecar.preflight)
        if app.state.sidecars_closing:
            raise RuntimeError("the terminal is shutting down")
        sidecar.spawn()
        timeout = getattr(app.state, "sidecar_ready_timeout_s", DEFAULT_SIDECAR_READY_TIMEOUT_S)
        await asyncio.to_thread(sidecar.wait_ready, timeout)
    except Exception as exc:  # a dead panel must not block startup
        # The readiness failures already quote the tail in their own message;
        # the preflight ones carry none, so it is appended only when it is new.
        tail = getattr(sidecar, "stderr_tail", "")
        detail = f"{exc}\n{tail}" if tail and tail not in str(exc) else str(exc)
        logger.warning("%s sidecar failed to start: %s", panel_id, detail)
        if sidecar is not None:
            await _stop_sidecar(panel_id, sidecar)
            if app.state.sidecars.get(panel_id) is sidecar:
                app.state.sidecars.pop(panel_id)
        setattr(app.state, attr, None)
        if app.state.sidecars_closing:
            return  # shutdown clears the record; a failure it caused is not one
        reason = failure_reason(str(exc), tail, getattr(sidecar, "token", None))
        _record_sidecar_status(app, panel_id, SidecarStatus.failed(reason))
        return

    app.state.panel_auth_headers[panel_id] = sidecar.auth_headers
    setattr(app.state, attr, sidecar.url)
    logger.info("%s sidecar available at %s", panel_id, sidecar.url)
    _record_sidecar_status(app, panel_id, SidecarStatus.running())

    loop = asyncio.get_running_loop()

    def retract() -> None:
        # A late exit of a sidecar that has since been replaced owns nothing.
        if app.state.sidecars.get(panel_id) is not sidecar:
            return
        setattr(app.state, attr, None)
        app.state.panel_auth_headers.pop(panel_id, None)
        reason = failure_reason(
            f"exited with status {sidecar.exit_status}", sidecar.stderr_tail, sidecar.token
        )
        _record_sidecar_status(app, panel_id, SidecarStatus.failed(reason))
        logger.info("%s panel retracted; opening its tab starts it again", panel_id)

    def on_exit() -> None:
        # Runs on the sidecar's watcher thread; ``app.state`` belongs to the loop.
        # the loop is closed: shutdown has already torn the state down
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(retract)

    # Registered last: a setter that fires when the exit already happened
    # closes the gap between readiness and this line.
    sidecar.on_exit = on_exit


async def request_sidecar_start(app: FastAPI, panel_id: str) -> SidecarStatus:
    """Start the sidecar behind *panel_id* again, one attempt at a time.

    The operator's retry, and the only one: nothing calls this on its own. A
    request while an attempt is in flight joins it, and a request for a
    published sidecar does nothing; both answer the current status. Otherwise
    one task stops the dead sidecar left in ``app.state.sidecars``, which reaps
    its tempdir and any straggling kernels, and then runs a fresh
    :func:`_launch_sidecar`. That is the same function the lifespan's first
    attempt awaits, so the retry waits exactly as long.

    Args:
        app: The web-terminal application.
        panel_id: An enabled sidecar panel id, a key of ``SIDECAR_PANELS``.

    Returns:
        The status recorded once the request is handled: ``starting`` for a
        new or joined attempt, ``running`` for a published sidecar.
    """
    tasks: dict[str, asyncio.Task[None]] = app.state.sidecar_start_tasks
    recorded: dict[str, SidecarStatus] = app.state.sidecar_status
    task = tasks.get(panel_id)
    if task is not None and not task.done():
        return recorded[panel_id]
    if getattr(app.state, panel_url_state_attr(panel_id), None):
        return recorded[panel_id]

    # Registered before anything awaits, so a second request joins this attempt.
    # Recorded here as well as by the launch, so the answer below is the new
    # attempt's and not the failure it replaces.
    dead = app.state.sidecars.pop(panel_id, None)
    _record_sidecar_status(app, panel_id, SidecarStatus.starting())
    task = asyncio.create_task(_relaunch_sidecar(app, panel_id, dead))
    tasks[panel_id] = task

    def _forget(done: asyncio.Task[None]) -> None:
        if tasks.get(panel_id) is done:
            tasks.pop(panel_id)

    task.add_done_callback(_forget)
    return recorded[panel_id]


async def _relaunch_sidecar(app: FastAPI, panel_id: str, dead: object | None) -> None:
    """Stop the dead sidecar *dead*, then launch a fresh one for *panel_id*.

    Args:
        app: The web-terminal application.
        panel_id: The sidecar's panel id.
        dead: The sidecar the failed attempt left behind, or ``None``.
    """
    if dead is not None:
        await _stop_sidecar(panel_id, dead)
    if app.state.sidecars_closing:
        return
    await _launch_sidecar(app, panel_id)


def _record_sidecar_status(app: FastAPI, panel_id: str, status: SidecarStatus) -> None:
    """Record *status* for *panel_id*: in memory for the routes, on disk for other processes.

    The disk write never raises (:func:`~osprey.interfaces.web_terminal.sidecar_status.write_status`),
    so recording cannot fail a launch.

    Args:
        app: The web-terminal application holding the in-memory copy.
        panel_id: The sidecar's panel id.
        status: The outcome to record.
    """
    app.state.sidecar_status[panel_id] = status
    write_status(app.state.panel_status_root, panel_id, status)


async def _stop_sidecar(panel_id: str, sidecar: object) -> None:
    """Stop one sidecar off the loop, and never raise doing it.

    Both callers have to survive a stop that fails: the launch path is already
    handling one failure and must still retract the panel, and the lifespan
    shutdown must reach the sidecars after this one. ``stop()`` also blocks
    while the process is reaped, so it runs in a worker thread.

    Args:
        panel_id: The sidecar's panel id, for the log line.
        sidecar: The sidecar object to stop.
    """
    try:
        await asyncio.to_thread(sidecar.stop)  # type: ignore[attr-defined]
    except Exception:  # one stuck sidecar must not block the rest
        logger.warning("Could not stop the %s sidecar", panel_id, exc_info=True)


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


#: Onboarding-tour invite policy (``web.tour``, per-user override via the
#: ``OSPREY_WEB_TOUR`` environment variable — the ``web.theme`` /
#: ``OSPREY_WEB_THEME`` shape, so a roster can arm the invite per user).
#: ``once``: the invite shows until this browser dismisses it ("Don't show
#: this again", or completing the tour). ``always``: the invite shows on
#: every load and offers no permanent dismissal — the shared read-only
#: screen case. ``never``: no automatic invite; the tour stays reachable
#: from the sidebar's More menu and the command palette.
TOUR_POLICIES = ("once", "always", "never")
DEFAULT_TOUR_POLICY = "once"


def resolve_tour_policy(configured: str | None) -> str:
    """Resolve ``web.tour`` / ``OSPREY_WEB_TOUR`` into a concrete policy.

    Mirrors the warn+fallback shape of the ``web.theme`` resolver: anything
    outside :data:`TOUR_POLICIES` — a typo, ``None``, an empty string — is
    logged as a warning and resolved to :data:`DEFAULT_TOUR_POLICY`, so a bad
    value degrades to the safe default instead of blocking server startup.

    Args:
        configured: The raw value, environment override already applied.

    Returns:
        A concrete policy string in :data:`TOUR_POLICIES`.
    """
    if configured in TOUR_POLICIES:
        return configured

    logger.warning(
        "Unknown web.tour %r (expected one of %s); falling back to %r.",
        configured,
        list(TOUR_POLICIES),
        DEFAULT_TOUR_POLICY,
    )
    return DEFAULT_TOUR_POLICY


#: How each value of the forwarded role-source header reads in the session
#: menu. The keys are the auth sidecar's closed vocabulary, spelled here
#: rather than imported for the same reason the header names are — the
#: terminal has no business pulling a service package into its import
#: closure — and ``tests/interfaces/web_terminal/test_terminal_user.py``
#: pins this key set against the sidecar's constants. Any other value
#: renders nothing at all: an unrecognised source is a source this build
#: cannot name, and a chip that named it anyway would be guessing.
_ROLE_SOURCE_LABELS: dict[str, str] = {"roster": "roster", "claim": "ID token"}


def control_target_picker_available(config_path: str | Path | None) -> bool:
    """Whether this deployment's sidebar offers the control-target block.

    Read from ``web.control_target_picker``, default ``True``. A render whose
    terminal reaches no machine sets it ``false``, and the chip and its popover
    are then absent.

    A config that cannot be read keeps the picker. This is a display setting,
    not a privilege gate: it grants and removes no access, so the shipped
    default is the honest answer when the file says nothing legible.
    :func:`resolve_privilege_gates` closes on the same failure because it
    guards edits.

    Args:
        config_path: The project config, or None for the default lookup.

    Returns:
        False only when the config says ``false``.
    """
    try:
        raw = _load_web_ui_config(config_path).get("control_target_picker")
    except Exception:
        logger.warning(
            "web.control_target_picker could not be read; showing the control-target picker."
        )
        return True
    return coerce_config_flag("web.control_target_picker", raw, True)


#: Words a human writes when they mean a boolean but YAML kept a string —
#: ``enabled: "false"`` (quoted) is the one that matters, because for a
#: default-ON switch a bare ``bool("false")`` is ``True``: the deployment
#: would read as having asked for the switch OFF and get it left ON.
_FALSE_WORDS = frozenset({"false", "no", "off", "0"})
_TRUE_WORDS = frozenset({"true", "yes", "on", "1"})


@dataclass(frozen=True)
class PrivilegeGates:
    """The two privilege gates a surface resolves out of its config file.

    Attributes:
        config_panel_enabled: Whether the Config panel's server surface
            (``/api/config`` and ``/api/claude-setup``) is live.
        scaffold_write_enabled: Whether the scaffold gallery's write and delete
            routes under ``/api/scaffold`` are live.
        config_unreadable_path: The resolved config file that exists but
            could not be read, or ``None``.
    """

    config_panel_enabled: bool
    scaffold_write_enabled: bool
    config_unreadable_path: Path | None = None


def resolve_privilege_gates(config_path: str | Path | None) -> PrivilegeGates:
    """Read both privilege gates out of the config file a surface resolved.

    No file resolved means the shipped defaults, read out of nothing: a
    privilege gate answered out of a file the surface was not pointed at is
    worse than one at its default. A resolved file that cannot be read closes
    both gates, because an unreadable config is not permission to author what
    the agent obeys. A readable file holding a value nobody can interpret takes
    the default through :func:`coerce_config_flag`.

    Both keys are read in one pass, so one file gives one answer for both
    gates and a failure is reported once. A surface refusing because the file
    could not be read words it with :func:`unreadable_config_refusal`.

    Args:
        config_path: The config file this surface resolved, or ``None``.

    Returns:
        The two gates, and the unreadable file when there is one.
    """
    if config_path is None:
        return PrivilegeGates(True, True)
    from osprey_connectors.config import get_config_value

    try:
        panel_raw = get_config_value("web.config_panel.enabled", True, str(config_path))
        scaffold_raw = get_config_value(
            "web.scaffold_gallery.write_enabled", True, str(config_path)
        )
    except Exception:
        logger.error(
            "Could not read the config file %s; the Config panel and scaffold gallery "
            "writes are closed until it reads cleanly",
            config_path,
            exc_info=True,
        )
        return PrivilegeGates(False, False, Path(config_path))
    return PrivilegeGates(
        coerce_config_flag("web.config_panel.enabled", panel_raw, True),
        coerce_config_flag("web.scaffold_gallery.write_enabled", scaffold_raw, True),
    )


#: The refusal every terminal-surface route gives when ``web.terminal`` is
#: false. One sentence, so the websocket's error frame and the log say the same.
TERMINAL_REFUSAL = "The terminal is not offered on this deployment (web.terminal: false)."


def resolve_terminal_offered(config_path: str | Path | None) -> bool:
    """Whether this deployment offers the terminal on the Agent tile (``web.terminal``).

    Default ``True``: a deployment that never mentions the key keeps the
    runtime opt-in. ``false`` takes the terminal surface away — the browser
    drops the entry to it, and ``/ws/terminal`` refuses to open a PTY session.

    Read the way :func:`resolve_privilege_gates` reads its gates, because it
    is one in kind — the server refuses a route on it, not only the page: no
    file resolved means the default, a resolved file that cannot be read means
    no terminal, and a value nobody can interpret takes the default through
    :func:`coerce_config_flag`.

    Args:
        config_path: The config file this surface resolved, or ``None``.

    Returns:
        False when the config says ``false`` or cannot be read.
    """
    if config_path is None:
        return True
    from osprey_connectors.config import get_config_value

    try:
        raw = get_config_value("web.terminal", True, str(config_path))
    except Exception:
        logger.error(
            "Could not read the config file %s; the terminal is closed until it reads cleanly",
            config_path,
            exc_info=True,
        )
        return False
    return coerce_config_flag("web.terminal", raw, True)


def unreadable_config_refusal(path: str | Path) -> str:
    """The refusal every surface gives when an existing config file could not be read.

    It names the file and the fix. The privilege gates are resolved once at
    startup, so restarting the server after the file is fixed is what reopens
    them.

    Args:
        path: The config file that exists but could not be read.

    Returns:
        The refusal sentence, naming the file and the fix.
    """
    return f"{path} could not be read; fix it and restart the server"


def coerce_config_flag(key: str, value: object, default: bool) -> bool:
    """Return a configured boolean switch, falling back to *default*.

    An **absent** key (``None``) means "use the default" — the shipped
    behaviour of a deployment that never mentions the switch.

    A real YAML boolean is honoured as written. A quoted ``"false"`` is honoured
    too: that spelling is a human writing a boolean, and reading it as truthy
    would leave a switch ON that the config says is OFF.

    Anything else (a number, a mapping from a mis-indented file, an unrecognized
    word) is reported and discarded, exactly as :func:`coerce_config_str` does
    with a non-string — a value nobody can interpret must not silently become
    one of the two postures.

    Args:
        key: Dotted config key, used only for the warning message.
        value: Whatever the config reader returned.
        default: The shipped default for *key*.

    Returns:
        The configured boolean, or *default*.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_WORDS:
            return True
        if token in _FALSE_WORDS:
            return False
    logger.warning("%s is %r, not a boolean; using %r instead", key, value, default)
    return default


#: Placeholders a config-declared panel's ``path`` may carry, resolved once per
#: container from :func:`compute_url_prefix` (``/u/<user>`` behind the
#: multi-user front door, ``""`` in the single-origin shape):
#:
#: * ``{url_prefix}`` — the mount as the proxy spells it, ``/u/<user>`` or
#:   empty. For a value that is itself a root-absolute path.
#: * ``{url_prefix_dir}`` — the same mount spelled as a directory with no
#:   leading and a trailing slash, ``u/<user>/`` or empty. For a value the
#:   backend roots at the origin itself and to which it prepends its own ``/``
#:   — noVNC's ``?path=`` is the canonical case: ``vnc.html?path=
#:   {url_prefix_dir}panel/<id>/websockify`` reaches
#:   ``/u/<user>/panel/<id>/websockify`` behind the front door and
#:   ``/panel/<id>/websockify`` without it, never a doubled slash.
#:
#: The proxy's content rewrite (:mod:`routes.proxy`) covers root-absolute
#: string literals in what a backend *serves*; a URL the browser assembles at
#: runtime from a query parameter is invisible to it, and the correct value is
#: per user, so the profile cannot spell it (#784). Nothing else in ``path`` is
#: touched.
_URL_PREFIX_PLACEHOLDERS = ("{url_prefix}", "{url_prefix_dir}")


def _substitute_url_prefix(path: str) -> str:
    """Resolve :data:`_URL_PREFIX_PLACEHOLDERS` in a custom panel's ``path``.

    A path without a placeholder is returned byte-identical, so every existing
    profile renders exactly as before.
    """
    if not isinstance(path, str) or not any(p in path for p in _URL_PREFIX_PLACEHOLDERS):
        return path
    prefix = compute_url_prefix()
    prefix_dir = f"{prefix.lstrip('/')}/" if prefix else ""
    return path.replace("{url_prefix}", prefix).replace("{url_prefix_dir}", prefix_dir)


def _load_panel_config() -> tuple[set[str], list[dict], str | None]:
    """Read web.panels and web.default_panel from config.yml.

    Returns:
        (enabled_builtin_ids, custom_panel_defs, default_panel_id_or_None)

        The default panel id is returned as declared by the profile/config;
        it is **not** validated here — the frontend treats an unknown id as
        a request to fall back to DEFAULT_PANEL_FALLBACK so a typo doesn't
        leave the user staring at a blank tabset.

    Raises:
        ValueError: When a declared panel id is one no request for that panel
            could be routed or forwarded with (:func:`panel_id_refusal`). The
            message names the config key, the id and the class, because the
            operator who wrote the id is the only one who can change it. Also a
            block that spells the retired ``arm_writes``, naming the tier key.
    """
    try:
        from osprey_connectors.workspace import load_osprey_config

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
        # Every declared id, whether its block is on or off. The id is one URL
        # path segment and a header value on every hop the proxy makes for the
        # panel, so an id outside that class answers 500 for every request that
        # panel makes while the rest of the terminal serves normally: one
        # refusal at boot names the block, and none of those 500s does.
        # `enabled` is a flag an operator flips, so an id that cannot be served
        # is refused before the flip rather than after it.
        refusal = panel_id_refusal(str(panel_id))
        if refusal is not None:
            raise ValueError(refusal)

        # An application is armed by the login's write tier, never by a second
        # key on its panel: a block still spelling one, on or off, is refused
        # rather than read as a grant it no longer is.
        if isinstance(spec, dict) and "arm_writes" in spec:
            raise ValueError(
                f"web.panels.{panel_id}.arm_writes is retired: an application panel "
                f"is armed for writes by the login's `{WRITES_KEY}` tier"
            )

        # One predicate for builtin and custom blocks alike: the build writes
        # `enabled` onto every block from the profile's `web_panels` selection,
        # so a block this render carries for a tab it does not select is off.
        if not panel_spec_enabled(spec):
            continue
        if panel_id in BUILTIN_PANELS:
            enabled.add(panel_id)
        elif isinstance(spec, dict):
            panel = {
                "id": panel_id,
                "label": spec.get("label", panel_id.upper()),
                "url": spec.get("url", ""),
                "healthEndpoint": spec.get("health_endpoint"),
                "path": _substitute_url_prefix(spec.get("path", "/")),
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
                # Root-absolute prefixes rewritten in this panel's responses on top
                # of its own `path` (see routes/proxy.py) — for a backend that
                # serves a prefix its `path` does not cover.
                "rewritePrefixes": spec.get("rewrite_prefixes") or [],
                # The application this panel serves (armed for writes by the
                # login's tier, routes/proxy.py) and the exact routes the
                # panel credential may reach on it.
                "app": spec.get("app"),
                "agentRoutes": _panel_agent_routes(str(panel_id), spec.get("agent_routes")),
            }
            # The NAME of the variable holding this panel's launch token, never
            # its value: this dict reaches the browser. Absent on a render that
            # does not arm the application.
            if spec.get("launch_token_env"):
                panel["launchTokenEnv"] = str(spec["launch_token_env"])
            # The read bound of this panel's proxied requests (see routes/proxy.py).
            # Absent and ``null`` say different things — absent keeps the shared
            # client's default, ``null`` asks for no bound — so the key is carried
            # only when the block writes it.
            if "read_timeout_s" in spec:
                panel["readTimeoutS"] = _panel_read_timeout(panel_id, spec["read_timeout_s"])
            custom.append(panel)

    return enabled, custom, default_panel


def _hold_panel_launch_tokens(app: FastAPI, custom_panels: list[dict]) -> None:
    """Hold each panel's launch token where the proxy finds its credentials.

    Read from this process's environment: compose hands it to a roster's
    container, and a roster-less terminal loads it from the deploy ``.env``. An
    empty variable holds nothing, so the backend refuses the launch; the warning
    names the variable and never a value. A variable that is not the application's
    own is never read: what is held goes upstream as a header.
    """
    for panel in custom_panels:
        name = panel.get("launchTokenEnv")
        if not name:
            continue
        try:
            own = launch_token_env(str(panel.get("app") or ""))
        except ValueError:
            own = None
        if name != own:
            logger.warning(
                "Panel %s names %s, which is not its application's launch token", panel["id"], name
            )
            continue
        token = os.environ.get(name, "")
        if token.strip():
            app.state.panel_auth_headers[panel["id"]] = {LAUNCH_TOKEN_HEADER: token}
        else:
            logger.warning("Panel %s names %s, which holds no value", panel["id"], name)


def _panel_agent_routes(panel_id: str, routes: object) -> list[dict[str, str]]:
    """Validate ``web.panels.<id>.agent_routes`` into ``{method, path}`` entries.

    Each entry is exactly ``{method, path}``: the method is upper-cased, the
    path is backend-root-relative (leading ``/``, no query) and is composed
    under ``/panel/<id>`` by the proxy, never here.

    Raises:
        ValueError: Any entry of another shape, naming the config key.
    """
    if routes is None:
        return []
    key = f"web.panels.{panel_id}.agent_routes"
    if not isinstance(routes, list):
        raise ValueError(f"{key} must be a list of {{method, path}} entries")
    entries = []
    for route in routes:
        if (
            not isinstance(route, dict)
            or set(route) != {"method", "path"}
            or not isinstance(route["method"], str)
            or not route["method"].strip()
            or not isinstance(route["path"], str)
            or not route["path"].startswith("/")
            or "?" in route["path"]
            or "#" in route["path"]
        ):
            raise ValueError(
                f"{key}: {route!r} is not a {{method, path}} entry with a "
                "backend-root-relative path"
            )
        entries.append({"method": route["method"].strip().upper(), "path": route["path"]})
    return entries


def _panel_read_timeout(panel_id: str, value: object) -> float | None:
    """Validate a panel's ``read_timeout_s``: positive seconds, or ``None`` for unbounded.

    Args:
        panel_id: The panel id, for the refusal message.
        value: The raw ``web.panels.<id>.read_timeout_s`` value.

    Raises:
        ValueError: When the value is neither ``null`` nor a positive number.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(
            f"web.panels.{panel_id}.read_timeout_s must be a positive number of seconds "
            f"or null (unbounded), got {value!r}"
        )
    return value


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
        from osprey_connectors.workspace import load_osprey_config

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
        from osprey_connectors.workspace import load_osprey_config

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
            value = config.get(section, {}) if isinstance(config, dict) else {}
            return value if isinstance(value, dict) else {}

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


def _settle_agent_cli(argv: list[str], project_cwd: str) -> None:
    """Settle and name the Claude Code the two surfaces run.

    The terminal spawns the CLI named by ``argv`` in a PTY; the chat
    goes through the Agent SDK, which OSPREY points at the installed ``claude``
    of the same version. The executable the chat's runs execute is resolved
    here, once for the life of the server, and named beside the terminal's
    argv. Blocking: the version probe spawns the CLI.

    When the project pins a version that is not installed, the chat cannot
    run: that is one error line carrying the operator's message, and every
    chat session is refused with the same message until the version is
    installed. The terminal view does not depend on that install, so the
    server still starts.

    Args:
        argv: The argv prefix the PTY will spawn.
        project_cwd: The project the chat's runs execute in.
    """
    from osprey.agent_runner import launcher
    from osprey.agent_runner.errors import AgentCliUnavailableError

    try:
        path = launcher.project_agent_cli_path(Path(project_cwd))
    except (AgentCliUnavailableError, ValueError) as exc:
        logger.error(
            "Claude Code CLI — terminal spawns %s; chat cannot run: %s", shlex.join(argv), exc
        )
        return
    logger.info(
        "Claude Code CLI — terminal spawns %s; chat runs %s (%s)",
        shlex.join(argv),
        path,
        launcher.cli_version_of(path) or "version unknown",
    )


#: Basename of the hook that reports the terminal agent's turn edges. The
#: rendered settings name it inside a shell command, so it is matched as a
#: substring rather than as a whole command: the interpreter, the project-dir
#: variable and the fail-open suffix all vary between deployments.
TURN_STATE_HOOK = "osprey_turn_state.py"

#: The ``SessionStart`` sources the turn hook must be registered for. An entry
#: without this matcher also runs on ``compact``, which fires inside a running
#: turn — its idle report would say the process is between turns while it is
#: mid-answer, so a registration missing the matcher is not turn reporting we
#: can act on.
TURN_HOOK_SESSION_START_MATCHER = "startup|resume|clear"


def _turn_hook_matchers(settings: dict) -> dict[str, list[str | None]]:
    """Map each hook event that runs the turn-state hook to its matchers.

    Args:
        settings: A parsed ``.claude/settings.json``.

    Returns:
        One entry per event whose registered commands name the turn-state hook,
        holding the ``matcher`` of each registration — ``None`` for one that
        carries none, which is how Claude Code spells "every source". Every
        level is shape-checked: these settings are a rendered artifact a
        deployment may have edited, and a malformed one must read as "no hook"
        rather than raise during a boot.
    """
    found: dict[str, list[str | None]] = {}
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return found
    for event, entries in hooks.items():
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            registered = entry.get("hooks")
            for hook in registered if isinstance(registered, list) else []:
                if isinstance(hook, dict) and TURN_STATE_HOOK in str(hook.get("command", "")):
                    found.setdefault(event, []).append(entry.get("matcher"))
    return found


def _detect_turn_hook(app: FastAPI) -> None:
    """Decide whether this project's settings install turn reporting.

    The answer lands on ``app.state.turn_hook_present`` and it is a permission,
    not a prediction: readers of ``app.state.turn_state`` may wait for an idle
    edge only where one can actually arrive. A deployment whose settings carry
    no turn hook — an older render, or a hand-edited one — never sends an edge,
    and a reader that waited for one would hang instead of falling back to the
    evidence it can gather itself.

    Two registrations are required. ``Stop`` is the edge that ends a turn, and
    ``SessionStart`` must carry the matcher that keeps ``compact`` out. A
    missing ``StopFailure`` is warned about rather than disqualifying: it only
    reports a turn that errored out, and a CLI build that never emits the event
    would otherwise make every deployment look broken.

    Args:
        app: The application, carrying ``state.project_cwd``.
    """
    # Established here, not inherited: every early return below documents a
    # False result, and this function must produce it on its own rather than
    # leaving whatever a caller happened to set beforehand.
    app.state.turn_hook_present = False

    settings_path = Path(app.state.project_cwd) / ".claude" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.debug("No %s — the terminal reports no turn edges", settings_path)
        return
    except (OSError, ValueError):
        logger.warning(
            "Could not read %s; assuming the terminal reports no turn edges",
            settings_path,
            exc_info=True,
        )
        return
    if not isinstance(settings, dict):
        return

    events = _turn_hook_matchers(settings)
    has_stop = "Stop" in events
    has_session_start = TURN_HOOK_SESSION_START_MATCHER in events.get("SessionStart", [])
    app.state.turn_hook_present = has_stop and has_session_start

    if not app.state.turn_hook_present:
        missing = [
            name for name, ok in (("Stop", has_stop), ("SessionStart", has_session_start)) if not ok
        ]
        logger.info(
            "%s is not wired for %s in %s; view switching will fall back to its "
            "own idleness evidence",
            TURN_STATE_HOOK,
            ", ".join(missing),
            settings_path,
        )
        return

    if "StopFailure" not in events:
        logger.warning(
            "%s is wired for Stop but not StopFailure in %s; a turn that ends in "
            "a failure reports no idle edge",
            TURN_STATE_HOOK,
            settings_path,
        )


def _start_watchdogs(app: FastAPI, config_file: Path | None) -> None:
    """Resolve, once, whether this build runs watchdogs, and where their holds live.

    ``app.state.session_hold_root`` and ``app.state.watchdog_runner`` are both set
    or both ``None``, and every in-process hold call goes through that root. A
    value inherited in this process's environment is dropped, because a child
    could otherwise read it and hold on a store no watchdog here ever writes.

    A rendered ``watchdogs`` block that breaks a rule stops the server with
    :class:`BuildProfileError`: serving on with the deployment's watchdogs quietly
    off is the one failure an operator could not see.
    """
    os.environ.pop(HOLD_DIR_ENV_VAR, None)
    app.state.session_hold_root = None
    app.state.watchdog_runner = None
    if config_file is None:
        return
    try:
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        logger.warning("No watchdogs: %s is unreadable (%s)", config_file, type(exc).__name__)
        return
    if not isinstance(config, dict):
        return
    build_dir = config_file.parent
    definitions = load_watchdogs(config, build_dir)
    if not definitions:
        return

    from osprey.services.watchdog.runner import WatchdogRunner

    root = session_hold_dir(repo_root_for_config(config_file))
    app.state.session_hold_root = root
    app.state.watchdog_runner = WatchdogRunner(
        definitions,
        hold_root=root,
        notify=make_notifier(app),
        config=config,
        build_dir=build_dir,
    )


def _create_lifespan(
    config_path: str | Path | None = None,
    shell_command: list[str] | None = None,
    project_dir: str | Path | None = None,
):
    """Create a lifespan context manager for the app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from osprey.agent_runner.launcher import build_claude_launch_argv
        from osprey.utils.shell_resolver import normalize_shell_command

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
            app.state.shell_command = normalize_shell_command(config["shell"])
        else:
            app.state.shell_command = build_claude_launch_argv(
                _load_claude_code_config(config_path)
            )
        max_bg = int(config.get("max_background_sessions", 5))
        app.state.pty_registry = PtyRegistry(max_background=max_bg)

        # Which transcript each session key currently points at, read once so a
        # recreated container resumes a cleared-and-continued conversation
        # rather than starting a fresh one under the same key.
        load_transcript_map(app)

        # What the terminal's Claude Code process last reported it was
        # doing, keyed by session key (routes/agent_turn.py). Empty at
        # startup: no report is not a claim of idleness, and every reader
        # treats a missing entry as "unknown" rather than "free to take over".
        app.state.turn_state = {}
        # Whether that reporting is actually installed. Left False here and
        # decided by the rendered hook settings, so a deployment whose
        # settings carry no turn-state hook never waits for an edge that
        # cannot arrive.
        app.state.turn_hook_present = False

        # ── Operator-chat pool bounds ──
        # Three knobs bound the operator-chat pool; each fails open to its
        # default so a missing/broken config never blocks startup. Read from
        # the top-level `web` section (same section as web.theme).
        # The route handlers re-read the two timeouts off app.state via getattr
        # with these same defaults, so the attribute names are load-bearing.
        try:
            from osprey_connectors.config import get_config_value

            chat_turn_timeout_s = float(get_config_value("web.chat_turn_timeout_s", 600))
            chat_idle_timeout_s = float(get_config_value("web.chat_idle_timeout_s", 1800))
            chat_max_sessions = int(get_config_value("web.chat_max_sessions", 5))
        except Exception:  # never let config load block startup
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
        # Uploads nobody sent with a message are deleted once they are a day old;
        # before any upload is counted, so the caps measure the swept tree.
        await asyncio.to_thread(sweep_uploads_at_startup, app)
        # Which Claude Code the two views run, settled and named once at
        # startup. Off the event loop: the probe spawns a large binary.
        await asyncio.to_thread(_settle_agent_cli, app.state.shell_command, app.state.project_cwd)
        # Read once, now that the project the terminal serves is known: the
        # answer is a property of that project's rendered settings, and nothing
        # rewrites them while the server runs.
        _detect_turn_hook(app)
        app.state.broadcaster = FileEventBroadcaster()
        app.state.active_panel = None
        # Bounded history of agent-activity events. The SSE stream only reaches
        # browsers that are already connected, so the buffer is what a browser
        # opened (or reloaded) mid-session reads to catch up on recent actions.
        app.state.agent_activity_buffer = deque(maxlen=ACTIVITY_BUFFER_MAX)
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
        # Whether the render zone (``config.yml`` + ``.claude/``) is read-only to
        # this process. In a privilege-split container the render zone is
        # root-owned and the root entrypoint performs the regen and the scaffold
        # restore *before* dropping to the non-root app user, so the server must
        # not attempt either write. ``OSPREY_RENDER_ZONE_READONLY=1`` is that
        # marker; absent (or any other value) leaves startup behaviour unchanged.
        # Later readers use ``getattr(app.state, "render_zone_readonly", False)``.
        # Read through ownership.is_container_render, not re-spelled here: the
        # same marker decides that module's ownership surface, and a server that
        # concluded it was on a host while the gallery concluded it was in a
        # container would disagree about which writes are even possible.
        from osprey.interfaces.web_terminal.ownership import is_container_render

        render_zone_readonly = is_container_render()
        app.state.render_zone_readonly = render_zone_readonly

        # Ensure OSPREY_CONFIG is set before any load_osprey_config() call
        if "OSPREY_CONFIG" not in os.environ:
            candidate = Path(app.state.project_cwd) / "config.yml"
            if candidate.exists():
                os.environ["OSPREY_CONFIG"] = str(candidate)
                logger.debug("Auto-set OSPREY_CONFIG=%s", candidate)

        # Clear any stale config cache (e.g. from web_cmd.py pre-lifespan call)
        from osprey_connectors.workspace import reset_config_cache

        reset_config_cache()

        # Put volume-owned artifact bodies back into the project tree before
        # anything reads it. In a deployed container the tree comes back
        # image-fresh on every recreation, while the operator's claimed
        # versions live on the claude-config volume; without this the agent
        # would run the framework's originals while the gallery showed the
        # operator's, and nothing would report the divergence. No-op elsewhere.
        # Skipped entirely when the render zone is read-only: the container
        # entrypoint already ran this restore as root, and repeating it here
        # would only fail on a tree this process cannot write.
        if render_zone_readonly:
            logger.info(
                "Render zone is read-only (OSPREY_RENDER_ZONE_READONLY=1); "
                "skipping scaffold restore — the container entrypoint already ran it"
            )
        else:
            from osprey.interfaces.web_terminal.scaffold_gallery_service import (
                restore_scaffold_bodies,
            )

            try:
                restore_scaffold_bodies(Path(app.state.project_cwd))
            except Exception as exc:  # never block startup on this
                logger.warning("Could not restore user-owned artifacts from the volume: %s", exc)

        # Resolve and store config_path for the settings API
        resolved_config_path: Path | None = None
        for config_candidate in [
            Path(config_path) if config_path else None,
            Path(os.environ.get("CONFIG_FILE", "")) if os.environ.get("CONFIG_FILE") else None,
            Path("config.yml"),
        ]:
            if config_candidate and config_candidate.exists() and config_candidate.is_file():
                resolved_config_path = config_candidate.resolve()
                break
        app.state.config_path = resolved_config_path
        _start_watchdogs(app, resolved_config_path)

        # ── Web theme (SSR no-FOUC attribute) ──
        # Resolved once at startup and server-rendered onto <html data-theme>
        # so the generated theme-boot.js first-paints with no
        # flash. Fails open on any load error — a missing/broken theme
        # registry must never block server startup.
        try:
            from osprey.interfaces.design_system.theme_config import (
                resolve_configured_web_theme,
            )

            # The shared environment → ``web.theme`` → id/pin/family chain, so
            # this page and the artifact pages the gallery serves cannot
            # disagree about what a configured value means.
            web_theme = resolve_configured_web_theme()
            app.state.web_theme_id = web_theme.id
            # Whether the configured value pinned a mode (a concrete id) or only
            # a palette (a family). Server-rendered alongside data-theme because
            # without it theme-manager.js's hub assumes 'auto' on a first visit
            # and re-resolves the mode from the OS one frame after paint,
            # silently discarding the pin.
            app.state.web_theme_mode = web_theme.pinned_mode
        except Exception:  # never let config/theme-registry load block startup
            logger.warning(
                "Could not resolve web.theme (config or theme-registry load failed); "
                "server-rendering fallback theme 'dark'",
                exc_info=True,
            )
            app.state.web_theme_id = "dark"
            app.state.web_theme_mode = None

        # ── The terminal gate (web.terminal) ──
        # Server-rendered onto <html data-terminal="off"> when false, and
        # /ws/terminal refuses to open a PTY session (routes/websocket.py). Read
        # like the privilege gates below — the server refuses a route on it —
        # so an unreadable config closes it.
        app.state.web_terminal_offered = resolve_terminal_offered(resolved_config_path)

        # ── Onboarding-tour invite policy ──
        # OSPREY_WEB_TOUR (the per-user roster path) outranks web.tour, the
        # same precedence OSPREY_WEB_THEME has over web.theme. Resolved once
        # at startup; GET /api/panels echoes it to the browser. Fails open to
        # the default policy on any config-read error.
        try:
            from osprey_connectors.workspace import load_osprey_config

            configured_tour = os.environ.get(
                "OSPREY_WEB_TOUR", ""
            ).strip() or load_osprey_config().get("web", {}).get("tour", DEFAULT_TOUR_POLICY)
            app.state.web_tour_policy = resolve_tour_policy(configured_tour)
        except Exception:  # never let config load block startup
            logger.warning(
                "Could not resolve web.tour (config load failed); falling back to policy %r",
                DEFAULT_TOUR_POLICY,
                exc_info=True,
            )
            app.state.web_tour_policy = DEFAULT_TOUR_POLICY

        # ── Config panel (server-side tier gate) ──
        # `web.config_panel.enabled: false` takes the Config panel's whole
        # SERVER surface away, not just its tab: /api/config and
        # /api/claude-setup refuse every verb with 403, and GET /api/panels
        # stops advertising the panel. Hiding the tab alone would leave a tier
        # that may not re-render this deployment's agent able to reach the edit
        # routes by typing their URLs, which is the class of gap this key
        # exists to close — config.yml and .claude/ are what the agent's
        # permission surface is rendered from.
        #
        # Resolved once here and read back as
        # ``getattr(app.state, "config_panel_enabled", False)``: an app that
        # skipped this lifespan has made no tier decision, and the routes
        # refuse it. The value is resolved together with the scaffold gate
        # below.

        # ── Scaffold gallery writes (server-side tier gate) ──
        # `web.scaffold_gallery.write_enabled: false` closes the gallery's whole
        # WRITE surface: create, claim, save, unoverride, register and delete
        # under /api/scaffold all refuse with 403, while every read route stays
        # open. What the gallery authors is `.claude/rules|skills|agents`
        # content, which the agent loads at PROJECT scope — instruction it
        # obeys, not decoration — so "may this tier author it" is a privilege
        # boundary. Hiding the buttons alone would be the same cosmetic gate
        # a client-only guard would be: it is undone by curl.
        #
        # Resolved once here and read back as
        # ``getattr(app.state, "scaffold_write_enabled", False)``: an app that
        # skipped this lifespan has made no tier decision, and the write routes
        # refuse it.
        #
        # Both gates answer out of the file this terminal resolved: with none
        # resolved, the shipped defaults (both open); with one that cannot be
        # read, closed, because an unreadable config is not permission to
        # author what the agent obeys. An absent key is the default every
        # single-user deployment has always had. They come from one read, so
        # they cannot disagree about the file.
        gates = resolve_privilege_gates(resolved_config_path)
        app.state.config_panel_enabled = gates.config_panel_enabled
        app.state.scaffold_write_enabled = gates.scaffold_write_enabled
        app.state.config_unreadable_path = gates.config_unreadable_path

        # ── Regenerate stale Claude Code artifacts on launch ──
        # config.yml is a build-time input: safety-critical fields (e.g. the
        # writes_enabled kill-switch baked into settings.json's permissions.deny)
        # only take effect once the artifacts are re-rendered. Regenerating here
        # — mirroring `osprey chat` — means an edited config.yml is honored
        # on the next server start. Fail open so a regen error never blocks launch.
        try:
            from osprey.cli.templates.manager import TemplateManager

            project_dir_for_regen = Path(app.state.project_cwd)
            if render_zone_readonly:
                # regen_if_drift writes even when nothing drifted — its no-op
                # path stamps settings.json with os.utime to clear the
                # SessionStart drift hook — so a read-only render zone gets the
                # dry-run preview and a warning instead. The entrypoint already
                # regenerated as root; anything still listed here means the
                # render on disk does not match config.yml and only a rebuild
                # (or a root-side regen) can fix it.
                preview = TemplateManager().regenerate_claude_code(
                    project_dir_for_regen, dry_run=True
                )
                would_change = list(preview.get("changed") or [])
                logger.warning(
                    "Render zone is read-only (OSPREY_RENDER_ZONE_READONLY=1); "
                    "on-launch Claude Code artifact regen skipped. Would change: %s",
                    ", ".join(would_change) if would_change else "nothing",
                )
            else:
                changed = TemplateManager().regen_if_drift(project_dir_for_regen)
                if changed:
                    logger.info(
                        "Regenerated %d stale Claude Code artifact(s): %s",
                        len(changed),
                        ", ".join(changed),
                    )
        except Exception:  # never let regen block server startup
            logger.warning("Claude Code artifact regen on launch failed", exc_info=True)

        # ── Provider env injection ──
        from osprey.agent_runner.provider_env import (
            detect_managed_policy_conflicts,
            format_managed_policy_conflicts,
            inject_provider_env,
            load_provider_spec,
        )

        _spec = None
        if app.state.config_path:
            from osprey_connectors.workspace import repo_root_for_config

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
                    from osprey.infrastructure.proxy.lifecycle import start_proxy_for

                    proxy_port = start_proxy_for(_spec, os.environ)
                    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"
                    logger.info(
                        "Translation proxy on :%d → %s",
                        proxy_port,
                        _spec.upstream_base_url,
                    )

        # Managed (enterprise) policy settings outrank the process environment
        # and --setting-sources project alike, so a policy value that differs
        # from the deployment's would silently redirect the operator-facing
        # terminal. The check reads the finished environment, translation-proxy
        # loopback included, because that is what the terminal's agent would
        # otherwise run on. A server with no provider compares against nothing,
        # so every policy provider key refuses. A refusal here leaves only the
        # proxy daemon thread behind, which exits with the process.
        _policy_conflicts = detect_managed_policy_conflicts(os.environ if _spec else {})
        if _policy_conflicts:
            raise RuntimeError(
                "Refusing to start the Web Terminal.\n"
                + format_managed_policy_conflicts(_policy_conflicts)
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
            from osprey_connectors.workspace import (
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
        # Read once here and
        # echoed to the browser by GET /api/panels. Every read fails open: a
        # facility with no config.yml still gets working Documentation and
        # Feedback controls pointed at the project defaults.
        #
        # The raw values are read together — a failure here means the config is
        # unreadable, which really does concern all of them — but each is
        # validated SEPARATELY by the resolver. A single unusable value (a
        # hand-written "256MB" ceiling, say) must not drag the others back to
        # project defaults: silently redirecting a facility's feedback address
        # to the upstream maintainer is exactly the failure a fail-open path
        # must not produce.
        #
        # The two keys `web.feedback.owner` can stand in for are read with a
        # None sentinel rather than their own default, because the resolver has
        # to tell "absent" from "spelled, and equal to the default" to know
        # whether the leaf outranks the owner block. Every other key reads with
        # its default, as before.
        try:
            from osprey_connectors.config import get_config_value

            raw_docs_url = get_config_value("web.docs_url", DEFAULT_DOCS_URL)
            raw_github_repo = get_config_value("web.feedback.github_repo", None)
            raw_email = get_config_value("web.feedback.email", None)
            raw_trackers = get_config_value("web.feedback.trackers", None)
            raw_owner = get_config_value("web.feedback.owner", None)
        except Exception:  # never let config load block startup
            logger.warning(
                "Could not read web.docs_url / web.feedback.* config keys; using defaults",
                exc_info=True,
            )
            raw_docs_url = raw_github_repo = raw_email = raw_trackers = None
            raw_owner = None
        # One resolver, so this block and the `GET /api/panels` fallback cannot
        # disagree: `trackers` is derived from `github_repo` and the owner
        # block, and two copies of that derivation are two chances to drift.
        destination = resolve_feedback_destination(
            docs_url=raw_docs_url,
            email=raw_email,
            github_repo=raw_github_repo,
            trackers=raw_trackers,
            owner=raw_owner,
        )
        app.state.docs_url = destination.docs_url
        app.state.feedback_github_repo = destination.github_repo
        app.state.feedback_trackers = destination.trackers
        app.state.feedback_email = destination.email
        app.state.feedback_owner_name = destination.owner_name

        # ── Deployment identity + the escalation link ──
        # A user files to whoever owns this deployment, because a user cannot
        # know whether a bug is OSPREY's code or this facility's config. The
        # maintainer who CAN tell then forwards the framework bugs upstream —
        # and only will if that is nearly free, so the identity upstream would
        # otherwise have to ask for rides along with every report, and the
        # forwarding link is prefilled.
        #
        # Nothing here varies per report, so it is resolved once at startup:
        # no per-submission composition, and no fitting to a URL length cap.
        try:
            from osprey_connectors.config import get_config_value

            raw_preset = get_config_value("provenance.preset", None)
            raw_preset_hash = get_config_value("provenance.preset_hash", None)
            raw_finder_mode = get_config_value("channel_finder.pipeline_mode", None)
        except Exception:  # an unreadable identity is not fatal
            logger.warning("Could not read the deployment identity keys", exc_info=True)
            raw_preset = raw_preset_hash = raw_finder_mode = None
        app.state.deployment_identity = resolve_deployment_identity(
            osprey_version=osprey_version(),
            preset=raw_preset,
            preset_hash=raw_preset_hash,
            channel_finder_mode=raw_finder_mode,
        )
        # "" for a deployment the OSPREY project already owns — a link inviting
        # a maintainer to forward a report to themselves is noise.
        app.state.feedback_escalation_url = upstream_escalation_url(
            app.state.deployment_identity, destination
        )

        # The server-side stores are sited on the CONFIGURED agent-data root
        # and deliberately NOT on workspace_dir: with web_terminal.watch_dir
        # set, the watched tree is somewhere else entirely and anything written
        # there would land outside the {user}-agent-data volume, where
        # the host cannot reach it. Never resolve_agent_data_root() either — that
        # appends sessions/<id>, and the store spans sessions: the sidecars'
        # start records. Feedback is not a store here: a submission is an entry
        # in the agent record.
        try:
            from osprey_connectors.workspace import resolve_shared_data_root

            shared_data_root = resolve_shared_data_root()
        except Exception:  # never let config load block startup
            shared_data_root = workspace_dir
            logger.warning(
                "Could not resolve the shared data root; siting the server-side stores under %s",
                shared_data_root,
                exc_info=True,
            )
        # What the terminal recorded about each sidecar's start, read back by
        # `osprey health` from this same root.
        app.state.panel_status_root = shared_data_root
        app.state.panel_status_dir = shared_data_root / PANEL_STATUS_DIRNAME
        # Workspace-relative form of the store: the *file watcher's* form,
        # used below to drop change events for writes into it. The file
        # browser is not a consumer — routes/files.py derives its own predicate
        # from the store directory itself, because it must also
        # handle symlink aliases and session-scoped roots that a single relative
        # path cannot express.
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
        app.state.panel_status_rel = resolve_store_rel(app.state.panel_status_dir, workspace_dir)
        # Recording a sidecar's start must be silent: it writes one file, and an
        # unconcealed store would push an SSE change frame to every connected
        # browser the moment a sidecar changed state. This seam silences the
        # watcher and only that; the file panel's listing and content reads
        # hide the store through routes/files.py's own predicate.
        app.state.concealed_store_rels = tuple(
            rel for rel in (app.state.panel_status_rel,) if rel is not None
        )

        app.state.watcher = WorkspaceWatcher(
            workspace_dir, app.state.broadcaster, concealed=app.state.concealed_store_rels
        )
        app.state.watcher.start()

        # Load panel config and conditionally launch servers
        enabled_panels, custom_panels, default_panel = _load_panel_config()
        app.state.enabled_panels = enabled_panels
        app.state.custom_panels = custom_panels
        app.state.default_panel = default_panel
        # The declared agent routes of the config-defined loopback panels are
        # what the panel credential may reach under /panel/<id>; stated once
        # per load, so a route the config drops stops admitting.
        set_panel_agent_routes(configured_agent_routes(custom_panels))

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

        # Whether the sidebar offers the control-target block
        # (web.control_target_picker). root() stamps it on the page.
        app.state.control_target_picker_available = control_target_picker_available(
            resolved_config_path
        )

        # ── Tour capabilities ──
        # The "Ask in plain language" tour card lists what THIS deployment's
        # agent can do — derived here, never claimed in the browser. Python
        # analysis and plots ride the core executor + workspace pipeline every
        # deployment carries; the logbook line appears only when the ARIEL
        # panel is enabled. No reading sentence is emitted here: a configured
        # ``control_system.type`` says a connector exists, not what is behind
        # it, so the server cannot tell a live machine apart from a stand-in
        # or a demo. The browser derives that wording from the active control
        # target's ``kind`` instead. Deployment-dependent sentences the server
        # cannot verify are not emitted at all.
        tour_capabilities = ["run Python analysis", "make plots"]
        if "ariel" in enabled_panels:
            tour_capabilities.append("search the logbook")
        app.state.tour_capabilities = tour_capabilities
        # Logbook availability as its own fact, so the browser can offer a
        # logbook starter prompt without parsing the capability sentences.
        app.state.tour_logbook = "ariel" in enabled_panels

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

        # The sidecar startup wait, resolved once here after the config cache
        # reset; every sidecar launch, startup or retry, reads it from app.state.
        try:
            from osprey_connectors.config import get_config_value

            app.state.sidecar_ready_timeout_s = resolve_sidecar_ready_timeout(
                get_config_value(SIDECAR_READY_TIMEOUT_KEY, DEFAULT_SIDECAR_READY_TIMEOUT_S)
            )
        except Exception:  # never let config load block startup
            logger.warning(
                "Could not resolve %s; using %s s",
                SIDECAR_READY_TIMEOUT_KEY,
                DEFAULT_SIDECAR_READY_TIMEOUT_S,
                exc_info=True,
            )
            app.state.sidecar_ready_timeout_s = DEFAULT_SIDECAR_READY_TIMEOUT_S

        # A sidecar's running process and the credential the proxy re-issues for
        # it, keyed by panel id. Both stay empty when no sidecar panel is
        # enabled, and a launch that failed adds to neither.
        app.state.sidecars = {}
        app.state.panel_auth_headers = {}
        _hold_panel_launch_tokens(app, custom_panels)
        # The start outcome the terminal last recorded for each sidecar panel;
        # the panel routes read it, `osprey health` reads the disk copy.
        app.state.sidecar_status = {}
        # One start task per sidecar panel while a retry is in flight, and the
        # flag that tells a failing start it was shutdown that stopped it.
        app.state.sidecar_start_tasks = {}
        app.state.sidecars_closing = False

        _launch_enabled_panel_servers(app, enabled_panels)
        await _launch_enabled_sidecars(app, enabled_panels)

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
        # chat tab does not pin a pool slot indefinitely. Fail-open at
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
                except Exception:  # one bad cycle must not kill the reaper
                    logger.warning("Idle chat reaper cycle failed", exc_info=True)

        reaper_task = asyncio.create_task(_reap_idle_chats())

        # ── Control-context ownership ──
        # This login keeps one control context, and it has one writer: the
        # web terminal whenever there is one, a controls server otherwise. The
        # claim is made here and renewed once a second, because the owner is
        # also who answers the switch requests other processes file. Fail-open
        # at every level: a claim that cannot be made leaves app.state without
        # `control_context_owner` — the roster renders read-only, the write
        # routes answer 503 — and the task retries on its next tick.
        app.state.control_context_task = await start_control_context_owner(app)

        yield

        # First, so the runs it ends can still reach the sessions that are live.
        if app.state.watchdog_runner is not None:
            await app.state.watchdog_runner.shutdown()

        await app.state.control_context_task.stop()

        reaper_task.cancel()
        with suppress(asyncio.CancelledError):
            await reaper_task

        await app.state.proxy_client.aclose()

        # Stop translation proxy if it was started
        from osprey.infrastructure.proxy.lifecycle import stop_proxy

        stop_proxy()

        # Stop every sidecar, one still coming up included: a stopped process
        # ends its start's wait within one poll. Then no record may outlive the
        # terminal claiming a sidecar runs or is starting.
        app.state.sidecars_closing = True
        for panel_id, sidecar in list(app.state.sidecars.items()):
            await _stop_sidecar(panel_id, sidecar)
        start_tasks = list(app.state.sidecar_start_tasks.values())
        if start_tasks:
            await asyncio.gather(*start_tasks, return_exceptions=True)
        for panel_id in sorted(enabled_panels & set(SIDECAR_PANELS)):
            clear_status(app.state.panel_status_root, panel_id)

        app.state.watcher.stop()
        app.state.pty_registry.cleanup_all()
        await app.state.operator_registry.cleanup_all()

    return lifespan


async def _scaffold_claim_conflict(_request: Request, exc: Exception) -> JSONResponse:
    """Render a refused claim as a 409 carrying the refusal verbatim.

    A refused claim is a conflict with the state of the project, and the
    message names what to do about it — so it is surfaced as written rather
    than being let through to become a bare 500 with the message stripped.
    Saving over a generated file raises the same refusal in the same words,
    naming the channel that actually owns the file, and reads the same way.

    Args:
        _request: The request whose route raised. Unused; part of the Starlette
            handler signature.
        exc: The :class:`~osprey.cli.scaffold_cmd.ScaffoldClaimError` raised.

    Returns:
        A 409 whose ``detail`` is the exception's message.
    """
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def _ownership_store_conflict(_request: Request, exc: Exception) -> JSONResponse:
    """Render a store that would not take the write as a 409.

    Nothing was recorded, so this must not read as success. Surfacing the
    reason beats the bare 500 an uncaught store error would otherwise give.

    Args:
        _request: The request whose route raised. Unused; part of the Starlette
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
    # and its literal-prefixed-path 404.
    app = FastAPI(
        title="OSPREY Web Terminal",
        description="Browser-based terminal with live workspace viewer",
        version="1.0.0",
        lifespan=_create_lifespan(config_path, shell_command, project_dir),
    )
    app.state.url_prefix = url_prefix

    app.include_router(router)
    app.include_router(watchdog_router)
    register_scaffold_conflict_handlers(app)

    @app.get("/")
    async def root(request: Request):
        app_name = getattr(request.app.state, "app_name", "")
        web_theme_id = getattr(request.app.state, "web_theme_id", "dark")
        web_theme_mode = getattr(request.app.state, "web_theme_mode", None)
        web_terminal_offered = bool(getattr(request.app.state, "web_terminal_offered", True))
        terminal_user = getattr(request.app.state, "terminal_user", "")
        landing_url = getattr(request.app.state, "landing_url", "")
        # The role nginx forwarded on THIS request, and where that role came
        # from, off the headers rather than app.state: per-login facts, and the
        # page GET is itself a gated request that carries them. Decoded by the
        # audit ledger's own bound; a value that fails it is shown as nothing,
        # not as the ledger's sentinel — with no nginx in front any client can
        # send these headers, and the chip is a display, not an authorization
        # surface. The source is shown only through :data:`_ROLE_SOURCE_LABELS`,
        # so a value outside the sidecar's vocabulary renders nothing.
        forwarded = forwarded_identity(request.headers)
        auth_role, auth_role_source = forwarded.role, forwarded.role_source
        if auth_role == UNSAFE_FORWARDED_VALUE:
            auth_role = None
        if auth_role_source == UNSAFE_FORWARDED_VALUE:
            auth_role_source = None
        auth_role_source_label = _ROLE_SOURCE_LABELS.get(auth_role_source or "", "")
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "app_name": app_name,
                "web_theme_id": web_theme_id,
                "web_theme_mode": web_theme_mode or "",
                # Stamped as data-terminal="off" only when false: the pre-paint
                # boot script and frame-params.js read it to keep the page out
                # of the terminal, whatever the tab's storage says.
                "web_terminal_offered": web_terminal_offered,
                "terminal_user": terminal_user,
                # Namespace for everything this page keeps in localStorage.
                # Empty on a single-user deployment, and the template omits
                # the attribute entirely in that case — see
                # resolve_storage_scope().
                "storage_scope": resolve_storage_scope(terminal_user),
                # The zone system.timezone names, resolved by the call the
                # operator chat's system prompt uses, so page and agent read one clock.
                "facility_timezone": get_facility_timezone().key,
                "landing_url": landing_url,
                "auth_role": auth_role or "",
                "auth_role_source_label": auth_role_source_label,
                "url_prefix": url_prefix,
                # Whether the sidebar offers the control-target block
                # (web.control_target_picker); the template stamps the off
                # state on <html> so app.js skips the chip and its popover.
                "control_target_available": bool(
                    getattr(request.app.state, "control_target_picker_available", True)
                ),
                # The registry's own labels for the built-in panels, stamped so
                # panel-catalog.js can read them at module load. /api/panels
                # answers for the ENABLED panels only and resolves after that
                # module's array is built, so it cannot seed the roster.
                "builtin_panel_labels": BUILTIN_PANEL_LABELS,
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
        # The storage scope travels with this page too, not just the index:
        # session.js's import closure reaches the modules that own the PTY
        # session id and the dock layout, so an unstamped
        # session page would write those keys unscoped and collide with the
        # neighbouring user's on the shared origin.
        return templates.TemplateResponse(
            request,
            "session.html",
            {
                "url_prefix": url_prefix,
                "storage_scope": resolve_storage_scope(
                    getattr(request.app.state, "terminal_user", "")
                ),
                "facility_timezone": get_facility_timezone().key,
                "web_theme_id": getattr(request.app.state, "web_theme_id", "dark"),
                "web_theme_mode": getattr(request.app.state, "web_theme_mode", None) or "",
                # activity-strip.js reaches panel-manager.js for the labels it
                # words panel actions with, so the pop-out page carries the
                # same roster stamp the index does.
                "builtin_panel_labels": BUILTIN_PANEL_LABELS,
            },
        )

    # frame_ancestors: console-origin HTML, proxied panel pages included, may
    # only be framed by the console itself (Q1-C6).
    configure_interface_app(app, static_dir=STATIC_DIR, frame_ancestors=True)

    return app


def _open_browser_when_ready(url: str, timeout: float = 15.0) -> None:
    """Wait for the server to accept connections, then open the browser.

    Args:
        url: The URL to open. Its port is what this waits on; a URL that names
            no port is probed at the ``web`` slot of the layout's default base,
            which is the port :func:`run_web` binds when nothing tells it
            otherwise.
        timeout: Seconds to wait for the server to start answering before
            giving up and skipping the browser open.
    """
    import socket
    import threading
    import time
    import webbrowser
    from urllib.parse import urlparse

    def _wait_and_open():
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or default_port("web")
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
    port: int = default_port("web"),
    shell_command: list[str] | None = None,
    config_path: str | None = None,
    project_dir: str | None = None,
    *,
    browser_url: str | None = None,
    sock: socket.socket | None = None,
) -> None:
    """Run the web terminal server.

    Args:
        host: Host to bind to.
        port: Port to run on. The default is the ``web`` slot at the layout's
            *default* base, which is right only for a programmatic caller with
            no config to resolve a base from. ``osprey web`` — the one caller —
            passes the port it resolved from this deployment's
            ``deployment.port_base``.
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
        sock: The listening socket to serve on, already bound to *host* and
            *port*; the caller that passes one owns its close. Omitted, the
            server binds *host*:*port* itself.
    """
    import uvicorn
    from uvicorn.main import STARTUP_FAILURE

    from osprey.interfaces._serving import listening_socket

    owned = sock is None
    if sock is None:
        sock = listening_socket(host, port)
    try:
        _open_browser_when_ready(browser_url or f"http://{host}:{port}")

        app = create_app(
            config_path=config_path, shell_command=shell_command, project_dir=project_dir
        )
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
        server.run(sockets=[sock])
    finally:
        if owned:
            sock.close()
    if not server.started:
        sys.exit(STARTUP_FAILURE)
