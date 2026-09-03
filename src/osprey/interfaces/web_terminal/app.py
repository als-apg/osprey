"""OSPREY Web Terminal — FastAPI Application.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Callable
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
from osprey.interfaces.common_middleware import (
    UNSAFE_FORWARDED_VALUE,
    apply_url_prefix,
    compute_url_prefix,
    forwarded_identity,
)
from osprey.interfaces.vendor import vendor_url
from osprey.interfaces.web_terminal.bar_items_store import (
    BarVocabulary,
    layout_path,
    load_layout,
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
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX
from osprey.port_layout import default_port
from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS, panel_spec_enabled
from osprey.registry.web import PANEL_ID_TO_REGISTRY_KEY, panel_url_state_attr

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jinja2.runtime import Context

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

    Mirrors the warn+fallback shape of the ``web.theme`` resolver
    (:func:`~osprey.interfaces.design_system.theme_config.resolve_theme_id`): it never
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


def resolve_storage_scope(terminal_user: str | None) -> str:
    """Resolve the per-user namespace for the browser's ``localStorage``.

    Multi-user deployments put one container per user behind a shared nginx
    front door at ``/u/<user>/`` — **same origin**, so every user shares one
    ``localStorage``. Without a namespace, one user's dock layout, rail
    position, palette history and active PTY session id are read and
    overwritten by the next user to log in on that browser.

    The namespace is decided here rather than in the browser: the served
    documents stamp it onto ``<html data-osprey-storage-scope>`` and every JS
    storage site reads it from there, so no client-side code has to parse
    ``location.pathname`` to work out which mount it is running under (a page
    fetched through a rewriting proxy, or opened at a path nginx normalised,
    would parse the wrong answer out of it).

    Reads the same value :func:`~osprey.interfaces.common_middleware.compute_url_prefix`
    reads, with the same blank-means-unset rule, so the scope and the
    ``/u/<user>`` prefix can never name different users.

    Args:
        terminal_user: The deployment's mount user (``OSPREY_TERMINAL_USER``,
            as captured on ``app.state.terminal_user``). ``None``, empty or
            blank is a single-user/dev deployment.

    Returns:
        The namespace token, or ``""`` when there is no mount user. Callers
        must render the attribute **only** for a truthy result: an empty
        ``data-osprey-storage-scope=""`` reads as "scoped to nothing" rather
        than "unscoped", and single-user markup must stay exactly as it was.
    """
    return str(terminal_user or "").strip()


#: The two supported rail positions. ``left`` is the icon-rail column;
#: ``top`` renders the same rail as a horizontal strip under the header.
RAIL_POSITIONS = ("left", "top")
DEFAULT_RAIL_POSITION = "left"

#: Onboarding-tour invite policy (``web.tour``, per-user override via the
#: ``OSPREY_WEB_TOUR`` environment variable — the ``web.theme`` /
#: ``OSPREY_WEB_THEME`` shape, so a roster can arm the invite per user).
#: ``once``: the invite shows until this browser dismisses it ("Don't show
#: this again", or completing the tour). ``always``: the invite shows on
#: every load and offers no permanent dismissal — the shared read-only
#: screen case. ``never``: no automatic invite; the tour stays reachable
#: from the rail's Tour control and the command palette.
TOUR_POLICIES = ("once", "always", "never")
DEFAULT_TOUR_POLICY = "once"


def resolve_tour_policy(configured: str | None) -> str:
    """Resolve ``web.tour`` / ``OSPREY_WEB_TOUR`` into a concrete policy.

    Mirrors the warn+fallback shape of :func:`resolve_ui_mode`: anything
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

#: Tracker kinds ``web.feedback.trackers`` accepts, each with the radio caption
#: used when an entry names no ``label`` of its own. The client's URL builders
#: (``static/js/feedback-prefill.js``) are keyed by the same two words.
FEEDBACK_TRACKER_LABELS: dict[str, str] = {"github": "GitHub", "gitlab": "GitLab"}

#: Recipient of the prefilled ``mailto:`` draft when ``web.feedback.email``
#: is absent — the OSPREY maintainers.
DEFAULT_FEEDBACK_EMAIL = "thellert@lbl.gov"

#: Ceiling (bytes) on the on-disk feedback store before the oldest saved
#: contexts are pruned; 256 MB unless ``web.feedback.max_store_bytes`` says
#: otherwise. Submission headers are never pruned, only their contexts.
DEFAULT_FEEDBACK_MAX_STORE_BYTES = 256 * 1024 * 1024

#: How each value of the forwarded role-source header reads in the session
#: menu. The keys are the auth sidecar's closed vocabulary, spelled here
#: rather than imported for the same reason the header names are — the
#: terminal has no business pulling a service package into its import
#: closure — and ``tests/interfaces/web_terminal/test_terminal_user.py``
#: pins this key set against the sidecar's constants. Any other value
#: renders nothing at all: an unrecognised source is a source this build
#: cannot name, and a chip that named it anyway would be guessing.
_ROLE_SOURCE_LABELS: dict[str, str] = {"roster": "roster", "claim": "ID token"}


# ── Bar items: the server side of the two item hosts ──────────────────────
#
# The global header and the status bar are item HOSTS: each renders one
# ordered run of ``.bar-item`` shells, and ``root()`` emits that run from the
# *effective layout* so the operator's order is on the page before any module
# script runs. Everything in this section is the SSR half of the contract the
# browser catalog (``static/js/bar-catalog.js``) and the layout normalizer
# (``static/js/bar-layout.js``) own on the client.

#: Schema version of the layout document. Mirrors ``BAR_LAYOUT_VERSION`` in
#: ``static/js/bar-layout.js``: the browser refuses a document whose version
#: it cannot read, so the two constants move together or not at all.
BAR_LAYOUT_VERSION = 1

#: The two hosts, in render order. Mirrors ``BAR_HOSTS`` in ``bar-catalog.js``.
BAR_HOSTS: tuple[str, ...] = ("header", "status")

#: The deployment's default layout — the order every request renders until a
#: saved document says otherwise.
#:
#: The header is the header as it stood before the bars became hosts: this
#: build moves nothing an operator already reaches for. Its ``space`` item is
#: what used to be the ``.header-left`` / ``.header-right`` split — spacing an
#: operator can see and move, instead of a wrapper they cannot.
#:
#: The status bar ships almost empty: a space, so the rest sits at the right
#: edge, then the system-health dot and the clock. Everything else the bar can
#: show — the documentation link, the stopwatch — stays in the catalog and is one drag away in Customize; a screen an
#: operator watches all day should start with what they read, not with what
#: the terminal knows about itself. ``system-health`` renders only where this
#: deployment enables the SYSTEM panel — :func:`bar_render_plan` drops it
#: otherwise rather than painting an empty shell — so a deployment without
#: that panel gets ``space · clock``.
#:
#: ``web.bar_items`` overrides this per deployment, and the sheet's one preset,
#: "Default", returns a user to whichever of the two applies.
#:
#: Every type here must exist in ``bar-catalog.js`` and declare the host it is
#: placed in; the catalog is the authority and this list is validated against
#: it by the client on every load, so a type removed there simply stops
#: rendering rather than breaking the page.
DEFAULT_BAR_LAYOUT: dict = {
    "version": BAR_LAYOUT_VERSION,
    "rev": 0,
    "header": [
        {"type": "logo"},
        {"type": "identity"},
        {"type": "space"},
        {"type": "control-target"},
        {"type": "search"},
        {"type": "display"},
    ],
    "status": [
        {"type": "space"},
        {"type": "system-health"},
        {"type": "clock"},
    ],
    "header_visible": True,
    "status_visible": True,
}

#: The ADOPTED item types: the ones whose body is a literal block in
#: ``index.html`` rather than a JS-built one.
#:
#: These are the nodes other modules resolve **by id** — ``#docs-link``,
#: ``#command-palette-btn``, ``#display-menu-settings``,
#: ``#logout-btn``, the identity trigger and menu. They are therefore
#: rendered on EVERY request whether or not the layout places
#: them: into their shell when placed, into ``#bar-item-pool`` when not. A
#: layout that drops an item must never make an id go dark — the module that
#: looks it up has no way to know the difference between "the operator removed
#: this item" and "this build is broken".
ADOPTED_BAR_ITEM_TYPES: tuple[str, ...] = (
    "logo",
    "identity",
    "search",
    "display",
    "docs",
)

#: Every item type this build renders — the Python mirror of the keys of
#: ``BAR_CATALOG`` in ``static/js/bar-catalog.js``. Any type may sit in either
#: bar; there is no placement axis on either side.
#:
#: The server needs its own copy because it renders first: a type this build
#: does not know is dropped here exactly as the client's normalizer drops it,
#: so the page never paints a shell the first reconcile takes away.
#: ``tests/interfaces/web_terminal/test_bar_items_ssr.py`` pins the tuple
#: against the JS catalog so the two cannot drift.
BAR_ITEM_TYPES: tuple[str, ...] = (
    "logo",
    "identity",
    "control-target",
    "display",
    "search",
    "system-health",
    "bluesky-queue",
    "docs",
    "feedback",
    "clock",
    "stopwatch",
    "space",
    "separator",
)

#: The types a layout may place more than once — the Python mirror of every
#: entry's ``multi`` in ``static/js/bar-catalog.js``. Every other type is one
#: server-rendered node or one id-owning dot, and a second shell for it could
#: only ever be empty: :func:`bar_render_plan` skips the duplicate, the store
#: refuses it, and the browser's normalizer drops it, all from this one list.
BAR_ITEM_MULTI: frozenset[str] = frozenset({"clock", "stopwatch", "space", "separator"})

#: Each item type's option specs — the Python mirror of every entry's
#: ``options`` in ``static/js/bar-catalog.js``, in the same three shapes its
#: ``numberSpec`` / ``booleanSpec`` / ``enumSpec`` builders produce. Only the
#: types that declare options appear; every other entry is ``NO_OPTIONS``
#: there and is absent here, which :func:`bar_item_vocabulary` reads as "no
#: options".
#:
#: The server needs its own copy for the same reason it mirrors
#: :data:`BAR_ITEM_TYPES`: the store validates a saved layout against these
#: specs, and without them a bad option value would be written to disk and then
#: refused by the browser on the next paint. ``step`` and ``unit`` are the
#: catalog's, not the store's — the customize sheet renders them, so they are
#: carried rather than dropped. ``values`` is a tuple because the vocabulary is
#: frozen and shared across requests.
BAR_ITEM_OPTIONS: dict[str, dict[str, dict]] = {
    "clock": {
        "zone": {"kind": "enum", "values": ("none", "local", "utc", "both"), "default": "none"},
        "format": {"kind": "enum", "values": ("24h", "12h"), "default": "24h"},
        "seconds": {"kind": "boolean", "default": False},
    },
    # Width 0 is the flexible space; anything else is a fixed width the
    # operator dragged the space to. The ceiling is the catalog's
    # ``SPACE_MAX_WIDTH``.
    "space": {
        "width": {"kind": "number", "min": 0, "max": 2000, "step": 1, "unit": "px", "default": 0},
    },
    "system-health": {
        "text": {"kind": "enum", "values": ("none", "status"), "default": "none"},
        "detail": {"kind": "enum", "values": ("categories", "checks"), "default": "categories"},
    },
    "bluesky-queue": {
        "controls": {"kind": "enum", "values": ("none", "stop", "full"), "default": "none"},
        "progress": {"kind": "boolean", "default": True},
        "count": {"kind": "boolean", "default": True},
    },
}

#: The built-in panel the system-health item reads through: its ``/checks``
#: report is proxied at ``/panel/system-health/`` exactly where the panel is
#: enabled, so the panel's presence is the one fact the item depends on.
SYSTEM_HEALTH_PANEL_ID = "system-health"

#: Which item types are gated on a deployment fact, and how — the Python
#: mirror of every catalog entry whose ``available`` is not ``ALWAYS``.
#:
#: The context these read is the one :func:`bar_availability_context` builds
#: and ``root()`` stamps on ``<html>``, in the catalog's own spelling, so the
#: server's answer and the browser's are the same answer rather than two
#: guesses at it. A type absent from this table is available everywhere, which
#: is what ``ALWAYS`` means in the catalog.
#:
#: ``tests/interfaces/web_terminal/test_bar_items_ssr.py`` pins the key set
#: against the catalog's declarations, so an item that grows a condition there
#: cannot quietly stay unconditional here.
BAR_ITEM_AVAILABILITY: dict[str, Callable[[dict], bool]] = {
    "identity": lambda ctx: ctx.get("identityAvailable") is True,
    "bluesky-queue": lambda ctx: ctx.get("blueskyAvailable") is True,
    "system-health": lambda ctx: ctx.get("systemHealthAvailable") is True,
}


def bar_availability_context(
    *,
    identity_available: bool,
    bluesky_available: bool,
    system_health_available: bool,
) -> dict:
    """What this deployment offers, in the vocabulary the catalog asks in.

    One evaluation, two readers: :func:`bar_render_plan` renders from it, and
    ``root()`` stamps it on ``<html>`` for ``bar-sync.js`` to hand to
    ``bar-layout.js``'s ``normalize()``. The browser used to infer these
    facts from the page instead — identity and the plan queue from whether a
    shell had been rendered — and an inference that goes wrong in either
    direction costs something real: a
    false "unavailable" drops the item and latches the whole document
    read-only for the session, so saving dies silently, while a false
    "available" lets the client put back an item this deployment cannot render.

    Args:
        identity_available: Whether the deployment renders an identity block.
        bluesky_available: Whether it declares the Bluesky panel, whose proxy
            is where the plan-queue item reads the queue.
        system_health_available: Whether it enables the SYSTEM panel, whose
            proxy is where the system-health item reads the report.

    Returns:
        The context, JSON-serializable exactly as stamped.
    """
    return {
        "identityAvailable": bool(identity_available),
        "blueskyAvailable": bool(bluesky_available),
        "systemHealthAvailable": bool(system_health_available),
    }


def bluesky_panel_declared(custom_panels: list[dict] | None) -> bool:
    """Whether this deployment declares the Bluesky panel.

    The plan-queue item reads the queue through that panel's proxy
    (``/panel/bluesky/queue`` and its event stream), so the panel's presence is
    the one fact the item depends on: where the panel is declared the queue
    can be reached, and where it is not there is nothing to reach. The panel
    declaration is also the bridge entitlement itself (see
    :data:`~osprey.deployment.web_terminals.personas.BLUESKY_PANEL_ID`), so
    this asks no second key for the same fact.

    Args:
        custom_panels: The config-declared panels (``app.state.custom_panels``),
            or None before the lifespan has run.

    Returns:
        True when a declared panel carries the Bluesky panel's id.
    """
    # Lazy, like every other reach into the deployment package from here: the
    # terminal app must not import deployment code to boot.
    from osprey.deployment.web_terminals.personas import BLUESKY_PANEL_ID

    return any(panel.get("id") == BLUESKY_PANEL_ID for panel in custom_panels or ())


class BarRenderPlan(NamedTuple):
    """Everything ``index.html`` needs to render both hosts and the pool.

    Attributes:
        header: Ordered header shells, each ``{"type", "adopted", "follows"}``.
        status: Ordered status-bar shells, same shape.
        pooled: Adopted types this deployment renders but this layout does not
            place — their nodes go into ``#bar-item-pool``.
        header_visible: False ⇒ ``<html data-header-bar="hidden">``.
        status_visible: False ⇒ ``<html data-status-bar="hidden">``.
    """

    header: list[dict]
    status: list[dict]
    pooled: list[str]
    header_visible: bool
    status_visible: bool


def effective_bar_layout(app: FastAPI) -> dict:
    """The layout document this request renders.

    THE seam, and now with both its sources wired: the operator's own saved
    arrangement (``app.state.bar_items_effective``) is read ahead of the
    deployment default (``app.state.bar_layout``), and neither ``root()`` nor
    the template learns that anything changed.

    The cache is populated once at startup and refreshed by the layout routes
    after every accepted save, so a render never touches the disk. ``None``
    there is the honest reading of "this operator has saved nothing" — a reset
    puts it back — and the deployment default is then what renders.

    ``app.state.bar_layout`` is honoured when present, which is what lets a
    test hand a document in without reaching into the renderer.

    A document whose ``version`` this build cannot read is refused here, for
    the same reason ``bar-layout.js`` refuses it on the client: a document
    written by a newer build that the server rendered and the client discarded
    would paint one arrangement and hydrate into another — and the server's
    pool membership would have been computed from a layout the client no
    longer holds, so an adopted node could be parked on the page while the
    client believes it is placed. Refusing in both halves keeps first paint and
    hydration talking about the same document.

    Args:
        app: The application whose state may carry a stored document.

    Returns:
        A readable layout document. Never None; falls back to
        :data:`DEFAULT_BAR_LAYOUT`. It is the shared cache itself, not a copy,
        so callers must treat it as immutable — a route that normalized or
        annotated it in place would corrupt every later render in the process.
        The writers are already safe: ``save_layout`` returns its own copy.
    """
    for candidate in (
        getattr(app.state, "bar_items_effective", None),
        getattr(app.state, "bar_layout", None),
    ):
        if isinstance(candidate, dict) and candidate.get("version") == BAR_LAYOUT_VERSION:
            return candidate
    return DEFAULT_BAR_LAYOUT


def _load_stored_bar_layout(
    store_dir: Path, vocabulary: BarVocabulary, default: dict
) -> dict | None:
    """This operator's saved arrangement, or ``None`` when they have none.

    ``None`` rather than a copy of *default*, because "nothing saved" and
    "saved something that happens to match the deployment" are different
    states: only the first one follows the deployment when an operator edits
    ``web.bar_items``, and only the first one is what a reset returns to.
    :func:`effective_bar_layout` reads it exactly that way.

    The existence check is what keeps a first-ever boot from paying for a read
    and a fallback copy. A document that exists but cannot be read — truncated,
    not JSON, or written at a schema version this build does not know — answers
    ``None`` too: the store hands back *default* there, and a cache holding a
    copy of the deployment default would say "this operator saved something"
    about a file nothing can read. Rendering the deployment's own arrangement
    is the right outcome for a damaged file, and it is what ``None`` already
    means. The store's revisions are what make the two cases separable: every
    accepted save lands at 1 or above, so ``rev 0`` is precisely "nothing
    readable was stored".

    Args:
        store_dir: The bar-items store directory, which need not exist.
        vocabulary: The facts the stored envelope is checked against.
        default: This deployment's configured layout, used when a document is
            present but unreadable.

    Returns:
        The stored document, or ``None`` when nothing readable is stored.
        ``None`` also covers a store that could not be reached at all, so it
        does not prove the mount is writable — the routes must treat a failure
        from ``save_layout`` as authoritative rather than inferring anything
        from what the boot managed to read.
    """
    try:
        if not layout_path(store_dir).exists():
            return None
    except OSError:  # pragma: no cover — an unreadable mount is not a boot failure
        logger.warning("Could not reach the bar-items store at %s", store_dir, exc_info=True)
        return None
    document = load_layout(store_dir, vocabulary=vocabulary, default=default)
    return document if document["rev"] else None


def bar_render_plan(layout: dict, *, context: dict) -> BarRenderPlan:
    """Turn a layout document into the two ordered runs plus the pool.

    Three rules do all the work here, and each one is a refusal the client's
    normalizer (``bar-layout.js``) already makes. The SSR is the first paint of
    what that normalizer will produce, so anything it renders that the client
    would drop is a rearrangement the operator watches happen.

    **Adopted nodes are conserved.** Every adopted type this deployment would
    render is emitted exactly once — in its shell if the layout places it, in
    the pool if it does not. Never twice (a layout naming ``docs`` twice would
    otherwise emit ``#docs-link`` twice, and ``getElementById`` would silently
    pick one), and never zero times.

    **An unavailable item is ABSENT, not empty.** An identity block needs
    something to identify with, a plan queue needs a Bluesky bridge to have a
    queue, and the system-health chip needs the SYSTEM panel to read from.
    Where the fact is absent, so is the shell — not just the body, and whether
    or not the item is one of the adopted ones. An empty shell is not free:
    ``bars.css``'s ``[data-follows="logo"]`` rule is keyed on the SHELL, so an
    empty identity shell paints a dangling middot after the wordmark on every
    single-user deployment, and each empty shell costs the run another gap. An
    adopted node is left out of the pool too, because seeding an id this
    deployment never renders would tell every reader it exists.

    Availability is asked of *context* — the deployment facts
    :func:`bar_availability_context` evaluates once and ``root()`` stamps on
    the page — through :data:`BAR_ITEM_AVAILABILITY`, which mirrors the
    catalog's own ``available()`` predicates. Server and browser therefore
    answer the question from the same facts instead of each guessing.

    **A type this build does not know is skipped.** Either bar may hold any
    known type (see :data:`BAR_ITEM_TYPES`); an unknown one is dropped exactly
    as the client's normalizer drops it.

    **A single-node type renders once.** A second entry for a type outside
    :data:`BAR_ITEM_MULTI` is skipped — the client's normalizer drops it, and
    a shell rendered for it would be an empty box the first reconcile takes
    away again.

    ``follows`` chains off the previous EMITTED item rather than the previous
    layout entry, so a skipped item cannot leave the middot stranded on a shell
    that is no longer next to the logo.

    Args:
        layout: The effective layout document.
        context: The deployment context from :func:`bar_availability_context`.

    Returns:
        The render plan the template consumes.
    """

    def _is_available(item_type: str) -> bool:
        predicate = BAR_ITEM_AVAILABILITY.get(item_type)
        return True if predicate is None else predicate(context)

    seen: set[str] = set()
    runs: dict[str, list[dict]] = {}
    for host in BAR_HOSTS:
        shells: list[dict] = []
        previous = ""
        for raw in layout.get(host) or []:
            if not isinstance(raw, dict):
                continue
            item_type = str(raw.get("type") or "")
            if item_type not in BAR_ITEM_TYPES:
                continue
            adopted = item_type in ADOPTED_BAR_ITEM_TYPES
            if not _is_available(item_type):
                # Absent, not empty. An adopted type is marked seen as well, so
                # the pool does not put back a node this deployment never
                # renders; a JS-built one has no node to pool in the first
                # place, and an empty shell would only be a box the client
                # takes away again on the first reconcile.
                seen.add(item_type)
                continue
            if item_type not in BAR_ITEM_MULTI:
                if item_type in seen:
                    continue
                seen.add(item_type)
            shells.append({"type": item_type, "adopted": adopted, "follows": previous})
            previous = item_type
        runs[host] = shells

    pooled = [
        item_type
        for item_type in ADOPTED_BAR_ITEM_TYPES
        if item_type not in seen and _is_available(item_type)
    ]
    return BarRenderPlan(
        header=runs["header"],
        status=runs["status"],
        pooled=pooled,
        header_visible=bool(layout.get("header_visible", True)),
        status_visible=bool(layout.get("status_visible", True)),
    )


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


def coerce_feedback_trackers(value: object) -> list[dict[str, str]]:
    """Return ``web.feedback.trackers`` as a list of normalised tracker entries.

    Each usable entry becomes ``{"kind", "label", "repo"}`` (GitHub, ``repo`` an
    ``owner/name``) or ``{"kind", "label", "url"}`` (GitLab, ``url`` the
    project's base URL — gitlab.com or self-hosted, trailing slash dropped).
    A missing ``label`` takes the kind's own name.

    Lenient per entry, strict per field: one malformed line — an unknown
    ``kind``, a GitHub entry without an ``owner/name`` repo, a GitLab entry
    whose ``url`` is not ``http(s)://`` — is reported and dropped while the rest
    of the list stands, because one typo must not retire every tracker the
    facility configured. A value that is not a list at all is reported and
    reads as no list.

    Args:
        value: Whatever the config reader returned for the key.

    Returns:
        The usable entries, in the order written.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        logger.warning("web.feedback.trackers is %r, not a list; ignoring it", value)
        return []
    trackers: list[dict[str, str]] = []
    for index, entry in enumerate(value):
        tracker = _coerce_feedback_tracker(entry)
        if tracker is None:
            logger.warning("web.feedback.trackers[%d] is %r; dropping it", index, entry)
            continue
        trackers.append(tracker)
    return trackers


def _coerce_feedback_tracker(entry: object) -> dict[str, str] | None:
    """One entry of :func:`coerce_feedback_trackers`, or ``None`` when unusable."""
    if not isinstance(entry, dict):
        return None
    kind = entry.get("kind")
    if not isinstance(kind, str) or kind.strip() not in FEEDBACK_TRACKER_LABELS:
        return None
    kind = kind.strip()
    label = entry.get("label")
    label = label.strip() if isinstance(label, str) and label.strip() else ""
    tracker = {"kind": kind, "label": label or FEEDBACK_TRACKER_LABELS[kind]}
    if kind == "github":
        repo = entry.get("repo")
        repo = repo.strip() if isinstance(repo, str) else ""
        if repo.count("/") != 1 or any(ch.isspace() for ch in repo) or not all(repo.split("/")):
            return None
        tracker["repo"] = repo
    else:
        url = entry.get("url")
        url = url.strip().rstrip("/") if isinstance(url, str) else ""
        if not url.startswith(("http://", "https://")) or any(ch.isspace() for ch in url):
            return None
        tracker["url"] = url
    return tracker


def resolve_feedback_trackers(
    trackers: list[dict[str, str]], github_repo: str
) -> list[dict[str, str]]:
    """The tracker list the dialog offers: the configured list plus the sugar.

    ``web.feedback.github_repo`` keeps its meaning as a single GitHub tracker,
    appended after the facility-authored list (blank retires it — that is the
    posture :func:`coerce_config_str` preserves). Two entries naming the same
    target collapse to the first, so a facility that lists the upstream repo
    under its own label does not get it rendered twice by the sugar.

    Args:
        trackers: Output of :func:`coerce_feedback_trackers`.
        github_repo: Resolved ``web.feedback.github_repo`` (``""`` when blank).

    Returns:
        The de-duplicated list, in render order.
    """
    candidates = list(trackers)
    if github_repo:
        candidates.append(
            {"kind": "github", "label": FEEDBACK_TRACKER_LABELS["github"], "repo": github_repo}
        )
    seen: set[tuple[str, str]] = set()
    resolved: list[dict[str, str]] = []
    for tracker in candidates:
        key = (tracker["kind"], tracker.get("repo") or tracker.get("url") or "")
        if key in seen:
            continue
        seen.add(key)
        resolved.append(dict(tracker))
    return resolved


#: Words a human writes when they mean a boolean but YAML kept a string —
#: ``enabled: "false"`` (quoted) is the one that matters, because for a
#: default-ON switch a bare ``bool("false")`` is ``True``: the deployment
#: would read as having asked for the switch OFF and get it left ON.
_FALSE_WORDS = frozenset({"false", "no", "off", "0"})
_TRUE_WORDS = frozenset({"true", "yes", "on", "1"})


def resolve_config_flag(key: str, default: bool, on_error: str) -> bool:
    """Read a configured boolean switch at startup, failing OPEN to *default*.

    The read and the coercion are one step because the two failure modes want
    the same answer: a config that cannot be loaded and a value nobody can
    interpret both leave the switch at the deployment's shipped posture. A
    startup switch that silently revoked a surface because the config file was
    briefly unreadable would be the worst possible failure here.

    Args:
        key: Dotted config key, read and reported verbatim.
        default: Posture for a deployment that never mentions the key.
        on_error: Warning logged when the config cannot be read at all; it says
            which switch was left at its default and what that means.

    Returns:
        The configured boolean, or *default*.
    """
    try:
        from osprey.utils.config import get_config_value

        raw = get_config_value(key, default)
    except Exception:  # noqa: BLE001 — never let config load block startup
        logger.warning(on_error, exc_info=True)
        raw = None
    return coerce_config_flag(key, raw, default)


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
        # One predicate for builtin and custom blocks alike: the build writes
        # `enabled` onto every block from the profile's `web_panels` selection,
        # so a block this render carries for a tab it does not select is off.
        if not panel_spec_enabled(spec):
            continue
        if panel_id in BUILTIN_PANELS:
            enabled.add(panel_id)
        elif isinstance(spec, dict):
            custom.append(
                {
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


#: Most items one host may hold. The Python mirror of ``MAX_ITEMS_PER_HOST``
#: in ``static/js/bar-layout.js``: the client drops everything past this
#: number, so a deployment that configured more would paint a bar the browser
#: immediately shortens — the first-paint flash the SSR exists to prevent. Per
#: host, not per document, for the reason stated there: two capped lists
#: already bound the total, and one shared cap would make a legal header edit
#: fail because of the status bar.
MAX_BAR_ITEMS_PER_HOST = 20


def bar_item_vocabulary() -> BarVocabulary:
    """The deployment facts :mod:`.bar_items_store` validates a layout against.

    Assembled from the tables above rather than restated: the store takes its
    vocabulary as a parameter precisely so it cannot become a second authority
    on item names, the schema version or the per-host cap, and so that
    ``tests/interfaces/web_terminal/test_bar_items_ssr.py``'s pin against
    ``bar-catalog.js`` guards the store's answers too.

    This is also where the cap's direction is settled: the store never imports
    :data:`MAX_BAR_ITEMS_PER_HOST`, it is handed it. ``app.py`` imports the
    store; a constant travelling the other way would close the cycle.

    Returns:
        A vocabulary describing every type this build can render.
    """
    return BarVocabulary(
        items={
            item_type: {
                "options": BAR_ITEM_OPTIONS.get(item_type, {}),
                "multi": item_type in BAR_ITEM_MULTI,
            }
            for item_type in BAR_ITEM_TYPES
        },
        version=BAR_LAYOUT_VERSION,
        max_items_per_host=MAX_BAR_ITEMS_PER_HOST,
        hosts=BAR_HOSTS,
    )


def _coerce_bar_item(host: str, index: int, raw: object) -> dict | None:
    """One configured entry as a layout item, or ``None`` if it cannot be one.

    Two spellings are accepted, because both read naturally in YAML: a bare
    string (``- clock``) for an item with no options, and a mapping
    (``- {type: clock, options: {zone: utc}}``) for one with them. Option
    VALUES are not validated here — the catalog owns each type's option spec
    and the browser applies it — but a non-mapping ``options`` is dropped,
    since nothing downstream could read it.

    Args:
        host: The bar the entry was written under, for the warning text.
        index: The entry's position in the raw list, so a warning points at
            the config the operator wrote rather than at the result.
        raw: The entry, exactly as YAML produced it.

    Returns:
        A ``{"type": ...}`` item, optionally carrying ``options``; ``None``
        when the entry is malformed or names a type this build does not know.
    """
    where = f"web.bar_items.{host}[{index}]"
    options: object = None
    if isinstance(raw, str):
        item_type = raw
    elif isinstance(raw, dict) and isinstance(raw.get("type"), str):
        item_type = raw["type"]
        options = raw.get("options")
    else:
        logger.warning("%s is not an item type or a mapping with one; dropping it.", where)
        return None

    if item_type not in BAR_ITEM_TYPES:
        logger.warning("%s names unknown bar item %r; dropping it.", where, item_type)
        return None

    item: dict = {"type": item_type}
    if options is not None:
        if isinstance(options, dict):
            item["options"] = dict(options)
        else:
            logger.warning("%s has non-mapping options; dropping them.", where)
    return item


def _load_bar_items(config_path: str | Path | None = None) -> dict:
    """Read ``web.bar_items`` into this deployment's default bar layout.

    The deployment's half of the bar-items contract: an operator's saved
    layout wins over this, and this wins over :data:`DEFAULT_BAR_LAYOUT`. It is
    the same fail-open coercion :func:`_load_panel_presets` performs for
    ``web.presets``, and for the same reason — there is no config schema
    anywhere, so a typo must cost the operator one item and never the boot.

    Four keys, each degrading on its own:

    * ``header`` / ``status`` — ordered lists of item types. A list that is not
      a list falls back to the shipped order for that bar; an entry that is
      malformed or names an unknown type is warned about and dropped while its
      neighbours survive. An explicitly
      empty list means an empty bar and is honoured. A bar that is not
      configured at all keeps the shipped order, so naming one bar never
      silently empties the other.
    * ``header_visible`` / ``status_visible`` — hide a bar without emptying
      it. Anything that is not a boolean falls back to visible.

    Args:
        config_path: Explicit ``config.yml`` to read; falls back to the
            ``CONFIG_FILE`` environment variable and then ``./config.yml``.

    Returns:
        The resolved layout document. Falls open to :data:`DEFAULT_BAR_LAYOUT`
        on any read error or when the key is absent.
    """
    try:
        raw = _load_web_ui_config(config_path).get("bar_items")
    except Exception:  # noqa: BLE001 — an unreadable config renders the shipped bars
        logger.warning("web.bar_items could not be read; using the default layout.")
        return DEFAULT_BAR_LAYOUT

    if raw is None:
        return DEFAULT_BAR_LAYOUT
    if not isinstance(raw, dict):
        logger.warning("web.bar_items is not a mapping; using the default layout.")
        return DEFAULT_BAR_LAYOUT

    layout: dict = {"version": BAR_LAYOUT_VERSION, "rev": 0}
    # Single-node types are counted across both bars, header first, exactly as
    # the browser's normalizer counts them.
    placed: set[str] = set()
    for host in BAR_HOSTS:
        configured = raw.get(host)
        if configured is None:
            layout[host] = list(DEFAULT_BAR_LAYOUT[host])
            placed.update(item["type"] for item in layout[host])
            continue
        if not isinstance(configured, list):
            logger.warning(
                "web.bar_items.%s is not a list of item types; using the default order.", host
            )
            layout[host] = list(DEFAULT_BAR_LAYOUT[host])
            placed.update(item["type"] for item in layout[host])
            continue
        items: list[dict] = []
        for index, entry in enumerate(configured):
            item = _coerce_bar_item(host, index, entry)
            if item is None:
                continue
            if item["type"] not in BAR_ITEM_MULTI:
                if item["type"] in placed:
                    logger.warning(
                        "web.bar_items.%s[%d] places %r a second time; dropping it.",
                        host,
                        index,
                        item["type"],
                    )
                    continue
                placed.add(item["type"])
            items.append(item)
        if len(items) > MAX_BAR_ITEMS_PER_HOST:
            logger.warning(
                "web.bar_items.%s holds %d items; keeping the first %d.",
                host,
                len(items),
                MAX_BAR_ITEMS_PER_HOST,
            )
            items = items[:MAX_BAR_ITEMS_PER_HOST]
        layout[host] = items

    for host in BAR_HOSTS:
        flag = f"{host}_visible"
        visible = raw.get(flag, DEFAULT_BAR_LAYOUT[flag])
        if not isinstance(visible, bool):
            logger.warning("web.bar_items.%s is not true or false; showing the bar.", flag)
            visible = DEFAULT_BAR_LAYOUT[flag]
        layout[flag] = visible

    return layout


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
        from osprey.utils.workspace import reset_config_cache

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
            # Kept for the rail-position block below (an unconfigured rail
            # follows the family — see FAMILY_RAIL_DEFAULTS).
            app.state.web_theme_family = web_theme.family
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

        # ── Onboarding-tour invite policy ──
        # OSPREY_WEB_TOUR (the per-user roster path) outranks web.tour, the
        # same precedence OSPREY_WEB_THEME has over web.theme. Resolved once
        # at startup; GET /api/panels echoes it to the browser. Fails open to
        # the default policy on any config-read error.
        try:
            from osprey.utils.workspace import load_osprey_config

            configured_tour = os.environ.get(
                "OSPREY_WEB_TOUR", ""
            ).strip() or load_osprey_config().get("web", {}).get("tour", DEFAULT_TOUR_POLICY)
            app.state.web_tour_policy = resolve_tour_policy(configured_tour)
        except Exception:  # noqa: BLE001 — never let config load block startup
            logger.warning(
                "Could not resolve web.tour (config load failed); falling back to policy %r",
                DEFAULT_TOUR_POLICY,
                exc_info=True,
            )
            app.state.web_tour_policy = DEFAULT_TOUR_POLICY

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
        # ``getattr(app.state, "config_panel_enabled", True)``, so an app built
        # without this lifespan (the route unit suites) behaves like a
        # deployment that never mentioned the key. Fails OPEN, deliberately:
        # the default posture is the panel every single-user deployment has
        # always had, and an unreadable config must not silently take an
        # operator's own config editor away.
        app.state.config_panel_enabled = resolve_config_flag(
            "web.config_panel.enabled",
            True,
            "Could not read web.config_panel.enabled; leaving the Config panel enabled",
        )

        # ── Scaffold gallery writes (server-side tier gate) ──
        # `web.scaffold_gallery.write_enabled: false` closes the gallery's whole
        # WRITE surface: create, claim, save, unoverride, register and delete
        # under /api/scaffold all refuse with 403, while every read route stays
        # open. What the gallery authors is `.claude/rules|skills|agents`
        # content, which the agent loads at PROJECT scope — instruction it
        # obeys, not decoration — so "may this tier author it" is a privilege
        # boundary. Hiding the buttons alone would be the same cosmetic gate
        # `ui_mode: simple` was: a client-only guard is undone by curl.
        #
        # Resolved once here and read back as
        # ``getattr(app.state, "scaffold_write_enabled", True)``, so an app
        # built without this lifespan (the route unit suites) behaves like a
        # deployment that never mentioned the key. Fails OPEN, deliberately:
        # the default posture is the gallery every single-user deployment has
        # always had, and a config-read error must not silently revoke it.
        app.state.scaffold_write_enabled = resolve_config_flag(
            "web.scaffold_gallery.write_enabled",
            True,
            "Could not read web.scaffold_gallery.write_enabled; "
            "leaving the scaffold gallery writable",
        )

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
            raw_trackers = get_config_value("web.feedback.trackers", None)
            raw_max_store_bytes = get_config_value(
                "web.feedback.max_store_bytes", DEFAULT_FEEDBACK_MAX_STORE_BYTES
            )
        except Exception:  # noqa: BLE001 — never let config load block startup
            logger.warning(
                "Could not read web.docs_url / web.feedback.* config keys; using defaults",
                exc_info=True,
            )
            raw_docs_url = raw_github_repo = raw_email = raw_trackers = raw_max_store_bytes = None
        app.state.docs_url = coerce_config_str("web.docs_url", raw_docs_url, DEFAULT_DOCS_URL)
        app.state.feedback_github_repo = coerce_config_str(
            "web.feedback.github_repo", raw_github_repo, DEFAULT_FEEDBACK_GITHUB_REPO
        )
        app.state.feedback_trackers = resolve_feedback_trackers(
            coerce_feedback_trackers(raw_trackers), app.state.feedback_github_repo
        )
        app.state.feedback_email = coerce_config_str(
            "web.feedback.email", raw_email, DEFAULT_FEEDBACK_EMAIL
        )
        app.state.feedback_max_store_bytes = coerce_store_ceiling(raw_max_store_bytes)

        # The server-side stores are sited on the CONFIGURED agent-data root
        # and deliberately NOT on workspace_dir: with web_terminal.watch_dir
        # set, the watched tree is somewhere else entirely and anything written
        # there would land outside the {user}-agent-data volume, where
        # `osprey feedback` on the host cannot reach it. Never
        # resolve_agent_data_root() either — that appends sessions/<id>, and
        # both stores span sessions. They are siblings under one root: feedback
        # records, and the per-user bar arrangement.
        try:
            from osprey.utils.workspace import resolve_shared_data_root

            shared_data_root = resolve_shared_data_root()
        except Exception:  # noqa: BLE001 — never let config load block startup
            shared_data_root = workspace_dir
            logger.warning(
                "Could not resolve the shared data root; siting the feedback and "
                "bar-items stores under %s",
                shared_data_root,
                exc_info=True,
            )
        feedback_dir = shared_data_root / "feedback"
        bar_items_dir = shared_data_root / "bar_items"
        app.state.feedback_dir = feedback_dir
        app.state.bar_items_dir = bar_items_dir
        # Workspace-relative form of each store: the *file watcher's* form,
        # used below to drop change events for writes into them. The file
        # browser is not a consumer — routes/files.py derives its own predicate
        # from ``feedback_dir``, because it must also handle symlink aliases and
        # session-scoped roots that a single relative path cannot express.
        # ``None`` when a store lies outside the watched tree (the watch_dir
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
        app.state.bar_items_rel = resolve_store_rel(bar_items_dir, workspace_dir)
        # One collection, in the order the stores are resolved above. Saving a
        # bar arrangement must be as silent as filing feedback: a layout PUT
        # writes one file, and an unconcealed store would push an SSE change
        # frame to every connected browser the moment anyone rearranged a bar.
        # This seam does that and only that. It does NOT hide either store from
        # the file panel — routes/files.py filters the listing and the content
        # read through its own predicate, which knows about the feedback store
        # alone — so bar_items/layout.json is still listed on a refresh. That is
        # deliberate for now: a layout is the operator's own preference, not
        # submitted session context, so it is clutter rather than disclosure.
        app.state.concealed_store_rels = tuple(
            rel for rel in (app.state.feedback_rel, app.state.bar_items_rel) if rel is not None
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

        # The deployment's default bar arrangement (web.bar_items), resolved
        # once at boot. This is the document effective_bar_layout() renders
        # until a user's saved layout is loaded ahead of it.
        app.state.bar_layout = _load_bar_items(resolved_config_path)

        # The per-user layer on top of that default, read from the store beside
        # the feedback records. Three pieces of state, and the layout routes
        # need all three:
        #
        # * ``bar_items_vocabulary`` — the deployment facts a saved document is
        #   validated against, built once here so every request validates
        #   against the same tables the SSR renders from.
        # * ``bar_items_effective`` — the cache ``effective_bar_layout()``
        #   reads. The store is touched once, at boot; a render never goes to
        #   disk. ``None`` means this operator has saved nothing, so the
        #   deployment default renders — which is also what a reset restores.
        # * ``bar_items_lock`` — taken by the routes around load-validate-save,
        #   so two tabs saving at once serialize rather than interleaving a
        #   read with the other's write. It guards the cache as well as the
        #   file, so the two cannot disagree about what is stored.
        app.state.bar_items_vocabulary = bar_item_vocabulary()
        app.state.bar_items_lock = asyncio.Lock()
        app.state.bar_items_effective = _load_stored_bar_layout(
            bar_items_dir, app.state.bar_items_vocabulary, app.state.bar_layout
        )

        # Whether this deployment declares the Bluesky panel: the plan-queue
        # item reads the queue through that panel's proxy, so the panel's
        # presence is what makes the item offerable. It says nothing about
        # whether the sidecar is answering; the item says that itself.
        app.state.bluesky_available = bluesky_panel_declared(custom_panels)

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
        # The two bars, rendered from the effective layout. Server-rendered
        # rather than fetched: the order is chrome, and chrome that arrives a
        # frame late is chrome the operator watches rearrange itself on every
        # load. `identityAvailable` is the same condition the identity block
        # itself renders under (a user to name, or a deployment name to show),
        # stated once so the item and its body can never disagree.
        #
        # The context is evaluated once and used twice: this render drops what
        # the deployment cannot show, and the template stamps the same facts on
        # <html> so bar-sync.js hands the browser's normalizer the server's
        # answer instead of inferring one from the page it just received.
        enabled_panels = getattr(request.app.state, "enabled_panels", None) or set()
        bar_context = bar_availability_context(
            identity_available=bool(terminal_user or app_name),
            bluesky_available=bool(getattr(request.app.state, "bluesky_available", False)),
            system_health_available=SYSTEM_HEALTH_PANEL_ID in enabled_panels,
        )
        plan = bar_render_plan(effective_bar_layout(request.app), context=bar_context)
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
                # Namespace for everything this page keeps in localStorage.
                # Empty on a single-user deployment, and the template omits
                # the attribute entirely in that case — see
                # resolve_storage_scope().
                "storage_scope": resolve_storage_scope(terminal_user),
                "landing_url": landing_url,
                "auth_role": auth_role or "",
                "auth_role_source_label": auth_role_source_label,
                "url_prefix": url_prefix,
                "bar_header": plan.header,
                "bar_status": plan.status,
                "bar_pooled": plan.pooled,
                "bar_header_visible": plan.header_visible,
                "bar_status_visible": plan.status_visible,
                "bar_context": bar_context,
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
        # session id, the dock layout and the rail position, so an unstamped
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
            },
        )

    configure_interface_app(app, static_dir=STATIC_DIR)

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
    """
    import uvicorn

    _open_browser_when_ready(browser_url or f"http://{host}:{port}")

    app = create_app(config_path=config_path, shell_command=shell_command, project_dir=project_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
