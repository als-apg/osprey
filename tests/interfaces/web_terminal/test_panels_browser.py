"""Browser smoke tests: the docked (dockview) workspace and its server bridge.

The service panels do not live in a fixed left/right split — they are docked
in a dockview grid (``dock-workspace.js``) whose panel content follows an overlay
iframe layer (``dock-iframe.js``), with a state bridge (``dock-sync.js``) that
turns human dock gestures into the same server POSTs the agent's MCP calls make.
These tests drive that stack through a real browser and a live SSE stream — the
parts the FastAPI TestClient cannot reach.

One panel per tile is a workspace invariant: the rail is the tab system, so an
activation REPLACES the focused service tile's panel (the evicted panel closes
server-side and dims on the rail), the "+" add menu and the rail entry's
"Open in a new tile" are the verbs that open a NEW tile beside the current one,
and no drop geometry may stack tabs.

What is proven here, grouped:

  * SSE / rail flows materialize as dockview panels: focusing a service panel
    docks a placeholder whose overlay iframe tracks the group geometry — taking
    the focused tile over from its previous panel; hiding a panel removes its
    placeholder; register appends a rail entry WITHOUT auto-docking.
  * The echo guard (``dock-sync.js``): a human dock-tab focus POSTs setPanelFocus
    exactly once, a human tab-close POSTs visibility(false), and a server-driven
    SSE focus is applied WITHOUT POSTing back (no feedback loop).
  * Drag rearrange via real Playwright drags — a tile moves to a new edge
    position, and a drop that would stack (onto another tile's tab bar) is
    vetoed — asserted with getBoundingClientRect geometry, and the overlay
    iframe re-following the dragged panel's new rectangle.
  * Layout persistence keyed by project_key: an expert arrangement survives a
    reload, two project cwds keep distinct layouts on one origin, reset restores
    the default, and a corrupt stored value falls back cleanly.
  * Locked simple mode: no drag affordance (a drag is a no-op) and no per-tab
    close control; an SSE register during simple mode is folded in with no dead
    tabs once the operator flips back to expert.
  * The "+" add menu: unclipped geometry beside the rail, URL-row gating, reveal a
    hidden panel INTO A NEW TILE beside the active one, register from URL, and
    inline register errors.
  * Tile header-bar contributions (the embed contract): a panel's own nav, search
    and overflow items render in ITS bar only, actions round-trip into the iframe,
    and the bar's drag ghost keeps its icons bar-sized.
  * The right-click context menu (``panel-context-menu.js``), the only home left
    for the open-beside and pop-out verbs: the rows a rail entry and a tile header
    offer, the surfaces that DECLINE (an offline entry by mouse or keyboard, a
    contributed header input, the terminal in simple mode), and the dismissals
    that need real focus and real event targets — a click into a panel iframe,
    and the scroll scoping that leaves the menu standing while xterm scrolls.
  * The command palette: it opens with the input focused and typing filters
    immediately, its Panels rows cover the ACTIVE panel and answer domain
    synonyms, a pop-out row really opens the proxied panel URL, and executed
    items come back under "Recent" on an empty query.

Each test launches a real uvicorn server in a background thread, drives events
through the REST API, and asserts the DOM via Playwright auto-waiting.

Run:
    uv run pytest tests/interfaces/web_terminal/test_panels_browser.py -q

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import asyncio
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock, patch

import pytest
import requests

from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _free_port, _run_app_server

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Live-server context manager (preserved from the pre-dock suite)
# ---------------------------------------------------------------------------

# Custom panel used by tests that need a second, immediately-healthy tab.
# healthEndpoint=None → panel becomes healthy immediately (no health polling).
# The raw URL is stored server-side; the browser receives /panel/data-viz.
_CUSTOM_DATA_VIZ: dict = {
    "id": "data-viz",
    "label": "DATA VIZ",
    "url": "http://data-viz.internal:8080",
    "healthEndpoint": None,
    "path": "/",
}


@contextmanager
def _stub_backend():
    """Serve 200 on every path, for a custom panel that must become healthy.

    Panels with a ``healthEndpoint`` are the only ones that reach the
    auto-activate branch in ``pollHealth``, and that branch needs a real
    unhealthy→healthy transition — so it needs a real backend to poll.

    Yields:
        base URL of the stub, e.g. ``"http://127.0.0.1:54321"``.
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):  # silence per-request stderr logging
            pass

    server = HTTPServer(("127.0.0.1", _free_port()), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _live_server(
    workspace_dir,
    enabled_panels,
    custom_panels=None,
    allow_runtime: bool = False,
    artifact_url: str | None = "http://127.0.0.1:8086",
    artifact_config_delay: float = 0.0,
    project_cwd: str | None = None,
    ui_mode: str | None = None,
):
    """Launch a real web terminal server on a free port in a background thread.

    Companion backends (artifact server, ARIEL, etc.) are bypassed via patches
    so no external process dependencies are required.  The artifacts panel
    reports its URL as http://127.0.0.1:8086 (the standard fallback).

    Pass ``artifact_url=None`` to make /api/artifact-server report no URL, which
    leaves the default panel loaded-but-unhealthy.  That is the only way to keep
    ``activeTabId`` null: a panel with ``healthEndpoint: null`` is marked healthy
    unconditionally on load, so pointing it at a dead port would NOT keep it
    unhealthy — only withholding the URL does.

    ``artifact_config_delay`` holds /api/artifact-server open for that many
    seconds, so the default panel settles *after* another panel's health poll.
    That ordering decides which panel gets the empty slot, and it is the order a
    loaded CI runner produces naturally while a fast dev box hides it.

    ``project_cwd`` seeds ``app.state.project_cwd`` before startup so the
    ``/api/panels`` ``project_key`` (the ``osprey-dock-layout-<key>`` localStorage
    suffix) is deterministic — persistence tests set it to control which stored
    layout a page load resolves to.  ``ui_mode`` seeds the server-rendered
    ``<html data-ui-mode>`` for the simple-mode tests.

    Yields:
        (base_url: str, app: FastAPI) — live server address and the FastAPI app
        object for post-startup state manipulation.
    """
    if custom_panels is None:
        custom_panels = []

    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled_panels, custom_panels, None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(artifact_url),
        ),
    ):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])

        if artifact_config_delay:

            @app.middleware("http")
            async def _delay_artifact_config(request, call_next):
                if request.url.path == "/api/artifact-server":
                    await asyncio.sleep(artifact_config_delay)
                return await call_next(request)

        # _run_app_server yields only after the port accepts connections (lifespan
        # done), so app.state is safe to mutate here; and it joins the server
        # thread while the patches above are still live, avoiding timing races.
        with _run_app_server(app) as base_url:
            # These are set AFTER the server is up: lifespan startup assigns both
            # project_cwd and web_ui_mode from config, so a pre-startup value would
            # be clobbered. root() re-reads them per request, so setting them here
            # (before any page load) is authoritative.
            if project_cwd is not None:
                app.state.project_cwd = project_cwd
            if ui_mode is not None:
                app.state.web_ui_mode = ui_mode
            if allow_runtime:
                app.state.allow_runtime_panels = True
                app.state.runtime_panel_allowlist = None  # any non-loopback host allowed

            yield base_url, app


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------

#: Seeded into every page before load: marks the one-time rail hint as already
#: dismissed. The hint floats over the dock tab strip (dropdown z-index), so on
#: the fresh profile these tests run under it would intercept the tab clicks and
#: drags they drive. It has its own dedicated coverage (rail-hint.test.mjs).
_DISMISS_RAIL_HINT = (
    "try { localStorage.setItem('osprey-rail-hint-dismissed-v1', '1') } catch (e) {}"
)


def _open_page(browser, base_url: str) -> Page:
    """Open a new browser page and wait for the rail + dock grid to render.

    panel-manager.js renders the rail asynchronously (it fetches /api/panels then
    each panel's config endpoint) and dock-workspace.js builds the dockview grid;
    this helper blocks until both are present so individual tests can assume a
    stable starting DOM.
    """
    page = browser.new_page()
    page.add_init_script(_DISMISS_RAIL_HINT)
    page.goto(base_url, wait_until="domcontentloaded")
    # Artifacts is always enabled and the DEFAULT_PANEL_FALLBACK, so its rail
    # button appears quickly after the async init path completes. Iframes also
    # carry data-panel-id, so target the button element specifically.
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    # The dockview grid is up once at least one group is on screen.
    expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
    return page


# ---------------------------------------------------------------------------
# Dock DOM helpers
# ---------------------------------------------------------------------------
#
# Every tile renders the same header bar (dock-tab.js): identity on the left,
# actions right-anchored (the bar itself is the drag handle — no grip badge).
# A service tile's identity is its visible .tile-tab-title, mirrored
# into the bar's aria-label and kept current through onDidTitleChange — the
# aria-label is the stable handle these tests address tabs by.
# The terminal tab is the one exception: it renders no aria-label — it adopts
# the live .terminal-header, which names it — so it is identified by that
# adoption and given the synthetic "SESSION" label below, the stable handle for
# it in these tests.

# Extract every dockview group with its tab labels and pixel rectangle. Used by
# the geometry assertions (split / restack / persistence ordering).
_GROUPS_JS = r"""() => [...document.querySelectorAll('.dv-groupview')].map(g => {
  // The terminal tile's tab is identified by its adopted .terminal-header
  // (it carries no accessible name of its own); its label is synthesized here
  // as "SESSION" — the same handle dock-workspace.js's TERMINAL_TITLE stamps on
  // the underlying dockview panel title.
  const tabs = [...g.querySelectorAll('.dv-tab')].map(t =>
    t.querySelector('.terminal-header')
      ? 'SESSION'
      : (t.querySelector('.tile-tab')?.getAttribute('aria-label') ?? ''));
  const r = g.getBoundingClientRect();
  return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), tabs };
})"""

# Align an overlay iframe against its dock group's content rectangle — the core
# adapter contract (the iframe follows the placeholder group). Returns null when
# either the placeholder or the iframe is absent.
_ALIGN_JS = r"""async (sid) => {
  const m = await import('/static/js/dock-workspace.js');
  const api = m.getDockApi();
  const panel = api && api.getPanel('iframe:' + sid);
  const iframe = document.querySelector(`.dock-iframe-overlay iframe[data-panel-id="${sid}"]`);
  if (!api || !panel || !iframe) return null;
  const content = panel.group && panel.group.element
    && panel.group.element.querySelector('.dv-content-container');
  if (!content) return null;
  const c = content.getBoundingClientRect();
  const i = iframe.getBoundingClientRect();
  return {
    contentLeft: Math.round(c.left), contentTop: Math.round(c.top),
    contentW: Math.round(c.width), contentH: Math.round(c.height),
    iframeLeft: Math.round(i.left), iframeTop: Math.round(i.top),
    iframeW: Math.round(i.width), iframeH: Math.round(i.height),
    display: iframe.style.display,
    active: panel.group && panel.group.activePanel === panel,
  };
}"""


def _dock_groups(page: Page) -> list[dict]:
    return page.evaluate(_GROUPS_JS)


def _service_tab(page: Page, label: str):
    """A dockview tile-strip whose accessible name equals ``label`` exactly.

    The strip has no visible text, so the name is the handle. Matched as a raw
    attribute (not ``get_by_label``, which only resolves labelable form
    controls, never an aria-label on a plain div) and exactly, so one service
    label like "WORKSPACE" cannot substring-match a longer one.
    """
    strip = page.locator(f'.tile-tab[aria-label="{label}"]')
    return page.locator(".dv-tab").filter(has=strip)


def _focus_service_tab(page: Page, label: str) -> None:
    """Focus a service tile by clicking its tab TITLE — never the whole tab.

    A panel may contribute controls into its own tile header (artifacts renders
    a search box and filter chips through tile-header-contrib.js), which widens
    the tab far past its title. Playwright clicks an element's CENTRE, so a
    click on the tab as a whole lands inside a contributed control — for
    artifacts, squarely in the search input — which is not a focus gesture and
    leaves the tile untouched. The title is the region that always means
    "focus this tile", whatever a panel contributes beside it.
    """
    _service_tab(page, label).locator(".tile-tab-title").click()


def _terminal_tab(page: Page):
    """The terminal tile's header bar — the tab that adopted .terminal-header."""
    return page.locator(".dv-tab").filter(has=page.locator(".terminal-header"))


_STRIP_HEIGHT_JS = r"""(terminal) => {
  const strips = [...document.querySelectorAll('.dv-tabs-and-actions-container')];
  const match = strips.find(s => !!s.querySelector('.tile-tab-terminal') === terminal);
  return match ? Math.round(match.getBoundingClientRect().height) : -1;
}"""


def _strip_height(page: Page, *, terminal: bool) -> int:
    """Rendered height of a tile's header strip, in px.

    Every tile carries the same fixed-height bar in expert mode
    (dockview-overrides.css); simple mode collapses the service bars to zero.
    Returns -1 when no such tile is docked.
    """
    return page.evaluate(_STRIP_HEIGHT_JS, terminal)


def _overlay_iframe(page: Page, panel_id: str):
    return page.locator(f'.dock-iframe-overlay iframe[data-panel-id="{panel_id}"]')


def _focus_service_panel(page: Page, panel_id: str, label: str) -> None:
    """Click a rail entry to focus (and thereby dock) a service panel."""
    page.locator(f'button.panel-rail-button[data-panel-id="{panel_id}"]').click()
    expect(_service_tab(page, label)).to_have_count(1, timeout=5_000)


def _track_panel_posts(page: Page) -> list[str]:
    """Collect the path tail of every POST to a /api/panel* endpoint, in order."""
    posts: list[str] = []

    def _record(req):
        if req.method == "POST" and "/api/panel" in req.url:
            posts.append(req.url.rsplit("/api/", 1)[-1])

    page.on("request", _record)
    return posts


def _command_posts(posts: list[str]) -> list[str]:
    """The COMMAND POSTs among ``posts``, with occupancy reports filtered out.

    ``/api/panel-layout`` is a report, not a command: the dock-sync occupancy
    reporter POSTs it whenever tile occupancy changes — including on a purely
    LOCAL tile close, which is the feature working rather than a leak. The
    invariant these tests guard is what a gesture *commands* (focus/visibility,
    the calls that move every client's workspace), so reports are filtered out
    before the assertion. A report is safe here precisely because the server
    stores it last-writer-wins and broadcasts nothing.
    """
    return [p for p in posts if p != "panel-layout"]


# ---------------------------------------------------------------------------
# Context-menu helpers
# ---------------------------------------------------------------------------
#
# The rail's ⊞ and ↗ corner glyphs are gone: "open in a new tile" and "open in
# a new window" live only in the right-click context menu
# (panel-context-menu.js), a body-level popover carrying one .rail-context-item
# per verb. Every gesture that used to click a corner goes through the menu
# now, so these helpers are the one place that knows the popover's DOM.

_RAIL_MENU = ".rail-context-menu"
_RAIL_MENU_ITEM = ".rail-context-item"

#: Row labels of the open context menu, in render order. Read from the label
#: span rather than the row so the trailing glyph never joins the text.
_MENU_LABELS_JS = (
    "() => [...document.querySelectorAll('.rail-context-menu .rail-context-label')]"
    ".map(e => e.textContent)"
)


def _rail_entry(page: Page, panel_id: str):
    return page.locator(f'button.panel-rail-button[data-panel-id="{panel_id}"]')


def _context_menu(page: Page):
    return page.locator(_RAIL_MENU)


def _menu_labels(page: Page) -> list[str]:
    return page.evaluate(_MENU_LABELS_JS)


def _menu_row(page: Page, label: str):
    """The menu row whose label contains ``label``."""
    return page.locator(_RAIL_MENU_ITEM).filter(has_text=label)


def _open_rail_menu(page: Page, panel_id: str):
    """Right-click a rail entry and wait for its context menu to render.

    The entry is the click target rather than one of its corners: a disabled
    entry is ``pointer-events: none``, so Playwright's actionability check also
    waits out the boot window in which the panel has not yet reported healthy —
    the same implicit wait the hover-corner gesture used to provide.
    """
    _rail_entry(page, panel_id).click(button="right")
    menu = _context_menu(page)
    expect(menu).to_have_count(1, timeout=5_000)
    return menu


def _open_second_tile(page: Page, base_url: str, panel_id: str, label: str) -> None:
    """Open ``panel_id`` as a SECOND tile beside the current one from its menu.

    A rail click would REPLACE the focused tile's panel (one panel per tile), so
    tests that need two service tiles side by side use the entry's context-menu
    row "Open in a new tile" — the open-beside verb, whose hover ⊞ corner is
    gone. ``base_url`` is unused (kept so call sites read uniformly with the old
    add-menu dance).
    """
    del base_url  # the menu path is purely client-side
    _open_rail_menu(page, panel_id)
    _menu_row(page, "Open in a new tile").click()
    expect(_service_tab(page, label)).to_have_count(1, timeout=5_000)


def _drag_with_dock_shield(page: Page, source, target, **kwargs) -> None:
    """``source.drag_to(target)`` with the app's real-drag state pre-raised.

    During a genuine pointer drag, dockview's onWillDragPanel raises the drag
    state (dock-workspace.js + dock-iframe.js): the ``.dock-dragging`` class on
    .main-container and the iframe pointer shield, so overlay iframes can't
    swallow the drag events over a covered group. Playwright's synthetic
    drag_to hit-tests the drop point WITHOUT any dragstart having raised that
    state, so a drop over an overlay iframe is judged "intercepted" and times
    out. Pre-raise both halves for the gesture, then restore them.
    """
    shield = (
        "(v) => { document.querySelectorAll('.dock-iframe-overlay iframe')"
        ".forEach(i => { i.style.pointerEvents = v; });"
        " document.querySelector('.main-container').classList"
        ".toggle('dock-dragging', v === 'none'); }"
    )
    page.evaluate(shield, "none")
    try:
        source.drag_to(target, **kwargs)
    finally:
        page.evaluate(shield, "auto")


def _reset_dock_layout(page: Page) -> None:
    """Invoke the exported resetDockLayout() on the live dock module singleton.

    A dynamic import of the same module URL app.js already loaded returns the same
    cached instance, so this drives the real DockviewApi — the seam a "Reset
    layout" control binds to.
    """
    page.evaluate(
        "async () => { const m = await import('/static/js/dock-workspace.js'); m.resetDockLayout(); }"
    )


def _dock_locked(page: Page) -> bool:
    return page.evaluate(
        "async () => { const m = await import('/static/js/dock-workspace.js');"
        " const a = m.getDockApi(); return !!(a && a.locked); }"
    )


# ===========================================================================
# Group 1 — SSE / rail flows materialize as dockview panels
# ===========================================================================


def test_focus_docks_service_panel_with_overlay_geometry(tmp_path, chromium_browser):
    """A rail click REPLACES the focused tile's panel; the overlay tracks the tile.

    A rail click on data-viz takes over the tile artifacts occupied (one panel
    per tile — the rail is the tab system): data-viz's overlay iframe becomes
    visible and its rectangle aligns to the dock group's content area (the
    adapter's follow contract), while the evicted artifacts panel loses its dock
    tab entirely and its overlay is concealed — a purely LOCAL vacate: its rail
    entry stays in place at full brightness, one click from taking the tile back.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        # Artifacts is auto-activated at boot: its dock tab + overlay iframe exist.
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)

        # Act — focus data-viz from the rail; it docks and becomes the active tab.
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

        # Assert — the data-viz overlay iframe aligns to its dock group content.
        geo = page.evaluate(_ALIGN_JS, "data-viz")
        assert geo is not None, "data-viz placeholder/overlay iframe missing"
        assert geo["active"] is True, f"data-viz is not the active dock tab: {geo}"
        assert geo["display"] != "none", f"data-viz overlay iframe hidden: {geo}"
        assert abs(geo["iframeLeft"] - geo["contentLeft"]) <= 2, geo
        assert abs(geo["iframeTop"] - geo["contentTop"]) <= 2, geo
        assert abs(geo["iframeW"] - geo["contentW"]) <= 2, geo
        assert abs(geo["iframeH"] - geo["contentH"]) <= 2, geo

        # The evicted artifacts panel: overlay concealed, dock tab gone — but
        # its rail entry survives untouched (eviction is local occupancy, not a
        # membership change) — the tile holds exactly one panel again.
        expect(_overlay_iframe(page, "artifacts")).to_be_hidden(timeout=5_000)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(0, timeout=5_000)
        expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
            timeout=5_000
        )
        groups = _dock_groups(page)
        service_groups = [g for g in groups if "SESSION" not in g["tabs"]]
        assert all(len(g["tabs"]) == 1 for g in service_groups), groups

        page.close()


def test_hide_active_panel_removes_dock_tab_and_expands_terminal(tmp_path, chromium_browser):
    """CC-1: hiding the sole visible panel drops its dock tab; the terminal expands.

    With only artifacts enabled it is the sole visible+healthy tab (auto-docked).
    Hiding it removes its dock placeholder and conceals its overlay iframe; with
    no service group left, the terminal card is the dock's only group and takes
    the full workspace width (no vestigial empty pane).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)

        # Act — hide the only visible+healthy panel.
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "artifacts", "visible": False},
        )
        assert r.status_code == 200

        # Assert — the artifacts dock tab is gone and its overlay iframe hidden.
        expect(_service_tab(page, "WORKSPACE")).to_have_count(0, timeout=5_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_hidden(timeout=5_000)

        # Assert — the terminal is the sole remaining group and fills the dock.
        page.wait_for_function(
            "() => document.querySelectorAll('.dv-groupview').length === 1", timeout=5_000
        )
        groups = _dock_groups(page)
        assert groups[0]["tabs"] == ["SESSION"], groups
        dock_w = page.evaluate(
            "() => document.getElementById('dock-root').getBoundingClientRect().width"
        )
        assert abs(groups[0]["w"] - dock_w) <= 2, (groups, dock_w)

        page.close()


def test_visibility_hide_and_show_toggles_rail_entry(tmp_path, chromium_browser):
    """SSE panel_visibility removes/re-adds a rail entry without a reload.

    data-viz is never focused here (so it has no dock placeholder yet); the
    visibility echo drives only its rail MEMBERSHIP — hiding removes the entry
    from the rail entirely (it moves to the "+" catalog), showing rebuilds it —
    proving the SSE→rail path is intact under the docked shell. No entry is
    ever dimmed: membership is binary.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        data_viz_tab = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        expect(data_viz_tab).to_be_visible(timeout=10_000)

        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200
        expect(data_viz_tab).to_have_count(0, timeout=5_000)  # removed, not dimmed

        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": True},
        )
        assert r.status_code == 200
        expect(data_viz_tab).to_be_visible(timeout=5_000)
        expect(page.locator(".panel-rail-closed")).to_have_count(0)

        page.close()


def test_register_appends_rail_entry_without_docking(tmp_path, chromium_browser):
    """CF-2/CC-3: panel_register adds a rail entry but does NOT auto-dock it.

    CF-2: exactly one rail entry is appended (non-destructive; the artifacts
          entry keeps its active state).
    CC-3: the new panel is not auto-activated — no dock placeholder tab appears
          for it and its rail entry is not marked active (its URL may not be
          ready; the user docks it when they choose).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        allow_runtime=True,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(
            page.locator('button.panel-rail-button[data-panel-id="artifacts"].active')
        ).to_be_attached(timeout=10_000)
        initial_tab_count = page.locator(".panel-rail-button").count()

        with patch(
            "osprey.interfaces.web_terminal.routes.panels._validate_panel_url",
            new=AsyncMock(return_value=None),
        ):
            r = requests.post(
                f"{base_url}/api/panels/register",
                json={
                    "id": "monitor",
                    "label": "MONITOR",
                    "url": "http://grafana.internal:3000",
                    "path": "/",
                },
            )
        assert r.status_code == 200, r.text

        # CF-2 — exactly one new rail entry appended, artifacts still active.
        monitor_rail = page.locator('button.panel-rail-button[data-panel-id="monitor"]')
        expect(monitor_rail).to_be_visible(timeout=5_000)
        assert page.locator(".panel-rail-button").count() == initial_tab_count + 1
        expect(
            page.locator('button.panel-rail-button[data-panel-id="artifacts"].active')
        ).to_be_attached(timeout=2_000)

        # CC-3 — the registered panel is NOT docked and NOT active.
        expect(monitor_rail).not_to_have_class(re.compile(r"\bactive\b"))
        expect(_service_tab(page, "MONITOR")).to_have_count(0, timeout=2_000)

        page.close()


def test_reopen_after_empty_state_redocks_panel(tmp_path, chromium_browser):
    """Closing the only panel then reopening it re-docks a live iframe, not a blank.

    Close the sole artifacts panel → its dock tab is gone and the terminal is
    the only group; reopen it from the "+" menu → its dock tab and overlay
    iframe return beside the terminal.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        artifacts_rail = page.locator('button.panel-rail-button[data-panel-id="artifacts"]')
        expect(artifacts_rail).to_be_visible(timeout=10_000)
        # Wait for the panel to actually be DOCKED before closing it — the
        # premise of this test is that a panel is open. `_open_page` only waits
        # for the rail entry and the first dockview group, and the rail renders
        # before the initial dock completes. Closing in that window closes
        # nothing, the dock then lands, and the tab this test expects to
        # disappear is instead created after the fact — so the assertion below
        # sees 1 for its whole timeout and reads as a product regression. Every
        # other test in this file asserts the same precondition first.
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)

        # Close the only panel via its rail "×" → terminal is the sole group.
        #
        # The click removes NOTHING by itself. `panel-commands.setPanelVisibility`
        # is fire-and-forget: it POSTs /api/panel-visibility, returns immediately,
        # and swallows failures — the DOM is driven by the server's
        # panel_visibility SSE echo, so what these waits span is a full
        # click → POST → broadcast → client-handler round trip, not a local
        # re-render. A short fixed budget here fails under whole-suite load with
        # the tile count pinned at 1 for the entire window and no error to show
        # for it, which reads as a product regression.
        #
        # So: wait on the echo's OWN first effect. The handler removes the rail
        # entry and then drops the tile (panel-manager.js's panel_visibility
        # branch, removeEntry before hidePanel), so a vanished rail entry means
        # the echo arrived and was applied; the tile assertion after it is then
        # about the handler's work, not about the network. Both at the 10s budget
        # this file already uses for round trips that must survive suite load.
        artifacts_rail.hover()
        page.locator('button[data-panel-id="artifacts"] .panel-rail-close').click()
        expect(artifacts_rail).to_have_count(0, timeout=10_000)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(0, timeout=10_000)
        page.wait_for_function(
            "() => document.querySelectorAll('.dv-groupview').length === 1", timeout=5_000
        )

        # Reopen from the "+" menu.
        page.locator("#panel-add-btn").click()
        page.locator('.panel-add-item[data-panel-id="artifacts"]').click()

        # The dock tab + overlay iframe are back beside the terminal.
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=5_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        assert len(_dock_groups(page)) == 2, _dock_groups(page)

        page.close()


# ===========================================================================
# Group 2 — Human dock gestures bridge to server POSTs (the echo guard)
# ===========================================================================


def test_dock_tab_focus_posts_setfocus_once(tmp_path, chromium_browser):
    """A human dock-tab focus POSTs setPanelFocus exactly once — no echo loop.

    With the two service panels in side-by-side tiles (via the "+" menu — a rail
    click would replace, not add), clicking the inactive artifacts dock tab
    fires one panel-focus POST. The server's SSE echo is applied under the echo
    guard, so it does not POST focus back again.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        # Open data-viz as a second tile beside artifacts; data-viz ends active.
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        # Drain the docking focus POST (async fetch) before counting.
        page.wait_for_timeout(800)

        posts = _track_panel_posts(page)
        # Human focus: click the artifacts dock tab (currently inactive).
        _focus_service_tab(page, "WORKSPACE")
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        # Let any (wrongly-)looping echo settle before counting.
        page.wait_for_timeout(600)

        assert _command_posts(posts) == ["panel-focus"], f"expected one focus POST, got {posts}"

        page.close()


def test_dock_tab_close_is_local_vacate_no_posts(tmp_path, chromium_browser):
    """Re-clicking a surfaced panel's rail entry is a LOCAL vacate.

    Tile occupancy is per-client layout state under the launcher-rail model:
    retiring the data-viz tile clears the local active state, but the panel
    keeps its rail membership — its entry stays in the rail at full brightness
    and NO panel POST fires (neither visibility nor a bounced focus). The
    re-click is the toggle shortcut equivalent of the tile header's own "×";
    the entry's "×" corner is the separate, membership-removing action.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        page.wait_for_timeout(600)

        posts = _track_panel_posts(page)
        # Re-click the already-surfaced entry to retire its tile.
        page.locator('button.panel-rail-button[data-panel-id="data-viz"]').click()

        # Its dock tab retires; the rail entry survives untouched and unmarked
        # active; no server POST of any kind is fired by the close.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0, timeout=5_000)
        rail_entry = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        expect(rail_entry).to_be_attached(timeout=5_000)
        expect(rail_entry).not_to_have_class(re.compile(r"\bactive\b"), timeout=5_000)
        page.wait_for_timeout(600)
        assert _command_posts(posts) == [], f"a local tile close must not command, got {posts}"

        page.close()


def test_service_tile_close_button_is_local_vacate(tmp_path, chromium_browser):
    """The service tile header's own "×" closes JUST the tile — never membership.

    Every tile now carries a close control (dock-tab.js). Clicking a service
    tile's "×" retires the tile locally: the rail entry survives at full
    brightness with no POST of any kind, and a plain rail click brings the
    panel straight back.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        # The unified bar: visible title + close, no grip badge (popout stays
        # rail-only).
        tab = _service_tab(page, "DATA VIZ")
        expect(tab.locator(".tile-tab-grip")).to_have_count(0)
        expect(tab.locator(".tile-tab-title")).to_have_text("DATA VIZ")
        expect(tab.locator(".tile-tab-close")).to_have_count(1)
        expect(tab.locator(".tile-tab-popout")).to_have_count(0)
        page.wait_for_timeout(600)

        posts = _track_panel_posts(page)
        tab.locator(".tile-tab-close").click()

        # The tile is gone; the rail entry survives, inactive; zero POSTs.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0, timeout=5_000)
        rail_entry = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        expect(rail_entry).to_be_attached(timeout=5_000)
        expect(rail_entry).not_to_have_class(re.compile(r"\bactive\b"), timeout=5_000)
        page.wait_for_timeout(600)
        assert _command_posts(posts) == [], f"a tile close must not command, got {posts}"

        # One rail click reopens the panel — close is never a removal.
        rail_entry.click()
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

        page.close()


def test_open_beside_moves_docked_panel_instead_of_duplicating(tmp_path, chromium_browser):
    """The new-tile verb on an ALREADY-OPEN panel moves its tile beside the active one.

    The old dance (remove from rail, re-add via "+") is gone: dockPanelAt's
    move semantics relocate the existing placeholder, so the panel never
    duplicates and never loses its rail membership.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
        # Focus the artifacts tile so data-viz's own tile is NOT the active one.
        _focus_service_tab(page, "WORKSPACE")
        expect(
            page.locator('button.panel-rail-button[data-panel-id="artifacts"].active')
        ).to_have_count(1, timeout=5_000)
        groups_before = len(_dock_groups(page))

        # Act — "Open in a new tile" on the already-open data-viz entry.
        _open_rail_menu(page, "data-viz")
        _menu_row(page, "Open in a new tile").click()

        # Still exactly one data-viz tab, and the tile count is unchanged —
        # the tile MOVED (beside artifacts), it was not duplicated.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
        assert len(_dock_groups(page)) == groups_before, _dock_groups(page)
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

        page.close()


# Synthesize the HTML5 drag a rail entry emits, dropped in the bottom quadrant
# of the terminal group's content — Playwright's mouse-based drag_to cannot
# produce HTML5 dnd events, so the sequence is dispatched in-page with a real
# DataTransfer. dragstart runs panel-rail's real handler (payload + shields);
# dragover/drop land on dockview's group drop target, which fires
# onUnhandledDragOver (accepted by rail-drag.js) and then onDidDrop. The drop
# only registers once a dragover has been PROCESSED — dockview settles its
# overlay state asynchronously and needs more than one dragover to resolve a
# target — so each attempt polls for dockview's own drop-target overlay before
# dropping rather than sleeping a fixed budget: a drop dispatched before that
# overlay exists is silently ignored, a no-op the assertions can only report
# as a missing tab. How long the whole gesture takes is load-dependent, so it
# is also RETRIED until the dock actually happened. The landing signal is the
# tile bar dock-tab.js stamps for the placeholder
# (`.tile-tab[data-panel-id="iframe:<id>"]`), the same DOM the caller's tab
# assertions read; the caller still owns WHERE the tile landed, so a drop that
# settles in the wrong place fails there as a real regression rather than
# being retried away.
_RAIL_DRAG_DROP_JS = """async (panelId) => {
    const entry = document.querySelector(
        `button.panel-rail-button[data-panel-id="${panelId}"]`);
    const dt = new DataTransfer();
    entry.dispatchEvent(new DragEvent('dragstart',
        { bubbles: true, cancelable: true, dataTransfer: dt }));
    const term = [...document.querySelectorAll('.dv-groupview')].find(g =>
        g.querySelector('.terminal-header'));
    const content = term.querySelector('.dv-content-container');
    const r = content.getBoundingClientRect();
    // Just inside the grid's bottom edge band → dockview resolves
    // { target: 'edge', position: 'bottom' } — a split against the grid edge
    // (the group-targeted variant is covered by the ⊞ tests + unit suite;
    // points deeper inside the content resolve 'center', which is refused).
    const x = r.left + r.width / 2;
    const y = r.bottom - 5;
    const opts = { bubbles: true, cancelable: true, dataTransfer: dt,
                   clientX: x, clientY: y };
    const target = document.elementFromPoint(x, y) ?? content;
    // dockview paints .dv-drop-target-selection once it has resolved the drop
    // position; the '-bottom' modifier is the edge split this drag intends, so
    // gating each drop on it proves the geometry resolved the way the test
    // claims. 20 attempts with a 500ms overlay budget each — an unloaded
    // machine lands on the first attempt in ~150ms, a loaded one gets an order
    // of magnitude more slack than any single hardcoded wait could give it.
    const ATTEMPTS = 20;
    const OVERLAY_BUDGET_MS = 500;
    const SETTLE_MS = 80;
    const overlay = () => document.querySelector(
        '.dv-drop-target-selection.dv-drop-target-bottom');
    const landed = () => !!document.querySelector(
        `.dv-groupview .dv-tab .tile-tab[data-panel-id="iframe:${panelId}"]`);
    let tries = 0;
    while (tries < ATTEMPTS && !landed()) {
        tries += 1;
        // dockview clears its drop target on drop, so each attempt re-arms it
        // with a fresh dragenter, then holds the dragover until dockview has
        // painted the bottom-edge overlay before dispatching the drop.
        target.dispatchEvent(new DragEvent('dragenter', opts));
        const deadline = performance.now() + OVERLAY_BUDGET_MS;
        while (!overlay() && performance.now() < deadline) {
            target.dispatchEvent(new DragEvent('dragover', opts));
            await new Promise((res) => setTimeout(res, 50));
        }
        if (!overlay()) {
            continue;  // never resolved this attempt; re-arm and try again
        }
        target.dispatchEvent(new DragEvent('drop', opts));
        await new Promise((res) => setTimeout(res, SETTLE_MS));
    }
    entry.dispatchEvent(new DragEvent('dragend', { bubbles: true, dataTransfer: dt }));
    if (!landed()) {
        throw new Error(
            `RailDropNeverSettled: no tile appeared for 'iframe:${panelId}' after ` +
            `${tries} overlay-gated drop attempts — the rail drop was never ` +
            `accepted by dockview.`);
    }
    return tries;
}"""


def test_drag_rail_entry_to_tile_edge_opens_split(tmp_path, chromium_browser):
    """Dragging a rail entry onto a tile edge opens the panel as a new tile there.

    The precise half of the open-beside verb (rail-drag.js): the entry's HTML5
    drag carries the panel id; dropping it near the bottom edge of the terminal
    tile splits that tile and the panel fills the new half, then the normal
    reveal tail focuses it.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        groups_before = len(_dock_groups(page))

        page.evaluate(_RAIL_DRAG_DROP_JS, "data-viz")

        # A new tile exists holding data-viz, below the terminal group.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
        assert len(_dock_groups(page)) == groups_before + 1, _dock_groups(page)
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == ["SESSION"])
        dv = next(g for g in groups if g["tabs"] == ["DATA VIZ"])
        assert dv["y"] > term["y"], (term, dv)

        page.close()


def test_server_sse_focus_is_applied_without_posting_back(tmp_path, chromium_browser):
    """A server-originated panel_focus re-activates an already-docked panel silently.

    Posting /api/panel-focus directly (as an agent MCP call would) drives the SSE
    handler, which applies the focus through the echo guard. When the target is
    already docked (its placeholder exists), the applied focus is a pure setActive
    the guard suppresses — the client emits no POST of its own. (First-dock of a
    panel legitimately emits one benign idempotent focus POST as its placeholder
    is created; that quirk is isolated out here by pre-docking data-viz.)
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        # Pre-dock data-viz as a SECOND tile (a rail click would replace), then
        # make artifacts active again, so the later server focus only has to
        # re-activate an existing placeholder.
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
        _focus_service_tab(page, "WORKSPACE")
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        # Let the WORKSPACE-click's own focus POST fully drain before tracking —
        # fetch() dispatches on a later tick, so it can otherwise land after the
        # tracker attaches and be misread as a POST-back.
        page.wait_for_timeout(800)

        posts = _track_panel_posts(page)
        # Server-driven focus back to the already-docked data-viz. The source
        # tag is what makes the server broadcast at all — a source-less human
        # report is mirrored without a frame.
        r = requests.post(
            f"{base_url}/api/panel-focus", json={"panel": "data-viz", "source": "agent"}
        )
        assert r.status_code == 200

        # It is applied — data-viz's tile takes the active focus (artifacts keeps
        # its own tile and overlay: side-by-side tiles are both on screen)...
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        expect(
            page.locator('button.panel-rail-button[data-panel-id="data-viz"].active')
        ).to_have_count(1, timeout=5_000)
        # ...with zero POSTs echoed back out (the guard suppressed the setActive).
        page.wait_for_timeout(600)
        assert _command_posts(posts) == [], f"server-driven focus POSTed back: {posts}"

        page.close()


def test_agent_hide_of_active_panel_posts_no_focus(tmp_path, chromium_browser):
    """An SSE-driven hide of the ACTIVE panel emits ZERO focus POSTs from the client.

    Regression guard for the echo bug: dropping the active service panel's
    placeholder makes dockview auto-activate a neighboring tile's panel, and
    that programmatic active-panel change must be recognised as a server-applied
    echo (panel-manager wraps the hide in the echo guard). If it leaked a
    setPanelFocus for the neighbor, the client would overwrite the server's own
    active_panel. The deliberate non-POSTing PANELS-order fallback owns where
    focus lands — here that is artifacts.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        # Two side-by-side tiles with data-viz active; artifacts is the visible
        # healthy neighbor the fallback must hand focus to.
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

        # Drain the docking focus POST before tracking.
        page.wait_for_timeout(800)

        posts = _track_panel_posts(page)
        # Agent/server-driven hide of the active panel (an MCP hide, over SSE).
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200

        # The active panel retires and the server-order fallback (artifacts) takes over.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0, timeout=5_000)
        expect(
            page.locator('button.panel-rail-button[data-panel-id="artifacts"].active')
        ).to_have_count(1, timeout=5_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)

        # The client emitted NO focus POST for the auto-activated neighbor.
        page.wait_for_timeout(600)
        assert "panel-focus" not in posts, f"agent-hide leaked a focus POST: {posts}"

        page.close()


# ===========================================================================
# Group 3 — Drag rearrange (real Playwright drags, geometry asserts)
# ===========================================================================


def test_drag_moves_tile_to_new_edge_position(tmp_path, chromium_browser):
    """Dragging a tile's tab to another group's edge moves the tile there.

    The artifacts tile starts left of the terminal. Dragging its tab to the
    right edge of the terminal group moves the tile to the far right; its
    overlay iframe re-follows to the new rectangle. (Edge drops are the only
    accepted drop geometry — see the veto test below.)
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)

        # Precondition — artifacts left of the terminal (the default split).
        groups = _dock_groups(page)
        assert len(groups) == 2, groups
        left, right = sorted(groups, key=lambda g: g["x"])
        assert left["tabs"] == ["WORKSPACE"] and right["tabs"] == ["SESSION"], groups

        # Act — drag the artifacts tab to the right edge of the terminal group.
        term_group = page.locator(".dv-groupview").last
        box = term_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _service_tab(page, "WORKSPACE"),
            term_group,
            target_position={"x": box["width"] - 8, "y": box["height"] / 2},
        )

        # Assert — the artifacts tile now sits rightmost, alone in its group.
        page.wait_for_function(
            """() => {
                const gs = [...document.querySelectorAll('.dv-groupview')].map(g => ({
                    x: g.getBoundingClientRect().x,
                    tabs: [...g.querySelectorAll('.tile-tab')]
                        .map(t => t.getAttribute('aria-label')).filter(Boolean),
                }));
                const ws = gs.find(g => g.tabs.length === 1 && g.tabs[0] === 'WORKSPACE');
                return ws && Math.max(...gs.map(g => g.x)) === ws.x;
            }""",
            timeout=5_000,
        )

        # The overlay iframe re-followed to the new group content rectangle.
        geo = page.evaluate(_ALIGN_JS, "artifacts")
        assert geo is not None and geo["active"] is True, geo
        assert abs(geo["iframeLeft"] - geo["contentLeft"]) <= 2, geo
        assert abs(geo["iframeW"] - geo["contentW"]) <= 2, geo

        page.close()


def test_drag_onto_tab_bar_is_vetoed_no_stack(tmp_path, chromium_browser):
    """A drop that would stack two panels into one tile is vetoed.

    One panel per tile: dropping the data-viz tab onto the artifacts tile's tab
    bar (the old restack gesture) is rejected by the onWillShowOverlay veto —
    the arrangement is unchanged, every tile still holds exactly one panel.
    A missed synthetic drop would also leave the arrangement unchanged, so this
    can pass trivially on a contended runner — the veto wiring itself is a
    one-line dockview event handler, and the unit suites pin the rest.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")

        before = _dock_groups(page)
        assert all(len(g["tabs"]) == 1 for g in before), before

        # Act — try to restack: drag the data-viz tab onto the artifacts tab.
        _drag_with_dock_shield(
            page, _service_tab(page, "DATA VIZ"), _service_tab(page, "WORKSPACE")
        )
        page.wait_for_timeout(700)

        # Assert — vetoed: no group holds two tabs, arrangement unchanged.
        after = _dock_groups(page)
        assert all(len(g["tabs"]) == 1 for g in after), (
            f"a drop stacked two panels into one tile: {after}"
        )
        shared = [g for g in after if "DATA VIZ" in g["tabs"] and "WORKSPACE" in g["tabs"]]
        assert not shared, after

        page.close()


# ===========================================================================
# Group 4 — Layout persistence keyed by project_key
# ===========================================================================


def test_layout_persists_across_reload(tmp_path, chromium_browser):
    """An expert arrangement survives a reload (keyed by project_key).

    Drag the native terminal to the far-left; after a reload the terminal is
    still leftmost — the persisted layout was restored over the default.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        # Default order is workspace/services left, terminal on the right.
        groups = _dock_groups(page)
        term_before = next(g for g in groups if g["tabs"] == ["SESSION"])
        assert term_before["x"] == max(g["x"] for g in groups), groups

        # Act — drag the terminal to the far-left of the first group.
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        first_group = page.locator(".dv-groupview").first
        fb = first_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _terminal_tab(page),
            first_group,
            target_position={"x": 8, "y": fb["height"] / 2},
        )
        page.wait_for_function(
            """() => { const gs = [...document.querySelectorAll('.dv-groupview')];
                const term = gs.find(g => g.querySelector('.terminal-header'));
                return term && Math.min(...gs.map(g => g.getBoundingClientRect().x)) === term.getBoundingClientRect().x; }""",
            timeout=5_000,
        )
        # Let the debounced persist write flush (schedulePersist ~150ms).
        page.wait_for_timeout(500)

        # Reload; the arrangement must be restored.
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        groups = _dock_groups(page)
        term_after = next(g for g in groups if g["tabs"] == ["SESSION"])
        assert term_after["x"] == min(g["x"] for g in groups), (
            f"terminal did not stay leftmost after reload: {groups}"
        )

        page.close()


def test_distinct_project_key_isolates_layouts(tmp_path, chromium_browser):
    """Two project cwds keep distinct dock layouts on one browser origin.

    Rearrange under cwd A (terminal → left); switch app.state.project_cwd to B and
    reload — the page shows the DEFAULT (terminal right), because B has no stored
    layout. Switching back to A restores A's arrangement (terminal left).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    cwd_a = str(workspace / "project-a")
    cwd_b = str(workspace / "project-b")

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        project_cwd=cwd_a,
    ) as (base_url, app):
        page = _open_page(chromium_browser, base_url)

        # Arrange A: terminal to the far-left, persisted under key(A).
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        first_group = page.locator(".dv-groupview").first
        fb = first_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _terminal_tab(page),
            first_group,
            target_position={"x": 8, "y": fb["height"] / 2},
        )
        page.wait_for_function(
            """() => { const gs = [...document.querySelectorAll('.dv-groupview')];
                const term = gs.find(g => g.querySelector('.terminal-header'));
                return term && Math.min(...gs.map(g => g.getBoundingClientRect().x)) === term.getBoundingClientRect().x; }""",
            timeout=5_000,
        )
        page.wait_for_timeout(500)

        keys_a = page.evaluate(
            "() => Object.keys(localStorage).filter(k => k.startsWith('osprey-dock-layout-'))"
        )
        assert len(keys_a) == 1, keys_a

        # Switch to project B and reload — a different key, no stored layout.
        app.state.project_cwd = cwd_b
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        groups = _dock_groups(page)
        term_b = next(g for g in groups if g["tabs"] == ["SESSION"])
        assert term_b["x"] == max(g["x"] for g in groups), (
            f"project B should show the default (terminal right), got {groups}"
        )
        keys_ab = page.evaluate(
            "() => Object.keys(localStorage).filter(k => k.startsWith('osprey-dock-layout-'))"
        )
        assert len(keys_ab) == 2, f"expected two distinct project keys, got {keys_ab}"

        # Switch back to A — its arrangement (terminal left) returns.
        app.state.project_cwd = cwd_a
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        groups = _dock_groups(page)
        term_a = next(g for g in groups if g["tabs"] == ["SESSION"])
        assert term_a["x"] == min(g["x"] for g in groups), (
            f"project A's stored layout (terminal left) was not isolated: {groups}"
        )

        page.close()


def test_reset_restores_default_layout(tmp_path, chromium_browser):
    """resetDockLayout() clears a custom arrangement back to the default.

    Rearrange (terminal → left), then reset: the grid returns to the default
    split (service tile left, terminal right) and the reset survives a
    reload (the stored value is the default, not the discarded custom one).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        first_group = page.locator(".dv-groupview").first
        fb = first_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _terminal_tab(page),
            first_group,
            target_position={"x": 8, "y": fb["height"] / 2},
        )
        page.wait_for_timeout(600)

        # Act — reset.
        _reset_dock_layout(page)
        page.wait_for_function(
            "() => document.querySelectorAll('.dv-groupview').length === 2", timeout=5_000
        )

        # Assert — default: the service tab-stack on the left, terminal right.
        groups = _dock_groups(page)
        assert len(groups) == 2, groups
        left, right = sorted(groups, key=lambda g: g["x"])
        assert left["tabs"] == ["WORKSPACE"], groups
        assert right["tabs"] == ["SESSION"], groups

        # And the reset persists across a reload (custom arrangement is gone).
        page.wait_for_timeout(500)
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == ["SESSION"])
        assert term["x"] == max(g["x"] for g in groups), (
            f"reset did not stick — terminal not back on the right: {groups}"
        )

        page.close()


def test_corrupt_stored_layout_falls_back_to_default(tmp_path, chromium_browser):
    """A corrupt stored layout is ignored; the default arrangement loads cleanly.

    Persist a custom arrangement, overwrite the stored value with unparseable
    JSON, then reload: reconcile rejects it and the default layout loads (terminal
    back on the right) with a working dock grid.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        # Establish a stored key by making + persisting a custom arrangement.
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        first_group = page.locator(".dv-groupview").first
        fb = first_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _terminal_tab(page),
            first_group,
            target_position={"x": 8, "y": fb["height"] / 2},
        )
        page.wait_for_timeout(600)
        key = page.evaluate(
            "() => Object.keys(localStorage).find(k => k.startsWith('osprey-dock-layout-'))"
        )
        assert key, "no stored layout key was written"

        # Corrupt it and reload.
        page.evaluate("(k) => localStorage.setItem(k, '{ not valid json')", key)
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        # Default arrangement loaded (terminal on the right), grid is functional.
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == ["SESSION"])
        assert term["x"] == max(g["x"] for g in groups), (
            f"corrupt layout did not fall back to default: {groups}"
        )
        assert _dock_locked(page) is False

        page.close()


# ===========================================================================
# Group 5 — Locked simple mode and mode↔dock transitions
# ===========================================================================


def test_simple_mode_locks_layout_and_hides_close_controls(tmp_path, chromium_browser):
    """Simple mode is a locked layout: no per-tab close, and drag is a no-op.

    The dock is api.locked with drag disabled and the per-tab close control
    (.tile-tab-close) hidden by the simple-mode CSS; a service tile's strip
    collapses to nothing at all there, since a bar that can neither drag nor
    close is dead chrome. A Playwright drag of a tab leaves the arrangement
    unchanged.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    # Non-empty workspace: an empty one makes a Simple hub boot chat-only
    # (workspace suppressed) — this test's subject is the LOCKED simple
    # layout, which needs the workspace docked.
    (workspace / "seed_artifact.txt").write_text("seed\n")

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
        ui_mode="simple",
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")
        page.wait_for_timeout(1_000)

        # Locked, and no per-tab close control is visible anywhere — neither
        # on the single service tile the locked simple layout docks (per
        # applySimpleLayout) nor on the terminal beside it.
        assert _dock_locked(page) is True
        expect(page.locator(".dv-tab .tile-tab-close:visible")).to_have_count(0)
        # The service tile's whole strip collapses; the terminal's stays, since
        # its bar is content (session id, "+ New") rather than a drag handle.
        assert _strip_height(page, terminal=False) == 0
        assert _strip_height(page, terminal=True) > 0

        # The chat/terminal card keeps the right-hand column in simple mode
        # (the single service tile on the left).
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == ["SESSION"])
        assert term["x"] == max(g["x"] for g in groups), groups

        # A drag is a no-op — the arrangement is byte-identical before/after.
        # Dragged BY THE TERMINAL's bar: the service tile has no drag surface
        # left to grab in simple mode (its strip is the 0px asserted above), so
        # the terminal's is the only tab an operator could still attempt this
        # with, and therefore the only one worth proving inert.
        before = _dock_groups(page)
        first_group = page.locator(".dv-groupview").first
        fb = first_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _terminal_tab(page),
            first_group,
            target_position={"x": 8, "y": fb["height"] / 2},
        )
        page.wait_for_timeout(700)
        assert _dock_groups(page) == before, "a drag rearranged the locked simple layout"

        page.close()


def test_mode_flip_restores_expert_layout_and_folds_in_simple_registration(
    tmp_path, chromium_browser
):
    """expert→simple→expert restores expert; a register during simple folds in clean.

    In simple mode a panel registered over SSE appears on the rail. Flipping back
    to expert unlocks the dock and restores the expert arrangement with no dead or
    duplicate tabs; the newly-registered panel then docks as a single clean tab
    when focused.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    # Non-empty workspace — see test_simple_mode_locks_layout_and_hides_close_controls.
    (workspace / "seed_artifact.txt").write_text("seed\n")

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
        allow_runtime=True,
        ui_mode="simple",
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")
        page.wait_for_timeout(1_000)
        assert _dock_locked(page) is True

        # Register a panel while simple mode is active → rail entry appears.
        with patch(
            "osprey.interfaces.web_terminal.routes.panels._validate_panel_url",
            new=AsyncMock(return_value=None),
        ):
            r = requests.post(
                f"{base_url}/api/panels/register",
                json={
                    "id": "monitor",
                    "label": "MONITOR",
                    "url": "http://grafana.internal:3000",
                    "path": "/",
                },
            )
        assert r.status_code == 200, r.text
        expect(page.locator('button.panel-rail-button[data-panel-id="monitor"]')).to_be_visible(
            timeout=5_000
        )

        # Flip to expert — dock unlocks and the expert layout is restored.
        # (The mode toggle lives inside the display-menu popover — open it first.)
        page.locator("#display-menu-btn").click()
        page.locator('#mode-toggle .mode-segment[data-mode="expert"]').click()
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "expert", timeout=5_000)
        page.wait_for_timeout(1_200)
        assert _dock_locked(page) is False

        # No dead tabs: every dock tab is a known panel; monitor is not docked yet.
        groups = _dock_groups(page)
        all_tabs = [t for g in groups for t in g["tabs"]]
        assert all(t in ("SESSION", "WORKSPACE") for t in all_tabs), (
            f"unexpected/dead dock tab after restore: {groups}"
        )
        assert all_tabs.count("MONITOR") == 0, groups

        # Focusing the panel registered during simple mode docks one clean tab.
        _focus_service_panel(page, "monitor", "MONITOR")
        groups = _dock_groups(page)
        monitor_tabs = [t for g in groups for t in g["tabs"] if t == "MONITOR"]
        assert len(monitor_tabs) == 1, f"expected exactly one MONITOR tab: {groups}"

        page.close()


def test_simple_mode_empty_workspace_boots_chat_only_until_agent_reveal(tmp_path, chromium_browser):
    """Simple UX first boot with an EMPTY workspace is chat-only; the rail add reveals.

    With no artifact in the agent workspace the hub docks only the chat/terminal
    card — no WORKSPACE tab, a single dock group. An agent ``add_panel_to_rail`` (the
    panel-visibility POST the MCP tool issues, tagged ``source: agent``) then
    brings the workspace up live on the shown panel.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()  # deliberately empty — the chat-only precondition

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        ui_mode="simple",
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")
        page.wait_for_timeout(1_000)

        # Chat-only boot: the locked layout holds only the terminal/chat card.
        assert _dock_locked(page) is True
        expect(_service_tab(page, "WORKSPACE")).to_have_count(0)
        groups = _dock_groups(page)
        assert [g["tabs"] for g in groups] == [["SESSION"]], groups

        # Agent reveal: add_panel_to_rail('artifacts') docks + activates the workspace.
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "artifacts", "visible": True, "source": "agent"},
        )
        assert r.status_code == 200
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=5_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)

        page.close()


# ===========================================================================
# Group 6 — The "+" add menu (rail-region behaviors, dock-aware)
# ===========================================================================


def test_add_menu_reveals_hidden_panel(tmp_path, chromium_browser):
    """Clicking a hidden panel in the "+" menu un-hides its rail entry and docks it.

    data-viz starts visible, is hidden via the API, then revealed from the menu —
    which docks it as a NEW tile beside the active group (the add menu is the
    grow-the-layout verb; a rail click would replace) and POSTs the reveal.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        data_viz_rail = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        expect(data_viz_rail).to_be_visible(timeout=10_000)

        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200
        expect(data_viz_rail).to_have_count(0, timeout=5_000)

        # Open the "+" menu; the removed panel is offered as an add target.
        page.locator("#panel-add-btn").click()
        menu_item = page.locator('.panel-add-item[data-panel-id="data-viz"]')
        expect(menu_item).to_be_visible(timeout=5_000)
        menu_item.click()

        # The rail entry returns and the panel docks (tab + overlay iframe).
        expect(data_viz_rail).to_be_visible(timeout=5_000)
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

        page.close()


def test_add_menu_opens_unclipped_beside_rail(tmp_path, chromium_browser):
    """The "+" popover renders fully on-screen, to the right of the rail.

    The rail column is a narrow scroll container; if it also clips horizontally,
    the popover (absolutely positioned outside it) is invisible, and focusing the
    first menu item drags the whole rail off-screen via focus auto-scroll.
    Playwright's visibility check does not see ancestor clipping, so this asserts
    geometry: menu box inside the viewport and past the rail edge, with no
    horizontal scroll displacement of the rail region.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(page.locator("#panel-add-btn")).to_be_visible(timeout=10_000)

        # Remove a panel from the rail so the menu has a focusable "Show panel"
        # item — the focus auto-scroll only fires when there is something to focus.
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200
        expect(page.locator('button.panel-rail-button[data-panel-id="data-viz"]')).to_have_count(
            0, timeout=5_000
        )

        page.locator("#panel-add-btn").click()
        expect(page.locator(".panel-add-menu.open")).to_be_visible(timeout=5_000)

        geo = page.evaluate(
            """() => {
                const menu = document.getElementById('panel-add-menu').getBoundingClientRect();
                const btn = document.getElementById('panel-add-btn').getBoundingClientRect();
                const region = document.querySelector('.panel-rail-region');
                return {
                    menuLeft: menu.left,
                    menuRight: menu.right,
                    btnLeft: btn.left,
                    railWidth: region.getBoundingClientRect().width,
                    railScrollLeft: region.scrollLeft,
                    viewportWidth: window.innerWidth,
                };
            }"""
        )
        assert geo["railScrollLeft"] == 0, f"opening the menu horizontally scrolled the rail: {geo}"
        assert geo["btnLeft"] >= 0, f'"+" button pushed off-screen: {geo}'
        assert geo["menuLeft"] >= geo["railWidth"], f"menu not clear of the rail column: {geo}"
        assert geo["menuRight"] <= geo["viewportWidth"], f"menu overflows viewport: {geo}"

        page.close()


def test_add_menu_url_row_hidden_when_disabled(tmp_path, chromium_browser):
    """With allow_runtime_panels off (default), the "+" menu shows no URL input."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        page.locator("#panel-add-btn").click()
        expect(page.locator(".panel-add-menu.open")).to_be_visible(timeout=5_000)
        expect(page.locator(".panel-add-input")).to_have_count(0)
        page.close()


def test_add_menu_url_row_shown_when_enabled(tmp_path, chromium_browser):
    """With allow_runtime_panels on, the "+" menu offers the URL input."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        allow_runtime=True,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        page.locator("#panel-add-btn").click()
        expect(page.locator('.panel-add-menu input[name="url"]')).to_be_visible(timeout=5_000)
        page.close()


def test_add_menu_registers_url_panel(tmp_path, chromium_browser):
    """Filling the URL form and clicking Add appends a new rail entry.

    SSRF validation is patched to pass (URL validation is unit-tested elsewhere);
    this exercises the human form → POST /api/panels/register → SSE → rail path.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        allow_runtime=True,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        with patch(
            "osprey.interfaces.web_terminal.routes.panels._validate_panel_url",
            new=AsyncMock(return_value=None),
        ):
            page.locator("#panel-add-btn").click()
            page.locator('.panel-add-menu input[name="url"]').fill("http://grafana.internal:3000")
            page.locator('.panel-add-menu input[name="label"]').fill("Monitor")
            page.locator(".panel-add-submit").click()

            # id derives from the label → "monitor"; the register SSE adds the rail entry.
            monitor_rail = page.locator('button.panel-rail-button[data-panel-id="monitor"]')
            expect(monitor_rail).to_be_visible(timeout=5_000)

        page.close()


def test_add_menu_shows_register_error(tmp_path, chromium_browser):
    """A register 4xx (here: built-in id collision) is shown inline in the menu."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        allow_runtime=True,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        page.locator("#panel-add-btn").click()
        page.locator('.panel-add-menu input[name="url"]').fill("http://x.internal:3000")
        # Label "Lattice" derives id "lattice", which collides with a built-in →
        # the server 422s and the menu must display the reason, not fail silently.
        page.locator('.panel-add-menu input[name="label"]').fill("Lattice")
        page.locator(".panel-add-submit").click()

        error = page.locator(".panel-add-url-error")
        expect(error).to_be_visible(timeout=5_000)
        expect(error).to_contain_text("built-in")

        page.close()


# ===========================================================================
# Group 7 — Health-driven activation guards (unchanged under the docked shell)
# ===========================================================================


def test_hidden_default_panel_falls_back_to_visible_panel(tmp_path, chromium_browser):
    """A hidden DEFAULT_PANEL hands the slot off to a visible healthy panel.

    `web.panels: {artifacts: {hidden: true}}` is supported, so the default panel
    can be healthy yet hidden. data-viz (visible, polling a live stub) must take
    the slot and dock; the hidden default must not surface itself.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _stub_backend() as backend_url:
        visible_panel = {
            "id": "data-viz",
            "label": "DATA VIZ",
            "url": backend_url,
            "healthEndpoint": "/health",
            "path": "/",
        }
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[visible_panel],
        ) as (base_url, app):
            # Hide the DEFAULT_PANEL itself — artifacts stays healthy, only hidden.
            app.state.visible_panels = ["data-viz"]

            page = chromium_browser.new_page()
            page.add_init_script(_DISMISS_RAIL_HINT)
            page.goto(base_url, wait_until="domcontentloaded")
            expect(page.locator('button[data-panel-id="data-viz"]')).to_be_attached(timeout=10_000)

            # The visible, healthy panel is the one docked and on screen.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="data-viz"].active')
            ).to_have_count(1, timeout=10_000)
            expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
            expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

            # The hidden default never surfaced itself: no rail entry (it is not
            # a member), no dock tab.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="artifacts"]')
            ).to_have_count(0)
            expect(_service_tab(page, "WORKSPACE")).to_have_count(0)

            page.close()


def test_hidden_panel_does_not_auto_activate(tmp_path, chromium_browser):
    """A hidden panel never surfaces itself, even as the only healthy one left.

    artifacts (DEFAULT_PANEL) reports no URL, so it never activates and keeps
    activeTabId null — the state the buggy fallback branch keys on. data-viz polls
    a live stub (a real unhealthy→healthy transition) but is hidden server-side.
    It must stay hidden: never docked, never active, no overlay iframe shown.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _stub_backend() as backend_url:
        hidden_panel = {
            "id": "data-viz",
            "label": "DATA VIZ",
            "url": backend_url,
            "healthEndpoint": "/health",
            "path": "/",
        }
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[hidden_panel],
            artifact_url=None,
        ) as (base_url, app):
            app.state.visible_panels = ["artifacts"]

            page = chromium_browser.new_page()
            page.add_init_script(_DISMISS_RAIL_HINT)
            page.goto(base_url, wait_until="domcontentloaded")
            expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)

            # Give the async init + health poll time to (wrongly) surface it —
            # data-viz's poll against the live stub goes healthy in this window,
            # which is exactly the transition the buggy fallback keyed on.
            page.wait_for_timeout(3_000)

            # A non-member has NO rail entry at all (no dimmed placeholder),
            # and its healthy transition must not have docked or activated it.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
            ).to_have_count(0)
            expect(_service_tab(page, "DATA VIZ")).to_have_count(0)
            expect(_overlay_iframe(page, "data-viz")).not_to_be_visible(timeout=5_000)

            page.close()


# ===========================================================================
# Group 8 — First-class terminal tile + launcher-rail semantics
# ===========================================================================


def test_terminal_rail_entry_close_and_reopen(tmp_path, chromium_browser):
    """The terminal is the rail's permanent anchor: tile closes, entry never changes.

    The rail's FIRST entry is the terminal ('SESSION') and it renders no "×"
    (the anchor is not removable — its tile closes via the dock tab or the
    Delete key on the focused entry). Closing the tile leaves the entry in
    place at full brightness; clicking it re-docks the same live terminal card.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        term_rail = page.locator('button.panel-rail-button[data-panel-id="terminal"]')
        expect(term_rail).to_be_visible(timeout=10_000)

        # The terminal entry leads the rail (the session tile is the anchor).
        first_id = page.locator(".panel-rail-button").first.get_attribute("data-panel-id")
        assert first_id == "terminal", first_id

        # The anchor offers no close affordance, even on hover.
        term_rail.hover()
        expect(term_rail.locator(".panel-rail-close")).to_be_hidden()

        # Close via the keyboard shortcut on the focused entry: the dock tile
        # goes away, the entry itself is untouched.
        term_rail.focus()
        page.keyboard.press("Delete")
        expect(_terminal_tab(page)).to_have_count(0, timeout=5_000)
        expect(term_rail).to_be_visible()

        # Reopen via the entry: the tile and live terminal card return.
        term_rail.click()
        expect(_terminal_tab(page)).to_have_count(1, timeout=5_000)
        expect(page.locator(".terminal-card")).to_be_visible(timeout=5_000)

        page.close()


def test_evicted_panel_entry_click_redocks_it(tmp_path, chromium_browser):
    """An evicted panel's rail entry survives and one click takes the tile back.

    A rail click on data-viz evicts artifacts from the tile (local vacate —
    membership untouched); the artifacts entry stays in the rail, and clicking
    it re-docks artifacts by taking the tile over from data-viz in turn. No
    visibility POST fires in either direction: occupancy is per-client state.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        artifacts_rail = page.locator('button.panel-rail-button[data-panel-id="artifacts"]')
        expect(artifacts_rail).to_be_visible(timeout=10_000)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)

        posts = _track_panel_posts(page)

        # Evict artifacts by activating data-viz (one panel per tile).
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        expect(_service_tab(page, "WORKSPACE")).to_have_count(0, timeout=5_000)
        expect(artifacts_rail).to_be_attached()

        # Take the tile back through the surviving entry.
        artifacts_rail.click()
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=5_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0, timeout=5_000)

        # Focus POSTs only (the docking flow may report focus more than once —
        # pre-existing, benign) — evictions never touch server visibility.
        page.wait_for_timeout(600)
        assert "panel-visibility" not in posts, f"eviction must stay local, got {posts}"
        assert set(_command_posts(posts)) == {"panel-focus"}, f"expected only focus POSTs: {posts}"

        page.close()


def test_rail_click_on_open_panel_jumps_without_moving_it(tmp_path, chromium_browser):
    """Clicking the rail entry of an OPEN panel focuses its tile — no move, no replace.

    With artifacts and data-viz in side-by-side tiles (data-viz active), clicking
    the artifacts rail entry activates the artifacts tile in place: both tiles
    survive, nothing is evicted, and both rail entries stay lit (open).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        before = _dock_groups(page)

        # Act — rail click on the OPEN (but unfocused) artifacts panel.
        page.locator('button.panel-rail-button[data-panel-id="artifacts"]').click()

        # Assert — artifacts is active and visible in ITS OWN tile; the layout
        # is untouched and data-viz keeps its tile and its lit rail entry.
        expect(
            page.locator('button.panel-rail-button[data-panel-id="artifacts"].active')
        ).to_have_count(1, timeout=5_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1)
        expect(page.locator('button.panel-rail-button[data-panel-id="data-viz"]')).to_be_attached()
        assert _dock_groups(page) == before, "a jump-to-open-panel changed the layout"

        page.close()


# JS: the tab strip of the group holding the terminal tile, with its current
# pixel height — the fixed-height assertions poll this. The terminal group is
# found by its adopted .terminal-header (its tab carries no accessible name).
_TERMINAL_STRIP_H_JS = """() => {
    const term = [...document.querySelectorAll('.dv-groupview')].find(g =>
        g.querySelector('.terminal-header'));
    const bar = term && term.querySelector('.dv-tabs-and-actions-container');
    return bar ? bar.getBoundingClientRect().height : null;
}"""


def test_tile_bar_fixed_height_no_hover_reflow(tmp_path, chromium_browser):
    """Tile header bars never change height — hover must not reflow content.

    Replaces the retired collapse/reveal behavior. Every tile carries the same
    fixed-height bar (identity + actions); hovering one (the old expand
    trigger) must leave the height and the content's top edge unmoved.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        expect(_terminal_tab(page)).to_have_count(1, timeout=10_000)

        # Exactly one terminal header row, and it lives in the tab, not the card.
        expect(page.locator(".terminal-header")).to_have_count(1)
        expect(page.locator(".dv-tab .terminal-header")).to_have_count(1)

        h_rest = page.evaluate(_TERMINAL_STRIP_H_JS)
        assert h_rest is not None and h_rest >= 30, f"bar height at rest: {h_rest}"

        content_top = page.evaluate(
            "() => document.querySelector('.terminal-body').getBoundingClientRect().top"
        )

        _terminal_tab(page).hover()
        page.wait_for_timeout(300)  # outlast any (now-deleted) transition
        assert page.evaluate(_TERMINAL_STRIP_H_JS) == h_rest, "bar height changed on hover"
        assert (
            page.evaluate(
                "() => document.querySelector('.terminal-body').getBoundingClientRect().top"
            )
            == content_top
        ), "tile content shifted on hover"

        # A service tile carries the unified bar: visible title + close, no
        # grip badge (popout stays a rail-entry affordance).
        ws = _service_tab(page, "WORKSPACE")
        expect(ws.locator(".tile-tab-grip")).to_have_count(0)
        expect(ws.locator(".tile-tab-title")).to_have_text("WORKSPACE")
        expect(ws.locator(".tile-tab-close")).to_have_count(1)
        expect(ws.locator(".tile-tab-popout")).to_have_count(0)

        # Same bar height as the terminal's — one header system, one size.
        svc_h = _strip_height(page, terminal=False)
        assert svc_h == h_rest, f"service bar {svc_h}px vs terminal {h_rest}px"

        # And it does not reflow on hover either.
        ws.hover()
        page.wait_for_timeout(300)
        assert _strip_height(page, terminal=False) == svc_h, "service bar grew on hover"

        page.close()


def test_overlay_iframe_follows_live_sash_drag(tmp_path, chromium_browser):
    """Mid-sash-drag (button still down) the overlay iframe tracks its group.

    Press on the sash between the service group and the terminal, move the
    pointer WITHOUT releasing, and assert the artifacts overlay iframe already
    matches its group's content rectangle — the live-resize contract (no
    freeze-until-mouseup snap).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=10_000)

        before = page.evaluate(_ALIGN_JS, "artifacts")
        assert before is not None, "artifacts placeholder/iframe missing"

        sash = page.evaluate(
            """() => {
                const s = [...document.querySelectorAll('.dv-sash')].find(el => {
                    const r = el.getBoundingClientRect();
                    return r.width > 0 && r.height > 50;
                });
                if (!s) return null;
                const r = s.getBoundingClientRect();
                return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
            }"""
        )
        assert sash is not None, "no vertical sash found"

        page.mouse.move(sash["x"], sash["y"])
        page.mouse.down()
        try:
            page.mouse.move(sash["x"] - 150, sash["y"], steps=8)
            # One rAF/RO tick to let the live follower re-apply geometry.
            page.wait_for_timeout(150)

            mid = page.evaluate(_ALIGN_JS, "artifacts")
            assert mid is not None, "alignment probe lost mid-drag"
            # The group actually resized…
            assert abs(mid["contentW"] - before["contentW"]) >= 100, (before, mid)
            # …and the overlay iframe is already aligned to it, mid-gesture.
            assert abs(mid["iframeLeft"] - mid["contentLeft"]) <= 2, mid
            assert abs(mid["iframeW"] - mid["contentW"]) <= 2, mid
        finally:
            page.mouse.up()

        page.close()


# ===========================================================================
# Group 8 — Tile header-bar contributions (embed contract v2)
# ===========================================================================
#
# An embedded panel contributes its toolbar controls INTO the tile bar via
# postMessage (design_system/js/header-contrib.js → tile-header-contrib.js);
# clicks round-trip back as osprey-header-action. Proven here with a custom
# panel whose (proxied, same-origin) page contributes a nav + a button on
# load and records incoming actions — the full pipeline the unit suites can
# only cover in halves.

_CONTRIB_PANEL_HTML = b"""<!doctype html>
<html><body><script>
  window.__actions = [];
  window.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'osprey-header-action') {
      window.__actions.push([e.data.id, e.data.value ?? null]);
    }
  });
  parent.postMessage({
    type: 'osprey-header-contribution',
    version: 1,
    items: [
      { kind: 'nav', id: 'view', priority: 1, items: [
        { id: 'browse', label: 'Browse', active: true },
        { id: 'create', label: 'New Entry' },
      ] },
      { kind: 'button', id: 'refresh', label: 'Refresh', priority: 2 },
    ],
  }, window.location.origin);
</script></body></html>"""


@contextmanager
def _contrib_panel_backend():
    """Serve a page that contributes to the tile bar and records actions."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_CONTRIB_PANEL_HTML)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", _free_port()), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_header_contribution_renders_and_round_trips(tmp_path, chromium_browser):
    """A panel's contribution renders in ITS tile bar and actions round-trip.

    The custom panel's page (reverse-proxied same-origin) posts a nav + button
    contribution on load; the bar must render them beside the tile title, a
    bar click must deliver osprey-header-action back into the iframe, and the
    artifacts tile (which contributes nothing) must keep an empty region.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _contrib_panel_backend() as stub_url:
        custom = {
            "id": "contrib-demo",
            "label": "CONTRIB",
            "url": stub_url,
            "healthEndpoint": None,
            "path": "/",
        }
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[custom],
            # The premise below is that artifacts contributes NOTHING, which
            # only holds while its iframe fails to load. The default fallback
            # URL is the artifacts server's standard port, so a real artifacts
            # server running on this host (a live web terminal, a demo stack)
            # would load for real and contribute its own nav items — turning
            # that assertion into a report about the machine. A free port keeps
            # the panel dead by construction.
            artifact_url=f"http://127.0.0.1:{_free_port()}",
        ) as (base_url, _app):
            page = _open_page(chromium_browser, base_url)
            _open_second_tile(page, base_url, "contrib-demo", "CONTRIB")

            # Contribution renders inside the CONTRIB tile's bar only.
            tab = _service_tab(page, "CONTRIB")
            expect(tab.locator(".contrib-nav-link")).to_have_count(2, timeout=10_000)
            expect(tab.locator(".contrib-nav-link.active")).to_have_text("Browse")
            expect(tab.locator(".contrib-btn")).to_have_text("Refresh")
            expect(tab.locator(".tile-tab-contrib.has-items")).to_have_count(1)

            ws = _service_tab(page, "WORKSPACE")
            expect(ws.locator(".tile-tab-contrib.has-items")).to_have_count(0)
            expect(ws.locator(".contrib-nav-link")).to_have_count(0)

            # Round-trip: bar clicks land in the panel as osprey-header-action.
            tab.locator(".contrib-nav-link", has_text="New Entry").click()
            tab.locator(".contrib-btn", has_text="Refresh").click()
            frame = page.frame_locator('.dock-iframe-overlay iframe[data-panel-id="contrib-demo"]')
            actions = None
            for _ in range(50):
                actions = frame.locator("body").evaluate("() => window.__actions")
                if actions and len(actions) >= 2:
                    break
                page.wait_for_timeout(100)
            assert actions == [["view", "create"], ["refresh", None]], actions

            page.close()


# A second contributing panel, this one exercising the `search` and `menu`
# kinds — and, unlike the one above, behaving like a REAL panel: it re-sends
# its whole contribution every time an action arrives. That is what makes this
# the only test that can prove the reconcile, because a rebuild-on-replace
# renderer would drop the caret on the first keystroke.
_SEARCH_MENU_PANEL_HTML = b"""<!doctype html>
<html><body><script>
  window.__actions = [];
  function contribution() {
    return {
      type: 'osprey-header-contribution',
      version: 1,
      items: [
        // Higher priority survives a narrow tile longer: the filter outranks
        // the overflow menu, whose entries are all reachable another way.
        { kind: 'search', id: 'filter', placeholder: 'Filter things\\u2026', priority: 3 },
        { kind: 'menu', id: 'more', priority: 1, items: [
          { id: 'all', label: 'All sessions', checked: false },
          { id: 'refresh', label: 'Refresh' },
        ] },
      ],
    };
  }
  window.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'osprey-header-action') {
      window.__actions.push([e.data.id, e.data.value ?? null]);
      // A real panel reacts and re-sends the WHOLE contribution.
      parent.postMessage(contribution(), window.location.origin);
    }
  });
  parent.postMessage(contribution(), window.location.origin);
</script></body></html>"""


@contextmanager
def _search_menu_panel_backend():
    """Serve a page contributing a search box and an overflow menu."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_SEARCH_MENU_PANEL_HTML)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", _free_port()), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _poll_actions(page, frame, minimum):
    """Wait for the panel iframe to have recorded at least `minimum` actions."""
    actions = None
    for _ in range(50):
        actions = frame.locator("body").evaluate("() => window.__actions")
        if actions and len(actions) >= minimum:
            return actions
        page.wait_for_timeout(100)
    return actions


def test_header_search_survives_re_contribution_and_menu_round_trips(tmp_path, chromium_browser):
    """Typing in a contributed search box keeps focus; the ⋯ menu round-trips.

    Two things only a real browser can settle. First, the tile bar IS the
    dockview drag surface, so clicking into a contributed input must focus it
    rather than start a tab drag (dock-tab.js's INTERACTIVE guard). Second,
    the panel re-sends its whole contribution on every action — so the caret
    surviving a full round-trip is what proves the render pass reconciles
    instead of rebuilding.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _search_menu_panel_backend() as stub_url:
        custom = {
            "id": "search-demo",
            "label": "SEARCHY",
            "url": stub_url,
            "healthEndpoint": None,
            "path": "/",
        }
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[custom],
        ) as (base_url, _app):
            page = _open_page(chromium_browser, base_url)
            _open_second_tile(page, base_url, "search-demo", "SEARCHY")

            tab = _service_tab(page, "SEARCHY")
            search = tab.locator(".contrib-search-input")
            expect(search).to_have_attribute("placeholder", "Filter things…", timeout=10_000)

            # Click into the field (not a drag) and type.
            search.click()
            page.keyboard.type("grid")

            frame = page.frame_locator('.dock-iframe-overlay iframe[data-panel-id="search-demo"]')
            actions = _poll_actions(page, frame, 1)
            assert actions and actions[0] == ["filter", "grid"], actions

            # The panel answered that action with a fresh whole contribution.
            # If the bar rebuilt its DOM, both of these would now fail.
            expect(search).to_have_value("grid")
            focused = page.evaluate("() => document.activeElement?.className ?? ''")
            assert "contrib-search-input" in focused, focused

            # Overflow menu: opens body-level, and an entry click round-trips.
            tab.locator(".contrib-menu-btn").click()
            popover = page.locator(".contrib-menu-popover")
            expect(popover).to_be_visible(timeout=5_000)
            expect(popover.locator(".contrib-menu-item")).to_have_count(2)

            popover.locator(".contrib-menu-item", has_text="Refresh").click()
            expect(page.locator(".contrib-menu-popover")).to_have_count(0)

            actions = _poll_actions(page, frame, 2)
            assert actions and actions[-1] == ["more", "refresh"], actions

            page.close()


# The drag ghost dockview builds for a tab is a deep clone of that tab, parented
# to document.body — OUTSIDE #dock-root (verified against the vendored 7.0.2
# bundle: the Tab's `createGhost` calls `_buildGhostElement()`, which clones the
# element and inlines the computed style ON THE ROOT ONLY, then `addGhostImage`
# appends the clone to document.body before `setDragImage`; the pointer-drag
# path parents its ghost to document.body too).
#
# So a tile's drag image is styled by whatever still matches once the bar is out
# of #dock-root. Two things keep that honest, and this test guards both:
#
#   * The tile-bar rules in dockview-overrides.css anchor on `.tile-tab` — a
#     class the clone carries — not on `.dock-root`. Anchoring on the dock root
#     leaves the ghost as unstyled DOM (the bar lays out `block`, controls fall
#     back to intrinsic sizes).
#   * svg-icons.js gives each icon an intrinsic width/height. A viewBox alone
#     gives an SVG an intrinsic RATIO but no intrinsic SIZE, so `width: auto`
#     resolves to the containing block's width and the 1:1 ratio squares it —
#     the search icon painted as a ~400px magnifying glass riding the cursor
#     (the close × collapsed to 0 instead, its container being zero-width).
#
# The two are independent: either one regressing alone still wrecks the drag
# image, so the assertions below cover them separately.
_GHOST_ICON_PROBE = """
() => {
  // The tab under test is the one whose panel contributed the search box —
  // NOT document's first .dv-tab, which is another tile entirely.
  const tab = document.querySelector('.contrib-search-input')?.closest('.dv-tab');
  if (!tab) return { error: 'no .dv-tab carrying a contributed search' };
  const liveIcon = tab.querySelector('.contrib-search-icon .bar-icon');
  if (!liveIcon) return { error: 'no search icon in the live bar' };
  const liveRect = liveIcon.getBoundingClientRect();
  const liveCloseRect = tab.querySelector('.tile-tab-actions button')
    ?.getBoundingClientRect();
  if (!liveCloseRect) return { error: 'no close control in the live bar' };
  // Drive dockview's REAL html5 drag source. addGhostImage removes the ghost in
  // a setTimeout(0), so it is still attached when this synchronous dispatch
  // returns — measure before yielding to the event loop.
  tab.dispatchEvent(new DragEvent('dragstart', {
    bubbles: true, cancelable: true, dataTransfer: new DataTransfer(),
  }));
  const ghost = document.querySelector('body > .dv-tab-ghost-drag');
  if (!ghost) return { error: 'dockview built no ghost element' };
  const box = (el) => {
    const r = el.getBoundingClientRect();
    return { w: Math.round(r.width), h: Math.round(r.height) };
  };
  const icons = [...ghost.querySelectorAll('.bar-icon')].map((el) => ({
    cls: el.getAttribute('class'),
    ...box(el),
    parent: el.parentElement.getAttribute('class') || el.parentElement.tagName,
    parentBox: box(el.parentElement),
  }));
  const ghostBar = ghost.querySelector('.tile-tab');
  const ghostClose = ghost.querySelector('.tile-tab-actions button');
  return {
    icons,
    ghost: box(ghost),
    parent: ghost.parentElement.tagName,
    live: { w: Math.round(liveRect.width), h: Math.round(liveRect.height) },
    liveClose: { w: Math.round(liveCloseRect.width), h: Math.round(liveCloseRect.height) },
    ghostClose: ghostClose ? box(ghostClose) : null,
    ghostBarDisplay: ghostBar ? getComputedStyle(ghostBar).display : null,
  };
}
"""


def test_tile_bar_drag_ghost_keeps_icons_bar_sized(tmp_path, chromium_browser):
    """A dragged tile's drag image must still look like the tile's header bar.

    dockview parents the ghost to document.body, so anything the bar's styling
    hangs off #dock-root is gone the moment the operator picks the tile up.
    What the ghost must keep: the bar's own layout, and icons at bar size
    rather than a magnifying glass scaled to the width of the whole tile.

    Only a real browser settles this: it needs layout, the replaced-element
    sizing fallback, and dockview's own ghost-construction path.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _search_menu_panel_backend() as stub_url:
        custom = {
            "id": "search-demo",
            "label": "SEARCHY",
            "url": stub_url,
            "healthEndpoint": None,
            "path": "/",
        }
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[custom],
        ) as (base_url, _app):
            page = _open_page(chromium_browser, base_url)
            _open_second_tile(page, base_url, "search-demo", "SEARCHY")

            # Wait for the contribution to land so the bar really holds a
            # magnifier — otherwise this could pass on an empty bar.
            tab = _service_tab(page, "SEARCHY")
            expect(tab.locator(".contrib-search-input")).to_have_attribute(
                "placeholder", "Filter things…", timeout=10_000
            )

            probe = page.evaluate(_GHOST_ICON_PROBE)
            assert "error" not in probe, probe
            assert probe["parent"] == "BODY", probe

            # In the bar itself the icon keeps the 12px `.tile-tab
            # .contrib-search-icon .bar-icon` gives it: CSS beats the intrinsic
            # width/height attributes, so sizing the SVG for the ghost must not
            # have resized anything the operator actually looks at.
            assert probe["live"] == {"w": 12, "h": 12}, probe["live"]

            icons = probe["icons"]
            # The bar always renders a close ×, and this panel adds a search
            # magnifier — a ghost with no icons would be a vacuous pass.
            assert len(icons) >= 2, icons
            assert any("search" in i["cls"] for i in icons), icons

            oversized = [i for i in icons if i["w"] > 24 or i["h"] > 24]
            assert not oversized, f"drag-ghost icons rendered oversized: {oversized}"

            # …and the ghost is the BAR, not a pile of unstyled DOM: the tile-bar
            # rules anchor on `.tile-tab`, a class the clone carries, so they
            # travel with it out of #dock-root. An unstyled clone would lay the
            # bar out as a block and leave the close control at its intrinsic
            # size instead of the bar's 24px control language.
            assert probe["ghostBarDisplay"] == "flex", probe["ghostBarDisplay"]
            assert probe["ghostClose"] == probe["liveClose"], probe

            page.close()


# ===========================================================================
# Group 9 — The rail / tile-header context menu (FR1-FR6)
# ===========================================================================
#
# The panel verbs that used to be hover-revealed corner glyphs now live in a
# right-click menu, and only there. What only a real browser can settle:
#
#   * FR1 - a right-click on a healthy rail entry really renders the popover,
#     with every verb in words, and its rows really drive the panel.
#   * FR3/FR5 - the surfaces that must DECLINE: a disabled entry (mouse on its
#     still-live "×" and keyboard alike) and a contributed header input, where
#     the browser's own text menu has to survive. No test can see that native
#     menu; what it can hold the app to is that the app's own popover stays
#     away and the event is left to the browser.
#   * FR4 - the dismissals that depend on real focus and real event targets: a
#     click landing inside a panel iframe (which never reaches this document —
#     only `window` blur does), and the scroll scoping that keeps the menu
#     standing while the terminal streams output.
#   * FR5 - the keyboard route, opened by the app rather than the platform.


def test_rail_context_menu_lists_every_panel_verb(tmp_path, chromium_browser):
    """Right-clicking a healthy rail entry opens the menu with all four verbs.

    The popover is body-level and viewport-positioned (it has to escape the
    rail's own overflow), names the panel in its accessible name, and carries
    the verbs in words with the rail's glyphs kept as trailing decoration. The
    membership verb is the only destructive one and is styled apart.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)

        menu = _open_rail_menu(page, "data-viz")

        assert _menu_labels(page) == [
            "Focus DATA VIZ",
            "Open in a new tile",
            "Open in a new window",
            "Remove from rail",
        ], _menu_labels(page)
        expect(menu).to_have_attribute("role", "menu")
        expect(menu).to_have_attribute("aria-label", "DATA VIZ actions")
        expect(menu.locator(".rail-context-divider")).to_have_count(1)
        expect(menu.locator(".rail-context-glyph")).to_have_count(3)
        expect(menu.locator(f"{_RAIL_MENU_ITEM}.danger")).to_have_text(
            re.compile(r"Remove from rail")
        )
        # data-viz's URL resolved at boot, so the popout row is live.
        expect(menu.locator(f'{_RAIL_MENU_ITEM}[aria-disabled="true"]')).to_have_count(0)

        placement = page.evaluate(
            "() => { const m = document.querySelector('.rail-context-menu');"
            " return { parent: m.parentElement.tagName,"
            " position: getComputedStyle(m).position }; }"
        )
        assert placement == {"parent": "BODY", "position": "fixed"}, placement

        page.close()


def test_rail_context_menu_keyboard_open_and_escape_restores_focus(tmp_path, chromium_browser):
    """Shift+F10 on a focused entry opens ONE menu; Escape hands focus back.

    macOS synthesises no `contextmenu` event from the keyboard, so the rail
    opens the menu itself — and cancels the keydown's default action, which on
    the platforms that DO synthesise one is what keeps a single keypress from
    stacking a second, native menu. Escape is a keyboard dismissal, so focus
    returns to whatever held it when the menu opened.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=10_000)
        # The entry has to be enabled before the keyboard route is allowed —
        # the same gate the mouse route passes through.
        expect(_rail_entry(page, "data-viz")).not_to_have_class(
            re.compile(r"\bdisabled\b"), timeout=10_000
        )

        _rail_entry(page, "data-viz").focus()
        page.keyboard.press("Shift+F10")

        # Exactly one popover, and the keyboard user lands on its first row.
        expect(_context_menu(page)).to_have_count(1, timeout=5_000)
        assert page.evaluate("() => document.activeElement?.textContent ?? ''").startswith(
            "Focus DATA VIZ"
        )

        page.keyboard.press("Escape")
        expect(_context_menu(page)).to_have_count(0, timeout=5_000)
        focused = page.evaluate("() => document.activeElement?.getAttribute('data-panel-id') ?? ''")
        assert focused == "data-viz", focused

        page.close()


def test_rail_context_menu_remove_from_rail_drops_membership(tmp_path, chromium_browser):
    """The menu's "Remove from rail" runs the same membership POST the "×" does.

    Like the corner it replaces, this is a SERVER change applied through the
    panel_visibility echo, so the wait spans a full click → POST → broadcast →
    handler round trip (see the rail "×" test for why the budget is 10s).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _focus_service_panel(page, "data-viz", "DATA VIZ")

        _open_rail_menu(page, "data-viz")
        _menu_row(page, "Remove from rail").click()

        # The menu closes before the action runs, and the panel loses both its
        # rail entry (membership) and its tile.
        expect(_context_menu(page)).to_have_count(0, timeout=5_000)
        expect(_rail_entry(page, "data-viz")).to_have_count(0, timeout=10_000)
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0, timeout=10_000)

        page.close()


def test_rail_context_menu_popout_row_inert_until_url_resolves(tmp_path, chromium_browser):
    """The new-window row renders inert while the panel has no standalone URL.

    A panel's URL and its health arrive together at boot (one with no URL never
    polls, so it never enables), which is why the entry is force-enabled here:
    the menu's enabled gate is a DOM-class question, and taking the class off is
    the only way to reach an entry that HAS a menu while its popout target is
    still unresolved — the state an operator meets when a companion server is
    slow to answer.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        artifact_url=None,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        entry = _rail_entry(page, "artifacts")
        expect(entry).to_have_class(re.compile(r"\bdisabled\b"), timeout=10_000)
        page.evaluate(
            "() => document.querySelector('button.panel-rail-button"
            "[data-panel-id=\"artifacts\"]').classList.remove('disabled')"
        )

        menu = _open_rail_menu(page, "artifacts")
        popout = _menu_row(page, "Open in a new window")
        expect(popout).to_have_attribute("aria-disabled", "true")
        # Every other row stays live — the gate is per-verb, not per-menu.
        expect(menu.locator(f'{_RAIL_MENU_ITEM}[aria-disabled="true"]')).to_have_count(1)

        pages_before = len(page.context.pages)
        # Forced, because Playwright reads `aria-disabled` on a menuitem as
        # "not enabled" and would otherwise refuse the press outright — which
        # is itself half the assertion. The press is still delivered, so the
        # row is held to doing nothing rather than merely looking inert.
        popout.click(force=True)
        page.wait_for_timeout(500)
        assert len(page.context.pages) == pages_before, "an inert row opened a window"
        # An inert row is not a dismissal either: the press landed inside the
        # menu, so the menu is still standing.
        expect(_context_menu(page)).to_have_count(1)

        page.close()


def test_disabled_rail_entry_has_no_context_menu(tmp_path, chromium_browser):
    """An offline panel's entry offers no menu — not by its "×", not by keyboard.

    The gate is in the handler rather than the rail's CSS on purpose: the
    disabled entry's "×" keeps `pointer-events: auto` so a dead panel can always
    be dismissed, and a keyboard menu request never consults CSS at all. Both
    routes are driven here; neither may reach a menu whose verbs act on a panel
    that has never answered. Declining leaves the event untouched, so what an
    operator gets on those presses is the browser's own menu.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        artifact_url=None,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        entry = _rail_entry(page, "artifacts")
        expect(entry).to_have_class(re.compile(r"\bdisabled\b"), timeout=10_000)

        # The still-live corner: a right-click on it lands on the entry.
        entry.locator(".panel-rail-close").click(button="right")
        page.wait_for_timeout(500)
        expect(_context_menu(page)).to_have_count(0)

        # The keyboard route bypasses `pointer-events` entirely.
        entry.focus()
        page.keyboard.press("Shift+F10")
        page.wait_for_timeout(500)
        expect(_context_menu(page)).to_have_count(0)

        page.close()


def test_simple_mode_drops_new_tile_row_and_terminal_menu(tmp_path, chromium_browser):
    """Simple mode: no new-tile row, and the terminal surfaces decline outright.

    The layout is locked to a single service tile there, so the open-beside verb
    has nowhere to go; and the xterm card is replaced by the operator console,
    so every terminal verb would act on a surface the operator cannot see — the
    same reason the palette drops those actions in simple mode.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    # Non-empty workspace — see test_simple_mode_locks_layout_and_hides_close_controls.
    (workspace / "seed_artifact.txt").write_text("seed\n")

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        ui_mode="simple",
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")

        _open_rail_menu(page, "artifacts")
        assert _menu_labels(page) == [
            "Focus WORKSPACE",
            "Open in a new window",
            "Remove from rail",
        ], _menu_labels(page)

        page.keyboard.press("Escape")
        expect(_context_menu(page)).to_have_count(0, timeout=5_000)

        # The SESSION entry declines: no rows means no menu at all, never an
        # empty popover.
        _rail_entry(page, "terminal").click(button="right")
        page.wait_for_timeout(500)
        expect(_context_menu(page)).to_have_count(0)

        page.close()


def test_terminal_context_menu_survives_streaming_output(tmp_path, chromium_browser):
    """The terminal's menu stays open while xterm scrolls its own viewport.

    Scroll dismissal is scoped to the menu's anchor: only a scroll of the
    document, or of a container that HOLDS the anchor, may close the menu. An
    unscoped listener would close it on xterm's viewport scroll — which fires on
    every streamed line, i.e. the terminal's normal state, making the menu
    unusable exactly where it is most useful.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(page.locator(".xterm-viewport")).to_have_count(1, timeout=10_000)

        _open_rail_menu(page, "terminal")
        assert _menu_labels(page) == [
            "Restart terminal",
            "New session",
            "Close terminal tile",
        ], _menu_labels(page)

        # The event xterm emits from its viewport on every streamed line. It
        # does not bubble, which is why the module listens in the capture phase
        # — and why it has to tell this target apart from the document.
        page.evaluate(
            "() => document.querySelector('.xterm-viewport').dispatchEvent(new Event('scroll'))"
        )
        page.wait_for_timeout(300)
        expect(_context_menu(page)).to_have_count(1)

        # A document scroll DOES move the anchor, so it dismisses — the contrast
        # that keeps the assertion above from passing on a missing listener.
        page.evaluate("() => document.dispatchEvent(new Event('scroll'))")
        expect(_context_menu(page)).to_have_count(0, timeout=5_000)

        page.close()


def test_tile_header_menu_yields_to_contributed_search_input(tmp_path, chromium_browser):
    """A right-click inside a contributed header input keeps the browser's menu.

    The tile bar owns a menu of the panel's verbs, but its interactive children
    do not: right-clicking a contributed search box has to leave copy/paste and
    the spell-check entries alone (dock-tab.js's INTERACTIVE guard). The title
    beside it is the contrast — the same press there does open the panel's menu.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _search_menu_panel_backend() as stub_url:
        custom = {
            "id": "search-demo",
            "label": "SEARCHY",
            "url": stub_url,
            "healthEndpoint": None,
            "path": "/",
        }
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[custom],
        ) as (base_url, _app):
            page = _open_page(chromium_browser, base_url)
            _open_second_tile(page, base_url, "search-demo", "SEARCHY")

            tab = _service_tab(page, "SEARCHY")
            search = tab.locator(".contrib-search-input")
            expect(search).to_have_attribute("placeholder", "Filter things…", timeout=10_000)

            search.click(button="right")
            page.wait_for_timeout(500)
            expect(_context_menu(page)).to_have_count(0)

            # The bar itself is a menu surface, with the same verbs the rail
            # entry offers.
            tab.locator(".tile-tab-title").click(button="right")
            menu = _context_menu(page)
            expect(menu).to_have_count(1, timeout=5_000)
            expect(menu).to_have_attribute("aria-label", "SEARCHY actions")
            assert _menu_labels(page) == [
                "Focus SEARCHY",
                "Open in a new tile",
                "Open in a new window",
                "Remove from rail",
            ], _menu_labels(page)

            page.close()


def test_context_menu_closes_when_a_click_lands_in_a_panel_iframe(tmp_path, chromium_browser):
    """A click inside a panel iframe closes the menu — without stealing focus back.

    A press inside an overlay iframe never reaches this document, so the outside
    pointerdown listener cannot see it; `window` blur is the only signal, and it
    arrives AFTER focus has entered the iframe. Restoring focus on that path
    would yank it straight back out of the panel the operator just clicked into,
    so this dismissal deliberately leaves focus alone.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _stub_backend() as stub_url:
        custom = {
            "id": "data-viz",
            "label": "DATA VIZ",
            "url": stub_url,
            "healthEndpoint": None,
            "path": "/",
        }
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[custom],
        ) as (base_url, _app):
            page = _open_page(chromium_browser, base_url)
            _focus_service_panel(page, "data-viz", "DATA VIZ")
            expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=10_000)

            _open_rail_menu(page, "data-viz")

            # A real click in the panel's own document (the stub answers the
            # proxied /panel/data-viz, so the iframe is same-origin and has a
            # body to click).
            page.frame_locator('.dock-iframe-overlay iframe[data-panel-id="data-viz"]').locator(
                "body"
            ).click()

            expect(_context_menu(page)).to_have_count(0, timeout=5_000)
            # Focus stayed where the operator put it: inside the iframe, not
            # back on the rail entry the menu was opened from.
            focused = page.evaluate("() => document.activeElement?.tagName ?? ''")
            assert focused == "IFRAME", focused

            page.close()


# ---------------------------------------------------------------------------
# Command palette (FR7-FR9)
# ---------------------------------------------------------------------------
#
# What only a real browser can settle about the palette:
#
#   * FR7 - focus. The input is focused inside the same rAF that reveals the
#     overlay, because focusing a node in a `visibility: hidden` subtree is a
#     spec'd no-op. happy-dom has no visibility model, so it cannot tell a
#     working focus call from the broken one; a real keystroke landing in the
#     input is the only proof.
#   * FR8 - the Panels rows. The registry unit tests prove the row set from
#     injected getters; these prove the LIVE getters produce it: ARIEL's
#     standalone URL really resolved through /api/ariel-server, and the active
#     panel really is included in the popout source and excluded from the
#     new-tile source.
#   * FR8 - the popout verb end to end: `window.open` opens a real page on the
#     proxied, same-origin /panel/<id> URL.
#   * FR9 - Recent. Real localStorage, and the section really renders first on
#     an empty query and survives a reload.
#
# ARIEL is the panel under test because it is the one the synonym table has
# domain vocabulary for ("logbook"/"elog"). It is enabled as a built-in panel
# with its companion server URL published post-startup - the catalog entry
# carries no healthEndpoint, so it is healthy the moment its config fetch
# resolves, and the stub backend behind it lets the proxied /panel/ariel URL
# answer for real when a popout navigates to it.

_PALETTE_OVERLAY = ".command-palette-overlay.visible"
_PALETTE_INPUT = ".command-palette-input"
_PALETTE_ITEM = ".command-palette-item"

#: The same platform split palette-boot.js applies to the hotkey: Cmd+K on
#: macOS, Ctrl+K everywhere else. Read from the browser rather than from the
#: host's ``sys.platform`` so the test follows the page's own decision.
_IS_MAC_JS = r"""() => {
  const nav = navigator;
  const platform = (nav.userAgentData && nav.userAgentData.platform) || nav.platform || '';
  return /Mac|iPhone|iPad|iPod/i.test(platform);
}"""

#: The palette list's group headings, in render order. "Recent" is a
#: presentation-level section, so it shows up here as a heading like any group.
_PALETTE_HEADINGS_JS = (
    "() => [...document.querySelectorAll('.command-palette-group-heading')].map(h => h.textContent)"
)

#: Row texts of the "Recent" section only - the nodes between the Recent
#: heading and the next heading. Returns None when no Recent section rendered,
#: which distinguishes "the section is absent" from "the section is empty".
_PALETTE_RECENT_ROWS_JS = r"""() => {
  const list = document.querySelector('.command-palette-list');
  if (!list) return null;
  const rows = [];
  let inRecent = false;
  for (const node of list.children) {
    if (node.classList.contains('command-palette-group-heading')) {
      if (inRecent) break;
      if (node.textContent === 'Recent') inRecent = true;
      continue;
    }
    if (inRecent) rows.push(node.textContent);
  }
  return inRecent ? rows : null;
}"""


@contextmanager
def _palette_live_server(workspace):
    """A live server whose ARIEL panel is a rail member with a resolved URL.

    ``_launch_panel_server`` is stubbed to publish only the gallery URL, so
    ARIEL's is set here - after startup, before any page load, which is when
    the browser first fetches /api/ariel-server. It points at the always-200
    stub so the proxied ``/panel/ariel`` a popout navigates to answers for real
    instead of erroring behind the proxy.

    Yields:
        (base_url, app) - as ``_live_server``.
    """
    with _stub_backend() as stub_url:
        with _live_server(
            workspace,
            enabled_panels={"artifacts", "ariel"},
        ) as (base_url, app):
            app.state.ariel_server_url = stub_url
            yield base_url, app


def _wait_for_ariel_ready(page: Page) -> None:
    """Block until ARIEL's standalone URL has resolved through /api/ariel-server.

    The palette builds its rows once per open (palette.js re-renders per open,
    not per keystroke), so an open that outruns ARIEL's config fetch is missing
    the ARIEL rows for as long as it stays open — no amount of typing brings
    them back. The rail entry sheds ``.disabled`` in the same settle that
    publishes the URL (initPanel → assumeHealthy), so waiting on the class is
    waiting on the rows' precondition. Every test that opens the palette and
    expects an ARIEL row must pass through here first.
    """
    expect(
        page.locator('button.panel-rail-button[data-panel-id="ariel"]:not(.disabled)')
    ).to_be_attached(timeout=10_000)


def _palette_input(page: Page):
    return page.locator(_PALETTE_INPUT)


def _palette_row(page: Page, label: str):
    """Palette rows whose text contains ``label`` (substring, case-insensitive).

    A row's label is split across highlight spans when the query matched it, so
    the text is read from the row rather than from a single text node.
    """
    return page.locator(_PALETTE_ITEM).filter(has_text=label)


def _open_palette(page: Page, *, hotkey: bool = False) -> None:
    """Open the command palette and wait for the revealed overlay.

    The default entry point is the header trigger; ``hotkey=True`` drives the
    Cmd/Ctrl+K path instead. Either way this returns only once the overlay
    carries ``.visible`` - the class the focus call shares a frame with, so a
    caller may type immediately afterwards without a further wait.

    The hotkey path first focuses a rail entry: off macOS the shortcut
    deliberately yields to readline's Ctrl+K while focus is inside the
    terminal, so the gesture has to start where the operator's would.
    """
    if hotkey:
        page.locator('button.panel-rail-button[data-panel-id="artifacts"]').focus()
        page.keyboard.press("Meta+k" if page.evaluate(_IS_MAC_JS) else "Control+k")
    else:
        page.locator("#command-palette-btn").click()
    expect(page.locator(_PALETTE_OVERLAY)).to_be_visible(timeout=5_000)


def _wait_for_panel_rows(page: Page) -> None:
    """Wait until the rail's ARIEL entry is usable before opening the palette.

    The ARIEL rows — and the popout row in particular — exist only once the
    panel-config fetch has resolved (``getPanelStandaloneUrl`` answers null
    until then). In isolation that takes well under a second; at the tail of a
    long single-process browser run it can exceed the palette's own 5 s budget.
    Waiting on the rail entry, which the same fetch enables, anchors the palette
    assertions to the state they actually depend on instead of to a timer. The
    rail signals availability with the ``disabled`` CSS class, not the HTML
    attribute (panel-rail.js ``setEntryEnabled``), so this waits on the class.
    """
    expect(
        page.locator('button.panel-rail-button[data-panel-id="ariel"]:not(.disabled)')
    ).to_be_attached(timeout=15_000)


def _palette_query(page: Page, query: str) -> None:
    """Put ``query`` in the search box and let the list re-render.

    Typed rather than pressed: whether the open FOCUSED the box is FR7's
    question, and exactly one test above owns it. Every other palette test is
    about what the list then contains, so they set the query outright instead of
    inheriting a focus dependency and failing for someone else's reason.
    """
    _palette_input(page).fill(query)
    expect(_palette_input(page)).to_have_value(query)


def _palette_reload(page: Page) -> None:
    """Reload the page and wait for the same steady state ``_open_page`` does.

    A reload tears down the rail and the dock grid and rebuilds both
    asynchronously, so a test that reaches for the palette straight after one
    would race the rebuild rather than the behaviour it means to assert.
    """
    page.reload(wait_until="domcontentloaded")
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)


def test_palette_hotkey_opens_with_input_focused_and_types_immediately(tmp_path, chromium_browser):
    """FR7: Cmd/Ctrl+K opens the palette focused - the next keystroke filters.

    The regression this guards is silent: ``inputEl.focus()`` ran while the
    overlay was still ``visibility: hidden``, which the spec makes a no-op, so
    the palette opened looking usable and swallowed everything the operator
    typed. Nothing here clicks the input - the characters are pressed straight
    at the document, so they can only reach the input if the open really moved
    focus there.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _palette_live_server(workspace) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        _wait_for_panel_rows(page)
        _open_palette(page, hotkey=True)

        # Real focus, not merely a rendered overlay. The diagnostic carries the
        # computed visibility because that is the whole story when this fails:
        # a `visible` class on the overlay is not the same as a computed
        # `visibility: visible` the browser will focus into.
        state = page.evaluate(
            """() => {
  const overlay = document.querySelector('.command-palette-overlay');
  const active = document.activeElement;
  return {
    focused: !!active && active.classList.contains('command-palette-input'),
    active: active ? active.tagName + '.' + active.className : null,
    overlayClass: overlay ? overlay.className : null,
    overlayVisibility: overlay ? getComputedStyle(overlay).visibility : null,
  };
}"""
        )
        assert state["focused"] is True, f"palette opened without the search input focused: {state}"

        # Act - type without touching the input first.
        page.keyboard.type("ariel")

        # Assert - the keystrokes landed AND re-filtered the list.
        expect(_palette_input(page)).to_have_value("ariel")
        expect(_palette_row(page, "Open ARIEL in a new window")).to_have_count(1, timeout=5_000)
        expect(_palette_row(page, "Open Settings")).to_have_count(0)

        page.close()


def test_palette_query_ariel_offers_popout_of_the_active_panel(tmp_path, chromium_browser):
    """FR8: with ARIEL active, "ariel" finds its popout row but no new-tile row.

    The two "Open ..." verbs draw on different sources, and the live getters
    are what this proves: ``getPopoutPanels`` includes the ACTIVE panel
    (popping out what you are looking at is the verb's commonest use, and it
    needed ARIEL's standalone URL to have resolved through /api/ariel-server),
    while ``getVisiblePanels`` - the new-tile source - excludes it, because
    opening a panel beside itself is a no-op. The second query proves the
    new-tile verb is present in this (expert) mode at all, so ARIEL's missing
    row is the active-panel exclusion rather than a whole verb going absent.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _palette_live_server(workspace) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _wait_for_ariel_ready(page)
        _focus_service_panel(page, "ariel", "ARIEL")

        _open_palette(page)
        _palette_query(page, "ariel")

        expect(_palette_row(page, "Open ARIEL in a new window")).to_have_count(1, timeout=5_000)
        # The query really narrowed the list, so the absences below mean the
        # rows do not exist rather than that everything is still on screen.
        expect(_palette_row(page, "Open Settings")).to_have_count(0)
        expect(_palette_row(page, "Open ARIEL in a new tile")).to_have_count(0)
        # Focus is the same story as the new tile: the active panel is not
        # offered a verb that would do nothing.
        expect(_palette_row(page, "Focus ARIEL")).to_have_count(0)

        # The new-tile verb itself is alive here - it just skips the active panel.
        _palette_query(page, "tile")
        expect(_palette_row(page, "Open WORKSPACE in a new tile")).to_have_count(1, timeout=5_000)
        expect(_palette_row(page, "Open ARIEL in a new tile")).to_have_count(0)

        page.close()


def test_palette_logbook_synonym_matches_ariel(tmp_path, chromium_browser):
    """FR8: "logbook" - the domain word, not the label - finds the ARIEL rows.

    ARIEL's rail label says nothing about logbooks, so an operator who types
    what the panel IS gets nothing unless the synonym tokens really reach the
    live rows' searchText.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _palette_live_server(workspace) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _wait_for_ariel_ready(page)

        _wait_for_panel_rows(page)
        _open_palette(page)
        _palette_query(page, "logbook")

        expect(_palette_row(page, "Open ARIEL in a new window")).to_have_count(1, timeout=5_000)
        # A word ARIEL's own rows are the only ones carrying: WORKSPACE, the
        # other rail member, is filtered out entirely.
        expect(_palette_row(page, "WORKSPACE")).to_have_count(0)

        page.close()


def test_palette_popout_row_opens_the_proxied_panel_url(tmp_path, chromium_browser):
    """FR8: running the popout row opens a real page on same-origin /panel/ariel.

    happy-dom can only prove ``window.open`` was called with some string. What
    matters to the operator is the page that opens: the proxied, root-relative
    standalone URL resolved against the terminal's own origin - never the
    panel's internal address, which the browser cannot reach.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _palette_live_server(workspace) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _wait_for_ariel_ready(page)

        _wait_for_panel_rows(page)
        _open_palette(page)
        _palette_query(page, "ariel window")
        row = _palette_row(page, "Open ARIEL in a new window")
        expect(row).to_have_count(1, timeout=5_000)

        with page.context.expect_page() as popup_info:
            row.click()
        popup = popup_info.value
        popup.wait_for_url(re.compile(r"/panel/ariel$"), timeout=10_000)

        assert popup.url == f"{base_url}/panel/ariel", popup.url

        popup.close()
        page.close()


def test_palette_recent_section_leads_the_empty_query_after_an_execution(
    tmp_path, chromium_browser
):
    """FR9: an executed item comes back first under "Recent", and survives reload.

    Recent is a presentation-level section rebuilt from localStorage keys that
    are re-resolved against the live registry on each render, so this drives
    the whole loop through a real browser store: execute, reopen on an empty
    query, reload, reopen again. "Move panel rail to left" is the item because
    it is offered in every mode and is a no-op at the default rail position, so
    the proof does not hinge on the workspace being left in some other state.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _palette_live_server(workspace) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        # No history yet: an empty query renders no Recent section at all.
        _open_palette(page)
        assert page.evaluate(_PALETTE_RECENT_ROWS_JS) is None
        page.keyboard.press("Escape")
        expect(page.locator(_PALETTE_OVERLAY)).to_have_count(0, timeout=5_000)

        # Act - run one item.
        _open_palette(page)
        _palette_query(page, "rail to left")
        row = _palette_row(page, "Move panel rail to left")
        expect(row).to_have_count(1, timeout=5_000)
        row.click()
        expect(page.locator(_PALETTE_OVERLAY)).to_have_count(0, timeout=5_000)

        # Assert - it leads the empty-query render, ahead of every group.
        _open_palette(page)
        headings = page.evaluate(_PALETTE_HEADINGS_JS)
        assert headings and headings[0] == "Recent", headings
        assert page.evaluate(_PALETTE_RECENT_ROWS_JS) == ["Move panel rail to left"]
        # The normal groups still render underneath - Recent never replaces them.
        assert "Panels" in headings, headings
        page.keyboard.press("Escape")
        expect(page.locator(_PALETTE_OVERLAY)).to_have_count(0, timeout=5_000)

        # Assert - the history lives in the browser store, not in module memory.
        _palette_reload(page)
        _open_palette(page)
        assert page.evaluate(_PALETTE_RECENT_ROWS_JS) == ["Move panel rail to left"]

        page.close()
