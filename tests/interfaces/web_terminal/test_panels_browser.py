"""Browser smoke tests: the docked (dockview) workspace and its server bridge.

The service panels do not live in a fixed left/right split — they are docked
in a dockview grid (``dock-workspace.js``) whose panel content follows an overlay
iframe layer (``dock-iframe.js``), with a state bridge (``dock-sync.js``) that
turns human dock gestures into the same server POSTs the agent's MCP calls make.
These tests drive that stack through a real browser and a live SSE stream — the
parts the FastAPI TestClient cannot reach.

An open — a rail click, the "+" add menu, an agent focus — replaces the panel
of the service tile used last, so nothing resizes; the replaced panel keeps
its rail entry and its cached iframe. The rail entry's "Open in a new tile" is
the one verb that opens a NEW tile beside the current one. A tile holds one
panel: the dock engine (``dock-drop.js``) decides every drop, an edge splits a
tile, and a tile's header or centre takes nothing.

What is proven here, grouped:

  * SSE / rail flows materialize as dockview panels: focusing a service panel
    docks a placeholder whose overlay iframe tracks the group geometry — in
    the focused service tile, in place of the panel it showed; hiding a panel
    removes its placeholder; register appends a rail entry WITHOUT auto-docking.
  * The echo guard (``dock-sync.js``): a human dock-tab focus POSTs setPanelFocus
    exactly once, a human tab-close POSTs visibility(false), and a server-driven
    SSE focus is applied WITHOUT POSTing back (no feedback loop).
  * Drag rearrange via real Playwright drags — a tile moves to a new edge
    position, and a tile's header or centre is no drop target — asserted
    with getBoundingClientRect geometry, and the overlay iframe re-following
    the dragged panel's new rectangle.
  * Layout persistence keyed by project_key: an arrangement survives a
    reload, two project cwds keep distinct layouts on one origin, reset restores
    the default, and a corrupt stored value falls back cleanly.
  * The "+" add menu: unclipped geometry beside the rail, URL-row gating, reveal a
    hidden panel into the active service tile, register from URL, and
    inline register errors.
  * Tile header-bar contributions (the embed contract): a panel's own nav, search
    and overflow items render in ITS bar only, actions round-trip into the iframe,
    and the bar's drag ghost keeps its icons bar-sized.
  * The right-click context menu (``panel-context-menu.js``), the only home left
    for the open-beside and pop-out verbs: the rows a rail entry and a tile header
    offer, the surfaces that DECLINE (an offline entry by mouse or keyboard, a
    contributed header input), and the dismissals
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
import json
import re
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from osprey.interfaces._serving import run_app_server

# Aliased to the historical private names: the shared helpers are the same
# two waits this module used to spell itself, and the alias keeps its fifty
# call sites reading as they did.
from tests.interfaces._browser import open_hub_page as _open_page
from tests.interfaces._browser import rail_entry as _rail_entry
from tests.interfaces._browser import wait_for_dock_settled, wait_for_page_ready
from tests.interfaces._hub import hub_app
from tests.interfaces._panel_launch import (
    UNSERVED,
    Unserved,
    publish_artifact_url,
    unserved_url,
)
from tests.interfaces.web_terminal._surface import arm_terminal_tab

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

# A third immediately-healthy panel, for tests that need two rail entries
# beside the artifacts panel, which has none (its artifacts are in the
# session tree).
_CUSTOM_OPS: dict = {
    **_CUSTOM_DATA_VIZ,
    "id": "ops",
    "label": "OPS",
    "url": "http://ops.internal:8080",
}


@contextmanager
def _stub_backend(hits: list[str] | None = None, *, down: threading.Event | None = None):
    """Serve 200 on every path, for a custom panel that must become healthy.

    Panels with a ``healthEndpoint`` are the only ones that reach the
    auto-activate branch in ``pollHealth``, and that branch needs a real
    unhealthy→healthy transition — so it needs a real backend to poll.

    Args:
        hits: When given, every request path the stub answers is appended to
            it, so a caller can prove a health poll really reached the stub.
        down: When given, the stub answers ``503`` on every path while the
            event is set, so a caller can take the backend dark mid-test.

    Yields:
        base URL of the stub, e.g. ``"http://127.0.0.1:54321"``.
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if hits is not None:
                hits.append(self.path)
            if down is not None and down.is_set():
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"down")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):  # silence per-request stderr logging
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
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
    # The shared unserved address — never the artifact slot at the default
    # base, which a real deployment on this host serves for real.
    artifact_url: str | None | Unserved = UNSERVED,
    artifact_config_delay: float = 0.0,
    project_cwd: str | None = None,
):
    """Launch a real web terminal server on a free port in a background thread.

    Companion backends (artifact server, ARIEL, etc.) are bypassed via patches
    so no external process dependencies are required.  The artifacts panel
    reports its URL as the artifact slot of the default block (the standard
    fallback).

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
    layout a page load resolves to.

    Yields:
        (base_url: str, app: FastAPI) — live server address and the FastAPI app
        object for post-startup state manipulation.
    """
    with hub_app(
        workspace_dir,
        enabled_panels=enabled_panels,
        custom_panels=custom_panels or [],
        launch=publish_artifact_url(artifact_url),
    ) as app:
        if artifact_config_delay:

            @app.middleware("http")
            async def _delay_artifact_config(request, call_next):
                if request.url.path == "/api/artifact-server":
                    await asyncio.sleep(artifact_config_delay)
                return await call_next(request)

        # run_app_server yields only after the port accepts connections (lifespan
        # done), so app.state is safe to mutate here; and it joins the server
        # thread while the patches above are still live, avoiding timing races.
        with run_app_server(app) as base_url:
            # Set AFTER the server is up: lifespan startup assigns project_cwd from
            # config, so a pre-startup value would be clobbered. It is re-read per
            # request, so setting it here (before any page load) is authoritative.
            if project_cwd is not None:
                app.state.project_cwd = project_cwd
            if allow_runtime:
                app.state.allow_runtime_panels = True
                app.state.runtime_panel_allowlist = None  # any non-loopback host allowed

            yield base_url, app


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
# a live bar that names it (the chat console's .op-session-bar, the
# .terminal-header on the terminal surface) — so it is identified by the .tile-tab-terminal
# class dock-tab.js stamps on it in both surfaces, and given the synthetic "Agent"
# label below, the stable handle for it in these tests.

# Extract every dockview group with its tab labels and pixel rectangle. Used by
# the geometry assertions (split / restack / persistence ordering).
_GROUPS_JS = r"""() => [...document.querySelectorAll('.dv-groupview')].map(g => {
  // The terminal tile's tab is identified by .tile-tab-terminal, not by the
  // bar it adopted: that bar differs between the views. It carries no
  // accessible name of its own; its label is synthesized here as "Agent" —
  // the same handle dock-workspace.js's TERMINAL_TITLE stamps on the
  // underlying dockview panel title.
  const tabs = [...g.querySelectorAll('.dv-tab')].map(t =>
    t.querySelector('.tile-tab-terminal')
      ? 'Agent'
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


def _title_case(label: str) -> str:
    """The sidebar's reading of a panel label (panel-catalog.js ``titleCase``).

    An all-caps label such as ``DATA VIZ`` is shown as ``Data Viz``; a label
    with any lower-case letter is the deployment's own spelling and is kept.
    Tests name panels the way the server does, so every tile and row lookup
    goes through this to match what the page renders.
    """
    if label != label.upper():
        return label
    return re.sub(r"(^|[\s-])(\w)", lambda m: m.group(1) + m.group(2).upper(), label.lower())


def _service_tab(page: Page, label: str):
    """A dockview tile bar whose accessible name equals ``label`` exactly.

    The name is the handle. Matched as a raw attribute (not ``get_by_label``,
    which only resolves labelable form controls, never an aria-label on a
    plain div) and exactly, so one service label like "Artifacts" cannot
    substring-match a longer one. The name is the title-cased label the page
    renders (see ``_title_case``).
    """
    bar = page.locator(f'.tile-tab[aria-label="{_title_case(label)}"]')
    return page.locator(".dv-tab").filter(has=bar)


def _focus_service_tab(page: Page, label: str) -> None:
    """Focus a service tile by clicking its tab TITLE — never the whole tab.

    A panel may contribute controls into its own tile header, which widens the
    tab far past its title. Playwright clicks an element's CENTRE, so a click on
    the tab as a whole can land inside a contributed control, which is not a
    focus gesture and leaves the tile untouched. The title is the region that
    always means "focus this tile", whatever a panel contributes beside it. The
    artifacts tile hides its own title while the shell shows its breadcrumb, so
    there the crumb's root (a plain span; the click bubbles to the tab) stands
    in for it.
    """
    _service_tab(page, label).locator(
        ".osprey-crumb-root:visible, .tile-tab-title:visible"
    ).first.click()


def _terminal_tab(page: Page):
    """The terminal tile's header bar, in either view — the .tile-tab-terminal tab."""
    return page.locator(".dv-tab").filter(has=page.locator(".tile-tab-terminal"))


_STRIP_HEIGHT_JS = r"""(terminal) => {
  const strips = [...document.querySelectorAll('.dv-tabs-and-actions-container')];
  const match = strips.find(s => !!s.querySelector('.tile-tab-terminal') === terminal);
  return match ? Math.round(match.getBoundingClientRect().height) : -1;
}"""


def _strip_height(page: Page, *, terminal: bool) -> int:
    """Rendered height of a tile's header strip, in px.

    Every tile carries the same fixed-height bar
    (dockview-overrides.css). Returns -1 when no such tile is docked.
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
    stores it last-writer-wins and broadcasts nothing. ``/api/panel-op-result``
    is the page's answer to an op the server relayed to it, not a command.
    """
    return [p for p in posts if p not in ("panel-layout", "panel-op-result")]


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


def _context_menu(page: Page):
    return page.locator(_RAIL_MENU)


def _pick_tile_menu_row(page: Page, tab, label: str) -> None:
    """Open the tile's ⋯ and run the row labelled ``label``; the menu closes."""
    tab.locator(".tile-tab-more").click()
    menu = _context_menu(page)
    expect(menu).to_be_visible(timeout=5_000)
    # The first match: "This panel" lists the panel's rows above the Tile verbs.
    menu.locator(_RAIL_MENU_ITEM, has_text=label).first.click()
    expect(menu).to_have_count(0, timeout=5_000)


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


def _open_agent_header_menu(page: Page):
    """Right-click the Agent tile's header bar and wait for its context menu.

    The press lands on the bar's bare spacer: the session label beside it is
    the session switcher, an interactive child the bar's menu yields to.
    """
    _terminal_tab(page).locator(".agent-bar-spacer:visible").first.click(button="right")
    menu = _context_menu(page)
    expect(menu).to_have_count(1, timeout=5_000)
    return menu


def _open_second_tile(page: Page, base_url: str, panel_id: str, label: str) -> None:
    """Open ``panel_id`` as a SECOND tile beside the current one.

    A rail click replaces the panel the focused service tile shows, so tests
    that need two service tiles side by side split instead. A panel already in
    a tile offers "Open in a new tile" on its rail entry; a panel in no tile is
    opened from the last docked service tile's ⋯ → Open panel ▸ Right ▸ <panel>, the
    one open that splits for it (with no service tile docked, its rail click
    opens a tile of its own). ``base_url`` is unused (kept so call sites read
    uniformly with the old add-menu dance).
    """
    del base_url  # the menu path is purely client-side
    _open_rail_menu(page, panel_id)
    beside = _menu_row(page, "Open in a new tile")
    if beside.count():
        beside.click()
    else:
        page.keyboard.press("Escape")
        expect(_context_menu(page)).to_have_count(0, timeout=5_000)
        _split_beside_a_service_tile(page, panel_id)
    expect(_service_tab(page, label)).to_have_count(1, timeout=5_000)


def _split_beside_a_service_tile(page: Page, panel_id: str) -> None:
    """Open ``panel_id`` (in no tile) right of the last docked service tile,
    through that tile's ⋯ → Open panel ▸ Right; with no service tile docked, the
    entry's click opens a tile of its own."""
    bars = page.locator('#dock-root .tile-tab[data-panel-key]:not([data-panel-key="terminal"])')
    if bars.count() == 0:
        _rail_entry(page, panel_id).click()
        return
    # The submenu names each panel as its sidebar row reads.
    shown = _rail_entry(page, panel_id).locator(".osprey-nav-label").inner_text().strip()
    _open_panel_from_tile(page, bars.last, "Right", shown)


def _open_panel_from_tile(page: Page, bar, place: str, label: str) -> None:
    """Run a tile bar's ⋯ → Open panel ▸ ``place`` ▸ ``label``."""
    bar.locator(".tile-tab-more").click()
    expect(_context_menu(page)).to_have_count(1, timeout=5_000)
    root = page.locator(_RAIL_MENU).first
    root.locator(f'{_RAIL_MENU_ITEM}[aria-haspopup="menu"]').filter(has_text="Open panel").click()
    depth1 = page.locator(f'{_RAIL_MENU}[data-depth="1"]')
    expect(depth1).to_have_count(1, timeout=5_000)
    depth1.get_by_role("menuitem", name=place, exact=True).click()
    depth2 = page.locator(f'{_RAIL_MENU}[data-depth="2"]')
    expect(depth2).to_have_count(1, timeout=5_000)
    depth2.get_by_role("menuitem", name=label, exact=True).click()
    expect(_context_menu(page)).to_have_count(0, timeout=5_000)


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


def _dock_locked(page: Page) -> bool:
    return page.evaluate(
        "async () => { const m = await import('/static/js/dock-workspace.js');"
        " const a = m.getDockApi(); return !!(a && a.locked); }"
    )


# ===========================================================================
# Group 1 — SSE / rail flows materialize as dockview panels
# ===========================================================================


def test_focus_replaces_the_service_tile_panel_with_overlay_geometry(tmp_path, chromium_browser):
    """A rail click puts the panel in the focused service tile, in place of the one it showed.

    A rail click on data-viz lands it in the tile artifacts occupies: data-viz's
    overlay iframe becomes visible and its rectangle aligns to the dock group's
    content area (the adapter's follow contract), while artifacts leaves the
    tile with its overlay hidden. A rail click on ops then replaces data-viz the
    same way, and data-viz's rail entry brings it back into the same tile, in
    the same iframe. Nothing resizes. Artifacts has no rail entry at any point.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ, _CUSTOM_OPS],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        artifacts_rail = page.locator('button.panel-rail-button[data-panel-id="artifacts"]')

        # Artifacts is auto-activated at boot: its dock tile + overlay iframe exist.
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        expect(artifacts_rail).to_have_count(0)
        before = _dock_groups(page)

        # Act — focus data-viz from the rail; it takes artifacts' place in the tile.
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

        # Assert — the data-viz overlay iframe aligns to its dock group content.
        geo = page.evaluate(_ALIGN_JS, "data-viz")
        assert geo is not None, "data-viz placeholder/overlay iframe missing"
        assert geo["active"] is True, f"data-viz is not its tile's panel: {geo}"
        assert geo["display"] != "none", f"data-viz overlay iframe hidden: {geo}"
        assert abs(geo["iframeLeft"] - geo["contentLeft"]) <= 2, geo
        assert abs(geo["iframeTop"] - geo["contentTop"]) <= 2, geo
        assert abs(geo["iframeW"] - geo["contentW"]) <= 2, geo
        assert abs(geo["iframeH"] - geo["contentH"]) <= 2, geo

        # The replaced artifacts panel: overlay hidden, its tile gone.
        expect(_overlay_iframe(page, "artifacts")).to_be_hidden(timeout=5_000)
        expect(_service_tab(page, "Artifacts")).to_have_count(0)
        expect(artifacts_rail).to_have_count(0)
        groups = _dock_groups(page)
        service_groups = [g for g in groups if "Agent" not in g["tabs"]]
        assert [g["tabs"] for g in service_groups] == [["Data Viz"]], groups
        # The same tiles at the same rectangles: an open never resizes.
        assert [(g["x"], g["y"], g["w"], g["h"]) for g in groups] == [
            (g["x"], g["y"], g["w"], g["h"]) for g in before
        ], (before, groups)

        # Mark the data-viz iframe element, so its return can be told from a rebuild.
        _overlay_iframe(page, "data-viz").evaluate("(f) => { f.dataset.kept = 'yes'; }")

        # Ops replaces data-viz; data-viz keeps its rail entry.
        _focus_service_panel(page, "ops", "OPS")
        expect(_overlay_iframe(page, "data-viz")).to_be_hidden(timeout=5_000)
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0)
        expect(page.locator('button.panel-rail-button[data-panel-id="data-viz"]')).to_be_attached(
            timeout=5_000
        )

        # Data-viz's rail entry brings it back into the same tile, in the same iframe.
        page.locator('button.panel-rail-button[data-panel-id="data-viz"]').click()
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        expect(_overlay_iframe(page, "data-viz")).to_have_attribute("data-kept", "yes")
        expect(_overlay_iframe(page, "ops")).to_be_hidden(timeout=5_000)
        groups = _dock_groups(page)
        assert [g["tabs"] for g in groups if "Agent" not in g["tabs"]] == [["Data Viz"]], groups
        assert [(g["x"], g["y"], g["w"], g["h"]) for g in groups] == [
            (g["x"], g["y"], g["w"], g["h"]) for g in before
        ], (before, groups)

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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)

        # Act — hide the only visible+healthy panel.
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "artifacts", "visible": False},
        )
        assert r.status_code == 200

        # Assert — the artifacts dock tab is gone and its overlay iframe hidden.
        expect(_service_tab(page, "Artifacts")).to_have_count(0, timeout=10_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_hidden(timeout=5_000)

        # Assert — the terminal is the sole remaining group and fills the dock.
        # The wait reads the width the assert below reads: the dock's, less the
        # gutter the sole group's view computes.
        page.wait_for_function(
            """() => {
              const groups = document.querySelectorAll('.dv-groupview');
              const dock = document.getElementById('dock-root');
              const view = document.querySelector('.dock-root .dv-view:has(> .dv-groupview)');
              if (groups.length !== 1 || !dock || !view) return false;
              const css = getComputedStyle(view);
              const room = dock.getBoundingClientRect().width
                - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight);
              return Math.abs(groups[0].getBoundingClientRect().width - room) <= 2;
            }""",
            timeout=5_000,
        )
        groups = _dock_groups(page)
        assert groups[0]["tabs"] == ["Agent"], groups
        # The card sits inside its view's gutter, so "fills the dock" means the
        # dock's width less the padding the view itself computes.
        room = page.evaluate(
            """() => {
              const dock = document.getElementById('dock-root').getBoundingClientRect();
              const view = document.querySelector('.dock-root .dv-view:has(> .dv-groupview)');
              const css = getComputedStyle(view);
              const left = parseFloat(css.paddingLeft);
              return { x: dock.x + left, w: dock.width - left - parseFloat(css.paddingRight) };
            }"""
        )
        assert abs(groups[0]["x"] - room["x"]) <= 2, (groups, room)
        assert abs(groups[0]["w"] - room["w"]) <= 2, (groups, room)

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
        # A full POST → broadcast → SSE handler round trip: the 10 s budget this
        # file gives every such echo under suite load (see the rail-menu close test).
        expect(data_viz_tab).to_have_count(0, timeout=10_000)  # removed, not dimmed

        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": True},
        )
        assert r.status_code == 200
        expect(data_viz_tab).to_be_visible(timeout=10_000)

        page.close()


def test_hide_made_before_the_event_stream_opens_reaches_the_rail(tmp_path, chromium_browser):
    """A panel hidden while the page's event stream is still connecting leaves the rail.

    The page is ready to drive before its stream is open, and SSE has no replay:
    the ``panel_visibility`` frame of a change made in that window is never
    delivered. The read the page makes when the stream opens is what applies it.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = chromium_browser.new_page()
        held = []
        page.route("**/api/files/events*", lambda route: held.append(route))
        page.goto(base_url, wait_until="domcontentloaded")
        wait_for_page_ready(page)
        data_viz_rail = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        expect(data_viz_rail).to_be_visible(timeout=10_000)
        assert held, "the event stream opened before the hide could be made"

        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200
        expect(data_viz_rail).to_have_count(1)

        for route in held:
            route.continue_()
        expect(data_viz_rail).to_have_count(0, timeout=10_000)

        page.close()


def test_hide_that_overtakes_the_stream_open_read_stays_applied(tmp_path, chromium_browser):
    """A panel hidden while the page re-reads the panel list stays off the rail.

    When its event stream opens the page reads ``/api/panels`` to pick up what it
    missed. A hide made after the server answered that read, whose frame the page
    applies before the answer arrives, is newer than the answer: the answer still
    lists the panel, and applying it would put the entry back for good.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = chromium_browser.new_page()
        stream = []
        page.route("**/api/files/events*", lambda route: stream.append(route))
        page.goto(base_url, wait_until="domcontentloaded")
        wait_for_page_ready(page)
        wait_for_dock_settled(page)
        data_viz_rail = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        expect(data_viz_rail).to_be_visible(timeout=10_000)
        assert stream, "the event stream opened before its read could be held"

        # Every read of the panel list from here on is the stream-open one. The
        # first is answered by the server at once and its answer held back.
        answered = []

        def hold_first_read(route):
            if answered:
                route.continue_()
            else:
                answered.append((route, route.fetch()))

        page.route("**/api/panels", hold_first_read)
        with page.expect_request("**/api/panels", timeout=10_000):
            for route in stream:
                route.continue_()
        # Stay in page calls, so the route handler runs, until the answer is held.
        deadline = time.monotonic() + 10
        while not answered:
            assert time.monotonic() < deadline, "the stream-open read was never answered"
            page.wait_for_timeout(50)
        route, stale = answered[0]
        assert "data-viz" in stale.json()["visible"]

        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200
        expect(data_viz_rail).to_have_count(0, timeout=10_000)

        # The held answer arrives now, older than the frame just applied. The
        # page reads the list again instead of applying it.
        with page.expect_request("**/api/panels", timeout=10_000):
            route.fulfill(response=stale)
        page.evaluate("() => fetch('/api/panels').then((r) => r.json())")
        expect(data_viz_rail).to_have_count(0)

        page.close()


def test_register_appends_rail_entry_without_docking(tmp_path, chromium_browser):
    """CF-2/CC-3: panel_register adds a rail entry but does NOT auto-dock it.

    CF-2: exactly one rail entry is appended (non-destructive; artifacts stays
          the active panel).
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
        # Artifacts has no rail entry; the client's active stamp shows the focus.
        artifacts_active = page.locator('#right-panel[data-active-panel="artifacts"]')
        expect(artifacts_active).to_have_count(1, timeout=10_000)
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
        expect(artifacts_active).to_have_count(1, timeout=2_000)

        # CC-3 — the registered panel is NOT docked and NOT active.
        expect(monitor_rail).not_to_have_class(re.compile(r"\bactive\b"))
        expect(_service_tab(page, "MONITOR")).to_have_count(0, timeout=2_000)

        page.close()


def test_reopen_after_empty_state_redocks_panel(tmp_path, chromium_browser):
    """Closing the only panel then reopening it re-docks a live iframe, not a blank.

    Close the sole data-viz panel → its dock tab is gone and the terminal is
    the only group; reopen it from the "+" menu → its dock tab and overlay
    iframe return beside the terminal. Artifacts reports no URL, so it never
    claims the slot and data-viz is the only service panel.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
        artifact_url=None,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        data_viz_rail = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        expect(data_viz_rail).to_be_visible(timeout=10_000)
        # Wait for the panel to actually be DOCKED before closing it — the
        # premise of this test is that a panel is open. `_open_page` only waits
        # for the rail entry and the first dockview group, and the rail renders
        # before the initial dock completes. Closing in that window closes
        # nothing, the dock then lands, and the tab this test expects to
        # disappear is instead created after the fact — so the assertion below
        # sees 1 for its whole timeout and reads as a product regression. Every
        # other test in this file asserts the same precondition first.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=10_000)

        # Close the only panel via its rail menu's "Remove from rail" → terminal
        # is the sole group.
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
        _open_rail_menu(page, "data-viz")
        _menu_row(page, "Remove from rail").click()
        expect(data_viz_rail).to_have_count(0, timeout=10_000)
        assert "data-viz" not in _rail_order(page)
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0, timeout=10_000)
        page.wait_for_function(
            "() => document.querySelectorAll('.dv-groupview').length === 1", timeout=5_000
        )

        # Reopen from the "+" menu.
        page.locator("#panel-add-btn").click()
        page.locator('.panel-add-item[data-panel-id="data-viz"]').click()

        # The dock tab + overlay iframe are back beside the terminal.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        assert len(_dock_groups(page)) == 2, _dock_groups(page)

        page.close()


# ===========================================================================
# Group 2 — Human dock gestures bridge to server POSTs (the echo guard)
# ===========================================================================


def test_dock_tab_focus_posts_setfocus_once(tmp_path, chromium_browser):
    """A human dock-tab focus POSTs setPanelFocus exactly once — no echo loop.

    With the two service panels in side-by-side tiles (via "Open in a new tile"
    — a rail click would replace artifacts, not add a tile), clicking the inactive artifacts tile
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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        # Open data-viz as a second tile beside artifacts; data-viz ends active.
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        # Drain the docking focus POST (async fetch) before counting.
        page.wait_for_timeout(800)

        posts = _track_panel_posts(page)
        # Human focus: click the artifacts dock tab (currently inactive).
        _focus_service_tab(page, "Artifacts")
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
        # The unified bar: visible title + pop out + close, no grip badge.
        tab = _service_tab(page, "DATA VIZ")
        expect(tab.locator(".tile-tab-grip")).to_have_count(0)
        expect(tab.locator(".tile-tab-title")).to_have_text(_title_case("DATA VIZ"))
        expect(tab.locator(".tile-tab-close")).to_have_count(1)
        expect(tab.locator(".tile-tab-window")).to_have_count(1)
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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
        # Focus the artifacts tile so data-viz's own tile is NOT the active one.
        _focus_service_tab(page, "Artifacts")
        expect(page.locator('#right-panel[data-active-panel="artifacts"]')).to_have_count(
            1, timeout=5_000
        )
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
# target — so each attempt polls for the drop bridge's split highlight before
# dropping rather than sleeping a fixed budget: a drop dispatched before the
# bridge has hit-tested the point is a no-op the assertions can only report as
# a missing tab. How long the whole gesture takes is load-dependent, so it
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
        g.querySelector('.tile-tab-terminal'));
    const content = term.querySelector('.dv-content-container');
    const r = content.getBoundingClientRect();
    // Just inside the dock's bottom strip → the engine resolves an outer
    // bottom target: a new tile along the dock's bottom (points deeper inside
    // the content are the tile's edge band or its centre, which a row never
    // lands on).
    const x = r.left + r.width / 2;
    const y = r.bottom - 5;
    const opts = { bubbles: true, cancelable: true, dataTransfer: dt,
                   clientX: x, clientY: y };
    const target = document.elementFromPoint(x, y) ?? content;
    // The bridge paints its split highlight once it has hit-tested the point,
    // so gating each drop on it proves the geometry resolved the way the test
    // claims. 20 attempts with a 500ms highlight budget each — an unloaded
    // machine lands on the first attempt in ~150ms, a loaded one gets an order
    // of magnitude more slack than any single hardcoded wait could give it.
    const ATTEMPTS = 20;
    const OVERLAY_BUDGET_MS = 500;
    const SETTLE_MS = 80;
    const overlay = () => document.querySelector('.dock-drop-preview__rect--split');
    const landed = () => !!document.querySelector(
        `.dv-groupview .dv-tab .tile-tab[data-panel-id="iframe:${panelId}"]`);
    let tries = 0;
    while (tries < ATTEMPTS && !landed()) {
        tries += 1;
        // The bridge clears its highlight on drop, so each attempt re-arms it
        // with a fresh dragenter, then holds the dragover until the split
        // highlight is painted before dispatching the drop.
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
            `${tries} highlight-gated drop attempts — the rail drop was never ` +
            `accepted by the drop bridge.`);
    }
    return tries;
}"""


def test_drag_rail_entry_to_tile_edge_opens_split(tmp_path, chromium_browser):
    """Dragging a rail entry onto a tile edge opens the panel as a new tile there.

    The precise half of the open-beside verb (rail-drag.js): the entry's HTML5
    drag carries the panel id; dropping it in the dock's bottom strip under the
    terminal tile opens the panel as a new tile along the dock's bottom, then
    the normal reveal tail focuses it.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        groups_before = len(_dock_groups(page))

        page.evaluate(_RAIL_DRAG_DROP_JS, "data-viz")

        # A new tile exists holding data-viz, below the terminal group.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
        assert len(_dock_groups(page)) == groups_before + 1, _dock_groups(page)
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == ["Agent"])
        dv = next(g for g in groups if g["tabs"] == [_title_case("DATA VIZ")])
        assert dv["y"] > term["y"], (term, dv)

        page.close()


_RAIL_REORDER_JS = """(args) => {
    const [panelId, beforeId] = args;
    const rail = document.querySelector('#panel-rail');
    const entry = rail.querySelector(`[data-panel-id="${panelId}"]`);
    const target = rail.querySelector(`[data-panel-id="${beforeId}"]`).getBoundingClientRect();
    const dt = new DataTransfer();
    const at = { bubbles: true, cancelable: true, dataTransfer: dt,
                 clientX: target.left + 2, clientY: target.top + 2 };
    entry.dispatchEvent(new DragEvent('dragstart',
        { bubbles: true, cancelable: true, dataTransfer: dt }));
    rail.dispatchEvent(new DragEvent('dragover', at));
    const marked = rail.querySelector('.drop-before')?.getAttribute('data-panel-id');
    rail.dispatchEvent(new DragEvent('drop', at));
    entry.dispatchEvent(new DragEvent('dragend', { bubbles: true, dataTransfer: dt }));
    return marked;
}"""


def _rail_order(page: Page) -> list[str]:
    """The ids of the rows in view, top to bottom (a parked row is not one)."""
    return page.eval_on_selector_all(
        "#panel-rail .panel-rail-button",
        "els => els.filter(e => e.checkVisibility()).map(e => e.dataset.panelId)",
    )


def test_drag_along_the_rail_reorders_it_for_every_client_and_survives_reload(
    tmp_path, chromium_browser
):
    """Dropping a rail entry on the rail itself moves it there.

    The order is the server's, so a second client follows the move and a
    reload draws the same rail; Agent stays at the head throughout. The two rows
    share the Tools group, because a drop into another group is a group move
    rather than a pure reorder.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ, _CUSTOM_OPS],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        other = _open_page(chromium_browser, base_url)
        # Artifacts is off the rail: its artifacts are in the session tree.
        before = ["terminal", "data-viz", "ops"]
        for client in (page, other):
            expect(client.locator('#panel-rail [data-panel-id="ops"]')).to_have_count(1)
            assert _rail_order(client) == before

        marked = page.evaluate(_RAIL_REORDER_JS, ["ops", "data-viz"])
        assert marked == "data-viz"

        after = ["terminal", "ops", "data-viz"]
        for client in (page, other):
            client.wait_for_function(
                """(want) => JSON.stringify([...document.querySelectorAll(
                    '#panel-rail .panel-rail-button')].filter(e => e.checkVisibility())
                    .map(e => e.dataset.panelId)) === JSON.stringify(want)""",
                arg=after,
                timeout=5_000,
            )
        # The drop opened no tile: the rail gesture is not the workspace one.
        expect(_service_tab(page, "OPS")).to_have_count(0)

        page.reload()
        wait_for_page_ready(page)
        expect(page.locator('#panel-rail [data-panel-id="ops"]')).to_have_count(1)
        assert _rail_order(page) == after

        other.close()
        page.close()


def test_server_sse_focus_is_applied_without_posting_back(tmp_path, chromium_browser):
    """An agent's open of an already-docked panel brings it forward silently.

    Posting /api/panel-op/open directly (as the agent's ``open_panel`` does)
    relays a ``panel_op`` frame to this page, which shows the panel and answers
    the op (/api/panel-op-result, the op's own answer). It commands nothing of
    its own: the surfaced focus runs under the echo guard, so no focus POST
    goes back out.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        # Pre-dock data-viz as a SECOND tile (a rail click would replace artifacts), then
        # make artifacts active again, so the later server focus only has to
        # re-activate an existing placeholder.
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
        _focus_service_tab(page, "Artifacts")
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        # Let the Artifacts-click's own focus POST fully drain before tracking —
        # fetch() dispatches on a later tick, so it can otherwise land after the
        # tracker attaches and be misread as a POST-back.
        page.wait_for_timeout(800)

        posts = _track_panel_posts(page)
        # The agent's open of the already-docked data-viz: relayed to this page
        # as an op, answered with the layout it shows.
        r = requests.post(
            f"{base_url}/api/panel-op/open",
            json={"request_id": "panels-open-1", "member": "data-viz"},
            timeout=10,
        )
        assert r.status_code == 200, r.text

        # It is applied — data-viz's tile takes the active focus (artifacts keeps
        # its own tile and overlay: side-by-side tiles are both on screen)...
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        expect(
            page.locator('button.panel-rail-button[data-panel-id="data-viz"].active')
        ).to_have_count(1, timeout=5_000)
        # ...with zero POSTs commanded back out (the guard suppressed the setActive).
        page.wait_for_timeout(600)
        assert _command_posts(posts) == [], f"an agent open POSTed a command back: {posts}"

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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
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
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0, timeout=10_000)
        expect(page.locator('#right-panel[data-active-panel="artifacts"]')).to_have_count(
            1, timeout=5_000
        )
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
    overlay iframe re-follows to the new rectangle.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)

        # Precondition — artifacts left of the terminal (the default split).
        groups = _dock_groups(page)
        assert len(groups) == 2, groups
        left, right = sorted(groups, key=lambda g: g["x"])
        assert left["tabs"] == ["Artifacts"] and right["tabs"] == ["Agent"], groups

        # Act — drag the artifacts tab to the right edge of the terminal group.
        term_group = page.locator(".dv-groupview").last
        box = term_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _service_tab(page, "Artifacts"),
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
                const ws = gs.find(g => g.tabs.length === 1 && g.tabs[0] === 'Artifacts');
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


# Hold an HTML5 drag of one service tile over three targets and report what the
# drop bridge (dock-drop.js) painted. The drag starts on the source bar's real
# dragstart handler (dockview's own header drag, which the bridge records as
# the tile). The first target is the terminal group's right-edge band: a split,
# so the bridge paints its `.dock-drop-preview__rect--split` highlight — the
# proof that the synthetic drag is one the bridge recognises at all. The other
# two are the target service tile's title and the centre of its content: a tile
# holds one panel, so neither is a target and nothing is painted, and the drop
# dispatched on the title changes nothing. Each target is held with repeated
# dragovers for a fixed budget. Events go to the element under the pointer, as
# a real drag would deliver them.
_HEADER_DRAG_PROBE_JS = r"""async ({ source, target }) => {
    const tabOf = (label) => [...document.querySelectorAll('.dv-tab')].find(t =>
        t.querySelector(`.tile-tab[aria-label="${label}"]`));
    const src = tabOf(source);
    const dst = tabOf(target);
    const term = [...document.querySelectorAll('.dv-groupview')].find(g =>
        g.querySelector('.tile-tab-terminal'));
    if (!src || !dst || !term) return { error: 'missing tab or terminal group' };
    const dt = new DataTransfer();
    src.dispatchEvent(new DragEvent('dragstart',
        { bubbles: true, cancelable: true, dataTransfer: dt }));
    const kind = () => {
        const box = document.querySelector('.dock-drop-preview__rect');
        return box ? [...box.classList].find((c) => c.startsWith('dock-drop-preview__rect--'))?.slice(25) ?? null : null;
    };
    const hold = async (el, x, y) => {
        const opts = { bubbles: true, cancelable: true, dataTransfer: dt,
                       clientX: x, clientY: y };
        el.dispatchEvent(new DragEvent('dragenter', opts));
        let painted = null;
        const deadline = performance.now() + 600;
        while (performance.now() < deadline) {
            el.dispatchEvent(new DragEvent('dragover', opts));
            await new Promise((res) => setTimeout(res, 50));
            painted = kind() ?? painted;
        }
        return { painted, opts };
    };
    const content = term.querySelector('.dv-content-container');
    const c = content.getBoundingClientRect();
    const ex = c.right - 5, ey = c.top + c.height / 2;
    const edgeEl = document.elementFromPoint(ex, ey) ?? content;
    const edge = await hold(edgeEl, ex, ey);
    edgeEl.dispatchEvent(new DragEvent('dragleave', { ...edge.opts, relatedTarget: document.body }));
    await new Promise((res) => setTimeout(res, 100));
    const clearedAfterEdge = kind() === null;
    const at = async (el) => {
        const r = el.getBoundingClientRect();
        const x = r.left + r.width / 2, y = r.top + r.height / 2;
        const hit = document.elementFromPoint(x, y) ?? el;
        return { hit, ...(await hold(hit, x, y)) };
    };
    const centre = await at(dst.closest('.dv-groupview').querySelector('.dv-content-container'));
    // With an artifact shown, the artifacts tile hides its name behind the
    // shell's breadcrumb root; the visible one of the two is the title.
    const titleEl = [...dst.querySelectorAll('.osprey-crumb-root, .tile-tab-title')]
        .find((el) => el.getClientRects().length > 0);
    const title = await at(titleEl);
    title.hit.dispatchEvent(new DragEvent('drop', title.opts));
    src.dispatchEvent(new DragEvent('dragend', { bubbles: true, dataTransfer: dt }));
    return { edgePainted: edge.painted, clearedAfterEdge,
             centrePainted: centre.painted, titlePainted: title.painted };
}"""


def test_a_tile_header_or_centre_is_no_drop_target(tmp_path, chromium_browser):
    """A tile dragged over another service tile's header or centre gets nothing.

    Dragging the data-viz tile over the artifacts tile's title or its centre
    paints no highlight, and the drop on the title leaves both tiles exactly
    where they were: a tile holds one panel. The same drag held over the
    terminal's right edge paints a split, so the bridge did recognise the drag.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        _open_second_tile(page, base_url, "data-viz", "DATA VIZ")

        before = _dock_groups(page)
        assert all(len(g["tabs"]) == 1 for g in before), before

        # Act — the header and centre drops, with a split target as contrast.
        probe = page.evaluate(
            _HEADER_DRAG_PROBE_JS, {"source": _title_case("DATA VIZ"), "target": "Artifacts"}
        )
        assert "error" not in probe, probe
        assert probe["edgePainted"] == "split", f"the bridge never recognised the drag: {probe}"
        assert probe["clearedAfterEdge"] is True, probe
        assert probe["centrePainted"] is None, f"a tile centre painted a highlight: {probe}"
        assert probe["titlePainted"] is None, f"a tile title painted a highlight: {probe}"

        # Assert — the drop on the title changed nothing.
        page.wait_for_timeout(500)
        after = _dock_groups(page)
        assert after == before, (before, after)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible()
        expect(_overlay_iframe(page, "data-viz")).to_be_visible()

        page.close()


# ===========================================================================
# Group 4 — Layout persistence keyed by project_key
# ===========================================================================


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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
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
                const term = gs.find(g => g.querySelector('.tile-tab-terminal'));
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
        wait_for_page_ready(page)
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        groups = _dock_groups(page)
        term_b = next(g for g in groups if g["tabs"] == ["Agent"])
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
        wait_for_page_ready(page)
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        groups = _dock_groups(page)
        term_a = next(g for g in groups if g["tabs"] == ["Agent"])
        assert term_a["x"] == min(g["x"] for g in groups), (
            f"project A's stored layout (terminal left) was not isolated: {groups}"
        )

        page.close()


def _reset_layout_from_palette(page: Page) -> None:
    """Run "Reset layout" from the command palette, as an operator would."""
    _open_palette(page)
    _palette_query(page, "reset layout")
    row = _palette_row(page, "Reset layout")
    expect(row).to_have_count(1, timeout=5_000)
    row.click()
    expect(page.locator(_PALETTE_OVERLAY)).to_have_count(0, timeout=5_000)


def _reset_layout_from_sidebar_menu(page: Page) -> None:
    """Run "Reset layout" from the sidebar header's right-click menu."""
    page.locator("#sidebar-toggle").click(button="right")
    expect(page.get_by_role("menu", name="Sidebar")).to_have_count(1, timeout=5_000)
    _menu_row(page, "Reset layout").click()
    expect(_context_menu(page)).to_have_count(0, timeout=5_000)


@pytest.mark.parametrize(
    "reset",
    [_reset_layout_from_palette, _reset_layout_from_sidebar_menu],
    ids=["palette", "sidebar-menu"],
)
def test_reset_layout_restores_the_default(tmp_path, chromium_browser, reset):
    """The "Reset layout" control clears a custom arrangement back to the default.

    Offered in the command palette and on the sidebar header's menu. Rearrange
    (terminal → left), then reset: the grid returns to the default split
    (service tile left, terminal right) and the reset survives a reload (the
    stored value is the default, not the discarded custom one).
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

        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        first_group = page.locator(".dv-groupview").first
        fb = first_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _terminal_tab(page),
            first_group,
            target_position={"x": 8, "y": fb["height"] / 2},
        )
        # The custom arrangement really landed — a reset from the default would
        # prove nothing.
        page.wait_for_function(
            """() => { const gs = [...document.querySelectorAll('.dv-groupview')];
                const term = gs.find(g => g.querySelector('.tile-tab-terminal'));
                return term && Math.min(...gs.map(g => g.getBoundingClientRect().x)) === term.getBoundingClientRect().x; }""",
            timeout=5_000,
        )
        page.wait_for_timeout(500)

        # Act — reset through the operator's control.
        reset(page)
        page.wait_for_function(
            """() => { const gs = [...document.querySelectorAll('.dv-groupview')];
                const term = gs.find(g => g.querySelector('.tile-tab-terminal'));
                return gs.length === 2 && term && Math.max(...gs.map(g => g.getBoundingClientRect().x)) === term.getBoundingClientRect().x; }""",
            timeout=5_000,
        )

        # Assert — default: the service tab-stack on the left, terminal right.
        groups = _dock_groups(page)
        assert len(groups) == 2, groups
        left, right = sorted(groups, key=lambda g: g["x"])
        assert left["tabs"] == ["Artifacts"], groups
        assert right["tabs"] == ["Agent"], groups

        # And the reset persists across a reload (custom arrangement is gone).
        page.wait_for_timeout(500)
        page.reload(wait_until="domcontentloaded")
        wait_for_page_ready(page)
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == ["Agent"])
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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
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
        wait_for_page_ready(page)
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        # Default arrangement loaded (terminal on the right), grid is functional.
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == ["Agent"])
        assert term["x"] == max(g["x"] for g in groups), (
            f"corrupt layout did not fall back to default: {groups}"
        )
        assert _dock_locked(page) is False

        page.close()


# ===========================================================================
# Group 5 — Agent menu rows
# ===========================================================================


# ===========================================================================
# Group 6 — The "+" add menu (rail-region behaviors, dock-aware)
# ===========================================================================


def test_add_menu_reveals_hidden_panel(tmp_path, chromium_browser):
    """Clicking a hidden panel in the "+" menu un-hides its rail entry and docks it.

    data-viz starts visible, is hidden via the API, then revealed from the menu —
    which puts it in the active service tile in place of the panel it showed,
    like every other open, and POSTs the reveal.
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
        expect(data_viz_rail).to_have_count(0, timeout=10_000)

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
        expect(page.locator("#panel-add-menu")).to_be_visible(timeout=5_000)

        geo = page.evaluate(
            """() => {
                const menu = document.getElementById('panel-add-menu').getBoundingClientRect();
                const btn = document.getElementById('panel-add-btn').getBoundingClientRect();
                const region = document.getElementById('sidebar');
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
        expect(page.locator("#panel-add-menu")).to_be_visible(timeout=5_000)
        expect(page.locator(".panel-add-input")).to_have_count(0)
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
        # Label "Ariel" derives id "ariel", which collides with a built-in →
        # the server 422s and the menu must display the reason, not fail silently.
        page.locator('.panel-add-menu input[name="label"]').fill("Ariel")
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
            page.goto(base_url, wait_until="domcontentloaded")
            wait_for_page_ready(page)
            expect(page.locator('button[data-panel-id="data-viz"]')).to_be_attached(timeout=10_000)

            # The visible, healthy panel is the one docked and on screen.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="data-viz"].active')
            ).to_have_count(1, timeout=10_000)
            expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
            expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

            # The hidden default never surfaced itself: no rail entry in view
            # (artifacts is off the rail), no dock tab.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="artifacts"]')
            ).to_be_hidden()
            assert "artifacts" not in _rail_order(page)
            expect(_service_tab(page, "Artifacts")).to_have_count(0)

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

    hits: list[str] = []
    with _stub_backend(hits) as backend_url:
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
            page.goto(base_url, wait_until="domcontentloaded")
            wait_for_page_ready(page)
            expect(page.locator('button[data-panel-id="terminal"]')).to_be_attached(timeout=10_000)

            # The premise: data-viz's health poll really reaches the live stub
            # and goes healthy, which is the transition the buggy fallback
            # keyed on. Without a hit the negatives below could not fail.
            deadline = time.monotonic() + 10.0
            while not any(h.endswith("/health") for h in hits):
                assert time.monotonic() < deadline, f"no health poll reached the stub: {hits}"
                page.wait_for_timeout(100)
            # One settle after the healthy answer for a wrong surface to land.
            page.wait_for_timeout(1_000)

            # A non-member has NO rail entry at all (no dimmed placeholder),
            # and its healthy transition must not have docked or activated it.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
            ).to_have_count(0)
            expect(_service_tab(page, "DATA VIZ")).to_have_count(0)
            expect(_overlay_iframe(page, "data-viz")).not_to_be_visible(timeout=5_000)

            page.close()


def test_tile_notice_follows_backend_outage(tmp_path, chromium_browser):
    """A polled panel whose backend stops answering says so over its own tile.

    The notice covers data-viz's whole tile, not its neighbour's, paints its
    grey wash, takes no pointer events (the page under it still receives
    clicks and scrolls), and clears on the backend's next answer. The test
    waits on the real 10 s health cadence, about 45 s in total: faking the
    page clock would also fake the dock's own timers.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    down = threading.Event()
    with _stub_backend(down=down) as backend_url:
        polled_panel = {
            "id": "data-viz",
            "label": "DATA VIZ",
            "url": backend_url,
            "healthEndpoint": "/health",
            "path": "/",
        }
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[polled_panel],
        ) as (base_url, app):
            app.state.visible_panels = ["artifacts", "data-viz"]

            page = _open_page(chromium_browser, base_url)
            expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
            expect(
                page.locator('button.panel-rail-button[data-panel-id="data-viz"]:not(.disabled)')
            ).to_have_count(1, timeout=15_000)
            _open_second_tile(page, base_url, "data-viz", "DATA VIZ")
            frame = _overlay_iframe(page, "data-viz")
            expect(frame).to_be_visible(timeout=5_000)

            down.set()

            notice = page.locator('.tile-notice[data-panel="data-viz"][data-unreachable]')
            expect(notice).to_be_visible(timeout=35_000)

            # The wash covers the whole page: the notice spans the iframe's box.
            nbox = notice.bounding_box()
            fbox = frame.bounding_box()
            assert nbox is not None and fbox is not None
            for key in ("x", "y", "width", "height"):
                assert abs(nbox[key] - fbox[key]) <= 1, (key, nbox, fbox)

            look = notice.evaluate(
                "el => { const cs = getComputedStyle(el);"
                " return { bg: cs.backgroundColor, pe: cs.pointerEvents }; }"
            )
            assert look["bg"] != "rgba(0, 0, 0, 0)", look
            assert look["pe"] == "none", look

            # Clicks and scrolls still reach the page under the wash.
            hit = page.evaluate(
                "([x, y]) => { const el = document.elementFromPoint(x, y);"
                " return el ? (el.dataset.panelId ?? el.className) : null; }",
                [fbox["x"] + fbox["width"] / 2, fbox["y"] + fbox["height"] / 2],
            )
            assert hit == "data-viz", hit

            expect(
                page.locator('.tile-notice[data-panel="artifacts"][data-unreachable]')
            ).to_have_count(0)

            down.clear()
            expect(page.locator(".tile-notice[data-unreachable]")).to_have_count(0, timeout=15_000)

            page.close()


# ===========================================================================
# Group 8 — First-class terminal tile + launcher-rail semantics
# ===========================================================================


def test_terminal_rail_entry_close_and_reopen(tmp_path, chromium_browser):
    """The terminal is the rail's permanent anchor: tile closes, entry never changes.

    The rail's FIRST entry is the terminal ('Agent'); the anchor is not
    removable — its tile closes via the dock tab or the Delete key on the
    focused entry. Closing the tile leaves the entry in place at full
    brightness; clicking it re-docks the same live terminal card.
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


def test_rail_click_on_a_replaced_panel_brings_it_back(tmp_path, chromium_browser):
    """A replaced panel keeps its rail entry, and one click puts it back in the tile.

    A rail click on data-viz replaces artifacts in its tile, and one on ops
    replaces data-viz; clicking the data-viz entry replaces ops with data-viz
    again, in place. Every entry stays; no visibility POST fires in either
    direction: which panel a tile shows is per-client state.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ, _CUSTOM_OPS],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        data_viz_rail = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        ops_rail = page.locator('button.panel-rail-button[data-panel-id="ops"]')
        expect(data_viz_rail).to_be_visible(timeout=10_000)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

        posts = _track_panel_posts(page)

        # Replace data-viz by activating ops.
        _focus_service_panel(page, "ops", "OPS")
        expect(_overlay_iframe(page, "data-viz")).to_be_hidden(timeout=5_000)
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0)
        expect(data_viz_rail).to_be_attached()

        # Bring data-viz back through its entry.
        data_viz_rail.click()
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        expect(_overlay_iframe(page, "ops")).to_be_hidden(timeout=5_000)
        expect(_service_tab(page, "OPS")).to_have_count(0)
        expect(ops_rail).to_be_attached()
        groups = _dock_groups(page)
        assert [g["tabs"] for g in groups if "Agent" not in g["tabs"]] == [["Data Viz"]], groups

        # Focus POSTs only — which panel a tile shows never touches server visibility.
        expect(page.locator('#right-panel[data-active-panel="data-viz"]')).to_have_count(
            1, timeout=5_000
        )
        assert "panel-visibility" not in posts, f"a panel switch must stay local, got {posts}"
        assert set(_command_posts(posts)) == {"panel-focus"}, f"expected only focus POSTs: {posts}"

        page.close()


def test_rail_click_on_open_panel_jumps_without_moving_it(tmp_path, chromium_browser):
    """Clicking the rail entry of an OPEN panel focuses its tile — no move, no replace.

    With data-viz and ops in side-by-side tiles (ops active), clicking the
    data-viz rail entry activates the data-viz tile in place: both tiles
    survive, nothing is evicted, and both rail entries stay lit (open).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ, _CUSTOM_OPS],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        _open_second_tile(page, base_url, "ops", "OPS")
        expect(_overlay_iframe(page, "ops")).to_be_visible(timeout=5_000)
        before = _dock_groups(page)

        # Act — rail click on the OPEN (but unfocused) data-viz panel.
        page.locator('button.panel-rail-button[data-panel-id="data-viz"]').click()

        # Assert — data-viz is active and visible in ITS OWN tile; the layout
        # is untouched and ops keeps its tile and its lit rail entry.
        expect(page.locator('#right-panel[data-active-panel="data-viz"]')).to_have_count(
            1, timeout=5_000
        )
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        expect(_service_tab(page, "OPS")).to_have_count(1)
        expect(page.locator('button.panel-rail-button[data-panel-id="ops"]')).to_be_attached()
        assert _dock_groups(page) == before, "a jump-to-open-panel changed the layout"

        page.close()


# JS: the tab strip of the group holding the terminal tile, with its current
# pixel height — the fixed-height assertions poll this. The terminal group is
# found by its .tile-tab-terminal (its tab carries no accessible name).
_TERMINAL_STRIP_H_JS = """() => {
    const term = [...document.querySelectorAll('.dv-groupview')].find(g =>
        g.querySelector('.tile-tab-terminal'));
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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        expect(_terminal_tab(page)).to_have_count(1, timeout=10_000)

        # Exactly one terminal header row, and it lives in the tab, not the card.
        expect(page.locator(".op-session-bar")).to_have_count(1)
        expect(page.locator(".dv-tab .op-session-bar")).to_have_count(1)

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

        # A service tile carries the unified bar: visible title + pop out +
        # close, no grip badge. No artifact is shown here, so the artifacts
        # tile keeps its own name as its visible title (with an artifact shown,
        # the bar's breadcrumb "Artifact › <title>" takes the name's place).
        ws = _service_tab(page, "Artifacts")
        expect(ws.locator(".tile-tab-grip")).to_have_count(0)
        expect(
            ws.locator(".osprey-crumb-root:visible, .tile-tab-title:visible").first
        ).to_have_text("Artifacts")
        expect(ws.locator(".tile-tab-close")).to_have_count(1)
        expect(ws.locator(".tile-tab-window")).to_have_count(1)

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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=10_000)
        # The boot layout replaces the default arrangement; a rectangle or sash
        # read before that is one of the arrangement about to be dropped.
        wait_for_dock_settled(page)

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
            # Wait, button still down, for the group to have resized and the
            # live follower to have re-applied the overlay's geometry to it.
            page.wait_for_function(
                f"""async ([sid, w0]) => {{
                    const r = await ({_ALIGN_JS})(sid);
                    return !!r && Math.abs(r.contentW - w0) >= 100
                        && Math.abs(r.iframeLeft - r.contentLeft) <= 2
                        && Math.abs(r.iframeW - r.contentW) <= 2;
                }}""",
                arg=["artifacts", before["contentW"]],
            )

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
# clicks round-trip back as canopy-host-header-action. Proven here with a custom
# panel whose (proxied, same-origin) page contributes a nav + a button on
# load and records incoming actions — the full pipeline the unit suites can
# only cover in halves.

_CONTRIB_PANEL_HTML = b"""<!doctype html>
<html><body><script>
  window.__actions = [];
  window.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'canopy-host-header-action') {
      window.__actions.push([e.data.id, e.data.value ?? null]);
    }
  });
  parent.postMessage({
    type: 'canopy-host-header-contribution',
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


def _contributing_panel(panel_id: str, label: str, url: str) -> dict:
    """A custom panel entry for a page that posts a header contribution.

    Config-declared, as the real loader stamps it: a first-party panel is
    framed same-origin, so its page can post the header contribution to the
    hub. A runtime registration would be opaque-origin and its contribution is
    a separate contract.
    """
    return {
        "id": panel_id,
        "label": label,
        "url": url,
        "healthEndpoint": None,
        "path": "/",
        "configDefined": True,
    }


@contextmanager
def _contrib_panel_backend():
    """Serve a page that contributes to the tile bar and records actions."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_CONTRIB_PANEL_HTML)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_header_contribution_renders_and_round_trips(reserve_port, tmp_path, chromium_browser):
    """A panel's contribution renders in ITS tile bar and actions round-trip.

    The custom panel's page (reverse-proxied same-origin) posts a nav + button
    contribution on load; the bar must render them beside the tile title, a
    bar click must deliver canopy-host-header-action back into the iframe, and the
    artifacts tile (which contributes nothing) must keep an empty region.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _contrib_panel_backend() as stub_url:
        custom = _contributing_panel("contrib-demo", "CONTRIB", stub_url)
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[custom],
            # The premise below is that artifacts contributes NOTHING, which
            # only holds while its iframe fails to load. The default fallback
            # URL is the artifacts server's standard port, so a real artifacts
            # server running on this host (a live web terminal, a demo stack)
            # would load for real and contribute its own nav items — turning
            # that assertion into a report about the machine. A reserved port
            # keeps the panel dead by construction.
            artifact_url=f"http://127.0.0.1:{reserve_port()}",
        ) as (base_url, _app):
            page = _open_page(chromium_browser, base_url)
            # Three tiles beside the sidebar: at the default width a tile folds
            # its lowest-priority item into its overflow menu. Give them the
            # room to show every item in the bar, and prove nothing folded.
            page.set_viewport_size({"width": 1600, "height": 800})
            _open_second_tile(page, base_url, "contrib-demo", "CONTRIB")

            # Contribution renders inside the CONTRIB tile's bar only.
            tab = _service_tab(page, "CONTRIB")
            expect(tab.locator(".contrib-nav-link")).to_have_count(2, timeout=10_000)
            expect(tab.locator(".contrib-nav-link").first).to_be_visible()
            expect(tab.locator(".contrib-hidden")).to_have_count(0)
            expect(tab.locator(".contrib-nav-link.active")).to_have_text("Browse")
            expect(tab.locator(".contrib-btn")).to_have_text("Refresh")
            expect(tab.locator(".tile-tab-contrib.has-items")).to_have_count(1)

            # The CONTRIB panel's items never leak into another tile. (The
            # artifacts tile has items of its own: the shell's crumb and pager.)
            ws = _service_tab(page, "Artifacts")
            expect(ws.locator(".contrib-nav-link")).to_have_count(0)
            expect(ws.locator(".contrib-btn")).to_have_count(0)

            # Round-trip: bar clicks land in the panel as canopy-host-header-action.
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


def test_header_contribution_reaches_the_fallback_bar_without_dockview(
    reserve_port, tmp_path, chromium_browser
):
    """With no dockview, a panel's contributed controls still have a bar.

    A panel on embed contract v2 hides its in-body copies of what it
    contributes, so a hub whose dockview bundle failed to load (a firewalled
    host, a blocked CDN) must still host the contribution, or the controls
    (an Open session button) are reachable nowhere. The bundle is blocked for
    this page, so the hub genuinely boots its fallback layout; the panel opens
    in #panel-content, its contribution must render in the fallback tile bar,
    and a bar click must round-trip into the iframe.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _contrib_panel_backend() as stub_url:
        custom = _contributing_panel("contrib-demo", "CONTRIB", stub_url)
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[custom],
            artifact_url=f"http://127.0.0.1:{reserve_port()}",
        ) as (base_url, _app):
            page = chromium_browser.new_page()
            errors: list[str] = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.route("**/dockview-core*", lambda route: route.abort())
            page.goto(base_url, wait_until="domcontentloaded")
            wait_for_page_ready(page)
            rail = page.locator('button.panel-rail-button[data-panel-id="contrib-demo"]')
            expect(rail).not_to_have_class(re.compile(r"\bdisabled\b"), timeout=15_000)
            assert (
                page.evaluate(
                    "async () => { const m = await import('/static/js/dock-workspace.js');"
                    " return m.getDockApi(); }"
                )
                is None
            ), "the dock shell was supposed to be unavailable on this page"
            assert any("dockview-core global unavailable" in e for e in errors), errors

            rail.click()
            frame_el = page.locator('#panel-content iframe[data-panel-id="contrib-demo"]')
            expect(frame_el).to_be_visible(timeout=15_000)

            bar = page.locator("#right-panel > .fallback-tile-bar")
            expect(bar).to_have_count(1)
            expect(bar).to_be_visible()
            expect(bar.locator(".tile-tab-title")).to_have_text(_title_case("CONTRIB"))
            expect(bar.locator(".contrib-nav-link")).to_have_count(2, timeout=10_000)
            refresh = bar.locator('[data-contrib-key="button:refresh"]')
            expect(refresh).to_be_visible()

            bar.locator(".contrib-nav-link", has_text="New Entry").click()
            refresh.click()
            frame = page.frame_locator('#panel-content iframe[data-panel-id="contrib-demo"]')
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
      type: 'canopy-host-header-contribution',
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
    if (e.data && e.data.type === 'canopy-host-header-action') {
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
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_SEARCH_MENU_PANEL_HTML)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
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
        custom = _contributing_panel("search-demo", "SEARCHY", stub_url)
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

            # The contributed menu is the tile ⋯'s "This panel" section, and an
            # entry click round-trips.
            tab.locator(".tile-tab-more").click()
            menu = _context_menu(page)
            expect(menu).to_be_visible(timeout=5_000)
            headings = menu.locator(".rail-context-heading")
            expect(headings).to_have_text(["This panel", "Tile"])
            menu.locator(_RAIL_MENU_ITEM, has_text="Refresh").click()
            expect(menu).to_have_count(0)

            actions = _poll_actions(page, frame, 2)
            assert actions and actions[-1] == ["more", "refresh"], actions

            page.close()


# CANOPY is not a custom panel: it is a SIDECAR, so its URL comes from
# app.state.canopy_server_url and its same-origin standing from the sidecar's
# own entry pages (sidecar_panels.py), not from a config marker. This stub
# stands in for the sidecar at the real mount: it answers /health, and the
# shell at / moves itself to one embedded page under /embed/ — an entry page,
# so served unsandboxed — which posts a header contribution and records every
# action the hub routes back.
#
# Under header protocol v1 the hub renders exactly the controls a panel
# contributes, so each CANOPY page shape shows every control once. CANOPY
# still sends an inert ``text`` item (its subtitle); the hub drops it, since
# the tile bar carries the tile's name and controls only. Two shapes are
# pinned, each in its own hub boot:
#
#   * a one-tile SCREEN: subtitle, the screen verbs (Print, Export PNG),
#     Search, ⋯ and Save view;
#   * the WORKSPACE page, at a distinct /embed/workspace/… path: a summary
#     text, Search as a ``button`` at priority 1, and ⋯ with Reset layout,
#     Show sub-screens and Collections.
#
# Priorities follow the fold ladder: the tile ⋯ lowest, verbs one higher than
# their right neighbour, Save view above the verbs.
_CANOPY_PAGES = {
    "screen": {
        "path": "/embed/demo/overview",
        "items": [
            {"kind": "text", "id": "subtitle", "text": "Trace · 2 curves · live", "priority": 5},
            {"kind": "button", "id": "print", "label": "Print", "priority": 3},
            {"kind": "button", "id": "export-png", "label": "Export PNG", "priority": 2},
            {"kind": "button", "id": "search", "label": "Search ⌘K", "priority": 1},
            {
                "kind": "menu",
                "id": "more",
                "priority": 0,
                "items": [
                    {"id": "menu:reset-layout", "label": "Reset layout"},
                    {"id": "tool:export-csv", "label": "Export CSV"},
                ],
            },
            {
                "kind": "button",
                "id": "save-view",
                "label": "Save view",
                "title": "Save view (⌘S)",
                "tone": "accent",
                "priority": 4,
            },
        ],
    },
    "workspace": {
        "path": "/embed/workspace/demo",
        "items": [
            {"kind": "text", "id": "summary", "text": "3 screens · 2 collections", "priority": 5},
            {"kind": "button", "id": "search", "label": "Search ⌘K", "priority": 1},
            {
                "kind": "menu",
                "id": "more",
                "priority": 0,
                "items": [
                    {"id": "menu:reset-layout", "label": "Reset layout"},
                    {"id": "menu:show-sub-screens", "label": "Show sub-screens"},
                    {"id": "menu:collections", "label": "Collections"},
                ],
            },
        ],
    },
}


def _canopy_shell_html(embed_path: str) -> bytes:
    """The sidecar root: move to ``embed_path`` beneath the panel's mount."""
    return (
        b"""<!doctype html>
<html><body><script>
  // The hub frames the panel's root; move to an embedded page beneath it.
  const base = location.pathname.replace(/\\/?$/, '/');
  location.replace(base + """
        + json.dumps(embed_path.lstrip("/")).encode()
        + b""" + location.search);
</script></body></html>"""
    )


def _canopy_embed_html(items: list[dict]) -> bytes:
    """An embed page that contributes ``items`` and records routed actions."""
    return (
        b"""<!doctype html>
<html><body><script>
  window.__actions = [];
  window.addEventListener('message', (e) => {
    if (e.source === window.parent && e.data && e.data.type === 'canopy-host-header-action') {
      window.__actions.push([e.data.id, e.data.value ?? null]);
    }
  });
  parent.postMessage({
    type: 'canopy-host-header-contribution',
    version: 1,
    items: """
        + json.dumps(items).encode()
        + b""",
  }, window.location.origin);
</script></body></html>"""
    )


@contextmanager
def _canopy_sidecar_stub(page_name: str):
    """Serve a stand-in CANOPY sidecar whose shell redirects to ``page_name``.

    Both embed pages are routed — ``/embed/workspace/…`` to the workspace page,
    any other ``/embed/…`` to the screen — so the redirect alone picks the page.
    """
    shell = _canopy_shell_html(_CANOPY_PAGES[page_name]["path"])
    workspace = _canopy_embed_html(_CANOPY_PAGES["workspace"]["items"])
    screen = _canopy_embed_html(_CANOPY_PAGES["screen"]["items"])

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/health":
                body, ctype = b'{"status":"ok"}', "application/json"
            elif path.startswith("/embed/workspace/"):
                body, ctype = workspace, "text/html"
            elif path.startswith("/embed/"):
                body, ctype = screen, "text/html"
            elif path == "/":
                body, ctype = shell, "text/html"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _canopy_hub(tmp_path, chromium_browser, page_name: str):
    """Boot a hub whose CANOPY sidecar serves ``page_name``; yield (page, base_url)."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _canopy_sidecar_stub(page_name) as stub_url:

        async def _publish_stub(app, enabled_panels):
            # What _launch_sidecar publishes once a real sidecar answers. CANOPY
            # takes no operator secret, so no panel_auth_headers entry is owed.
            assert "canopy" in enabled_panels
            app.state.canopy_server_url = stub_url

        with (
            patch(
                "osprey.interfaces.web_terminal.app._launch_enabled_sidecars",
                new=AsyncMock(side_effect=_publish_stub),
            ),
            _live_server(
                workspace,
                enabled_panels={"artifacts", "canopy"},
                # Keep artifacts dead by construction, as the neighbour above.
                artifact_url=unserved_url(),
            ) as (base_url, _app),
        ):
            page = _open_page(chromium_browser, base_url)
            try:
                _open_second_tile(page, base_url, "canopy", "CANOPY")
                yield page, base_url
            finally:
                page.close()


_CANOPY_IFRAME = '.dock-iframe-overlay iframe[data-panel-id="canopy"]'


def _canopy_activate(page: Page, tab, label: str) -> None:
    """Click the control labelled ``label`` wherever the bar put it.

    On a wide tile it is a button in the bar; a menu entry, or a button the
    ladder folded on a narrow tile, is a row of the tile ⋯. Either way it
    dispatches the same action, so the round-trip is width-independent.
    """
    button = tab.locator(".contrib-btn:visible", has_text=label)
    if button.count():
        button.click()
        return
    _pick_tile_menu_row(page, tab, label)


@pytest.mark.parametrize("page_name", ["screen", "workspace"])
def test_canopy_header_contribution_renders_exactly_and_round_trips(
    tmp_path, chromium_browser, page_name
):
    """Each CANOPY page shape renders exactly its contributed control set.

    CANOPY's subtitle ``text`` item is part of what the stub sends and none of
    what the bar shows.

    The sidecar is replaced by a stub published at ``app.state.canopy_server_url``
    (what a started sidecar publishes), so the iframe is CANOPY's real mount,
    ``/panel/canopy/embed/…`` through the panel proxy. The hub must render one
    element per contributed item but the retired ``text`` — counted by
    ``[data-contrib-key]``, folded ones included, so the count does not depend
    on the tile's width — with the tile's one ⋯, and every button and every ⋯
    entry must reach the
    frame as ``canopy-host-header-action`` carrying CANOPY's ids: a button sends
    ``{id}``, a menu entry ``{id: "more", value: <entry id>}``.
    """
    spec = _CANOPY_PAGES[page_name]
    # The bar shows no subtitle: a `text` item is dropped like an unknown kind.
    items = [it for it in spec["items"] if it["kind"] != "text"]
    expected_keys = sorted(f"{it['kind']}:{it['id']}" for it in items)

    with _canopy_hub(tmp_path, chromium_browser, page_name) as (page, base_url):
        tab = _service_tab(page, "CANOPY")
        keyed = tab.locator("[data-contrib-key]")
        expect(keyed).to_have_count(len(items), timeout=10_000)
        expect(tab.locator(".contrib-text")).to_have_count(0)

        # Exactly the contributed set: no item dropped, none duplicated.
        keys = keyed.evaluate_all("(els) => els.map((el) => el.dataset.contribKey)")
        assert sorted(keys) == expected_keys, keys
        # One ⋯ only, the tile's: a contributed menu renders no button.
        expect(tab.locator(".tile-tab-more")).to_have_count(1)
        expect(tab.locator("[data-contrib-key^='menu:']:visible")).to_have_count(0)

        # The frame sits at this page's path under CANOPY's mount, same origin.
        frame_url = page.locator(_CANOPY_IFRAME).evaluate("(el) => el.contentWindow.location.href")
        assert frame_url.startswith(f"{base_url}/panel/canopy{spec['path']}"), frame_url

        save = [it for it in items if it["id"] == "save-view"]
        if save:
            accent = tab.locator(".contrib-btn.contrib-btn-accent")
            expect(accent).to_have_count(1)
            expect(accent).to_have_text(save[0]["label"])
            expect(accent).to_have_attribute("title", save[0]["title"])

        expected_actions = []
        for item in items:
            if item["kind"] == "button":
                _canopy_activate(page, tab, item["label"])
                expected_actions.append([item["id"], None])
            elif item["kind"] == "menu":
                for entry in item["items"]:
                    _canopy_activate(page, tab, entry["label"])
                    expected_actions.append([item["id"], entry["id"]])

        frame = page.frame_locator(_CANOPY_IFRAME)
        actions = _poll_actions(page, frame, len(expected_actions))
        assert actions == expected_actions, actions


# The drag ghost dockview builds for a tab is a deep clone of that tab, parented
# to document.body — OUTSIDE #dock-root (verified against the vendored 7.0.2
# bundle: the Tab's `createGhost` calls `_buildGhostElement()`, which clones the
# element and inlines the computed style ON THE ROOT ONLY, then `addGhostImage`
# appends the clone to document.body before `setDragImage`; the pointer-drag
# path parents its ghost to document.body too).
#
# So a tile's drag image is styled by whatever still matches once the bar is out
# of #dock-root. Two things keep that honest, and this test guards both.
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
  const icons = [...ghost.querySelectorAll('.bar-icon, svg.osprey-ic')].map((el) => ({
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
        custom = _contributing_panel("search-demo", "SEARCHY", stub_url)
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
            # The bar always renders its ⋯ and close ×, and this panel adds a
            # search magnifier — a ghost with no icons would be a vacuous pass.
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
    """Right-clicking a healthy rail entry opens the menu with every verb.

    The popover is body-level and viewport-positioned (it has to escape the
    rail's own overflow), names the panel in its accessible name, and carries
    the verbs in words with the rail's glyphs kept as trailing decoration. A
    panel in no tile offers Open; the group moves sit in the Move to submenu,
    and the membership verb, under a divider, is the only destructive one and
    is styled apart.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)

        menu = _open_rail_menu(page, "data-viz")

        assert _menu_labels(page) == [
            "Open",
            "Open in new window",
            "Move to",
            "Remove from rail",
        ], _menu_labels(page)
        expect(menu).to_have_attribute("role", "menu")
        expect(menu).to_have_attribute("aria-label", f"{_title_case('DATA VIZ')} actions")
        expect(menu.locator(".rail-context-divider")).to_have_count(1)
        expect(menu.locator(".rail-context-glyph")).to_have_count(2)
        expect(_menu_row(page, "Move to")).to_have_attribute("aria-haspopup", "menu")
        expect(menu.locator(f"{_RAIL_MENU_ITEM}.danger")).to_have_text(
            re.compile(r"Remove from rail")
        )
        # data-viz's URL resolved at boot, so the window row is live.
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
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=10_000)
        # The entry has to be enabled before the keyboard route is allowed —
        # the same gate the mouse route passes through.
        expect(_rail_entry(page, "data-viz")).not_to_have_class(
            re.compile(r"\bdisabled\b"), timeout=10_000
        )

        _rail_entry(page, "data-viz").focus()
        page.keyboard.press("Shift+F10")

        # Exactly one popover, and the keyboard user lands on its first row.
        expect(_context_menu(page)).to_have_count(1, timeout=5_000)
        assert (
            page.evaluate(
                "() => document.activeElement?.querySelector('.rail-context-label')?.textContent ?? ''"
            )
            == "Open"
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


def test_remove_from_rail_undo_restores_the_entry_at_its_old_position(tmp_path, chromium_browser):
    """ "Remove from rail" offers Undo, and Undo puts the entry back where it was.

    The removed entry sits in the MIDDLE of the rail, so a restore that merely
    re-added it would land it at the foot instead — the position is what this
    proves, over the real POST → broadcast → handler round trip both ways.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    ops = {**_CUSTOM_DATA_VIZ, "id": "ops", "label": "OPS", "url": "http://ops.internal:8080"}

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ, ops],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        rail_ids = (
            "() => [...document.querySelectorAll('.panel-rail-button')].map(e => e.dataset.panelId)"
        )
        expect(_rail_entry(page, "ops")).to_be_visible(timeout=10_000)
        before = page.evaluate(rail_ids)
        assert before.index("data-viz") < before.index("ops"), before

        _open_rail_menu(page, "data-viz")
        _menu_row(page, "Remove from rail").click()

        notice = page.locator("#undo-notice")
        expect(notice).to_contain_text(
            f"{_title_case('DATA VIZ')} removed from the rail", timeout=5_000
        )
        expect(_rail_entry(page, "data-viz")).to_have_count(0, timeout=10_000)

        notice.get_by_role("button", name="Undo").click()

        expect(_rail_entry(page, "data-viz")).to_have_count(1, timeout=10_000)
        expect(notice).to_have_count(0)
        assert page.evaluate(rail_ids) == before

        page.close()


def test_rail_context_menu_popout_row_inert_until_url_resolves(tmp_path, chromium_browser):
    """The Open in new window row renders inert while the panel has no frame to move.

    A panel's URL and its health arrive together at boot (one with no URL never
    polls, so it never enables), which is why the entry is force-enabled here:
    the menu's enabled gate is a DOM-class question, and taking the class off is
    the only way to reach an entry that HAS a menu while its frame does not exist
    yet — the state an operator meets when a companion server is slow to answer.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts", "ariel"},
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        # ARIEL's server publishes no URL, so its entry never enables.
        entry = _rail_entry(page, "ariel")
        expect(entry).to_have_class(re.compile(r"\bdisabled\b"), timeout=10_000)
        page.evaluate(
            "() => document.querySelector('button.panel-rail-button"
            "[data-panel-id=\"ariel\"]').classList.remove('disabled')"
        )

        menu = _open_rail_menu(page, "ariel")
        popout = _menu_row(page, "Open in new window")
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
    """An offline panel's entry offers no menu — not by a press, not by keyboard.

    The gate is in the handler rather than the rail's CSS on purpose: the
    disabled entry's `pointer-events: none` stops a real mouse press from
    reaching it, but a keyboard menu request never consults CSS at all, and
    the CSS is not what this guarantee should rest on. Both routes are driven
    here; neither may reach a menu whose verbs act on a panel that has never
    answered. Declining leaves the event untouched, so what an operator gets on
    those presses is the browser's own menu.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts", "ariel"},
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        # ARIEL's server publishes no URL, so its entry never enables.
        entry = _rail_entry(page, "ariel")
        expect(entry).to_have_class(re.compile(r"\bdisabled\b"), timeout=10_000)

        # The press, delivered to the entry itself: `pointer-events: none`
        # would make a real right-click land on whatever lies beneath (and
        # Playwright's click would wait for a hit forever), so the event is
        # dispatched straight at the handler the gate lives in.
        entry.dispatch_event("contextmenu")
        page.wait_for_timeout(500)
        expect(_context_menu(page)).to_have_count(0)

        # The keyboard route bypasses `pointer-events` entirely.
        entry.focus()
        page.keyboard.press("Shift+F10")
        page.wait_for_timeout(500)
        expect(_context_menu(page)).to_have_count(0)

        page.close()


def test_chat_trims_the_agent_menu_to_the_layout_rows(tmp_path, chromium_browser):
    """In chat the PTY verb would act on a surface the operator cannot see.

    The Agent tile header keeps Switch to terminal, Export conversation and the tile
    rows; the Agent rail row holds the session rows.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        _open_rail_menu(page, "data-viz")
        assert _menu_labels(page) == [
            "Open",
            "Open in new window",
            "Move to",
            "Remove from rail",
        ], _menu_labels(page)

        page.keyboard.press("Escape")
        expect(_context_menu(page)).to_have_count(0, timeout=5_000)

        _open_agent_header_menu(page)
        labels = _menu_labels(page)
        # "Agent": the way to the terminal and the chat's own rows (Export
        # conversation, Desktop notifications); "Tile": the tile rows, no PTY
        # verb. The notifications row names a browser that blocks them;
        # headless may.
        assert labels[:2] == ["Switch to terminal", "Export conversation"], labels
        assert labels[2].startswith("Desktop notifications"), labels
        assert labels[3:] == [
            "Maximize",
            "Zoom",
            "Open in new window",
            "Reveal in sidebar",
            "Close Agent pane",
        ], labels

        page.keyboard.press("Escape")
        expect(_context_menu(page)).to_have_count(0, timeout=5_000)

        _open_rail_menu(page, "terminal")
        assert _menu_labels(page) == [
            "New session",
            "Artifacts by",
            "Show pinned sessions",
            "Collapse all",
        ], _menu_labels(page)

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
        page = chromium_browser.new_page()
        arm_terminal_tab(page)
        page.goto(base_url, wait_until="domcontentloaded")
        wait_for_page_ready(page)
        expect(page.locator(".xterm-viewport")).to_have_count(1, timeout=10_000)
        # The Agent tile's header is re-parented while the dock boots; a press
        # landing between the two moves opens no menu.
        wait_for_dock_settled(page)

        _open_agent_header_menu(page)
        # The pane's "This panel" rows come first; the Tile rows close the menu.
        assert _menu_labels(page)[-6:] == [
            "Maximize",
            "Zoom",
            "Open in new window",
            "Reveal in sidebar",
            "Close Agent pane",
            "Restart terminal",
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


def test_agent_header_menu_opens_while_the_panel_list_is_still_loading(tmp_path, chromium_browser):
    """The Agent tile is on screen before ``/api/panels`` answers, and so is its menu.

    The dock boots beside the panel manager, which then waits on the panel list.
    Under load that wait is seconds long, and a right-click in it must not fall
    through to the browser's own menu: the Agent's verbs need no panel list.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(workspace, enabled_panels={"artifacts"}) as (base_url, _app):
        page = chromium_browser.new_page()
        held = []
        page.route("**/api/panels", lambda route: held.append(route))
        arm_terminal_tab(page)
        page.goto(base_url, wait_until="domcontentloaded")  # page-ready: its panel list is held
        expect(page.locator(".xterm-viewport")).to_have_count(1, timeout=10_000)
        assert held, "the panel list was answered before the menu could be tried"

        _open_agent_header_menu(page)
        assert "Close Agent pane" in _menu_labels(page), _menu_labels(page)

        page.keyboard.press("Escape")
        for route in held:
            route.continue_()
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
        custom = _contributing_panel("search-demo", "SEARCHY", stub_url)
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

            # The bar itself is a menu surface: the same menu as its ⋯, the
            # panel's own rows over the tile verbs.
            tab.locator(".tile-tab-title").click(button="right")
            menu = _context_menu(page)
            expect(menu).to_have_count(1, timeout=5_000)
            expect(menu).to_have_attribute("aria-label", "Searchy actions")
            assert _menu_labels(page) == [
                "All sessions",
                "Refresh",
                "Open panel",
                "Maximize",
                "Zoom",
                "Open in new window",
                "Reload",
                "Reveal in sidebar",
                "Close tile",
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
#     panel really is included in the window source and excluded from the
#     new-tile source.
#   * FR8 - the window verb end to end: `window.open` opens the hub's shell page,
#     and the panel's own frame moves into it, on the proxied, same-origin
#     /panel/<id> URL.
#   * FR9 - Recent. Real localStorage, and the section really renders first on
#     an empty query and survives a reload.
#
# ARIEL is the panel under test because it is the one the synonym table has
# domain vocabulary for ("elog"). It is enabled as a built-in panel
# with its companion server URL published post-startup - the catalog entry
# carries no healthEndpoint, so it is healthy the moment its config fetch
# resolves, and the stub backend behind it lets the proxied /panel/ariel URL
# answer for real when its frame loads in the window.

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
    // The label, not the row: a row's detail ("new tab") is a separate span.
    if (inRecent) rows.push(node.querySelector('.command-palette-item-label').textContent);
  }
  return inRecent ? rows : null;
}"""


@contextmanager
def _palette_live_server(workspace):
    """A live server whose ARIEL panel is a rail member with a resolved URL.

    ``_launch_panel_server`` is stubbed to publish only the gallery URL, so
    ARIEL's is set here - after startup, before any page load, which is when
    the browser first fetches /api/ariel-server. It points at the always-200
    stub so the proxied ``/panel/ariel`` its frame loads answers for real
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
    the ARIEL rows — the popout row in particular — for as long as it stays
    open, and no amount of typing brings them back. The rail entry sheds
    ``.disabled`` (a CSS class, not the HTML attribute: panel-rail.js
    ``setEntryEnabled``) in the same settle that publishes the URL (initPanel →
    assumeHealthy), so waiting on the class is waiting on the rows'
    precondition. In isolation that takes well under a second; at the tail of a
    long single-process browser run it can exceed the palette's own 5 s budget,
    hence the wider wait here. Every test that opens the palette and expects an
    ARIEL row must pass through here first.
    """
    expect(
        page.locator('button.panel-rail-button[data-panel-id="ariel"]:not(.disabled)')
    ).to_be_attached(timeout=15_000)


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

    The hotkey path first focuses a rail entry (ARIEL's, which the palette
    server always enables): off macOS the shortcut deliberately yields to
    readline's Ctrl+K while focus is inside the terminal, so the gesture has
    to start where the operator's would.
    """
    if hotkey:
        _rail_entry(page, "ariel").focus()
        page.keyboard.press("Meta+k" if page.evaluate(_IS_MAC_JS) else "Control+k")
    else:
        page.locator("#command-palette-btn").click()
    expect(page.locator(_PALETTE_OVERLAY)).to_be_visible(timeout=5_000)


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
    wait_for_page_ready(page)
    expect(_rail_entry(page, "terminal", enabled_only=True)).to_be_attached(timeout=10_000)
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

        _wait_for_ariel_ready(page)
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
        expect(_palette_row(page, "Open Logbook in new window")).to_have_count(1, timeout=5_000)
        expect(_palette_row(page, "Open Settings")).to_have_count(0)

        page.close()


def test_palette_query_ariel_offers_popout_of_the_active_panel(tmp_path, chromium_browser):
    """FR8: with ARIEL active, "ariel" finds its popout row but no new-tile row.

    The two "Open ..." verbs draw on different sources, and the live getters
    are what this proves: ``getPopoutPanels`` includes the ACTIVE panel
    (moving what you are looking at to a window is the verb's commonest use, and
    it needed ARIEL's frame to exist, once its URL resolved through
    /api/ariel-server),
    while ``getVisiblePanels`` - the new-tile source - excludes it, because
    opening a panel beside itself is a no-op. The second query proves the
    new-tile verb is present at all, so ARIEL's missing
    row is the active-panel exclusion rather than a whole verb going absent.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _palette_live_server(workspace) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _wait_for_ariel_ready(page)
        _focus_service_panel(page, "ariel", "LOGBOOK")

        _open_palette(page)
        _palette_query(page, "ariel")

        expect(_palette_row(page, "Open Logbook in new window")).to_have_count(1, timeout=5_000)
        # The query really narrowed the list, so the absences below mean the
        # rows do not exist rather than that everything is still on screen.
        expect(_palette_row(page, "Open Settings")).to_have_count(0)
        expect(_palette_row(page, "Open Logbook in a new tile")).to_have_count(0)
        # Focus is the same story as the new tile: the active panel is not
        # offered a verb that would do nothing.
        expect(_palette_row(page, "Focus Logbook")).to_have_count(0)

        # The new-tile verb itself is alive here - it just skips the active panel.
        _palette_query(page, "tile")
        expect(_palette_row(page, "Open Artifacts in a new tile")).to_have_count(1, timeout=5_000)
        expect(_palette_row(page, "Open Logbook in a new tile")).to_have_count(0)

        page.close()


def test_palette_elog_synonym_matches_logbook(tmp_path, chromium_browser):
    """FR8: "elog" - the domain word, not the label - finds the Logbook rows.

    The rail label is not the only word an operator reaches for, so one who
    types "elog" gets nothing unless the synonym tokens really reach the live
    rows' searchText.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _palette_live_server(workspace) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _wait_for_ariel_ready(page)
        _open_palette(page)
        _palette_query(page, "elog")

        expect(_palette_row(page, "Open Logbook in new window")).to_have_count(1, timeout=5_000)
        # A word ARIEL's own rows are the only ones carrying: Artifacts, the
        # other rail member, is filtered out entirely.
        expect(_palette_row(page, "Artifacts")).to_have_count(0)

        page.close()


def test_palette_window_row_moves_the_panel_frame_on_its_proxied_url(tmp_path, chromium_browser):
    """FR8: running the Open in new window row moves the panel's frame into a hub window.

    happy-dom can only prove ``window.open`` was called with some string. What
    matters to the operator is what shows: the hub's own shell page, holding the
    panel's frame on the proxied, root-relative URL resolved against the terminal's
    origin - never the panel's internal address, which the browser cannot reach -
    and the tile gone from the dock.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _palette_live_server(workspace) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        _wait_for_ariel_ready(page)
        _focus_service_panel(page, "ariel", "LOGBOOK")
        _open_palette(page)
        _palette_query(page, "ariel window")
        row = _palette_row(page, "Open Logbook in new window")
        expect(row).to_have_count(1, timeout=5_000)

        with page.context.expect_page() as popup_info:
            row.click()
        popup = popup_info.value
        wait_for_page_ready(popup)
        frame = popup.locator('iframe.panel-iframe[data-panel-id="ariel"]')
        expect(frame).to_be_visible(timeout=10_000)

        assert urlsplit(popup.url).path == "/static/attached-window.html"
        opened = urlsplit(frame.get_attribute("src") or "")
        query = parse_qs(opened.query)
        assert f"{opened.scheme}://{opened.netloc}{opened.path}" == f"{base_url}/panel/ariel"
        assert query.get("embedded") == ["true"], opened
        assert "theme" in query, opened
        assert "popout" not in query, opened
        expect(_service_tab(page, "LOGBOOK")).to_have_count(0)

        popup.close()
        page.close()


def test_palette_recent_section_leads_the_empty_query_after_an_execution(
    tmp_path, chromium_browser
):
    """FR9: an executed item comes back first under "Recent", and survives reload.

    Recent is a presentation-level section rebuilt from localStorage keys that
    are re-resolved against the live registry on each render, so this drives
    the whole loop through a real browser store: execute, reopen on an empty
    query, reload, reopen again. "Open activity log" is the item because it is
    offered in every mode and opens a new tab, so the proof does not hinge on
    this page being left in some other state.
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
        _palette_query(page, "activity log")
        row = _palette_row(page, "Open activity log")
        expect(row).to_have_count(1, timeout=5_000)
        row.click()
        expect(page.locator(_PALETTE_OVERLAY)).to_have_count(0, timeout=5_000)

        # Assert - it leads the empty-query render, ahead of every group.
        _open_palette(page)
        headings = page.evaluate(_PALETTE_HEADINGS_JS)
        assert headings and headings[0] == "Recent", headings
        assert page.evaluate(_PALETTE_RECENT_ROWS_JS) == ["Open activity log"]
        # The normal groups still render underneath - Recent never replaces them.
        assert "Panels" in headings, headings
        page.keyboard.press("Escape")
        expect(page.locator(_PALETTE_OVERLAY)).to_have_count(0, timeout=5_000)

        # Assert - the history lives in the browser store, not in module memory.
        _palette_reload(page)
        _open_palette(page)
        assert page.evaluate(_PALETTE_RECENT_ROWS_JS) == ["Open activity log"]

        page.close()
