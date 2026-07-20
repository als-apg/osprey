"""Browser smoke tests: the docked (dockview) workspace and its server bridge.

The service panels no longer live in a fixed left/right split — they are docked
in a dockview grid (``dock-workspace.js``) whose panel content follows an overlay
iframe layer (``dock-iframe.js``), with a state bridge (``dock-sync.js``) that
turns human dock gestures into the same server POSTs the agent's MCP calls make.
These tests drive that stack through a real browser and a live SSE stream — the
parts the FastAPI TestClient cannot reach.

What is proven here, grouped:

  * SSE / rail flows now materialize as dockview panels: focusing a service panel
    docks a placeholder whose overlay iframe tracks the group geometry; hiding it
    removes the placeholder; register appends a rail entry WITHOUT auto-docking.
  * The echo guard (``dock-sync.js``): a human dock-tab focus POSTs setPanelFocus
    exactly once, a human tab-close POSTs visibility(false), and a server-driven
    SSE focus is applied WITHOUT POSTing back (no feedback loop).
  * Drag rearrange via real Playwright drags — split to a new group, restack onto
    another group's tab bar — asserted with getBoundingClientRect geometry, and
    the overlay iframe re-following the dragged panel's new rectangle.
  * Layout persistence keyed by project_key: an expert arrangement survives a
    reload, two project cwds keep distinct layouts on one origin, reset restores
    the default, and a corrupt stored value falls back cleanly.
  * Locked simple mode: no drag affordance (a drag is a no-op) and no per-tab
    close control; an SSE register during simple mode is folded in with no dead
    tabs once the operator flips back to expert.
  * The "+" add menu: unclipped geometry beside the rail, URL-row gating, reveal a
    hidden panel, register from URL, and inline register errors.

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
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            # Set the artifact URL without actually spawning a server process.
            side_effect=lambda a: setattr(a.state, "artifact_server_url", artifact_url),
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


def _open_page(browser, base_url: str) -> Page:
    """Open a new browser page and wait for the rail + dock grid to render.

    panel-manager.js renders the rail asynchronously (it fetches /api/panels then
    each panel's config endpoint) and dock-workspace.js builds the dockview grid;
    this helper blocks until both are present so individual tests can assume a
    stable starting DOM.
    """
    page = browser.new_page()
    page.goto(base_url, wait_until="domcontentloaded")
    # Artifacts is always enabled and the DEFAULT_PANEL_FALLBACK, so its rail
    # button appears quickly after the async init path completes. Iframes also
    # carry data-panel-id, so target the button element specifically.
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    # The dockview grid is up once at least one group is on screen.
    expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
    # The first-visit welcome overlay intercepts pointer events; remove it so
    # tests that click header controls get a genuinely interactable starting DOM.
    page.evaluate("document.getElementById('welcome-overlay')?.remove()")
    return page


# ---------------------------------------------------------------------------
# Dock DOM helpers
# ---------------------------------------------------------------------------
#
# dockview does not stamp its panel ids on the tab DOM, so tabs are addressed by
# their visible label text (the panel title). The native terminal tab is titled
# empty on purpose (the card self-labels) and carries aria-label="Session", which
# is the stable handle for it.

# Extract every dockview group with its tab labels and pixel rectangle. Used by
# the geometry assertions (split / restack / persistence ordering).
_GROUPS_JS = r"""() => [...document.querySelectorAll('.dv-groupview')].map(g => {
  const tabs = [...g.querySelectorAll('.dv-tab .dv-default-tab-content')].map(t => t.textContent);
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
    """A dockview tab whose visible label equals ``label`` exactly (case-sensitive).

    The ^…$ regex keeps one service label (e.g. "WORKSPACE") from substring-
    matching another panel's longer label.
    """
    content = page.locator(".dv-default-tab-content", has_text=re.compile(rf"^{re.escape(label)}$"))
    return page.locator(".dv-tab").filter(has=content)


def _terminal_tab(page: Page):
    """The native terminal panel's dock tab (empty title → aria-label='Session')."""
    return page.locator('.dv-tab[aria-label="Session"]')


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


def _drag_with_dock_shield(page: Page, source, target, **kwargs) -> None:
    """``source.drag_to(target)`` with the app's real-drag iframe shield pre-raised.

    During a genuine pointer drag, dockview's onWillDragPanel raises the pointer
    shield (dock-workspace.js + dock-iframe.js) so overlay iframes can't swallow
    the drag events over a covered group. Playwright's synthetic drag_to hit-tests
    the drop point BEFORE any dragstart could raise that shield, so a drop onto a
    group covered by an overlay iframe is judged "intercepted" and times out.
    Pre-raise the same shield for the gesture, then restore it.
    """
    shield = (
        "(v) => document.querySelectorAll('.dock-iframe-overlay iframe')"
        ".forEach(i => { i.style.pointerEvents = v; })"
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
    """Focusing a service panel docks a placeholder whose overlay iframe tracks it.

    A rail click on data-viz makes it the active dock tab; its overlay iframe
    becomes visible and its rectangle aligns to the dock group's content area
    (the adapter's follow contract). The previously-active artifacts overlay is
    concealed because it is no longer the active tab in the shared group.
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

        # The artifacts overlay iframe is behind the now-active data-viz tab.
        expect(_overlay_iframe(page, "artifacts")).to_be_hidden(timeout=5_000)

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
        assert groups[0]["tabs"] == [""], groups
        dock_w = page.evaluate(
            "() => document.getElementById('dock-root').getBoundingClientRect().width"
        )
        assert abs(groups[0]["w"] - dock_w) <= 2, (groups, dock_w)

        page.close()


def test_visibility_hide_and_show_toggles_rail_entry(tmp_path, chromium_browser):
    """SSE panel_visibility toggles a rail entry hidden→visible without a reload.

    data-viz is never focused here (so it has no dock placeholder yet); the
    visibility echo drives only its rail entry, proving the SSE→rail path is
    intact under the docked shell.
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
        expect(data_viz_tab).not_to_be_visible(timeout=5_000)

        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": True},
        )
        assert r.status_code == 200
        expect(data_viz_tab).to_be_visible(timeout=5_000)

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

        # Close the only panel via its rail "×" → terminal is the sole group.
        artifacts_rail.hover()
        page.locator('button[data-panel-id="artifacts"] .panel-rail-close').click()
        expect(_service_tab(page, "WORKSPACE")).to_have_count(0, timeout=5_000)
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

    With both service panels stacked in one group, clicking the (inactive)
    artifacts dock tab fires one panel-focus POST. The server's SSE echo is
    applied under the echo guard, so it does not POST focus back again.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        # Dock data-viz so both service tabs share a group; data-viz ends active.
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        # Drain the docking focus POST (async fetch) before counting.
        page.wait_for_timeout(800)

        posts = _track_panel_posts(page)
        # Human focus: click the artifacts dock tab (currently inactive).
        _service_tab(page, "WORKSPACE").click()
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        # Let any (wrongly-)looping echo settle before counting.
        page.wait_for_timeout(600)

        assert posts == ["panel-focus"], f"expected one focus POST, got {posts}"

        page.close()


def test_dock_tab_close_posts_visibility_false(tmp_path, chromium_browser):
    """A human tab-close ('×' → .dv-default-tab-action) POSTs visibility(false) once.

    Closing the ACTIVE data-viz tab reveals artifacts behind it, which is a
    legitimate focus change and may emit one benign focus POST for artifacts; the
    invariant under test is that the close fires the visibility POST exactly once
    (no echo loop), not the absence of that follow-on focus.
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
        # Let the freshly-docked tab's layout settle before clicking its close
        # control: dock-sync resolves the tab→panel id by DOM position in capture
        # phase, so a click mid-settle can map to nothing and drop the POST.
        expect(_service_tab(page, "DATA VIZ").locator(".dv-default-tab-action")).to_be_visible(
            timeout=5_000
        )
        page.wait_for_timeout(600)

        posts = _track_panel_posts(page)
        # Close the data-viz dock tab via its close control.
        _service_tab(page, "DATA VIZ").locator(".dv-default-tab-action").click()

        # Its dock tab and rail entry both retire; exactly one visibility POST.
        expect(_service_tab(page, "DATA VIZ")).to_have_count(0, timeout=5_000)
        expect(
            page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        ).not_to_be_visible(timeout=5_000)
        page.wait_for_timeout(600)
        assert posts.count("panel-visibility") == 1, f"expected one visibility POST, got {posts}"

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
        # Pre-dock data-viz, then make artifacts active again, so the later
        # server focus only has to re-activate an existing placeholder.
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        _service_tab(page, "WORKSPACE").click()
        expect(_overlay_iframe(page, "artifacts")).to_be_visible(timeout=5_000)
        # Let the WORKSPACE-click's own focus POST fully drain before tracking —
        # fetch() dispatches on a later tick, so it can otherwise land after the
        # tracker attaches and be misread as a POST-back.
        page.wait_for_timeout(800)

        posts = _track_panel_posts(page)
        # Server-driven focus back to the already-docked data-viz.
        r = requests.post(f"{base_url}/api/panel-focus", json={"panel": "data-viz"})
        assert r.status_code == 200

        # It is applied — data-viz's overlay iframe surfaces again...
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)
        expect(_overlay_iframe(page, "artifacts")).to_be_hidden(timeout=5_000)
        # ...with zero POSTs echoed back out (the guard suppressed the setActive).
        page.wait_for_timeout(600)
        assert posts == [], f"server-driven focus POSTed back: {posts}"

        page.close()


def test_agent_hide_of_active_panel_posts_no_focus(tmp_path, chromium_browser):
    """An SSE-driven hide of the ACTIVE panel emits ZERO focus POSTs from the client.

    Regression guard for the echo bug: dropping the active service panel's
    placeholder makes dockview auto-activate the stacked neighbor, and that
    programmatic active-panel change must be recognised as a server-applied echo
    (panel-manager wraps the hide in the echo guard). If it leaked a
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
        # Dock data-viz and leave it active; artifacts is the visible healthy neighbor.
        _focus_service_panel(page, "data-viz", "DATA VIZ")
        expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

        # Precondition — the two service panels MUST share one group with data-viz
        # active. Only then does hiding data-viz drop its placeholder and make
        # dockview auto-activate the stacked artifacts neighbor — the programmatic
        # focus change the echo guard has to suppress. If a future default layout
        # split them into separate groups, artifacts would already be active in its
        # own group and this test would pass trivially without exercising C1.
        groups = _dock_groups(page)
        shared = [g for g in groups if "DATA VIZ" in g["tabs"] and "WORKSPACE" in g["tabs"]]
        assert len(shared) == 1, f"expected data-viz stacked with artifacts: {groups}"

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


def test_drag_splits_panel_into_new_group(tmp_path, chromium_browser):
    """Dragging a stacked tab to a group's right edge splits it into a new group.

    Both service panels start stacked in one group. Dragging the data-viz tab to
    the right edge of the terminal group moves it into its own group at the far
    right; its overlay iframe re-follows to the new rectangle.
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

        # Precondition — artifacts + data-viz share one group (a tab stack).
        groups = _dock_groups(page)
        shared = [g for g in groups if "DATA VIZ" in g["tabs"] and "WORKSPACE" in g["tabs"]]
        assert len(shared) == 1, f"expected data-viz stacked with artifacts: {groups}"

        # Act — drag the data-viz tab to the right edge of the terminal group.
        term_group = page.locator(".dv-groupview").last
        box = term_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _service_tab(page, "DATA VIZ"),
            term_group,
            target_position={"x": box["width"] - 8, "y": box["height"] / 2},
        )

        # Assert — data-viz now sits alone in the rightmost group.
        page.wait_for_function(
            """() => {
                const gs = [...document.querySelectorAll('.dv-groupview')].map(g => ({
                    x: g.getBoundingClientRect().x,
                    tabs: [...g.querySelectorAll('.dv-default-tab-content')].map(t => t.textContent),
                }));
                const dv = gs.find(g => g.tabs.length === 1 && g.tabs[0] === 'DATA VIZ');
                return dv && Math.max(...gs.map(g => g.x)) === dv.x;
            }""",
            timeout=5_000,
        )
        groups = _dock_groups(page)
        dv_group = next(g for g in groups if g["tabs"] == ["DATA VIZ"])
        artifacts_group = next(g for g in groups if "WORKSPACE" in g["tabs"])
        assert dv_group["x"] > artifacts_group["x"], groups
        assert "DATA VIZ" not in artifacts_group["tabs"], groups

        # The overlay iframe re-followed to the new group content rectangle.
        geo = page.evaluate(_ALIGN_JS, "data-viz")
        assert geo is not None and geo["active"] is True, geo
        assert abs(geo["iframeLeft"] - geo["contentLeft"]) <= 2, geo
        assert abs(geo["iframeW"] - geo["contentW"]) <= 2, geo

        page.close()


def test_drag_restacks_panel_onto_another_tab(tmp_path, chromium_browser):
    """Dropping a split-out tab back onto another group's tab restacks them.

    Split data-viz into its own group, then drag its tab onto the artifacts
    ('WORKSPACE') tab: the two share one group again (same x/width, two tabs).
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

        # Split data-viz out to the right first.
        term_group = page.locator(".dv-groupview").last
        box = term_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _service_tab(page, "DATA VIZ"),
            term_group,
            target_position={"x": box["width"] - 8, "y": box["height"] / 2},
        )
        page.wait_for_function(
            """() => [...document.querySelectorAll('.dv-groupview')]
                .some(g => { const t = [...g.querySelectorAll('.dv-default-tab-content')].map(e=>e.textContent);
                            return t.length === 1 && t[0] === 'DATA VIZ'; })""",
            timeout=5_000,
        )

        # Act — drag the data-viz tab onto the artifacts tab to restack.
        _drag_with_dock_shield(
            page, _service_tab(page, "DATA VIZ"), _service_tab(page, "WORKSPACE")
        )

        # Assert — they occupy one group again (same rectangle, two tabs).
        page.wait_for_function(
            """() => [...document.querySelectorAll('.dv-groupview')]
                .some(g => { const t = [...g.querySelectorAll('.dv-default-tab-content')].map(e=>e.textContent);
                            return t.includes('DATA VIZ') && t.includes('WORKSPACE'); })""",
            timeout=5_000,
        )
        groups = _dock_groups(page)
        shared = [g for g in groups if "DATA VIZ" in g["tabs"] and "WORKSPACE" in g["tabs"]]
        assert len(shared) == 1, f"expected data-viz restacked with artifacts: {groups}"

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
        term_before = next(g for g in groups if g["tabs"] == [""])
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
                const term = gs.find(g => [...g.querySelectorAll('.dv-default-tab-content')]
                    .every(t => t.textContent === '') );
                return term && Math.min(...gs.map(g => g.getBoundingClientRect().x)) === term.getBoundingClientRect().x; }""",
            timeout=5_000,
        )
        # Let the debounced persist write flush (schedulePersist ~150ms).
        page.wait_for_timeout(500)

        # Reload; the arrangement must be restored.
        page.reload(wait_until="domcontentloaded")
        page.evaluate("document.getElementById('welcome-overlay')?.remove()")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        groups = _dock_groups(page)
        term_after = next(g for g in groups if g["tabs"] == [""])
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
                const term = gs.find(g => [...g.querySelectorAll('.dv-default-tab-content')].every(t => t.textContent === ''));
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
        page.evaluate("document.getElementById('welcome-overlay')?.remove()")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        groups = _dock_groups(page)
        term_b = next(g for g in groups if g["tabs"] == [""])
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
        page.evaluate("document.getElementById('welcome-overlay')?.remove()")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        groups = _dock_groups(page)
        term_a = next(g for g in groups if g["tabs"] == [""])
        assert term_a["x"] == min(g["x"] for g in groups), (
            f"project A's stored layout (terminal left) was not isolated: {groups}"
        )

        page.close()


def test_reset_restores_default_layout(tmp_path, chromium_browser):
    """resetDockLayout() clears a custom arrangement back to the default.

    Rearrange (terminal → left), then reset: the grid returns to the default
    split (service tab-stack left, terminal right) and the reset survives a
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
        assert right["tabs"] == [""], groups

        # And the reset persists across a reload (custom arrangement is gone).
        page.wait_for_timeout(500)
        page.reload(wait_until="domcontentloaded")
        page.evaluate("document.getElementById('welcome-overlay')?.remove()")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == [""])
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
        page.evaluate("document.getElementById('welcome-overlay')?.remove()")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
        page.wait_for_timeout(1_500)

        # Default arrangement loaded (terminal on the right), grid is functional.
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == [""])
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
    (.dv-default-tab-action) hidden by the simple-mode CSS; a Playwright drag of a
    tab leaves the arrangement unchanged.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
        ui_mode="simple",
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")
        page.wait_for_timeout(1_000)

        # Locked, and no per-tab close control is visible.
        assert _dock_locked(page) is True
        expect(page.locator(".dv-tab .dv-default-tab-action:visible")).to_have_count(0)

        # The chat/terminal card keeps the right-hand column in simple mode
        # (service tabs stack on the left).
        groups = _dock_groups(page)
        term = next(g for g in groups if g["tabs"] == [""])
        assert term["x"] == max(g["x"] for g in groups), groups

        # A drag is a no-op — the arrangement is byte-identical before/after.
        before = _dock_groups(page)
        first_group = page.locator(".dv-groupview").first
        fb = first_group.bounding_box()
        _drag_with_dock_shield(
            page,
            _service_tab(page, "WORKSPACE"),
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
        page.locator('#mode-toggle .mode-segment[data-mode="expert"]').click()
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "expert", timeout=5_000)
        page.wait_for_timeout(1_200)
        assert _dock_locked(page) is False

        # No dead tabs: every dock tab is a known panel; monitor is not docked yet.
        groups = _dock_groups(page)
        all_tabs = [t for g in groups for t in g["tabs"]]
        assert all(t in ("", "WORKSPACE") for t in all_tabs), (
            f"unexpected/dead dock tab after restore: {groups}"
        )
        assert all_tabs.count("MONITOR") == 0, groups

        # Focusing the panel registered during simple mode docks one clean tab.
        _focus_service_panel(page, "monitor", "MONITOR")
        groups = _dock_groups(page)
        monitor_tabs = [t for g in groups for t in g["tabs"] if t == "MONITOR"]
        assert len(monitor_tabs) == 1, f"expected exactly one MONITOR tab: {groups}"

        page.close()


# ===========================================================================
# Group 6 — The "+" add menu (rail-region behaviors, dock-aware)
# ===========================================================================


def test_add_menu_reveals_hidden_panel(tmp_path, chromium_browser):
    """Clicking a hidden panel in the "+" menu un-hides its rail entry and docks it.

    data-viz starts visible, is hidden via the API, then revealed from the menu —
    which docks it beside the active group and POSTs the visibility reveal.
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
        expect(data_viz_rail).not_to_be_visible(timeout=5_000)

        # Open the "+" menu; the hidden panel is offered as an add target.
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

        # Hide a panel so the menu has a focusable "Show panel" item — the focus
        # auto-scroll only fires when there is something to focus.
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200
        expect(
            page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        ).not_to_be_visible(timeout=5_000)

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
            page.goto(base_url, wait_until="domcontentloaded")
            expect(page.locator('button[data-panel-id="data-viz"]')).to_be_attached(timeout=10_000)
            page.evaluate("document.getElementById('welcome-overlay')?.remove()")

            # The visible, healthy panel is the one docked and on screen.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="data-viz"].active')
            ).to_have_count(1, timeout=10_000)
            expect(_service_tab(page, "DATA VIZ")).to_have_count(1, timeout=5_000)
            expect(_overlay_iframe(page, "data-viz")).to_be_visible(timeout=5_000)

            # The hidden default never surfaced itself.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="artifacts"]')
            ).not_to_be_visible()
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
            page.goto(base_url, wait_until="domcontentloaded")
            expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)
            page.evaluate("document.getElementById('welcome-overlay')?.remove()")

            # Give the async init + health poll time to (wrongly) surface it.
            page.wait_for_timeout(3_000)

            # Guard: the arrangement only tests anything if data-viz went healthy.
            expect(
                page.locator('button.panel-rail-button[data-panel-id="data-viz"]:not(.disabled)')
            ).to_have_count(1, timeout=10_000)

            data_viz_rail = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
            expect(data_viz_rail).not_to_be_visible(timeout=5_000)
            expect(
                page.locator('button.panel-rail-button[data-panel-id="data-viz"].active')
            ).to_have_count(0)
            # Never docked, never overlaid.
            expect(_service_tab(page, "DATA VIZ")).to_have_count(0)
            expect(_overlay_iframe(page, "data-viz")).not_to_be_visible(timeout=5_000)

            page.close()
