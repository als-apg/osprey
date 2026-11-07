"""Browser e2e: a pane in a window of its own, on the hermetic hub.

The tile header's Open in new window moves a pane's live content into a window the hub
drives, opened on the hub's shell page (pane-window.js, attached-window.js), and
the tile leaves the dock. Pins:

- The Agent pane: the window is at the hub's address, with a bar naming the pane
  and Reattach above the chat; a message typed in the window goes through the
  hub's own chat client; a click on the session in use brings the window forward;
  Reset layout leaves no Agent tile; Reattach docks it again, with the
  conversation; closing the window closes the pane, and its row reopens it.
- A service panel: its frame survives a resize of the hub's dock, keeps its page,
  follows the hub's theme, and after its window closed stays out of the dock, over
  a reload too, until its sidebar row opens it (R4). A reload of the hub with the
  window open brings it back docked (R7).
- An agent's arrangement naming a detached panel is refused by name and changes
  nothing.
- The window's bar is the tile's own: the panel's contributed search and menu work
  there and its overflow ladder follows the window's width, a right-click opens
  there without the dock's verbs, Reattach is top right, and a reload of the window
  keeps all of it on the same page (R10).
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager, suppress

import pytest
from docs.screenshots.contact_sheet import hermetic_hub

from tests.interfaces.web_terminal.test_panels_browser import (
    _contributing_panel,
    _live_server,
    _open_palette,
    _open_second_tile,
    _palette_input,
    _palette_live_server,
    _palette_row,
    _search_menu_panel_backend,
    _service_tab,
)

pytestmark = [pytest.mark.browser, pytest.mark.slow]

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if not _PLAYWRIGHT_AVAILABLE:  # pragma: no cover
    pytest.skip("playwright not installed", allow_module_level=True)

_TIMEOUT_MS = 20_000
_COMPOSER = '#operator-container textarea[aria-label="Message the agent"]'
_AGENT_TILE = '.tile-tab[data-panel-id="terminal"]'
_ARTIFACTS_TILE = '.tile-tab[data-panel-id="iframe:artifacts"]'
_ARTIFACTS_FRAME = 'iframe.panel-iframe[data-panel-id="artifacts"]'
_BAR = ".attached-window-root > .tile-tab"
_BAR_REATTACH = f"{_BAR} button.tile-tab-reattach"
_MENU = ".rail-context-menu"

#: The dock as it stands: every tile bar's panel id, and the stored layouts.
_DOCK_JS = """() => ({
  tiles: [...document.querySelectorAll('.tile-tab[data-panel-id]')]
    .map((el) => el.dataset.panelId).sort(),
  stored: Object.keys(localStorage).filter((k) => k.startsWith('osprey-dock-layout-'))
    .sort().map((k) => localStorage.getItem(k)),
})"""


@pytest.fixture
def hub():
    with hermetic_hub(e2e=True) as running:
        yield running


@pytest.fixture
def page(chromium_browser):
    ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
    yield ctx.new_page()
    ctx.close()


def _boot(page: Page, hub) -> None:
    page.goto(hub.base_url, wait_until="domcontentloaded")
    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(f"{_AGENT_TILE} .tile-tab-window")).to_be_visible(timeout=_TIMEOUT_MS)


def _boot_with_artifacts(page: Page, hub) -> None:
    _boot(page, hub)
    expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(_ARTIFACTS_FRAME)).to_be_visible(timeout=_TIMEOUT_MS)


def _dock(page: Page) -> dict:
    """The dock once its debounced layout write (150 ms) has had time to land."""
    page.wait_for_timeout(700)
    return page.evaluate(_DOCK_JS)


def _tiles(page: Page) -> list[str]:
    return _dock(page)["tiles"]


def _send(box, prompt: str) -> None:
    """Send *prompt*; a composer still binding keeps the text, so Enter is
    pressed until the composer takes it."""
    box.fill(prompt)
    for _ in range(20):
        box.press("Enter")
        with suppress(AssertionError):
            expect(box).to_have_value("", timeout=500)
            return
    raise AssertionError(f"the composer never sent {prompt!r}")


def _detach_agent(page: Page) -> Page:
    with page.expect_popup() as info:
        page.locator(f"{_AGENT_TILE} .tile-tab-window").click()
    popup = info.value
    expect(popup.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    return popup


def _detach_artifacts(page: Page) -> Page:
    with page.expect_popup() as info:
        page.locator(f"{_ARTIFACTS_TILE} .tile-tab-window").click()
    popup = info.value
    expect(popup.locator(_ARTIFACTS_FRAME)).to_be_visible(timeout=_TIMEOUT_MS)
    return popup


def _spy_on_window_focus(page: Page) -> None:
    """Count every focus the hub asks of the window it opens: patched on the open."""
    page.evaluate(
        """() => {
          window.__focused = 0;
          const realOpen = window.open;
          window.open = (...args) => {
            const win = realOpen.apply(window, args);
            const proxy = new Proxy(win, {
              get(target, name) {
                if (name === 'focus') return () => { window.__focused += 1; target.focus(); };
                const value = Reflect.get(target, name);
                return typeof value === 'function' ? value.bind(target) : value;
              },
            });
            return proxy;
          };
        }"""
    )


def test_the_agent_pane_leaves_the_dock_chats_from_the_window_and_reattaches(hub, page, tmp_path):
    _boot(page, hub)
    popup = _detach_agent(page)

    # The window is on the hub's own shell page, not about:blank, and its bar
    # names the pane above the chat, which fills the window.
    assert popup.url == f"{hub.base_url.rstrip('/')}/static/attached-window.html"
    expect(popup.locator(f"{_BAR} .op-session-bar .op-session-led")).to_be_visible()
    expect(popup.locator(_BAR_REATTACH)).to_have_text("Reattach")
    # The mirrored stylesheets load on their own time, so the box settles once they have.
    popup.wait_for_function(
        """() => {
          const bar = document
            .querySelector('.attached-window-root > .tile-tab')
            .getBoundingClientRect();
          const pane = document
            .querySelector('.attached-window-root > .terminal-panel')
            .getBoundingClientRect();
          return bar.top === 0 && pane.left === 0 && pane.top === bar.bottom;
        }""",
        timeout=_TIMEOUT_MS,
    )
    popup.screenshot(path=str(tmp_path / "agent-window.png"))

    # R2: no hollow shell. The tile is gone from the dock.
    expect(page.locator(_COMPOSER)).to_have_count(0)
    expect(page.locator(_AGENT_TILE)).to_have_count(0)
    expect(page.locator("html")).to_have_attribute("data-agent", "detached")
    assert popup.evaluate("document.documentElement.dataset.theme") == page.evaluate(
        "document.documentElement.dataset.theme"
    )
    assert popup.evaluate("document.styleSheets.length") > 0

    # Typed in the window, the message goes through the hub's chat client and the
    # scripted child's reply streams back into the window.
    _send(popup.locator(_COMPOSER), "Check the beam from the window")
    expect(popup.locator("#operator-container")).to_contain_text(
        "Done: Check the beam from the window", timeout=_TIMEOUT_MS
    )

    with popup.expect_event("close", timeout=_TIMEOUT_MS):
        popup.locator(_BAR_REATTACH).click()

    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(_AGENT_TILE)).to_have_count(1)
    expect(page.locator("html")).not_to_have_attribute("data-agent", "detached")
    expect(page.locator("#operator-container")).to_contain_text(
        "Done: Check the beam from the window"
    )


def test_closing_the_agent_window_closes_the_pane_and_its_row_reopens_it(hub, page):
    _boot(page, hub)
    popup = _detach_agent(page)
    _send(popup.locator(_COMPOSER), "Check the beam from the window")
    expect(popup.locator("#operator-container")).to_contain_text(
        "Done: Check the beam from the window", timeout=_TIMEOUT_MS
    )

    popup.close()

    expect(page.locator("html")).to_have_attribute("data-agent", "closed", timeout=_TIMEOUT_MS)
    assert "terminal" not in _tiles(page)

    page.locator("#sidebar-agent-row").click()

    expect(page.locator(_AGENT_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator("#operator-container")).to_contain_text(
        "Done: Check the beam from the window"
    )


def test_a_session_click_with_the_agent_detached_focuses_the_window_and_docks_nothing(hub, page):
    _boot(page, hub)
    # A session to click on: the one the composer starts.
    _send(page.locator(_COMPOSER), "Start a session")
    expect(page.locator("#operator-container")).to_contain_text(
        "Done: Start a session", timeout=_TIMEOUT_MS
    )
    row = page.locator("[data-session-tree] .osprey-tree-title").first
    expect(row).to_be_visible(timeout=_TIMEOUT_MS)
    _spy_on_window_focus(page)
    popup = _detach_agent(page)
    before = _tiles(page)

    row.click()

    page.wait_for_function("() => window.__focused >= 1", timeout=_TIMEOUT_MS)
    assert not popup.is_closed()
    assert "terminal" not in before
    assert _tiles(page) == before
    expect(page.locator("html")).to_have_attribute("data-agent", "detached")


def test_reset_layout_with_the_agent_detached_keeps_it_out_of_the_dock(hub, page):
    _boot_with_artifacts(page, hub)
    popup = _detach_agent(page)

    # The page's own module instance: it holds the dock.
    assert page.evaluate(
        """async () => {
          const { getDockApi, resetDockLayout } = await import('/static/js/dock-workspace.js');
          resetDockLayout();
          return getDockApi() !== null;
        }"""
    )

    page.wait_for_timeout(500)
    assert "terminal" not in _tiles(page)
    expect(page.locator(_COMPOSER)).to_have_count(0)
    expect(popup.locator(_COMPOSER)).to_be_visible()
    expect(page.locator("html")).to_have_attribute("data-agent", "detached")


def test_a_hub_reload_closes_the_agent_window_and_docks_the_pane(hub, page):
    _boot(page, hub)
    popup = _detach_agent(page)

    page.reload(wait_until="domcontentloaded")

    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    page.wait_for_timeout(500)
    assert popup.is_closed()
    expect(page.locator(_AGENT_TILE)).to_have_count(1)


def test_a_service_panel_leaves_with_its_frame_and_survives_a_dock_resize(hub, page):
    _boot_with_artifacts(page, hub)
    popup = _detach_artifacts(page)

    # The hub's own frame element is in the window, at a real size, and the tile
    # has gone with it.
    expect(popup.frame_locator(_ARTIFACTS_FRAME).locator("body")).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(_ARTIFACTS_FRAME)).to_have_count(0)
    expect(page.locator(_ARTIFACTS_TILE)).to_have_count(0)
    box = popup.locator(_ARTIFACTS_FRAME).bounding_box()
    assert box is not None and box["width"] > 100 and box["height"] > 100
    assert popup.locator(".attached-window-root").get_attribute("data-osprey-detached") == "true"

    # Every geometry pass the hub runs skips it.
    page.set_viewport_size({"width": 1100, "height": 700})
    page.wait_for_timeout(500)
    expect(popup.locator(_ARTIFACTS_FRAME)).to_be_visible()
    after = popup.locator(_ARTIFACTS_FRAME).bounding_box()
    assert after == box


def test_a_detached_frame_follows_the_theme(hub, page):
    _boot_with_artifacts(page, hub)
    popup = _detach_artifacts(page)
    expect(popup.frame_locator(_ARTIFACTS_FRAME).locator("body")).to_be_visible(timeout=_TIMEOUT_MS)

    theme = page.evaluate(
        """async () => {
          const { setTheme, getTheme } = await import('/design-system/js/theme-manager.js');
          const next = getTheme() === 'dark' ? 'light' : 'dark';
          setTheme(next);
          return next;
        }"""
    )
    popup.wait_for_function(
        """([sel, theme]) => document.querySelector(sel).contentDocument
            ?.documentElement.getAttribute('data-theme') === theme""",
        arg=[_ARTIFACTS_FRAME, theme],
        timeout=_TIMEOUT_MS,
    )


def test_ariel_keeps_its_entry_and_after_its_window_closes_only_its_row_opens_it(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    tile = '.tile-tab[data-panel-id="iframe:ariel"]'
    frame = 'iframe.panel-iframe[data-panel-id="ariel"]'
    row = 'button.panel-rail-button[data-panel-id="ariel"]'
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(row)).not_to_have_class(
            re.compile(r"\bdisabled\b"), timeout=_TIMEOUT_MS
        )
        # Artifacts keeps a tile of its own, so the empty-slot landing has no say below.
        page.locator(row).click(button="right")
        page.get_by_role("menuitem", name="Open in a new tile").click()
        expect(page.locator(tile)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1)
        expect(page.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
        # On its own page first: a hash set on the blank page a frame starts on is lost.
        page.wait_for_function(
            """(sel) => { const w = document.querySelector(sel).contentWindow;
                return w.location.pathname.startsWith('/panel/ariel')
                  && w.document.readyState === 'complete'; }""",
            arg=frame,
            timeout=_TIMEOUT_MS,
        )
        page.locator(frame).evaluate("(el) => { el.contentWindow.location.hash = '#entry?id=X'; }")

        with page.expect_popup() as info:
            page.locator(f"{tile} .tile-tab-window").click()
        popup = info.value

        # R1: the same frame, in the window, on the entry it showed.
        expect(popup.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
        assert "#entry?id=X" in (popup.locator(frame).get_attribute("src") or "")
        popup.wait_for_function(
            "(sel) => document.querySelector(sel).contentWindow.location.hash === '#entry?id=X'",
            arg=frame,
            timeout=_TIMEOUT_MS,
        )
        expect(page.locator(tile)).to_have_count(0)

        popup.close()

        # R4: closed, not docked again, its frame out of sight; a reload changes nothing.
        page.wait_for_function(
            """(sel) => { const el = document.querySelector(sel);
                return !!el && el.style.display === 'none'; }""",
            arg=frame,
            timeout=_TIMEOUT_MS,
        )
        assert "iframe:ariel" not in _tiles(page)
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(row)).not_to_have_class(
            re.compile(r"\bdisabled\b"), timeout=_TIMEOUT_MS
        )
        page.wait_for_timeout(1500)
        assert "iframe:ariel" not in _tiles(page)

        page.locator(row).click()
        expect(page.locator(tile)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
        ctx.close()


def test_the_hub_navigating_a_detached_frame_that_navigates_in_place_changes_its_page(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    frame = 'iframe.panel-iframe[data-panel-id="ariel"]'
    row = 'button.panel-rail-button[data-panel-id="ariel"]'
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        expect(page.locator(row)).not_to_have_class(
            re.compile(r"\bdisabled\b"), timeout=_TIMEOUT_MS
        )
        page.locator(row).click(button="right")
        with page.expect_popup() as info:
            page.get_by_role("menuitem", name="Open in new window").click()
        popup = info.value
        expect(popup.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
        # Its document says it navigates in place, as ARIEL's router does; a router
        # takes the message only from its parent, which is no longer the hub.
        popup.wait_for_function(
            "(sel) => document.querySelector(sel).contentWindow.location.pathname"
            ".startsWith('/panel/ariel')",
            arg=frame,
            timeout=_TIMEOUT_MS,
        )
        popup.locator(frame).evaluate(
            """(el) => el.contentWindow.eval(
              "parent.postMessage({type: 'osprey-navigate-ready'}, parent.location.origin)"
            )"""
        )
        page.wait_for_timeout(300)

        # The path a session switch takes to a detached panel.
        page.evaluate(
            """async () => {
              const { restorePanelPage } = await import('/static/js/panel-lifecycle.js');
              restorePanelPage('ariel', '/panel/ariel#entry?id=Y');
            }"""
        )

        popup.wait_for_function(
            "(sel) => document.querySelector(sel).contentWindow.location.hash === '#entry?id=Y'",
            arg=frame,
            timeout=_TIMEOUT_MS,
        )
        ctx.close()


def test_a_panel_with_no_tile_opens_in_a_window_from_its_row_and_reattaches(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    tile = '.tile-tab[data-panel-id="iframe:ariel"]'
    frame = 'iframe.panel-iframe[data-panel-id="ariel"]'
    row = 'button.panel-rail-button[data-panel-id="ariel"]'
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(row)).not_to_have_class(
            re.compile(r"\bdisabled\b"), timeout=_TIMEOUT_MS
        )
        expect(page.locator(tile)).to_have_count(0)

        page.locator(row).click(button="right")
        with page.expect_popup() as info:
            page.get_by_role("menuitem", name="Open in new window").click()
        popup = info.value

        expect(popup.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
        box = popup.locator(frame).bounding_box()
        assert box is not None and box["width"] > 100 and box["height"] > 100
        expect(page.locator(tile)).to_have_count(0)
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1)

        with popup.expect_event("close", timeout=_TIMEOUT_MS):
            popup.locator(_BAR_REATTACH).click()

        expect(page.locator(tile)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
        ctx.close()


def test_a_hub_reload_docks_a_panel_that_left_with_no_tile_at_no_panels_cost(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    tile = '.tile-tab[data-panel-id="iframe:ariel"]'
    row = 'button.panel-rail-button[data-panel-id="ariel"]'
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(row)).not_to_have_class(
            re.compile(r"\bdisabled\b"), timeout=_TIMEOUT_MS
        )
        page.locator(row).click(button="right")
        with page.expect_popup() as info:
            page.get_by_role("menuitem", name="Open in new window").click()
        popup = info.value
        expect(popup.locator('iframe.panel-iframe[data-panel-id="ariel"]')).to_be_visible(
            timeout=_TIMEOUT_MS
        )

        page.reload(wait_until="domcontentloaded")

        expect(page.locator(tile)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        assert popup.is_closed()
        ctx.close()


def test_the_closed_agent_pane_opens_in_a_window_from_its_row_and_reattaches(hub, page):
    _boot(page, hub)
    popup = _detach_agent(page)
    popup.close()
    expect(page.locator("html")).to_have_attribute("data-agent", "closed", timeout=_TIMEOUT_MS)

    page.locator("#sidebar-agent-row").click(button="right")
    with page.expect_popup() as info:
        page.get_by_role("menuitem", name="Open in new window").click()
    popup = info.value

    expect(popup.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator("html")).to_have_attribute("data-agent", "detached")
    with popup.expect_event("close", timeout=_TIMEOUT_MS):
        popup.locator(_BAR_REATTACH).click()
    expect(page.locator(_AGENT_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)


def test_a_hub_reload_brings_a_detached_service_panel_back_docked(hub, page):
    _boot_with_artifacts(page, hub)
    before = _tiles(page)
    popup = _detach_artifacts(page)
    assert "iframe:artifacts" not in _tiles(page)

    page.reload(wait_until="domcontentloaded")

    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(_ARTIFACTS_FRAME)).to_be_visible(timeout=_TIMEOUT_MS)
    assert popup.is_closed()
    assert _tiles(page) == before


def test_an_agent_arrangement_naming_a_detached_panel_is_refused_and_changes_nothing(hub, page):
    _boot_with_artifacts(page, hub)
    popup = _detach_artifacts(page)
    before = _tiles(page)
    # The server has heard the pop-out once its report lands.
    page.wait_for_function(
        """async () => (await (await fetch('/api/panels')).json()).popouts?.includes('artifacts')""",
        timeout=_TIMEOUT_MS,
    )

    status, body = page.evaluate(
        """async () => {
          const r = await fetch('/api/panel-op/open-set', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              request_id: 'detached-test',
              members: ['artifacts'],
              as: 'side',
            }),
          });
          return [r.status, await r.text()];
        }"""
    )

    assert status == 409, body
    assert json.loads(body)["reason"] == "popped_out"
    assert _tiles(page) == before
    assert not popup.is_closed()


@contextmanager
def _panel_with_controls_in_a_window(tmp_path, chromium_browser):
    """A panel contributing a search and a menu, detached, its window at 900 wide."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with _search_menu_panel_backend() as stub_url:
        custom = _contributing_panel("search-demo", "SEARCHY", stub_url)
        with _live_server(workspace, enabled_panels={"artifacts"}, custom_panels=[custom]) as (
            base_url,
            _app,
        ):
            ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
            page = ctx.new_page()
            page.goto(base_url, wait_until="domcontentloaded")
            expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
            _open_second_tile(page, base_url, "search-demo", "SEARCHY")
            tab = _service_tab(page, "SEARCHY")
            expect(tab.locator(".contrib-search-input")).to_be_visible(timeout=_TIMEOUT_MS)
            with page.expect_popup() as info:
                tab.locator(".tile-tab-window").click()
            popup = info.value
            expect(popup.locator(f"{_BAR} .contrib-search-input")).to_be_visible(
                timeout=_TIMEOUT_MS
            )
            popup.set_viewport_size({"width": 900, "height": 600})
            try:
                yield page, popup
            finally:
                ctx.close()


def test_a_detached_panel_shows_its_search_and_the_ladder_follows_the_window(
    tmp_path, chromium_browser
):
    with _panel_with_controls_in_a_window(tmp_path, chromium_browser) as (_page, popup):
        expect(popup.locator(f"{_BAR} .tile-tab-title")).to_have_text("Searchy")
        search = popup.locator(f"{_BAR} .contrib-search")
        expect(search).not_to_have_class(re.compile(r"contrib-search-collapsed"))

        # The observer that hears this is the window's: the hub's never fires here.
        popup.set_viewport_size({"width": 240, "height": 600})
        expect(search).to_have_class(re.compile(r"contrib-search-collapsed"), timeout=5_000)

        popup.set_viewport_size({"width": 900, "height": 600})
        expect(search).not_to_have_class(re.compile(r"contrib-search-collapsed"), timeout=5_000)


def test_a_contributed_menu_opens_in_the_window_and_closes_on_a_click_into_the_panel(
    tmp_path, chromium_browser
):
    with _panel_with_controls_in_a_window(tmp_path, chromium_browser) as (page, popup):
        popover = popup.locator(".contrib-menu-popover")
        popup.locator(f"{_BAR} .contrib-menu-btn").click()
        expect(popover).to_be_visible()
        expect(popover.locator(".contrib-menu-item")).to_have_count(2)
        expect(page.locator(".contrib-menu-popover")).to_have_count(0)

        frame = popup.locator('iframe.panel-iframe[data-panel-id="search-demo"]').bounding_box()
        popup.mouse.click(frame["x"] + frame["width"] / 2, frame["y"] + frame["height"] / 2)
        expect(popover).to_have_count(0, timeout=5_000)

        # A press on the bar itself: the window's own node, of the window's realm.
        popup.locator(f"{_BAR} .contrib-menu-btn").click()
        expect(popover).to_be_visible()
        popup.locator(f"{_BAR} .tile-tab-title").click()
        expect(popover).to_have_count(0, timeout=5_000)


def test_a_right_click_on_the_window_bar_opens_there_without_the_docks_verbs(
    tmp_path, chromium_browser
):
    with _panel_with_controls_in_a_window(tmp_path, chromium_browser) as (page, popup):
        popup.locator(f"{_BAR} .tile-tab-title").click(button="right")
        menu = popup.locator(_MENU)
        expect(menu).to_be_visible(timeout=5_000)
        expect(page.locator(_MENU)).to_have_count(0)
        labels = menu.locator(".rail-context-label").all_inner_texts()
        for verb in ("Focus", "Open in a new tile", "Move to", "Reset layout"):
            assert not any(verb in label for label in labels), labels
        assert labels, "the menu has rows"

        popup.keyboard.press("Escape")
        expect(menu).to_have_count(0, timeout=5_000)


def test_a_right_click_on_the_agent_windows_bar_has_no_layout_rows(hub, page):
    _boot(page, hub)
    popup = _detach_agent(page)
    popup.locator(f"{_BAR} .op-session-led").click(button="right")
    menu = popup.locator(_MENU)
    expect(menu).to_be_visible(timeout=5_000)
    labels = menu.locator(".rail-context-label").all_inner_texts()
    assert "Export conversation" in labels, labels
    assert "Reset layout" not in labels, labels
    assert "Close Agent pane" not in labels, labels


def test_reattach_is_top_right_and_the_window_has_no_dock_buttons(hub, page):
    _boot_with_artifacts(page, hub)
    popups = [_detach_agent(page), _detach_artifacts(page)]
    for popup in popups:
        popup.set_viewport_size({"width": 800, "height": 600})
        box = popup.locator(_BAR_REATTACH).bounding_box()
        assert box["y"] < 40, box
        assert box["x"] + box["width"] > 800 - 24, box
        expect(popup.locator(f"{_BAR} .tile-tab-window")).to_have_count(0)
        expect(popup.locator(f"{_BAR} .tile-tab-close")).to_have_count(0)


def test_reloading_a_service_panels_window_keeps_it_detached_on_its_page_with_its_bar(
    tmp_path, chromium_browser
):
    with _panel_with_controls_in_a_window(tmp_path, chromium_browser) as (page, popup):
        frame = 'iframe.panel-iframe[data-panel-id="search-demo"]'
        # The bar shows the cached contribution before the moved frame has loaded,
        # and a hash set on a frame still loading is lost with its first document.
        popup.wait_for_function(
            """(sel) => { const w = document.querySelector(sel).contentWindow;
                return w.location.pathname.startsWith('/panel/search-demo')
                  && w.document.readyState === 'complete'; }""",
            arg=frame,
            polling=100,
            timeout=_TIMEOUT_MS,
        )
        popup.evaluate(
            "(sel) => { document.querySelector(sel).contentWindow.location.hash = '#page-two'; }",
            frame,
        )
        popup.reload(wait_until="domcontentloaded")

        expect(popup.locator(f"{_BAR} .tile-tab-title")).to_have_text(
            "Searchy", timeout=_TIMEOUT_MS
        )
        expect(popup.locator(f"{_BAR} .contrib-search-input")).to_be_visible(timeout=_TIMEOUT_MS)
        expect(popup.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
        popup.wait_for_function(
            "(sel) => document.querySelector(sel).contentWindow.location.hash === '#page-two'",
            arg=frame,
            polling=100,
            timeout=_TIMEOUT_MS,
        )
        assert not any("search-demo" in tile for tile in _tiles(page))

        # What was bound to the old document is bound to this one.
        popup.locator(f"{_BAR} .contrib-menu-btn").click()
        expect(popup.locator(".contrib-menu-popover")).to_be_visible()
        popup.keyboard.press("Escape")
        popup.locator(f"{_BAR} .tile-tab-title").click(button="right")
        expect(popup.locator(_MENU)).to_be_visible(timeout=5_000)
        popup.keyboard.press("Escape")

        popup.locator(_BAR_REATTACH).click()
        expect(_service_tab(page, "SEARCHY")).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(_service_tab(page, "SEARCHY").locator(".contrib-search-input")).to_be_visible(
            timeout=_TIMEOUT_MS
        )


def test_reloading_the_agent_window_keeps_it_detached_with_its_bar_and_reattaches(hub, page):
    _boot(page, hub)
    popup = _detach_agent(page)
    # A switcher left open over the reload must not swallow the next click.
    popup.locator(f"{_BAR} .op-session-label").click()
    expect(popup.locator("#session-switcher-menu")).to_be_visible(timeout=5_000)
    popup.reload(wait_until="domcontentloaded")

    expect(popup.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(popup.locator(f"{_BAR} .op-session-bar .op-session-led")).to_be_visible(
        timeout=_TIMEOUT_MS
    )
    expect(popup.locator(_BAR_REATTACH)).to_be_visible()
    assert "terminal" not in _tiles(page)

    # The console and its bar kept their listeners over the reload: a discarded
    # document takes every listener on its nodes with it.
    popup.locator(f"{_BAR} .op-session-led").click(button="right")
    expect(popup.locator(_MENU)).to_be_visible(timeout=5_000)
    popup.keyboard.press("Escape")
    popup.locator(f"{_BAR} .op-session-label").click()
    expect(popup.locator("#session-switcher-menu")).to_be_visible(timeout=5_000)
    popup.keyboard.press("Escape")
    expect(popup.locator("#session-switcher-menu")).to_be_hidden(timeout=5_000)
    _send(popup.locator(_COMPOSER), "after the reload")
    expect(popup.locator(".op-messages")).to_contain_text("after the reload", timeout=_TIMEOUT_MS)

    popup.locator(_BAR_REATTACH).click()
    expect(page.locator(_AGENT_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(f"{_AGENT_TILE} .op-session-bar .op-session-led")).to_be_visible()
    expect(page.locator(_COMPOSER)).to_be_visible()


def test_the_console_bar_comes_home_when_the_agent_window_closes(hub, page):
    _boot(page, hub)
    popup = _detach_agent(page)
    expect(popup.locator(f"{_BAR} .op-session-new")).to_be_visible()
    expect(page.locator(".op-session-bar")).to_have_count(0)

    popup.close()
    expect(page.locator("html")).to_have_attribute("data-agent", "closed", timeout=_TIMEOUT_MS)

    # Reopened, the tile adopts the same bar, buttons and all.
    page.evaluate("() => document.querySelector('#sidebar-agent-row')?.click()")
    expect(page.locator(f"{_AGENT_TILE} .op-session-new")).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(".op-session-bar")).to_have_count(1)


_ARIEL_TILE = '.tile-tab[data-panel-id="iframe:ariel"]'
_ARIEL_FRAME = 'iframe.panel-iframe[data-panel-id="ariel"]'
_ARIEL_ROW = 'button.panel-rail-button[data-panel-id="ariel"]'
_TRAIL = "[data-detached-trail]"


def _boot_ariel_beside_artifacts(page: Page, base_url: str) -> None:
    """ARIEL docked in a tile of its own, right of the Artifacts tile."""
    page.goto(base_url, wait_until="domcontentloaded")
    expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(_ARIEL_ROW)).not_to_have_class(
        re.compile(r"\bdisabled\b"), timeout=_TIMEOUT_MS
    )
    page.locator(_ARIEL_ROW).click(button="right")
    page.get_by_role("menuitem", name="Open in a new tile").click()
    expect(page.locator(_ARIEL_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(_ARIEL_FRAME)).to_be_visible(timeout=_TIMEOUT_MS)


def _detach_ariel(page: Page) -> Page:
    with page.expect_popup() as info:
        page.locator(f"{_ARIEL_TILE} .tile-tab-window").click()
    popup = info.value
    expect(popup.locator(_ARIEL_FRAME)).to_be_visible(timeout=_TIMEOUT_MS)
    return popup


def _beside(page: Page, left: str, right: str) -> bool:
    """Whether the tile bar *right* starts where the tile bar *left* ends."""
    first, second = page.locator(left).bounding_box(), page.locator(right).bounding_box()
    return (
        abs(second["y"] - first["y"]) < 2
        and first["x"] < second["x"]
        and second["x"] - (first["x"] + first["width"]) < 16
    )


def test_a_panels_row_focuses_its_window_and_its_icon_docks_it_beside_a_reminted_neighbour(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        _boot_ariel_beside_artifacts(page, base_url)
        assert _beside(page, _ARTIFACTS_TILE, _ARIEL_TILE)
        _spy_on_window_focus(page)
        popup = _detach_ariel(page)
        expect(page.locator(_ARIEL_ROW)).to_have_class(re.compile(r"\bdetached\b"))
        expect(page.locator(f"{_ARIEL_ROW} {_TRAIL}")).to_be_visible()
        expect(page.locator(_ARIEL_ROW)).to_have_attribute(
            "aria-description", "In its own window — click to bring it to the front"
        )

        # One row says what the window can do, both ways.
        page.locator(_ARIEL_ROW).click(button="right")
        labels = page.locator(f"{_MENU} .rail-context-label").all_inner_texts()
        assert "Reattach" in labels, labels
        assert "Open in new window" not in labels, labels
        page.keyboard.press("Escape")

        # A click on the row brings the window to the front, and moves nothing.
        before = _tiles(page)
        page.locator(_ARIEL_ROW).click()
        page.wait_for_function("() => window.__focused >= 1", timeout=_TIMEOUT_MS)
        assert _tiles(page) == before
        assert not popup.is_closed()

        # The neighbour's tile goes and comes back, under a new tile id.
        page.locator(f"{_ARTIFACTS_TILE} .tile-tab-close").click()
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(0, timeout=_TIMEOUT_MS)
        # Off the rail here, so the path an agent's focus takes brings it back.
        page.evaluate(
            """async () => {
              const { activateTab } = await import('/static/js/panel-manager.js');
              activateTab('artifacts', { userInitiated: true });
            }"""
        )
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)

        # A click on the icon reattaches, beside the neighbour by its key.
        with popup.expect_event("close", timeout=_TIMEOUT_MS):
            page.locator(f"{_ARIEL_ROW} {_TRAIL}").click()
        expect(page.locator(_ARIEL_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(_ARIEL_FRAME)).to_be_visible(timeout=_TIMEOUT_MS)
        expect(page.locator(f"{_ARIEL_ROW} {_TRAIL}")).to_have_count(0)
        expect(page.locator(_ARIEL_ROW)).not_to_have_class(re.compile(r"\bdetached\b"))
        assert _beside(page, _ARTIFACTS_TILE, _ARIEL_TILE)
        ctx.close()


def test_the_sessions_row_focuses_the_agent_window_and_its_icon_reattaches(hub, page):
    _boot(page, hub)
    _spy_on_window_focus(page)
    popup = _detach_agent(page)
    row = page.locator("#sidebar-agent-row")
    expect(row).to_have_class(re.compile(r"\bdetached\b"))
    # The row's own icon hides in the expanded sidebar; this one is a different element.
    expect(row.locator(_TRAIL)).to_be_visible()
    expect(row).to_have_attribute(
        "data-tip", "Sessions · Agent pane in its own window — click to bring it to the front"
    )

    before = _tiles(page)
    row.click()
    page.wait_for_function("() => window.__focused >= 1", timeout=_TIMEOUT_MS)
    assert _tiles(page) == before
    assert not popup.is_closed()

    row.click(button="right")
    labels = page.locator(f"{_MENU} .rail-context-label").all_inner_texts()
    assert "Reattach" in labels, labels
    assert "Close Agent pane" in labels, labels
    assert "Open in new window" not in labels, labels
    page.keyboard.press("Escape")

    with popup.expect_event("close", timeout=_TIMEOUT_MS):
        row.locator(_TRAIL).click()
    expect(page.locator(_AGENT_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(row.locator(_TRAIL)).to_have_count(0)
    expect(row).not_to_have_class(re.compile(r"\bdetached\b"))
    expect(row).to_have_attribute("data-tip", "Sessions")
    assert row.get_attribute("aria-description") is None


def test_the_tile_button_and_the_palette_say_open_in_new_window_and_reattach(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        _boot_ariel_beside_artifacts(page, base_url)
        button = page.locator(f"{_ARIEL_TILE} .tile-tab-window")
        expect(button).to_have_attribute("aria-label", "Open in new window")

        popup = _detach_ariel(page)
        _open_palette(page)
        _palette_input(page).fill("reattach logbook")
        row = _palette_row(page, "Reattach Logbook")
        expect(row).to_be_visible(timeout=_TIMEOUT_MS)
        with popup.expect_event("close", timeout=_TIMEOUT_MS):
            row.click()
        expect(page.locator(_ARIEL_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        ctx.close()


def test_a_chat_message_and_an_approval_both_work_from_the_agent_window(hub, page):
    hub.chat.asks["Please carry out the plan"] = ("ExitPlanMode", {"plan": "Scan the beam."})
    _boot(page, hub)
    popup = _detach_agent(page)

    _send(popup.locator(_COMPOSER), "Check the beam from the window")
    expect(popup.locator("#operator-container")).to_contain_text(
        "Done: Check the beam from the window", timeout=_TIMEOUT_MS
    )

    # The card comes up in the window, and its answer reaches the turn waiting on it.
    _send(popup.locator(_COMPOSER), "Please carry out the plan")
    approve = popup.locator(".op-card-actions button", has_text="Approve plan")
    expect(approve).to_be_visible(timeout=_TIMEOUT_MS)
    approve.click()
    expect(popup.locator("#operator-container")).to_contain_text(
        "The operator said PermissionResultAllow", timeout=_TIMEOUT_MS
    )
    assert "terminal" not in _tiles(page)


def _boot_custom_panel_row(page: Page, base_url: str, row: str) -> None:
    page.goto(base_url, wait_until="domcontentloaded")
    expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    expect(page.locator(row)).not_to_have_class(re.compile(r"\bdisabled\b"), timeout=_TIMEOUT_MS)


@contextmanager
def _a_never_loaded_panel(tmp_path, chromium_browser):
    """A first-party panel with no frame yet, and the hub's question about it held back."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    row = 'button.panel-rail-button[data-panel-id="never-loaded"]'
    with _search_menu_panel_backend() as stub_url:
        custom = _contributing_panel("never-loaded", "NEVER", stub_url)
        with _live_server(workspace, enabled_panels={"artifacts"}, custom_panels=[custom]) as (
            base_url,
            _app,
        ):
            ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
            page = ctx.new_page()
            held = []
            page.route("**/api/panels/never-loaded/cap*", lambda route: held.append(route))
            _boot_custom_panel_row(page, base_url, row)
            page.locator(row).click(button="right")
            item = page.get_by_role("menuitem", name="Open in new window")
            expect(item).not_to_have_attribute("aria-disabled", "true")
            with page.expect_popup() as info:
                item.click()
            popup = info.value
            # The window is open inside the gesture; the frame waits in the hub.
            page.wait_for_timeout(300)
            assert not popup.is_closed()
            assert len(held) == 1
            expect(popup.locator("iframe.panel-iframe")).to_have_count(0)
            try:
                yield page, popup, held[0], row
            finally:
                ctx.close()


def test_a_first_click_on_a_never_loaded_panel_opens_the_window_and_the_frame_follows(
    tmp_path, chromium_browser
):
    with _a_never_loaded_panel(tmp_path, chromium_browser) as (page, popup, cap, row):
        cap.fulfill(status=404)

        frame = 'iframe.panel-iframe[data-panel-id="never-loaded"]'
        expect(popup.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
        popup.wait_for_function(
            """(sel) => { const w = document.querySelector(sel).contentWindow;
                return w.location.pathname.startsWith('/panel/never-loaded')
                  && w.document.readyState === 'complete'; }""",
            arg=frame,
            polling=100,
            timeout=_TIMEOUT_MS,
        )
        expect(page.locator(row)).to_have_class(re.compile(r"\bdetached\b"))
        expect(popup.locator(_BAR_REATTACH)).to_be_visible()


def test_a_first_click_on_a_panel_that_resolves_opaque_closes_the_window_cleanly(
    tmp_path, chromium_browser
):
    with _a_never_loaded_panel(tmp_path, chromium_browser) as (page, popup, cap, row):
        before = _tiles(page)
        with popup.expect_event("close", timeout=_TIMEOUT_MS):
            cap.fulfill(status=200, json={"cap_url": None})

        expect(page.locator(".undo-notice-text")).to_have_text(
            "This panel runs at an opaque origin and cannot open in a new window.",
            timeout=_TIMEOUT_MS,
        )
        expect(page.locator(row)).not_to_have_class(re.compile(r"\bdetached\b"))
        assert _tiles(page) == before


def test_remove_from_rail_in_the_windows_menu_closes_the_window_and_the_row_goes(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        _boot_ariel_beside_artifacts(page, base_url)
        popup = _detach_ariel(page)

        popup.locator(f"{_BAR} .tile-tab-title").click(button="right")
        with popup.expect_event("close", timeout=_TIMEOUT_MS):
            popup.locator(_MENU).get_by_role("menuitem", name="Remove from rail").click()

        expect(page.locator(_ARIEL_ROW)).to_have_count(0, timeout=_TIMEOUT_MS)
        expect(page.locator(_ARIEL_TILE)).to_have_count(0)
        # The window is gone from the pane's account too: a reload keeps it closed.
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        page.wait_for_timeout(1000)
        expect(page.locator(_ARIEL_TILE)).to_have_count(0)
        ctx.close()


#: The pages the server holds for the session in use's layout; {} until it holds one.
_SAVED_PAGES_JS = """async () => {
  const rows = await (await fetch('/api/sessions')).json();
  for (const r of rows.sessions ?? rows.rows ?? rows ?? []) {
    if (!r.transcript_id) continue;
    const saved = await (await fetch(`/api/sessions/${r.transcript_id}/layout`)).json();
    if (saved?.layout?.pages) return saved.layout.pages;
  }
  return {};
}"""


def _saved_page(page: Page, panel_id: str, part: str) -> None:
    """Wait until the session's saved layout has *panel_id* on a page containing *part*."""
    page.wait_for_function(
        f"""async () => ((await ({_SAVED_PAGES_JS})())[{json.dumps(panel_id)}] ?? '')
            .includes({json.dumps(part)})""",
        polling=250,
        timeout=_TIMEOUT_MS,
    )


def _settled_after_reload(page: Page) -> None:
    """The hub reloaded, and session-layout's arrive plus SETTLE_MS (400 ms) have passed."""
    page.reload(wait_until="domcontentloaded")
    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    page.wait_for_timeout(1200)


def _x_order(page: Page) -> list[str]:
    """The tiles' panel ids, left to right."""
    return page.evaluate(
        """() => [...document.querySelectorAll('.tile-tab[data-panel-id]')]
          .map((el) => [el.getBoundingClientRect().x, el.dataset.panelId])
          .sort((a, b) => a[0] - b[0]).map(([, id]) => id)"""
    )


def test_r7_the_agent_comes_back_docked_beside_its_neighbour(hub, page):
    _boot_with_artifacts(page, hub)
    order = _x_order(page)
    popup = _detach_agent(page)
    _send(popup.locator(_COMPOSER), "Check the beam from the window")
    expect(popup.locator("#operator-container")).to_contain_text(
        "Done: Check the beam from the window", timeout=_TIMEOUT_MS
    )
    page.wait_for_timeout(500)

    _settled_after_reload(page)

    assert popup.is_closed()
    expect(page.locator(_AGENT_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
    assert _x_order(page) == order
    expect(page.locator("#operator-container")).to_contain_text(
        "Done: Check the beam from the window", timeout=_TIMEOUT_MS
    )


def test_r7_two_panes_detached_in_neighbour_order_both_return_in_place(tmp_path, chromium_browser):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        _boot_ariel_beside_artifacts(page, base_url)
        order = _x_order(page)
        first, second = _detach_ariel(page), _detach_artifacts(page)
        assert _tiles(page) == ["terminal"]
        page.wait_for_timeout(500)

        _settled_after_reload(page)

        assert first.is_closed()
        assert second.is_closed()
        expect(page.locator(_ARIEL_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(1, timeout=_TIMEOUT_MS)
        assert _x_order(page) == order
        ctx.close()


def test_r7_with_the_neighbour_closed_every_other_docked_panel_stays_docked(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        _boot_ariel_beside_artifacts(page, base_url)
        popup = _detach_ariel(page)
        page.locator(f"{_ARTIFACTS_TILE} .tile-tab-close").click()
        expect(page.locator(_ARTIFACTS_TILE)).to_have_count(0, timeout=_TIMEOUT_MS)
        page.wait_for_timeout(500)

        _settled_after_reload(page)

        assert popup.is_closed()
        tiles = _tiles(page)
        assert "terminal" in tiles, tiles
        assert "iframe:ariel" in tiles, tiles
        ctx.close()
