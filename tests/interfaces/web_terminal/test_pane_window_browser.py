"""Browser e2e: a pane in a window of its own, on the hermetic hub.

The tile header's Pop out moves a pane's live content into a window the hub
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
"""

from __future__ import annotations

import json
import re
from contextlib import suppress

import pytest
from docs.screenshots.contact_sheet import hermetic_hub

from tests.interfaces.web_terminal.test_panels_browser import _palette_live_server

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
_BAR_TITLE = ".attached-window-bar .attached-window-title"
_BAR_REATTACH = ".attached-window-bar button.attached-window-reattach"

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
    expect(page.locator(f"{_AGENT_TILE} .tile-tab-popout")).to_be_visible(timeout=_TIMEOUT_MS)


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
        page.locator(f"{_AGENT_TILE} .tile-tab-popout").click()
    popup = info.value
    expect(popup.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    return popup


def _detach_artifacts(page: Page) -> Page:
    with page.expect_popup() as info:
        page.locator(f"{_ARTIFACTS_TILE} .tile-tab-popout").click()
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
    expect(popup.locator(_BAR_TITLE)).to_have_text("Agent")
    expect(popup.locator(_BAR_REATTACH)).to_have_text("Reattach")
    # The mirrored stylesheets load on their own time, so the box settles once they have.
    popup.wait_for_function(
        """() => {
          const bar = document.querySelector('.attached-window-bar').getBoundingClientRect();
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
            page.locator(f"{tile} .tile-tab-popout").click()
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
