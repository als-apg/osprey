"""Browser e2e: the Agent pane popped out into an attached window on the hermetic hub.

The tile header's Pop out moves the Agent pane's live subtree into a blank window
the hub drives (attached-popout.js, attached-window.js). Pins: the window holds
the chat and the tile a stand-in; a message typed in the window goes through the
hub's own chat client (the scripted chat child's reply comes back into it); the
layout never changes; closing the window puts the pane back in its tile, with
the conversation; a reload of the hub closes the window and shows the pane in
its tile.
"""

from __future__ import annotations

from contextlib import suppress

import pytest
from docs.screenshots.contact_sheet import hermetic_hub

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
_STAND_IN = ".dock-panel-host > .attached-standin"

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


def _dock(page: Page) -> dict:
    """The dock once its debounced layout write (150 ms) has had time to land."""
    page.wait_for_timeout(700)
    return page.evaluate(_DOCK_JS)


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


def _pop_out(page: Page) -> Page:
    with page.expect_popup() as info:
        page.locator(f"{_AGENT_TILE} .tile-tab-popout").click()
    popup = info.value
    expect(popup.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    return popup


def test_the_agent_pane_pops_out_chats_from_the_window_and_comes_back(hub, page):
    _boot(page, hub)
    before = _dock(page)

    popup = _pop_out(page)

    # The window holds the chat, styled and themed like the hub; the tile stays.
    expect(page.locator(_COMPOSER)).to_have_count(0)
    expect(page.locator(_STAND_IN)).to_contain_text("Agent is in its own window.")
    expect(page.locator("html")).to_have_attribute("data-agent", "popped")
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
    # The layout never saw the pop-out.
    assert _dock(page) == before

    popup.close()

    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(_STAND_IN)).to_have_count(0)
    expect(page.locator("html")).not_to_have_attribute("data-agent", "popped")
    expect(page.locator(f"{_AGENT_TILE}")).to_have_count(1)
    expect(page.locator("#operator-container")).to_contain_text(
        "Done: Check the beam from the window"
    )
    assert _dock(page) == before


def test_a_hub_reload_closes_the_window_and_the_pane_is_in_its_tile(hub, page):
    _boot(page, hub)
    popup = _pop_out(page)

    page.reload(wait_until="domcontentloaded")

    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    page.wait_for_timeout(500)
    assert popup.is_closed()
    expect(page.locator(_STAND_IN)).to_have_count(0)
    expect(page.locator(_AGENT_TILE)).to_have_count(1)


def test_a_service_panel_pops_out_with_its_frame_and_comes_back_to_its_tile(hub, page):
    _boot(page, hub)
    tile = page.locator('.tile-tab[data-panel-id="iframe:artifacts"]')
    expect(tile).to_have_count(1, timeout=_TIMEOUT_MS)
    frame = 'iframe.panel-iframe[data-panel-id="artifacts"]'
    expect(page.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
    before = _dock(page)

    with page.expect_popup() as info:
        tile.locator(".tile-tab-popout").click()
    popup = info.value

    # The hub's own frame element is in the window and loads there; the tile
    # keeps its place and shows the stand-in.
    expect(popup.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(popup.frame_locator(frame).locator("body")).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(frame)).to_have_count(0)
    expect(page.locator(".dock-iframe-overlay > .attached-standin")).to_contain_text(
        "is in its own window."
    )
    assert _dock(page) == before

    page.locator(".dock-iframe-overlay > .attached-standin .attached-standin-back").click()

    assert popup.is_closed() or popup.wait_for_event("close", timeout=_TIMEOUT_MS)
    expect(page.locator(frame)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(".attached-standin")).to_have_count(0)
    assert _dock(page) == before
