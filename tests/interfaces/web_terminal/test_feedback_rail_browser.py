"""Browser tests: where the sidebar's footer (Docs, Feedback, ☰ More) lands.

The footer sits at the bottom of the sidebar, below the Tools group, and never
scrolls off: only the Workspace section gives up height. That is a geometry
fact no unit test can see, because only a real layout engine resolves the flex
column.

Coverage:

  (1) at 1280×700 the footer is inside the sidebar and the window, below the
      ``+ Add panel`` row that closes the Tools group.
  (2) each footer label fits its row, rather than being quietly ellipsised.
  (3) the ☰ More menu opens beside the sidebar, stays in the window, and closes
      on Escape with focus back on its button.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_feedback_rail_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.interfaces._hub import hub_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: Sub-pixel slack for a "these two edges coincide" comparison. Chromium
#: reports fractional device-pixel geometry, so an exact equality on box edges
#: is a flake waiting to happen.
_EDGE_TOLERANCE_PX = 1.0

#: The smallest window the sidebar's height rule is written for.
_SMALL_VIEWPORT = {"width": 1280, "height": 700}


@contextmanager
def _hub_server(workspace_dir: Path) -> Iterator[str]:
    """Launch a real web-terminal hub with just the artifacts panel enabled.

    The same patch set every hub browser suite uses: the companion backends are
    bypassed and the artifacts panel reports the standard fallback URL, so the
    sidebar renders one real panel row without a backend process.

    Yields:
        The hub's base URL.
    """
    # The server-side stores live under the shared data root; pinning it to
    # this test's workspace keeps one suite's state out of the next.
    shared_root = patch(
        "osprey_connectors.workspace.resolve_shared_data_root", return_value=workspace_dir
    )
    with hub_server(workspace_dir, shared_root) as (base_url, _app):
        yield base_url


def _open_hub_page(browser: Browser, url: str) -> Page:
    """Open a hub URL at 1280×700 and wait for the panel rows AND the footer.

    The footer is static markup, but its position only settles once
    ``createRail()`` has filled the Workspace and Tools groups above it.
    """
    page = browser.new_page(viewport=_SMALL_VIEWPORT)
    page.goto(url, wait_until="domcontentloaded")
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    expect(page.locator("#panel-feedback-btn")).to_be_visible(timeout=10_000)
    return page


def _box(page: Page, selector: str) -> dict[str, float]:
    """The element's bounding box, with ``right``/``bottom`` filled in."""
    box = page.locator(selector).bounding_box()
    assert box is not None, f"{selector} has no layout box"
    return {**box, "right": box["x"] + box["width"], "bottom": box["y"] + box["height"]}


def test_footer_sits_below_tools_inside_the_window(tmp_path, chromium_browser):
    """The footer is the sidebar's last block and stays on screen at 1280×700."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            sidebar = _box(page, "#sidebar")
            foot = _box(page, ".sidebar-foot")
            add = _box(page, "#panel-add-btn")

            assert foot["y"] >= add["bottom"] - _EDGE_TOLERANCE_PX, (
                "the footer is interleaved with the Tools group"
            )
            assert foot["bottom"] <= sidebar["bottom"] + _EDGE_TOLERANCE_PX, (foot, sidebar)
            assert foot["bottom"] <= _SMALL_VIEWPORT["height"] + _EDGE_TOLERANCE_PX, (
                "the footer scrolled off the bottom of the window"
            )
            expect(page.locator("#panel-feedback-btn")).to_be_in_viewport()
            expect(page.locator("#more-btn")).to_be_in_viewport()
        finally:
            page.close()


def test_footer_labels_fit_their_rows_uncut(tmp_path, chromium_browser):
    """The expanded footer's labels are fully readable, not ellipsised away.

    ``text-overflow: ellipsis`` means a label that outgrows its row degrades
    silently rather than overflowing visibly — so the failure this catches is
    one that no layout exception and no unit test reports.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            overflow = page.evaluate(
                "() => Array.from("
                "  document.querySelectorAll('.sidebar-foot .osprey-nav-label')"
                ").filter((el) => el.offsetParent !== null)"
                " .filter((el) => el.scrollWidth > el.clientWidth + 1)"
                " .map((el) => el.textContent)"
            )
            assert overflow == [], f"footer labels are cut off: {overflow}"
        finally:
            page.close()


def _menu_fits_viewport(page: Page) -> None:
    """The open ☰ More menu lies wholly inside the window."""
    menu = _box(page, "#more-menu")
    size = page.viewport_size
    assert size is not None
    assert menu["x"] >= -_EDGE_TOLERANCE_PX, menu
    assert menu["y"] >= -_EDGE_TOLERANCE_PX, menu
    assert menu["right"] <= size["width"] + _EDGE_TOLERANCE_PX, menu
    assert menu["bottom"] <= size["height"] + _EDGE_TOLERANCE_PX, menu


def test_more_menu_opens_beside_the_sidebar(tmp_path, chromium_browser):
    """The menu opens right of the sidebar, inside the window, and Escape closes it."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            page.locator("#more-btn").click()
            expect(page.locator("#more-menu")).to_be_visible()
            expect(page.locator("#more-btn")).to_have_attribute("aria-expanded", "true")

            menu = _box(page, "#more-menu")
            sidebar = _box(page, "#sidebar")
            assert menu["x"] >= sidebar["right"] - _EDGE_TOLERANCE_PX, (menu, sidebar)
            _menu_fits_viewport(page)

            page.keyboard.press("Escape")
            expect(page.locator("#more-menu")).to_be_hidden()
            expect(page.locator("#more-btn")).to_be_focused()
        finally:
            page.close()
