"""Browser tests: the sidebar's resize handle in a real layout engine.

The unit suite (``sidebar-resize.test.mjs``) pins the clamp, storage, snap and
keys against happy-dom, which lays nothing out. What only a browser can show
is that the grid column follows the handle and that a reload restores the
width before the first paint.

Coverage:

  (1) a mouse drag on the handle widens ``#sidebar`` to the dragged width, and
      a reload comes back at that width with no transition from the default.
  (2) a drag well below the minimum collapses the sidebar to the 56 px icon
      rail, and a drag back out restores the dragged width.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_sidebar_resize_browser.py -v

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

#: Sub-pixel slack for a width comparison.
_TOLERANCE_PX = 1.0

_VIEWPORT = {"width": 1280, "height": 800}


@contextmanager
def _hub_server(workspace_dir: Path) -> Iterator[str]:
    """Launch a real web-terminal hub with just the artifacts panel enabled.

    Yields:
        The hub's base URL.
    """
    shared_root = patch(
        "osprey_connectors.workspace.resolve_shared_data_root", return_value=workspace_dir
    )
    with hub_server(workspace_dir, shared_root) as (base_url, _app):
        yield base_url


def _open(browser: Browser, url: str) -> Page:
    """Open the hub at 1280×800 and wait for the sidebar to be wired."""
    page = browser.new_page(viewport=_VIEWPORT)
    page.goto(url, wait_until="domcontentloaded")
    expect(page.locator("#sidebar-resize[aria-valuetext]")).to_be_attached(timeout=10_000)
    return page


def _sidebar_width(page: Page) -> float:
    box = page.locator("#sidebar").bounding_box()
    assert box is not None, "#sidebar has no layout box"
    return box["width"]


def _drag_handle(page: Page, x: float, steps: int = 4) -> None:
    """Press on the handle's centre, move to ``x`` in ``steps`` moves, and release."""
    box = page.locator("#sidebar-resize").bounding_box()
    assert box is not None, "#sidebar-resize has no layout box"
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + box["width"] / 2, y)
    page.mouse.down()
    page.mouse.move(x, y, steps=steps)
    page.mouse.up()


def _stored_width(page: Page) -> int:
    return int(page.evaluate("localStorage.getItem('osprey-sidebar-width')"))


def test_drag_sets_the_width_and_a_reload_keeps_it(tmp_path, chromium_browser):
    """A drag to x=320 widens the sidebar by the drag, before and after a reload."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open(chromium_browser, base_url)
        try:
            assert abs(_sidebar_width(page) - 232) <= _TOLERANCE_PX
            _drag_handle(page, 320)
            dragged = _stored_width(page)
            # The handle's centre sits a few px inside the sidebar's edge.
            assert abs(dragged - 320) <= 6, dragged
            assert abs(_sidebar_width(page) - dragged) <= _TOLERANCE_PX

            page.reload(wait_until="domcontentloaded")
            # The pre-paint restore: the property is on <html> before any module runs.
            assert (
                page.evaluate(
                    "document.documentElement.style.getPropertyValue('--sidebar-user-width')"
                )
                == f"{round(dragged)}px"
            )
            expect(page.locator("#sidebar-resize[aria-valuetext]")).to_be_attached()
            assert abs(_sidebar_width(page) - dragged) <= _TOLERANCE_PX
        finally:
            page.close()


def test_drag_past_the_minimum_collapses_and_back_out_expands(tmp_path, chromium_browser):
    """Dragging to x=60 shows icons only; dragging back out restores the stored width."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open(chromium_browser, base_url)
        try:
            _drag_handle(page, 300)
            stored = _stored_width(page)

            _drag_handle(page, 60)
            expect(page.locator("html")).to_have_attribute("data-sidebar", "collapsed")
            # No transition while dragging: the rail is already 56 px.
            assert abs(_sidebar_width(page) - 56) <= _TOLERANCE_PX
            assert _stored_width(page) == stored

            # One move, so the drag ends on the snap rather than carrying on past it.
            _drag_handle(page, 160, steps=1)
            expect(page.locator("html")).to_have_attribute("data-sidebar", "expanded")
            assert abs(_sidebar_width(page) - stored) <= _TOLERANCE_PX
        finally:
            page.close()
