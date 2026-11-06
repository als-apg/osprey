"""Browser tests: the sidebar group headings in a real layout engine.

The unit suite (``sidebar-groups.test.mjs``) pins the menu's rows and what
they do against happy-dom, which lays nothing out. What only a browser can
show is that the heading answers a right-click across the full width of the
sidebar row, not only over the letters of its name, and that a real
Shift+F10 on the focused name opens the same menu.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_sidebar_group_headings_browser.py -v

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

_TOOLS_HEAD = '#panel-rail > [data-group="tools"] > .sidebar-group-head'
_MENU = ".rail-context-menu"


@contextmanager
def _page(browser: Browser, workspace_dir: Path) -> Iterator[Page]:
    """A real web-terminal hub over an empty project, opened at 1280×800."""
    shared_root = patch(
        "osprey_connectors.workspace.resolve_shared_data_root", return_value=workspace_dir
    )
    with hub_server(workspace_dir, shared_root) as (base_url, _app):
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        try:
            page.goto(base_url, wait_until="domcontentloaded")
            expect(page.locator(_TOOLS_HEAD)).to_be_visible(timeout=10_000)
            yield page
        finally:
            ctx.close()


def test_right_click_anywhere_across_the_heading_opens_the_group_menu(tmp_path, chromium_browser):
    """The heading spans the rail row; a right-click at its far end, past the name, opens the menu."""
    with _page(chromium_browser, tmp_path) as page:
        head = page.locator(_TOOLS_HEAD)
        assert head.locator("button").count() == 0, "the ⋯ button is gone"
        widths = page.evaluate(
            """(sel) => {
              const head = document.querySelector(sel);
              const rail = document.getElementById('panel-rail');
              const name = head.querySelector('.sidebar-group-name');
              const range = document.createRange();
              range.selectNodeContents(name);
              return {
                head: head.getBoundingClientRect().width,
                rail: rail.clientWidth,
                text: range.getBoundingClientRect().width,
              };
            }""",
            _TOOLS_HEAD,
        )
        assert widths["head"] >= widths["rail"] - 1
        assert widths["text"] < widths["head"] / 2

        box = head.bounding_box()
        assert box is not None
        # Near the right end of the row, well past the letters of "Tools" (the
        # sidebar's resize handle covers its last few pixels).
        head.click(button="right", position={"x": box["width"] - 16, "y": box["height"] / 2})
        menu = page.locator(_MENU)
        expect(menu).to_be_visible()
        expect(menu).to_have_attribute("aria-label", "Tools group")
        expect(menu.locator('[role="menuitem"]')).to_have_text(
            ["Rename", "New group below", "Move up", "Move down"]
        )


def test_shift_f10_on_the_focused_name_opens_the_menu_and_rename_works(tmp_path, chromium_browser):
    """From the keyboard: focus the name, Shift+F10, Enter on Rename opens the field."""
    with _page(chromium_browser, tmp_path) as page:
        name = page.locator(f"{_TOOLS_HEAD} .sidebar-group-name")
        name.focus()
        page.keyboard.press("Shift+F10")
        menu = page.locator(_MENU)
        expect(menu).to_be_visible()
        expect(menu).to_have_attribute("aria-label", "Tools group")

        page.keyboard.press("Enter")  # the first row, Rename, has focus
        expect(menu).to_have_count(0)
        field = page.locator(f"{_TOOLS_HEAD} .sidebar-group-rename")
        expect(field).to_be_focused()
        field.fill("Instruments")
        page.keyboard.press("Enter")
        expect(name).to_have_text("Instruments")
        expect(name).to_be_focused()
