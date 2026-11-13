"""Browser tests: the sidebar group headings in a real layout engine.

The unit suite (``sidebar-groups.test.mjs``) pins the menu's rows and what
they do against happy-dom, which lays nothing out. What only a browser can
show is that the heading answers a right-click across the full width of the
sidebar row, not only over the letters of its name, that a real Shift+F10 on
the focused name opens the same menu, and that the Sessions group holding the
Agent row lays out with no heading at all.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_sidebar_group_headings_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.interfaces._browser import wait_for_page_ready
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
_SESSIONS = '#panel-rail > [data-group="workspace"]'
_MENU = ".rail-context-menu"
_GROUPS_KEY = "osprey-sidebar-groups"


@contextmanager
def _page(browser: Browser, workspace_dir: Path, stored: dict | None = None) -> Iterator[Page]:
    """A real web-terminal hub over an empty project, opened at 1280×800, with
    ``stored`` seeded as the sidebar groups when given, handed over once the hub
    is ready: the rail's groups are drawn during the panel manager's start-up,
    and a heading can show before the last of them is."""
    shared_root = patch(
        "osprey_connectors.workspace.resolve_shared_data_root", return_value=workspace_dir
    )
    with hub_server(workspace_dir, shared_root) as (base_url, _app):
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        if stored is not None:
            value = json.dumps(json.dumps(stored))
            page.add_init_script(
                f"try {{ localStorage.setItem({json.dumps(_GROUPS_KEY)}, {value}); }} catch (e) {{}}"
            )
        try:
            page.goto(base_url, wait_until="domcontentloaded")
            wait_for_page_ready(page)
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
        # Near the right end of the row, well past the letters of "Apps" (the
        # sidebar's resize handle covers its last few pixels).
        head.click(button="right", position={"x": box["width"] - 16, "y": box["height"] / 2})
        menu = page.locator(_MENU)
        expect(menu).to_be_visible()
        expect(menu).to_have_attribute("aria-label", "Apps group")
        expect(menu.locator('[role="menuitem"]')).to_have_text(
            [
                "Rename…",
                "New group below",
                "Move up",
                "Move down",
                "Collapse group",
                "Delete group",
            ]
        )


def test_shift_f10_on_the_focused_name_opens_the_menu_and_rename_works(tmp_path, chromium_browser):
    """From the keyboard: focus the name, Shift+F10, Enter on Rename opens the field."""
    with _page(chromium_browser, tmp_path) as page:
        name = page.locator(f"{_TOOLS_HEAD} .sidebar-group-name")
        name.focus()
        page.keyboard.press("Shift+F10")
        menu = page.locator(_MENU)
        expect(menu).to_be_visible()
        expect(menu).to_have_attribute("aria-label", "Apps group")

        page.keyboard.press("Enter")  # the first row, Rename, has focus
        expect(menu).to_have_count(0)
        field = page.locator(f"{_TOOLS_HEAD} .sidebar-group-rename")
        expect(field).to_be_focused()
        field.fill("Instruments")
        page.keyboard.press("Enter")
        expect(name).to_have_text("Instruments")
        expect(name).to_be_focused()


def test_the_agent_row_leads_the_sidebar_with_no_heading(tmp_path, chromium_browser):
    """The Sessions group shows no heading: the Agent row sits at the top of the
    group, the divider still sits below it, and the group keeps its accessible name."""
    with _page(chromium_browser, tmp_path) as page:
        expect(page.locator(f"{_SESSIONS} > .sidebar-group-head")).to_be_hidden()
        expect(page.get_by_role("listitem", name="Sessions", exact=True)).to_have_count(1)
        layout = page.evaluate(
            """(sel) => {
              const group = document.querySelector(sel);
              const agent = document.getElementById('sidebar-agent-row');
              const divider = document.querySelector('#panel-rail > .sidebar-divider');
              return {
                gap: agent.getBoundingClientRect().top - group.getBoundingClientRect().top,
                dividerAfter: divider?.previousElementSibling === group,
              };
            }""",
            _SESSIONS,
        )
        # Only the group's own padding sits above the Agent row: no 24 px heading.
        assert layout["gap"] < 24
        assert layout["dividerAfter"]


def test_a_stored_layout_that_renamed_or_lowered_workspace_still_leads_headless(
    tmp_path, chromium_browser
):
    """A layout stored with Workspace renamed and moved down still puts the Agent
    row's group first, with no heading, and keeps the operator's other names."""
    stored = {
        "version": 3,
        "groups": [
            {"id": "tools", "label": "Instruments"},
            {"id": "workspace", "label": "Operations"},
            {"id": "system", "label": "System"},
        ],
        "placement": {},
    }
    with _page(chromium_browser, tmp_path, stored=stored) as page:
        groups = page.eval_on_selector_all(
            "#panel-rail > .sidebar-group[data-group]", "els => els.map(e => e.dataset.group)"
        )
        assert groups == ["workspace", "tools", "system"]
        expect(page.locator(f"{_SESSIONS} > .sidebar-group-head")).to_be_hidden()
        expect(page.locator(f"{_SESSIONS} #sidebar-agent-row")).to_be_attached()
        expect(page.locator(f"{_TOOLS_HEAD} .sidebar-group-name")).to_have_text("Instruments")


_GROUPS = "#panel-rail > .sidebar-group[data-group]"
_GROUP_ORDER_JS = "els => els.map(e => e.dataset.group)"


def test_dragging_a_heading_moves_its_whole_group_and_survives_reload(tmp_path, chromium_browser):
    """A real mouse drag of the System heading onto the top of Apps moves the
    whole System group, rows and all, above Apps, and a reload draws that order
    again; Sessions stays first throughout."""
    system = '#panel-rail > [data-group="system"]'
    with _page(chromium_browser, tmp_path) as page:
        assert page.eval_on_selector_all(_GROUPS, _GROUP_ORDER_JS) == [
            "workspace",
            "tools",
            "system",
        ]
        system_rows = page.locator(f"{system} [data-panel-id]")
        rows_before = system_rows.count()

        page.locator(f"{system} > .sidebar-group-head").drag_to(
            page.locator(_TOOLS_HEAD), target_position={"x": 20, "y": 2}
        )

        expect(page.locator(_GROUPS).nth(1)).to_have_attribute("data-group", "system")
        assert page.eval_on_selector_all(_GROUPS, _GROUP_ORDER_JS) == [
            "workspace",
            "system",
            "tools",
        ]
        assert system_rows.count() == rows_before
        expect(page.locator("#panel-rail .drop-before, #panel-rail .drop-after")).to_have_count(0)
        expect(page.locator("#panel-rail > .sidebar-group.dragging")).to_have_count(0)

        page.reload(wait_until="domcontentloaded")
        wait_for_page_ready(page)
        assert page.eval_on_selector_all(_GROUPS, _GROUP_ORDER_JS) == [
            "workspace",
            "system",
            "tools",
        ]
