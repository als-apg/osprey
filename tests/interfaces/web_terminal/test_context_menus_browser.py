"""Browser e2e: the hub's context menus at every place, by pointer and by keyboard.

Each place offers its OSPREY menu on a right-click and on the menu key /
Shift+F10 with focus on it, the menu taking focus on its first row and Escape
handing it back. Pins:

- A tile bar's ⋯ and a right-click on the bar offer the same rows.
- A pop-out's bar carries the window rows (Reattach, Zoom, Reload, Close window) and
  none of the dock's; Close window closes the window and the panel keeps its
  sidebar row, with no tile coming back.
- A right-click on the terminal body opens the Terminal menu.
- A chat message's menu opens in the popped-out Agent window, in that window's
  document, and still after the window reloads.
- Shift+right-click on a shell surface, and a right-click in the composer or in
  a sidebar rename field, leave the browser's own menu: no OSPREY menu opens
  and the event's default is not prevented.
- The menu key opens the menu of every in-scope place: rail entry (failed one
  too), group heading, Sessions row, session and artifact rows, pinned heading,
  sidebar header, service / Agent / pop-out tile bars, chat message, chat
  background, terminal and the empty dock slot. (The Channel Finder's
  background is pinned in channel_finder/test_graph_finder_browser.py.)
- Reveal in sidebar expands a collapsed sidebar.
- A submenu opens, closes one level and runs from the keyboard.

Run: pytest tests/interfaces/web_terminal/test_context_menus_browser.py -m browser
"""

from __future__ import annotations

import re
from contextlib import suppress
from typing import TYPE_CHECKING

import pytest
from docs.screenshots.contact_sheet import SEEDED_SESSIONS, hermetic_hub

from tests.interfaces._browser import rail_entry, wait_for_page_ready
from tests.interfaces.web_terminal._chat_server import live_chat_server
from tests.interfaces.web_terminal._surface import open_terminal
from tests.interfaces.web_terminal.test_artifact_sessions_browser import (
    _KIND_MODE_JS,
    _execute,
    _open_all,
    _open_group,
    _open_session,
    _run_row,
    _writers,
)
from tests.interfaces.web_terminal.test_dock_tiles_browser import _bar, _hub
from tests.interfaces.web_terminal.test_dock_tiles_browser import _open as _open_dock_hub
from tests.interfaces.web_terminal.test_pane_window_browser import (
    _ARIEL_ROW,
    _ARIEL_TILE,
    _boot_ariel_beside_artifacts,
    _detach_ariel,
)
from tests.interfaces.web_terminal.test_panels_browser import _palette_live_server

if TYPE_CHECKING:
    from playwright.sync_api import Locator

pytestmark = [pytest.mark.browser, pytest.mark.slow]

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if not _PLAYWRIGHT_AVAILABLE:  # pragma: no cover
    pytest.skip("playwright not installed", allow_module_level=True)

_TIMEOUT_MS = 20_000
#: The hub's menus: the rail-context skin (panel-context-menu.js) and the design
#: system's default skin, which the place menus (chat, terminal, dock) open with.
_MENU = ".rail-context-menu, .osprey-context-menu"
_RAIL_MENU = ".rail-context-menu"
_COMPOSER = '#operator-container textarea[aria-label="Message the agent"]'
_AGENT_TILE = '.tile-tab[data-panel-id="terminal"]'
_POP_BAR = ".attached-window-root > .tile-tab"
_TREE = "#sidebar-session-tree"

#: Records, per contextmenu event reaching the window last, whether a handler
#: prevented its default: the browser shows its own menu only when none did.
_RECORD_PREVENTED_JS = """() => {
  window.__prevented = [];
  window.addEventListener('contextmenu', (e) => {
    window.__prevented.push(e.defaultPrevented);
    // The browser's own menu would block the page; nothing else here needs it.
    e.preventDefault();
  });
}"""


# --- helpers ------------------------------------------------------------------


def _labels(menu: Locator) -> list[str]:
    labels = menu.locator(".rail-context-label, .osprey-context-label")
    return [t.strip() for t in labels.all_inner_texts()]


def _focus(target: Locator) -> Locator:
    """Focus *target*, or its first focusable descendant, as Tab would. Returns a
    locator of what took the focus.

    Resolved by locator, never through a handle: a tree redraw replaces a row by
    an equal one, and a handle taken before it (as ``Locator.evaluate`` takes
    one, then runs) focuses the detached old row, which takes no focus."""
    own = target.evaluate("(el) => el.matches('button, a[href], input, textarea, [tabindex]')")
    focused = target if own else target.locator("button, a[href], [tabindex]").first
    focused.focus(timeout=5_000)
    expect(focused, "the place takes no focus").to_be_focused(timeout=5_000)
    return focused


def _menu_key(page: Page, target: Locator, *, aria: str | None = None) -> list[str]:
    """Focus *target*, press Shift+F10: one menu opens with focus on a row, and
    Escape closes it and gives focus back. Returns the menu's row labels."""
    focused = _focus(target)
    page.keyboard.press("Shift+F10")
    menu = page.locator(_MENU)
    expect(menu).to_have_count(1, timeout=5_000)
    if aria is not None:
        expect(menu).to_have_attribute("aria-label", aria)
    # The menu takes the focus once it is drawn: waited on, not read at once.
    page.wait_for_function(
        "(sel) => document.activeElement?.closest(sel) != null", arg=_MENU, timeout=5_000
    )
    labels = _labels(menu)
    page.keyboard.press("Escape")
    expect(menu).to_have_count(0, timeout=5_000)
    # Focus comes back to the place the menu opened from, found again here
    # rather than held from before: a tree redraw while the menu is open
    # replaces a row by an equal one, and the focus goes to the new row.
    expect(focused).to_be_focused(timeout=5_000)
    return labels


def _right_click_menu(page: Page, target: Locator, **click) -> Locator:
    target.click(button="right", **click)
    menu = page.locator(_MENU)
    expect(menu).to_have_count(1, timeout=5_000)
    return menu


def _open_service(page: Page, key: str) -> None:
    rail_entry(page, key, enabled_only=True).click()
    expect(_bar(page, key)).to_have_count(1, timeout=10_000)


def _nav_label(page: Page, key: str) -> str:
    return rail_entry(page, key).locator(".osprey-nav-label").inner_text().strip()


def _active_label(page: Page) -> str:
    return page.evaluate(
        "() => document.activeElement?.querySelector('.rail-context-label')?.textContent ?? ''"
    )


def _press_until_active(page: Page, label: str, key: str = "ArrowDown") -> None:
    for _ in range(20):
        if _active_label(page) == label:
            return
        page.keyboard.press(key)
    raise AssertionError(f"no row {label!r}; focus on {_active_label(page)!r}")


def _send(box: Locator, prompt: str) -> None:
    """Send *prompt*; a composer still binding keeps the text, so Enter is pressed
    until the composer takes it."""
    box.fill(prompt)
    for _ in range(20):
        box.press("Enter")
        with suppress(AssertionError):
            expect(box).to_have_value("", timeout=500)
            return
    raise AssertionError(f"the composer never sent {prompt!r}")


def _boot_hermetic(page: Page, hub) -> None:
    page.goto(hub.base_url, wait_until="domcontentloaded")
    wait_for_page_ready(page)
    expect(page.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(page.locator(f"{_AGENT_TILE} .tile-tab-more")).to_be_visible(timeout=_TIMEOUT_MS)


def _detach_agent(page: Page) -> Page:
    with page.expect_popup() as info:
        page.locator(f"{_AGENT_TILE} .tile-tab-window").click()
    popup = info.value
    wait_for_page_ready(popup)
    expect(popup.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    return popup


def _converse(page: Page, prompt: str) -> None:
    _send(page.locator(_COMPOSER), prompt)
    expect(page.locator("#operator-container")).to_contain_text(
        f"Done: {prompt}", timeout=_TIMEOUT_MS
    )


def _entry(page: Page, role: str) -> Locator:
    return page.locator(f".op-entry.{role}").last


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def hub():
    with hermetic_hub(e2e=True) as running:
        yield running


@pytest.fixture
def page(chromium_browser):
    ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
    yield ctx.new_page()
    ctx.close()


# --- tile bars ----------------------------------------------------------------


def test_a_tile_bars_more_button_and_a_right_click_on_the_bar_offer_the_same_rows(
    tmp_path, chromium_browser
):
    with _hub(tmp_path) as dock:
        page = _open_dock_hub(chromium_browser, dock)
        _open_service(page, "svc-a")
        bar = _bar(page, "svc-a")

        bar.locator(".tile-tab-more").click()
        from_more = _labels(page.locator(_MENU))
        page.keyboard.press("Escape")
        expect(page.locator(_MENU)).to_have_count(0, timeout=5_000)
        from_bar = _labels(_right_click_menu(page, bar.locator(".tile-tab-title")))
        page.keyboard.press("Escape")

        assert from_more == from_bar, (from_more, from_bar)
        assert from_bar == [
            "Open panel",
            "Maximize",
            "Zoom",
            "Open in new window",
            "Open in new session",
            "Reload",
            "Reveal in sidebar",
            "Close tile",
        ], from_bar

        agent = page.locator(_AGENT_TILE)
        agent.locator(".tile-tab-more").click()
        agent_more = _labels(page.locator(_MENU))
        page.keyboard.press("Escape")
        expect(page.locator(_MENU)).to_have_count(0, timeout=5_000)
        agent_bar = _labels(_right_click_menu(page, agent.locator(".agent-bar-spacer:visible")))
        page.keyboard.press("Escape")
        assert agent_more == agent_bar, (agent_more, agent_bar)
        assert "Close Agent pane" in agent_bar, agent_bar
        assert dock.errors == []


def test_a_pop_out_bar_has_the_window_rows_and_close_window_keeps_the_sidebar_row(
    tmp_path, chromium_browser
):
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with _palette_live_server(workspace) as (base_url, _app):
        ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        _boot_ariel_beside_artifacts(page, base_url)
        popup = _detach_ariel(page)
        bar = popup.locator(_POP_BAR)
        expect(bar).to_be_visible(timeout=_TIMEOUT_MS)

        menu = _right_click_menu(popup, bar.locator(".tile-tab-title"))
        labels = _labels(menu)
        expect(page.locator(_MENU)).to_have_count(0)
        # The window rows, Close window last; nothing that acts on a dock or
        # on the rail membership of a panel already on it.
        assert labels[labels.index("Reattach") :] in (
            ["Reattach", "Zoom", "Reload", "Close window"],
            ["Reattach", "Zoom", "Reload", "Help", "Close window"],
        ), labels
        for verb in (
            "Open panel",
            "Maximize",
            "Open in new session",
            "Reveal in sidebar",
            "Close tile",
            "Remove from rail",
        ):
            assert verb not in labels, labels
        popup.keyboard.press("Escape")
        expect(menu).to_have_count(0, timeout=5_000)
        bar.locator(".tile-tab-more").click()
        assert _labels(popup.locator(_MENU)) == labels

        with popup.expect_event("close", timeout=_TIMEOUT_MS):
            # The click closes this window, which can fail a click call still waiting on it.
            popup.locator(_MENU).get_by_role("menuitem", name="Close window").evaluate(
                "b => setTimeout(() => b.click())"
            )

        row = page.locator(_ARIEL_ROW)
        expect(row).to_have_count(1)
        expect(row).not_to_have_class(re.compile(r"\bdetached\b"), timeout=_TIMEOUT_MS)
        page.wait_for_timeout(700)
        expect(page.locator(_ARIEL_TILE)).to_have_count(0)
        ctx.close()


# --- terminal -----------------------------------------------------------------


def test_a_right_click_on_the_terminal_body_opens_the_terminal_menu(tmp_path, chromium_browser):
    with live_chat_server(tmp_path) as (base_url, _app):
        page = chromium_browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(base_url, wait_until="domcontentloaded")
        wait_for_page_ready(page)
        expect(page.locator(".op-session-bar")).to_be_visible(timeout=_TIMEOUT_MS)
        open_terminal(page)
        screen = page.locator("#terminal-container .xterm-screen")
        expect(screen).to_be_visible(timeout=_TIMEOUT_MS)

        menu = _right_click_menu(page, screen)
        expect(menu).to_have_attribute("aria-label", "Terminal")
        assert _labels(menu) == [
            "Copy",
            "Paste",
            "Select all",
            "Clear terminal",
            "Switch to chat",
            "Restart terminal",
        ], _labels(menu)
        page.keyboard.press("Escape")
        expect(menu).to_have_count(0, timeout=5_000)

        # The menu key from the terminal's own input.
        labels = _menu_key(page, page.locator("#terminal-container .xterm-helper-textarea"))
        assert labels[0] == "Copy", labels
        page.close()


# --- the popped-out Agent window ------------------------------------------------


def test_a_chat_message_menu_opens_in_the_popped_out_agent_window(hub, page):
    _boot_hermetic(page, hub)
    popup = _detach_agent(page)
    _converse(popup, "Check the beam from the window")

    menu = _right_click_menu(popup, _entry(popup, "assistant"), position={"x": 4, "y": 4})
    expect(menu).to_have_attribute("aria-label", "Agent message")
    assert _labels(menu) == ["Copy", "Quote in prompt"], _labels(menu)
    expect(page.locator(_MENU)).to_have_count(0)
    popup.keyboard.press("Escape")
    expect(menu).to_have_count(0, timeout=5_000)

    menu = _right_click_menu(popup, _entry(popup, "operator"), position={"x": 4, "y": 4})
    expect(menu).to_have_attribute("aria-label", "Your message")
    popup.locator(_MENU).get_by_role("menuitem", name="Quote in prompt").click()
    expect(popup.locator(_COMPOSER)).to_have_value(re.compile("Check the beam from the window"))


def test_a_pop_out_reload_keeps_the_chat_menus(hub, page):
    _boot_hermetic(page, hub)
    popup = _detach_agent(page)
    popup.reload(wait_until="domcontentloaded")
    wait_for_page_ready(popup)
    expect(popup.locator(_COMPOSER)).to_be_visible(timeout=_TIMEOUT_MS)
    expect(popup.locator(f"{_POP_BAR} button.tile-tab-reattach")).to_be_visible(timeout=_TIMEOUT_MS)
    _converse(popup, "after the reload")

    menu = _right_click_menu(popup, _entry(popup, "assistant"), position={"x": 4, "y": 4})
    expect(menu).to_have_attribute("aria-label", "Agent message")
    popup.keyboard.press("Escape")
    expect(menu).to_have_count(0, timeout=5_000)
    labels = _menu_key(popup, popup.locator(".op-messages"), aria="Conversation")
    assert labels == ["Scroll to latest", "Export conversation", "New session"], labels


# --- the browser's own menu -----------------------------------------------------


def test_shift_right_click_on_a_shell_surface_keeps_the_browser_menu(tmp_path, chromium_browser):
    with _hub(tmp_path) as dock:
        page = _open_dock_hub(chromium_browser, dock)
        _open_service(page, "svc-a")
        page.evaluate(_RECORD_PREVENTED_JS)
        targets = [
            _bar(page, "svc-a").locator(".tile-tab-title"),
            rail_entry(page, "svc-b"),
            page.locator('#panel-rail > [data-group="tools"] .sidebar-group-name'),
            page.locator("#sidebar-toggle"),
        ]
        for target in targets:
            target.click(button="right", modifiers=["Shift"])
            page.wait_for_timeout(200)
            expect(page.locator(_MENU)).to_have_count(0)
        assert page.evaluate("() => window.__prevented") == [False] * len(targets)
        # The contrast: without Shift the same bar opens its menu.
        _right_click_menu(page, targets[0])
        assert dock.errors == []


def test_a_right_click_in_the_composer_or_a_rename_field_opens_no_osprey_menu(hub, page):
    _boot_hermetic(page, hub)
    page.evaluate(_RECORD_PREVENTED_JS)

    page.locator(_COMPOSER).click(button="right")
    page.wait_for_timeout(200)
    expect(page.locator(_MENU)).to_have_count(0)

    # The heading's menu is wired by panel-rail.js's render, which also adds
    # the System group; index.html alone serves an unwired heading.
    expect(page.locator('#panel-rail > [data-group="system"]')).to_be_attached(timeout=_TIMEOUT_MS)
    head = page.locator('#panel-rail > [data-group="tools"] > .sidebar-group-head')
    head.locator(".sidebar-group-name").focus()
    page.keyboard.press("Shift+F10")
    expect(page.locator(_MENU)).to_have_count(1, timeout=5_000)
    page.keyboard.press("Enter")  # Rename… is the first row
    field = head.locator(".sidebar-group-rename")
    expect(field).to_be_focused(timeout=5_000)
    page.evaluate("() => { window.__prevented = []; }")
    field.click(button="right")
    page.wait_for_timeout(200)
    expect(page.locator(_MENU)).to_have_count(0)
    assert page.evaluate("() => window.__prevented") == [False]
    page.keyboard.press("Escape")


# --- the menu key at every place --------------------------------------------------


def test_the_menu_key_opens_the_rail_tile_and_dock_menus(tmp_path, chromium_browser):
    with _hub(tmp_path) as dock:
        page = _open_dock_hub(chromium_browser, dock)
        expect(rail_entry(page, "svc-a", enabled_only=True)).to_have_count(1, timeout=10_000)

        labels = _menu_key(page, rail_entry(page, "svc-a"))
        assert labels[0] == "Open", labels
        assert labels[labels.index("Open in new window") + 1] == "Open in new session", labels

        page.evaluate(
            """async () => {
              const { setEntryStatus } = await import('/static/js/rail-row-state.js');
              setEntryStatus(document.getElementById('panel-rail'), 'svc-b',
                { failed: true, message: 'The service did not start' });
            }"""
        )
        expect(rail_entry(page, "svc-b")).to_have_class(re.compile(r"\bfailed\b"))
        assert _menu_key(page, rail_entry(page, "svc-b")) == ["Retry start", "Remove from rail"]

        name = page.locator('#panel-rail > [data-group="tools"] .sidebar-group-name')
        assert _menu_key(page, name)[0] == "Rename…"

        _open_service(page, "svc-a")
        labels = _menu_key(page, _bar(page, "svc-a").locator(".tile-tab-more"))
        assert labels[-1] == "Close tile", labels
        # The Agent's bar is the console bar it adopts; its + New is in it.
        labels = _menu_key(page, page.locator(f"{_AGENT_TILE} .agent-bar-new"))
        assert "Close Agent pane" in labels, labels

        # Close every tile, one at a time, each close seen before the next: the
        # dock is one empty slot.
        tiles = page.locator("#dock-root .tile-tab")
        for left in range(tiles.count() - 1, -1, -1):
            tiles.first.locator('.tile-tab-actions button[aria-label^="Close"]').click()
            expect(tiles).to_have_count(left, timeout=5_000)
        labels = _menu_key(page, page.locator("#dock-root"), aria="Dock")
        assert labels == ["Open panel", "Reset layout"], labels
        assert dock.errors == []


def test_escape_gives_focus_to_the_rail_entry_a_redraw_drew_while_its_menu_was_open(
    tmp_path, chromium_browser
):
    with _hub(tmp_path) as dock:
        page = _open_dock_hub(chromium_browser, dock)
        entry = rail_entry(page, "svc-a", enabled_only=True)
        expect(entry).to_have_count(1, timeout=10_000)
        _focus(entry)
        page.evaluate("() => { window.__railEntryBefore = document.activeElement; }")
        page.keyboard.press("Shift+F10")
        menu = page.locator(_MENU)
        expect(menu).to_have_count(1, timeout=5_000)

        # The rail's full render builds every row anew while the menu is open.
        page.evaluate(
            """async () => {
              const { renderRail } = await import('/static/js/panel-lifecycle.js');
              renderRail();
            }"""
        )
        assert page.evaluate("() => !window.__railEntryBefore.isConnected"), (
            "the render kept the old row"
        )
        page.keyboard.press("Escape")
        expect(menu).to_have_count(0, timeout=5_000)
        expect(rail_entry(page, "svc-a")).to_be_focused(timeout=5_000)
        assert dock.errors == []


def test_the_menu_key_opens_the_sidebar_menus(hub, page):
    _boot_hermetic(page, hub)
    tree_session = page.locator(
        f'{_TREE} .osprey-tree-session[data-session-id="{SEEDED_SESSIONS[1].transcript_id}"]'
    )
    expect(tree_session).to_be_visible(timeout=_TIMEOUT_MS)

    labels = _menu_key(page, page.locator("#sidebar-agent-row"), aria="Sessions actions")
    assert labels[0] == "New session", labels
    assert "Open in new session" not in labels, labels

    labels = _menu_key(page, page.locator("#sidebar-toggle"), aria="Sidebar")
    assert "Collapse sidebar" in labels and "Reset layout" in labels, labels

    labels = _menu_key(page, tree_session)
    assert "Open in new session" not in labels, labels
    assert "Rename…" in labels, labels

    # An artifact row: the seeded session holding the beam profile, opened.
    sid = SEEDED_SESSIONS[2].transcript_id
    session = page.locator(f'{_TREE} .osprey-tree-session[data-session-id="{sid}"]')
    session.locator(".osprey-tree-title").click()
    expect(page.locator(f"{_TREE} .osprey-tree-session.active")).to_have_attribute(
        "data-session-id", sid, timeout=_TIMEOUT_MS
    )
    _open_all(page)
    first = page.locator(f"{_TREE} .osprey-list-row[data-id]:not(.osprey-tree-run)").first
    expect(first).to_be_visible(timeout=_TIMEOUT_MS)
    # Pinned by its id: `.first` is re-run on every use, and a redraw can put
    # another row first.
    artifact = page.locator(f'{_TREE} .osprey-list-row[data-id="{first.get_attribute("data-id")}"]')
    labels = _menu_key(page, artifact)
    assert labels[:2] == ["Open", "Open in new session"], labels

    # A pinned session draws the Pinned group, whose heading has a menu.
    page.evaluate(
        """async (sid) => {
          const m = await import('/static/js/session-model.js');
          await m.setPinned(sid, true);
        }""",
        sid,
    )
    pinned = page.locator(".sidebar-pinned-head")
    expect(pinned).to_be_visible(timeout=_TIMEOUT_MS)
    labels = _menu_key(page, pinned)
    assert "Hide pinned sessions" in labels, labels


def test_the_menu_key_opens_a_run_rows_menu(hub, page):
    seeded = SEEDED_SESSIONS[2]
    call = "toolu_e2e_menu_key_run"
    with _writers(hub, seeded.session_key):
        _execute(call)
    page.add_init_script(_KIND_MODE_JS)
    _boot_hermetic(page, hub)
    sid = seeded.session_key
    _open_session(page, sid)
    _open_group(page, sid, "runs")
    run = _run_row(page, call)
    expect(run).to_be_visible(timeout=_TIMEOUT_MS)
    assert _menu_key(page, run), "the run row's menu has rows"


def _empty_sidebar(page: Page) -> tuple[Locator, dict]:
    """The rail's own box below its last group: empty sidebar space."""
    rail = page.locator("#panel-rail")
    box = rail.bounding_box()
    last = page.locator("#panel-rail > *:visible").last.bounding_box()
    assert box is not None and last is not None
    y = last["y"] + last["height"] + 12 - box["y"]
    assert y < box["height"] - 4, "the rail has no empty space below its groups"
    return rail, {"x": 24, "y": y}


def test_a_right_click_on_empty_sidebar_space_opens_the_sidebar_menu(hub, page):
    _boot_hermetic(page, hub)
    rail, at = _empty_sidebar(page)
    menu = _right_click_menu(page, rail, position=at)
    expect(menu).to_have_attribute("aria-label", "Sidebar")
    assert "New group…" in _labels(menu), _labels(menu)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="empty sidebar space (#sidebar, #panel-rail) takes no focus, so the menu key "
    "cannot reach its menu from the keyboard",
)
def test_the_menu_key_opens_the_empty_sidebar_menu(hub, page):
    _boot_hermetic(page, hub)
    assert _menu_key(page, page.locator("#panel-rail"), aria="Sidebar")


def test_the_menu_key_opens_the_chat_and_pop_out_menus(hub, page):
    _boot_hermetic(page, hub)
    _converse(page, "Status of the ring")

    assert _menu_key(page, _entry(page, "assistant"), aria="Agent message") == [
        "Copy",
        "Quote in prompt",
    ]
    assert _menu_key(page, _entry(page, "operator"), aria="Your message")[0] == "Copy"
    labels = _menu_key(page, page.locator(".op-messages"), aria="Conversation")
    assert labels[0] == "Scroll to latest", labels

    popup = _detach_agent(page)
    labels = _menu_key(popup, popup.locator(f"{_POP_BAR} .agent-bar-new"))
    assert "Reattach" in labels and labels[-1] == "Close window", labels
    expect(page.locator(_MENU)).to_have_count(0)


# --- reveal and submenus ----------------------------------------------------------


def test_reveal_in_sidebar_expands_a_collapsed_sidebar(tmp_path, chromium_browser):
    with _hub(tmp_path) as dock:
        page = _open_dock_hub(chromium_browser, dock)
        _open_service(page, "svc-a")
        page.locator("#sidebar-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-sidebar", "collapsed")

        _right_click_menu(page, _bar(page, "svc-a").locator(".tile-tab-title"))
        page.locator(_MENU).get_by_role("menuitem", name="Reveal in sidebar").click()

        expect(page.locator("html")).to_have_attribute("data-sidebar", "expanded", timeout=5_000)
        expect(rail_entry(page, "svc-a").locator(".osprey-nav-label")).to_be_visible()
        assert dock.errors == []


def test_a_submenu_opens_closes_a_level_and_runs_from_the_keyboard(tmp_path, chromium_browser):
    with _hub(tmp_path) as dock:
        page = _open_dock_hub(chromium_browser, dock)
        _open_service(page, "svc-a")
        target = _nav_label(page, "svc-b")
        _focus(_bar(page, "svc-a").locator(".tile-tab-more"))
        page.keyboard.press("Shift+F10")
        expect(page.locator(_MENU)).to_have_count(1, timeout=5_000)

        _press_until_active(page, "Open panel")
        page.keyboard.press("ArrowRight")
        depth1 = page.locator(f'{_RAIL_MENU}[data-depth="1"]')
        expect(depth1).to_have_count(1, timeout=5_000)
        assert _labels(depth1) == ["Right", "Down", "Replace here"]
        assert _active_label(page) == "Right"

        page.keyboard.press("ArrowRight")
        depth2 = page.locator(f'{_RAIL_MENU}[data-depth="2"]')
        expect(depth2).to_have_count(1, timeout=5_000)
        assert target in _labels(depth2), _labels(depth2)
        # ArrowLeft closes one level, back on the row that opened it.
        page.keyboard.press("ArrowLeft")
        expect(depth2).to_have_count(0, timeout=5_000)
        expect(depth1).to_have_count(1)
        assert _active_label(page) == "Right"

        page.keyboard.press("ArrowRight")
        expect(depth2).to_have_count(1, timeout=5_000)
        _press_until_active(page, target)
        page.keyboard.press("Enter")

        expect(page.locator(_MENU)).to_have_count(0, timeout=5_000)
        expect(_bar(page, "svc-b")).to_have_count(1, timeout=10_000)
        expect(_bar(page, "svc-a")).to_have_count(1)
        a = _bar(page, "svc-a").bounding_box()
        b = _bar(page, "svc-b").bounding_box()
        assert b["x"] >= a["x"] + a["width"] - 2, (a, b)
        assert dock.errors == []
