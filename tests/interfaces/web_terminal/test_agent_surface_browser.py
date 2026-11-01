"""Browser tests: the Agent tile opens on the chat, and the terminal is an opt-in.

What only a real browser can prove: the pre-paint boot script and the CSS gate,
that no PTY socket opens until the ⋯ entry is clicked, that the choice lives per
tab, and that ``web.terminal: false`` leaves no way in.

Run:
    uv run pytest tests/interfaces/web_terminal/test_agent_surface_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pytest

from tests.interfaces.web_terminal._chat_server import _PTY_SPAWNS, live_chat_server
from tests.interfaces.web_terminal._surface import back_to_chat, open_terminal

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_CHAT = "#operator-container"
_TERMINAL = "#terminal-container"
_ENTRY = ".op-surface-row"


def _open(opener, base_url: str) -> tuple[Page, list[str]]:
    """Open the hub; returns the page and the PTY sockets it opens from here on."""
    page = opener.new_page()
    sockets: list[str] = []
    page.on(
        "websocket",
        lambda ws: sockets.append(ws.url) if "/ws/terminal" in ws.url else None,
    )
    page.goto(base_url, wait_until="domcontentloaded")
    expect(page.locator(".op-session-bar")).to_be_visible(timeout=10_000)
    return page, sockets


def _on_chat(page: Page) -> None:
    expect(page.locator(_CHAT)).to_be_visible()
    expect(page.locator(_TERMINAL)).to_be_hidden()
    expect(page.locator(".op-session-bar")).to_be_visible()


def _on_terminal(page: Page) -> None:
    expect(page.locator(_TERMINAL)).to_be_visible()
    expect(page.locator(_CHAT)).to_be_hidden()
    expect(page.locator(".terminal-header")).to_be_visible()


def test_boot_is_the_chat_and_opens_no_pty_socket(tmp_path, chromium_browser):
    with live_chat_server(tmp_path) as (base_url, _app):
        page, sockets = _open(chromium_browser, base_url)
        _on_chat(page)
        expect(page.locator("html")).not_to_have_attribute("data-agent-surface", "terminal")
        page.wait_for_timeout(1_000)
        assert sockets == []
        assert _PTY_SPAWNS == []
        page.close()


def test_the_entry_opens_the_terminal_and_back_to_chat_returns(tmp_path, chromium_browser):
    with live_chat_server(tmp_path) as (base_url, _app):
        page, sockets = _open(chromium_browser, base_url)

        open_terminal(page)
        _on_terminal(page)
        expect(page.locator("#terminal-container .xterm")).to_be_visible(timeout=10_000)
        page.wait_for_function("() => document.querySelector('#session-led.active') !== null")
        assert len(sockets) == 1

        back_to_chat(page)
        _on_chat(page)
        page.close()


def test_the_choice_survives_a_reload_and_a_new_tab_starts_in_chat(tmp_path, chromium_browser):
    with live_chat_server(tmp_path) as (base_url, _app):
        context = chromium_browser.new_context()
        page, _sockets = _open(context, base_url)
        open_terminal(page)

        page.reload(wait_until="domcontentloaded")
        _on_terminal(page)

        other, sockets = _open(context, base_url)
        _on_chat(other)
        assert sockets == []

        back_to_chat(page)
        page.reload(wait_until="domcontentloaded")
        _on_chat(page)
        context.close()


def test_web_terminal_false_leaves_no_entry(tmp_path, chromium_browser):
    with live_chat_server(tmp_path) as (base_url, app):
        app.state.web_terminal_offered = False
        page, sockets = _open(chromium_browser, base_url)
        expect(page.locator("html")).to_have_attribute("data-terminal", "off")

        page.locator(".op-session-more").click()
        expect(page.locator("#agent-conversations-menu")).to_be_visible()
        expect(page.locator(_ENTRY)).to_have_count(0)
        assert sockets == []
        page.close()


def test_a_kept_choice_is_ignored_under_web_terminal_false(tmp_path, chromium_browser):
    with live_chat_server(tmp_path) as (base_url, app):
        app.state.web_terminal_offered = False
        context = chromium_browser.new_context()
        context.add_init_script("sessionStorage.setItem('osprey-agent-surface', 'terminal')")
        page, sockets = _open(context, base_url)
        _on_chat(page)
        page.wait_for_timeout(500)
        assert sockets == []
        context.close()


def test_the_more_menu_and_palette_have_no_way_to_the_terminal(tmp_path, chromium_browser):
    with live_chat_server(tmp_path) as (base_url, _app):
        page, _sockets = _open(chromium_browser, base_url)

        page.locator("#more-btn").click()
        menu = page.locator("#more-menu")
        expect(menu).to_be_visible()
        assert "terminal" not in menu.inner_text().lower()
        page.keyboard.press("Escape")

        page.locator("#command-palette-btn").click()
        palette = page.locator(".command-palette-dialog")
        expect(palette).to_be_visible()
        assert "terminal" not in palette.inner_text().lower()
        page.close()
