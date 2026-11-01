"""Browser helpers for the Agent tile's two surfaces.

The tile opens on the chat. The terminal is a per-tab opt-in from the tile's ⋯
menu (agent-surface.js), and a tab that opted in reopens on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page

#: Seeds the tab's choice before the page loads, once, so a later "Back to chat"
#: and reload stays in chat. The hub's own frame only: a panel iframe shares the
#: tab's sessionStorage.
_TERMINAL_TAB_INIT = """
if (window === window.top && !sessionStorage.getItem('__terminal_tab_seeded')) {
  sessionStorage.setItem('__terminal_tab_seeded', '1');
  sessionStorage.setItem('KEY', 'terminal');
}
"""


def arm_terminal_tab(page: Page, *, scope: str = "") -> None:
    """Make *page* boot like a tab that already opted into the terminal.

    Call before ``goto``. Nothing else on the page is staged: the boot script,
    the PTY socket and the terminal header all run as they do on a reload.

    Args:
        page: The page about to load the hub.
        scope: The storage scope the hub stamps, on a multi-user mount.
    """
    key = f"osprey-agent-surface--{scope}" if scope else "osprey-agent-surface"
    page.add_init_script(_TERMINAL_TAB_INIT.replace("KEY", key))


def open_terminal(page: Page) -> None:
    """Open the terminal from the chat's ⋯ menu, the way an operator does."""
    from playwright.sync_api import expect

    page.locator(".op-session-more").click()
    page.locator(".op-surface-row", has_text="Open terminal").click()
    expect(page.locator("html")).to_have_attribute("data-agent-surface", "terminal")
    expect(page.locator("#terminal-container")).to_be_visible()


def back_to_chat(page: Page) -> None:
    """Go back to the chat from the terminal header's ⋯ menu."""
    from playwright.sync_api import expect

    page.locator("#terminal-more-btn").click()
    page.locator(".op-surface-row", has_text="Back to chat").click()
    expect(page.locator("html")).not_to_have_attribute("data-agent-surface", "terminal")
    expect(page.locator("#operator-container")).to_be_visible()
