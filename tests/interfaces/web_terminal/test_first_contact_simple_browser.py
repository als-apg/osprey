"""Browser test: the Simple view's first-contact block, in a real page.

The empty state is what a newcomer reads before they type anything, so the two
things it must never get wrong are pinned here rather than in a DOM-double
unit test: it has to actually APPEAR on a live page (it renders only once the
server's facts, a session and a control-target render have all landed — a
sequence no happy-dom harness reproduces), and a starter chip has to INSERT its
question into the composer without sending it. A chip that submitted would turn
one click that only said "ask me this" into a turn the operator never chose.

The sentence is checked for what it must not claim as much as for what it says:
this deployment stands on no known machine, so nothing may name one.

Run:
    uv run pytest tests/interfaces/web_terminal/test_first_contact_simple_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.interfaces._browser import assert_page_loads_clean

# The faked-SDK live server from the chat suite: a real web terminal whose only
# replaced part is the agent subprocess, so a turn can be sent without a model.
from tests.interfaces.web_terminal.test_chat_browser import _PLANS, _live_chat_server

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# The onboarding tour's invite veil covers the shell on a fresh profile and
# intercepts pointer events, so every page seeds the dismissal flag before its
# first script runs — the same seed the ui-mode and drawer suites carry.
_DISMISS_TOUR = "try { localStorage.setItem('osprey-tour-dismissed-v1', '1') } catch (e) {}"

_OP = "#operator-container"
_INPUT = f"{_OP} .op-input-area textarea"
_EMPTY = f"{_OP} .op-messages .op-empty"

# The one prompt every deployment offers, because what the agent is ALLOWED to
# do is per-session posture the page cannot answer for itself.
_ALLOWED_PROMPT = "What are you allowed to do in this session?"

# A capability the server publishes so the sentence has something to say; the
# read phrase would be the chip's contribution, and this hub has no roster.
_CAPABILITY = "run analysis scripts you approve"


def _open_simple_hub(browser: Browser, base_url: str) -> tuple[Page, list[str], list[str]]:
    """Open the hub in Simple view and start watching the page's behaviour.

    Returns:
        (page, errors, chat_posts) — the page, a running list of uncaught
        exceptions and ``console.error`` texts for the WHOLE flow (the load
        itself is gated by ``assert_page_loads_clean``), and every ``POST``
        this page made to the chat endpoint.
    """
    page = browser.new_page()
    page.add_init_script(_DISMISS_TOUR)

    errors: list[str] = []
    chat_posts: list[str] = []

    def _on_console(msg) -> None:  # noqa: ANN001 - Playwright ConsoleMessage
        if msg.type == "error":
            errors.append(f"console: {msg.text}")

    def _on_request(request) -> None:  # noqa: ANN001 - Playwright Request
        if request.method == "POST" and "/api/chat" in request.url:
            chat_posts.append(request.url)

    page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
    page.on("console", _on_console)
    page.on("request", _on_request)

    assert_page_loads_clean(page, base_url)
    expect(page.locator(_INPUT)).to_be_visible(timeout=10_000)
    return page, errors, chat_posts


def test_empty_state_renders_and_a_chip_only_inserts(tmp_path, chromium_browser):
    """The block appears, claims nothing about a machine, and its chip inserts.

    One test because the second half needs the first: the chip only exists once
    the block has rendered, and re-reaching that settled moment in a second page
    load buys nothing. The insert is asserted on the composer's exact value (no
    trailing newline, which would be a submitted turn's residue elsewhere), on
    focus landing in the same input, and on the negative that matters most — no
    request reached the chat endpoint.
    """
    with _live_chat_server(tmp_path, ui_mode="simple") as (base_url, app):
        app.state.tour_capabilities = [_CAPABILITY]
        page, errors, chat_posts = _open_simple_hub(chromium_browser, base_url)

        block = page.locator(_EMPTY)
        expect(block).to_be_visible(timeout=10_000)
        # It is the first thing in the log, above anything the session adds.
        assert page.evaluate(
            f"() => document.querySelector('{_OP} .op-messages').firstElementChild"
            ".classList.contains('op-empty')"
        ), "the first-contact block must be the first child of the message log"

        intro = page.locator(f"{_EMPTY} .op-empty-intro")
        expect(intro).to_have_count(1)
        intro_text = intro.inner_text()
        assert _CAPABILITY in intro_text, (
            f"the sentence must carry the published capability; read {intro_text!r}"
        )
        # No roster means no machine kind, and an unnamed machine may not be
        # described as the facility's own.
        assert "live machine" not in intro_text, (
            f"nothing may name a machine this session was never told about; read {intro_text!r}"
        )

        chip = page.locator(f"{_EMPTY} button.tour-chip", has_text=_ALLOWED_PROMPT)
        expect(chip).to_have_count(1)
        chip.click()

        textarea = page.locator(_INPUT)
        expect(textarea).to_have_value(_ALLOWED_PROMPT)
        assert page.evaluate(
            f"() => document.activeElement === document.querySelector('{_INPUT}')"
        ), "the cursor must land in the composer the chip just filled"

        # Give a mistaken submit time to leave the page before calling it absent.
        page.wait_for_timeout(500)
        assert chat_posts == [], f"a chip must never send; the page POSTed {chat_posts!r}"
        # The invitation is still standing: nothing was said yet.
        expect(block).to_be_visible()

        assert errors == [], f"the page reported errors during first contact: {errors}"
        page.close()


def test_first_message_clears_the_empty_state(tmp_path, chromium_browser):
    """Sending a turn removes the block: the log is the conversation from here."""
    with _live_chat_server(tmp_path, ui_mode="simple") as (base_url, app):
        app.state.tour_capabilities = [_CAPABILITY]
        _PLANS["hello there"] = [("text", "hi back"), ("result",)]
        page, errors, chat_posts = _open_simple_hub(chromium_browser, base_url)

        block = page.locator(_EMPTY)
        expect(block).to_be_visible(timeout=10_000)

        textarea = page.locator(_INPUT)
        textarea.fill("hello there")
        textarea.press("Enter")

        expect(block).to_have_count(0, timeout=10_000)
        expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
            "hi back", timeout=10_000
        )

        # The same recorder the chip test reads as empty did see this turn leave,
        # so its silence there is evidence and not a broken listener.
        assert chat_posts, "no POST reached the chat endpoint; the request recorder is blind"

        assert errors == [], f"the page reported errors while sending the first turn: {errors}"
        page.close()
