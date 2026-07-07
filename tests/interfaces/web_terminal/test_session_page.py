"""Behavioral pin for session.html's four-view nav and periodic refresh.

Complements test_contract_params.py's page-7 chrome-contract cases (which
cover the shell: branding/switcher/theme) with the page's actual content
behavior: session.js's nav click handler (the four ``.pill``/``.view`` pairs
toggle in lockstep and each view fetches its own endpoint) and the 12s
``setInterval`` refresh loop that repaints whichever view is currently
active.

Each view's renderer fetches a distinct same-origin endpoint (session-views.js):
agents -> ``/api/session-agents``, toollog -> ``/api/session-log``,
artifacts -> ``/api/session-summary``, conversation -> ``/api/session-chat``.
A real network request to that endpoint is the observable signal that
switching (or the refresh loop) actually invoked the view's renderer, not
just toggled a CSS class.

The refresh loop is driven via Playwright's Clock API (``page.clock``,
installed *before* navigation so it captures the ``setInterval`` session.js
registers at module-eval time) rather than a real 12-second wait or any
change to session.js itself.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_session_page.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pytest

from tests.interfaces.test_load_smokes import _launch_web_terminal

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# view name -> the endpoint its renderer fetches (session-views.js).
_VIEW_ENDPOINTS = {
    "agents": "/api/session-agents",
    "toollog": "/api/session-log",
    "artifacts": "/api/session-summary",
    "conversation": "/api/session-chat",
}


def test_session_nav_pills_switch_views(tmp_path, monkeypatch, chromium_browser):
    """Clicking a nav pill activates its view, fetches its endpoint, and deactivates the rest.

    Cycles through all four views (starting and ending on the default
    'agents'), asserting after each click that exactly the clicked pill and
    its matching ``.view`` section carry ``active`` and every other pill/view
    does not -- session.js's handler toggles all of them together in one
    click handler, so a partial update would be a real regression.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Arrange -- initial load triggers refreshActive() for the default view.
        with page.expect_request(lambda r: "/api/session-agents" in r.url):
            page.goto(f"{base_url}/static/session.html", wait_until="load")

        expect(page.locator('.pill[data-view="agents"]')).to_have_class("pill active")
        expect(page.locator("#view-agents")).to_have_class("view active")

        # Act + Assert -- switch through every view in turn.
        for view in ("toollog", "artifacts", "conversation", "agents"):
            endpoint = _VIEW_ENDPOINTS[view]
            with page.expect_request(lambda r, endpoint=endpoint: endpoint in r.url):
                page.locator(f'.pill[data-view="{view}"]').click()

            expect(page.locator(f'.pill[data-view="{view}"]')).to_have_class("pill active")
            expect(page.locator(f"#view-{view}")).to_have_class("view active")
            for other in _VIEW_ENDPOINTS:
                if other == view:
                    continue
                expect(page.locator(f'.pill[data-view="{other}"]')).to_have_class("pill")
                expect(page.locator(f"#view-{other}")).to_have_class("view")

        page.close()


def test_session_refresh_loop_repaints_active_view(tmp_path, monkeypatch, chromium_browser):
    """The 12s ``setInterval`` re-fetches whichever view is currently active.

    Fake timers are installed *before* navigation: session.js's
    ``setInterval(refreshActive, 12000)`` runs synchronously at module-eval
    time (deferred module scripts run after the DOM parses, but still before
    any interaction), so the interval must already be registered against the
    faked clock by the time that line executes -- installing afterward would
    leave it bound to a real timer instead.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.clock.install()

        with page.expect_request(lambda r: "/api/session-agents" in r.url):
            page.goto(f"{base_url}/static/session.html", wait_until="load")

        # Act -- advance virtual time past the 12s interval.
        with page.expect_request(lambda r: "/api/session-agents" in r.url) as repaint_info:
            page.clock.fast_forward(12_000)

        # Assert -- a second request for the still-active 'agents' view fired
        # from the interval callback, not merely the initial on-load refresh
        # above (which the `with` block already consumed).
        assert "/api/session-agents" in repaint_info.value.url

        page.close()
