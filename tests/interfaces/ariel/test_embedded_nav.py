"""Browser tests for ARIEL's embedded-mode navigation.

When the panel runs inside the web-terminal iframe (``?embedded=true``),
logbook search goes through the agent, so the Search nav link is hidden
(``layout.css``) and the router opens Browse by default and redirects stray
``#search`` hashes to Browse (``app.js``). Standalone keeps Search as the
default view.

Run:
    .venv/bin/python -m pytest tests/interfaces/ariel/test_embedded_nav.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pytest

from tests.interfaces.test_load_smokes import _launch_ariel

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_embedded_hides_search_and_defaults_to_browse(tmp_path, monkeypatch, chromium_browser):
    """``?embedded=true`` hides the Search nav link and lands on Browse."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true", wait_until="load")

        # layout.css hides the Search link off body.embedded.
        expect(page.locator('.nav-link[data-view="search"]')).to_be_hidden()
        # The other three links stay visible.
        for view in ("browse", "create", "status"):
            expect(page.locator(f'.nav-link[data-view="{view}"]')).to_be_visible()
        # defaultView() routed the empty hash to Browse, not Search.
        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page.locator("#view-search.active")).to_have_count(0)

        page.close()


def test_embedded_redirects_search_hash_to_browse(tmp_path, monkeypatch, chromium_browser):
    """A stray ``#search`` deep link is rerouted to Browse when embedded."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true#search", wait_until="load")

        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page).to_have_url(f"{base_url}/?embedded=true#browse")

        page.close()


def test_standalone_keeps_search_as_default(tmp_path, monkeypatch, chromium_browser):
    """Without ``?embedded=true`` the panel still opens on a visible Search."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(base_url, wait_until="load")

        expect(page.locator('.nav-link[data-view="search"]')).to_be_visible()
        expect(page.locator("#view-search.active")).to_have_count(1)

        page.close()
