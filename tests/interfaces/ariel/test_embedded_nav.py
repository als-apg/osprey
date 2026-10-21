"""Browser tests for ARIEL's embedded-mode navigation.

When the panel runs inside the web-terminal iframe (``?embedded=true``) the
host tile bar is its only header: ``layout.css`` hides the panel's own header
outright and ``app.js`` contributes a Browse / New Entry nav into the tile bar
instead, with Search in front of them where ``ariel.embedded_search`` is on.
Embedded without the switch, the router opens Browse and redirects a stray
``#search`` there; ``#status`` is redirected either way.
Standalone keeps the full header and Search as the default view.

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


def _embedded_search_on(page) -> None:
    """Answer ``/api/capabilities`` as a deployment with ``ariel.embedded_search: true``."""
    page.route(
        "**/api/capabilities",
        lambda route: route.fulfill(json={"embedded_search": True}),
    )


def test_embedded_hides_header_and_defaults_to_browse(tmp_path, monkeypatch, chromium_browser):
    """``?embedded=true`` hides the panel's own header and lands on Browse."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true", wait_until="load")

        # layout.css drops the whole header off body.embedded — the tile bar
        # carries the nav instead (contributed by app.js).
        expect(page.locator(".header")).to_be_hidden()
        # Search is off by default when embedded, so the empty hash opens Browse.
        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page.locator("#view-search.active")).to_have_count(0)

        page.close()


def test_embedded_redirects_search_hash_to_browse(tmp_path, monkeypatch, chromium_browser):
    """A stray ``#search`` deep link is rerouted to Browse when Search is off."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true#search", wait_until="load")

        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page).to_have_url(f"{base_url}/?embedded=true#browse")

        page.close()


def test_embedded_search_switch_defaults_to_search(tmp_path, monkeypatch, chromium_browser):
    """With ``ariel.embedded_search`` on, the embedded panel opens on Search."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        _embedded_search_on(page)

        page.goto(f"{base_url}?embedded=true", wait_until="load")

        expect(page.locator("#view-search.active")).to_have_count(1)
        expect(page.locator("#view-browse.active")).to_have_count(0)

        page.close()


def test_embedded_search_switch_keeps_search_hash(tmp_path, monkeypatch, chromium_browser):
    """With ``ariel.embedded_search`` on, a ``#search`` deep link stays on Search."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        _embedded_search_on(page)

        page.goto(f"{base_url}?embedded=true#search", wait_until="load")

        expect(page.locator("#view-search.active")).to_have_count(1)
        expect(page).to_have_url(f"{base_url}/?embedded=true#search")

        page.close()


def test_embedded_redirects_status_hash_to_browse(tmp_path, monkeypatch, chromium_browser):
    """Status is standalone-only: ``#status`` reroutes, so its poll never starts."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true#status", wait_until="load")

        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page.locator("#view-status.active")).to_have_count(0)
        expect(page).to_have_url(f"{base_url}/?embedded=true#browse")

        page.close()


def test_standalone_keeps_header_and_search_default(tmp_path, monkeypatch, chromium_browser):
    """Without ``?embedded=true`` the panel keeps its header and opens on Search."""
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(base_url, wait_until="load")

        expect(page.locator(".header")).to_be_visible()
        for view in ("search", "browse", "create", "status"):
            expect(page.locator(f'.nav-link[data-view="{view}"]')).to_be_visible()
        expect(page.locator("#view-search.active")).to_have_count(1)

        page.close()


@pytest.mark.parametrize("query", ["?embedded=true", ""], ids=["embedded", "standalone"])
def test_entry_route_opens_the_detail_card_over_browse(
    tmp_path, monkeypatch, chromium_browser, query
):
    """``#entry?id=`` opens the detail card over Browse; closing hands the hash back.

    The launched panel has no database, so the entry cannot load and the card
    shows the panel's ordinary failure message.
    """
    with _launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}/{query}#entry?id=no-such-entry&attachment=att-x", wait_until="load")

        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page.locator("#entry-modal")).to_be_visible()
        expect(page.locator("#entry-modal-body")).to_contain_text("Failed to Load Entry")
        expect(page.locator("#image-lightbox")).to_have_count(0)
        expect(page).to_have_url(f"{base_url}/{query}#entry?id=no-such-entry&attachment=att-x")

        page.locator("#entry-modal-close").click()

        expect(page.locator("#entry-modal")).to_be_hidden()
        expect(page).to_have_url(f"{base_url}/{query}#browse")

        page.close()
