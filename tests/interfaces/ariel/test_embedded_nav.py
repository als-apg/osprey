"""Browser tests for ARIEL's embedded-mode navigation.

When the panel runs inside the web-terminal iframe (``?embedded=true``) the
host tile bar is its only header: ``layout.css`` hides the panel's own header
and each view's title row (the Search hero, the Browse and New Entry
headings) outright, and ``app.js`` contributes a Browse / New Entry nav into
the tile bar instead, with Search in front of them where
``ariel.embedded_search`` is on.
Embedded without the switch, the router opens Browse and redirects a stray
``#search`` there; ``#status`` is redirected either way.
Standalone keeps the full header, the view headings and Search as the
default view.

Run:
    .venv/bin/python -m pytest tests/interfaces/ariel/test_embedded_nav.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pytest

from tests.interfaces._launchers import launch_ariel

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


#: Stands in for the hub: the page is top-level in these tests, so
#: ``window.parent`` is replaced with a recorder before any script runs, and
#: ``contributeHeader`` (a no-op unless ``window.parent !== window``) posts
#: each contribution into ``window.__contributions``.
_RECORD_CONTRIBUTIONS = """
(() => {
  window.__contributions = [];
  const hub = { postMessage: (message) => { window.__contributions.push(message); } };
  Object.defineProperty(window, 'parent', { configurable: true, get: () => hub });
})();
"""


def _last_nav(page) -> list[dict]:
    """Entries of the ``view`` nav in the panel's latest tile-bar contribution."""
    page.wait_for_function("window.__contributions.length > 0")
    items = page.evaluate("window.__contributions.at(-1).items")
    navs = [item for item in items if item["kind"] == "nav" and item["id"] == "view"]
    assert len(navs) == 1, items
    return navs[0]["items"]


def test_embedded_hides_header_and_defaults_to_browse(tmp_path, monkeypatch, chromium_browser):
    """``?embedded=true`` hides the panel's own header and lands on Browse."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true", wait_until="load")

        # layout.css drops the whole header off body.embedded — the tile bar
        # carries the nav instead (contributed by app.js).
        expect(page.locator(".header")).to_be_hidden()
        # Search is off by default when embedded, so the empty hash opens Browse.
        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page.locator("#view-search.active")).to_have_count(0)
        # The tile bar names the panel and marks the view: no title row in the body.
        expect(page.locator("#view-browse .dashboard-header")).to_be_hidden()

        page.close()


def test_embedded_new_entry_has_no_title_row(tmp_path, monkeypatch, chromium_browser):
    """Embedded New Entry opens straight on its form, with no "New Entry" heading."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true#create", wait_until="load")

        expect(page.locator("#view-create.active")).to_have_count(1)
        expect(page.locator("#view-create .dashboard-header")).to_be_hidden()
        expect(page.locator("#entry-subject")).to_be_visible()

        page.close()


def test_embedded_nav_carries_browse(tmp_path, monkeypatch, chromium_browser):
    """The tile-bar nav offers Browse, and a click there opens it."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.add_init_script(_RECORD_CONTRIBUTIONS)

        page.goto(f"{base_url}?embedded=true#create", wait_until="load")

        expect(page.locator("#view-create.active")).to_have_count(1)
        nav = _last_nav(page)
        assert [entry["id"] for entry in nav] == ["browse", "create"]
        assert [entry["id"] for entry in nav if entry["active"]] == ["create"]

        # The hub posts the clicked entry back; the panel switches and re-publishes.
        page.evaluate(
            "window.postMessage("
            "{type: 'osprey-header-action', id: 'view', value: 'browse'}, location.origin)"
        )

        expect(page.locator("#view-browse.active")).to_have_count(1)
        page.wait_for_function(
            "window.__contributions.at(-1).items[0].items.find((e) => e.active)?.id === 'browse'"
        )

        page.close()


def test_embedded_redirects_search_hash_to_browse(tmp_path, monkeypatch, chromium_browser):
    """A stray ``#search`` deep link is rerouted to Browse when Search is off."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true#search", wait_until="load")

        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page).to_have_url(f"{base_url}/?embedded=true#browse")

        page.close()


def test_embedded_search_switch_defaults_to_search(tmp_path, monkeypatch, chromium_browser):
    """With ``ariel.embedded_search`` on, the embedded panel opens on Search."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        _embedded_search_on(page)

        page.goto(f"{base_url}?embedded=true", wait_until="load")

        expect(page.locator("#view-search.active")).to_have_count(1)
        expect(page.locator("#view-browse.active")).to_have_count(0)
        # No "Search the Logbook" hero: the search box is the first thing in the body.
        expect(page.locator(".search-hero")).to_be_hidden()
        expect(page.locator("#search-input")).to_be_visible()

        page.close()


def test_embedded_search_switch_keeps_search_hash(tmp_path, monkeypatch, chromium_browser):
    """With ``ariel.embedded_search`` on, a ``#search`` deep link stays on Search."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        _embedded_search_on(page)

        page.goto(f"{base_url}?embedded=true#search", wait_until="load")

        expect(page.locator("#view-search.active")).to_have_count(1)
        expect(page).to_have_url(f"{base_url}/?embedded=true#search")

        page.close()


def test_embedded_redirects_status_hash_to_browse(tmp_path, monkeypatch, chromium_browser):
    """Status is standalone-only: ``#status`` reroutes, so its poll never starts."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(f"{base_url}?embedded=true#status", wait_until="load")

        expect(page.locator("#view-browse.active")).to_have_count(1)
        expect(page.locator("#view-status.active")).to_have_count(0)
        expect(page).to_have_url(f"{base_url}/?embedded=true#browse")

        page.close()


def test_standalone_keeps_header_and_search_default(tmp_path, monkeypatch, chromium_browser):
    """Without ``?embedded=true`` the panel keeps its header and opens on Search."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        page.goto(base_url, wait_until="load")

        expect(page.locator(".header")).to_be_visible()
        for view in ("search", "browse", "create", "status"):
            expect(page.locator(f'.nav-link[data-view="{view}"]')).to_be_visible()
        expect(page.locator("#view-search.active")).to_have_count(1)
        # Standalone, the view heading is the page's own title and stays.
        expect(page.locator(".search-hero h1")).to_be_visible()

        page.close()


@pytest.mark.parametrize("query", ["?embedded=true", ""], ids=["embedded", "standalone"])
def test_entry_route_opens_the_detail_card_over_browse(
    tmp_path, monkeypatch, chromium_browser, query
):
    """``#entry?id=`` opens the detail card over Browse; closing hands the hash back.

    The launched panel has no database, so the entry cannot load and the card
    shows the panel's ordinary failure message.
    """
    with launch_ariel(tmp_path, monkeypatch) as base_url:
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
