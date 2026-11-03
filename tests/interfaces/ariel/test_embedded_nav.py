"""Browser tests for ARIEL's embedded contract.

When the panel runs inside the web-terminal iframe (``?embedded=true``) the
host tile bar is its only header: ``layout.css`` hides the panel's own header,
and ``header.js`` contributes ``[search, Compose, ⋯ menu]`` to the tile bar
instead. The list pane then carries no search box and no compose button of
its own, and the rail starts as an icon strip.

The search item re-keys (a new id, seeded with the new value) whenever the
panel itself changes the query (hash navigation, a saved search), and keeps
its id for the tile bar's own reports, so the hub never overwrites what the
operator is typing. Every legacy hash alias lands embedded too; ``#status``
shows the list there, because Status is standalone-only.

Run:
    uv run pytest tests/interfaces/ariel/test_embedded_nav.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

from tests.interfaces.ariel import test_shell as _shell
from tests.interfaces.ariel.test_shell import (
    INVALID_CAPABILITIES,
    FakeBackend,
    app,
    canonical,
    reader_title,
    row,
    wait_for_call,
    wait_for_hash,
)

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

try:
    from playwright.sync_api import expect
except ImportError:  # pragma: no cover
    expect = None  # type: ignore[assignment]

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: The shell suite's panel fixtures: one DB-less ARIEL per module, re-authorized per test.
_ariel_server = _shell._ariel_server
ariel_url = _shell.ariel_url


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


def _open_embedded(
    browser: Browser, url: str, backend: FakeBackend | None = None, *, record: bool = True
) -> tuple[Page, FakeBackend]:
    """An embedded page on *url* behind a fake backend, recording contributions."""
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    if record:
        page.add_init_script(_RECORD_CONTRIBUTIONS)
    backend = (backend or FakeBackend()).install(page)
    page.goto(url, wait_until="load")
    expect(page.locator("#ariel-list-body")).to_have_attribute("aria-busy", "false")
    return page, backend


def _items(page: Page) -> list[dict[str, Any]]:
    """Items of the panel's latest tile-bar contribution."""
    page.wait_for_function("window.__contributions.length > 0")
    return page.evaluate("window.__contributions.at(-1).items")


def _item(page: Page, kind: str) -> dict[str, Any]:
    found = [item for item in _items(page) if item["kind"] == kind]
    assert len(found) == 1, _items(page)
    return found[0]


def _action(page: Page, item_id: str, value: str | None = None) -> None:
    """Post a tile-bar action the way the hub does."""
    page.evaluate(
        "([id, value]) => window.postMessage("
        "{type: 'osprey-header-action', id, value}, location.origin)",
        [item_id, value],
    )


def test_embedded_hides_header_and_contributes_search_compose_and_menu(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """The panel's header is gone; the tile bar gets search, Compose and the ⋯ menu."""
    page, _ = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true")

    expect(page.locator(".header")).to_be_hidden()
    # The box and the compose button belong to the host now.
    expect(page.locator("#ariel-search-input")).to_have_count(0)
    expect(page.locator("#ariel-compose-btn")).to_have_count(0)
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
    expect(row(page, "E1")).to_be_visible()

    items = _items(page)
    assert [item["kind"] for item in items] == ["search", "button", "menu"], items
    search, compose, menu = items
    assert search["id"] == "ariel-search-0"
    assert search["value"] == ""
    assert compose["id"] == "ariel-compose"
    assert compose["label"] == "Compose"
    assert menu["id"] == "ariel-more"
    assert [entry["id"] for entry in menu["items"]] == [
        "toggle-rail",
        "save-search",
        "toggle-strip",
    ]
    assert menu["items"][0]["label"] == "Show sidebar"
    page.close()


def test_tile_bar_search_runs_the_query_without_re_keying(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A tile-bar search report runs the search and keeps the field's id."""
    page, backend = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true")
    search_id = _item(page, "search")["id"]

    _action(page, search_id, "vacuum")

    expect(page.locator("#ariel-list .results-count")).to_be_visible()
    assert backend.calls("POST", "/api/search")[-1].json()["query"] == "vacuum"
    wait_for_hash(page, canonical(q="vacuum"))
    assert _item(page, "search")["id"] == search_id
    page.close()


def test_a_panel_set_query_re_keys_the_search_item(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A query the panel sets itself (here a hash) re-keys the field, seeded with it."""
    page, _ = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true")
    assert _item(page, "search")["id"] == "ariel-search-0"

    page.evaluate("location.hash = '#q=beam'")

    page.wait_for_function(
        "window.__contributions.at(-1).items[0].id === 'ariel-search-1'"
        " && window.__contributions.at(-1).items[0].value === 'beam'"
    )
    expect(page.locator("#ariel-list .results-count")).to_be_visible()
    page.close()


def test_tile_bar_compose_opens_the_form_over_the_list(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Compose from the tile bar opens the create form; Cancel returns to the list."""
    page, _ = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true")

    _action(page, "ariel-compose")
    expect(page.locator("#ariel-compose")).to_be_visible()
    expect(page.locator("#entry-subject")).to_be_visible()
    page.click("#create-cancel-btn")
    expect(page.locator("#ariel-compose")).to_be_hidden()
    page.close()


def test_menu_toggles_the_rail_and_relabels(ariel_url: str, chromium_browser: Browser) -> None:
    """⋯ → sidebar toggles the rail, and the entry's label follows the state."""
    page, _ = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true")

    _action(page, "ariel-more", "toggle-rail")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    page.wait_for_function(
        "window.__contributions.at(-1).items[2].items[0].label === 'Hide sidebar'"
    )

    _action(page, "ariel-more", "toggle-rail")
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
    page.wait_for_function(
        "window.__contributions.at(-1).items[2].items[0].label === 'Show sidebar'"
    )
    page.close()


def test_menu_save_search_opens_the_name_field(ariel_url: str, chromium_browser: Browser) -> None:
    """⋯ → Save search expands the rail and asks for a name; the saved search re-keys."""
    page, _ = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true")
    _action(page, _item(page, "search")["id"], "beam")
    wait_for_hash(page, canonical(q="beam"))

    _action(page, "ariel-more", "save-search")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    name = page.locator('[data-rail-field="save"]')
    expect(name).to_be_visible()
    name.fill("Beam")
    name.press("Enter")
    saved = page.locator('#ariel-rail button[data-box="saved:Beam"]')
    expect(saved).to_be_visible()

    # Running it changes the query from the panel's side: the field re-keys.
    _action(page, _item(page, "search")["id"], "other")
    wait_for_hash(page, canonical(q="other"))
    key = _item(page, "search")["id"]
    saved.click()
    page.wait_for_function(
        "(key) => window.__contributions.at(-1).items[0].id !== key"
        " && window.__contributions.at(-1).items[0].value === 'beam'",
        arg=key,
    )
    page.close()


def test_invalid_configuration_omits_the_search_item(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """With search blocked the tile bar carries Compose and the menu only."""
    page, _ = _open_embedded(
        chromium_browser,
        f"{ariel_url}/?embedded=true",
        FakeBackend(capabilities=dict(INVALID_CAPABILITIES)),
    )
    expect(page.locator("#config-invalid-banner")).to_be_visible()
    assert [item["kind"] for item in _items(page)] == ["button", "menu"]
    page.close()


def test_slash_is_left_to_the_host(ariel_url: str, chromium_browser: Browser) -> None:
    """Embedded, ``/`` moves no focus: the search box is the host's."""
    page, _ = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true")
    page.keyboard.press("/")
    assert page.evaluate("document.activeElement === document.body")
    page.close()


# ---------------------------------------------------------------------------
# Every legacy alias, embedded
# ---------------------------------------------------------------------------


def test_embedded_entry_alias_opens_reader_and_lightbox(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``#entry?id=&attachment=`` selects the entry and opens the picture."""
    page, _ = _open_embedded(
        chromium_browser, f"{ariel_url}/?embedded=true#entry?id=E1&attachment=A1"
    )
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    expect(page.locator("#image-lightbox")).to_be_visible()
    wait_for_hash(page, canonical(sel="E1", att="A1"))
    page.close()


def test_embedded_unknown_entry_shows_the_reader_error(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """An unknown id renders the reader's failure message, embedded as standalone."""
    page, _ = _open_embedded(
        chromium_browser, f"{ariel_url}/?embedded=true#entry?id=no-such-entry&attachment=att-x"
    )
    expect(page.locator("#ariel-reader-body")).to_contain_text("Failed to Load Entry")
    expect(page.locator("#image-lightbox")).to_have_count(0)
    page.close()


@pytest.mark.parametrize(
    ("alias", "subject"), [("#create", ""), ("#create?draft=D1", "Drafted by the agent")]
)
def test_embedded_create_alias_opens_compose(
    ariel_url: str, chromium_browser: Browser, alias: str, subject: str
) -> None:
    """``#create`` and ``#create?draft=`` open the form embedded."""
    page, _ = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true{alias}")
    expect(page.locator("#ariel-compose")).to_be_visible()
    expect(page.locator("#entry-subject")).to_have_value(subject)
    page.close()


@pytest.mark.parametrize("alias", ["#search", "#browse", "#status"])
def test_embedded_list_aliases_show_the_list(
    ariel_url: str, chromium_browser: Browser, alias: str
) -> None:
    """``#search``, ``#browse`` and (embedded) ``#status`` all land on the list."""
    page, _ = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true{alias}")
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(3)
    expect(page.locator("osprey-drawer#status-drawer[open]")).to_have_count(0)
    expect(page.locator("osprey-drawer#settings-drawer[open]")).to_have_count(0)
    wait_for_hash(page, canonical())
    page.close()


def test_standalone_keeps_its_header_and_contributes_nothing(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Without ``?embedded=true`` the panel keeps its header, box and compose button."""
    page = chromium_browser.new_page()
    page.add_init_script(_RECORD_CONTRIBUTIONS)
    FakeBackend().install(page)
    page.goto(f"{ariel_url}/", wait_until="load")

    expect(page.locator(".header")).to_be_visible()
    expect(page.locator("#ariel-gear-btn")).to_be_visible()
    expect(page.locator("#ariel-search-input")).to_be_visible()
    expect(page.locator("#ariel-compose-btn")).to_be_visible()
    expect(row(page, "E1")).to_be_visible()
    assert page.evaluate("window.__contributions.length") == 0
    page.close()


def test_tile_bar_operator_is_lifted_after_settle(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``vacuum author:x`` with no trailing space is lifted once the text settles."""
    page, backend = _open_embedded(chromium_browser, f"{ariel_url}/?embedded=true")
    _action(page, _item(page, "search")["id"], "vacuum author:x")

    call = wait_for_call(
        page,
        backend,
        "POST",
        "/api/search",
        lambda r: r.filters().get("author") == ["x"],
        timeout=8.0,
    )
    assert call.json()["query"] == "vacuum"
    wait_for_hash(page, canonical(q="vacuum", f={"author": ["x"]}))
    page.close()
