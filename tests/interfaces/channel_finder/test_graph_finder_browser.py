"""Browser tests: the graph finder's columns and menus, driven for real.

The column gestures are what the unit suites cannot settle: a header drag is a
pointer gesture with ``setPointerCapture``, its drop slot is decided from live
``getBoundingClientRect()`` boxes, a resize writes a ``<col>`` width that only
a layout engine turns into a wider cell, and the arrangement has to survive a
reload. ``tests/interfaces/channel_finder/*.test.mjs`` pin every decision in
isolation; this suite proves they are wired to a pointer and a paint.

Coverage (one test each, every test in Chromium and in WebKit):

  (a) dragging a header onto another moves the column there, and the order
      survives a reload; the drag sweeps no text selection across the rows.
  (b) dragging a column's edge widens the drawn cell by the distance dragged,
      and the width survives a reload; double-clicking the edge fits it to
      its widest value.
  (c) a click on a header sorts on the server — the first row changes to the
      one that sorts first — and the header says so; the click that ends a
      drag does not also sort.
  (d) a right-click on a row opens the shared context menu inside the window,
      and its "Only section" row narrows the result.
  (e) a device opens in the inspector beside the results, its rows are marked,
      and Escape closes it.

The server is the graph-mode Channel Finder over the shared demo corpus
(``launch_graph_channel_finder``); no graph database is dialled.

Run:
    .venv/bin/pytest tests/interfaces/channel_finder/test_graph_finder_browser.py -m browser -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.interfaces.conftest import launch_graph_channel_finder

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from playwright.sync_api import Browser, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

ROWS = "#graph-finder-table .result-table tbody tr[data-pv]"
VIEWPORT = {"width": 1280, "height": 800}


@pytest.fixture(params=["chromium_browser", "webkit_browser"], ids=["chromium", "webkit"])
def engine_browser(request: pytest.FixtureRequest) -> Browser:
    """Each browser engine in turn; each skips on its own when absent."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def finder_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A graph-mode Channel Finder over the demo corpus."""
    with launch_graph_channel_finder(tmp_path, monkeypatch) as url:
        yield url


def _open(browser: Browser, url: str) -> Page:
    page = browser.new_page(viewport=VIEWPORT)
    page.goto(f"{url}/#explore")
    page.wait_for_selector(ROWS)
    return page


def _columns(page: Page) -> list[str]:
    return page.eval_on_selector_all(
        ".result-table thead th[data-col]", "els => els.map(e => e.dataset.col)"
    )


def _box(page: Page, selector: str) -> dict:
    box = page.locator(selector).bounding_box()
    assert box is not None, f"{selector} is not laid out"
    return box


def test_a_dragged_header_moves_its_column_and_the_order_is_kept(
    engine_browser: Browser, finder_url: str
):
    page = _open(engine_browser, finder_url)
    assert _columns(page)[1:4] == ["device", "section", "address"]

    src = _box(page, '.result-table th[data-col="signal"]')
    dst = _box(page, '.result-table th[data-col="device"]')
    page.mouse.move(src["x"] + 20, src["y"] + src["height"] / 2)
    page.mouse.down()
    page.mouse.move(src["x"] - 20, src["y"] + src["height"] / 2, steps=4)
    page.mouse.move(dst["x"] + 8, dst["y"] + dst["height"] / 2, steps=10)
    assert page.locator(".col-drag-ghost").count() == 1
    assert page.locator(".col-drop-line").count() == 1
    assert page.evaluate("String(getSelection())") == ""
    page.mouse.up()

    assert _columns(page)[1:3] == ["signal", "device"]
    assert page.locator(".col-drag-ghost").count() == 0
    # The drag's closing click did not sort.
    assert page.locator('th[data-col="signal"]').get_attribute("aria-sort") == "none"

    page.reload()
    page.wait_for_selector(ROWS)
    assert _columns(page)[1:3] == ["signal", "device"]


def test_a_dragged_edge_resizes_its_column_and_the_width_is_kept(
    engine_browser: Browser, finder_url: str
):
    page = _open(engine_browser, finder_url)
    before = _box(page, '.result-table th[data-col="address"]')["width"]

    grip = _box(page, '.result-table th[data-col="address"] .col-resize')
    y = grip["y"] + grip["height"] / 2
    page.mouse.move(grip["x"] + grip["width"] / 2, y)
    page.mouse.down()
    page.mouse.move(grip["x"] + grip["width"] / 2 + 90, y, steps=6)
    page.mouse.up()

    after = _box(page, '.result-table th[data-col="address"]')["width"]
    assert after == pytest.approx(before + 90, abs=3)

    page.reload()
    page.wait_for_selector(ROWS)
    kept = _box(page, '.result-table th[data-col="address"]')["width"]
    assert kept == pytest.approx(after, abs=1)

    # Double-clicking the edge fits the column to its widest cell. Narrow it
    # first so every address truncates: a clipped cell's scrollWidth is its
    # full text width, which is what the fit has to reach in any engine.
    grip = _box(page, '.result-table th[data-col="address"] .col-resize')
    y = grip["y"] + grip["height"] / 2
    page.mouse.move(grip["x"] + grip["width"] / 2, y)
    page.mouse.down()
    page.mouse.move(grip["x"] + grip["width"] / 2 - (kept - 80), y, steps=6)
    page.mouse.up()
    narrow = _box(page, '.result-table th[data-col="address"]')["width"]
    widest = page.eval_on_selector_all(
        "#graph-finder-table .result-table tbody td.pv",
        "els => Math.max(...els.map(e => e.scrollWidth))",
    )
    assert widest > narrow

    page.locator('.result-table th[data-col="address"] .col-resize').dblclick()
    fitted = _box(page, '.result-table th[data-col="address"]')["width"]
    assert fitted == pytest.approx(widest + 2, abs=2)


def test_a_header_click_sorts_on_the_server(engine_browser: Browser, finder_url: str):
    page = _open(engine_browser, finder_url)
    first = page.locator(ROWS).first.get_attribute("data-pv")

    page.locator('.result-table th[data-col="section"] .th-label').click()
    page.wait_for_selector('th[data-col="section"][aria-sort="ascending"]')
    page.locator('.result-table th[data-col="section"] .th-label').click()
    page.wait_for_selector('th[data-col="section"][aria-sort="descending"]')
    page.wait_for_function(f"document.querySelector('{ROWS}').dataset.pv !== {first!r}")

    sections = page.eval_on_selector_all(
        "#graph-finder-table .result-table tbody td.sec", "els => els.map(e => e.textContent)"
    )
    assert sections == sorted(sections, reverse=True)


def test_a_row_menu_opens_inside_the_window_and_narrows_the_search(
    engine_browser: Browser, finder_url: str
):
    page = _open(engine_browser, finder_url)
    total = page.locator(".result-counts strong").first.inner_text()

    row = page.locator(ROWS).nth(2)
    section = row.locator("td.sec").inner_text()
    box = row.bounding_box()
    assert box is not None
    # Near the right edge: the menu must flip inward rather than overflow.
    # Aim inside the pane's client box, short of a classic scrollbar (WebKit).
    right = page.eval_on_selector(
        ".result-wrap", "el => el.getBoundingClientRect().left + el.clientLeft + el.clientWidth"
    )
    assert right > VIEWPORT["width"] - 60
    page.mouse.click(right - 12, box["y"] + box["height"] / 2, button="right")
    menu = page.locator(".osprey-context-menu")
    menu.wait_for()
    menu_box = menu.bounding_box()
    assert menu_box is not None
    assert menu_box["x"] + menu_box["width"] <= VIEWPORT["width"]

    menu.locator(".osprey-context-item", has_text=f"Only section {section}").click()
    page.wait_for_selector(f'.active-filter[data-chip="section:{section}"]')
    assert page.locator(".result-counts strong").first.inner_text() != total
    assert set(
        page.eval_on_selector_all(
            "#graph-finder-table .result-table tbody td.sec", "els => els.map(e => e.textContent)"
        )
    ) == {section}


def test_a_device_opens_beside_the_results_and_escape_closes_it(
    engine_browser: Browser, finder_url: str
):
    page = _open(engine_browser, finder_url)

    page.locator("#graph-finder-table td.dev button.dev").first.click()
    card = page.locator(".finder-inspector .device-card")
    card.wait_for()
    results = _box(page, ".finder-results")
    inspector = _box(page, ".finder-inspector")
    assert inspector["x"] >= results["x"] + results["width"]
    assert page.locator("#graph-finder-table tbody tr.open").count() >= 1

    page.keyboard.press("Escape")
    card.wait_for(state="detached")
    assert page.locator("#graph-finder-table tbody tr.open").count() == 0
