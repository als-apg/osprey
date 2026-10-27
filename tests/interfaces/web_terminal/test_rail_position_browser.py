"""Browser tests: the rail-position axis, end to end.

``?rail=top`` (or ``web.rail_position: top`` / a stored choice — one ladder,
resolved pre-paint by rail-boot.js) renders the SAME rail DOM as a horizontal
strip under the header instead of the default left column. That flip is pure
CSS off ``html[data-rail-position]``, so none of it is reachable from the
FastAPI TestClient — only a real browser computes the flex direction and
proves the strip actually lies horizontally.

Coverage, one parametrized test:

  * ``?rail=top`` → the attribute is stamped and .panel-rail computes
    ``flex-direction: row``, with the rail-button contract selectors intact.
  * no query → the default left column (``flex-direction: column``).

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_rail_position_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pytest

from tests.interfaces._hub import hub_server

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def _open_hub_page(browser: Browser, url: str) -> Page:
    """Open a hub URL and wait for the rail to be populated (panel init done)."""
    page = browser.new_page()
    page.goto(url, wait_until="domcontentloaded")
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    return page


def _rail_flex_direction(page: Page) -> str:
    return page.evaluate("getComputedStyle(document.querySelector('.panel-rail')).flexDirection")


@pytest.mark.parametrize(
    ("query", "position", "direction"),
    [("/?rail=top", "top", "row"), ("", "left", "column")],
    ids=["top", "default"],
)
def test_rail_position_lays_out_the_rail(tmp_path, chromium_browser, query, position, direction):
    """The stamped position decides the rail's computed flex direction.

    ``?rail=top`` lays the rail out as a row under the header; with no query the
    server-stamped default keeps the left column. Both are the same rail DOM
    restyled — the rail-button contract selector the page waits on survives
    either way, so the top strip is never a parallel widget.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with hub_server(workspace) as (base_url, _app):
        page = _open_hub_page(chromium_browser, f"{base_url}{query}")
        try:
            assert page.get_attribute("html", "data-rail-position") == position
            assert _rail_flex_direction(page) == direction
        finally:
            page.close()
