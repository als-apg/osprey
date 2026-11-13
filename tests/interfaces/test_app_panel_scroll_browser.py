"""Browser Playwright suite: an app panel taller than its tile can be scrolled.

The design system's ``base.css`` pins ``html, body`` to the tile and hides
their overflow, so a panel page scrolls only if its own root does. A root that
merely grows (``min-height: 100vh``) leaves everything below the tile's edge
out of reach.

Each bundled panel, and the scaffold ``osprey app new`` copies, is served here
with the real design-system stylesheets and no backend: the layout under test
is the stylesheet's, whatever the panel's script drew.

Run:
    .venv/bin/pytest tests/interfaces/test_app_panel_scroll_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import osprey
from osprey.interfaces._serving import run_app_server

if TYPE_CHECKING:
    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_SRC = Path(osprey.__file__).parent
_DESIGN_SYSTEM = _SRC / "interfaces" / "design_system" / "static"
_PANELS = sorted(
    [*(_SRC / "profiles" / "presets" / "apps").glob("*/panel"), _SRC / "apps/scaffold-app/panel"]
)

# A short tile, and content well past its lower edge.
_TILE = {"width": 800, "height": 320}
_TALL_PX = 2400


def _panel_page(panel: Path) -> FastAPI:
    app = FastAPI()
    app.mount("/design-system", StaticFiles(directory=_DESIGN_SYSTEM))
    app.mount("/static", StaticFiles(directory=panel))
    app.get("/", response_class=HTMLResponse)(lambda: (panel / "index.html").read_text("utf-8"))
    return app


def test_every_app_panel_is_found():
    assert {panel.parent.name for panel in _PANELS} >= {"lattice", "orbit-bump", "scaffold-app"}


@pytest.mark.parametrize("panel", _PANELS, ids=lambda panel: panel.parent.name)
def test_content_below_the_tile_is_reached_by_scrolling(chromium_browser: Browser, panel: Path):
    with run_app_server(_panel_page(panel)) as base:
        page = chromium_browser.new_page(viewport=_TILE)
        page.goto(f"{base}/")
        page.locator("main").wait_for()
        page.evaluate(
            """(tall) => {
              const root = document.querySelector('main');
              const filler = document.createElement('div');
              filler.style.height = `${tall}px`;
              const last = document.createElement('p');
              last.id = 'last-line';
              last.textContent = 'last line';
              root.append(filler, last);
            }""",
            _TALL_PX,
        )
        last = page.locator("#last-line")
        assert last.bounding_box()["y"] > _TILE["height"]

        page.mouse.move(_TILE["width"] / 2, _TILE["height"] / 2)
        page.mouse.wheel(0, _TALL_PX * 2)

        page.wait_for_function(
            "(edge) => document.getElementById('last-line').getBoundingClientRect().bottom <= edge",
            arg=_TILE["height"],
            timeout=3000,
        )
