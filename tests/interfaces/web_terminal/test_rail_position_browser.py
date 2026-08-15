"""Browser tests: the rail-position axis, end to end.

``?rail=top`` (or ``web.rail_position: top`` / a stored choice — one ladder,
resolved pre-paint by rail-boot.js) renders the SAME rail DOM as a horizontal
strip under the header instead of the default left column. That flip is pure
CSS off ``html[data-rail-position]``, so none of it is reachable from the
FastAPI TestClient — only a real browser computes the flex direction and
proves the strip actually lies horizontally.

Coverage:

  (1) ``?rail=top`` → the attribute is stamped and .panel-rail computes
      ``flex-direction: row``, with the rail-button contract selectors intact.
  (2) no query → the default left column (``flex-direction: column``).

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_rail_position_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _apply_all, _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


@contextmanager
def _hub_server(workspace_dir: Path) -> Iterator[str]:
    """Launch a real web-terminal hub with just the artifacts panel enabled.

    The companion backends are bypassed via the same patches every hub browser
    suite uses; the artifacts panel reports the standard fallback URL so the
    rail renders without a real backend process.

    Yields:
        The hub's base URL.
    """
    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(),
        ),
    ]
    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url


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


def test_rail_top_renders_horizontal_strip(tmp_path, chromium_browser):
    """?rail=top stamps the attribute and lays the rail out as a row.

    The rail-button contract selector must survive the flip — the top strip
    is the same DOM restyled, not a parallel widget.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, f"{base_url}/?rail=top")
        try:
            assert page.get_attribute("html", "data-rail-position") == "top"
            assert _rail_flex_direction(page) == "row"
        finally:
            page.close()


def test_rail_defaults_to_left_column(tmp_path, chromium_browser):
    """Without a query the server-stamped default keeps the left column."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            assert page.get_attribute("html", "data-rail-position") == "left"
            assert _rail_flex_direction(page) == "column"
        finally:
            page.close()
