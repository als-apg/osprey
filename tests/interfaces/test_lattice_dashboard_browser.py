"""Browser test: the lattice dashboard over a render's simulator view.

Each page is served by the real dashboard app over a render root: the built
control-assistant render, whose view serves the SR deck; a synthetic render
whose view serves the texture model alone; and a render with no simulator
view. A real Chromium page is the oracle for what the operator sees: the
optics plot and tune readout of a selected model, the model selector's
entries and the banner above the figures.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from osprey.interfaces.lattice_dashboard.catalog import NO_SERVED_MODEL_TEXT, NO_VIEW_TEXT
from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.sync_api import Browser

    from tests._builds import BuiltProject

pytestmark = [pytest.mark.browser, pytest.mark.slow]

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

#: The element count of the control-assistant SR deck.
SR_ELEMENTS = 802

#: Upper bound for a figure worker to finish on the full SR deck.
FIGURE_TIMEOUT_MS = 60_000

# The page's own origin posts the select, as the selector does.
_SELECT = """
async (name) => {
  const response = await fetch('/api/models/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  return { status: response.status, body: await response.json() };
}
"""

_OPTICS_POINTS = """
() => {
  const el = document.getElementById('plot-optics');
  return el && el.calcdata && el.calcdata.length ? el.calcdata[0].length : 0;
}
"""


def _serve(render_root: Path | None, workspace: Path):
    from osprey.interfaces.lattice_dashboard.app import create_app

    return _run_app_server(create_app(workspace_root=workspace, render_root=render_root))


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_selected_sr_draws_its_optics_and_tunes(
    tmp_path: Path, built_control_assistant: BuiltProject, chromium_browser: Browser
) -> None:
    """An explicit select of SR draws the optics over every deck element and the tunes."""
    with _serve(built_control_assistant.build_dir, tmp_path / "ws") as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="load")
        expect(page.locator("#model-select option")).not_to_have_count(0, timeout=10_000)

        selected = page.evaluate(_SELECT, "SR")
        assert selected["status"] == 200, selected
        assert selected["body"]["model"] == "SR"

        # The optics solve samples each element's entrance and the lattice end.
        page.wait_for_function(
            f"({_OPTICS_POINTS})() === {SR_ELEMENTS + 1}", timeout=FIGURE_TIMEOUT_MS
        )
        expect(page.locator("#plot-optics .main-svg").first).to_be_visible()

        for chip in ("#stat-nux", "#stat-nuy"):
            expect(page.locator(f"{chip} .stat-value")).to_have_text(
                re.compile(r"^\d+\.\d{4}$"), timeout=FIGURE_TIMEOUT_MS
            )
        expect(page.locator("#model-notice")).to_be_hidden()

        page.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_texture_only_lists_every_deck_model_unserved(
    tmp_path: Path, chromium_browser: Browser
) -> None:
    """A view serving texture alone lists each deck-bearing model as unserved."""
    from tests.interfaces.lattice_dashboard.test_app import _write_render

    render = _write_render(
        tmp_path / "render",
        served=["texture"],
        models={
            "BOOSTER": {"solve": "periodic"},
            "SPARE": None,
            "SR": {"solve": "periodic"},
        },
    )
    with _serve(render, tmp_path / "ws") as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="load")

        options = page.locator("#model-select option")
        expect(options).to_have_count(2, timeout=10_000)
        assert options.evaluate_all("els => els.map(e => e.value)") == ["BOOSTER", "SR"]
        for text in options.all_text_contents():
            assert "not served" in text, text

        notice = page.locator("#model-notice")
        expect(notice).to_be_visible()
        expect(notice).to_have_text(NO_SERVED_MODEL_TEXT)

        page.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_a_render_without_the_view_says_so(tmp_path: Path, chromium_browser: Browser) -> None:
    """A render with no simulator view shows the no-view banner."""
    render = tmp_path / "render"
    render.mkdir()
    with _serve(render, tmp_path / "ws") as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="load")

        notice = page.locator("#model-notice")
        expect(notice).to_be_visible(timeout=10_000)
        expect(notice).to_have_text(NO_VIEW_TEXT)

        page.close()
