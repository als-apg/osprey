"""Browser test: lattice Plotly figures re-theme live on a theme switch.

Regression guard for the "dark-locked plots" class of bug (the audit's live-
reswitch findings): a Plotly figure's chrome AND its opt-in themed marker must
follow a theme toggle without a reload. render.js wires this through the
design-system theme ``subscribe()`` — this exercises the whole path in a real
browser (module-cached ``render.js`` instance, real ``osprey-theme-switcher``
toggle, real ``Plotly.relayout``/``restyle``), which a TestClient can't see.

It replaces the standalone tuning-panel live-reswitch test that was removed
when the tuning panel was retired, repointed at the lattice ``resonance``
figure (whose worker tags the working-point star ``meta='themed-fg-marker'``).

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pytest

from tests.interfaces.test_load_smokes import _launch_lattice_dashboard

pytestmark = [pytest.mark.browser, pytest.mark.slow]

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

# A minimal resonance-shaped figure: one working-point star tagged for the
# theme follower, plus an uncolored paper annotation (the "Area = …" overlay).
_RENDER_FIGURE = """
async () => {
  const mod = await import(new URL('/static/js/render.js', window.location.origin));
  mod.renderPlotly('resonance', {
    data: [{
      x: [0.2], y: [0.3], mode: 'markers',
      marker: { size: 14, symbol: 'star' },
      meta: 'themed-fg-marker',
    }],
    layout: {
      annotations: [{
        text: 'Area = 1.0', xref: 'paper', yref: 'paper',
        x: 0.02, y: 0.98, showarrow: false, font: { size: 13 },
      }],
    },
  });
  const el = document.getElementById('plot-resonance');
  return { marker: el.data[0].marker.color, paper: el._fullLayout.paper_bgcolor };
}
"""

_READ_STATE = """
() => {
  const el = document.getElementById('plot-resonance');
  return { marker: el.data[0].marker.color, paper: el._fullLayout.paper_bgcolor };
}
"""


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_resonance_figure_rethemes_live_on_toggle(tmp_path, monkeypatch, chromium_browser):
    """A rendered resonance figure's star marker and paper re-theme on a live toggle.

    Seeds the dark theme, renders the figure (so the app's ``subscribe()`` — set
    up over ``ALL_FIGURES`` at load — owns ``plot-resonance``), captures the dark
    marker/paper colors, toggles the hub to light via the switcher, and asserts
    both colors actually changed rather than staying dark-locked.
    """
    with _launch_lattice_dashboard(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(f"{base_url}?theme=dark", wait_until="load")

        # The renderer + its theme subscriber are wired at module load; the plot
        # container exists once createUI(ALL_FIGURES) has run.
        page.wait_for_selector("#plot-resonance", timeout=10_000)
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)

        dark = page.evaluate(_RENDER_FIGURE)
        assert dark["marker"], "themed star marker should resolve to a non-empty dark foreground"
        assert dark["paper"], "resonance paper_bgcolor should resolve in the dark theme"

        # Toggle the theme to light via the real switcher control.
        page.click("osprey-theme-switcher .theme-switcher-mode")
        page.wait_for_function(
            "document.documentElement.getAttribute('data-theme') === 'light'", timeout=5_000
        )

        # The subscribe() handler must re-drive BOTH the plot chrome and the
        # opt-in star marker — not just at first render.
        page.wait_for_function(
            "(dark) => { const el = document.getElementById('plot-resonance');"
            " return el && el.data[0].marker.color !== dark.marker"
            " && el._fullLayout.paper_bgcolor !== dark.paper; }",
            arg=dark,
            timeout=5_000,
        )

        light = page.evaluate(_READ_STATE)
        assert light["marker"] != dark["marker"], "star marker stayed dark-locked after switching to light"
        assert light["paper"] != dark["paper"], "resonance paper stayed dark-locked after switching to light"

        page.close()
