"""Browser Playwright suite: a served Plotly artifact follows the OSPREY theme.

Regression pins for the dark-locked-artifact bug. The theme bridge the gallery
injects into ``plot_html`` artifacts used to carry a private ``{dark, light}``
hex palette and read ``window.parent``'s ``data-theme`` with a hardcoded
``'dark'`` fallback, so:

  - a plot opened in its own tab ("Open in new tab") had no parent and always
    rendered dark, whatever the operator's theme or OS preference;
  - every theme outside the ``main`` family (``high-contrast-light``,
    ``retro-light``, ``desy-light``, ...) missed the two-entry map and fell
    to the dark palette even inside the gallery.

These tests drive the real served page in a real browser and check the colors
Plotly actually applied against the ``--chart-*`` tokens of the theme in
force, for a figure whose author baked a dark palette into the layout (the
shape of the reported artifact: a 3D scatter with dark ``paper_bgcolor``,
``scene.bgcolor`` and white text).

Every page here is opened through the web-terminal hub's panel proxy
(``/panel/artifacts/…``) with the hub's auth gate on, the way a deployment's
browser reaches it. The artifact pages are served opaque (sandboxed, origin
``"null"``), so the design-system modules and the Plotly bundle they import
must come through the C-ASSETS routes without a cookie; a path that list leaves
out leaves the plot unthemed here.

Run:
    .venv/bin/pytest tests/interfaces/artifacts/test_plot_theme_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from tests.interfaces.artifacts.test_opaque_origin_browser import (
    GALLERY,
    artifact_frame,
    hub_with_gallery,
    open_gallery,
    select_artifact,
    set_gallery_theme,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Frame, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# A Plotly page the way ``go.Figure.to_html(include_plotlyjs=False)`` writes
# one, with the author's dark palette baked into the layout. A 2D trace plus
# a 3D scene so both the cartesian and the scene branches of the bridge are
# exercised; the 3D trace is what the reported artifact used. The scene's
# axis panes are switched on the way plotly.py's default template does it --
# Plotly only carries ``backgroundcolor`` into the full layout for an axis
# whose ``showbackground`` is on, so this is what makes the pane assertion
# observable.
_DARK_BAKED_PLOT = """<html>
<head><meta charset="utf-8" /></head>
<body>
<div style="height:600px; width:800px;">
<div id="plot-a" class="plotly-graph-div" style="height:100%; width:100%;"></div>
<script>
Plotly.newPlot("plot-a",
  [{"type":"scatter","x":[0,1,2],"y":[1,3,2],"mode":"lines"}],
  {"paper_bgcolor":"rgba(15, 15, 25, 1)","plot_bgcolor":"rgba(15, 15, 25, 1)",
   "font":{"color":"white"},"width":800,"height":600});
</script>
<div id="plot-b" class="plotly-graph-div" style="height:100%; width:100%;"></div>
<script>
Plotly.newPlot("plot-b",
  [{"type":"scatter3d","x":[0,1],"y":[0,1],"z":[0,1],"mode":"markers"}],
  {"paper_bgcolor":"rgba(15, 15, 25, 1)",
   "scene":{"bgcolor":"rgba(10, 10, 20, 0.95)",
            "xaxis":{"showbackground":true},"yaxis":{"showbackground":true},
            "zaxis":{"showbackground":true}},
   "font":{"color":"white"},"width":800,"height":600});
</script>
</div>
</body>
</html>"""

# JS that reports what Plotly actually applied next to the tokens the page's
# own stylesheet resolves for the theme in force -- one read, one object.
_PROBE = """() => {
  const css = getComputedStyle(document.documentElement);
  const tok = (n) => css.getPropertyValue(n).trim();
  const a = document.getElementById('plot-a'), b = document.getElementById('plot-b');
  const fa = a && a._fullLayout, fb = b && b._fullLayout;
  return {
    theme: document.documentElement.getAttribute('data-theme'),
    tokens: { paper: tok('--chart-paper-bg'), plot: tok('--chart-plot-bg'),
              text: tok('--chart-axis-text'), pane: tok('--chart-pane-bg') },
    a: fa && { paper: fa.paper_bgcolor, plot: fa.plot_bgcolor, font: fa.font.color,
               visible: a.style.visibility },
    b: fb && { paper: fb.paper_bgcolor, font: fb.font.color,
               scene: fb.scene && fb.scene.bgcolor,
               pane: fb.scene && fb.scene.xaxis && fb.scene.xaxis.backgroundcolor,
               hasCartesianAxis: 'xaxis' in b.layout },
    bodyBg: getComputedStyle(document.body).backgroundColor,
  };
}"""


def _save(workspace, content: bytes, filename: str, artifact_type: str, title: str):
    from osprey.stores.artifact_store import ArtifactStore

    return ArtifactStore(workspace_root=workspace).save_file(
        file_content=content,
        filename=filename,
        artifact_type=artifact_type,
        title=title,
        description="fixture",
        mime_type="text/html",
        tool_source="test_fixture",
        category="visualization",
    )


PLOT_TITLE = "Dark-baked 3D correlation"


@contextmanager
def _launch(tmp_path, monkeypatch, *, web_theme: str | None = None) -> Iterator[tuple[str, str]]:
    """Serve the gallery behind the hub with one dark-baked ``plot_html`` seeded.

    Yields ``(base_url, file_path)``: the hub's address and the artifact's
    proxied ``/panel/artifacts/files/...`` route. The hub serves the Plotly
    bundle offline, from the artifacts app's vendor tree through the proxy
    (``osprey vendor fetch`` has run on the gate host).
    """

    def seed(workspace):
        entry = _save(
            workspace, _DARK_BAKED_PLOT.encode(), "dark-baked.html", "plot_html", PLOT_TITLE
        )
        return entry.id, f"{GALLERY}files/{entry.id}/{entry.filename}"

    with hub_with_gallery(tmp_path, monkeypatch, seed=seed, web_theme=web_theme) as (
        base_url,
        (_artifact_id, path),
    ):
        yield base_url, path


def _probe(page: Page | Frame) -> dict:
    # The bridge reveals a chart only after its relayout resolved; wait for
    # that rather than for a fixed delay.
    page.wait_for_function(
        "() => { const a = document.getElementById('plot-a');"
        " return a && a._fullLayout && a.style.visibility === 'visible'; }",
        timeout=15_000,
    )
    return page.evaluate(_PROBE)


def _assert_follows_tokens(probe: dict, expected_theme: str) -> None:
    assert probe["theme"] == expected_theme
    tokens = probe["tokens"]
    assert tokens["paper"], "tokens.css did not resolve on the served page"
    assert probe["a"]["paper"] == tokens["paper"]
    assert probe["a"]["plot"] == tokens["plot"]
    assert probe["a"]["font"] == tokens["text"]
    assert probe["bodyBg"] != "rgba(0, 0, 0, 0)", "page background must be painted"
    # The 3D scene: box and axis panes re-themed, and the bridge must not
    # have conjured a cartesian axis onto a scene-only figure.
    assert probe["b"]["paper"] == tokens["paper"]
    assert probe["b"]["scene"] == tokens["plot"]
    assert probe["b"]["pane"] == tokens["pane"]
    assert probe["b"]["hasCartesianAxis"] is False


def test_standalone_page_follows_query_theme_outside_main_family(
    tmp_path, monkeypatch, chromium_browser
):
    """No parent frame, a non-main light theme requested: the plot must take
    that theme's tokens, not a dark fallback."""
    with _launch(tmp_path, monkeypatch) as (base_url, path):
        page = chromium_browser.new_page()
        page.goto(f"{base_url}{path}?theme=high-contrast-light")
        _assert_follows_tokens(_probe(page), "high-contrast-light")


def test_standalone_page_follows_os_light_preference(tmp_path, monkeypatch, chromium_browser):
    """Nothing stored, nothing requested, OS prefers light: the plot is light."""
    with _launch(tmp_path, monkeypatch) as (base_url, path):
        ctx = chromium_browser.new_context(color_scheme="light")
        page = ctx.new_page()
        page.goto(f"{base_url}{path}")
        _assert_follows_tokens(_probe(page), "light")


def test_standalone_page_honors_pinned_web_theme(tmp_path, monkeypatch, chromium_browser):
    """A deployment pin (``web.theme: light``) wins over a dark OS preference
    on a first visit, exactly as it does for the terminal itself."""
    with _launch(tmp_path, monkeypatch, web_theme="light") as (base_url, path):
        ctx = chromium_browser.new_context(color_scheme="dark")
        page = ctx.new_page()
        page.goto(f"{base_url}{path}")
        _assert_follows_tokens(_probe(page), "light")


def test_unstyled_html_artifact_gets_themed_defaults(tmp_path, monkeypatch, chromium_browser):
    """A served table/html artifact with no styling of its own paints the
    theme's background and text color (zero-specificity defaults), instead of
    browser-default black-on-white."""

    def seed(workspace):
        from osprey.stores.artifact_store import ArtifactStore

        entry = ArtifactStore(workspace_root=workspace).save_file(
            file_content=(
                b'<html><head></head><body><table class="artifact-table">'
                b"<tr><th>ch</th></tr><tr><td>SR01C:HCM1</td></tr></table></body></html>"
            ),
            filename="readings.html",
            artifact_type="table_html",
            title="Readings",
            description="fixture",
            mime_type="text/html",
            tool_source="test_fixture",
        )
        return f"{GALLERY}files/{entry.id}/{entry.filename}"

    with hub_with_gallery(tmp_path, monkeypatch, seed=seed) as (base_url, path):
        page = chromium_browser.new_page()
        page.goto(f"{base_url}{path}?theme=dark")
        page.wait_for_function(
            "() => document.documentElement.getAttribute('data-theme') === 'dark'"
        )
        probe = page.evaluate(
            """() => {
              const css = getComputedStyle(document.documentElement);
              return {
                bgToken: css.getPropertyValue('--bg-primary').trim(),
                htmlBg: getComputedStyle(document.documentElement).backgroundColor,
                bodyColor: getComputedStyle(document.body).color,
                textToken: css.getPropertyValue('--text-primary').trim(),
              };
            }"""
        )
        assert probe["bgToken"], "tokens.css did not resolve"

        def _rgb(hex_color: str) -> str:
            hex_color = hex_color.lstrip("#")
            r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgb({r}, {g}, {b})"

        assert probe["htmlBg"] == _rgb(probe["bgToken"])
        assert probe["bodyColor"] == _rgb(probe["textToken"])


def test_embedded_page_retheme_on_hub_broadcast(tmp_path, monkeypatch, chromium_browser):
    """Inside the gallery the plot is an opaque frame. A theme change in the
    gallery is relayed to it (target ``"*"``: an opaque frame cannot be named),
    and the bridge must re-theme live to whatever id arrives -- including one
    outside the main family -- through the same token path. The frame takes the
    message by its source (its parent), since its own origin proves nothing."""
    with _launch(tmp_path, monkeypatch) as (base_url, path):
        page = chromium_browser.new_page()
        open_gallery(page, base_url, theme="dark")
        select_artifact(page, PLOT_TITLE)
        frame = artifact_frame(page, path.split("/")[-2])
        assert frame.evaluate("() => String(self.origin)") == "null"
        _assert_follows_tokens(_probe(frame), "dark")

        set_gallery_theme(page, "retro-light")
        frame.wait_for_function(
            "() => document.documentElement.getAttribute('data-theme') === 'retro-light'"
            " && document.getElementById('plot-a')._fullLayout.paper_bgcolor"
            "    === getComputedStyle(document.documentElement)"
            "         .getPropertyValue('--chart-paper-bg').trim()",
            timeout=15_000,
        )
        _assert_follows_tokens(frame.evaluate(_PROBE), "retro-light")
