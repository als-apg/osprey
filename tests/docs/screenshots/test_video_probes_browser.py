"""Browser test: the rotate probe reads the camera the viewer sees.

A drag that ends over the colorbar releases the mouse on a ``div``, not on the
3D canvas. The scene has turned, but Plotly only copies the camera into
``_fullLayout`` on a canvas ``mouseup``, so a probe reading the layout would
report no rotation for a plot that visibly rotated.

Skips cleanly when the Chromium headless binary is not installed; it never
downloads one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docs.screenshots import video_landing, video_probes

from tests._playwright import chromium_available

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# How long a test waits for the state it asserts: far above a healthy browser's
# reaction, so only a page that never gets there fails.
_DEADLINE_MS = 10_000


def _scatter3d_page(tmp_path: Path) -> Path:
    go = pytest.importorskip("plotly.graph_objects")
    fig = go.Figure(
        go.Scatter3d(
            x=list(range(50)),
            y=[i % 7 for i in range(50)],
            z=[i % 11 for i in range(50)],
            mode="markers",
            marker={"color": list(range(50)), "colorbar": {"title": "t"}},
        )
    )
    fig.update_layout(width=800, height=800, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    path = tmp_path / "plot.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def test_a_drag_released_over_the_colorbar_still_reads_as_rotation(tmp_path):
    if not chromium_available():
        pytest.skip("Playwright Chromium is not installed")
    from playwright.sync_api import sync_playwright

    page_path = _scatter3d_page(tmp_path)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 800, "height": 800})
            page.goto(page_path.as_uri())  # page-ready: a file:// fixture, not an OSPREY page
            page.wait_for_function("() => !!document.querySelector('.js-plotly-plot')._fullLayout")
            page.wait_for_timeout(500)
            before = video_probes.camera_azimuth(page.main_frame)
            # From the scene's centre to the colorbar at the right edge.
            page.mouse.move(300, 400)
            page.mouse.down()
            for step in range(1, video_landing.DRAG_STEPS + 1):
                page.mouse.move(300 + 490 * step / video_landing.DRAG_STEPS, 400)
            page.mouse.up()
            waited = 0
            while True:
                page.wait_for_timeout(video_landing.DRAG_SETTLE_MS)
                waited += video_landing.DRAG_SETTLE_MS
                after = video_probes.camera_azimuth(page.main_frame)
                if video_probes.azimuth_change(before, after) >= 45 or waited >= _DEADLINE_MS:
                    break
        finally:
            browser.close()
    assert video_probes.azimuth_change(before, after) >= 45


# The hub's session-tree artifact row (inside an open session's box), and the
# embedded viewer's preview pane as the gallery's renderPreview writes it. The
# probes run the row check on the hub and the preview checks inside the
# artifacts iframe; one page holds both.
_PANE = """
<div class="sidebar-kids osprey-tree" id="sidebar-session-tree" data-session-tree role="tree">
  <div class="osprey-tree-box" role="group" data-session-box="s1">
    <button class="osprey-list-row osprey-tree-d2" role="treeitem" data-id="abc123"
            aria-selected="{selected}">
      <span class="osprey-list-title">Channels</span>
    </button>
  </div>
</div>
<div id="browse-preview-pane">
  <div id="preview-empty">No artifact selected</div>
  <div id="preview-content" class="{hidden}">
    <div class="preview-viewport">{viewport}</div>
  </div>
</div>
"""


# The archiver card's time-series viewer: its badges and toolbar appear at
# once, the Plotly chart is drawn into ``[data-ts-chart]`` afterwards.
_TS = (
    '<div class="ts-viewport-container"><div class="ts-viewer">'
    '<div class="ts-info-bar"><span class="ts-badge">CH</span></div>'
    '<div class="ts-chart-container" data-ts-chart>{chart}</div></div></div>'
)
_TS_LOADING = '<div class="ts-viewport-container"><div class="ts-loading">Loading timeseries data...</div></div>'
_DRAWN = (
    '<div class="scatterlayer"><g class="trace"></g></div>'
    "<script>document.querySelector('[data-ts-chart]')._fullLayout = {};</script>"
)


@pytest.mark.parametrize(
    ("hidden", "viewport", "expected"),
    [
        ("hidden", '<div class="md-preview-container"><p>x</p></div>', False),
        ("", '<div class="md-preview-container"></div>', False),
        ("", '<div class="md-preview-container"><p>SR BPM 1</p></div>', True),
        ("", '<div class="json-viewer">{"a": 1}</div>', True),
        ("", '<img src="data:," alt="broken">', False),
        ("", '<iframe srcdoc="<div>plot</div>"></iframe>', True),
        ("", '<iframe srcdoc=""></iframe>', False),
        ("", '<iframe sandbox="allow-scripts" srcdoc="<div>plot</div>"></iframe>', True),
        ("", '<iframe src="/files/other9/plot.html" srcdoc="<div>plot</div>"></iframe>', False),
        ("", '<iframe src="/files/abc123/plot.html" srcdoc="<div>plot</div>"></iframe>', True),
        ("", _TS.format(chart=""), False),
        ("", _TS_LOADING, False),
        ("", _TS.format(chart=_DRAWN), True),
    ],
    ids=[
        "pane-hidden",
        "empty-markdown",
        "markdown",
        "json",
        "broken-image",
        "loaded-iframe",
        "empty-iframe",
        "sandboxed-iframe",
        "other-artifact-file",
        "own-file",
        "timeseries-controls-only",
        "timeseries-loading",
        "timeseries-drawn",
    ],
)
def test_the_preview_probe_waits_for_the_clicked_artifact_to_render(hidden, viewport, expected):
    if not chromium_available():
        pytest.skip("Playwright Chromium is not installed")
    from playwright.sync_api import sync_playwright

    html = _PANE.format(hidden=hidden, selected="true", viewport=viewport)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html)
            frame = page.main_frame
            waited = 0
            while video_probes.preview_rendered(frame, "abc123") is not expected:
                assert waited < _DEADLINE_MS
                page.wait_for_timeout(video_probes.POLL_MS)
                waited += video_probes.POLL_MS
            page.wait_for_timeout(video_probes.POLL_MS)
            assert video_probes.preview_rendered(frame, "abc123") is expected
            assert frame.evaluate(video_probes._CARD_SHOWN_JS, "abc123") is True
            assert frame.evaluate(video_probes._CARD_SHOWN_JS, "zzz") is False
            assert frame.evaluate(video_probes._PREVIEW_STATE_JS).startswith("preview: ")
            assert frame.evaluate(video_probes._ROW_SELECTED_JS, "abc123") is True
            page.locator('[data-id="abc123"]').evaluate(
                "(row) => row.setAttribute('aria-selected', 'false')"
            )
            assert frame.evaluate(video_probes._ROW_SELECTED_JS, "abc123") is False
        finally:
            browser.close()


def test_the_closing_hold_scrolls_the_attachment_into_view():
    if not chromium_available():
        pytest.skip("Playwright Chromium is not installed")
    from playwright.sync_api import sync_playwright

    html = (
        '<div style="height:600px;overflow:auto" id="form">'
        '<div style="height:2000px">fields</div>'
        '<div id="file-preview"><img alt="plot" style="width:200px;height:120px" '
        'src="data:image/gif;base64,R0lGODlhAQABAAAAACw="></div></div>'
    )
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 800, "height": 600})
            page.set_content(html)
            assert page.evaluate(video_probes._REVEAL_ATTACHMENT_JS) is True
            page.wait_for_function(
                "() => { const t = document.querySelector('#file-preview img')"
                ".getBoundingClientRect().top; return t >= 0 && t < 600; }",
                timeout=_DEADLINE_MS,
            )
            page.set_content("<div>no draft</div>")
            assert page.evaluate(video_probes._REVEAL_ATTACHMENT_JS) is False
        finally:
            browser.close()
