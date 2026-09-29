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
from docs.screenshots import video_probes, video_take

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    p = sync_playwright().start()
    try:
        return Path(p.chromium.executable_path).exists()
    finally:
        p.stop()


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
    if not _chromium_available():
        pytest.skip("Playwright Chromium is not installed")
    from playwright.sync_api import sync_playwright

    page_path = _scatter3d_page(tmp_path)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 800, "height": 800})
            page.goto(page_path.as_uri())
            page.wait_for_function("() => !!document.querySelector('.js-plotly-plot')._fullLayout")
            page.wait_for_timeout(500)
            before = video_probes.camera_azimuth(page.main_frame)
            # From the scene's centre to the colorbar at the right edge.
            page.mouse.move(300, 400)
            page.mouse.down()
            for step in range(1, video_take.DRAG_STEPS + 1):
                page.mouse.move(300 + 490 * step / video_take.DRAG_STEPS, 400)
            page.mouse.up()
            page.wait_for_timeout(video_take.DRAG_SETTLE_MS)
            after = video_probes.camera_azimuth(page.main_frame)
        finally:
            browser.close()
    assert video_probes.azimuth_change(before, after) >= 45


# The preview pane's markup as the gallery's renderPreview writes it.
_PANE = """
<div class="tree-item" data-id="abc123"></div>
<div id="preview-empty">Select an artifact to preview</div>
<div id="preview-content" class="{hidden}">
  <div class="preview-header"><span class="preview-header-title">Channels</span></div>
  <span class="preview-path-text">var/agent_data/artifacts/{shown}_channels.md</span>
  <div class="preview-viewport">{viewport}</div>
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
    ("hidden", "shown", "viewport", "expected"),
    [
        ("hidden", "abc123", '<div class="md-preview-container"><p>x</p></div>', False),
        ("", "other9", '<div class="md-preview-container"><p>x</p></div>', False),
        ("", "abc123", '<div class="md-preview-container"></div>', False),
        ("", "abc123", '<div class="md-preview-container"><p>SR BPM 1</p></div>', True),
        ("", "abc123", '<div class="json-viewer">{"a": 1}</div>', True),
        ("", "abc123", '<img src="data:," alt="broken">', False),
        ("", "abc123", '<iframe srcdoc="<div>plot</div>"></iframe>', True),
        ("", "abc123", '<iframe srcdoc=""></iframe>', False),
        ("", "abc123", _TS.format(chart=""), False),
        ("", "abc123", _TS_LOADING, False),
        ("", "abc123", _TS.format(chart=_DRAWN), True),
    ],
    ids=[
        "pane-hidden",
        "other-artifact",
        "empty-markdown",
        "markdown",
        "json",
        "broken-image",
        "loaded-iframe",
        "empty-iframe",
        "timeseries-controls-only",
        "timeseries-loading",
        "timeseries-drawn",
    ],
)
def test_the_preview_probe_waits_for_the_clicked_artifact_to_render(
    hidden, shown, viewport, expected
):
    if not _chromium_available():
        pytest.skip("Playwright Chromium is not installed")
    from playwright.sync_api import sync_playwright

    html = _PANE.format(hidden=hidden, shown=shown, viewport=viewport)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html)
            page.wait_for_timeout(200)
            frame = page.main_frame
            assert frame.evaluate(video_probes._CARD_SHOWN_JS, "abc123") is True
            assert frame.evaluate(video_probes._CARD_SHOWN_JS, "zzz") is False
            assert frame.evaluate(video_probes._PREVIEW_SHOWS_JS, "abc123") is expected
            assert frame.evaluate(video_probes._PREVIEW_STATE_JS).startswith("preview: ")
        finally:
            browser.close()


def test_the_closing_hold_scrolls_the_attachment_into_view():
    if not _chromium_available():
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
            page.wait_for_timeout(800)
            top = page.evaluate(
                "() => document.querySelector('#file-preview img').getBoundingClientRect().top"
            )
            assert 0 <= top < 600
            page.set_content("<div>no draft</div>")
            assert page.evaluate(video_probes._REVEAL_ATTACHMENT_JS) is False
        finally:
            browser.close()
