"""Browser test: exporting a saved Plotly artifact renders the plot, not a blank page.

A figure saved through ``serialize_object`` is written with
``include_plotlyjs=False``, so the page calls ``Plotly.newPlot`` without loading
Plotly. ``convert_html_to_image`` must inject the library and wait for the plot
to draw; otherwise the exported PNG (the image attached to a logbook entry) is
an empty white page.

This drives the real converter in a real headless Chromium and counts the
colours in the centre of the PNG: a drawn 3D scatter has axis panes, grid lines,
markers and anti-aliased text, while a blank page has one colour.

Run:
    uv run pytest tests/interfaces/artifacts/test_plot_export_browser.py -v

Skips cleanly when the Chromium headless binary is not installed; it never
downloads one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# A drawn 3D scene yields hundreds of distinct colours; a blank page yields one.
_MIN_DISTINCT_COLOURS = 10


def _chromium_available() -> bool:
    """Whether Playwright's Chromium binary is present on this host."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    p = sync_playwright().start()
    try:
        return Path(p.chromium.executable_path).exists()
    finally:
        p.stop()


def test_saved_scatter3d_artifact_exports_a_drawn_plot(tmp_path, monkeypatch):
    if not _chromium_available():
        pytest.skip("Playwright Chromium is not installed")

    go = pytest.importorskip("plotly.graph_objects")
    image = pytest.importorskip("PIL.Image")

    from osprey.mcp_server.export import converter
    from osprey.stores.artifact_store import serialize_object

    def _no_install() -> None:
        pytest.skip("Chromium launch failed as missing; this test never installs a browser")

    monkeypatch.setattr(converter, "_install_chromium", _no_install)

    figure = go.Figure(
        go.Scatter3d(
            x=[0, 1, 2, 3, 4, 5],
            y=[5, 3, 4, 1, 2, 0],
            z=[1, 4, 2, 5, 3, 0],
            mode="markers+lines",
            marker={"size": 8, "color": [0, 1, 2, 3, 4, 5], "colorscale": "Viridis"},
        )
    )
    content, artifact_type, filename, _mime = serialize_object(figure, "BPM positions")
    assert artifact_type == "plot_html"
    assert b"cdn.plot.ly" not in content and b"plotly.min.js" not in content

    html_path = tmp_path / filename
    html_path.write_bytes(content)
    png_path = tmp_path / "export.png"

    result = asyncio.run(converter.convert_html_to_image(html_path, png_path, fmt="png"))

    assert result == png_path.resolve()
    with image.open(result) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        centre = rgb.crop((w // 4, h // 8, 3 * w // 4, h // 2))
        colours = centre.getcolors(maxcolors=w * h)
    assert colours is not None
    assert len(colours) > _MIN_DISTINCT_COLOURS, (
        f"central plot region has {len(colours)} colours; the plot did not draw"
    )
