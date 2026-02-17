"""Convert HTML files to images using Playwright (headless Chromium).

Provides a graceful fallback when Playwright is not installed —
callers get a clear ``PlaywrightNotInstalledError`` instead of a raw
``ImportError``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("osprey.mcp_server.export.converter")

SUPPORTED_FORMATS = {"png", "jpeg"}


class PlaywrightNotInstalledError(Exception):
    """Raised when Playwright or its browsers are not available."""


async def convert_html_to_image(
    html_path: str | Path,
    output_path: str | Path,
    fmt: str = "png",
    width: int = 1200,
    height: int = 800,
) -> Path:
    """Render an HTML file to an image via headless Chromium.

    Args:
        html_path: Path to the source HTML file.
        output_path: Destination path for the rendered image.
        fmt: Image format — ``"png"`` or ``"jpeg"``.
        width: Viewport width in pixels.
        height: Viewport height in pixels.

    Returns:
        Resolved ``Path`` of the written image file.

    Raises:
        PlaywrightNotInstalledError: Playwright or Chromium not available.
        FileNotFoundError: *html_path* does not exist.
        ValueError: Unsupported *fmt*.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}'. Use one of: {SUPPORTED_FORMATS}")

    source = Path(html_path)
    if not source.exists():
        raise FileNotFoundError(f"HTML file not found: {source}")

    dest = Path(output_path)

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise PlaywrightNotInstalledError(
            "Playwright is not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.goto(source.as_uri(), wait_until="networkidle")
            await page.screenshot(path=str(dest), type=fmt, full_page=True)
            await browser.close()
    except Exception as exc:
        if "Executable doesn't exist" in str(exc) or "browserType.launch" in str(exc):
            raise PlaywrightNotInstalledError(
                "Chromium browser not installed. Run: playwright install chromium"
            ) from exc
        raise

    logger.info("Converted %s → %s (%s)", source.name, dest.name, fmt)
    return dest.resolve()
