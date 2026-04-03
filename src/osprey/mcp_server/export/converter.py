"""Convert HTML files to images using Playwright (headless Chromium).

Auto-installs Chromium on first use. Raises ``PlaywrightNotInstalledError``
instead of a raw ``ImportError`` when Playwright is missing.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("osprey.mcp_server.export.converter")

SUPPORTED_FORMATS = {"png", "jpeg"}


class PlaywrightNotInstalledError(Exception):
    """Raised when Playwright or its browsers are not available."""


def _ensure_chromium_installed() -> None:
    """Auto-install Chromium browser if not already present."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            # Check if the executable exists
            if not Path(p.chromium.executable_path).exists():
                raise FileNotFoundError
    except (FileNotFoundError, Exception):
        logger.info("Chromium not found — installing via 'playwright install chromium'...")
        try:
            subprocess.run(
                ["playwright", "install", "chromium"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Chromium installed successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise PlaywrightNotInstalledError(
                "Failed to auto-install Chromium. Run manually: playwright install chromium"
            ) from exc


async def convert_html_to_image(
    html_path: str | Path,
    output_path: str | Path,
    fmt: str = "png",
    width: int = 1200,
    height: int = 800,
) -> Path:
    """Render an HTML file to an image via headless Chromium.

    On first use, automatically installs Chromium if the browser
    binary is missing.

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

    # Auto-install Chromium on first use
    _ensure_chromium_installed()

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
