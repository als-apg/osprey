"""Convert HTML files to images using Playwright (headless Chromium).

Chromium is installed on demand, but only ever in *reaction* to a launch that
failed because the binary is genuinely absent — the launch itself is the
availability check, so there is no second, weaker check that can disagree with
it. The install is attempted at most once per process and its verdict is cached,
so a host that cannot reach the browser CDN pays for one failed attempt rather
than one per conversion.

Raises ``PlaywrightNotInstalledError`` instead of a raw ``ImportError`` when
Playwright itself is missing. A launch that fails for any other reason (missing
system libraries, a sandbox denial) propagates unchanged rather than being
reported as a missing browser.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("osprey.mcp_server.export.converter")

SUPPORTED_FORMATS = {"png", "jpeg"}

# The only launch-error text that means the browser binary is not there —
# Playwright raises "browserType.launch: Executable doesn't exist at <path>".
# Matching this and nothing else keeps a launch that failed for an unrelated
# reason from being misread as a missing browser (and from triggering a pointless
# network install).
_MISSING_BROWSER_MARKER = "Executable doesn't exist"

# Run through the *current* interpreter rather than a bare "playwright" console
# script, which need not be on PATH in a container or in a venv the server was
# not launched from.
_INSTALL_CMD = [sys.executable, "-m", "playwright", "install", "chromium"]

# At-most-once-per-process install, and its cached verdict. Guarded by a plain
# threading lock (not asyncio) so the state stays correct across event loops and
# across threads; it is only ever held inside a worker thread.
_install_lock = threading.Lock()
_install_attempted = False
_install_error: str | None = None


class PlaywrightNotInstalledError(Exception):
    """Raised when Playwright or its browsers are not available."""


def _is_missing_browser(exc: BaseException) -> bool:
    """Whether *exc* is Playwright reporting an absent browser binary."""
    return _MISSING_BROWSER_MARKER in str(exc)


def _run_install() -> str | None:
    """Run the Chromium install; return an error message, or ``None`` on success."""
    logger.info("Chromium not found — installing via '%s'...", " ".join(_INSTALL_CMD))
    try:
        proc = subprocess.run(_INSTALL_CMD, capture_output=True, text=True)
    except OSError as exc:
        return f"Failed to run '{' '.join(_INSTALL_CMD)}': {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return (
            f"Failed to auto-install Chromium (exit {proc.returncode}). "
            f"Run manually: playwright install chromium\n{detail}"
        )
    logger.info("Chromium installed successfully.")
    return None


def _install_chromium() -> None:
    """Install Chromium once per process; re-raise the cached failure thereafter.

    Blocking — call it via :func:`asyncio.to_thread` so a conversion running on an
    event loop does not stall it for the length of a browser download.

    Raises:
        PlaywrightNotInstalledError: The install failed, now or on the earlier
            attempt whose verdict is cached.
    """
    global _install_attempted, _install_error
    with _install_lock:
        if not _install_attempted:
            _install_attempted = True
            _install_error = _run_install()
        if _install_error is not None:
            raise PlaywrightNotInstalledError(_install_error)


async def _render(
    async_playwright: Callable[[], Any],
    source: Path,
    dest: Path,
    fmt: str,
    width: int,
    height: int,
) -> None:
    """Screenshot *source* into *dest* with one headless Chromium page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.goto(source.as_uri(), wait_until="networkidle")
            await page.screenshot(path=str(dest), type=fmt, full_page=True)
        finally:
            await browser.close()


async def convert_html_to_image(
    html_path: str | Path,
    output_path: str | Path,
    fmt: str = "png",
    width: int = 1200,
    height: int = 800,
) -> Path:
    """Render an HTML file to an image via headless Chromium.

    If the browser binary is missing it is installed (once per process) and the
    render retried. A browser that launches is never re-checked and never
    re-installed.

    Args:
        html_path: Path to the source HTML file.
        output_path: Destination path for the rendered image.
        fmt: Image format — ``"png"`` or ``"jpeg"``.
        width: Viewport width in pixels.
        height: Viewport height in pixels.

    Returns:
        Resolved ``Path`` of the written image file.

    Raises:
        PlaywrightNotInstalledError: Playwright is not importable, or Chromium is
            absent and could not be installed.
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
        await _render(async_playwright, source, dest, fmt, width, height)
    except Exception as exc:
        if not _is_missing_browser(exc):
            raise
        # The launch is the availability check, so only a genuinely absent binary
        # reaches here: install it and retry exactly once.
        await asyncio.to_thread(_install_chromium)
        try:
            await _render(async_playwright, source, dest, fmt, width, height)
        except Exception as retry_exc:
            if _is_missing_browser(retry_exc):
                raise PlaywrightNotInstalledError(
                    "Chromium browser not installed. Run: playwright install chromium"
                ) from retry_exc
            raise

    logger.info("Converted %s → %s (%s)", source.name, dest.name, fmt)
    return dest.resolve()
