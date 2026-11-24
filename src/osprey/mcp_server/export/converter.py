"""Convert HTML files to images using Playwright (headless Chromium).

The launch is the availability check, so there is no second, weaker check
that can disagree with it. The converter never installs a browser: the image
build installs it, or whoever sets up a bare host installs it once. A launch
that failed because the browser binary is absent raises
``PlaywrightNotInstalledError`` with a message that names the install command;
a launch that fails for any other reason (missing system libraries, a sandbox
denial) propagates unchanged.
"""

from __future__ import annotations

import logging
import re
import shlex
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("osprey.mcp_server.export.converter")

SUPPORTED_FORMATS = {"png", "jpeg"}

# The only launch-error text that means the browser binary is not there —
# Playwright raises "browserType.launch: Executable doesn't exist at <path>".
# Matching this and nothing else keeps a launch that failed for an unrelated
# reason from being misread as a missing browser.
_MISSING_BROWSER_MARKER = "Executable doesn't exist"

# What every absent-browser error opens with.
_BROWSER_MISSING = "Headless Chromium is not installed, so the page cannot be rendered to an image"

# The interpreter this server runs under, because a bare ``python`` or
# ``playwright`` on PATH need not be OSPREY's; the headless shell is the browser
# the image build installs.
BROWSER_INSTALL_COMMAND = shlex.join(
    [sys.executable, "-m", "playwright", "install", "--only-shell", "chromium"]
)

# The one sentence every surface relays when the browser is missing.
BROWSER_MISSING_MESSAGE = (
    f"{_BROWSER_MISSING}: install it once with {BROWSER_INSTALL_COMMAND}, or, in a "
    "container, rebuild the image without images.skip_browser."
)


# The Plotly bundle exists only where ``osprey vendor fetch`` wrote it. Its
# absence is the normal case (an installed wheel, an unfetched checkout), and
# the CDN URL is used then.
_PLOTLY_VENDOR_REL = "artifacts/static/js/vendor/plotly-3.3.1.min.js"


def _plotly_vendor_path() -> Path:
    """Path of the vendored Plotly bundle under ``osprey.interfaces``."""
    import osprey.interfaces

    return Path(osprey.interfaces.__file__).parent / _PLOTLY_VENDOR_REL


def _plotly_script_src() -> str:
    """Script ``src`` for Plotly in a page rendered from a ``file://`` URL.

    The vendored bundle when present, else the absolute CDN URL. ``vendor_url``
    is not used: in offline mode it returns a server-relative path, which does
    not resolve for a page loaded from disk.
    """
    bundle = _plotly_vendor_path()
    if bundle.exists():
        return bundle.resolve().as_uri()
    from osprey.interfaces.vendor import asset_cdn_url

    return asset_cdn_url("Plotly.js")


# The deadline of a whole export. A plain page's load is held to all of it; an
# injected page's load and its plot's draw are both held to it. The draw gets its own budget, timed from the page's
# load; the load (a local file and the Plotly bundle, slow to parse on a loaded
# host) gets the rest, so a slow load is never read as a plot that did not draw.
_EXPORT_DEADLINE_MS = 60_000
_PLOTLY_BUDGET_MS = 10_000

# True once the first Plotly graph has drawn — its layout is computed and no
# transition is in flight — not merely once ``window.Plotly`` exists.
_PLOT_DRAWN_JS = (
    "() => { const g = document.querySelector('.js-plotly-plot'); "
    "return !!(g && g._fullLayout && !g._transitioning); }"
)

# Resolves after two animation frames, so the drawn plot has been painted.
_TWO_FRAMES_JS = "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"

_HEAD_OPEN = re.compile(rb"<head\b[^>]*>", re.IGNORECASE)
_HTML_OPEN = re.compile(rb"<html\b[^>]*>", re.IGNORECASE)


def _inject_plotly(html: bytes, base_href: str, script_src: str) -> bytes:
    """Insert a ``<base>`` and the Plotly ``<script>`` right after ``<head>``.

    Falls back to the first child of ``<html>``, then to the start of the page.
    """
    tags = f'<base href="{base_href}"><script src="{script_src}"></script>'.encode()
    match = _HEAD_OPEN.search(html) or _HTML_OPEN.search(html)
    at = match.end() if match else 0
    return html[:at] + tags + html[at:]


class PlaywrightNotInstalledError(Exception):
    """Raised when the headless browser is not installed."""


class ExportDeadlineError(Exception):
    """Raised when an export missed its deadline; nothing is written then.

    A screenshot of a page that did not load, or of a plot that did not draw, is
    a blank image a reader would take for the plot.
    """

    _WHAT = "The page {name} did not finish within {ms} ms"

    def __init__(self, source: Path, budget_ms: int) -> None:
        what = self._WHAT.format(name=source.name, ms=budget_ms)
        super().__init__(f"{what}; no image was written.")
        self.source = source
        self.budget_ms = budget_ms


class PageNotLoadedError(ExportDeadlineError):
    """Raised when the page did not load within its share of the export deadline."""

    _WHAT = "The page {name} did not load within {ms} ms"


class PlotNotDrawnError(ExportDeadlineError):
    """Raised when the page's Plotly plot did not draw within its budget after the load."""

    _WHAT = "The Plotly plot in {name} did not draw within {ms} ms"


def _is_missing_browser(exc: BaseException) -> bool:
    """Whether *exc* is Playwright reporting an absent browser binary."""
    return _MISSING_BROWSER_MARKER in str(exc)


async def _render(
    async_playwright: Callable[[], Any],
    source: Path,
    dest: Path,
    fmt: str,
    width: int,
    height: int,
) -> None:
    """Screenshot *source* into *dest* with one headless Chromium page.

    A page that calls Plotly without loading it is rendered from a temporary
    copy in ``dest.parent`` that loads the library, then waited on until its
    plot has drawn. The source file itself is never modified.

    Raises:
        PageNotLoadedError: The page had not loaded within the export deadline
            (less the draw budget, for an injected page); no image is written.
        PlotNotDrawnError: The plot had not drawn within the draw budget, timed
            from the page's load; no image is written.
    """
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    from osprey.interfaces.vendor import html_needs_plotly

    html = source.read_bytes()
    if not html_needs_plotly(html):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": width, "height": height})
                try:
                    await page.goto(
                        source.as_uri(), wait_until="networkidle", timeout=_EXPORT_DEADLINE_MS
                    )
                except PlaywrightTimeoutError as exc:
                    raise PageNotLoadedError(source, _EXPORT_DEADLINE_MS) from exc
                await page.screenshot(path=str(dest), type=fmt, full_page=True)
            finally:
                await browser.close()
        return

    injected = _inject_plotly(html, f"{source.parent.as_uri()}/", _plotly_script_src())
    with tempfile.NamedTemporaryFile(suffix=".html", dir=dest.parent, delete=False) as tmp:
        tmp.write(injected)
    copy = Path(tmp.name)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": width, "height": height})
                load_ms = _EXPORT_DEADLINE_MS - _PLOTLY_BUDGET_MS
                try:
                    await page.goto(copy.as_uri(), wait_until="load", timeout=load_ms)
                except PlaywrightTimeoutError as exc:
                    raise PageNotLoadedError(source, load_ms) from exc
                try:
                    await page.wait_for_function(_PLOT_DRAWN_JS, timeout=_PLOTLY_BUDGET_MS)
                    await page.evaluate(_TWO_FRAMES_JS)
                except PlaywrightTimeoutError as exc:
                    raise PlotNotDrawnError(source, _PLOTLY_BUDGET_MS) from exc
                await page.screenshot(path=str(dest), type=fmt, full_page=True)
            finally:
                await browser.close()
    finally:
        copy.unlink(missing_ok=True)


async def convert_html_to_image(
    html_path: str | Path,
    output_path: str | Path,
    fmt: str = "png",
    width: int = 1200,
    height: int = 800,
) -> Path:
    """Render an HTML file to an image via headless Chromium.

    A browser binary that is missing is reported, never installed.

    Args:
        html_path: Path to the source HTML file.
        output_path: Destination path for the rendered image.
        fmt: Image format — ``"png"`` or ``"jpeg"``.
        width: Viewport width in pixels.
        height: Viewport height in pixels.

    Returns:
        Resolved ``Path`` of the written image file.

    Raises:
        PlaywrightNotInstalledError: The headless browser is not installed; the
            message names the command that installs it.
        PageNotLoadedError: The page did not load within the export deadline.
        PlotNotDrawnError: The page's Plotly plot did not draw in time.
        FileNotFoundError: *html_path* does not exist.
        ValueError: Unsupported *fmt*.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}'. Use one of: {SUPPORTED_FORMATS}")

    source = Path(html_path)
    if not source.exists():
        raise FileNotFoundError(f"HTML file not found: {source}")

    dest = Path(output_path)

    from playwright.async_api import async_playwright

    try:
        await _render(async_playwright, source, dest, fmt, width, height)
    except Exception as exc:
        if _is_missing_browser(exc):
            raise PlaywrightNotInstalledError(BROWSER_MISSING_MESSAGE) from exc
        raise

    logger.info("Converted %s → %s (%s)", source.name, dest.name, fmt)
    return dest.resolve()
