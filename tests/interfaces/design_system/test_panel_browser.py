"""Browser Playwright suite: the reference panel actually re-themes on ``?theme=``.

This is the empirical replacement for the *undecidable* static "honors
``?theme=``" check on a panel. A static scan cannot prove that a panel's
pre-paint boot script plus token stylesheet combine to put the requested
theme on ``<html>`` — only a real browser can. So this suite stands the
reference panel (Task 2.3, ``panels/reference/index.html``) up in a live
server, loads it with ``?theme=<family>``, and asserts the applied
``data-theme`` on ``document.documentElement`` matches the requested id.

The reference panel is not yet wired into any app's routes (that is Phase 3),
so the suite builds its own minimal FastAPI app that (a) serves the panel's
``index.html`` at ``GET /panel`` and (b) mounts ``/design-system`` via the
shared :func:`configure_interface_app` so the panel's ``<head>`` assets —
``theme-boot.js``, ``tokens.css``, ``theme-manager.js``, ``frame-params.js`` —
resolve instead of 404ing (a 404 there means the boot script never runs and
theming silently no-ops, which is exactly the failure this test must catch).

Mirrors the ``browser``-marked model of ``test_osprey_drawer.py`` (marker set,
live uvicorn server on a free port, real chromium page) and the applied-theme
assertion of ``test_visual.py``.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_panel_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed (the shared
``chromium_browser`` fixture owns that skip).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from starlette.responses import FileResponse

import osprey.interfaces.design_system.panels as _panels_pkg
from osprey.interfaces._app_setup import configure_interface_app
from tests.interfaces._browser import assert_page_loads_clean
from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# The four valid theme ids, one per (family, mode) cell — the exact set
# VALID_IDS in theme-boot.js accepts. Covering both families (osprey /
# high-contrast) and both modes (light / dark) proves the reference panel
# re-themes across the whole matrix, not just the default family.
THEME_IDS = ["light", "dark", "high-contrast-light", "high-contrast-dark"]

# The reference panel directory, resolved at runtime off the panels package so
# it tracks the package rather than a hardcoded path (Task 2.3 layout).
_REFERENCE_DIR = Path(_panels_pkg.__file__).parent / "reference"


# ---------------------------------------------------------------------------
# Live server launcher
# ---------------------------------------------------------------------------


@contextmanager
def _launch_reference_panel() -> Iterator[str]:
    """Minimal app: reference ``index.html`` at ``/panel`` + ``/design-system`` mount.

    ``configure_interface_app`` derives and mounts ``/design-system`` from its
    own location (the shared design-system static dir), so the panel's head
    assets resolve. ``static_dir`` is the panel dir itself — unused by this
    suite but a real, existing directory so the ``/static`` mount is valid.
    """
    index_html = _REFERENCE_DIR / "index.html"
    assert index_html.exists(), f"reference panel missing at {index_html}"

    app = FastAPI()

    @app.get("/panel")
    def _panel() -> FileResponse:  # pragma: no cover - exercised via browser
        return FileResponse(index_html)

    configure_interface_app(app, static_dir=_REFERENCE_DIR)

    with _run_app_server(app) as base_url:
        yield base_url


def _applied_theme(page) -> str | None:  # noqa: ANN001 - Playwright Page
    """Return the ``data-theme`` attribute on ``<html>`` after boot."""
    return page.evaluate("document.documentElement.getAttribute('data-theme')")


# ---------------------------------------------------------------------------
# Test 1: ?theme=<id> is applied to <html> for every valid family/mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme_id", THEME_IDS)
def test_reference_panel_applies_requested_theme(chromium_browser, theme_id: str):
    """Loading the reference panel with ``?theme=<id>`` stamps that id on ``<html>``."""
    with _launch_reference_panel() as base_url:
        page = chromium_browser.new_page()
        try:
            # A clean load also proves the panel's /design-system head assets
            # (theme-boot.js, tokens.css, theme-manager.js, frame-params.js)
            # resolved — if any 404'd, boot would not run and theming would
            # silently no-op rather than fail visibly.
            assert_page_loads_clean(page, f"{base_url}/panel?theme={theme_id}")

            applied = _applied_theme(page)
            assert applied == theme_id, (
                f"reference panel: expected data-theme={theme_id!r}, got {applied!r}"
            )
        finally:
            page.close()


# ---------------------------------------------------------------------------
# Test 2: without ?theme=, the panel still boots to a valid known theme
# ---------------------------------------------------------------------------


def test_reference_panel_boots_to_valid_theme_without_query(chromium_browser):
    """With no ``?theme=``, theme-boot still resolves ``<html>`` to a valid id.

    Proves the pre-paint boot ran (the ``auto`` → concrete-id resolution),
    rather than leaving ``data-theme`` null/empty. The exact id depends on the
    browser's ``prefers-color-scheme``, so this asserts membership in the valid
    set, not a specific value.
    """
    with _launch_reference_panel() as base_url:
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/panel", wait_until="domcontentloaded")
            applied = _applied_theme(page)
            assert applied in THEME_IDS, (
                f"reference panel booted to unexpected data-theme: {applied!r}"
            )
        finally:
            page.close()
