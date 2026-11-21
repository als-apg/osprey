"""Browser suite: the shared page-ready wait on a pending page that throws.

A late page declares ``data-page-ready="pending"`` and marks itself; when its
script throws, or leaves a rejection unhandled, before that call, the writer
(``design-system/js/page-ready.js``) marks it ``failed``. The wait every browser
test makes after a navigation returns on ``true`` and fails at once on
``failed``, naming it, instead of running into its timeout.

Run:
    .venv/bin/pytest tests/interfaces/test_page_ready_wait_browser.py -m browser -v
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from starlette.responses import HTMLResponse

import osprey.interfaces.design_system.panels as _panels_pkg
from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces._serving import run_app_server
from tests.interfaces._browser import _PAGE_READY_TIMEOUT_MS, wait_for_page_ready

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_IMPORT = "import { markPageReady } from '/design-system/js/page-ready.js';"

#: Each page's inline module, after the writer's import.
_PAGES = {
    "throws": "throw new Error('boom');",
    "rejects": "Promise.reject(new Error('x'));",
    "marks-on-failure": "void Promise.reject(new Error('x')).finally(() => markPageReady());",
}


def _page(body: str) -> str:
    return (
        '<!doctype html><html data-page-ready="pending"><head><meta charset="utf-8">'
        f'<script type="module">{_IMPORT}\n{body}</script></head><body></body></html>'
    )


@contextmanager
def _launch() -> Iterator[str]:
    """A minimal app: the three pending pages and the ``/design-system`` mount."""
    app = FastAPI()

    @app.get("/page/{name}")
    def _serve(name: str) -> HTMLResponse:  # pragma: no cover - exercised via browser
        return HTMLResponse(_page(_PAGES[name]))

    configure_interface_app(app, static_dir=Path(_panels_pkg.__file__).parent)

    with run_app_server(app) as base_url:
        yield base_url


def test_a_pending_page_that_throws_fails_the_wait_at_once_naming_failed(chromium_browser):
    with _launch() as base_url:
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/page/throws", wait_until="domcontentloaded")
            started = time.monotonic()
            with pytest.raises(AssertionError, match="failed"):
                wait_for_page_ready(page)
            assert (time.monotonic() - started) * 1000 < _PAGE_READY_TIMEOUT_MS / 2
        finally:
            page.close()


def test_a_pending_page_with_an_unhandled_rejection_fails_the_wait(chromium_browser):
    with _launch() as base_url:
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/page/rejects", wait_until="domcontentloaded")
            with pytest.raises(AssertionError, match="failed"):
                wait_for_page_ready(page)
        finally:
            page.close()


def test_a_page_that_marks_itself_on_its_failure_path_is_ready(chromium_browser):
    with _launch() as base_url:
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/page/marks-on-failure", wait_until="domcontentloaded")
            wait_for_page_ready(page)
            assert page.locator("html").get_attribute("data-page-ready") == "true"
        finally:
            page.close()
