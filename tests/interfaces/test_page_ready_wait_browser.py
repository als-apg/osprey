"""Browser suite: the shared page-ready wait on a pending page that fails.

A late page declares ``data-page-ready="pending"``, loads the writer
(``design-system/js/page-ready.js``) by its own tag first, and marks itself;
when its script throws, leaves a rejection unhandled, or one of its own scripts
fails to load, before that call, the writer marks it ``failed``. The wait every
browser test makes after a navigation returns on ``true`` and fails at once on
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
from playwright.sync_api import expect
from starlette.responses import HTMLResponse, Response

import osprey.interfaces.design_system.panels as _panels_pkg
from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces._serving import run_app_server
from osprey.interfaces.design_system.panels.page_ready import WRITER_TAG
from tests.interfaces._browser import _PAGE_READY_TIMEOUT_MS, wait_for_page_ready

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_IMPORT = "import { markPageReady } from '/design-system/js/page-ready.js';"


def _module(body: str) -> str:
    """An inline module that imports the writer's call, then runs *body*."""
    return f'<script type="module">{_IMPORT}\n{body}</script>'


#: Each page's head after the writer's tag.
_PAGES = {
    "throws": _module("throw new Error('boom');"),
    "rejects": _module("Promise.reject(new Error('x'));"),
    "marks-on-failure": _module(
        "void Promise.reject(new Error('x')).finally(() => markPageReady());"
    ),
    "module-missing": '<script type="module" src="/page-assets/no-such.js"></script>'
    + _module("markPageReady();"),
    "inline-import-missing": '<script type="module">import "/page-assets/no-such.js";'
    "\nmarkPageReady();</script>",
    "parse-error": '<script type="module">import "/page-assets/imports-broken.js";'
    "\nmarkPageReady();</script>",
    "vendor-missing": '<script defer src="/page-assets/vendor/no-such.js"></script>'
    + _module("markPageReady();"),
    "missing-after-mark": _module(
        "markPageReady();\n"
        "const late = document.createElement('script');\n"
        "late.src = '/page-assets/late.js';\n"
        "late.onerror = () => document.body.setAttribute('data-late', 'failed');\n"
        "document.head.append(late);"
    ),
}

#: The page's own modules the app serves; any other ``/page-assets/`` file is missing.
_ASSETS = {
    "broken.js": "export const = ;\n",
    "imports-broken.js": "import { x } from '/page-assets/broken.js';\nexport default x;\n",
}


def _page(head: str) -> str:
    return (
        '<!doctype html><html data-page-ready="pending"><head><meta charset="utf-8">'
        f"{WRITER_TAG}{head}</head><body></body></html>"
    )


@contextmanager
def _launch() -> Iterator[str]:
    """A minimal app: the pending pages, their own files and the ``/design-system`` mount."""
    app = FastAPI()

    @app.get("/page/{name}")
    def _serve(name: str) -> HTMLResponse:  # pragma: no cover - exercised via browser
        return HTMLResponse(_page(_PAGES[name]))

    @app.get("/page-assets/{path:path}")
    def _asset(path: str) -> Response:  # pragma: no cover - exercised via browser
        if path not in _ASSETS:
            return Response(status_code=404)
        return Response(_ASSETS[path], media_type="text/javascript")

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


def _fails_at_once(chromium_browser, name: str) -> None:
    with _launch() as base_url:
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/page/{name}", wait_until="domcontentloaded")
            started = time.monotonic()
            with pytest.raises(AssertionError, match="failed"):
                wait_for_page_ready(page)
            assert (time.monotonic() - started) * 1000 < _PAGE_READY_TIMEOUT_MS / 2
        finally:
            page.close()


def test_a_pending_page_whose_module_is_missing_fails_the_wait_at_once(chromium_browser):
    _fails_at_once(chromium_browser, "module-missing")


def test_a_pending_page_whose_inline_module_imports_a_missing_file_fails_the_wait(
    chromium_browser,
):
    _fails_at_once(chromium_browser, "inline-import-missing")


def test_a_pending_page_with_a_parse_error_in_its_module_graph_fails_the_wait(chromium_browser):
    _fails_at_once(chromium_browser, "parse-error")


def test_a_missing_vendor_bundle_leaves_the_page_ready(chromium_browser):
    with _launch() as base_url:
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/page/vendor-missing", wait_until="domcontentloaded")
            wait_for_page_ready(page)
            assert page.locator("html").get_attribute("data-page-ready") == "true"
        finally:
            page.close()


def test_a_script_missing_after_the_page_marked_itself_changes_nothing(chromium_browser):
    with _launch() as base_url:
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/page/missing-after-mark", wait_until="domcontentloaded")
            wait_for_page_ready(page)
            # The late script's own report is the state that says its failure was heard.
            expect(page.locator("body")).to_have_attribute("data-late", "failed")
            assert page.locator("html").get_attribute("data-page-ready") == "true"
        finally:
            page.close()
