"""Guard tests for ``assert_page_loads_clean`` (tests/interfaces/_browser.py).

Proves the helper actually catches breakage rather than rubber-stamping every
page: a missing module subresource (the classic "renamed dom.js and forgot a
route" mistake), a genuine uncaught JS exception, and — as a control — that a
truly clean page passes.

Run:
    .venv/bin/pytest tests/interfaces/test_browser_helper.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from tests.interfaces._browser import assert_page_loads_clean
from tests.interfaces.conftest import _run_app_server

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_missing_module_subresource_is_caught(chromium_browser) -> None:
    """A 404'd `<script type="module">` is caught by the response-arm."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _index() -> str:
        return (
            "<html><head>"
            '<script type="module" src="/design-system/js/dom.js"></script>'
            "</head><body><h1>page</h1></body></html>"
        )

    with _run_app_server(app) as base_url:
        page = chromium_browser.new_page()
        with pytest.raises(AssertionError, match="dom.js"):
            assert_page_loads_clean(page, base_url)


def test_uncaught_pageerror_is_caught(chromium_browser) -> None:
    """A real uncaught JS exception is caught by the pageerror-arm."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _index() -> str:
        return "<html><head></head><body><script>throw new Error('boom');</script></body></html>"

    with _run_app_server(app) as base_url:
        page = chromium_browser.new_page()
        with pytest.raises(AssertionError, match="boom"):
            assert_page_loads_clean(page, base_url)


def test_clean_page_passes(chromium_browser) -> None:
    """A genuinely clean page returns without raising."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _index() -> str:
        return "<html><head></head><body><h1>ok</h1></body></html>"

    with _run_app_server(app) as base_url:
        page = chromium_browser.new_page()
        assert assert_page_loads_clean(page, base_url) is None


def test_allowlist_suppresses_matching_item(chromium_browser) -> None:
    """An allowlist predicate matching the dom.js URL suppresses the failure."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _index() -> str:
        return (
            "<html><head>"
            '<script type="module" src="/design-system/js/dom.js"></script>'
            "</head><body><h1>page</h1></body></html>"
        )

    with _run_app_server(app) as base_url:
        page = chromium_browser.new_page()
        assert (
            assert_page_loads_clean(
                page,
                base_url,
                allowlist=lambda kind, detail: kind == "response" and "dom.js" in detail,
            )
            is None
        )
