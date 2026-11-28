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

import re
from contextlib import AbstractContextManager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import expect

from osprey.interfaces._serving import run_app_server
from tests.interfaces import _browser
from tests.interfaces._browser import (
    LAST_ACTIVE_DEBOUNCE_MS,
    TOAST_TIMEOUT_MS,
    assert_page_loads_clean,
    record_toasts,
    toasts,
    wait_for_last_active,
    wait_for_toast,
)

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: A page whose module subresource this bare app does not serve.
_MISSING_MODULE_PAGE = (
    "<html><head>"
    '<script type="module" src="/design-system/js/dom.js"></script>'
    "</head><body><h1>page</h1></body></html>"
)


def _serve(html: str) -> AbstractContextManager[str]:
    """Serve ``html`` at ``/`` and nothing else."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _index() -> str:
        return html

    return run_app_server(app)


def test_missing_module_subresource_is_caught(chromium_browser) -> None:
    """A 404'd `<script type="module">` is caught by the response-arm."""
    with _serve(_MISSING_MODULE_PAGE) as base_url:
        page = chromium_browser.new_page()
        with pytest.raises(AssertionError, match="dom.js"):
            assert_page_loads_clean(page, base_url)


def test_uncaught_pageerror_is_caught(chromium_browser) -> None:
    """A real uncaught JS exception is caught by the pageerror-arm."""
    with _serve(
        "<html><head></head><body><script>throw new Error('boom');</script></body></html>"
    ) as base_url:
        page = chromium_browser.new_page()
        with pytest.raises(AssertionError, match="boom"):
            assert_page_loads_clean(page, base_url)


def test_clean_page_passes(chromium_browser) -> None:
    """A genuinely clean page returns without raising."""
    with _serve("<html><head></head><body><h1>ok</h1></body></html>") as base_url:
        page = chromium_browser.new_page()
        assert assert_page_loads_clean(page, base_url) is None


def test_allowlist_suppresses_matching_item(chromium_browser) -> None:
    """An allowlist predicate matching the dom.js URL suppresses the failure."""
    with _serve(_MISSING_MODULE_PAGE) as base_url:
        page = chromium_browser.new_page()
        assert (
            assert_page_loads_clean(
                page,
                base_url,
                allowlist=lambda kind, detail: kind == "response" and "dom.js" in detail,
            )
            is None
        )


def test_last_active_debounce_is_read_from_the_page_source() -> None:
    """The wait's ceiling is scaled from the debounce the page itself declares."""
    source = (
        Path(__file__).resolve().parents[2]
        / "src/osprey/interfaces/web_terminal/static/js/session-model.js"
    ).read_text(encoding="utf-8")
    match = re.search(r"^export const LAST_ACTIVE_DEBOUNCE_MS = (\d+);$", source, re.M)
    assert match, "LAST_ACTIVE_DEBOUNCE_MS not found in session-model.js"
    assert int(match.group(1)) == LAST_ACTIVE_DEBOUNCE_MS
    assert LAST_ACTIVE_DEBOUNCE_MS > 0


_SERVER_KEY = "11111111-2222-3333-4444-555555555555"
_OTHER_KEY = "66666666-7777-8888-9999-000000000000"


def _serve_last_active(key: str) -> AbstractContextManager[str]:
    """Serve a bare page at ``/`` and a last-active answer that always names ``key``."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _index() -> str:
        return "<html><head></head><body><h1>ok</h1></body></html>"

    @app.get("/api/sessions/last-active")
    def _last_active() -> dict[str, str]:
        return {"session_key": key, "at": "2026-01-01T00:00:00Z"}

    return run_app_server(app)


def test_wait_for_last_active_returns_on_the_servers_key(chromium_browser) -> None:
    """The wait returns once the server names the expected key."""
    with _serve_last_active(_SERVER_KEY) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url)  # page-ready: a bare page this test serves
        assert wait_for_last_active(page, _SERVER_KEY) is None


def test_wait_for_last_active_names_both_keys_when_the_server_holds_another(
    chromium_browser,
) -> None:
    """A key the server never names fails with the expected key and the server's answer."""
    with _serve_last_active(_SERVER_KEY) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url)  # page-ready: a bare page this test serves
        with pytest.raises(AssertionError) as raised:
            wait_for_last_active(page, _OTHER_KEY)
        assert _OTHER_KEY in str(raised.value)
        assert _SERVER_KEY in str(raised.value)


def test_wait_for_last_active_refuses_an_empty_key(chromium_browser) -> None:
    """An empty key is refused before any wait."""
    page = chromium_browser.new_page()
    with pytest.raises(ValueError, match="key"):
        wait_for_last_active(page, "")


_DESIGN_SYSTEM_STATIC = (
    Path(__file__).resolve().parents[2] / "src/osprey/interfaces/design_system/static"
)

#: Raises two short toasts through the real toast layer, the second queued
#: behind the first.
_SHOW_TWO_TOASTS_JS = """
async () => {
  const { showToast } = await import('/design-system/js/toast.js');
  showToast('first', { timeout: 50 });
  showToast('second', { kind: 'error', timeout: 50 });
}
"""


def _serve_toast_layer() -> AbstractContextManager[str]:
    """Serve a bare page at ``/`` and the design system's static files."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _index() -> str:
        return "<html><head></head><body><h1>ok</h1></body></html>"

    app.mount("/design-system", StaticFiles(directory=_DESIGN_SYSTEM_STATIC))
    return run_app_server(app)


def test_toast_recorder_keeps_a_toast_that_came_and_went(chromium_browser) -> None:
    """Toasts that have already hidden are still in the record, in the order shown."""
    with _serve_toast_layer() as base_url:
        page = chromium_browser.new_page()
        record_toasts(page)
        page.goto(base_url)  # page-ready: a bare page this test serves
        page.evaluate(_SHOW_TWO_TOASTS_JS)
        expect(page.locator(".osprey-toast")).to_have_count(0)
        assert toasts(page) == [
            {"message": "first", "kind": "info"},
            {"message": "second", "kind": "error"},
        ]


def test_wait_for_toast_names_the_record_when_the_message_never_shows(
    chromium_browser, monkeypatch
) -> None:
    """A message no toast carried fails with that message and what the page did show."""
    # The smallest ceiling Playwright honours: the subject is the failure, not the wait.
    monkeypatch.setattr(_browser, "_TOAST_SHOWN_TIMEOUT_MS", 1)
    with _serve_toast_layer() as base_url:
        page = chromium_browser.new_page()
        record_toasts(page)
        page.goto(base_url)  # page-ready: a bare page this test serves
        page.evaluate(_SHOW_TWO_TOASTS_JS)
        expect(page.locator(".osprey-toast")).to_have_count(0)
        with pytest.raises(AssertionError, match="never raised") as raised:
            wait_for_toast(page, "never raised")
        assert "'second'" in str(raised.value)


def test_toasts_refuses_a_frame_without_the_recorder(chromium_browser) -> None:
    """A page loaded without the recorder is refused, never read as "no toast"."""
    with _serve_toast_layer() as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url)  # page-ready: a bare page this test serves
        with pytest.raises(AssertionError, match="not installed"):
            toasts(page)


def test_toast_timeout_is_read_from_the_page_source() -> None:
    """The toast wait's ceiling is scaled from the timeout the toast layer declares."""
    source = (_DESIGN_SYSTEM_STATIC / "js/toast.js").read_text(encoding="utf-8")
    match = re.search(r"^export const TOAST_TIMEOUT_MS = (\d+);$", source, re.M)
    assert match, "TOAST_TIMEOUT_MS not found in toast.js"
    assert int(match.group(1)) == TOAST_TIMEOUT_MS
    assert TOAST_TIMEOUT_MS > 0
