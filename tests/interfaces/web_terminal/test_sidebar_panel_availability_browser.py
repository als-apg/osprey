"""Browser test: the sidebar reads a panel's routes exactly while the panel is served.

A panel can be enabled with nothing behind it, and the hub's proxy answers 404
for such a panel. So the shell may not ask: no list, stream or health read
reaches ``/panel/artifacts`` or ``/panel/system-health`` until the panel's config
endpoint names a url, and they stop when it no longer does. Each case runs the
real hub and proxy; a served panel is a small backend that answers the routes the
sidebar reads. A restarted hub is two things here: what the hub publishes
changes, and the page's event stream reopens.

Run:
    uv run pytest tests/interfaces/web_terminal/test_sidebar_panel_availability_browser.py \
        -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from osprey.interfaces._serving import run_app_server
from tests.interfaces.web_terminal._chat_server import live_chat_server

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_ARTIFACTS = "/panel/artifacts/"
_HEALTH = "/panel/system-health/"
_BOTH = frozenset({"artifacts", "system-health"})

_TITLE = "Orbit report"
_ROW = f"#sidebar-artifacts-list .osprey-list-row:has-text('{_TITLE}')"
_DOT = "#sidebar-target-health"

#: Keeps the page's EventSources reachable, so a test can drop one the way a
#: restarting hub does. The wrapper in api.js then reconnects it for real.
_RECORD_SOURCES = """
const Real = window.EventSource;
window.__sources = [];
window.EventSource = class extends Real {
  constructor(...args) {
    super(...args);
    window.__sources.push(this);
  }
};
"""

_DROP_PANEL_STREAM = """
() => {
  for (const source of window.__sources) {
    if (!source.url.includes('/api/files/events') || source.readyState === 2) continue;
    source.close();
    source.onerror?.(new Event('error'));
  }
}
"""

#: How long a page is watched before saying it asked for nothing. Long enough
#: for a boot's reads and a stream retry to have shown themselves.
_QUIET_MS = 3_000


def _companion() -> FastAPI:
    """A backend answering what the sidebar reads, for either panel."""
    app = FastAPI()

    @app.get("/")
    async def root():
        return HTMLResponse("<!doctype html><title>panel</title>")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/checks")
    async def checks():
        return {"results": [{"status": "ok"}], "interval_s": 5}

    @app.get("/api/artifacts")
    async def artifacts(run: bool = False):
        entry = {
            "id": "a1",
            "artifact_type": "markdown",
            "timestamp": "2026-10-30T11:00:00Z",
            "title": _TITLE,
            "filename": "a1.md",
            "run_record": False,
        }
        return (
            {"artifacts": [], "total": 0, "next_cursor": None}
            if run
            else {"artifacts": [entry], "total": 1, "next_cursor": None}
        )

    @app.get("/api/focus")
    async def focus():
        return {"focused": False}

    @app.get("/api/events")
    async def events(request: Request):
        async def stream():
            yield ": open\n\n"
            while not await request.is_disconnected():
                await asyncio.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def _open(browser: Browser, base_url: str) -> tuple[Page, list[str]]:
    """Open the hub, recording every request the page makes under the two panels' paths."""
    context = browser.new_context()
    page = context.new_page()
    asked: list[str] = []

    def _on_request(request) -> None:
        if request.frame != page.main_frame:
            return
        if _ARTIFACTS in request.url or _HEALTH in request.url:
            asked.append(request.url)

    page.on("request", _on_request)
    page.add_init_script(_RECORD_SOURCES)
    page.goto(base_url, wait_until="load")
    return page, asked


def _reopen_stream(page: Page) -> None:
    """Drop the page's panel event stream, as a hub going away does: it reconnects."""
    page.evaluate(_DROP_PANEL_STREAM)


def _sidebar_routes(asked: list[str]) -> list[str]:
    """The recorded requests to the routes the sidebar reads (not a health poll of the rail)."""
    return [url for url in asked if "/api/" in url or url.endswith("/checks")]


@pytest.mark.parametrize(
    "enabled",
    [_BOTH, frozenset()],
    ids=["enabled-not-launched", "not-enabled"],
)
def test_an_unserved_panel_is_never_asked(tmp_path, chromium_browser, enabled):
    """Nothing reaches either panel's routes, and the dot stays hidden."""
    with live_chat_server(tmp_path, ui_mode="expert", enabled_panels=enabled) as (base_url, _app):
        page, asked = _open(chromium_browser, base_url)
        page.wait_for_timeout(_QUIET_MS)

        assert asked == [], f"the page asked a panel that is not served: {asked}"
        expect(page.locator(_DOT)).to_be_hidden()
        expect(page.locator(_ROW)).to_have_count(0)
        page.context.close()


def test_a_served_panel_works_as_ever(tmp_path, chromium_browser):
    """Served from the start: the lists fill, the stream opens and the dot reads."""
    with (
        run_app_server(_companion()) as companion,
        live_chat_server(
            tmp_path,
            ui_mode="expert",
            enabled_panels=_BOTH,
            panel_urls={"artifact": companion, "system_health": companion},
        ) as (base_url, _app),
    ):
        page, asked = _open(chromium_browser, base_url)

        expect(page.locator(_ROW)).to_be_visible(timeout=10_000)
        expect(page.locator(_DOT)).to_have_attribute(
            "aria-label", "Service health: ok", timeout=10_000
        )
        routes = _sidebar_routes(asked)
        for route in ("/api/artifacts", "/api/focus", "/api/events", "/checks"):
            assert any(route in url for url in routes), f"never read {route}: {routes}"
        page.context.close()


def test_a_panel_appearing_and_leaving_is_followed_without_a_reload(tmp_path, chromium_browser):
    """Start on the hub serving it, stop when it does not, the stream's retry included."""
    with (
        run_app_server(_companion()) as companion,
        live_chat_server(tmp_path, ui_mode="expert", enabled_panels=_BOTH) as (base_url, app),
    ):
        page, asked = _open(chromium_browser, base_url)
        page.wait_for_timeout(1_000)
        assert asked == [], f"the page asked a panel that is not served: {asked}"
        boot = page.evaluate("() => performance.timeOrigin")

        app.state.artifact_server_url = companion
        app.state.system_health_server_url = companion
        _reopen_stream(page)

        expect(page.locator(_ROW)).to_be_visible(timeout=20_000)
        expect(page.locator(_DOT)).to_have_attribute(
            "aria-label", "Service health: ok", timeout=20_000
        )
        assert any("/api/events" in url for url in asked), f"no stream opened: {asked}"

        app.state.artifact_server_url = None
        app.state.system_health_server_url = None
        _reopen_stream(page)

        expect(page.locator(_ROW)).to_have_count(0, timeout=20_000)
        expect(page.locator(_DOT)).to_be_hidden()

        # Let what was in flight land, then nothing may be asked any more.
        page.wait_for_timeout(1_000)
        asked.clear()
        page.wait_for_timeout(_QUIET_MS)
        assert asked == [], f"the page kept asking a panel that left: {asked}"
        assert page.evaluate("() => performance.timeOrigin") == boot, "the page reloaded"
        page.context.close()
