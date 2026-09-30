"""Browser suite: a timeline mark's tooltip reads the facility clock.

The dispatch dashboard's timeline marks name the instant a trigger fired. That
instant renders in the facility zone the dashboard state carries, in the
viewer's locale, and names the zone only when the viewer's own clock reads
differently.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_dashboard_facility_time.py \
        -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1100, "height": 900}

DESIGN_SYSTEM_STATIC_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
)

TRIGGER_NAME = "hourly-check"
FACILITY_ZONE = "Asia/Tokyo"

#: The components ``toLocaleString()`` renders by default.
_COMPONENTS = (
    "year: 'numeric', month: 'numeric', day: 'numeric', "
    "hour: 'numeric', minute: '2-digit', second: '2-digit'"
)


def _create_dispatch_dashboard_app() -> FastAPI:
    """Serve the real dashboard beside the real design system and one recent mark.

    The mark sits 30 minutes before each request, so it is inside the 24-hour
    window whatever the wall clock. The emitted ISO is kept on the app.
    """
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )
    app.state.mark_iso = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(render_dashboard_html(facility_name="Test Facility", channel_strip_prefix="SR:"))

    @app.get("/dashboard/state")
    async def dashboard_state() -> dict[str, Any]:
        return {
            "pool": {"running": 0, "queued": 0, "max": 2},
            "triggers": [
                {
                    "name": TRIGGER_NAME,
                    "status": "armed",
                    "source": "webhook",
                    "on_error": "drop",
                    "allowed_tools": [],
                    "prompt": "Check the hour",
                    "source_config": {},
                    "last_fired": None,
                    "next_fire": None,
                },
            ],
            "runs": [],
            "timeline": {
                TRIGGER_NAME: [{"timestamp": app.state.mark_iso, "status": "dispatched"}],
            },
            "worker_error": None,
            "server_time_iso": datetime.now(UTC).isoformat(),
            "facility_timezone": FACILITY_ZONE,
        }

    return app


@pytest.fixture
def dashboard() -> Iterator[tuple[str, FastAPI]]:
    app = _create_dispatch_dashboard_app()
    with _run_app_server(app) as base_url:
        yield base_url, app


def _mark_title(chromium_browser: Browser, base_url: str, viewer_zone: str, expected_js: str):
    page = chromium_browser.new_page(viewport=VIEWPORT, timezone_id=viewer_zone, locale="en-US")
    try:
        page.goto(f"{base_url}/?mode=expert", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => document.querySelector('.lane-mark')?.title.includes(' — ')",
            timeout=10_000,
        )
        expected = page.evaluate(expected_js) + " — dispatched"
        page.wait_for_function(
            "(want) => document.querySelector('.lane-mark')?.title === want",
            arg=expected,
            timeout=10_000,
        )
        return page.evaluate("() => document.querySelector('.lane-mark').title"), expected
    finally:
        page.close()


def test_a_timeline_mark_reads_the_facility_clock_and_names_its_zone(
    chromium_browser: Browser, dashboard: tuple[str, FastAPI]
) -> None:
    base_url, app = dashboard
    expected_js = (
        f"() => new Intl.DateTimeFormat(undefined, {{{_COMPONENTS}, "
        f"timeZone: '{FACILITY_ZONE}', timeZoneName: 'short'}})"
        f".format(new Date('{app.state.mark_iso}'))"
    )
    title, expected = _mark_title(chromium_browser, base_url, "America/Los_Angeles", expected_js)
    assert title == expected


def test_a_viewer_on_the_facility_clock_reads_the_mark_without_a_zone_name(
    chromium_browser: Browser, dashboard: tuple[str, FastAPI]
) -> None:
    base_url, app = dashboard
    expected_js = (
        f"() => new Intl.DateTimeFormat(undefined, {{{_COMPONENTS}, "
        f"timeZone: '{FACILITY_ZONE}'}})"
        f".format(new Date('{app.state.mark_iso}'))"
    )
    title, expected = _mark_title(chromium_browser, base_url, FACILITY_ZONE, expected_js)
    assert title == expected
