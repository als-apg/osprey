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

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from osprey.interfaces._serving import run_app_server
from tests.interfaces.design_system._dispatch_dashboard import create_dispatch_dashboard_app

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1100, "height": 900}

TRIGGER_NAME = "hourly-check"
FACILITY_ZONE = "Asia/Tokyo"

#: The components ``toLocaleString()`` renders by default.
_COMPONENTS = (
    "year: 'numeric', month: 'numeric', day: 'numeric', "
    "hour: 'numeric', minute: '2-digit', second: '2-digit'"
)


def _state(mark_iso: str) -> dict[str, Any]:
    """One armed trigger with one mark at ``mark_iso``, on a server clock read per request."""
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
        "timeline": {TRIGGER_NAME: [{"timestamp": mark_iso, "status": "dispatched"}]},
        "worker_error": None,
        "server_time_iso": datetime.now(UTC).isoformat(),
        "facility_timezone": FACILITY_ZONE,
    }


@pytest.fixture
def dashboard() -> Iterator[tuple[str, str]]:
    """The dashboard's URL and the ISO of its one timeline mark.

    The mark sits 30 minutes before the server starts, so it is inside the
    24-hour window whatever the wall clock.
    """
    mark_iso = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    with run_app_server(create_dispatch_dashboard_app(lambda: _state(mark_iso))) as base_url:
        yield base_url, mark_iso


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
    chromium_browser: Browser, dashboard: tuple[str, str]
) -> None:
    base_url, mark_iso = dashboard
    expected_js = (
        f"() => new Intl.DateTimeFormat(undefined, {{{_COMPONENTS}, "
        f"timeZone: '{FACILITY_ZONE}', timeZoneName: 'short'}})"
        f".format(new Date('{mark_iso}'))"
    )
    title, expected = _mark_title(chromium_browser, base_url, "America/Los_Angeles", expected_js)
    assert title == expected


def test_a_viewer_on_the_facility_clock_reads_the_mark_without_a_zone_name(
    chromium_browser: Browser, dashboard: tuple[str, str]
) -> None:
    base_url, mark_iso = dashboard
    expected_js = (
        f"() => new Intl.DateTimeFormat(undefined, {{{_COMPONENTS}, "
        f"timeZone: '{FACILITY_ZONE}'}})"
        f".format(new Date('{mark_iso}'))"
    )
    title, expected = _mark_title(chromium_browser, base_url, FACILITY_ZONE, expected_js)
    assert title == expected
