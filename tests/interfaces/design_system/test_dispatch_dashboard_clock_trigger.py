"""Browser suite: a clock trigger's row shows its times and its next fire.

A ``source: cron`` trigger with ``at``/``days`` fires at clock times in the
facility zone. The trigger list summarises it by those times and days, and its
sub-line names the next fire, rendered in the facility zone the dashboard state
carries.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_dashboard_clock_trigger.py \
        -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

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

TRIGGER_NAME = "morning-report"

#: One armed clock trigger; its next fire is a Monday 07:45 in Berlin (CEST).
STATE = {
    "pool": {"running": 0, "queued": 0, "max": 2},
    "triggers": [
        {
            "name": TRIGGER_NAME,
            "status": "armed",
            "source": "cron",
            "on_error": "drop",
            "allowed_tools": [],
            "prompt": "Summarise the night",
            "source_config": {"at": ["07:45"], "days": ["mon"]},
            "last_fired": None,
            "next_fire": "2026-09-28T05:45:00+00:00",
        },
    ],
    "runs": [],
    "timeline": {TRIGGER_NAME: []},
    "worker_error": None,
    "server_time_iso": "2026-09-27T12:00:00+00:00",
    "facility_timezone": "Europe/Berlin",
}


def _create_dispatch_dashboard_app() -> FastAPI:
    """Serve the real dashboard beside the real design system and a fixed state."""
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(render_dashboard_html(facility_name="Test Facility", channel_strip_prefix="SR:"))

    @app.get("/dashboard/state")
    async def dashboard_state() -> dict:
        return STATE

    return app


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    with _run_app_server(_create_dispatch_dashboard_app()) as base_url:
        yield base_url


def test_a_clock_trigger_shows_its_times_and_next_fire(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """In Expert mode, the trigger row names the times, the days and the next fire."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    page.goto(f"{dashboard_url}/?mode=expert", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.documentElement.getAttribute('data-ui-mode') === 'expert'",
        timeout=10_000,
    )
    page.click("#tab-triggers")
    page.wait_for_selector("#trigger-list .trigger-row", timeout=10_000)
    # The next fire renders once the facility-time module has loaded.
    page.wait_for_function(
        "() => (document.querySelector('#trigger-list .trigger-row .trigger-sub')"
        " || {}).textContent?.includes('next')",
        timeout=10_000,
    )
    stamped = page.evaluate("() => document.documentElement.getAttribute('data-facility-timezone')")
    assert stamped == "Europe/Berlin"

    row_text = page.inner_text("#trigger-list .trigger-row")
    assert "07:45 · mon" in row_text, row_text
    assert "next" in row_text, row_text
    assert "07:45" in row_text.split("next", 1)[1], row_text
    page.close()
