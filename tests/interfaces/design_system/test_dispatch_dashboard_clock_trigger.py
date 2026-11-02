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

from typing import TYPE_CHECKING

import pytest

from osprey.interfaces._serving import run_app_server
from tests.interfaces.design_system._dispatch_dashboard import create_dispatch_dashboard_app

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1100, "height": 900}


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


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    with run_app_server(create_dispatch_dashboard_app(STATE)) as base_url:
        yield base_url


def test_a_clock_trigger_shows_its_times_and_next_fire(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """The trigger row names the times, the days and the next fire."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    page.goto(f"{dashboard_url}/", wait_until="domcontentloaded")
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
