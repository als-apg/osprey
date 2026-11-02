"""Browser suite: a trigger's Tools field names the tools its runs may use.

The trigger detail view lists a trigger's ``allowed_tools``. That list is the
whole tool surface of the run's main thread, so an empty list grants no tool and
the field reads ``none``.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_dashboard_trigger_tools.py \
        -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from osprey.interfaces._serving import run_app_server
from tests.interfaces.design_system._dispatch_dashboard import create_dispatch_dashboard_app

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1100, "height": 900}


def _trigger(name: str, allowed_tools: list[str]) -> dict[str, Any]:
    """An armed webhook trigger with the given tool list."""
    return {
        "name": name,
        "status": "armed",
        "source": "webhook",
        "on_error": "drop",
        "allowed_tools": allowed_tools,
        "prompt": "Check the readbacks",
        "source_config": {},
        "last_fired": None,
        "next_fire": None,
    }


#: One trigger that grants no tool and one that grants two.
STATE = {
    "pool": {"running": 0, "queued": 0, "max": 2},
    "triggers": [
        _trigger("no-tools", []),
        _trigger("reader", ["read_pv", "get_archive"]),
    ],
    "runs": [],
    "timeline": {"no-tools": [], "reader": []},
    "worker_error": None,
    "server_time_iso": "2026-09-27T12:00:00+00:00",
    "facility_timezone": "UTC",
}


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    with run_app_server(create_dispatch_dashboard_app(STATE, channel_strip_prefix="")) as base_url:
        yield base_url


@pytest.mark.parametrize(
    ("trigger_name", "expected"),
    [("no-tools", "none"), ("reader", "read_pv, get_archive")],
)
def test_the_tools_field_names_the_granted_tools(
    chromium_browser: Browser, dashboard_url: str, trigger_name: str, expected: str
) -> None:
    """The Tools cell lists the granted tools, and reads ``none`` for an empty list."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    try:
        page.goto(
            f"{dashboard_url}/?view=trigger&name={trigger_name}",
            wait_until="domcontentloaded",
        )
        cell = "#trigger-detail .field-label:text-is('Tools') + .field-value"
        page.wait_for_selector(cell, timeout=10_000)
        assert page.inner_text(cell) == expected
    finally:
        page.close()
