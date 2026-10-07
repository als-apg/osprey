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

import pathlib
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


def _create_dispatch_dashboard_app() -> FastAPI:
    """Serve the real dashboard beside the real design system and a fixed state."""
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(render_dashboard_html(facility_name="Test Facility"))

    @app.get("/dashboard/state")
    async def dashboard_state() -> dict:
        return STATE

    return app


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    with _run_app_server(_create_dispatch_dashboard_app()) as base_url:
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
            f"{dashboard_url}/?mode=expert&view=trigger&name={trigger_name}",
            wait_until="domcontentloaded",
        )
        cell = "#trigger-detail .field-label:text-is('Tools') + .field-value"
        page.wait_for_selector(cell, timeout=10_000)
        assert page.inner_text(cell) == expected
    finally:
        page.close()
