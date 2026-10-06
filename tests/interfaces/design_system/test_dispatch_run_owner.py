"""Browser suite: a run detail names the person who fired the run.

The worker's dashboard feed projects ``owner`` on every run: the person the fire
was attributed to, or ``null`` for an owner-less fire (cron, webhook, chat
bridge). The run detail's Expert summary line says who fired an attributed run
and says nothing for an owner-less one. Simple mode shows no owner: its summary
is the subtraction of the Expert one, so the line adds no Simple surface.

These tests drive the real rendered dashboard in a browser and assert on the
rendered summary text, not on the markup the renderer might never reach.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_run_owner.py -m browser -v

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

    from playwright.sync_api import Browser, Page

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

TRIGGER_NAME = "hello-dispatch"
ATTRIBUTED_RUN_ID = "run-fired-by-alice"
OWNER_LESS_RUN_ID = "run-fired-by-nobody"

#: One trigger and two runs: one a person fired, one nobody did.
STATE = {
    "triggers": [
        {"name": TRIGGER_NAME, "status": "armed", "source": "manual", "on_error": "drop"},
    ],
    "runs": [
        {
            "run_id": ATTRIBUTED_RUN_ID,
            "trigger_name": TRIGGER_NAME,
            "owner": "alice",
            "status": "completed",
            "created_at": 1785744790,
            "duration_sec": 18.4,
            "age_sec": 30,
            "num_turns": 3,
            "tool_count": 2,
            "text_output": "done",
            "tool_calls": [],
        },
        {
            "run_id": OWNER_LESS_RUN_ID,
            "trigger_name": TRIGGER_NAME,
            "owner": None,
            "status": "completed",
            "created_at": 1785744790,
            "duration_sec": 18.4,
            "age_sec": 30,
            "num_turns": 3,
            "tool_count": 2,
            "text_output": "done",
            "tool_calls": [],
        },
    ],
}


def _create_dispatch_dashboard_app() -> FastAPI:
    """Serve the real ``render_dashboard_html()`` output beside the real design system."""
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(
            render_dashboard_html(
                facility_name="Test Facility",
                channel_strip_prefix="SR:",
                telemetry_url="",
            )
        )

    return app


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    with _run_app_server(_create_dispatch_dashboard_app()) as base_url:
        yield base_url


def _open_run(page: Page, base_url: str, run_id: str, mode: str) -> None:
    """Load the dashboard in *mode*, install a known state, and route to one run.

    The state poll has no dispatcher behind it here, so ``state`` is assigned
    directly and the router is driven through ``navigate()``, which is what
    repaints the screen.
    """
    page.goto(f"{base_url}/?mode={mode}", wait_until="domcontentloaded")
    page.wait_for_function(
        "mode => document.documentElement.getAttribute('data-ui-mode') === mode",
        arg=mode,
        timeout=10_000,
    )
    page.wait_for_function("() => typeof navigate === 'function'", timeout=10_000)
    page.evaluate(
        """([nextState, runId]) => {
            state.triggers = nextState.triggers;
            state.runs = nextState.runs;
            navigate({ name: 'run', runId });
        }""",
        [STATE, run_id],
    )
    page.wait_for_selector("#run-detail .detail-summary", state="attached", timeout=10_000)


def _visible_summaries(page: Page) -> list[str]:
    """Text of every run-detail summary line the operator can see."""
    return page.evaluate(
        """() => Array.from(
             document.querySelectorAll('#run-detail .detail-summary')
           ).filter(e => e.checkVisibility()).map(e => e.textContent)"""
    )


def test_run_detail_names_who_fired_it(chromium_browser: Browser, dashboard_url: str) -> None:
    """In Expert mode, an attributed run's summary names its owner."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, ATTRIBUTED_RUN_ID, "expert")

    summaries = _visible_summaries(page)
    assert any("fired by alice" in text for text in summaries), summaries
    page.close()


def test_run_detail_names_nobody_for_an_owner_less_run(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """In Expert mode, an owner-less run's summary names nobody."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, OWNER_LESS_RUN_ID, "expert")

    summaries = _visible_summaries(page)
    assert summaries, "the run detail should show a summary line"
    assert not any("fired by" in text for text in summaries), summaries
    page.close()


def test_simple_mode_does_not_show_the_owner(chromium_browser: Browser, dashboard_url: str) -> None:
    """In Simple mode, the attributed run's visible summary does not name the owner."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, ATTRIBUTED_RUN_ID, "simple")

    summaries = _visible_summaries(page)
    assert summaries, "the run detail should show a summary line"
    assert not any("fired by" in text for text in summaries), summaries
    page.close()
