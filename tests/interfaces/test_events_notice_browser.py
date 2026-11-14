"""The EVENTS panel's warning banner.

The dispatcher dashboard has no status line and the tile bar carries no text,
so the warnings nothing else on screen says (the dispatcher or its worker is
unreachable, the queue is full) show as a banner at the top of the body, and
only while they hold. Standalone and embedded show the same body. The page is
served from dashboard.html as is, with the design-system assets from the tree
and ``/dashboard/state`` stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import Route, expect

from osprey.dispatch.dashboard import render_dashboard_html
from tests.interfaces._browser import wait_for_page_ready

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_ORIGIN = "http://events.test"
_DESIGN_SYSTEM = Path(__file__).resolve().parents[2] / "src/osprey/interfaces/design_system/static"
_CONTENT_TYPES = {".js": "text/javascript", ".css": "text/css"}


def _state(*, worker_error: str | None = None, queued: int = 0) -> dict:
    return {
        "pool": {"running": 0, "queued": queued, "max": 2},
        "triggers": [],
        "runs": [],
        "timeline": {},
        "server_time_iso": "2026-11-06T12:00:00Z",
        "facility_timezone": None,
        "worker_error": worker_error,
    }


def _open(chromium_browser, state: dict, *, embedded: bool = True):
    page = chromium_browser.new_page()

    def handle(route: Route) -> None:
        path = route.request.url.removeprefix(_ORIGIN).split("?")[0]
        if path == "/":
            route.fulfill(body=render_dashboard_html(), content_type="text/html")
        elif path == "/dashboard/state":
            route.fulfill(body=json.dumps(state), content_type="application/json")
        elif path.startswith("/design-system/"):
            file = _DESIGN_SYSTEM / path.removeprefix("/design-system/")
            if file.is_file():
                route.fulfill(
                    body=file.read_bytes(),
                    content_type=_CONTENT_TYPES.get(file.suffix, "application/octet-stream"),
                )
            else:
                route.fulfill(status=404, body="")
        else:
            route.fulfill(status=404, body="")

    page.route(f"{_ORIGIN}/**", handle)
    page.goto(f"{_ORIGIN}/{'?embedded=true' if embedded else ''}")
    wait_for_page_ready(page)
    # The first poll has landed once the panel-wide "Connecting" state is gone.
    expect(page.locator("#panel-state")).to_be_hidden()
    return page


@pytest.mark.parametrize(
    ("state", "lines"),
    [
        (
            _state(worker_error="connection refused"),
            ["Cannot reach the dispatch worker — the runs below may be out of date."],
        ),
        (_state(queued=3), ["Queue full — 3 waiting."]),
        (
            _state(worker_error="connection refused", queued=1),
            [
                "Cannot reach the dispatch worker — the runs below may be out of date.",
                "Queue full — 1 waiting.",
            ],
        ),
    ],
    ids=["worker-unreachable", "queue-full", "both"],
)
def test_the_banner_shows_while_a_warning_holds(chromium_browser, state, lines):
    page = _open(chromium_browser, state)
    banner = page.locator("#embed-notice")
    expect(banner).to_be_visible()
    expect(banner.locator("div")).to_have_text(lines)


def test_no_banner_when_all_is_well(chromium_browser):
    page = _open(chromium_browser, _state())
    expect(page.locator("#embed-notice")).to_be_hidden()


def test_standalone_shows_the_same_banner(chromium_browser):
    page = _open(chromium_browser, _state(worker_error="connection refused"), embedded=False)
    expect(page.locator("#status-text")).to_have_count(0)
    expect(page.locator("#embed-notice div")).to_have_text(
        ["Cannot reach the dispatch worker — the runs below may be out of date."]
    )
