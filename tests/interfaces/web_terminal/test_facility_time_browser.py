"""Web terminal times read on the facility clock in a real browser.

The server stamps the facility zone on ``<html data-facility-timezone>``, and
the hook log and the Activity page render every stamp through the design
system's facility-time formatter. A viewer whose browser runs in
another zone must still read the facility's wall clock, and the hook log's
time column names the zone; a viewer already on the facility clock sees a
plain ``Time`` header.

The instant ``2026-01-15T20:04:05.123Z`` reads ``05:04`` in Tokyo and ``12:04``
in Los Angeles, so a viewer-zone formatter fails these tests.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_facility_time_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from osprey.agent_record.client import ClientAnswer
from osprey.interfaces.web_terminal.routes import session as session_routes
from tests.agent_record.fakes import FakeRecordClient
from tests.interfaces._launchers import launch_web_terminal

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

FACILITY_ZONE = "Asia/Tokyo"
INSTANT = "2026-01-15T20:04:05.123Z"

_HOOK_LOG = {
    "entries": [{"ts": INSTANT, "hook": "PreToolUse", "tool": "Bash", "status": "allowed"}]
}
TRANSCRIPT = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
# One record entry of the session: a tool call at the instant.
_TIMELINE = [
    {
        "kind": "tool_call",
        "key": "k:call:t1",
        "ts": INSTANT,
        "tool_use_id": "t1",
        "body": {"id": "t1", "name": "mcp__controls__channel_read", "input": {}},
    }
]


@contextmanager
def _facility_web_terminal(tmp_path, monkeypatch):
    """A live web terminal whose facility zone is :data:`FACILITY_ZONE`."""
    with patch(
        "osprey.interfaces.web_terminal.app.get_facility_timezone",
        return_value=ZoneInfo(FACILITY_ZONE),
    ):
        with launch_web_terminal(tmp_path, monkeypatch) as base_url:
            yield base_url


def _fulfill_json(payload):
    body = json.dumps(payload)
    return lambda route: route.fulfill(status=200, content_type="application/json", body=body)


def _open_hook_log(page, base_url):
    """Load the index and expand the hook log in the Settings drawer."""
    page.route("**/api/hooks/debug-log**", _fulfill_json(_HOOK_LOG))
    page.goto(f"{base_url}/", wait_until="load")
    page.wait_for_selector("#hook-debug-log-toggle", state="attached")
    page.evaluate("document.getElementById('hook-debug-log-toggle').click()")
    page.wait_for_selector("td.log-ts", state="attached")


def test_hook_log_and_session_activity_read_the_facility_clock(
    tmp_path, monkeypatch, chromium_browser
):
    with _facility_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(timezone_id="America/Los_Angeles", locale="en-US")

        _open_hook_log(page, base_url)
        assert page.locator("td.log-ts").first.text_content() == "05:04:05.123"
        assert page.locator(".hook-debug-log-table thead th").first.text_content() == (
            "Time (GMT+9)"
        )

        record = FakeRecordClient({"timeline": ClientAnswer(data=_TIMELINE, complete=True)})
        monkeypatch.setattr(session_routes, "record_client", lambda: record)
        page.goto(f"{base_url}/static/session.html#{TRANSCRIPT}", wait_until="load")
        expect(page.locator("button.act-row .act-t").first).to_have_text("05:04:05")
        page.locator('#views button[data-view="tools"]').click()
        expect(page.locator("table.act-table tr.row .act-t").first).to_have_text("05:04:05")


def test_viewer_on_the_facility_clock_sees_no_zone_name(tmp_path, monkeypatch, chromium_browser):
    with _facility_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(timezone_id=FACILITY_ZONE, locale="en-US")

        _open_hook_log(page, base_url)
        assert page.locator("td.log-ts").first.text_content() == "05:04:05.123"
        assert page.locator(".hook-debug-log-table thead th").first.text_content() == "Time"
