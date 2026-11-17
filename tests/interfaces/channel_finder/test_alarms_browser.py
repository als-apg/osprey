"""Browser test: the Channels panel's Alarms view on the demo's alarm replay.

The build and the served panel are the All PVs browser test's: the demo preset
with the simulator's section, whose ``alarm_state`` replays the packaged alarm
feed. After the replay, two channels are in alarm: a latched MAJOR (the RF
cavity temperature, back to NO_ALARM and unacknowledged) and an acknowledged
MINOR (a vacuum gauge).

Run:
    .venv/bin/pytest tests/interfaces/channel_finder/test_alarms_browser.py -m browser -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("osprey_connectors.service_answers")

from tests.interfaces.channel_finder.test_all_pvs_browser import (
    LATCHED_PV,
    _open,
    panel_server,  # noqa: F401  (fixture)
    panel_url,  # noqa: F401  (fixture)
    render,  # noqa: F401  (fixture)
)

if TYPE_CHECKING:
    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

ROWS = ".alarms-list tbody tr[data-name]"
#: The replay's guidance for the latched cavity.
GUIDANCE = "Check cavity 1 cooling water flow"


def test_the_alarms_view_on_the_demo_replay(chromium_browser: Browser, panel_url: str):  # noqa: F811
    host = _open(chromium_browser, panel_url, "#/alarms")
    host.wait_for_selector(ROWS)

    pill = host.locator('#cf-views button[data-view="alarms"] .osprey-count')
    pill.wait_for()
    assert pill.inner_text() == "2"
    assert host.locator('[data-tile="MAJOR"] .alarms-tile-n').inner_text() == "1"

    latched = host.locator(f'{ROWS}[data-name="{LATCHED_PV}"]')
    assert "latched" in latched.locator(".alarms-flags").inner_text()

    inspector = host.locator(".alarms-inspector")
    host.wait_for_function(
        "text => document.querySelector('.alarms-inspector')?.textContent.includes(text)",
        arg=GUIDANCE,
    )
    guidance = inspector.locator(".alarms-guidance").first
    assert guidance.locator("strong").inner_text() == "Response"
    assert GUIDANCE in guidance.inner_text()
