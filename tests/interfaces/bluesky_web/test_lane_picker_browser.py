"""Real-browser tests for the BLUESKY panel's plan-lane picker.

The vitest suite (``lane-client.test.mjs``) pins the pure half — roster
parsing, the ``?lane=`` spelling, the search-string rewrite. What only a real
browser can prove is the composed behaviour this feature exists for: a panel
document pinned to the second lane sends EVERY bridge-bound request — boot
fetches and both SSE streams alike — to that lane's bridge and none to lane
1's, the picker renders from the sidecar's roster with the current lane held
down, and a single-lane deployment shows no picker at all.

Runs the composed sidecar for real (``bluesky_live_server``), then swaps in a
two-lane world the way every suite in this directory swaps the bridge: a
recording ``httpx.MockTransport`` answering for BOTH bridge hosts, plus the
``app.state`` lane map/roster the lifespan would have resolved from a
two-lane render.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from osprey.interfaces.bluesky_web.app import app
from tests.interfaces.bluesky_web.conftest import STUB_BRIDGE_URL, _stub_bridge

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if TYPE_CHECKING:
    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VA_BRIDGE_URL = "http://bridge-va.test"

_PANEL_SETTLE_MS = 15_000


def _wire_two_lanes() -> list[str]:
    """Swap the served app into a two-lane shape; return the recorded hosts.

    Must be called inside a ``bluesky_live_server()`` context (the lifespan
    has run). The recording handler answers for both bridge hosts with the
    same stub answers, so which HOST each request reached is the entire
    routing claim.
    """
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return _stub_bridge(request)

    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.state.bridge_url = STUB_BRIDGE_URL
    app.state.bridge_urls = {"bluesky": STUB_BRIDGE_URL, "bluesky_va": VA_BRIDGE_URL}
    app.state.lanes = [
        {"lane": "bluesky", "lane_target": "live"},
        {"lane": "bluesky_va", "lane_target": "va"},
    ]
    return hosts


def test_single_lane_deployment_shows_no_picker(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    """Every deployment until a second lane is opted in: panel unchanged."""
    with bluesky_live_server() as base_url:
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/bluesky/", wait_until="load")
            expect(page.locator("#queue-state-badge")).to_be_visible(timeout=_PANEL_SETTLE_MS)
            expect(page.locator("#lane-strip")).to_be_hidden()
            expect(page.locator("#lane-banner")).to_be_hidden()
        finally:
            page.close()


def test_second_lane_document_routes_every_bridge_request_to_that_lane(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    """The claim the feature exists for, end to end.

    A panel opened on ``?lane=bluesky_va`` boots its plans, draft, queue,
    runs, capability read and both SSE streams against the VA bridge and
    touches lane 1's bridge not once — the wrong-machine mixture the one-lane-
    per-document design makes unrepresentable.
    """
    with bluesky_live_server() as base_url:
        hosts = _wire_two_lanes()
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/bluesky/?lane=bluesky_va", wait_until="load")

            strip = page.locator("#lane-strip")
            expect(strip).to_be_visible(timeout=_PANEL_SETTLE_MS)
            buttons = strip.locator("button")
            expect(buttons).to_have_count(2)
            # Labelled by target — which machine, not which index.
            expect(buttons.nth(0)).to_have_text("live")
            expect(buttons.nth(1)).to_have_text("va")
            expect(buttons.nth(1)).to_have_attribute("aria-pressed", "true")
            expect(page.locator("#lane-banner")).to_be_hidden()

            # The boot fetches have long since landed once the strip rendered;
            # the routing claim is over everything recorded.
            assert hosts, "the panel booted without a single bridge fetch"
            assert set(hosts) == {"bridge-va.test"}
        finally:
            page.close()


def test_picker_switches_back_to_lane_one_by_navigation(
    chromium_browser: Browser, bluesky_live_server
) -> None:
    """Clicking the other lane is a navigation to a fresh document whose URL
    no longer names a lane — lane 1 stays off the wire, as it always was."""
    with bluesky_live_server() as base_url:
        hosts = _wire_two_lanes()
        page = chromium_browser.new_page()
        try:
            page.goto(f"{base_url}/bluesky/?lane=bluesky_va", wait_until="load")
            strip = page.locator("#lane-strip")
            expect(strip).to_be_visible(timeout=_PANEL_SETTLE_MS)

            hosts.clear()
            with page.expect_navigation(wait_until="load"):
                strip.locator("button", has_text="live").click()

            assert "lane=" not in page.url
            expect(page.locator("#lane-strip")).to_be_visible(timeout=_PANEL_SETTLE_MS)
            expect(page.locator("#lane-strip button", has_text="live")).to_have_attribute(
                "aria-pressed", "true"
            )
            assert hosts, "the reloaded panel booted without a single bridge fetch"
            assert set(hosts) == {"bridge.test"}
        finally:
            page.close()
