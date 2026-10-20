"""Real-browser tests for the trajectory panel's BPM Control drawer.

The vitest suite pins what the drawer renders from the shared settings model
and the body it commits. What only a real browser through the stack proves is
the composed path: Open BPM control slides the page's ``<osprey-drawer>`` in,
styled in either theme, its readback plots drawn by Plotly at a real size once
the drawer is open, and a setting committed from the drawer reaching the setup
route as settings only and reading back in both the drawer and the tab.

Every page runs through :mod:`tests.interfaces.trajectory._stack`. Skips
cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.interfaces.trajectory._stack import (
    SETTLE_MS,
    Stack,
    run_stack,
)

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

THEMES = ("light", "dark")
DRAWER = "osprey-drawer#bpm-control-drawer"
READBACK_PLOTS = ("plot-bpm-atten", "plot-bpm-pt-state")


@pytest.fixture
def stack(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Stack]:
    with run_stack(chromium_browser, tmp_path, monkeypatch) as served:
        yield served


def _in_drawer(page: Page, name: str):
    return page.locator(f"{DRAWER} [data-control={name}]")


def _in_tab(page: Page, name: str):
    return page.locator(f"[data-area=setup] [data-control={name}]")


def _open_drawer(page: Page) -> None:
    _in_tab(page, "open-bpm-control").click()
    drawer = page.locator(DRAWER)
    expect(drawer).to_have_attribute("open", "", timeout=SETTLE_MS)
    expect(page.locator("#drawer-backdrop")).to_have_attribute("open", "", timeout=SETTLE_MS)
    # The slide-in has finished: the drawer's right edge is the viewport's.
    page.wait_for_function(
        """sel => { const r = document.querySelector(sel).getBoundingClientRect();
                    return Math.abs(r.right - window.innerWidth) < 1; }""",
        arg=DRAWER,
        timeout=SETTLE_MS,
    )


def _open_session(page: Page) -> None:
    page.locator("[data-view=session] [data-control=session-open]").click()
    expect(page.locator("[data-view=session] [data-control=session-end]")).to_be_enabled(
        timeout=SETTLE_MS
    )


@pytest.mark.parametrize("theme", THEMES)
def test_open_bpm_control_opens_the_drawer_styled_with_sized_plots(
    stack: Stack, theme: str
) -> None:
    page = stack.open("alice", theme=theme)
    expect(page.locator("html")).to_have_attribute("data-theme", theme, timeout=SETTLE_MS)
    drawer = page.locator(DRAWER)
    expect(drawer).not_to_have_attribute("open", "")

    _open_drawer(page)

    assert drawer.evaluate("e => getComputedStyle(e).position") == "fixed", "drawer CSS missing"
    background = drawer.evaluate("e => getComputedStyle(e).backgroundColor")
    assert background not in ("", "rgba(0, 0, 0, 0)", "transparent"), "drawer has no background"
    expect(_in_drawer(page, "stored-current-readback")).to_have_text("0.000 mA", timeout=SETTLE_MS)
    for name in ("setting-pilot-tone", "setting-bpm-mode", "setting-attenuation"):
        expect(_in_drawer(page, name)).to_be_visible(timeout=SETTLE_MS)
    for plot_id in READBACK_PLOTS:
        plot = _in_drawer(page, plot_id)
        expect(plot.locator(".main-svg").first).to_be_attached(timeout=SETTLE_MS)
        size = plot.evaluate(
            "e => { const s = e.querySelector('.main-svg').getBoundingClientRect();"
            " return [s.width, s.height]; }"
        )
        assert size[0] > 0 and size[1] > 0, f"{plot_id} drew at size {size}"
        # The plot fills the drawer body, not Plotly's default 700 px.
        body = page.locator(f"{DRAWER} .drawer-body").evaluate(
            "e => e.getBoundingClientRect().width"
        )
        assert size[0] <= body, f"{plot_id} is {size[0]} wide in a {body} wide drawer"

    page.locator(f"{DRAWER} .drawer-close-btn").click()
    expect(drawer).not_to_have_attribute("open", "", timeout=SETTLE_MS)
    stack.assert_loaded_clean()


def test_a_drawer_setting_commits_settings_only_and_reads_back_in_both_views(
    stack: Stack,
) -> None:
    page = stack.open("alice")
    _open_session(page)
    _open_drawer(page)
    pilot = _in_drawer(page, "setting-pilot-tone")
    expect(pilot).to_be_enabled(timeout=SETTLE_MS)
    expect(pilot).to_have_value("0")

    pilot.fill("1")
    pilot.press("Enter")
    pilot.blur()
    with page.expect_request(lambda r: r.url.endswith("/api/setup") and r.method == "POST") as sent:
        _in_drawer(page, "bpm-control-set").click()

    assert sent.value.post_data_json == {"settings": {"pilot_tone": 1}}
    expect(_in_drawer(page, "setting-pilot-tone")).to_have_value("1", timeout=SETTLE_MS)
    expect(_in_tab(page, "setting-pilot-tone")).to_have_value("1", timeout=SETTLE_MS)
    expect(_in_drawer(page, "bpm-control-set")).to_be_disabled(timeout=SETTLE_MS)
    for device in stack.machine.devices:
        address = f"{device.replace('/', ':')}:PILOT_TONE:RB"
        assert stack.machine.values[address] == 1
    stack.assert_loaded_clean()
