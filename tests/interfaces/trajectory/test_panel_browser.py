"""Real-browser tests for the trajectory panel, through the in-process stack.

The vitest suites pin the panel's pure halves — what the controls offer for a
snapshot, how a session is kept or dropped, which URL each call builds. What
only a real browser against the real service proves is the composed path: the
panel loads through its operator's terminal behind ``/u/<user>/`` styled and
with no refused request, standalone and as a hub tile; another operator, or a
second tab of the same one, takes the session over; and a reload while an
action runs leaves that action running, with Take over offered only once it
has ended.

Every page runs through :mod:`tests.interfaces.trajectory._stack`: the real
trajectory app on a fake machine, each operator's real web terminal, and a
front that strips ``/u/<user>``. The stack fails any page that sees a 4xx on
load, and refuses to start without the vendored Plotly.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from osprey.services.trajectory.constants import CLIENT_HEADER
from tests.interfaces.trajectory._stack import (
    PANEL_ID,
    SETTLE_MS,
    TURNS_TRACKED,
    Stack,
    connected,
    run_stack,
    tile,
)

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if TYPE_CHECKING:
    from playwright.sync_api import Browser, FrameLocator, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

THEMES = ("light", "dark")


@pytest.fixture
def stack(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Stack]:
    with run_stack(chromium_browser, tmp_path, monkeypatch) as served:
        yield served


# -- helpers --------------------------------------------------------------------


def _control(page: Page | FrameLocator, name: str):
    return page.locator(f"[data-view=session] [data-control={name}]")


def _holder(page: Page | FrameLocator):
    return page.locator("[data-fact=holder] .fact-value")


def _open_session(page: Page, owner: str) -> str:
    """Open the session from *page*; answers the client id the grant handed it."""
    with page.expect_response(
        lambda r: r.url.endswith("/api/session") and r.request.method == "POST"
    ) as granted:
        _control(page, "session-open").click()
    expect(_control(page, "session-end")).to_be_enabled(timeout=SETTLE_MS)
    expect(_holder(page)).to_have_text(f"panel of {owner} (this tab)", timeout=SETTLE_MS)
    return granted.value.json()["client_id"]


def _assert_styled(page: Page | FrameLocator, theme: str) -> None:
    """The design system's tokens and the panel's own sheet apply, in *theme*."""
    root = page.locator("html")
    expect(root).to_have_attribute("data-theme", theme, timeout=SETTLE_MS)
    font = root.evaluate("e => getComputedStyle(e).getPropertyValue('--font-display').trim()")
    assert font, "the design-system tokens did not load"
    body = page.locator("body")
    assert "Outfit" in body.evaluate("e => getComputedStyle(e).fontFamily")
    grid = page.locator(".traj-grid")
    assert grid.evaluate("e => getComputedStyle(e).display") == "grid", "panel.css did not load"
    assert page.locator("html").evaluate("() => typeof window.Plotly") == "object"


# -- A0: the panel loads through the proxy ------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
def test_the_panel_loads_styled_through_the_proxy_standalone(stack: Stack, theme: str) -> None:
    page = stack.open("alice", theme=theme)

    assert page.url.startswith(f"{stack.front}/u/alice/panel/{PANEL_ID}/")
    _assert_styled(page, theme)
    stack.assert_loaded_clean()


@pytest.mark.parametrize("mode", ["expert", "simple"])
@pytest.mark.parametrize("theme", THEMES)
def test_the_panel_loads_styled_as_a_hub_tile(stack: Stack, theme: str, mode: str) -> None:
    page = stack.open_tile("alice", theme=theme, mode=mode)

    expect(page.locator("html")).to_have_attribute("data-ui-mode", mode)
    frame = page.locator(f'.dock-iframe-overlay iframe[data-panel-id="{PANEL_ID}"]')
    assert f"/u/alice/panel/{PANEL_ID}/" in frame.evaluate("e => e.src")
    panel = tile(page)
    _assert_styled(panel, theme)
    expect(panel.locator("html")).to_have_attribute("data-ui-mode", mode, timeout=SETTLE_MS)
    stack.assert_loaded_clean()


# -- the fake machine --------------------------------------------------------------


def _call(page: Page, method: str, path: str, client_id: str | None = None, body=None):
    """Send one panel-door call from *page*; answers ``[status, json-or-byte-count]``."""
    return page.evaluate(
        """async ([method, path, header, id, body]) => {
             const headers = {'Content-Type': 'application/json'};
             if (id) headers[header] = id;
             const r = await fetch(path, {method, headers,
               body: body == null ? undefined : JSON.stringify(body)});
             const type = r.headers.get('content-type') || '';
             return [r.status, type.includes('json') ? await r.json()
                                                       : (await r.arrayBuffer()).byteLength];
           }""",
        [method, path, CLIENT_HEADER, client_id, body],
    )


def test_the_fake_machine_serves_what_the_panel_reads(stack: Stack) -> None:
    """The chain, catalog, machine facts, solve, settings readbacks and a raw
    buffer all reach the panel door through the proxy."""
    page = stack.open("alice")
    status, state = _call(page, "GET", "api/state")
    assert status == 200
    assert [step["id"] for step in state["chain"]] == ["arm", "inject", "clean", "readout"]
    assert [entry["label"] for entry in state["catalog"]["bpms"]] == ["BPM1", "BPM2", "BPM3"]
    assert state["catalog"]["bpms"][1]["places"][-1] == {"id": "SR/SECT2", "level": "sector"}
    assert "families" not in state
    assert state["machine"] == {"rf_on": 1, "stored_current": 0}
    assert state["setup"] is None
    assert state["solve"] is not None and state["solve"]["singular_values"]

    status, readbacks = _call(page, "GET", "api/setup")
    assert status == 200
    assert {item["id"] for item in readbacks["settings"]} == {
        "attenuation",
        "bpm_mode",
        "pilot_tone",
        "turns_tracked",
    }

    client_id = _open_session(page, "alice")
    status, answer = _call(page, "POST", "api/setup", client_id, {"settings": {}})
    assert status == 200, answer
    assert _call(page, "GET", "api/state")[1]["setup"] is not None
    status, shot = _call(
        page, "POST", "api/shot", client_id, {"steps": ["arm", "inject", "readout"]}
    )
    assert status == 200, shot
    shot_id = shot["shot"]["shot_id"]
    status, listed = _call(page, "GET", "api/shots")
    assert status == 200
    assert [item["shot_id"] for item in listed["shots"]] == [shot_id]
    status, size = _call(page, "GET", f"api/shots/{shot_id}/raw")
    assert status == 200 and size > 0

    stack.machine.stage("nominal")
    assert stack.machine.staged == ["nominal", "nominal"]
    with pytest.raises(KeyError, match="known"):
        stack.machine.stage("no-such-state")


# -- take-over -------------------------------------------------------------------


def test_another_operator_and_a_second_tab_of_the_same_one_take_over(stack: Stack) -> None:
    alice = stack.open("alice")
    _open_session(alice, "alice")

    bob = stack.open("bob")
    expect(_holder(bob)).to_have_text("panel of alice")
    expect(_control(bob, "session-open")).to_be_disabled()
    expect(_control(bob, "session-end")).to_be_disabled()
    _control(bob, "session-takeover").click()
    expect(_holder(bob)).to_have_text("panel of bob (this tab)", timeout=SETTLE_MS)
    expect(_control(bob, "session-end")).to_be_enabled()
    expect(_control(bob, "session-takeover")).to_be_disabled()

    expect(_control(alice, "session-takeover")).to_be_enabled(timeout=SETTLE_MS)
    expect(_control(alice, "session-end")).to_be_disabled()
    expect(_holder(alice)).to_have_text("panel of bob")
    expect(alice.locator(".notice-taken-over")).to_have_text(
        "Taken over by panel of bob from panel of alice."
    )

    second = stack.open("alice")
    expect(_control(second, "session-end")).to_be_disabled()
    _control(second, "session-takeover").click()
    expect(_holder(second)).to_have_text("panel of alice (this tab)", timeout=SETTLE_MS)
    expect(_control(bob, "session-takeover")).to_be_enabled(timeout=SETTLE_MS)
    expect(_control(bob, "session-end")).to_be_disabled()
    # The first tab of the same operator is a different client: it is not handed
    # the session back by its owner's take-over.
    expect(_holder(alice)).to_have_text("panel of alice")
    expect(_control(alice, "session-takeover")).to_be_enabled()
    expect(_control(alice, "session-end")).to_be_disabled()


# -- a reload never stops the action ---------------------------------------------


def test_a_reload_mid_setup_leaves_the_setup_running(stack: Stack) -> None:
    """Reload while a setup is held mid-write: it keeps running, and the
    reloaded tab offers Take over only after it ended."""
    page = stack.open("alice")
    client_id = _open_session(page, "alice")
    stack.machine.hold()
    body = json.dumps({"settings": {"turns_tracked": TURNS_TRACKED + 10}})
    page.evaluate(
        """([header, id, body]) => { void fetch('api/setup', {method: 'POST',
             headers: {'Content-Type': 'application/json', [header]: id}, body}); }""",
        [CLIENT_HEADER, client_id, body],
    )

    assert stack.machine.waiting.wait(SETTLE_MS / 1000), "the setup never reached its write"
    page.reload(wait_until="load")
    connected(page)
    expect(page.locator("[data-running=setup]")).to_be_visible(timeout=SETTLE_MS)
    expect(_holder(page)).to_have_text("panel of alice", timeout=SETTLE_MS)
    expect(_control(page, "session-takeover")).to_be_disabled()
    expect(_control(page, "session-open")).to_be_disabled()
    expect(_control(page, "session-end")).to_be_disabled()

    stack.machine.release()
    expect(_control(page, "session-takeover")).to_be_enabled(timeout=SETTLE_MS)
    expect(page.locator("[data-view=status] .running")).to_have_text("Nothing running")
    assert stack.machine.values["SR:DIAG:TBT:TURNS:RB"] == TURNS_TRACKED + 10
