"""Real-browser tests for the trajectory panel.

The vitest suites pin the panel's pure halves — what the controls offer for a
snapshot, how a session is kept or dropped, which URL each call builds. What
only a real browser against the real service proves is the composed loop: an
operator opens a session, sets the machine up, shoots, applies the step, puts
it back and runs continuously, with every request reaching the panel door;
another operator, or a second tab of the same one, takes the session over; and
a reload while an action runs leaves that action running, with Take over
offered only once it has ended.

The app is the real one behind the real shared web gate. Each browser context
presents a roster secret, so the gate stamps that operator as owner. The
admission gate reads a real control-context tree under a throwaway deployment
root and the run lock is the real one; the connector is a stand-in whose
injection fills turn-by-turn waveforms and whose checked writes can be held
open, the correction tool answers a set step, and the limits validator is
stubbed.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np
import pytest

from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import WebCredentials
from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory import actions
from osprey.services.trajectory.app import LAUNCH_TOKEN_ENV, create_app
from osprey.services.trajectory.sequence import SEQUENCE_MODULE_FILE, load_sequence
from osprey.services.trajectory.startup import (
    TrajectoryCommands,
    TrajectorySettings,
    TrajectoryStartup,
)
from osprey.services.trajectory.step import TrajectoryTool
from osprey.services.trajectory.store import STATE_DIR_ENV_VAR
from osprey_connectors import posture_store
from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult, WriteOutcome
from osprey_connectors.identity import AUDIT_IDENTITY_ENV
from tests._guarded_run import reset_run_state
from tests.interfaces.conftest import _run_app_server

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Request

pytestmark = [pytest.mark.browser, pytest.mark.slow]

TARGET = "va"
OWN_SECRET = "own-secret-" + "x" * 32
ALICE_SECRET = "alice-secret-" + "a" * 32
BOB_SECRET = "bob-secret-" + "b" * 32

ARM, ARMED = "SR:BPM:ARM", "SR:BPM:ARMED"
REQUEST, BUSY = "SR:INJ:REQ", "SR:INJ:BUSY"
COMMANDS = TrajectoryCommands(
    model="SR", arm=(ARM,), injection_request=REQUEST, injection_busy=BUSY, armed=(ARMED,)
)
X_ROWS = ("SR:BPM1:X", "SR:BPM2:X", "SR:BPM3:X")
Y_ROWS = ("SR:BPM1:Y", "SR:BPM2:Y", "SR:BPM3:Y")
SUMS = ("SR:BPM1:SUM", "SR:BPM2:SUM", "SR:BPM3:SUM")
HCM1, HCM2, VCM1 = "SR:HCM1", "SR:HCM2", "SR:VCM1"
BEFORE = {HCM1: 1.0, HCM2: -2.0, VCM1: 0.5}
STEP = {HCM1: 0.2, HCM2: -0.1, VCM1: 0.05}
#: What an apply adds to each corrector: the correction undoes the tool's step.
DELTAS = {name: -value for name, value in STEP.items()}
KICKER = "SR:BUNCH:KICKER:ENABLE"

#: Turn-by-turn samples per record, and the sample the beam arrives at.
SAMPLES, ARRIVAL = 8, 3
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
#: Seconds each injection takes, so a continuous run lasts until Stop.
INJECTION_S = 0.1
#: A cap a continuous run never reaches before the test presses Stop.
CONTINUOUS_CAP = "1000"

#: The panel's sequence module: an injection, and a bunch clean that pulses
#: the kicker enable through a checked write.
SEQUENCE_SOURCE = f'''
SEQUENCE_METADATA = {{
    "slots": {{"kicker_enable": "{KICKER}"}},
    "writes": {{"bunch_clean": ["kicker_enable"]}},
    "max_pause_s": 0.5,
}}


async def inject(handle):
    await handle.write("injection_request", 1)


async def bunch_clean(handle):
    await handle.write("kicker_enable", 1)
    await handle.write("kicker_enable", 0)
'''

SETTLE_MS = 15_000


class _Beamline:
    """Corrector and kicker settings, one arming device and three BPMs.

    An arm sets the armed flag; an injection request, once armed, posts a
    fresh turn-by-turn shot on every BPM record, stamped later than the last.
    A checked write waits while the gate is held (:meth:`hold`), so a test can
    keep a setup or an apply in flight.
    """

    _control_target = TARGET

    def __init__(self) -> None:
        self.values: dict[str, Any] = {**BEFORE, KICKER: 0, BUSY: 0, ARMED: 0}
        self.stamps: dict[str, datetime] = {}
        for address in (*X_ROWS, *Y_ROWS, *SUMS):
            self.values[address] = [0.0] * SAMPLES
        self.shots = 0
        self.writes: list[tuple[str, Any]] = []
        self._open = threading.Event()
        self._open.set()
        self.waiting = threading.Event()
        self._limits_validator: Any = None

    def hold(self) -> None:
        self.waiting.clear()
        self._open.clear()

    def release(self) -> None:
        self._open.set()

    def _current_value_reader(self) -> None:
        return None

    async def read_channel(self, address: str, timeout: float | None = None) -> ChannelValue:  # noqa: ARG002
        return ChannelValue(
            value=self.values.get(address, 0), timestamp=self.stamps.get(address, EPOCH)
        )

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        return {address: await self.read_channel(address, timeout) for address in channel_addresses}

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> Any:
        while not self._open.is_set():
            self.waiting.set()
            await asyncio.sleep(0.02)
        self.writes.append((address, value))
        self.values[address] = value
        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )

    async def write_multiple_channels(
        self, operations: list[tuple[str, Any]], **_: Any
    ) -> list[ChannelWriteResult]:
        for address, value in operations:
            if address == ARM and value == 1:
                self.values[ARMED] = 1
            elif address == REQUEST and value == 1:
                await asyncio.sleep(INJECTION_S)
                self._inject()
        return [
            ChannelWriteResult(
                channel_address=address, value_written=value, outcome=WriteOutcome.UNREQUESTED
            )
            for address, value in operations
        ]

    def _inject(self) -> None:
        if self.values[ARMED] != 1:
            return
        self.shots += 1
        stamp = EPOCH + timedelta(seconds=10 * self.shots)
        for index in range(len(X_ROWS)):
            for address, level in (
                (X_ROWS[index], 1.0 + index),
                (Y_ROWS[index], -1.0 - index),
                (SUMS[index], 2.0),
            ):
                self.values[address] = [0.0] * ARRIVAL + [level] * (SAMPLES - ARRIVAL)
                self.stamps[address] = stamp
        self.values[ARMED] = 0
        self.stamps[ARMED] = stamp


class _Correct:
    """A fake pyAML tool answering ``correct`` with a set step."""

    def correct(self, x: Any, y: Any, reference: Any = None, **_: Any) -> Any:  # noqa: ARG002
        inverse = SimpleNamespace(
            no_response_names=(),
            rank_discarded=0,
            singular_values=np.array([1.0]),
            inverse_weights=np.array([1.0]),
            spectrum_kind=(SimpleNamespace(value="LIVE"),),
            n_discard=0,
        )
        return SimpleNamespace(
            step=np.array(list(STEP.values())),
            names=tuple(STEP),
            reason=SimpleNamespace(value="OK"),
            status="",
            all_zero=False,
            zeroed_h=(),
            zeroed_v=(),
            pseudo_inverse=inverse,
        )


TOOL = TrajectoryTool(
    model="SR",
    tool=_Correct(),
    layout=SimpleNamespace(
        bpm_names_h=X_ROWS, bpm_names_v=Y_ROWS, hcorr_names=(HCM1, HCM2), vcorr_names=(VCM1,)
    ),
    bpm_index=(1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
    sum_names=(*SUMS, *SUMS),
    hcorr_index=(0.5, 1.5),
    vcorr_index=(2.5,),
    unsolved={},
    alpha=0.0,
    n_discard=0,
    gain=1.0,
)


class _Validator:
    failsafe_reason = None

    def validate(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass
class _Panel:
    """The served app, its stand-in machine, and every request a page sent."""

    base: str
    machine: _Beamline
    browser: Browser
    requests: list[str] = field(default_factory=list)
    contexts: list[BrowserContext] = field(default_factory=list)

    def open(self, secret: str) -> Page:
        """A fresh tab of the operator *secret* names, on the panel, stream connected."""
        context = self.browser.new_context(extra_http_headers={OPERATOR_SECRET_HEADER: secret})
        self.contexts.append(context)
        context.on("request", self._seen)
        page = context.new_page()
        page.goto(f"{self.base}/panel/", wait_until="load")
        _connected(page)
        return page

    def _seen(self, request: Request) -> None:
        self.requests.append(request.url)


@pytest.fixture
def _deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_CONTEXT_DIR_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        ENV_CONTROL_IDENTITY,
        AUDIT_IDENTITY_ENV,
        "OSPREY_EXECUTION_MODE",
        LAUNCH_TOKEN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "deploy"
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text(
        "control_system:\n  writes_enabled: true\n", encoding="utf-8"
    )
    monkeypatch.chdir(root)
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()
    reset_run_state(monkeypatch)
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    monkeypatch.setenv(STATE_DIR_ENV_VAR, str(tmp_path / "state"))
    posture_store.invalidate_cache()
    monkeypatch.setattr(
        actions.LimitsValidator, "from_config", staticmethod(lambda **_kw: _Validator())
    )
    yield root
    posture_store.invalidate_cache()
    reset_config_cache()


@pytest.fixture
def panel(chromium_browser: Browser, _deployment: Path, tmp_path: Path) -> Iterator[_Panel]:
    """The trajectory app serving a machine that can shoot, for alice and bob.

    On the way out, asserts no page sent a request to the root-absolute
    ``/api/`` — the agent door, which the terminal proxy would credit to the
    agent.
    """
    sequence_dir = tmp_path / "sequences"
    sequence_dir.mkdir()
    (sequence_dir / SEQUENCE_MODULE_FILE).write_text(SEQUENCE_SOURCE, encoding="utf-8")
    sequence = load_sequence(sequence_dir, COMMANDS)
    machine = _Beamline()

    async def start() -> TrajectoryStartup:
        return TrajectoryStartup(
            target=TARGET,
            connector=machine,
            commands=COMMANDS,
            guarded_run_dir=Path("/unused"),
            settings=TrajectorySettings(armed_timeout_s=1.0),
            sequence_dir=sequence_dir,
            sequence=sequence,
            tool=TOOL,
        )

    app = create_app(start=start)
    app.state.web_credentials = WebCredentials(
        operator_secret=OWN_SECRET,
        panel_token="panel-token",
        roster_secrets=(ALICE_SECRET, BOB_SECRET),
        roster_owners=("alice", "bob"),
    )
    with _run_app_server(app) as base:
        served = _Panel(base=base, machine=machine, browser=chromium_browser)
        try:
            yield served
        finally:
            machine.release()
            for context in served.contexts:
                context.close()
    agent_door = [url for url in served.requests if urlsplit(url).path.startswith("/api/")]
    assert agent_door == []
    assert any(urlsplit(url).path.startswith("/panel/api/") for url in served.requests)


# -- helpers --------------------------------------------------------------------


def _connected(page: Page) -> None:
    expect(page.locator("[data-connection=connected]")).to_be_visible(timeout=SETTLE_MS)


def _control(page: Page, name: str):
    return page.locator(f"[data-control={name}]")


def _holder(page: Page):
    return page.locator("[data-view=header] dt:text-is('Holder') + dd")


def _last_action(page: Page):
    return page.locator("[data-view=last-action] p")


def _open_session(page: Page, owner: str) -> None:
    _control(page, "open").click()
    expect(_control(page, "release")).to_be_visible(timeout=SETTLE_MS)
    expect(_holder(page)).to_have_text(f"panel of {owner} (this tab)", timeout=SETTLE_MS)


def _shoot(page: Page) -> None:
    expect(_control(page, "shot")).to_be_enabled(timeout=SETTLE_MS)
    _control(page, "shot").click()
    expect(page.locator("[data-view=step] dt:text-is('Digest')")).to_be_visible(timeout=SETTLE_MS)
    expect(_control(page, "apply")).to_be_enabled(timeout=SETTLE_MS)


def _offers_take_over_only_once_idle(page: Page, machine: _Beamline, action: str) -> None:
    """Reload while *action* is held mid-write: it keeps running, and the
    reloaded tab offers Take over only after it ended."""
    assert machine.waiting.wait(SETTLE_MS / 1000), f"the {action} never reached its write"
    page.reload(wait_until="load")
    _connected(page)
    expect(page.locator(f"p.running[data-running={action}]")).to_be_visible(timeout=SETTLE_MS)
    expect(_holder(page)).to_have_text("panel of alice", timeout=SETTLE_MS)
    expect(_control(page, "takeover")).to_have_count(0)
    expect(_control(page, "open")).to_have_count(0)
    expect(_control(page, "release")).to_have_count(0)
    machine.release()
    expect(_control(page, "takeover")).to_be_visible(timeout=SETTLE_MS)
    expect(page.locator("p.running")).to_have_text("Nothing running")


# -- the loop --------------------------------------------------------------------


def test_one_operator_runs_the_whole_loop_through_the_panel(panel: _Panel) -> None:
    """Session, setup, shot, step, apply, put back, continuous, Stop — each
    control reaching the service and its outcome coming back on the stream."""
    page = panel.open(ALICE_SECRET)
    expect(page.locator("[data-view=header] dt:text-is('Operator') + dd")).to_have_text("alice")
    expect(_control(page, "setup")).to_be_disabled()
    _open_session(page, "alice")

    page.locator("input[name=bunch_clean]").check()
    _control(page, "setup").click()
    expect(_last_action(page)).to_have_text("setup, 2 setting(s) moved", timeout=SETTLE_MS)
    assert panel.machine.writes == [(KICKER, 1), (KICKER, 0)]

    _shoot(page)
    deltas = page.locator("[data-view=step] table.deltas tbody tr")
    expect(deltas).to_have_count(len(STEP))
    expect(page.locator("[data-view=status] dt:text-is('Included') + dd")).to_have_text("1")

    _control(page, "apply").click()
    expect(_last_action(page)).to_have_text("apply, 3 setting(s) moved", timeout=SETTLE_MS)
    expect(page.locator("[data-notice=moved] strong").first).to_have_text("step applied")
    for name, delta in DELTAS.items():
        assert panel.machine.values[name] == pytest.approx(BEFORE[name] + delta)

    expect(_control(page, "put-back")).to_be_enabled()
    _control(page, "put-back").click()
    expect(_last_action(page)).to_have_text("put_back, 3 setting(s) moved", timeout=SETTLE_MS)
    assert {name: panel.machine.values[name] for name in BEFORE} == BEFORE
    expect(_control(page, "put-back")).to_be_disabled()

    shots = panel.machine.shots
    page.locator("input[name=max_shots]").fill(CONTINUOUS_CAP)
    _control(page, "continuous").click()
    expect(page.locator("[data-view=continuous]")).to_contain_text("Running:", timeout=SETTLE_MS)
    expect(_control(page, "shot")).to_be_disabled()
    _control(page, "stop").click()
    expect(page.locator("[data-view=ended]")).to_contain_text(
        "Continuous run ended: stopped", timeout=SETTLE_MS
    )
    expect(page.locator("[data-view=status]")).to_contain_text("Not running.")
    assert panel.machine.shots > shots
    expect(_control(page, "release")).to_be_visible()
    expect(_control(page, "shot")).to_be_enabled(timeout=SETTLE_MS)


# -- take-over -------------------------------------------------------------------


def test_another_operator_and_a_second_tab_of_the_same_one_take_over(panel: _Panel) -> None:
    alice = panel.open(ALICE_SECRET)
    _open_session(alice, "alice")

    bob = panel.open(BOB_SECRET)
    expect(_holder(bob)).to_have_text("panel of alice")
    expect(_control(bob, "open")).to_have_count(0)
    expect(_control(bob, "setup")).to_be_disabled()
    _control(bob, "takeover").click()
    expect(_holder(bob)).to_have_text("panel of bob (this tab)", timeout=SETTLE_MS)
    expect(_control(bob, "release")).to_be_visible()
    expect(_control(bob, "setup")).to_be_enabled()

    expect(_control(alice, "takeover")).to_be_visible(timeout=SETTLE_MS)
    expect(_control(alice, "release")).to_have_count(0)
    expect(_holder(alice)).to_have_text("panel of bob")
    expect(alice.locator(".notice-taken-over")).to_have_text(
        "Taken over by panel of bob from panel of alice."
    )

    second = panel.open(ALICE_SECRET)
    expect(_control(second, "release")).to_have_count(0)
    _control(second, "takeover").click()
    expect(_holder(second)).to_have_text("panel of alice (this tab)", timeout=SETTLE_MS)
    expect(_control(bob, "takeover")).to_be_visible(timeout=SETTLE_MS)
    expect(_control(bob, "release")).to_have_count(0)
    # The first tab of the same operator is a different client: it is not handed
    # the session back by its owner's take-over.
    expect(_holder(alice)).to_have_text("panel of alice")
    expect(_control(alice, "takeover")).to_be_visible()
    expect(_control(alice, "release")).to_have_count(0)


# -- a reload never stops the action ---------------------------------------------


def test_a_reload_mid_setup_leaves_the_setup_running(panel: _Panel) -> None:
    page = panel.open(ALICE_SECRET)
    _open_session(page, "alice")
    page.locator("input[name=bunch_clean]").check()
    panel.machine.hold()
    _control(page, "setup").click()

    _offers_take_over_only_once_idle(page, panel.machine, "setup")
    expect(_last_action(page)).to_have_text("setup, 2 setting(s) moved", timeout=SETTLE_MS)
    assert panel.machine.writes == [(KICKER, 1), (KICKER, 0)]


def test_a_reload_mid_apply_leaves_the_apply_running(panel: _Panel) -> None:
    page = panel.open(ALICE_SECRET)
    _open_session(page, "alice")
    _shoot(page)
    panel.machine.hold()
    _control(page, "apply").click()

    _offers_take_over_only_once_idle(page, panel.machine, "apply")
    expect(_last_action(page)).to_have_text("apply, 3 setting(s) moved", timeout=SETTLE_MS)
    for name, delta in DELTAS.items():
        assert panel.machine.values[name] == pytest.approx(BEFORE[name] + delta)
