"""The in-process stack every trajectory browser test runs through.

A browser on a deployment reaches the panel only through its operator's web
terminal: nginx strips ``/u/<user>`` and the terminal's panel proxy forwards
``/panel/trajectory/...`` to the trajectory service, rewriting the page's
root-absolute design-system and font literals on the way. This module stands
that whole path up in one process:

* the **trajectory app**, the real one behind its real web gate, started on a
  fake machine (:class:`Beamline`) and a real pyAML trajectory tool over a
  small response matrix, under a throwaway deployment root with a real
  control-context tree and the real run lock; only the limits validator is
  stubbed;
* one **web-terminal app per operator**, each holding that operator's roster
  secret as its own operator secret, so the service's gate names the operator
  exactly as it does behind a per-user terminal container;
* a small **front** that plays nginx: it strips ``/u/<user>`` and hands the
  request to that user's terminal, which then spells its own mount prefix.

The front also serves each terminal's hub page, so an embedded test opens the
panel as a real hub tile, in Expert or Simple mode.

Every page opened through :class:`Stack` is watched by a context-level
listener: a response of 4xx or more during its load fails the test, iframes
included. The vendored Plotly is required, never optional: without it the
stack refuses to start.

The fake machine answers for the panel's behaviour, never for physics. A test
that needs a particular machine state stages it by name (:meth:`Beamline.stage`
over :data:`FAKE_STATES`), so frozen tests assert what the panel shows for a
state, and the physics stay with the deployed end-to-end tests.
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from starlette.applications import Starlette

from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER, url_mount_prefix
from osprey.interfaces.web_auth import WebCredentials
from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory import actions
from osprey.services.trajectory import app as trajectory_app
from osprey.services.trajectory.app import LAUNCH_TOKEN_ENV
from osprey.services.trajectory.sequence import SEQUENCE_MODULE_FILE, load_sequence
from osprey.services.trajectory.startup import (
    TrajectoryCommands,
    TrajectoryDevice,
    TrajectorySettings,
    TrajectoryStartup,
)
from osprey.services.trajectory.step import TOOL_NAME, TrajectoryTool
from osprey.services.trajectory.store import STATE_DIR_ENV_VAR
from osprey_connectors import posture_store
from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult, WriteOutcome
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from tests._guarded_run import reset_run_state
from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Frame, FrameLocator, Page, Response

TARGET = "va"
MODEL = "SR"
#: The trajectory panel's id in the web terminal; its URLs live under it.
PANEL_ID = "trajectory"
#: The panel bundle's directory and the Plotly file it loads from beside it.
PANEL_DIR = Path(trajectory_app.__file__).parent / "panel"
PLOTLY = PANEL_DIR / "vendor" / "plotly-3.3.1.min.js"

#: The operators of the stack: each has a terminal at ``/u/<name>`` whose
#: operator secret is that operator's roster secret at the service.
OPERATORS = {
    "alice": "alice-secret-" + "a" * 32,
    "bob": "bob-secret-" + "b" * 32,
}
#: The service's own operator secret, which no terminal holds.
SERVICE_SECRET = "service-secret-" + "s" * 32

#: Turn-by-turn samples per record, the trigger sample, and where beam arrives.
SAMPLES, TRIGGER_SAMPLE, ARRIVAL = 64, 10, 12
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
#: Seconds each injection takes, so a continuous run lasts until Stop.
INJECTION_S = 0.1
#: How long a browser waits for the panel to settle on something.
SETTLE_MS = 15_000

ARM, ARMED = "SR:BPM:ARM", "SR:BPM:ARMED"
REQUEST, BUSY = "SR:INJ:REQ", "SR:INJ:BUSY"
CLEAN = "SR:INJ:CLEAN"
RF_ON, CURRENT = "SR:RF:ON", "SR:DCCT:CURRENT"
TURNS_SP, TURNS_RB = "SR:DIAG:TBT:TURNS:SP", "SR:DIAG:TBT:TURNS:RB"
HCMS = ("SR:HCM1", "SR:HCM2")
VCMS = ("SR:VCM1",)
#: Every corrector's setting before a test moves it.
CORRECTORS = {"SR:HCM1": 1.0, "SR:HCM2": -2.0, "SR:VCM1": 0.5}

#: The per-device BPM settings the model declares, and each one's starting value.
DEVICE_SETTINGS = {"attenuation": 10.0, "bpm_mode": 0, "pilot_tone": 0}
#: Turns every BPM records after the trigger sample, at start.
TURNS_TRACKED = 40

#: The panel's sequence module: an injection, and the longest pause.
SEQUENCE_SOURCE = """
SEQUENCE_METADATA = {"max_pause_s": 0.5}


async def inject(handle):
    await handle.write("injection_request", 1)
"""


def bpm_names(n_bpms: int) -> tuple[str, ...]:
    """The device names of *n_bpms* BPMs, 1-based."""
    return tuple(f"SR/BPM{n}" for n in range(1, n_bpms + 1))


def _row(device: str, plane: str) -> str:
    return f"{device.replace('/', ':')}:{plane}"


def _setting(device: str, setting: str, end: str) -> str:
    return f"{device.replace('/', ':')}:{setting.upper()}:{end}"


def commands(n_bpms: int) -> TrajectoryCommands:
    """The model's commands for *n_bpms* BPMs sharing one arming device."""
    return TrajectoryCommands(
        model=MODEL,
        devices=tuple(
            TrajectoryDevice(
                name=device,
                x_row=_row(device, "X"),
                y_row=_row(device, "Y"),
                sum_row=_row(device, "SUM"),
                arm=ARM,
                armed=ARMED,
            )
            for device in bpm_names(n_bpms)
        ),
        injection_request=REQUEST,
        injection_busy=BUSY,
        trigger_sample=TRIGGER_SAMPLE,
    )


def _step(step_id: str, label: str, description: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": step_id,
        "label": label,
        "description": description,
        "command": None,
        "default": True,
        "position": 0,
        "timeout_s": None,
        **extra,
    }


def slots(n_bpms: int) -> dict[str, Any]:
    """The model's ``trajectory_slots`` entry, shaped as the facts view writes it.

    Declares the chain (arm, inject, bunch cleaning as a command step,
    readout), the BPM settings (three per device, turns tracked once), and the
    machine readbacks the header shows.
    """
    devices = bpm_names(n_bpms)
    settings: list[dict[str, Any]] = [
        {
            "id": name,
            "label": name.replace("_", " ").capitalize(),
            "description": None,
            "scope": "device",
            "setting": {device: _setting(device, name, "SP") for device in devices},
            "readback": {device: _setting(device, name, "RB") for device in devices},
            "tolerance": dict.fromkeys(devices, 0.01),
            "timeout_s": 1.0,
        }
        for name in DEVICE_SETTINGS
    ]
    settings.append(
        {
            "id": "turns_tracked",
            "label": "Turns tracked",
            "description": None,
            "scope": "global",
            "setting": TURNS_SP,
            "readback": TURNS_RB,
            "tolerance": None,
            "timeout_s": 1.0,
        }
    )
    return {
        "devices": {},
        "trigger_sample": TRIGGER_SAMPLE,
        "injection_request": REQUEST,
        "injection_busy": BUSY,
        "machine": {"rf_on": RF_ON, "stored_current": CURRENT},
        "settings": settings,
        "steps": [
            _step("arm", "Arm BPMs", "Arms every BPM's turn-by-turn acquisition."),
            _step("inject", "Inject Beam", "Fires one injected shot."),
            _step(
                "clean",
                "Bunch Cleaning",
                "Cleans the stored bunches before the shot.",
                command={"address": CLEAN, "value": 1},
                timeout_s=1.0,
            ),
            _step("readout", "Readout BPMs", "Reads every BPM's turn-by-turn buffer."),
        ],
    }


def tool(n_bpms: int) -> TrajectoryTool:
    """A real pyAML trajectory tool over a small, well-conditioned response matrix."""
    from pyaml.tuning_tools.trajectory import Trajectory

    devices = bpm_names(n_bpms)
    x_rows = [_row(device, "X") for device in devices]
    y_rows = [_row(device, "Y") for device in devices]
    matrix: list[list[float]] = []
    for index in range(n_bpms):
        matrix.append([1.0 / (1 + abs(index - 0)), 1.0 / (1 + abs(index - n_bpms // 2)), 0.0])
    for index in range(n_bpms):
        matrix.append([0.0, 0.0, 1.0 / (1 + abs(index - n_bpms // 2))])
    fields = {
        "matrix": matrix,
        "observable_names": [*x_rows, *y_rows],
        "variable_names": [*HCMS, *VCMS],
        "rf_response": None,
        "observable_planes": ["H"] * n_bpms + ["V"] * n_bpms,
        "variable_planes": ["H"] * len(HCMS) + ["V"] * len(VCMS),
    }
    built = Trajectory(TOOL_NAME, fields, alpha=0.0, n_discard=0, gain=1.0)
    positions = tuple(float(n) for n in range(1, n_bpms + 1))
    return TrajectoryTool(
        model=MODEL,
        tool=built,
        layout=built.layout,
        bpm_index=(*positions, *positions),
        hcorr_index=(0.5, float(n_bpms) / 2),
        vcorr_index=(float(n_bpms) / 2 + 0.5,),
        unsolved={},
        alpha=0.0,
        n_discard=0,
        gain=1.0,
    )


# -- the fake machine -------------------------------------------------------------------

#: A named fake state: what it does to the machine before a test drives the panel.
FakeState = Callable[["Beamline"], None]

#: Every named fake state :meth:`Beamline.stage` knows. A state sets machine
#: values the panel reads; it never models physics.
FAKE_STATES: dict[str, FakeState] = {}


def fake_state(name: str) -> Callable[[FakeState], FakeState]:
    """Register the decorated function as the fake state *name*."""

    def register(apply: FakeState) -> FakeState:
        if name in FAKE_STATES:
            raise ValueError(f"fake state {name!r} is already registered")
        FAKE_STATES[name] = apply
        return apply

    return register


@fake_state("nominal")
def _nominal(machine: Beamline) -> None:
    """RF on, an empty ring, pilot tone off: the state every test starts in."""
    machine.values[RF_ON] = 1
    machine.values[CURRENT] = 0.0
    machine.stored = 0.0


class Beamline:
    """The machine the stack's service drives: correctors, BPMs and their settings.

    An arm sets the armed flag; an injection request, once armed, posts a
    fresh turn-by-turn record of :data:`SAMPLES` samples on every BPM, stamped
    later than the last, with the beam arriving at :data:`ARRIVAL` (on top of
    any stored beam a staged state put in the ring). A setting's setpoint
    drives its readback; the bunch-cleaning command empties the ring. A
    checked write waits while the gate is held (:meth:`hold`), so a test can
    keep a setup or an apply in flight.
    """

    _control_target = TARGET

    def __init__(self, n_bpms: int = 3) -> None:
        self.devices = bpm_names(n_bpms)
        self.rows = [
            (_row(device, "X"), _row(device, "Y"), _row(device, "SUM")) for device in self.devices
        ]
        self.values: dict[str, Any] = {**CORRECTORS, BUSY: 0, ARMED: 0, CLEAN: 0}
        self.readback_of: dict[str, str] = {TURNS_SP: TURNS_RB}
        self.values[TURNS_SP] = self.values[TURNS_RB] = TURNS_TRACKED
        for device in self.devices:
            for name, value in DEVICE_SETTINGS.items():
                setpoint = _setting(device, name, "SP")
                readback = _setting(device, name, "RB")
                self.readback_of[setpoint] = readback
                self.values[setpoint] = self.values[readback] = value
        for row in (address for rows in self.rows for address in rows):
            self.values[row] = [0.0] * SAMPLES
        self.stamps: dict[str, datetime] = {}
        #: The stored beam's share of the sum, on every turn of a record.
        self.stored = 0.0
        self.shots = 0
        self.writes: list[tuple[str, Any]] = []
        self.staged: list[str] = []
        self._open = threading.Event()
        self._open.set()
        self.waiting = threading.Event()
        self._limits_validator: Any = None
        self.stage("nominal")

    # -- staging --------------------------------------------------------------

    def stage(self, name: str) -> None:
        """Put the machine in the named fake state *name* (see :data:`FAKE_STATES`).

        Raises:
            KeyError: No fake state of that name is registered; the message
                lists the known ones.
        """
        try:
            apply = FAKE_STATES[name]
        except KeyError:
            raise KeyError(f"no fake state {name!r}; known: {sorted(FAKE_STATES)}") from None
        apply(self)
        self.staged.append(name)

    # -- holding a write ------------------------------------------------------

    def hold(self) -> None:
        """Hold every checked write at the gate until :meth:`release`."""
        self.waiting.clear()
        self._open.clear()

    def release(self) -> None:
        self._open.set()

    # -- the connector surface ------------------------------------------------

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
        await self._effect(address, value)
        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )

    async def write_multiple_channels(
        self, operations: list[tuple[str, Any]], **_: Any
    ) -> list[ChannelWriteResult]:
        for address, value in operations:
            self.writes.append((address, value))
            await self._effect(address, value)
        return [
            ChannelWriteResult(
                channel_address=address, value_written=value, outcome=WriteOutcome.UNREQUESTED
            )
            for address, value in operations
        ]

    async def _effect(self, address: str, value: Any) -> None:
        if address == ARM and value == 1:
            self.values[ARMED] = 1
        elif address == REQUEST and value == 1:
            await asyncio.sleep(INJECTION_S)
            self._inject()
        elif address == CLEAN and value == 1:
            self.stored = 0.0
            self.values[CURRENT] = 0.0
        else:
            self.values[address] = value
            readback = self.readback_of.get(address)
            if readback is not None:
                self.values[readback] = value

    def _inject(self) -> None:
        if self.values[ARMED] != 1:
            return
        self.shots += 1
        stamp = EPOCH + timedelta(seconds=10 * self.shots)
        for index, (x_row, y_row, sum_row) in enumerate(self.rows):
            beam = SAMPLES - ARRIVAL
            self.values[x_row] = [0.0] * ARRIVAL + [1.0 + index] * beam
            self.values[y_row] = [0.0] * ARRIVAL + [-1.0 - index] * beam
            self.values[sum_row] = [self.stored] * ARRIVAL + [2.0 + self.stored] * beam
            for address in (x_row, y_row, sum_row):
                self.stamps[address] = stamp
        self.values[ARMED] = 0
        self.stamps[ARMED] = stamp


class _Validator:
    failsafe_reason = None

    def validate(self, *_args: Any, **_kwargs: Any) -> None:
        return None


# -- the front --------------------------------------------------------------------------

#: The operator whose terminal the current request reached through the front.
_USER: contextvars.ContextVar[str] = contextvars.ContextVar("stack_user", default="alice")

#: Every module that reads its terminal's mount prefix per call. One process
#: serves every operator's terminal here, so the prefix is the user the front
#: routed the request to, not the process environment's.
_PREFIX_READERS = (
    "osprey.interfaces.web_terminal.app",
    "osprey.interfaces.web_terminal.routes.proxy",
    "osprey.interfaces.web_terminal.routes.panels",
)


def _mount_prefix() -> str:
    return url_mount_prefix(_USER.get())


def _front(terminals: dict[str, Any]) -> Any:
    """An ASGI app playing nginx for *terminals*, keyed by user.

    ``/u/<user>/...`` reaches that user's terminal with the prefix stripped,
    and the bare ``/u/<user>`` reaches its root, as nginx's trailing-slash
    redirect would; any other path is 404. The lifespan starts every terminal's
    own lifespan.
    """

    @asynccontextmanager
    async def lifespan(_app: Any):
        async with AsyncExitStack() as stack:
            for user, terminal in terminals.items():
                _USER.set(user)
                await stack.enter_async_context(terminal.router.lifespan_context(terminal))
            yield

    lifespans = Starlette(lifespan=lifespan)

    async def asgi(scope, receive, send):
        if scope["type"] == "lifespan":
            await lifespans(scope, receive, send)
            return
        path = scope.get("path", "")
        for user, terminal in terminals.items():
            prefix = url_mount_prefix(user)
            if path == prefix or path.startswith(f"{prefix}/"):
                _USER.set(user)
                scope = dict(scope, path=path[len(prefix) :] or "/")
                await terminal(scope, receive, send)
                return
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"not behind the front"})

    return asgi


# -- the stack --------------------------------------------------------------------------


@dataclass
class Stack:
    """The running stack, its fake machine, and every page it opened.

    Attributes:
        front: The front's base URL; each operator's terminal is under ``/u/<user>``.
        service: The trajectory service's own base URL, which no browser uses.
        machine: The fake machine the service drives.
        browser: The browser every page opens in.
        terminals: Each operator's terminal app, for state a test sets (UI mode).
        responses: Every ``(status, url)`` a page of the stack received.
    """

    front: str
    service: str
    machine: Beamline
    browser: Browser
    terminals: dict[str, Any]
    responses: list[tuple[int, str]] = field(default_factory=list)
    contexts: list[BrowserContext] = field(default_factory=list)

    def terminal_url(self, user: str) -> str:
        """*user*'s terminal: the hub page."""
        return f"{self.front}{url_mount_prefix(user)}/"

    def panel_url(self, user: str) -> str:
        """The panel as *user* opens it standalone, through their terminal's proxy."""
        return f"{self.front}{url_mount_prefix(user)}/panel/{PANEL_ID}/panel/"

    def context(self, user: str, *, theme: str = "dark") -> BrowserContext:
        """A fresh browser context of *user* in *theme*, every response recorded."""
        context = self.browser.new_context(
            extra_http_headers={OPERATOR_SECRET_HEADER: OPERATORS[user]},
            color_scheme="dark" if "dark" in theme else "light",
        )
        context.add_init_script(_theme_script(user, theme))
        context.on("response", self._seen)
        self.contexts.append(context)
        return context

    def open(self, user: str, *, theme: str = "dark") -> Page:
        """A fresh standalone tab of *user* on the panel, loaded, stream connected.

        Fails when any response of its load was a 4xx or more.
        """
        page = self.context(user, theme=theme).new_page()
        page.goto(self.panel_url(user), wait_until="load")
        connected(page)
        self.assert_loaded_clean()
        return page

    def open_tile(self, user: str, *, theme: str = "dark", mode: str = "expert") -> Page:
        """*user*'s hub in *mode* with the panel open as a tile, stream connected.

        Answers the hub page; :func:`tile` reaches the panel inside it. Fails
        when any response of the hub's or the tile's load was a 4xx or more.
        """
        self.terminals[user].state.web_ui_mode = mode
        page = self.context(user, theme=theme).new_page()
        page.goto(self.terminal_url(user), wait_until="load")
        page.locator(f'button.panel-rail-button[data-panel-id="{PANEL_ID}"]').click()
        iframe = page.locator(f'.dock-iframe-overlay iframe[data-panel-id="{PANEL_ID}"]')
        iframe.wait_for(state="visible", timeout=SETTLE_MS)
        connected(tile(page))
        self.assert_loaded_clean()
        return page

    def assert_loaded_clean(self) -> None:
        """Fail when any page of the stack received a 4xx or more so far."""
        refused = [(status, url) for status, url in self.responses if status >= 400]
        assert refused == [], f"responses of 4xx or more on load: {refused}"

    def _seen(self, response: Response) -> None:
        self.responses.append((response.status, response.url))


def _theme_script(user: str, theme: str) -> str:
    """An init script storing *theme* as the hub's and the panel's preference."""
    keys = ["osprey-theme", f"osprey-theme--{user}"]
    return (
        "(() => { try { for (const key of "
        + repr(keys).replace("'", '"')
        + f') window.localStorage.setItem(key, "{theme}"); }} catch (e) {{}} }})();'
    )


def connected(page: Page | Frame | FrameLocator) -> None:
    """Wait until the panel's event stream reports connected."""
    from playwright.sync_api import expect

    expect(page.locator("[data-connection=connected]")).to_be_visible(timeout=SETTLE_MS)


def tile(page: Page) -> FrameLocator:
    """The panel inside the hub *page*'s trajectory tile."""
    return page.frame_locator(f'.dock-iframe-overlay iframe[data-panel-id="{PANEL_ID}"]')


def _require_plotly() -> None:
    if not PLOTLY.is_file():
        pytest.fail(
            f"the vendored Plotly is missing at {PLOTLY}; fetch it with "
            "`uv run osprey vendor fetch --target services/trajectory/panel/vendor/`",
            pytrace=False,
        )


def _deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway deployment root: config, control-context tree, state dir."""
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
    return root


def _workspace(tmp_path: Path, user: str) -> Path:
    """The directory *user*'s terminal watches; each terminal watches its own."""
    return tmp_path / f"_agent_data_{user}"


#: A PTY command that stays alive and tolerates the arguments a resume appends.
_LONG_LIVED_SHELL = [sys.executable, "-c", "import time; time.sleep(3600)"]


@contextmanager
def run_stack(
    browser: Browser,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_bpms: int = 3,
) -> Iterator[Stack]:
    """Run the stack for one test; see the module docstring.

    Args:
        browser: The browser every page opens in.
        tmp_path: The test's own directory: deployment root, state, workspace.
        monkeypatch: The test's monkeypatch, which undoes every patch here.
        n_bpms: How many BPMs the fake machine and the model carry.

    Yields:
        The running stack. On the way out every context is closed and any
        held write released, and every page is checked to have reached the
        service through its panel door and never through the agent door.
    """
    _require_plotly()
    root = _deployment(tmp_path, monkeypatch)
    monkeypatch.setenv("CONFIG_FILE", str(root / "build" / "config.yml"))
    from osprey_connectors.workspace import reset_config_cache

    sequence_dir = tmp_path / "sequences"
    sequence_dir.mkdir()
    (sequence_dir / SEQUENCE_MODULE_FILE).write_text(SEQUENCE_SOURCE, encoding="utf-8")
    sequence = load_sequence(sequence_dir)
    machine = Beamline(n_bpms)
    model_tool = tool(n_bpms)
    declared = slots(n_bpms)
    monkeypatch.setattr(trajectory_app, "_trajectory_slots", lambda _config, _model: declared)

    async def start() -> TrajectoryStartup:
        return TrajectoryStartup(
            target=TARGET,
            connector=machine,
            commands=commands(n_bpms),
            guarded_run_dir=Path("/unused"),
            settings=TrajectorySettings(armed_timeout_s=1.0),
            sequence_dir=sequence_dir,
            sequence=sequence,
            tool=model_tool,
        )

    service = trajectory_app.create_app(start=start)
    service.state.web_credentials = WebCredentials(
        operator_secret=SERVICE_SECRET,
        panel_token="panel-token-" + "p" * 32,
        roster_secrets=tuple(OPERATORS.values()),
        roster_owners=tuple(OPERATORS),
    )
    for user in OPERATORS:
        workspace = _workspace(tmp_path, user)
        workspace.mkdir()
        (workspace / "seed_artifact.txt").write_text("seed\n", encoding="utf-8")

    with ExitStack() as stack:
        service_base = stack.enter_context(_run_app_server(service))
        panel = {
            "id": PANEL_ID,
            "label": "TRAJECTORY",
            "url": service_base,
            "path": "/panel/",
            "healthEndpoint": None,
            "configDefined": True,
        }
        monkeypatch.setenv(TERMINAL_USER_ENV, next(iter(OPERATORS)))
        for module in _PREFIX_READERS:
            monkeypatch.setattr(f"{module}.compute_url_prefix", _mount_prefix)
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                side_effect=lambda *_a, **_k: {"watch_dir": str(_workspace(tmp_path, _USER.get()))},
            )
        )
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._load_panel_config",
                return_value=(set(), [panel], None),
            )
        )
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._launch_panel_server",
                side_effect=publish_artifact_url(None),
            )
        )
        from osprey.interfaces.web_terminal.app import create_app as create_terminal

        terminals: dict[str, Any] = {}
        for user in OPERATORS:
            token = _USER.set(user)
            try:
                terminal = create_terminal(shell_command=list(_LONG_LIVED_SHELL))
            finally:
                _USER.reset(token)
            terminals[user] = terminal
        front = stack.enter_context(_run_app_server(_front(terminals)))
        for user, terminal in terminals.items():
            # Each terminal holds its operator's roster secret as its own, as a
            # per-user terminal container does; the proxy re-issues it upstream.
            terminal.state.terminal_user = user
            terminal.state.web_credentials = WebCredentials(
                operator_secret=OPERATORS[user], panel_token=f"panel-token-{user}-" + "t" * 32
            )
        served = Stack(
            front=front,
            service=service_base,
            machine=machine,
            browser=browser,
            terminals=terminals,
        )
        try:
            yield served
        finally:
            machine.release()
            for context in served.contexts:
                context.close()
    reset_config_cache()
    posture_store.invalidate_cache()
    agent_door = [url for _status, url in served.responses if f"/panel/{PANEL_ID}/api/" in url]
    assert agent_door == [], "a page reached the service's agent door"
