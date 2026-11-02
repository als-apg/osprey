"""How a CANOPY process on the control session ends: 78, 75, or not at all.

The first half runs the real thing: ``python -m uvicorn canopy.main:app --fd N``
as a child on a listening socket this test holds, with the backend selected by
its ``module:Class`` path, and asserts the status the child leaves with. A
child sees no test double, so its deployment is an empty config — a ``live``
target nothing can derive, which is an unservable binding: it builds no
connector and touches no network, and it still reads, compares and watches the
record, which is all these tests are about.

The second half pins what must have happened before the exit — the session
closed, nothing left connected — in process, with the exit itself replaced.

The third pins what a running process says when the write state changes: a real
broker over the backend, stepped by an injected clock. A switch flip, a position
change or a hold is no move: the claim changes on the next tick and the process
stays. Only a move of the record ends it (75), and the outgoing process pushes
its ``switching`` claim before the drain ends and before the exit.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from canopy.contracts import load_heartbeat_vectors
from canopy.epics.base import PVConnectionFailed
from canopy.epics.config import EpicsConfig
from canopy.epics.gateway import build_connector
from canopy.hare.config import HareConfig
from canopy.hare.services.broker import PVBroker

from osprey.integrations.canopy import backend as backend_module
from osprey.integrations.canopy import sidecar as sidecar_module
from osprey.integrations.canopy.backend import (
    NO_RECORD,
    REACH_SINCE_ENV,
    RENDERED_FOR_ENV,
    SESSIONS_FILENAME,
    STATEMENT_FILENAME,
    PVWriteRefused,
    record_identity,
)
from osprey.runtime import session as session_module
from osprey.runtime.session import (
    EXIT_BINDING_MOVED,
    EXIT_START_REFUSED,
    EXIT_TRANSPORT_STALE,
    Binding,
    SessionEnvironmentError,
)
from osprey_connectors import control_context, posture_store, session_hold, types
from osprey_connectors.control_system import _base as connector_base
from osprey_connectors.control_system import _epics_connector as epics_connector_module
from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
from osprey_connectors.errors import ConnectorUnavailableError
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._harness._control_context_fixtures import _set_switch_to

# start() arms the real raw-put block in this process; put back what it patched.
pytestmark = pytest.mark.usefixtures("raw_put_block_restored")

BACKEND_PATH = "osprey.integrations.canopy.backend:OspreyPVConnector"

#: How long a child gets to come up, or to leave.
CHILD_TIMEOUT_S = 120.0

STARTED = "Application startup complete"


def write_record(state_dir: Path, *, target="live", generation=3) -> Path:
    """Write the record with *target*'s switch allowed, so the connector's own verdict allows."""
    path = state_dir / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation), path=path
    )
    control_context.invalidate_cache()
    _set_switch_to(path, dict.fromkeys({target} & set(types.CONTROL_TARGETS), "allowed"))
    control_context.invalidate_cache()
    return path


def test_the_exit_statuses_are_the_contract_numbers():
    """A supervisor matches on these two integers."""
    assert (EXIT_START_REFUSED, EXIT_BINDING_MOVED) == (78, 75)


class TestRecordIdentity:
    """What the rendered-for variable is compared with."""

    @pytest.mark.parametrize(
        ("binding", "identity"),
        [
            (Binding("va", 3, "virtual_accelerator", "Virtual accelerator", "bound"), "va@3"),
            (Binding("live", 7, None, "Live machine", "unservable"), "live@7"),
            (Binding("va", None, "virtual_accelerator", "Virtual accelerator", "fallback"), "none"),
            (Binding("live", None, None, "Live machine", "unservable"), "none"),
        ],
    )
    def test_it_names_the_record_read_and_not_the_binding_state(self, binding, identity):
        """A degraded binding still names its record; no record is ``none``."""
        assert record_identity(binding) == identity
        assert NO_RECORD == "none"


# -- a real child ------------------------------------------------------------------


class _Child:
    """``python -m uvicorn canopy.main:app --fd N`` on a socket the test holds."""

    def __init__(self, tmp_path: Path, **environment: str | None) -> None:
        self.state_dir = tmp_path / "control_target"
        self.state_dir.mkdir(exist_ok=True)
        config = tmp_path / "config.yml"
        config.write_text("{}\n", encoding="utf-8")
        self._output = tmp_path / "child.log"

        env = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("OSPREY_", "CANOPY_", "EPICS_")) and name != "CONFIG_FILE"
        }
        env.update(
            {
                posture_store.CONTROL_CONTEXT_DIR_ENV_VAR: str(self.state_dir),
                AUDIT_IDENTITY_ENV: "test-operator",
                "OSPREY_CONFIG": str(config),
                "CONFIG_FILE": str(config),
                "CANOPY_EPICS__BACKEND": BACKEND_PATH,
                "CANOPY_MODULES": "hare",
            }
        )
        for name, value in environment.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(16)
        self.port = self._listener.getsockname()[1]
        fd = self._listener.fileno()
        with self._output.open("wb") as output:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "canopy.main:app", "--fd", str(fd)],
                cwd=tmp_path,
                env=env,
                pass_fds=(fd,),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )

    @property
    def output(self) -> str:
        return self._output.read_text(encoding="utf-8", errors="replace")

    def wait(self) -> int:
        try:
            return self.process.wait(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            pytest.fail(f"the child did not leave within {CHILD_TIMEOUT_S} s:\n{self.output}")

    def wait_serving(self) -> None:
        deadline = time.monotonic() + CHILD_TIMEOUT_S
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                pytest.fail(
                    f"the child left with {self.process.returncode} before serving:\n{self.output}"
                )
            if STARTED in self.output:
                try:
                    with urllib.request.urlopen(url, timeout=2.0) as response:
                        if response.status == 200:
                            return
                except (urllib.error.URLError, OSError):
                    pass
            time.sleep(0.05)
        pytest.fail(f"the child was not serving within {CHILD_TIMEOUT_S} s:\n{self.output}")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=30)
        self._listener.close()


@pytest.fixture
def child(tmp_path):
    """Start children in ``tmp_path``; whatever is still running is killed afterwards."""
    started: list[_Child] = []

    def start(**environment: str | None) -> _Child:
        process = _Child(tmp_path, **environment)
        started.append(process)
        return process

    yield start
    for process in started:
        process.stop()
    control_context.invalidate_cache()


class TestServedProcess:
    """The status a real CANOPY process leaves with."""

    def test_a_refused_environment_exits_78_before_serving(self, child):
        """A forbidden variable is a refused start: status 78, the reason logged, nothing served."""
        process = child(
            **{RENDERED_FOR_ENV: "none", posture_store.CONTROL_OWNER_ENV_VAR: "someone"}
        )
        assert process.wait() == 78
        assert f"{posture_store.CONTROL_OWNER_ENV_VAR} must not be set" in process.output
        assert STARTED not in process.output

    def test_an_environment_that_names_no_record_exits_78(self, child):
        """Without the rendered-for variable the start is refused, not retried."""
        process = child(**{RENDERED_FOR_ENV: None})
        assert process.wait() == 78
        assert f"{RENDERED_FOR_ENV} is not set" in process.output
        assert STARTED not in process.output

    def test_an_environment_rendered_for_another_record_exits_75_before_serving(
        self, child, tmp_path
    ):
        """The record read is ``live@3``, the environment says ``live@2``: status 75."""
        (tmp_path / "control_target").mkdir()
        write_record(tmp_path / "control_target", generation=3)
        process = child(**{RENDERED_FOR_ENV: "live@2"})
        assert process.wait() == 75
        assert STARTED not in process.output

    def test_an_environment_rendered_for_a_record_that_is_gone_exits_75(self, child):
        """No record is read, the environment names one: status 75."""
        process = child(**{RENDERED_FOR_ENV: "live@3"})
        assert process.wait() == 75
        assert STARTED not in process.output

    def test_a_matching_unservable_process_serves_until_the_record_moves(self, child, tmp_path):
        """Rendered for the record it read, it serves; a new generation ends it with 75."""
        (tmp_path / "control_target").mkdir()
        write_record(tmp_path / "control_target", generation=3)
        process = child(**{RENDERED_FOR_ENV: "live@3"})
        process.wait_serving()
        assert process.process.poll() is None
        write_record(tmp_path / "control_target", generation=4)
        assert process.wait() == 75

    def test_a_matching_fallback_process_serves_until_a_record_appears(self, child):
        """With no record and ``none`` it serves; a record appearing ends it with 75."""
        process = child(**{RENDERED_FOR_ENV: "none"})
        process.wait_serving()
        assert process.process.poll() is None
        write_record(process.state_dir, generation=1)
        assert process.wait() == 75


# -- what happened before the exit -------------------------------------------------

VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}


class _Exited(BaseException):
    """Stands in for the process ending."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


class _TrackedFake(FakeConnector):
    """A fake that knows how many instances are connected."""

    connected = 0
    connects = 0

    async def connect(self, config):
        await super().connect(config)
        type(self).connected += 1
        type(self).connects += 1

    async def disconnect(self):
        type(self).connected -= 1
        await super().disconnect()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A session environment over :class:`_TrackedFake`, with the exit replaced."""
    directory = tmp_path / "control_target"
    directory.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    for name in list(os.environ):
        if name.startswith(("OSPREY_CONTROL_TARGET", "EPICS_", "CANOPY_EPICS__")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        session_hold.HOLD_DIR_ENV_VAR,
        session_module.SESSION_KEY_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    monkeypatch.chdir(tmp_path)

    def get_config_value(path, default=None, _config_path=None):
        return VA_SECTION if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)

    exits: list[int] = []

    def fake_exit(status):
        exits.append(status)
        raise _Exited(status)

    monkeypatch.setattr(backend_module.os, "_exit", fake_exit)
    _TrackedFake.connected = _TrackedFake.connects = 0
    control_context.invalidate_cache()
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _TrackedFake)
        yield directory, exits
    control_context.invalidate_cache()


def _backend():
    return build_connector(EpicsConfig(backend=BACKEND_PATH))


class TestBeforeTheExit:
    """What the backend has done by the time the process ends."""

    async def test_a_refused_environment_builds_no_connector(self, deployment, monkeypatch):
        """Status 78 with nothing connected and no session kept."""
        _, exits = deployment
        monkeypatch.setenv(RENDERED_FOR_ENV, "none")
        monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "someone")
        backend = _backend()
        with pytest.raises(_Exited):
            await backend.start()
        assert exits == [78]
        assert _TrackedFake.connects == 0
        assert backend._session is None

    async def test_a_missing_rendered_for_variable_opens_no_session(self, deployment, monkeypatch):
        """Status 78 before the session is opened."""
        _, exits = deployment
        monkeypatch.delenv(RENDERED_FOR_ENV, raising=False)
        with pytest.raises(_Exited):
            await _backend().start()
        assert exits == [78]
        assert _TrackedFake.connects == 0

    async def test_a_refusal_that_does_not_end_the_process_still_raises(
        self,
        deployment,  # noqa: ARG002 - the fixture sets up the session environment
        monkeypatch,
    ):
        """Were the exit ever to return, ``start`` does not carry on into serving."""
        monkeypatch.setenv(RENDERED_FOR_ENV, "none")
        monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "someone")
        monkeypatch.setattr(backend_module.os, "_exit", lambda status: None)
        with pytest.raises(SessionEnvironmentError):
            await _backend().start()

    async def test_a_rendered_for_difference_closes_the_session_first(
        self, deployment, monkeypatch
    ):
        """Status 75, after the connector that was built has been disconnected."""
        directory, exits = deployment
        write_record(directory, target="va", generation=3)
        monkeypatch.setenv(RENDERED_FOR_ENV, "va@2")
        backend = _backend()
        with pytest.raises(_Exited):
            await backend.start()
        assert exits == [75]
        assert (_TrackedFake.connects, _TrackedFake.connected) == (1, 0)
        assert backend._session is None

    async def test_a_difference_that_does_not_end_the_process_still_raises(
        self, deployment, monkeypatch
    ):
        """Were the exit ever to return, ``start`` does not carry on into serving."""
        directory, _ = deployment
        write_record(directory, target="va", generation=3)
        monkeypatch.setenv(RENDERED_FOR_ENV, "none")
        monkeypatch.setattr(backend_module.os, "_exit", lambda status: None)
        with pytest.raises(PVConnectionFailed):
            await _backend().start()

    async def test_a_move_closes_the_session_and_exits_75(self, deployment, monkeypatch):
        """The watcher sees the new generation, disconnects, and ends the process.

        No broker runs, so no push ever comes: the watcher drains once the push
        wait has run out.
        """
        directory, exits = deployment
        monkeypatch.setattr(backend_module, "STATEMENT_PUSH_WAIT_S", 0.05)
        write_record(directory, target="va", generation=3)
        monkeypatch.setenv(RENDERED_FOR_ENV, "va@3")
        backend = _backend()
        await backend.start()
        assert (exits, _TrackedFake.connected) == ([], 1)
        watcher = backend._watcher
        write_record(directory, target="va", generation=4)
        with pytest.raises(_Exited):
            await asyncio.wait_for(watcher, 5.0)
        assert exits == [75]
        assert _TrackedFake.connected == 0

    async def test_closing_stops_the_watcher_without_an_exit(self, deployment, monkeypatch):
        """An ordinary shutdown is not a move: no exit, and a later record change is ignored."""
        directory, exits = deployment
        write_record(directory, target="va", generation=3)
        monkeypatch.setenv(RENDERED_FOR_ENV, "va@3")
        backend = _backend()
        await backend.start()
        watcher = backend._watcher
        await backend.close()
        assert watcher.cancelled()
        write_record(directory, target="va", generation=4)
        await asyncio.sleep(0.1)
        assert exits == []
        assert _TrackedFake.connected == 0


# -- what the outgoing process says before it leaves --------------------------------

ARMED_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "writes": "allowed",
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}

CHANNEL = "SR:QF:01:SP"
OTHER_CHANNEL = "SR:QF:02:SP"

#: The broker's flush interval, in the injected clock's seconds.
TICK = 0.01

#: The claims the bound ``va`` target (a simulator) is stated with.
ON = {"state": "on", "reason": None, "cause": None}
ASKED = {"state": "asked", "reason": None, "cause": "asked"}
OFF = {"state": "off", "reason": refusal_phrase("switch_off", "simulator"), "cause": "switch_off"}
SWITCHING = {
    "state": "applying",
    "reason": refusal_phrase("switching", "simulator"),
    "cause": "switching",
}


class _FakeClock:
    """The broker loops' clock: a sleep returns only when the test advances past it."""

    def __init__(self) -> None:
        self.now = 0.0
        self._sleepers: list[tuple[float, asyncio.Future]] = []

    async def sleep(self, delay: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self._sleepers.append((self.now + delay, future))
        await future

    async def advance(self, delay: float) -> None:
        """Move time forward, wake every due sleeper, and let the woken loops run."""
        self.now += delay
        due = [f for t, f in self._sleepers if t <= self.now + 1e-9]
        self._sleepers = [(t, f) for t, f in self._sleepers if t > self.now + 1e-9]
        for future in due:
            if not future.done():
                future.set_result(None)
        for _ in range(20):
            await asyncio.sleep(0)


class _HeldMachine(_TrackedFake):
    """A writing fake; while ``hold`` is set to an unset event every write waits for it."""

    hold: asyncio.Event | None = None
    writes: list = []

    async def write_channel(self, channel_address, value, **kwargs):
        type(self).writes.append((channel_address, value))
        hold = type(self).hold
        if hold is not None:
            await hold.wait()
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
            observed_value=value,
        )


class _Recorder:
    """One ordered timeline: the frames a socket got, the drain's end, the exit."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    async def send_json(self, data) -> None:
        self.events.append(("frame", data))

    def frames(self) -> list[dict]:
        return [d for kind, d in self.events if kind == "frame" and d.get("type") == "heartbeat"]

    def writes(self) -> list[dict]:
        return [frame["writes"] for frame in self.frames()]

    def index(self, kind: str) -> int:
        return next(i for i, (k, _) in enumerate(self.events) if k == kind)

    def first_frame_index(self, predicate) -> int:
        return next(
            i
            for i, (k, d) in enumerate(self.events)
            if k == "frame" and d.get("type") == "heartbeat" and predicate(d["writes"])
        )

    def has(self, kind: str) -> bool:
        return any(k == kind for k, _ in self.events)


@pytest.fixture
def flip(deployment, monkeypatch):
    """The deployment with the tier allowed, bound to ``va@3`` with its switch allowed.

    Yields the state directory. What the session's verdict says is the real
    record's switch and the real hold store's records, set by each test.
    """
    directory, _ = deployment
    monkeypatch.setattr(connector_base, "_long_lived_session", False)

    def get_config_value(path, default=None, _config_path=None):
        return ARMED_SECTION if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)
    ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _HeldMachine)
    _HeldMachine.hold, _HeldMachine.writes = None, []
    write_record(directory, target="va", generation=3)
    monkeypatch.setenv(RENDERED_FOR_ENV, "va@3")
    yield directory
    if _HeldMachine.hold is not None:
        _HeldMachine.hold.set()
    _HeldMachine.hold = None


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("the condition never held")
        await asyncio.sleep(0.005)


class _Run:
    """One DATA: the backend, a clocked broker, one recorded socket."""

    def __init__(self, monkeypatch) -> None:
        self.recorder = _Recorder()
        self.clock = _FakeClock()
        self.backend = _backend()
        self.broker = PVBroker(
            self.backend,
            HareConfig(
                ws_batch_interval_ms=int(TICK * 1000),
                heartbeat_interval_s=100.0,
                monitor_grace_seconds=0.0,
            ),
        )
        self.broker._sleep = self.clock.sleep
        self.broker.add_client("c0", self.recorder)
        monkeypatch.setattr(
            backend_module,
            "_exit_process",
            lambda status: self.recorder.events.append(("exit", status)),
        )

    async def start(self, monkeypatch) -> None:
        await self.broker.start()
        await self.clock.advance(0)
        session = self.backend._session
        drain = session.drain

        async def recorded_drain(timeout):
            try:
                return await drain(timeout)
            finally:
                self.recorder.events.append(("drain_end", None))

        monkeypatch.setattr(session, "drain", recorded_drain)

    async def until_watcher_acts(self) -> None:
        """Real time only: the move check runs on ``MOVE_POLL_S``, never on the broker's clock."""
        watcher = self.backend._watcher
        await _until(lambda: watcher in self.backend._acting)
        # Let it reach its wait for the push.
        for _ in range(20):
            await asyncio.sleep(0)

    async def until_claim(self, claim: dict) -> None:
        """Real time for the move watcher to read the change, then one push tick."""
        await _until(lambda: self.backend.writes == claim)
        await self.clock.advance(TICK)

    async def until_exit(self) -> None:
        await _until(lambda: self.recorder.has("exit"))

    def stayed(self) -> bool:
        """Nothing drained, nothing exited, the move watcher still watching."""
        return (
            not self.recorder.has("drain_end")
            and not self.recorder.has("exit")
            and not self.backend._watcher.done()
        )

    async def stop(self) -> None:
        await self.broker.stop()


def _flip_record(directory: Path, *, target="va", generation=3) -> None:
    """Rewrite the record, so the move watcher reads it and the verdict again."""
    time.sleep(0.002)  # a new modification time even on a coarse clock
    write_record(directory, target=target, generation=generation)


def _switch(directory: Path, switch: dict[str, str]) -> None:
    """Set the record's switch to exactly *switch*; the record itself stays ``va@3``."""
    time.sleep(0.002)
    _set_switch_to(directory / control_context.RECORD_FILENAME, switch)
    control_context.invalidate_cache()


def _hold_store(directory: Path, monkeypatch, *keys: str) -> Path:
    """A hold store with an empty record per key, and the terminal naming *keys*."""
    root = directory.parent / "holds"
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    for key in keys:
        session_hold.ensure_session(key, root=root)
    _name_sessions(keys)
    return root


def _name_sessions(keys) -> None:
    (Path.cwd() / SESSIONS_FILENAME).write_text(json.dumps(sorted(keys)), encoding="utf-8")


def _raise_hold(root: Path, key: str) -> None:
    assert session_hold.open_flag(
        key,
        flag_id="flag-1",
        source="watchdog:limits",
        reason="a limit was refused",
        prompt_ref="turn-7",
        prompt_excerpt="set the current to 900",
        root=root,
    )


#: The claim a hold raised by ``watchdog:limits`` is stated with.
HELD = {
    "state": "held",
    "reason": refusal_phrase("session_held", "simulator") + " Raised by: watchdog:limits.",
    "cause": "session_held",
}


class TestAChangeIsShownInPlace:
    """A switch flip, a position change or a hold: the next tick's claim, and no exit."""

    async def test_writes_turned_on_are_pushed_as_on(self, flip, monkeypatch):
        _switch(flip, {})
        run = _Run(monkeypatch)
        await run.start(monkeypatch)
        try:
            assert run.backend.writes == OFF
            _switch(flip, {"va": "allowed"})
            await run.until_claim(ON)
            assert run.recorder.writes() == [ON]
            await asyncio.sleep(0.05)
            assert run.stayed()
            await run.backend.put(CHANNEL, 1.0, binding="va@3")
            assert _HeldMachine.writes == [(CHANNEL, 1.0)]
        finally:
            await run.stop()
            await run.backend.close()

    async def test_writes_turned_off_are_pushed_and_refuse_at_once(self, flip, monkeypatch):
        """ON→OFF with a write in flight: ``off`` is pushed, a new write refused, the first lands."""
        run = _Run(monkeypatch)
        await run.start(monkeypatch)
        try:
            assert run.backend.writes == ON
            _HeldMachine.hold = asyncio.Event()
            in_flight = asyncio.create_task(run.backend.put(CHANNEL, 1.0, binding="va@3"))
            await _until(lambda: _HeldMachine.writes != [])

            _switch(flip, {})
            await run.until_claim(OFF)
            assert run.recorder.writes() == [OFF]

            with pytest.raises(PVWriteRefused) as caught:
                await run.backend.put(OTHER_CHANNEL, 2.0, binding="va@3")
            assert caught.value.detail == OFF["reason"]
            assert _HeldMachine.writes == [(CHANNEL, 1.0)]

            _HeldMachine.hold.set()
            assert await in_flight is None
            await asyncio.sleep(0.05)
            assert run.stayed()
        finally:
            await run.stop()
            await run.backend.close()

    async def test_a_move_to_asked_is_pushed_as_asked(self, flip, monkeypatch):
        run = _Run(monkeypatch)
        await run.start(monkeypatch)
        try:
            _switch(flip, {"va": "asked"})
            await run.until_claim(ASKED)
            assert run.recorder.writes() == [ASKED]
            with pytest.raises(PVWriteRefused):
                await run.backend.put(CHANNEL, 1.0, binding="va@3")
            await run.backend.put(CHANNEL, 2.0, binding="va@3", confirmed=True)
            assert _HeldMachine.writes == [(CHANNEL, 2.0)]
            _switch(flip, {"va": "allowed"})
            await run.until_claim(ON)
            assert run.recorder.writes() == [ASKED, ON]
            await asyncio.sleep(0.05)
            assert run.stayed()
        finally:
            await run.stop()
            await run.backend.close()

    async def test_a_hold_is_pushed_as_held_and_lifts_in_place(self, flip, monkeypatch):
        root = _hold_store(flip, monkeypatch, "alice-1")
        run = _Run(monkeypatch)
        await run.start(monkeypatch)
        try:
            assert run.backend.writes == ON
            _raise_hold(root, "alice-1")
            await run.until_claim(HELD)
            with pytest.raises(PVWriteRefused) as caught:
                await run.backend.put(CHANNEL, 1.0, binding="va@3", confirmed=True)
            assert caught.value.detail == HELD["reason"]
            session_hold.answer_flags(
                "alice-1", ["flag-1"], answer="continue", operator="alice", root=root
            )
            await run.until_claim(ON)
            assert run.recorder.writes() == [HELD, ON]
            await asyncio.sleep(0.05)
            assert run.stayed()
            assert _HeldMachine.writes == []
        finally:
            await run.stop()
            await run.backend.close()

    async def test_a_held_session_joining_the_terminal_is_pushed_as_held(self, flip, monkeypatch):
        """``sessions.json`` names one more session, which is held: ``held``; it ends: ``on``."""
        root = _hold_store(flip, monkeypatch, "alice-1")
        session_hold.ensure_session("alice-2", root=root)
        _raise_hold(root, "alice-2")
        run = _Run(monkeypatch)
        await run.start(monkeypatch)
        try:
            assert run.backend.writes == ON
            _name_sessions(["alice-1", "alice-2"])
            await run.until_claim(HELD)
            _name_sessions(["alice-1"])
            await run.until_claim(ON)
            assert run.recorder.writes() == [HELD, ON]
            assert run.stayed()
        finally:
            await run.stop()
            await run.backend.close()


class TestOnlyAMoveExits:
    """The record names another ``(target, generation)``: ``switching`` is pushed, then 75."""

    async def test_another_generation_is_pushed_as_switching(self, flip, monkeypatch):
        run = _Run(monkeypatch)
        await run.start(monkeypatch)
        try:
            _flip_record(flip, generation=4)
            await run.until_watcher_acts()
            # The move is seen without a single flush tick; nothing has left
            # yet, and with nothing in flight the exit still waits for the push.
            await asyncio.sleep(0.05)
            assert run.recorder.writes() == []
            assert not run.recorder.has("exit")
            await run.clock.advance(TICK)
            assert run.recorder.writes() == [SWITCHING]
            await run.until_exit()
            await run.clock.advance(TICK)
        finally:
            await run.stop()
        writes = run.recorder.writes()
        assert writes == [SWITCHING]
        assert ("exit", EXIT_BINDING_MOVED) in run.recorder.events
        assert (
            run.recorder.first_frame_index(lambda w: True)
            < run.recorder.index("drain_end")
            < run.recorder.index("exit")
        )

    async def test_the_successor_s_first_frame_says_on(self, flip, monkeypatch):
        """After a move to ``va@4``, a DATA opened on the new record states plain ``on``."""
        run = _Run(monkeypatch)
        await run.start(monkeypatch)
        try:
            _flip_record(flip, generation=4)
            await run.until_watcher_acts()
            await run.clock.advance(TICK)
            await run.until_exit()
        finally:
            await run.stop()
            await run.backend.close()

        monkeypatch.setenv(RENDERED_FOR_ENV, "va@4")
        successor = _Run(monkeypatch)
        await successor.start(monkeypatch)
        try:
            first = successor.broker.heartbeat_frame()
            assert first["binding"] == "va@4"
            assert first["writes"] == ON
        finally:
            await successor.stop()
            await successor.backend.close()

    async def test_a_move_with_a_write_in_flight_never_says_on(self, flip, monkeypatch):
        """A write holds the drain: every frame until the exit is ``switching``."""
        run = _Run(monkeypatch)
        await run.start(monkeypatch)
        release = asyncio.Event()
        session = run.backend._session
        drain = session.drain

        async def held_drain(timeout):
            # A write still in the connector: the drain does not end until it does.
            await release.wait()
            return await drain(timeout)

        monkeypatch.setattr(session, "drain", held_drain)
        try:
            _flip_record(flip, generation=4)
            await run.until_watcher_acts()
            for _ in range(10):
                await run.clock.advance(TICK)
            assert run.recorder.writes() == [SWITCHING]
            assert run.broker.heartbeat_frame()["writes"] == SWITCHING
            assert not run.recorder.has("drain_end")
            assert not run.recorder.has("exit")
            release.set()
            await run.until_exit()
            await run.clock.advance(TICK)
        finally:
            await run.stop()
        assert run.recorder.writes() == [SWITCHING]
        assert run.recorder.first_frame_index(lambda w: True) < run.recorder.index("drain_end")

    async def test_a_broker_that_never_reports_a_push_holds_the_exit_only_for_the_bound(
        self, flip, monkeypatch
    ):
        """No ``statement_sent()``: the watcher drains once ``STATEMENT_PUSH_WAIT_S`` has run out."""
        monkeypatch.setattr(backend_module, "STATEMENT_PUSH_WAIT_S", 0.2)
        run = _Run(monkeypatch)
        run.backend.statement_sent = lambda: None
        await run.start(monkeypatch)
        try:
            # The bound runs from when the watcher acts, which is after the move
            # and before this test can observe it; timing from the move never
            # reads the wait as shorter than it was.
            moved = time.monotonic()
            _flip_record(flip, generation=4)
            await run.until_watcher_acts()
            await run.clock.advance(TICK)
            assert run.recorder.writes()[0] == SWITCHING
            assert not run.recorder.has("drain_end")
            await run.until_exit()
            waited = time.monotonic() - moved
        finally:
            await run.stop()
        assert waited >= 0.2 - 0.01
        assert run.recorder.index("drain_end") < run.recorder.index("exit")

    async def test_a_failed_watch_is_pushed_as_switching(self, flip, monkeypatch):
        """A move check that raises leaves as a move would, saying ``switching`` first."""
        run = _Run(monkeypatch)
        await run.start(monkeypatch)

        def broken():
            raise OSError("the record cannot be read")

        monkeypatch.setattr(run.backend._session, "_refresh", broken)
        try:
            _flip_record(flip)
            await run.until_watcher_acts()
            await run.clock.advance(TICK)
            assert run.recorder.writes() == [SWITCHING]
            await run.until_exit()
        finally:
            await run.stop()
        assert run.recorder.first_frame_index(lambda w: True) < run.recorder.index("drain_end")


# -- what the next process on the same record starts in -----------------------------

#: Where the backend's wall clock starts in the tests below: epoch seconds.
WALL_START = 1_760_700_000.0

HOLD_S = epics_connector_module.STALE_HOLD_DOWN_S

SAME_RECORD = control_context.ControlContext(target="va", generation=3, posture={})


class _ReachClock:
    """The backend's two clocks, stepped by the test."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return 1000.0 + self.now

    def time(self) -> float:
        return WALL_START + self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ReachMachine(_HeldMachine):
    """A machine whose stale watch raises the stale transport as a lost connector.

    ``wait_stale`` raises :class:`ConnectorUnavailableError` once :meth:`go_stale`
    is called, as the EPICS connector does; the session returns on it, and the
    backend's stale watch states ``no`` before the exit.
    """

    async def wait_stale(self) -> None:
        await self._stale.wait()
        raise ConnectorUnavailableError("stale transport")


@pytest.fixture
def reach(flip):
    """The flip deployment over :class:`_ReachMachine`; yields the state directory."""
    ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _ReachMachine)
    return flip


def _clocked(monkeypatch, clock: _ReachClock) -> _Run:
    run = _Run(monkeypatch)
    run.backend._clock = clock.monotonic
    run.backend._wall_clock = clock.time
    return run


def _machine(run: _Run) -> _ReachMachine:
    return run.backend._session._connector


def _statement_file() -> dict:
    return json.loads((Path.cwd() / STATEMENT_FILENAME).read_text(encoding="utf-8"))


def _record_statement_at_exit(run: _Run, monkeypatch) -> list[dict]:
    """Replace the run's exit with one that also reads the statement file as the process ends."""
    seen: list[dict] = []

    def exit_process(status):
        seen.append(_statement_file())
        run.recorder.events.append(("exit", status))

    monkeypatch.setattr(backend_module, "_exit_process", exit_process)
    return seen


def _reachables(run: _Run) -> list[dict]:
    return [frame["reachable"] for frame in run.recorder.frames()]


def _shape(steps: list) -> list:
    """State and ``since`` identity of a reachable sequence, repeats collapsed.

    Each ``since`` becomes the order it first appeared in, so two sequences with
    different clocks compare by whether a ``since`` is kept or replaced.
    """
    names: dict[float, int] = {}
    shape: list = []
    for step in steps:
        if step == "close":
            item: object = "close"
        else:
            since = step["since"]
            item = (step["state"], None if since is None else names.setdefault(since, len(names)))
        if not shape or shape[-1] != item:
            shape.append(item)
    return shape


#: OSPREY's own sequence for an exit 76 and a restart on the same binding.
EXIT_76_SEQUENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "backend-statement.json").read_text(encoding="utf-8")
)["exit_76_sequence"]


def _vector_shape(sequence_id: str) -> list:
    """The reachable shape of CANOPY's sequence *sequence_id*, or of OSPREY's exit-76 one."""
    if sequence_id == EXIT_76_SEQUENCE["id"]:
        sequence = EXIT_76_SEQUENCE
    else:
        sequence = next(s for s in load_heartbeat_vectors()["sequences"] if s["id"] == sequence_id)
    return _shape(
        [
            "close" if step["event"] == "close" else step["frame"]["reachable"]
            for step in sequence["steps"]
        ]
    )


async def _outage(run: _Run, clock: _ReachClock) -> float:
    """Connect the channel, drop it past the hold-down; returns the drop's ``since``."""
    await run.backend.subscribe(CHANNEL, lambda _event: None)
    await run.clock.advance(TICK)
    clock.advance(7.0)
    dropped_at = clock.time()
    _machine(run).drop(CHANNEL)
    clock.advance(HOLD_S + 1.0)
    await run.clock.advance(TICK)
    return dropped_at


async def _successor(monkeypatch, clock: _ReachClock, carried: str | None) -> _Run:
    if carried is None:
        monkeypatch.delenv(REACH_SINCE_ENV, raising=False)
    else:
        monkeypatch.setenv(REACH_SINCE_ENV, carried)
    successor = _clocked(monkeypatch, clock)
    await successor.start(monkeypatch)
    return successor


@pytest.mark.usefixtures("reach")
class TestTheNextProcessStartsInTheLastStatement:
    """The statement file a DATA leaves, and the carry its successor starts in."""

    async def test_exit_76_with_a_name_up_writes_no_since_the_probe(self, monkeypatch):
        """The adapter still says ``yes``; the stale watch sets ``no`` and writes it before the exit."""
        clock = _ReachClock()
        run = _clocked(monkeypatch, clock)
        at_exit = _record_statement_at_exit(run, monkeypatch)
        await run.start(monkeypatch)
        try:
            await run.backend.subscribe(CHANNEL, lambda _event: None)
            await run.clock.advance(TICK)
            assert run.backend.reachable["state"] == "yes"
            assert _statement_file()["state"] == "yes"
            clock.advance(30.0)
            probe = clock.time()
            _machine(run).go_stale()
            await run.until_exit()
            assert run.backend.reachable == {"state": "no", "since": probe, "reason": None}
        finally:
            await run.stop()
            await run.backend.close()
        assert ("exit", EXIT_TRANSPORT_STALE) in run.recorder.events
        assert at_exit == [
            {"binding_key": "va@3", "state": "no", "since": probe, "written_at": probe}
        ]

    async def test_exit_76_after_a_held_drop_keeps_the_drop_s_since(self, monkeypatch):
        """A ``no`` already stated outranks the probe: the file keeps the drop's ``since``."""
        clock = _ReachClock()
        run = _clocked(monkeypatch, clock)
        at_exit = _record_statement_at_exit(run, monkeypatch)
        await run.start(monkeypatch)
        try:
            dropped_at = await _outage(run, clock)
            clock.advance(30.0)
            _machine(run).go_stale()
            await run.until_exit()
        finally:
            await run.stop()
            await run.backend.close()
        assert [(s["state"], s["since"]) for s in at_exit] == [("no", dropped_at)]

    async def test_a_successor_after_exit_76_holds_the_carried_no_until_its_channel_connects(
        self, monkeypatch
    ):
        """Started into the outage: the carried ``no`` and ``since`` from its first frame on.

        Its channel never connects, so nothing of its own says otherwise: the
        ``since`` stays the predecessor's, and no stale exit comes from it. Its
        first channel up is ``yes``. The frames across the restart have the
        vector's ``recovery-through-exit-76`` shape.
        """
        clock = _ReachClock()
        run = _clocked(monkeypatch, clock)
        await run.start(monkeypatch)
        try:
            dropped_at = await _outage(run, clock)
            _machine(run).go_stale()
            await run.until_exit()
        finally:
            await run.stop()
            await run.backend.close()
        before = _reachables(run)

        carried = sidecar_module.carried_reach(Path.cwd(), SAME_RECORD)
        assert carried == f"no,{dropped_at!r}"
        clock.advance(2.0)
        successor = await _successor(monkeypatch, clock, carried)
        try:
            first = successor.broker.heartbeat_frame()["reachable"]
            assert first == {"state": "no", "since": dropped_at, "reason": None}
            _machine(successor).drop(CHANNEL)
            await successor.backend.subscribe(CHANNEL, lambda _event: None)
            seen = []
            for _ in range(30):
                clock.advance(1.0)
                await successor.clock.advance(TICK)
                seen.append(successor.backend.reachable)
            assert {(s["state"], s["since"]) for s in seen} == {("no", dropped_at)}
            assert not successor.recorder.has("exit")
            _machine(successor).restore(CHANNEL)
            await successor.clock.advance(TICK)
            assert successor.backend.reachable == {"state": "yes", "since": None, "reason": None}
            after = [first, *_reachables(successor)]
        finally:
            await successor.stop()
            await successor.backend.close()
        assert _shape([*before, "close", *after]) == _vector_shape("recovery-through-exit-76")

    async def test_a_switch_flip_during_an_outage_keeps_the_process_and_its_no(
        self, reach, monkeypatch
    ):
        """No exit: the outage's ``no`` and ``since`` stand, the claim turns ``off``."""
        clock = _ReachClock()
        run = _clocked(monkeypatch, clock)
        await run.start(monkeypatch)
        try:
            dropped_at = await _outage(run, clock)
            _switch(reach, {})
            await run.until_claim(OFF)
            assert run.recorder.frames()[-1]["reachable"] == {
                "state": "no",
                "since": dropped_at,
                "reason": None,
            }
            await asyncio.sleep(0.05)
            assert run.stayed()
            assert (_statement_file()["state"], _statement_file()["since"]) == ("no", dropped_at)
        finally:
            await run.stop()
            await run.backend.close()

    async def test_a_restart_during_an_outage_hands_on_the_since(
        self,
        reach,  # noqa: ARG002 - the deployment over the reach machine
        monkeypatch,
    ):
        """A DATA replaced on the same binding mid-outage: the successor's first frame is its ``no``.

        Its own channel never connects, so the carried ``no`` stands. The frames
        across the restart have the vector's ``same-binding-restart-during-outage``
        shape.
        """
        clock = _ReachClock()
        run = _clocked(monkeypatch, clock)
        await run.start(monkeypatch)
        try:
            dropped_at = await _outage(run, clock)
            await run.clock.advance(TICK)
        finally:
            await run.stop()
            await run.backend.close()
        assert not run.recorder.has("exit")
        before = _reachables(run)

        carried = sidecar_module.carried_reach(Path.cwd(), SAME_RECORD)
        assert carried == f"no,{dropped_at!r}"
        clock.advance(2.0)
        successor = await _successor(monkeypatch, clock, carried)
        try:
            first = successor.broker.heartbeat_frame()["reachable"]
            assert first == {"state": "no", "since": dropped_at, "reason": None}
            _machine(successor).drop(CHANNEL)
            await successor.backend.subscribe(CHANNEL, lambda _event: None)
            for _ in range(30):
                clock.advance(1.0)
                await successor.clock.advance(TICK)
            assert successor.backend.reachable == first
            after = [first, *_reachables(successor)]
        finally:
            await successor.stop()
            await successor.backend.close()
        assert _shape([*before, "close", *after]) == _vector_shape(
            "same-binding-restart-during-outage"
        )

    async def test_a_changed_record_identity_starts_the_successor_without_a_carry(
        self, reach, monkeypatch
    ):
        """The record moved to another generation: no carry, the successor starts ``unknown``."""
        clock = _ReachClock()
        run = _clocked(monkeypatch, clock)
        await run.start(monkeypatch)
        try:
            await _outage(run, clock)
            _flip_record(reach, generation=4)
            await run.until_watcher_acts()
            await run.clock.advance(TICK)
            await run.until_exit()
        finally:
            await run.stop()
            await run.backend.close()
        assert _statement_file()["binding_key"] == "va@3"

        moved = control_context.ControlContext(target="va", generation=4, posture={})
        carried = sidecar_module.carried_reach(Path.cwd(), moved)
        assert carried is None
        assert not (Path.cwd() / STATEMENT_FILENAME).exists()
        monkeypatch.setenv(RENDERED_FOR_ENV, "va@4")
        successor = await _successor(monkeypatch, clock, carried)
        try:
            assert successor.broker.heartbeat_frame()["reachable"]["state"] == "unknown"
        finally:
            await successor.stop()
            await successor.backend.close()

    async def test_a_carried_yes_holds_until_the_successor_s_own_evidence(self, monkeypatch):
        """A ``yes`` carried over a restart: ``yes`` from the first frame, not ``unknown``."""
        clock = _ReachClock()
        successor = await _successor(monkeypatch, clock, "yes")
        try:
            assert successor.broker.heartbeat_frame()["reachable"]["state"] == "yes"
            await successor.backend.subscribe(CHANNEL, lambda _event: None)
            assert successor.backend.reachable["state"] == "yes"
        finally:
            await successor.stop()
            await successor.backend.close()

    @pytest.mark.parametrize("value", ["", "maybe", "no", "no,soon", "no,nan", "yes,1.0"])
    async def test_a_carry_that_says_nothing_definite_is_ignored(self, monkeypatch, value):
        successor = await _successor(monkeypatch, _ReachClock(), value)
        try:
            assert successor.broker.heartbeat_frame()["reachable"]["state"] == "unknown"
        finally:
            await successor.stop()
            await successor.backend.close()

    async def test_releasing_the_last_screen_ends_a_carried_no(self, monkeypatch):
        """With nothing subscribed the successor says ``unknown``, as its predecessor would."""
        successor = await _successor(monkeypatch, _ReachClock(), f"no,{WALL_START!r}")
        try:
            _machine(successor).drop(CHANNEL)
            handle = await successor.backend.subscribe(CHANNEL, lambda _event: None)
            assert successor.backend.reachable["state"] == "no"
            await handle.close()
            assert successor.backend.reachable["state"] == "unknown"
        finally:
            await successor.stop()
            await successor.backend.close()

    async def test_every_definite_change_rewrites_the_file(self, monkeypatch):
        """``yes`` at the first channel, ``no`` once the drop is held; ``unknown`` writes nothing."""
        clock = _ReachClock()
        run = _clocked(monkeypatch, clock)
        await run.start(monkeypatch)
        try:
            await run.clock.advance(TICK)
            assert not (Path.cwd() / STATEMENT_FILENAME).exists()
            dropped_at = await _outage(run, clock)
            assert (_statement_file()["state"], _statement_file()["since"]) == ("no", dropped_at)
            _machine(run).restore(CHANNEL)
            await run.clock.advance(TICK)
            assert (_statement_file()["state"], _statement_file()["since"]) == ("yes", None)
        finally:
            await run.stop()
            await run.backend.close()
