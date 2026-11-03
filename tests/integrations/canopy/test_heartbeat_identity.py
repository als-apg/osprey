"""The first frame of a CANOPY PV socket names the binding OSPREY's backend serves.

CANOPY's ``ws_pvs`` route opens every socket with the broker's heartbeat frame,
which reads ``label``, ``notice`` and ``binding`` off the PV connector. This
module proves the seam between the two repositories: a real
:class:`~canopy.hare.services.broker.PVBroker` is started over
:class:`~osprey.integrations.canopy.backend.OspreyPVConnector` (loaded by its
``module:Class`` path, as CANOPY loads it), the real route is driven over a
small fake socket, and the first frame it sends must carry the session's
identity in each of the three states a session can be in — bound to a record,
falling back to the baseline, or unservable.

The same broker is then run against the installed CANOPY wheel's
``heartbeat_vectors.json``: every frame OSPREY sends has the vector frames'
shape. Every row of OSPREY's writes claim table (``writes_claims`` in
``fixtures/backend-statement.json``) is reached through the real session — the
real record, switch, hold store and sessions file wherever one reaches it — and
the frame's ``writes`` equals the row's claim. A changed statement reaches the
connector's ``statement_sent()`` with no socket connected, and a switch flip
under a running process is such a change: no exit, the claim changes.

Everything runs on the test's own event loop, so the session the broker starts
is the one the route reads from.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from importlib import resources
from pathlib import Path

import canopy.contracts
import pytest
from canopy.contracts import HEARTBEAT_VECTORS_FILE, load_heartbeat_vectors
from canopy.epics.config import EpicsConfig
from canopy.epics.gateway import build_connector
from canopy.hare.api.v1 import ws as ws_module
from canopy.hare.config import HareConfig
from canopy.hare.services.broker import PVBroker
from fastapi import WebSocketDisconnect

from osprey.integrations.canopy import SESSIONS_FILENAME
from osprey.integrations.canopy import backend as backend_module
from osprey.runtime import session as session_module
from osprey.runtime.session import STATE_BOUND, STATE_FALLBACK, STATE_UNSERVABLE
from osprey.runtime.session_outcomes import refusal_message
from osprey_connectors import control_context, posture_store, session_hold, types
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.posture_store import WriteVerdict
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._harness._control_context_fixtures import _set_switch_to

# start() arms the real raw-put block in this process; put back what it patched.
pytestmark = pytest.mark.usefixtures("raw_put_block_restored")

BACKEND_PATH = "osprey.integrations.canopy.backend:OspreyPVConnector"

#: The keys of a heartbeat frame, as CANOPY's route documents them.
FRAME_KEYS = {"type", "timestamp", "backend", "label", "notice", "binding", "reachable", "writes"}

#: A deployment whose baseline is the virtual accelerator.
VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}


#: The same deployment with the write tier allowed.
ARMED_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "writes": "allowed",
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}


def _refused(reason: str) -> WriteVerdict:
    """The execution verdict refusing the bound ``va`` target for *reason*."""
    return WriteVerdict(allowed=False, reason=reason, target="va", kind="simulator")


#: The installed CANOPY wheel's heartbeat vectors, read through the package.
VECTORS = load_heartbeat_vectors()
#: OSPREY's writes claim table.
CLAIMS = json.loads(
    (Path(__file__).parent / "fixtures" / "backend-statement.json").read_text(encoding="utf-8")
)["writes_claims"]
ROWS = {row["id"]: row for row in CLAIMS["rows"]}
VECTOR_FRAMES = [
    case["frame"] for case in (*VECTORS["reachable_frames"], *VECTORS["writes_frames"])
]

#: Who raises the hold the ``session_held`` row is reached with.
HOLD_SOURCE = "watchdog:limits"


def _expected_claim(row: dict, verdict: WriteVerdict) -> dict:
    """*row*'s claim with its reason token filled from the session's *verdict*."""
    claim = dict(row["claim"])
    if claim["reason"] == "<refusal_message>":
        claim["reason"] = refusal_message(verdict)
    elif claim["reason"] == "<phrase:switching>":
        claim["reason"] = refusal_phrase("switching", verdict.kind)
    return claim


class _RefusingConnector(FakeConnector):
    """A connector whose control system is not there."""

    async def connect(self, config):  # noqa: ARG002 - the connector connect signature
        raise ConnectionError("no gateway answers")


class _FakeSocket:
    """Just enough of a WebSocket for ``ws_pvs``: records frames, then hangs up."""

    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[dict] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data) -> None:
        self.sent.append(data)

    async def receive_json(self):
        raise WebSocketDisconnect(code=1000)

    async def close(self) -> None:
        pass


def _write_record(state_dir: Path, switch: dict[str, str] | None = None) -> None:
    """Write the ``va@3`` record; with *switch*, make its switch exactly that."""
    path = state_dir / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target="va", generation=3, posture={}), path=path
    )
    control_context.invalidate_cache()
    if switch is not None:
        _set_switch_to(path, switch)
        control_context.invalidate_cache()


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """A session process's environment, serving :data:`VA_SECTION`, with no record yet."""
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

    def no_exit(status):
        raise AssertionError(f"the backend tried to end the test process with status {status}")

    monkeypatch.setattr(backend_module.os, "_exit", no_exit)
    control_context.invalidate_cache()
    yield directory
    control_context.invalidate_cache()


def _broker(connector) -> PVBroker:
    return PVBroker(
        connector,
        HareConfig(ws_batch_interval_ms=10, heartbeat_interval_s=100.0, monitor_grace_seconds=0.0),
    )


async def _first_frame(
    connector_class,
    *,
    state: str,
    before_socket: Callable[[backend_module.OspreyPVConnector], Awaitable[None]] | None = None,
    seen: dict | None = None,
) -> tuple[dict, object]:
    """Start a broker over the backend, open one socket, return its first frame and binding.

    *before_socket*, when given, runs on the started connector before the socket opens.
    *seen*, when given, gets the session's verdict as the socket opened, under ``verdict``.
    """
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, connector_class)
        connector = build_connector(EpicsConfig(backend=BACKEND_PATH, connect_report_s=5.0))
        broker = _broker(connector)
        await broker.start()
        try:
            binding = connector._session.binding
            assert binding.state == state
            if before_socket is not None:
                await before_socket(connector)
            if seen is not None:
                seen["verdict"] = connector._session.verdict
            socket = _FakeSocket()
            await ws_module.ws_pvs(socket, broker)
        finally:
            await broker.stop()
    assert socket.accepted
    assert socket.sent, "the route sent no frame"
    return socket.sent[0], binding


def _check_frame(frame: dict, binding, *, notice, binding_id: str) -> None:
    # The frame crosses the wire as JSON, so check what a client would decode.
    decoded = json.loads(json.dumps(frame))
    assert set(decoded) == FRAME_KEYS
    assert decoded["type"] == "heartbeat"
    assert decoded["backend"] == "osprey"
    assert decoded["label"] == binding.display_name
    assert isinstance(decoded["label"], str) and decoded["label"]
    assert decoded["notice"] == notice
    assert decoded["binding"] == binding.id == binding_id


class TestFirstFrameIdentity:
    """One binding state per test; the first frame is checked in each."""

    async def test_a_bound_backend_opens_the_socket_with_its_record(self, state_dir, monkeypatch):
        """Bound to ``va@3``: the target's name, no notice, the record as the binding."""
        _write_record(state_dir)
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, "va@3")
        frame, binding = await _first_frame(FakeConnector, state=STATE_BOUND)
        _check_frame(frame, binding, notice=None, binding_id="va@3")

    async def test_a_fallback_backend_opens_the_socket_saying_so(self, state_dir, monkeypatch):  # noqa: ARG002 - the fixture sets the environment
        """No record: the baseline's name, notice and binding both ``fallback``."""
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, backend_module.NO_RECORD)
        frame, binding = await _first_frame(FakeConnector, state=STATE_FALLBACK)
        _check_frame(frame, binding, notice="fallback", binding_id="fallback")

    async def test_an_unservable_backend_opens_the_socket_saying_so(self, state_dir, monkeypatch):
        """A record whose control system refuses: notice and binding both ``unservable``."""
        _write_record(state_dir)
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, "va@3")
        frame, binding = await _first_frame(_RefusingConnector, state=STATE_UNSERVABLE)
        _check_frame(frame, binding, notice="unservable", binding_id="unservable")


# -- the installed wheel's heartbeat vectors ----------------------------------


def _use_section(monkeypatch, section: dict) -> None:
    def get_config_value(path, default=None, _config_path=None):
        return section if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)


def _verdict(monkeypatch, verdict: WriteVerdict) -> None:
    """The execution verdict, for a row no record reaches on this host."""
    monkeypatch.setattr(session_module, "_verdict", lambda target, session_keys=None: verdict)


def _armed(state_dir: Path, monkeypatch, switch: dict[str, str] | None = None) -> None:
    """Bound to ``va@3``, the tier allowed, the switch *switch* (default: ``va`` allowed)."""
    _write_record(state_dir, {"va": "allowed"} if switch is None else switch)
    monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, "va@3")
    _use_section(monkeypatch, ARMED_SECTION)


def _held(tmp_path: Path, monkeypatch) -> None:
    """The terminal names two sessions in its sessions file; the second is held."""
    root = tmp_path / "holds"
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    for key in ("alice-1", "alice-2"):
        session_hold.ensure_session(key, root=root)
    (tmp_path / SESSIONS_FILENAME).write_text('["alice-1", "alice-2"]', encoding="utf-8")
    assert session_hold.open_flag(
        "alice-2",
        flag_id="flag-1",
        source=HOLD_SOURCE,
        reason="a limit was refused",
        prompt_ref="turn-7",
        prompt_excerpt="set the current to 900",
        root=root,
    )


async def _move_record(connector) -> None:
    """Point the record at ``va@4``, then wait until the move watcher has seen it and left.

    The watcher leaves only once the broker has pushed the changed statement,
    so it is waited for while the broker still runs.
    """
    state_dir = Path(os.environ[posture_store.CONTROL_CONTEXT_DIR_ENV_VAR])
    control_context.write_record(
        control_context.ControlContext(target="va", generation=4, posture={}),
        path=state_dir / control_context.RECORD_FILENAME,
    )
    control_context.invalidate_cache()
    session = connector._session
    watcher = connector._watcher
    for _ in range(500):
        if session.target_moving and connector.writes["cause"] == "switching" and watcher.done():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the move watcher never saw the record move")


def _setup(row_id: str, state_dir: Path, monkeypatch, exits: list[int]):
    """Put the process in the state that makes *row_id* the claim it states.

    Returns ``(connector class, binding state, hook run before the socket opens)``.
    """
    bound = (FakeConnector, STATE_BOUND, None)
    tmp_path = state_dir.parent
    if row_id == "on":
        _armed(state_dir, monkeypatch)
        return bound
    if row_id == "asked":
        _armed(state_dir, monkeypatch, {"va": "asked"})
        return bound
    if row_id == "session_held":
        _armed(state_dir, monkeypatch)
        _held(tmp_path, monkeypatch)
        return bound
    if row_id == "switch_off":
        _armed(state_dir, monkeypatch, {})
        return bound
    if row_id in ("stopped_since", "switching"):
        _armed(state_dir, monkeypatch)
        _verdict(monkeypatch, _refused(row_id))
        return bound
    if row_id == "target_moving":
        _armed(state_dir, monkeypatch)
        monkeypatch.setattr(backend_module, "_exit_process", exits.append)
        return FakeConnector, STATE_BOUND, _move_record
    if row_id == "fallback":
        _armed(state_dir, monkeypatch)
        (state_dir / control_context.RECORD_FILENAME).unlink()
        control_context.invalidate_cache()
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, backend_module.NO_RECORD)
        return FakeConnector, STATE_FALLBACK, None
    if row_id == "unservable":
        _armed(state_dir, monkeypatch)
        return _RefusingConnector, STATE_UNSERVABLE, None
    if row_id == "tier_never":
        _armed(state_dir, monkeypatch)
        _use_section(monkeypatch, VA_SECTION)
        return bound
    if row_id == "readonly_run":
        _armed(state_dir, monkeypatch)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        return bound
    if row_id == "record_unavailable":
        _armed(state_dir, monkeypatch)
        (tmp_path / SESSIONS_FILENAME).write_text("[not json", encoding="utf-8")
        return bound
    raise AssertionError(f"no setup reaches the writes_claims row {row_id!r}")


class TestHeartbeatVectors:
    """The real broker over the backend, checked against the wheel's frames and OSPREY's causes."""

    def test_the_vectors_are_the_installed_wheel_s_file(self):
        installed = resources.files(canopy.contracts) / HEARTBEAT_VECTORS_FILE
        assert VECTORS == json.loads(installed.read_text(encoding="utf-8"))
        assert ROWS, "OSPREY's cause table has no rows"

    def test_the_claim_table_covers_every_claim_state(self):
        states = {row["claim"]["state"] for row in ROWS.values()}
        assert states == {"on", "asked", "held", "off", "applying", "locked"}

    @pytest.mark.parametrize("row_id", list(ROWS))
    async def test_each_claim_is_stated_as_its_row(self, state_dir, monkeypatch, row_id):
        """Every row is reached through the real session; the frame's ``writes`` is its claim."""
        exits: list[int] = []
        row = ROWS[row_id]
        connector_class, state, hook = _setup(row_id, state_dir, monkeypatch, exits)
        assert state == row["binding_state"]
        seen: dict = {}
        frame, binding = await _first_frame(
            connector_class, state=state, before_socket=hook, seen=seen
        )
        decoded = json.loads(json.dumps(frame))

        # The frame has the vector frames' shape, key for key.
        for vector in VECTOR_FRAMES:
            assert set(decoded) == set(vector)
            assert set(decoded["reachable"]) == set(vector["reachable"])
            if vector["writes"] is not None:
                assert set(decoded["writes"]) == set(vector["writes"])
        assert decoded["label"] == binding.display_name

        verdict = seen["verdict"]
        if state == STATE_BOUND and not row["target_moving"]:
            # The verdict the row names is the one the session read.
            assert (verdict.allowed, verdict.reason, verdict.position) == (
                row["verdict"]["allowed"],
                row["verdict"]["reason"],
                row["verdict"]["position"],
            )
        assert decoded["writes"] == _expected_claim(row, verdict)
        if row_id == "on":
            on = next(case for case in VECTORS["writes_frames"] if case["case"] == "on")
            assert decoded["writes"] == on["frame"]["writes"]
        if row_id == "session_held":
            assert decoded["writes"]["reason"].endswith(f"Raised by: {HOLD_SOURCE}.")
        if row_id == "target_moving":
            assert exits == [session_module.EXIT_BINDING_MOVED]
        else:
            assert exits == []

    async def test_a_statement_change_with_no_socket_calls_statement_sent(
        self, state_dir, monkeypatch
    ):
        """No socket: an unchanged statement leaves the event clear; a change sets it."""
        _armed(state_dir, monkeypatch)
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, FakeConnector)
            connector = build_connector(EpicsConfig(backend=BACKEND_PATH, connect_report_s=5.0))
            sent = 0
            original = connector.statement_sent

            def counting() -> None:
                nonlocal sent
                sent += 1
                original()

            connector.statement_sent = counting
            broker = _broker(connector)
            await broker.start()
            try:
                assert connector.writes["state"] == "on"
                await asyncio.sleep(0.1)  # ten flush ticks, nothing changed
                assert sent == 0
                assert not connector._statement_sent.is_set()

                # A switch flip is no move: the claim changes, the process stays.
                _write_record(state_dir, {})
                await asyncio.wait_for(connector._statement_sent.wait(), timeout=5.0)
                assert not broker._channels
                assert sent == 1
                assert connector.writes["cause"] == "switch_off"
                assert not connector._watcher.done()
            finally:
                await broker.stop()

    async def test_no_session_makes_no_writes_claim(self):
        connector = backend_module.OspreyPVConnector()
        assert connector.writes is None
        connector.statement_sent()
        assert connector._statement_sent.is_set()
