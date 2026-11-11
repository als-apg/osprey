"""The Channels value child: its wire, its reads, and how it ends.

The in-process half opens a real session on a real record in ``tmp_path``, with
a fake connector behind it, and serves on a real AF_UNIX socket. ``serve``
returns the status ``main`` would exit with, so nothing is replaced to read it.

The real-child half runs ``python -m`` on a socket this test holds. A child sees
no test double, so its deployment is an empty config: a ``live`` target nothing
can derive, which is an unservable binding that touches no network and still
reads and watches the record.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from osprey.interfaces.channel_finder import directory_values_child as child_module
from osprey.interfaces.channel_finder import directory_values_ipc as ipc
from osprey.interfaces.channel_finder.directory_values_ipc import ValueRow
from osprey.runtime import session as session_module
from osprey.runtime.session import EXIT_BINDING_MOVED, EXIT_START_REFUSED, EXIT_TRANSPORT_STALE
from osprey.services.python_executor.write_surface import _ARMED_CHECKED
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.control_system import _base as connector_base
from osprey_connectors.control_system import _epics_connector as epics_connector_module
from osprey_connectors.control_system._base import ChannelMetadata, ChannelValue
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.posture_env import POSTURE_ENV_VAR, SANDBOX_MODE, is_readonly_run
from osprey_connectors_tests._epics_fakes import FakeAioca, FakeChannel, InlineCaLoop
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._harness._control_context_fixtures import _set_switch_to
from tests._audit_capture import capture_audit_bodies

#: How long a real child gets to come up or to leave.
CHILD_TIMEOUT_S = 120.0

#: How long an orphan gets to notice: about two polls and its way out.
ORPHAN_BOUND_S = 15.0

VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}


def _write_record(state_dir: Path, *, target: str = "va", generation: int = 3) -> None:
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation),
        path=state_dir / control_context.RECORD_FILENAME,
    )
    control_context.invalidate_cache()


@pytest.fixture(autouse=True)
def _inherits_a_writing_posture(monkeypatch) -> None:
    """``serve`` sets the posture on this process: restored here, and it starts writable."""
    monkeypatch.setenv(POSTURE_ENV_VAR, "readwrite")


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """A 0700 directory with a path short enough for AF_UNIX, unlike ``tmp_path``."""
    directory = tempfile.mkdtemp(prefix="dv")
    yield Path(directory)
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def listener(short_dir: Path) -> Iterator[socket.socket]:
    """The listening socket a supervisor holds across respawns."""
    held = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    held.bind(str(short_dir / "v.sock"))
    held.listen(16)
    yield held
    held.close()


# -- the wire -----------------------------------------------------------------


class _Reader(asyncio.StreamReader):
    def __init__(self, data: bytes) -> None:
        super().__init__()
        self.feed_data(data)
        self.feed_eof()


class TestFraming:
    async def test_a_request_round_trips(self):
        frame = ipc.encode_request(["A:B", "C:D"])
        message = await ipc.read_frame(_Reader(frame))
        assert ipc.decode_request(message) == ["A:B", "C:D"]

    async def test_a_response_round_trips(self):
        rows = [
            ValueRow("A", ipc.OUTCOME_FOUND, 1.5, "2026-11-06T00:00:00+00:00", 1, "HIGH"),
            ValueRow("B", ipc.OUTCOME_NO_ANSWER, reason="no answer within 1 s"),
            ValueRow("C", ipc.OUTCOME_UNAVAILABLE, reason="nothing to read"),
        ]
        message = await ipc.read_frame(_Reader(ipc.encode_response(rows)))
        assert ipc.decode_response(message) == rows

    def test_a_row_has_the_contract_fields_and_reason_only_when_there_is_one(self):
        fields = {"name", "outcome", "value", "timestamp", "severity", "status"}
        assert set(ValueRow("A", ipc.OUTCOME_FOUND).to_wire()) == fields
        with_reason = ValueRow("A", ipc.OUTCOME_NO_ANSWER, reason="x").to_wire()
        assert set(with_reason) == fields | {"reason"}

    async def test_a_close_between_frames_is_no_message(self):
        assert await ipc.read_frame(_Reader(b"")) is None

    @pytest.mark.parametrize(
        "data",
        [
            b"\x00\x00",
            struct.pack(">I", 10) + b"{}",
            struct.pack(">I", 3) + b"{]}",
            struct.pack(">I", 3) + b"NaN",
            struct.pack(">I", 8) + b"Infinity",
            struct.pack(">I", 200_000) + b"[" * 100_000 + b"]" * 100_000,
            struct.pack(">I", 2 * (ipc.MAX_FRAME_DEPTH + 1))
            + b"[" * (ipc.MAX_FRAME_DEPTH + 1)
            + b"]" * (ipc.MAX_FRAME_DEPTH + 1),
            struct.pack(">I", ipc.MAX_FRAME_BYTES + 1),
        ],
        ids=[
            "header cut",
            "body cut",
            "not json",
            "NaN",
            "Infinity",
            "too deep",
            "past the depth bound",
            "over the limit",
        ],
    )
    async def test_a_broken_frame_is_refused(self, data):
        with pytest.raises(ipc.FrameError):
            await ipc.read_frame(_Reader(data))

    @pytest.mark.parametrize(
        "message",
        [
            ["A"],
            {},
            {"names": "A"},
            {"names": [1]},
            {"names": [""]},
            {"names": ["A"] * (ipc.MAX_NAMES + 1)},
            {"names": ["A"], "value": 1},
            {"names": ["A"], "op": "write"},
            {"write": {"A": 1}},
        ],
    )
    def test_anything_but_a_bounded_read_is_refused(self, message):
        with pytest.raises(ipc.FrameError):
            ipc.decode_request(message)

    def test_exactly_the_cap_is_a_request(self):
        names = [f"PV{i}" for i in range(ipc.MAX_NAMES)]
        assert ipc.decode_request({"names": names}) == names

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.5, 1.5),
            (True, True),
            (None, None),
            ("text", "text"),
            (math.nan, "NaN"),
            (math.inf, "Infinity"),
            (-math.inf, "-Infinity"),
            ([1.0, math.nan], [1.0, "NaN"]),
            (b"bytes", "bytes"),
        ],
    )
    def test_every_value_is_strict_json(self, value, expected):
        assert ipc.wire_value(value) == expected
        json.dumps(ipc.wire_value(value), allow_nan=False)

    def test_an_array_value_is_a_list(self):
        np = pytest.importorskip("numpy")
        assert ipc.wire_value(np.array([1.0, math.nan])) == [1.0, "NaN"]
        assert ipc.wire_value(np.float64(2.5)) == 2.5


# -- in process, on a fake connector --------------------------------------------


class _ScriptedFake(FakeConnector):
    """A fake with an alarm on one channel, one that hangs, and odd readings."""

    instances: list[_ScriptedFake] = []
    writes = 0
    rendezvous: threading.Barrier | None = None

    def __init__(self) -> None:
        super().__init__()
        type(self).instances.append(self)

    def _read_value(self, channel_address: str) -> ChannelValue:
        reading = super()._read_value(channel_address)
        if channel_address == "ALARMED":
            reading.metadata = ChannelMetadata(alarm_severity=2, alarm_status="HIHI")
        return reading

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        if channel_address == "HANGS":
            await asyncio.Event().wait()
        if channel_address.startswith("TOGETHER"):
            # A PVA read's shape: a worker thread for the whole read.
            await asyncio.to_thread(type(self).rendezvous.wait)
        if channel_address == "HUGE":
            return ChannelValue(value=[0.0] * 100_000, timestamp=datetime.now(UTC))
        if channel_address == "NO_TIMESTAMP":
            reading = await super().read_channel(channel_address, timeout)
            reading.timestamp = None
            return reading
        return await super().read_channel(channel_address, timeout)

    async def write_channel(self, *args: Any, **kwargs: Any):
        type(self).writes += 1
        return await super().write_channel(*args, **kwargs)


@pytest.fixture
def session_env(tmp_path, monkeypatch) -> Iterator[Path]:
    """A session's environment, the record bound to the VA, and the fake behind it."""
    state_dir = tmp_path / "control_target"
    state_dir.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state_dir))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    for name in list(os.environ):
        if name.startswith(("OSPREY_CONTROL_TARGET", "EPICS_")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)

    def get_config_value(path, default=None, _config_path=None):
        return VA_SECTION if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    monkeypatch.setattr(connector_base, "_long_lived_session", False)
    # The block and the watcher are process-wide; each has its own test below.
    monkeypatch.setattr(child_module, "_arm_raw_put_block", lambda: None)
    monkeypatch.setattr(child_module, "_start_parent_watcher", lambda *_: None)
    monkeypatch.setattr(_ScriptedFake, "instances", [])
    monkeypatch.setattr(_ScriptedFake, "writes", 0)
    _write_record(state_dir)
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _ScriptedFake)
        yield state_dir
    control_context.invalidate_cache()


class _Client:
    """One connection to a child on its socket."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader, self.writer = reader, writer

    async def read(self, names: list[str]) -> list[ValueRow]:
        self.writer.write(ipc.encode_request(names))
        await self.writer.drain()
        message = await asyncio.wait_for(ipc.read_frame(self.reader), 10.0)
        return ipc.decode_response(message)

    async def send_raw(self, message: Any) -> Any:
        self.writer.write(ipc.encode_frame(message))
        await self.writer.drain()
        return await asyncio.wait_for(ipc.read_frame(self.reader), 10.0)


@pytest.fixture
async def served(session_env, listener):  # noqa: ARG001 - the session needs its environment
    """The child's ``serve`` running as a task, and a client connected to it."""
    task = asyncio.create_task(child_module.serve(listener))
    reader, writer = await asyncio.open_unix_connection(listener.getsockname())
    client = _Client(reader, writer)
    await client.read([])  # answered only once the session is open
    yield task, client
    writer.close()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


class TestReads:
    async def test_a_batch_reads_through_the_connector_in_request_order(self, served):
        _task, client = served
        _ScriptedFake.instances[-1].set_value("A", 4.5)
        rows = await client.read(["A", "ALARMED"])
        assert [row.name for row in rows] == ["A", "ALARMED"]
        assert [row.outcome for row in rows] == [ipc.OUTCOME_FOUND, ipc.OUTCOME_FOUND]
        assert rows[0].value == 4.5
        assert rows[0].timestamp is not None
        assert (rows[1].severity, rows[1].status) == (2, "HIHI")

    async def test_a_channel_that_hangs_is_no_answer_within_the_cap(self, served, monkeypatch):
        _task, client = served
        monkeypatch.setattr(child_module, "PV_READ_BOUND_S", 0.05)
        started = time.monotonic()
        rows = await client.read(["A", "HANGS"])
        assert time.monotonic() - started < 2.0
        assert rows[0].outcome == ipc.OUTCOME_FOUND
        assert rows[1].outcome == ipc.OUTCOME_NO_ANSWER
        assert rows[1].value is None
        assert "no answer" in rows[1].reason

    def test_the_per_channel_cap_is_one_second(self):
        assert child_module.PV_READ_BOUND_S == 1.0

    async def test_a_channel_that_is_down_is_no_answer_and_the_rest_read(self, served):
        _task, client = served
        _ScriptedFake.instances[-1].drop("DOWN")
        rows = await client.read(["DOWN", "A"])
        assert rows[0].outcome == ipc.OUTCOME_NO_ANSWER
        assert "DOWN" in rows[0].reason
        assert rows[1].outcome == ipc.OUTCOME_FOUND

    async def test_a_value_over_the_bound_is_not_sent(self, served):
        _task, client = served
        rows = await client.read(["HUGE"])
        assert rows[0].outcome == ipc.OUTCOME_FOUND
        assert rows[0].value is None
        assert "too large" in rows[0].reason

    async def test_a_reading_that_cannot_be_shown_costs_only_its_own_row(self, served):
        _task, client = served
        rows = await client.read(["NO_TIMESTAMP", "A"])
        assert rows[0].outcome == ipc.OUTCOME_NO_ANSWER
        assert rows[0].reason
        assert rows[1].outcome == ipc.OUTCOME_FOUND

    @pytest.mark.usefixtures("session_env")
    async def test_a_session_with_no_connector_answers_unavailable(self):
        class _NoConnector:
            binding = session_module.Binding("live", 1, None, "Live", "unservable")

            async def read(self, _name):
                raise session_module.SessionUnservableError("nothing reachable")

        rows = await child_module.read_batch(_NoConnector(), ["A", "B"])
        assert [row.outcome for row in rows] == [ipc.OUTCOME_UNAVAILABLE] * 2
        assert rows[0].reason == "nothing reachable"

    async def test_a_connection_can_ask_more_than_once(self, served):
        _task, client = served
        assert len(await client.read(["A"])) == 1
        assert len(await client.read(["A", "B"])) == 2

    async def test_every_row_of_a_full_batch_has_a_worker_at_once(self, served, monkeypatch):
        """Each read waits for all the others: a smaller pool leaves them all no answer."""
        monkeypatch.setattr(
            _ScriptedFake, "rendezvous", threading.Barrier(ipc.MAX_NAMES, timeout=5.0)
        )
        _task, client = served
        rows = await client.read([f"TOGETHER:{i}" for i in range(ipc.MAX_NAMES)])
        assert {row.outcome for row in rows} == {ipc.OUTCOME_FOUND}

    async def test_a_batch_at_the_cap_reads_every_name(self, served):
        _task, client = served
        names = [f"PV{i}" for i in range(ipc.MAX_NAMES)]
        rows = await client.read(names)
        assert [row.name for row in rows] == names


class TestNoWriteOverIpc:
    @pytest.mark.parametrize(
        "message",
        [
            {"names": ["A"], "value": 1},
            {"op": "write", "name": "A", "value": 1},
            {"write": {"A": 1}},
            {"names": ["A"] * (ipc.MAX_NAMES + 1)},
        ],
    )
    async def test_a_request_that_is_not_a_read_closes_the_connection(self, served, message):
        _task, client = served
        before = _ScriptedFake.instances[-1]._state.copy()
        assert await client.send_raw(message) is None
        assert _ScriptedFake.writes == 0
        assert _ScriptedFake.instances[-1]._state == before

    async def test_a_refused_client_does_not_end_the_child(self, served, listener):
        task, client = served
        assert await client.send_raw({"write": {"A": 1}}) is None
        reader, writer = await asyncio.open_unix_connection(listener.getsockname())
        assert len(await _Client(reader, writer).read(["A"])) == 1
        writer.close()
        assert not task.done()

    def test_the_child_touches_only_the_reading_half_of_a_session(self):
        import ast

        tree = ast.parse(Path(child_module.__file__).read_text(encoding="utf-8"))
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "session"
        }
        assert used == {"binding", "read", "wait_moved", "wait_stale"}


class TestAudit:
    async def test_each_batch_records_actor_target_and_count(self, served, monkeypatch):
        bodies = capture_audit_bodies(monkeypatch, "channels")
        _task, client = served
        await client.read(["A", "B", "C"])
        assert len(bodies) == 1
        body = bodies[0]
        assert body["actor"] == "test-operator"
        assert body["subject"] == child_module.READ_SUBJECT
        assert body["decision"] == "allowed"
        assert body["detail"] == "target=va pv_count=3"

    async def test_two_batches_are_two_records(self, served, monkeypatch):
        bodies = capture_audit_bodies(monkeypatch, "channels")
        _task, client = served
        await client.read(["A"])
        await client.read(["B"])
        assert len(bodies) == 2

    async def test_a_refused_request_reads_nothing_and_records_nothing(self, served, monkeypatch):
        bodies = capture_audit_bodies(monkeypatch, "channels")
        _task, client = served
        assert await client.send_raw({"names": ["A"] * (ipc.MAX_NAMES + 1)}) is None
        assert bodies == []


def test_arming_installs_the_armed_block_and_refuses_rpc(monkeypatch):
    from osprey.runtime import raw_put_block

    installed: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        raw_put_block, "install", lambda mode, **contract: installed.append((mode, contract))
    )
    child_module._arm_raw_put_block()
    assert [mode for mode, _ in installed] == ["armed"]
    contract = installed[0][1]
    assert contract["refuse_rpc"] is True
    assert ("tango.DeviceProxy", ("command_inout", "command_inout_asynch")) in (
        contract["rpc_targets"]
    )
    blocked = {(dotted, attr) for dotted, attrs in contract["blocked_targets"] for attr in attrs}
    assert set(_ARMED_CHECKED) <= blocked
    assert contract["ca_provider_targets"] == ()


#: Every PVAccess client an armed process could hold, shaped as the real ones are.
_FAKE_P4P_CONTEXT = """
class Context:
    def put(self, name, value):
        return "reached the network"

    def get(self, name):
        return "read"

    def rpc(self, name, value):
        return "reached the network"
"""

_FAKE_P4P_ASYNC_CONTEXT = """
class Context:
    async def put(self, name, value):
        return "reached the network"

    async def get(self, name):
        return "read"

    async def rpc(self, name, value):
        return "reached the network"
"""

_FAKE_PVACCESS = """
PVA = "pva"
CA = "ca"


class Channel:
    def __init__(self, name, provider=PVA):
        self.name = name

    def get(self):
        return "read"

    def put(self, value):
        return "reached the network"

    def putGet(self, value):
        return "reached the network"

    def asyncPut(self, value):
        return "reached the network"

    def parsePut(self, value):
        return "reached the network"

    def parsePutGet(self, value):
        return "reached the network"

    def putDouble(self, value):
        return "reached the network"
"""

#: Arms the child's block, then imports each client and calls one verb on it.
_ARMED_PROBE = """
import asyncio, importlib, inspect, json, sys

from osprey.interfaces.channel_finder import directory_values_child as child

child._arm_raw_put_block()


def call(owner, verb):
    if owner == "pvaccess.Channel":
        import pvaccess
        instance = pvaccess.Channel("A:B", getattr(pvaccess, sys.argv[1]))
    else:
        instance = importlib.import_module(owner.rsplit(".", 1)[0]).Context()
    args = ("A:B",) if verb in ("put", "get", "rpc") and owner != "pvaccess.Channel" else ()
    args += (1.0,) if verb != "get" else ()
    try:
        result = getattr(instance, verb)(*args)
        return asyncio.run(result) if inspect.isawaitable(result) else result
    except Exception as exc:
        return type(exc).__name__


contexts = ["p4p.client." + kind + ".Context" for kind in ("raw", "thread", "asyncio", "cothread")]
verbs = ["put", "putGet", "asyncPut", "parsePut", "parsePutGet", "putDouble", "get"]
answers = {f"{owner}.{verb}": call(owner, verb) for owner in contexts for verb in ("put", "get", "rpc")}
answers.update({f"pvaccess.Channel.{verb}": call("pvaccess.Channel", verb) for verb in verbs})
print("ANSWERS " + json.dumps(answers))
"""


def _fake_pvaccess_clients(root: Path) -> None:
    """The client packages, on a path ahead of any real one, for a process to import."""
    client = root / "p4p" / "client"
    client.mkdir(parents=True)
    (root / "p4p" / "__init__.py").write_text("", encoding="utf-8")
    (client / "__init__.py").write_text("", encoding="utf-8")
    for kind in ("raw", "thread", "cothread"):
        (client / f"{kind}.py").write_text(_FAKE_P4P_CONTEXT, encoding="utf-8")
    (client / "asyncio.py").write_text(_FAKE_P4P_ASYNC_CONTEXT, encoding="utf-8")
    (root / "pvaccess.py").write_text(_FAKE_PVACCESS, encoding="utf-8")


def _armed_answers(tmp_path: Path, provider: str) -> dict[str, Any]:
    """What each client entry point did in a process the child's own arming ran in."""
    _fake_pvaccess_clients(tmp_path)
    (tmp_path / "probe.py").write_text(_ARMED_PROBE, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
    completed = subprocess.run(
        [sys.executable, str(tmp_path / "probe.py"), provider],
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )
    assert completed.returncode == 0, completed.stderr
    line = next(ln for ln in completed.stdout.splitlines() if ln.startswith("ANSWERS "))
    return json.loads(line.removeprefix("ANSWERS "))


class TestArmedBlockRefusesPvaccessPuts:
    """Each runs in a process of its own: the block cannot be taken off again."""

    @pytest.mark.parametrize("provider", ["PVA", "CA"])
    def test_every_pvaccess_put_is_refused_and_reads_still_pass(self, tmp_path, provider):
        answers = _armed_answers(tmp_path, provider)
        for dotted, attr in _ARMED_CHECKED:
            assert answers[f"{dotted}.{attr}"] == "ChannelWriteBlockedError", (dotted, attr)
        # The pvaPy sweep's typed setter, which no table row names.
        assert answers["pvaccess.Channel.putDouble"] == "ChannelWriteBlockedError"
        for kind in ("raw", "thread", "asyncio", "cothread"):
            assert answers[f"p4p.client.{kind}.Context.rpc"] == "RuntimeError"
            assert answers[f"p4p.client.{kind}.Context.get"] == "read"
        assert answers["pvaccess.Channel.get"] == "read"

    def test_the_real_p4p_client_refuses_a_put_and_still_reads(self, tmp_path):
        pytest.importorskip("p4p.client.thread")
        probe = tmp_path / "probe.py"
        probe.write_text(
            textwrap.dedent(
                """
                from osprey.interfaces.channel_finder import directory_values_child as child
                child._arm_raw_put_block()
                from p4p.client.thread import Context
                context = Context("pva")
                for call in (
                    lambda: context.put("NO:SUCH:PV", 1.0, timeout=0.2),
                    lambda: context.rpc("NO:SUCH:PV", None, timeout=0.2),
                    lambda: context.get("NO:SUCH:PV", timeout=0.2),
                ):
                    try:
                        call()
                    except Exception as exc:
                        print("ANSWER", type(exc).__name__)
                context.close()
                """
            ),
            encoding="utf-8",
        )
        env = {**os.environ, "EPICS_PVA_ADDR_LIST": "127.0.0.1", "EPICS_PVA_AUTO_ADDR_LIST": "NO"}
        completed = subprocess.run(
            [sys.executable, str(probe)],
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
        answers = [
            line.split()[1] for line in completed.stdout.splitlines() if line.startswith("ANSWER ")
        ]
        assert answers == ["ChannelWriteBlockedError", "RuntimeError", "TimeoutError"], (
            completed.stderr
        )


@pytest.mark.usefixtures("session_env")
class TestStart:
    async def test_the_block_is_armed_before_the_session_opens(self, listener, monkeypatch):
        order: list[str] = []
        real_open = session_module.ControlSession.open.__func__

        async def spying_open(cls, **kwargs):
            order.append("open")
            assert kwargs["surface"] == "channels"
            return await real_open(cls, **kwargs)

        monkeypatch.setattr(child_module, "_arm_raw_put_block", lambda: order.append("arm"))
        monkeypatch.setattr(child_module, "_start_parent_watcher", lambda *_: order.append("watch"))
        monkeypatch.setattr(session_module.ControlSession, "open", classmethod(spying_open))
        task = asyncio.create_task(child_module.serve(listener))
        reader, writer = await asyncio.open_unix_connection(listener.getsockname())
        await _Client(reader, writer).read([])
        writer.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert order == ["watch", "arm", "open"]

    async def test_a_block_that_cannot_be_armed_refuses_the_start(
        self, listener, monkeypatch, capsys
    ):
        def fail():
            raise RuntimeError("no table")

        monkeypatch.setattr(child_module, "_arm_raw_put_block", fail)
        assert await child_module.serve(listener) == EXIT_START_REFUSED
        assert "no table" in capsys.readouterr().err

    async def test_a_forbidden_variable_refuses_the_start(self, listener, monkeypatch, capsys):
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "sandbox")
        opened: list[bool] = []
        monkeypatch.setattr(session_module.ControlSession, "open", lambda **_: opened.append(True))
        assert await child_module.serve(listener) == EXIT_START_REFUSED
        err = capsys.readouterr().err
        assert err.startswith(ipc.START_REFUSED_PREFIX)
        assert posture_store.LAUNCH_POSTURE_ENV_VAR in err
        assert opened == []


class TestLeaving:
    async def test_a_moved_record_exits_75(self, served, session_env):
        task, _client = served
        _write_record(session_env, generation=4)
        assert await asyncio.wait_for(task, 10.0) == EXIT_BINDING_MOVED

    async def test_a_move_leaves_without_closing_the_session(
        self, served, session_env, monkeypatch
    ):
        """Closing first would answer a batch in that window with unavailable rows."""
        closes: list[Any] = []

        async def close(session):
            closes.append(session)

        monkeypatch.setattr(session_module.ControlSession, "close", close)
        task, _client = served
        _write_record(session_env, generation=4)
        assert await asyncio.wait_for(asyncio.shield(task), 10.0) == EXIT_BINDING_MOVED
        assert closes == []

    async def test_a_stale_transport_exits_76(self, served):
        task, _client = served
        _ScriptedFake.instances[-1].go_stale()
        assert await asyncio.wait_for(task, 10.0) == EXIT_TRANSPORT_STALE

    @pytest.mark.usefixtures("session_env")
    async def test_a_watch_that_fails_is_a_move(self, listener, monkeypatch):
        async def broken(_self):
            raise OSError("record unreadable")

        monkeypatch.setattr(session_module.ControlSession, "wait_moved", broken)
        assert await asyncio.wait_for(child_module.serve(listener), 10.0) == EXIT_BINDING_MOVED

    @pytest.mark.usefixtures("session_env")
    async def test_a_stale_probe_that_fails_ends_nothing(self, listener, monkeypatch):
        async def broken(_self):
            raise OSError("probe failed")

        monkeypatch.setattr(session_module.ControlSession, "wait_stale", broken)
        task = asyncio.create_task(child_module.serve(listener))
        reader, writer = await asyncio.open_unix_connection(listener.getsockname())
        client = _Client(reader, writer)
        assert await client.read([]) == []
        await asyncio.sleep(0.05)
        assert len(await client.read(["A"])) == 1
        assert not task.done()
        writer.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def test_the_exit_statuses_are_the_contract_numbers(self):
        assert (EXIT_BINDING_MOVED, EXIT_TRANSPORT_STALE, EXIT_START_REFUSED) == (75, 76, 78)


# -- the real EPICS connector, on a fake aioca ------------------------------------


class _DeadlineAioca(FakeAioca):
    """A connect to a channel nobody answers for takes its whole deadline, as aioca's does."""

    async def connect(self, name, *, timeout=None, **kwargs):
        if self._channel(name) is None:
            await asyncio.sleep(max(0.0, timeout[0] - time.time()))
        return await super().connect(name, timeout=timeout, **kwargs)


#: A deployment whose live machine is EPICS behind a name server, the shape whose
#: transport can go stale; its calls would take the connector's 5 s default.
EPICS_SECTION = {
    "type": types.EPICS,
    "connector": {
        types.EPICS: {
            "gateways": {"read_only": {"address": "gw", "port": 5075, "use_name_server": True}}
        }
    },
}


@pytest.fixture
def dead_aioca(session_env, monkeypatch) -> FakeAioca:
    """The session bound to a real ``EPICSConnector`` on a fake aioca, with short windows."""
    _write_record(session_env, target="live", generation=1)
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda path, default=None, _config_path=None: (
            EPICS_SECTION if path == "control_system" else default
        ),
    )
    fake = _DeadlineAioca()
    monkeypatch.setitem(sys.modules, "aioca", fake)
    monkeypatch.setattr(epics_connector_module, "get_ca_loop", InlineCaLoop)
    monkeypatch.setattr(epics_connector_module, "STALE_PROBE_INTERVAL_S", 0.01)
    monkeypatch.setattr(epics_connector_module, "STALE_SILENT_S", 0.05)
    # Far below the connector's own default, as the real 1 s is: a child that only
    # cancels at this bound never lets the connector see its connect fail.
    monkeypatch.setattr(child_module, "PV_READ_BOUND_S", 0.1)
    return fake


class TestStaleOnTheEpicsConnector:
    async def test_a_channel_that_drops_ends_the_child_with_76(self, dead_aioca, listener):
        channel = dead_aioca.channels["SR:CH"] = FakeChannel()
        task = asyncio.create_task(child_module.serve(listener))
        reader, writer = await asyncio.open_unix_connection(listener.getsockname())
        client = _Client(reader, writer)
        try:
            assert (await client.read(["SR:CH"]))[0].outcome == ipc.OUTCOME_FOUND
            channel.connected = False
            rows = await client.read(["SR:CH"])
            assert rows[0].outcome == ipc.OUTCOME_NO_ANSWER
            assert await asyncio.wait_for(asyncio.shield(task), 10.0) == EXIT_TRANSPORT_STALE
        finally:
            writer.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.usefixtures("dead_aioca")
    async def test_the_connector_is_bound_to_the_channel_cap(self, listener, monkeypatch):
        opened: list[session_module.ControlSession] = []
        real_open = session_module.ControlSession.open.__func__

        async def spying_open(cls, **kwargs):
            opened.append(await real_open(cls, **kwargs))
            return opened[-1]

        monkeypatch.setattr(session_module.ControlSession, "open", classmethod(spying_open))
        task = asyncio.create_task(child_module.serve(listener))
        reader, writer = await asyncio.open_unix_connection(listener.getsockname())
        try:
            await _Client(reader, writer).read([])
            assert opened[0]._connector._timeout == child_module.PV_READ_BOUND_S
        finally:
            writer.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


# -- the read gateway, whatever the switch says -----------------------------------


READ_GATEWAY = {"address": "read-gw", "port": 5064}
WRITE_GATEWAY = {"address": "write-gw", "port": 5084}


def _epics_section(**gateways: dict[str, Any]) -> dict[str, Any]:
    """A live EPICS machine whose write tier allows writes, so only the switch decides."""
    return {
        "type": types.EPICS,
        "writes": "allowed",
        "connector": {types.EPICS: {"gateways": gateways}},
    }


def _set_switch(state_dir: Path, position: str) -> None:
    _write_record(state_dir, target="live", generation=1)
    wanted = {} if position == "off" else {"live": position}
    _set_switch_to(state_dir / control_context.RECORD_FILENAME, wanted)
    control_context.invalidate_cache()


class _Gateways:
    """The real ``EPICSConnector`` on a fake aioca, and what it pointed Channel Access at."""

    def __init__(self, monkeypatch, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.monkeypatch = monkeypatch
        self.ca_loop = InlineCaLoop()
        self.aioca = FakeAioca()
        self.opened: list[session_module.ControlSession] = []
        self.aioca.channels["SR:CH"] = FakeChannel()
        monkeypatch.setitem(sys.modules, "aioca", self.aioca)
        monkeypatch.setattr(epics_connector_module, "get_ca_loop", lambda: self.ca_loop)
        real_open = session_module.ControlSession.open.__func__

        async def spying_open(cls, **kwargs):
            self.opened.append(await real_open(cls, **kwargs))
            return self.opened[-1]

        monkeypatch.setattr(session_module.ControlSession, "open", classmethod(spying_open))

    def serve_section(self, section: dict[str, Any]) -> None:
        self.monkeypatch.setattr(
            "osprey_connectors.config.get_config_value",
            lambda path, default=None, _config_path=None: (
                section if path == "control_system" else default
            ),
        )

    @property
    def session(self) -> session_module.ControlSession:
        return self.opened[-1]

    @property
    def addresses(self) -> list[Any]:
        return [env.get("EPICS_CA_ADDR_LIST") for env in self.ca_loop.pointed_at]


@pytest.fixture
def gateways(session_env, monkeypatch) -> _Gateways:
    return _Gateways(monkeypatch, session_env)


@contextlib.asynccontextmanager
async def _serving(listener: socket.socket):
    """``serve`` as a task, and a client whose first answer says the session is open."""
    task = asyncio.create_task(child_module.serve(listener))
    reader, writer = await asyncio.open_unix_connection(listener.getsockname())
    client = _Client(reader, writer)
    try:
        await client.read([])
        yield client
    finally:
        writer.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class TestReadGatewayOnly:
    """Ruling 21: read-only whatever the operator's switch says."""

    @pytest.mark.parametrize("position", ["off", "asked", "allowed"])
    async def test_every_switch_position_selects_the_read_gateway(
        self, gateways, listener, position
    ):
        gateways.serve_section(_epics_section(read_only=READ_GATEWAY, write_access=WRITE_GATEWAY))
        _set_switch(gateways.state_dir, position)
        async with _serving(listener) as client:
            connector = gateways.session.connector
            assert connector._selected_role == "read_only"
            assert gateways.addresses == ["read-gw"]
            assert is_readonly_run()
            writes = gateways.session.writes
            assert writes["allowed"] is False
            assert writes["reason"] == "readonly_run"
            assert (await client.read(["SR:CH"]))[0].outcome == ipc.OUTCOME_FOUND

    @pytest.mark.parametrize("position", ["asked", "allowed"])
    async def test_an_open_switch_would_select_the_write_gateway_without_the_child(
        self, gateways, position
    ):
        """What the test above would see if the child stopped declaring its run."""
        gateways.serve_section(_epics_section(read_only=READ_GATEWAY, write_access=WRITE_GATEWAY))
        _set_switch(gateways.state_dir, position)
        session = await session_module.ControlSession.open(surface=child_module.SURFACE)
        assert session.connector._selected_role == "write_access"
        assert gateways.addresses == ["write-gw"]
        await session.close()

    @pytest.mark.parametrize(
        ("before", "after"), [("off", "asked"), ("asked", "off"), ("asked", "allowed")]
    )
    async def test_a_switch_flip_is_not_a_move(self, gateways, listener, before, after):
        gateways.serve_section(_epics_section(read_only=READ_GATEWAY, write_access=WRITE_GATEWAY))
        _set_switch(gateways.state_dir, before)
        async with _serving(listener):
            _set_switch_to(
                gateways.state_dir / control_context.RECORD_FILENAME,
                {} if after == "off" else {"live": after},
            )
            control_context.invalidate_cache()
            assert gateways.session.moved() is False

    @pytest.mark.parametrize("position", ["off", "asked", "allowed"])
    async def test_a_block_with_only_a_write_gateway_is_never_connected(
        self, gateways, listener, position
    ):
        """It cannot read through the read gateway, so it does not read at all."""
        gateways.serve_section(_epics_section(write_access=WRITE_GATEWAY))
        _set_switch(gateways.state_dir, position)
        async with _serving(listener) as client:
            assert gateways.session.connector is None
            assert gateways.session.binding.state == session_module.STATE_UNSERVABLE
            assert gateways.ca_loop.pointed_at == []
            assert (await client.read(["SR:CH"]))[0].outcome == ipc.OUTCOME_UNAVAILABLE

    async def test_the_audit_record_reads_as_a_process_that_cannot_write(
        self, gateways, listener, monkeypatch
    ):
        gateways.serve_section(_epics_section(read_only=READ_GATEWAY, write_access=WRITE_GATEWAY))
        _set_switch(gateways.state_dir, "allowed")
        bodies = capture_audit_bodies(monkeypatch, "channels")
        async with _serving(listener) as client:
            await client.read(["SR:CH"])
        assert [body["posture"] for body in bodies] == ["sandbox"]
        assert os.environ[POSTURE_ENV_VAR] == SANDBOX_MODE


# -- a real child -----------------------------------------------------------------


def _child_env(tmp_path: Path, **environment: str | None) -> tuple[dict[str, str], Path]:
    state_dir = tmp_path / "control_target"
    state_dir.mkdir(exist_ok=True)
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OSPREY_", "EPICS_")) and name != "CONFIG_FILE"
    }
    env.update(
        {
            posture_store.CONTROL_CONTEXT_DIR_ENV_VAR: str(state_dir),
            AUDIT_IDENTITY_ENV: "test-operator",
            "OSPREY_CONFIG": str(config),
            "CONFIG_FILE": str(config),
        }
    )
    for name, value in environment.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env, state_dir


def _child_argv(fd: int) -> list[str]:
    return [sys.executable, "-m", ipc.CHILD_MODULE, ipc.SOCKET_FD_ARG, str(fd)]


def _start_child(listener: socket.socket, env: dict[str, str], log: Path) -> subprocess.Popen:
    fd = listener.fileno()
    with log.open("wb") as output:
        return subprocess.Popen(
            _child_argv(fd),
            env=env,
            pass_fds=(fd,),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )


def _ask(connection: socket.socket, names: list[str]) -> list[ValueRow]:
    """One blocking batch on *connection*."""
    connection.sendall(ipc.encode_request(names))
    length = struct.unpack(">I", _receive(connection, 4))[0]
    return ipc.decode_response(json.loads(_receive(connection, length)))


def _receive(connection: socket.socket, count: int) -> bytes:
    data = b""
    while len(data) < count:
        chunk = connection.recv(count - len(data))
        if not chunk:
            raise AssertionError("the child closed the connection")
        data += chunk
    return data


def _connect(listener: socket.socket) -> socket.socket:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(CHILD_TIMEOUT_S)
    connection.connect(listener.getsockname())
    return connection


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=30)


class TestRealChild:
    def test_a_forbidden_variable_exits_78_with_its_reason(self, tmp_path, listener):
        env, _ = _child_env(tmp_path, OSPREY_LAUNCH_POSTURE="sandbox")
        log = tmp_path / "child.log"
        process = _start_child(listener, env, log)
        try:
            assert process.wait(timeout=CHILD_TIMEOUT_S) == EXIT_START_REFUSED
        finally:
            _stop(process)
        last_line = log.read_text(encoding="utf-8").strip().splitlines()[-1]
        assert last_line.startswith(ipc.START_REFUSED_PREFIX)
        assert "OSPREY_LAUNCH_POSTURE" in last_line

    def test_a_config_mismatch_exits_78(self, tmp_path, listener):
        env, _ = _child_env(tmp_path, CONFIG_FILE=str(tmp_path / "other.yml"))
        process = _start_child(listener, env, tmp_path / "child.log")
        try:
            assert process.wait(timeout=CHILD_TIMEOUT_S) == EXIT_START_REFUSED
        finally:
            _stop(process)

    def test_a_move_exits_75_and_the_next_child_serves_the_same_socket(self, tmp_path, listener):
        env, state_dir = _child_env(tmp_path)
        _write_record(state_dir, target="live", generation=1)
        first = _start_child(listener, env, tmp_path / "first.log")
        second = None
        try:
            with _connect(listener) as connection:
                rows = _ask(connection, ["A"])
            # An empty config cannot derive a connector: no channel, and it says why.
            assert rows[0].outcome == ipc.OUTCOME_UNAVAILABLE
            _write_record(state_dir, target="live", generation=2)
            assert first.wait(timeout=CHILD_TIMEOUT_S) == EXIT_BINDING_MOVED
            assert Path(listener.getsockname()).exists()

            second = _start_child(listener, env, tmp_path / "second.log")
            with _connect(listener) as connection:
                assert _ask(connection, []) == []
        finally:
            _stop(first)
            if second is not None:
                _stop(second)

    def test_a_pending_executor_job_does_not_hold_the_exit(self, tmp_path, listener):
        """An abandoned PVA read sits in the default executor: the exit must not join it."""
        env, _ = _child_env(tmp_path)
        fd = listener.fileno()
        launcher = textwrap.dedent(
            f"""
            import asyncio, time
            from osprey.interfaces.channel_finder import directory_values_child as child

            async def serve(_listener):
                asyncio.get_running_loop().run_in_executor(None, time.sleep, 600)
                await asyncio.sleep(0)
                return {EXIT_BINDING_MOVED}

            child.serve = serve
            child.main([{ipc.SOCKET_FD_ARG!r}, "{fd}"])
            """
        )
        process = subprocess.Popen(
            [sys.executable, "-c", launcher],
            env=env,
            pass_fds=(fd,),
            stdin=subprocess.DEVNULL,
        )
        try:
            assert process.wait(timeout=CHILD_TIMEOUT_S) == EXIT_BINDING_MOVED
        finally:
            _stop(process)

    def test_a_child_leaves_when_its_parent_is_killed(self, tmp_path, listener):
        """No stop call: only the parent's death can end it."""
        env, state_dir = _child_env(tmp_path)
        _write_record(state_dir, target="live", generation=1)
        fd = listener.fileno()
        log = tmp_path / "child.log"
        launcher = textwrap.dedent(
            f"""
            import subprocess, sys, time
            with open({str(log)!r}, "wb") as output:
                child = subprocess.Popen(
                    {_child_argv(fd)!r},
                    pass_fds=({fd},),
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            print(child.pid, flush=True)
            time.sleep(600)
            """
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", launcher],
            env=env,
            pass_fds=(fd,),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        child_pid = int(parent.stdout.readline())
        try:
            with _connect(listener) as connection:
                assert _ask(connection, []) == []
                parent.send_signal(signal.SIGKILL)
                parent.wait(timeout=30)
                # The accepted connection lives only in the child: it ends with it.
                connection.settimeout(ORPHAN_BOUND_S)
                started = time.monotonic()
                assert connection.recv(1) == b"", log.read_text(encoding="utf-8")
            assert time.monotonic() - started < ORPHAN_BOUND_S
        finally:
            _stop(parent)
            parent.stdout.close()
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
