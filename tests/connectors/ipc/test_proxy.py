"""Parent-side proxy connector: what a tool sees when the connector is a child process.

These tests drive the real :class:`ConnectorHostProxy` against a scripted fake
child that speaks the frame codec directly, over a socketpair. That keeps the
process boundary honest — every assertion below travels through real encoding,
a real byte stream, and real demultiplexing — while staying deterministic: the
fake child answers exactly when, and in whatever order, the test says.

Assertions are on concrete payloads (the value that came back, the fields on
the re-raised exception, the exact frames the child observed), never merely
that a call "didn't raise".
"""

import asyncio
import inspect
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
)
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from osprey_connectors.ipc import frames
from osprey_connectors.ipc.proxy import CHILD, ConnectorHostProxy

TS = datetime(2026, 8, 22, 14, 30, 5, 123456, tzinfo=UTC)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _channel_value(value, units="A"):
    return ChannelValue(value=value, timestamp=TS, metadata=ChannelMetadata(units=units))


# ---------------------------------------------------------------- fake child


class _FakeChild:
    """The other end of the pipe, scripted by the test.

    ``handler`` is an async callable ``(child, request_frame)``; it decides what
    (if anything) comes back, so a test can answer out of order, answer late,
    answer with a typed error, or die without answering at all.
    """

    def __init__(self, reader, writer, handler):
        self._reader = reader
        self._writer = writer
        self._handler = handler
        self.requests: list[frames.RequestFrame] = []
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._serve())

    async def _serve(self):
        stream = frames.FrameReader()
        while True:
            chunk = await self._reader.read(65536)
            if not chunk:
                return
            for frame in stream.feed(chunk):
                self.requests.append(frame)
                await self._handler(self, frame)

    def send_result(self, request_id, value):
        self._writer.write(frames.encode_result(request_id, value))

    def send_error(self, request_id, exc):
        self._writer.write(frames.encode_error(request_id, exc))

    async def die(self):
        """Close the pipe without answering — the child crashing, from outside."""
        self._writer.close()
        if self._task is not None:
            self._task.cancel()

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
        self._writer.close()


async def _proxy_with_child(handler, **proxy_kwargs):
    """A live proxy wired to a scripted fake child over a real socketpair."""
    parent_sock, child_sock = socket.socketpair()
    parent_reader, parent_writer = await asyncio.open_connection(sock=parent_sock)
    child_reader, child_writer = await asyncio.open_connection(sock=child_sock)

    child = _FakeChild(child_reader, child_writer, handler)
    child.start()
    proxy = ConnectorHostProxy(parent_reader, parent_writer, **proxy_kwargs)
    return proxy, child


def _echo_handler(values):
    """Answer each request by popping the next scripted value, in arrival order."""
    queue = list(values)

    async def handler(child, frame):
        child.send_result(frame.request_id, queue.pop(0))

    return handler


@pytest.fixture
async def teardown():
    """Close whatever the test opened, even when it asserted its way out."""
    items = []
    yield items
    for proxy, child in items:
        # Short ack: most fake children are not scripted to answer a disconnect,
        # and waiting the real default out in every teardown costs seconds.
        await proxy.disconnect(ack_timeout=0.05)
        await child.stop()


# ---------------------------------------------------------------- happy path


async def test_read_channel_round_trips_a_real_channel_value(teardown):
    proxy, child = await _proxy_with_child(_echo_handler([_channel_value(1.25)]))
    teardown.append((proxy, child))

    result = await proxy.read_channel("SR:BEND:1:CUR")

    assert isinstance(result, ChannelValue)
    assert result.value == 1.25
    assert result.timestamp == TS
    assert result.metadata.units == "A"
    assert len(child.requests) == 1
    assert child.requests[0].method == "read_channel"
    assert child.requests[0].kwargs == {"channel_address": "SR:BEND:1:CUR", "timeout": None}


async def test_ndarray_payload_decodes_to_numpy_with_dtype_and_shape(teardown):
    waveform = np.arange(12, dtype=np.float32).reshape(3, 4)
    proxy, child = await _proxy_with_child(_echo_handler([_channel_value(waveform)]))
    teardown.append((proxy, child))

    result = await proxy.read_channel("SR:BPM:1:WAVE")

    assert isinstance(result.value, np.ndarray)
    assert result.value.dtype == np.float32
    assert result.value.shape == (3, 4)
    assert np.array_equal(result.value, waveform)


async def test_read_multiple_channels_issues_exactly_one_request(teardown):
    addresses = ["SR:A", "SR:B", "SR:C", "SR:D"]
    batch = {address: _channel_value(index) for index, address in enumerate(addresses)}
    proxy, child = await _proxy_with_child(_echo_handler([batch]))
    teardown.append((proxy, child))

    result = await proxy.read_multiple_channels(addresses, timeout=2.0)

    assert len(child.requests) == 1, "a batched read must not fan out into per-channel frames"
    assert child.requests[0].method == "read_multiple_channels"
    assert child.requests[0].kwargs == {"channel_addresses": addresses, "timeout": 2.0}
    assert sorted(result) == addresses
    assert [result[address].value for address in addresses] == [0, 1, 2, 3]


async def test_write_channel_round_trips_a_write_result(teardown):
    written = ChannelWriteResult(channel_address="SR:BEND:1:SP", value_written=2.5, success=True)
    proxy, child = await _proxy_with_child(_echo_handler([written]))
    teardown.append((proxy, child))

    result = await proxy.write_channel("SR:BEND:1:SP", 2.5, verification_level="readback")

    assert isinstance(result, ChannelWriteResult)
    assert (result.channel_address, result.value_written, result.success) == (
        "SR:BEND:1:SP",
        2.5,
        True,
    )
    assert child.requests[0].method == "write_channel"
    assert child.requests[0].kwargs == {
        "channel_address": "SR:BEND:1:SP",
        "value": 2.5,
        "timeout": None,
        "verification_level": "readback",
    }


async def test_omitted_verification_keywords_never_reach_the_wire(teardown):
    written = ChannelWriteResult(channel_address="SR:X:SP", value_written=1, success=True)
    proxy, child = await _proxy_with_child(_echo_handler([written]))
    teardown.append((proxy, child))

    await proxy.write_channel("SR:X:SP", 1)

    # Omission is a sentinel, not a value: forwarding None would override the
    # child connector's own per-channel resolution.
    assert "verification_level" not in child.requests[0].kwargs
    assert "tolerance" not in child.requests[0].kwargs


async def test_write_multiple_channels_issues_one_batched_request(teardown):
    results = [
        ChannelWriteResult(channel_address="SR:A:SP", value_written=1, success=True),
        ChannelWriteResult(channel_address="SR:B:SP", value_written=2, success=True),
    ]
    proxy, child = await _proxy_with_child(_echo_handler([results]))
    teardown.append((proxy, child))

    returned = await proxy.write_multiple_channels([("SR:A:SP", 1), ("SR:B:SP", 2)])

    assert len(child.requests) == 1
    assert child.requests[0].method == "write_multiple_channels"
    # Tuples cross the wire as lists; the pairing is what matters.
    assert child.requests[0].kwargs["operations"] == [["SR:A:SP", 1], ["SR:B:SP", 2]]
    assert [item.channel_address for item in returned] == ["SR:A:SP", "SR:B:SP"]


async def test_out_of_order_responses_route_to_the_right_caller(teardown):
    held: list[frames.RequestFrame] = []

    async def handler(child, frame):
        held.append(frame)
        if len(held) == 2:
            # Answer in reverse arrival order — the demultiplexer must not care.
            child.send_result(held[1].request_id, _channel_value("second"))
            child.send_result(held[0].request_id, _channel_value("first"))

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    first, second = await asyncio.gather(
        proxy.read_channel("SR:FIRST"), proxy.read_channel("SR:SECOND")
    )

    assert first.value == "first"
    assert second.value == "second"


# ---------------------------------------------------------------- typed errors


async def test_limits_violation_re_raises_with_its_bounds_intact(teardown):
    original = ChannelLimitsViolationError(
        channel_address="SR:BEND:1:SP",
        value=12.0,
        violation_type="max",
        violation_reason="above configured maximum",
        min_value=0.0,
        max_value=10.0,
        max_step=1.0,
        current_value=9.5,
    )

    async def handler(child, frame):
        child.send_error(frame.request_id, original)

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    with pytest.raises(ChannelLimitsViolationError) as caught:
        await proxy.write_channel("SR:BEND:1:SP", 12.0)

    exc = caught.value
    assert exc.channel_address == "SR:BEND:1:SP"
    assert exc.attempted_value == 12.0
    assert exc.violation_type == "max"
    assert exc.violation_reason == "above configured maximum"
    assert (exc.min_value, exc.max_value, exc.max_step, exc.current_value) == (
        0.0,
        10.0,
        1.0,
        9.5,
    )


async def test_write_blocked_re_raises_with_its_reason(teardown):
    original = ChannelWriteBlockedError(
        "SR:BEND:1:SP", "WRITES_DISABLED", message="writes are disabled in this deployment"
    )

    async def handler(child, frame):
        child.send_error(frame.request_id, original)

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    with pytest.raises(ChannelWriteBlockedError) as caught:
        await proxy.write_channel("SR:BEND:1:SP", 1.0)

    assert caught.value.reason == "WRITES_DISABLED"
    assert caught.value.channel_address == "SR:BEND:1:SP"
    assert "writes are disabled in this deployment" in str(caught.value)


async def test_one_error_does_not_disturb_a_concurrent_request(teardown):
    held: list[frames.RequestFrame] = []

    async def handler(child, frame):
        held.append(frame)
        if len(held) == 2:
            child.send_error(held[0].request_id, ChannelWriteBlockedError("SR:A:SP", "LIMITS"))
            child.send_result(held[1].request_id, _channel_value(7))

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    failing = asyncio.ensure_future(proxy.write_channel("SR:A:SP", 1.0))
    reading = asyncio.ensure_future(proxy.read_channel("SR:B"))

    with pytest.raises(ChannelWriteBlockedError):
        await failing
    assert (await reading).value == 7


# ---------------------------------------------------------------- child death


async def test_child_death_mid_request_surfaces_as_connection_error(teardown):
    async def handler(child, frame):
        await child.die()

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    with pytest.raises(ConnectionError) as caught:
        await proxy.read_channel("SR:BEND:1:CUR")

    assert CHILD in str(caught.value)


async def test_every_outstanding_request_fails_when_the_child_dies(teardown):
    held: list[frames.RequestFrame] = []

    async def handler(child, frame):
        held.append(frame)
        if len(held) == 3:
            await child.die()

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    calls = [asyncio.ensure_future(proxy.read_channel(f"SR:{index}")) for index in range(3)]
    outcomes = await asyncio.gather(*calls, return_exceptions=True)

    assert len(outcomes) == 3
    for outcome in outcomes:
        assert isinstance(outcome, ConnectionError)
        assert CHILD in str(outcome)


async def test_calls_after_the_child_dies_raise_connection_error(teardown):
    async def handler(child, frame):
        await child.die()

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    with pytest.raises(ConnectionError):
        await proxy.read_channel("SR:FIRST")

    with pytest.raises(ConnectionError) as caught:
        await proxy.read_channel("SR:SECOND")
    assert CHILD in str(caught.value)

    with pytest.raises(ConnectionError):
        await proxy.read_multiple_channels(["SR:A", "SR:B"])


async def test_a_garbled_stream_kills_the_proxy_rather_than_hanging(teardown):
    async def handler(child, frame):
        child._writer.write(b"this is not a frame at all")

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    with pytest.raises(ConnectionError) as caught:
        await proxy.read_channel("SR:BEND:1:CUR")
    assert CHILD in str(caught.value)


# ---------------------------------------------------------------- timeouts


async def test_per_request_timeout_raises_timeout_error_and_leaves_the_proxy_usable(teardown):
    answer_now = False

    async def handler(child, frame):
        if answer_now:
            child.send_result(frame.request_id, _channel_value(3.5))

    proxy, child = await _proxy_with_child(handler, timeout_grace_s=0.0)
    teardown.append((proxy, child))

    with pytest.raises(TimeoutError) as caught:
        await proxy.read_channel("SR:SLOW", timeout=0.05)
    assert "read_channel" in str(caught.value)

    answer_now = True
    assert (await proxy.read_channel("SR:FAST", timeout=5.0)).value == 3.5


async def test_a_timeout_reported_by_the_child_is_raised_as_is(teardown):
    async def handler(child, frame):
        child.send_error(frame.request_id, TimeoutError("channel SR:GONE did not respond"))

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    with pytest.raises(TimeoutError) as caught:
        await proxy.read_channel("SR:GONE", timeout=5.0)
    assert "SR:GONE did not respond" in str(caught.value)


# ------------------------------------------------------- drain / refuse (3.1)


async def test_drain_returns_true_once_everything_completes(teardown):
    held: list[frames.RequestFrame] = []

    async def handler(child, frame):
        held.append(frame)

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    calls = [asyncio.ensure_future(proxy.read_channel(f"SR:{index}")) for index in range(2)]
    await asyncio.sleep(0.05)
    proxy.refuse_new_requests("switching to the live target")

    drain = asyncio.ensure_future(proxy.drain(2.0))
    await asyncio.sleep(0.01)
    for frame in held:
        child.send_result(frame.request_id, _channel_value(0))

    assert await drain is True
    assert [(await call).value for call in calls] == [0, 0]


async def test_drain_returns_false_when_a_request_outlives_the_deadline(teardown):
    async def handler(child, frame):
        pass  # never answers

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    call = asyncio.ensure_future(proxy.read_channel("SR:STUCK"))
    await asyncio.sleep(0.05)

    assert await proxy.drain(0.1) is False
    call.cancel()


async def test_drain_of_an_idle_proxy_is_immediately_true(teardown):
    proxy, child = await _proxy_with_child(_echo_handler([]))
    teardown.append((proxy, child))

    assert await proxy.drain(0.0) is True


async def test_refused_proxy_names_the_supplied_reason(teardown):
    proxy, child = await _proxy_with_child(_echo_handler([]))
    teardown.append((proxy, child))

    proxy.refuse_new_requests("target switch in progress: draining to the live gateway")

    with pytest.raises(ConnectionError) as caught:
        await proxy.read_channel("SR:BEND:1:CUR")
    assert "target switch in progress: draining to the live gateway" in str(caught.value)
    assert child.requests == []


async def test_outstanding_requests_survive_a_refusal(teardown):
    held: list[frames.RequestFrame] = []

    async def handler(child, frame):
        held.append(frame)

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    call = asyncio.ensure_future(proxy.read_channel("SR:INFLIGHT"))
    await asyncio.sleep(0.05)
    proxy.refuse_new_requests("draining")

    child.send_result(held[0].request_id, _channel_value(42))
    assert (await call).value == 42


# ---------------------------------------------------------------- disconnect


async def test_disconnect_sends_the_request_and_is_idempotent(teardown):
    async def handler(child, frame):
        if frame.method == "disconnect":
            child.send_result(frame.request_id, None)

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    await proxy.disconnect()
    await proxy.disconnect()  # never raises, never re-sends

    assert [frame.method for frame in child.requests] == ["disconnect"]

    with pytest.raises(ConnectionError):
        await proxy.read_channel("SR:BEND:1:CUR")


async def test_disconnect_never_raises_when_the_child_is_already_gone(teardown):
    async def handler(child, frame):
        await child.die()

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    with pytest.raises(ConnectionError):
        await proxy.read_channel("SR:BEND:1:CUR")

    await proxy.disconnect()  # the assertion is that this returns


async def test_disconnect_gives_up_on_a_silent_child_without_hanging(teardown):
    async def handler(child, frame):
        pass  # never acknowledges

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    await asyncio.wait_for(proxy.disconnect(ack_timeout=0.05), timeout=2.0)


async def test_disconnect_fails_outstanding_requests(teardown):
    async def handler(child, frame):
        pass

    proxy, child = await _proxy_with_child(handler)
    teardown.append((proxy, child))

    call = asyncio.ensure_future(proxy.read_channel("SR:INFLIGHT"))
    await asyncio.sleep(0.05)
    await proxy.disconnect(ack_timeout=0.05)

    with pytest.raises(ConnectionError):
        await call


# ---------------------------------------------------------------- contract


def test_the_proxy_mirrors_the_connector_call_surface():
    """A tool must be able to hold a proxy where it held a connector."""
    for name in (
        "read_channel",
        "write_channel",
        "read_multiple_channels",
        "write_multiple_channels",
    ):
        proxy_params = inspect.signature(getattr(ConnectorHostProxy, name)).parameters
        base_params = inspect.signature(getattr(ControlSystemConnector, name)).parameters
        assert list(proxy_params) == list(base_params), name
        assert [param.default for param in proxy_params.values()] == [
            param.default for param in base_params.values()
        ], name

    # disconnect takes an extra keyword-only knob, so it is checked by call shape.
    disconnect = inspect.signature(ConnectorHostProxy.disconnect).parameters
    assert list(disconnect)[:1] == ["self"]
    assert all(
        param.kind is inspect.Parameter.KEYWORD_ONLY
        and param.default is not inspect.Parameter.empty
        for name, param in disconnect.items()
        if name != "self"
    )


def test_the_proxy_does_not_subscribe():
    """No tool calls subscribe(); a proxy that offered one would be a lie."""
    assert not hasattr(ConnectorHostProxy, "subscribe")


def test_the_import_closure_never_reaches_a_control_system_client():
    """The whole point of the child: no libca in the parent's address space."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src")]
    )
    probe = (
        "import osprey_connectors.ipc.proxy, sys; "
        "sys.exit(1 if any(m in sys.modules for m in ('epics', 'pyepics', 'p4p')) else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, (
        f"importing the proxy pulled a control-system client into the parent: "
        f"{completed.stdout}{completed.stderr}"
    )
