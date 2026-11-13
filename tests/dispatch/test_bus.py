"""The dispatch worker bus end to end: broker, link and client over real loopback TCP.

Every case runs a real :class:`~osprey.dispatch.bus.broker.WorkerBus` and real
:class:`~osprey.dispatch.bus.link.WorkerLink` sockets on ``127.0.0.1``, with
:class:`~tests.dispatch._bus_fakes.FakeWorker` handlers standing in for the dispatch worker, and fast
heartbeats so expiry takes a fraction of a second.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest
import zmq
import zmq.asyncio

from osprey.dispatch.bus import broker as bus_broker
from osprey.dispatch.bus import keys as bus_keys
from osprey.dispatch.bus import protocol
from osprey.dispatch.bus.broker import WorkerBus, is_local_endpoint
from osprey.dispatch.bus.client import BusWorkerClient
from osprey.dispatch.bus.keys import BusKeyError
from osprey.dispatch.bus.link import WorkerLink
from osprey.dispatch.worker_client import (
    WorkerAuthRejectedError,
    WorkerRecordUnavailableError,
    WorkerRejectedRequestError,
    WorkerUnavailableError,
    WorkerUnreachableError,
)
from tests.dispatch._bus_fakes import FakeWorker

HB = 0.1


class Harness:
    """Starts buses and links, and stops every one of them at the end of a case."""

    def __init__(self) -> None:
        self.buses: list[WorkerBus] = []
        self.links: list[asyncio.Task[None]] = []

    async def bus(self, bind: str = "tcp://127.0.0.1:*", **kwargs: Any) -> WorkerBus:
        bus = WorkerBus(bind, keys=kwargs.pop("keys", None), heartbeat_sec=HB, **kwargs)
        await bus.start()
        self.buses.append(bus)
        return bus

    def join(self, endpoint: str, worker: FakeWorker, **kwargs: Any) -> asyncio.Task[None]:
        link = WorkerLink(
            endpoint,
            worker.name,
            worker.handlers(),
            hello=worker.hello,
            keys=kwargs.pop("keys", None),
            heartbeat_sec=HB,
            **kwargs,
        )
        worker.link = link
        task = asyncio.create_task(link.run())
        self.links.append(task)
        return task

    @staticmethod
    async def leave(task: asyncio.Task[None]) -> None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def close(self) -> None:
        for task in self.links:
            await self.leave(task)
        for bus in self.buses:
            await bus.stop()


@pytest.fixture
async def harness():
    h = Harness()
    try:
        yield h
    finally:
        await h.close()


async def until(probe: Callable[[], Any], timeout: float = 5.0) -> Any:
    """Await until ``probe()`` is truthy, and return it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = probe()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


async def joined(harness: Harness, *names: str) -> tuple[WorkerBus, BusWorkerClient, list]:
    """A bus with one fake worker per name, every one of them live."""
    bus = await harness.bus()
    workers = [FakeWorker(name) for name in names]
    for worker in workers:
        harness.join(bus.endpoint, worker)
    await until(lambda: bus.live() == sorted(names))
    return bus, BusWorkerClient(bus, worker_wait_sec=1.0), workers


# -- wire format --------------------------------------------------------------


def test_a_message_round_trips_with_its_blob():
    frames = protocol.encode(protocol.ARTIFACT, {"rid": "r", "n": 1}, b"\x00\xff")
    message = protocol.decode(frames)
    assert message == protocol.Message(protocol.ARTIFACT, {"rid": "r", "n": 1}, b"\x00\xff")


@pytest.mark.parametrize(
    "frames",
    [[b"OK"], [b"OK", b"not json"], [b"OK", b"[1]"], [b"OK", b"{}", b"", b""]],
)
def test_a_malformed_message_is_refused(frames):
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(frames)


@pytest.mark.parametrize(
    ("endpoint", "local"),
    [
        ("tcp://127.0.0.1:5555", True),
        ("tcp://localhost:5555", True),
        ("tcp://[::1]:5555", True),
        ("ipc:///tmp/bus", True),
        ("inproc://bus", True),
        ("tcp://0.0.0.0:5555", False),
        ("tcp://event-dispatcher:5555", False),
    ],
)
def test_only_a_loopback_endpoint_counts_as_local(endpoint, local):
    assert is_local_endpoint(endpoint) is local


def test_an_unauthenticated_bus_refuses_a_public_bind():
    with pytest.raises(BusKeyError, match="without CURVE keys"):
        WorkerBus("tcp://0.0.0.0:5555", keys=None)


def test_an_unauthenticated_link_refuses_a_remote_dispatcher():
    with pytest.raises(BusKeyError, match="without CURVE keys"):
        WorkerLink("tcp://event-dispatcher:5555", "w1", {}, hello=dict, keys=None)


# -- start order --------------------------------------------------------------


async def test_a_dispatch_waits_for_a_worker_that_joins_late(harness):
    bus = await harness.bus()
    client = BusWorkerClient(bus, worker_wait_sec=5.0)
    pending = asyncio.create_task(client.dispatch(prompt="p", allowed_tools=[]))
    await asyncio.sleep(0.2)
    assert not pending.done()

    harness.join(bus.endpoint, FakeWorker("w1"))
    answer = await asyncio.wait_for(pending, 5)
    assert answer == {"status": "accepted", "run_id": "w1-run-1"}


async def test_a_dispatch_with_no_worker_is_unreachable_and_retryable(harness):
    bus = await harness.bus()
    client = BusWorkerClient(bus, worker_wait_sec=0.1)
    with pytest.raises(WorkerUnreachableError) as caught:
        await client.dispatch(prompt="p", allowed_tools=[])
    assert caught.value.retryable


async def test_a_worker_rejoins_a_dispatcher_that_restarted(harness):
    first, _, workers = await joined(harness, "w1")
    workers[0].in_flight.add("w1-run-7")
    endpoint = first.endpoint
    await first.stop()
    harness.buses.remove(first)

    second = await harness.bus(endpoint)
    await until(lambda: second.live() == ["w1"])
    # The heartbeat carried the run in flight, so the new broker knows its owner.
    assert second.owner_of("w1-run-7") == "w1"
    answer = await BusWorkerClient(second).dispatch(prompt="p", allowed_tools=[])
    assert answer["run_id"] == "w1-run-1"


# -- distribution and routing ---------------------------------------------------


async def test_dispatches_spread_over_the_least_loaded_workers(harness):
    bus, client, (w1, w2) = await joined(harness, "w1", "w2")
    owners = []
    for _ in range(4):
        answer = await client.dispatch(prompt="p", allowed_tools=[])
        owners.append(bus.owner_of(answer["run_id"]))
    assert sorted(owners) == ["w1", "w1", "w2", "w2"]
    assert w1.asked(protocol.DISPATCH) == w2.asked(protocol.DISPATCH) == 2


async def test_a_worker_at_capacity_is_passed_over(harness):
    bus = await harness.bus()
    full, free = FakeWorker("full"), FakeWorker("free")
    full.in_flight = {"x"}
    full_hello = full.hello
    full.hello = lambda: {**full_hello(), "capacity": 1}  # type: ignore[method-assign]
    harness.join(bus.endpoint, full)
    harness.join(bus.endpoint, free)
    await until(lambda: bus.live() == ["free", "full"])
    client = BusWorkerClient(bus)
    for _ in range(3):
        await client.dispatch(prompt="p", allowed_tools=[])
    assert full.asked(protocol.DISPATCH) == 0
    assert free.asked(protocol.DISPATCH) == 3


async def test_a_cancel_goes_to_the_owner_only(harness):
    _, client, (w1, w2) = await joined(harness, "w1", "w2")
    run_id = (await client.dispatch(prompt="p", allowed_tools=[]))["run_id"]
    owner, other = (w1, w2) if run_id.startswith("w1") else (w2, w1)

    assert await client.cancel(run_id) == {"run_id": run_id, "cancelled": True}
    assert owner.asked(protocol.CANCEL) == 1
    assert other.asked(protocol.CANCEL) == 0


async def test_a_run_with_no_known_owner_is_asked_of_every_worker(harness):
    _, client, (w1, w2) = await joined(harness, "w1", "w2")
    w2.statuses["old-run"] = {"run_id": "old-run", "status": "completed"}

    assert await client.status("old-run") == {"run_id": "old-run", "status": "completed"}
    assert w1.asked(protocol.STATUS) == w2.asked(protocol.STATUS) == 1


async def test_a_run_no_worker_holds_is_not_found(harness):
    _, client, _ = await joined(harness, "w1", "w2")
    with pytest.raises(WorkerRejectedRequestError) as caught:
        await client.status("nobody-has-it")
    assert not caught.value.retryable


async def test_an_artifact_crosses_the_bus_byte_for_byte(harness):
    _, client, (w1,) = await joined(harness, "w1")
    w1.statuses["r"] = {}
    meta, data = await client.artifact("r", "a1")
    assert meta == {"mime": "image/png", "filename": "plot.png"}
    assert data == w1.blob


async def test_the_run_list_merges_every_worker_newest_first(harness):
    _, client, (w1, w2) = await joined(harness, "w1", "w2")
    w1.rows = [{"run_id": "a", "created_at": 1}, {"run_id": "c", "created_at": 3}]
    w2.rows = [{"run_id": "b", "created_at": 2}]
    assert [r["run_id"] for r in await client.runs()] == ["c", "b", "a"]


async def test_the_run_list_survives_one_worker_that_cannot_read_the_record(harness):
    _, client, (w1, w2) = await joined(harness, "w1", "w2")
    w1.reject = protocol.BusError(503, "record unavailable: down")
    w2.rows = [{"run_id": "b", "created_at": 2}]
    assert [r["run_id"] for r in await client.runs()] == ["b"]


async def test_the_run_list_reports_a_record_no_worker_can_read(harness):
    _, client, (w1,) = await joined(harness, "w1")
    w1.reject = protocol.BusError(503, "record unavailable: down")
    with pytest.raises(WorkerRecordUnavailableError, match="record unavailable"):
        await client.runs()


# -- errors keep the HTTP client's taxonomy --------------------------------------


@pytest.mark.parametrize(
    ("error", "expected", "code"),
    [
        (
            protocol.BusError(400, "input_files_cap_exceeded"),
            WorkerRejectedRequestError,
            "input_files_cap_exceeded",
        ),
        (
            protocol.BusError(403, "Tools blocked by server denylist: ['Bash']"),
            WorkerRejectedRequestError,
            None,
        ),
        (protocol.BusError(401, "Invalid bearer token"), WorkerAuthRejectedError, None),
        (protocol.BusError(503, "busy"), WorkerUnavailableError, None),
        (RuntimeError("secret internal detail"), WorkerUnavailableError, None),
    ],
)
async def test_a_refused_dispatch_raises_what_the_http_client_raises(
    harness, error, expected, code
):
    _, client, (w1,) = await joined(harness, "w1")
    w1.reject = error
    with pytest.raises(expected) as caught:
        await client.dispatch(prompt="p", allowed_tools=[])
    assert type(caught.value) is expected
    assert caught.value.error_code == code
    # Only the whitelisted code crosses; the worker's own text never does.
    assert "secret internal detail" not in str(caught.value)
    assert "denylist" not in str(caught.value)


async def test_an_oversize_request_is_refused_before_it_is_read(harness):
    bus = await harness.bus()
    worker = FakeWorker("w1")
    harness.join(bus.endpoint, worker, max_request_bytes=1024)
    await until(lambda: bus.live() == ["w1"])
    with pytest.raises(WorkerRejectedRequestError):
        await BusWorkerClient(bus).dispatch(prompt="x" * 4096, allowed_tools=[])
    assert worker.asked(protocol.DISPATCH) == 0


# -- liveness ---------------------------------------------------------------------


async def test_a_worker_that_goes_silent_loses_its_runs(harness):
    bus = await harness.bus()
    worker = FakeWorker("w1")
    link = harness.join(bus.endpoint, worker)
    await until(lambda: bus.live() == ["w1"])
    client = BusWorkerClient(bus)
    run_id = (await client.dispatch(prompt="p", allowed_tools=[]))["run_id"]
    worker.hold = asyncio.Event()
    asked = asyncio.create_task(client.status(run_id))
    await until(lambda: worker.asked(protocol.STATUS))
    viewer = asyncio.create_task(_collect(client, run_id))

    await harness.leave(link)

    with pytest.raises(WorkerUnreachableError):
        await asyncio.wait_for(asked, 2)
    chunks = await asyncio.wait_for(viewer, 2)
    assert chunks[-1].startswith(b'data: {"type": "error", "message": "worker lost: w1')
    assert bus.live() == []


async def test_a_worker_that_restarted_loses_the_runs_of_its_previous_life(harness):
    bus, client, (w1,) = await joined(harness, "w1")
    run_id = (await client.dispatch(prompt="p", allowed_tools=[]))["run_id"]
    viewer = asyncio.create_task(_collect(client, run_id))

    w1.boot_nonce = 2.0
    w1.in_flight.clear()

    chunks = await asyncio.wait_for(viewer, 2)
    assert b"restarted" in chunks[-1]
    assert bus.live() == ["w1"]


# -- streams ----------------------------------------------------------------------


async def _collect(client: BusWorkerClient, run_id: str) -> list[bytes]:
    return [chunk async for chunk in client.stream(run_id) if not chunk.startswith(b":")]


async def test_every_viewer_sees_the_whole_stream(harness):
    _, client, (w1,) = await joined(harness, "w1")
    run_id = (await client.dispatch(prompt="p", allowed_tools=[]))["run_id"]
    early = asyncio.create_task(_collect(client, run_id))
    await asyncio.sleep(0.05)

    assert w1.link is not None
    w1.link.emit(run_id, {"type": "text", "content": "one"})
    await asyncio.sleep(0.05)
    late = asyncio.create_task(_collect(client, run_id))
    await asyncio.sleep(0.05)
    w1.link.emit(run_id, {"type": "done"})

    expected = [
        b'data: {"type": "text", "content": "one"}\n\n',
        b'data: {"type": "done"}\n\n',
    ]
    assert await asyncio.wait_for(early, 2) == expected
    assert await asyncio.wait_for(late, 2) == expected
    # A viewer after the end still gets the replay while the stream is kept.
    assert await _collect(client, run_id) == expected


async def test_a_stream_of_an_unknown_run_is_unavailable(harness):
    _, client, _ = await joined(harness, "w1")
    with pytest.raises(WorkerUnavailableError, match="not found"):
        await _collect(client, "nope")


async def test_a_quiet_stream_yields_a_keepalive(harness):
    bus, client, (w1,) = await joined(harness, "w1")
    run_id = (await client.dispatch(prompt="p", allowed_tools=[]))["run_id"]
    events = bus.stream(run_id, idle=0.05)
    assert await anext(events) is None

    assert w1.link is not None
    w1.link.emit(run_id, {"type": "done"})
    assert [event async for event in events if event is not None] == [{"type": "done"}]


async def test_a_finished_runs_stream_is_let_go_but_its_owner_is_kept(harness, monkeypatch):
    monkeypatch.setattr(bus_broker, "STREAM_TTL_SEC", 0.0)
    bus, client, (w1,) = await joined(harness, "w1")
    run_id = (await client.dispatch(prompt="p", allowed_tools=[]))["run_id"]
    # Off the heartbeat, or the next one would bring the stream back.
    w1.in_flight.clear()
    assert w1.link is not None
    w1.link.emit(run_id, {"type": "done"})

    await until(lambda: run_id not in bus._streams)
    with pytest.raises(WorkerUnavailableError, match="no longer streams"):
        await _collect(client, run_id)
    # Only the owner can serve the run's artifacts, so it is remembered.
    assert bus.owner_of(run_id) == "w1"


# -- misbehaving peers ------------------------------------------------------------


class RawPeer:
    """A bare ``DEALER`` that sends the broker whatever a case hands it.

    It joins full (capacity 1, one run in flight), so no dispatch is ever sent
    its way, and after each message it sends a heartbeat naming a fresh run:
    once the broker owns that run, it has handled everything sent before it.
    """

    def __init__(self, bus: WorkerBus, name: str) -> None:
        self._bus = bus
        self.name = name
        self._marks = 0
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.ROUTING_ID, name.encode())
        self._sock.connect(bus.endpoint)

    async def __aenter__(self) -> RawPeer:
        await self._mark()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self._sock.close(linger=0)
        self._ctx.term()

    async def send(self, *messages: list[bytes]) -> None:
        for frames in messages:
            await self._sock.send_multipart(frames)
            await self._mark()

    async def _mark(self) -> None:
        self._marks += 1
        run_id = f"{self.name}-mark-{self._marks}"
        hello = {"capacity": 1, "in_flight": [run_id]}
        await self._sock.send_multipart(protocol.encode(protocol.HELLO, hello))
        await until(lambda: self._bus.owner_of(run_id) == self.name)


async def test_garbage_from_a_peer_is_dropped_and_the_broker_keeps_serving(harness):
    _, client, _ = await joined(harness, "w1")
    async with RawPeer(harness.buses[0], "noisy") as peer:
        await peer.send(
            [b"OK"],
            [b"OK", b"not json"],
            protocol.encode(b"PING", {}),
            protocol.ok("0" * 32, {"answers": "no request"}),
            protocol.encode(protocol.EVENT, {"run_id": 7, "event": "not an event"}),
        )
        answer = await client.dispatch(prompt="p", allowed_tools=[])
    assert answer["run_id"] == "w1-run-1"


async def test_a_worker_cannot_answer_a_request_sent_to_another(harness):
    bus, client, (w1,) = await joined(harness, "w1")
    run_id = (await client.dispatch(prompt="p", allowed_tools=[]))["run_id"]
    w1.statuses[run_id] = {"run_id": run_id, "status": "completed"}
    w1.hold = asyncio.Event()
    asked = asyncio.create_task(client.status(run_id))
    await until(lambda: w1.asked(protocol.STATUS))
    rid = w1.calls[-1][1]["rid"]

    async with RawPeer(bus, "w2") as impostor:
        await impostor.send(protocol.ok(rid, {"run_id": run_id, "status": "forged"}))
        assert not asked.done()
        w1.hold.set()
        assert await asyncio.wait_for(asked, 2) == {"run_id": run_id, "status": "completed"}


async def test_a_dispatch_answer_without_a_run_id_starts_no_run(harness):
    _, client, (w1,) = await joined(harness, "w1")

    async def no_run_id(_body):
        return {"status": "accepted"}, None

    w1.dispatch = no_run_id
    with pytest.raises(WorkerUnavailableError):
        await client.dispatch(prompt="p", allowed_tools=[])


async def test_a_call_the_worker_does_not_serve_is_unavailable_not_rejected(harness):
    # A worker older than the call answers 501: a fault of the deployment, not of the request.
    bus = await harness.bus()
    worker = FakeWorker("w1")
    serves = worker.handlers()
    del serves[protocol.ARTIFACT]
    worker.handlers = lambda: serves  # type: ignore[method-assign]
    harness.join(bus.endpoint, worker)
    await until(lambda: bus.live() == ["w1"])
    with pytest.raises(WorkerUnavailableError) as caught:
        await BusWorkerClient(bus).artifact("r", "a1")
    assert type(caught.value) is WorkerUnavailableError


async def test_a_verb_the_worker_has_never_heard_of_is_answered_501_not_dropped(harness):
    # A newer dispatcher's verb: the worker answers at once instead of leaving
    # the dispatcher to wait out its timeout and read the worker as unreachable.
    bus, _, _ = await joined(harness, "w1")
    with pytest.raises(protocol.BusError) as caught:
        await bus.request("w1", b"FUTURE_VERB", {}, timeout=2.0)
    assert caught.value.status == 501


@pytest.mark.parametrize(
    "call",
    [
        lambda client: client.runs(),
        lambda client: client.status("r"),
        lambda client: client.cancel("r"),
        lambda client: client.artifacts("r"),
    ],
    ids=["runs", "status", "cancel", "artifacts"],
)
async def test_a_call_with_no_worker_connected_is_unreachable_and_retryable(harness, call):
    bus = await harness.bus()
    with pytest.raises(WorkerUnreachableError) as caught:
        await call(BusWorkerClient(bus))
    assert caught.value.retryable


async def test_a_worker_whose_state_cannot_be_read_still_joins(harness):
    bus = await harness.bus()
    worker = FakeWorker("w1")

    def unreadable() -> dict[str, Any]:
        raise RuntimeError("state unreadable")

    worker.hello = unreadable  # type: ignore[method-assign]
    harness.join(bus.endpoint, worker)
    await until(lambda: bus.live() == ["w1"])


def test_an_event_emitted_before_the_link_connects_is_counted_as_dropped():
    link = WorkerLink("tcp://127.0.0.1:1", "w1", {}, hello=dict, keys=None)
    link.emit("r", {"type": "text", "content": "lost"})
    assert link.dropped_events == 1


# -- CURVE ----------------------------------------------------------------------


@pytest.fixture
def curve():
    secrets = {name: bus_keys.new_secret() for name in ("server", "w1", "w2", "rogue")}
    server = bus_keys.server_keys(
        secrets["server"], {name: bus_keys.public_of(secrets[name]) for name in ("w1", "w2")}
    )

    def client_keys(name: str) -> bus_keys.ClientKeys:
        return bus_keys.client_keys(secrets[name], bus_keys.public_of(secrets["server"]))

    return server, client_keys


async def test_a_worker_with_a_listed_key_joins_under_its_name(harness, curve):
    server, client_keys = curve
    bus = await harness.bus(keys=server)
    harness.join(bus.endpoint, FakeWorker("w1"), keys=client_keys("w1"))
    await until(lambda: bus.live() == ["w1"])
    answer = await BusWorkerClient(bus).dispatch(prompt="p", allowed_tools=[])
    assert answer["run_id"] == "w1-run-1"


async def test_a_worker_with_an_unlisted_key_never_joins(harness, curve):
    server, client_keys = curve
    bus = await harness.bus(keys=server)
    harness.join(bus.endpoint, FakeWorker("w1"), keys=client_keys("rogue"))
    await asyncio.sleep(HB * 8)
    assert bus.live() == []


async def test_a_worker_cannot_join_under_another_workers_name(harness, curve):
    server, client_keys = curve
    bus = await harness.bus(keys=server)
    # w2's own, listed key, presented under w1's name.
    harness.join(bus.endpoint, FakeWorker("w1"), keys=client_keys("w2"))
    await asyncio.sleep(HB * 8)
    assert bus.live() == []


def test_a_public_key_is_derived_from_its_secret():
    secret = bus_keys.new_secret()
    assert len(secret) == 64
    assert bus_keys.public_of(secret) == bus_keys.public_of(secret) != secret


@pytest.mark.parametrize(
    ("workers", "match"),
    [({}, "no worker key"), ({"w1": "zz"}, "not hex"), ({"w1": "00" * 16}, "16 bytes")],
)
def test_server_keys_refuse_what_no_worker_could_join_with(workers, match):
    with pytest.raises(BusKeyError, match=match):
        bus_keys.server_keys(bus_keys.new_secret(), workers)


def test_two_workers_may_not_share_a_key():
    public = bus_keys.public_of(bus_keys.new_secret())
    with pytest.raises(BusKeyError, match="the same key"):
        bus_keys.server_keys(bus_keys.new_secret(), {"w1": public, "w2": public})


def test_the_worker_key_list_round_trips():
    workers = {"dispatch-worker-2": "ab" * 32, "dispatch-worker-1": "cd" * 32}
    text = bus_keys.format_worker_keys(workers)
    assert text.startswith("dispatch-worker-1=")
    assert bus_keys.parse_worker_keys(text) == workers
    assert bus_keys.parse_worker_keys(f"{text},") == workers


@pytest.mark.parametrize("text", ["w1", "w1=", "=ab", "w1=ab,w1=cd"])
def test_a_malformed_worker_key_list_is_refused(text):
    with pytest.raises(BusKeyError):
        bus_keys.parse_worker_keys(text)


def test_the_worker_takes_its_secret_out_of_the_environment():
    from osprey.dispatch.bus import settings

    secret = bus_keys.new_secret()
    environ = {
        settings.CONNECT_ENV: "tcp://event-dispatcher:10050",
        settings.WORKER_NAME_ENV: "dispatch-worker-1",
        settings.SECRET_KEY_ENV: secret,
        settings.SERVER_KEY_ENV: bus_keys.public_of(bus_keys.new_secret()),
    }
    worker = settings.worker_settings(environ)
    assert worker is not None and worker.keys is not None
    assert settings.SECRET_KEY_ENV not in environ


def test_a_dispatcher_with_a_secret_requires_the_worker_list():
    from osprey.dispatch.bus import settings

    environ = {settings.BIND_ENV: "tcp://0.0.0.0:1", settings.SECRET_KEY_ENV: bus_keys.new_secret()}
    with pytest.raises(BusKeyError, match=settings.WORKER_KEYS_ENV):
        settings.server_settings(environ)
