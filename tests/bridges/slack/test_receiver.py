"""Tests for the Slack receiver: the ingest thread and the serve loop.

The two halves meet only at the durable :class:`~osprey.bridges.slack.inbox.Inbox`
(a real one on ``tmp_path``), so every test drives them through it: a fake socket
connection yields envelopes and idle ticks, and a fake runtime holds, returns or
raises per ``event_id``. Waits are bounded polls on observable state, never
sleeps that guess at timing.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.bridges.slack.client import SlackAuthError
from osprey.bridges.slack.inbox import Inbox
from osprey.bridges.slack.receiver import POOL_SIZE, WAKE_INTERVAL, Ingest, serve
from osprey.bridges.slack.socket import Envelope

JOIN_TIMEOUT = 5.0
TICK = 0.02

_END = object()


# -- fakes ----------------------------------------------------------------------


class FakeConnection:
    """Iterates what the test pushes; an empty queue is an idle tick (``None``).

    Push an :class:`Envelope`, an exception instance (raised from the iterator),
    or :data:`_END` (iteration ends, as when the link closes for good).
    """

    def __init__(self, on_ack: Callable[[str], None] | None = None) -> None:
        self.items: queue.Queue[Any] = queue.Queue()
        self.acks: list[str] = []
        self.closed = threading.Event()
        self._on_ack = on_ack

    def push(self, item: Any) -> None:
        self.items.put(item)

    def __iter__(self) -> Iterator[Envelope | None]:
        while not self.closed.is_set():
            try:
                item = self.items.get(timeout=TICK)
            except queue.Empty:
                yield None
                continue
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def ack(self, envelope_id: str) -> bool:
        if self._on_ack is not None:
            self._on_ack(envelope_id)
        self.acks.append(envelope_id)
        return True

    def close(self) -> None:
        self.closed.set()


class FakeRuntime:
    """``handle_event`` per ``event_id``: held until released, raising, or returning."""

    def __init__(self, *, hold: bool = False) -> None:
        self.hold = hold
        self.raises: set[str] = set()
        self.calls: list[str] = []
        self.active: set[str] = set()
        self.max_active = 0
        self.supervised = 0
        self._gates: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def gate(self, event_id: str) -> threading.Event:
        with self._lock:
            return self._gates.setdefault(event_id, threading.Event())

    def release(self, event_id: str) -> None:
        self.gate(event_id).set()

    def release_all(self) -> None:
        self.hold = False
        with self._lock:
            gates = list(self._gates.values())
        for g in gates:
            g.set()

    def handle_event(self, event: Any) -> str:
        eid = event["event_id"]
        with self._lock:
            self.calls.append(eid)
            self.active.add(eid)
            self.max_active = max(self.max_active, len(self.active))
        try:
            if self.hold:
                assert self.gate(eid).wait(JOIN_TIMEOUT), f"{eid} never released"
            if eid in self.raises:
                raise RuntimeError(f"boom {eid}")
            return "handled"
        finally:
            with self._lock:
                self.active.discard(eid)

    def supervise(self) -> threading.Thread:
        self.supervised += 1
        return threading.current_thread()


class FakeClock:
    """Monotonic clock that advances ``step`` on every read."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class ServeThread(threading.Thread):
    """Runs :func:`serve` and keeps what it raised."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(daemon=True, name="test-serve")
        kwargs.setdefault("tick", TICK)
        self.kwargs = kwargs
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            serve(**self.kwargs)
        except BaseException as exc:
            self.error = exc


def payload(event_id: str, text: str = "hi") -> dict[str, Any]:
    return {
        "type": "event_callback",
        "event_id": event_id,
        "team_id": "T0TEAM",
        "event": {"type": "message", "channel": "D0DIRECT", "ts": "1.0", "text": text},
    }


def envelope(event_id: str, kind: str = "events_api") -> Envelope:
    return Envelope(f"env-{event_id}", kind, payload(event_id))


def wait_until(predicate: Callable[[], bool], timeout: float = JOIN_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition never became true")
        time.sleep(0.005)


def record_file(inbox: Inbox, key: str) -> dict[str, Any]:
    return json.loads((inbox.directory / f"{key}.json").read_bytes())


@pytest.fixture
def stop() -> Iterator[threading.Event]:
    event = threading.Event()
    yield event
    event.set()


@pytest.fixture
def fatal() -> threading.Event:
    return threading.Event()


@pytest.fixture
def inbox(tmp_path: Path) -> Inbox:
    return Inbox(tmp_path / "inbox")


def start_ingest(conn: FakeConnection, inbox: Inbox, stop, fatal) -> Ingest:
    ingest = Ingest(conn, inbox, stop=stop, fatal=fatal)
    ingest.start()
    return ingest


# -- module constants -----------------------------------------------------------


def test_bounds_match_the_plan():
    assert POOL_SIZE == 4
    assert WAKE_INTERVAL == 60.0


# -- ingest ---------------------------------------------------------------------


def test_ack_follows_a_durable_put(inbox, stop, fatal):
    seen_on_disk: list[bool] = []
    conn = FakeConnection(
        on_ack=lambda env_id: seen_on_disk.append(
            env_id.removeprefix("env-") in dict(inbox.pending())
        )
    )
    ingest = start_ingest(conn, inbox, stop, fatal)
    conn.push(envelope("Ev1"))
    wait_until(lambda: conn.acks == ["env-Ev1"])
    assert seen_on_disk == [True]
    assert inbox.pending() == [("Ev1", payload("Ev1"))]
    stop.set()
    ingest.join(JOIN_TIMEOUT)
    assert not ingest.is_alive()


def test_cap_withholds_the_ack(tmp_path, stop, fatal, caplog):
    inbox = Inbox(tmp_path / "inbox", cap=1)
    conn = FakeConnection()
    ingest = start_ingest(conn, inbox, stop, fatal)
    conn.push(envelope("Ev1"))
    conn.push(envelope("Ev2"))
    conn.push(envelope("Ev3", kind="hello_marker"))  # processed after Ev2
    wait_until(lambda: "env-Ev3" in conn.acks)
    assert conn.acks == ["env-Ev1", "env-Ev3"]
    assert [k for k, _ in inbox.pending()] == ["Ev1"]
    assert any("Ev2" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    stop.set()
    ingest.join(JOIN_TIMEOUT)


def test_redelivery_of_a_stored_event_is_acked_again(inbox, stop, fatal):
    conn = FakeConnection()
    ingest = start_ingest(conn, inbox, stop, fatal)
    conn.push(envelope("Ev1"))
    conn.push(Envelope("env-retry", "events_api", payload("Ev1")))
    wait_until(lambda: len(conn.acks) == 2)
    assert [k for k, _ in inbox.pending()] == ["Ev1"]
    stop.set()
    ingest.join(JOIN_TIMEOUT)


def test_other_envelope_types_are_acked_and_ignored(inbox, stop, fatal):
    conn = FakeConnection()
    ingest = start_ingest(conn, inbox, stop, fatal)
    conn.push(Envelope("env-cmd", "slash_commands", {"command": "/x"}))
    conn.push(Envelope("env-int", "interactive", {"type": "block_actions"}))
    wait_until(lambda: len(conn.acks) == 2)
    assert inbox.pending() == []
    stop.set()
    ingest.join(JOIN_TIMEOUT)


def test_events_api_without_event_id_is_acked_and_dropped_loudly(inbox, stop, fatal, caplog):
    conn = FakeConnection()
    ingest = start_ingest(conn, inbox, stop, fatal)
    conn.push(Envelope("env-bad", "events_api", {"type": "event_callback"}))
    wait_until(lambda: conn.acks == ["env-bad"])
    assert inbox.pending() == []
    assert any(r.levelname == "WARNING" for r in caplog.records)
    stop.set()
    ingest.join(JOIN_TIMEOUT)


def test_ingest_stops_on_stop_and_closes_the_connection(inbox, stop, fatal):
    conn = FakeConnection()
    ingest = start_ingest(conn, inbox, stop, fatal)
    stop.set()
    ingest.join(JOIN_TIMEOUT)
    assert not ingest.is_alive()
    assert ingest.error is None
    assert conn.closed.is_set()


def test_ingest_raises_on_the_fatal_flag(inbox, stop, fatal):
    conn = FakeConnection()
    ingest = start_ingest(conn, inbox, stop, fatal)
    fatal.set()
    ingest.join(JOIN_TIMEOUT)
    assert not ingest.is_alive()
    assert isinstance(ingest.error, SlackAuthError)


def test_ingest_records_an_iterator_failure(inbox, stop, fatal):
    conn = FakeConnection()
    ingest = start_ingest(conn, inbox, stop, fatal)
    conn.push(SlackAuthError("link disabled", error="link_disabled"))
    ingest.join(JOIN_TIMEOUT)
    assert not ingest.is_alive()
    assert isinstance(ingest.error, SlackAuthError)
    assert ingest.error.error == "link_disabled"


# -- serve ----------------------------------------------------------------------


def test_ack_never_waits_on_a_busy_pool(inbox, stop, fatal):
    conn = FakeConnection()
    runtime = FakeRuntime(hold=True)
    ingest = start_ingest(conn, inbox, stop, fatal)
    srv = ServeThread(
        inbox=inbox, runtime=runtime, stop=stop, fatal=fatal, ingest=ingest, pool_size=1
    )
    srv.start()
    for i in range(1, 5):
        conn.push(envelope(f"Ev{i}"))
    wait_until(lambda: len(conn.acks) == 4)
    wait_until(lambda: runtime.calls == ["Ev1"])
    assert runtime.max_active == 1
    # Every record is durable while only one is being handled.
    assert [k for k, _ in inbox.pending()] == ["Ev1", "Ev2", "Ev3", "Ev4"]
    runtime.release_all()
    wait_until(lambda: inbox.pending() == [])
    assert runtime.calls == ["Ev1", "Ev2", "Ev3", "Ev4"]
    stop.set()
    srv.join(JOIN_TIMEOUT)
    ingest.join(JOIN_TIMEOUT)
    assert srv.error is None


def test_pool_bounds_concurrent_handlers(inbox, stop, fatal):
    for i in range(1, 7):
        inbox.put(f"Ev{i}", payload(f"Ev{i}"))
    runtime = FakeRuntime(hold=True)
    srv = ServeThread(inbox=inbox, runtime=runtime, stop=stop, fatal=fatal, pool_size=2)
    srv.start()
    wait_until(lambda: len(runtime.calls) == 2)
    time.sleep(TICK * 5)  # several passes, each re-reading pending()
    assert runtime.calls == ["Ev1", "Ev2"]  # in-flight records are not resubmitted
    runtime.release_all()
    wait_until(lambda: inbox.pending() == [])
    assert sorted(runtime.calls) == [f"Ev{i}" for i in range(1, 7)]
    assert runtime.max_active <= 2
    stop.set()
    srv.join(JOIN_TIMEOUT)
    assert srv.error is None


def test_record_is_removed_only_when_handle_event_returns(inbox, stop, fatal):
    inbox.put("Ev1", payload("Ev1"))
    runtime = FakeRuntime(hold=True)
    srv = ServeThread(inbox=inbox, runtime=runtime, stop=stop, fatal=fatal)
    srv.start()
    wait_until(lambda: runtime.calls == ["Ev1"])
    assert [k for k, _ in inbox.pending()] == ["Ev1"]
    runtime.release("Ev1")
    wait_until(lambda: inbox.pending() == [])
    stop.set()
    srv.join(JOIN_TIMEOUT)


def test_record_is_kept_on_a_raise_and_counted(inbox, stop, fatal):
    inbox.put("Ev1", payload("Ev1"))
    inbox.put("Ev2", payload("Ev2"))
    runtime = FakeRuntime()
    runtime.raises.add("Ev1")
    srv = ServeThread(
        inbox=inbox, runtime=runtime, stop=stop, fatal=fatal, clock=FakeClock(step=0.0)
    )
    srv.start()
    wait_until(lambda: [k for k, _ in inbox.pending()] == ["Ev1"])
    wait_until(lambda: record_file(inbox, "Ev1")["attempts"] == 1)
    time.sleep(TICK * 5)
    # Not hot-looped: a failed record waits for the next wake before a retry.
    assert runtime.calls.count("Ev1") == 1
    stop.set()
    srv.join(JOIN_TIMEOUT)
    assert srv.error is None


def test_failed_record_is_retried_at_the_wake_until_quarantined(inbox, stop, fatal):
    inbox.put("Ev1", payload("Ev1"))
    runtime = FakeRuntime()
    runtime.raises.add("Ev1")
    srv = ServeThread(
        inbox=inbox,
        runtime=runtime,
        stop=stop,
        fatal=fatal,
        wake_interval=10.0,
        clock=FakeClock(step=1.0),
    )
    srv.start()
    wait_until(lambda: inbox.pending() == [])
    assert runtime.calls == ["Ev1", "Ev1", "Ev1"]  # max_attempts = 3
    assert len(list((inbox.directory / "quarantine").iterdir())) == 1
    stop.set()
    srv.join(JOIN_TIMEOUT)


def test_replay_after_a_simulated_crash_feeds_leftovers_first(tmp_path, stop, fatal):
    directory = tmp_path / "inbox"
    # The previous process stored and acked two events, then died.
    crashed = Inbox(directory)
    crashed.put("Old1", payload("Old1"))
    crashed.put("Old2", payload("Old2"))
    del crashed

    inbox = Inbox(directory)
    conn = FakeConnection()
    runtime = FakeRuntime(hold=True)
    ingest = start_ingest(conn, inbox, stop, fatal)
    conn.push(envelope("New1"))
    wait_until(lambda: conn.acks == ["env-New1"])  # ingest runs before serve starts
    srv = ServeThread(
        inbox=inbox, runtime=runtime, stop=stop, fatal=fatal, ingest=ingest, pool_size=1
    )
    srv.start()
    for eid in ("Old1", "Old2", "New1"):
        wait_until(lambda eid=eid: runtime.calls[-1:] == [eid])
        runtime.release(eid)
    wait_until(lambda: inbox.pending() == [])
    assert runtime.calls == ["Old1", "Old2", "New1"]
    stop.set()
    srv.join(JOIN_TIMEOUT)
    ingest.join(JOIN_TIMEOUT)
    assert srv.error is None


def test_idle_socket_still_supervises_and_stops_promptly(inbox, stop, fatal):
    conn = FakeConnection()  # never yields anything but idle ticks
    runtime = FakeRuntime()
    ingest = start_ingest(conn, inbox, stop, fatal)
    srv = ServeThread(
        inbox=inbox,
        runtime=runtime,
        stop=stop,
        fatal=fatal,
        ingest=ingest,
        wake_interval=60.0,
        clock=FakeClock(step=30.0),
    )
    srv.start()
    wait_until(lambda: runtime.supervised >= 2)
    started = time.monotonic()
    stop.set()
    srv.join(JOIN_TIMEOUT)
    ingest.join(JOIN_TIMEOUT)
    assert not srv.is_alive() and not ingest.is_alive()
    assert time.monotonic() - started < 1.0
    assert srv.error is None
    assert conn.closed.is_set()


def test_supervise_waits_for_the_wake_interval(inbox, stop, fatal):
    runtime = FakeRuntime()
    srv = ServeThread(
        inbox=inbox,
        runtime=runtime,
        stop=stop,
        fatal=fatal,
        wake_interval=60.0,
        clock=FakeClock(step=0.0),
    )
    srv.start()
    time.sleep(TICK * 5)
    assert runtime.supervised == 0
    stop.set()
    srv.join(JOIN_TIMEOUT)


def test_fatal_flag_raises(inbox, stop, fatal):
    conn = FakeConnection()
    runtime = FakeRuntime()
    ingest = start_ingest(conn, inbox, stop, fatal)
    srv = ServeThread(inbox=inbox, runtime=runtime, stop=stop, fatal=fatal, ingest=ingest)
    srv.start()
    fatal.set()
    srv.join(JOIN_TIMEOUT)
    ingest.join(JOIN_TIMEOUT)
    assert not srv.is_alive()
    assert isinstance(srv.error, SlackAuthError)


def test_fatal_flag_raises_without_an_ingest(inbox, stop, fatal):
    fatal.set()
    with pytest.raises(SlackAuthError):
        serve(inbox, FakeRuntime(), stop=stop, fatal=fatal, tick=TICK)


def test_ingest_death_raises_through_serve(inbox, stop, fatal):
    conn = FakeConnection()
    runtime = FakeRuntime()
    ingest = start_ingest(conn, inbox, stop, fatal)
    srv = ServeThread(inbox=inbox, runtime=runtime, stop=stop, fatal=fatal, ingest=ingest)
    srv.start()
    conn.push(SlackAuthError("link disabled", error="link_disabled"))
    srv.join(JOIN_TIMEOUT)
    assert not srv.is_alive()
    assert isinstance(srv.error, SlackAuthError)
    assert srv.error.error == "link_disabled"


def test_ingest_ending_for_any_reason_raises_through_serve(inbox, stop, fatal):
    conn = FakeConnection()
    runtime = FakeRuntime()
    ingest = start_ingest(conn, inbox, stop, fatal)
    srv = ServeThread(inbox=inbox, runtime=runtime, stop=stop, fatal=fatal, ingest=ingest)
    srv.start()
    conn.push(_END)
    srv.join(JOIN_TIMEOUT)
    assert not srv.is_alive()
    assert isinstance(srv.error, SlackAuthError)


def test_stop_drains_the_pool_before_returning(inbox, stop, fatal):
    inbox.put("Ev1", payload("Ev1"))
    runtime = FakeRuntime(hold=True)
    srv = ServeThread(inbox=inbox, runtime=runtime, stop=stop, fatal=fatal)
    srv.start()
    wait_until(lambda: runtime.calls == ["Ev1"])
    stop.set()
    time.sleep(TICK * 5)
    assert srv.is_alive()  # still waiting for the in-flight handler
    runtime.release("Ev1")
    srv.join(JOIN_TIMEOUT)
    assert not srv.is_alive()
    assert srv.error is None
    assert inbox.pending() == []
