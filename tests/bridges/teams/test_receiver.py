"""Service Bus ingestion: the ``QueueReceiver`` seam, body decoding, and the serve loop.

Everything runs over a hand-written receiver, message and runtime. No Azure
library is imported at collection or at run time: the two Service Bus exception
classes the settlement path recognises are matched by *name*, so this module
defines look-alikes of its own, and ``make_receiver`` -- the one place
``azure.servicebus`` is imported -- is exercised against a stub module installed
under the SDK's name, which is what lets the factory's arithmetic be read back
off the call it made.

The threaded tests are deterministic. Every handler blocks on an
:class:`threading.Event` the test releases, the serve loop's wake clock is a
fake the test advances, and "the loop went round without pulling" is proven by
counting the fake clock's reads rather than by sleeping. No test waits on a
timeout that has to expire for the assertion to hold.

Properties pinned here, each silent in production when it regresses:

*  **every pull is gated on a free handler slot**, never on a count: with
   ``POOL_SIZE`` handlers in flight the receiver is not asked for another
   message until one of them returns, and no pull ever asks for fewer than one
   message;
*  **every receiver call happens under the one lock** — the pull, the
   registration and both settlements;
*  **``register`` precedes and ``complete`` follows ``handle_event``**, and the
   settlement runs from a ``finally`` so a raising handler still settles;
*  **a lost message lock is logged, not raised**, because redelivery is
   de-duplicated by the engine's claim;
*  **a dead receiver fails the loop loudly** instead of leaving a bridge that
   looks alive and answers no one;
*  **the lock is renewed for the whole poll budget plus the settlement margin**,
   and the queue is opened in peek-lock, both read back off what
   ``make_receiver`` asked the SDK for.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import threading
from collections.abc import Callable, Iterable
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from osprey.bridges.core import CoreConfig
from osprey.bridges.teams import receiver as receiver_module
from osprey.bridges.teams.config import TeamsBridgeConfig
from osprey.bridges.teams.receiver import (
    POOL_SIZE,
    RECEIVE_WAIT_SEC,
    SETTLE_MARGIN_SEC,
    SLOT_TIMEOUT_SEC,
    WAKE_INTERVAL,
    QueueReceiver,
    decode_body,
    make_receiver,
    serve,
)

# Short enough that a slot miss never dominates a test, long enough that the
# loop is not a busy spin. Correctness never depends on it: every assertion
# below is driven by an Event or a fake clock, not by this expiring.
TEST_SLOT_TIMEOUT = 0.005

# Joins bound how long a broken implementation can hang the suite. A passing
# run never comes near them.
JOIN_TIMEOUT = 10.0


# --- fakes ------------------------------------------------------------------


class ServiceBusError(Exception):
    """Same name as the SDK's base class; the loop recognises it by name only."""


class MessageLockLostError(ServiceBusError):
    """Same name and parentage as the SDK's lock-lost error."""


_DEFAULT_BODY = object()
"""Sentinel for "give the message a well-formed activity body" — ``None`` is a
real (rejected) body value under test."""


class FakeMessage:
    """A received message: an id for the timeline and a body like the SDK's.

    The SDK exposes a data body as an *iterator of byte sections*, which is the
    shape ``body`` takes here by default so the loop is proven against the
    least convenient form it will meet in production.
    """

    def __init__(self, mid: str, body: Any = _DEFAULT_BODY, *, sections: bool = True):
        self.id = mid
        raw = json.dumps({"type": "message", "id": mid}).encode() if body is _DEFAULT_BODY else body
        self._raw = raw
        self._sections = sections

    @property
    def body(self) -> Any:
        if self._sections and isinstance(self._raw, bytes):
            return iter([self._raw[:3], self._raw[3:]])
        return self._raw


class FakeReceiver:
    """A scripted :class:`QueueReceiver` that records every call on a shared timeline.

    ``messages`` is dealt out one per pull, then pulls return empty batches.
    ``receive`` can be scripted to raise on a given call (``die_on``) to model a
    receiver that has gone away; ``settle_raises`` makes ``complete`` raise for
    the named message ids.

    Every method asserts the serve loop's lock is held when it is entered — the
    lock is the test's own, handed to ``serve`` — and records overlapping
    entries, so a call made outside the lock fails the test even when no other
    thread happened to be inside at the time.
    """

    def __init__(
        self,
        messages: list[FakeMessage],
        *,
        lock: threading.Lock,
        timeline: list[tuple[str, str]],
        die_on: int | None = None,
        settle_raises: dict[str, BaseException] | None = None,
        register_raises: BaseException | None = None,
    ):
        self._pending = list(messages)
        self.lock = lock
        self.timeline = timeline
        self.receive_calls: list[tuple[int, float]] = []
        self.completed: list[str] = []
        self.dead_lettered: list[tuple[str, str]] = []
        self.registered: list[str] = []
        self.die_on = die_on
        self.settle_raises = settle_raises or {}
        self.register_raises = register_raises
        self.pulled = threading.Event()
        self._active = 0
        self.max_overlap = 0
        self._entry_guard = threading.Lock()

    def _enter(self) -> None:
        assert self.lock.locked(), "receiver called outside the serve lock"
        with self._entry_guard:
            self._active += 1
            self.max_overlap = max(self.max_overlap, self._active)

    def _exit(self) -> None:
        with self._entry_guard:
            self._active -= 1

    def receive(self, max_messages: int, max_wait: float) -> list[Any]:
        self._enter()
        try:
            assert max_messages >= 1, f"receive asked for {max_messages} messages"
            self.receive_calls.append((max_messages, max_wait))
            self.pulled.set()
            if self.die_on is not None and len(self.receive_calls) >= self.die_on:
                raise RuntimeError("receiver died")
            if not self._pending:
                return []
            return [self._pending.pop(0)]
        finally:
            self._exit()

    def complete(self, msg: Any) -> None:
        self._enter()
        try:
            self.timeline.append(("complete", msg.id))
            self.completed.append(msg.id)
            exc = self.settle_raises.get(msg.id)
            if exc is not None:
                raise exc
        finally:
            self._exit()

    def dead_letter(self, msg: Any, reason: str) -> None:
        self._enter()
        try:
            self.timeline.append(("dead_letter", msg.id))
            self.dead_lettered.append((msg.id, reason))
        finally:
            self._exit()

    def register(self, msg: Any) -> None:
        self._enter()
        try:
            self.timeline.append(("register", msg.id))
            self.registered.append(msg.id)
            if self.register_raises is not None:
                raise self.register_raises
        finally:
            self._exit()


_static: QueueReceiver = FakeReceiver([], lock=threading.Lock(), timeline=[])
"""Type-level proof that the fake satisfies the Protocol; ``mypy`` checks it."""


class FakeRuntime:
    """Stands in for :class:`~osprey.bridges.core.BridgeRuntime`.

    Only the two members the loop is allowed to drive exist. Each handled event
    blocks on its own :class:`threading.Event` until the test releases it, which
    is how a test holds ``POOL_SIZE`` handlers in flight at once; ``blocked``
    is signalled every time a handler parks so the test can wait for that
    moment instead of guessing at it.
    """

    def __init__(
        self,
        *,
        timeline: list[tuple[str, str]],
        raises: BaseException | None = None,
        hold: bool = False,
    ):
        self.timeline = timeline
        self.raises = raises
        self.hold = hold
        self.events: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []
        self.supervised = 0
        self.on_supervise: Callable[[], None] | None = None
        self.releases: dict[str, threading.Event] = {}
        self.blocked = threading.Semaphore(0)
        self._guard = threading.Lock()

    def handle_event(self, event: Any) -> str:
        mid = event["id"]
        with self._guard:
            self.events.append(event)
            self.threads.append(threading.current_thread())
            release = self.releases.setdefault(mid, threading.Event())
        self.timeline.append(("handle", mid))
        if self.hold:
            self.blocked.release()
            assert release.wait(JOIN_TIMEOUT), f"handler {mid} was never released"
        if self.raises is not None:
            raise self.raises
        return "handled"

    def release(self, mid: str) -> None:
        with self._guard:
            self.releases.setdefault(mid, threading.Event()).set()

    def wait_blocked(self, count: int) -> None:
        for _ in range(count):
            assert self.blocked.acquire(timeout=JOIN_TIMEOUT), "handlers never parked"

    def supervise(self) -> None:
        self.supervised += 1
        if self.on_supervise is not None:
            self.on_supervise()


class FakeClock:
    """A monotonic clock the test advances; every read is counted.

    ``step`` seconds elapse per read. The read count doubles as an iteration
    counter for the serve loop, which reads the clock once per pass.
    """

    def __init__(self, step: float = 0.0):
        self.step = step
        self.reads = 0
        self.now = 1000.0
        self._guard = threading.Lock()
        self.read = threading.Condition(self._guard)

    def __call__(self) -> float:
        with self._guard:
            self.reads += 1
            self.now += self.step
            self.read.notify_all()
            return self.now

    def wait_reads(self, target: int) -> None:
        with self._guard:
            assert self.read.wait_for(lambda: self.reads >= target, timeout=JOIN_TIMEOUT), (
                f"loop never reached {target} clock reads"
            )


class ServeThread:
    """Run :func:`serve` on a thread and record when it returned on the timeline."""

    def __init__(
        self,
        receiver: FakeReceiver,
        runtime: FakeRuntime,
        *,
        stop: threading.Event,
        lock: threading.Lock,
        clock: FakeClock | None = None,
        wake_interval: float = WAKE_INTERVAL,
    ):
        self.error: BaseException | None = None
        self.timeline = receiver.timeline

        def target() -> None:
            try:
                serve(
                    receiver,
                    runtime,
                    stop=stop,
                    wake_interval=wake_interval,
                    slot_timeout=TEST_SLOT_TIMEOUT,
                    clock=clock if clock is not None else FakeClock(),
                    lock=lock,
                )
            except BaseException as exc:  # re-raised by join()
                self.error = exc
            finally:
                self.timeline.append(("serve_returned", ""))

        self.thread = threading.Thread(target=target, name="serve-under-test", daemon=True)
        self.thread.start()

    def join(self) -> None:
        self.thread.join(JOIN_TIMEOUT)
        assert not self.thread.is_alive(), "serve did not return"
        if self.error is not None:
            raise self.error


@pytest.fixture
def lock() -> threading.Lock:
    return threading.Lock()


@pytest.fixture
def timeline() -> list[tuple[str, str]]:
    return []


def stop_after(stop: threading.Event, wakes: int) -> Callable[[], None]:
    """A supervise hook that sets ``stop`` on the ``wakes``-th wake, as a signal
    handler would — from outside the loop's own control flow."""
    seen = 0

    def hook() -> None:
        nonlocal seen
        seen += 1
        if seen >= wakes:
            stop.set()

    return hook


def stop_when_completed(stop: threading.Event, receiver: FakeReceiver, count: int) -> None:
    """Spin the serve loop until ``count`` messages are settled, then stop it.

    Runs on the test thread: waits on the receiver's ``pulled`` signal (set on
    every pull) and re-checks the settled count each time, so no time-based
    wait is involved.
    """
    while len(receiver.completed) + len(receiver.dead_lettered) < count:
        assert receiver.pulled.wait(JOIN_TIMEOUT), "loop stopped pulling"
        receiver.pulled.clear()
    stop.set()


# --- the seam ---------------------------------------------------------------


class TestConstants:
    def test_the_documented_constants_are_what_the_proposal_pins(self) -> None:
        assert POOL_SIZE == 4
        assert SETTLE_MARGIN_SEC == 120
        assert WAKE_INTERVAL == 60.0
        assert RECEIVE_WAIT_SEC == 5.0
        assert SLOT_TIMEOUT_SEC == 1.0

    def test_the_fake_satisfies_the_runtime_protocol_check(self) -> None:
        assert isinstance(_static, QueueReceiver)


# --- body decoding ----------------------------------------------------------


class TestDecodeBody:
    def test_a_body_of_byte_sections_is_joined_and_decoded(self) -> None:
        msg = FakeMessage("m1")
        assert decode_body(msg) == {"type": "message", "id": "m1"}

    def test_a_single_bytes_body_is_decoded(self) -> None:
        msg = FakeMessage("m1", b'{"a": 1}', sections=False)
        assert decode_body(msg) == {"a": 1}

    def test_a_str_body_is_decoded(self) -> None:
        msg = FakeMessage("m1", '{"a": 1}', sections=False)
        assert decode_body(msg) == {"a": 1}

    def test_a_mapping_body_is_returned_unchanged(self) -> None:
        msg = FakeMessage("m1", {"a": 1}, sections=False)
        assert decode_body(msg) == {"a": 1}

    def test_a_sequence_of_str_sections_is_joined(self) -> None:
        msg = FakeMessage("m1", ['{"a"', ": 1}"], sections=False)
        assert decode_body(msg) == {"a": 1}

    @pytest.mark.parametrize(
        "body",
        [b"not json", b"[1, 2]", b"42", b"\xff\xfe", 17, None, [b"{", 3]],
        ids=["text", "array", "number", "bad-utf8", "int", "none", "mixed-sections"],
    )
    def test_anything_that_is_not_a_json_object_is_rejected(self, body: Any) -> None:
        with pytest.raises(ValueError):
            decode_body(FakeMessage("m1", body, sections=False))

    def test_a_message_whose_body_cannot_even_be_read_is_rejected(self) -> None:
        class Unreadable:
            @property
            def body(self) -> Any:
                raise OSError("link detached")

        with pytest.raises(ValueError):
            decode_body(Unreadable())

    def test_a_message_without_a_body_attribute_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            decode_body(object())


# --- the serve loop ---------------------------------------------------------


def settled(ids: Iterable[str]) -> list[str]:
    """Message ids from a settlement or dispatch record, order-normalised.

    With two messages in flight they are handled on two pool threads, so which
    of them finishes first is the scheduler's business and not a property
    :func:`serve` promises. Sorting keeps the assertion a multiset equality —
    every message present exactly once — and drops only the ordering nothing
    guarantees. The ordering that *is* guaranteed is per message (register
    before the handler, completion after it) and is pinned on the timeline,
    where the guarantee actually lives.
    """
    return sorted(ids)


class TestServe:
    def test_a_stop_event_set_before_serving_never_pulls(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        stop = threading.Event()
        stop.set()
        receiver = FakeReceiver([FakeMessage("m1")], lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline)

        ServeThread(receiver, runtime, stop=stop, lock=lock).join()

        assert receiver.receive_calls == []
        assert runtime.events == []
        assert runtime.supervised == 0

    def test_every_pull_asks_for_one_message_with_the_documented_wait(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        stop = threading.Event()
        receiver = FakeReceiver(
            [FakeMessage("m1"), FakeMessage("m2")], lock=lock, timeline=timeline
        )
        runtime = FakeRuntime(timeline=timeline)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        stop_when_completed(stop, receiver, 2)
        serving.join()

        assert receiver.receive_calls, "nothing was pulled"
        assert set(receiver.receive_calls) == {(1, RECEIVE_WAIT_SEC)}
        assert settled(receiver.completed) == ["m1", "m2"]

    def test_register_precedes_and_complete_follows_the_handler_for_every_message(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        stop = threading.Event()
        receiver = FakeReceiver(
            [FakeMessage("m1"), FakeMessage("m2")], lock=lock, timeline=timeline
        )
        runtime = FakeRuntime(timeline=timeline)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        stop_when_completed(stop, receiver, 2)
        serving.join()

        for mid in ("m1", "m2"):
            steps = [step for step, who in timeline if who == mid]
            assert steps == ["register", "handle", "complete"], (mid, timeline)
        assert settled(e["id"] for e in runtime.events) == ["m1", "m2"]

    def test_the_handler_runs_off_the_loop_thread_and_completion_waits_for_it(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        """While the handler is parked nothing has been completed: settlement is
        after the handler returns, which is what holds the lock for the whole
        run rather than the first instant of it."""
        stop = threading.Event()
        receiver = FakeReceiver([FakeMessage("m1")], lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline, hold=True)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        runtime.wait_blocked(1)
        assert receiver.registered == ["m1"]
        assert receiver.completed == []
        assert runtime.threads[0] is not serving.thread

        runtime.release("m1")
        stop_when_completed(stop, receiver, 1)
        serving.join()

        assert receiver.completed == ["m1"]

    def test_with_pool_size_handlers_in_flight_no_further_pull_is_made_until_one_returns(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        """The pull is gated on a free slot, never on a count. With every
        handler parked the loop keeps going round (the clock proves it) but
        never asks the receiver for a message; releasing one handler is what
        allows the next pull."""
        stop = threading.Event()
        messages = [FakeMessage(f"m{i}") for i in range(1, POOL_SIZE + 3)]
        receiver = FakeReceiver(messages, lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline, hold=True)
        clock = FakeClock()
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock, clock=clock)

        runtime.wait_blocked(POOL_SIZE)
        pulls_at_saturation = len(receiver.receive_calls)
        assert pulls_at_saturation == POOL_SIZE
        # Let the loop go round at least twenty more times with every slot taken.
        clock.wait_reads(clock.reads + 20)
        assert len(receiver.receive_calls) == pulls_at_saturation

        runtime.release("m1")
        runtime.wait_blocked(1)  # m5 is now parked, which means it was pulled
        assert len(receiver.receive_calls) == POOL_SIZE + 1
        assert receiver.completed == ["m1"]

        for mid in ("m2", "m3", "m4", "m5"):
            runtime.release(mid)
        runtime.wait_blocked(1)  # m6
        runtime.release("m6")
        stop_when_completed(stop, receiver, POOL_SIZE + 2)
        serving.join()

        assert sorted(receiver.completed) == sorted(m.id for m in messages)
        assert receiver.max_overlap == 1

    def test_an_empty_batch_frees_the_slot_for_the_next_pull(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        """An empty pull must give its slot back, or four empty pulls in a row
        would leave a bridge that never pulls again."""
        stop = threading.Event()
        receiver = FakeReceiver([], lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        for _ in range(POOL_SIZE * 3):
            assert receiver.pulled.wait(JOIN_TIMEOUT)
            receiver.pulled.clear()
        stop.set()
        serving.join()

        assert len(receiver.receive_calls) >= POOL_SIZE * 3
        assert runtime.events == []

    def test_an_undecodable_body_is_dead_lettered_under_the_lock_and_never_handled(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        stop = threading.Event()
        poison = FakeMessage("bad", b"not json", sections=False)
        receiver = FakeReceiver([poison, FakeMessage("m1")], lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        stop_when_completed(stop, receiver, 2)
        serving.join()

        assert [mid for mid, _ in receiver.dead_lettered] == ["bad"]
        assert receiver.dead_lettered[0][1]
        assert "bad" not in receiver.registered
        assert "bad" not in receiver.completed
        assert [e["id"] for e in runtime.events] == ["m1"]
        assert receiver.completed == ["m1"]

    def test_a_raising_handler_still_completes_and_does_not_kill_the_loop(
        self,
        lock: threading.Lock,
        timeline: list[tuple[str, str]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        stop = threading.Event()
        receiver = FakeReceiver(
            [FakeMessage("m1"), FakeMessage("m2")], lock=lock, timeline=timeline
        )
        runtime = FakeRuntime(timeline=timeline, raises=RuntimeError("engine blew up"))
        with caplog.at_level(logging.ERROR, logger="osprey.bridges.teams.receiver"):
            serving = ServeThread(receiver, runtime, stop=stop, lock=lock)
            stop_when_completed(stop, receiver, 2)
            serving.join()

        assert settled(receiver.completed) == ["m1", "m2"]
        assert any("engine blew up" in (r.exc_text or "") for r in caplog.records)

    @pytest.mark.parametrize("error", [MessageLockLostError, ServiceBusError])
    def test_a_service_bus_error_on_complete_is_logged_not_raised(
        self,
        error: type[Exception],
        lock: threading.Lock,
        timeline: list[tuple[str, str]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Matched by class name: the SDK's exception classes are never
        imported here. A lost lock means the message is redelivered, and the
        engine's claim de-duplicates the redelivery, so nothing is owed."""
        stop = threading.Event()
        receiver = FakeReceiver(
            [FakeMessage("m1"), FakeMessage("m2")],
            lock=lock,
            timeline=timeline,
            settle_raises={"m1": error("lock expired")},
        )
        runtime = FakeRuntime(timeline=timeline)
        with caplog.at_level(logging.WARNING, logger="osprey.bridges.teams.receiver"):
            serving = ServeThread(receiver, runtime, stop=stop, lock=lock)
            stop_when_completed(stop, receiver, 2)
            serving.join()

        assert settled(receiver.completed) == ["m1", "m2"]
        assert settled(e["id"] for e in runtime.events) == ["m1", "m2"]
        assert any(error.__name__ in r.getMessage() for r in caplog.records)

    def test_an_unexpected_error_on_complete_is_logged_with_its_traceback_not_raised(
        self,
        lock: threading.Lock,
        timeline: list[tuple[str, str]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Settlement runs on a pool thread; an exception escaping there is
        swallowed by the pool, so logging it is the only way it is ever seen."""
        stop = threading.Event()
        receiver = FakeReceiver(
            [FakeMessage("m1"), FakeMessage("m2")],
            lock=lock,
            timeline=timeline,
            settle_raises={"m1": KeyError("not a service bus error")},
        )
        runtime = FakeRuntime(timeline=timeline)
        with caplog.at_level(logging.ERROR, logger="osprey.bridges.teams.receiver"):
            serving = ServeThread(receiver, runtime, stop=stop, lock=lock)
            stop_when_completed(stop, receiver, 2)
            serving.join()

        assert settled(receiver.completed) == ["m1", "m2"]
        assert any("not a service bus error" in (r.exc_text or "") for r in caplog.records)

    def test_a_receiver_that_dies_on_pull_propagates_out_of_serve(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        """A bridge that has gone deaf must fail loudly and be restarted, not
        sit in a loop looking alive."""
        stop = threading.Event()
        receiver = FakeReceiver([FakeMessage("m1")], lock=lock, timeline=timeline, die_on=2)
        runtime = FakeRuntime(timeline=timeline)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        with pytest.raises(RuntimeError, match="receiver died"):
            serving.join()

        assert receiver.completed == ["m1"]

    def test_a_dying_receiver_still_lets_in_flight_handlers_settle_before_serve_returns(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        stop = threading.Event()
        receiver = FakeReceiver([FakeMessage("m1")], lock=lock, timeline=timeline, die_on=2)
        runtime = FakeRuntime(timeline=timeline, hold=True)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        runtime.wait_blocked(1)
        runtime.release("m1")
        with pytest.raises(RuntimeError, match="receiver died"):
            serving.join()

        assert timeline.index(("complete", "m1")) < timeline.index(("serve_returned", ""))

    def test_a_failing_register_propagates_out_of_serve(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        """An unregistered message would outlive its lock mid-handling; that is
        the receiver-side machinery failing, and the loop treats it as such."""
        stop = threading.Event()
        receiver = FakeReceiver(
            [FakeMessage("m1")],
            lock=lock,
            timeline=timeline,
            register_raises=RuntimeError("renewer closed"),
        )
        runtime = FakeRuntime(timeline=timeline)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        with pytest.raises(RuntimeError, match="renewer closed"):
            serving.join()

        assert runtime.events == []
        assert receiver.completed == []

    def test_a_shutdown_waits_for_in_flight_handlers_to_settle(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        stop = threading.Event()
        messages = [FakeMessage(f"m{i}") for i in range(1, POOL_SIZE + 1)]
        receiver = FakeReceiver(messages, lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline, hold=True)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock)

        runtime.wait_blocked(POOL_SIZE)
        stop.set()
        for m in messages:
            runtime.release(m.id)
        serving.join()

        returned = timeline.index(("serve_returned", ""))
        for m in messages:
            assert timeline.index(("complete", m.id)) < returned
        assert sorted(receiver.completed) == sorted(m.id for m in messages)

    def test_the_loop_supervises_the_drain_every_wake_interval(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        """Wakes are driven by the injected clock: each read advances it by a
        full interval, so every pass is a wake, and the third wake stops the
        loop from inside — as a signal handler would — proving the hook runs on
        the loop's own thread between pulls."""
        stop = threading.Event()
        receiver = FakeReceiver([], lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline)
        runtime.on_supervise = stop_after(stop, 3)
        clock = FakeClock(step=WAKE_INTERVAL)
        serving = ServeThread(
            receiver, runtime, stop=stop, lock=lock, clock=clock, wake_interval=WAKE_INTERVAL
        )

        serving.join()

        assert runtime.supervised == 3

    def test_no_wake_happens_before_an_interval_has_elapsed(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        stop = threading.Event()
        receiver = FakeReceiver([FakeMessage("m1")], lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline)
        clock = FakeClock(step=0.0)
        serving = ServeThread(receiver, runtime, stop=stop, lock=lock, clock=clock)

        # Let the loop go round several more times (a pull each) before stopping
        # it; the clock never advances, so none of those passes may wake.
        clock.wait_reads(clock.reads + 5)
        stop_when_completed(stop, receiver, 1)
        serving.join()

        assert runtime.supervised == 0

    def test_serve_uses_the_documented_wake_interval_by_default(
        self, lock: threading.Lock, timeline: list[tuple[str, str]]
    ) -> None:
        """The clock advances just under one interval per read, so a wake on
        the second read proves the default is ``WAKE_INTERVAL`` and not shorter."""
        stop = threading.Event()
        receiver = FakeReceiver([], lock=lock, timeline=timeline)
        runtime = FakeRuntime(timeline=timeline)
        clock = FakeClock(step=WAKE_INTERVAL / 2)
        start = clock.now
        woke_at: list[float] = []

        def on_supervise() -> None:
            woke_at.append(clock.now)
            stop.set()

        runtime.on_supervise = on_supervise

        def target() -> None:
            serve(
                receiver, runtime, stop=stop, slot_timeout=TEST_SLOT_TIMEOUT, clock=clock, lock=lock
            )

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(JOIN_TIMEOUT)
        assert not thread.is_alive()

        # The first read arms the deadline (+30 s); the wake fires on the first
        # read at or past one full interval after that, i.e. at +90 s. A shorter
        # default would have fired at +60 s.
        assert woke_at == [start + clock.step + WAKE_INTERVAL]


# --- the real receiver ------------------------------------------------------


CONNECTION_STRING = (
    "Endpoint=sb://example.servicebus.windows.net/;"
    "SharedAccessKeyName=bridge-listen;SharedAccessKey=secret"
)
QUEUE = "teams-activities"

# Deliberately not `CoreConfig`'s default: a renewal duration hard-coded to the
# 450 s the default budget happens to produce would pass every assertion below
# if they were written against it.
POLL_BUDGET = 900.0

PEEK_LOCK = "stub-peek-lock"
"""The stub module's stand-in for ``ServiceBusReceiveMode.PEEK_LOCK``.

An opaque sentinel on purpose. Asserting the factory passed *this object*
asserts it read the mode off the SDK's enum; a hand-written ``"peeklock"`` in
the source would satisfy an assertion against the string and would silently
stop meaning peek-lock the day the SDK spelled it differently."""


def make_cfg(poll_budget: float = POLL_BUDGET) -> TeamsBridgeConfig:
    """A config carrying the three fields ``make_receiver`` reads."""
    return TeamsBridgeConfig(
        servicebus_connection_string=CONNECTION_STRING,
        servicebus_queue=QUEUE,
        core=CoreConfig(poll_budget=poll_budget),
    )


class StubSDKReceiver:
    """Stands in for the SDK's ``ServiceBusReceiver``, recording every call made on it."""

    def __init__(self) -> None:
        self.messages: list[Any] = []
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False
        self.settle_raises: BaseException | None = None

    def receive_messages(
        self, max_message_count: int | None = None, max_wait_time: float | None = None
    ) -> list[Any]:
        self.calls.append(("receive_messages", max_message_count, max_wait_time))
        return self.messages

    def complete_message(self, message: Any) -> None:
        self.calls.append(("complete_message", message))
        if self.settle_raises is not None:
            raise self.settle_raises

    def dead_letter_message(self, message: Any, reason: str | None = None) -> None:
        self.calls.append(("dead_letter_message", message, reason))
        if self.settle_raises is not None:
            raise self.settle_raises

    def close(self) -> None:
        self.closed = True


class StubAutoLockRenewer:
    """Stands in for the SDK's ``AutoLockRenewer``, keeping the kwargs it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.registered: list[tuple[Any, Any]] = []
        self.closed = False

    def register(self, receiver: Any, renewable: Any) -> None:
        self.registered.append((receiver, renewable))

    def close(self, wait: bool = True) -> None:
        self.closed = True


class StubSDKClient:
    """Stands in for the SDK's ``ServiceBusClient``."""

    def __init__(self, conn_str: str, kwargs: dict[str, Any]) -> None:
        self.conn_str = conn_str
        self.kwargs = kwargs
        self.receiver_kwargs: dict[str, Any] | None = None
        self.receiver = StubSDKReceiver()
        self.closed = False

    def get_queue_receiver(self, **kwargs: Any) -> StubSDKReceiver:
        self.receiver_kwargs = kwargs
        return self.receiver

    def close(self) -> None:
        self.closed = True


class StubServiceBus:
    """A stand-in ``azure.servicebus`` module, plus the record of what was asked of it.

    The stub is what makes these tests a proof rather than a restatement: the
    renewal duration and the receive mode are read back from the call the
    factory made, so the numbers can only agree by the factory having computed
    them. Nothing here dials anything, and nothing imports the real SDK — the
    module this builds is installed in ``sys.modules`` under the SDK's name.
    """

    def __init__(self) -> None:
        self.clients: list[StubSDKClient] = []
        self.renewers: list[StubAutoLockRenewer] = []
        record = self

        class ServiceBusClient(StubSDKClient):
            @classmethod
            def from_connection_string(cls, conn_str: str, **kwargs: Any) -> Any:
                client = cls(conn_str, kwargs)
                record.clients.append(client)
                return client

        class AutoLockRenewer(StubAutoLockRenewer):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                record.renewers.append(self)

        module = ModuleType("azure.servicebus")
        module.ServiceBusClient = ServiceBusClient  # type: ignore[attr-defined]
        module.AutoLockRenewer = AutoLockRenewer  # type: ignore[attr-defined]
        module.ServiceBusReceiveMode = SimpleNamespace(  # type: ignore[attr-defined]
            PEEK_LOCK=PEEK_LOCK
        )
        self.module = module

    @property
    def client(self) -> StubSDKClient:
        """The one client the factory built; an assertion if it built another."""
        assert len(self.clients) == 1, f"expected one client, got {len(self.clients)}"
        return self.clients[0]

    @property
    def renewer(self) -> StubAutoLockRenewer:
        """The one renewer the factory built; an assertion if it built another."""
        assert len(self.renewers) == 1, f"expected one renewer, got {len(self.renewers)}"
        return self.renewers[0]


@pytest.fixture
def servicebus(monkeypatch: pytest.MonkeyPatch) -> StubServiceBus:
    """Install the stub in place of ``azure.servicebus`` for the duration of a test."""
    stub = StubServiceBus()
    azure = ModuleType("azure")
    azure.__path__ = []  # type: ignore[attr-defined]
    azure.servicebus = stub.module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.servicebus", stub.module)
    return stub


class TestMakeReceiver:
    """What the factory builds, read back off the stub it built it from."""

    def test_it_opens_the_configured_queue(self, servicebus: StubServiceBus) -> None:
        make_receiver(make_cfg())

        assert servicebus.client.conn_str == CONNECTION_STRING
        assert servicebus.client.receiver_kwargs == {
            "queue_name": QUEUE,
            "receive_mode": PEEK_LOCK,
        }

    def test_the_receiver_is_peek_lock(self, servicebus: StubServiceBus) -> None:
        # Receive-and-delete would settle every activity the instant it was
        # pulled: a crash between the pull and the answer would lose the
        # question outright instead of leaving it to be redelivered.
        make_receiver(make_cfg())

        assert servicebus.client.receiver_kwargs is not None
        assert servicebus.client.receiver_kwargs["receive_mode"] is PEEK_LOCK

    def test_the_lock_is_renewed_past_the_poll_budget(self, servicebus: StubServiceBus) -> None:
        # The headline. A handler holds its message for the whole dispatch wait
        # and then posts the answer, so renewal has to outlast the budget by the
        # settlement margin or a long run loses its lock mid-answer.
        make_receiver(make_cfg(poll_budget=900.0))

        assert servicebus.renewer.kwargs["max_lock_renewal_duration"] == 900.0 + SETTLE_MARGIN_SEC

    def test_the_renewal_duration_follows_the_configured_budget(
        self, servicebus: StubServiceBus
    ) -> None:
        # The other half of the test above: the duration moves with the config
        # rather than being a constant that happens to match one budget.
        make_receiver(make_cfg(poll_budget=1800.0))

        assert servicebus.renewer.kwargs["max_lock_renewal_duration"] == 1800.0 + SETTLE_MARGIN_SEC

    def test_the_renewer_has_a_thread_per_handler_and_one_to_spare(
        self, servicebus: StubServiceBus
    ) -> None:
        # Every handler can be holding a message whose lock is being renewed;
        # the spare keeps a renewal from queueing behind the full pool.
        make_receiver(make_cfg())

        assert servicebus.renewer.kwargs["max_workers"] == POOL_SIZE + 1

    @pytest.mark.usefixtures("servicebus")
    def test_what_it_returns_satisfies_the_seam(self) -> None:
        assert isinstance(make_receiver(make_cfg()), QueueReceiver)

    def test_building_pulls_nothing(self, servicebus: StubServiceBus) -> None:
        # The factory is called at wiring time, before the engine has started
        # and before crash recovery has run; a pull here would take delivery of
        # an activity with nothing yet able to answer it.
        make_receiver(make_cfg())

        assert servicebus.client.receiver.calls == []


class TestTheReceiverAdapter:
    """The Protocol members, each mapped onto the SDK call that implements it."""

    def test_receive_maps_onto_the_sdk_pull(self, servicebus: StubServiceBus) -> None:
        # `serve` calls receive(1, RECEIVE_WAIT_SEC); the SDK spells the same two
        # arguments `max_message_count` and `max_wait_time`.
        receiver = make_receiver(make_cfg())
        sdk = servicebus.client.receiver
        sdk.messages = [FakeMessage("a")]

        pulled = receiver.receive(1, RECEIVE_WAIT_SEC)

        assert pulled == sdk.messages
        assert sdk.calls == [("receive_messages", 1, RECEIVE_WAIT_SEC)]

    def test_the_message_is_handed_back_untouched(self, servicebus: StubServiceBus) -> None:
        # `serve` reads only `msg.body`, and the object it settles is the object
        # it was given: anything the adapter wrapped around a message would have
        # to be unwrapped again before the SDK could settle it.
        receiver = make_receiver(make_cfg())
        message = FakeMessage("a")
        servicebus.client.receiver.messages = [message]

        assert receiver.receive(1, RECEIVE_WAIT_SEC)[0] is message

    def test_complete_settles_the_message(self, servicebus: StubServiceBus) -> None:
        receiver = make_receiver(make_cfg())
        message = FakeMessage("a")

        receiver.complete(message)

        assert servicebus.client.receiver.calls == [("complete_message", message)]

    def test_dead_letter_carries_the_reason(self, servicebus: StubServiceBus) -> None:
        # The reason is the only thing an operator staring at the dead-letter
        # queue has to go on, so it has to reach the broker, not just the log.
        receiver = make_receiver(make_cfg())
        message = FakeMessage("a")

        receiver.dead_letter(message, "undecodable body: not JSON")

        assert servicebus.client.receiver.calls == [
            ("dead_letter_message", message, "undecodable body: not JSON")
        ]

    def test_register_renews_this_message_on_this_receiver(
        self, servicebus: StubServiceBus
    ) -> None:
        # The renewer renews a lock by asking the receiver that holds it, so it
        # needs both halves; handing it the message alone renews nothing.
        receiver = make_receiver(make_cfg())
        message = FakeMessage("a")

        receiver.register(message)

        assert servicebus.renewer.registered == [(servicebus.client.receiver, message)]

    @pytest.mark.parametrize("settle", ["complete", "dead_letter"])
    def test_settlement_errors_reach_the_loop_unwrapped(
        self, servicebus: StubServiceBus, settle: str
    ) -> None:
        # `serve` classifies a settlement failure by walking the exception's MRO
        # for the SDK class names. An adapter-owned wrapper around it would turn
        # the one-line "the broker will redeliver" warning into a traceback.
        receiver = make_receiver(make_cfg())
        lost = MessageLockLostError("the lock expired")
        servicebus.client.receiver.settle_raises = lost
        args: tuple[Any, ...] = (FakeMessage("a"),)
        if settle == "dead_letter":
            args = (*args, "undecodable body: not JSON")

        with pytest.raises(MessageLockLostError) as raised:
            getattr(receiver, settle)(*args)

        assert raised.value is lost

    def test_close_shuts_down_the_renewer_the_receiver_and_the_client(
        self, servicebus: StubServiceBus
    ) -> None:
        # All three, in that order: a renewal thread still running against a
        # closed receiver has nothing to renew and something to raise about.
        receiver = make_receiver(make_cfg())

        receiver.close()  # type: ignore[attr-defined]

        assert servicebus.renewer.closed
        assert servicebus.client.receiver.closed
        assert servicebus.client.closed


# --- with no azure library installed -----------------------------------------


def load_receiver_copy() -> Any:
    """Load a fresh, private copy of the ingestion module.

    A copy rather than a reload, so blocking imports for one test cannot leave
    the canonical module rebound for the rest of the suite. The alias keeps the
    package prefix, which is what makes its ``from .config import ...`` resolve.
    """
    alias = "osprey.bridges.teams._receiver_import_probe"
    spec = importlib.util.spec_from_file_location(alias, receiver_module.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reimport_adapter_package(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Drop the adapter package from the module cache and import it again.

    Both halves of the damage are undone on teardown: ``delitem`` puts the
    cached modules back, and the ``setattr`` puts back the attribute the fresh
    import rebinds on ``osprey.bridges`` — without it the package reachable by
    attribute and the one in ``sys.modules`` would be two different objects for
    the rest of the session.
    """
    import osprey.bridges as bridges

    cached = sys.modules["osprey.bridges.teams"]
    for name in list(sys.modules):
        if name == "osprey.bridges.teams" or name.startswith("osprey.bridges.teams."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(bridges, "teams", cached)
    return importlib.import_module("osprey.bridges.teams")


class TestWithoutTheAzureLibrary:
    """The adapter on a machine with no ``teams`` extra installed.

    Which is every dev checkout, every ``osprey build`` run rendering the
    compose template, and most of CI. A module-level ``azure.servicebus`` import
    anywhere under the package would make all of those fail at import.
    """

    @pytest.mark.usefixtures("no_servicebus")
    def test_the_block_is_real(self) -> None:
        # Proof the three tests below are not passing vacuously on a machine
        # that has azure-servicebus installed after all.
        with pytest.raises(ImportError, match="blocked in this test"):
            importlib.import_module("azure.servicebus")

    @pytest.mark.usefixtures("no_servicebus")
    def test_the_adapter_package_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fresh = reimport_adapter_package(monkeypatch)

        assert fresh.TeamsBridgeConfig is not None

    @pytest.mark.usefixtures("no_servicebus")
    def test_the_ingestion_module_imports_and_works(self) -> None:
        fresh = load_receiver_copy()

        assert fresh.decode_body(FakeMessage("a")) == {"type": "message", "id": "a"}

    @pytest.mark.usefixtures("no_servicebus")
    def test_make_receiver_names_the_extra_to_install(self) -> None:
        # The one function that needs the SDK, and the one place an operator
        # meets the missing extra: the message has to say which extra it is.
        fresh = load_receiver_copy()

        with pytest.raises(ImportError, match="teams"):
            fresh.make_receiver(make_cfg())
