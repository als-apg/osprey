"""RoomPoller: the ingestion loop's offset semantics, hold, and refusal to die.

Almost every test here drives :meth:`RoomPoller.poll_once` directly — one cycle,
on the calling thread, no timing at all. That is deliberate: the properties worth
pinning are about *ordering* (when the persisted offset moves relative to
``handle_event``) and a sleep-based test could only ever approximate them. The
handful of tests that genuinely need a thread synchronise on an
:class:`threading.Event` the fake Talk sets, never on elapsed time, and every
thread is stopped and joined before the test returns.

The failures these guard against are all silent in production:

* a persisted offset that advances past an unhandled message — the message is
  gone after a crash, with nothing in any log to say so;
* an in-memory cursor that advances past a *failed* message — the persisted
  offset looks correct, yet the running process never re-fetches, so the message
  is answered only if the bridge happens to restart;
* a room thread that exits on an exception — that room goes deaf permanently
  while the process and every other room look healthy;
* a message consumed while its room's type was still unresolved.
"""

import threading

import httpx
import pytest

from osprey.bridges.core import BridgeRuntime
from osprey.bridges.nextcloud_talk import (
    NextcloudBridgeConfig,
    OffsetStore,
    RoomDirectory,
    RoomTypeUnresolved,
    TalkClient,
)
from osprey.bridges.nextcloud_talk.client import LAST_GIVEN_HEADER
from osprey.bridges.nextcloud_talk.poller import RoomPoller
from osprey.bridges.nextcloud_talk.rooms import BACKOFF_CAP

ROOM = "roomA"

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account="osprey-bot",
    app_password="app-pw",
    rooms=(ROOM,),
)

JOIN_TIMEOUT = 10.0
"""Generous upper bound for a thread to notice the stop event. Never used as a
delay — only as the point at which a hung thread fails the test instead of
hanging the suite."""


def _message(message_id, text="hi"):
    """A minimal Talk chat message. The poller passes these through untouched."""
    return {"id": message_id, "token": ROOM, "actorId": "alice", "message": text}


def _ocs(data, *, last_given=None):
    headers = {} if last_given is None else {LAST_GIVEN_HEADER: str(last_given)}
    return httpx.Response(
        200,
        json={"ocs": {"meta": {"status": "ok", "statuscode": 200}, "data": data}},
        headers=headers,
    )


class FakeTalk:
    """A scripted Talk: a room payload plus a queue of chat answers.

    Each queued chat answer is one of
      * ``list[dict]`` — a batch of messages (with an optional ``last_given``
        cursor when given as ``(messages, last_given)``),
      * ``None`` — Talk's ``304``, the idle outcome,
      * an ``Exception`` instance — raised, standing in for a transport failure.

    Once the queue is empty every further poll answers ``304``, so a loop under
    test idles instead of replaying the script.
    """

    def __init__(self, *, room_type=2, last_message_id=None, chat=(), room_failures=0):
        self.room_type = room_type
        self.last_message_id = last_message_id
        self.chat = list(chat)
        self.room_failures = room_failures
        self.requested = []
        """``lastKnownMessageId`` of every chat poll, in order — the window each
        poll actually asked for."""
        self.room_calls = 0
        self.reached = threading.Event()
        """Set once :attr:`stop_after` chat polls have been answered."""
        self.stop_after = None

    def handler(self, request):
        if "/api/v4/room/" in request.url.path:
            self.room_calls += 1
            if self.room_calls <= self.room_failures:
                raise httpx.ConnectError("nextcloud unreachable", request=request)
            payload = {"type": self.room_type, "displayName": "Ops"}
            if self.last_message_id is not None:
                payload["lastMessage"] = {"id": self.last_message_id}
            return _ocs(payload)

        self.requested.append(int(request.url.params["lastKnownMessageId"]))
        if self.stop_after is not None and len(self.requested) >= self.stop_after:
            self.reached.set()
        answer = self.chat.pop(0) if self.chat else None
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            return httpx.Response(httpx.codes.NOT_MODIFIED)
        if isinstance(answer, tuple):
            messages, last_given = answer
            return _ocs(messages, last_given=last_given)
        return _ocs(answer)


class FakeRuntime:
    """A BridgeRuntime stand-in recording what the poller drove it with.

    Only the two members the poller is allowed to touch. ``fail_on`` maps a
    message id to an exception raised *after* the call is recorded, which is how
    "message 2 of 3 fails" is expressed.
    """

    def __init__(self, *, fail_on=None, outcome="handled"):
        self.timeline = []
        """Interleaved record of ``handle:<id>`` and ``advance:<offset>`` — the
        only way to assert the persisted offset moves after the WHOLE batch."""
        self.handled = []
        self.supervise_calls = 0
        self.fail_on = dict(fail_on or {})
        self.outcome = outcome

    def handle_event(self, event):
        self.handled.append(event)
        self.timeline.append(f"handle:{event.get('id')}")
        exc = self.fail_on.pop(event.get("id"), None)
        if exc is not None:
            raise exc
        return self.outcome

    def supervise(self):
        self.supervise_calls += 1
        return None


class RecordingOffsets(OffsetStore):
    """An OffsetStore that also writes into a shared timeline."""

    def __init__(self, path, timeline):
        super().__init__(path)
        self._timeline = timeline

    def advance(self, room, message_id):
        self._timeline.append(f"advance:{message_id}")
        super().advance(room, message_id)


def _harness(tmp_path, talk, *, runtime=None, seed=None, sleeps=None, **poller_kwargs):
    """``(poller, runtime, offsets)`` wired over ``talk``.

    The directory starts *unresolved*, so every test exercises the real hold on
    its first cycle rather than a pre-warmed cache.
    """
    runtime = runtime or FakeRuntime()
    client = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(talk.handler)))
    rooms = RoomDirectory(client, sleep=lambda _: None)
    offsets = RecordingOffsets(str(tmp_path / "offsets.json"), runtime.timeline)
    if seed is not None:
        offsets.advance(ROOM, seed)
        runtime.timeline.clear()
    poller = RoomPoller(
        ROOM,
        client=client,
        rooms=rooms,
        offsets=offsets,
        runtime=runtime,
        sleep=(sleeps.append if sleeps is not None else (lambda _: None)),
        poll_hold=0,
        **poller_kwargs,
    )
    return poller, runtime, offsets


# ==========================================================================
# Where a room starts reading
# ==========================================================================


def test_first_ever_start_skips_history_from_the_rooms_tail(tmp_path):
    talk = FakeTalk(last_message_id=7, chat=[None])
    poller, _, offsets = _harness(tmp_path, talk)

    poller.poll_once()

    # Polled from the tail Talk reported, so the room's backlog is never answered.
    assert talk.requested == [7]
    # Persisted at once: "first-ever" must mean once, not once per restart.
    assert offsets.get(ROOM) == 7


def test_first_ever_start_in_an_empty_room_starts_from_zero(tmp_path):
    talk = FakeTalk(last_message_id=None, chat=[None])
    poller, _, _ = _harness(tmp_path, talk)

    poller.poll_once()

    assert talk.requested == [0]


def test_a_persisted_offset_resumes_instead_of_skipping(tmp_path):
    talk = FakeTalk(last_message_id=999, chat=[None])
    poller, _, _ = _harness(tmp_path, talk, seed=42)

    poller.poll_once()

    # The room's current tail is irrelevant once an offset exists — resuming is
    # what stops a restart from skipping everything posted while it was down.
    assert talk.requested == [42]


def test_the_cursor_is_resolved_once_not_per_cycle(tmp_path):
    talk = FakeTalk(last_message_id=7, chat=[None, None, None])
    poller, _, _ = _harness(tmp_path, talk)

    for _ in range(3):
        poller.poll_once()

    assert talk.requested == [7, 7, 7]


# ==========================================================================
# Offset advance — the asymmetry that protects messages
# ==========================================================================


def test_the_persisted_offset_advances_only_after_the_whole_batch(tmp_path):
    talk = FakeTalk(chat=[([_message(11), _message(12), _message(13)], 13), None])
    poller, runtime, offsets = _harness(tmp_path, talk, seed=10)

    assert poller.poll_once() == 3

    # The single advance lands AFTER the last handle — no interleaving, so a
    # crash mid-batch can only ever re-fetch, never skip.
    assert runtime.timeline == ["handle:11", "handle:12", "handle:13", "advance:13"]
    assert offsets.get(ROOM) == 13


def test_messages_are_dispatched_in_talks_order(tmp_path):
    talk = FakeTalk(chat=[([_message(11), _message(12), _message(13)], 13)])
    poller, runtime, _ = _harness(tmp_path, talk, seed=10)

    poller.poll_once()

    assert [event["id"] for event in runtime.handled] == [11, 12, 13]


def test_raw_messages_are_passed_through_untouched(tmp_path):
    """The engine calls ``ops.parse_event`` itself, so the poller must hand over
    Talk's object as it arrived."""
    message = _message(11, text="what is the beam current?")
    talk = FakeTalk(chat=[([message], 11)])
    poller, runtime, _ = _harness(tmp_path, talk, seed=10)

    poller.poll_once()

    assert runtime.handled == [message]


def test_an_exception_leaves_the_persisted_offset_unadvanced(tmp_path):
    talk = FakeTalk(chat=[([_message(11), _message(12), _message(13)], 13)])
    runtime = FakeRuntime(fail_on={12: RuntimeError("dispatcher down")})
    poller, runtime, offsets = _harness(tmp_path, talk, runtime=runtime, seed=10)

    assert poller.poll_once() == 0

    assert offsets.get(ROOM) == 10
    assert "advance:13" not in runtime.timeline
    # 13 was never reached; the batch is re-fetched whole rather than resumed
    # mid-way, and dedup claims absorb 11.
    assert [event["id"] for event in runtime.handled] == [11, 12]


def test_an_exception_re_fetches_the_same_window(tmp_path):
    """The in-memory rollback. Without it the persisted offset would be correctly
    unadvanced while the running process sailed past message 12 — it would be
    answered only if the bridge happened to restart."""
    talk = FakeTalk(
        chat=[
            ([_message(11), _message(12)], 12),
            ([_message(11), _message(12)], 12),
        ]
    )
    runtime = FakeRuntime(fail_on={12: RuntimeError("dispatcher down")})
    poller, runtime, offsets = _harness(tmp_path, talk, runtime=runtime, seed=10)

    poller.poll_once()
    assert poller.poll_once() == 2

    assert talk.requested == [10, 10]
    assert offsets.get(ROOM) == 12
    assert [event["id"] for event in runtime.handled] == [11, 12, 11, 12]


def test_a_room_type_unresolved_escaping_the_engine_defers_the_message(tmp_path):
    """The end-to-end deferral. If ``parse_event`` is ever reached for a room
    whose type has not resolved it raises rather than returning None, and that
    raise has to land here as "try again", not as a consumed message."""
    talk = FakeTalk(chat=[([_message(11)], 11), ([_message(11)], 11)])
    runtime = FakeRuntime(fail_on={11: RoomTypeUnresolved("room 'roomA' has no cached type")})
    poller, runtime, offsets = _harness(tmp_path, talk, runtime=runtime, seed=10)

    poller.poll_once()
    assert offsets.get(ROOM) == 10

    assert poller.poll_once() == 1
    assert talk.requested == [10, 10]
    assert offsets.get(ROOM) == 11


@pytest.mark.parametrize("outcome", ["handled", "duplicate", "ignored"])
def test_offset_advance_is_independent_of_the_handle_event_outcome(tmp_path, outcome):
    """A claimed message that ended up parked in the retry queue still advances
    the cursor — the retry queue owns it from then on, and holding the cursor for
    it would re-deliver it forever."""
    talk = FakeTalk(chat=[([_message(11)], 11)])
    poller, _, offsets = _harness(tmp_path, talk, runtime=FakeRuntime(outcome=outcome), seed=10)

    poller.poll_once()

    assert offsets.get(ROOM) == 11


def test_an_idle_poll_advances_nothing(tmp_path):
    talk = FakeTalk(chat=[None])
    poller, runtime, offsets = _harness(tmp_path, talk, seed=10)

    assert poller.poll_once() == 0

    assert offsets.get(ROOM) == 10
    assert runtime.handled == []


# ==========================================================================
# Cursor arithmetic
# ==========================================================================


def test_talks_own_cursor_hint_is_preferred(tmp_path):
    """``X-Chat-Last-Given`` accounts for messages Talk considered but did not
    return, so it may legitimately run ahead of the batch's highest id."""
    talk = FakeTalk(chat=[([_message(11)], 20), None])
    poller, _, offsets = _harness(tmp_path, talk, seed=10)

    poller.poll_once()
    poller.poll_once()

    assert offsets.get(ROOM) == 20
    assert talk.requested == [10, 20]


def test_the_highest_id_is_the_fallback_when_talk_sends_no_cursor(tmp_path):
    talk = FakeTalk(chat=[[_message(11), _message(13), _message(12)]])
    poller, _, offsets = _harness(tmp_path, talk, seed=10)

    poller.poll_once()

    assert offsets.get(ROOM) == 13


def test_a_cursor_hint_behind_the_request_cannot_loop_forever(tmp_path):
    """A server answering with a cursor behind the one asked for would otherwise
    make the same batch arrive on every poll."""
    talk = FakeTalk(chat=[([_message(11)], 5), None])
    poller, _, offsets = _harness(tmp_path, talk, seed=10)

    poller.poll_once()
    poller.poll_once()

    assert offsets.get(ROOM) == 10
    assert talk.requested == [10, 10]


def test_a_message_without_a_usable_id_is_still_dispatched(tmp_path):
    talk = FakeTalk(chat=[[{"token": ROOM, "message": "hi"}, _message(11)]])
    poller, runtime, offsets = _harness(tmp_path, talk, seed=10)

    assert poller.poll_once() == 2

    assert len(runtime.handled) == 2
    assert offsets.get(ROOM) == 11


# ==========================================================================
# The room-type hold
# ==========================================================================


def test_a_held_room_is_not_polled_at_all(tmp_path):
    talk = FakeTalk(room_failures=99, chat=[[_message(11)]])
    poller, runtime, offsets = _harness(tmp_path, talk)

    assert poller.poll_once() == 0

    # No chat request while the type is unknown, so nothing was even read —
    # let alone judged by a mention filter that could not know the room type.
    assert talk.requested == []
    assert runtime.handled == []
    assert offsets.get(ROOM) is None


def test_messages_arriving_before_the_type_resolves_are_dispatched_after(tmp_path):
    """Deferred, never dropped: the batch waits on the server because the cursor
    never moved, and arrives whole on the cycle after the type resolves."""
    talk = FakeTalk(room_failures=1, last_message_id=10, chat=[([_message(11)], 11)])
    poller, runtime, offsets = _harness(tmp_path, talk)

    assert poller.poll_once() == 0
    assert talk.requested == []

    assert poller.poll_once() == 1
    assert [event["id"] for event in runtime.handled] == [11]
    assert offsets.get(ROOM) == 11


def test_supervise_runs_even_while_a_room_is_held(tmp_path):
    """A room stuck on its type must not stall the engine's housekeeping — that
    housekeeping is what keeps the retry drain alive while Nextcloud is unwell."""
    talk = FakeTalk(room_failures=99)
    poller, runtime, _ = _harness(tmp_path, talk)

    for _ in range(3):
        poller.poll_once()

    assert runtime.supervise_calls == 3
    assert talk.requested == []


# ==========================================================================
# supervise on every wake
# ==========================================================================


def test_supervise_is_called_on_every_wake_including_idle_ones(tmp_path):
    talk = FakeTalk(chat=[None, None, None])
    poller, runtime, _ = _harness(tmp_path, talk)

    for _ in range(3):
        poller.poll_once()

    # Idle 304s are the overwhelmingly common wake, so if they did not drive
    # supervision an idle bridge would never supervise at all.
    assert runtime.supervise_calls == 3


def test_supervise_is_called_even_when_the_poll_fails(tmp_path):
    talk = FakeTalk(chat=[httpx.ConnectError("down")])
    poller, runtime, _ = _harness(tmp_path, talk)

    poller.poll_once()

    assert runtime.supervise_calls == 1


def test_supervise_precedes_the_poll(tmp_path):
    """It runs at the top of the cycle, so housekeeping happens even on the cycle
    where the poll or the dispatch is about to fail."""
    talk = FakeTalk(chat=[([_message(11)], 11)])
    runtime = FakeRuntime(fail_on={11: RuntimeError("boom")})
    poller, runtime, _ = _harness(tmp_path, talk, runtime=runtime, seed=10)

    poller.poll_once()

    assert runtime.supervise_calls == 1


# ==========================================================================
# Backoff — the same policy the room-type fetch uses
# ==========================================================================


def test_backoff_escalates_and_caps(tmp_path):
    talk = FakeTalk(chat=[httpx.ConnectError("down")] * 8)
    sleeps = []
    poller, _, _ = _harness(tmp_path, talk, seed=10, sleeps=sleeps)

    for _ in range(8):
        poller.poll_once()

    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, BACKOFF_CAP, BACKOFF_CAP]


def test_backoff_resets_after_a_cycle_that_worked(tmp_path):
    talk = FakeTalk(
        chat=[
            httpx.ConnectError("down"),
            httpx.ConnectError("down"),
            None,
            httpx.ConnectError("down"),
        ]
    )
    sleeps = []
    poller, _, _ = _harness(tmp_path, talk, seed=10, sleeps=sleeps)

    for _ in range(4):
        poller.poll_once()

    # A room that recovers must not carry a minute-long delay into its next
    # hiccup.
    assert sleeps == [1.0, 2.0, 1.0]


def test_a_dispatch_failure_also_backs_off(tmp_path):
    talk = FakeTalk(chat=[([_message(11)], 11)])
    runtime = FakeRuntime(fail_on={11: RuntimeError("boom")})
    sleeps = []
    poller, _, _ = _harness(tmp_path, talk, runtime=runtime, seed=10, sleeps=sleeps)

    poller.poll_once()

    assert sleeps == [1.0]


def test_the_backoff_schedule_is_configurable(tmp_path):
    talk = FakeTalk(chat=[httpx.ConnectError("down")] * 4)
    sleeps = []
    poller, _, _ = _harness(
        tmp_path, talk, seed=10, sleeps=sleeps, backoff_start=0.5, backoff_cap=2.0
    )

    for _ in range(4):
        poller.poll_once()

    assert sleeps == [0.5, 1.0, 2.0, 2.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"backoff_start": 0.0}, id="zero_start"),
        pytest.param({"backoff_start": -1.0}, id="negative_start"),
        pytest.param({"backoff_start": 10.0, "backoff_cap": 5.0}, id="cap_below_start"),
    ],
)
def test_backoff_settings_that_would_hot_loop_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        _harness(tmp_path, FakeTalk(), **kwargs)


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(httpx.ConnectError("refused"), id="transport_error"),
        pytest.param(httpx.ReadTimeout("timed out"), id="read_timeout"),
        pytest.param(RuntimeError("something unforeseen"), id="unforeseen"),
        pytest.param(ValueError("bad value"), id="value_error"),
    ],
)
def test_poll_once_never_raises(tmp_path, failure):
    talk = FakeTalk(chat=[failure])
    poller, _, _ = _harness(tmp_path, talk, seed=10)

    assert poller.poll_once() == 0


def test_a_failed_cycle_is_logged_with_its_retry_delay(tmp_path, caplog):
    talk = FakeTalk(chat=[httpx.ConnectError("down")])
    poller, _, _ = _harness(tmp_path, talk, seed=10)

    with caplog.at_level("WARNING", logger="osprey.bridges.nextcloud_talk.poller"):
        poller.poll_once()

    assert ROOM in caplog.text
    assert "poll cycle failed" in caplog.text


# ==========================================================================
# Threading — a room thread never exits on its own
# ==========================================================================


def _run_until(poller, talk, cycles):
    """Start ``poller``, wait for ``cycles`` chat polls, then stop and join.

    Synchronises on the event the fake Talk sets, never on elapsed time. Returns
    whether the thread was still alive at the moment the target was reached.
    """
    talk.stop_after = cycles
    poller.start()
    try:
        assert talk.reached.wait(timeout=JOIN_TIMEOUT), "poller never reached the target cycle"
        alive = poller.is_alive()
    finally:
        poller.stop()
        poller.join(timeout=JOIN_TIMEOUT)
    assert not poller.is_alive(), "poller did not stop"
    return alive


def test_the_thread_survives_repeated_dispatch_failures(tmp_path):
    """The single most important property: nothing in core restarts a dead
    ingestion thread, so an escaping exception would make this room deaf forever
    while the process still looked healthy."""
    talk = FakeTalk(chat=[([_message(11)], 11)] * 3)
    runtime = FakeRuntime(fail_on={11: RuntimeError("boom")})
    poller, runtime, offsets = _harness(tmp_path, talk, runtime=runtime, seed=10)
    # Every batch re-delivers 11, so re-arm the failure each time.
    runtime.fail_on = {}
    original = runtime.handle_event

    def always_fail(event):
        original(event)
        raise RuntimeError("boom")

    runtime.handle_event = always_fail

    assert _run_until(poller, talk, cycles=4) is True
    assert offsets.get(ROOM) == 10


def test_the_thread_survives_repeated_transport_failures(tmp_path):
    talk = FakeTalk(chat=[httpx.ConnectError("down")] * 3)
    poller, _, _ = _harness(tmp_path, talk, seed=10)

    assert _run_until(poller, talk, cycles=4) is True


def test_the_thread_keeps_polling_while_idle(tmp_path):
    talk = FakeTalk(chat=[None] * 3)
    poller, runtime, _ = _harness(tmp_path, talk, seed=10)

    assert _run_until(poller, talk, cycles=3) is True
    assert runtime.supervise_calls >= 3


def test_stop_ends_the_loop(tmp_path):
    talk = FakeTalk(chat=[None])
    poller, _, _ = _harness(tmp_path, talk, seed=10)
    talk.stop_after = 1

    poller.start()
    assert talk.reached.wait(timeout=JOIN_TIMEOUT)
    poller.stop()
    poller.join(timeout=JOIN_TIMEOUT)

    assert not poller.is_alive()


def test_one_shared_stop_event_stops_every_room(tmp_path):
    """The SIGTERM shape: the entrypoint sets one event and every room winds
    down."""
    stop = threading.Event()
    pollers = []
    talks = []
    for index in range(3):
        talk = FakeTalk(chat=[None])
        talk.stop_after = 1
        poller, _, _ = _harness(tmp_path / f"room{index}", talk, seed=10, stop=stop)
        pollers.append(poller)
        talks.append(talk)

    for poller in pollers:
        poller.start()
    try:
        for talk in talks:
            assert talk.reached.wait(timeout=JOIN_TIMEOUT)
    finally:
        stop.set()
        for poller in pollers:
            poller.join(timeout=JOIN_TIMEOUT)

    assert not any(poller.is_alive() for poller in pollers)


def test_the_stop_event_is_the_default_backoff_wait(tmp_path):
    """A shutdown during a minute-long backoff must not be waited out."""
    talk = FakeTalk(chat=[httpx.ConnectError("down")] * 50)
    client = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(talk.handler)))
    rooms = RoomDirectory(client, sleep=lambda _: None)
    offsets = OffsetStore(str(tmp_path / "offsets.json"))
    offsets.advance(ROOM, 10)
    # No injected sleep: the default must be the stop event's wait.
    poller = RoomPoller(
        ROOM,
        client=client,
        rooms=rooms,
        offsets=offsets,
        runtime=FakeRuntime(),
        poll_hold=0,
        backoff_start=30.0,
        backoff_cap=30.0,
    )
    talk.stop_after = 1

    poller.start()
    try:
        assert talk.reached.wait(timeout=JOIN_TIMEOUT)
    finally:
        poller.stop()
        # With time.sleep this would take 30 seconds and the join would fail.
        poller.join(timeout=JOIN_TIMEOUT)

    assert not poller.is_alive()


def test_start_twice_does_not_double_poll_a_room(tmp_path):
    talk = FakeTalk(chat=[None])
    poller, _, _ = _harness(tmp_path, talk, seed=10)
    talk.stop_after = 1

    first = poller.start()
    try:
        second = poller.start()
        assert second is first
    finally:
        poller.stop()
        poller.join(timeout=JOIN_TIMEOUT)


def test_join_before_start_is_a_no_op(tmp_path):
    poller, _, _ = _harness(tmp_path, FakeTalk(), seed=10)

    poller.join(timeout=0)

    assert not poller.is_alive()


# ==========================================================================
# The fake runtime is a faithful stand-in
# ==========================================================================


def test_the_fake_runtime_matches_the_real_bridge_runtime_surface():
    """These tests substitute a fake for BridgeRuntime, so the two members the
    poller drives must actually exist on the real class. mypy checks the calls
    themselves, since the poller's parameter is typed as BridgeRuntime."""
    for name in ("handle_event", "supervise"):
        assert callable(getattr(BridgeRuntime, name)), name
