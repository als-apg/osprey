"""RoomDirectory: the fail-closed room-type hold and its capped backoff.

Every test drives an :class:`httpx.MockTransport` and an injected sleep — nothing
here opens a socket or waits a real second. Four properties get the most
attention, because each one is a way the live bridge would fail *silently*:

* **cache-read purity** — ``parse_event`` calls the read path per message, so the
  reads must issue no request at all;
* **unresolved is not "group room"** — :meth:`RoomDirectory.is_one_to_one` raises
  instead of answering, so the mention filter can never fail open;
* **the fetch never raises out** — a Nextcloud that is down at startup leaves
  rooms unresolved, not a dead thread;
* **the lock is not held across the HTTP call** — otherwise one unreachable room
  would stall every other room's poll thread behind a socket timeout.
"""

import threading
import time

import httpx
import pytest

from osprey.bridges.nextcloud_talk import (
    NextcloudBridgeConfig,
    RoomDirectory,
    RoomTypeUnresolved,
    TalkApiError,
    TalkClient,
)
from osprey.bridges.nextcloud_talk.client import ROOM_TYPE_ONE_TO_ONE
from osprey.bridges.nextcloud_talk.rooms import BACKOFF_CAP, BACKOFF_START

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account="osprey-bot",
    app_password="app-pw",
    rooms=("roomA", "roomB"),
)

GROUP_ROOM = {"type": 2, "displayName": "Ops", "lastMessage": {"id": 7}}
ONE_TO_ONE_ROOM = {"type": ROOM_TYPE_ONE_TO_ONE, "displayName": "alice", "lastMessage": {"id": 3}}


def _ocs(data, *, status=200):
    """An OCS v2 envelope response carrying ``data``."""
    return httpx.Response(
        status,
        json={"ocs": {"meta": {"status": "ok", "statuscode": status}, "data": data}},
    )


def _room_token(request):
    """The room token addressed by a v4 room request."""
    return request.url.path.rsplit("/", 1)[-1]


def _directory(handler, **kwargs):
    """``(directory, sleeps, requests)`` over a MockTransport running ``handler``.

    ``sleeps`` records every backoff wait in order — the whole assertion surface
    for the backoff schedule — and ``requests`` every HTTP request the directory
    made, which is how the read path is pinned as network-free.
    """
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    def recorder(request):
        requests.append(request)
        return handler(request)

    client = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(recorder)))
    return RoomDirectory(client, sleep=sleeps.append, **kwargs), sleeps, requests


def _always(payload):
    """A Talk that answers every room request with ``payload``."""
    return lambda request: _ocs(payload)


def _fails_then(failures, payload=GROUP_ROOM, exc=None):
    """A Talk that fails a room's first ``failures`` requests, then answers.

    Failures are counted per room, so one room's outage does not consume
    another's budget.
    """
    seen: dict[str, int] = {}

    def handler(request):
        room = _room_token(request)
        seen[room] = seen.get(room, 0) + 1
        if seen[room] <= failures:
            raise exc or httpx.ConnectError("nextcloud unreachable", request=request)
        return _ocs(payload)

    return handler


# ==========================================================================
# Resolution and caching — one fetch per room, ever
# ==========================================================================


def test_resolve_once_returns_type_and_caches_it():
    directory, sleeps, requests = _directory(_always(GROUP_ROOM))

    assert directory.resolve_once("roomA") == 2
    assert directory.type_of("roomA") == 2
    assert len(requests) == 1
    # A resolved room never waits and never fetches again, so a poll thread can
    # call resolve_once at the top of every cycle for free.
    assert directory.resolve_once("roomA") == 2
    assert len(requests) == 1
    assert sleeps == []


def test_resolution_is_per_room():
    def handler(request):
        return _ocs(ONE_TO_ONE_ROOM if _room_token(request) == "roomB" else GROUP_ROOM)

    directory, _, _ = _directory(handler)

    assert directory.resolve_once("roomA") == 2
    # roomB is still unresolved: resolving one room says nothing about another.
    assert directory.type_of("roomB") is None
    assert directory.resolve_once("roomB") == ROOM_TYPE_ONE_TO_ONE


def test_info_of_caches_display_name_and_tail_anchor():
    directory, _, _ = _directory(_always(GROUP_ROOM))
    directory.resolve_once("roomA")

    info = directory.info_of("roomA")
    assert info is not None
    assert (info.token, info.type, info.display_name) == ("roomA", 2, "Ops")
    # The room's tail at resolution time — the first-ever-start skip-history anchor.
    assert info.last_message_id == 7
    assert directory.display_name_of("roomA") == "Ops"


# ==========================================================================
# The hold: messages arriving before the type resolves
# ==========================================================================


def test_fetch_fails_then_succeeds_holds_until_resolved():
    """The messages-arriving-before-resolution case.

    While the fetch is failing the room stays unresolved and asking what kind of
    room it is raises, which is what defers those messages. Once the fetch
    succeeds the same reads answer, and the deferred messages are dispatchable.
    """
    directory, sleeps, requests = _directory(_fails_then(1))

    # Cycle 1: fetch fails, room still unresolved, one backoff wait spent.
    assert directory.resolve_once("roomA") is None
    assert directory.type_of("roomA") is None
    with pytest.raises(RoomTypeUnresolved):
        directory.is_one_to_one("roomA")
    assert sleeps == [BACKOFF_START]

    # Cycle 2: fetch succeeds; the hold lifts and no further wait is spent.
    assert directory.resolve_once("roomA") == 2
    assert directory.is_one_to_one("roomA") is False
    assert sleeps == [BACKOFF_START]
    assert len(requests) == 2


def test_unresolved_reads_are_answerable_without_guessing():
    directory, _, requests = _directory(_always(GROUP_ROOM))

    # None means "not known yet" — never "group room". type_of is the poller's
    # form (it tests for None); is_one_to_one is parse_event's and refuses.
    assert directory.type_of("roomA") is None
    assert directory.info_of("roomA") is None
    # Log-safe: a name is a string even for a room nothing is known about.
    assert directory.display_name_of("roomA") == ""
    with pytest.raises(RoomTypeUnresolved, match="roomA"):
        directory.is_one_to_one("roomA")
    assert requests == []


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(ONE_TO_ONE_ROOM, True, id="one_to_one"),
        pytest.param(GROUP_ROOM, False, id="group"),
        pytest.param({"type": 3, "displayName": "Public"}, False, id="public_is_group"),
    ],
)
def test_is_one_to_one_maps_talk_types(payload, expected):
    directory, _, _ = _directory(_always(payload))
    directory.resolve_once("roomA")

    assert directory.is_one_to_one("roomA") is expected


# ==========================================================================
# Cache reads never touch the network (parse_event calls these per message)
# ==========================================================================


def test_reads_issue_no_requests_after_resolution():
    directory, sleeps, requests = _directory(_always(ONE_TO_ONE_ROOM))
    directory.resolve_once("roomA")
    requests.clear()

    for _ in range(50):
        assert directory.type_of("roomA") == ROOM_TYPE_ONE_TO_ONE
        assert directory.is_one_to_one("roomA") is True
        assert directory.display_name_of("roomA") == "alice"
        assert directory.info_of("roomA") is not None

    assert requests == []
    assert sleeps == []


def test_reads_issue_no_requests_while_unresolved():
    """The purity that matters most: an unresolved room must not turn the
    per-message read path into a fetch — that is how HTTP latency (and an HTTP
    timeout) would end up on every inbound message."""
    directory, sleeps, requests = _directory(_always(GROUP_ROOM))

    for _ in range(50):
        assert directory.type_of("roomA") is None
        assert directory.display_name_of("roomA") == ""
        assert directory.info_of("roomA") is None
        with pytest.raises(RoomTypeUnresolved):
            directory.is_one_to_one("roomA")

    assert requests == []
    assert sleeps == []


# ==========================================================================
# Capped exponential backoff
# ==========================================================================


def test_backoff_escalates_and_caps():
    # Never resolves: eight consecutive failures walk the schedule to its ceiling
    # and stay there.
    directory, sleeps, requests = _directory(_fails_then(999))

    for _ in range(8):
        assert directory.resolve_once("roomA") is None

    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, BACKOFF_CAP, BACKOFF_CAP]
    assert max(sleeps) == BACKOFF_CAP
    # One request per attempt: the retry loop is the caller's, so resolve_once
    # never fetches twice in one call.
    assert len(requests) == 8


def test_backoff_schedule_is_configurable():
    directory, sleeps, _ = _directory(_fails_then(999), backoff_start=0.5, backoff_cap=2.0)

    for _ in range(5):
        directory.resolve_once("roomA")

    assert sleeps == [0.5, 1.0, 2.0, 2.0, 2.0]


def test_backoff_is_tracked_per_room():
    """One unreachable room must not put another room on a long delay — each
    room's poll thread is retrying its own fetch."""
    directory, sleeps, _ = _directory(_fails_then(999))

    for _ in range(3):
        directory.resolve_once("roomA")
    sleeps.clear()
    directory.resolve_once("roomB")

    assert sleeps == [BACKOFF_START]


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"backoff_start": 0.0}, id="zero_start"),
        pytest.param({"backoff_start": -1.0}, id="negative_start"),
        pytest.param({"backoff_start": 10.0, "backoff_cap": 5.0}, id="cap_below_start"),
    ],
)
def test_rejects_backoff_settings_that_would_hot_loop(kwargs):
    with pytest.raises(ValueError):
        _directory(_always(GROUP_ROOM), **kwargs)


# ==========================================================================
# The fetch never raises out — the process stays up, the room stays unresolved
# ==========================================================================


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(
            _fails_then(999, exc=httpx.ConnectError("refused")),
            id="transport_error",
        ),
        pytest.param(lambda request: httpx.Response(500), id="server_error"),
        pytest.param(lambda request: httpx.Response(404), id="not_found"),
        pytest.param(lambda request: httpx.Response(200, text="not json"), id="non_json_body"),
        pytest.param(lambda request: _ocs({"displayName": "Ops"}), id="payload_without_type"),
        pytest.param(lambda request: _ocs(["wrong", "shape"]), id="payload_not_an_object"),
        pytest.param(_fails_then(999, exc=TalkApiError("boom")), id="talk_api_error"),
    ],
)
def test_every_fetch_failure_becomes_unresolved_not_an_exception(handler):
    directory, sleeps, _ = _directory(handler)

    assert directory.resolve_once("roomA") is None
    assert directory.type_of("roomA") is None
    assert sleeps == [BACKOFF_START]


def test_failed_fetch_is_logged_with_the_retry_delay(caplog):
    directory, _, _ = _directory(_fails_then(999))

    with caplog.at_level("WARNING", logger="osprey.bridges.nextcloud_talk.rooms"):
        directory.resolve_once("roomA")

    assert "roomA" in caplog.text
    assert "type fetch failed" in caplog.text


# ==========================================================================
# Thread safety — room threads and the drain thread share one instance
# ==========================================================================


def _run_concurrently(work, threads):
    """Run ``work(i)`` on ``threads`` threads, returning results and exceptions."""
    results: list[object] = []
    errors: list[Exception] = []
    lock = threading.Lock()
    ready = threading.Barrier(threads)

    def runner(index):
        try:
            ready.wait(timeout=5)
            value = work(index)
        except Exception as exc:  # collected so a thread failure fails the test
            with lock:
                errors.append(exc)
        else:
            with lock:
                results.append(value)

    workers = [threading.Thread(target=runner, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert not any(worker.is_alive() for worker in workers)
    return results, errors


def test_concurrent_population_of_one_room_agrees():
    directory, _, requests = _directory(_always(GROUP_ROOM))

    results, errors = _run_concurrently(lambda _: directory.resolve_once("roomA"), threads=8)

    assert errors == []
    # A racing duplicate fetch is harmless, but every caller must see one answer.
    assert set(results) == {2}
    assert 1 <= len(requests) <= 8
    assert directory.type_of("roomA") == 2


def test_concurrent_reads_and_population_across_rooms():
    rooms = ("roomA", "roomB", "roomC", "roomD")

    def handler(request):
        return _ocs({"type": 2, "displayName": _room_token(request)})

    directory, _, _ = _directory(handler)

    def work(index):
        room = rooms[index % len(rooms)]
        for _ in range(50):
            # Interleave the read path with population, the way the drain thread
            # and the room threads do.
            directory.type_of(room)
            directory.display_name_of(room)
            directory.resolve_once(room)
        return directory.type_of(room)

    results, errors = _run_concurrently(work, threads=len(rooms) * 3)

    assert errors == []
    assert set(results) == {2}
    for room in rooms:
        assert directory.type_of(room) == 2
        assert directory.display_name_of(room) == room


def test_a_slow_fetch_does_not_block_readers():
    """The lock must not be held across the HTTP call.

    If it were, an unreachable room would hold it for the client's full socket
    timeout and every other room's thread — plus every per-message read — would
    queue up behind that one room.
    """
    in_flight = threading.Event()
    release = threading.Event()

    def handler(request):
        in_flight.set()
        release.wait(timeout=5)
        return _ocs(GROUP_ROOM)

    directory, _, _ = _directory(handler)
    fetcher = threading.Thread(target=directory.resolve_once, args=("roomA",))
    fetcher.start()
    try:
        assert in_flight.wait(timeout=5)
        # The fetch is parked inside the transport right now.
        started = time.monotonic()
        assert directory.type_of("roomA") is None
        assert directory.type_of("roomB") is None
        assert directory.display_name_of("roomA") == ""
        elapsed = time.monotonic() - started
    finally:
        release.set()
        fetcher.join(timeout=10)

    assert elapsed < 2.0, f"reads waited {elapsed:.2f}s on an in-flight fetch"
    assert directory.type_of("roomA") == 2
