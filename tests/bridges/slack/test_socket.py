"""Tests for the Slack bridge's Socket Mode connection.

Every test drives the real :class:`SocketModeConnection` through an injected
``connect`` that hands out scripted fake connections, and a fake Web API client
that answers ``apps.connections.open`` from a per-test queue — no network, no
workspace. Time is injected too: the fake clock advances when the connection
sleeps or a receive times out, so backoff and idle ticks are observed through
what was asked, never by waiting.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosed

from osprey.bridges.slack.client import SlackApiError, SlackAuthError
from osprey.bridges.slack.socket import (
    BACKOFF_CAP_SEC,
    BACKOFF_RESET_SEC,
    BACKOFF_START_SEC,
    HELLO_TIMEOUT_SEC,
    PING_INTERVAL_SEC,
    RECV_TIMEOUT_SEC,
    Envelope,
    SocketModeConnection,
)

APP_TOKEN = "xapp-1-A111-3333-app-secret"  # gitleaks:allow - a fake test token
HELLO = json.dumps({"type": "hello", "num_connections": 1})


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeConn:
    """A scripted websocket: ``recv`` pops frames (str), raises scripted exceptions,
    and times out (advancing the clock) once the script is empty."""

    def __init__(self, clock: FakeClock, *frames: Any, name: str = "") -> None:
        self.clock = clock
        self.script = list(frames)
        self.sent: list[str] = []
        self.closed = False
        self.recv_timeouts: list[float | None] = []
        self.name = name

    def recv(self, timeout: float | None = None) -> str:
        self.recv_timeouts.append(timeout)
        if self.closed:
            raise ConnectionClosed(None, None)
        if not self.script:
            self.clock.now += timeout or 0.0
            raise TimeoutError
        item = self.script.pop(0)
        if isinstance(item, TimeoutError):
            self.clock.now += timeout or 0.0
            raise item
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item()
        return item

    def send(self, message: str) -> None:
        if self.closed:
            raise ConnectionClosed(None, None)
        self.sent.append(message)

    def close(self) -> None:
        self.closed = True


class FakeConnect:
    """Stands in for ``websockets.sync.client.connect``."""

    def __init__(self, *conns: Any, clock: FakeClock | None = None) -> None:
        self.conns = list(conns)
        self.clock = clock
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.marks: list[float] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeConn:
        self.calls.append((url, kwargs))
        if self.clock is not None:
            self.marks.append(self.clock.now)
        item = self.conns.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeWeb:
    """Answers ``apps.connections.open`` with successive URLs (or raises)."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers) or ["wss://wss.example/link/?ticket=1"]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fatal = threading.Event()

    def call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, kwargs))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, dict):
            return answer
        return {"ok": True, "url": answer}


def envelope(eid: str, payload: Any = None, kind: str = "events_api") -> str:
    return json.dumps(
        {
            "envelope_id": eid,
            "type": kind,
            "payload": {"event_id": f"Ev{eid}"} if payload is None else payload,
            "accepts_response_payload": False,
        }
    )


def disconnect(reason: str) -> str:
    return json.dumps({"type": "disconnect", "reason": reason})


def make(
    web: FakeWeb,
    connect: FakeConnect,
    clock: FakeClock,
    *,
    trust_env: bool = False,
    rng: Any = lambda: 1.0,
) -> SocketModeConnection:
    return SocketModeConnection(
        web,
        APP_TOKEN,
        connect=connect,
        trust_env=trust_env,
        clock=clock,
        sleep=clock.sleep,
        rng=rng,
    )


def take(it: Iterator[Any], n: int) -> list[Any]:
    return [next(it) for _ in range(n)]


def envelopes_until(it: Iterator[Any], count: int, limit: int = 500) -> list[Envelope]:
    got: list[Envelope] = []
    for _ in range(limit):
        item = next(it)
        if item is not None:
            got.append(item)
            if len(got) == count:
                return got
    raise AssertionError(f"only {len(got)} envelopes within {limit} items")


# -- connecting ---------------------------------------------------------------


def test_it_opens_with_the_app_token_connects_to_the_url_and_yields_envelopes() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO, envelope("e1"))
    web = FakeWeb("wss://wss.example/link/?ticket=abc")
    connect = FakeConnect(conn)
    sock = make(web, connect, clock)

    item = next(iter(sock))

    assert web.calls == [("apps.connections.open", {"token": APP_TOKEN})]
    assert connect.calls[0][0] == "wss://wss.example/link/?ticket=abc"
    assert item == Envelope("e1", "events_api", {"event_id": "Eve1"})
    envelope_id, kind, payload = item
    assert (envelope_id, kind, payload["event_id"]) == ("e1", "events_api", "Eve1")


@pytest.mark.parametrize(("trust_env", "proxy"), [(True, True), (False, None)])
def test_the_proxy_follows_trust_env(trust_env: bool, proxy: Any) -> None:
    clock = FakeClock()
    connect = FakeConnect(FakeConn(clock, HELLO, envelope("e1")))
    sock = make(FakeWeb(), connect, clock, trust_env=trust_env)

    next(iter(sock))

    assert connect.calls[0][1]["proxy"] is proxy


def test_the_ping_interval_is_short() -> None:
    clock = FakeClock()
    connect = FakeConnect(FakeConn(clock, HELLO, envelope("e1")))
    sock = make(FakeWeb(), connect, clock)

    next(iter(sock))

    kwargs = connect.calls[0][1]
    assert kwargs["ping_interval"] == PING_INTERVAL_SEC
    assert 0 < PING_INTERVAL_SEC <= 30
    assert kwargs["ping_timeout"] is not None


def test_no_envelope_is_yielded_before_hello() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, envelope("early"), HELLO, envelope("e1"))
    sock = make(FakeWeb(), FakeConnect(conn), clock)

    assert envelopes_until(iter(sock), 1) == [Envelope("e1", "events_api", {"event_id": "Eve1"})]


def test_a_missing_hello_times_out_closes_and_retries_with_backoff() -> None:
    clock = FakeClock()
    silent = FakeConn(clock)
    good = FakeConn(clock, HELLO, envelope("e1"))
    connect = FakeConnect(silent, good)
    sock = make(FakeWeb(), connect, clock)

    got = envelopes_until(iter(sock), 1)

    assert got[0].envelope_id == "e1"
    assert silent.closed
    assert len(connect.calls) == 2
    assert sum(silent.recv_timeouts) >= HELLO_TIMEOUT_SEC  # type: ignore[arg-type]
    assert clock.sleeps and sum(clock.sleeps) == pytest.approx(BACKOFF_START_SEC)


def test_the_wait_for_hello_still_yields_idle_ticks() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, TimeoutError(), TimeoutError(), HELLO, envelope("e1"))
    sock = make(FakeWeb(), FakeConnect(conn), clock)

    assert take(iter(sock), 3) == [None, None, Envelope("e1", "events_api", {"event_id": "Eve1"})]


# -- idle ticks ---------------------------------------------------------------


def test_a_quiet_socket_yields_an_idle_tick_per_receive_timeout() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO)
    sock = make(FakeWeb(), FakeConnect(conn), clock)

    assert take(iter(sock), 3) == [None, None, None]
    assert 0 < RECV_TIMEOUT_SEC <= 1.0
    assert all(t is not None and t <= 1.0 for t in conn.recv_timeouts)


def test_a_stream_of_non_envelope_frames_still_yields_an_idle_tick_every_second() -> None:
    clock = FakeClock()

    def chatter() -> str:
        clock.now += 0.4
        return json.dumps({"type": "something_else"})

    conn = FakeConn(clock, HELLO, *[chatter] * 10)
    sock = make(FakeWeb(), FakeConnect(conn), clock)
    it = iter(sock)

    start = clock.now
    first = next(it)

    assert first is None
    assert clock.now - start <= 1.0 + 0.4


# -- frames that are not envelopes ---------------------------------------------


def test_malformed_and_unaddressed_frames_are_skipped() -> None:
    clock = FakeClock()
    conn = FakeConn(
        clock,
        HELLO,
        "not json",
        json.dumps(["a", "list"]),
        json.dumps({"type": "events_api", "payload": {}}),  # no envelope_id
        json.dumps({"envelope_id": 7, "type": "events_api"}),  # id not a string
        HELLO,
        envelope("e1"),
    )
    sock = make(FakeWeb(), FakeConnect(conn), clock)

    assert envelopes_until(iter(sock), 1)[0].envelope_id == "e1"


def test_a_non_mapping_payload_is_handed_on_as_empty() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO, envelope("e1", payload="oops", kind="slash_commands"))
    sock = make(FakeWeb(), FakeConnect(conn), clock)

    assert next(iter(sock)) == Envelope("e1", "slash_commands", {})


# -- acks -----------------------------------------------------------------------


def test_ack_sends_the_envelope_id_on_the_current_connection() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO, envelope("e1"))
    sock = make(FakeWeb(), FakeConnect(conn), clock)
    item = next(iter(sock))

    assert sock.ack(item.envelope_id) is True

    assert [json.loads(m) for m in conn.sent] == [{"envelope_id": "e1"}]


def test_ack_without_a_connection_returns_false_and_does_not_raise() -> None:
    clock = FakeClock()
    sock = make(FakeWeb(), FakeConnect(), clock)

    assert sock.ack("e1") is False


def test_ack_on_a_dead_connection_returns_false_and_does_not_raise() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO, envelope("e1"))
    sock = make(FakeWeb(), FakeConnect(conn), clock)
    next(iter(sock))
    conn.closed = True

    assert sock.ack("e1") is False


def test_ack_is_safe_from_another_thread_while_iterating() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO, envelope("e1"))
    sock = make(FakeWeb(), FakeConnect(conn), clock)
    it = iter(sock)
    item = next(it)

    results: list[bool] = []
    t = threading.Thread(target=lambda: results.append(sock.ack(item.envelope_id)))
    t.start()
    next(it)
    t.join(timeout=5)

    assert results == [True]


# -- disconnect reasons ----------------------------------------------------------


@pytest.mark.parametrize("reason", ["refresh_requested", "warning", "something_new"])
def test_a_refresh_opens_the_new_connection_before_closing_the_old(reason: str) -> None:
    clock = FakeClock()
    old = FakeConn(clock, HELLO, envelope("e1"), disconnect(reason), name="old")
    states_at_new_connect: list[bool] = []

    def new_hello() -> str:
        states_at_new_connect.append(old.closed)
        return HELLO

    new = FakeConn(clock, new_hello, envelope("e2"), name="new")
    web = FakeWeb("wss://one", "wss://two")
    connect = FakeConnect(old, new)
    sock = make(web, connect, clock)

    got = envelopes_until(iter(sock), 2)

    assert [e.envelope_id for e in got] == ["e1", "e2"]
    assert [c[0] for c in connect.calls] == ["wss://one", "wss://two"]
    assert len(web.calls) == 2
    assert states_at_new_connect == [False]
    assert old.closed and not new.closed
    assert clock.sleeps == []
    sock.ack("e2")
    assert new.sent and not old.sent


def test_a_failed_refresh_keeps_serving_on_the_old_connection() -> None:
    clock = FakeClock()
    old = FakeConn(clock, HELLO, disconnect("warning"), envelope("e1"))
    connect = FakeConnect(old, OSError("refused"))
    sock = make(FakeWeb("wss://one", "wss://two"), connect, clock)

    got = envelopes_until(iter(sock), 1)

    assert got[0].envelope_id == "e1"
    assert not old.closed


def test_link_disabled_raises_slack_auth_error_sets_fatal_and_closes() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO, disconnect("link_disabled"))
    web = FakeWeb()
    sock = make(web, FakeConnect(conn), clock)

    with pytest.raises(SlackAuthError) as raised:
        list(iter(sock))

    assert raised.value.error == "link_disabled"
    assert web.fatal.is_set()
    assert conn.closed
    assert len(web.calls) == 1


def test_link_disabled_before_hello_is_fatal_too() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, disconnect("link_disabled"))
    web = FakeWeb()
    sock = make(web, FakeConnect(conn), clock)

    with pytest.raises(SlackAuthError):
        list(iter(sock))
    assert conn.closed


# -- abnormal close and backoff ----------------------------------------------------


def test_an_abnormal_close_reconnects_after_a_jittered_backoff_with_idle_ticks() -> None:
    clock = FakeClock()
    first = FakeConn(clock, HELLO, envelope("e1"), ConnectionClosed(None, None))
    second = FakeConn(clock, HELLO, envelope("e2"))
    connect = FakeConnect(first, second)
    sock = make(FakeWeb("wss://one", "wss://two"), connect, clock, rng=lambda: 0.0)
    it = iter(sock)

    assert next(it).envelope_id == "e1"
    items: list[Any] = []
    while not items or items[-1] is None:
        items.append(next(it))

    assert items[-1].envelope_id == "e2"
    assert None in items[:-1]  # the wait is visible as idle ticks
    assert len(connect.calls) == 2
    waited = sum(clock.sleeps)
    assert BACKOFF_START_SEC * 0.5 <= waited < BACKOFF_START_SEC
    assert all(s <= RECV_TIMEOUT_SEC for s in clock.sleeps)


def test_backoff_grows_per_consecutive_failure_and_is_capped() -> None:
    clock = FakeClock()
    failures = [OSError("down")] * 10
    connect = FakeConnect(*failures, FakeConn(clock, HELLO, envelope("e1")), clock=clock)
    sock = make(FakeWeb(), connect, clock)

    assert envelopes_until(iter(sock), 1)[0].envelope_id == "e1"

    gaps = [b - a for a, b in zip(connect.marks, connect.marks[1:], strict=False)]
    assert gaps[0] == pytest.approx(BACKOFF_START_SEC)
    assert gaps[1] == pytest.approx(BACKOFF_START_SEC * 2)
    assert all(b >= a for a, b in zip(gaps, gaps[1:], strict=False))
    assert max(gaps) == pytest.approx(BACKOFF_CAP_SEC)


def test_backoff_resets_after_a_connection_stayed_up() -> None:
    clock = FakeClock()

    def long_quiet() -> str:
        clock.now += BACKOFF_RESET_SEC + 1
        raise ConnectionClosed(None, None)

    first = FakeConn(clock, HELLO, ConnectionClosed(None, None))
    second = FakeConn(clock, HELLO, long_quiet)
    third = FakeConn(clock, HELLO, envelope("e1"))
    sock = make(FakeWeb(), FakeConnect(first, second, third), clock)

    envelopes_until(iter(sock), 1)

    # The first drop waits START. The second came after a stable connection, so
    # the streak restarts and it waits START again rather than doubling.
    assert sum(clock.sleeps) == pytest.approx(BACKOFF_START_SEC * 2)


def test_an_open_failure_is_retried_with_backoff() -> None:
    clock = FakeClock()
    web = FakeWeb(SlackApiError("boom", method="apps.connections.open"), "wss://two")
    connect = FakeConnect(FakeConn(clock, HELLO, envelope("e1")))
    sock = make(web, connect, clock)

    assert envelopes_until(iter(sock), 1)[0].envelope_id == "e1"
    assert len(web.calls) == 2
    assert sum(clock.sleeps) == pytest.approx(BACKOFF_START_SEC)


def test_an_open_answer_without_a_url_is_retried() -> None:
    clock = FakeClock()
    web = FakeWeb({"ok": True}, "wss://two")
    connect = FakeConnect(FakeConn(clock, HELLO, envelope("e1")))
    sock = make(web, connect, clock)

    assert envelopes_until(iter(sock), 1)[0].envelope_id == "e1"
    assert [c[0] for c in connect.calls] == ["wss://two"]


def test_an_auth_fatal_open_propagates() -> None:
    clock = FakeClock()
    web = FakeWeb(SlackAuthError("dead", error="invalid_auth"))
    connect = FakeConnect()
    sock = make(web, connect, clock)

    with pytest.raises(SlackAuthError):
        next(iter(sock))
    assert connect.calls == []


def test_no_token_or_url_reaches_the_log(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    secret_url = "wss://wss.example/link/?ticket=SECRET-TICKET"
    connect = FakeConnect(
        OSError(f"cannot reach {secret_url}"), FakeConn(clock, HELLO, envelope("e1"))
    )
    sock = make(FakeWeb(secret_url), connect, clock)

    with caplog.at_level("DEBUG"):
        envelopes_until(iter(sock), 1)

    assert "SECRET-TICKET" not in caplog.text
    assert APP_TOKEN not in caplog.text


# -- closing -----------------------------------------------------------------------


def test_close_ends_iteration_and_closes_the_connection() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO, envelope("e1"))
    sock = make(FakeWeb(), FakeConnect(conn), clock)
    it = iter(sock)
    next(it)

    sock.close()

    assert list(it) == []
    assert conn.closed
    assert sock.ack("e1") is False


def test_close_during_a_backoff_ends_iteration_without_reconnecting() -> None:
    clock = FakeClock()
    conn = FakeConn(clock, HELLO, ConnectionClosed(None, None))
    connect = FakeConnect(conn)
    sock = make(FakeWeb(), connect, clock)
    it = iter(sock)

    assert next(it) is None  # first tick of the backoff wait
    sock.close()

    assert list(it) == []
    assert len(connect.calls) == 1


def test_close_before_iterating_never_connects() -> None:
    clock = FakeClock()
    web = FakeWeb()
    connect = FakeConnect()
    sock = make(web, connect, clock)

    sock.close()

    assert list(iter(sock)) == []
    assert web.calls == [] and connect.calls == []
