"""The alarm relay end to end: a fake broker, the real server, the real HTTP backend."""

from __future__ import annotations

import asyncio
import json

import pytest

from osprey_connectors.alarm_state import AlarmStates, _phoebus_kafka
from osprey_connectors.alarm_state._phoebus_kafka import PhoebusAlarmState
from osprey_connectors.alarm_state._relay import RelayAlarmState
from osprey_connectors.alarm_state.relay import serve
from osprey_connectors.service_answers import BackendSpec, Unavailable

from .test_kafka_backend import (
    CONFIG_KEY,
    LOADING,
    PV,
    STATE_KEY,
    FakeConsumer,
    Record,
    _backend,
    _caught_up,
    _major,
    clock,  # noqa: F401  (the fake clock the Kafka feed reads, for every test here)
)

OK_PV = "DEMO:VAC1:GAUGE:Pres-I"
DEADLINE = 5.0
MALFORMED = Unavailable("malformed response")


def _records() -> list[Record]:
    config = {"enabled": True, "latching": True, "guidance": [{"title": "Act", "details": "Now"}]}
    ok = json.dumps({"severity": "OK", "current_severity": "OK"}).encode()
    return [
        Record(CONFIG_KEY, json.dumps(config).encode()),
        Record(STATE_KEY, _major()),
        Record(f"state:/Demo/Vac/{OK_PV}".encode(), ok),
    ]


def _reader(url: str, **seams) -> RelayAlarmState:
    spec = BackendSpec(
        kind="alarm_state",
        type="osprey_alarm_relay",
        target="live",
        settings={"url": url},
        rehearsal=False,
    )
    return RelayAlarmState(spec, **seams)


class Served:
    """A transport that answers every request with one body and records what was asked."""

    def __init__(self, body: object) -> None:
        self.body = body
        self.asked: list[tuple] = []

    async def request(self, method, path, *, params=None, deadline):
        self.asked.append((method, path, params, deadline))
        return self.body


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch) -> None:
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")


class Relay:
    """A feed over a fake consumer, served on a free loopback port."""

    def __init__(self, feed: PhoebusAlarmState, server: asyncio.Server) -> None:
        self.feed = feed
        self.server = server
        self.port = server.sockets[0].getsockname()[1]
        # A path prefix is what a front end adds; the relay itself serves from the root.
        self.reader = _reader(f"http://127.0.0.1:{self.port}/")

    async def raw(self, request: str) -> tuple[int, dict]:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(request.encode())
        await writer.drain()
        answer = await reader.read()
        writer.close()
        head, _, body = answer.partition(b"\r\n\r\n")
        return int(head.split(b" ")[1]), json.loads(body)


@pytest.fixture
async def relay():
    relays = []

    async def start(consumer: FakeConsumer, *, catch_up: bool = True) -> Relay:
        feed = _backend(consumer)
        await feed.start()
        if catch_up:
            await _caught_up(feed)
        relays.append(Relay(feed, await serve(feed, port=0)))
        return relays[-1]

    yield start
    for made in relays:
        made.server.close()
        await made.server.wait_closed()
        await made.feed.aclose()


async def test_the_reader_answers_what_the_feed_beside_the_broker_answers(relay):
    made = await relay(FakeConsumer(_records()))
    feed, reader = made.feed, made.reader

    assert await reader.states(DEADLINE) == await feed.states(DEADLINE)
    assert await reader.state_for(PV, DEADLINE) == await feed.state_for(PV, DEADLINE)
    assert await reader.config_for(PV, DEADLINE) == await feed.config_for(PV, DEADLINE)
    assert await reader.state_for("OTHER:PV", DEADLINE) is None
    assert (await reader.config_for("OTHER:PV", DEADLINE)).in_tree == "no"

    states = await reader.states(DEADLINE)
    assert isinstance(states, AlarmStates)
    assert states.counts == {"MAJOR": 1, "OK": 1}
    assert states.stale is False and states.last_contact is not None
    assert (await reader.config_for(PV, DEADLINE)).guidance == [{"title": "Act", "details": "Now"}]
    assert "relay answered: read 3 messages" in await reader.probe(DEADLINE)


async def test_a_relay_that_lost_the_broker_serves_its_last_state_as_stale(relay, clock):  # noqa: F811
    consumer = FakeConsumer(_records())
    made = await relay(consumer)
    consumer.end_offset_error = OSError("broker gone")
    clock.now += _phoebus_kafka._STALE_WINDOW_S + 1

    states = await made.reader.states(DEADLINE)
    state = await made.reader.state_for(PV, DEADLINE)

    assert states.stale is True and all(row.state.stale for row in states.rows)
    assert state.stale is True and state.severity == "MAJOR"


async def test_a_loading_relay_answers_loading_never_a_clear_state(relay):
    consumer = FakeConsumer(_records())
    consumer.deliver.clear()
    made = await relay(consumer, catch_up=False)

    assert await made.reader.states(DEADLINE) == LOADING
    assert await made.reader.state_for(PV, DEADLINE) == LOADING
    assert await made.reader.config_for("OTHER:PV", DEADLINE) == LOADING
    assert await made.reader.probe(DEADLINE) == LOADING


async def test_a_relay_that_is_gone_is_unavailable(relay):
    made = await relay(FakeConsumer(_records()))
    made.server.close()
    await made.server.wait_closed()

    answer = await made.reader.states(DEADLINE)

    assert isinstance(answer, Unavailable)
    assert answer.reason.startswith("could not reach the service")


async def test_the_reader_asks_for_a_path_under_the_configured_url():
    transport = Served({"state": None})
    reader = _reader("https://front.example/alarm-relay", transport=transport)

    assert await reader.state_for(PV, DEADLINE) is None
    assert transport.asked == [("GET", "/state", {"channel": PV}, DEADLINE)]


@pytest.mark.parametrize(
    "request_line, status",
    [
        ("POST /states HTTP/1.1", 405),
        ("PUT /state?channel=X HTTP/1.1", 405),
        ("DELETE / HTTP/1.1", 405),
        ("GET /command HTTP/1.1", 404),
        ("GET /state HTTP/1.1", 400),
        ("GET /config?channel=A&channel=B HTTP/1.1", 400),
        ("nonsense", 400),
    ],
)
async def test_the_relay_refuses_everything_but_its_reads(relay, request_line, status):
    made = await relay(FakeConsumer(_records()))

    answered, body = await made.raw(f"{request_line}\r\nHost: relay\r\n\r\n")

    assert answered == status
    assert "error" in body


async def test_the_root_names_the_service(relay):
    made = await relay(FakeConsumer(_records()))

    status, body = await made.raw("GET / HTTP/1.1\r\nHost: relay\r\n\r\n")

    assert status == 200
    assert body["service"] == "osprey-alarm-relay"
    assert "read 3 messages" in body["evidence"]


@pytest.mark.parametrize(
    "served",
    [
        ["not", "a", "mapping"],
        {"rows": [], "stale": "no", "last_contact": None},
        {"rows": [{"channel": "", "path": None, "state": {}}], "stale": False},
        {"rows": [{"channel": PV, "path": None, "state": {"latched": 1}}], "stale": False},
        {"unavailable": ""},
    ],
)
async def test_an_answer_no_relay_writes_is_malformed(served):
    assert await _reader("http://relay", transport=Served(served)).states(DEADLINE) == MALFORMED


async def test_another_service_at_the_url_does_not_probe_as_a_relay():
    logger = Served({"name": "Alarm logging Service", "version": "4.7.3"})

    answer = await _reader("http://logger", transport=logger).probe(DEADLINE)

    assert answer == Unavailable("the url does not answer as an osprey-alarm-relay")
