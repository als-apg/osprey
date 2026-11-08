"""The Kafka alarm feed's lifecycle, catch-up answers and staleness, over a fake consumer."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from collections import namedtuple

import pytest

from osprey_connectors.alarm_state import _phoebus_kafka
from osprey_connectors.alarm_state._phoebus_kafka import PhoebusAlarmState
from osprey_connectors.pv_directory.contract import (
    ALARM_STALE_WINDOW_S,
    AlarmStateRecord,
    BackendSpec,
    Unavailable,
)

PV = "DEMO:RF1:CAV:Temp-I"
STATE_KEY = f"state:/Demo/RF/{PV}".encode()
CONFIG_KEY = f"config:/Demo/RF/{PV}".encode()
PARTITION = "topic-0"
LOADING = Unavailable("alarm feed still loading")
EMPTY = Unavailable("no message from the feed yet")
Record = namedtuple("Record", "key value")


def _major() -> bytes:
    return json.dumps({"severity": "MAJOR", "current_severity": "MAJOR"}).encode()


def _spec(**settings) -> BackendSpec:
    return BackendSpec(
        kind="alarm_state",
        type="phoebus_alarm_server",
        target="live",
        settings=settings,
        rehearsal=False,
    )


class FakeConsumer:
    """An aiokafka consumer over one partition; records every call it receives."""

    def __init__(self, records=(), *, start_error=None):
        self.records = list(records)
        self.start_error = start_error
        self.position_at = 0
        self.end_offset_error = None
        self.end_offset_calls = 0
        self.polls = 0
        self.stopped = False
        self.seeked = False
        self.deliver = asyncio.Event()
        self.deliver.set()
        self.batch_size = None

    async def start(self):
        if self.start_error is not None:
            raise self.start_error

    async def stop(self):
        self.stopped = True

    def assignment(self):
        return {PARTITION}

    async def end_offsets(self, partitions):
        self.end_offset_calls += 1
        if self.end_offset_error is not None:
            raise self.end_offset_error
        return dict.fromkeys(partitions, len(self.records))

    async def seek_to_beginning(self, *partitions):
        self.seeked = True
        self.position_at = 0

    async def position(self, _partition):
        return self.position_at

    async def getmany(self, *_partitions, **_options):
        self.polls += 1
        await asyncio.sleep(0)
        if not self.deliver.is_set() or self.position_at >= len(self.records):
            return {}
        batch = self.records[self.position_at :][: self.batch_size]
        self.position_at += len(batch)
        if self.position_at == self.batch_size:
            self.deliver.clear()
        return {PARTITION: batch}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(_phoebus_kafka, "_now", clock)
    monkeypatch.setattr(_phoebus_kafka, "_POLL_TIMEOUT_MS", 0)
    monkeypatch.setattr(_phoebus_kafka, "_RETRY_S", 0.0)
    return clock


def _backend(consumer: FakeConsumer) -> PhoebusAlarmState:
    return PhoebusAlarmState(_spec(), consumer_factory=lambda _settings: consumer)


async def _until(predicate, timeout: float = 5.0) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout)


async def _answer(backend: PhoebusAlarmState, pv: str = PV, deadline: float = 1.0):
    return await backend.state_for(pv, deadline=deadline)


def _elapsed(since: float) -> float:
    return asyncio.get_running_loop().time() - since


async def _caught_up(backend: PhoebusAlarmState) -> None:
    await _until(lambda: backend._caught_up)


async def test_reads_answer_from_the_fold_after_catch_up():
    consumer = FakeConsumer([Record(CONFIG_KEY, b'{"enabled": true}'), Record(STATE_KEY, _major())])
    backend = _backend(consumer)
    await backend.start()
    await _caught_up(backend)

    state = await _answer(backend)
    assert isinstance(state, AlarmStateRecord)
    assert state.severity == "MAJOR"
    assert state.stale is False
    config = await backend.config_for(PV, deadline=1.0)
    assert config.in_tree == "yes"
    assert config.label == "live configuration"
    assert await _answer(backend, "OTHER:PV") is None
    assert (await backend.config_for("OTHER:PV", deadline=1.0)).in_tree == "no"
    assert "2 messages" in await backend.probe(deadline=1.0)
    assert consumer.seeked is True
    await backend.aclose()


async def test_a_read_before_start_is_still_loading_at_once():
    backend = _backend(FakeConsumer([Record(STATE_KEY, _major())]))
    began = asyncio.get_running_loop().time()
    assert await _answer(backend, deadline=5.0) == LOADING
    assert await backend.config_for(PV, deadline=5.0) == LOADING
    assert await backend.probe(deadline=5.0) == LOADING
    assert _elapsed(began) < 1.0


async def test_a_read_before_catch_up_is_loading_never_clear_or_out_of_the_tree():
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    consumer.deliver.clear()
    backend = _backend(consumer)
    await backend.start()
    await _until(lambda: consumer.polls > 3)

    assert await _answer(backend, deadline=0.0) == LOADING
    assert await backend.config_for("OTHER:PV", deadline=0.0) == LOADING

    consumer.deliver.set()
    await _caught_up(backend)
    assert (await _answer(backend)).severity == "MAJOR"
    await backend.aclose()


async def test_a_partly_read_topic_is_still_loading():
    consumer = FakeConsumer([Record(STATE_KEY, _major()), Record(CONFIG_KEY, b"{}")])
    consumer.batch_size = 1
    backend = _backend(consumer)
    await backend.start()
    await _until(lambda: consumer.position_at == 1)
    polls = consumer.polls
    await _until(lambda: consumer.polls > polls + 3)

    assert await _answer(backend, deadline=0.0) == LOADING

    consumer.batch_size = None
    consumer.deliver.set()
    await _caught_up(backend)
    assert (await _answer(backend)).severity == "MAJOR"
    await backend.aclose()


async def test_a_read_before_catch_up_waits_and_answers_once_the_feed_catches_up():
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    consumer.deliver.clear()
    backend = _backend(consumer)
    await backend.start()
    await _until(lambda: consumer.polls > 3)

    async def release_after_the_read_began():
        await asyncio.sleep(0.05)
        assert not backend._caught_up
        consumer.deliver.set()

    releasing = asyncio.create_task(release_after_the_read_began())
    state = await _answer(backend, deadline=5.0)
    await releasing

    assert isinstance(state, AlarmStateRecord)
    assert state.severity == "MAJOR"
    await backend.aclose()


async def test_a_read_whose_deadline_runs_out_first_answers_loading_and_leaves_no_task():
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    consumer.deliver.clear()
    backend = _backend(consumer)
    await backend.start()
    await _until(lambda: consumer.polls > 3)
    before = asyncio.all_tasks()

    began = asyncio.get_running_loop().time()
    assert await _answer(backend, deadline=0.05) == LOADING
    assert await backend.config_for(PV, deadline=0.05) == LOADING
    assert _elapsed(began) >= 0.09

    assert asyncio.all_tasks() - before == set()
    await backend.aclose()


async def test_an_unreachable_broker_answers_at_once_without_spending_the_deadline():
    backend = _backend(FakeConsumer(start_error=ConnectionError("broker down")))
    await backend.start()

    began = asyncio.get_running_loop().time()
    state = await _answer(backend, deadline=5.0)
    config = await backend.config_for(PV, deadline=5.0)

    assert _elapsed(began) < 1.0
    for answer in (state, config):
        assert isinstance(answer, Unavailable)
        assert "cannot be reached" in answer.reason
    await backend.aclose()


async def test_an_empty_topic_answers_no_message_after_catch_up():
    backend = _backend(FakeConsumer())
    await backend.start()
    await _caught_up(backend)
    assert await _answer(backend) == EMPTY
    assert await backend.config_for(PV, deadline=1.0) == EMPTY
    assert isinstance(await backend.probe(deadline=1.0), str)
    await backend.aclose()


async def test_an_unreachable_broker_answers_with_that_reason():
    consumer = FakeConsumer(start_error=ConnectionError("broker down"))
    backend = _backend(consumer)
    await backend.start()
    await _until(lambda: backend._unreachable is not None)

    for answer in (
        await _answer(backend),
        await backend.config_for(PV, deadline=1.0),
        await backend.probe(deadline=1.0),
    ):
        assert isinstance(answer, Unavailable)
        assert "cannot be reached" in answer.reason
        assert "broker down" in answer.reason
    await backend.aclose()


async def _assert_unreachable(backend: PhoebusAlarmState) -> None:
    for answer in (
        await _answer(backend),
        await backend.config_for(PV, deadline=1.0),
        await backend.probe(deadline=1.0),
    ):
        assert isinstance(answer, Unavailable)
        assert "cannot be reached" in answer.reason


async def test_a_broker_lost_while_loading_stops_answering_loading(clock):
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    consumer.deliver.clear()
    backend = _backend(consumer)
    await backend.start()
    await _until(lambda: consumer.polls > 3)
    assert await _answer(backend, deadline=0.0) == LOADING

    clock.now += _phoebus_kafka._LOAD_DEADLINE_S
    await _until(lambda: backend._unreachable is not None)

    await _assert_unreachable(backend)
    assert consumer.stopped is True
    await backend.aclose()


async def test_a_message_while_loading_restarts_the_load_deadline(clock):
    consumer = FakeConsumer([Record(STATE_KEY, _major()), Record(CONFIG_KEY, b"{}")])
    consumer.batch_size = 1
    consumer.deliver.clear()
    backend = _backend(consumer)
    await backend.start()
    await _until(lambda: consumer.polls > 3)

    most_of_it = _phoebus_kafka._LOAD_DEADLINE_S * 0.75
    clock.now += most_of_it
    consumer.deliver.set()
    await _until(lambda: consumer.position_at == 1)
    clock.now += most_of_it
    polls = consumer.polls
    await _until(lambda: consumer.polls > polls + 3)
    assert await _answer(backend, deadline=0.0) == LOADING

    clock.now += most_of_it
    await _until(lambda: backend._unreachable is not None)
    await _assert_unreachable(backend)
    await backend.aclose()


async def test_a_position_that_never_resolves_counts_against_the_load_deadline(clock, monkeypatch):
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    consumer.deliver.clear()
    asked = asyncio.Event()

    async def never(_partition):
        asked.set()
        await asyncio.Event().wait()

    consumer.position = never
    monkeypatch.setattr(_phoebus_kafka, "_LOAD_DEADLINE_S", 0.01)
    backend = _backend(consumer)
    await backend.start()
    await asyncio.wait_for(asked.wait(), 5.0)
    clock.now += 1.0

    await _until(lambda: backend._unreachable is not None)
    await _assert_unreachable(backend)
    await backend.aclose()


async def test_a_broker_that_comes_back_is_read():
    consumer = FakeConsumer([Record(STATE_KEY, _major())], start_error=ConnectionError("down"))
    backend = _backend(consumer)
    await backend.start()
    await _until(lambda: backend._unreachable is not None)
    consumer.start_error = None
    await _caught_up(backend)
    assert (await _answer(backend)).severity == "MAJOR"
    assert backend._unreachable is None
    await backend.aclose()


async def test_a_feed_silent_after_catch_up_turns_stale(clock):
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    backend = _backend(consumer)
    await backend.start()
    await _caught_up(backend)
    assert (await _answer(backend)).stale is False

    consumer.end_offset_error = ConnectionError("gone")
    clock.now += ALARM_STALE_WINDOW_S + 1
    polls = consumer.polls
    await _until(lambda: consumer.polls > polls + 3)

    state = await _answer(backend)
    assert state.stale is True
    assert state.severity == "MAJOR"
    await backend.aclose()


async def test_a_successful_end_offset_check_keeps_a_quiet_feed_fresh(clock):
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    backend = _backend(consumer)
    await backend.start()
    await _caught_up(backend)

    calls = consumer.end_offset_calls
    clock.now += ALARM_STALE_WINDOW_S + 1
    await _until(lambda: consumer.end_offset_calls > calls)
    await _until(lambda: backend._last_contact == clock.now)
    assert (await _answer(backend)).stale is False
    await backend.aclose()


async def test_a_message_after_catch_up_keeps_the_feed_fresh_and_updates(clock):
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    consumer.end_offset_error = None
    backend = _backend(consumer)
    await backend.start()
    await _caught_up(backend)

    consumer.end_offset_error = ConnectionError("gone")
    clock.now += ALARM_STALE_WINDOW_S + 1
    consumer.records.append(Record(STATE_KEY, b'{"severity": "OK", "current_severity": "OK"}'))
    await _until(lambda: backend._last_contact == clock.now)

    state = await _answer(backend)
    assert state.severity == "OK"
    assert state.stale is False
    await backend.aclose()


async def test_aclose_stops_consuming_and_is_idempotent():
    consumer = FakeConsumer([Record(STATE_KEY, _major())])
    backend = _backend(consumer)
    await backend.start()
    await _caught_up(backend)

    await backend.aclose()
    await backend.aclose()

    assert consumer.stopped is True
    polls = consumer.polls
    for _ in range(5):
        await asyncio.sleep(0)
    assert consumer.polls == polls
    assert await _answer(backend) == LOADING


async def test_start_is_idempotent():
    built = []

    def factory(_settings):
        built.append(FakeConsumer())
        return built[-1]

    backend = PhoebusAlarmState(_spec(), consumer_factory=factory)
    await backend.start()
    await backend.start()
    await _caught_up(backend)
    assert len(built) == 1
    await backend.aclose()


async def test_a_lost_consumer_keeps_the_old_snapshot_until_the_new_one_catches_up():
    first = FakeConsumer([Record(STATE_KEY, _major())])
    second = FakeConsumer([Record(STATE_KEY, b'{"severity": "OK"}')])
    second.deliver.clear()
    consumers = iter([first, second])
    backend = PhoebusAlarmState(_spec(), consumer_factory=lambda _settings: next(consumers))
    await backend.start()
    await _caught_up(backend)

    async def lost(*_partitions, **_options):
        raise ConnectionError("lost")

    first.getmany = lost
    await _until(lambda: second.polls > 1)
    assert first.stopped is True
    assert (await _answer(backend)).severity == "MAJOR"

    second.deliver.set()
    await _until(lambda: backend._unreachable is None)
    assert (await _answer(backend)).severity == "OK"
    await backend.aclose()


async def test_a_missing_client_answers_the_install_reason_on_every_read(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiokafka", None)
    backend = PhoebusAlarmState(_spec(bootstrap_servers="broker:9092", topic="Demo"))
    await backend.start()

    for answer in (
        await _answer(backend),
        await backend.config_for(PV, deadline=1.0),
        await backend.probe(deadline=1.0),
    ):
        assert answer == Unavailable(
            "Kafka client not installed (install osprey-framework[alarm-state]; "
            "for an image, through OSPREY_PIP_SPEC)"
        )
    assert backend._task is None
    await backend.aclose()


async def test_a_factory_needs_no_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiokafka", None)
    backend = _backend(FakeConsumer())
    await backend.start()
    await _caught_up(backend)
    assert await _answer(backend) == EMPTY
    await backend.aclose()


async def test_missing_settings_answer_why(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiokafka", types.ModuleType("aiokafka"))
    backend = PhoebusAlarmState(_spec(topic="Demo"))
    await backend.start()
    answer = await _answer(backend)
    assert isinstance(answer, Unavailable)
    assert "bootstrap_servers" in answer.reason
    assert backend._task is None


async def test_the_default_consumer_has_no_group_reads_from_the_start_and_never_produces(
    monkeypatch,
):
    built = {}
    consumer = FakeConsumer([Record(STATE_KEY, _major())])

    def consumer_class(*topics, **options):
        built.update(topics=topics, options=options)
        return consumer

    def producer_class(*args, **kwargs):  # pragma: no cover - failing here is the assertion
        raise AssertionError("the backend must never produce")

    module = types.ModuleType("aiokafka")
    module.AIOKafkaConsumer = consumer_class
    module.AIOKafkaProducer = producer_class
    monkeypatch.setitem(sys.modules, "aiokafka", module)

    backend = PhoebusAlarmState(_spec(bootstrap_servers="broker:9092", topic="Demo"))
    await backend.start()
    await _caught_up(backend)
    await backend.aclose()

    assert built["topics"] == ("Demo",)
    assert built["options"]["bootstrap_servers"] == "broker:9092"
    assert built["options"]["group_id"] is None
    assert built["options"]["enable_auto_commit"] is False
    assert built["options"]["auto_offset_reset"] == "earliest"
    assert consumer.seeked is True


def test_the_constructor_does_no_io_and_takes_seams_it_ignores():
    def factory(_settings):  # pragma: no cover - building must not call it
        raise AssertionError("the constructor does no I/O")

    backend = PhoebusAlarmState(_spec(), consumer_factory=factory, transport=object())
    assert backend.source.target == "live"
    assert backend.source.service == "phoebus_alarm_server"
    assert backend.source.rehearsal is False
