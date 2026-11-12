"""The replay alarm feed: the real fold over a file, with no broker and no ``aiokafka``."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey_connectors.alarm_state import (
    AlarmConfigRecord,
    AlarmStateRecord,
    _replay,
)
from osprey_connectors.alarm_state._fold import FeedSnapshot, fold_message
from osprey_connectors.alarm_state._phoebus_kafka import PhoebusAlarmState
from osprey_connectors.alarm_state._replay import ReplayAlarmState
from osprey_connectors.service_answers import BackendSpec, Unavailable

from .test_kafka_backend import FakeConsumer, Record

FIXTURES = Path(__file__).parent / "fixtures"
OLD_FEED = FIXTURES / "old-feed.ndjson"
STOPPED_FEED = FIXTURES / "stopped-feed.ndjson"
PV = "DEMO:RF1:CAV:Temp-I"
ACKED_PV = "DEMO:VAC1:GAUGE:Pres-I"
LOADING = Unavailable("alarm feed still loading")
EMPTY = Unavailable("no message from the feed yet")
LOAD_MOMENT = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)

MESSAGES = [
    {"key": f"config:/Demo/RF/{PV}", "value": {"enabled": True, "guidance": [{"title": "Act"}]}},
    {"key": f"state:/Demo/RF/{PV}", "value": {"severity": "MINOR", "current_severity": "MINOR"}},
    {"key": f"state:/Demo/RF/{PV}", "value": {"severity": "MAJOR", "current_severity": "OK"}},
    {"key": f"state:/Demo/Vac/{ACKED_PV}", "value": {"severity": "MINOR_ACK"}},
    {"key": "state:/Demo/Vac/DEMO:VAC1:GONE", "value": {"severity": "MAJOR"}},
    {"key": "state:/Demo/Vac/DEMO:VAC1:GONE", "value": None},
    {"key": "command:/Demo/RF", "value": {"command": "acknowledge"}},
]


def _spec(replay_file: object = None) -> BackendSpec:
    settings = {} if replay_file is None else {"replay_file": replay_file}
    return BackendSpec(
        kind="alarm_state",
        type="alarm_replay",
        target="va",
        settings=settings,
        rehearsal=False,
    )


def _backend(path: object = OLD_FEED, *, project_dir: Path | None = None) -> ReplayAlarmState:
    replay_file = str(path) if isinstance(path, Path) else path
    return ReplayAlarmState(_spec(replay_file), project_dir=project_dir)


def _write(path: Path, *lines: object) -> Path:
    path.write_text("".join(f"{json.dumps(line)}\n" for line in lines), encoding="utf-8")
    return path


@pytest.fixture
def load_moment(monkeypatch) -> list[datetime]:
    moment = [LOAD_MOMENT]
    monkeypatch.setattr(_replay, "_now", lambda: moment[0])
    return moment


@pytest.fixture
def held_load(monkeypatch) -> tuple[threading.Event, threading.Event]:
    """Hold the file read until released: the first event says it began, the second frees it."""
    entered = threading.Event()
    release = threading.Event()
    read_replay = _replay._read_replay

    def held(path):
        entered.set()
        release.wait(5.0)
        return read_replay(path)

    monkeypatch.setattr(_replay, "_read_replay", held)
    return entered, release


async def test_the_snapshot_equals_the_kafka_fold_of_the_same_messages(tmp_path):
    backend = _backend(_write(tmp_path / "feed.jsonl", *MESSAGES))
    await backend.start()

    records = [
        Record(m["key"].encode(), None if m["value"] is None else json.dumps(m["value"]).encode())
        for m in MESSAGES
    ]
    kafka = PhoebusAlarmState(
        BackendSpec(
            kind="alarm_state",
            type="phoebus_alarm_server",
            target="live",
            settings={},
            rehearsal=False,
        ),
        consumer_factory=lambda _settings: FakeConsumer(records),
    )
    await kafka.start()
    try:

        async def caught_up():
            while not kafka._caught_up:
                await asyncio.sleep(0)

        await asyncio.wait_for(caught_up(), 5.0)
        folded = FeedSnapshot()
        for record in records:
            fold_message(folded, record.key, record.value)

        assert backend._snapshot == kafka._snapshot == folded
        for pv in (PV, ACKED_PV, "DEMO:VAC1:GONE", "DEMO:NOT:IN:THE:TREE"):
            assert await backend.state_for(pv, 1.0) == await kafka.state_for(pv, 1.0)
            assert await backend.config_for(pv, 1.0) == await kafka.config_for(pv, 1.0)
    finally:
        await kafka.aclose()
        await backend.aclose()


@pytest.mark.usefixtures("load_moment")
async def test_reads_answer_the_folded_record():
    backend = _backend()
    await backend.start()

    assert await backend.state_for(PV, 1.0) == AlarmStateRecord(
        severity="MAJOR",
        status="NO_ALARM",
        latched=True,
        acknowledged=False,
        message="HIHI_ALARM",
        time="2026-11-06T11:59:30.000Z",
        stale=False,
    )
    acked = await backend.state_for(ACKED_PV, 1.0)
    assert acked.severity == "MINOR"
    assert acked.acknowledged is True
    config = await backend.config_for(PV, 1.0)
    assert isinstance(config, AlarmConfigRecord)
    assert config.in_tree == "yes"
    assert await backend.state_for("DEMO:NOT:IN:THE:TREE", 1.0) is None
    assert (await backend.config_for("DEMO:NOT:IN:THE:TREE", 1.0)).in_tree == "no"


def _elapsed(since: float) -> float:
    return asyncio.get_running_loop().time() - since


async def test_a_read_before_start_is_still_loading_at_once():
    backend = _backend()
    began = asyncio.get_running_loop().time()

    assert await backend.state_for(PV, 5.0) == LOADING
    assert await backend.config_for(PV, 5.0) == LOADING
    assert await backend.probe(5.0) == LOADING
    assert _elapsed(began) < 1.0


async def test_a_read_whose_deadline_runs_out_during_the_load_is_still_loading(held_load):
    entered, release = held_load
    backend = _backend()
    starting = asyncio.create_task(backend.start())
    try:
        await asyncio.to_thread(entered.wait, 5.0)
        before = asyncio.all_tasks()

        began = asyncio.get_running_loop().time()
        assert await backend.state_for(PV, 0.05) == LOADING
        assert await backend.config_for(PV, 0.05) == LOADING
        assert _elapsed(began) >= 0.09
        assert asyncio.all_tasks() - before == set()
    finally:
        release.set()
    await starting
    assert (await backend.state_for(PV, 1.0)).severity == "MAJOR"


async def test_a_read_during_the_load_waits_and_answers_once_the_load_lands(held_load):
    entered, release = held_load
    backend = _backend()
    starting = asyncio.create_task(backend.start())
    await asyncio.to_thread(entered.wait, 5.0)

    async def release_after_the_read_began():
        await asyncio.sleep(0.05)
        assert backend._snapshot is None
        release.set()

    releasing = asyncio.create_task(release_after_the_read_began())
    try:
        state = await backend.state_for(PV, 5.0)
    finally:
        release.set()
        await releasing
        await starting

    assert isinstance(state, AlarmStateRecord)
    assert state.severity == "MAJOR"
    await backend.aclose()


async def test_a_cancelled_start_leaves_the_load_running(held_load):
    entered, release = held_load
    backend = _backend()
    starting = asyncio.create_task(backend.start())
    await asyncio.to_thread(entered.wait, 5.0)
    starting.cancel()
    await asyncio.gather(starting, return_exceptions=True)
    release.set()

    await backend.start()

    assert (await backend.state_for(PV, 1.0)).severity == "MAJOR"


@pytest.mark.usefixtures("load_moment")
async def test_times_are_rebased_so_the_newest_is_the_load_moment():
    backend = _backend()
    await backend.start()

    newest = await backend.state_for(ACKED_PV, 1.0)
    older = await backend.state_for(PV, 1.0)
    assert newest.time == "2026-11-06T12:00:00.000Z"
    assert older.time == "2026-11-06T11:59:30.000Z"


@pytest.mark.usefixtures("load_moment")
async def test_the_newest_message_need_not_be_the_last_line(tmp_path):
    backend = _backend(
        _write(
            tmp_path / "feed.jsonl",
            {
                "key": f"state:/Demo/{PV}",
                "value": {"severity": "MAJOR"},
                "time": "2020-01-01T00:01:00Z",
            },
            {
                "key": f"state:/Demo/{ACKED_PV}",
                "value": {"severity": "MINOR"},
                "time": "2020-01-01T00:00:00Z",
            },
        )
    )
    await backend.start()

    assert (await backend.state_for(PV, 1.0)).time == "2026-11-06T12:00:00.000Z"
    assert (await backend.state_for(ACKED_PV, 1.0)).time == "2026-11-06T11:59:00.000Z"


@pytest.mark.usefixtures("load_moment")
async def test_a_message_with_no_time_keeps_what_its_value_recorded(tmp_path):
    recorded = {"seconds": 1793243647, "nano": 0}
    backend = _backend(
        _write(
            tmp_path / "feed.jsonl",
            {"key": f"state:/Demo/{PV}", "value": {"severity": "MAJOR", "time": recorded}},
        )
    )
    await backend.start()

    assert (await backend.state_for(PV, 1.0)).time == "2026-10-29T03:14:07.000Z"


async def test_a_replay_of_old_timestamps_is_not_stale_when_fresh_nor_a_minute_later(load_moment):
    backend = _backend()
    await backend.start()

    assert (await backend.state_for(PV, 1.0)).stale is False
    load_moment[0] = LOAD_MOMENT + timedelta(minutes=1, seconds=1)
    assert (await backend.state_for(PV, 1.0)).stale is False
    assert (await backend.state_for(PV, 1.0)).time == "2026-11-06T11:59:30.000Z"


async def test_the_stopped_feed_is_stale():
    backend = _backend(STOPPED_FEED)
    await backend.start()

    record = await backend.state_for(PV, 1.0)
    assert record.severity == "MAJOR"
    assert record.stale is True
    assert "stopped" in await backend.probe(1.0)


async def test_a_marker_followed_by_messages_is_not_stale(tmp_path):
    backend = _backend(
        _write(
            tmp_path / "feed.jsonl",
            {"feed": "stopped"},
            {"key": f"state:/Demo/{PV}", "value": {"severity": "MAJOR"}},
        )
    )
    await backend.start()

    assert (await backend.state_for(PV, 1.0)).stale is False


@pytest.mark.parametrize("lines", [(), ({"feed": "stopped"},)])
async def test_a_replay_with_no_message_answers_no_message(tmp_path, lines):
    backend = _backend(_write(tmp_path / "feed.jsonl", *lines))
    await backend.start()

    assert await backend.state_for(PV, 1.0) == EMPTY
    assert await backend.config_for(PV, 1.0) == EMPTY
    assert isinstance(await backend.probe(1.0), str)


async def test_an_unresolved_demo_name_is_unavailable_at_every_read():
    backend = _backend("demo:example-alarms")

    before = await backend.state_for(PV, 1.0)
    assert isinstance(before, Unavailable)
    assert before != LOADING
    assert "demo:example-alarms" in before.reason

    await backend.start()
    assert await backend.state_for(PV, 1.0) == before
    assert await backend.config_for(PV, 1.0) == before
    assert await backend.probe(1.0) == before
    assert backend._task is None


async def test_a_relative_path_resolves_against_the_project_dir(tmp_path):
    _write(tmp_path / "feed.jsonl", {"key": f"state:/Demo/{PV}", "value": {"severity": "MAJOR"}})
    backend = _backend("feed.jsonl", project_dir=tmp_path)
    await backend.start()

    assert (await backend.state_for(PV, 1.0)).severity == "MAJOR"


async def test_a_relative_path_without_a_project_dir_is_unavailable():
    backend = _backend("feed.jsonl")
    await backend.start()

    answer = await backend.state_for(PV, 1.0)
    assert isinstance(answer, Unavailable)
    assert "feed.jsonl" in answer.reason


async def test_no_replay_file_is_unavailable():
    backend = _backend(None)
    await backend.start()

    answer = await backend.state_for(PV, 1.0)
    assert isinstance(answer, Unavailable)
    assert "replay_file" in answer.reason


async def test_a_missing_file_answers_with_the_reason_and_never_a_clear_state(tmp_path):
    backend = _backend(tmp_path / "absent.jsonl")
    await backend.start()

    for answer in (
        await backend.state_for(PV, 1.0),
        await backend.config_for(PV, 1.0),
        await backend.probe(1.0),
    ):
        assert isinstance(answer, Unavailable)
        assert "absent.jsonl" in answer.reason


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[1, 2]",
        '{"key": "state:/Demo/X", "value": {}, "time": "not a time"}',
        '{"key": "state:/Demo/X", "value": {}, "time": 12}',
        '{"key": "state:/Demo/X", "value": 12}',
    ],
)
async def test_a_malformed_line_fails_the_whole_load(tmp_path, line):
    path = tmp_path / "feed.jsonl"
    good = json.dumps({"key": f"state:/Demo/{PV}", "value": {"severity": "OK"}})
    path.write_text(f"{good}\n{line}\n", encoding="utf-8")
    backend = _backend(path)
    await backend.start()

    answer = await backend.state_for(PV, 1.0)
    assert isinstance(answer, Unavailable)
    assert answer != LOADING
    assert isinstance(await backend.probe(1.0), Unavailable)


async def test_start_is_idempotent(monkeypatch):
    reads = []
    read_replay = _replay._read_replay

    def counted(path):
        reads.append(path)
        return read_replay(path)

    monkeypatch.setattr(_replay, "_read_replay", counted)
    backend = _backend()

    await asyncio.gather(backend.start(), backend.start())
    await backend.start()

    assert len(reads) == 1


async def test_aclose_releases_the_load_and_is_idempotent():
    backend = _backend()
    await backend.aclose()
    await backend.start()
    assert (await backend.state_for(PV, 1.0)).severity == "MAJOR"

    await backend.aclose()
    await backend.aclose()

    assert backend._task is None
    assert backend._snapshot is None
    assert await backend.state_for(PV, 1.0) == LOADING

    await backend.start()
    assert (await backend.state_for(PV, 1.0)).severity == "MAJOR"


async def test_aclose_during_the_load_ends_the_start_without_an_answer(held_load):
    entered, release = held_load
    backend = _backend()
    starting = asyncio.create_task(backend.start())
    await asyncio.to_thread(entered.wait, 5.0)

    await backend.aclose()
    await asyncio.wait_for(starting, 5.0)
    release.set()
    await asyncio.sleep(0)

    assert await backend.state_for(PV, 1.0) == LOADING


async def test_it_runs_with_aiokafka_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiokafka", None)
    backend = _backend()
    await backend.start()

    assert (await backend.state_for(PV, 1.0)).severity == "MAJOR"
    assert isinstance(await backend.probe(1.0), str)


def test_importing_it_loads_neither_aiokafka_nor_the_kafka_backend():
    code = (
        "import sys; sys.modules['aiokafka'] = None\n"
        "import osprey_connectors.alarm_state._replay\n"
        "assert 'osprey_connectors.alarm_state._phoebus_kafka' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


# -- the list of every channel's state ----------------------------------------------


async def test_the_list_before_start_is_still_loading():
    assert await _backend().states(1.0) == LOADING


async def test_the_list_of_an_empty_replay_is_no_message(tmp_path):
    backend = _backend(_write(tmp_path / "feed.ndjson"))
    await backend.start()
    assert await backend.states(1.0) == EMPTY


@pytest.mark.usefixtures("load_moment")
async def test_the_list_holds_every_channel_its_counts_and_the_load_moment():
    backend = _backend()
    await backend.start()

    states = await backend.states(1.0)

    assert {row.channel: row.path for row in states.rows} == {
        PV: f"/Demo/RF/{PV}",
        ACKED_PV: f"/Demo/Vac/{ACKED_PV}",
    }
    for row in states.rows:
        assert row.state == await backend.state_for(row.channel, 1.0)
    assert dict(states.counts) == {"MAJOR": 1, "MINOR": 1}
    assert states.stale is False
    assert states.last_contact == "2026-11-06T12:00:00.000Z"


async def test_the_list_of_a_stopped_feed_is_stale():
    backend = _backend(STOPPED_FEED)
    await backend.start()

    states = await backend.states(1.0)
    assert states.stale is True
    assert all(row.state.stale for row in states.rows)


async def test_a_list_read_during_the_load_waits_for_it(held_load):
    entered, release = held_load
    backend = _backend()
    starting = asyncio.create_task(backend.start())
    await asyncio.to_thread(entered.wait, 5.0)
    assert await backend.states(0.05) == LOADING

    release.set()
    await starting
    assert len((await backend.states(1.0)).rows) == 2
