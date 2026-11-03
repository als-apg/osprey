"""The recorder adapter's raw reads against a throwaway archive store.

Pins ``fetch_window`` and ``fetch_last_before``: the ``max_raw`` cap (400, never
a truncated answer), the exact-nanosecond post-filter around the
millisecond-floored/ceiled Mongo bounds, value mapping, the last-before
lookback, and a document written by the recorder itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import pytest
from canopy.contracts.sample import AlarmSeverity, Sample
from canopy.core.errors import BadRequest
from canopy.mole.adapters.base import PVNotFoundInArchiver
from canopy.mole.config import ArchiverInstanceConfig
from tests.integrations.canopy import _recorder_store as store

from osprey.integrations.canopy.mongo_recorder import RECORDER_TYPE, OspreyMongoRecorderAdapter

PV_A = "BTS:DIAG:BPM:01:POSITION:X"
PV_B = "BTS:DIAG:BPM:02:POSITION:X"
PV_S = "BTS:DIAG:STATE"

#: A fixed whole-second epoch (2026-09-30T12:00:00Z) so expectations are exact.
BASE_S = 1_790_769_600
NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
BASE_NS = BASE_S * NS_PER_S

Make = Callable[[ArchiverInstanceConfig], OspreyMongoRecorderAdapter]


def _cfg(url: str, **extras: object) -> ArchiverInstanceConfig:
    return ArchiverInstanceConfig(
        type=RECORDER_TYPE,
        url=url,
        api_key_env=store.PASSWORD_ENV,
        timeout_s=10.0,
        extras={"collection": store.COLLECTION, **extras},
    )


@pytest.fixture
async def make_adapter() -> AsyncIterator[Make]:
    """Build adapters and close every one of them at teardown."""
    made: list[OspreyMongoRecorderAdapter] = []

    def make(cfg: ArchiverInstanceConfig) -> OspreyMongoRecorderAdapter:
        adapter = OspreyMongoRecorderAdapter.from_config("osprey", cfg)
        assert adapter.config_error is None, adapter.config_error
        made.append(adapter)
        return adapter

    yield make
    for adapter in made:
        await adapter.close()


def _seed_seconds(archive, count: int, *, pv: str = PV_A) -> None:
    """One recorder sample per second for *count* seconds, value = index."""
    store.seed_samples(
        archive.writer, ((store.utc_slot(BASE_S + i), {pv: float(i)}) for i in range(count))
    )


def _at_ms(ms_after_base: int) -> datetime:
    return datetime.fromtimestamp(BASE_S, tz=UTC) + timedelta(milliseconds=ms_after_base)


async def test_fetch_window_reads_every_sample_in_order(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 10)
    store.write_manifest(recorder_archive.collection)
    adapter = make_adapter(_cfg(recorder_archive.url))

    samples = await adapter.fetch_window(PV_A, BASE_NS, BASE_NS + 9 * NS_PER_S)
    assert [s.ts_ns for s in samples] == [BASE_NS + i * NS_PER_S for i in range(10)]
    assert [s.value for s in samples] == [float(i) for i in range(10)]
    assert all(s.severity is AlarmSeverity.NO_ALARM for s in samples)
    assert all(s.source == "archive" for s in samples)


async def test_fetch_window_is_inclusive_and_bounded(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 10)
    adapter = make_adapter(_cfg(recorder_archive.url))

    samples = await adapter.fetch_window(PV_A, BASE_NS + 2 * NS_PER_S, BASE_NS + 5 * NS_PER_S)
    assert [s.value for s in samples] == [2.0, 3.0, 4.0, 5.0]


async def test_fetch_window_post_filters_to_the_exact_nanosecond(
    recorder_archive, make_adapter
) -> None:
    """Mongo bounds are floored/ceiled to the ms; the answer is exact to the ns."""
    for ms, value in ((0, 1.0), (5, 2.0), (10, 3.0)):
        recorder_archive.writer.write_sample(_at_ms(ms), {PV_A: value})
    adapter = make_adapter(_cfg(recorder_archive.url))

    samples = await adapter.fetch_window(PV_A, BASE_NS + 1, BASE_NS + 10 * NS_PER_MS - 1)
    assert [(s.ts_ns, s.value) for s in samples] == [(BASE_NS + 5 * NS_PER_MS, 2.0)]

    t = BASE_NS + 5 * NS_PER_MS
    assert await adapter.fetch_window(PV_A, t + 1, t + 999) == []
    assert await adapter.fetch_window(PV_A, t - 999, t - 1) == []
    assert [s.value for s in await adapter.fetch_window(PV_A, t, t)] == [2.0]


async def test_fetch_window_skips_documents_without_the_pv(recorder_archive, make_adapter) -> None:
    store.seed_samples(
        recorder_archive.writer,
        [
            (store.utc_slot(BASE_S), {PV_A: 1.0, PV_B: 10.0}),
            (store.utc_slot(BASE_S + 1), {PV_B: 11.0}),
            (store.utc_slot(BASE_S + 2), {PV_A: 3.0, PV_B: 12.0}),
        ],
    )
    adapter = make_adapter(_cfg(recorder_archive.url))

    samples = await adapter.fetch_window(PV_A, BASE_NS, BASE_NS + 2 * NS_PER_S)
    assert [(s.ts_ns, s.value) for s in samples] == [(BASE_NS, 1.0), (BASE_NS + 2 * NS_PER_S, 3.0)]


async def test_fetch_window_maps_bool_and_int_to_float_and_keeps_strings(
    recorder_archive, make_adapter
) -> None:
    recorder_archive.collection.insert_many(
        [
            {"date": store.utc_slot(BASE_S), PV_A: True, PV_S: "OPEN"},
            {"date": store.utc_slot(BASE_S + 1), PV_A: 7, PV_S: "CLOSED"},
            {"date": store.utc_slot(BASE_S + 2), PV_A: 2.5, PV_S: "OPEN"},
            {"date": store.utc_slot(BASE_S + 3), PV_A: False, PV_S: "MOVING"},
        ]
    )
    adapter = make_adapter(_cfg(recorder_archive.url))

    numeric = await adapter.fetch_window(PV_A, BASE_NS, BASE_NS + 3 * NS_PER_S)
    assert [s.value for s in numeric] == [1.0, 7.0, 2.5, 0.0]
    assert all(type(s.value) is float for s in numeric)

    strings = await adapter.fetch_window(PV_S, BASE_NS, BASE_NS + 3 * NS_PER_S)
    assert [s.value for s in strings] == ["OPEN", "CLOSED", "OPEN", "MOVING"]


async def test_fetch_window_at_the_cap_answers(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 5)
    adapter = make_adapter(_cfg(recorder_archive.url, max_raw=5))

    assert len(await adapter.fetch_window(PV_A, BASE_NS, BASE_NS + 10 * NS_PER_S)) == 5


async def test_fetch_window_over_the_cap_is_a_400_not_a_truncation(
    recorder_archive, make_adapter
) -> None:
    _seed_seconds(recorder_archive, 6)
    adapter = make_adapter(_cfg(recorder_archive.url, max_raw=5))

    with pytest.raises(BadRequest) as info:
        await adapter.fetch_window(PV_A, BASE_NS, BASE_NS + 10 * NS_PER_S)
    assert info.value.status_code == 400
    assert "max_raw" in info.value.detail and "5" in info.value.detail

    assert len(await adapter.fetch_window(PV_A, BASE_NS, BASE_NS + 4 * NS_PER_S)) == 5


async def test_fetch_window_cap_comes_from_extras_as_a_string(
    recorder_archive, make_adapter
) -> None:
    """The Mole env delivers extras as strings; ``from_config`` coerces them."""
    _seed_seconds(recorder_archive, 4)
    adapter = make_adapter(_cfg(recorder_archive.url, max_raw="3"))

    with pytest.raises(BadRequest):
        await adapter.fetch_window(PV_A, BASE_NS, BASE_NS + 10 * NS_PER_S)


async def test_fetch_window_unknown_pv_is_not_found(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    with pytest.raises(PVNotFoundInArchiver):
        await adapter.fetch_window(PV_B, BASE_NS, BASE_NS + 3 * NS_PER_S)


async def test_fetch_window_empty_and_inverted_ranges(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    assert (
        await adapter.fetch_window(PV_A, BASE_NS + 100 * NS_PER_S, BASE_NS + 200 * NS_PER_S) == []
    )
    assert await adapter.fetch_window(PV_A, BASE_NS + 2 * NS_PER_S, BASE_NS) == []


async def test_fetch_last_before_is_the_newest_earlier_sample(
    recorder_archive, make_adapter
) -> None:
    _seed_seconds(recorder_archive, 5)
    adapter = make_adapter(_cfg(recorder_archive.url))

    prior = await adapter.fetch_last_before(PV_A, BASE_NS + 3 * NS_PER_S + 500)
    assert prior == Sample(
        ts_ns=BASE_NS + 3 * NS_PER_S, value=3.0, severity=AlarmSeverity.NO_ALARM, source="archive"
    )


async def test_fetch_last_before_includes_an_exact_hit(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 5)
    adapter = make_adapter(_cfg(recorder_archive.url))

    prior = await adapter.fetch_last_before(PV_A, BASE_NS + 2 * NS_PER_S)
    assert prior is not None
    assert (prior.ts_ns, prior.value) == (BASE_NS + 2 * NS_PER_S, 2.0)


async def test_fetch_last_before_one_ns_before_a_stored_ms(recorder_archive, make_adapter) -> None:
    """``$lte`` is floored to the ms, so a stored ms one ns later is never returned."""
    recorder_archive.writer.write_sample(_at_ms(0), {PV_A: 1.0})
    recorder_archive.writer.write_sample(_at_ms(7), {PV_A: 2.0})
    adapter = make_adapter(_cfg(recorder_archive.url))

    prior = await adapter.fetch_last_before(PV_A, BASE_NS + 7 * NS_PER_MS - 1)
    assert prior is not None
    assert (prior.ts_ns, prior.value) == (BASE_NS, 1.0)


async def test_fetch_last_before_within_the_lookback(recorder_archive, make_adapter) -> None:
    recorder_archive.writer.write_sample(store.utc_slot(BASE_S), {PV_A: 4.0})
    adapter = make_adapter(_cfg(recorder_archive.url, last_before_lookback_s=60))

    prior = await adapter.fetch_last_before(PV_A, BASE_NS + 59 * NS_PER_S)
    assert prior is not None and prior.value == 4.0
    edge = await adapter.fetch_last_before(PV_A, BASE_NS + 60 * NS_PER_S)
    assert edge is not None and edge.ts_ns == BASE_NS
    assert await adapter.fetch_last_before(PV_A, BASE_NS + 61 * NS_PER_S) is None


async def test_fetch_last_before_nothing_earlier_is_none(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    assert await adapter.fetch_last_before(PV_A, BASE_NS - 1) is None


async def test_fetch_last_before_skips_documents_without_the_pv(
    recorder_archive, make_adapter
) -> None:
    store.seed_samples(
        recorder_archive.writer,
        [
            (store.utc_slot(BASE_S), {PV_A: 1.0, PV_B: 10.0}),
            (store.utc_slot(BASE_S + 1), {PV_B: 11.0}),
        ],
    )
    adapter = make_adapter(_cfg(recorder_archive.url))

    prior = await adapter.fetch_last_before(PV_A, BASE_NS + 5 * NS_PER_S)
    assert prior is not None
    assert (prior.ts_ns, prior.value) == (BASE_NS, 1.0)


async def test_fetch_last_before_unknown_pv_is_not_found(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    with pytest.raises(PVNotFoundInArchiver):
        await adapter.fetch_last_before(PV_B, BASE_NS + NS_PER_S)


async def test_reads_back_what_the_recorder_wrote_beside_the_manifest(
    recorder_archive, make_adapter
) -> None:
    recorder_archive.writer.write_sample(store.utc_slot(BASE_S), {PV_A: 0.125, PV_B: -3.5})
    store.write_manifest(recorder_archive.collection)
    adapter = make_adapter(_cfg(recorder_archive.url))

    window = await adapter.fetch_window(PV_A, BASE_NS - NS_PER_S, BASE_NS + NS_PER_S)
    assert window == [
        Sample(ts_ns=BASE_NS, value=0.125, severity=AlarmSeverity.NO_ALARM, source="archive")
    ]
    prior = await adapter.fetch_last_before(PV_B, BASE_NS + NS_PER_S)
    assert prior == Sample(
        ts_ns=BASE_NS, value=-3.5, severity=AlarmSeverity.NO_ALARM, source="archive"
    )
