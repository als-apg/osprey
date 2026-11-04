"""The recorder adapter's binned reads against a throwaway archive store.

Pins ``fetch_binned``: the ``max_binned_scan`` cap (400, never a truncated
answer), the ``auto`` width and bucket-count rules (target clamped to 5,000),
the 5,000-bucket ceiling for ``time`` binning, buckets anchored on ``t0`` and
the population standard deviation.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import AsyncIterator, Callable

import pytest
from canopy.contracts.sample import AlarmSeverity
from canopy.core.errors import BadRequest
from canopy.mole.adapters.base import BinningNotSupported, PVNotFoundInArchiver
from canopy.mole.config import ArchiverInstanceConfig
from tests.integrations.canopy import _recorder_store as store

from osprey.integrations.canopy.mongo_recorder import RECORDER_TYPE, OspreyMongoRecorderAdapter

PV_A = "BTS:DIAG:BPM:01:POSITION:X"
PV_S = "BTS:DIAG:STATE"

#: A fixed whole-second epoch (2026-09-30T12:00:00Z) so expectations are exact.
BASE_S = 1_790_769_600
NS_PER_S = 1_000_000_000
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


async def _auto(adapter, t0: int, t1: int, target: int, pv: str = PV_A):
    return await adapter.fetch_binned(
        pv, t0_ns=t0, t1_ns=t1, binning="auto", target_buckets=target, bin_size_ns=None
    )


async def _time(adapter, t0: int, t1: int, bin_size: int, pv: str = PV_A):
    return await adapter.fetch_binned(
        pv, t0_ns=t0, t1_ns=t1, binning="time", target_buckets=None, bin_size_ns=bin_size
    )


async def test_auto_buckets_are_anchored_on_t0_with_population_stats(
    recorder_archive, make_adapter
) -> None:
    _seed_seconds(recorder_archive, 10)
    store.write_manifest(recorder_archive.collection)
    adapter = make_adapter(_cfg(recorder_archive.url))
    t0, t1 = BASE_NS, BASE_NS + 9 * NS_PER_S

    points = await _auto(adapter, t0, t1, 3)

    # w = ceil(9 s / 3) = 3 s; n = min(ceil((9 s + 1) / 3 s), 3) = 3; the last
    # bucket absorbs the sample at t1.
    w = 3 * NS_PER_S
    assert [p.ts_ns for p in points] == [t0, t0 + w, t0 + 2 * w]
    assert all(p.bin_ns == w for p in points)
    assert [p.count for p in points] == [3, 3, 4]
    groups = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]
    for p, values in zip(points, groups, strict=True):
        assert p.mean == pytest.approx(statistics.fmean(values))
        assert p.std_dev == pytest.approx(statistics.pstdev(values))
        assert p.min == min(values) and p.max == max(values)
        assert p.severity is AlarmSeverity.NO_ALARM


async def test_one_value_bucket_has_zero_std_dev(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    points = await _time(adapter, BASE_NS, BASE_NS + 2 * NS_PER_S, NS_PER_S)

    assert [p.count for p in points] == [1, 1, 1]
    assert all(p.std_dev == 0.0 for p in points)
    assert all(p.mean == p.min == p.max for p in points)


async def test_empty_buckets_are_omitted(recorder_archive, make_adapter) -> None:
    store.seed_samples(
        recorder_archive.writer,
        [(store.utc_slot(BASE_S), {PV_A: 1.0}), (store.utc_slot(BASE_S + 9), {PV_A: 2.0})],
    )
    adapter = make_adapter(_cfg(recorder_archive.url))

    points = await _time(adapter, BASE_NS, BASE_NS + 9 * NS_PER_S, NS_PER_S)

    assert [p.ts_ns for p in points] == [BASE_NS, BASE_NS + 9 * NS_PER_S]
    assert all(p.bin_ns > 0 for p in points)


async def test_t0_equal_t1_with_one_doc_is_one_bucket(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))
    ts = BASE_NS + NS_PER_S

    (point,) = await _auto(adapter, ts, ts, 5000)

    assert point.ts_ns == ts and point.bin_ns == 1 and point.count == 1
    assert point.mean == 1.0 and point.std_dev == 0.0


async def test_inverted_window_is_empty(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    assert await _auto(adapter, BASE_NS + 2 * NS_PER_S, BASE_NS, 10) == []


async def test_window_over_max_raw_bins_fully(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 50)
    adapter = make_adapter(_cfg(recorder_archive.url, max_raw=5))
    t0, t1 = BASE_NS, BASE_NS + 49 * NS_PER_S

    with pytest.raises(BadRequest):
        await adapter.fetch_window(PV_A, t0, t1)
    points = await _auto(adapter, t0, t1, 7)

    assert sum(p.count for p in points) == 50
    assert points[-1].ts_ns <= t1 and points[-1].max == 49.0
    assert all(p.bin_ns > 0 for p in points)
    assert [p.ts_ns for p in points] == sorted(p.ts_ns for p in points)


async def test_scan_over_max_binned_scan_is_a_400_not_a_truncation(
    recorder_archive, make_adapter
) -> None:
    _seed_seconds(recorder_archive, 6)
    adapter = make_adapter(_cfg(recorder_archive.url, max_binned_scan=5))

    with pytest.raises(BadRequest) as info:
        await _auto(adapter, BASE_NS, BASE_NS + 10 * NS_PER_S, 10)
    assert info.value.status_code == 400
    assert "max_binned_scan" in info.value.detail and "5" in info.value.detail

    points = await _auto(adapter, BASE_NS, BASE_NS + 4 * NS_PER_S, 10)
    assert sum(p.count for p in points) == 5


async def test_time_binning_over_5000_buckets_is_a_400(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    with pytest.raises(BadRequest) as info:
        await _time(adapter, BASE_NS, BASE_NS + 5000 * NS_PER_S, NS_PER_S)
    assert info.value.status_code == 400
    assert "5000" in info.value.detail.replace(",", "")

    points = await _time(adapter, BASE_NS, BASE_NS + 5000 * NS_PER_S - 1, NS_PER_S)
    assert sum(p.count for p in points) == 3


async def test_auto_at_span_5000_w_returns_at_most_5000_points(
    recorder_archive, make_adapter
) -> None:
    """Span = 5000 w: the raw bucket count would be 5001; it is capped at 5000."""
    count = 5001
    _seed_seconds(recorder_archive, count)
    adapter = make_adapter(_cfg(recorder_archive.url))
    t0, t1 = BASE_NS, BASE_NS + 5000 * NS_PER_S

    points = await _auto(adapter, t0, t1, 5000)

    assert len(points) <= 5000
    assert all(p.bin_ns == NS_PER_S for p in points)
    assert sum(p.count for p in points) == count
    last = points[-1]
    assert last.ts_ns == t0 + 4999 * NS_PER_S and last.count == 2 and last.max == 5000.0


async def test_auto_target_is_clamped_to_5000(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))
    t0, t1 = BASE_NS, BASE_NS + 10_000 * NS_PER_S

    points = await _auto(adapter, t0, t1, 1_000_000)

    w = math.ceil((t1 - t0) / 5000)
    assert all(p.bin_ns == w for p in points)
    assert sum(p.count for p in points) == 3


@pytest.mark.parametrize(
    ("binning", "target", "bin_size"),
    [("auto", None, None), ("auto", 0, None), ("time", None, None), ("time", None, 0), ("x", 5, 5)],
)
async def test_unservable_binning_arguments_are_binning_not_supported(
    recorder_archive, make_adapter, binning, target, bin_size
) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    with pytest.raises(BinningNotSupported):
        await adapter.fetch_binned(
            PV_A,
            t0_ns=BASE_NS,
            t1_ns=BASE_NS + 2 * NS_PER_S,
            binning=binning,
            target_buckets=target,
            bin_size_ns=bin_size,
        )


async def test_unknown_pv_is_not_found(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    adapter = make_adapter(_cfg(recorder_archive.url))

    with pytest.raises(PVNotFoundInArchiver):
        await _auto(adapter, BASE_NS, BASE_NS + 2 * NS_PER_S, 5, pv="NOPE:PV")


async def test_string_samples_are_not_binned(recorder_archive, make_adapter) -> None:
    _seed_seconds(recorder_archive, 3)
    recorder_archive.collection.update_many({}, {"$set": {PV_S: "OPEN"}})
    adapter = make_adapter(_cfg(recorder_archive.url))

    assert await _auto(adapter, BASE_NS, BASE_NS + 2 * NS_PER_S, 5, pv=PV_S) == []
