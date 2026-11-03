"""The recorder adapter before it reads a sample: config, refusal, errors, channel index.

CANOPY loads the adapter by its ``module:Class`` path from the Mole env. A
misconfigured adapter is unhealthy and never connects; pymongo failures map onto
Mole's 503/504/502; the channel index is the union of the oldest and newest
documents' fields minus the bookkeeping ones.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import pytest
from canopy.mole.adapters.base import (
    AdapterHealth,
    ArchiverBadResponse,
    ArchiverTimeout,
    ArchiverUnavailable,
    PVNotFoundInArchiver,
)
from canopy.mole.adapters.registry import _resolve_adapter_class, build_registry
from canopy.mole.config import ArchiverInstanceConfig, MoleConfig, get_mole_config
from pymongo import errors as mongo_errors
from tests.integrations.canopy import _recorder_store as store

from osprey.integrations.canopy import mongo_recorder
from osprey.integrations.canopy.mongo_recorder import (
    RECORDER_TYPE,
    OspreyMongoRecorderAdapter,
    map_mongo_error,
)
from osprey.integrations.canopy.sidecar import MOLE_TYPES
from osprey_connectors import types

PV_A = "BTS:DIAG:BPM:01:POSITION:X"
PV_B = "BTS:DIAG:BPM:02:POSITION:X"
PV_C = "BTS:DIAG:BPM:03:POSITION:X"

#: A URL nothing listens on; a misconfigured adapter must never get that far.
UNREACHABLE_URL = "mongodb://osprey@127.0.0.1:1/osprey_archive?authSource=admin"

Make = Callable[[ArchiverInstanceConfig], OspreyMongoRecorderAdapter]


def _cfg(url: str = UNREACHABLE_URL, **overrides: object) -> ArchiverInstanceConfig:
    return ArchiverInstanceConfig(
        type=RECORDER_TYPE,
        url=url,
        api_key_env=overrides.pop("api_key_env", store.PASSWORD_ENV),
        timeout_s=10.0,
        extras=overrides.pop("extras", {"collection": store.COLLECTION}),
    )


@pytest.fixture
def forbid_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the adapter ever constructs a client."""

    def _boom(*args: object, **kwargs: object) -> None:  # pragma: no cover - a regression
        raise AssertionError("a misconfigured adapter must never create a MongoDB client")

    monkeypatch.setattr(mongo_recorder, "AsyncMongoClient", _boom)


@pytest.fixture
async def make_adapter() -> AsyncIterator[Make]:
    """Build adapters and close every one of them at teardown."""
    made: list[OspreyMongoRecorderAdapter] = []

    def make(cfg: ArchiverInstanceConfig) -> OspreyMongoRecorderAdapter:
        adapter = OspreyMongoRecorderAdapter.from_config("osprey", cfg)
        made.append(adapter)
        return adapter

    yield make
    for adapter in made:
        await adapter.close()


def test_canopy_resolves_the_type_the_sidecar_renders() -> None:
    assert MOLE_TYPES[types.MONGODB_ARCHIVER] == RECORDER_TYPE
    assert _resolve_adapter_class(RECORDER_TYPE) is OspreyMongoRecorderAdapter


# --------------------------------------------------------------------------- #
# The Mole env
# --------------------------------------------------------------------------- #

MOLE_ENV = {
    "CANOPY_MOLE__DEFAULT_ARCHIVER": "osprey",
    "CANOPY_MOLE__ARCHIVERS__OSPREY__TYPE": RECORDER_TYPE,
    "CANOPY_MOLE__ARCHIVERS__OSPREY__URL": (
        "mongodb://osprey@archiver-mongodb:27017/osprey_archive?authSource=admin"
    ),
    "CANOPY_MOLE__ARCHIVERS__OSPREY__EXTRAS__COLLECTION": "pv_history",
    "CANOPY_MOLE__ARCHIVERS__OSPREY__API_KEY_ENV": "MONGO_ROOT_PASSWORD",
    "CANOPY_MOLE__ARCHIVERS__OSPREY__TIMEOUT_S": "10",
}


def _load_mole_env(monkeypatch: pytest.MonkeyPatch, tmp_path, **override: str) -> MoleConfig:
    monkeypatch.chdir(tmp_path)  # no stray .env file leaks in
    for key, value in {**MOLE_ENV, **override}.items():
        monkeypatch.setenv(key, value)
    return get_mole_config()


@pytest.mark.usefixtures("forbid_connect")
async def test_the_mole_env_builds_the_adapter_without_connecting(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MONGO_ROOT_PASSWORD", "s3cret")
    config = _load_mole_env(monkeypatch, tmp_path)

    adapter = build_registry(config).get("osprey")
    assert isinstance(adapter, OspreyMongoRecorderAdapter)
    assert adapter.database == "osprey_archive"
    assert adapter.collection_name == "pv_history"
    assert adapter.config_error is None
    assert adapter.client_options() == {
        "password": "s3cret",
        "serverSelectionTimeoutMS": 2000,
        "timeoutMS": 10000,
        "tz_aware": True,
    }
    assert adapter.caps == {
        "max_raw": 50_000,
        "max_binned_scan": 1_000_000,
        "last_before_lookback_s": 86_400,
    }
    assert adapter._client is None
    await adapter.close()


@pytest.mark.usefixtures("forbid_connect")
async def test_env_string_caps_are_coerced_to_int(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MONGO_ROOT_PASSWORD", "s3cret")
    config = _load_mole_env(
        monkeypatch,
        tmp_path,
        CANOPY_MOLE__ARCHIVERS__OSPREY__EXTRAS__MAX_RAW="1200",
        CANOPY_MOLE__ARCHIVERS__OSPREY__EXTRAS__MAX_BINNED_SCAN="5000",
        CANOPY_MOLE__ARCHIVERS__OSPREY__EXTRAS__LAST_BEFORE_LOOKBACK_S="60",
    )
    adapter = OspreyMongoRecorderAdapter.from_config("osprey", config.archivers["osprey"])
    assert adapter.config_error is None
    assert adapter.caps == {"max_raw": 1200, "max_binned_scan": 5000, "last_before_lookback_s": 60}


# --------------------------------------------------------------------------- #
# Misconfiguration: unhealthy, never connects, every operation 503s
# --------------------------------------------------------------------------- #

MISCONFIGS = [
    pytest.param({"api_key_env": "OSPREY_TEST_UNSET_PW"}, "password", id="password-unset"),
    pytest.param({"api_key_env": "OSPREY_TEST_EMPTY_PW"}, "password", id="password-empty"),
    pytest.param({"api_key_env": None}, "api_key_env", id="no-api-key-env"),
    pytest.param(
        {"url": "mongodb://osprey@127.0.0.1:1/?authSource=admin"}, "database", id="no-database"
    ),
    pytest.param({"url": "http://127.0.0.1:1/osprey_archive"}, "URI", id="bad-uri"),
    pytest.param({"extras": {}}, "collection", id="no-collection"),
    pytest.param({"extras": {"collection": ""}}, "collection", id="empty-collection"),
    pytest.param({"extras": {"collection": "c", "max_raw": "lots"}}, "max_raw", id="cap-non-int"),
    pytest.param(
        {"extras": {"collection": "c", "max_binned_scan": "1.5"}},
        "max_binned_scan",
        id="cap-float-string",
    ),
    pytest.param(
        {"extras": {"collection": "c", "last_before_lookback_s": None}},
        "last_before_lookback_s",
        id="cap-none",
    ),
    pytest.param({"extras": {"collection": "c", "max_raw": "0"}}, "max_raw", id="cap-not-positive"),
]


@pytest.mark.parametrize(("overrides", "mentions"), MISCONFIGS)
@pytest.mark.usefixtures("forbid_connect")
async def test_a_misconfigured_adapter_is_unhealthy_without_connecting(
    overrides, mentions, monkeypatch, make_adapter
) -> None:
    monkeypatch.setenv(store.PASSWORD_ENV, "pw")
    monkeypatch.delenv("OSPREY_TEST_UNSET_PW", raising=False)
    monkeypatch.setenv("OSPREY_TEST_EMPTY_PW", "")
    adapter = make_adapter(_cfg(**overrides))

    assert adapter.config_error is not None
    assert mentions in adapter.config_error

    health = await asyncio.wait_for(adapter.health(), 0.5)
    assert isinstance(health, AdapterHealth)
    assert health.healthy is False
    assert health.last_error == adapter.config_error
    assert health.last_check_ts_ns > 0

    operations = [
        adapter.fetch_window(PV_A, 0, 1),
        adapter.fetch_binned(PV_A, 0, 1, binning="auto", target_buckets=10, bin_size_ns=None),
        adapter.fetch_last_before(PV_A, 1),
        adapter.metadata(PV_A),
        adapter.search("BTS:*", limit=10),
    ]
    for op in operations:
        with pytest.raises(ArchiverUnavailable) as info:
            await op
        assert type(info.value) is ArchiverUnavailable
        assert info.value.status_code == 503
        assert info.value.detail == health.last_error
    assert adapter._client is None


@pytest.mark.usefixtures("forbid_connect")
async def test_the_password_never_appears_in_messages(monkeypatch, make_adapter) -> None:
    monkeypatch.setenv(store.PASSWORD_ENV, "hunter2-very-secret")
    adapter = make_adapter(_cfg(extras={}))
    assert "hunter2" not in (adapter.config_error or "")
    assert "hunter2" not in repr(adapter)


@pytest.mark.usefixtures("forbid_connect")
async def test_close_is_idempotent_without_a_client(monkeypatch, make_adapter) -> None:
    monkeypatch.setenv(store.PASSWORD_ENV, "pw")
    adapter = make_adapter(_cfg())
    await adapter.close()
    await adapter.close()
    assert adapter._client is None


# --------------------------------------------------------------------------- #
# Error mapping, in its stated order
# --------------------------------------------------------------------------- #


class _TimeoutFlagged(mongo_errors.PyMongoError):
    @property
    def timeout(self) -> bool:
        return True


ERROR_TABLE = [
    # server selection timed out -> 503 (even though .timeout is True)
    pytest.param(
        mongo_errors.ServerSelectionTimeoutError("no servers"),
        ArchiverUnavailable,
        503,
        id="1-server-selection",
    ),
    # timeouts -> 504
    pytest.param(
        mongo_errors.NetworkTimeout("socket timed out"),
        ArchiverTimeout,
        504,
        id="2-network-timeout",
    ),
    pytest.param(
        mongo_errors.ExecutionTimeout("maxTimeMS", code=50),
        ArchiverTimeout,
        504,
        id="2-execution-timeout",
    ),
    pytest.param(TimeoutError(), ArchiverTimeout, 504, id="2-asyncio-timeout"),
    pytest.param(_TimeoutFlagged("csot"), ArchiverTimeout, 504, id="2-timeout-flag"),
    # bare AutoReconnect -> 503
    pytest.param(
        mongo_errors.AutoReconnect("connection reset"),
        ArchiverUnavailable,
        503,
        id="3-autoreconnect",
    ),
    # unauthorized / auth failed -> 503
    pytest.param(
        mongo_errors.OperationFailure("not authorized", code=13),
        ArchiverUnavailable,
        503,
        id="4-unauthorized",
    ),
    pytest.param(
        mongo_errors.OperationFailure("auth failed", code=18),
        ArchiverUnavailable,
        503,
        id="4-auth-failed",
    ),
    # anything else -> 502
    pytest.param(
        mongo_errors.OperationFailure("bad query", code=2),
        ArchiverBadResponse,
        502,
        id="5-operation-failure",
    ),
    pytest.param(mongo_errors.PyMongoError("weird"), ArchiverBadResponse, 502, id="5-pymongo"),
    pytest.param(
        mongo_errors.ConnectionFailure("refused"),
        ArchiverBadResponse,
        502,
        id="5-connection-failure",
    ),
]


@pytest.mark.parametrize(("exc", "expected_type", "status"), ERROR_TABLE)
def test_error_mapping_order(exc, expected_type, status) -> None:
    mapped = map_mongo_error(exc, "osprey")
    assert type(mapped) is expected_type
    assert mapped.status_code == status
    assert "osprey" in mapped.detail


@pytest.mark.parametrize(("exc", "expected_type", "status"), ERROR_TABLE)
async def test_errors_surface_through_operations(
    exc, expected_type, status, monkeypatch, make_adapter
) -> None:
    monkeypatch.setenv(store.PASSWORD_ENV, "pw")
    adapter = make_adapter(_cfg())

    async def _raise(*args: object, **kwargs: object) -> None:
        raise exc

    monkeypatch.setattr(adapter, "_query_channel_index", _raise)
    with pytest.raises(expected_type) as info:
        await adapter.fetch_window(PV_A, 0, 1)
    assert type(info.value) is expected_type
    assert info.value.status_code == status
    assert info.value.__cause__ is exc


async def test_health_reports_the_mapped_error_message(monkeypatch, make_adapter) -> None:
    monkeypatch.setenv(store.PASSWORD_ENV, "pw")
    adapter = make_adapter(_cfg())

    async def _ping() -> None:
        raise mongo_errors.OperationFailure("Authentication failed.", code=18)

    monkeypatch.setattr(adapter, "_ping", _ping)
    health = await adapter.health()
    assert health.healthy is False
    assert "Authentication failed" in (health.last_error or "")


async def test_the_health_ping_is_bounded_below_the_registry_window(
    monkeypatch, make_adapter
) -> None:
    monkeypatch.setenv(store.PASSWORD_ENV, "pw")
    adapter = make_adapter(_cfg())

    async def _hang() -> None:
        await asyncio.sleep(10)

    monkeypatch.setattr(adapter, "_ping", _hang)
    started = asyncio.get_running_loop().time()
    health = await asyncio.wait_for(adapter.health(), 2.0)
    elapsed = asyncio.get_running_loop().time() - started
    assert health.healthy is False
    assert elapsed < 1.9
    assert "1.5" in (health.last_error or "")


async def test_an_unreachable_server_answers_before_the_registry_window(
    monkeypatch, make_adapter
) -> None:
    monkeypatch.setenv(store.PASSWORD_ENV, "pw")
    adapter = make_adapter(_cfg())
    health = await asyncio.wait_for(adapter.health(), 2.0)
    assert health.healthy is False
    assert health.last_error


@pytest.mark.parametrize(
    "pv", ["BTS:BPM.X", "$where", "$gt", "_id", "date", "expireAt", "osprey_densified", ""]
)
async def test_invalid_or_reserved_names_are_not_found_before_any_lookup(
    pv, monkeypatch, make_adapter
) -> None:
    monkeypatch.setenv(store.PASSWORD_ENV, "pw")
    adapter = make_adapter(_cfg())

    async def _no_lookup(*args: object, **kwargs: object) -> None:  # pragma: no cover
        raise AssertionError("an invalid name must be rejected before any query")

    monkeypatch.setattr(adapter, "_query_channel_index", _no_lookup)
    for op in (
        adapter.fetch_window(pv, 0, 1),
        adapter.fetch_last_before(pv, 1),
        adapter.metadata(pv),
    ):
        with pytest.raises(PVNotFoundInArchiver) as info:
            await op
        assert info.value.status_code == 404
        assert info.value.error_code == "pv_not_found_in_archiver"
    assert adapter._client is None


# --------------------------------------------------------------------------- #
# Channel index against a throwaway archive store
# --------------------------------------------------------------------------- #


def _now_slot() -> datetime:
    return store.utc_slot(datetime.now(UTC).timestamp())


async def test_the_index_is_the_oldest_and_newest_fields_minus_bookkeeping(
    recorder_archive, make_adapter
) -> None:
    now = _now_slot()
    store.seed_samples(
        recorder_archive.writer,
        [
            (now - timedelta(minutes=10), {PV_A: 1.0}),
            (now - timedelta(minutes=5), {PV_C: 3.0}),  # neither oldest nor newest
            (now, {PV_B: 2.0}),
        ],
    )
    store.write_manifest(recorder_archive.collection)
    recorder_archive.collection.insert_one(
        {"date": now - timedelta(minutes=20), PV_A: 0.5, "osprey_densified": True}
    )

    adapter = make_adapter(_cfg(recorder_archive.url))
    assert await adapter.channel_index() == frozenset({PV_A, PV_B})

    assert await adapter.fetch_window(PV_A, 0, 1) == []
    with pytest.raises(PVNotFoundInArchiver):
        await adapter.fetch_window(PV_C, 0, 1)
    with pytest.raises(PVNotFoundInArchiver):
        await adapter.metadata("fingerprint")

    health = await adapter.health()
    assert health.healthy is True
    assert health.last_error is None
    assert health.latency_ms is not None


async def test_a_densified_newest_documents_marker_stays_out(
    recorder_archive, make_adapter
) -> None:
    recorder_archive.collection.insert_one(
        {"date": _now_slot(), PV_A: 1.0, "osprey_densified": True}
    )
    adapter = make_adapter(_cfg(recorder_archive.url))
    assert await adapter.channel_index() == frozenset({PV_A})


async def test_the_index_is_cached_for_thirty_seconds(
    recorder_archive, make_adapter, monkeypatch
) -> None:
    now = _now_slot()
    recorder_archive.writer.write_sample(now, {PV_A: 1.0})
    adapter = make_adapter(_cfg(recorder_archive.url))

    clock = [1000.0]
    monkeypatch.setattr(mongo_recorder.time, "monotonic", lambda: clock[0])
    assert await adapter.channel_index() == frozenset({PV_A})

    recorder_archive.writer.write_sample(now + timedelta(seconds=1), {PV_B: 2.0})
    clock[0] += 29.0
    assert await adapter.channel_index() == frozenset({PV_A})
    clock[0] += 2.0
    assert await adapter.channel_index() == frozenset({PV_A, PV_B})


async def test_an_undated_collection_is_unhealthy_with_a_message(
    recorder_archive, make_adapter
) -> None:
    store.write_manifest(recorder_archive.collection)
    adapter = make_adapter(_cfg(recorder_archive.url))
    health = await adapter.health()
    assert health.healthy is False
    assert health.last_error == (
        f"{recorder_archive.database}.{store.COLLECTION} has no dated documents"
    )
    assert await adapter.channel_index() == frozenset()
    with pytest.raises(PVNotFoundInArchiver):
        await adapter.fetch_window(PV_A, 0, 1)


async def test_a_wrong_password_maps_to_503(recorder_archive, make_adapter, monkeypatch) -> None:
    monkeypatch.setenv("OSPREY_TEST_WRONG_PW", "definitely-not-it")
    adapter = make_adapter(_cfg(recorder_archive.url, api_key_env="OSPREY_TEST_WRONG_PW"))
    with pytest.raises(ArchiverUnavailable) as info:
        await adapter.fetch_window(PV_A, 0, 1)
    assert type(info.value) is ArchiverUnavailable
    assert info.value.status_code == 503
    assert (await adapter.health()).healthy is False


async def test_close_is_idempotent_after_connecting(recorder_archive, make_adapter) -> None:
    recorder_archive.writer.write_sample(_now_slot(), {PV_A: 1.0})
    adapter = make_adapter(_cfg(recorder_archive.url))
    assert (await adapter.health()).healthy
    assert adapter._client is not None
    await adapter.close()
    assert adapter._client is None
    await adapter.close()
