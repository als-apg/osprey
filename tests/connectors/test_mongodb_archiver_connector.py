"""Tests for the MongoDB archive."""

import copy
import json
import logging
import re
import ssl
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from osprey.port_layout import default_port
from osprey_connectors.archive_store import (
    _HOST_OVERRIDE_ENV,
    _PORT_OVERRIDE_ENV,
    address_overrides,
    field_name,
)
from osprey_connectors.archiver._base import ArchiveCoverage, ArchiveExtent
from osprey_connectors.archiver._mongodb_archiver_connector import (
    DEPLOY_HINT,
    MongoDBArchiverConnector,
)
from osprey_connectors.errors import ArchiveQueryError, ConnectorUnavailableError
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import BackendSpec, Unavailable
from tests.connectors._bundled_mongo import BUNDLED_CLIENT_KWARGS, bundled_block

# xdist_group("docker"): the session ``mongodb_container`` fixture starts a real
# container, and this file shares the group with the Postgres-backed ARIEL tests so
# every container start lands on one worker. Each worker is its own testcontainers
# session with its own ryuk reaper; two workers starting containers at once race the
# Docker daemon's port mapper and fail the reaper's port 8080 publish.
pytestmark = pytest.mark.xdist_group("docker")


def _spec(settings):
    """The ``va`` archive spec of a ``mongodb_archiver`` block."""
    return BackendSpec(
        kind="archive", type="mongodb_archiver", target="va", settings=settings, rehearsal=False
    )


def _archive(settings):
    """The archive *settings* configure; its settings are read here, nothing is dialled."""
    return MongoDBArchiverConnector(_spec(settings))


#: A block naming its store by url: it reads no login and no environment.
_STUB_BLOCK = {"url": "mongodb://stub.example.invalid", "name": "testdb", "collection": "testcoll"}


def _wired(collection, waveforms=None, *, timeout_s=10):
    """An archive already open on the stub *collection* (and its waveform sibling)."""
    archive = _archive({**_STUB_BLOCK, "timeout_s": timeout_s})
    archive._collection = collection
    archive._waveforms = waveforms
    return archive


@pytest.fixture
def static_mongo_config(monkeypatch):
    """A complete, archive-ready config that needs no container.

    For tests that fail when the archive is built or opened, before a store is
    reached. The in-network address overrides are cleared so a developer shell
    that sets them cannot supply the host a test has deleted.
    """
    monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
    monkeypatch.delenv(_HOST_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(_PORT_OVERRIDE_ENV, raising=False)
    return {
        "host": "mongodb.example.invalid",
        "port": 27017,
        "name": "testdb",
        "collection": "testcoll",
        "auth": {
            "source": "admin",
            "username": "user",
            "password_env": "MONGODB_MOCK_PASSWORD",
        },
        "timeout_s": 10,
    }


def _unreachable_client():
    """A patched ``MongoClient`` whose ping fails as an unreachable store does."""
    pymongo_errors = pytest.importorskip("pymongo.errors")
    patched = patch("pymongo.MongoClient")
    mock_client_cls = patched.start()
    mock_client_cls.return_value.admin.command.side_effect = pymongo_errors.ConnectionFailure(
        "refused"
    )
    return patched, mock_client_cls


class TestOpenAndCloseLifecycle:
    """The archive opens its store on the first call and closes it on ``aclose``."""

    @pytest.mark.asyncio
    async def test_the_first_call_opens_the_store(self, mongodb_config):
        """Building reads the settings only; the first call opens and pings the store."""
        archive = _archive(mongodb_config)
        assert archive._client is None

        evidence = await archive.probe()

        assert isinstance(evidence, str)
        assert "answered a ping" in evidence
        assert archive._client is not None
        assert archive._collection is not None

        await archive.aclose()

    @pytest.mark.parametrize(
        ("config_timeout", "expected"), [(None, 60), (120, 120)], ids=["default", "configured"]
    )
    def test_timeout(self, static_mongo_config, config_timeout, expected):
        """The timeout is 60 s unless the config names one. A settings read, so
        nothing is dialled."""
        config = static_mongo_config.copy()
        del config["timeout_s"]
        if config_timeout is not None:
            config["timeout_s"] = config_timeout

        archive = _archive(config)

        assert archive._timeout == expected

    @pytest.mark.asyncio
    async def test_aclose_clears_state(self, mongodb_config):
        """``aclose`` drops the client and the collection handles."""
        archive = _archive(mongodb_config)
        assert isinstance(await archive.probe(), str)

        await archive.aclose()

        assert archive._client is None
        assert archive._collection is None

    @pytest.mark.asyncio
    async def test_aclose_when_never_opened(self, static_mongo_config):
        """``aclose`` is safe on an archive that never opened its store."""
        archive = _archive(static_mongo_config)
        # Should not raise any exception
        await archive.aclose()

        assert archive._client is None
        assert archive._collection is None

    @pytest.mark.asyncio
    async def test_a_store_that_did_not_open_is_tried_again_on_the_next_call(
        self, static_mongo_config
    ):
        """No connect step to have failed: a store that is down answers, and reopens later."""
        archive = _archive(static_mongo_config)
        patched, mock_client_cls = _unreachable_client()
        try:
            first = await archive.probe()
            mock_client_cls.return_value.admin.command.side_effect = None
            second = await archive.probe()
        finally:
            patched.stop()

        assert isinstance(first, Unavailable)
        assert first.cause == "unreachable"
        assert isinstance(second, str)
        assert mock_client_cls.call_count == 2
        assert archive._collection is not None


class TestSettingsValidationWithoutDocker:
    """The settings read when the archive is built, which fails before any client is built."""

    @pytest.mark.parametrize(
        ("key", "match"),
        [
            (("host",), "host is required"),
            (("name",), "name.*database name.*required"),
            (("collection",), "collection is required"),
            (("auth", "username"), r"auth\.username is required"),
            (("auth", "password_env"), r"auth\.password_env is required"),
            (("auth", "source"), r"auth\.source \(authentication database\) is required"),
        ],
        ids=[
            "missing_host",
            "missing_database_name",
            "missing_collection",
            "missing_auth_username",
            "missing_auth_password_env",
            "missing_auth_source",
        ],
    )
    def test_a_missing_key_raises_value_error(self, static_mongo_config, key, match):
        """Each required key missing from the config (with no environment
        override to stand in for it) is refused as an authoring error."""
        config = copy.deepcopy(static_mongo_config)
        *parents, leaf = key
        section = config
        for parent in parents:
            section = section[parent]
        del section[leaf]

        with pytest.raises(ValueError, match=match):
            _archive(config)

    def test_an_unset_password_env_var_raises_connection_error(self, static_mongo_config):
        """An unset password variable is a deployment state, not a config error.

        ``osprey up`` mints the password into the project's ``.env``, so this is
        exactly what a built-but-never-deployed project hits on its first
        archiver call. It must raise ConnectorUnavailableError, which reaches the
        agent as an actionable ``connection_error`` envelope naming the fix,
        rather than as an opaque internal error.
        """
        config = static_mongo_config.copy()
        config["auth"] = {**config["auth"], "password_env": "NONEXISTENT_ENV_VAR"}

        with pytest.raises(
            ConnectorUnavailableError, match="NONEXISTENT_ENV_VAR.*is not set"
        ) as exc_info:
            _archive(config)

        assert "osprey up" in str(exc_info.value)


class TestImportErrorHandling:
    """Tests for import error handling when pymongo is missing."""

    def test_building_raises_import_error_when_pymongo_missing(self, static_mongo_config):
        """Building the archive raises ImportError with a helpful message when pymongo is missing."""
        spec = _spec(static_mongo_config)
        original_modules = sys.modules.copy()

        # Mock the import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == "pymongo" or name.startswith("pymongo."):
                raise ImportError("No module named 'pymongo'")
            return original_modules.get(name)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                MongoDBArchiverConnector(spec)

            # pymongo is a declared dependency of osprey-connectors, so the
            # message must name the package to reinstall — not an extra to
            # select, which would send the reader after one that no longer
            # exists.
            assert "pymongo is required" in str(exc_info.value)
            assert "osprey-connectors" in str(exc_info.value)
            assert "archiver-mongodb" not in str(exc_info.value)


class TestHistoryMethod:
    """Tests for the history method."""

    @pytest.mark.asyncio
    async def test_history_multiple_pvs(self, mongodb_config, mongodb_test_data):
        """history returns data for multiple PVs."""
        archive = _archive(mongodb_config)

        start_date = mongodb_test_data["start_date"]
        end_date = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        channels = mongodb_test_data["channels"]

        df = await archive.history(
            channels,
            start_at=start_date,
            end_at=end_date,
        )

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "address", "value"]
        # All PVs should be represented as channels, long-format
        assert set(df["address"]) == set(channels)
        assert len(df) > 0

        await archive.aclose()

    @pytest.mark.asyncio
    async def test_history_empty_time_range(self, mongodb_config):
        """history returns an empty DataFrame for a time range with no data."""
        archive = _archive(mongodb_config)

        # Use a time range that definitely has no data
        start_date = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        end_date = datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC)

        df = await archive.history(
            ["BEAM:CURRENT"],
            start_at=start_date,
            end_at=end_date,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["timestamp", "address", "value"]
        assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert df["value"].dtype == "float64"

        await archive.aclose()

    @pytest.mark.asyncio
    async def test_history_of_an_unreachable_store_answers_unavailable(self, static_mongo_config):
        """A store that cannot be reached is an answer, with the deploy hint, never a raise."""
        archive = _archive(static_mongo_config)
        patched, _client = _unreachable_client()
        try:
            answer = await archive.history(
                ["BEAM:CURRENT"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
            )
        finally:
            patched.stop()

        assert isinstance(answer, Unavailable)
        assert answer.cause == "unreachable"
        assert answer.reason.startswith(
            "Cannot connect to MongoDB at mongodb.example.invalid:27017"
        )
        assert DEPLOY_HINT in answer.reason

    @pytest.mark.asyncio
    async def test_history_empty_channels_raises_archive_query_error(self, static_mongo_config):
        """An empty channel list is the caller's argument: ArchiveQueryError, before any I/O."""
        archive = _archive(static_mongo_config)

        with pytest.raises(ArchiveQueryError, match="channels cannot be empty") as exc_info:
            await archive.history(
                [],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 2, tzinfo=UTC),
            )

        assert exc_info.value.parameter == "addresses"
        assert archive._client is None

    @pytest.mark.asyncio
    async def test_history_time_range_filtering(self, mongodb_config, mongodb_test_data):
        """history correctly filters by time range."""
        archive = _archive(mongodb_config)

        # Get first 6 hourly samples. The archive's range query is inclusive
        # on both ends ($gte/$lte), so end_date is the timestamp of the last
        # sample we want, not the exclusive upper bound. precision_ms matches
        # the seeded hourly cadence; bounds are tz-aware since the returned
        # timestamps are UTC.
        start_date = mongodb_test_data["start_date"].replace(tzinfo=UTC)
        end_date = datetime(2024, 1, 1, 5, 0, 0, tzinfo=UTC)

        df = await archive.history(
            ["BEAM:CURRENT"],
            start_at=start_date,
            end_at=end_date,
            precision_ms=3_600_000,
        )

        assert len(df) == 6  # hourly samples at 00:00..05:00 inclusive
        assert df["timestamp"].iloc[0] >= start_date
        assert df["timestamp"].iloc[-1] <= end_date

        await archive.aclose()


class TestCoverageMethod:
    """Tests for the coverage method."""

    @pytest.mark.asyncio
    async def test_coverage_reports_what_the_collection_holds(
        self, mongodb_config, mongodb_test_data
    ):
        """coverage reports the span the collection actually holds.

        The fixture seeds hourly documents from ``start_date`` up to (not
        including) ``end_date``, so the newest stored sample is one hour before
        ``end_date``. Asserting that exact boundary is what proves the bounds
        are read from the store rather than echoed from the request.
        """
        archive = _archive(mongodb_config)

        channel = mongodb_test_data["channels"][0]
        coverage = (await archive.coverage([channel]))[channel]

        assert isinstance(coverage, ArchiveCoverage)
        assert coverage.address == channel
        assert coverage.is_archived is True  # Should be True since we have test data
        # A known channel needs no reason: the store said.
        assert coverage.unknown_reason is None
        assert coverage.first_sample_at == mongodb_test_data["start_date"]
        assert coverage.last_sample_at == mongodb_test_data["end_date"] - timedelta(hours=1)

        await archive.aclose()

    @pytest.mark.asyncio
    async def test_coverage_nonexistent_pv(self, mongodb_config):
        """coverage answers is_archived=False for a nonexistent PV."""
        archive = _archive(mongodb_config)

        coverage = (await archive.coverage(["NONEXISTENT:PV"]))["NONEXISTENT:PV"]

        assert isinstance(coverage, ArchiveCoverage)
        assert coverage.address == "NONEXISTENT:PV"
        assert coverage.is_archived is False  # Should be False for nonexistent PV
        # No stored samples means no coverage window — not an invented one.
        assert coverage.first_sample_at is None
        assert coverage.last_sample_at is None

        await archive.aclose()

    @pytest.mark.asyncio
    async def test_coverage_mixed_pvs(self, mongodb_config, mongodb_test_data):
        """coverage correctly identifies archived and unarchived PVs."""
        archive = _archive(mongodb_config)

        # Mix of existing and nonexistent PVs
        channels = mongodb_test_data["channels"] + ["NONEXISTENT:PV"]
        coverage = await archive.coverage(channels)

        assert not isinstance(coverage, Unavailable)
        assert len(coverage) == len(channels)

        # Existing PVs should be True
        for pv in mongodb_test_data["channels"]:
            assert coverage[pv].is_archived is True

        # Nonexistent PV should be False
        assert coverage["NONEXISTENT:PV"].is_archived is False

        await archive.aclose()

    @pytest.mark.asyncio
    async def test_coverage_empty_list(self, mongodb_config):
        """coverage answers an empty mapping for empty input."""
        archive = _archive(mongodb_config)

        coverage = await archive.coverage([])

        assert coverage == {}

        await archive.aclose()

    @pytest.mark.asyncio
    async def test_coverage_of_an_unreachable_store_answers_unavailable(self, static_mongo_config):
        """A store that cannot be opened answers why, for every channel at once."""
        archive = _archive(static_mongo_config)
        patched, _client = _unreachable_client()
        try:
            answer = await archive.coverage(["BEAM:CURRENT"])
        finally:
            patched.stop()

        assert isinstance(answer, Unavailable)
        assert answer.cause == "unreachable"
        assert DEPLOY_HINT in answer.reason


def _test_collection(mongodb_container):
    """The fixture collection, through a client the caller closes."""
    from pymongo import MongoClient

    client = MongoClient(
        host=mongodb_container["host"],
        port=mongodb_container["port"],
        username=mongodb_container["username"],
        password=mongodb_container["password"],
        authSource=mongodb_container["auth_db"],
    )
    return client, client[mongodb_container["db_name"]][mongodb_container["collection_name"]]


class TestDottedChannelAddresses:
    """A channel address carrying a field name is read under its own address."""

    @pytest.mark.asyncio
    async def test_a_dotted_channel_reads_back_by_its_address(
        self, mongodb_config, mongodb_container, mongodb_test_data
    ):
        start_date = mongodb_test_data["start_date"]
        client, collection = _test_collection(mongodb_container)
        try:
            collection.insert_many(
                [
                    {
                        "date": start_date + timedelta(minutes=30 + 60 * hour),
                        field_name("BEAM:REC.RBV"): float(hour),
                    }
                    for hour in range(3)
                ]
            )
        finally:
            client.close()

        archive = _archive(mongodb_config)
        try:
            df = await archive.history(
                ["BEAM:REC.RBV", "BEAM:CURRENT"],
                start_at=start_date.replace(tzinfo=UTC),
                end_at=(start_date + timedelta(hours=4)).replace(tzinfo=UTC),
            )
            coverage = await archive.coverage(["BEAM:REC.RBV"])
        finally:
            await archive.aclose()

        assert set(df["address"]) == {"BEAM:REC.RBV", "BEAM:CURRENT"}
        dotted = df[df["address"] == "BEAM:REC.RBV"]
        assert list(dotted["value"]) == [0.0, 1.0, 2.0]
        assert coverage["BEAM:REC.RBV"].is_archived is True
        assert coverage["BEAM:REC.RBV"].address == "BEAM:REC.RBV"
        assert list(coverage) == ["BEAM:REC.RBV"]

    @pytest.mark.asyncio
    async def test_a_nested_document_is_not_read_as_a_dotted_channel(
        self, mongodb_config, mongodb_container, mongodb_test_data
    ):
        start_date = mongodb_test_data["start_date"]
        client, collection = _test_collection(mongodb_container)
        try:
            collection.insert_one(
                {"date": start_date + timedelta(minutes=30), "BEAM:REC": {"RBV": 9.0}}
            )
        finally:
            client.close()

        archive = _archive(mongodb_config)
        try:
            df = await archive.history(
                ["BEAM:REC.RBV"],
                start_at=start_date.replace(tzinfo=UTC),
                end_at=(start_date + timedelta(hours=4)).replace(tzinfo=UTC),
            )
            coverage = await archive.coverage(["BEAM:REC.RBV"])
        finally:
            await archive.aclose()

        assert len(df) == 0
        assert {address: c.is_archived for address, c in coverage.items()} == {
            "BEAM:REC.RBV": False
        }


class TestFactoryIntegration:
    """Tests for factory integration."""

    @pytest.fixture(autouse=True)
    def setup_factory(self):
        """Ensure built-in archives (including mongodb_archiver) are registered.

        Idempotent: ``ConnectorFactory._ensure_builtins()`` no-ops only when every
        required built-in is already present, and otherwise re-registers the
        missing ones, so this is safe regardless of prior test state. No
        teardown — we leave built-ins in place so subsequent tests can rely on
        them.
        """
        from osprey_connectors.factory import ConnectorFactory

        ConnectorFactory._ensure_builtins()
        yield

    @pytest.mark.asyncio
    async def test_factory_builds_the_mongodb_archive(self, mongodb_config):
        """create_service builds a MongoDBArchiverConnector from an archive spec; it opens on use."""
        archive = await ConnectorFactory.create_service(_spec(mongodb_config))

        assert isinstance(archive, MongoDBArchiverConnector)
        assert isinstance(await archive.probe(), str)

        await archive.aclose()

    @pytest.mark.asyncio
    async def test_factory_with_missing_host_raises_error(self, static_mongo_config):
        """create_service propagates the settings reader's ValueError for a missing host."""
        config_without_host = static_mongo_config.copy()
        del config_without_host["host"]

        with pytest.raises(ValueError, match="host is required"):
            await ConnectorFactory.create_service(_spec(config_without_host))


class TestQueryShapeWithoutDocker:
    """Query construction, asserted against a stub collection — no container needed."""

    def _stub_connector(self, documents):
        """Return an archive open on an in-memory stub collection."""
        captured = {}

        class _Cursor:
            def sort(self, *_args):
                return iter(documents)

        collection = MagicMock()

        def _find(query, projection):
            captured["query"] = query
            captured["projection"] = projection
            return _Cursor()

        collection.find = _find

        return _wired(collection), captured

    @pytest.mark.asyncio
    async def test_pv_existence_is_ored_not_anded(self):
        """PVs archived in separate documents must not produce an empty frame."""
        connector, captured = self._stub_connector([])

        await connector.history(
            ["BEAM:CURRENT", "BEAM:LIFETIME"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert captured["query"]["$or"] == [
            {"BEAM:CURRENT": {"$exists": True}},
            {"BEAM:LIFETIME": {"$exists": True}},
        ]
        # ANDing a top-level key per PV is wrong; none may appear.
        assert "BEAM:CURRENT" not in captured["query"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("channels", "expected"),
        [
            (["BEAM:CURRENT"], {"date": 1, "BEAM:CURRENT": 1}),
            (
                ["BEAM:CURRENT", "BEAM:LIFETIME"],
                {"date": 1, "BEAM:CURRENT": 1, "BEAM:LIFETIME": 1},
            ),
            # A PV literally named "date" is projected under its own field,
            # beside the document's instant; a repeated PV collapses into one key.
            (["date"], {"date": 1, "%64ate": 1}),
            (["BEAM:CURRENT", "BEAM:CURRENT"], {"date": 1, "BEAM:CURRENT": 1}),
        ],
    )
    async def test_projection_requests_date_plus_exactly_the_requested_pvs(
        self, channels, expected
    ):
        """``date`` must be projected alongside the PVs, and nothing else.

        Drop ``date`` and every document is skipped as missing it; projecting
        more than the requested PVs pulls whole documents over the wire.
        """
        connector, captured = self._stub_connector([])

        await connector.history(
            channels,
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert captured["projection"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("processing", "expected"),
        [("max", [1.0, 3.0, 5.0]), ("mean", [0.5, 2.5, 4.5])],
        ids=["max", "mean"],
    )
    async def test_precision_ms_bins_with_the_selected_aggregation(self, processing, expected):
        """precision_ms must actually downsample a non-raw mode rather than being a
        no-op, and ``processing`` selects the aggregation applied to each bin."""
        documents = [
            {"date": datetime(2024, 1, 1, 0, 0, s, tzinfo=UTC), "BEAM:CURRENT": float(s)}
            for s in range(6)
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            precision_ms=2000,
            processing=processing,
        )

        # Six 1-second samples binned at 2 s, i.e. three bins of two samples.
        assert df.loc[df["address"] == "BEAM:CURRENT", "value"].tolist() == expected

    @pytest.mark.asyncio
    async def test_raw_processing_decimates_to_last_real_sample_per_bin(self):
        """ "raw" with a real precision_ms keeps one real sample per bin, at its
        own real timestamp. Six 1s-apart documents at precision_ms=2000 -> three
        2s bins."""
        documents = [
            {"date": datetime(2024, 1, 1, 0, 0, s, tzinfo=UTC), "BEAM:CURRENT": float(s)}
            for s in range(6)
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            precision_ms=2000,
        )

        rows = df.loc[df["address"] == "BEAM:CURRENT"]
        assert rows["value"].tolist() == [1.0, 3.0, 5.0]
        # Real archived timestamps (t=1,3,5s), not the bin edges (t=0,2,4s).
        assert rows["timestamp"].tolist() == [
            documents[1]["date"],
            documents[3]["date"],
            documents[5]["date"],
        ]

    @pytest.mark.asyncio
    async def test_raw_processing_at_full_resolution_returns_every_sample(self):
        """precision_ms <= 0 is "raw" processing's full-resolution case: every
        real sample, unbinned."""
        documents = [
            {"date": datetime(2024, 1, 1, 0, 0, s, tzinfo=UTC), "BEAM:CURRENT": float(s)}
            for s in range(6)
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            precision_ms=0,
        )

        assert df.loc[df["address"] == "BEAM:CURRENT", "value"].tolist() == [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("processing", ["raw", "mean"])
    async def test_empty_window_returns_the_typed_empty_long_frame(self, processing):
        # Regression: the non-numeric guard ran before the emptiness check, so
        # an empty (object-dtype) channel raised a misleading "non-numeric".
        connector, _ = self._stub_connector([])

        df = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
            processing=processing,
        )

        assert list(df.columns) == ["timestamp", "address", "value"]
        assert df.empty
        assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert df["value"].dtype == "float64"

    @pytest.mark.asyncio
    async def test_timeout_zero_respected_not_falsy(self):
        """timeout=0 must mean zero, not fall back to the 60 s default."""
        connector, _ = self._stub_connector([])
        connector._timeout = 60

        answer = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
            timeout_s=0,
        )

        assert answer == Unavailable(cause="timeout", reason="MongoDB query timed out after 0s")

    @pytest.mark.asyncio
    async def test_sparse_documents_are_not_upsampled(self):
        """A fine precision_ms must not manufacture samples faster than the ingester wrote them."""
        documents = [
            {"date": datetime(2024, 1, 1, h, 0, 0, tzinfo=UTC), "BEAM:CURRENT": float(h)}
            for h in range(6)
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 1, 5, 0, 0, tzinfo=UTC),
            precision_ms=1000,
        )

        # Documents are 1 hour apart; naive 1 s resampling would reindex onto
        # an 18,001-row grid.
        values = df.loc[df["address"] == "BEAM:CURRENT", "value"]
        assert len(values) == 6
        assert values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("processing", ["raw", "mean"])
    async def test_absent_pv_contributes_no_rows_not_nan(self, processing):
        """A PV present in zero matched documents contributes no rows at all -- not a NaN row.

        Regression: under "mean", one requested PV matching zero documents used
        to raise "non-numeric" and kill the whole query.
        """
        documents = [
            {"date": datetime(2024, 1, 1, 0, 0, s, tzinfo=UTC), "BEAM:CURRENT": float(s)}
            for s in range(3)
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.history(
            ["BEAM:CURRENT", "BEAM:LIFETIME"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            processing=processing,
        )

        assert set(df["address"]) == {"BEAM:CURRENT"}

    @pytest.mark.asyncio
    async def test_document_missing_date_is_skipped(self):
        """A document without a 'date' field is skipped; its values contribute no rows."""
        documents = [
            {"BEAM:CURRENT": 99.0},  # no 'date' field: must not become a row
            {"date": datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC), "BEAM:CURRENT": 1.0},
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
            precision_ms=0,
        )

        assert df["value"].tolist() == [1.0]

    @pytest.mark.asyncio
    async def test_mixed_cadence_channels_aggregate_independently(self):
        """A 10 Hz and a much sparser channel queried together must each be
        aggregated over only their own samples.

        Regression: the deleted median-gap floor picked one bin width from the
        combined timestamp union of every requested PV; the 50 SLOW samples
        1500 ms apart dominate that median and corrupted FAST's bins (a single
        sparse sample would not have).
        """
        base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        fast_docs = [
            {"date": base + timedelta(milliseconds=i * 100), "FAST": float(i)}
            for i in range(40)  # 10 Hz, 4 true 1s bins of 10 samples each
        ]
        slow_docs = [
            {"date": base + timedelta(milliseconds=i * 1500), "SLOW": 500.0 + i}
            for i in range(50)  # far sparser, but numerous and evenly spread
        ]
        documents = sorted(fast_docs + slow_docs, key=lambda d: d["date"])
        connector, _ = self._stub_connector(documents)

        df = await connector.history(
            ["FAST", "SLOW"],
            start_at=base,
            end_at=base + timedelta(milliseconds=49 * 1500 + 100),
            precision_ms=1000,
            processing="mean",
        )

        fast_values = df.loc[df["address"] == "FAST", "value"].tolist()
        slow_values = df.loc[df["address"] == "SLOW", "value"].tolist()

        # FAST: the true per-1s-bin means.
        assert fast_values == pytest.approx([4.5, 14.5, 24.5, 34.5])
        # SLOW: its own 50 real samples, each its own bin.
        assert len(slow_values) == 50
        assert slow_values == pytest.approx([500.0 + i for i in range(50)])


class TestClientKwargsWithoutDocker:
    """The keyword arguments the connector hands pymongo, with the client patched."""

    @pytest.mark.asyncio
    async def test_a_bundled_block_builds_the_six_keyword_client(self, monkeypatch):
        monkeypatch.setenv("MONGO_ROOT_PASSWORD", "pw")
        monkeypatch.delenv(_HOST_OVERRIDE_ENV, raising=False)
        monkeypatch.delenv(_PORT_OVERRIDE_ENV, raising=False)

        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await _archive(bundled_block()).probe(), str)

        assert mock_client_cls.call_args.kwargs == BUNDLED_CLIENT_KWARGS

    @pytest.mark.asyncio
    async def test_a_ca_bundle_becomes_tls_ca_file(self, monkeypatch):
        monkeypatch.setenv("MONGO_ROOT_PASSWORD", "pw")
        monkeypatch.delenv(_HOST_OVERRIDE_ENV, raising=False)
        monkeypatch.delenv(_PORT_OVERRIDE_ENV, raising=False)
        block = {**bundled_block(), "tls": {"ca_bundle": "/etc/ssl/certs/site-ca.pem"}}

        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await _archive(block).probe(), str)

        assert mock_client_cls.call_args.kwargs == {
            **BUNDLED_CLIENT_KWARGS,
            "tlsCAFile": "/etc/ssl/certs/site-ca.pem",
        }

    @pytest.mark.asyncio
    async def test_no_ca_bundle_passes_no_tls_key(self, monkeypatch):
        monkeypatch.setenv("MONGO_ROOT_PASSWORD", "pw")
        monkeypatch.delenv(_HOST_OVERRIDE_ENV, raising=False)
        monkeypatch.delenv(_PORT_OVERRIDE_ENV, raising=False)

        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await _archive({**bundled_block(), "tls": {}}).probe(), str)

        assert mock_client_cls.call_args.kwargs == BUNDLED_CLIENT_KWARGS


class TestUrlWithoutDocker:
    """A block that names its store by a connection string, with the client patched."""

    URL = "mongodb://h1.example.org:27017,h2.example.org:27018/?replicaSet=rs0&tls=true"

    @staticmethod
    def _config(url, **extra):
        return {"url": url, "name": "testdb", "collection": "testcoll", **extra}

    @pytest.mark.asyncio
    async def test_a_url_is_passed_verbatim_as_host(self, monkeypatch):
        monkeypatch.delenv(_HOST_OVERRIDE_ENV, raising=False)
        monkeypatch.delenv(_PORT_OVERRIDE_ENV, raising=False)

        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await _archive(self._config(self.URL)).probe(), str)

        assert mock_client_cls.call_args.kwargs == {
            "host": self.URL,
            "serverSelectionTimeoutMS": 60000,
        }

    @pytest.mark.asyncio
    async def test_a_url_ignores_the_address_overrides(self, monkeypatch):
        monkeypatch.setenv(_HOST_OVERRIDE_ENV, "archiver-mongodb")
        monkeypatch.setenv(_PORT_OVERRIDE_ENV, "27017")

        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await _archive(self._config(self.URL, host="localhost")).probe(), str)

        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["host"] == self.URL
        assert "port" not in kwargs

    @pytest.mark.asyncio
    async def test_a_url_with_a_login_passes_it_as_keywords(self, monkeypatch):
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        auth = {"source": "archive", "username": "reader", "password_env": "MONGODB_MOCK_PASSWORD"}

        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await _archive(self._config(self.URL, auth=auth)).probe(), str)

        assert mock_client_cls.call_args.kwargs == {
            "host": self.URL,
            "username": "reader",
            "password": "secret",
            "authSource": "archive",
            "serverSelectionTimeoutMS": 60000,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "mongodb://u:p@h.example.org/",
            "mongodb://u@h.example.org/",
            "mongodb://h.example.org/?tlsCertificateKeyFilePassword=x",
            "mongodb://h.example.org/?tls=true;tlsCertificateKeyFilePassword=x",
        ],
    )
    async def test_a_url_carrying_a_credential_is_refused(self, url):
        with pytest.raises(ValueError, match=r"archiver\.settings\.url") as exc_info:
            _archive(self._config(url))

        assert url not in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("option", "key"),
        [
            ("authSource", "auth.source"),
            ("authsource", "auth.source"),
            ("tlsCAFile", "tls.ca_bundle"),
            ("serverSelectionTimeoutMS", "timeout_s"),
        ],
    )
    async def test_a_url_option_with_its_own_key_is_refused(self, option, key):
        url = f"mongodb://h.example.org/?{option}=x"

        with pytest.raises(ValueError, match=rf"`{option}`.*`{re.escape(key)}`"):
            _archive(self._config(url))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "option", ["tlsInsecure", "tlsAllowInvalidCertificates", "tlsAllowInvalidHostnames"]
    )
    async def test_a_url_option_that_turns_verification_off_is_refused(self, option):
        url = f"mongodb://h.example.org/?tls=true&{option}=true"

        with pytest.raises(ValueError, match=rf"`{option}`.*certificate verification"):
            _archive(self._config(url))

    @pytest.mark.asyncio
    async def test_a_malformed_url_is_refused_by_key(self):
        url = "mongodb://[::1/"

        with pytest.raises(
            ValueError, match=r"archiver\.settings\.url.*not a valid connection string"
        ):
            _archive(self._config(url))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("url", "option", "key"),
        [
            ("mongodb://h.example.org/?tls=true;authSource=x", "authSource", "auth.source"),
            (
                "mongodb://h.example.org/?tls=true;serverSelectionTimeoutMS=5",
                "serverSelectionTimeoutMS",
                "timeout_s",
            ),
        ],
    )
    async def test_an_option_after_a_semicolon_separator_is_refused(self, url, option, key):
        with pytest.raises(ValueError, match=rf"`{option}`.*`{re.escape(key)}`"):
            _archive(self._config(url))

    @pytest.mark.asyncio
    async def test_a_url_with_another_scheme_is_refused(self):
        with pytest.raises(ValueError, match="mongodb://"):
            _archive(self._config("https://h.example.org/"))

    @pytest.mark.asyncio
    async def test_the_route_is_logged_once_per_process(self, monkeypatch, caplog):
        from osprey_connectors.archiver import _mongodb_archiver_connector as module

        monkeypatch.setattr(module, "_ROUTE_LOGGED", False)
        monkeypatch.setattr(module.logger, "propagate", True, raising=False)
        caplog.set_level(logging.INFO)

        _archive(self._config(self.URL))
        _archive(self._config(self.URL))

        routes = [r.getMessage() for r in caplog.records if "reaches its store" in r.getMessage()]
        assert routes == [
            "MongoDB archiver reaches its store by `url`; host, port and the "
            "OSPREY_ARCHIVER_MONGODB_* overrides are not applied"
        ]
        assert self.URL not in caplog.text


class TestErrorHandlingWithoutDocker:
    """Exception mapping and degradation paths, with the pymongo client mocked."""

    @staticmethod
    def _erroring_connector(**side_effects):
        """Return an open archive whose named collection methods raise."""
        connector = _wired(MagicMock())
        for method, exception in side_effects.items():
            getattr(connector._collection, method).side_effect = exception
        return connector

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raised", "cause", "match"),
        [
            ("ConnectionFailure", "unreachable", "Cannot connect to MongoDB"),
            ("ConfigurationError", "misconfigured", "MongoDB configuration error"),
            (TimeoutError("unreachable"), "unreachable", "MongoDB connection failed"),
            (OSError("unreachable"), "unreachable", "MongoDB connection failed"),
            (RuntimeError("driver bug"), "failed", "MongoDB connection failed"),
        ],
    )
    async def test_open_failures_answer_unavailable(self, raised, cause, match, monkeypatch):
        """Every failure to open the store is the first call's answer, cause and words per failure."""
        if isinstance(raised, str):  # pymongo exception class, imported lazily
            pymongo_errors = pytest.importorskip("pymongo.errors")
            raised = getattr(pymongo_errors, raised)("refused")
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        config = {
            "host": "mongodb.example.invalid",
            "port": default_port("mongo"),
            "name": "testdb",
            "collection": "testcoll",
            "auth": {
                "source": "admin",
                "username": "user",
                "password_env": "MONGODB_MOCK_PASSWORD",
            },
        }

        connector = _archive(config)
        with patch("pymongo.MongoClient") as mock_client_cls:
            mock_client_cls.return_value.admin.command.side_effect = raised
            answer = await connector.probe()

        assert isinstance(answer, Unavailable)
        assert answer.cause == cause
        assert re.match(match, answer.reason)
        # A client whose ping failed is closed and not kept: the next call opens afresh.
        mock_client_cls.return_value.close.assert_called_once()
        assert connector._client is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised",
        [
            ssl.SSLError("no certificate"),
            FileNotFoundError(2, "No such file or directory"),
            IsADirectoryError(21, "Is a directory"),
            PermissionError(13, "Permission denied"),
        ],
    )
    async def test_an_unreadable_ca_bundle_is_named(self, raised, monkeypatch):
        """pymongo reads the CA file while it builds the client, before any ping."""
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        config = {
            "host": "mongodb.example.invalid",
            "port": default_port("mongo"),
            "name": "testdb",
            "collection": "testcoll",
            "auth": {
                "source": "admin",
                "username": "user",
                "password_env": "MONGODB_MOCK_PASSWORD",
            },
            "tls": {"ca_bundle": "/etc/ssl/certs/missing-site-ca.pem"},
        }

        connector = _archive(config)
        with patch("pymongo.MongoClient", side_effect=raised):
            answer = await connector.probe()

        assert isinstance(answer, Unavailable)
        assert answer.cause == "misconfigured"
        assert "tls.ca_bundle" in answer.reason
        assert str(raised) in answer.reason
        assert DEPLOY_HINT not in answer.reason

    @pytest.mark.asyncio
    async def test_a_missing_file_without_a_ca_bundle_is_not_blamed_on_one(self, monkeypatch):
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        config = {
            "host": "mongodb.example.invalid",
            "port": default_port("mongo"),
            "name": "testdb",
            "collection": "testcoll",
            "auth": {
                "source": "admin",
                "username": "user",
                "password_env": "MONGODB_MOCK_PASSWORD",
            },
        }

        connector = _archive(config)
        with patch("pymongo.MongoClient", side_effect=FileNotFoundError(2, "missing")):
            answer = await connector.probe()

        assert isinstance(answer, Unavailable)
        assert answer.cause == "unreachable"
        assert answer.reason.startswith("MongoDB connection failed")
        assert "tls.ca_bundle" not in answer.reason

    @pytest.mark.asyncio
    async def test_a_bearer_token_is_refused_by_name(self, monkeypatch):
        """The store authenticates a user and a password; a token has nowhere to go."""
        monkeypatch.setenv("MONGODB_MOCK_TOKEN", "secret")

        with pytest.raises(ValueError, match="not a bearer token") as exc_info:
            _archive(
                {
                    "host": "mongodb.example.invalid",
                    "port": default_port("mongo"),
                    "name": "testdb",
                    "collection": "testcoll",
                    "auth": {"source": "admin", "token_env": "MONGODB_MOCK_TOKEN"},
                }
            )

        assert "archiver.settings.auth" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_a_missing_port_is_refused_rather_than_guessed(self, monkeypatch):
        """No layout to derive from here, so a missing port is an authoring error.

        ``osprey-connectors`` is a separate wheel with no dependency on
        ``osprey``, so it cannot compute the ``mongo`` slot of the deployment's
        block. Guessing a number would dial a port that, on a host running two
        deployments, belongs to the other one's archive.
        """
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        monkeypatch.delenv("OSPREY_ARCHIVER_MONGODB_PORT", raising=False)

        with pytest.raises(ValueError, match="port is required for MongoDB archiver"):
            _archive(
                {
                    "host": "mongodb.example.invalid",
                    "name": "testdb",
                    "collection": "testcoll",
                    "auth": {
                        "source": "admin",
                        "username": "user",
                        "password_env": "MONGODB_MOCK_PASSWORD",
                    },
                }
            )

    @pytest.mark.asyncio
    async def test_the_environment_port_alone_satisfies_the_requirement(self, monkeypatch):
        """A container is given its address entirely through the environment."""
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        monkeypatch.setenv("OSPREY_ARCHIVER_MONGODB_HOST", "archiver-mongodb")
        monkeypatch.setenv("OSPREY_ARCHIVER_MONGODB_PORT", "27017")
        connector = _archive(
            {
                "name": "testdb",
                "collection": "testcoll",
                "auth": {
                    "source": "admin",
                    "username": "user",
                    "password_env": "MONGODB_MOCK_PASSWORD",
                },
            }
        )

        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await connector.probe(), str)

        assert mock_client_cls.call_args.kwargs["port"] == 27017

    @pytest.mark.asyncio
    async def test_aclose_swallows_close_error_and_clears_state(self):
        """A failing client.close() is logged, not raised, and state is still cleared."""
        connector = self._erroring_connector()
        connector._client = MagicMock()
        connector._client.close.side_effect = RuntimeError("socket already closed")

        await connector.aclose()

        assert connector._client is None
        assert connector._collection is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raised", "cause", "reason"),
        [
            # The configured timeout_s reads as a float, so the reason says 10.0s.
            (TimeoutError("slow server"), "timeout", "MongoDB query timed out after 10.0s"),
            (
                ConnectionError("connection reset"),
                "unreachable",
                "Network connectivity issue with MongoDB: connection reset",
            ),
            (
                ValueError("bad document"),
                "bad_reply",
                "Error retrieving data from MongoDB: bad document",
            ),
            (
                TypeError("bad document"),
                "bad_reply",
                "Error retrieving data from MongoDB: bad document",
            ),
            pytest.param(
                RuntimeError("cursor lost"),
                "failed",
                "Error retrieving data from MongoDB: cursor lost",
                id="an_unclassified_failure",
            ),
        ],
    )
    async def test_history_failures_answer_per_exception_type(self, raised, cause, reason):
        """Each failure of a history read is an answer: the documented cause and words."""
        connector = self._erroring_connector(find=raised)

        answer = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert answer == Unavailable(cause=cause, reason=reason)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("bound", "start", "end"),
        [
            ("start_at", "2024-01-01", datetime(2024, 1, 2, tzinfo=UTC)),
            ("end_at", datetime(2024, 1, 1, tzinfo=UTC), "2024-01-02"),
        ],
        ids=["start", "end"],
    )
    async def test_non_datetime_bound_is_rejected(self, bound, start, end):
        """history raises TypeError naming whichever bound is not a datetime."""
        connector = self._erroring_connector()

        with pytest.raises(TypeError, match=f"{bound} must be a datetime object"):
            await connector.history(["BEAM:CURRENT"], start_at=start, end_at=end)

        connector._collection.find.assert_not_called()

    @pytest.mark.asyncio
    async def test_coverage_query_errors_answer_unavailable(self):
        """A coverage read that fails answers why instead of raising, and claims no channel.

        An unreadable store reports no coverage rather than inventing one: not
        "not archived", and no bounds, for any channel asked.
        """
        connector = self._erroring_connector(
            find_one=RuntimeError("cursor lost"),
            count_documents=RuntimeError("cursor lost"),
        )

        answer = await connector.coverage(["BEAM:CURRENT", "BEAM:LIFETIME"])

        assert answer == Unavailable(
            cause="failed", reason="the MongoDB archive could not be read: RuntimeError"
        )

    @pytest.mark.asyncio
    async def test_history_pymongo_connection_loss_answers_unreachable(self):
        """A pymongo connection-class failure mid-query is the store being unreachable.

        Answered as ``unreachable``, not as the generic failure the last branch
        gives; the client is kept, and pymongo reconnects it on the next call.
        """
        pymongo_errors = pytest.importorskip("pymongo.errors")
        # ServerSelectionTimeoutError -> AutoReconnect -> ConnectionFailure: the
        # deepest subclass, so this proves the whole family is covered.
        lost = pymongo_errors.ServerSelectionTimeoutError("no replica set members available")

        connector = self._erroring_connector(find=lost)

        answer = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert answer == Unavailable(
            cause="unreachable",
            reason="Lost connection to MongoDB: no replica set members available",
        )
        assert connector._collection is not None

    @pytest.mark.asyncio
    async def test_a_lost_connection_is_answered_and_the_archive_is_kept(self):
        """The controls server holds one archive per target and never drops it.

        The archive answers a lost store as ``Unavailable``, so nothing raises
        for a supervisor to recover from: the tool's envelope says the archive
        could not be reached, names no recovery, and the next call is served by
        the same archive, which asks the store again.
        """
        from fastmcp.exceptions import ToolError

        from osprey.mcp_server.control_system import server_context
        from osprey.mcp_server.control_system.error_handling import (
            connector_error_handler,
            refuse_unavailable,
        )

        pymongo_errors = pytest.importorskip("pymongo.errors")
        connector = self._erroring_connector(find=pymongo_errors.AutoReconnect("primary stepped"))

        context = server_context.ControlSystemContext()
        context._config = server_context.MCPServerConfig(raw={})
        context._archives["va"] = connector
        spec = _spec(_STUB_BLOCK)

        with pytest.raises(ToolError) as raised:
            async with connector_error_handler("archiver_read", connector_name="archiver"):
                archive = await context.archive(spec)
                answer = await archive.history(
                    ["BEAM:CURRENT"],
                    start_at=datetime(2024, 1, 1, tzinfo=UTC),
                    end_at=datetime(2024, 1, 2, tzinfo=UTC),
                )
                assert isinstance(answer, Unavailable)
                refuse_unavailable("archiver_read", "archiver", answer)

        envelope = json.loads(str(raised.value))
        assert envelope["error_type"] == "connection_error"
        assert envelope["error_message"] == (
            "Failed to connect to the archiver: Lost connection to MongoDB: primary stepped."
        )
        assert "details" not in envelope
        assert await context.archive(spec) is connector
        again = await connector.history(
            ["BEAM:CURRENT"],
            start_at=datetime(2024, 1, 1, tzinfo=UTC),
            end_at=datetime(2024, 1, 2, tzinfo=UTC),
        )
        assert isinstance(again, Unavailable)
        assert connector._collection.find.call_count == 2


class TestHonestCoverageWithoutDocker:
    """``coverage`` reports the span the store actually holds."""

    @staticmethod
    def _connector_with_extent(oldest, newest):
        """An open archive whose find_one returns the given boundary docs."""
        connector = _wired(MagicMock())
        calls = []

        def _find_one(query, projection, sort):  # noqa: ARG001 - stands in for the collection's find_one, whose caller names sort
            calls.append(sort)
            if oldest is None:
                return None
            return oldest if sort == [("date", 1)] else newest

        connector._collection.find_one = _find_one
        return connector, calls

    @pytest.mark.asyncio
    async def test_reports_the_oldest_and_newest_stored_sample(self):
        """first_sample_at/end come from the store, not from a declared window."""
        first = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
        last = datetime(2025, 7, 1, 12, 0, tzinfo=UTC)
        connector, calls = self._connector_with_extent({"date": first}, {"date": last})

        metadata = (await connector.coverage(["BEAM:CURRENT"]))["BEAM:CURRENT"]

        assert metadata.is_archived is True
        assert metadata.first_sample_at == first
        assert metadata.last_sample_at == last
        # Ascending then descending: the two ends of the {date: 1} index.
        assert calls == [[("date", 1)], [("date", -1)]]

    @pytest.mark.asyncio
    async def test_unstored_pv_reports_no_coverage(self):
        """A PV with no documents is not archived and claims no coverage window.

        The mock archiver's habit of reporting first_sample_at=2000 for anything
        asked of it is precisely the fiction this replaces.
        """
        connector, calls = self._connector_with_extent(None, None)

        metadata = (await connector.coverage(["NOT:STORED"]))["NOT:STORED"]

        assert metadata.is_archived is False
        assert metadata.first_sample_at is None
        assert metadata.last_sample_at is None
        # No second lookup once the first comes back empty.
        assert calls == [[("date", 1)]]


class TestInNetworkAddressOverride:
    """The config block is the host-side truth; the environment is the
    per-container translation, and it wins.

    Both halves matter. A container handed the host-side address dials its own
    loopback and reaches nothing; a host process handed a compose alias dials a
    name that does not resolve. One config, two vantage points, and the
    environment is what says which one this process is at.
    """

    @staticmethod
    def _config(**overrides):
        config = {
            "host": "localhost",
            "port": 27017,
            "name": "testdb",
            "collection": "testcoll",
            "auth": {
                "source": "admin",
                "username": "user",
                "password_env": "MONGODB_MOCK_PASSWORD",
            },
        }
        config.update(overrides)
        return config

    def test_no_environment_means_no_override(self, monkeypatch):
        monkeypatch.delenv(_HOST_OVERRIDE_ENV, raising=False)
        monkeypatch.delenv(_PORT_OVERRIDE_ENV, raising=False)

        assert address_overrides() == (None, None)

    def test_the_environment_is_read_as_host_and_typed_port(self, monkeypatch):
        monkeypatch.setenv(_HOST_OVERRIDE_ENV, "archiver-mongodb")
        monkeypatch.setenv(_PORT_OVERRIDE_ENV, "27017")

        assert address_overrides() == ("archiver-mongodb", 27017)

    def test_whitespace_is_unset_not_an_address_of_nothing(self, monkeypatch):
        """A compose file that renders an empty value must not point the reader
        at the empty string — it must leave the configured value alone."""
        monkeypatch.setenv(_HOST_OVERRIDE_ENV, "   ")
        monkeypatch.setenv(_PORT_OVERRIDE_ENV, "  ")

        assert address_overrides() == (None, None)

    def test_the_halves_are_independent(self, monkeypatch):
        monkeypatch.setenv(_HOST_OVERRIDE_ENV, "archiver-mongodb")
        monkeypatch.delenv(_PORT_OVERRIDE_ENV, raising=False)

        assert address_overrides() == ("archiver-mongodb", None)

    def test_a_port_that_is_not_a_port_is_refused_not_ignored(self, monkeypatch):
        """Falling back to the host-side port would send a container to an
        address nothing serves, for a reason nobody is looking at."""
        monkeypatch.setenv(_PORT_OVERRIDE_ENV, "27017a")

        with pytest.raises(ValueError, match=_PORT_OVERRIDE_ENV):
            address_overrides()

    @pytest.mark.asyncio
    async def test_the_connector_dials_the_override_over_its_config(self, monkeypatch):
        """The end this exists for: the worker's `archiver_read` reaches the
        store by its network alias while config.yml still says localhost."""
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        monkeypatch.setenv(_HOST_OVERRIDE_ENV, "archiver-mongodb")
        monkeypatch.setenv(_PORT_OVERRIDE_ENV, "27017")

        connector = _archive(self._config(host="localhost", port=27117))
        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await connector.probe(), str)

        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["host"] == "archiver-mongodb"
        assert kwargs["port"] == 27017

    @pytest.mark.asyncio
    async def test_the_connector_keeps_its_config_when_no_override_is_set(self, monkeypatch):
        """The host-side path, which is every agent running outside a container."""
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        monkeypatch.delenv(_HOST_OVERRIDE_ENV, raising=False)
        monkeypatch.delenv(_PORT_OVERRIDE_ENV, raising=False)

        connector = _archive(self._config(host="facility-mongo.example.org", port=27117))
        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await connector.probe(), str)

        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["host"] == "facility-mongo.example.org"
        assert kwargs["port"] == 27117

    @pytest.mark.asyncio
    async def test_an_environment_only_address_is_a_legitimate_one(self, monkeypatch):
        """The required-ness check is about having an address, not about where
        it came from."""
        monkeypatch.setenv("MONGODB_MOCK_PASSWORD", "secret")
        monkeypatch.setenv(_HOST_OVERRIDE_ENV, "archiver-mongodb")
        config = self._config()
        del config["host"]

        connector = _archive(config)
        with patch("pymongo.MongoClient") as mock_client_cls:
            assert isinstance(await connector.probe(), str)

        assert mock_client_cls.call_args.kwargs["host"] == "archiver-mongodb"

    def test_the_recorder_reads_this_same_contract(self):
        """Two consumers, one reader — so they cannot come to disagree about
        what an empty value means or how the port is typed."""
        from osprey.services.archiver_recorder import config as recorder_config

        assert recorder_config.address_overrides is address_overrides


# ---------------------------------------------------------------------------
# Waveform channels
# ---------------------------------------------------------------------------

#: A pulsed-supply trace channel, as the demo facility names it.
WAVEFORM_CHANNEL = "SR:SEK:WAVEFORM"
WAVEFORM_T0 = datetime(2024, 1, 1, 6, 0, 0, tzinfo=UTC)
WAVEFORM_EXPIRES = datetime(2030, 1, 1, tzinfo=UTC)


def _sek_documents(count: int, spacing: timedelta = timedelta(milliseconds=100)):
    """``count`` archived SEK traces, built the way every writer builds them."""
    from osprey_connectors.simulation.archive import waveform_document
    from osprey_connectors.simulation.pulse import trace

    documents = []
    for i in range(count):
        date = WAVEFORM_T0 + i * spacing
        values = trace(WAVEFORM_CHANNEL, date.timestamp(), driver=2.5, enable=True)
        assert values.shape == (500,)
        documents.append(
            waveform_document(WAVEFORM_CHANNEL, date, values, expire_at=WAVEFORM_EXPIRES)
        )
    return documents


class _StubStore:
    """An in-memory stand-in for one pymongo collection.

    ``find`` and ``find_one`` honour the ``$exists`` field test and the date
    sort, which is all the connector asks of a store; ``raises`` makes every
    method fail the way a dropped connection does.
    """

    def __init__(self, documents=(), raises: BaseException | None = None):
        self.documents = list(documents)
        self.raises = raises
        self.calls: list[str] = []

    def _matching(self, query):
        self.calls.append("read")
        if self.raises is not None:
            raise self.raises
        fields = [next(iter(clause)) for clause in query.get("$or", [])]
        fields += [key for key in query if key not in ("$or", "date")]
        return [doc for doc in self.documents if any(field in doc for field in fields)]

    def find(self, query, projection):  # noqa: ARG002 - the stub ignores projection
        matching = self._matching(query)

        class _Cursor:
            def sort(self, key, direction):
                return iter(sorted(matching, key=lambda doc: doc[key], reverse=direction < 0))

        return _Cursor()

    def find_one(self, query, projection, sort):  # noqa: ARG002 - the stub ignores projection
        matching = sorted(self._matching(query), key=lambda doc: doc["date"])
        if not matching:
            return None
        return matching[0] if sort == [("date", 1)] else matching[-1]

    def count_documents(self, query, limit):
        return min(len(self._matching(query)), limit)


def _waveform_connector(scalars=(), waveforms=(), waveform_raises=None):
    """An open archive reading the two stub stores."""
    return _wired(_StubStore(scalars), _StubStore(waveforms, raises=waveform_raises))


def _scalar_documents():
    """Hourly scalar snapshots of one numeric channel."""
    return [{"date": WAVEFORM_T0 + timedelta(hours=h), "BEAM:CURRENT": 100.0 + h} for h in range(4)]


class TestWaveformReadWithoutDocker:
    """The waveform contract of ``history`` and ``coverage``."""

    @pytest.mark.asyncio
    async def test_raw_returns_each_stored_trace_as_a_float_list(self):
        documents = _sek_documents(3)
        connector = _waveform_connector(waveforms=documents)

        df = await connector.history(
            [WAVEFORM_CHANNEL],
            start_at=WAVEFORM_T0 - timedelta(minutes=1),
            end_at=WAVEFORM_T0 + timedelta(minutes=1),
        )

        assert list(df["address"]) == [WAVEFORM_CHANNEL] * 3
        for sample, document in zip(df["value"], documents, strict=True):
            assert isinstance(sample, list)
            assert len(sample) == 500
            assert all(type(x) is float for x in sample)
            assert sample == document[field_name(WAVEFORM_CHANNEL)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("precision_ms", [0, 1000, 60_000])
    async def test_raw_is_unbinned_whatever_precision_ms(self, precision_ms):
        """Ten traces 100 ms apart are all returned, at their own instants."""
        documents = _sek_documents(10)
        connector = _waveform_connector(waveforms=documents)

        df = await connector.history(
            [WAVEFORM_CHANNEL],
            start_at=WAVEFORM_T0,
            end_at=WAVEFORM_T0 + timedelta(seconds=5),
            precision_ms=precision_ms,
        )

        assert len(df) == 10
        expected = [pd.Timestamp(doc["date"]) for doc in documents]
        assert list(df["timestamp"]) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("processing", ["mean", "min", "max", "median", "std", "count"])
    async def test_non_raw_on_a_waveform_channel_is_refused(self, processing):
        connector = _waveform_connector(waveforms=_sek_documents(3))

        with pytest.raises(ArchiveQueryError, match=r"waveform.*processing='raw'") as excinfo:
            await connector.history(
                [WAVEFORM_CHANNEL],
                start_at=WAVEFORM_T0,
                end_at=WAVEFORM_T0 + timedelta(minutes=1),
                processing=processing,
            )
        assert WAVEFORM_CHANNEL in str(excinfo.value)
        assert (excinfo.value.parameter, excinfo.value.address) == ("processing", WAVEFORM_CHANNEL)

    @pytest.mark.asyncio
    async def test_non_raw_is_refused_even_with_no_trace_in_the_window(self):
        """A waveform channel is a waveform channel outside its stored range too."""
        connector = _waveform_connector(waveforms=_sek_documents(3))

        with pytest.raises(ArchiveQueryError, match="waveform"):
            await connector.history(
                [WAVEFORM_CHANNEL],
                start_at=WAVEFORM_T0 + timedelta(days=10),
                end_at=WAVEFORM_T0 + timedelta(days=11),
                processing="mean",
            )

    @pytest.mark.asyncio
    async def test_non_raw_on_a_scalar_channel_still_works(self):
        connector = _waveform_connector(scalars=_scalar_documents(), waveforms=_sek_documents(3))

        df = await connector.history(
            ["BEAM:CURRENT"],
            start_at=WAVEFORM_T0,
            end_at=WAVEFORM_T0 + timedelta(hours=4),
            precision_ms=7_200_000,
            processing="mean",
        )

        assert list(df["value"]) == [100.5, 102.5]

    @pytest.mark.asyncio
    async def test_scalar_and_waveform_channels_in_one_request(self):
        connector = _waveform_connector(scalars=_scalar_documents(), waveforms=_sek_documents(2))

        df = await connector.history(
            ["BEAM:CURRENT", WAVEFORM_CHANNEL],
            start_at=WAVEFORM_T0,
            end_at=WAVEFORM_T0 + timedelta(hours=4),
        )

        scalar = df[df["address"] == "BEAM:CURRENT"]["value"]
        waveform = df[df["address"] == WAVEFORM_CHANNEL]["value"]
        assert list(scalar) == [100.0, 101.0, 102.0, 103.0]
        assert [len(sample) for sample in waveform] == [500, 500]

    @pytest.mark.asyncio
    async def test_a_waveform_only_channel_is_available(self):
        connector = _waveform_connector(scalars=_scalar_documents(), waveforms=_sek_documents(1))

        coverage = await connector.coverage(["BEAM:CURRENT", WAVEFORM_CHANNEL, "NOT:STORED"])

        assert {address: c.is_archived for address, c in coverage.items()} == {
            "BEAM:CURRENT": True,
            WAVEFORM_CHANNEL: True,
            "NOT:STORED": False,
        }

    @pytest.mark.asyncio
    async def test_a_waveform_only_channel_reports_its_stored_extent(self):
        documents = _sek_documents(4, spacing=timedelta(minutes=10))
        connector = _waveform_connector(scalars=_scalar_documents(), waveforms=documents)

        metadata = (await connector.coverage([WAVEFORM_CHANNEL]))[WAVEFORM_CHANNEL]

        assert metadata.is_archived is True
        assert metadata.first_sample_at == documents[0]["date"]
        assert metadata.last_sample_at == documents[-1]["date"]

    @pytest.mark.asyncio
    async def test_the_extent_spans_both_collections(self):
        """A channel held in both stores reports the union of its coverage."""
        early = {"date": WAVEFORM_T0 - timedelta(days=1), WAVEFORM_CHANNEL: [0.0, 1.0]}
        documents = _sek_documents(2, spacing=timedelta(hours=1))
        connector = _waveform_connector(scalars=[early], waveforms=documents)

        metadata = (await connector.coverage([WAVEFORM_CHANNEL]))[WAVEFORM_CHANNEL]

        assert metadata.first_sample_at == early["date"]
        assert metadata.last_sample_at == documents[-1]["date"]

    @pytest.mark.asyncio
    async def test_an_interrupted_read_answers_unreachable_not_an_error(self):
        """A connection lost mid-read is an answer for the whole request, never "not archived"."""
        pymongo_errors = pytest.importorskip("pymongo.errors")
        lost = pymongo_errors.AutoReconnect("connection reset")
        connector = _waveform_connector(waveforms=_sek_documents(1), waveform_raises=lost)

        answer = await connector.coverage([WAVEFORM_CHANNEL])

        assert answer == Unavailable(
            cause="unreachable", reason="the MongoDB archive could not be read: AutoReconnect"
        )

    @pytest.mark.asyncio
    async def test_a_connector_without_a_waveform_handle_reads_scalars_only(self):
        connector = _waveform_connector(scalars=_scalar_documents())
        connector._waveforms = None

        df = await connector.history(
            ["BEAM:CURRENT"],
            start_at=WAVEFORM_T0,
            end_at=WAVEFORM_T0 + timedelta(hours=4),
            processing="max",
            precision_ms=3_600_000 * 4,
        )

        assert list(df["value"]) == [103.0]
        coverage = await connector.coverage([WAVEFORM_CHANNEL])
        assert coverage[WAVEFORM_CHANNEL].is_archived is False

    @pytest.mark.asyncio
    async def test_aclose_drops_the_waveform_handle(self):
        connector = _waveform_connector(waveforms=_sek_documents(1))

        await connector.aclose()

        assert connector._waveforms is None


class TestWaveformReadAgainstMongo:
    """The waveform sibling collection, read through a real store."""

    @pytest.fixture
    def waveform_store(self, mongodb_container, mongodb_test_data):  # noqa: ARG002 - seeds the scalars
        from osprey_connectors.simulation.archive import _prepare_collection, waveform_collection

        client, scalars = _test_collection(mongodb_container)
        waveforms = scalars.database[waveform_collection(scalars.name)]
        waveforms.delete_many({})
        _prepare_collection(scalars, waveforms=waveforms)
        documents = _sek_documents(5)
        waveforms.insert_many([dict(doc) for doc in documents])
        try:
            yield documents
        finally:
            waveforms.drop()
            client.close()

    @pytest.mark.asyncio
    async def test_raw_reads_the_sek_traces(self, mongodb_config, waveform_store):
        connector = _archive(mongodb_config)
        try:
            df = await connector.history(
                [WAVEFORM_CHANNEL],
                start_at=WAVEFORM_T0,
                end_at=WAVEFORM_T0 + timedelta(minutes=1),
                precision_ms=60_000,
            )
        finally:
            await connector.aclose()

        assert len(df) == len(waveform_store)
        for sample, document in zip(df["value"], waveform_store, strict=True):
            assert isinstance(sample, list)
            assert len(sample) == 500
            assert all(type(x) is float for x in sample)
            assert sample == document[field_name(WAVEFORM_CHANNEL)]

    @pytest.mark.asyncio
    async def test_coverage_and_processing(self, mongodb_config, waveform_store):
        connector = _archive(mongodb_config)
        try:
            coverage = await connector.coverage([WAVEFORM_CHANNEL, "BEAM:CURRENT"])
            with pytest.raises(ArchiveQueryError, match="waveform"):
                await connector.history(
                    [WAVEFORM_CHANNEL],
                    start_at=WAVEFORM_T0,
                    end_at=WAVEFORM_T0 + timedelta(minutes=1),
                    processing="mean",
                )
            scalar = await connector.history(
                ["BEAM:CURRENT"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 1, 4, tzinfo=UTC),
                precision_ms=3_600_000,
                processing="mean",
            )
        finally:
            await connector.aclose()

        assert {address: c.is_archived for address, c in coverage.items()} == {
            WAVEFORM_CHANNEL: True,
            "BEAM:CURRENT": True,
        }
        metadata = coverage[WAVEFORM_CHANNEL]
        assert pd.Timestamp(metadata.first_sample_at).tz_localize(None) == pd.Timestamp(
            waveform_store[0]["date"]
        ).tz_localize(None)
        assert len(scalar) > 0


class _MemoryStore:
    """An in-memory collection answering the reads ``extent`` and ``coverage`` make."""

    def __init__(self, documents, *, fail: bool = False):
        self._documents = list(documents)
        self._fail = fail

    def _check(self):
        if self._fail:
            raise RuntimeError("store down")

    def find_one(self, query, _projection=None, sort=None):
        self._check()
        rows = [d for d in self._documents if self._matches(d, query)]
        if sort:
            key, direction = sort[0]
            rows.sort(key=lambda d: d[key], reverse=direction < 0)
        return rows[0] if rows else None

    def find(self, query, _projection=None):
        self._check()
        return [d for d in self._documents if self._matches(d, query)]

    def count_documents(self, query, **_options):
        self._check()
        return sum(1 for d in self._documents if self._matches(d, query))

    @staticmethod
    def _matches(document, query):
        for key, condition in query.items():
            if isinstance(condition, dict) and "$exists" in condition:
                if (key in document) != condition["$exists"]:
                    return False
            elif document.get(key) != condition:
                return False
        return True


def _memory_connector(scalars, waveforms=(), *, fail=False):
    return _wired(_MemoryStore(scalars, fail=fail), _MemoryStore(waveforms, fail=fail))


class TestStoreExtentWithoutDocker:
    """``extent`` answers the store's channels and its sample span."""

    @pytest.mark.asyncio
    async def test_a_two_tier_store_answers_every_channel_and_its_span(self):
        base = datetime(2026, 1, 1, tzinfo=UTC)
        coarse = [
            {"_id": n, "date": base + timedelta(minutes=n), "SR:A": 1.0, "OLD:ONLY": 3.0}
            for n in range(0, 60, 10)
        ]
        dense = [
            {
                "_id": 100 + n,
                "date": base + timedelta(minutes=60, seconds=10 * n),
                "SR:A": 1.0,
                field_name("REC.RBV"): 2.0,
                "expireAt": base,
            }
            for n in range(6)
        ]
        traces = [
            {"date": base + timedelta(minutes=30), field_name("SR:WAVE"): [1.0, 2.0]},
            {"date": base + timedelta(minutes=50), field_name("SR:WAVE"): [1.0, 2.0]},
            {"date": base + timedelta(minutes=50), field_name("SR:TRACE.X"): [3.0]},
        ]
        connector = _memory_connector(coarse + dense, traces)

        extent = await connector.extent(timeout_s=10)

        # The newest documents name the channels: one dropped since is not listed.
        assert isinstance(extent, ArchiveExtent)
        assert extent.addresses == frozenset({"SR:A", "REC.RBV", "SR:WAVE", "SR:TRACE.X"})
        assert extent.oldest_at == base
        assert extent.newest_at == base + timedelta(minutes=60, seconds=50)

    @pytest.mark.asyncio
    async def test_a_failing_store_answers_unavailable_with_a_reason(self):
        connector = _memory_connector([], fail=True)

        extent = await connector.extent(timeout_s=10)

        assert extent == Unavailable(
            cause="failed", reason="the MongoDB archive could not be read: RuntimeError"
        )

    @pytest.mark.asyncio
    async def test_an_empty_store_holds_no_channels(self):
        extent = await _memory_connector([]).extent(timeout_s=10)

        assert isinstance(extent, ArchiveExtent)
        assert extent.addresses == frozenset()
        assert extent.newest_at is None

    @pytest.mark.asyncio
    async def test_a_failed_read_is_unavailable_not_unarchived(self):
        connector = _memory_connector([], fail=True)

        coverage = await connector.coverage(["SR:A", "SR:B"])

        assert isinstance(coverage, Unavailable)
        assert "could not be read" in coverage.reason

    @pytest.mark.asyncio
    async def test_a_failed_read_never_answers_the_exception_text(self):
        connector = _memory_connector([])
        leak = ConnectionError("cannot reach mongodb://reader:secret@archive.example:27017")

        class Leaking:
            def find_one(self, *_args, **_kwargs):
                raise leak

            find = find_one

        connector._collection = connector._waveforms = Leaking()

        coverage = await connector.coverage(["SR:A"])
        extent = await connector.extent(timeout_s=10)

        reason = "the MongoDB archive could not be read: ConnectionError"
        assert coverage == Unavailable(cause="unreachable", reason=reason)
        assert extent == Unavailable(cause="unreachable", reason=reason)
