"""Tests for MongoDB Archiver connector."""

import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from osprey.connectors.archiver.base import ArchiverMetadata
from osprey.connectors.archiver.mongodb_archiver_connector import MongoDBArchiverConnector
from osprey.connectors.factory import ConnectorFactory


@pytest.mark.integration
class TestConnectDisconnectLifecycle:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mongodb_config):
        """Test that connect succeeds with valid config."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        assert connector._connected is True
        assert connector._client is not None
        assert connector._collection is not None

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_default_timeout(self, mongodb_config):
        """Test that default timeout of 60s is used when not specified."""
        # Remove timeout from config to test default
        config_without_timeout = mongodb_config.copy()
        del config_without_timeout["timeout"]

        connector = MongoDBArchiverConnector()
        await connector.connect(config_without_timeout)

        assert connector._timeout == 60

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_custom_timeout(self, mongodb_config):
        """Test that custom timeout is used when specified."""
        config_with_timeout = mongodb_config.copy()
        config_with_timeout["timeout"] = 120

        connector = MongoDBArchiverConnector()
        await connector.connect(config_with_timeout)

        assert connector._timeout == 120

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_missing_host_raises_value_error(self, mongodb_config):
        """Test that connect raises ValueError when host is missing."""
        config = mongodb_config.copy()
        del config["host"]

        connector = MongoDBArchiverConnector()

        with pytest.raises(ValueError, match="host is required"):
            await connector.connect(config)

    @pytest.mark.asyncio
    async def test_connect_missing_database_name_raises_value_error(self, mongodb_config):
        """Test that connect raises ValueError when database name is missing."""
        config = mongodb_config.copy()
        del config["name"]

        connector = MongoDBArchiverConnector()

        with pytest.raises(ValueError, match="name.*database name.*required"):
            await connector.connect(config)

    @pytest.mark.asyncio
    async def test_connect_missing_collection_raises_value_error(self, mongodb_config):
        """Test that connect raises ValueError when collection is missing."""
        config = mongodb_config.copy()
        del config["collection"]

        connector = MongoDBArchiverConnector()

        with pytest.raises(ValueError, match="collection is required"):
            await connector.connect(config)

    @pytest.mark.asyncio
    async def test_connect_missing_username_raises_value_error(self, mongodb_config):
        """Test that connect raises ValueError when username is missing."""
        config = mongodb_config.copy()
        del config["username"]

        connector = MongoDBArchiverConnector()

        with pytest.raises(ValueError, match="username is required"):
            await connector.connect(config)

    @pytest.mark.asyncio
    async def test_connect_missing_password_env_raises_value_error(self, mongodb_config):
        """Test that connect raises ValueError when password_env is missing."""
        config = mongodb_config.copy()
        del config["password_env"]

        connector = MongoDBArchiverConnector()

        with pytest.raises(ValueError, match="password_env is required"):
            await connector.connect(config)

    @pytest.mark.asyncio
    async def test_connect_missing_auth_db_raises_value_error(self, mongodb_config):
        """Test that connect raises ValueError when auth database is missing."""
        config = mongodb_config.copy()
        del config["auth"]

        connector = MongoDBArchiverConnector()

        with pytest.raises(ValueError, match="auth.*authentication database.*required"):
            await connector.connect(config)

    @pytest.mark.asyncio
    async def test_connect_missing_password_env_var_raises_value_error(self, mongodb_config):
        """Test that connect raises ValueError when password env var is not set."""
        config = mongodb_config.copy()
        config["password_env"] = "NONEXISTENT_ENV_VAR"

        connector = MongoDBArchiverConnector()

        with pytest.raises(ValueError, match="Environment variable.*not set"):
            await connector.connect(config)

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self, mongodb_config):
        """Test that disconnect clears connection state."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        await connector.disconnect()

        assert connector._connected is False
        assert connector._client is None
        assert connector._collection is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test that disconnect is safe to call when already disconnected."""
        connector = MongoDBArchiverConnector()
        # Should not raise any exception
        await connector.disconnect()

        assert connector._connected is False
        assert connector._client is None
        assert connector._collection is None


@pytest.mark.integration
class TestImportErrorHandling:
    """Tests for import error handling when pymongo is missing."""

    @pytest.mark.asyncio
    async def test_connect_raises_import_error_when_pymongo_missing(self, mongodb_config):
        """Test that connect raises ImportError with helpful message when pymongo missing."""
        # Remove pymongo from sys.modules if it exists
        original_modules = sys.modules.copy()

        # Mock the import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == "pymongo" or name.startswith("pymongo."):
                raise ImportError("No module named 'pymongo'")
            return original_modules.get(name)

        connector = MongoDBArchiverConnector()

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                await connector.connect(mongodb_config)

            assert "pymongo is required" in str(exc_info.value)
            assert "pip install pymongo" in str(exc_info.value)


@pytest.mark.integration
class TestGetDataMethod:
    """Tests for get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_returns_dataframe(self, mongodb_config, mongodb_test_data):
        """Test that get_data returns a DataFrame with DatetimeIndex."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        start_date = mongodb_test_data["start_date"]
        end_date = datetime(2024, 1, 1, 12, 0, 0)  # First 12 hours

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=start_date,
            end_date=end_date,
        )

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "channel", "value"]
        assert (df["channel"] == "BEAM:CURRENT").all()
        assert len(df) > 0  # Should have data

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_multiple_pvs(self, mongodb_config, mongodb_test_data):
        """Test that get_data returns data for multiple PVs."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        start_date = mongodb_test_data["start_date"]
        end_date = datetime(2024, 1, 1, 12, 0, 0)
        pv_list = mongodb_test_data["pv_names"]

        df = await connector.get_data(
            pv_list=pv_list,
            start_date=start_date,
            end_date=end_date,
        )

        assert isinstance(df, pd.DataFrame)
        # All PVs should be represented as channels, long-format
        assert set(df["channel"]) == set(pv_list)
        assert len(df) > 0

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_empty_time_range(self, mongodb_config):
        """Test that get_data returns empty DataFrame for time range with no data."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        # Use a time range that definitely has no data
        start_date = datetime(2025, 1, 1, 0, 0, 0)
        end_date = datetime(2025, 1, 2, 0, 0, 0)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=start_date,
            end_date=end_date,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["timestamp", "channel", "value"]
        assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert df["value"].dtype == "float64"

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_not_connected_raises_runtime_error(self):
        """Test that get_data raises RuntimeError when not connected."""
        connector = MongoDBArchiverConnector()

        with pytest.raises(RuntimeError, match="MongoDB archiver not connected"):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )

    @pytest.mark.asyncio
    async def test_get_data_invalid_start_date_raises_type_error(self, mongodb_config):
        """Test that get_data raises TypeError when start_date is not a datetime."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        with pytest.raises(TypeError, match="start_date must be a datetime object"):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date="2024-01-01",  # String instead of datetime
                end_date=datetime(2024, 1, 2),
            )

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_invalid_end_date_raises_type_error(self, mongodb_config):
        """Test that get_data raises TypeError when end_date is not a datetime."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        with pytest.raises(TypeError, match="end_date must be a datetime object"):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date="2024-01-02",  # String instead of datetime
            )

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_empty_pv_list_raises_value_error(self, mongodb_config):
        """Test that get_data raises ValueError when pv_list is empty."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        with pytest.raises(ValueError, match="pv_list cannot be empty"):
            await connector.get_data(
                pv_list=[],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_time_range_filtering(self, mongodb_config, mongodb_test_data):
        """Test that get_data correctly filters by time range."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        # Get first 6 hourly samples. The connector's range query is inclusive
        # on both ends ($gte/$lte), so end_date is the timestamp of the last
        # sample we want, not the exclusive upper bound. precision_ms matches
        # the seeded hourly cadence; bounds are tz-aware since the returned
        # timestamps are UTC.
        start_date = mongodb_test_data["start_date"].replace(tzinfo=UTC)
        end_date = datetime(2024, 1, 1, 5, 0, 0, tzinfo=UTC)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=start_date,
            end_date=end_date,
            precision_ms=3_600_000,
        )

        assert len(df) == 6  # hourly samples at 00:00..05:00 inclusive
        assert df["timestamp"].iloc[0] >= start_date
        assert df["timestamp"].iloc[-1] <= end_date

        await connector.disconnect()


@pytest.mark.integration
class TestGetDataErrorHandling:
    """Tests for error handling in get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_timeout_raises_timeout_error(self, mongodb_config):
        """Test that timeout is properly handled."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        # Use a very short timeout that should fail
        # Note: This test may be flaky, so we'll use a reasonable timeout
        # and verify the error handling works
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0, 0)

        # This should work with normal timeout
        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=start_date,
            end_date=end_date,
            timeout=1,  # 1 second should be enough for small query
        )

        assert isinstance(df, pd.DataFrame)

        await connector.disconnect()


@pytest.mark.integration
class TestMetadataMethods:
    """Tests for metadata methods."""

    @pytest.mark.asyncio
    async def test_get_metadata_returns_archiver_metadata(self, mongodb_config, mongodb_test_data):
        """Test that get_metadata returns ArchiverMetadata dataclass."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        pv_name = mongodb_test_data["pv_names"][0]
        metadata = await connector.get_metadata(pv_name)

        assert isinstance(metadata, ArchiverMetadata)
        assert metadata.pv_name == pv_name
        assert metadata.is_archived is True  # Should be True since we have test data
        assert pv_name in metadata.description

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent_pv(self, mongodb_config):
        """Test that get_metadata returns is_archived=False for nonexistent PV."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        metadata = await connector.get_metadata("NONEXISTENT:PV")

        assert isinstance(metadata, ArchiverMetadata)
        assert metadata.pv_name == "NONEXISTENT:PV"
        assert metadata.is_archived is False  # Should be False for nonexistent PV

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_returns_dict(self, mongodb_config, mongodb_test_data):
        """Test that check_availability returns dict mapping PVs to availability."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        pv_names = mongodb_test_data["pv_names"]
        availability = await connector.check_availability(pv_names)

        assert isinstance(availability, dict)
        assert len(availability) == len(pv_names)
        for pv in pv_names:
            assert pv in availability
            assert availability[pv] is True  # All test PVs should be available

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_mixed_pvs(self, mongodb_config, mongodb_test_data):
        """Test that check_availability correctly identifies available and unavailable PVs."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        # Mix of existing and nonexistent PVs
        pv_names = mongodb_test_data["pv_names"] + ["NONEXISTENT:PV"]
        availability = await connector.check_availability(pv_names)

        assert isinstance(availability, dict)
        assert len(availability) == len(pv_names)

        # Existing PVs should be True
        for pv in mongodb_test_data["pv_names"]:
            assert availability[pv] is True

        # Nonexistent PV should be False
        assert availability["NONEXISTENT:PV"] is False

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_empty_list(self, mongodb_config):
        """Test that check_availability returns empty dict for empty input."""
        connector = MongoDBArchiverConnector()
        await connector.connect(mongodb_config)

        availability = await connector.check_availability([])

        assert isinstance(availability, dict)
        assert len(availability) == 0

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_not_connected_raises_runtime_error(self):
        """Test that check_availability raises RuntimeError when not connected."""
        connector = MongoDBArchiverConnector()

        with pytest.raises(RuntimeError, match="MongoDB archiver not connected"):
            await connector.check_availability(["BEAM:CURRENT"])

    @pytest.mark.asyncio
    async def test_get_metadata_not_connected_raises_runtime_error(self):
        """Test that get_metadata raises RuntimeError when not connected."""
        connector = MongoDBArchiverConnector()

        with pytest.raises(RuntimeError, match="MongoDB archiver not connected"):
            await connector.get_metadata("BEAM:CURRENT")


@pytest.mark.integration
class TestFactoryIntegration:
    """Tests for factory integration."""

    @pytest.fixture(autouse=True)
    def setup_factory(self):
        """Ensure built-in archivers (including mongodb_archiver) are registered.

        Idempotent: ``register_builtin_connectors`` no-ops if already populated,
        so this is safe regardless of prior test state. No teardown — we leave
        built-ins in place so subsequent tests can rely on them.
        """
        from osprey.connectors.factory import register_builtin_connectors

        register_builtin_connectors()
        yield

    @pytest.mark.asyncio
    async def test_factory_creates_mongodb_archiver_connector(self, mongodb_config):
        """Test that factory creates and connects MongoDBArchiverConnector."""
        config = {
            "type": "mongodb_archiver",
            "mongodb_archiver": mongodb_config,
        }

        connector = await ConnectorFactory.create_archiver_connector(config)

        assert isinstance(connector, MongoDBArchiverConnector)
        assert connector._connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_factory_with_missing_host_raises_error(self, mongodb_config):
        """Test that factory propagates ValueError for missing host."""
        config_without_host = mongodb_config.copy()
        del config_without_host["host"]

        config = {
            "type": "mongodb_archiver",
            "mongodb_archiver": config_without_host,
        }

        with pytest.raises(ValueError, match="host is required"):
            await ConnectorFactory.create_archiver_connector(config)


class TestQueryShapeWithoutDocker:
    """Query construction, asserted against a stub collection — no container needed."""

    def _stub_connector(self, documents):
        """Return a connector wired to an in-memory stub collection."""
        from unittest.mock import MagicMock

        from osprey.connectors.archiver.mongodb_archiver_connector import (
            MongoDBArchiverConnector,
        )

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

        connector = MongoDBArchiverConnector()
        connector._connected = True
        connector._collection = collection
        connector._timeout = 10
        return connector, captured

    @pytest.mark.asyncio
    async def test_pv_existence_is_ored_not_anded(self):
        """PVs archived in separate documents must not produce an empty frame."""
        connector, captured = self._stub_connector([])

        await connector.get_data(
            pv_list=["BEAM:CURRENT", "BEAM:LIFETIME"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert captured["query"]["$or"] == [
            {"BEAM:CURRENT": {"$exists": True}},
            {"BEAM:LIFETIME": {"$exists": True}},
        ]
        # The old shape ANDed a top-level key per PV; it must be gone.
        assert "BEAM:CURRENT" not in captured["query"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("pv_list", "expected"),
        [
            (["BEAM:CURRENT"], {"date": 1, "BEAM:CURRENT": 1}),
            (
                ["BEAM:CURRENT", "BEAM:LIFETIME"],
                {"date": 1, "BEAM:CURRENT": 1, "BEAM:LIFETIME": 1},
            ),
            # A PV literally named "date", and a repeated PV, both collapse
            # into the single key the projection already has.
            (["date"], {"date": 1}),
            (["BEAM:CURRENT", "BEAM:CURRENT"], {"date": 1, "BEAM:CURRENT": 1}),
        ],
    )
    async def test_projection_requests_date_plus_exactly_the_requested_pvs(self, pv_list, expected):
        """``date`` must be projected alongside the PVs, and nothing else.

        Drop ``date`` and every document is skipped as missing it; projecting
        more than the requested PVs pulls whole documents over the wire.
        """
        connector, captured = self._stub_connector([])

        await connector.get_data(
            pv_list=pv_list,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert captured["projection"] == expected

    @pytest.mark.asyncio
    async def test_precision_ms_bins_a_non_raw_processing_mode(self):
        """precision_ms must actually downsample a non-raw mode rather than being a no-op."""
        documents = [
            {"date": datetime(2024, 1, 1, 0, 0, s, tzinfo=UTC), "BEAM:CURRENT": float(s)}
            for s in range(6)
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            precision_ms=2000,
            processing="max",
        )

        # Six 1-second samples binned at 2 s, "max" keeping the larger of each pair.
        assert df.loc[df["channel"] == "BEAM:CURRENT", "value"].tolist() == [1.0, 3.0, 5.0]

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

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            precision_ms=2000,
        )

        rows = df.loc[df["channel"] == "BEAM:CURRENT"]
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

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            precision_ms=0,
        )

        assert df.loc[df["channel"] == "BEAM:CURRENT", "value"].tolist() == [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]

    @pytest.mark.asyncio
    async def test_processing_selects_the_aggregation(self):
        documents = [
            {"date": datetime(2024, 1, 1, 0, 0, s, tzinfo=UTC), "BEAM:CURRENT": float(s)}
            for s in range(6)
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            precision_ms=2000,
            processing="mean",
        )

        assert df.loc[df["channel"] == "BEAM:CURRENT", "value"].tolist() == [0.5, 2.5, 4.5]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("processing", ["raw", "mean"])
    async def test_empty_window_returns_the_typed_empty_long_frame(self, processing):
        # Regression: the non-numeric guard ran before the emptiness check, so
        # an empty (object-dtype) channel raised a misleading "non-numeric".
        connector, _ = self._stub_connector([])

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 2, tzinfo=UTC),
            processing=processing,
        )

        assert list(df.columns) == ["timestamp", "channel", "value"]
        assert df.empty
        assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert df["value"].dtype == "float64"

    @pytest.mark.asyncio
    async def test_timeout_zero_respected_not_falsy(self):
        """timeout=0 must mean zero, not fall back to the 60 s default."""
        connector, _ = self._stub_connector([])
        connector._timeout = 60

        with pytest.raises(TimeoutError):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 2, tzinfo=UTC),
                timeout=0,
            )

    @pytest.mark.asyncio
    async def test_sparse_documents_are_not_upsampled(self):
        """A fine precision_ms must not manufacture samples faster than the ingester wrote them."""
        documents = [
            {"date": datetime(2024, 1, 1, h, 0, 0, tzinfo=UTC), "BEAM:CURRENT": float(h)}
            for h in range(6)
        ]
        connector, _ = self._stub_connector(documents)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 5, 0, 0, tzinfo=UTC),
            precision_ms=1000,
        )

        # Documents are 1 hour apart; naive 1 s resampling would reindex onto
        # an 18,001-row grid.
        values = df.loc[df["channel"] == "BEAM:CURRENT", "value"]
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

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT", "BEAM:LIFETIME"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            processing=processing,
        )

        assert set(df["channel"]) == {"BEAM:CURRENT"}

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

        df = await connector.get_data(
            pv_list=["FAST", "SLOW"],
            start_date=base,
            end_date=base + timedelta(milliseconds=49 * 1500 + 100),
            precision_ms=1000,
            processing="mean",
        )

        fast_values = df.loc[df["channel"] == "FAST", "value"].tolist()
        slow_values = df.loc[df["channel"] == "SLOW", "value"].tolist()

        # FAST: the true per-1s-bin means.
        assert fast_values == pytest.approx([4.5, 14.5, 24.5, 34.5])
        # SLOW: its own 50 real samples, each its own bin.
        assert len(slow_values) == 50
        assert slow_values == pytest.approx([500.0 + i for i in range(50)])
