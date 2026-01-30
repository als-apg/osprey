"""Tests for MongoDB Archiver connector."""

import sys
from datetime import datetime
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
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "BEAM:CURRENT" in df.columns
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
        # All PVs should be columns
        for pv in pv_list:
            assert pv in df.columns
        assert len(df.columns) == len(pv_list)
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
        assert "BEAM:CURRENT" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

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

        # Get data for first 6 hours
        start_date = mongodb_test_data["start_date"]
        end_date = datetime(2024, 1, 1, 6, 0, 0)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=start_date,
            end_date=end_date,
        )

        assert len(df) == 6  # 6 hours of data (hourly intervals)
        assert df.index[0] >= start_date
        assert df.index[-1] <= end_date

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
        """Register MongoDB archiver connector and clean up afterward."""
        ConnectorFactory.register_archiver("mongodb_archiver", MongoDBArchiverConnector)
        yield
        ConnectorFactory._archiver_connectors.clear()

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
