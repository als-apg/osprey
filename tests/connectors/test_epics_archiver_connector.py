"""Tests for EPICS Archiver Appliance connector."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from osprey.connectors.archiver.base import ArchiverMetadata
from osprey.connectors.archiver.epics_archiver_connector import EPICSArchiverConnector
from osprey.connectors.factory import ConnectorFactory


@pytest.fixture
def mock_archiver_client():
    """
    Create a mock ArchiverClient that returns synthetic DataFrames.

    The real archivertools library (als-archiver-client) returns DataFrames with:
    - Columns: [secs, nanos, PV1, PV2, ...]
    - Index: Default RangeIndex (0, 1, 2, ...)

    See: https://github.com/andrea-pollastro/als-archiver-client

    Note: The current connector implementation assumes the DataFrame has a
    time-based index and calls pd.to_datetime(data.index). This mock returns
    a DataFrame with a DatetimeIndex to match what the connector expects,
    though this may not match the real library's exact format.
    """
    mock_client = MagicMock()

    def mock_match_data(pv_list, precision, start, end, verbose):
        # Generate timestamps as the real library does (secs/nanos columns)
        # but return with DatetimeIndex since the connector expects that
        timestamps = pd.date_range(start=start, end=end, periods=100)
        data = {pv: [float(i) for i in range(100)] for pv in pv_list}
        return pd.DataFrame(data, index=timestamps)

    mock_client.match_data = MagicMock(side_effect=mock_match_data)
    return mock_client


@pytest.fixture
def mock_archiver_client_realistic():
    """
    Create a mock ArchiverClient that returns DataFrames in the REAL format.

    The real archivertools library returns:
    - Columns: [secs, nanos, PV1, PV2, ...]
    - Index: Default RangeIndex

    Use this fixture to test against the actual library format.
    """
    mock_client = MagicMock()

    def mock_match_data(pv_list, precision, start, end, verbose):
        num_points = 100
        start_ts = start.timestamp()
        end_ts = end.timestamp()
        step = (end_ts - start_ts) / num_points

        secs = [int(start_ts + i * step) for i in range(num_points)]
        nanos = [0] * num_points
        data = {"secs": secs, "nanos": nanos}
        for pv in pv_list:
            data[pv] = [float(i) for i in range(num_points)]
        return pd.DataFrame(data)  # Default RangeIndex

    mock_client.match_data = MagicMock(side_effect=mock_match_data)
    return mock_client


@pytest.fixture
def mock_archivertools_module(mock_archiver_client):
    """Create a mock archivertools module."""
    mock_module = MagicMock()
    mock_module.ArchiverClient = MagicMock(return_value=mock_archiver_client)
    return mock_module


class TestConnectDisconnectLifecycle:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_archivertools_module):
        """Test that connect succeeds with valid config."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            assert connector._connected is True
            assert connector._archiver_client is not None

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_default_timeout(self, mock_archivertools_module):
        """Test that default timeout of 60s is used when not specified."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            assert connector._timeout == 60

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_custom_timeout(self, mock_archivertools_module):
        """Test that custom timeout is used when specified."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com", "timeout": 120})

            assert connector._timeout == 120

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_missing_url_raises_value_error(self, mock_archivertools_module):
        """Test that connect raises ValueError when URL is missing."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()

            with pytest.raises(ValueError, match="archiver URL is required"):
                await connector.connect({})

    @pytest.mark.asyncio
    async def test_connect_empty_url_raises_value_error(self, mock_archivertools_module):
        """Test that connect raises ValueError when URL is empty string."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()

            with pytest.raises(ValueError, match="archiver URL is required"):
                await connector.connect({"url": ""})

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self, mock_archivertools_module):
        """Test that disconnect clears connection state."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            await connector.disconnect()

            assert connector._connected is False
            assert connector._archiver_client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test that disconnect is safe to call when already disconnected."""
        connector = EPICSArchiverConnector()
        # Should not raise any exception
        await connector.disconnect()

        assert connector._connected is False
        assert connector._archiver_client is None


class TestImportErrorHandling:
    """Tests for import error handling when archivertools is missing."""

    @pytest.mark.asyncio
    async def test_connect_raises_import_error_when_archivertools_missing(self):
        """Test that connect raises ImportError with helpful message when archivertools missing."""
        # Remove archivertools from sys.modules if it exists
        original_modules = sys.modules.copy()

        # Mock the import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == "archivertools":
                raise ImportError("No module named 'archivertools'")
            return original_modules.get(name)

        connector = EPICSArchiverConnector()

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                await connector.connect({"url": "https://archiver.example.com"})

            assert "archivertools is required" in str(exc_info.value)
            assert "pip install archivertools" in str(exc_info.value)


class TestGetDataMethod:
    """Tests for get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_returns_dataframe(
        self, mock_archivertools_module, mock_archiver_client
    ):
        """Test that get_data returns a DataFrame with DatetimeIndex."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            start_date = datetime(2024, 1, 1, 0, 0, 0)
            end_date = datetime(2024, 1, 1, 1, 0, 0)

            df = await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=start_date,
                end_date=end_date,
            )

            assert isinstance(df, pd.DataFrame)
            assert isinstance(df.index, pd.DatetimeIndex)
            assert "BEAM:CURRENT" in df.columns

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_calls_match_data_correctly(
        self, mock_archivertools_module, mock_archiver_client
    ):
        """Test that get_data passes correct parameters to ArchiverClient.match_data()."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            start_date = datetime(2024, 1, 1, 0, 0, 0)
            end_date = datetime(2024, 1, 1, 1, 0, 0)
            pv_list = ["PV:1", "PV:2"]

            await connector.get_data(
                pv_list=pv_list,
                start_date=start_date,
                end_date=end_date,
                precision_ms=500,
            )

            mock_archiver_client.match_data.assert_called_once_with(
                pv_list=pv_list,
                precision=500,
                start=start_date,
                end=end_date,
                verbose=0,
            )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_not_connected_raises_runtime_error(self):
        """Test that get_data raises RuntimeError when not connected."""
        connector = EPICSArchiverConnector()

        with pytest.raises(RuntimeError, match="Archiver not connected"):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
                timeout=60,  # Provide timeout since _timeout attribute doesn't exist before connect
            )

    @pytest.mark.asyncio
    async def test_get_data_invalid_start_date_raises_type_error(
        self, mock_archivertools_module, mock_archiver_client
    ):
        """Test that get_data raises TypeError when start_date is not a datetime."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            with pytest.raises(TypeError, match="start_date must be a datetime object"):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date="2024-01-01",  # String instead of datetime
                    end_date=datetime(2024, 1, 2),
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_invalid_end_date_raises_type_error(
        self, mock_archivertools_module, mock_archiver_client
    ):
        """Test that get_data raises TypeError when end_date is not a datetime."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            with pytest.raises(TypeError, match="end_date must be a datetime object"):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date=datetime(2024, 1, 1),
                    end_date="2024-01-02",  # String instead of datetime
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_uses_default_precision(
        self, mock_archivertools_module, mock_archiver_client
    ):
        """Test that get_data uses default precision_ms=1000."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            start_date = datetime(2024, 1, 1, 0, 0, 0)
            end_date = datetime(2024, 1, 1, 1, 0, 0)

            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=start_date,
                end_date=end_date,
            )

            # Check that precision=1000 was passed (default)
            call_kwargs = mock_archiver_client.match_data.call_args[1]
            assert call_kwargs["precision"] == 1000

            await connector.disconnect()


class TestRealLibraryFormat:
    """
    Tests using the realistic library format.

    The real archivertools library (als-archiver-client) returns DataFrames with
    columns [secs, nanos, PV1, PV2, ...] and a RangeIndex, NOT a DatetimeIndex.

    The connector properly converts secs/nanos to a DatetimeIndex.
    """

    @pytest.fixture
    def mock_archivertools_realistic(self, mock_archiver_client_realistic):
        """Create mock module with realistic client."""
        mock_module = MagicMock()
        mock_module.ArchiverClient = MagicMock(return_value=mock_archiver_client_realistic)
        return mock_module

    @pytest.mark.asyncio
    async def test_get_data_with_real_library_format(self, mock_archivertools_realistic):
        """
        Test connector properly handles real library DataFrame format.

        The real library returns DataFrame with columns [secs, nanos, PV1, ...].
        The connector should convert secs/nanos to a proper DatetimeIndex and
        remove those columns from the result.
        """
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_realistic}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            start_date = datetime(2024, 1, 1, 0, 0, 0)
            end_date = datetime(2024, 1, 1, 1, 0, 0)

            df = await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=start_date,
                end_date=end_date,
            )

            # The connector returns a DataFrame
            assert isinstance(df, pd.DataFrame)

            # secs/nanos columns should be removed
            assert "secs" not in df.columns
            assert "nanos" not in df.columns

            # Only PV columns remain
            assert "BEAM:CURRENT" in df.columns
            assert len(df.columns) == 1

            # Index is proper DatetimeIndex with actual timestamps (not epoch 1970)
            assert isinstance(df.index, pd.DatetimeIndex)

            # Verify timestamps are in 2024, not 1970 (epoch)
            # The mock uses start.timestamp() which gives UTC seconds
            assert df.index[0].year == 2024
            assert df.index[-1].year == 2024

            # Timestamps should span roughly the requested time range
            # (accounting for timezone differences in the mock)
            time_span = df.index[-1] - df.index[0]
            expected_span = pd.Timedelta(hours=1)
            assert abs(time_span - expected_span) < pd.Timedelta(minutes=1)

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_multiple_pvs_real_format(self, mock_archivertools_realistic):
        """Test connector handles multiple PVs with real library format."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_realistic}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            start_date = datetime(2024, 1, 1, 0, 0, 0)
            end_date = datetime(2024, 1, 1, 1, 0, 0)
            pv_list = ["PV:1", "PV:2", "PV:3"]

            df = await connector.get_data(
                pv_list=pv_list,
                start_date=start_date,
                end_date=end_date,
            )

            # All PVs should be columns
            for pv in pv_list:
                assert pv in df.columns

            # secs/nanos should not be present
            assert "secs" not in df.columns
            assert "nanos" not in df.columns

            # Should have exactly 3 columns (the PVs)
            assert len(df.columns) == 3

            await connector.disconnect()


class TestGetDataErrorHandling:
    """Tests for error handling in get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_timeout_raises_timeout_error(self, mock_archivertools_module):
        """Test that timeout is properly wrapped as TimeoutError."""
        mock_client = MagicMock()

        def slow_match_data(*args, **kwargs):
            import time

            time.sleep(5)  # Will be interrupted by timeout

        mock_client.match_data = slow_match_data
        mock_archivertools_module.ArchiverClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com", "timeout": 0.1})

            with pytest.raises(TimeoutError, match="timed out"):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 2),
                    timeout=0.1,
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_connection_refused_raises_connection_error(
        self, mock_archivertools_module
    ):
        """Test that ConnectionRefusedError is wrapped as ConnectionError."""
        mock_client = MagicMock()
        mock_client.match_data = MagicMock(side_effect=ConnectionRefusedError("Connection refused"))
        mock_archivertools_module.ArchiverClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            with pytest.raises(ConnectionError, match="Cannot connect to the archiver"):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 2),
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_network_error_raises_connection_error(self, mock_archivertools_module):
        """Test that generic connection errors are wrapped as ConnectionError."""
        mock_client = MagicMock()
        mock_client.match_data = MagicMock(
            side_effect=Exception("Connection timed out: could not reach server")
        )
        mock_archivertools_module.ArchiverClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            with pytest.raises(ConnectionError, match="Network connectivity issue"):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 2),
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_unexpected_format_raises_value_error(self, mock_archivertools_module):
        """Test that non-DataFrame return value raises ValueError."""
        mock_client = MagicMock()
        mock_client.match_data = MagicMock(return_value={"unexpected": "format"})
        mock_archivertools_module.ArchiverClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            with pytest.raises(ValueError, match="Unexpected data format"):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 2),
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_client_init_failure_raises_connection_error(
        self, mock_archivertools_module
    ):
        """Test that ArchiverClient initialization failure raises ConnectionError."""
        mock_archivertools_module.ArchiverClient = MagicMock(
            side_effect=Exception("Failed to initialize client")
        )

        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()

            with pytest.raises(ConnectionError, match="ArchiverClient initialization failed"):
                await connector.connect({"url": "https://archiver.example.com"})


class TestMetadataMethods:
    """Tests for metadata methods."""

    @pytest.mark.asyncio
    async def test_get_metadata_returns_archiver_metadata(
        self, mock_archivertools_module, mock_archiver_client
    ):
        """Test that get_metadata returns ArchiverMetadata dataclass."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            metadata = await connector.get_metadata("BEAM:CURRENT")

            assert isinstance(metadata, ArchiverMetadata)
            assert metadata.pv_name == "BEAM:CURRENT"
            assert metadata.is_archived is True
            assert "BEAM:CURRENT" in metadata.description

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_returns_dict(
        self, mock_archivertools_module, mock_archiver_client
    ):
        """Test that check_availability returns dict mapping PVs to True."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            pv_names = ["PV:1", "PV:2", "PV:3"]
            availability = await connector.check_availability(pv_names)

            assert isinstance(availability, dict)
            assert len(availability) == len(pv_names)
            for pv in pv_names:
                assert pv in availability
                assert availability[pv] is True

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_empty_list(
        self, mock_archivertools_module, mock_archiver_client
    ):
        """Test that check_availability returns empty dict for empty input."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            availability = await connector.check_availability([])

            assert isinstance(availability, dict)
            assert len(availability) == 0

            await connector.disconnect()


class TestFactoryIntegration:
    """Tests for factory integration."""

    @pytest.fixture(autouse=True)
    def setup_factory(self):
        """Register EPICS archiver connector and clean up afterward."""
        ConnectorFactory.register_archiver("epics_archiver", EPICSArchiverConnector)
        yield
        ConnectorFactory._archiver_connectors.clear()

    @pytest.mark.asyncio
    async def test_factory_creates_epics_archiver_connector(self, mock_archivertools_module):
        """Test that factory creates and connects EPICSArchiverConnector."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            config = {
                "type": "epics_archiver",
                "epics_archiver": {"url": "https://archiver.example.com", "timeout": 30},
            }

            connector = await ConnectorFactory.create_archiver_connector(config)

            assert isinstance(connector, EPICSArchiverConnector)
            assert connector._connected is True
            assert connector._timeout == 30

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_factory_with_missing_url_raises_error(self, mock_archivertools_module):
        """Test that factory propagates ValueError for missing URL."""
        with patch.dict("sys.modules", {"archivertools": mock_archivertools_module}):
            config = {
                "type": "epics_archiver",
                "epics_archiver": {},  # Missing URL
            }

            with pytest.raises(ValueError, match="archiver URL is required"):
                await ConnectorFactory.create_archiver_connector(config)
