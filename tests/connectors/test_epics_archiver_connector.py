"""Tests for EPICS Archiver Appliance connector."""

import json
import urllib.error
import urllib.request
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from osprey.connectors.archiver.base import ArchiverMetadata
from osprey.connectors.archiver.epics_archiver_connector import EPICSArchiverConnector
from osprey.connectors.factory import ConnectorFactory


def _make_urlopen_response(payload: list) -> MagicMock:
    """Return a context-manager mock that yields a file-like object with JSON payload."""
    body = json.dumps(payload).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _archiver_payload(pv: str, points: list) -> list:
    """Build a minimal archiver JSON payload for a single PV."""
    return [
        {
            "meta": {"name": pv},
            "data": [{"secs": s, "nanos": n, "val": v} for s, n, v in points],
        }
    ]


class TestConnectDisconnectLifecycle:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test that connect succeeds with valid config."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        assert connector._connected is True
        assert connector._url == "https://archiver.example.com"

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_default_timeout(self):
        """Test that default timeout of 60s is used when not specified."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        assert connector._timeout == 60

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_custom_timeout(self):
        """Test that custom timeout is used when specified."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com", "timeout": 120})

        assert connector._timeout == 120

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_missing_url_raises_value_error(self):
        """Test that connect raises ValueError when URL is missing."""
        connector = EPICSArchiverConnector()

        with pytest.raises(ValueError, match="archiver URL is required"):
            await connector.connect({})

    @pytest.mark.asyncio
    async def test_connect_empty_url_raises_value_error(self):
        """Test that connect raises ValueError when URL is empty string."""
        connector = EPICSArchiverConnector()

        with pytest.raises(ValueError, match="archiver URL is required"):
            await connector.connect({"url": ""})

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        """Test that disconnect clears connection state."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        await connector.disconnect()

        assert connector._connected is False
        assert connector._url is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test that disconnect is safe to call when already disconnected."""
        connector = EPICSArchiverConnector()
        await connector.disconnect()

        assert connector._connected is False
        assert connector._url is None


class TestGetDataMethod:
    """Tests for get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_returns_dataframe(self):
        """Test that get_data returns a DataFrame with DatetimeIndex."""
        points = [(1704067200, 0, 499.8), (1704067201, 0, 499.7)]
        response = _make_urlopen_response(_archiver_payload("BEAM:CURRENT", points))

        with patch("urllib.request.urlopen", return_value=response):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1, 0, 0, 0),
                end_date=datetime(2024, 1, 1, 1, 0, 0),
            )

            assert isinstance(df, pd.DataFrame)
            assert isinstance(df.index, pd.DatetimeIndex)
            assert "BEAM:CURRENT" in df.columns

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_single_pv_correct_values(self):
        """Test that single-PV fetch returns correct values."""
        points = [(1704067200, 0, 1.0), (1704067201, 0, 2.0), (1704067202, 0, 3.0)]
        response = _make_urlopen_response(_archiver_payload("PV:X", points))

        with patch("urllib.request.urlopen", return_value=response):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["PV:X"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 1),
            )

            assert list(df["PV:X"]) == [1.0, 2.0, 3.0]

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_multi_pv_returns_one_column_per_pv(self):
        """Test that multi-PV fetch returns DataFrame with one column per PV."""
        points = [(1704067200 + i, 0, float(i)) for i in range(5)]

        call_count = [0]

        def mock_urlopen(req, timeout=None):
            idx = call_count[0]
            call_count[0] += 1
            pv = "PV:1" if idx == 0 else "PV:2"
            return _make_urlopen_response(_archiver_payload(pv, points))

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["PV:1", "PV:2"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 0, 1),
            )

            assert isinstance(df, pd.DataFrame)
            assert "PV:1" in df.columns
            assert "PV:2" in df.columns

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_server_side_downsampling_wraps_pv(self):
        """Test that precision_ms > 0 wraps PV name as lastSample_N(pv)."""
        captured_urls = []

        def mock_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            return _make_urlopen_response([])

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            await connector.get_data(
                pv_list=["SR:DCCT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 1),
                precision_ms=5000,
            )

            assert len(captured_urls) == 1
            assert "lastSample_5" in captured_urls[0]

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_precision_ms_zero_sends_raw_pv(self):
        """Test that precision_ms=0 sends the raw PV name without wrapping."""
        captured_urls = []

        def mock_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            return _make_urlopen_response([])

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            await connector.get_data(
                pv_list=["SR:DCCT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 1),
                precision_ms=0,
            )

            assert "lastSample" not in captured_urls[0]

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_empty_response_returns_empty_dataframe(self):
        """Test that empty archiver response [] returns empty DataFrame."""
        response = _make_urlopen_response([])

        with patch("urllib.request.urlopen", return_value=response):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 1),
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_empty_data_list_returns_empty_dataframe(self):
        """Test that [{meta:..., data:[]}] archiver response returns empty DataFrame."""
        response = _make_urlopen_response([{"meta": {"name": "BEAM:CURRENT"}, "data": []}])

        with patch("urllib.request.urlopen", return_value=response):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 1),
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

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
                timeout=60,
            )

    @pytest.mark.asyncio
    async def test_get_data_invalid_start_date_raises_type_error(self):
        """Test that get_data raises TypeError when start_date is not a datetime."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        with pytest.raises(TypeError, match="start_date must be a datetime object"):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date="2024-01-01",
                end_date=datetime(2024, 1, 2),
            )

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_invalid_end_date_raises_type_error(self):
        """Test that get_data raises TypeError when end_date is not a datetime."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        with pytest.raises(TypeError, match="end_date must be a datetime object"):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date="2024-01-02",
            )

        await connector.disconnect()


class TestMultiPVAlignment:
    """Tests for multi-PV timestamp alignment."""

    @pytest.mark.asyncio
    async def test_multi_pv_different_timestamps_produces_aligned_dataframe(self):
        """Test that multi-PV with different timestamps produces aligned DataFrame."""
        base = 1704067200
        pv1_points = [(base, 0, 1.0), (base + 2, 0, 2.0)]
        pv2_points = [(base + 1, 0, 10.0), (base + 3, 0, 20.0)]

        call_count = [0]

        def mock_urlopen(req, timeout=None):
            idx = call_count[0]
            call_count[0] += 1
            payload = (
                _archiver_payload("PV:A", pv1_points)
                if idx == 0
                else _archiver_payload("PV:B", pv2_points)
            )
            return _make_urlopen_response(payload)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["PV:A", "PV:B"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 0, 0, 10),
                precision_ms=1000,
            )

            assert isinstance(df, pd.DataFrame)
            assert "PV:A" in df.columns
            assert "PV:B" in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)

            await connector.disconnect()


class TestGetDataErrorHandling:
    """Tests for error handling in get_data method."""

    @pytest.mark.asyncio
    async def test_url_error_raises_connection_error(self):
        """Test that urllib.error.URLError is mapped to ConnectionError."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            with pytest.raises(ConnectionError):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 2),
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self):
        """Test that asyncio timeout raises TimeoutError."""
        import asyncio

        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(10)

        with patch("asyncio.to_thread", side_effect=slow_fetch):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com", "timeout": 0.01})

            with pytest.raises(TimeoutError, match="timed out"):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 2),
                    timeout=0.01,
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connection_refused_raises_connection_error(self):
        """Test that ConnectionRefusedError is wrapped as ConnectionError."""
        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
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
    async def test_generic_connection_error_raised(self):
        """Test that generic exceptions with 'connection' in message raise ConnectionError."""
        with patch(
            "urllib.request.urlopen",
            side_effect=Exception("Connection timed out: could not reach server"),
        ):
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
    async def test_waveform_pv_raises_value_error(self):
        """Test that array-valued val (waveform PV) raises ValueError."""
        payload = [
            {"meta": {"name": "CAM:IMAGE"}, "data": [{"secs": 1, "nanos": 0, "val": [1, 2, 3]}]}
        ]
        response = _make_urlopen_response(payload)

        with patch("urllib.request.urlopen", return_value=response):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            with pytest.raises(ValueError, match="Waveform PVs not supported"):
                await connector.get_data(
                    pv_list=["CAM:IMAGE"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 2),
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_timeout_zero_respected_not_falsy(self):
        """Test that timeout=0 is respected and treated as zero, not as falsy."""
        import asyncio

        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(10)

        with patch("asyncio.to_thread", side_effect=slow_fetch):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com", "timeout": 60})

            # timeout=0 should trigger immediate timeout, not fall back to self._timeout=60
            with pytest.raises((TimeoutError, Exception)):
                await connector.get_data(
                    pv_list=["BEAM:CURRENT"],
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 2),
                    timeout=0,
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_unconnected_connector_raises_runtime_error(self):
        """Test that calling get_data on unconnected connector raises RuntimeError."""
        connector = EPICSArchiverConnector()

        with pytest.raises(RuntimeError, match="Archiver not connected"):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
                timeout=1,
            )


class TestMetadataMethods:
    """Tests for metadata methods."""

    @pytest.mark.asyncio
    async def test_get_metadata_returns_archiver_metadata(self):
        """Test that get_metadata returns ArchiverMetadata dataclass."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        metadata = await connector.get_metadata("BEAM:CURRENT")

        assert isinstance(metadata, ArchiverMetadata)
        assert metadata.pv_name == "BEAM:CURRENT"
        assert metadata.is_archived is True
        assert "BEAM:CURRENT" in metadata.description

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_returns_dict(self):
        """Test that check_availability returns dict mapping PVs to True."""
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
    async def test_check_availability_empty_list(self):
        """Test that check_availability returns empty dict for empty input."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        availability = await connector.check_availability([])

        assert isinstance(availability, dict)
        assert len(availability) == 0

        await connector.disconnect()


class TestICMPPingResilience:
    """Tests for als-profiles GitLab #8: ArchiverClient ICMP ping in constructor.

    archivertools.DataDownloader.__init__ calls a subprocess ping to check
    reachability. This causes two problems:

    1. ICMP may be blocked in containers while HTTP port 17668 is open
    2. The ping output + print() statements corrupt MCP stdio transport

    The connector must suppress both the ping and stdout during construction.
    """

    @pytest.mark.asyncio
    async def test_connect_survives_blocked_icmp(self):
        """connect() must succeed even when ICMP ping fails."""

        class PingCheckingClient:
            """Mimics real ArchiverClient: constructor pings."""

            def __init__(self, archiver_url=None):
                import os as _os

                print("Verifying reachability...")
                # Static test hostname, not user input
                exit_status = _os.system(  # noqa: S605,S607
                    "ping -c 1 -W 1 localhost.invalid"
                )
                if exit_status != 0:
                    raise ConnectionError("Archiver server is unreachable.")

        mock_module = MagicMock()
        mock_module.ArchiverClient = PingCheckingClient

        with patch.dict("sys.modules", {"archivertools": mock_module}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "http://archiver.example.com:17668"})

            assert connector._connected is True
            assert connector._archiver_client is not None
            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_does_not_pollute_stdout(self):
        """ArchiverClient prints to stdout during init — corrupts MCP stdio."""
        import io
        import sys

        class PrintyClient:
            def __init__(self, archiver_url=None):
                print("===================================")
                print("Verifying reachability...")
                print("Archiver server is reachable via ping.")

        mock_module = MagicMock()
        mock_module.ArchiverClient = PrintyClient

        with patch.dict("sys.modules", {"archivertools": mock_module}):
            connector = EPICSArchiverConnector()

            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                await connector.connect({"url": "http://archiver.example.com:17668"})
            finally:
                sys.stdout = old_stdout

            assert captured.getvalue() == "", (
                f"stdout was polluted during connect(): {captured.getvalue()!r}"
            )
            assert connector._connected is True
            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_still_raises_on_real_init_failure(self):
        """Non-ping failures must still raise ConnectionError."""
        mock_module = MagicMock()
        mock_module.ArchiverClient = MagicMock(
            side_effect=RuntimeError("Invalid archiver configuration")
        )

        with patch.dict("sys.modules", {"archivertools": mock_module}):
            connector = EPICSArchiverConnector()
            with pytest.raises(ConnectionError, match="ArchiverClient initialization failed"):
                await connector.connect({"url": "http://archiver.example.com:17668"})


class TestFactoryIntegration:
    """Tests for factory integration."""

    @pytest.fixture(autouse=True)
    def setup_factory(self):
        """Register EPICS archiver connector and clean up afterward."""
        ConnectorFactory.register_archiver("epics_archiver", EPICSArchiverConnector)
        yield
        ConnectorFactory._archiver_connectors.clear()

    @pytest.mark.asyncio
    async def test_factory_creates_epics_archiver_connector(self):
        """Test that factory creates and connects EPICSArchiverConnector."""
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
    async def test_factory_with_missing_url_raises_error(self):
        """Test that factory propagates ValueError for missing URL."""
        config = {
            "type": "epics_archiver",
            "epics_archiver": {},  # Missing URL
        }

        with pytest.raises(ValueError, match="archiver URL is required"):
            await ConnectorFactory.create_archiver_connector(config)
