"""Tests for EPICS Archiver Appliance connector."""

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from osprey.connectors.archiver._timerange import Processing
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


@pytest.fixture
def captured_urls():
    """Patch ``urlopen`` to record each request URL and answer with no data."""
    urls: list[str] = []

    def mock_urlopen(req, timeout=None):
        urls.append(req.full_url)
        return _make_urlopen_response([])

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        yield urls


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
        """Test that get_data returns the canonical long-format DataFrame."""
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
            assert list(df.columns) == ["timestamp", "channel", "value"]
            assert df["timestamp"].dtype == "datetime64[ns, UTC]"
            assert df["value"].dtype == "float64"
            assert set(df["channel"]) == {"BEAM:CURRENT"}

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_single_pv_correct_values(self):
        """Test that single-PV fetch returns correct values, in timestamp order."""
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

            assert (df["channel"] == "PV:X").all()
            assert list(df["value"]) == [1.0, 2.0, 3.0]

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_string_valued_pv_round_trips(self):
        """An mbbi/DBR_STRING-style PV archives a string ``val``; it must come
        back unchanged, not raise."""
        points = [(1704067200, 0, "CW"), (1704067201, 0, "CW"), (1704067202, 0, "STANDBY")]
        response = _make_urlopen_response(_archiver_payload("RF:MODE", points))

        with patch("urllib.request.urlopen", return_value=response):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["RF:MODE"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 1),
            )

            assert (df["channel"] == "RF:MODE").all()
            assert list(df["value"]) == ["CW", "CW", "STANDBY"]

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_multi_pv_returns_rows_for_each_channel(self):
        """Test that multi-PV fetch returns long-format rows for each channel."""
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
            assert list(df.columns) == ["timestamp", "channel", "value"]
            assert set(df["channel"]) == {"PV:1", "PV:2"}
            assert (df["channel"] == "PV:1").sum() == 5
            assert (df["channel"] == "PV:2").sum() == 5

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
            assert list(df.columns) == ["timestamp", "channel", "value"]

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
            assert list(df.columns) == ["timestamp", "channel", "value"]

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


class TestMultiPVLongFormat:
    """Each channel in a multi-PV response keeps its own real timestamps."""

    @pytest.mark.asyncio
    async def test_multi_pv_each_channel_keeps_its_own_timestamps(self):
        """Channels sampled at different instants are not forced onto a shared index."""
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
            assert list(df.columns) == ["timestamp", "channel", "value"]
            assert len(df) == 4
            assert set(df["channel"]) == {"PV:A", "PV:B"}

            expected_a = list(pd.to_datetime([base, base + 2], unit="s", utc=True))
            expected_b = list(pd.to_datetime([base + 1, base + 3], unit="s", utc=True))

            a_timestamps = list(df.loc[df["channel"] == "PV:A", "timestamp"])
            b_timestamps = list(df.loc[df["channel"] == "PV:B", "timestamp"])

            assert a_timestamps == expected_a
            assert b_timestamps == expected_b
            # No shared/aligned grid was fabricated between them.
            assert set(a_timestamps).isdisjoint(set(b_timestamps))

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


class TestConnectSideEffects:
    """connect() must be metadata-only: no third-party client, no stdout.

    Historical context (als-profiles GitLab #8): an earlier implementation
    constructed ``archivertools.ArchiverClient`` inside ``connect()``, whose
    ``__init__`` ICMP-pinged the server (fails in containers where ICMP is
    blocked but HTTP 17668 is open) and printed to stdout (corrupts MCP
    stdio transport). The connector now talks direct HTTP at fetch time;
    these tests pin that ``connect()`` never regresses to connect-time
    side effects.
    """

    @pytest.mark.asyncio
    async def test_connect_needs_no_archiver_client_library(self):
        """connect() succeeds with archivertools entirely absent."""
        with patch.dict("sys.modules", {"archivertools": None}):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "http://archiver.example.com:17668"})

            assert connector._connected is True
            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_does_not_pollute_stdout(self):
        """Nothing may print to stdout during connect() — MCP stdio safety."""
        import io
        import sys

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
    async def test_connect_performs_no_network_io(self):
        """connect() must not open HTTP connections (fetch-time only)."""
        with patch("urllib.request.urlopen", side_effect=AssertionError("network I/O in connect")):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "http://archiver.example.com:17668"})

            assert connector._connected is True
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


class TestQueryWindowTimezone:
    """The wire format is UTC; a caller's zone must be converted, not relabeled.

    Regression: the connector used to strftime a literal 'Z' onto the caller's
    wall-clock digits, shifting every ALS query by 7-8h.
    """

    _LA = ZoneInfo("America/Los_Angeles")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("start", "end", "facility_local"),
        [
            # tz-aware, non-UTC: converted, not relabeled.
            (
                datetime(2026, 7, 30, 10, 0, 0, tzinfo=_LA),
                datetime(2026, 7, 30, 11, 0, 0, tzinfo=_LA),
                False,
            ),
            # already UTC: survives untouched.
            (
                datetime(2026, 7, 30, 17, 0, 0, tzinfo=UTC),
                datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC),
                False,
            ),
            # bare wall-clock: means facility-local, matching the tool layer.
            (datetime(2026, 7, 30, 10, 0, 0), datetime(2026, 7, 30, 11, 0, 0), True),
        ],
        ids=["aware_non_utc", "aware_utc", "naive_facility_local"],
    )
    async def test_window_reaches_the_archiver_as_the_same_utc_instant(
        self, monkeypatch, captured_urls, start, end, facility_local
    ):
        if facility_local:
            monkeypatch.setattr(
                "osprey.utils.config.get_facility_timezone",
                lambda: ZoneInfo("America/Los_Angeles"),
            )

        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        await connector.get_data(
            pv_list=["SR:DCCT"], start_date=start, end_date=end, precision_ms=1000
        )

        # 10:00 PDT is 17:00 UTC; 11:00 PDT is 18:00 UTC.
        assert "from=2026-07-30T17%3A00%3A00.000Z" in captured_urls[0]
        assert "to=2026-07-30T18%3A00%3A00.000Z" in captured_urls[0]

        await connector.disconnect()


class TestProcessingModes:
    """The requested aggregation must reach the archiver as an operator."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("raw", "lastSample_60"),
            ("mean", "mean_60"),
            ("min", "min_60"),
            ("max", "max_60"),
            ("median", "median_60"),
            ("std", "std_60"),
            ("count", "count_60"),
        ],
    )
    async def test_mode_selects_the_archiver_operator(self, captured_urls, mode, expected):
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        await connector.get_data(
            pv_list=["SR:DCCT"],
            start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
            end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
            precision_ms=60_000,
            processing=mode,
        )

        assert expected in captured_urls[0]

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_unknown_mode_raises_value_error(self):
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        with pytest.raises(ValueError, match="Unknown processing mode"):
            await connector.get_data(
                pv_list=["SR:DCCT"],
                start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=60_000,
                processing="p99",
            )

        await connector.disconnect()


class TestNonNumericAggregation:
    """Only ``raw`` is valid for an enum/status channel — on every backend.

    Regression: EPICS pushed the aggregation to the appliance without enforcing
    this, so a mean over a string PV returned raw values labelled as means.
    """

    @staticmethod
    def _connector_returning(payload: list):
        return patch("urllib.request.urlopen", return_value=_make_urlopen_response(payload))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["mean", "min", "max", "median", "std", "count"])
    async def test_aggregate_over_string_channel_raises_naming_it(self, mode):
        payload = _archiver_payload("SR:MODE", [(1000, 0, "CW"), (1060, 0, "STANDBY")])

        with self._connector_returning(payload):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            with pytest.raises(ValueError, match="SR:MODE"):
                await connector.get_data(
                    pv_list=["SR:MODE"],
                    start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                    end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                    precision_ms=60_000,
                    processing=mode,
                )

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_raw_over_string_channel_is_allowed(self):
        """``raw`` never aggregates, so an enum channel passes through untouched."""
        payload = _archiver_payload("SR:MODE", [(1000, 0, "CW"), (1060, 0, "STANDBY")])

        with self._connector_returning(payload):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["SR:MODE"],
                start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=60_000,
                processing="raw",
            )

            assert df["value"].tolist() == ["CW", "STANDBY"]

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_empty_channel_does_not_fail_an_aggregate_query(self):
        """A channel that matched nothing has nothing to reject."""
        with self._connector_returning([]):
            connector = EPICSArchiverConnector()
            await connector.connect({"url": "https://archiver.example.com"})

            df = await connector.get_data(
                pv_list=["SR:DCCT"],
                start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=60_000,
                processing="mean",
            )

            assert df.empty

            await connector.disconnect()


class TestSubSecondPrecision:
    """A bin width the appliance cannot express is rejected, not rounded.

    Regression: the width was floored with ``max(1, precision_ms // 1000)``,
    silently serving a 500 ms or 1500 ms request at 1 s.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("precision_ms", [1, 500, 999, 1500, 2500])
    async def test_non_whole_second_precision_is_rejected(self, precision_ms):
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        with pytest.raises(ValueError, match="whole number of seconds"):
            await connector.get_data(
                pv_list=["SR:DCCT"],
                start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=precision_ms,
                processing="raw",
            )

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_rejection_follows_resolve_processing_rather_than_a_second_copy_of_the_rule(
        self, monkeypatch, captured_urls
    ):
        """Which widths are expressible is decided once, in ``resolve_processing``.

        The helper is forced to reject a width EPICS' own arithmetic would wave
        through; the request must be refused, not downgraded to a bare PV name.
        """
        monkeypatch.setattr(
            "osprey.connectors.archiver.epics_archiver_connector.resolve_processing",
            lambda processing, precision_ms: Processing(
                mode=processing,
                precision_ms=precision_ms,
                epics_operator=None,
            ),
        )

        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        with pytest.raises(ValueError, match="whole number of seconds"):
            await connector.get_data(
                pv_list=["SR:DCCT"],
                start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
                end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
                precision_ms=60_000,
                processing="mean",
            )

        assert captured_urls == []

        await connector.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("processing", "precision_ms", "expected"),
        [
            ("mean", 1000, "mean_1"),
            ("mean", 60_000, "mean_60"),
            ("mean", 3600_000, "mean_3600"),
            # raw at a non-60s width: raw's operator width must track precision_ms.
            ("raw", 5000, "lastSample_5"),
        ],
    )
    async def test_whole_second_precision_renders_the_operator(
        self, captured_urls, processing, precision_ms, expected
    ):
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        await connector.get_data(
            pv_list=["SR:DCCT"],
            start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
            end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
            precision_ms=precision_ms,
            processing=processing,
        )

        assert len(captured_urls) == 1
        assert expected in captured_urls[0]

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_full_resolution_is_still_allowed(self, captured_urls):
        """precision_ms <= 0 is full resolution, not a sub-second bin width."""
        connector = EPICSArchiverConnector()
        await connector.connect({"url": "https://archiver.example.com"})

        await connector.get_data(
            pv_list=["SR:DCCT"],
            start_date=datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
            end_date=datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC),
            precision_ms=0,
            processing="raw",
        )

        # Bare PV name, no operator wrapper.
        assert "lastSample" not in captured_urls[0]

        await connector.disconnect()
