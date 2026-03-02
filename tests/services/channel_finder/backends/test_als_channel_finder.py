"""Tests for ALSChannelFinderBackend with mocked HTTP responses."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp

from osprey.services.channel_finder.backends.als_channel_finder import (
    ALSChannelFinderBackend,
    _channel_to_pvrecord,
)
from osprey.services.channel_finder.backends.base import PVRecord, SearchResult


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Realistic ChannelFinder JSON fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CHANNEL = {
    "name": "SR:01:BPM:01:X",
    "owner": "controls",
    "properties": [
        {"name": "pvName", "value": "SR:01:BPM:01:X"},
        {"name": "recordType", "value": "ai"},
        {"name": "recordDesc", "value": "Horizontal beam position"},
        {"name": "hostName", "value": "als-srv01.lbl.gov"},
        {"name": "iocName", "value": "IOC:BPM:01"},
        {"name": "engineer", "value": "jdoe"},
    ],
    "tags": [],
}

_SAMPLE_CHANNEL_2 = {
    "name": "SR:01:BPM:01:Y",
    "owner": "controls",
    "properties": [
        {"name": "pvName", "value": "SR:01:BPM:01:Y"},
        {"name": "recordType", "value": "ai"},
        {"name": "recordDesc", "value": "Vertical beam position"},
        {"name": "hostName", "value": "als-srv01.lbl.gov"},
        {"name": "iocName", "value": "IOC:BPM:01"},
    ],
    "tags": [],
}

_SAMPLE_CHANNEL_AO = {
    "name": "SR:01:HCM:01:SP",
    "owner": "controls",
    "properties": [
        {"name": "recordType", "value": "ao"},
        {"name": "recordDesc", "value": "Current setpoint"},
        {"name": "hostName", "value": "als-srv01.lbl.gov"},
        {"name": "iocName", "value": "IOC:COR:01"},
    ],
    "tags": [],
}


# ---------------------------------------------------------------------------
# _channel_to_pvrecord unit tests
# ---------------------------------------------------------------------------


class TestChannelToPVRecord:
    def test_basic_mapping(self):
        rec = _channel_to_pvrecord(_SAMPLE_CHANNEL)
        assert rec.name == "SR:01:BPM:01:X"
        assert rec.record_type == "ai"
        assert rec.description == "Horizontal beam position"
        assert rec.host == "als-srv01.lbl.gov"
        assert rec.ioc == "IOC:BPM:01"
        assert rec.units == ""  # not available from ChannelFinder

    def test_extra_properties_go_to_tags(self):
        rec = _channel_to_pvrecord(_SAMPLE_CHANNEL)
        assert "engineer" in rec.tags
        assert rec.tags["engineer"] == "jdoe"

    def test_known_props_excluded_from_tags(self):
        rec = _channel_to_pvrecord(_SAMPLE_CHANNEL)
        for known in ("recordType", "recordDesc", "hostName", "iocName", "pvName"):
            assert known not in rec.tags

    def test_missing_properties(self):
        rec = _channel_to_pvrecord({"name": "TEST:PV", "properties": []})
        assert rec.name == "TEST:PV"
        assert rec.record_type == ""
        assert rec.description == ""
        assert rec.tags == {}

    def test_empty_channel(self):
        rec = _channel_to_pvrecord({})
        assert rec.name == ""
        assert isinstance(rec, PVRecord)


# ---------------------------------------------------------------------------
# Helper to build a mock backend with patched aiohttp
# ---------------------------------------------------------------------------


def _make_mock_response(json_data=None, text_data=None, status=200):
    """Create a mock aiohttp response as an async context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=status,
        )
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)
    if text_data is not None:
        resp.text = AsyncMock(return_value=text_data)
    return resp


class _AsyncCtxMgr:
    """Minimal async context manager wrapping a response."""

    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


class _MockSessionCtx:
    """Replaces aiohttp.ClientSession for testing.

    get() is a sync method returning an async context manager,
    matching aiohttp.ClientSession.get() behavior.
    """

    def __init__(self, responses: dict[str, object]):
        self._responses = responses
        self.closed = False

    def get(self, url, params=None):
        # Match by path prefix — sync method, returns async ctx mgr
        for key, resp in self._responses.items():
            if key in url:
                return _AsyncCtxMgr(resp)
        raise aiohttp.ClientError(f"Unexpected URL: {url}")

    async def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# ALSChannelFinderBackend.search tests
# ---------------------------------------------------------------------------


class TestALSSearch:
    def _backend_with_responses(self, count: int, channels: list):
        """Create a backend with mocked count + channel responses."""
        backend = ALSChannelFinderBackend("https://localhost:8443/ChannelFinder")
        count_resp = _make_mock_response(text_data=str(count))
        channels_resp = _make_mock_response(json_data=channels)
        backend._session = _MockSessionCtx(
            {"channels/count": count_resp, "channels": channels_resp}
        )
        return backend

    def test_basic_search(self):
        backend = self._backend_with_responses(2, [_SAMPLE_CHANNEL, _SAMPLE_CHANNEL_2])
        result = _run(backend.search("SR:01:BPM:*"))

        assert isinstance(result, SearchResult)
        assert result.total_count == 2
        assert len(result.records) == 2
        assert result.records[0].name == "SR:01:BPM:01:X"
        assert result.page == 1
        assert result.has_more is False

    def test_empty_search(self):
        backend = self._backend_with_responses(0, [])
        result = _run(backend.search("NONEXISTENT:*"))

        assert result.total_count == 0
        assert result.records == []
        assert result.has_more is False

    def test_pagination_math(self):
        """Verify page→offset conversion and has_more calculation."""
        backend = ALSChannelFinderBackend("https://localhost:8443/ChannelFinder")
        count_resp = _make_mock_response(text_data="150")
        channels_resp = _make_mock_response(json_data=[_SAMPLE_CHANNEL] * 50)

        # Track the params sent to verify offset calculation
        captured_params = {}

        def mock_get(url, params=None):
            if "count" in url:
                return _AsyncCtxMgr(count_resp)
            captured_params.update(params or {})
            return _AsyncCtxMgr(channels_resp)

        session = MagicMock()
        session.closed = False
        session.get = mock_get
        backend._session = session

        # Page 3, size 50 → offset should be 100
        result = _run(backend.search("*", page=3, page_size=50))
        assert captured_params["~from"] == "100"
        assert captured_params["~size"] == "50"
        assert result.page == 3
        assert result.has_more is False  # 100+50 = 150 = total

    def test_page_size_capped(self):
        backend = ALSChannelFinderBackend("https://localhost:8443/ChannelFinder")
        count_resp = _make_mock_response(text_data="10")
        channels_resp = _make_mock_response(json_data=[_SAMPLE_CHANNEL] * 10)

        captured_params = {}

        def mock_get(url, params=None):
            if "count" in url:
                return _AsyncCtxMgr(count_resp)
            captured_params.update(params or {})
            return _AsyncCtxMgr(channels_resp)

        session = MagicMock()
        session.closed = False
        session.get = mock_get
        backend._session = session

        result = _run(backend.search("*", page_size=999))
        assert result.page_size == 200
        assert captured_params["~size"] == "200"

    def test_has_more_true(self):
        backend = self._backend_with_responses(250, [_SAMPLE_CHANNEL] * 100)
        result = _run(backend.search("*", page=1, page_size=100))
        assert result.has_more is True

    def test_offset_beyond_total_returns_empty(self):
        """If page is beyond available results, return empty."""
        backend = self._backend_with_responses(5, [])
        result = _run(backend.search("*", page=100, page_size=100))
        assert result.records == []
        assert result.has_more is False


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------


class TestFilterParams:
    def test_record_type_filter(self):
        backend = ALSChannelFinderBackend()
        params = backend._build_query_params("*", record_type="ao")
        assert params["recordType"] == "ao"
        assert params["~name"] == "*"

    def test_ioc_filter(self):
        backend = ALSChannelFinderBackend()
        params = backend._build_query_params("SR:*", ioc="IOC:BPM:01")
        assert params["iocName"] == "IOC:BPM:01"

    def test_no_filters(self):
        backend = ALSChannelFinderBackend()
        params = backend._build_query_params("SR:*")
        assert params == {"~name": "SR:*"}

    def test_combined_filters(self):
        backend = ALSChannelFinderBackend()
        params = backend._build_query_params("*", record_type="ai", ioc="IOC:BPM:01")
        assert params["~name"] == "*"
        assert params["recordType"] == "ai"
        assert params["iocName"] == "IOC:BPM:01"


# ---------------------------------------------------------------------------
# description_contains client-side filtering
# ---------------------------------------------------------------------------


class TestDescriptionContains:
    def test_filters_client_side(self):
        channels = [_SAMPLE_CHANNEL, _SAMPLE_CHANNEL_2, _SAMPLE_CHANNEL_AO]
        backend = ALSChannelFinderBackend()
        count_resp = _make_mock_response(text_data="3")
        channels_resp = _make_mock_response(json_data=channels)
        backend._session = _MockSessionCtx(
            {"channels/count": count_resp, "channels": channels_resp}
        )

        result = _run(backend.search("*", description_contains="horizontal"))
        assert len(result.records) == 1
        assert result.records[0].name == "SR:01:BPM:01:X"

    def test_case_insensitive(self):
        channels = [_SAMPLE_CHANNEL]
        backend = ALSChannelFinderBackend()
        count_resp = _make_mock_response(text_data="1")
        channels_resp = _make_mock_response(json_data=channels)
        backend._session = _MockSessionCtx(
            {"channels/count": count_resp, "channels": channels_resp}
        )

        result = _run(backend.search("*", description_contains="HORIZONTAL"))
        assert len(result.records) == 1

    def test_has_more_with_description_filter(self):
        """has_more reflects server-side count even with client-side filtering."""
        channels = [_SAMPLE_CHANNEL, _SAMPLE_CHANNEL_2]
        backend = ALSChannelFinderBackend()
        # Server says 500 total, we only fetched 2 (page_size=2)
        count_resp = _make_mock_response(text_data="500")
        channels_resp = _make_mock_response(json_data=channels)
        backend._session = _MockSessionCtx(
            {"channels/count": count_resp, "channels": channels_resp}
        )

        result = _run(backend.search("*", description_contains="beam", page_size=2))
        # Both match "beam position"
        assert len(result.records) == 2
        # has_more should be True (server has 500, we only fetched 2)
        assert result.has_more is True


# ---------------------------------------------------------------------------
# get_metadata tests
# ---------------------------------------------------------------------------


class TestGetMetadata:
    def test_single_pv(self):
        backend = ALSChannelFinderBackend()
        channels_resp = _make_mock_response(json_data=[_SAMPLE_CHANNEL])
        backend._session = _MockSessionCtx({"channels": channels_resp})

        records = _run(backend.get_metadata(["SR:01:BPM:01:X"]))
        assert len(records) == 1
        assert records[0].name == "SR:01:BPM:01:X"
        assert records[0].record_type == "ai"

    def test_missing_pv(self):
        backend = ALSChannelFinderBackend()
        channels_resp = _make_mock_response(json_data=[])
        backend._session = _MockSessionCtx({"channels": channels_resp})

        records = _run(backend.get_metadata(["DOES:NOT:EXIST"]))
        assert records == []

    def test_empty_list(self):
        backend = ALSChannelFinderBackend()
        records = _run(backend.get_metadata([]))
        assert records == []

    def test_caps_at_100(self):
        backend = ALSChannelFinderBackend()
        channels_resp = _make_mock_response(json_data=[])
        backend._session = _MockSessionCtx({"channels": channels_resp})

        names = [f"PV:{i}" for i in range(150)]
        records = _run(backend.get_metadata(names))
        # Should not error, just cap at 100
        assert isinstance(records, list)

    def test_handles_connection_error(self):
        """Connection errors for individual PVs are logged, not raised."""
        backend = ALSChannelFinderBackend()

        def failing_get(url, params=None):
            raise aiohttp.ClientError("Connection refused")

        session = MagicMock()
        session.closed = False
        session.get = failing_get
        backend._session = session

        # Should not raise
        records = _run(backend.get_metadata(["SR:01:BPM:01:X"]))
        assert records == []


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


class TestConnectionManagement:
    def test_close(self):
        backend = ALSChannelFinderBackend()
        mock_session = AsyncMock()
        mock_session.closed = False
        backend._session = mock_session

        _run(backend.close())
        mock_session.close.assert_awaited_once()

    def test_close_idempotent(self):
        backend = ALSChannelFinderBackend()
        # No session — close should be a no-op
        _run(backend.close())

    def test_lazy_session_creation(self):
        backend = ALSChannelFinderBackend()
        assert backend._session is None
