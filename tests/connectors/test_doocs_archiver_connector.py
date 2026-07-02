"""
Unit tests for DOOCSArchiverConnector.

All tests mock doocs4py so no installed DOOCS environment is required.
"""

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

UTC = timezone.utc

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_END = datetime(2026, 1, 1, 1, 0, 0, tzinfo=UTC)  # 1-hour window

_START_TS = _START.timestamp()
_END_TS = _END.timestamp()


def _make_raw_chunk(n=20, t_start=None, t_end=None):
    """Return a list of (timestamp, _, _, value) tuples like DOOCS TTII data."""
    t_start = t_start or _START_TS
    t_end = t_end or _END_TS
    times = np.linspace(t_start, t_end, n)
    return [(float(t), 0, 0, float(i)) for i, t in enumerate(times)]


def _make_doocs4py(chunk=None, names_result=None):
    """Return a mock doocs4py module wired for history reads.

    _read_history exits after the first full-range chunk because
    ``current_stop <= start_ts`` becomes true immediately.  A single
    return value is enough; no empty sentinel is needed.
    """
    chunk = chunk if chunk is not None else _make_raw_chunk()

    d = MagicMock()
    d.__version__ = "2.0.0"
    d.names.return_value = names_result or [("FACILITY", "XFEL")]

    result_with_data = MagicMock()
    result_with_data.value = chunk
    d.get.return_value = result_with_data
    return d


# --------------------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------------------


@pytest.fixture
async def archiver():
    """DOOCSArchiverConnector connected with a mock doocs4py."""
    mock_d4py = _make_doocs4py()

    with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
        from osprey.connectors.archiver.doocs_archiver_connector import (
            DOOCSArchiverConnector,
        )

        conn = DOOCSArchiverConnector()
        await conn.connect({})
        yield conn, mock_d4py
        await conn.disconnect()


# --------------------------------------------------------------------------------------
# connect / disconnect
# --------------------------------------------------------------------------------------


class TestArchiverConnect:
    async def test_connect_sets_connected(self):
        mock_d4py = _make_doocs4py()
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            assert conn._connected is False
            await conn.connect({})
            assert conn._connected is True
            await conn.disconnect()

    async def test_connect_raises_import_error_without_doocs4py(self):
        with patch.dict(sys.modules, {"doocs4py": None}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            with pytest.raises(ImportError, match="doocs4py"):
                await conn.connect({})

    async def test_connect_raises_on_ens_failure(self):
        mock_d4py = _make_doocs4py()
        mock_d4py.names.side_effect = RuntimeError("ENS down")
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            with pytest.raises(Exception, match="ENS"):
                await conn.connect({})

    async def test_connect_stores_avg_window(self):
        mock_d4py = _make_doocs4py()
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({"avg_window": 30})
            assert conn._avg_window == 30
            await conn.disconnect()

    async def test_disconnect_clears_connected(self, archiver):
        conn, _ = archiver
        await conn.disconnect()
        assert conn._connected is False


# --------------------------------------------------------------------------------------
# get_data — type validation / guard rails
# --------------------------------------------------------------------------------------


class TestGetDataValidation:
    async def test_raises_when_not_connected(self):
        mock_d4py = _make_doocs4py()
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            # Never called connect()
            with pytest.raises(RuntimeError, match="not connected"):
                await conn.get_data(["ADDR"], _START, _END)

    async def test_raises_on_invalid_start_date(self, archiver):
        conn, _ = archiver
        with pytest.raises(TypeError, match="start_date"):
            await conn.get_data(["ADDR"], "2026-01-01", _END)

    async def test_raises_on_invalid_end_date(self, archiver):
        conn, _ = archiver
        with pytest.raises(TypeError, match="end_date"):
            await conn.get_data(["ADDR"], _START, 1234567890)


# --------------------------------------------------------------------------------------
# get_data — single PV
# --------------------------------------------------------------------------------------


class TestGetDataSinglePV:
    async def test_returns_dataframe(self, archiver):
        conn, _ = archiver
        df = await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)
        assert isinstance(df, pd.DataFrame)

    async def test_single_pv_column_present(self, archiver):
        conn, _ = archiver
        df = await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)
        assert "FAC/DEV/LOC/PROP" in df.columns

    async def test_single_pv_has_data(self, archiver):
        conn, _ = archiver
        df = await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)
        assert len(df) > 0

    async def test_hist_suffix_appended_to_address(self, archiver):
        conn, mock_d4py = archiver
        await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)
        # Address object should have been created with .HIST suffix
        addr_calls = [str(c) for c in mock_d4py.Address.call_args_list]
        assert any("HIST" in c for c in addr_calls)

    async def test_hist_suffix_not_doubled(self, archiver):
        conn, mock_d4py = archiver
        await conn.get_data(["FAC/DEV/LOC/PROP.HIST"], _START, _END)
        addr_calls = [str(c) for c in mock_d4py.Address.call_args_list]
        assert not any("HIST.HIST" in c for c in addr_calls)


# --------------------------------------------------------------------------------------
# get_data — multi-PV alignment
# --------------------------------------------------------------------------------------


class TestGetDataMultiPV:
    async def test_multi_pv_all_columns_present(self):
        """Two PVs: each gets its own mock get() sequence.

        _read_history fetches chunks in reverse chronological order.  A single
        chunk that spans [start, end] already sets current_stop <= start_ts, so
        the loop exits after ONE get() call per PV — no second "empty" sentinel
        is needed.
        """
        chunk_a = _make_raw_chunk(n=10)
        chunk_b = _make_raw_chunk(n=15)

        mock_d4py = MagicMock()
        mock_d4py.__version__ = "2.0.0"
        mock_d4py.names.return_value = [("FACILITY", "XFEL")]

        # One get() per PV — the full-range chunk causes the loop to exit.
        r_a = MagicMock()
        r_a.value = chunk_a
        r_b = MagicMock()
        r_b.value = chunk_b
        mock_d4py.get.side_effect = [r_a, r_b]

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                ["FAC/DEV/LOC/A", "FAC/DEV/LOC/B"], _START, _END, precision_ms=1000
            )
            await conn.disconnect()

        assert "FAC/DEV/LOC/A" in df.columns
        assert "FAC/DEV/LOC/B" in df.columns


# --------------------------------------------------------------------------------------
# _read_history — internal unit tests
# --------------------------------------------------------------------------------------


class TestReadHistory:
    def _make_connector_with_d4py(self, mock_d4py):
        """Return a DOOCSArchiverConnector with _doocs4py already set (no connect)."""
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            conn._doocs4py = mock_d4py
            conn._connected = True
            conn._avg_window = None
            return conn

    def test_returns_none_on_empty_data(self):
        mock_d4py = MagicMock()
        result_empty = MagicMock()
        result_empty.value = []
        mock_d4py.get.return_value = result_empty

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)
        assert out is None

    def test_returns_none_on_exception(self):
        mock_d4py = MagicMock()
        mock_d4py.get.side_effect = RuntimeError("DOOCS error")

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)
        assert out is None

    def _mock_get(self, mock_d4py, chunk):
        """Wire mock_d4py.get to return chunk once."""
        r = MagicMock()
        r.value = chunk
        mock_d4py.get.return_value = r

    def test_raw_data_returned_when_no_max_points(self):
        chunk = _make_raw_chunk(n=20)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)

        assert out is not None
        assert "time" in out and "data" in out
        assert len(out["time"]) == 20
        assert len(out["data"]) == 20

    def test_resampling_respects_max_points(self):
        chunk = _make_raw_chunk(n=100)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS, max_points=10)

        assert out is not None
        assert len(out["time"]) == 10
        assert len(out["data"]) == 10

    def test_smoothing_applied_with_avg_window(self):
        # Use a step-function signal: 25 zeros then 25 ones.
        # A moving average will blur the step, producing values in (0, 1) near
        # the boundary — demonstrably different from the raw step.
        n = 50
        times = np.linspace(_START_TS, _END_TS, n)
        values = [0.0] * 25 + [1.0] * 25
        chunk = [(float(t), 0, 0, float(v)) for t, v in zip(times, values)]

        mock_d4py = MagicMock()
        r = MagicMock()
        r.value = chunk
        mock_d4py.get.return_value = r

        conn = self._make_connector_with_d4py(mock_d4py)
        # avg_window=600s → win ≈ 8 samples over 1-hour / 50-point grid
        out_smooth = conn._read_history(
            "FAC/DEV/LOC/PROP", _START_TS, _END_TS, max_points=50, avg_window=600.0
        )

        assert out_smooth is not None
        # Smoothed step has values between 0 and 1 near the transition
        assert np.any((out_smooth["data"] > 0.0) & (out_smooth["data"] < 1.0))

    def test_hist_suffix_appended(self):
        chunk = _make_raw_chunk(n=5)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)

        addr_arg = mock_d4py.Address.call_args[0][0]
        assert addr_arg.endswith(".HIST")

    def test_hist_suffix_not_doubled(self):
        chunk = _make_raw_chunk(n=5)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        conn._read_history("FAC/DEV/LOC/PROP.HIST", _START_TS, _END_TS)

        addr_arg = mock_d4py.Address.call_args[0][0]
        assert addr_arg.count(".HIST") == 1

    def test_metadata_fields_present(self):
        chunk = _make_raw_chunk(n=10)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)

        assert "metadata" in out
        meta = out["metadata"]
        assert "raw_count" in meta
        assert "max_points" in meta
        assert "avg_window" in meta


# --------------------------------------------------------------------------------------
# check_availability
# --------------------------------------------------------------------------------------


class TestCheckAvailability:
    async def test_available_when_names_returns_result(self, archiver):
        conn, mock_d4py = archiver
        mock_d4py.names.return_value = [("FAC/DEV/LOC/PROP.HIST", "some_value")]

        avail = await conn.check_availability(["FAC/DEV/LOC/PROP"])

        assert avail["FAC/DEV/LOC/PROP"] is True

    async def test_unavailable_when_names_returns_empty(self, archiver):
        conn, mock_d4py = archiver
        mock_d4py.names.return_value = []

        avail = await conn.check_availability(["FAC/DEV/LOC/MISSING"])

        assert avail["FAC/DEV/LOC/MISSING"] is False

    async def test_unavailable_when_names_raises(self, archiver):
        conn, mock_d4py = archiver
        mock_d4py.names.side_effect = RuntimeError("lookup failed")

        avail = await conn.check_availability(["FAC/DEV/LOC/PROP"])

        assert avail["FAC/DEV/LOC/PROP"] is False

    async def test_hist_suffix_appended_for_lookup(self, archiver):
        conn, mock_d4py = archiver
        mock_d4py.names.return_value = []

        await conn.check_availability(["FAC/DEV/LOC/PROP"])

        call_arg = mock_d4py.names.call_args[0][0]
        assert call_arg.endswith(".HIST")

    async def test_hist_suffix_not_doubled_for_lookup(self, archiver):
        conn, mock_d4py = archiver
        mock_d4py.names.return_value = []

        await conn.check_availability(["FAC/DEV/LOC/PROP.HIST"])

        call_arg = mock_d4py.names.call_args[0][0]
        assert call_arg.count(".HIST") == 1


# --------------------------------------------------------------------------------------
# get_metadata
# --------------------------------------------------------------------------------------


class TestGetMetadata:
    async def test_get_metadata_returns_archiver_metadata(self, archiver):
        from osprey.connectors.archiver.base import ArchiverMetadata

        conn, _ = archiver
        meta = await conn.get_metadata("FAC/DEV/LOC/PROP")

        assert isinstance(meta, ArchiverMetadata)
        assert meta.pv_name == "FAC/DEV/LOC/PROP"
        assert meta.is_archived is True
