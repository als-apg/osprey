"""
Unit tests for DOOCSArchiverConnector.

All tests mock doocs4py so no installed DOOCS environment is required.
"""

import sys
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

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
        chunk = [(float(t), 0, 0, float(v)) for t, v in zip(times, values, strict=False)]

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


# --------------------------------------------------------------------------------------
# Disconnected guard / timezone
# --------------------------------------------------------------------------------------


class TestDisconnectedGuard:
    """A disconnected connector must not reach the ENS."""

    async def test_never_connected_returns_all_false_without_lookups(self):
        from osprey.connectors.archiver.doocs_archiver_connector import (
            DOOCSArchiverConnector,
        )

        conn = DOOCSArchiverConnector()
        avail = await conn.check_availability(["FAC/DEV/LOC/P"])

        assert avail == {"FAC/DEV/LOC/P": False}

    async def test_after_disconnect_returns_all_false_without_lookups(self):
        mock_d4py = _make_doocs4py(names_result=[("FAC/DEV/LOC/P.HIST", "value")])

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            await conn.disconnect()

            mock_d4py.names.reset_mock()
            avail = await conn.check_availability(["FAC/DEV/LOC/P"])

        assert avail == {"FAC/DEV/LOC/P": False}
        assert mock_d4py.names.call_count == 0

    async def test_double_disconnect_is_safe(self):
        mock_d4py = _make_doocs4py()

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            await conn.disconnect()
            await conn.disconnect()

        assert conn._doocs4py is None
        assert conn._connected is False


class TestQueryWindowTimezone:
    """Naive bounds must resolve against the facility zone, not the host's."""

    async def test_naive_bounds_use_the_facility_zone(self, archiver, monkeypatch, request):
        from zoneinfo import ZoneInfo

        # Force the host zone to UTC so this test is RED on every machine if the
        # fix regresses -- not only on hosts whose zone happens to differ from the
        # facility zone patched below. This repo's users are largely at LBNL, in
        # America/Los_Angeles -- exactly the zone patched here -- so without this
        # a reintroduced host-zone .timestamp() bug would show green on a
        # developer's own machine. monkeypatch restores the TZ env var
        # automatically; tzset() must be called again on the way out so the C
        # library actually forgets the override.
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()
        request.addfinalizer(time.tzset)

        conn, mock_d4py = archiver
        monkeypatch.setattr(
            "osprey.connectors.archiver._timerange.get_facility_timezone",
            lambda: ZoneInfo("America/Los_Angeles"),
        )

        captured = {}
        original = conn._read_history

        def _spy(address, start_time, end_time, max_points=None, avg_window=None):
            captured["start"] = start_time
            captured["end"] = end_time
            return original(address, start_time, end_time, max_points, avg_window)

        conn._read_history = _spy

        await conn.get_data(
            pv_list=["FAC/DEV/LOC/P"],
            start_date=datetime(2026, 7, 30, 10, 0, 0),
            end_date=datetime(2026, 7, 30, 11, 0, 0),
        )

        # 10:00 PDT is 17:00 UTC.
        assert captured["start"] == datetime(2026, 7, 30, 17, 0, 0, tzinfo=UTC).timestamp()
        assert captured["end"] == datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC).timestamp()


class TestProcessingSparseData:
    """A non-raw processing mode must not manufacture samples DOOCS never recorded.

    ``_read_history`` bounds "raw" by construction (it never returns more than
    ``num_points`` samples), but when the archive itself is sparser than the
    requested precision_ms, its fallback-to-full-resolution path returns however
    many real samples exist, spaced however far apart they really are — which
    can be much wider than precision_ms. Resampling that at the raw precision_ms
    would upsample onto a much finer grid than the data has, the same exposure
    fixed for the Mock and MongoDB archiver connectors.
    """

    async def test_sparse_history_is_not_upsampled(self):
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 1, 1, 6, 0, 0, tzinfo=UTC)  # 6-hour window
        # Only 6 archived samples, one per hour -- far sparser than the
        # requested 1s precision_ms below.
        times = np.linspace(start.timestamp(), end.timestamp(), 6)
        chunk = [(float(t), 0, 0, float(i)) for i, t in enumerate(times)]
        mock_d4py = _make_doocs4py(chunk=chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAC/DEV/LOC/P"],
                start_date=start,
                end_date=end,
                precision_ms=1000,
                processing="mean",
            )
            await conn.disconnect()

        # Naively resampling at the requested 1 s precision against data
        # actually spaced ~1 hour apart would inflate this to ~21,601 rows,
        # almost entirely NaN. The row count must stay near the archived
        # sample count instead, with no NaN rows.
        assert len(df) <= 10
        assert not df["FAC/DEV/LOC/P"].isna().any()


class TestProcessingGenuineAggregation:
    """A non-raw processing mode must aggregate over the real archived samples,
    not a single zero-order-hold sample per bin.

    Regression coverage for finding #2 reintroduced inside DOOCS: get_data used
    to pass ``num_points`` (derived from precision_ms) into ``_read_history`` for
    every processing mode, decimating to one zero-order-hold sample per bin
    *before* the resampler ever saw the data. mean/count/std then degenerated
    to a relabeled last-sample value instead of a genuine aggregate.
    """

    @staticmethod
    def _dense_chunk(n_bins: int, samples_per_bin: int):
        """``samples_per_bin`` samples per 1-second bin, values 0..N-1 in order."""
        chunk = []
        v = 0
        for b in range(n_bins):
            for s in range(samples_per_bin):
                t = _START_TS + b + s / samples_per_bin
                chunk.append((float(t), 0, 0, float(v)))
                v += 1
        return chunk

    async def test_mean_aggregates_true_within_bin_average(self):
        chunk = self._dense_chunk(n_bins=3, samples_per_bin=10)
        mock_d4py = _make_doocs4py(chunk=chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAC/DEV/LOC/P"],
                start_date=_START,
                end_date=datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                precision_ms=1000,
                processing="mean",
            )
            await conn.disconnect()

        # Bin 0: values 0..9 -> true mean 4.5; bin 1: 10..19 -> 14.5;
        # bin 2: 20..29 -> 24.5. A zero-order-hold-then-resample bug instead
        # reports a single raw sample's value, relabeled, per bin.
        values = df["FAC/DEV/LOC/P"].dropna().tolist()
        assert values == pytest.approx([4.5, 14.5, 24.5])

    async def test_count_returns_true_per_bin_sample_count(self):
        chunk = self._dense_chunk(n_bins=3, samples_per_bin=10)
        mock_d4py = _make_doocs4py(chunk=chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAC/DEV/LOC/P"],
                start_date=_START,
                end_date=datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                precision_ms=1000,
                processing="count",
            )
            await conn.disconnect()

        # 10 real samples land in each 1 s bin; a zero-order-hold-then-resample
        # bug instead reports (at most) 1 per bin.
        counts = df["FAC/DEV/LOC/P"].dropna().tolist()
        assert counts == [10, 10, 10]

    @staticmethod
    def _two_pv_mock(chunk_a, chunk_b):
        """A mock doocs4py wired to return ``chunk_a`` then ``chunk_b`` on two
        successive ``get()`` calls, matching the fetch order for a 2-PV request."""
        mock_d4py = MagicMock()
        mock_d4py.__version__ = "2.0.0"
        mock_d4py.names.return_value = [("FACILITY", "XFEL")]
        r_a = MagicMock()
        r_a.value = chunk_a
        r_b = MagicMock()
        r_b.value = chunk_b
        mock_d4py.get.side_effect = [r_a, r_b]
        return mock_d4py

    async def test_multi_pv_mean_aggregates_true_within_bin_average_for_both_columns(self):
        """Regression for finding #1 reached via alignment: align_to_grid forward-fills
        onto one value per grid point, so resampling *after* alignment (the brief's
        original order) would aggregate over an already-decimated series, same as the
        single-PV bug. Both columns must carry the true within-bin means.
        """
        chunk = self._dense_chunk(n_bins=3, samples_per_bin=10)
        mock_d4py = self._two_pv_mock(chunk, chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAC/DEV/LOC/A", "FAC/DEV/LOC/B"],
                start_date=_START,
                end_date=datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                precision_ms=1000,
                processing="mean",
            )
            await conn.disconnect()

        # Bin 0: values 0..9 -> 4.5; bin 1: 10..19 -> 14.5; bin 2: 20..29 -> 24.5.
        # align_to_grid's query-window grid runs one step past the last real bin
        # (00:00:03), forward-filling the last bin's mean there -- expected and
        # unrelated to aggregation correctness, which is what this test pins.
        assert df["FAC/DEV/LOC/A"].tolist() == pytest.approx([4.5, 14.5, 24.5, 24.5])
        assert df["FAC/DEV/LOC/B"].tolist() == pytest.approx([4.5, 14.5, 24.5, 24.5])

    async def test_multi_pv_count_returns_true_per_bin_sample_count_for_both_columns(self):
        chunk = self._dense_chunk(n_bins=3, samples_per_bin=10)
        mock_d4py = self._two_pv_mock(chunk, chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAC/DEV/LOC/A", "FAC/DEV/LOC/B"],
                start_date=_START,
                end_date=datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                precision_ms=1000,
                processing="count",
            )
            await conn.disconnect()

        # 10 real samples land in each 1 s bin, on both PVs.
        assert df["FAC/DEV/LOC/A"].tolist() == [10, 10, 10, 10]
        assert df["FAC/DEV/LOC/B"].tolist() == [10, 10, 10, 10]

    async def test_multi_pv_raw_output_unchanged_by_the_aggregation_restructure(self):
        """Pin the default (raw) multi-PV path so the aggregation restructure above
        cannot silently change it: raw still decimates via _read_history's
        num_points zero-order hold, then align_to_grid unions at precision_ms --
        exactly as before this fix pass.
        """
        chunk = self._dense_chunk(n_bins=3, samples_per_bin=10)
        mock_d4py = self._two_pv_mock(chunk, chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAC/DEV/LOC/A", "FAC/DEV/LOC/B"],
                start_date=_START,
                end_date=datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                precision_ms=1000,
                # processing defaults to "raw"
            )
            await conn.disconnect()

        assert df["FAC/DEV/LOC/A"].tolist() == pytest.approx([0.0, 9.0, 19.0, 29.0])
        assert df["FAC/DEV/LOC/B"].tolist() == pytest.approx([0.0, 9.0, 19.0, 29.0])
