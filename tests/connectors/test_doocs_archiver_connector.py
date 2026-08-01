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
        """The canonical long frame, and a populated one.

        ``isinstance(df, pd.DataFrame)`` alone passed even when ``get_data``
        returned an empty frame for every query -- and an empty frame still
        carries the three long-format columns, so the column check does not
        rescue it either. The emptiness assertion is what makes this test able
        to fail.
        """
        conn, _ = archiver
        df = await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "channel", "value"]
        assert not df.empty

    async def test_single_pv_channel_present(self, archiver):
        """Exactly the requested channel comes back, and it carries rows.

        Asserted as set equality rather than ``(df["channel"] == …).all()``:
        over an empty selection ``.all()`` is vacuously True, so the original
        form passed when ``get_data`` returned an empty frame for every query --
        the same antipattern this branch calls out in
        ``test_simulation_integration.py``. ``set(...) == {…}`` is empty-frame
        sensitive in both directions: it fails on a missing channel and on a
        stray extra one.
        """
        conn, _ = archiver
        df = await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)

        assert set(df["channel"]) == {"FAC/DEV/LOC/PROP"}

    async def test_single_pv_has_data(self, archiver):
        conn, _ = archiver
        df = await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)
        assert len(df) > 0


class TestGetDataTimeout:
    """``timeout`` reaches ``asyncio.wait_for``, with the configured default applied."""

    @staticmethod
    def _spy_wait_for(monkeypatch):
        """Capture the timeout handed to asyncio.wait_for, then run it for real."""
        from osprey.connectors.archiver import doocs_archiver_connector as mod

        seen = {}
        real = mod.asyncio.wait_for

        async def spy(awaitable, timeout):
            seen["timeout"] = timeout
            return await real(awaitable, timeout)

        monkeypatch.setattr(mod.asyncio, "wait_for", spy)
        return seen

    async def test_omitted_timeout_uses_configured_default(self, archiver, monkeypatch):
        """An omitted timeout must not mean "wait forever".

        ``get_data`` passes its ``timeout`` straight to ``asyncio.wait_for``,
        where ``None`` blocks indefinitely — so a caller that omitted the
        argument would hang on an unresponsive ENS with no way out. EPICS and
        MongoDB both fall back to a configured default; DOOCS had neither the
        fallback nor the attribute to fall back to. Asserted at the value that
        actually reaches ``wait_for``, so the default has to be *used*, not
        merely stored on the connector.
        """
        conn, _ = archiver
        seen = self._spy_wait_for(monkeypatch)

        await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)

        assert seen["timeout"] == 60

    async def test_configured_default_is_honored(self, archiver, monkeypatch):
        """A ``timeout`` in the connector config reaches ``wait_for``.

        Configured through ``connect()`` rather than by assigning ``_timeout``
        directly: that keeps the config key itself under test, so nothing is
        left asserting only that a private attribute was stored.
        """
        conn, _ = archiver
        await conn.connect({"timeout": 7})
        seen = self._spy_wait_for(monkeypatch)

        await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END)

        assert seen["timeout"] == 7

    async def test_explicit_timeout_wins_over_default(self, archiver, monkeypatch):
        conn, _ = archiver
        seen = self._spy_wait_for(monkeypatch)

        await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END, timeout=3)

        assert seen["timeout"] == 3

    async def test_explicit_zero_timeout_is_not_treated_as_unset(self, archiver, monkeypatch):
        """``timeout=0`` is a real request, not a missing argument.

        The same distinction the MongoDB connector was fixed for: a falsy check
        here would silently substitute the 60-second default.
        """
        conn, _ = archiver
        seen = self._spy_wait_for(monkeypatch)

        with pytest.raises(TimeoutError):
            await conn.get_data(["FAC/DEV/LOC/PROP"], _START, _END, timeout=0)

        assert seen["timeout"] == 0


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

    def test_no_caller_can_request_a_manufactured_grid(self):
        """``_read_history`` has no grid knob left to turn on.

        The removed ``max_points`` branch built an ``np.linspace`` grid and
        forward-filled real samples onto it with a zero-order hold, so every
        timestamp it returned was a grid position rather than an archived one --
        measured over an 8h window, only 2 of the 10,000 returned timestamps
        were real archived samples. That is exactly what this framework's
        "nothing is manufactured" contract forbids, so the branch could never be
        switched back on for ``get_data``; keeping the parameter only preserved
        a way to violate the contract. Asking for it is now a TypeError.
        """
        chunk = _make_raw_chunk(n=100)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        with pytest.raises(TypeError, match="max_points"):
            conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS, max_points=10)

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

    def test_raw_arrays_are_float64_built_from_the_time_and_value_fields(self):
        """Positions 0 and 3 of a DOOCS TTII entry are the time and the value.

        Pins the field selection, the float64 coercion and the sample order
        independently of the construction used to extract them, so the
        extraction can be rewritten (list comprehension, ``np.fromiter``, ...)
        without silently sliding onto the wrong tuple positions, losing the
        int→float coercion, or reordering the samples.
        """
        # Positions 1 and 2 carry values that would be obvious if selected.
        chunk = [(10.5, 111.0, 222.0, 5.5), (20.25, 333.0, 444.0, 6.25), (30.0, 555.0, 666.0, 7)]
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)

        assert out is not None
        assert out["time"].dtype == np.float64
        assert out["data"].dtype == np.float64
        assert out["time"].tolist() == [10.5, 20.25, 30.0]
        # The trailing int 7 must come back coerced to 7.0, not dropped.
        assert out["data"].tolist() == [5.5, 6.25, 7.0]

    def test_malformed_entry_is_reported_as_no_data(self):
        """An entry too short to carry a value must not half-build a series.

        The extraction raises ``IndexError`` on ``entry[3]``, which the method's
        own ``except Exception`` turns into "no data retrieved". Pinned so a
        rewritten extraction cannot substitute a different failure mode (a
        truncated array, a ``ValueError`` from a mis-sized buffer, ...).
        """
        chunk = [(10.5, 111.0, 222.0, 5.5), (20.25, 333.0)]
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)

        assert conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS) is None

    def test_every_archived_sample_is_returned(self):
        chunk = _make_raw_chunk(n=20)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)

        assert out is not None
        assert "time" in out and "data" in out
        assert len(out["time"]) == 20
        assert len(out["data"]) == 20

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
        # avg_window=600s → ~8 of the 50 samples fall inside the window
        out_smooth = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS, avg_window=600.0)

        assert out_smooth is not None
        # Smoothed step has values between 0 and 1 near the transition
        assert np.any((out_smooth["data"] > 0.0) & (out_smooth["data"] < 1.0))

    def test_smoothing_keeps_real_irregular_timestamps(self):
        """avg_window must not resample onto a uniform grid.

        The moving average used to require a constant dt, so supplying
        avg_window forced a ``np.linspace`` grid plus a zero-order hold. Every
        returned timestamp was then a grid position rather than an archived one,
        and every value had been forward-filled onto it — precisely what this
        framework's "nothing is manufactured" contract forbids, reintroduced
        through a config knob. This is the path a real DOOCS deployment with
        ``avg_window`` set actually takes.
        """
        # Deliberately bursty sampling: three clusters with long gaps between.
        # A uniform grid over this span looks nothing like the real timestamps.
        real_times = [
            _START_TS,
            _START_TS + 1,
            _START_TS + 2,
            _START_TS + 1800,
            _START_TS + 1801,
            _END_TS,
        ]
        chunk = [(float(t), 0, 0, float(i)) for i, t in enumerate(real_times)]
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS, avg_window=600.0)

        assert out is not None
        assert list(out["time"]) == [float(t) for t in real_times]
        assert len(out["data"]) == len(real_times)
        # Smoothing still happened: the isolated first cluster averages its own
        # three samples (0, 1, 2) rather than passing 0 through untouched.
        assert out["data"][0] != 0.0

    def test_smoothing_window_wider_than_span_keeps_arrays_aligned(self):
        """An avg_window wider than the queried span must not desynchronize the arrays.

        The moving average used ``np.convolve(..., mode="same")``, which returns
        ``max(len(data), win)`` elements — not ``len(data)``. So whenever the
        configured ``avg_window`` covered more samples than the query returned,
        ``data`` came back longer than ``time`` and ``get_data`` died building
        the per-channel Series ("Length of values does not match length of
        index"). A one-hour ``avg_window`` on a connector queried for a few
        minutes is an ordinary configuration, not a pathological one.
        """
        n = 20
        chunk = _make_raw_chunk(n=n)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        # The 20 samples span a full hour, so avg_window=36000 s (10 h) is far
        # wider than the queried span — every sample sees the whole series.
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS, avg_window=36000.0)

        assert out is not None
        assert len(out["data"]) == len(out["time"]) == n
        # min_periods=1 shrinks the window at the edges rather than padding with
        # zeros, so every point is a real average of real samples.
        assert np.all(np.isfinite(out["data"]))

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
        # developer's own machine. Finalizers run LIFO, so a bare
        # ``request.addfinalizer(time.tzset)`` would run *before*
        # monkeypatch's own teardown restores the TZ env var, re-reading the
        # still-patched "UTC" and leaving every later test in the session on
        # a UTC host zone. Undo the monkeypatch first, then tzset(), so the
        # C library re-reads the real, restored TZ.
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()

        def _restore_tz():
            monkeypatch.undo()
            time.tzset()

        request.addfinalizer(_restore_tz)

        conn, mock_d4py = archiver
        monkeypatch.setattr(
            "osprey.connectors.archiver._timerange.get_facility_timezone",
            lambda: ZoneInfo("America/Los_Angeles"),
        )

        captured = {}
        original = conn._read_history

        def _spy(address, start_time, end_time, avg_window=None):
            captured["start"] = start_time
            captured["end"] = end_time
            return original(address, start_time, end_time, avg_window)

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
        values = df.loc[df["channel"] == "FAC/DEV/LOC/P", "value"]
        assert len(values) <= 10
        assert not values.isna().any()


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

    async def test_multi_pv_mean_aggregates_true_within_bin_average_for_both_channels(self):
        """Each channel must carry its own true within-bin means, independently.

        Formerly a regression test for finding #1 reached via alignment:
        align_to_grid forward-filled onto one value per grid point, so resampling
        *after* alignment aggregated over an already-decimated series, same as the
        single-PV bug. align_to_grid is gone entirely now -- long_frame just
        assembles each channel's own aggregate_series output, so there is no
        shared grid left to forward-fill onto.
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
        # No trailing duplicate: with align_to_grid gone there is no query-window
        # grid point past the last real bin to forward-fill onto.
        expected = pytest.approx([4.5, 14.5, 24.5])
        assert df.loc[df["channel"] == "FAC/DEV/LOC/A", "value"].tolist() == expected
        assert df.loc[df["channel"] == "FAC/DEV/LOC/B", "value"].tolist() == expected

    async def test_multi_pv_count_returns_true_per_bin_sample_count_for_both_channels(self):
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
        assert df.loc[df["channel"] == "FAC/DEV/LOC/A", "value"].tolist() == [10, 10, 10]
        assert df.loc[df["channel"] == "FAC/DEV/LOC/B", "value"].tolist() == [10, 10, 10]

    async def test_multi_pv_raw_output_is_each_channels_own_decimated_series(self):
        """ "raw" now reaches _read_history with max_points=None (I4), so every
        real archived sample survives _read_history untouched; decimate_raw
        (inside aggregate_series) then keeps each 1s bin's own last real
        sample, at its own real timestamp. 30 real samples across 3 true 1s
        bins (10 samples/bin) decimate to exactly 3 rows per channel: the
        bin's last sample, value 9/19/29, each at its own real timestamp --
        never a bin-edge timestamp resample().agg("last") would fabricate.
        long_frame then just assembles each channel's own decimated series
        with no shared grid, so with identical input chunks the two channels
        must decimate identically.
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

        # The last real sample of each of the 3 true 1s bins: chunk indices
        # 9, 19, 29 (values 9.0, 19.0, 29.0), at their own real timestamps.
        expected_values = [chunk[9][3], chunk[19][3], chunk[29][3]]
        expected_times = pd.to_datetime(
            [chunk[9][0], chunk[19][0], chunk[29][0]], unit="s", utc=True
        )

        assert len(df) == 6  # exactly 3 rows per channel, no more, no fewer
        for channel in ("FAC/DEV/LOC/A", "FAC/DEV/LOC/B"):
            rows = df.loc[df["channel"] == channel]
            assert rows["value"].tolist() == pytest.approx(expected_values)
            assert list(rows["timestamp"]) == list(expected_times)

    async def test_raw_returns_only_real_archived_timestamps(self):
        """Every "raw"-mode timestamp must be a real archived sample, never a
        manufactured grid point (I4). Before this fix, "raw" passed
        ``max_points=num_points`` into ``_read_history``, which built a
        ``np.linspace`` grid and zero-order-held onto it -- measured over
        288,000 archived samples in an 8h window, only 2 of the 10,000
        returned timestamps were real archived ones. ``max_points=None`` now
        lets every real sample reach ``decimate_raw``, which only ever keeps
        samples that were actually archived.
        """
        chunk = self._dense_chunk(n_bins=3, samples_per_bin=10)
        mock_d4py = _make_doocs4py(chunk=chunk)
        archived_timestamps = set(pd.to_datetime([c[0] for c in chunk], unit="s", utc=True))

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
                # processing defaults to "raw"
            )
            await conn.disconnect()

        assert len(df) > 0
        assert set(df["timestamp"]) <= archived_timestamps

    @staticmethod
    def _mixed_cadence_mock(fast_chunk, slow_chunk):
        """A mock doocs4py wired for one dense and one sparse channel.

        The dense channel's chunk spans the whole window (its oldest sample is
        at ``start_ts``), so its single ``get()`` call terminates
        ``_read_history``'s pagination loop immediately, per the convention
        established for ``_two_pv_mock`` above. The sparse channel's one real
        sample sits mid-window, so its oldest-in-chunk timestamp is still
        *after* ``start_ts`` -- the loop does not yet know it has reached the
        start of the range, and pages once more. A second, empty result
        terminates it, matching real DOOCS pagination semantics.
        """
        mock_d4py = MagicMock()
        mock_d4py.__version__ = "2.0.0"
        mock_d4py.names.return_value = [("FACILITY", "XFEL")]
        r_fast = MagicMock()
        r_fast.value = fast_chunk
        r_slow = MagicMock()
        r_slow.value = slow_chunk
        r_slow_empty = MagicMock()
        r_slow_empty.value = []
        mock_d4py.get.side_effect = [r_fast, r_slow, r_slow_empty]
        return mock_d4py

    async def test_multi_pv_mixed_cadence_channels_aggregate_independently(self):
        """A 10 Hz and a 0.1 Hz channel queried together must each be aggregated over
        only their own samples -- the exact defect this redesign eliminates. Under
        the old grid, the common bin width was floored at the *sparsest* channel's
        own floor, so the dense channel's fine per-bin means got forward-filled
        onto that coarse grid instead of keeping their true within-bin average.
        """
        fast_chunk = self._dense_chunk(n_bins=10, samples_per_bin=10)  # 10 Hz, 10 s
        slow_chunk = [(_START_TS + 5.0, 0, 0, 500.0)]  # 0.1 Hz: one real sample
        mock_d4py = self._mixed_cadence_mock(fast_chunk, slow_chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAC/DEV/LOC/FAST", "FAC/DEV/LOC/SLOW"],
                start_date=_START,
                end_date=datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC),
                precision_ms=1000,
                processing="mean",
            )
            await conn.disconnect()

        fast_values = df.loc[df["channel"] == "FAC/DEV/LOC/FAST", "value"].tolist()
        slow_values = df.loc[df["channel"] == "FAC/DEV/LOC/SLOW", "value"].tolist()

        # FAST: 10 true per-bin means over its own 10 samples/bin, not deflated
        # to a single value borrowed via SLOW's 10 s cadence.
        assert fast_values == pytest.approx([4.5 + 10 * b for b in range(10)])
        # SLOW: its own single real sample, unaffected by FAST's cadence.
        assert slow_values == pytest.approx([500.0])

    async def test_multi_pv_mixed_cadence_dense_channel_count_not_deflated(self):
        """Same mixed-cadence scenario via "count": the dense channel's true
        per-bin sample count must not be deflated by the sparse channel's
        coarser cadence (the "count came back 10x low" symptom of the old bug).
        """
        fast_chunk = self._dense_chunk(n_bins=10, samples_per_bin=10)
        slow_chunk = [(_START_TS + 5.0, 0, 0, 500.0)]
        mock_d4py = self._mixed_cadence_mock(fast_chunk, slow_chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAC/DEV/LOC/FAST", "FAC/DEV/LOC/SLOW"],
                start_date=_START,
                end_date=datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC),
                precision_ms=1000,
                processing="count",
            )
            await conn.disconnect()

        fast_counts = df.loc[df["channel"] == "FAC/DEV/LOC/FAST", "value"].tolist()
        slow_counts = df.loc[df["channel"] == "FAC/DEV/LOC/SLOW", "value"].tolist()

        assert fast_counts == [10] * 10
        assert slow_counts == [1]

    async def test_multi_pv_sparse_dominated_mixed_cadence_defect_is_gone(self):
        """Reproduces the real DOOCS defect the earlier mixed-cadence tests above
        didn't actually exercise. A single SLOW sample never engaged
        ``_floor_bin_ms``'s multi-sample branch (it needs ``len(series) > 1`` to
        compute a gap at all), so the earlier fixture passed for the wrong
        reason -- it never widened SLOW's own floor past ``precision_ms``.

        Reconstructing the deleted DOOCS logic (``_floor_bin_ms`` +
        ``align_to_grid``) against a fixture with >=2 SLOW samples spaced wider
        than ``precision_ms`` confirms a genuine aggregation-content defect: with
        FAST truly at 10 Hz over 4 real 1s bins (count 10 each) and SLOW at just
        2 samples 5 s apart, ``_floor_bin_ms(SLOW)`` computes 5000ms, so
        ``grid_ms = max(1000, 5000) = 5000``. FAST's own correctly-resampled
        4-bin series (``[10, 10, 10, 10]``) then gets forward-filled onto that
        coarse shared grid and reindexed down to just 2 points, reproducing
        ``FAST: [10.0, 10.0]`` -- 2 of FAST's 4 real bins vanish, and the second
        surviving value is relabeled at t=5s, where FAST has zero real samples.
        That is a real content defect (not a shape/KeyError failure), driven
        entirely by SLOW's cadence dominating the shared bin width.
        """
        base_ts = _START.timestamp()
        fast_chunk = [
            (base_ts + i * 0.1, 0, 0, float(i)) for i in range(40)
        ]  # 10 Hz, 4 true 1s bins
        slow_chunk = [(base_ts, 0, 0, 500.0), (base_ts + 5.0, 0, 0, 501.0)]  # 5 s apart

        mock_d4py = MagicMock()
        mock_d4py.__version__ = "2.0.0"
        mock_d4py.names.return_value = [("FACILITY", "XFEL")]
        r_fast = MagicMock()
        r_fast.value = fast_chunk
        r_slow = MagicMock()
        r_slow.value = slow_chunk
        r_slow_empty = MagicMock()
        r_slow_empty.value = []
        # FAST's chunk spans the whole window -> 1 get() call. SLOW's first
        # sample sits at start_ts too, so its oldest-in-chunk also reaches
        # start_ts on the first call -> also 1 get() call; the spare empty
        # result is unused but keeps this robust to either outcome.
        mock_d4py.get.side_effect = [r_fast, r_slow, r_slow_empty]

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey.connectors.archiver.doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector()
            await conn.connect({})
            df = await conn.get_data(
                pv_list=["FAST", "SLOW"],
                start_date=_START,
                end_date=datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC),
                precision_ms=1000,
                processing="count",
            )
            await conn.disconnect()

        fast_counts = df.loc[df["channel"] == "FAST", "value"].tolist()
        slow_counts = df.loc[df["channel"] == "SLOW", "value"].tolist()

        # All 4 of FAST's true 1s bins survive, each with its true count of 10
        # -- not collapsed to 2 points by SLOW's much coarser cadence.
        assert fast_counts == [10, 10, 10, 10]
        assert slow_counts == [1, 1]
