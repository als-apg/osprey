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

from osprey_connectors.archiver import ArchiveCoverage
from osprey_connectors.errors import NaiveDatetimeError
from osprey_connectors.service_answers import BackendSpec, Unavailable

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


def _spec(settings: dict | None = None) -> BackendSpec:
    return BackendSpec(
        kind="archive",
        type="doocs_archiver",
        target="live",
        settings=settings or {},
        rehearsal=False,
    )


def _no_answer(answer: object, cause: str) -> str:
    """*answer*'s reason, after asserting it is ``Unavailable`` with *cause*."""
    assert isinstance(answer, Unavailable), answer
    assert answer.cause == cause, answer
    return answer.reason


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
    """DOOCSArchiverConnector built with a mock doocs4py."""
    mock_d4py = _make_doocs4py()

    with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
        from osprey_connectors.archiver._doocs_archiver_connector import (
            DOOCSArchiverConnector,
        )

        conn = DOOCSArchiverConnector(_spec())
        yield conn, mock_d4py
        await conn.aclose()


# --------------------------------------------------------------------------------------
# connect / disconnect
# --------------------------------------------------------------------------------------


class TestArchiverBuild:
    async def test_building_asks_the_ens_nothing(self):
        mock_d4py = _make_doocs4py()
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            assert conn._doocs4py is mock_d4py
            assert mock_d4py.names.call_count == 0
            await conn.aclose()

    async def test_build_raises_import_error_without_doocs4py(self):
        with patch.dict(sys.modules, {"doocs4py": None}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            with pytest.raises(ImportError, match="doocs4py"):
                DOOCSArchiverConnector(_spec())

    async def test_the_probe_names_an_ens_failure(self):
        mock_d4py = _make_doocs4py()
        mock_d4py.names.side_effect = RuntimeError("ENS down")
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            reason = _no_answer(await conn.probe(), "unreachable")
            assert reason == "DOOCS archiver connector failed to connect to the ENS."

    async def test_the_probe_counts_the_facilities_the_ens_lists(self, archiver):
        conn, mock_d4py = archiver
        mock_d4py.names.return_value = [("FACILITY", "XFEL"), ("FACILITY", "FLASH")]

        assert await conn.probe() == "the ENS lists 2 facilities"
        assert mock_d4py.names.call_args[0][0] == "*"

    async def test_build_stores_avg_window(self):
        mock_d4py = _make_doocs4py()
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec({"avg_window": 30}))
            assert conn._avg_window == 30
            await conn.aclose()


# --------------------------------------------------------------------------------------
# get_data — type validation / guard rails
# --------------------------------------------------------------------------------------


class TestConnectionSettings:
    """The shared connection keys: a login or CA doocs4py cannot send is refused."""

    @pytest.mark.parametrize(
        ("block", "key"),
        [
            ({"auth": {"token_env": "ARCHIVER_TOKEN"}}, "auth"),
            ({"tls": {"ca_bundle": "/etc/ssl/certs/site-ca.pem"}}, "tls"),
        ],
        ids=["auth", "tls"],
    )
    async def test_a_login_or_ca_is_refused_by_name(self, block, key):
        with patch.dict(sys.modules, {"doocs4py": _make_doocs4py()}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            with pytest.raises(ValueError, match=rf"`archiver\.settings\.{key}`.*doocs4py"):
                DOOCSArchiverConnector(_spec(block))

    async def test_a_flat_timeout_is_refused_naming_timeout_s(self):
        with patch.dict(sys.modules, {"doocs4py": _make_doocs4py()}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            with pytest.raises(ValueError, match=r"`archiver\.settings\.timeout_s`"):
                DOOCSArchiverConnector(_spec({"timeout": 7}))


class TestGetDataValidation:
    @pytest.mark.parametrize(
        ("bound", "start", "end"),
        [("start_at", "2026-01-01", _END), ("end_at", _START, 1234567890)],
        ids=["start", "end"],
    )
    async def test_non_datetime_bound_is_rejected(self, archiver, bound, start, end):
        conn, _ = archiver
        with pytest.raises(TypeError, match=bound):
            await conn.history(["ADDR"], start_at=start, end_at=end)


# --------------------------------------------------------------------------------------
# get_data — single PV
# --------------------------------------------------------------------------------------


class TestGetDataSinglePV:
    async def test_single_pv_returns_populated_long_frame(self, archiver):
        """The canonical long frame, carrying exactly the requested channel and
        every one of the fixture chunk's 20 archived samples."""
        conn, _ = archiver
        df = await conn.history(["FAC/DEV/LOC/PROP"], start_at=_START, end_at=_END)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "address", "value"]
        assert set(df["address"]) == {"FAC/DEV/LOC/PROP"}
        assert len(df) == 20

    async def test_unreadable_history_answers_failed(self, archiver):
        """A channel whose history cannot be read answers why, not an empty frame."""
        conn, mock_d4py = archiver
        mock_d4py.get.side_effect = RuntimeError("DOOCS error")

        answer = await conn.history(["FAC/DEV/LOC/PROP"], start_at=_START, end_at=_END)
        assert _no_answer(answer, "failed") == (
            "DOOCS archiver connector: Cannot read history for FAC/DEV/LOC/PROP"
        )


class TestGetDataTimeout:
    """``timeout`` reaches ``asyncio.wait_for``, with the configured default applied."""

    @staticmethod
    def _spy_wait_for(monkeypatch):
        """Capture the timeout handed to asyncio.wait_for, then run it for real."""
        from osprey_connectors.archiver import _doocs_archiver_connector as mod

        seen = {}
        real = mod.asyncio.wait_for

        async def spy(awaitable, timeout):
            seen["timeout"] = timeout
            return await real(awaitable, timeout)

        monkeypatch.setattr(mod.asyncio, "wait_for", spy)
        return seen

    async def test_omitted_timeout_uses_configured_default(self, archiver, monkeypatch):
        """An omitted timeout must not mean "wait forever".

        Regression: DOOCS passed ``timeout=None`` straight to
        ``asyncio.wait_for``, hanging indefinitely on an unresponsive ENS.
        """
        conn, _ = archiver
        seen = self._spy_wait_for(monkeypatch)

        await conn.history(["FAC/DEV/LOC/PROP"], start_at=_START, end_at=_END)

        assert seen["timeout"] == 60

    async def test_configured_default_is_honored(self, archiver, monkeypatch):
        """A ``timeout_s`` in the connector config reaches ``wait_for``."""
        _, mock_d4py = archiver
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec({"timeout_s": 7}))
        seen = self._spy_wait_for(monkeypatch)

        await conn.history(["FAC/DEV/LOC/PROP"], start_at=_START, end_at=_END)

        assert seen["timeout"] == 7

    async def test_explicit_timeout_wins_over_default(self, archiver, monkeypatch):
        conn, _ = archiver
        seen = self._spy_wait_for(monkeypatch)

        await conn.history(["FAC/DEV/LOC/PROP"], start_at=_START, end_at=_END, timeout_s=3)

        assert seen["timeout"] == 3

    async def test_explicit_zero_timeout_is_not_treated_as_unset(self, archiver, monkeypatch):
        """``timeout=0`` is a real request; a falsy check would substitute the default."""
        conn, _ = archiver
        seen = self._spy_wait_for(monkeypatch)

        answer = await conn.history(["FAC/DEV/LOC/PROP"], start_at=_START, end_at=_END, timeout_s=0)
        assert _no_answer(answer, "timeout") == "DOOCS archiver request timed out after 0s"

        assert seen["timeout"] == 0


# --------------------------------------------------------------------------------------
# _read_history — internal unit tests
# --------------------------------------------------------------------------------------


class TestReadHistory:
    def _make_connector_with_d4py(self, mock_d4py):
        """Return a DOOCSArchiverConnector built with *mock_d4py* as its doocs4py."""
        mock_d4py.__version__ = "2.0.0"
        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            conn._doocs4py = mock_d4py
            conn._avg_window = None
            return conn

    def test_no_caller_can_request_a_manufactured_grid(self):
        """``_read_history`` has no grid knob left to turn on.

        Regression: the removed ``max_points`` branch zero-order-held real
        samples onto an ``np.linspace`` grid of manufactured timestamps;
        asking for it is a TypeError.
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
        """Positions 0 and 3 of a DOOCS TTII entry are the time and the value."""
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
        """An entry too short to carry a value must not half-build a series."""
        chunk = [(10.5, 111.0, 222.0, 5.5), (20.25, 333.0)]
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)

        assert conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS) is None

    def test_every_archived_sample_is_returned_as_the_parallel_arrays_get_data_consumes(self):
        """``_read_history`` hands back every sample, as exactly ``time`` and
        ``data`` and nothing else."""
        chunk = _make_raw_chunk(n=20)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)

        assert out is not None
        assert set(out) == {"time", "data"}
        assert len(out["time"]) == len(out["data"]) == 20

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

        Regression: the moving average forced an ``np.linspace`` grid plus a
        zero-order hold, so no returned timestamp was a real archived one.
        """
        # Deliberately bursty sampling: three clusters with long gaps between.
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
        # Smoothing still happened: the first cluster averages its own samples.
        assert out["data"][0] != 0.0

    def test_smoothing_window_wider_than_span_keeps_arrays_aligned(self):
        """An avg_window wider than the queried span must not desynchronize the arrays.

        Regression: ``np.convolve(..., mode="same")`` returns
        ``max(len(data), win)`` elements, so ``data`` came back longer than
        ``time`` and ``get_data`` died building the per-channel Series.
        """
        n = 20
        chunk = _make_raw_chunk(n=n)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        # avg_window=36000 s (10 h) is far wider than the 1 h data span.
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS, avg_window=36000.0)

        assert out is not None
        assert len(out["data"]) == len(out["time"]) == n
        # min_periods=1 shrinks the window at the edges rather than zero-padding.
        assert np.all(np.isfinite(out["data"]))

    def test_pagination_breaks_when_oldest_timestamp_stops_advancing(self):
        """The failsafe: a chunk whose oldest timestamp equals the current stop
        must end pagination instead of re-requesting the same window forever."""
        mid_ts = (_START_TS + _END_TS) / 2
        mock_d4py = MagicMock()
        # The first chunk covers only the newer half of the window, so pagination
        # continues with current_stop = mid_ts; the archive then keeps answering
        # with that same oldest sample. No third result: without the failsafe the
        # loop would spin into StopIteration and report no data at all.
        mock_d4py.get.side_effect = [
            MagicMock(value=_make_raw_chunk(n=10, t_start=mid_ts, t_end=_END_TS)),
            MagicMock(value=[(mid_ts, 0, 0, 99.0)]),
        ]

        conn = self._make_connector_with_d4py(mock_d4py)
        out = conn._read_history("FAC/DEV/LOC/PROP", _START_TS, _END_TS)

        assert out is not None
        assert mock_d4py.get.call_count == 2
        # The duplicated mid-window sample is deduplicated, not double-counted.
        assert len(out["time"]) == 10
        assert np.all(np.diff(out["time"]) > 0)

    @pytest.mark.parametrize(
        "address", ["FAC/DEV/LOC/PROP", "FAC/DEV/LOC/PROP.HIST"], ids=["bare", "already_hist"]
    )
    def test_history_is_read_from_the_hist_address(self, address):
        """``.HIST`` is appended to a bare address and not doubled on one that has it."""
        chunk = _make_raw_chunk(n=5)
        mock_d4py = MagicMock()
        self._mock_get(mock_d4py, chunk)

        conn = self._make_connector_with_d4py(mock_d4py)
        conn._read_history(address, _START_TS, _END_TS)

        assert mock_d4py.Address.call_args[0][0] == "FAC/DEV/LOC/PROP.HIST"


# --------------------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------------------


class TestCoverage:
    async def test_archived_when_names_returns_result(self, archiver):
        conn, mock_d4py = archiver
        mock_d4py.names.return_value = [("FAC/DEV/LOC/PROP.HIST", "some_value")]

        coverage = await conn.coverage(["FAC/DEV/LOC/PROP"])

        # DOOCS states no bounds.
        assert coverage == {
            "FAC/DEV/LOC/PROP": ArchiveCoverage(address="FAC/DEV/LOC/PROP", is_archived=True)
        }

    async def test_not_archived_when_names_returns_empty(self, archiver):
        conn, mock_d4py = archiver
        mock_d4py.names.return_value = []

        coverage = await conn.coverage(["FAC/DEV/LOC/MISSING"])

        assert coverage["FAC/DEV/LOC/MISSING"].is_archived is False

    async def test_a_failed_lookup_is_unknown(self, archiver):
        """A lookup that raises answers ``None``: the archiver could not say, and why."""
        conn, mock_d4py = archiver
        mock_d4py.names.side_effect = RuntimeError("lookup failed")

        coverage = await conn.coverage(["FAC/DEV/LOC/PROP"])

        assert coverage["FAC/DEV/LOC/PROP"] == ArchiveCoverage(
            address="FAC/DEV/LOC/PROP",
            is_archived=None,
            unknown_reason="the DOOCS name service could not be asked: RuntimeError",
        )

    @pytest.mark.parametrize(
        "address", ["FAC/DEV/LOC/PROP", "FAC/DEV/LOC/PROP.HIST"], ids=["bare", "already_hist"]
    )
    async def test_lookup_uses_the_hist_address(self, archiver, address):
        """``.HIST`` is appended to a bare address and not doubled on one that has it."""
        conn, mock_d4py = archiver
        mock_d4py.names.return_value = []

        await conn.coverage([address])

        assert mock_d4py.names.call_args[0][0] == "FAC/DEV/LOC/PROP.HIST"


# --------------------------------------------------------------------------------------
# Closing / timezone
# --------------------------------------------------------------------------------------


class TestClose:
    async def test_double_close_is_safe_and_reads_go_on(self):
        """The archive holds nothing open: closing twice is safe, and it still answers."""
        mock_d4py = _make_doocs4py(names_result=[("FAC/DEV/LOC/P.HIST", "value")])

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            await conn.aclose()
            await conn.aclose()

        coverage = await conn.coverage(["FAC/DEV/LOC/P"])
        assert coverage["FAC/DEV/LOC/P"].is_archived is True


class TestQueryWindowTimezone:
    """Aware bounds reach the archive as their own instant; naive ones are refused."""

    async def test_a_naive_bound_is_refused(self, archiver):
        conn, _ = archiver
        with pytest.raises(NaiveDatetimeError, match="start_at"):
            await conn.history(["FAC/DEV/LOC/P"], start_at=datetime(2026, 7, 30), end_at=_END)

    async def test_aware_bounds_are_the_same_instant_on_any_host(
        self, archiver, monkeypatch, request
    ):
        from zoneinfo import ZoneInfo

        # Force the host zone to UTC so a reintroduced host-zone .timestamp()
        # bug fails on every machine, not only where the host zone differs
        # from the bounds' own zone.
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()

        def _restore_tz():
            # Undo the monkeypatch before tzset() so the C library re-reads
            # the restored TZ, not the patched one.
            monkeypatch.undo()
            time.tzset()

        request.addfinalizer(_restore_tz)

        conn, _ = archiver
        pacific = ZoneInfo("America/Los_Angeles")

        captured = {}
        original = conn._read_history

        def _spy(address, start_time, end_time, avg_window=None):
            captured["start"] = start_time
            captured["end"] = end_time
            return original(address, start_time, end_time, avg_window)

        conn._read_history = _spy

        await conn.history(
            ["FAC/DEV/LOC/P"],
            start_at=datetime(2026, 7, 30, 10, 0, 0, tzinfo=pacific),
            end_at=datetime(2026, 7, 30, 11, 0, 0, tzinfo=pacific),
        )

        # 10:00 PDT is 17:00 UTC.
        assert captured["start"] == datetime(2026, 7, 30, 17, 0, 0, tzinfo=UTC).timestamp()
        assert captured["end"] == datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC).timestamp()


class TestProcessingSparseData:
    """A non-raw processing mode must not manufacture samples DOOCS never recorded.

    An archive sparser than the requested precision_ms must not be upsampled
    onto a finer grid than the data has.
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
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            df = await conn.history(
                ["FAC/DEV/LOC/P"],
                start_at=start,
                end_at=end,
                precision_ms=1000,
                processing="mean",
            )
            await conn.aclose()

        # Naive 1 s resampling of ~1-hour-apart data would inflate this to
        # ~21,601 mostly-NaN rows.
        values = df.loc[df["address"] == "FAC/DEV/LOC/P", "value"]
        assert len(values) <= 10
        assert not values.isna().any()


class TestProcessingGenuineAggregation:
    """A non-raw processing mode must aggregate over the real archived samples,
    not a single zero-order-hold sample per bin.

    Regression: get_data passed ``num_points`` into ``_read_history`` for every
    mode, decimating before the resampler ever saw the data.
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

    @pytest.mark.parametrize(
        ("processing", "expected"),
        [("mean", [4.5, 14.5, 24.5]), ("count", [10, 10, 10])],
        ids=["mean", "count"],
    )
    async def test_multi_pv_aggregates_each_channel_over_its_own_bins(self, processing, expected):
        """Each channel must carry its own true within-bin aggregate, independently.

        Bin 0 holds values 0..9, bin 1 10..19, bin 2 20..29: means 4.5, 14.5,
        24.5, and 10 real samples per 1 s bin, on both PVs.
        """
        chunk = self._dense_chunk(n_bins=3, samples_per_bin=10)
        mock_d4py = self._two_pv_mock(chunk, chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            df = await conn.history(
                ["FAC/DEV/LOC/A", "FAC/DEV/LOC/B"],
                start_at=_START,
                end_at=datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                precision_ms=1000,
                processing=processing,
            )
            await conn.aclose()

        assert df.loc[df["address"] == "FAC/DEV/LOC/A", "value"].tolist() == pytest.approx(expected)
        assert df.loc[df["address"] == "FAC/DEV/LOC/B", "value"].tolist() == pytest.approx(expected)

    async def test_multi_pv_raw_output_is_each_channels_own_decimated_series(self):
        """Raw keeps each 1s bin's own last real sample, at its own real
        timestamp, identically for both channels."""
        chunk = self._dense_chunk(n_bins=3, samples_per_bin=10)
        mock_d4py = self._two_pv_mock(chunk, chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            df = await conn.history(
                ["FAC/DEV/LOC/A", "FAC/DEV/LOC/B"],
                start_at=_START,
                end_at=datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                precision_ms=1000,
                # processing defaults to "raw"
            )
            await conn.aclose()

        # The last real sample of each of the 3 true 1s bins: chunk indices
        # 9, 19, 29 (values 9.0, 19.0, 29.0), at their own real timestamps.
        expected_values = [chunk[9][3], chunk[19][3], chunk[29][3]]
        expected_times = pd.to_datetime(
            [chunk[9][0], chunk[19][0], chunk[29][0]], unit="s", utc=True
        )

        assert len(df) == 6  # exactly 3 rows per channel, no more, no fewer
        for channel in ("FAC/DEV/LOC/A", "FAC/DEV/LOC/B"):
            rows = df.loc[df["address"] == channel]
            assert rows["value"].tolist() == pytest.approx(expected_values)
            assert list(rows["timestamp"]) == list(expected_times)

    @staticmethod
    def _mixed_cadence_mock(fast_chunk, slow_chunk):
        """A mock doocs4py wired for one dense and one sparse channel.

        The dense chunk spans the whole window, so one ``get()`` ends its
        pagination; the sparse channel's sample sits mid-window, so it pages
        once more and the empty result terminates it.
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

    @pytest.mark.parametrize(
        ("processing", "fast_expected", "slow_expected"),
        [
            ("mean", [4.5 + 10 * b for b in range(10)], [500.0]),
            ("count", [10] * 10, [1]),
        ],
        ids=["mean", "count"],
    )
    async def test_multi_pv_mixed_cadence_channels_aggregate_independently(
        self, processing, fast_expected, slow_expected
    ):
        """A 10 Hz and a 0.1 Hz channel queried together must each be aggregated
        over only their own samples: FAST keeps its 10 true per-bin means, and its
        per-bin count is not deflated by SLOW's coarser cadence ("count came back
        10x low"); SLOW keeps its own single real sample.

        Regression: a shared grid would floor the bin width at the sparsest
        channel's cadence and forward-fill the dense channel onto it.
        """
        fast_chunk = self._dense_chunk(n_bins=10, samples_per_bin=10)  # 10 Hz, 10 s
        slow_chunk = [(_START_TS + 5.0, 0, 0, 500.0)]  # 0.1 Hz: one real sample
        mock_d4py = self._mixed_cadence_mock(fast_chunk, slow_chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            df = await conn.history(
                ["FAC/DEV/LOC/FAST", "FAC/DEV/LOC/SLOW"],
                start_at=_START,
                end_at=datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC),
                precision_ms=1000,
                processing=processing,
            )
            await conn.aclose()

        fast_values = df.loc[df["address"] == "FAC/DEV/LOC/FAST", "value"].tolist()
        slow_values = df.loc[df["address"] == "FAC/DEV/LOC/SLOW", "value"].tolist()

        assert fast_values == pytest.approx(fast_expected)
        assert slow_values == pytest.approx(slow_expected)

    async def test_multi_pv_sparse_dominated_mixed_cadence_defect_is_gone(self):
        """Mixed cadence with >=2 sparse samples spaced wider than precision_ms.

        Regression: ``_floor_bin_ms(SLOW)`` widened the shared grid to 5000 ms,
        so FAST's 4 real 1s bins were forward-filled down to 2 relabeled points.
        """
        base_ts = _START.timestamp()
        fast_chunk = [
            (base_ts + i * 0.1, 0, 0, float(i)) for i in range(40)
        ]  # 10 Hz, 4 true 1s bins
        slow_chunk = [(base_ts, 0, 0, 500.0), (base_ts + 5.0, 0, 0, 501.0)]  # 5 s apart

        # Both chunks reach start_ts on the first get(); the mock's spare empty
        # result keeps this robust to either pagination outcome.
        mock_d4py = self._mixed_cadence_mock(fast_chunk, slow_chunk)

        with patch.dict(sys.modules, {"doocs4py": mock_d4py}):
            from osprey_connectors.archiver._doocs_archiver_connector import (
                DOOCSArchiverConnector,
            )

            conn = DOOCSArchiverConnector(_spec())
            df = await conn.history(
                ["FAST", "SLOW"],
                start_at=_START,
                end_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC),
                precision_ms=1000,
                processing="count",
            )
            await conn.aclose()

        fast_counts = df.loc[df["address"] == "FAST", "value"].tolist()
        slow_counts = df.loc[df["address"] == "SLOW", "value"].tolist()

        # All 4 of FAST's true 1s bins survive with their true counts.
        assert fast_counts == [10, 10, 10, 10]
        assert slow_counts == [1, 1]
