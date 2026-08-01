"""Tests for the shared archiver time-range and processing helpers."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from osprey.connectors.archiver._timerange import (
    LONG_COLUMNS,
    PROCESSING_MODES,
    aggregate_series,
    decimate_raw,
    long_frame,
    require_datetime,
    resolve_processing,
    to_utc,
)


class TestRequireDatetime:
    """The shared query-bound guard every connector calls before ``to_utc``."""

    def test_two_datetimes_pass(self):
        require_datetime(datetime(2026, 7, 30), datetime(2026, 7, 31, tzinfo=UTC))

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            ("2026-07-30", datetime(2026, 7, 31), "start_date"),
            (datetime(2026, 7, 30), "2026-07-31", "end_date"),
            (None, datetime(2026, 7, 31), "start_date"),
            (datetime(2026, 7, 30), 1234567890, "end_date"),
        ],
    )
    def test_non_datetime_raises_naming_the_argument(self, start, end, expected):
        """The message must name which bound was wrong, not just that one was."""
        with pytest.raises(TypeError, match=f"{expected} must be a datetime object"):
            require_datetime(start, end)

    def test_start_is_reported_first_when_both_are_wrong(self):
        with pytest.raises(TypeError, match="start_date must be a datetime object"):
            require_datetime("a", "b")

    def test_date_is_not_accepted_as_datetime(self):
        """``datetime.date`` has no ``tzinfo``, so ``to_utc`` would fail on it."""
        from datetime import date

        with pytest.raises(TypeError, match="start_date must be a datetime object"):
            require_datetime(date(2026, 7, 30), datetime(2026, 7, 31))


class TestToUtc:
    def test_aware_non_utc_converts(self):
        dt = datetime(2026, 7, 30, 11, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        assert to_utc(dt) == datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC)

    def test_aware_utc_is_unchanged(self):
        dt = datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC)
        assert to_utc(dt) == dt

    def test_naive_reads_as_facility_zone(self, monkeypatch):
        monkeypatch.setattr(
            "osprey.connectors.archiver._timerange.get_facility_timezone",
            lambda: ZoneInfo("America/Los_Angeles"),
        )
        assert to_utc(datetime(2026, 7, 30, 11, 0, 0)) == datetime(
            2026, 7, 30, 18, 0, 0, tzinfo=UTC
        )

    def test_naive_with_no_configured_zone_is_utc(self):
        # tests/conftest.py clears CONFIG_FILE, so get_facility_timezone() -> UTC
        assert to_utc(datetime(2026, 7, 30, 18, 0, 0)) == datetime(
            2026, 7, 30, 18, 0, 0, tzinfo=UTC
        )


class TestResolveProcessing:
    @pytest.mark.parametrize(
        "mode,operator,agg",
        [
            # "raw" renders a server-side operator (the appliance's lastSample
            # is faithful — it returns a real archived sample) but NO pandas
            # aggregation: see test_raw_renders_no_pandas_aggregation.
            ("raw", "lastSample_60", None),
            ("mean", "mean_60", "mean"),
            ("min", "min_60", "min"),
            ("max", "max_60", "max"),
            ("median", "median_60", "median"),
            ("std", "std_60", "std"),
            ("count", "count_60", "count"),
        ],
    )
    def test_each_mode_renders_for_both_backends(self, mode, operator, agg):
        p = resolve_processing(mode, 60_000)
        assert p.epics_operator == operator
        assert p.pandas_agg == agg

    def test_raw_renders_no_pandas_aggregation(self):
        # There is no resample aggregation that does what "raw" means. The
        # closest, `agg("last")`, returns the bin's last value but relabels it
        # at the bin's leading edge — a timestamp nothing was recorded at.
        # decimate_raw exists precisely to avoid that, so "raw" must render no
        # pandas aggregation at all: a value here is a standing invitation for
        # a future caller to hand it to resample() and manufacture a timestamp.
        assert resolve_processing("raw", 60_000).pandas_agg is None

    def test_every_aggregate_mode_renders_a_pandas_aggregation(self):
        for mode in PROCESSING_MODES:
            if mode == "raw":
                continue
            assert resolve_processing(mode, 1000).pandas_agg

    def test_raw_at_full_resolution_has_no_operator(self):
        assert resolve_processing("raw", 0).epics_operator is None

    @pytest.mark.parametrize("precision_ms", [0, 1, 500, 1000, 7000, 60_000])
    def test_the_resolved_processing_carries_its_own_bin_width(self, precision_ms):
        # The bin width travels with the mode it was resolved against. Every
        # backend used to keep a local copy and hand it to aggregate_series
        # separately, so three call sites each re-asserted by hand that the
        # width resolved with is the width aggregated with. Nothing checked it.
        mode = "raw" if precision_ms <= 0 else "mean"
        assert resolve_processing(mode, precision_ms).precision_ms == precision_ms

    def test_unknown_mode_lists_the_valid_set(self):
        with pytest.raises(ValueError, match="Unknown processing mode"):
            resolve_processing("p99", 1000)

    def test_aggregate_without_bin_width_is_rejected(self):
        with pytest.raises(ValueError, match="requires precision_ms > 0"):
            resolve_processing("mean", 0)


class TestLongFrame:
    def test_columns_and_dtypes(self):
        idx = pd.to_datetime([0], unit="s", utc=True)
        series = {"PV": pd.Series([1.0], index=idx, name="PV")}
        frame = long_frame(series)
        assert list(frame.columns) == list(LONG_COLUMNS)
        assert frame["timestamp"].dtype == "datetime64[ns, UTC]"
        assert frame["channel"].dtype == "str"
        assert frame["value"].dtype == np.float64

    def test_two_channels_of_different_lengths_produce_one_row_per_sample(self):
        idx_a = pd.to_datetime([0, 60, 120], unit="s", utc=True)
        idx_b = pd.to_datetime([30], unit="s", utc=True)
        series = {
            "B:PV": pd.Series([2.0], index=idx_b, name="B:PV"),
            "A:PV": pd.Series([1.0, 2.0, 3.0], index=idx_a, name="A:PV"),
        }
        frame = long_frame(series)
        assert len(frame) == 4
        # Sorted by channel then timestamp: all A:PV rows (in timestamp order),
        # then all B:PV rows.
        assert frame["channel"].tolist() == ["A:PV", "A:PV", "A:PV", "B:PV"]
        assert frame["value"].tolist() == [1.0, 2.0, 3.0, 2.0]
        assert frame["timestamp"].tolist() == list(idx_a) + list(idx_b)

    def test_empty_channel_contributes_no_rows(self):
        idx = pd.to_datetime([0], unit="s", utc=True)
        series = {
            "A:PV": pd.Series([1.0], index=idx, name="A:PV"),
            "EMPTY:PV": pd.Series(dtype=float, name="EMPTY:PV"),
        }
        frame = long_frame(series)
        assert frame["channel"].tolist() == ["A:PV"]
        assert len(frame) == 1

    def test_empty_mapping_yields_the_typed_empty_frame(self):
        frame = long_frame({})
        assert list(frame.columns) == list(LONG_COLUMNS)
        assert len(frame) == 0
        assert frame["timestamp"].dtype == "datetime64[ns, UTC]"
        assert frame["channel"].dtype == "str"
        assert frame["value"].dtype == np.float64

    def test_all_channels_empty_yields_the_typed_empty_frame(self):
        series = {
            "A:PV": pd.Series(dtype=float, name="A:PV"),
            "B:PV": pd.Series(dtype=float, name="B:PV"),
        }
        frame = long_frame(series)
        assert list(frame.columns) == list(LONG_COLUMNS)
        assert len(frame) == 0
        assert frame["timestamp"].dtype == "datetime64[ns, UTC]"
        assert frame["channel"].dtype == "str"
        assert frame["value"].dtype == np.float64

    def test_non_numeric_channel_is_not_coerced_or_dropped(self):
        # An enum/status PV (e.g. machine mode) archives string values. value
        # is not dtype-constrained: nothing here coerces or drops it.
        idx = pd.to_datetime([0, 60], unit="s", utc=True)
        series = {"T:MODE": pd.Series(["CW", "STANDBY"], index=idx, name="T:MODE")}
        frame = long_frame(series)
        assert len(frame) == 2
        assert frame["value"].tolist() == ["CW", "STANDBY"]

    def test_mixed_numeric_and_string_channels_both_present(self):
        # Combining a numeric channel with a non-numeric one must not drop
        # either or force one to fit the other's dtype: pandas' ordinary
        # concat type promotion takes the value column to object, and both
        # channels' real values survive unchanged.
        idx = pd.to_datetime([0], unit="s", utc=True)
        series = {
            "A:NUM": pd.Series([1.5], index=idx, name="A:NUM"),
            "B:STR": pd.Series(["CW"], index=idx, name="B:STR"),
        }
        frame = long_frame(series)
        assert len(frame) == 2
        assert frame["value"].dtype == object
        num_row = frame.loc[frame["channel"] == "A:NUM", "value"]
        str_row = frame.loc[frame["channel"] == "B:STR", "value"]
        assert num_row.tolist() == [1.5]
        assert str_row.tolist() == ["CW"]


class TestAggregateSeries:
    def test_raw_at_full_resolution_returns_the_series_unchanged(self):
        # precision_ms <= 0 means full resolution: every real sample, untouched.
        # Raw *with* a positive precision_ms decimates instead -- see
        # test_raw_with_precision_ms_decimates_to_last_real_sample_per_bin below.
        idx = pd.to_datetime([0, 5, 9], unit="s", utc=True)
        s = pd.Series([1.0, 2.0, 3.0], index=idx, name="PV")
        resolved = resolve_processing("raw", 0)
        out = aggregate_series(s, resolved)
        pd.testing.assert_series_equal(out, s)

    def test_raw_with_precision_ms_decimates_to_last_real_sample_per_bin(self):
        # Six 1s-apart samples, precision_ms=2000 -> three 2s bins. Each kept row
        # must be the bin's own real last sample -- at its own real timestamp,
        # not the bin edge resample().agg("last") would relabel it at.
        idx = pd.to_datetime([0, 1, 2, 3, 4, 5], unit="s", utc=True)
        s = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="PV")
        resolved = resolve_processing("raw", 2000)
        out = aggregate_series(s, resolved)
        assert out.tolist() == [1.0, 3.0, 5.0]
        assert list(out.index) == [idx[1], idx[3], idx[5]]

    def test_raw_on_non_numeric_channel_still_decimates(self):
        # decimate_raw is dtype-agnostic: an enum/status channel round-trips
        # under processing="raw" even with a real precision_ms bin width.
        idx = pd.to_datetime([0, 1, 2], unit="s", utc=True)
        s = pd.Series(["CW", "CW", "STANDBY"], index=idx, name="T:MODE")
        resolved = resolve_processing("raw", 2000)
        out = aggregate_series(s, resolved)
        assert out.tolist() == ["CW", "STANDBY"]

    def test_empty_series_returns_unchanged_for_every_mode(self):
        # A channel that matched zero samples must not be rejected as
        # "non-numeric" (it has no values to inspect) for any mode, raw
        # included -- one empty requested channel must not fail every other
        # channel's aggregation in the same query.
        s = pd.Series([], index=pd.to_datetime([], utc=True), dtype=object, name="PV")
        for mode in PROCESSING_MODES:
            resolved = resolve_processing(mode, 1000)
            out = aggregate_series(s, resolved)
            assert out.empty
            assert not out.index.has_duplicates

    def test_mean_over_ten_samples_per_second_returns_true_per_bin_means(self):
        # 20 samples at 100ms spacing -> two 1s bins of 10 samples each.
        idx = pd.to_datetime(np.arange(20) * 100, unit="ms", utc=True)
        values = np.arange(20, dtype=float)
        s = pd.Series(values, index=idx, name="PV")
        resolved = resolve_processing("mean", 1000)
        out = aggregate_series(s, resolved)
        assert len(out) == 2
        assert out.iloc[0] == pytest.approx(values[:10].mean())
        assert out.iloc[1] == pytest.approx(values[10:].mean())

    def test_count_returns_true_per_bin_counts(self):
        # First bin gets 3 samples, second bin gets 1 sample.
        idx = pd.to_datetime([0, 100, 900, 1500], unit="ms", utc=True)
        s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="PV")
        resolved = resolve_processing("count", 1000)
        out = aggregate_series(s, resolved)
        assert out.tolist() == [3, 1]

    def test_sparse_channel_does_not_upsample_onto_the_grid(self):
        # 6 samples one hour apart, resampled at precision_ms=1000: without the
        # empty-bin drop this would emit 6+ hours worth of 1s bins (3,600+ rows).
        idx = pd.to_datetime(np.arange(6) * 3600, unit="s", utc=True)
        s = pd.Series(np.arange(6, dtype=float), index=idx, name="PV")
        resolved = resolve_processing("mean", 1000)
        out = aggregate_series(s, resolved)
        assert len(out) == 6
        assert out.tolist() == list(range(6))

    @pytest.mark.parametrize(
        "precision_ms,expected",
        [
            # Sub-second bins: each 1s-apart sample lands in its own bin.
            (100, [1.0, 2.0, 3.0, 4.0]),
            (500, [1.0, 2.0, 3.0, 4.0]),
            # Non-whole-second bins: the [0, 1.5)s bin holds both t=0 and t=1.
            (1500, [1.5, 3.0, 4.0]),
        ],
    )
    def test_non_nanosecond_index_resolution_does_not_crash(self, precision_ms, expected):
        # A source built with unit="s" (e.g. DOOCS) yields a datetime64[s, UTC]
        # index, not datetime64[ns, UTC]. Sub-second (100, 500) and
        # non-whole-second (1500) precision_ms used to raise ZeroDivisionError or
        # ValueError from pandas because resample() assumed nanosecond bins could
        # divide evenly into the index's native (coarser) unit.
        idx = pd.to_datetime([0, 1, 2, 3], unit="s", utc=True)
        assert idx.dtype == "datetime64[s, UTC]"
        s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="PV")
        resolved = resolve_processing("mean", precision_ms)
        out = aggregate_series(s, resolved)
        assert out.index.dtype == "datetime64[ns, UTC]"
        assert out.tolist() == expected

    def test_raw_mode_is_always_valid_for_non_numeric_channel(self):
        # raw performs no aggregation, so a non-numeric (enum/status) channel
        # is never rejected under processing="raw".
        idx = pd.to_datetime([0, 60], unit="s", utc=True)
        s = pd.Series(["CW", "STANDBY"], index=idx, name="T:MODE")
        resolved = resolve_processing("raw", 0)
        out = aggregate_series(s, resolved)
        pd.testing.assert_series_equal(out, s)

    def test_non_raw_mode_on_non_numeric_channel_raises_naming_channel_and_mode(self):
        idx = pd.to_datetime([0, 60], unit="s", utc=True)
        s = pd.Series(["CW", "STANDBY"], index=idx, name="T:MODE")
        resolved = resolve_processing("mean", 60_000)
        with pytest.raises(ValueError, match="'T:MODE'") as exc_info:
            aggregate_series(s, resolved)
        assert "'mean'" in str(exc_info.value)

    def test_non_raw_mode_on_unnamed_non_numeric_channel_names_it_as_unset(self):
        # s.name is the only way to identify the channel in the error message;
        # when it's unset the message must say so, not print the literal
        # string "None".
        idx = pd.to_datetime([0, 60], unit="s", utc=True)
        s = pd.Series(["CW", "STANDBY"], index=idx)
        assert s.name is None
        resolved = resolve_processing("mean", 60_000)
        with pytest.raises(ValueError) as exc_info:
            aggregate_series(s, resolved)
        message = str(exc_info.value)
        assert "None" not in message
        assert "unnamed" in message.lower()

    def test_std_on_single_sample_bins_emits_nan(self):
        # aggregate_series keeps a bin whenever it has at least one sample;
        # std (and any other spread statistic) over a single-sample bin is
        # mathematically undefined and pandas emits NaN for it. Pinned here so
        # this is a known, tested property rather than a surprise downstream.
        idx = pd.to_datetime([0, 1, 2], unit="s", utc=True)
        s = pd.Series([1.0, 2.0, 3.0], index=idx, name="PV")
        resolved = resolve_processing("std", 1000)
        out = aggregate_series(s, resolved)
        assert len(out) == 3
        assert out.isna().all()


class TestDecimateRaw:
    def test_non_positive_precision_ms_returns_every_sample_unchanged(self):
        idx = pd.to_datetime([0, 1, 2], unit="s", utc=True)
        s = pd.Series([1.0, 2.0, 3.0], index=idx, name="PV")
        pd.testing.assert_series_equal(decimate_raw(s, 0), s)
        pd.testing.assert_series_equal(decimate_raw(s, -1), s)

    def test_empty_series_returned_unchanged(self):
        s = pd.Series([], index=pd.to_datetime([], utc=True), dtype=float, name="PV")
        out = decimate_raw(s, 1000)
        assert out.empty

    def test_keeps_the_last_real_sample_with_its_own_timestamp_not_the_bin_edge(self):
        # Twenty samples at 100ms spacing (0..1900ms) -> two 1s bins of 10
        # samples each. The kept row must be the real 10th/20th sample's own
        # timestamp, never the bin-start edge (0s, 1s) resample().agg("last")
        # would relabel it at.
        idx = pd.to_datetime(np.arange(20) * 100, unit="ms", utc=True)
        s = pd.Series(np.arange(20, dtype=float), index=idx, name="PV")
        out = decimate_raw(s, 1000)
        assert out.tolist() == [9.0, 19.0]
        assert list(out.index) == [idx[9], idx[19]]
        assert idx[9] != idx[9].floor("1000ms")  # the real timestamp, not the edge

    def test_sparse_channel_is_left_bit_identical(self):
        # Already at most one real sample per bin: nothing to drop.
        idx = pd.to_datetime(np.arange(6) * 3600, unit="s", utc=True)
        s = pd.Series(np.arange(6, dtype=float), index=idx, name="PV")
        pd.testing.assert_series_equal(decimate_raw(s, 1000), s)

    def test_dtype_agnostic_enum_channel_round_trips(self):
        # Bins at [0, 2)s and [2, 4)s: t=0,1 share the first bin (keep the
        # later "CW" at t=1), t=2 is alone in the second bin.
        idx = pd.to_datetime([0, 1, 2], unit="s", utc=True)
        s = pd.Series(["CW", "CW", "STANDBY"], index=idx, name="T:MODE")
        out = decimate_raw(s, 2000)
        assert out.tolist() == ["CW", "STANDBY"]
        assert list(out.index) == [idx[1], idx[2]]


class TestBinOriginIsSharedByEveryMode:
    """``raw`` and the aggregate modes must bin on the *same* lattice.

    ``decimate_raw`` used to bin with ``DatetimeIndex.floor``, which is
    epoch-anchored, while ``aggregate_series`` bins with ``resample``, which is
    anchored at the start of the first sample's day. The two agree for every
    width that divides a day and disagree otherwise, so an operator comparing
    ``processing="raw"`` against ``processing="mean"`` at, say, 7000 ms got two
    grids offset from each other. Nothing was manufactured either way; the bins
    simply did not line up.
    """

    # 2026-07-31 is chosen deliberately. 86_400_000 ms is not a multiple of
    # 7000, so which side of the epoch lattice UTC midnight falls on is
    # date-dependent: on 2026-07-30 midnight happens to sit exactly on it and
    # the two conventions coincide even at 7000 ms, while on 2026-07-31 they
    # are 1 s apart. Pin the date or this test silently stops testing anything.
    BASE = "2026-07-31T10:00:00Z"

    def _samples(self):
        base = pd.Timestamp(self.BASE)
        idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=i) for i in range(40)])
        return pd.Series(np.arange(40, dtype=float), index=idx, name="PV")

    @pytest.mark.parametrize(
        "precision_ms",
        [
            # 7000 ms does not divide a day: the divergent case. This is the
            # only width family that can catch the defect.
            7000,
            # 1000 ms divides a day: the two conventions coincide here, which
            # is why every pre-existing test (all whole-second divisors) passed
            # while the lattices were different. Kept so a "fix" that merely
            # moves the divergence somewhere else still fails.
            1000,
        ],
    )
    def test_raw_bins_on_the_same_lattice_as_the_aggregate_modes(self, precision_ms):
        s = self._samples()
        width = pd.Timedelta(milliseconds=precision_ms)

        # aggregate_series indexes each bin at its own leading edge, so its
        # index *is* the aggregate lattice.
        edges = aggregate_series(s, resolve_processing("mean", precision_ms)).index
        kept = decimate_raw(s, precision_ms)

        assert len(kept) == len(edges)
        for edge, (timestamp, value) in zip(edges, kept.items(), strict=True):
            in_bin = s[(s.index >= edge) & (s.index < edge + width)]
            # The kept row must be the last REAL sample of the SAME bin the
            # aggregate modes used -- at its own real timestamp, never the edge.
            assert timestamp == in_bin.index[-1]
            assert value == in_bin.iloc[-1]
