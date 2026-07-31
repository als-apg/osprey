"""Tests for the timeseries downsampling and frame-extraction helpers.

``lttb_downsample`` is the Largest-Triangle-Three-Buckets reducer shared by the
Artifact Gallery and MCP tools. Two contracts matter most: it always keeps the
first and last points and returns exactly ``max_points`` when it downsamples,
and it applies the selected indices to the ORIGINAL data so ``None`` gap markers
(archiver disconnects / IOC reboots) survive into the output even though the
triangle-area math runs on a zero-filled working copy.
"""

from __future__ import annotations

from osprey.utils.timeseries import (
    extract_channel_series,
    is_numeric_channel,
    lttb_downsample,
    lttb_downsample_channel,
)


class TestLttbDownsampleIdentity:
    """Cases where no reduction happens and the inputs pass through."""

    def test_returns_input_when_n_below_max_points(self):
        index = [0, 1, 2]
        data = [[0.0], [1.0], [2.0]]
        out_index, out_data = lttb_downsample(index, data, max_points=10)
        assert out_index is index
        assert out_data is data

    def test_returns_input_when_n_equals_max_points(self):
        index = [0, 1, 2, 3]
        data = [[float(i)] for i in range(4)]
        out_index, out_data = lttb_downsample(index, data, max_points=4)
        assert out_index is index
        assert out_data is data

    def test_returns_input_when_max_points_below_three(self):
        """LTTB needs at least first/last plus one bucket; <3 is a no-op."""
        index = list(range(100))
        data = [[float(i)] for i in index]
        out_index, out_data = lttb_downsample(index, data, max_points=2)
        assert out_index is index
        assert out_data is data


class TestLttbDownsampleReduction:
    def _ramp(self, n):
        index = list(range(n))
        data = [[float(i)] for i in index]
        return index, data

    def test_reduces_to_exactly_max_points(self):
        index, data = self._ramp(1000)
        out_index, out_data = lttb_downsample(index, data, max_points=50)
        assert len(out_index) == 50
        assert len(out_data) == 50

    def test_first_and_last_points_always_kept(self):
        index, data = self._ramp(1000)
        out_index, out_data = lttb_downsample(index, data, max_points=25)
        assert out_index[0] == index[0]
        assert out_index[-1] == index[-1]
        assert out_data[0] == data[0]
        assert out_data[-1] == data[-1]

    def test_min_max_points_of_three(self):
        index, data = self._ramp(500)
        out_index, out_data = lttb_downsample(index, data, max_points=3)
        assert len(out_index) == 3
        assert out_index[0] == 0
        assert out_index[-1] == 499

    def test_output_index_is_ordered_subset_of_input(self):
        index, data = self._ramp(300)
        out_index, _ = lttb_downsample(index, data, max_points=20)
        assert out_index == sorted(out_index)
        assert set(out_index) <= set(index)

    def test_preserves_all_columns(self):
        n = 400
        index = list(range(n))
        # Three channels; first column drives selection, all are sliced together.
        data = [[float(i), float(i) * 2, float(i) * 3] for i in index]
        _, out_data = lttb_downsample(index, data, max_points=30)
        assert all(len(row) == 3 for row in out_data)
        # Every kept row is an original row (columns stay aligned on one x-axis).
        for row in out_data:
            assert row[1] == row[0] * 2
            assert row[2] == row[0] * 3

    def test_dominant_spike_is_selected(self):
        """The whole point of LTTB: a sharp feature survives downsampling."""
        n = 200
        index = list(range(n))
        data = [[0.0] for _ in index]
        data[100] = [1000.0]  # single large spike
        out_index, out_data = lttb_downsample(index, data, max_points=10)
        assert 100 in out_index
        assert [1000.0] in out_data

    def test_none_gaps_preserved_in_output(self):
        """Selected indices apply to the ORIGINAL data, so ``None`` survives.

        The triangle-area math runs on a zero-filled working copy, but the
        emitted rows are the untouched originals — an archiver gap marker in a
        channel column must not be silenced to 0.0.
        """
        n = 300
        index = list(range(n))
        # First column is a ramp (drives selection); second column is all None.
        data = [[float(i), None] for i in index]
        _, out_data = lttb_downsample(index, data, max_points=20)
        assert all(row[1] is None for row in out_data)

    def test_returns_tuple(self):
        index, data = self._ramp(50)
        result = lttb_downsample(index, data, max_points=10)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestExtractChannelSeries:
    """Normalizes all three artifact timeseries layouts to per-channel series.

    This is the sole extractor. It replaced a split-orient-only predecessor
    that had no branch for the long-format ``{"query": ..., "series": ...}``
    payload and returned it unchanged, leaving downstream readers with empty
    ``columns``/``index``/``data`` instead of the real per-channel arrays.
    """

    def test_new_series_layout_returned_as_is(self):
        series = {
            "PV:A": {"timestamps": ["t0", "t1"], "values": [1.0, 2.0]},
            "PV:B": {"timestamps": ["t0"], "values": ["CW"]},
        }
        query = {"channels": ["PV:A", "PV:B"], "start_time": "2026-01-01"}
        raw = {"query": query, "series": series}

        out_series, out_query = extract_channel_series(raw)

        assert out_series == series
        assert out_query == query

    def test_new_series_layout_under_data_wrapper(self):
        """The new layout may also arrive wrapped in a 'data' envelope."""
        series = {"PV:A": {"timestamps": ["t0"], "values": [1.0]}}
        raw = {"data": {"query": {"k": "v"}, "series": series}}

        out_series, out_query = extract_channel_series(raw)

        assert out_series == series
        assert out_query == {"k": "v"}

    def test_archiver_split_orient_layout_transposed_per_channel(self):
        frame = {
            "columns": ["PV:A", "PV:B"],
            "index": ["t0", "t1", "t2"],
            "data": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        }
        raw = {"dataframe": frame, "query": {"channels": ["PV:A", "PV:B"]}}

        series, query = extract_channel_series(raw)

        assert series == {
            "PV:A": {"timestamps": ["t0", "t1", "t2"], "values": [1.0, 2.0, 3.0]},
            "PV:B": {"timestamps": ["t0", "t1", "t2"], "values": [10.0, 20.0, 30.0]},
        }
        assert query == {"channels": ["PV:A", "PV:B"]}

    def test_flat_layout_transposed_per_channel(self):
        """Flat layout (no 'dataframe' wrapper) transposes the same way; query defaults empty."""
        payload = {
            "columns": ["PV:A"],
            "index": ["t0", "t1"],
            "data": [[5.0], [6.0]],
        }
        raw = {"data": payload}

        series, query = extract_channel_series(raw)

        assert series == {"PV:A": {"timestamps": ["t0", "t1"], "values": [5.0, 6.0]}}
        assert query == {}

    def test_drops_nulls_when_transposing_wide_layout(self):
        """A None in a wide-frame cell means that channel had no sample at that
        shared timestamp -- it must be dropped (not kept as a gap marker),
        unlike ``lttb_downsample`` which preserves None gaps in a channel's
        own column.
        """
        frame = {
            "columns": ["PV:A", "PV:B"],
            "index": ["t0", "t1", "t2"],
            "data": [[1.0, None], [None, 20.0], [3.0, 30.0]],
        }
        raw = {"dataframe": frame, "query": {}}

        series, _query = extract_channel_series(raw)

        assert series["PV:A"] == {"timestamps": ["t0", "t2"], "values": [1.0, 3.0]}
        assert series["PV:B"] == {"timestamps": ["t1", "t2"], "values": [20.0, 30.0]}

    def test_empty_wide_layout_produces_empty_series_per_declared_channel(self):
        frame = {"columns": ["PV:A", "PV:B"], "index": [], "data": []}
        raw = {"dataframe": frame, "query": {}}

        series, _query = extract_channel_series(raw)

        assert series == {
            "PV:A": {"timestamps": [], "values": []},
            "PV:B": {"timestamps": [], "values": []},
        }


class TestIsNumericChannel:
    """The dtype check callers (web API, archiver_downsample) use to report a
    channel's numeric-ness -- the same check ``lttb_downsample_channel`` runs
    internally to decide whether LTTB applies.
    """

    def test_all_floats_is_numeric(self):
        assert is_numeric_channel([1.0, 2.0, 3.0]) is True

    def test_all_none_is_numeric(self):
        assert is_numeric_channel([None, None]) is True

    def test_floats_with_none_gaps_is_numeric(self):
        assert is_numeric_channel([1.0, None, 3.0]) is True

    def test_any_string_makes_it_non_numeric(self):
        assert is_numeric_channel(["CW", "STANDBY"]) is False

    def test_mixed_numeric_and_string_is_non_numeric(self):
        """A single non-numeric sample marks the whole channel non-numeric."""
        assert is_numeric_channel([1.0, "FAULT", 3.0]) is False

    def test_empty_is_numeric(self):
        assert is_numeric_channel([]) is True


class TestLttbDownsampleChannel:
    """Per-channel LTTB variant: each channel has its own timestamps, so there
    is no shared x-axis to slice multiple columns against (contrast with
    ``lttb_downsample``, whose docstring documents that shared-axis slicing).
    """

    def test_returns_input_when_n_below_max_points(self):
        timestamps = ["t0", "t1", "t2"]
        values = [0.0, 1.0, 2.0]
        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=10)
        assert out_ts is timestamps
        assert out_vals is values

    def test_reduces_to_at_most_max_points(self):
        n = 500
        timestamps = [f"t{i}" for i in range(n)]
        values = [float(i) for i in range(n)]
        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=30)
        assert len(out_ts) == 30
        assert len(out_vals) == 30

    def test_preserves_first_and_last_points(self):
        n = 300
        timestamps = [f"t{i}" for i in range(n)]
        values = [float(i) for i in range(n)]
        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=20)
        assert out_ts[0] == timestamps[0]
        assert out_ts[-1] == timestamps[-1]
        assert out_vals[0] == values[0]
        assert out_vals[-1] == values[-1]

    def test_none_gaps_preserved_like_shared_axis_variant(self):
        n = 200
        timestamps = [f"t{i}" for i in range(n)]
        values: list[float | None] = [float(i) for i in range(n)]
        values[50:60] = [None] * 10
        _out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=20)
        # Some of the None-gap points may or may not be selected, but any that
        # are selected must still be None, never coerced to 0.0.
        for ts, val in zip(_out_ts, out_vals, strict=True):
            if ts in timestamps[50:60]:
                assert val is None

    def test_non_numeric_channel_passes_through_when_it_fits(self):
        """A status/enum channel under max_points is returned unchanged, not coerced."""
        timestamps = ["t0", "t1", "t2"]
        values = ["STANDBY", "CW", "CW"]
        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=200)
        assert out_ts == timestamps
        assert out_vals == values

    def test_non_numeric_channel_evenly_subsampled_without_coercion(self):
        """A status/enum channel over max_points is evenly subsampled, never
        coerced toward the LTTB triangle-area math (which would raise on
        strings, or be meaningless if strings were mapped to 0.0).
        """
        n = 400
        timestamps = [f"t{i}" for i in range(n)]
        values = ["CW" if i % 2 == 0 else "STANDBY" for i in range(n)]

        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=25)

        assert len(out_ts) <= 25
        assert out_ts[0] == timestamps[0]
        assert out_ts[-1] == timestamps[-1]
        # Every returned value is one of the channel's real strings -- never coerced.
        assert all(v in ("CW", "STANDBY") for v in out_vals)
        # Values line up with their own timestamps (a real, not fabricated, pairing).
        idx_by_ts = {ts: i for i, ts in enumerate(timestamps)}
        assert all(out_vals[i] == values[idx_by_ts[ts]] for i, ts in enumerate(out_ts))

    def test_mixed_none_and_string_values_treated_as_non_numeric(self):
        """A channel with None gaps alongside real string samples is still
        non-numeric overall -- it must not be coerced through the numeric path.
        """
        n = 300
        timestamps = [f"t{i}" for i in range(n)]
        values = [None if i % 3 == 0 else "FAULT" for i in range(n)]

        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=15)

        assert len(out_ts) <= 15
        assert out_ts[0] == timestamps[0]
        assert out_ts[-1] == timestamps[-1]
        assert all(v is None or v == "FAULT" for v in out_vals)

    def test_returns_tuple(self):
        timestamps = list(range(50))
        values = [float(i) for i in range(50)]
        result = lttb_downsample_channel(timestamps, values, max_points=10)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shorter_values_than_timestamps_does_not_raise_and_is_paired_by_zip(self):
        """Regression: a malformed channel whose `values` is shorter than its
        `timestamps` used to raise IndexError once past max_points -- real
        LTTB's bucket loop walks `range(len(timestamps))` but its helper
        arrays are built from `values`, so a shorter `values` indexed out of
        range. Tolerated the same way `_pivot_channel_series_to_table`
        tolerates the mismatch: `zip(..., strict=False)` pairs up to the
        shorter list rather than crashing.
        """
        timestamps = list(range(50))
        values = [float(i) for i in range(30)]

        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=10)

        assert len(out_ts) == 10
        assert len(out_vals) == 10
        # Paired against the (shorter) values list, so the output never
        # ranges past index 29 -- confirms zip-pairing, not a silent slice.
        assert out_ts[0] == 0
        assert out_ts[-1] == 29

    def test_longer_values_than_timestamps_does_not_raise_and_is_paired_by_zip(self):
        """Same tolerance, opposite mismatch direction: `values` longer than
        `timestamps`.

        NOT a regression test for the reported crash: with `values` longer,
        `n = len(timestamps)` is already the SHORTER list before this fix, so
        the pre-fix bucket loop (`range(n)`) never walked past the (longer)
        `values`-derived working copy and never raised. Only the sibling
        `test_shorter_values_than_timestamps_...` above reproduces the actual
        `IndexError`. This test exists to document the tolerance is symmetric,
        not to pin the fix.
        """
        timestamps = [f"t{i}" for i in range(30)]
        values = [float(i) for i in range(50)]

        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=10)

        assert len(out_ts) == 10
        assert len(out_vals) == 10
        assert out_ts[0] == "t0"
        assert out_ts[-1] == "t29"
        assert out_vals[-1] == 29.0
