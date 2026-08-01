"""Tests for the timeseries downsampling and frame-extraction helpers.

``lttb_downsample_channel`` is the Largest-Triangle-Three-Buckets reducer shared
by the Artifact Gallery and MCP tools. It reduces exactly one channel: each
channel carries its own timestamps, so there is no shared index and no
multi-column slicing. Two contracts matter most: it always keeps the first and
last points and returns exactly ``max_points`` when it downsamples, and it
applies the selected indices to the ORIGINAL values so ``None`` gap markers
(archiver disconnects / IOC reboots) survive into the output even though the
triangle-area math runs on a zero-filled working copy.

Index selection is tested directly against the private ``_lttb_select_indices``
because that is where the triangle-area rule lives; the public function's own
tests then pin what the caller actually observes.
"""

from __future__ import annotations

from osprey.utils.timeseries import (
    _lttb_select_indices,
    extract_channel_series,
    is_numeric_channel,
    lttb_downsample_channel,
)


class TestLttbSelectIndices:
    """The triangle-area selection rule, exercised on its own.

    The selector returns indices, never values -- that separation is what lets
    the caller slice the untouched originals and keep gaps as gaps. Callers
    only reach it with ``max_points >= 3`` and ``len(values) > max_points``.
    """

    def test_returns_exactly_max_points_indices(self):
        assert len(_lttb_select_indices([float(i) for i in range(1000)], 50)) == 50

    def test_first_and_last_indices_always_selected(self):
        selected = _lttb_select_indices([float(i) for i in range(1000)], 25)
        assert selected[0] == 0
        assert selected[-1] == 999

    def test_min_max_points_of_three(self):
        selected = _lttb_select_indices([float(i) for i in range(500)], 3)
        assert selected == [0, selected[1], 499]
        assert len(selected) == 3

    def test_indices_are_ascending_and_within_range(self):
        n = 300
        selected = _lttb_select_indices([float(i) for i in range(n)], 20)
        assert selected == sorted(selected)
        assert all(0 <= i < n for i in selected)

    def test_dominant_spike_is_selected(self):
        """The whole point of LTTB: a sharp feature survives downsampling."""
        values = [0.0] * 200
        values[100] = 1000.0  # single large spike
        assert 100 in _lttb_select_indices(values, 10)

    def test_spike_and_valley_are_both_selected_by_triangle_area(self):
        """Extrema of BOTH signs survive, and only via the area comparison.

        Unlike ``test_dominant_spike_is_selected`` above -- whose spike at 100
        happens to land exactly on a bucket's first index (bucket_size 198/8 =
        24.75, so i=5 gives b_start=100), meaning it is picked even if the
        triangle-area comparison never fires -- neither 50 nor 150 is a bucket
        start here (bucket_size 198/18 = 11.0, boundaries 1, 12, 23, ...). They
        are reached only because their triangle area is the largest in their
        bucket, so this pins the selection rule itself, not the bucket lattice.
        """
        values = [0.0] * 200
        values[50] = 100.0  # spike
        values[150] = -100.0  # valley
        selected = _lttb_select_indices(values, 20)
        assert 50 in selected
        assert 150 in selected

    def test_selected_indices_are_exactly_pinned(self):
        """Golden characterization of the selection rule itself.

        The property tests above (spike/valley survive, endpoints kept, exact
        count) all pass under arithmetic that is merely *close* to LTTB -- they
        pin the shape of the answer, not the triangle-area formula. This test
        pins the formula. Verified to move at least one index under each of:
        widening the look-ahead bucket by one, flipping the sign of the ``x``
        difference, and perturbing either the ``avg_x`` or ``avg_y`` term.

        The fixture is a pure-integer ramp so the expectation is reproducible
        on every platform and Python version.
        """
        values = [float(i) for i in range(120)]  # ramp
        values[17] = 50.0  # spike above the ramp
        values[88] = -50.0  # valley below it
        selected = _lttb_select_indices(values, 12)
        assert selected == [0, 11, 17, 24, 36, 48, 60, 82, 88, 95, 107, 119]

    def test_none_gaps_do_not_crash_the_area_math(self):
        """A gap is 0.0 in the working copy only, so the arithmetic never sees
        ``NoneType`` -- the selector still returns a full index list."""
        values: list[float | None] = [float(i) for i in range(200)]
        values[50:60] = [None] * 10
        assert len(_lttb_select_indices(values, 20)) == 20

    def test_all_none_channel_selects_normally(self):
        """Every value a gap: the zero-filled copy is flat, so selection falls
        back to bucket starts -- but still returns first and last."""
        selected = _lttb_select_indices([None] * 200, 20)
        assert len(selected) == 20
        assert selected[0] == 0
        assert selected[-1] == 199


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
        unlike ``lttb_downsample_channel`` which preserves None gaps in a
        channel's own values.
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
    """The public reducer: one channel's own timestamps and values in, the
    same pair reduced out. What the caller observes -- pass-through, first/last
    retention, gap survival, and the non-numeric fallback -- is pinned here;
    the index-selection rule itself is pinned in ``TestLttbSelectIndices``.
    """

    def test_returns_input_when_n_below_max_points(self):
        timestamps = ["t0", "t1", "t2"]
        values = [0.0, 1.0, 2.0]
        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=10)
        assert out_ts is timestamps
        assert out_vals is values

    def test_returns_input_when_n_equals_max_points(self):
        """The boundary: exactly ``max_points`` samples is already small enough."""
        timestamps = [f"t{i}" for i in range(4)]
        values = [float(i) for i in range(4)]
        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=4)
        assert out_ts is timestamps
        assert out_vals is values

    def test_returns_input_when_max_points_below_three(self):
        """LTTB needs at least first/last plus one bucket; <3 is a no-op."""
        timestamps = [f"t{i}" for i in range(100)]
        values = [float(i) for i in range(100)]
        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=2)
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

    def test_none_gaps_survive_into_the_output(self):
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

    def test_gap_on_an_always_kept_point_survives(self):
        """The unconditional version of the test above.

        Selection is data-dependent, so the sibling test only constrains gaps
        that happen to be picked. Indices 0 and n-1 are kept by construction,
        so putting a gap on each pins gap survival with no dependence on the
        triangle-area outcome: a disconnect must arrive at the chart as a
        break in the line, not as a fabricated 0.0 reading.
        """
        n = 300
        timestamps = [f"t{i}" for i in range(n)]
        values: list[float | None] = [float(i) for i in range(n)]
        values[0] = None
        values[-1] = None

        out_ts, out_vals = lttb_downsample_channel(timestamps, values, max_points=20)

        assert out_ts[0] == "t0"
        assert out_ts[-1] == f"t{n - 1}"
        assert out_vals[0] is None
        assert out_vals[-1] is None

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
