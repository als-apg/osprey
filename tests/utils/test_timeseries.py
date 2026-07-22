"""Tests for the timeseries downsampling and frame-extraction helpers.

``lttb_downsample`` is the Largest-Triangle-Three-Buckets reducer shared by the
Artifact Gallery and MCP tools. Two contracts matter most: it always keeps the
first and last points and returns exactly ``max_points`` when it downsamples,
and it applies the selected indices to the ORIGINAL data so ``None`` gap markers
(archiver disconnects / IOC reboots) survive into the output even though the
triangle-area math runs on a zero-filled working copy.
"""

from __future__ import annotations

from osprey.utils.timeseries import extract_timeseries_frame, lttb_downsample


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


class TestExtractTimeseriesFrame:
    def test_archiver_layout_with_query(self):
        frame = {"columns": ["t", "v"], "index": [1, 2], "data": [[1], [2]]}
        query = {"channels": ["A"], "start": "2026-01-01"}
        raw = {"data": {"dataframe": frame, "query": query}}
        out_frame, out_query = extract_timeseries_frame(raw)
        assert out_frame is frame
        assert out_query is query

    def test_archiver_layout_missing_query_defaults_empty(self):
        frame = {"columns": ["t"], "index": [1], "data": [[1]]}
        raw = {"data": {"dataframe": frame}}
        out_frame, out_query = extract_timeseries_frame(raw)
        assert out_frame is frame
        assert out_query == {}

    def test_flat_layout(self):
        payload = {"columns": ["t", "v"], "index": [1, 2], "data": [[1], [2]]}
        raw = {"data": payload}
        out_frame, out_query = extract_timeseries_frame(raw)
        assert out_frame is payload
        assert out_query == {}

    def test_no_data_or_dataframe_key_uses_raw_as_payload(self):
        """With neither ``data`` nor ``dataframe`` present, raw is the payload."""
        raw = {"columns": ["t"], "index": [1]}
        out_frame, out_query = extract_timeseries_frame(raw)
        assert out_frame is raw
        assert out_query == {}

    def test_no_data_key_with_nested_dataframe(self):
        frame = {"columns": ["t"], "index": [1], "data": [[1]]}
        raw = {"dataframe": frame, "query": {"k": "v"}}
        out_frame, out_query = extract_timeseries_frame(raw)
        assert out_frame is frame
        assert out_query == {"k": "v"}
