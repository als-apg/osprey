"""Tests for the graph comparison metrics library."""

import numpy as np
import pytest

from osprey.mcp_server.workspace.tools.graph_comparison import (
    _align_lengths,
    compare_datasets,
    compute_correlation,
    compute_peak_shift,
    compute_rmse,
)


class TestAlignLengths:
    """Tests for _align_lengths helper."""

    def test_same_length_passthrough(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        ra, rb = _align_lengths(a, b)
        np.testing.assert_array_equal(ra, a)
        np.testing.assert_array_equal(rb, b)

    def test_different_lengths_aligned(self):
        a = np.array([1.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        ra, rb = _align_lengths(a, b)
        assert len(ra) == len(rb) == 3

    def test_interpolation_preserves_endpoints(self):
        a = np.array([0.0, 10.0])
        b = np.array([0.0, 5.0, 10.0])
        ra, rb = _align_lengths(a, b)
        assert ra[0] == pytest.approx(0.0)
        assert ra[-1] == pytest.approx(10.0)


class TestComputeRMSE:
    """Tests for compute_rmse."""

    def test_identical_signals(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_rmse(sig, sig) == pytest.approx(0.0)

    def test_known_rmse(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        # RMSE = sqrt(mean([1, 1, 1])) = 1.0
        assert compute_rmse(a, b) == pytest.approx(1.0)

    def test_different_lengths(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 3.0])
        result = compute_rmse(a, b)
        assert isinstance(result, float)
        assert result >= 0


class TestComputeCorrelation:
    """Tests for compute_correlation."""

    def test_identical_signals(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_correlation(sig, sig) == pytest.approx(1.0)

    def test_inverted_signal(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert compute_correlation(a, b) == pytest.approx(-1.0)

    def test_constant_signal_returns_zero(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([1.0, 2.0, 3.0])
        assert compute_correlation(a, b) == pytest.approx(0.0)


class TestComputePeakShift:
    """Tests for compute_peak_shift."""

    def test_aligned_peaks(self):
        # Peak at index 5
        a = np.zeros(10)
        a[5] = 10.0
        b = np.zeros(10)
        b[5] = 8.0
        result = compute_peak_shift(a, b)
        assert result["shift"] == 0

    def test_shifted_peak(self):
        a = np.zeros(20)
        a[10] = 10.0
        b = np.zeros(20)
        b[8] = 10.0
        result = compute_peak_shift(a, b)
        assert result["shift"] == 2

    def test_no_peaks(self):
        a = np.zeros(10)
        b = np.zeros(10)
        result = compute_peak_shift(a, b)
        assert result["shift"] is None
        assert "note" in result


class TestCompareDatasets:
    """Tests for the orchestrator compare_datasets."""

    def test_all_metrics(self):
        current = [[0, 1.0], [1, 2.0], [2, 3.0], [3, 4.0], [4, 5.0]]
        reference = [[0, 1.1], [1, 2.1], [2, 3.1], [3, 4.1], [4, 5.1]]
        result = compare_datasets(current, reference)
        assert "metrics" in result
        assert "rmse" in result["metrics"]
        assert "correlation" in result["metrics"]
        assert result["current_points"] == 5
        assert result["reference_points"] == 5

    def test_selected_metrics(self):
        current = [[0, 1.0], [1, 2.0], [2, 3.0]]
        reference = [[0, 1.0], [1, 2.0], [2, 3.0]]
        result = compare_datasets(current, reference, metrics=["rmse"])
        assert "rmse" in result["metrics"]
        assert "correlation" not in result["metrics"]

    def test_identical_datasets(self):
        data = [[0, 1.0], [1, 2.0], [2, 3.0], [3, 4.0], [4, 5.0]]
        result = compare_datasets(data, data, metrics=["rmse", "correlation"])
        assert result["metrics"]["rmse"] == pytest.approx(0.0)
        assert result["metrics"]["correlation"] == pytest.approx(1.0)

    def test_single_column_data(self):
        """Data with only y-values (no x column)."""
        current = [[1.0], [2.0], [3.0]]
        reference = [[1.1], [2.1], [3.1]]
        result = compare_datasets(current, reference, metrics=["rmse"])
        assert isinstance(result["metrics"]["rmse"], float)

    def test_unknown_metric(self):
        current = [[0, 1.0]]
        reference = [[0, 1.0]]
        result = compare_datasets(current, reference, metrics=["bogus"])
        assert "error" in result["metrics"]["bogus"]
