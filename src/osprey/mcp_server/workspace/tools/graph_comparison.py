"""Pure comparison metrics for graph data.

Stateless computation — no MCP or DataContext dependencies.
Used by the graph_tools MCP module.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal, stats

logger = logging.getLogger("osprey.mcp_server.tools.graph_comparison")

# Available metrics
AVAILABLE_METRICS = ["rmse", "correlation", "dtw_distance", "peak_shift"]


def compute_rmse(current: np.ndarray, reference: np.ndarray) -> float:
    """Compute Root Mean Square Error between two 1D signals.

    Signals are interpolated to a common length if they differ.

    Args:
        current: Current signal (1D array).
        reference: Reference signal (1D array).

    Returns:
        RMSE value (lower is more similar).
    """
    current, reference = _align_lengths(current, reference)
    return float(np.sqrt(np.mean((current - reference) ** 2)))


def compute_correlation(current: np.ndarray, reference: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two 1D signals.

    Args:
        current: Current signal (1D array).
        reference: Reference signal (1D array).

    Returns:
        Correlation coefficient (-1 to 1). 1 means identical shape.
    """
    current, reference = _align_lengths(current, reference)

    # Handle constant signals
    if np.std(current) == 0 or np.std(reference) == 0:
        return 0.0

    r, _ = stats.pearsonr(current, reference)
    return float(r)


def compute_dtw_distance(current: np.ndarray, reference: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two 1D signals.

    DTW handles signals with different time bases or rates, making it
    robust for comparing traces from different sources.

    Args:
        current: Current signal (1D array).
        reference: Reference signal (1D array).

    Returns:
        DTW distance (lower is more similar).

    Raises:
        ImportError: If dtw-python is not installed.
    """
    try:
        from dtw import dtw as dtw_func
    except ImportError:
        raise ImportError(
            "dtw-python is required for DTW comparison. "
            "Install with: uv sync --extra graph"
        )

    alignment = dtw_func(current, reference)
    return float(alignment.distance)


def compute_peak_shift(current: np.ndarray, reference: np.ndarray) -> dict:
    """Compute peak position differences between two signals.

    Finds prominent peaks in both signals and reports the shift
    in the primary peak position.

    Args:
        current: Current signal (1D array).
        reference: Reference signal (1D array).

    Returns:
        Dict with current_peak_idx, reference_peak_idx, shift,
        and relative_shift (normalized to signal length).
    """
    current_peaks, _ = signal.find_peaks(current, prominence=np.std(current) * 0.5)
    ref_peaks, _ = signal.find_peaks(reference, prominence=np.std(reference) * 0.5)

    if len(current_peaks) == 0 or len(ref_peaks) == 0:
        return {
            "current_peak_idx": None,
            "reference_peak_idx": None,
            "shift": None,
            "relative_shift": None,
            "note": "No prominent peaks found in one or both signals",
        }

    # Use the highest peak from each signal
    current_main = current_peaks[np.argmax(current[current_peaks])]
    ref_main = ref_peaks[np.argmax(reference[ref_peaks])]

    # Normalize to common length basis
    current_norm = current_main / len(current)
    ref_norm = ref_main / len(reference)

    shift = int(current_main - ref_main)
    relative_shift = float(current_norm - ref_norm)

    return {
        "current_peak_idx": int(current_main),
        "reference_peak_idx": int(ref_main),
        "shift": shift,
        "relative_shift": round(relative_shift, 4),
    }


def compare_datasets(
    current_data: list[list],
    reference_data: list[list],
    metrics: list[str] | None = None,
) -> dict:
    """Run comparison metrics between two datasets.

    Extracts the y-values (second column) from each dataset and
    computes the requested metrics.

    Args:
        current_data: Current data as [[x,y], ...] pairs.
        reference_data: Reference data as [[x,y], ...] pairs.
        metrics: Which metrics to compute. Defaults to all available.

    Returns:
        Dict with metric names as keys and results as values,
        plus metadata about the comparison.
    """
    if metrics is None:
        metrics = list(AVAILABLE_METRICS)

    # Extract y-values
    current_y = np.array([row[1] if len(row) > 1 else row[0] for row in current_data], dtype=float)
    ref_y = np.array([row[1] if len(row) > 1 else row[0] for row in reference_data], dtype=float)

    results: dict = {
        "metrics": {},
        "current_points": len(current_y),
        "reference_points": len(ref_y),
    }

    for metric in metrics:
        if metric == "rmse":
            results["metrics"]["rmse"] = round(compute_rmse(current_y, ref_y), 6)
        elif metric == "correlation":
            results["metrics"]["correlation"] = round(compute_correlation(current_y, ref_y), 6)
        elif metric == "dtw_distance":
            try:
                results["metrics"]["dtw_distance"] = round(
                    compute_dtw_distance(current_y, ref_y), 6
                )
            except ImportError as e:
                results["metrics"]["dtw_distance"] = {"error": str(e)}
        elif metric == "peak_shift":
            results["metrics"]["peak_shift"] = compute_peak_shift(current_y, ref_y)
        else:
            results["metrics"][metric] = {"error": f"Unknown metric: {metric}"}

    return results


def _align_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate two 1D arrays to the same length (the longer one).

    Args:
        a: First array.
        b: Second array.

    Returns:
        Tuple of (a_resampled, b_resampled) with matching lengths.
    """
    if len(a) == len(b):
        return a, b

    target_len = max(len(a), len(b))

    if len(a) != target_len:
        x_old = np.linspace(0, 1, len(a))
        x_new = np.linspace(0, 1, target_len)
        a = np.interp(x_new, x_old, a)

    if len(b) != target_len:
        x_old = np.linspace(0, 1, len(b))
        x_new = np.linspace(0, 1, target_len)
        b = np.interp(x_new, x_old, b)

    return a, b
