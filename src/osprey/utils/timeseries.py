"""Timeseries utility functions.

Pure-math helpers for downsampling and extracting timeseries data,
used by both the Artifact Gallery interface and MCP server tools.
"""

import math


def lttb_downsample(
    index: list, data: list[list], max_points: int
) -> tuple[list, list[list]]:
    """Largest-Triangle-Three-Buckets downsampling.

    Operates on split-orient DataFrame arrays.  Uses the first data column as
    the representative for triangle-area index selection, then slices all
    columns at the same indices so every channel shares the same x-axis.

    Returns (downsampled_index, downsampled_data) where data keeps all columns.
    """
    n = len(index)
    if n <= max_points or max_points < 3:
        return index, data

    # Convert index to numeric for area calculation
    x = list(range(n))
    y = [row[0] if row else 0 for row in data]  # first channel

    selected = [0]  # Always keep first point
    bucket_size = (n - 2) / (max_points - 2)

    a_idx = 0
    for i in range(1, max_points - 1):
        # Next bucket boundaries
        b_start = int(math.floor((i - 1) * bucket_size)) + 1
        b_end = int(math.floor(i * bucket_size)) + 1
        b_end = min(b_end, n)

        # Average of the bucket after this one (lookahead)
        c_start = int(math.floor(i * bucket_size)) + 1
        c_end = int(math.floor((i + 1) * bucket_size)) + 1
        c_end = min(c_end, n)
        if c_start >= n:
            c_start = n - 1
        if c_end > n:
            c_end = n
        c_len = max(c_end - c_start, 1)
        avg_x = sum(x[c_start:c_end]) / c_len
        avg_y = sum(y[c_start:c_end]) / c_len

        # Pick point in current bucket with max triangle area
        max_area = -1.0
        best = b_start
        for j in range(b_start, b_end):
            area = abs(
                (x[a_idx] - avg_x) * (y[j] - y[a_idx])
                - (x[a_idx] - x[j]) * (avg_y - y[a_idx])
            )
            if area > max_area:
                max_area = area
                best = j

        selected.append(best)
        a_idx = best

    selected.append(n - 1)  # Always keep last point

    new_index = [index[i] for i in selected]
    new_data = [data[i] for i in selected]
    return new_index, new_data


def extract_timeseries_frame(raw: dict) -> tuple[dict, dict]:
    """Extract split-orient DataFrame + query metadata from a data context file.

    Handles two layouts:
      - Archiver: raw["data"]["dataframe"] = {columns, index, data},
                  raw["data"]["query"] = {...}
      - Flat:     raw["data"] = {columns, index, data}
    Returns (frame_dict, query_dict).
    """
    payload = raw.get("data", raw)
    if "dataframe" in payload:
        return payload["dataframe"], payload.get("query", {})
    return payload, {}
