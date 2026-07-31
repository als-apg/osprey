"""Timeseries utility functions.

Pure-math helpers for downsampling and extracting timeseries data,
used by both the Artifact Gallery interface and MCP server tools.
"""

import math


def lttb_downsample(index: list, data: list[list], max_points: int) -> tuple[list, list[list]]:
    """Largest-Triangle-Three-Buckets downsampling over a shared index.

    Takes a single index shared by every column -- the legacy split-orient
    layout, and the internal shape :func:`lttb_downsample_channel` reduces one
    channel through. It selects indices by triangle area using the first data
    column as the representative, then slices every column at those same
    indices, so all columns necessarily come out on one x-axis.

    Archiver artifacts no longer have a shared index: each channel carries its
    own timestamps. Reduce those with :func:`lttb_downsample_channel`, which
    calls this per channel with that channel's own index.

    Returns (downsampled_index, downsampled_data) where data keeps all columns.
    """
    n = len(index)
    if n <= max_points or max_points < 3:
        return index, data

    # Sanitized numeric working copy for triangle-area math ONLY.
    # None gap values (archiver disconnects / IOC reboots) -> 0.0 here so the
    # bucket-average and area arithmetic never see NoneType. The selected indices
    # are applied to the ORIGINAL `data` below, so gaps are preserved in the output.
    def _num(v: float | int | None) -> float:
        return float(v) if v is not None else 0.0

    clean = [[_num(v) for v in row] if row else [] for row in data]
    x = list(range(n))
    y = [clean[i][0] if clean[i] else 0.0 for i in range(n)]  # first channel

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
                (x[a_idx] - avg_x) * (y[j] - y[a_idx]) - (x[a_idx] - x[j]) * (avg_y - y[a_idx])
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


def extract_channel_series(raw: dict) -> tuple[dict[str, dict], dict]:
    """Normalize any artifact timeseries layout into per-channel series.

    Handles three layouts:
      - New (long-format archiver_read): payload["series"] =
        {channel: {"timestamps": [...], "values": [...]}},
        payload["query"] = {...}. Already per-channel -- returned as-is.
      - Archiver (legacy split-orient): payload["dataframe"] =
        {columns, index, data}, payload["query"] = {...}.
      - Flat (legacy split-orient, no wrapper): payload =
        {columns, index, data} directly.

    For the two legacy split-orient layouts every column shares one `index`,
    so each column is transposed into its own (timestamps, values) pair.
    A `None` value at a given row/column means that channel had no sample at
    that shared timestamp -- not a real value -- so it is dropped rather than
    kept as a gap marker (contrast with ``lttb_downsample``, which preserves
    a `None` gap that already survived into a channel's own column).

    Args:
        raw: Parsed JSON content of an artifact data file.

    Returns:
        Tuple of (series, query) where series maps channel name to
        {"timestamps": [...], "values": [...]}, in declaration order
        (dict/column order).
    """
    payload = raw.get("data", raw)

    if "series" in payload:
        return payload["series"], payload.get("query", {})

    if "dataframe" in payload:
        frame = payload["dataframe"]
        query = payload.get("query", {})
    else:
        frame = payload
        query = {}

    columns = frame.get("columns", [])
    index = frame.get("index", [])
    rows = frame.get("data", [])

    series: dict[str, dict] = {}
    for col_idx, channel in enumerate(columns):
        timestamps: list = []
        values: list = []
        for row_idx, row in enumerate(rows):
            value = row[col_idx] if col_idx < len(row) else None
            if value is None:
                continue
            timestamps.append(index[row_idx])
            values.append(value)
        series[channel] = {"timestamps": timestamps, "values": values}

    return series, query


def _is_numeric_sample(value: object) -> bool:
    """True for ``None`` or an int/float sample; False for an enum/status string."""
    return value is None or isinstance(value, int | float)


def is_numeric_channel(values: list) -> bool:
    """True if every value in a channel's series is numeric (``None`` counts as numeric).

    This is the same check ``lttb_downsample_channel`` performs internally to decide
    whether LTTB's triangle-area math applies to a channel. Exposed so callers (the
    artifacts web API, ``archiver_downsample``) can report a channel's dtype directly
    rather than re-deriving it or guessing from a sample of values.

    Args:
        values: A channel's own values (numeric, ``None``, or enum/status strings).

    Returns:
        True if the channel is numeric (safe for LTTB), False otherwise.
    """
    return all(_is_numeric_sample(v) for v in values)


def _even_subsample(timestamps: list, values: list, max_points: int) -> tuple[list, list]:
    """Evenly-spaced subsampling that always keeps the first and last points.

    Used in place of LTTB for non-numeric (enum/status) channels, where
    triangle-area math over the values is meaningless. Callers only reach
    this with ``max_points >= 3`` and ``len(timestamps) > max_points``.
    """
    # i=0 and i=max_points-1 land exactly on 0 and n-1, so first and last are
    # kept by construction. The set collapses the repeats rounding produces as
    # max_points approaches n; sorted restores order.
    n = len(timestamps)
    step = (n - 1) / (max_points - 1)
    selected = sorted({min(round(i * step), n - 1) for i in range(max_points)})
    return [timestamps[i] for i in selected], [values[i] for i in selected]


def lttb_downsample_channel(timestamps: list, values: list, max_points: int) -> tuple[list, list]:
    """Largest-Triangle-Three-Buckets downsampling for a single channel.

    Per-channel variant of ``lttb_downsample``: each channel carries its own
    timestamps, independent of every other channel's, so there is no shared
    x-axis to slice multiple columns against (see that function's docstring).

    LTTB's triangle-area math requires numeric values. A non-numeric
    (enum/status) channel is never coerced toward that math -- it passes
    through unchanged when it already fits under ``max_points``, and is
    evenly subsampled (always keeping the first and last points) when it
    does not.

    A channel whose ``timestamps``/``values`` lengths differ (a malformed
    artifact, not the expected shape) is tolerated the same way
    ``_pivot_channel_series_to_table`` already tolerates it: pairs are built
    with ``zip(..., strict=False)``, so the mismatch never crashes the
    caller. Without this, real LTTB's helper arrays are built from
    ``values`` while its bucket loop walks ``range(len(timestamps))``, so a
    shorter ``values`` raises ``IndexError`` once the point count exceeds
    ``max_points``.

    Args:
        timestamps: This channel's own timestamps, ascending.
        values: This channel's own values, one per timestamp -- numeric,
            ``None``, or non-numeric strings (enum/status channels).
        max_points: Maximum number of points to return.

    Returns:
        Tuple of (downsampled_timestamps, downsampled_values).
    """
    if len(values) != len(timestamps):
        paired = list(zip(timestamps, values, strict=False))
        timestamps = [ts for ts, _ in paired]
        values = [v for _, v in paired]

    n = len(timestamps)
    if n <= max_points or max_points < 3:
        return timestamps, values

    if not is_numeric_channel(values):
        return _even_subsample(timestamps, values, max_points)

    ds_timestamps, ds_rows = lttb_downsample(timestamps, [[v] for v in values], max_points)
    ds_values = [row[0] for row in ds_rows]
    return ds_timestamps, ds_values
