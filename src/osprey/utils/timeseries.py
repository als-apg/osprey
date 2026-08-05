"""Timeseries utility functions.

Pure-math helpers for downsampling and extracting timeseries data,
used by both the Artifact Gallery interface and MCP server tools.
"""

import math


def _lttb_select_indices(values: list, max_points: int) -> list[int]:
    """Largest-Triangle-Three-Buckets index selection over one channel's values.

    Returns the indices to keep, never the values themselves: the caller slices
    its own ``timestamps``/``values`` with them, so a ``None`` gap marker
    (archiver disconnect / IOC reboot) is carried through untouched rather than
    being reconstructed. Nothing is manufactured -- every returned index refers
    to a sample that really exists.

    The x coordinate of a point is simply its index, so there is no separate
    x array: a point's position on the time axis is its ordinal, not its
    timestamp. (LTTB over unevenly spaced samples would weight by real elapsed
    time; this implementation deliberately does not, and never has.)

    Only reached with ``max_points >= 3`` and ``len(values) > max_points``;
    ``lttb_downsample_channel`` handles the pass-through cases before calling.

    Returns ``max_points`` ascending indices, always starting at 0 and ending
    at ``len(values) - 1``.
    """
    n = len(values)

    # Sanitized numeric working copy for triangle-area math ONLY.
    # None gap values -> 0.0 here so the bucket-average and area arithmetic
    # never see NoneType. Only this local copy is zero-filled; the caller
    # slices the ORIGINAL values, so gaps survive into the output.
    y = [0.0 if v is None else float(v) for v in values]

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
        # x[i] == i, so the mean of the look-ahead bucket's x coordinates is
        # just the mean of its index range -- no identity list materialized.
        avg_x = sum(range(c_start, c_end)) / c_len
        avg_y = sum(y[c_start:c_end]) / c_len

        # Pick point in current bucket with max triangle area
        max_area = -1.0
        best = b_start
        for j in range(b_start, b_end):
            area = abs((a_idx - avg_x) * (y[j] - y[a_idx]) - (a_idx - j) * (avg_y - y[a_idx]))
            if area > max_area:
                max_area = area
                best = j

        selected.append(best)
        a_idx = best

    selected.append(n - 1)  # Always keep last point
    return selected


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
    kept as a gap marker (contrast with ``lttb_downsample_channel``, which
    preserves a `None` gap that already survived into a channel's own column).

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
    # `None` counts as numeric: it is a gap marker (archiver disconnect / IOC
    # reboot), not an enum/status value, and LTTB zero-fills it for the area math
    # while preserving it in the output.
    return all(v is None or isinstance(v, int | float) for v in values)


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


def lttb_downsample_channel(
    timestamps: list, values: list, max_points: int, numeric: bool | None = None
) -> tuple[list, list]:
    """Largest-Triangle-Three-Buckets downsampling for a single channel.

    Each channel carries its own timestamps, independent of every other
    channel's, so there is no shared x-axis and no multi-column slicing: this
    reduces exactly one (timestamps, values) pair. Callers with several
    channels call it once per channel.

    Selection and slicing are separate steps -- :func:`_lttb_select_indices`
    picks indices from a zero-filled numeric copy, and those indices are
    applied to the ORIGINAL ``timestamps``/``values``. That is what keeps a
    ``None`` gap a ``None`` in the output instead of a fabricated 0.0, and what
    guarantees the first and last samples are always returned.

    LTTB's triangle-area math requires numeric values. A non-numeric
    (enum/status) channel is never coerced toward that math -- it passes
    through unchanged when it already fits under ``max_points``, and is
    evenly subsampled (always keeping the first and last points) when it
    does not.

    A channel whose ``timestamps``/``values`` lengths differ (a malformed
    artifact, not the expected shape) is tolerated the same way
    ``_pivot_channel_series_to_table`` already tolerates it: both are truncated
    to the shorter length, so the mismatch never crashes the caller. Without
    this, real LTTB's helper arrays are built from ``values`` while its bucket
    loop walks ``range(len(timestamps))``, so a shorter ``values`` raises
    ``IndexError`` once the point count exceeds ``max_points``.

    Args:
        timestamps: This channel's own timestamps, ascending.
        values: This channel's own values, one per timestamp -- numeric,
            ``None``, or non-numeric strings (enum/status channels).
        max_points: Maximum number of points to return.
        numeric: Whether this channel is numeric, if the caller already knows.
            Callers that report a channel's dtype alongside its samples have
            already paid for :func:`is_numeric_channel`; passing the answer in
            keeps this from walking every value a second time. ``None`` means
            "work it out", which is what a caller that does not report dtype
            wants.

    Returns:
        Tuple of (downsampled_timestamps, downsampled_values).
    """
    if len(values) != len(timestamps):
        shared = min(len(timestamps), len(values))
        timestamps, values = timestamps[:shared], values[:shared]

    n = len(timestamps)
    if n <= max_points or max_points < 3:
        return timestamps, values

    if not (is_numeric_channel(values) if numeric is None else numeric):
        return _even_subsample(timestamps, values, max_points)

    selected = _lttb_select_indices(values, max_points)
    return [timestamps[i] for i in selected], [values[i] for i in selected]


def downsample_channel_map(
    series: dict[str, dict], max_points: int, channels: list[str] | None = None
) -> list[dict]:
    """Downsample each requested channel and report what that cost it.

    The shared body of every "give me a plottable version of this artifact"
    endpoint: pick the channels, reduce each one, and say how many points it
    started with, how many came back, and whether it is numeric. Callers rename
    these into their own published field names -- the artifacts web API and the
    ``archiver_downsample`` MCP tool each have their own wire vocabulary -- but
    none of them re-derive the numbers.

    Deciding ``numeric`` here rather than in each caller is also what keeps the
    channel from being walked twice: :func:`lttb_downsample_channel` needs the
    same answer internally, and takes it as an argument rather than recomputing
    it.

    Args:
        series: Per-channel series as returned by :func:`extract_channel_series`.
        max_points: Maximum points to return per channel.
        channels: Optional subset to keep, in the caller's requested order.
            ``None`` keeps every channel in the series' own order. Names not
            present in ``series`` are dropped, so an empty result means nothing
            requested matched.

    Returns:
        One record per selected channel, in selection order, with keys
        ``channel``, ``timestamps``, ``values``, ``original_points``,
        ``returned_points`` and ``numeric``.
    """
    selected = list(series) if channels is None else [c for c in channels if c in series]

    records = []
    for channel in selected:
        channel_series = series[channel]
        timestamps = channel_series.get("timestamps", [])
        values = channel_series.get("values", [])
        numeric = is_numeric_channel(values)
        ds_timestamps, ds_values = lttb_downsample_channel(
            timestamps, values, max_points, numeric=numeric
        )
        records.append(
            {
                "channel": channel,
                "timestamps": ds_timestamps,
                "values": ds_values,
                "original_points": len(timestamps),
                "returned_points": len(ds_timestamps),
                "numeric": numeric,
            }
        )
    return records
