"""MCP tool: archiver_downsample.

Returns per-channel downsampled timeseries data from an artifact entry.
Uses LTTB (Largest-Triangle-Three-Buckets) to preserve visual shape while
keeping the payload small enough for inline report generation. Each channel
carries its OWN timestamps -- channels have independent sample cadences and
are not aligned to a shared axis, so this is no longer a shared-``labels``
Chart.js config.
"""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.utils.timeseries import (
    extract_channel_series,
    is_numeric_channel,
    lttb_downsample_channel,
)

logger = logging.getLogger("osprey.mcp_server.tools.archiver_downsample")


@mcp.tool()
async def archiver_downsample(
    entry_id: str,
    max_points: int = 200,
    channels: list[str] | None = None,
) -> str:
    """Downsample a timeseries artifact entry for chart embedding.

    Uses LTTB (Largest-Triangle-Three-Buckets) to reduce each channel's point
    count independently while preserving its visual shape. A non-numeric
    (enum/status) channel's values are never coerced for the triangle-area
    math -- it passes through unchanged when it already fits under
    ``max_points`` and is evenly subsampled (keeping the first and last
    points) otherwise.

    Only works on category="archiver_data" entries.

    Args:
        entry_id: Artifact entry ID to downsample.
        max_points: Maximum number of points to return per channel (default 200).
        channels: Optional list of channel names to include. If omitted,
            all channels are included.

    Returns:
        JSON with ``datasets`` (each ``{"channel", "timestamps", "values",
        "original_points", "downsampled_points", "numeric"}``), plus top-level
        ``original_points``/``downsampled_points`` summed across channels
        and a ``time_range`` spanning all returned channels. Each dataset
        carries its own timestamps -- this is no longer a shared-``labels``
        Chart.js config. ``numeric`` is False for enum/status channels, which
        cannot share a numeric axis with the others.
    """
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.get_entry(entry_id)

    if entry is None:
        return make_error(
            "validation_error",
            f"Artifact entry '{entry_id}' not found.",
            suggestions=["Use session_summary to list available entries."],
        )

    if entry.category != "archiver_data":
        return make_error(
            "validation_error",
            f"Entry '{entry_id}' has category={entry.category!r}, not 'archiver_data'.",
            suggestions=["Only archiver_data entries can be downsampled."],
        )

    filepath = store.get_file_path(entry_id)
    if filepath is None:
        return make_error(
            "internal_error",
            f"Data file for entry '{entry_id}' not found on disk.",
        )

    try:
        raw = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return make_error(
            "internal_error",
            f"Could not read data file: {e}",
        )

    series, _query_meta = extract_channel_series(raw)
    all_channels = list(series.keys())

    # Filter channels if requested
    if channels:
        selected_channels = [c for c in channels if c in series]
        if not selected_channels:
            return make_error(
                "validation_error",
                f"None of the requested channels {channels} found in entry channels "
                f"{all_channels}.",
            )
    else:
        selected_channels = all_channels

    max_points = max(3, min(max_points, 10000))

    datasets = []
    for channel in selected_channels:
        chan_data = series[channel]
        timestamps = chan_data.get("timestamps", [])
        values = chan_data.get("values", [])

        ds_timestamps, ds_values = lttb_downsample_channel(timestamps, values, max_points)

        datasets.append(
            {
                "channel": channel,
                "timestamps": ds_timestamps,
                "values": ds_values,
                "original_points": len(timestamps),
                "downsampled_points": len(ds_timestamps),
                "numeric": is_numeric_channel(values),
            }
        )

    # A channel with no samples has no span to contribute -- but it must not
    # erase the others', hence filtering it out here rather than letting an
    # empty list into the min/max. `default=None` then covers the genuinely
    # spanless payload (no channels, or every channel empty). Downsampling
    # always keeps a channel's first and last point, so a dataset's own ends
    # are still the channel's real ends.
    spans = [d["timestamps"] for d in datasets if d["timestamps"]]

    result = {
        "datasets": datasets,
        "original_points": sum(d["original_points"] for d in datasets),
        "downsampled_points": sum(d["downsampled_points"] for d in datasets),
        "time_range": {
            "start": min((s[0] for s in spans), default=None),
            "end": max((s[-1] for s in spans), default=None),
        },
    }

    return json.dumps(result, indent=2)
