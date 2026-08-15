"""MCP tool: archiver_downsample.

Returns per-channel downsampled timeseries data from an artifact entry.
Uses LTTB (Largest-Triangle-Three-Buckets) to preserve visual shape while
keeping the payload small enough for inline report generation. Each channel
carries its own timestamps -- channels have independent sample cadences and
are not aligned to a shared axis.
"""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.utils.timeseries import (
    downsample_channel_map,
    extract_channel_series,
)

logger = logging.getLogger("osprey.mcp_server.tools.archiver_downsample")


@mcp.tool()
async def archiver_downsample(
    artifact_id: str,
    max_points: int = 200,
    channels: list[str] | None = None,
) -> str:
    """Downsample a timeseries artifact for chart embedding.

    Uses LTTB (Largest-Triangle-Three-Buckets) to reduce each channel's point
    count independently while preserving its visual shape. A non-numeric
    (enum/status) channel's values are never coerced for the triangle-area
    math -- it passes through unchanged when it already fits under
    ``max_points`` and is evenly subsampled (keeping the first and last
    points) otherwise.

    Only works on category="archiver_data" artifacts.

    Args:
        artifact_id: ID of the artifact to downsample.
        max_points: Maximum number of points to return per channel (default 200).
        channels: Optional list of channel names to include. If omitted,
            all channels are included.

    Returns:
        JSON with ``datasets`` (each ``{"channel", "timestamps", "values",
        "original_points", "downsampled_points", "numeric"}``), plus top-level
        ``original_points``/``downsampled_points`` summed across channels
        and a ``time_range`` spanning all returned channels. Each dataset
        carries its own timestamps. ``numeric`` is False for enum/status
        channels, which cannot share a numeric axis with the others.
    """
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.get_entry(artifact_id)

    if entry is None:
        return make_error(
            "validation_error",
            f"Artifact '{artifact_id}' not found.",
            suggestions=["Use artifact_list to see available artifacts."],
        )

    if entry.category != "archiver_data":
        return make_error(
            "validation_error",
            f"Artifact '{artifact_id}' has category={entry.category!r}, not 'archiver_data'.",
            suggestions=["Only archiver_data artifacts can be downsampled."],
        )

    filepath = store.get_file_path(artifact_id)
    if filepath is None:
        return make_error(
            "internal_error",
            f"File for artifact '{artifact_id}' not found on disk.",
        )

    try:
        raw = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return make_error(
            "internal_error",
            f"Could not read data file: {e}",
        )

    series, _query_meta = extract_channel_series(raw)

    max_points = max(3, min(max_points, 10000))
    records = downsample_channel_map(series, max_points, channels=channels or None)

    if channels and not records:
        return make_error(
            "validation_error",
            f"None of the requested channels {channels} found in entry channels {list(series)}.",
        )

    datasets = [
        {
            "channel": record["channel"],
            "timestamps": record["timestamps"],
            "values": record["values"],
            "original_points": record["original_points"],
            "downsampled_points": record["returned_points"],
            "numeric": record["numeric"],
        }
        for record in records
    ]

    # An empty channel must not erase the other channels' spans in the
    # min/max; default=None covers the case where every channel is empty.
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
