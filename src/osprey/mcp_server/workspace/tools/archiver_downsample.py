"""MCP tool: archiver_downsample.

Returns chart-ready downsampled timeseries data from an artifact entry.
Uses LTTB (Largest-Triangle-Three-Buckets) to preserve visual shape while
keeping the payload small enough for inline report generation.
"""

import json
import logging

from osprey.interfaces.artifacts.app import _extract_timeseries_frame, lttb_downsample
from osprey.mcp_server.errors import make_error
from osprey.utils.workspace import resolve_workspace_root
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.archiver_downsample")


@mcp.tool()
async def archiver_downsample(
    entry_id: str,
    max_points: int = 200,
    channels: list[str] | None = None,
) -> str:
    """Downsample a timeseries artifact entry for chart embedding.

    Uses LTTB (Largest-Triangle-Three-Buckets) to reduce point count while
    preserving the visual shape of the data. Returns a chart-ready payload
    with labels and datasets that can be directly embedded in Chart.js config.

    Only works on category="archiver_data" entries.

    Args:
        entry_id: Artifact entry ID to downsample.
        max_points: Maximum number of points to return (default 200).
        channels: Optional list of channel names to include. If omitted,
            all channels are included.

    Returns:
        JSON with labels, datasets, original_points, downsampled_points,
        and time_range suitable for Chart.js.
    """
    try:
        workspace_root = resolve_workspace_root()
    except Exception as e:
        return json.dumps(
            make_error(
                "internal_error",
                f"Could not resolve workspace root: {e}",
            )
        )

    from osprey.mcp_server.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=workspace_root)
    entry = store.get_entry(entry_id)

    if entry is None:
        return json.dumps(
            make_error(
                "validation_error",
                f"Artifact entry '{entry_id}' not found.",
                suggestions=["Use session_summary to list available entries."],
            )
        )

    if entry.category != "archiver_data":
        return json.dumps(
            make_error(
                "validation_error",
                f"Entry '{entry_id}' has category={entry.category!r}, not 'archiver_data'.",
                suggestions=["Only archiver_data entries can be downsampled."],
            )
        )

    filepath = store.get_file_path(entry_id)
    if filepath is None:
        return json.dumps(
            make_error(
                "internal_error",
                f"Data file for entry '{entry_id}' not found on disk.",
            )
        )

    try:
        raw = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return json.dumps(
            make_error(
                "internal_error",
                f"Could not read data file: {e}",
            )
        )

    frame, query_meta = _extract_timeseries_frame(raw)
    all_columns = frame.get("columns", [])
    index = frame.get("index", [])
    rows = frame.get("data", [])
    original_points = len(index)

    if original_points == 0:
        return json.dumps(
            {
                "labels": [],
                "datasets": [],
                "original_points": 0,
                "downsampled_points": 0,
                "time_range": {"start": None, "end": None},
            }
        )

    # Filter channels if requested
    if channels:
        col_indices = [i for i, c in enumerate(all_columns) if c in channels]
        if not col_indices:
            return json.dumps(
                make_error(
                    "validation_error",
                    f"None of the requested channels {channels} found in "
                    f"entry columns {all_columns}.",
                )
            )
        selected_columns = [all_columns[i] for i in col_indices]
        filtered_rows = [[row[i] for i in col_indices] for row in rows]
    else:
        selected_columns = all_columns
        filtered_rows = rows

    # Downsample
    max_points = max(3, min(max_points, 10000))
    ds_index, ds_rows = lttb_downsample(index, filtered_rows, max_points)
    downsampled_points = len(ds_index)

    # Build Chart.js-ready datasets
    datasets = []
    for col_idx, col_name in enumerate(selected_columns):
        values = [row[col_idx] if col_idx < len(row) else None for row in ds_rows]
        datasets.append(
            {
                "channel": col_name,
                "values": values,
            }
        )

    time_range = {
        "start": ds_index[0] if ds_index else None,
        "end": ds_index[-1] if ds_index else None,
    }

    result = {
        "labels": ds_index,
        "datasets": datasets,
        "original_points": original_points,
        "downsampled_points": downsampled_points,
        "time_range": time_range,
    }

    return json.dumps(result, indent=2)
