"""MCP tool: archiver_read — retrieve historical data from the archiver."""

import json
import logging
import math
from datetime import UTC, datetime, timedelta

from osprey.mcp_server.control_system.error_handling import ToolError, connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.tools.archiver_read")


def _parse_time(time_str: str) -> datetime:
    """Parse a time string into a timezone-aware datetime.

    Supports ISO-8601 and simple relative expressions like "1h ago", "30m ago", "2d ago".
    """
    if not time_str or time_str.strip().lower() == "now":
        return datetime.now(UTC)

    stripped = time_str.strip().lower()

    # Relative time: "1h ago", "30m ago", "2d ago"
    if stripped.endswith(" ago"):
        amount_unit = stripped[:-4].strip()
        unit_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
        for suffix, kwarg in unit_map.items():
            if amount_unit.endswith(suffix):
                try:
                    amount = float(amount_unit[:-1])
                    return datetime.now(UTC) - timedelta(**{kwarg: amount})
                except ValueError:
                    break

    # Fall back to dateutil for ISO / human strings
    from dateutil import parser as dateutil_parser

    dt = dateutil_parser.parse(time_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@mcp.tool()
async def archiver_read(
    channels: list[str],
    start_time: str,
    end_time: str = "now",
    processing: str = "raw",
    bin_size: int | None = None,
) -> str:
    """Retrieve historical archived data for one or more channels.

    Data is always saved to a workspace file and a compact summary is returned.

    Args:
        channels: List of PV/channel addresses to query.
        start_time: Start of the time range — ISO-8601 or relative (e.g. "2h ago").
        end_time: End of the time range (default "now").
        processing: Archiver processing mode (e.g. "raw", "mean", "max").
        bin_size: Bin size in seconds when using a processing mode other than "raw".

    Returns:
        JSON summary with row counts, per-channel stats, and the data file path.
    """
    if not channels:
        return json.dumps(
            make_error(
                "validation_error",
                "No channels provided.",
                ["Provide at least one channel address."],
            )
        )

    try:
        start_dt = _parse_time(start_time)
    except Exception as exc:
        return json.dumps(
            make_error(
                "validation_error",
                f"Could not parse start_time '{start_time}': {exc}",
                ["Use ISO-8601 format or relative expressions like '2h ago'."],
            )
        )

    try:
        end_dt = _parse_time(end_time)
    except Exception as exc:
        return json.dumps(
            make_error(
                "validation_error",
                f"Could not parse end_time '{end_time}': {exc}",
                ["Use ISO-8601 format or 'now'."],
            )
        )

    try:
        async with connector_error_handler("archiver_read", connector_name="archiver"):
            from osprey.mcp_server.control_system.server_context import get_server_context

            registry = get_server_context()
            connector = await registry.archiver()

            df = await connector.get_data(
                channels,
                start_dt,
                end_dt,
                precision_ms=1000 if bin_size is None else bin_size * 1000,
            )

            # Full data payload goes to file; compact summary returned inline
            records = json.loads(df.to_json(orient="split", date_format="iso"))
            data_payload = {
                "query": {
                    "channels": channels,
                    "start_time": str(start_dt),
                    "end_time": str(end_dt),
                    "processing": processing,
                    "bin_size": bin_size,
                },
                "dataframe": records,
            }

            def _safe_stat(value: float) -> float | None:
                """Convert NaN to None for JSON-safe serialization."""
                v = float(value)
                return None if math.isnan(v) else v

            per_channel = {}
            for ch in channels:
                if ch in df.columns:
                    col = df[ch]
                    mean = _safe_stat(col.mean())
                    per_channel[ch] = {
                        "points": int(col.count()),
                        "min": _safe_stat(col.min()),
                        "max": _safe_stat(col.max()),
                        "mean": round(mean, 6) if mean is not None else None,
                    }

            summary = {
                "channels_queried": len(channels),
                "total_rows": len(df),
                "time_range": {"start": str(start_dt), "end": str(end_dt)},
                "per_channel": per_channel,
            }
            access_details = {
                "data_file_structure": {
                    "root_keys": ["query", "dataframe"],
                    "dataframe_format": "pandas split-orient JSON",
                    "dataframe_keys": ["index", "columns", "data"],
                },
                "schema": {
                    "index": "list of ISO-8601 timestamp strings (one per row)",
                    "columns": f"list of channel names: {list(df.columns)}",
                    "data": (
                        "list of lists — each inner list has one value per column, "
                        "aligned with index"
                    ),
                },
                "access_patterns": {
                    "all_timestamps": 'json_data["dataframe"]["index"]',
                    "channel_names": 'json_data["dataframe"]["columns"]',
                    "all_rows": 'json_data["dataframe"]["data"]',
                    "single_channel_values": (
                        'col_idx = columns.index("CHANNEL"); [row[col_idx] for row in data]'
                    ),
                    "value_at_time": "data[row_idx][col_idx]",
                },
                "example_row": {
                    "timestamp": str(df.index[0]) if len(df) > 0 else None,
                    "values": (
                        {ch: _safe_stat(df[ch].iloc[0]) for ch in df.columns}
                        if len(df) > 0
                        else {}
                    ),
                },
                "row_count": len(df),
                "processing": processing,
                "bin_size": bin_size,
            }

            # Save via ArtifactStore (unified)
            from osprey.stores.artifact_store import get_artifact_store

            store = get_artifact_store()
            entry = store.save_data(
                tool="archiver_read",
                data=data_payload,
                title=f"Archiver data for {len(channels)} channel(s)",
                description=f"Archiver data for {len(channels)} channel(s), {len(df)} points",
                summary=summary,
                access_details=access_details,
                category="archiver_data",
                metadata={"data_type": "timeseries"},
            )

            return json.dumps(entry.to_tool_response(), default=str)

    except ToolError as exc:
        return exc.response
