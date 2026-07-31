"""MCP tool: archiver_read — retrieve historical data from the archiver."""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from osprey.connectors.archiver import PROCESSING_MODES
from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error
from osprey.utils.config import get_facility_timezone

logger = logging.getLogger("osprey.mcp_server.tools.archiver_read")


def _parse_time(time_str: str) -> datetime:
    """Parse a time string into a timezone-aware datetime.

    Supports ISO-8601 and simple relative expressions like "1h ago", "30m ago", "2d ago".
    Naive inputs (operator-provided wall-clock with no offset) are interpreted as
    facility-local — the agent rule promises operator times are facility-local, so the
    query window and the echoed range must honor the facility zone, not the box/UTC.
    """
    tz = get_facility_timezone()
    if not time_str or time_str.strip().lower() == "now":
        return datetime.now(tz)

    stripped = time_str.strip().lower()

    # Relative time: "1h ago", "30m ago", "2d ago"
    if stripped.endswith(" ago"):
        amount_unit = stripped[:-4].strip()
        unit_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
        for suffix, kwarg in unit_map.items():
            if amount_unit.endswith(suffix):
                try:
                    amount = float(amount_unit[:-1])
                    return datetime.now(tz) - timedelta(**{kwarg: amount})
                except ValueError:
                    break

    # Fall back to dateutil for ISO / human strings
    from dateutil import parser as dateutil_parser

    dt = dateutil_parser.parse(time_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
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
        processing: Aggregation within each bin — one of "raw", "mean", "min",
            "max", "median", "std", "count".
        bin_size: Bin size in seconds when using a processing mode other than
            "raw". ``0`` requests full resolution — every real archived sample
            in range, with no per-bin decimation — and is only valid with
            processing="raw" (an aggregate has no bin to aggregate over).
            ``None`` (the default) uses a 1-second bin. Negative values are
            rejected.

    Returns:
        JSON summary with per-channel point counts and stats, and the data
        file path.
    """
    if not channels:
        return make_error(
            "validation_error",
            "No channels provided.",
            ["Provide at least one channel address."],
        )

    if processing not in PROCESSING_MODES:
        return make_error(
            "validation_error",
            f"Unknown processing mode '{processing}'.",
            [f"Use one of: {', '.join(PROCESSING_MODES)}."],
        )

    if bin_size is not None and bin_size < 0:
        return make_error(
            "validation_error",
            f"bin_size must be >= 0 (got {bin_size}).",
            ["Use 0 for full resolution, or a positive number of seconds."],
        )

    if bin_size == 0 and processing != "raw":
        return make_error(
            "validation_error",
            f"bin_size=0 (full resolution) requires processing='raw' (got "
            f"processing={processing!r}); an aggregate has no bin to aggregate over.",
            [
                "Use processing='raw' with bin_size=0 for full resolution.",
                "Or choose a positive bin_size to use an aggregate processing mode.",
            ],
        )

    try:
        start_dt = _parse_time(start_time)
    except Exception as exc:
        return make_error(
            "validation_error",
            f"Could not parse start_time '{start_time}': {exc}",
            ["Use ISO-8601 format or relative expressions like '2h ago'."],
        )

    try:
        end_dt = _parse_time(end_time)
    except Exception as exc:
        return make_error(
            "validation_error",
            f"Could not parse end_time '{end_time}': {exc}",
            ["Use ISO-8601 format or 'now'."],
        )

    async with connector_error_handler("archiver_read", connector_name="archiver"):
        from osprey.mcp_server.control_system.server_context import get_server_context

        registry = get_server_context()
        connector = await registry.archiver()

        # Deduplicate before querying: a caller-repeated channel name must not
        # double-query the archiver, and every count derived below
        # (channels_queried, per_channel, total_points) must describe exactly
        # what was actually queried and returned — not the raw request list.
        unique_channels = list(dict.fromkeys(channels))

        precision_ms = 1000 if bin_size is None else bin_size * 1000
        df = await connector.get_data(
            unique_channels,
            start_dt,
            end_dt,
            precision_ms=precision_ms,
            processing=processing,
        )

        # Each channel has its own independent samples — filter the long
        # frame per channel rather than assuming a shared column/index.
        series: dict[str, dict[str, list[Any]]] = {}
        per_channel: dict[str, dict[str, Any]] = {}
        total_points = 0

        for ch in unique_channels:
            sub = df[df["channel"] == ch]
            timestamps = [ts.isoformat() for ts in sub["timestamp"]]
            values = [None if pd.isna(v) else v for v in sub["value"].tolist()]
            series[ch] = {"timestamps": timestamps, "values": values}

            points = len(sub)
            total_points += points

            stats: dict[str, Any] = {"points": points}
            # value may hold strings for enum/status channels; coerce before
            # computing stats so a non-numeric channel never raises or
            # reports NaN — it simply omits the numeric fields. "points"
            # counts every sample (including ones that turn out non-numeric);
            # min/max/mean below count only the numeric ones.
            numeric = pd.to_numeric(sub["value"], errors="coerce")
            if numeric.notna().any():
                stats["min"] = float(numeric.min())
                stats["max"] = float(numeric.max())
                stats["mean"] = round(float(numeric.mean()), 6)
            per_channel[ch] = stats

        # Full data payload goes to file; compact summary returned inline
        data_payload = {
            "query": {
                "channels": unique_channels,
                "start_time": str(start_dt),
                "end_time": str(end_dt),
                "processing": processing,
                "bin_size": bin_size,
            },
            "series": series,
        }

        summary = {
            "channels_queried": len(unique_channels),
            "total_points": total_points,
            "time_range": {"start": str(start_dt), "end": str(end_dt)},
            "per_channel": per_channel,
        }
        access_details = {
            "data_file_structure": {
                "root_keys": ["query", "series"],
                "series_format": (
                    "mapping of channel name to {'timestamps': [...], 'values': [...]}"
                ),
            },
            "schema": {
                "query": (
                    "the parameters this read actually used (after "
                    "deduplicating channels): channels, start_time/end_time, "
                    "processing, bin_size. start_time/end_time are "
                    "facility-local wall-clock strings (e.g. with a "
                    "'-08:00' offset) — NOT the UTC/ISO-8601 format used "
                    "below for series timestamps."
                ),
                "series": (
                    "dict keyed by channel name; each entry is {'timestamps': [...], "
                    "'values': [...]} — two parallel arrays of equal length holding "
                    "that channel's own real samples, timestamps ascending within "
                    "the channel. Channels have independent timestamps and sample "
                    "counts and are NOT aligned to each other or to any shared "
                    "index."
                ),
                "timestamps": (
                    "list of ISO-8601 UTC timestamp strings, one per sample, "
                    "ascending within the channel (series timestamps only — "
                    "see the query note above)"
                ),
                "values": (
                    "list of sample values, one per timestamp: numeric for "
                    "ordinary channels, the channel's own strings for "
                    "enum/status channels, or JSON null where the archived "
                    "sample itself was null. Non-numeric samples pass through "
                    "as their own values and are never nulled. A channel's "
                    "'points' count (and the summary's total) includes null "
                    "and non-numeric entries; min/max/mean are computed only "
                    "over the numeric values, so 'points' is not the count of "
                    "usable numbers."
                ),
            },
            "access_patterns": {
                ch: f'json_data["series"]["{ch}"]  ->  {{"timestamps": [...], "values": [...]}}'
                for ch in unique_channels
            },
            "total_points": total_points,
            "processing": processing,
            "bin_size": bin_size,
        }

        # Save via ArtifactStore (unified)
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.save_data(
            tool="archiver_read",
            data=data_payload,
            title=f"Archiver data for {len(unique_channels)} channel(s)",
            description=(
                f"Archiver data for {len(unique_channels)} channel(s), {total_points} points"
            ),
            summary=summary,
            access_details=access_details,
            category="archiver_data",
            metadata={"data_type": "timeseries"},
        )

        return json.dumps(entry.to_tool_response(), default=str)
