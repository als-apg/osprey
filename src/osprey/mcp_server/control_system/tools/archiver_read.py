"""MCP tool: archiver_read — retrieve historical data from the archiver."""

import json
import logging
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

import pandas as pd

from osprey.connectors.archiver import PROCESSING_MODES
from osprey.mcp_server.control_system import target_banner
from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error
from osprey.utils.config import get_facility_timezone
from osprey_connectors.types import resolve_archiver_type

logger = logging.getLogger("osprey.mcp_server.tools.archiver_read")

#: Points per channel the default bin aims for when the caller names no
#: ``bin_size``; ``archiver.auto_bin_points`` in config.yml overrides it.
DEFAULT_AUTO_BIN_POINTS = 10_000

#: ``target_source`` when the session has selected a target of its own — the
#: state file named it, and it differs from the deployment baseline.
TARGET_SOURCE_SESSION = "session_switch"

#: ``target_source`` when no session selection was in force: no state file, an
#: unreadable one, or a session sitting on the baseline. All three mean the same
#: thing about the data — it was read while the deployment baseline was current
#: — and saying "baseline" is the honest spelling of all three. The key is
#: always present, so a reader never has to guess from an absent one.
TARGET_SOURCE_BASELINE = "baseline"


def auto_bin_seconds(span: timedelta, target_points: int) -> int:
    """The whole-second bin that keeps a continuously archived channel near *target_points*.

    ``ceil(span / target_points)``, never below one second: 1 s for anything
    up to the budget (2.8 h at 10 000), 9 s for a day, ~53 min for a year. A
    degenerate or inverted window gets the smallest legal bin rather than an
    exception — the read itself reports the empty window honestly.

    Whole seconds because the EPICS Archiver Appliance only bins in whole
    seconds (``lastSample_N``); a sub-second width would be refused there.
    Rounding up, never down, keeps the result under budget.

    Widening the bin loses nothing on a slowly or intermittently archived
    channel: ``raw`` keeps one real sample per bin and an empty bin yields no
    row, so the channels issue #117 was filed about come back exactly as before.
    """
    seconds = span.total_seconds()
    if seconds <= 0:
        return 1
    return max(1, math.ceil(seconds / target_points))


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
        kwarg = unit_map.get(amount_unit[-1:])
        if kwarg is not None:
            try:
                return datetime.now(tz) - timedelta(**{kwarg: float(amount_unit[:-1])})
            except ValueError:
                pass  # unparseable amount — fall through to dateutil below

    # Fall back to dateutil for ISO / human strings
    from dateutil import parser as dateutil_parser

    dt = dateutil_parser.parse(time_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


class _CoverageProbe(NamedTuple):
    """What the connector could say about one empty channel."""

    available: bool | None
    metadata: Any  # ArchiverMetadata | None — untyped to keep the import lazy
    note: str | None


def _pin_utc(dt: datetime | None) -> datetime | None:
    """pymongo reads naive datetimes as UTC; make that explicit before comparing."""
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _classify_coverage(
    start_utc: datetime, end_utc: datetime, probe: _CoverageProbe
) -> dict[str, Any]:
    """One empty channel's entry: a verdict, plus bounds when they are known."""
    md = probe.metadata
    if probe.available is False or (md is not None and not md.is_archived):
        return {"verdict": "never_recorded"}
    a_start = _pin_utc(md.archival_start) if md is not None else None
    a_end = _pin_utc(md.archival_end) if md is not None else None
    if a_start is None or a_end is None:
        entry: dict[str, Any] = {"verdict": "coverage_unknown"}
        if probe.note:
            entry["note"] = probe.note
        return entry
    if end_utc < a_start:
        verdict = "window_precedes_archive"
    elif start_utc > a_end:
        verdict = "window_follows_archive"
    else:
        verdict = "gap_within_coverage"
    return {
        "verdict": verdict,
        "archive_start": a_start.isoformat(),
        "archive_end": a_end.isoformat(),
    }


def _coverage_message(
    start_utc: datetime, end_utc: datetime, overall: str, channels: dict[str, dict[str, Any]]
) -> str:
    """Plain language for the agent: the asked window, the real bounds, no guessing."""
    window = f"{start_utc.isoformat()} → {end_utc.isoformat()}"
    n = len(channels)
    noun = "1 channel" if n == 1 else f"{n} channels"
    if overall == "window_precedes_archive":
        oldest = min(entry["archive_start"] for entry in channels.values())
        return (
            f"No archived data for {noun} in {window}: archiving begins {oldest}, "
            f"after the queried window ends. The archive reports only what it holds — "
            f"nothing is extrapolated. Data exists from {oldest} onward."
        )
    if overall == "window_follows_archive":
        newest = max(entry["archive_end"] for entry in channels.values())
        return (
            f"No archived data for {noun} in {window}: the archive's newest sample is "
            f"{newest}, before the queried window begins. If this deployment records "
            f"continuously, recording may have stopped."
        )
    if overall == "never_recorded":
        return (
            f"Not in this archive: {', '.join(channels)}. No history was ever "
            f"recorded for {'this channel' if n == 1 else 'these channels'}."
        )
    if overall == "gap_within_coverage":
        return (
            f"The queried window {window} lies inside archive coverage but holds no "
            f"samples for {noun}: nothing was recorded there. A gap is recorded "
            f"silence — the channel was not answering — not data awaiting synthesis."
        )
    if overall == "coverage_unknown":
        return (
            f"No data in {window} for {noun}, and this archiver backend reports no "
            f"coverage bounds, so the reason cannot be determined from here."
        )
    counts = Counter(entry["verdict"] for entry in channels.values())
    parts = ", ".join(f"{count}× {verdict}" for verdict, count in counts.items())
    return (
        f"No archived data in {window} for {noun}, for differing reasons: {parts}. "
        f"See coverage.channels for each channel's verdict and bounds."
    )


async def _probe_coverage(connector: Any, channels: list[str]) -> dict[str, _CoverageProbe]:
    """Ask the connector about each empty channel; a failed probe is a note.

    Failures degrade to ``coverage_unknown`` downstream rather than raising:
    the data that DID come back must never be lost to its own explanation.
    """
    try:
        availability = await connector.check_availability(channels)
    except Exception:  # noqa: BLE001 — any probe failure degrades, none propagate
        availability = {}
    probes: dict[str, _CoverageProbe] = {}
    for ch in channels:
        try:
            md = await connector.get_metadata(ch)
        except Exception as exc:  # noqa: BLE001
            probes[ch] = _CoverageProbe(
                available=availability.get(ch), metadata=None, note=f"metadata probe failed: {exc}"
            )
        else:
            probes[ch] = _CoverageProbe(available=availability.get(ch), metadata=md, note=None)
    return probes


def _compose_coverage(
    start_utc: datetime, end_utc: datetime, probes: dict[str, _CoverageProbe]
) -> dict[str, Any] | None:
    """Explain the empty channels of a read, from facts — or admit not knowing.

    Returns ``None`` when nothing was empty (the common path adds no block, no
    tokens, no schema noise). Never raises: an explanation that failed to
    compose must not cost the agent the data that DID come back.
    """
    if not probes:
        return None
    channels = {ch: _classify_coverage(start_utc, end_utc, probe) for ch, probe in probes.items()}
    verdicts = {entry["verdict"] for entry in channels.values()}
    overall = next(iter(verdicts)) if len(verdicts) == 1 else "mixed"
    return {
        "verdict": overall,
        "message": _coverage_message(start_utc, end_utc, overall, channels),
        "channels": channels,
    }


def _provenance(archiver_section: Any, connector: Any) -> dict[str, str]:
    """Who served this read: the session's control-system target and the archiver.

    Four keys, always all four, stamped identically onto the saved query block
    and the artifact's metadata — the ``bin_size_source`` rule applied to
    identity: a fact the tool knows and the reader cannot recover later has to
    travel with the data.

    * ``target`` / ``target_source`` — the control-system target the session was
      on when the read ran. Archived data is not routed through that target (see
      :func:`archiver_read`), but a plot of it is read next to live values from
      one, so which one was current is part of what the numbers mean. The target
      comes from the shared resolver in
      :mod:`~osprey.mcp_server.control_system.target_banner`, never from a second
      reading of the state file; a session on the baseline (or with no readable
      state at all) stamps :data:`TARGET_SOURCE_BASELINE`.
    * ``archiver_type`` — the configured archiver type, resolved by the same
      :func:`~osprey_connectors.types.resolve_archiver_type` the factory uses, so
      the stamp cannot disagree with what was actually built.
    * ``archiver_backend`` — the connector class that served it. The class name
      and nothing else: an endpoint or connection string would put a host — or a
      credential — into an artifact that outlives the session.

    Args:
        archiver_section: The ``archiver`` config block.
        connector: The archiver connector instance that served this read.
    """
    situation = target_banner.resolve_target_situation()
    return {
        "target": situation.session_target,
        "target_source": TARGET_SOURCE_SESSION if situation.switched else TARGET_SOURCE_BASELINE,
        "archiver_type": resolve_archiver_type(archiver_section),
        "archiver_backend": type(connector).__name__,
    }


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
        bin_size: Bin size in seconds. ``None`` (the default) picks a
            whole-second bin from the span so a continuously archived channel
            returns at most about 10 000 points (``archiver.auto_bin_points``
            in config.yml): 1 s for anything under ~2.8 hours, 9 s for a day,
            ~53 minutes for a year. The bin actually used is reported as
            ``summary.bin_size`` with ``summary.bin_size_source`` ``"auto"``;
            pass ``bin_size`` explicitly to override it. ``0`` requests full
            resolution — every real archived sample in range, with no per-bin
            decimation — and is only valid with processing="raw" (an aggregate
            has no bin to aggregate over). Negative values are rejected.

    Returns:
        JSON summary with per-channel point counts and stats, the bin that was
        used and whether it was requested or chosen automatically, and the
        data file path. When a requested channel has zero points in the
        window, ``summary.coverage`` explains why — the window precedes/follows
        the archive's real bounds, the channel was never recorded, or the
        window holds an honest gap — so an empty answer is never a silent one.
        The saved query block and the artifact's metadata additionally carry a
        provenance stamp naming the session's control-system target and the
        archiver that served the data.
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

        # The connector contract forbids a backend from serving a width other
        # than the one asked for, so the tool — the only layer that knows the
        # span — is where a default bin gets chosen. Having chosen it, it says
        # so in the summary: an auto-picked parameter the agent cannot see
        # would be the dishonest version of this feature.
        if bin_size is None:
            target_points = registry.get("archiver.auto_bin_points", DEFAULT_AUTO_BIN_POINTS)
            if (
                isinstance(target_points, bool)
                or not isinstance(target_points, int)
                or target_points < 1
            ):
                return make_error(
                    "configuration_error",
                    f"archiver.auto_bin_points must be a positive integer (got {target_points!r}).",
                    [
                        "Fix archiver.auto_bin_points in config.yml, or remove it to use "
                        f"the default of {DEFAULT_AUTO_BIN_POINTS}.",
                    ],
                )
            resolved_bin = auto_bin_seconds(end_dt - start_dt, target_points)
            bin_source = "auto"
        else:
            resolved_bin = bin_size
            bin_source = "requested"

        # The archiver connector is HTTP/pymongo-class — an appliance or a
        # database, never Channel Access — so this read is NOT routed through
        # the connector-host child that serves the control system, and must
        # never be gated on that child being alive: history stays readable
        # exactly when a diagnosis needs it most. The only thing the session
        # target contributes here is provenance, stamped below.
        connector = await registry.archiver()
        provenance = _provenance(registry.config.archiver, connector)

        # Deduplicate: all counts below must describe what was actually queried.
        unique_channels = list(dict.fromkeys(channels))

        df = await connector.get_data(
            unique_channels,
            start_dt,
            end_dt,
            precision_ms=resolved_bin * 1000,
            processing=processing,
        )

        series: dict[str, dict[str, list[Any]]] = {}
        per_channel: dict[str, dict[str, Any]] = {}
        total_points = 0

        by_channel = dict(tuple(df.groupby("channel", sort=False)))
        no_samples = df.iloc[:0]

        # Iterate requested channels so key order matches the request and a
        # channel with no data still gets an empty entry.
        for ch in unique_channels:
            sub = by_channel.get(ch, no_samples)
            timestamps = [ts.isoformat() for ts in sub["timestamp"]]
            # NaN is not valid JSON; ``.astype(object)`` first is required or
            # ``where`` puts NaN back in place of None on float columns.
            values = sub["value"].astype(object).where(sub["value"].notna(), None).tolist()
            series[ch] = {"timestamps": timestamps, "values": values}

            points = len(sub)
            total_points += points

            stats: dict[str, Any] = {"points": points}
            # Enum/status channels hold strings: "points" counts every sample,
            # min/max/mean cover only the numeric ones.
            numeric = pd.to_numeric(sub["value"], errors="coerce")
            if numeric.notna().any():
                stats["min"] = float(numeric.min())
                stats["max"] = float(numeric.max())
                stats["mean"] = round(float(numeric.mean()), 6)
            per_channel[ch] = stats

        empty_channels = [ch for ch in unique_channels if per_channel[ch]["points"] == 0]
        coverage = None
        if empty_channels:
            probes = await _probe_coverage(connector, empty_channels)
            coverage = _compose_coverage(start_dt.astimezone(UTC), end_dt.astimezone(UTC), probes)

        # Full data payload goes to file; compact summary returned inline
        data_payload = {
            "query": {
                "channels": unique_channels,
                "start_time": str(start_dt),
                "end_time": str(end_dt),
                "processing": processing,
                "bin_size": resolved_bin,
                "bin_size_source": bin_source,
                **provenance,
            },
            "series": series,
        }

        summary = {
            "channels_queried": len(unique_channels),
            "total_points": total_points,
            "time_range": {"start": str(start_dt), "end": str(end_dt)},
            "bin_size": resolved_bin,
            "bin_size_source": bin_source,
            "per_channel": per_channel,
        }
        if coverage is not None:
            summary["coverage"] = coverage
        access_details = {
            "data_file_structure": {
                "root_keys": ["query", "series"],
                "series_format": (
                    "mapping of channel name to {'timestamps': [...], 'values': [...]}"
                ),
            },
            "access_pattern": (
                'json_data["series"]["<channel>"]  ->  {"timestamps": [...], "values": [...]}'
                " — one entry per requested channel; the channel names are the keys of"
                " summary.per_channel"
            ),
            "schema": {
                "query": (
                    "the parameters this read actually used, after deduplicating "
                    "channels. start_time/end_time are facility-local wall-clock "
                    "strings (e.g. with a '-08:00' offset) — NOT the UTC used for "
                    "series timestamps. It also carries the provenance stamp: "
                    "'target'/'target_source' name the control-system target the "
                    "session was on, and 'archiver_type'/'archiver_backend' name "
                    "the archiver that served the data."
                ),
                "series": (
                    "each channel's own real samples as two parallel arrays of equal "
                    "length, timestamps ascending within the channel. Channels are "
                    "NOT aligned to each other: independent timestamps and counts."
                ),
                "timestamps": "ISO-8601 UTC strings, one per sample.",
                "values": (
                    "one real sample per timestamp: numeric, the channel's own string "
                    "for an enum/status channel, or JSON null where the archived "
                    "sample itself was null. 'points' counts all of them; min/max/mean "
                    "are computed over the numeric ones only."
                ),
            },
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
            metadata={"data_type": "timeseries", **provenance},
        )

        return json.dumps(entry.to_tool_response(), default=str)
