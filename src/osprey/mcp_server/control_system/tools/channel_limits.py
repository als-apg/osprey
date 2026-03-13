"""MCP tool: channel_limits — query the safety limits database.

Read-only metadata lookup. No connector or approval needed.
"""

import json
import logging
import re

from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.tools.channel_limits")

VALID_FILTERS = frozenset(
    {"writable", "read_only", "has_step_limit", "has_range", "readback_verified"}
)


def _build_summary(validator) -> dict:
    """Build database-level statistics from the validator."""
    total = len(validator.limits)
    writable = sum(1 for c in validator.limits.values() if c.writable)
    read_only = total - writable
    has_step = sum(1 for c in validator.limits.values() if c.max_step is not None)
    has_range = sum(
        1 for c in validator.limits.values() if c.min_value is not None or c.max_value is not None
    )

    # Verification breakdown from raw db
    verification_counts: dict[str, int] = {}
    for addr, cfg in validator._raw_db.items():
        if addr.startswith("_") or addr == "defaults":
            continue
        if not isinstance(cfg, dict):
            continue
        level = cfg.get("verification", {}).get("level", "default")
        verification_counts[level] = verification_counts.get(level, 0) + 1

    return {
        "status": "success",
        "description": f"Limits database: {total} channels configured",
        "summary": {
            "total_channels": total,
            "writable": writable,
            "read_only": read_only,
            "has_step_limit": has_step,
            "has_range": has_range,
            "verification_breakdown": verification_counts,
            "version": validator._raw_db.get("_version"),
        },
        "access_details": {
            "policy": validator.policy,
            "defaults": validator._raw_db.get("defaults"),
        },
    }


def _build_channel_entry(validator, channel_address: str) -> dict:
    """Build a detailed entry for a single known channel."""
    cfg = validator.limits[channel_address]
    entry: dict = {
        "writable": cfg.writable,
        "min_value": cfg.min_value,
        "max_value": cfg.max_value,
        "max_step": cfg.max_step,
    }
    # Merge verification config from raw db
    raw = validator._raw_db.get(channel_address, {})
    if isinstance(raw, dict) and "verification" in raw:
        entry["verification"] = raw["verification"]
    return entry


def _match_channels(validator, pattern: str | None, filter_by: str | None) -> dict:
    """Match channels by regex pattern and/or property filter. Returns compact entries."""
    addresses = list(validator.limits.keys())

    # Apply regex filter
    if pattern:
        addresses = [a for a in addresses if re.search(pattern, a)]

    # Apply property filter
    if filter_by:
        filtered = []
        for addr in addresses:
            cfg = validator.limits[addr]
            if filter_by == "writable" and cfg.writable:
                filtered.append(addr)
            elif filter_by == "read_only" and not cfg.writable:
                filtered.append(addr)
            elif filter_by == "has_step_limit" and cfg.max_step is not None:
                filtered.append(addr)
            elif filter_by == "has_range" and (
                cfg.min_value is not None or cfg.max_value is not None
            ):
                filtered.append(addr)
            elif filter_by == "readback_verified":
                raw = validator._raw_db.get(addr, {})
                if isinstance(raw, dict) and raw.get("verification", {}).get("level") == "readback":
                    filtered.append(addr)
        addresses = filtered

    # Build compact entries (flat verification_level instead of nested dict)
    results = {}
    for addr in addresses:
        cfg = validator.limits[addr]
        entry: dict = {
            "writable": cfg.writable,
            "min_value": cfg.min_value,
            "max_value": cfg.max_value,
            "max_step": cfg.max_step,
        }
        raw = validator._raw_db.get(addr, {})
        if isinstance(raw, dict):
            entry["verification_level"] = raw.get("verification", {}).get("level")
        results[addr] = entry

    return results


@mcp.tool()
async def channel_limits(
    channels: list[str] | None = None,
    pattern: str | None = None,
    filter_by: str | None = None,
) -> str:
    """Query the channel safety limits database.

    Proactively look up allowed ranges, step limits, and writability BEFORE
    attempting writes. This tool reads a local metadata file — no control
    system connection needed, no approval required.

    Modes (selected by parameter combination):
      - No params: summary statistics and policy overview
      - channels: detailed config for specific channel addresses
      - pattern: regex search across all channel addresses
      - pattern + filter_by: regex search filtered by property
      - filter_by alone: all channels matching a property

    Args:
        channels: Exact channel addresses to look up.
        pattern: Regex to match against channel addresses.
        filter_by: Property filter — one of: writable, read_only,
                   has_step_limit, has_range, readback_verified.

    Returns:
        JSON with channel limits configuration or database summary.
    """
    # Validate parameter combinations
    if channels is not None and pattern is not None:
        return json.dumps(
            make_error(
                "validation_error",
                "Cannot combine 'channels' (exact lookup) with 'pattern' (regex search).",
                ["Use 'channels' for exact addresses or 'pattern' for regex matching, not both."],
            )
        )

    if filter_by is not None and filter_by not in VALID_FILTERS:
        return json.dumps(
            make_error(
                "validation_error",
                f"Invalid filter_by value: {filter_by!r}",
                [f"Valid values: {', '.join(sorted(VALID_FILTERS))}"],
            )
        )

    # Validate regex before loading validator
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as exc:
            return json.dumps(
                make_error(
                    "validation_error",
                    f"Invalid regex pattern: {exc}",
                    ["Provide a valid Python regular expression."],
                )
            )

    # Load validator
    try:
        from osprey.connectors.control_system.limits_validator import LimitsValidator
    except ImportError:
        LimitsValidator = None  # type: ignore[assignment,misc]

    validator = None
    if LimitsValidator is not None:
        validator = LimitsValidator.from_config()

    if validator is None:
        return json.dumps(
            {
                "status": "success",
                "description": "Channel limits checking is not enabled in this configuration.",
                "summary": {"limits_enabled": False},
                "access_details": {
                    "note": "Enable limits_checking in config.yml to use this tool."
                },
            }
        )

    # Dispatch to the appropriate mode
    if channels is None and pattern is None and filter_by is None:
        # Summary mode
        return json.dumps(_build_summary(validator), default=str)

    if channels is not None:
        # Lookup mode
        results = {}
        for addr in channels:
            if addr in validator.limits:
                results[addr] = _build_channel_entry(validator, addr)
            else:
                # Channel not in database — show what the policy would do
                allow_unlisted = validator.policy.get("allow_unlisted_channels", True)
                if allow_unlisted:
                    results[addr] = {
                        "in_database": False,
                        "policy_action": "allowed (no limits enforced)",
                    }
                else:
                    results[addr] = {
                        "in_database": False,
                        "policy_action": "BLOCKED (unlisted channels not allowed)",
                    }

        return json.dumps(
            {
                "status": "success",
                "description": f"Limits for {len(channels)} channel(s)",
                "summary": {"channels_queried": len(channels), "channels_found": len(results)},
                "access_details": {"channels": results},
            },
            default=str,
        )

    # Search / Filter mode (pattern and/or filter_by)
    matched = _match_channels(validator, pattern, filter_by)
    desc_parts = []
    if pattern:
        desc_parts.append(f"pattern={pattern!r}")
    if filter_by:
        desc_parts.append(f"filter={filter_by}")

    return json.dumps(
        {
            "status": "success",
            "description": f"Search ({', '.join(desc_parts)}): {len(matched)} match(es)",
            "summary": {"matches": len(matched)},
            "access_details": {"channels": matched},
        },
        default=str,
    )
