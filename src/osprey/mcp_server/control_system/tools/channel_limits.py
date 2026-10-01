"""MCP tool: channel_limits — query the write-safety limits database.

Read-only metadata lookup. No connector or approval needed. This is a database
of which channels are WRITE-GATED and under what constraints — it answers
writability, never membership. A channel absent from it is simply unconfigured
for limits checking, not nonexistent; discovering what channels exist on a
deployment is the channel-finder server's job, not this one's.
"""

import json
import logging
import re

from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error
from osprey_connectors import control_context

logger = logging.getLogger("osprey.mcp_server.tools.channel_limits")

VALID_FILTERS = frozenset({"writable", "read_only", "has_step_limit", "has_range"})

#: The key a posture that names no connector type was read from. A validator
#: built from a bare policy dict carries no key of its own; this is the honest
#: answer for it, and it is the same fallback ``LimitsValidator.validate``
#: quotes when it refuses such a write.
DEPLOYMENT_WIDE_MODE_KEY = "control_system.limits_checking.mode"


def _record_target() -> str | None:
    """The control target the deployment's record names, or ``None`` without one.

    Read from :func:`osprey_connectors.control_context.read_record` — the same
    record ``channel_write`` and the approval prompt read. Asking one record
    the same way everywhere is what makes the limits this tool reports the
    limits a write to that target is actually checked against.

    ``None`` is the single answer for every "there is no usable target" case —
    no record, an unreadable one, a record whose target is not a non-empty
    string. Every one of them means the same thing to the caller: ask
    :meth:`LimitsValidator.from_config` without a target and get the
    deployment-wide block, which is the posture a target that resolves to
    nothing gets anyway.

    Reading is wrapped because resolving the shared agent-data root can raise,
    and a read-only metadata lookup must not fail on the way to reporting what
    the deployment allows.
    """
    try:
        record = control_context.read_record()
    except Exception:  # a target that cannot be read is simply absent
        logger.debug("Could not read the control-context record", exc_info=True)
        return None
    if record is None:
        return None
    target = record.target
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def _mode_policy(validator) -> tuple[str | None, str]:
    """The limits mode as reported, plus the key that answered.

    The value is the validator's own, verbatim and with no default:
    ``optional`` writes a channel with no record with no limits, ``exclusive``
    refuses it, and ``None`` means no key states a mode — which also refuses.
    Defaulting an unstated answer to ``optional`` here would tell an operator
    their deployment permits writes that every write path in fact blocks.

    Returns:
        A ``(mode, answering_key)`` pair.
    """
    mode = validator.policy.get("mode")
    answering_key = validator.policy.get("mode_key") or DEPLOYMENT_WIDE_MODE_KEY
    return mode, answering_key


def _build_summary(validator) -> dict:
    """Build database-level statistics from the validator."""
    total = len(validator.limits)
    writable = sum(1 for c in validator.limits.values() if c.writable)
    read_only = total - writable
    has_step = sum(1 for c in validator.limits.values() if c.max_step is not None)
    has_range = sum(
        1 for c in validator.limits.values() if c.min_value is not None or c.max_value is not None
    )

    # How many channels resolve to confirmed writes
    confirmed = sum(1 for addr in validator.limits if validator.resolve_confirm(addr))

    mode, answering_key = _mode_policy(validator)

    return {
        "status": "success",
        "description": f"Limits database: {total} channels configured",
        "summary": {
            "total_channels": total,
            "writable": writable,
            "read_only": read_only,
            "has_step_limit": has_step,
            "has_range": has_range,
            "confirm_breakdown": {"true": confirmed, "false": total - confirmed},
            "version": validator._raw_db.get("_version"),
        },
        "access_details": {
            # The posture's own fields are restated so that the two the caller
            # reasons about are always present and always the stated value, even
            # for a validator whose policy dict was hand-built.
            "policy": {
                **validator.policy,
                "mode": mode,
                "mode_key": answering_key,
            },
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
        "confirm": validator.resolve_confirm(channel_address),
    }
    return entry


def _match_channels(
    validator,
    pattern: str | None,
    name_contains: str | None,
    filter_by: str | None,
) -> dict:
    """Match channels by regex pattern and/or property filter. Returns compact entries."""
    addresses = list(validator.limits.keys())

    # Apply regex filter
    if pattern:
        addresses = [a for a in addresses if re.search(pattern, a)]

    # Apply literal substring filter
    if name_contains:
        addresses = [a for a in addresses if name_contains in a]

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
        addresses = filtered

    # Build compact entries
    results = {}
    for addr in addresses:
        cfg = validator.limits[addr]
        results[addr] = {
            "writable": cfg.writable,
            "min_value": cfg.min_value,
            "max_value": cfg.max_value,
            "max_step": cfg.max_step,
            "confirm": validator.resolve_confirm(addr),
        }

    return results


@mcp.tool()
async def channel_limits(
    channels: list[str] | None = None,
    pattern: str | None = None,
    name_contains: str | None = None,
    filter_by: str | None = None,
) -> str:
    """Query the channel write-safety limits database.

    Proactively look up allowed ranges, step limits, and writability BEFORE
    attempting writes. This tool reads a local metadata file — no control
    system connection needed, no approval required.

    This database holds only the channels a deployment has configured limits
    for; it is not a roster of what channels exist. A channel this tool
    cannot find may still be perfectly real — it simply has no limits
    record, and the deployment's limits ``mode`` decides whether it can be
    written. To discover what channels exist, use the channel-finder
    server where one is configured for this deployment.

    Modes (selected by parameter combination):
      - No params: summary statistics and policy overview
      - channels: detailed config for specific channel addresses
      - pattern: regex search across configured channel addresses
      - name_contains: literal substring search across configured channel addresses
      - pattern/name_contains + filter_by: search filtered by property
      - filter_by alone: all configured channels matching a property

    Args:
        channels: Exact channel addresses to look up.
        pattern: Regex to match against configured channel addresses.
        name_contains: Literal substring to match against configured channel addresses.
                       Use this for names containing regex metacharacters such as
                       [], (), ., or ^.
        filter_by: Property filter — one of: writable, read_only,
                   has_step_limit, has_range.

    Returns:
        JSON with channel limits configuration or database summary. The
        reported ``mode`` is the limits mode of the control target this
        deployment is on: ``exclusive`` (only channels in the database can be
        written), ``optional`` (a channel with no record is written with no
        limits) or ``null`` — no config key states a mode, and a channel with
        no record is refused; ``mode_key`` names the key that answered.
    """
    # Validate parameter combinations
    if channels is not None and (pattern is not None or name_contains is not None):
        return make_error(
            "validation_error",
            "Cannot combine 'channels' (exact lookup) with search parameters.",
            [
                "Use 'channels' for exact addresses, 'pattern' for regex matching, "
                "or 'name_contains' for literal substring matching."
            ],
        )

    if pattern is not None and name_contains is not None:
        return make_error(
            "validation_error",
            "Cannot combine 'pattern' (regex search) with 'name_contains' (literal search).",
            ["Use either 'pattern' or 'name_contains', not both."],
        )

    if filter_by is not None and filter_by not in VALID_FILTERS:
        return make_error(
            "validation_error",
            f"Invalid filter_by value: {filter_by!r}",
            [f"Valid values: {', '.join(sorted(VALID_FILTERS))}"],
        )

    # Validate regex before loading validator
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as exc:
            return make_error(
                "validation_error",
                f"Invalid regex pattern: {exc}",
                ["Provide a valid Python regular expression."],
            )

    # Load validator
    try:
        from osprey.connectors.control_system.limits_validator import LimitsValidator
    except ImportError:
        LimitsValidator = None  # type: ignore[assignment,misc]

    validator = None
    if LimitsValidator is not None:
        # The posture reported is the one a write would land under: a
        # deployment may run its virtual accelerator alone in the optional
        # mode, and reporting the deployment-wide answer while the record is
        # on VA would describe a machine the caller is not pointed at.
        validator = LimitsValidator.from_config(target=_record_target())

    if validator is None:
        return json.dumps(
            {
                "status": "success",
                "description": "Channel limits checking is not enabled in this configuration.",
                "summary": {"limits_enabled": False},
                "access_details": {
                    "note": "Enable limits_checking in the build profile (profile.yml on the host), then rebuild and redeploy to use this tool."
                },
            }
        )

    # Dispatch to the appropriate mode
    if channels is None and pattern is None and name_contains is None and filter_by is None:
        # Summary mode
        return json.dumps(_build_summary(validator), default=str)

    if channels is not None:
        # Lookup mode
        mode, answering_key = _mode_policy(validator)
        results = {}
        for addr in channels:
            if addr in validator.limits:
                results[addr] = _build_channel_entry(validator, addr)
            else:
                # Channel not in database — show what the policy would do.
                # Only an explicit `optional` is permission, exactly as the validator
                # decides it: an unstated answer refuses, and says which key is
                # unstated rather than reporting a write that would be blocked.
                results[addr] = {
                    "in_database": False,
                    "mode": mode,
                    "mode_key": answering_key,
                    "policy_action": (
                        "allowed (no limits enforced)"
                        if mode == "optional"
                        else f"BLOCKED ('{answering_key}' is not 'optional')"
                    ),
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

    # Search / Filter mode (pattern/name_contains and/or filter_by)
    matched = _match_channels(validator, pattern, name_contains, filter_by)
    desc_parts = []
    if pattern:
        desc_parts.append(f"pattern={pattern!r}")
    if name_contains:
        desc_parts.append(f"name_contains={name_contains!r}")
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
