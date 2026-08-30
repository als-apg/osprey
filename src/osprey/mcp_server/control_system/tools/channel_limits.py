"""MCP tool: channel_limits — query the safety limits database.

Read-only metadata lookup. No connector or approval needed.
"""

import json
import logging
import re

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.tools.channel_limits")

VALID_FILTERS = frozenset({"writable", "read_only", "has_step_limit", "has_range"})

#: The key a posture that names no connector type was read from. A validator
#: built from a bare policy dict carries no key of its own; this is the honest
#: answer for it, and it is the same fallback ``LimitsValidator.validate``
#: quotes when it refuses such a write.
DEPLOYMENT_WIDE_UNLISTED_KEY = "control_system.limits_checking.allow_unlisted_channels"


def _session_target() -> str | None:
    """The control target this server is on, or ``None`` if there is no record.

    ``None`` is the single answer for every "there is no usable target" case —
    no state file, an unreadable state directory, a record whose target is not
    a non-empty string. Every one of them means the same thing to the caller:
    ask :meth:`LimitsValidator.from_config` without a target and get the
    deployment-wide block, which is the posture a target that resolves to
    nothing gets anyway.

    Reading is wrapped because :func:`target_state.state_dir` resolves a shared
    data root and can raise, and a read-only metadata lookup must not fail on
    the way to reporting what the deployment allows.
    """
    try:
        record = target_state.read()
    except Exception:  # noqa: BLE001 - a target that cannot be read is simply absent
        logger.debug("Could not read the control-system target state", exc_info=True)
        return None
    if not isinstance(record, dict):
        return None
    target = record.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def _unlisted_policy(validator) -> tuple[bool | None, str]:
    """The unlisted-channel posture as reported, plus the key that answered.

    The value is the validator's own tri-state, verbatim and with no default:
    ``True`` allows an unlisted channel, ``False`` refuses it, and ``None``
    means no key states an answer — which also refuses. Defaulting an unstated
    answer to ``True`` here would tell an operator their deployment permits
    writes that every write path in fact blocks.

    Returns:
        An ``(allow_unlisted, answering_key)`` pair.
    """
    allow_unlisted = validator.policy.get("allow_unlisted_channels")
    answering_key = validator.policy.get("allow_unlisted_key") or DEPLOYMENT_WIDE_UNLISTED_KEY
    return allow_unlisted, answering_key


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

    allow_unlisted, answering_key = _unlisted_policy(validator)

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
            # reasons about are always present and always the tri-state, even
            # for a validator whose policy dict was hand-built.
            "policy": {
                **validator.policy,
                "allow_unlisted_channels": allow_unlisted,
                "allow_unlisted_key": answering_key,
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
    """Query the channel safety limits database.

    Proactively look up allowed ranges, step limits, and writability BEFORE
    attempting writes. This tool reads a local metadata file — no control
    system connection needed, no approval required.

    Modes (selected by parameter combination):
      - No params: summary statistics and policy overview
      - channels: detailed config for specific channel addresses
      - pattern: regex search across all channel addresses
      - name_contains: literal substring search across all channel addresses
      - pattern/name_contains + filter_by: search filtered by property
      - filter_by alone: all channels matching a property

    Args:
        channels: Exact channel addresses to look up.
        pattern: Regex to match against channel addresses.
        name_contains: Literal substring to match against channel addresses. Use this for
                       names containing regex metacharacters such as [], (), ., or ^.
        filter_by: Property filter — one of: writable, read_only,
                   has_step_limit, has_range.

    Returns:
        JSON with channel limits configuration or database summary. The
        reported ``allow_unlisted_channels`` is the posture of the control
        target this session is on and may be ``null`` — no config key states
        an answer, and unlisted channels are refused; ``allow_unlisted_key``
        names the key that answered.
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
        # The posture reported is the one this session writes under: a
        # deployment may relax unlisted channels for its virtual accelerator
        # alone, and reporting the deployment-wide answer on a VA session would
        # describe a machine the caller is not on.
        validator = LimitsValidator.from_config(target=_session_target())

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
        allow_unlisted, answering_key = _unlisted_policy(validator)
        results = {}
        for addr in channels:
            if addr in validator.limits:
                results[addr] = _build_channel_entry(validator, addr)
            else:
                # Channel not in database — show what the policy would do.
                # Only an explicit True is permission, exactly as the validator
                # decides it: an unstated answer refuses, and says which key is
                # unstated rather than reporting a write that would be blocked.
                results[addr] = {
                    "in_database": False,
                    "allow_unlisted_channels": allow_unlisted,
                    "allow_unlisted_key": answering_key,
                    "policy_action": (
                        "allowed (no limits enforced)"
                        if allow_unlisted is True
                        else f"BLOCKED ('{answering_key}' does not allow unlisted channels)"
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
