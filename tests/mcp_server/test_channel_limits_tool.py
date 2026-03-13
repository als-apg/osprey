"""Tests for the channel_limits MCP tool.

Covers: summary mode, exact lookup (found/not-found/mixed), regex search,
property filters, combined search+filter, parameter validation, and limits-disabled.
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from tests.mcp_server.conftest import get_tool_fn

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

TEST_LIMITS_DB = {
    "_version": "1.0",
    "defaults": {"writable": True, "verification": {"level": "callback"}},
    "MAG:HCM01:CURRENT:SP": {
        "min_value": -10.0,
        "max_value": 10.0,
        "max_step": 2.0,
        "writable": True,
        "verification": {"level": "readback", "tolerance_percent": 0.1},
    },
    "MAG:QF01:CURRENT:SP": {"writable": False},
    "DIAG:TEMP:SP": {
        "min_value": 0.0,
        "max_value": 100.0,
        "writable": True,
    },
}


@dataclass
class FakeChannelLimitsConfig:
    channel_address: str
    min_value: float | None = None
    max_value: float | None = None
    max_step: float | None = None
    writable: bool = True


def _make_validator(allow_unlisted: bool = True) -> MagicMock:
    """Build a mock LimitsValidator from TEST_LIMITS_DB."""
    validator = MagicMock()
    validator._raw_db = TEST_LIMITS_DB

    limits = {}
    for addr, cfg in TEST_LIMITS_DB.items():
        if addr.startswith("_") or addr == "defaults" or not isinstance(cfg, dict):
            continue
        limits[addr] = FakeChannelLimitsConfig(
            channel_address=addr,
            min_value=cfg.get("min_value"),
            max_value=cfg.get("max_value"),
            max_step=cfg.get("max_step"),
            writable=cfg.get("writable", True),
        )
    validator.limits = limits
    validator.policy = {
        "allow_unlisted_channels": allow_unlisted,
        "on_violation": "error",
    }
    return validator


def _get_channel_limits():
    from osprey.mcp_server.control_system.tools.channel_limits import channel_limits

    return get_tool_fn(channel_limits)


# ---------------------------------------------------------------------------
# Summary mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_summary_mode():
    """No params → stats, policy, defaults, version."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn()

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["total_channels"] == 3
    assert data["summary"]["writable"] == 2
    assert data["summary"]["read_only"] == 1
    assert data["summary"]["has_step_limit"] == 1
    assert data["summary"]["version"] == "1.0"
    assert data["access_details"]["policy"]["allow_unlisted_channels"] is True
    assert data["access_details"]["defaults"]["writable"] is True


# ---------------------------------------------------------------------------
# Lookup mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_lookup_found():
    """Single known channel → full config with verification."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["MAG:HCM01:CURRENT:SP"])

    data = json.loads(result)
    assert data["status"] == "success"
    ch = data["access_details"]["channels"]["MAG:HCM01:CURRENT:SP"]
    assert ch["writable"] is True
    assert ch["min_value"] == -10.0
    assert ch["max_value"] == 10.0
    assert ch["max_step"] == 2.0
    assert ch["verification"]["level"] == "readback"


@pytest.mark.unit
async def test_lookup_not_found_blocked():
    """Unknown channel + allow_unlisted=false → BLOCKED."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(allow_unlisted=False),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["UNKNOWN:PV"])

    data = json.loads(result)
    assert data["status"] == "success"
    ch = data["access_details"]["channels"]["UNKNOWN:PV"]
    assert ch["in_database"] is False
    assert "BLOCKED" in ch["policy_action"]


@pytest.mark.unit
async def test_lookup_not_found_allowed():
    """Unknown channel + allow_unlisted=true → allowed."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(allow_unlisted=True),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["UNKNOWN:PV"])

    data = json.loads(result)
    ch = data["access_details"]["channels"]["UNKNOWN:PV"]
    assert ch["in_database"] is False
    assert "allowed" in ch["policy_action"]


@pytest.mark.unit
async def test_lookup_multiple_mixed():
    """Mix of found + not-found channels."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(allow_unlisted=False),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["MAG:HCM01:CURRENT:SP", "NONEXISTENT:PV"])

    data = json.loads(result)
    channels = data["access_details"]["channels"]
    assert channels["MAG:HCM01:CURRENT:SP"]["writable"] is True
    assert channels["NONEXISTENT:PV"]["in_database"] is False
    assert "BLOCKED" in channels["NONEXISTENT:PV"]["policy_action"]


@pytest.mark.unit
async def test_read_only_channel_details():
    """Read-only channel shows writable: false."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["MAG:QF01:CURRENT:SP"])

    data = json.loads(result)
    ch = data["access_details"]["channels"]["MAG:QF01:CURRENT:SP"]
    assert ch["writable"] is False


# ---------------------------------------------------------------------------
# Search / pattern mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_pattern_match():
    """MAG:.* → matches 2 MAG channels."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(pattern="MAG:.*")

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["matches"] == 2
    assert "MAG:HCM01:CURRENT:SP" in data["access_details"]["channels"]
    assert "MAG:QF01:CURRENT:SP" in data["access_details"]["channels"]


@pytest.mark.unit
async def test_pattern_no_match():
    """Non-matching pattern → empty results, still success."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(pattern="NONEXISTENT:.*")

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["matches"] == 0


@pytest.mark.unit
async def test_pattern_invalid_regex():
    """Invalid regex → validation_error."""
    fn = _get_channel_limits()
    result = await fn(pattern="[invalid")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "regex" in data["error_message"].lower()


# ---------------------------------------------------------------------------
# Filter mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_filter_writable():
    """filter_by=writable → writable channels only."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(filter_by="writable")

    data = json.loads(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 2
    assert all(ch["writable"] is True for ch in channels.values())


@pytest.mark.unit
async def test_filter_read_only():
    """filter_by=read_only → read-only channels only."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(filter_by="read_only")

    data = json.loads(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 1
    assert "MAG:QF01:CURRENT:SP" in channels


@pytest.mark.unit
async def test_filter_has_step_limit():
    """filter_by=has_step_limit → channels with max_step."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(filter_by="has_step_limit")

    data = json.loads(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 1
    assert "MAG:HCM01:CURRENT:SP" in channels


@pytest.mark.unit
async def test_filter_readback_verified():
    """filter_by=readback_verified → channels with readback verification."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(filter_by="readback_verified")

    data = json.loads(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 1
    assert "MAG:HCM01:CURRENT:SP" in channels


# ---------------------------------------------------------------------------
# Combined pattern + filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_combined_pattern_and_filter():
    """pattern + filter_by → intersection."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(pattern="MAG:.*", filter_by="writable")

    data = json.loads(result)
    channels = data["access_details"]["channels"]
    # MAG:HCM01 is writable, MAG:QF01 is read-only → only 1 match
    assert len(channels) == 1
    assert "MAG:HCM01:CURRENT:SP" in channels


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_channels_and_pattern_error():
    """Both channels and pattern → validation_error."""
    fn = _get_channel_limits()
    result = await fn(channels=["TEST:PV"], pattern="TEST:.*")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"


@pytest.mark.unit
async def test_invalid_filter_error():
    """Unknown filter_by value → validation_error."""
    fn = _get_channel_limits()
    result = await fn(filter_by="bogus_filter")

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "bogus_filter" in data["error_message"]


# ---------------------------------------------------------------------------
# Limits disabled
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_limits_disabled():
    """from_config() returns None → disabled response (not an error)."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=None,
    ):
        fn = _get_channel_limits()
        result = await fn()

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["limits_enabled"] is False
