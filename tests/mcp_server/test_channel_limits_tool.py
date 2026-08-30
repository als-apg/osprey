"""Tests for the channel_limits MCP tool.

Covers: summary mode, exact lookup (found/not-found/mixed), regex search,
property filters, combined search+filter, parameter validation, limits-disabled,
and the reported limits posture (tri-state allow_unlisted_channels plus the
config key that answered, resolved for the session's control target).
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

TEST_LIMITS_DB = {
    "_version": "1.0",
    "defaults": {"writable": True, "confirm": True},
    "MAG:HCM01:CURRENT:SP": {
        "min_value": -10.0,
        "max_value": 10.0,
        "max_step": 2.0,
        "writable": True,
    },
    "MAG:QF01:CURRENT:SP": {"writable": False},
    "DIAG:TEMP:SP": {
        "min_value": 0.0,
        "max_value": 100.0,
        "writable": True,
        "confirm": False,
    },
    "Amplifier 2 [J]": {
        "min_value": 0.0,
        "max_value": 50.0,
        "writable": True,
    },
    "DDG-AA-ShotCntrl Delay.Ch H": {
        "writable": True,
    },
    "Background density [1e18/cm^3]": {
        "writable": False,
    },
}


@dataclass
class FakeChannelLimitsConfig:
    channel_address: str
    min_value: float | None = None
    max_value: float | None = None
    max_step: float | None = None
    writable: bool = True


def _resolve_confirm(channel_address: str) -> bool:
    """Mirror LimitsValidator.resolve_confirm: channel → defaults → True."""
    cfg = TEST_LIMITS_DB.get(channel_address)
    if isinstance(cfg, dict) and "confirm" in cfg:
        return bool(cfg["confirm"])
    return bool(TEST_LIMITS_DB["defaults"].get("confirm", True))


DEPLOYMENT_WIDE_KEY = "control_system.limits_checking.allow_unlisted_channels"
VA_KEY = "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"


def _make_validator(
    allow_unlisted: bool | None = True,
    allow_unlisted_key: str = DEPLOYMENT_WIDE_KEY,
) -> MagicMock:
    """Build a mock LimitsValidator from TEST_LIMITS_DB.

    Args:
        allow_unlisted: The tri-state posture answer — ``None`` is an unset
            deployment-wide key, which refuses unlisted channels.
        allow_unlisted_key: The config key that answered, as
            ``LimitsValidator._from_posture`` puts it into ``policy``.
    """
    validator = MagicMock()
    validator._raw_db = TEST_LIMITS_DB
    validator.resolve_confirm.side_effect = _resolve_confirm

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
        "allow_unlisted_key": allow_unlisted_key,
        "on_violation": "error",
    }
    return validator


def _patch_target(target: str | None = None, raises: bool = False):
    """Patch the session control target the tool reads before building a validator."""
    if raises:
        return patch(
            "osprey.mcp_server.control_system.tools.channel_limits.target_state.read",
            side_effect=OSError("no shared data root"),
        )
    record = None if target is None else {"target": target, "generation": 3}
    return patch(
        "osprey.mcp_server.control_system.tools.channel_limits.target_state.read",
        return_value=record,
    )


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

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["total_channels"] == 6
    assert data["summary"]["writable"] == 4
    assert data["summary"]["read_only"] == 2
    assert data["summary"]["has_step_limit"] == 1
    assert data["summary"]["version"] == "1.0"
    assert data["access_details"]["policy"]["allow_unlisted_channels"] is True
    assert data["access_details"]["policy"]["allow_unlisted_key"] == DEPLOYMENT_WIDE_KEY
    assert data["access_details"]["defaults"]["writable"] is True


@pytest.mark.unit
async def test_summary_confirm_breakdown():
    """Summary reports how many channels resolve to confirm true vs false."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn()

    data = extract_response_dict(result)
    # Only DIAG:TEMP:SP opts out; the rest inherit defaults.confirm = true.
    assert data["summary"]["confirm_breakdown"] == {"true": 5, "false": 1}
    # The retired per-level breakdown is gone: no summary key names it any more.
    assert not [key for key in data["summary"] if "verification" in key]


# ---------------------------------------------------------------------------
# Lookup mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_lookup_found():
    """Single known channel → full config with the resolved confirm flag."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["MAG:HCM01:CURRENT:SP"])

    data = extract_response_dict(result)
    assert data["status"] == "success"
    ch = data["access_details"]["channels"]["MAG:HCM01:CURRENT:SP"]
    assert ch["writable"] is True
    assert ch["min_value"] == -10.0
    assert ch["max_value"] == 10.0
    assert ch["max_step"] == 2.0
    assert ch["confirm"] is True
    assert "verification" not in ch


@pytest.mark.unit
async def test_lookup_confirm_opt_out():
    """A channel with confirm: false reports it; defaults are not applied over it."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["DIAG:TEMP:SP"])

    data = extract_response_dict(result)
    ch = data["access_details"]["channels"]["DIAG:TEMP:SP"]
    assert ch["confirm"] is False


@pytest.mark.unit
async def test_lookup_not_found_blocked():
    """Unknown channel + allow_unlisted=false → BLOCKED."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(allow_unlisted=False),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["UNKNOWN:PV"])

    data = extract_response_dict(result)
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

    data = extract_response_dict(result)
    ch = data["access_details"]["channels"]["UNKNOWN:PV"]
    assert ch["in_database"] is False
    assert "allowed" in ch["policy_action"]


@pytest.mark.unit
async def test_summary_unset_reports_null_and_deployment_wide_key():
    """Deployment-wide key unset → the summary reports null, not a permissive default."""
    with (
        _patch_target(),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=_make_validator(allow_unlisted=None),
        ),
    ):
        fn = _get_channel_limits()
        result = await fn()

    data = extract_response_dict(result)
    policy = data["access_details"]["policy"]
    assert policy["allow_unlisted_channels"] is None
    assert policy["allow_unlisted_key"] == DEPLOYMENT_WIDE_KEY


@pytest.mark.unit
async def test_lookup_unset_is_refused_naming_the_deployment_wide_key():
    """Unset is nobody's permission: the unlisted channel is refused, key named."""
    with (
        _patch_target(),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=_make_validator(allow_unlisted=None),
        ),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["UNKNOWN:PV"])

    data = extract_response_dict(result)
    ch = data["access_details"]["channels"]["UNKNOWN:PV"]
    assert ch["in_database"] is False
    assert ch["allow_unlisted_channels"] is None
    assert ch["allow_unlisted_key"] == DEPLOYMENT_WIDE_KEY
    assert ch["policy_action"].startswith("BLOCKED")
    assert DEPLOYMENT_WIDE_KEY in ch["policy_action"]


@pytest.mark.unit
async def test_posture_is_resolved_for_the_session_target():
    """The tool asks for the posture of the target this server is on."""
    with (
        _patch_target("va"),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=_make_validator(allow_unlisted=True, allow_unlisted_key=VA_KEY),
        ) as from_config,
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["UNKNOWN:PV"])

    from_config.assert_called_once_with(target="va")
    data = extract_response_dict(result)
    ch = data["access_details"]["channels"]["UNKNOWN:PV"]
    assert ch["allow_unlisted_channels"] is True
    assert ch["allow_unlisted_key"] == VA_KEY
    assert ch["policy_action"] == "allowed (no limits enforced)"


@pytest.mark.unit
async def test_summary_reports_the_per_target_key():
    """A per-type block answers: the summary names that key, not the deployment-wide one."""
    with (
        _patch_target("va"),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=_make_validator(allow_unlisted=True, allow_unlisted_key=VA_KEY),
        ),
    ):
        fn = _get_channel_limits()
        result = await fn()

    data = extract_response_dict(result)
    assert data["access_details"]["policy"]["allow_unlisted_key"] == VA_KEY


@pytest.mark.unit
async def test_unreadable_target_state_falls_back_to_the_deployment_wide_block():
    """An unreadable state directory is not fatal: no target → deployment-wide posture."""
    with (
        _patch_target(raises=True),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=_make_validator(),
        ) as from_config,
    ):
        fn = _get_channel_limits()
        result = await fn()

    from_config.assert_called_once_with(target=None)
    assert extract_response_dict(result)["status"] == "success"


@pytest.mark.unit
async def test_hand_built_policy_without_a_key_names_the_deployment_wide_one():
    """A validator built from a bare policy dict carries no key; report the honest default."""
    validator = _make_validator(allow_unlisted=False)
    del validator.policy["allow_unlisted_key"]
    with (
        _patch_target("live"),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=validator,
        ),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["UNKNOWN:PV"])

    ch = extract_response_dict(result)["access_details"]["channels"]["UNKNOWN:PV"]
    assert ch["allow_unlisted_channels"] is False
    assert ch["allow_unlisted_key"] == DEPLOYMENT_WIDE_KEY


@pytest.mark.unit
async def test_lookup_multiple_mixed():
    """Mix of found + not-found channels."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(allow_unlisted=False),
    ):
        fn = _get_channel_limits()
        result = await fn(channels=["MAG:HCM01:CURRENT:SP", "NONEXISTENT:PV"])

    data = extract_response_dict(result)
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

    data = extract_response_dict(result)
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

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["matches"] == 2
    assert "MAG:HCM01:CURRENT:SP" in data["access_details"]["channels"]
    assert "MAG:QF01:CURRENT:SP" in data["access_details"]["channels"]


@pytest.mark.unit
async def test_search_entries_carry_confirm():
    """Compact search entries report confirm, never verification vocabulary."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(pattern=".*")

    data = extract_response_dict(result)
    channels = data["access_details"]["channels"]
    assert channels["MAG:HCM01:CURRENT:SP"]["confirm"] is True
    assert channels["DIAG:TEMP:SP"]["confirm"] is False
    assert all("verification" not in key for entry in channels.values() for key in entry)


@pytest.mark.unit
async def test_pattern_no_match():
    """Non-matching pattern → empty results, still success."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(pattern="NONEXISTENT:.*")

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["matches"] == 0


@pytest.mark.unit
async def test_pattern_invalid_regex():
    """Invalid regex → validation_error."""
    fn = _get_channel_limits()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(pattern="[invalid")

    data = _exc_ctx["envelope"]
    assert "regex" in data["error_message"].lower()


# ---------------------------------------------------------------------------
# Literal name search mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_name_contains_matches_regex_metacharacters_literally():
    """name_contains uses literal matching for names with [], (), ., ^, etc."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(name_contains="Amplifier 2 [J]")

    data = extract_response_dict(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 1
    assert "Amplifier 2 [J]" in channels


@pytest.mark.unit
async def test_name_contains_treats_dot_as_literal():
    """Literal matching should not treat '.' as a regex wildcard."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(name_contains="Delay.Ch")

    data = extract_response_dict(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 1
    assert "DDG-AA-ShotCntrl Delay.Ch H" in channels


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

    data = extract_response_dict(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 4
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

    data = extract_response_dict(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 2
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

    data = extract_response_dict(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 1
    assert "MAG:HCM01:CURRENT:SP" in channels


@pytest.mark.unit
async def test_the_filter_set_carries_no_retired_readback_filter():
    """Only the four live property filters are accepted; a readback one is not.

    The tool once offered a filter keyed on the retired per-channel readback
    setting. Pinning ``VALID_FILTERS`` itself is what keeps it from coming
    back — the rejection below only proves the guard still names what it
    refused.
    """
    from osprey.mcp_server.control_system.tools.channel_limits import VALID_FILTERS

    assert VALID_FILTERS == {"writable", "read_only", "has_step_limit", "has_range"}
    assert not [name for name in VALID_FILTERS if "readback" in name or "verif" in name]

    fn = _get_channel_limits()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(filter_by="readback")

    data = _exc_ctx["envelope"]
    assert "readback" in data["error_message"]


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

    data = extract_response_dict(result)
    channels = data["access_details"]["channels"]
    # MAG:HCM01 is writable, MAG:QF01 is read-only → only 1 match
    assert len(channels) == 1
    assert "MAG:HCM01:CURRENT:SP" in channels


@pytest.mark.unit
async def test_combined_name_contains_and_filter():
    """name_contains + filter_by → literal search filtered by property."""
    with patch(
        "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
        return_value=_make_validator(),
    ):
        fn = _get_channel_limits()
        result = await fn(name_contains="[", filter_by="read_only")

    data = extract_response_dict(result)
    channels = data["access_details"]["channels"]
    assert len(channels) == 1
    assert "Background density [1e18/cm^3]" in channels


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_channels_and_pattern_error():
    """Both channels and pattern → validation_error."""
    fn = _get_channel_limits()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(channels=["TEST:PV"], pattern="TEST:.*")

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_channels_and_name_contains_error():
    """Both channels and name_contains → validation_error."""
    fn = _get_channel_limits()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(channels=["TEST:PV"], name_contains="TEST")

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_pattern_and_name_contains_error():
    """Both pattern and name_contains → validation_error."""
    fn = _get_channel_limits()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(pattern="TEST:.*", name_contains="TEST")

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_invalid_filter_error():
    """Unknown filter_by value → validation_error."""
    fn = _get_channel_limits()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(filter_by="bogus_filter")

    data = _exc_ctx["envelope"]
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

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["limits_enabled"] is False
