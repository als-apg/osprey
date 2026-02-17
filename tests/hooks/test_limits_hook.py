"""Tests for the osprey_limits hook.

This hook validates channel write values against configured safety limits
using the LimitsValidator. When limits are violated, the hook blocks the write.
When the validator is disabled or unavailable, writes pass through.

NOTE: LimitsValidator.from_config() reads the channel database from a JSON file
specified by control_system.limits_checking.database_path. Tests must create
a proper database file for validation to work.
"""

import json

import pytest


def _make_limits_config(tmp_path, channels_db, enabled=True, allow_unlisted=False):
    """Create config.yml + channel_limits.json for limits testing."""
    # Write channel limits database
    db_path = tmp_path / "channel_limits.json"
    db_path.write_text(json.dumps(channels_db))

    # Write config
    import yaml

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.dump(
            {
                "control_system": {
                    "type": "mock",
                    "writes_enabled": True,
                    "limits_checking": {
                        "enabled": enabled,
                        "database_path": str(db_path),
                        "allow_unlisted_channels": allow_unlisted,
                    },
                },
            }
        )
    )
    return config_path


@pytest.mark.unit
def test_limits_violation_blocks_write(tmp_path, hook_runner):
    """Write exceeding channel limits is blocked."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 999.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_valid_value_passes(tmp_path, hook_runner):
    """Write within channel limits passes through."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        allow_unlisted=True,
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_limits_disabled_passes_through(tmp_path, hook_runner):
    """When limits_checking.enabled is false, all writes pass."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        enabled=False,
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 999999.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through (limits disabled)


@pytest.mark.unit
def test_non_write_tools_pass(tmp_path, hook_runner):
    """Non-write tools (channel_read, etc.) are not checked by limits hook."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["TEST:PV"]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Read tools pass through


@pytest.mark.unit
def test_unlisted_channel_blocked_by_default(tmp_path, hook_runner):
    """Unlisted channels are blocked when allow_unlisted_channels is false (default)."""
    config = _make_limits_config(
        tmp_path,
        {"KNOWN:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        allow_unlisted=False,
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "UNKNOWN:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_non_writable_channel_blocked(tmp_path, hook_runner):
    """Channel marked writable=false is blocked."""
    config = _make_limits_config(
        tmp_path,
        {"READONLY:PV": {"min_value": 0.0, "max_value": 100.0, "writable": False}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "READONLY:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_multiple_operations_any_violation_blocks(tmp_path, hook_runner):
    """If any operation in a batch violates limits, the entire batch is blocked."""
    config = _make_limits_config(
        tmp_path,
        {
            "PV:A": {"min_value": 0.0, "max_value": 100.0, "writable": True},
            "PV:B": {"min_value": 0.0, "max_value": 10.0, "writable": True},
        },
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__osprey-control-system__channel_write",
        {
            "operations": [
                {"channel": "PV:A", "value": 50.0},  # OK
                {"channel": "PV:B", "value": 999.0},  # Violation
            ]
        },
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
