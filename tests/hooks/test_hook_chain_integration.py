"""Integration tests for the hook chain execution order.

Claude Code runs hooks in the order they appear in settings.json.
The expected chain for osprey write operations is:
  1. osprey_writes_check — master kill switch (deny if writes disabled)
  2. osprey_limits — channel limits validation (deny if out of range)
  3. osprey_approval — human approval prompt (ask if required)
  4. osprey_audit — audit logging (always passes, logs the call)

If any hook denies, subsequent hooks do not run. These tests verify
the chain behavior by running hooks sequentially.
"""

import json
from pathlib import Path

import pytest
import yaml

HOOKS_DIR = Path(__file__).parents[2] / ".claude" / "hooks"

WRITE_HOOK_CHAIN = [
    "osprey_writes_check.py",
    "osprey_limits.py",
    "osprey_approval.py",
    "osprey_audit.py",
]


def _make_chain_config(
    tmp_path,
    writes_enabled=True,
    limits_enabled=True,
    approval_mode="selective",
    channels_db=None,
    allow_unlisted=True,
):
    """Create config.yml + optional channel_limits.json for chain testing."""
    config_dict = {
        "control_system": {
            "type": "mock",
            "writes_enabled": writes_enabled,
            "limits_checking": {
                "enabled": limits_enabled,
                "allow_unlisted_channels": allow_unlisted,
            },
        },
        "approval": {"global_mode": approval_mode},
    }

    if channels_db is not None and limits_enabled:
        db_path = tmp_path / "channel_limits.json"
        db_path.write_text(json.dumps(channels_db))
        config_dict["control_system"]["limits_checking"]["database_path"] = str(db_path)

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(config_dict))
    return config_path


def run_hook_chain(hook_runner, hook_names, tool_name, tool_input, config_path, cwd):
    """Run hooks in sequence, stopping at first deny."""
    for hook_name in hook_names:
        result = hook_runner(
            hook_name,
            tool_name,
            tool_input,
            config_path=config_path,
            cwd=cwd,
        )
        if result is not None:
            decision = result.get("hookSpecificOutput", {}).get("permissionDecision")
            if decision == "deny":
                return result, hook_name  # Blocked at this hook
            elif decision == "ask":
                return result, hook_name  # Paused for approval at this hook
    return None, None  # All hooks passed


@pytest.mark.integration
def test_writes_disabled_blocks_before_limits(tmp_path, hook_runner):
    """Writes-disabled hook blocks before limits hook ever runs."""
    config = _make_chain_config(
        tmp_path,
        writes_enabled=False,
        channels_db={"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        approval_mode="disabled",
    )

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert blocked_by == "osprey_writes_check.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.integration
def test_limits_violation_blocks_before_approval(tmp_path, hook_runner):
    """Limits violation blocks before approval hook runs."""
    config = _make_chain_config(
        tmp_path,
        writes_enabled=True,
        channels_db={"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        approval_mode="selective",
        allow_unlisted=False,
    )

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 999.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert blocked_by == "osprey_limits.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.integration
def test_valid_write_reaches_approval(tmp_path, hook_runner):
    """Valid write with enabled writes and valid limits reaches approval hook."""
    config = _make_chain_config(
        tmp_path,
        writes_enabled=True,
        channels_db={"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        approval_mode="selective",
        allow_unlisted=True,
    )

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert blocked_by == "osprey_approval.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


@pytest.mark.integration
def test_valid_write_disabled_approval_passes_all(tmp_path, hook_runner):
    """Valid write with disabled approval passes all hooks."""
    config = _make_chain_config(
        tmp_path,
        writes_enabled=True,
        channels_db={"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        approval_mode="disabled",
        allow_unlisted=True,
    )

    audit_dir = tmp_path / "osprey-workspace" / "audit"
    audit_dir.mkdir(parents=True)

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # All hooks passed
    assert blocked_by is None


@pytest.mark.integration
def test_read_tool_skips_write_checks(tmp_path, hook_runner):
    """Read tools pass through all write-focused hooks."""
    config = _make_chain_config(
        tmp_path,
        writes_enabled=False,
        limits_enabled=False,
        approval_mode="disabled",
    )

    audit_dir = tmp_path / "osprey-workspace" / "audit"
    audit_dir.mkdir(parents=True)

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
    )

    # Read operations pass through all hooks
    assert result is None
    assert blocked_by is None


@pytest.mark.integration
def test_non_osprey_tool_passes_entire_chain(tmp_path, hook_runner):
    """Non-osprey tools pass through the entire hook chain untouched."""
    config = _make_chain_config(
        tmp_path,
        writes_enabled=False,
        limits_enabled=False,
        approval_mode="all_capabilities",
    )

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "some_other_tool",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None
    assert blocked_by is None


@pytest.mark.integration
def test_python_execute_chain_with_framework_patterns(tmp_path, hook_runner):
    """Full hook chain triggers approval for Tango write patterns through python_execute."""
    config = _make_chain_config(
        tmp_path,
        writes_enabled=True,
        approval_mode="selective",
    )

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "mcp__osprey-python-executor__python_execute",
        {"code": "device.write_attribute('MOTOR:POS', 100)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert blocked_by == "osprey_approval.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"
