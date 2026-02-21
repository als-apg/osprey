"""Tests for the osprey_writes_check hook.

This hook enforces the master writes kill switch (control_system.writes_enabled).
When disabled, it blocks channel_write and write-mode python_execute.
Read-only tools and readonly python always pass through.
"""

import pytest


@pytest.mark.unit
def test_writes_disabled_blocks_channel_write(tmp_path, hook_runner, make_config):
    """Writes disabled blocks channel_write tool."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_writes_enabled_allows_channel_write(tmp_path, hook_runner, make_config):
    """Writes enabled allows channel_write through."""
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_writes_disabled_blocks_python_write_mode(tmp_path, hook_runner, make_config):
    """Writes disabled blocks python_execute in write mode."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "write"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_writes_disabled_allows_python_readonly(tmp_path, hook_runner, make_config):
    """Writes disabled still allows python_execute in readonly mode."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_writes_disabled_allows_channel_read(tmp_path, hook_runner, make_config):
    """Writes disabled does not affect channel_read (read-only tool)."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_non_osprey_tools_pass_through(tmp_path, hook_runner, make_config):
    """Non-osprey tools are not affected by the writes check hook."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "some_other_tool",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Not an osprey tool, passes through


@pytest.mark.unit
def test_deny_message_includes_reason(tmp_path, hook_runner, make_config):
    """Deny decision includes an informative message."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "permissionDecisionReason" in output
    assert "WRITES DISABLED" in output["permissionDecisionReason"]


# -- Config edge cases (gap fill) --


@pytest.mark.unit
def test_missing_config_file_denies(tmp_path, hook_runner):
    """If config.yml doesn't exist, writes_enabled defaults to False (fail-closed).

    The hook's load_osprey_config() returns {} when the file is missing,
    and writes_enabled defaults to False, which blocks writes. This is the
    safe default for a safety-critical system — fail-closed, not fail-open.
    """
    # Point to a non-existent config path
    nonexistent_config = tmp_path / "nonexistent" / "config.yml"

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=nonexistent_config,
        cwd=tmp_path,
    )

    # Missing config → writes_enabled=False → deny (fail-closed)
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_missing_writes_enabled_key_denies(tmp_path, hook_runner, make_config):
    """Config exists but has no writes_enabled key → defaults to False (deny).

    The hook uses .get("writes_enabled", False), so a missing key is treated
    as writes disabled. This is intentionally fail-closed.
    """
    config = make_config({"control_system": {"type": "mock"}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    # Missing writes_enabled key → defaults to False → deny
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
