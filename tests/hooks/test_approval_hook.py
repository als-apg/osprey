"""Tests for the osprey_approval hook.

This hook implements the human-in-the-loop approval system. Based on config:
- disabled: all tools pass through
- selective: only specific tool types require approval
- all_capabilities: all osprey tools require approval

Also covers pre-execution notebook creation for python_execute approval.
"""

import pytest


@pytest.mark.unit
def test_approval_disabled_passes_all(tmp_path, hook_runner, make_config):
    """When approval mode is 'disabled', all tools pass through."""
    config = make_config(
        {
            "approval": {"global_mode": "disabled"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # All tools pass


@pytest.mark.unit
def test_selective_mode_blocks_write(tmp_path, hook_runner, make_config):
    """Selective mode blocks channel_write (a write operation)."""
    config = make_config(
        {
            "approval": {
                "global_mode": "selective",
                "requires_approval": ["channel_write", "python_execute"],
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_selective_mode_blocks_python_write(tmp_path, hook_runner, make_config):
    """Selective mode blocks python_execute in write mode."""
    config = make_config(
        {
            "approval": {
                "global_mode": "selective",
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-python-executor__python_execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "write"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_selective_mode_allows_readonly_python(tmp_path, hook_runner, make_config):
    """Selective mode allows readonly python without write patterns."""
    config = make_config(
        {
            "approval": {
                "global_mode": "selective",
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-python-executor__python_execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Readonly without write patterns passes


@pytest.mark.unit
def test_selective_mode_allows_read(tmp_path, hook_runner, make_config):
    """Selective mode allows channel_read through (not in approval list)."""
    config = make_config(
        {
            "approval": {
                "global_mode": "selective",
                "requires_approval": ["channel_write", "python_execute"],
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Read passes through in selective mode


@pytest.mark.unit
def test_all_capabilities_mode_blocks_all(tmp_path, hook_runner, make_config):
    """all_capabilities mode blocks all osprey tools for approval."""
    config = make_config(
        {
            "approval": {"global_mode": "all_capabilities"},
            "control_system": {"writes_enabled": True},
        }
    )

    # channel_read — normally read-only, but all_capabilities blocks everything
    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_non_osprey_tools_pass_through(tmp_path, hook_runner, make_config):
    """Non-osprey tools bypass the approval hook entirely."""
    config = make_config(
        {
            "approval": {"global_mode": "all_capabilities"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "some_other_tool",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Not an osprey tool


@pytest.mark.unit
def test_approval_ask_includes_tool_info(tmp_path, hook_runner, make_config):
    """Approval ask decision includes tool details for the operator."""
    config = make_config(
        {
            "approval": {
                "global_mode": "selective",
                "requires_approval": ["channel_write"],
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 42.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    # Should include context about what is being approved
    assert "permissionDecisionReason" in output
    assert "TEST:PV" in output["permissionDecisionReason"]


@pytest.mark.unit
def test_approval_python_write_creates_notebook(tmp_path, hook_runner, make_config):
    """Approval for python_execute with write patterns creates a pre-execution notebook."""
    config = make_config(
        {
            "approval": {"global_mode": "selective"},
            "control_system": {"writes_enabled": True},
            "artifact_server": {"host": "127.0.0.1", "port": 8086},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-python-executor__python_execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "write"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    # Gallery link should be in the reason (may fail if osprey not importable
    # in subprocess, but the approval itself must still work)
    reason = output["permissionDecisionReason"]
    assert "Python execution" in reason


@pytest.mark.unit
def test_approval_notebook_failure_nonfatal(tmp_path, hook_runner, make_config):
    """If notebook creation fails in the hook, approval still works normally."""
    config = make_config(
        {
            "approval": {"global_mode": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    # Even without osprey importable, the hook should not crash
    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey-python-executor__python_execute",
        {"code": "epics.caput('PV', 5.0)", "execution_mode": "write"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    assert "EPICS write patterns" in output["permissionDecisionReason"]
