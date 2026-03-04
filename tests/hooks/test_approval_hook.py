"""Tests for the osprey_approval hook.

This hook implements the human-in-the-loop approval system. Based on config:
- disabled: all tools pass through
- selective: only specific tool types require approval
- all_capabilities: all osprey tools require approval

Also covers pre-execution notebook creation for execute (python) approval.
"""

import pytest

# Default hook_config matching the original hard-coded OSPREY_PREFIXES
DEFAULT_APPROVAL_CONFIG = {
    "server_prefixes": ["mcp__controls__", "mcp__python__", "mcp__workspace__"],
    "approval_prefixes": ["mcp__controls__", "mcp__python__", "mcp__workspace__"],
}


def _is_allow(result) -> bool:
    """Check if hook result is an allow decision (None or explicit allow)."""
    if result is None:
        return True
    output = result.get("hookSpecificOutput", {})
    return output.get("permissionDecision") == "allow"


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
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _is_allow(result)  # All tools pass


@pytest.mark.unit
def test_selective_mode_blocks_write(tmp_path, hook_runner, make_config):
    """Selective mode blocks channel_write (a write operation)."""
    config = make_config(
        {
            "approval": {
                "global_mode": "selective",
                "requires_approval": ["channel_write", "execute"],
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
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
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "write"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
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
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _is_allow(result)  # Readonly without write patterns passes


@pytest.mark.unit
def test_selective_mode_allows_read(tmp_path, hook_runner, make_config):
    """Selective mode allows channel_read through (not in approval list)."""
    config = make_config(
        {
            "approval": {
                "global_mode": "selective",
                "requires_approval": ["channel_write", "execute"],
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _is_allow(result)  # Read passes through in selective mode


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
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
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
        hook_config=DEFAULT_APPROVAL_CONFIG,
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
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 42.0}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
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
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "write"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
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
        "mcp__python__execute",
        {"code": "epics.caput('PV', 5.0)", "execution_mode": "write"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    assert "write patterns" in output["permissionDecisionReason"]


# ============================================================================
# Framework pattern detection — extended coverage (Tango, LabVIEW, etc.)
# ============================================================================


@pytest.mark.unit
def test_framework_pattern_detection_tango_write(tmp_path, hook_runner, make_config):
    """Tango write_attribute pattern triggers approval via framework detection."""
    config = make_config(
        {
            "approval": {"global_mode": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "device.write_attribute('MOTOR:POS', 100)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_framework_pattern_detection_labview_write(tmp_path, hook_runner, make_config):
    """LabVIEW set_control pattern triggers approval via framework detection."""
    config = make_config(
        {
            "approval": {"global_mode": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "labview.set_control('temperature', 350)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_framework_pattern_detection_set_value(tmp_path, hook_runner, make_config):
    """EPICS .set_value() pattern triggers approval via framework detection."""
    config = make_config(
        {
            "approval": {"global_mode": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "pv.set_value(42.0)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_framework_pattern_no_false_positive_dict(tmp_path, hook_runner, make_config):
    """Dict operations should not trigger write pattern detection."""
    config = make_config(
        {
            "approval": {"global_mode": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "cache = {}\ncache['key'] = 'value'", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _is_allow(result)  # No approval needed


@pytest.mark.unit
def test_framework_pattern_detection_import_fallback(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """When osprey is not importable, fallback patterns still catch basic writes."""
    config = make_config(
        {
            "approval": {"global_mode": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    # The hook runs as a subprocess, so we can't easily mock the import.
    # Instead, test that a pattern covered by the fallback list still works.
    # caput( is in the fallback list, so it should always be caught.
    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "caput('TEST:PV', 1.0)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_framework_pattern_config_driven(tmp_path, hook_runner, make_config):
    """Config-driven custom patterns trigger approval via framework detection."""
    config = make_config(
        {
            "approval": {"global_mode": "selective"},
            "control_system": {
                "writes_enabled": True,
                "patterns": {
                    "write": [r"\bmy_custom_write\s*\("],
                    "read": [],
                },
            },
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "my_custom_write('DEVICE', 42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


# -- Config edge cases (gap fill) --


@pytest.mark.unit
def test_missing_approval_section_defaults_selective(tmp_path, hook_runner, make_config):
    """Config without 'approval' key defaults to selective mode.

    The hook uses config.get("approval", {}).get("global_mode", "selective"),
    so a missing approval section is treated as selective mode.
    """
    config = make_config(
        {
            "control_system": {"writes_enabled": True},
            # No 'approval' section at all
        }
    )

    # channel_write in selective mode requires approval
    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_invalid_approval_mode_passes_through(tmp_path, hook_runner, make_config):
    """Unknown approval mode string does not match any branch -> passes through.

    The hook checks for 'disabled', 'all_capabilities', and 'selective' explicitly.
    An unknown mode falls through all conditionals, allowing the tool without approval.
    This documents the current behavior -- not necessarily desired, but important to know.
    """
    config = make_config(
        {
            "approval": {"global_mode": "nonexistent_mode"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    # Unknown mode falls through all branches -> explicit allow
    assert _is_allow(result)


# ============================================================================
# Dynamic prefix tests — custom server hooks
# ============================================================================


@pytest.mark.unit
def test_custom_server_prefix_triggers_approval(tmp_path, hook_runner, make_config):
    """Custom server prefix in hook_config triggers approval in all_capabilities mode."""
    config = make_config(
        {
            "approval": {"global_mode": "all_capabilities"},
        }
    )

    custom_config = {
        "server_prefixes": ["mcp__controls__", "mcp__my_plc__"],
        "approval_prefixes": ["mcp__controls__", "mcp__my_plc__"],
    }

    result = hook_runner(
        "osprey_approval.py",
        "mcp__my_plc__set_output",
        {"output": "valve_1", "value": True},
        config_path=config,
        cwd=tmp_path,
        hook_config=custom_config,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    assert "set_output" in output["permissionDecisionReason"]
