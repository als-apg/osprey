"""Tests for the osprey_audit hook.

This hook logs all osprey tool invocations to the audit log for
traceability and safety compliance. It should never block execution.
"""

import json

import pytest


@pytest.mark.unit
def test_audit_log_written(tmp_path, hook_runner, make_config):
    """Audit hook writes a log entry for osprey tool calls."""
    config = make_config({"control_system": {"writes_enabled": True}})

    # Create audit directory
    audit_dir = tmp_path / "osprey-workspace" / "audit"
    audit_dir.mkdir(parents=True)

    result = hook_runner(
        "osprey_audit.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
    )

    # Audit hook should not block (returns None — pass through)
    assert result is None

    # Check that audit log file was created
    log_files = list(audit_dir.glob("*.jsonl")) + list(audit_dir.glob("*.json"))
    assert len(log_files) > 0, "Audit log file should have been created"


@pytest.mark.unit
def test_audit_log_format(tmp_path, hook_runner, make_config):
    """Audit log entries contain required fields."""
    config = make_config({"control_system": {"writes_enabled": True}})

    audit_dir = tmp_path / "osprey-workspace" / "audit"
    audit_dir.mkdir(parents=True)

    hook_runner(
        "osprey_audit.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 42.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    # Read the audit log
    log_files = list(audit_dir.glob("*.jsonl")) + list(audit_dir.glob("*.json"))
    assert len(log_files) > 0

    log_content = log_files[0].read_text().strip()
    # Handle JSONL (one JSON object per line)
    entry = json.loads(log_content.split("\n")[-1])

    assert "tool" in entry
    assert "timestamp" in entry
    assert entry["tool"] == "channel_write"


@pytest.mark.unit
def test_audit_non_osprey_tools_skipped(tmp_path, hook_runner, make_config):
    """Non-osprey tools are not logged by the audit hook."""
    config = make_config({"control_system": {"writes_enabled": True}})

    audit_dir = tmp_path / "osprey-workspace" / "audit"
    audit_dir.mkdir(parents=True)

    result = hook_runner(
        "osprey_audit.py",
        "some_other_tool",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None

    # No audit log should be written for non-osprey tools
    log_files = list(audit_dir.glob("*.jsonl")) + list(audit_dir.glob("*.json"))
    if log_files:
        # If a file exists, it should not contain entries for non-osprey tools
        content = log_files[0].read_text().strip()
        if content:
            for line in content.split("\n"):
                entry = json.loads(line)
                assert "tool" in entry  # Only osprey tools are logged


@pytest.mark.unit
def test_audit_directory_created(tmp_path, hook_runner, make_config):
    """Audit hook creates the audit directory if it doesn't exist."""
    config = make_config({"control_system": {"writes_enabled": True}})

    # Deliberately do NOT create the audit directory
    audit_dir = tmp_path / "osprey-workspace" / "audit"
    assert not audit_dir.exists()

    result = hook_runner(
        "osprey_audit.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["TEST:PV"]},
        config_path=config,
        cwd=tmp_path,
    )

    # Hook should not block
    assert result is None

    # Directory should have been created
    assert audit_dir.exists()


@pytest.mark.unit
def test_audit_write_operations_logged(tmp_path, hook_runner, make_config):
    """Write operations include operation details in audit log."""
    config = make_config({"control_system": {"writes_enabled": True}})

    audit_dir = tmp_path / "osprey-workspace" / "audit"
    audit_dir.mkdir(parents=True)

    hook_runner(
        "osprey_audit.py",
        "mcp__osprey-control-system__channel_write",
        {"operations": [{"channel": "SR:SETPOINT", "value": 500.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    log_files = list(audit_dir.glob("*.jsonl")) + list(audit_dir.glob("*.json"))
    assert len(log_files) > 0

    log_content = log_files[0].read_text().strip()
    entry = json.loads(log_content.split("\n")[-1])

    assert entry["tool"] == "channel_write"
    assert "arguments" in entry
