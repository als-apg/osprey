"""Tests for the osprey_approval hook.

This hook implements the human-in-the-loop approval system. Based on the
`approval` section of config.yml:
- `enabled: false` — all tools pass through
- `tools: {name: policy}` with policies `skip` / `selective` / `always`
- `default_policy` — applied to tools absent from `tools`

Also covers pre-execution notebook creation for execute (python) approval.
"""

import json
import os
import re

import pytest

# Default hook_config matching the original hard-coded OSPREY_PREFIXES
DEFAULT_APPROVAL_CONFIG = {
    "server_prefixes": [
        "mcp__controls__",
        "mcp__python__",
        "mcp__osprey_workspace__",
        "mcp__ariel__",
    ],
    "approval_prefixes": [
        "mcp__controls__",
        "mcp__python__",
        "mcp__osprey_workspace__",
        "mcp__ariel__",
    ],
}

# Per-tool approval config for new tests
DEFAULT_TOOLS_CONFIG = {
    "enabled": True,
    "default_policy": "always",
    "tools": {
        "channel_write": "always",
        "channel_read": "skip",
        "archiver_read": "skip",
        "execute": "selective",
        "setup_patch": "always",
        "entry_create": "always",
    },
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
            "approval": {"enabled": False},
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
                "enabled": True,
                "default_policy": "selective",
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
                "enabled": True,
                "default_policy": "selective",
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "readwrite"},
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
                "enabled": True,
                "default_policy": "selective",
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
def test_per_tool_skip_allows_read(tmp_path, hook_runner, make_config):
    """A per-tool skip policy lets channel_read through, even when the
    default_policy would otherwise prompt. Mirrors production config which
    pins channel_read=skip while other tools fall through to selective."""
    config = make_config(
        {
            "approval": {
                "enabled": True,
                "default_policy": "selective",
                "tools": {"channel_read": "skip"},
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

    assert _is_allow(result)


@pytest.mark.unit
def test_channel_read_skip_emits_allow_decision(tmp_path, hook_runner, make_config):
    """DETERMINISTIC contract pin for the e2e read test (test_safety_reads.py).

    The default control_assistant config ships ``channel_read: skip``. This test
    drives the approval hook directly (no LLM/SDK) and asserts the skip branch
    emits an explicit ``permissionDecision: 'allow'`` — that is the decision the
    SDK later surfaces through can_use_tool as a recorded `allow` hook event,
    which is exactly why the e2e asserts "all read events are allow" (not zero).
    Asserting `allow` here (not merely "not ask") locks the product contract so
    any regression in the skip branch is caught without an API key.
    """
    config = make_config(
        {
            "approval": {
                "enabled": True,
                "default_policy": "always",
                "tools": {"channel_read": "skip"},
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:BEAM:CURRENT"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    # Skip branch must emit an EXPLICIT allow (not None) so the SDK records it.
    assert result is not None, "skip policy must emit an explicit allow decision"
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "allow"


@pytest.mark.unit
def test_channel_read_always_asks(tmp_path, hook_runner, make_config):
    """Paired contract: with ``default_policy: always`` and no ``tools`` map,
    even channel_read must `ask` — the fail-closed backstop (scenario 2d). This
    guards against the lazy 'fix' of statically allowlisting channel_read, which
    would short-circuit the PreToolUse hook and silently break default-always.
    """
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:BEAM:CURRENT"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_default_policy_always_blocks_all_tools(tmp_path, hook_runner, make_config):
    """`default_policy: always` (with no `tools` overrides) asks on every osprey tool."""
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )

    # channel_read — normally read-only, but default_policy=always asks on everything
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
            "approval": {"enabled": True, "default_policy": "always"},
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
                "enabled": True,
                "default_policy": "selective",
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
            "approval": {"enabled": True, "default_policy": "selective"},
            "control_system": {"writes_enabled": True},
            "artifact_server": {"host": "127.0.0.1", "port": 8086},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "readwrite"},
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
            "approval": {"enabled": True, "default_policy": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    # Even without osprey importable, the hook should not crash
    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "epics.caput('PV', 5.0)", "execution_mode": "readwrite"},
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
            "approval": {"enabled": True, "default_policy": "selective"},
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
            "approval": {"enabled": True, "default_policy": "selective"},
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
            "approval": {"enabled": True, "default_policy": "selective"},
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
            "approval": {"enabled": True, "default_policy": "selective"},
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
            "approval": {"enabled": True, "default_policy": "selective"},
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
            "approval": {"enabled": True, "default_policy": "selective"},
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


@pytest.mark.unit
def test_has_write_patterns_passes_config_to_framework(tmp_path, hook_runner, make_config):
    """Config patterns flow from hook config dict through to framework detection.

    When custom patterns are in config AND the framework module is importable,
    the hook must pass them to detect_control_system_operations() so they are
    merged with framework standards — not silently ignored.
    """
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "selective"},
            "control_system": {
                "writes_enabled": True,
                "patterns": {
                    "write": [r"\bfacility_hw_write\s*\("],
                    "read": [],
                },
            },
        }
    )

    # Custom pattern should trigger approval
    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "facility_hw_write('MOTOR', 100)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"

    # Framework EPICS pattern should ALSO still trigger approval (merged, not replaced)
    result2 = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "epics.caput('PV', 1.0)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result2 is not None
    output2 = result2["hookSpecificOutput"]
    assert output2["permissionDecision"] == "ask"


@pytest.mark.unit
def test_has_write_patterns_override_via_config(tmp_path, hook_runner, make_config):
    """Config with mode: override replaces framework patterns in the hook subprocess.

    When a facility sets mode: override, ONLY their custom patterns should be
    used. Framework patterns (e.g., epics.caput) should NOT trigger approval.
    """
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "selective"},
            "control_system": {
                "writes_enabled": True,
                "patterns": {
                    "mode": "override",
                    "write": [r"\bfacility_hw_write\s*\("],
                    "read": [],
                },
            },
        }
    )

    # Custom pattern should still trigger approval
    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "facility_hw_write('MOTOR', 100)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"

    # Framework EPICS pattern should NOT trigger approval (override mode)
    result2 = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "epics.caput('PV', 1.0)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _is_allow(result2)


@pytest.mark.unit
def test_fallback_merges_custom_patterns(hook_module):
    """Fallback path merges custom patterns with _FALLBACK_WRITE_PATTERNS by default.

    When osprey is not importable, the fallback regex path must merge
    custom patterns (extend mode) instead of replacing the fallback list.
    """
    fallback_patterns = hook_module("osprey_approval")._FALLBACK_WRITE_PATTERNS

    # Simulate the fallback merge logic (extend mode, the default)
    config = {
        "control_system": {
            "patterns": {
                "write": [r"\bfacility_hw_write\s*\("],
            }
        }
    }

    pat_config = config.get("control_system", {}).get("patterns", {})
    custom = pat_config.get("write")
    mode = pat_config.get("mode", "extend")
    patterns = list(fallback_patterns)
    if custom:
        if mode == "override":
            patterns = list(custom)
        else:
            patterns.extend(p for p in custom if p not in patterns)

    # Custom pattern present
    assert any(re.search(p, "facility_hw_write('X', 1)") for p in patterns)
    # Framework patterns still present (extend mode)
    assert any(re.search(p, "caput('PV', 1.0)") for p in patterns)
    assert len(patterns) == len(fallback_patterns) + 1


@pytest.mark.unit
def test_fallback_override_replaces_patterns(hook_module):
    """Fallback path with mode=override replaces _FALLBACK_WRITE_PATTERNS entirely."""
    fallback_patterns = hook_module("osprey_approval")._FALLBACK_WRITE_PATTERNS

    config = {
        "control_system": {
            "patterns": {
                "mode": "override",
                "write": [r"\bfacility_hw_write\s*\("],
            }
        }
    }

    pat_config = config.get("control_system", {}).get("patterns", {})
    custom = pat_config.get("write")
    mode = pat_config.get("mode", "extend")
    patterns = list(fallback_patterns)
    if custom:
        if mode == "override":
            patterns = list(custom)
        else:
            patterns.extend(p for p in custom if p not in patterns)

    # Custom pattern present
    assert any(re.search(p, "facility_hw_write('X', 1)") for p in patterns)
    # Framework patterns NOT present (override mode)
    assert not any(re.search(p, "caput('PV', 1.0)") for p in patterns)
    assert len(patterns) == 1


# -- Config edge cases (gap fill) --


@pytest.mark.unit
def test_missing_approval_section_defaults_to_always(tmp_path, hook_runner, make_config):
    """Config without 'approval' key falls through to default_policy='always' (fail-closed).

    The hook reads ``config.get("approval", {})`` so a missing section yields
    an empty dict. With no per-tool ``tools`` mapping, every tool resolves
    through the ``default_policy`` default of ``"always"`` -> ask.
    """
    config = make_config(
        {
            "control_system": {"writes_enabled": True},
            # No 'approval' section at all
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


# ============================================================================
# Dynamic prefix tests — custom server hooks
# ============================================================================


@pytest.mark.unit
def test_custom_server_prefix_triggers_approval(tmp_path, hook_runner, make_config):
    """Custom server prefix in hook_config triggers approval under `default_policy: always`."""
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
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


# ============================================================================
# Per-tool approval tests (new config format)
# ============================================================================


@pytest.mark.unit
def test_approval_enabled_false_allows_all(tmp_path, hook_runner, make_config):
    """When approval.enabled is false, all tools pass through."""
    tools_config = {**DEFAULT_TOOLS_CONFIG, "enabled": False}
    config = make_config({"approval": tools_config})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _is_allow(result)


@pytest.mark.unit
def test_tool_policy_always_asks(tmp_path, hook_runner, make_config):
    """Tool mapped to 'always' policy always requires approval."""
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

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
def test_tool_policy_skip_allows(tmp_path, hook_runner, make_config):
    """Tool mapped to 'skip' policy is allowed without approval."""
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _is_allow(result)


@pytest.mark.unit
def test_tool_policy_selective_execute_write_mode_asks(tmp_path, hook_runner, make_config):
    """Selective policy for execute blocks write-mode execution."""
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readwrite"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_tool_policy_selective_execute_readonly_allows(tmp_path, hook_runner, make_config):
    """Selective policy for execute allows readonly code without write patterns."""
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _is_allow(result)


@pytest.mark.unit
def test_tool_policy_selective_execute_write_patterns_asks(tmp_path, hook_runner, make_config):
    """Selective policy for execute blocks code with write patterns."""
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_unknown_tool_defaults_to_always(tmp_path, hook_runner, make_config):
    """Tools not in the tools map fall back to default_policy (always)."""
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    # Use a custom hook_config with a novel server prefix
    custom_config = {
        "server_prefixes": ["mcp__controls__", "mcp__my_plc__"],
        "approval_prefixes": ["mcp__controls__", "mcp__my_plc__"],
    }

    result = hook_runner(
        "osprey_approval.py",
        "mcp__my_plc__unknown_tool",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
        hook_config=custom_config,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_setup_patch_always_asks(tmp_path, hook_runner, make_config):
    """Workspace setup_patch tool requires approval through the hook."""
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__osprey_workspace__setup_patch",
        {"path": "control_system.writes_enabled", "value": True},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


@pytest.mark.unit
def test_entry_create_always_asks(tmp_path, hook_runner, make_config):
    """ARIEL entry_create tool requires approval through the hook."""
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__ariel__entry_create",
        {"title": "Test entry", "content": "Test content"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


# ============================================================================
# Pattern parity — fallback patterns must match framework patterns
# ============================================================================


@pytest.mark.unit
def test_fallback_pattern_parity_with_framework(hook_module):
    """Fallback write patterns in the hook must match framework standard patterns.

    This catches drift between the two pattern lists. If a pattern is added to
    get_framework_standard_patterns()["write"] but not to _FALLBACK_WRITE_PATTERNS,
    this test fails — preventing silent gaps in the fallback approval path.
    """
    from osprey.services.python_executor.analysis.pattern_detection import (
        get_framework_standard_patterns,
    )

    framework_patterns = get_framework_standard_patterns()["write"]
    fallback_patterns = hook_module("osprey_approval")._FALLBACK_WRITE_PATTERNS

    assert fallback_patterns == framework_patterns, (
        f"Fallback patterns ({len(fallback_patterns)}) differ from "
        f"framework patterns ({len(framework_patterns)}).\n"
        f"Missing from fallback: {set(framework_patterns) - set(fallback_patterns)}\n"
        f"Extra in fallback: {set(fallback_patterns) - set(framework_patterns)}"
    )


@pytest.mark.unit
def test_fallback_covers_p4p_write_idioms(hook_module):
    """The fallback list must carry the p4p (PVAccess) write spellings.

    Parity alone would stay green if both lists lost p4p together, so the
    idioms are pinned here as well: put and post anchored to p4p, the rpc
    round trip, and SharedPV (serving a PV puts values on the wire). There is
    deliberately no bare ``.post(`` entry - it would flag every requests.post()
    in ordinary analysis code.
    """
    fallback_patterns = hook_module("osprey_approval")._FALLBACK_WRITE_PATTERNS

    for pattern in (
        r"\bp4p\b[\s\S]*?\.put\s*\(",
        r"\bp4p\b[\s\S]*?\.post\s*\(",
        r"\.rpc\s*\(",
        r"\bSharedPV\b",
    ):
        assert pattern in fallback_patterns

    assert r"\.post\s*\(" not in fallback_patterns


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import p4p.client.thread\np4p.client.thread.Context('pva').put('SR:A:SP', 1)\n",
            id="p4p-put",
        ),
        pytest.param(
            "from p4p.server.thread import SharedPV\npv = SharedPV()\npv.post(1.0)\n",
            id="p4p-sharedpv-post",
        ),
        pytest.param(
            "from p4p.client.asyncio import Context\nr = await Context('pva').rpc('SR:C', a)\n",
            id="p4p-rpc",
        ),
    ],
)
def test_fallback_regexes_match_p4p_code(hook_module, code):
    """The pinned fallback regexes fire on code an agent would actually write."""
    import re as _re

    fallback_patterns = hook_module("osprey_approval")._FALLBACK_WRITE_PATTERNS

    assert any(_re.search(p, code) for p in fallback_patterns)


@pytest.mark.unit
@pytest.mark.parametrize(
    "stdin",
    ["", "{nope", "[]", "[1,2,3]"],
    ids=["empty", "invalid-json", "wrong-shape", "wrong-shape-truthy"],
)
def test_malformed_stdin_fails_open(tmp_path, hook_runner_raw, stdin):
    """Unusable stdin passes the tool through without asking for approval.

    A closed pipe, a truncated write and a non-object payload — falsy (``[]``)
    or truthy (``[1,2,3]``) — each leave the hook with no tool to gate, so it
    exits 0 and emits no decision. Emitting an "ask" here would stall every
    tool call behind a prompt nobody can answer. The truthy payload is the one
    an emptiness check lets through, so it has to be rejected on shape.
    """
    returncode, stdout, stderr = hook_runner_raw(
        "osprey_approval.py",
        tool_name=None,
        tool_input=None,
        cwd=tmp_path,
        stdin_override=stdin,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "Traceback" not in stderr


# ============================================================================
# readwrite mode always asks — approval does not rest on pattern detection
# ============================================================================


@pytest.mark.unit
def test_selective_readwrite_asks_without_write_patterns(tmp_path, hook_runner, make_config):
    """A readwrite run prompts even when the detector sees no write pattern.

    The agent asking for write mode is itself the signal: approval for
    readwrite must not depend on the regex recognising the spelling."""
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "print('nothing to see')", "execution_mode": "readwrite"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


@pytest.mark.unit
def test_selective_readwrite_asks_for_aliased_caput(tmp_path, hook_runner, make_config):
    """``from epics import caput as _w`` evades every write regex; readwrite still asks."""
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "selective"},
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {
            "code": "from epics import caput as _w\n_w('SR:MAG:QF:01:CURRENT:SP', 150)",
            "execution_mode": "readwrite",
        },
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


# ============================================================================
# Write posture — the short-circuit that keeps an unarmed write from prompting
# ============================================================================
# When a deployment is not armed for the target a call names, this hook must
# emit NO decision at all: the layer that refuses the call has already spoken,
# and an "ask" from here would reopen `can_use_tool` over it. Two things bound
# that: the three-way (a config stating no posture prompts exactly as before),
# and the rule that a defer is only safe where a refusal is GUARANTEED — by
# `osprey_writes_check`'s deny for the session-targeted tools, or by
# `queue_start`'s own pre-bridge lane gate for a start whose lane is placed.

#: A hook_config whose `write_tools` covers the two framework write tools and a
#: whole self-gated Bluesky server, the way a rendered project spells it.
POSTURE_HOOK_CONFIG = {
    "server_prefixes": ["mcp__controls__", "mcp__python__", "mcp__bluesky__"],
    "approval_prefixes": ["mcp__controls__", "mcp__python__", "mcp__bluesky__"],
    "write_tools": [
        "mcp__controls__channel_write",
        "mcp__python__execute",
        "mcp__bluesky__.*",
    ],
}

#: Global "yes" overridden by a "no" on the live machine's own connector block.
#: Its live answer (not armed) and its most-restrictive answer (not armed) are
#: the same, so a test using it alone cannot show the state file was read.
DISARMED_LIVE_CONFIG = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"writes_enabled": False},
        "virtual_accelerator": {},
    },
}

#: The mirror image: global "no", armed on the machine alone. Its live answer
#: (armed) and its most-restrictive answer (not armed) DIFFER, so a call that
#: prompts on target `live` proves the session's state file reached the hook.
ARMED_LIVE_ONLY_CONFIG = {
    "type": "epics",
    "writes_enabled": False,
    "connector": {
        "epics": {"writes_enabled": True},
        "virtual_accelerator": {},
    },
}

#: Not one word about posture anywhere — the shape every deployment had before
#: the per-type key existed.
NO_POSTURE_CONFIG = {"type": "epics", "connector": {"epics": {}}}

#: Two rendered plan lanes, one per target, over a config that arms neither.
TWO_LANE_SERVICES = {
    "bluesky": {"port": 8090, "target": "va"},
    "bluesky_live": {"port": 8091, "target": "live"},
}

#: One rendered plan lane, serving the machine.
ONE_LANE_SERVICES = {"bluesky": {"port": 8090, "target": "live"}}


def _posture_config(make_config, control_system, services=None):
    """A config that prompts for everything, over the given control_system block."""
    config = {
        "approval": {"enabled": True, "default_policy": "always"},
        "control_system": control_system,
    }
    if services is not None:
        config["services"] = services
    return make_config(config)


def _write_session_state(repo_root, target):
    """Point the session at *target* with a state file this process owns.

    ``owner_ppid`` is this pytest process, which is genuinely on the ancestor
    chain of the hook subprocess, and ``server_pid`` is alive by definition — so
    the reader's real parentage and liveness rules select this record.
    """
    directory = repo_root / "var" / "agent_data" / "control_target"
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "target": target,
        "generation": 1,
        "server_pid": os.getpid(),
        "owner_ppid": os.getpid(),
        "targets": {
            "live": {"label": "Storage ring", "endpoint": "pva://gw:5075", "real_machine": True},
            "va": {"label": "Virtual accelerator", "endpoint": "pva://127.0.0.1:5074"},
        },
    }
    (directory / f"target_state_{os.getpid()}.json").write_text(json.dumps(record))


def _decision(result):
    """The permission decision in a hook result, or ``None`` when it emitted none."""
    if result is None:
        return None
    return result["hookSpecificOutput"]["permissionDecision"]


@pytest.mark.unit
def test_readonly_execute_still_prompts_when_writes_are_not_armed(
    tmp_path, hook_runner, make_config
):
    """A readonly execution is not a write, so the posture never reaches it.

    Deferring here would delete the approval prompt for ordinary analysis code
    on every unarmed deployment — the tool `writes_check` deliberately lets
    through is the one this hook must keep asking about.
    """
    config = _posture_config(make_config, {"type": "mock", "writes_enabled": False})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "print(get_channel('SR:CURRENT'))", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=POSTURE_HOOK_CONFIG,
    )

    assert _decision(result) == "ask"


@pytest.mark.unit
@pytest.mark.parametrize("short_name", ["queue_add", "queue_stop"], ids=["queue_add", "queue_stop"])
def test_the_queue_tools_that_no_layer_denies_keep_their_prompt(
    tmp_path, hook_runner, make_config, short_name
):
    """Neither composing onto an idle queue nor halting one is ever deferred.

    `writes_check` allows every lane-addressed tool and neither of these refuses
    on posture, so this prompt is the only gate they have. `queue_add` starts
    nothing — the server withholds the launch token and the bridge refuses
    `launch_token_required` once the queue drains — and a plain stop is ungated
    everywhere by design.
    """
    config = _posture_config(make_config, {"type": "mock", "writes_enabled": False})

    result = hook_runner(
        "osprey_approval.py",
        f"mcp__bluesky__{short_name}",
        {"draft_revision": 3},
        config_path=config,
        cwd=tmp_path,
        hook_config=POSTURE_HOOK_CONFIG,
    )

    assert _decision(result) == "ask"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("control_system", "target", "decision"),
    [
        pytest.param(DISARMED_LIVE_CONFIG, "live", None, id="disarmed-block-on-the-machine"),
        pytest.param(DISARMED_LIVE_CONFIG, "va", "ask", id="global-yes-still-arms-the-sim"),
        pytest.param(ARMED_LIVE_ONLY_CONFIG, "live", "ask", id="armed-block-on-the-machine"),
        pytest.param(ARMED_LIVE_ONLY_CONFIG, "va", None, id="global-no-leaves-the-sim-unarmed"),
    ],
)
def test_channel_write_follows_the_session_target(
    tmp_path, hook_runner, make_config, control_system, target, decision
):
    """One tool, two configs, both targets — the posture is per target.

    Each config is the other's mirror, so no row can pass on the deployment-wide
    key alone. `ARMED_LIVE_ONLY_CONFIG` is what makes the state file
    load-bearing: its most-restrictive answer is "not armed", so the `live` row
    can only prompt if the hook actually read the session's target.
    """
    config = _posture_config(make_config, control_system)
    _write_session_state(tmp_path, target=target)

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "SR:QF:SP", "value": 1.5}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=POSTURE_HOOK_CONFIG,
    )

    assert _decision(result) == decision


@pytest.mark.unit
def test_a_config_that_states_no_posture_prompts_exactly_as_before(
    tmp_path, hook_runner, make_config
):
    """No `writes_enabled` anywhere is the shape every deployment used to have.

    Silence is not a refusal. Reading it as one would make the approval prompt
    disappear on every project written before the key existed.
    """
    config = _posture_config(make_config, NO_POSTURE_CONFIG)

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "SR:QF:SP", "value": 1.5}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=POSTURE_HOOK_CONFIG,
    )

    assert _decision(result) == "ask"


@pytest.mark.unit
def test_an_unidentifiable_target_takes_the_most_restrictive_posture(
    tmp_path, hook_runner, make_config
):
    """No state file means no target, and no target means both of them.

    The deployment is unarmed on either side here, so the defer stands and
    `writes_check`'s deny is what the agent sees. Picking the baseline instead
    would state a posture for a session that may have switched away from it.
    """
    config = _posture_config(make_config, {"type": "epics", "writes_enabled": False})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "SR:QF:SP", "value": 1.5}]},
        config_path=config,
        cwd=tmp_path,
        hook_config=POSTURE_HOOK_CONFIG,
    )

    assert _decision(result) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("services", "tool_input", "decision"),
    [
        pytest.param(TWO_LANE_SERVICES, {"lane": "bluesky_live"}, None, id="lane-on-the-machine"),
        pytest.param(TWO_LANE_SERVICES, {"lane": "bluesky"}, "ask", id="lane-on-the-sim"),
        pytest.param(TWO_LANE_SERVICES, {}, "ask", id="two-lanes-and-none-named"),
        pytest.param(TWO_LANE_SERVICES, {"lane": "bluesky_ghost"}, "ask", id="lane-not-rendered"),
        pytest.param(ONE_LANE_SERVICES, {}, None, id="one-lane-binds-itself"),
    ],
)
def test_queue_start_follows_the_target_its_lane_serves(
    tmp_path, hook_runner, make_config, services, tool_input, decision
):
    """A start is addressed by LANE, and only a PLACED lane may be deferred.

    Which machine a plan lane drives is render-time truth in the config, so a
    start bound to the disarmed lane defers whatever the session is pointed at:
    `queue_start` re-reads that same posture and refuses before its bridge is
    called. A start naming no lane binds to the one lane a single-lane
    deployment has, which is what the tool's own `_bind_lane` does.

    The two "cannot place it" rows are the ones that must NOT defer. Nothing
    else denies a lane-addressed tool — `writes_check` allows them all — so
    without a placed lane there is no guaranteed refusal behind a defer, and
    this prompt is the only gate the start has.
    """
    config = _posture_config(make_config, DISARMED_LIVE_CONFIG, services=services)

    result = hook_runner(
        "osprey_approval.py",
        "mcp__bluesky__queue_start",
        tool_input,
        config_path=config,
        cwd=tmp_path,
        hook_config=POSTURE_HOOK_CONFIG,
    )

    assert _decision(result) == decision


# ============================================================================
# The composed invariant: a defer must never be the last word
# ============================================================================

#: The layer that refuses a call this hook defers on. `None` is the state the
#: matrix forbids a defer in — nothing would stop the call.
_BY_WRITES_CHECK = "osprey_writes_check denies it"
_BY_LANE_GATE = "queue_start refuses before its bridge is called"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("control_system", "services", "tool_name", "tool_input", "target", "guarantor"),
    [
        pytest.param(
            DISARMED_LIVE_CONFIG,
            None,
            "mcp__controls__channel_write",
            {"operations": [{"channel": "SR:QF:SP", "value": 1.5}]},
            "live",
            _BY_WRITES_CHECK,
            id="channel_write-on-the-disarmed-machine",
        ),
        pytest.param(
            DISARMED_LIVE_CONFIG,
            None,
            "mcp__controls__channel_write",
            {"operations": [{"channel": "SR:QF:SP", "value": 1.5}]},
            "va",
            None,
            id="channel_write-on-the-armed-simulator",
        ),
        pytest.param(
            {"type": "mock", "writes_enabled": False},
            None,
            "mcp__python__execute",
            {"code": "caput('SR:QF:SP', 1.5)", "execution_mode": "readwrite"},
            None,
            _BY_WRITES_CHECK,
            id="readwrite-execute-unarmed",
        ),
        pytest.param(
            {"type": "mock", "writes_enabled": False},
            None,
            "mcp__python__execute",
            {"code": "print(1)", "execution_mode": "readonly"},
            None,
            None,
            id="readonly-execute-unarmed",
        ),
        pytest.param(
            NO_POSTURE_CONFIG,
            None,
            "mcp__controls__channel_write",
            {"operations": [{"channel": "SR:QF:SP", "value": 1.5}]},
            None,
            None,
            id="no-posture-stated",
        ),
        pytest.param(
            DISARMED_LIVE_CONFIG,
            TWO_LANE_SERVICES,
            "mcp__bluesky__queue_start",
            {"lane": "bluesky_live"},
            None,
            _BY_LANE_GATE,
            id="queue_start-on-the-disarmed-lane",
        ),
        pytest.param(
            DISARMED_LIVE_CONFIG,
            TWO_LANE_SERVICES,
            "mcp__bluesky__queue_start",
            {},
            None,
            None,
            id="queue_start-with-no-lane-to-place",
        ),
        pytest.param(
            DISARMED_LIVE_CONFIG,
            TWO_LANE_SERVICES,
            "mcp__bluesky__queue_add",
            {"draft_revision": 3},
            None,
            None,
            id="queue_add-composes-tokenless",
        ),
    ],
)
def test_a_defer_is_never_the_last_word(
    tmp_path,
    hook_runner,
    make_config,
    control_system,
    services,
    tool_input,
    tool_name,
    target,
    guarantor,
):
    """Both write hooks over one call: an approval defer must leave a refusal behind.

    Deferring emits no decision, so the call proceeds unless something else
    refuses it. Each row names the layer that would, and `None` names a row
    where nothing does — which is precisely where this hook must keep prompting.
    `osprey_writes_check` is the one guarantor a pytest process can observe, so
    rows naming it are checked against the deny it actually emits; the lane gate
    lives in `queue_start` itself, which no hook run reaches.
    """
    config = _posture_config(make_config, control_system, services=services)
    if target is not None:
        _write_session_state(tmp_path, target=target)

    def run(hook):
        return _decision(
            hook_runner(
                hook,
                tool_name,
                tool_input,
                config_path=config,
                cwd=tmp_path,
                hook_config=POSTURE_HOOK_CONFIG,
            )
        )

    approval_deferred = run("osprey_approval.py") is None

    if guarantor is None:
        assert approval_deferred is False, (
            "nothing refuses this call, so the approval prompt is its only gate"
        )
    else:
        assert approval_deferred is True
    if guarantor == _BY_WRITES_CHECK:
        assert run("osprey_writes_check.py") == "deny"
