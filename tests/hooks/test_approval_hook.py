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
import sys

import pytest

from tests._control_context_fixtures import write_control_context, write_server_report
from tests._guarded_run import (
    LINKED_DIRECTORY_REFUSAL,
    PLANTED_REFUSAL,
    PLANTED_REFUSED,
    held_lock,
    link_directory,
    lock_path,
    plant_kind,
)

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


#: The values a hand edit can leave on ``approval.enabled`` that are not a bool.
NON_BOOL_SWITCHES = [None, "false", 0, "", 1]

#: A write the policy asks about and a read the policy skips: the deny covers both.
SWITCH_CALLS = [
    pytest.param(
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        id="write",
    ),
    pytest.param("mcp__controls__channel_read", {"channels": ["TEST:PV"]}, id="skip-read"),
]


def _switch_config(make_config, approval):
    return make_config({"approval": approval, "control_system": {"writes_enabled": True}})


def _switch_sentence(value):
    return (
        f"approval.enabled is {value!r}, not true or false; "
        "set approval.enabled to true or false in profile.yml and rebuild"
    )


def _section_sentence(value):
    return (
        f"approval is {value!r}, not a mapping; "
        "set approval.enabled to true or false in profile.yml and rebuild"
    )


def _deny_reason(result):
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    return output["permissionDecisionReason"]


@pytest.mark.parametrize(("tool_name", "tool_input"), SWITCH_CALLS)
@pytest.mark.parametrize("enabled", NON_BOOL_SWITCHES)
def test_a_non_bool_switch_denies_every_call(
    tmp_path, hook_runner, make_config, enabled, tool_name, tool_input
):
    """A switch that is not true or false asks nobody and allows nothing."""
    config = _switch_config(make_config, {**DEFAULT_TOOLS_CONFIG, "enabled": enabled})

    result = hook_runner(
        "osprey_approval.py",
        tool_name,
        tool_input,
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _deny_reason(result) == _switch_sentence(enabled)


@pytest.mark.parametrize(("tool_name", "tool_input"), SWITCH_CALLS)
@pytest.mark.parametrize("section", [None, "disabled"], ids=["bare", "string"])
def test_a_section_that_is_not_a_mapping_denies_every_call(
    tmp_path, hook_runner, make_config, section, tool_name, tool_input
):
    """A bare ``approval:`` line is a deny with the fix, never a crash."""
    config = _switch_config(make_config, section)

    result = hook_runner(
        "osprey_approval.py",
        tool_name,
        tool_input,
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _deny_reason(result) == _section_sentence(section)


def test_a_non_bool_switch_deny_files_a_refused_record(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The deny is a refusal on the audit trail — `refused`, reason `approval_config_invalid`."""
    from pathlib import Path

    from tests._control_context_fixtures import pin_identity, state_dir_under

    pin_identity(monkeypatch)
    root = tmp_path / "agent_data"
    state_dir_under(root).mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=_switch_config(make_config, {**DEFAULT_TOOLS_CONFIG, "enabled": "false"}),
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _deny_reason(result) == _switch_sentence("false")
    records = [
        json.loads(line)
        for path in Path(tmp_path).rglob("hook_approval.jsonl")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["decision"] == "refused"
    assert records[0]["reason"] == "approval_config_invalid"
    assert records[0]["subject"] == "mcp__controls__channel_write"


def test_a_non_bool_switch_denies_under_dispatch(tmp_path, hook_runner, make_config, monkeypatch):
    """A headless dispatch run gets the same deny; nobody is there to ask."""
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=_switch_config(make_config, {**DEFAULT_TOOLS_CONFIG, "enabled": 0}),
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _deny_reason(result) == _switch_sentence(0)


@pytest.mark.parametrize("enabled", [*NON_BOOL_SWITCHES, "true"])
def test_the_hook_spells_the_switch_fault_like_the_build(enabled):
    """The standalone hook restates the build's sentence; the two never drift."""
    from osprey.cli.build_posture_check import approval_enabled_fault
    from tests.hooks.conftest import import_hook

    hook = import_hook("osprey_approval")
    rendered = {"approval": {**DEFAULT_TOOLS_CONFIG, "enabled": enabled}}
    assert hook._APPROVAL_CONFIG_INVALID_REASON.format(value=enabled) == approval_enabled_fault(
        rendered
    )


@pytest.mark.parametrize("section", [None, "disabled"], ids=["bare", "string"])
def test_the_hook_spells_the_section_fault_like_the_build(section):
    from osprey.cli.build_posture_check import approval_enabled_fault
    from tests.hooks.conftest import import_hook

    hook = import_hook("osprey_approval")
    assert hook._APPROVAL_SECTION_INVALID_REASON.format(value=section) == approval_enabled_fault(
        {"approval": section}
    )


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
        {"code": "write_channel('PV', 1.0)", "execution_mode": "readwrite"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"


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


def test_approval_python_write_creates_notebook(tmp_path, hook_runner, make_config):
    """Approval for python_execute with write patterns creates a pre-execution notebook."""
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "selective"},
            "control_system": {"writes_enabled": True},
            "artifact_server": {"host": "127.0.0.1", "port": 10200},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "write_channel('PV', 1.0)", "execution_mode": "readwrite"},
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
        {"code": "write_channel('PV', 5.0)", "execution_mode": "readwrite"},
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


def test_framework_pattern_detection_import_fallback(tmp_path, hook_runner, make_config):
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


def test_entry_publish_always_asks(tmp_path, hook_runner, make_config):
    """Publishing an entry reaches the facility's logbook, so it prompts.

    It falls to `default_policy` rather than naming a policy of its own, which
    is fail-closed and is what the shipped presets state explicitly.
    """
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    result = hook_runner(
        "osprey_approval.py",
        "mcp__ariel__entry_publish",
        {"entry_id": "e-1"},
        config_path=config,
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


@pytest.mark.parametrize(
    "tool_name",
    ["mcp__ring__channel_write", "mcp__ring__control_target_set"],
    ids=["channel_write", "control_target_set"],
)
def test_clone_approval_gated_tools_still_ask(tmp_path, hook_runner, make_config, tool_name):
    """A clone renames only the server prefix, never the tool.

    The registry rewrites each rule's matcher to `mcp__<clone>__<tool>`, so the
    hook fires; the policy has to be resolved from the SHORT name or every
    approval-gated controls tool on a clone falls through unrecognised.
    """
    config = make_config({"approval": DEFAULT_TOOLS_CONFIG})

    result = hook_runner(
        "osprey_approval.py",
        tool_name,
        {"channel": "TEST:PV", "value": 1.0},
        config_path=config,
        cwd=tmp_path,
        hook_config={
            "server_prefixes": ["mcp__ring__"],
            "approval_prefixes": ["mcp__ring__"],
        },
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


# ============================================================================
# Pattern parity — fallback patterns must match framework patterns
# ============================================================================


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


def test_fallback_covers_pvaccess_write_idioms(hook_module):
    """The fallback list must carry the pvaPy (``pvaccess``) write spellings.

    Pinned for the same reason p4p is: parity alone would stay green if both
    lists lost them together. The generic ``.put(`` misses the typed setters,
    asyncPut and parsePut, which are matched unanchored, for an import that
    never writes the token pvaccess; RpcClient is pvaPy's rpc.
    """
    fallback_patterns = hook_module("osprey_approval")._FALLBACK_WRITE_PATTERNS

    for pattern in (
        r"\.put(?:Get|Boolean|Byte|Double|Float|Int|Long|Short|String|ScalarArray"
        r"|UByte|UInt|ULong|UShort|AsDoubleArray)\w*\s*\(",
        r"\.asyncPut\s*\(",
        r"\.parsePut\w*\s*\(",
        r"\bRpcClient\s*\(",
        r"\bpvaccess\b[\s\S]*?\.invoke\s*\(",
        r"\b(?:PvaServer|PvaMirrorServer|RpcServer|CaIoc)\b",
    ):
        assert pattern in fallback_patterns


def test_fallback_covers_the_doocs_write_idiom(hook_module):
    """The fallback list must carry the DOOCS write spelling.

    ``doocs4py.set`` is what the shipped DOOCS connector writes through, so a
    deployment on that connector is guaranteed to have the library. Anchored
    to ``doocs4py`` on purpose: a bare ``.set(`` would flag every set() call in
    ordinary analysis code.
    """
    fallback_patterns = hook_module("osprey_approval")._FALLBACK_WRITE_PATTERNS

    assert r"\bdoocs4py\b[\s\S]*?\.set\s*\(" in fallback_patterns
    assert r"\.set\s*\(" not in fallback_patterns


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
        pytest.param(
            "import pvaccess as pva\npva.Channel('SR:A:SP', pva.CA).putDouble(1.0)\n",
            id="pvaccess-typed-put",
        ),
        pytest.param(
            "import pvaccess\npvaccess.Channel('SR:A:SP').asyncPut(pv, cb, err)\n",
            id="pvaccess-async-put",
        ),
        pytest.param(
            "import pvaccess\npvaccess.MultiChannel(['SR:A']).putAsDoubleArray([1.0])\n",
            id="pvaccess-multichannel-put",
        ),
        pytest.param(
            "from pvaccess import RpcClient\nRpcClient('SR:C').invoke(request)\n",
            id="pvaccess-rpc",
        ),
        pytest.param(
            "import importlib\n"
            "importlib.import_module('pva' + 'ccess').Channel('SR:A:SP').putDouble(1.0)\n",
            id="pvaccess-typed-put-built-import",
        ),
        pytest.param(
            "m = __import__('pva' + 'ccess')\nm.Channel('SR:A:SP').asyncPut(pv, cb, err)\n",
            id="pvaccess-async-put-built-import",
        ),
    ],
)
def test_fallback_regexes_match_pva_client_code(hook_module, code):
    """The pinned fallback regexes fire on code an agent would actually write."""
    import re as _re

    fallback_patterns = hook_module("osprey_approval")._FALLBACK_WRITE_PATTERNS

    assert any(_re.search(p, code) for p in fallback_patterns)


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
# `osprey_writes_check`'s deny for the tools that follow the control target, or by
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
    "bluesky": {"port": 10080, "target": "va"},
    "bluesky_live": {"port": 10081, "target": "live"},
}

#: One rendered plan lane, serving the machine.
ONE_LANE_SERVICES = {"bluesky": {"port": 10080, "target": "live"}}


def _posture_config(make_config, control_system, services=None):
    """A config that prompts for everything, over the given control_system block."""
    config = {
        "approval": {"enabled": True, "default_policy": "always"},
        "control_system": control_system,
    }
    if services is not None:
        config["services"] = services
    return make_config(config)


def _write_session_state(repo_root, target, posture=None):
    """Point the deployment at *target* with the record the hooks read.

    Lands under ``<repo_root>/var/agent_data``, which is the root the hooks
    derive when nothing stamps ``OSPREY_AGENT_DATA_ROOT`` and the subprocess
    runs with *repo_root* as its ``cwd`` — so the read is the real one.

    A live server's report is written beside it so the prompt's ``Target:``
    line can name the machine; the identity is the reports', while the target
    itself is the record's.
    """
    root = repo_root / "var" / "agent_data"
    write_control_context(root, target=target, generation=1, posture=posture)
    write_server_report(
        root,
        os.getpid(),
        applied_target=target,
        applied_generation=1,
        targets={
            "live": {"label": "Storage ring", "endpoint": "pva://gw:5075", "real_machine": True},
            "va": {"label": "Virtual accelerator", "endpoint": "pva://127.0.0.1:5074"},
        },
    )


def _decision(result):
    """The permission decision in a hook result, or ``None`` when it emitted none."""
    if result is None:
        return None
    return result["hookSpecificOutput"]["permissionDecision"]


def test_readonly_execute_still_prompts_when_writes_are_not_armed(
    tmp_path, hook_runner, make_config
):
    """A readonly execution is not a write, so the posture never reaches it.

    Deferring here would delete the approval prompt for ordinary analysis code
    on every unarmed deployment — the tool `writes_check` deliberately lets
    through is the one this hook must keep asking about.
    """
    config = _posture_config(
        make_config,
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
            "writes_enabled": False,
        },
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "print(get_channel('SR:CURRENT'))", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=POSTURE_HOOK_CONFIG,
    )

    assert _decision(result) == "ask"


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
    config = _posture_config(
        make_config,
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
            "writes_enabled": False,
        },
    )

    result = hook_runner(
        "osprey_approval.py",
        f"mcp__bluesky__{short_name}",
        {"draft_revision": 3},
        config_path=config,
        cwd=tmp_path,
        hook_config=POSTURE_HOOK_CONFIG,
    )

    assert _decision(result) == "ask"


@pytest.mark.parametrize(
    ("control_system", "target", "decision"),
    [
        pytest.param(DISARMED_LIVE_CONFIG, "live", None, id="disarmed-block-on-the-machine"),
        pytest.param(DISARMED_LIVE_CONFIG, "va", "ask", id="global-yes-still-arms-the-sim"),
        pytest.param(ARMED_LIVE_ONLY_CONFIG, "live", "ask", id="armed-block-on-the-machine"),
        pytest.param(ARMED_LIVE_ONLY_CONFIG, "va", None, id="global-no-leaves-the-sim-unarmed"),
    ],
)
def test_channel_write_follows_the_control_target(
    tmp_path, hook_runner, make_config, control_system, target, decision
):
    """One tool, two configs, both targets — the posture is per target.

    Each config is the other's mirror, so no row can pass on the deployment-wide
    key alone. `ARMED_LIVE_ONLY_CONFIG` is what makes the state file
    load-bearing: its most-restrictive answer is "not armed", so the `live` row
    can only prompt if the hook actually read the recorded control target.
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
    start bound to the disarmed lane defers whatever the deployment is pointed at:
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
            {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"serving": "in_process"}},
                "writes_enabled": False,
            },
            None,
            "mcp__python__execute",
            {"code": "caput('SR:QF:SP', 1.5)", "execution_mode": "readwrite"},
            None,
            _BY_WRITES_CHECK,
            id="readwrite-execute-unarmed",
        ),
        pytest.param(
            {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"serving": "in_process"}},
                "writes_enabled": False,
            },
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


def test_a_recorded_narrowing_defers_and_writes_check_denies(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """An operator narrowing composes like a config disarm: defer, with the deny behind it.

    The deployment arms this machine, so the config half of the posture answers
    "armed" — the narrowing recorded on the control-context record is the only
    thing refusing here, and ``osprey_writes_check`` reads the same record
    (``effective_writes_for``). Keeping the prompt would ask the human to
    approve a write the next hook is guaranteed to refuse; the matrix above
    pins the config half of this rule, this test pins the record half.
    """
    session_key = "4f1c2a7e-0000-4000-8000-000000000001"
    config = _posture_config(make_config, ARMED_LIVE_ONLY_CONFIG)
    _write_session_state(tmp_path, target="live")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", session_key)

    def run(hook):
        return _decision(
            hook_runner(
                hook,
                "mcp__controls__channel_write",
                {"operations": [{"channel": "SR:QF:SP", "value": 1.5}]},
                config_path=config,
                cwd=tmp_path,
                hook_config=POSTURE_HOOK_CONFIG,
            )
        )

    # Control: with nothing narrowed, the armed machine keeps its prompt.
    assert run("osprey_approval.py") == "ask"

    _write_session_state(tmp_path, target="live", posture={"live": "sandbox"})

    assert run("osprey_approval.py") is None, (
        "the narrowing is refused by writes_check, so the prompt must defer to that deny"
    )
    assert run("osprey_writes_check.py") == "deny"


def test_pre_execution_notebook_is_saved_without_launching_the_gallery(
    tmp_path, hook_module, monkeypatch
):
    """The review notebook lands in the store; no gallery server is started for it.

    The hook is a one-shot process. A server thread started from it dies with
    the process a few milliseconds later, so it can serve nobody, and its
    teardown races the interpreter's exit. A gallery that is already running
    picks the notebook up from the shared store; one that is not cannot be
    started from here.
    """
    import osprey.infrastructure.server_launcher as launcher
    from osprey.stores.artifact_store import ArtifactStore
    from osprey_connectors.workspace import reset_config_cache, resolve_shared_data_root

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"project_root: {tmp_path}\ncontrol_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n      serving: in_process\n"
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    reset_config_cache()

    launches: list[str] = []
    monkeypatch.setattr(launcher, "ensure_artifact_server", lambda: launches.append("artifact"))
    monkeypatch.setattr(launcher, "ensure_web_server", lambda key: launches.append(key))

    hook = hook_module("osprey_approval")
    hook._create_pre_execution_notebook("epics.caput('PV', 1.0)", "readonly", {})

    saved = [
        entry
        for entry in ArtifactStore(workspace_root=resolve_shared_data_root()).list_entries()
        if entry.tool_source == "osprey_approval"
    ]
    assert len(saved) == 1, "the review notebook was not saved"
    assert saved[0].artifact_type == "notebook"
    assert launches == []


# ============================================================================
# Raw client-write spellings in a readwrite `execute` are denied, not asked
# ============================================================================

RAW_PUT_CONFIG = {
    "approval": DEFAULT_TOOLS_CONFIG,
    "control_system": {"writes_enabled": True},
}


def _decision(result):
    return None if result is None else result["hookSpecificOutput"]["permissionDecision"]


def _run_execute(hook_runner, make_config, tmp_path, code, *, mode="readwrite", config=None):
    return hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": code, "execution_mode": mode},
        config_path=make_config(config or RAW_PUT_CONFIG),
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )


@pytest.mark.parametrize(
    ("code", "spelling"),
    [
        pytest.param("from epics import caput\ncaput('SR:QF:SP', 1.5)\n", "caput(", id="caput"),
        pytest.param("caput_many(['A', 'B'], [1, 2])\n", "caput_many(", id="caput_many"),
        pytest.param(
            "import epics\nepics.caput('SR:QF:SP', 1.5)\n", "epics.caput(", id="epics.caput"
        ),
        pytest.param(
            "from epics import PV\nPV('SR:QF:SP').put(1.5)\n", "PV(...).put(", id="PV-put"
        ),
        pytest.param(
            "import aioca\nawait aioca.caput('SR:QF:SP', 1.5)\n", "aioca.caput(", id="aioca.caput"
        ),
        pytest.param(
            "from osprey_connectors.control_system import write_door\n",
            "write_door",
            id="write_door",
        ),
        pytest.param(
            "from osprey_connectors.control_system import open_door\nwith open_door():\n    pass\n",
            "open_door",
            id="open_door",
        ),
    ],
)
def test_readwrite_raw_client_write_is_denied(tmp_path, hook_runner, make_config, code, spelling):
    """Each unambiguous raw spelling is denied with the write_channel remedy."""
    result = _run_execute(hook_runner, make_config, tmp_path, code)

    assert _decision(result) == "deny"
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason.startswith("raw client write refused")
    assert f"`{spelling}`" in reason
    assert "osprey.runtime.write_channel(address, value)" in reason
    assert "osprey.runtime.write_channels(" in reason


@pytest.mark.parametrize(
    "approval",
    [
        pytest.param({"enabled": False}, id="approval-disabled"),
        pytest.param({"enabled": True, "tools": {"execute": "skip"}}, id="execute-skip"),
        pytest.param({"enabled": True, "default_policy": "skip"}, id="default-skip"),
    ],
)
def test_raw_client_write_denied_even_when_approval_would_allow(
    tmp_path, hook_runner, make_config, approval
):
    """Neither a disabled approval gate nor a skip policy turns the deny into an allow."""
    config = {"approval": approval, "control_system": {"writes_enabled": True}}

    result = _run_execute(
        hook_runner, make_config, tmp_path, "caput('SR:QF:SP', 1.5)", config=config
    )

    assert _decision(result) == "deny"


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("# caput('SR:QF:SP', 1.5)\nprint(1)\n", id="comment"),
        pytest.param("note = \"caput('SR:QF:SP', 1.5) and open_door\"\n", id="string"),
        pytest.param('doc = """\nPV(\'X\').put(1)\nctxt.put(\'Y\', 2)\n"""\n', id="docstring"),
        pytest.param('msg = f"caput( with {x}"\n', id="fstring-text"),
    ],
)
def test_raw_spelling_in_data_still_asks(tmp_path, hook_runner, make_config, code):
    """A spelling only in a comment, a string or f-string text is not code: readwrite asks."""
    result = _run_execute(hook_runner, make_config, tmp_path, code)

    assert _decision(result) == "ask"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="f-string braces tokenize from 3.12")
def test_raw_spelling_inside_fstring_braces_is_denied(tmp_path, hook_runner, make_config):
    """From 3.12 the expression inside f-string braces is code, and is matched."""
    result = _run_execute(hook_runner, make_config, tmp_path, "s = f\"{caput('X', 1)}\"\n")

    assert _decision(result) == "deny"


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="before 3.12 an f-string is one token")
def test_raw_spelling_inside_fstring_braces_asks_before_312(tmp_path, hook_runner, make_config):
    """Before 3.12 the whole f-string is a STRING token and is stripped."""
    result = _run_execute(hook_runner, make_config, tmp_path, "s = f\"{caput('X', 1)}\"\n")

    assert _decision(result) == "ask"


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "from osprey.runtime import write_channel\nwrite_channel('SR:QF:SP', 1.5)\n",
            id="write_channel",
        ),
        pytest.param("write_channels({'SR:QF:SP': 1.5})\n", id="write_channels"),
        pytest.param("import queue\nq = queue.Queue()\nq.put(1)\n", id="queue.put"),
        pytest.param("mycaput('X', 1)\nrecaput_many(1)\n", id="caput-as-suffix"),
        pytest.param(
            "from p4p.client.thread import Context\nctxt = Context('pva')\nctxt.put('SR:QF:SP', 1.5)\n",
            id="p4p-put",
        ),
        pytest.param(
            "import pvaccess\npvaccess.Channel('SR:QF:SP').put(1.5)\n",
            id="pvaccess-put",
        ),
    ],
)
def test_sanctioned_and_ambiguous_spellings_still_ask(tmp_path, hook_runner, make_config, code):
    """write_channel(s), the ambiguous bare `.put(` and the PVAccess puts the
    runtime limits-checks rather than refuses keep the human ask."""
    result = _run_execute(hook_runner, make_config, tmp_path, code)

    assert _decision(result) == "ask"


def test_untokenizable_code_keeps_the_ask(tmp_path, hook_runner, make_config):
    """Source tokenize cannot read falls through to today's flow, never an allow."""
    result = _run_execute(hook_runner, make_config, tmp_path, "caput('X', 1\nx = '''unterminated")

    assert _decision(result) == "ask"


def test_readonly_raw_spelling_is_not_denied_here(tmp_path, hook_runner, make_config):
    """The deny is a readwrite rule; readonly keeps its pattern-detection ask."""
    result = _run_execute(
        hook_runner, make_config, tmp_path, "caput('SR:QF:SP', 1.5)", mode="readonly"
    )

    assert _decision(result) == "ask"


def test_execute_file_keeps_its_ask(tmp_path, hook_runner, make_config):
    """execute_file carries a path, not code, so it asks and the runtime block stands behind it."""
    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute_file",
        {"file_path": "open_door_caput.py", "code": "caput('X', 1)", "execution_mode": "readwrite"},
        config_path=make_config(RAW_PUT_CONFIG),
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )

    assert _decision(result) == "ask"


def test_raw_client_write_deny_files_a_refused_record(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The deny is a refusal on the audit trail — `refused`, reason `raw_client_write`."""
    import json
    from pathlib import Path

    from tests._control_context_fixtures import pin_identity, state_dir_under

    pin_identity(monkeypatch)
    root = tmp_path / "agent_data"
    state_dir_under(root).mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))

    result = _run_execute(hook_runner, make_config, tmp_path, "open_door()\n")

    assert _decision(result) == "deny"
    records = [
        json.loads(line)
        for path in Path(tmp_path).rglob("hook_approval.jsonl")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["decision"] == "refused"
    assert records[0]["reason"] == "raw_client_write"
    assert records[0]["subject"] == "mcp__python__execute"


# ---------------------------------------------------------------------------
# Guarded runs: the target's run lock on the execute prompt
# ---------------------------------------------------------------------------

#: Asks for every execute call, on a deployment whose baseline target is ``va``.
GUARDED_ALWAYS_CONFIG = {
    "approval": {
        "enabled": True,
        "default_policy": "always",
        "tools": {"execute": "always", "execute_file": "always"},
    },
    "control_system": {"type": "virtual_accelerator", "writes_enabled": True},
}

#: Asks only for readwrite or write-shaped execute calls.
GUARDED_SELECTIVE_CONFIG = {
    "approval": {
        "enabled": True,
        "default_policy": "always",
        "tools": {"execute": "selective", "execute_file": "selective"},
    },
    "control_system": {"type": "virtual_accelerator", "writes_enabled": True},
}


def _lock(repo_root, target="va"):
    """The run lock of *target* under *repo_root*, its directory created."""
    path = lock_path(target, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_guarded(hook_runner, make_config, tmp_path, tool_input, *, tool="execute", config=None):
    return hook_runner(
        "osprey_approval.py",
        f"mcp__python__{tool}",
        tool_input,
        config_path=make_config(config or GUARDED_ALWAYS_CONFIG),
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
    )


def _reason(result):
    return result["hookSpecificOutput"]["permissionDecisionReason"]


def _updated(result):
    return result["hookSpecificOutput"].get("updatedInput")


@pytest.mark.parametrize("present", [False, True], ids=["no-lock-file", "free-lock-file"])
def test_a_free_target_binds_its_name(tmp_path, hook_runner, make_config, present):
    if present:
        _lock(tmp_path).touch()
    tool_input = {"code": "print(1)", "execution_mode": "readwrite", "approved_target": "live"}

    result = _run_guarded(hook_runner, make_config, tmp_path, tool_input)

    assert _decision(result) == "ask"
    assert _updated(result) == {
        "code": "print(1)",
        "execution_mode": "readwrite",
        "approved_target": "va",
    }


def test_execute_file_carries_the_binding(tmp_path, hook_runner, make_config):
    result = _run_guarded(
        hook_runner,
        make_config,
        tmp_path,
        {"file_path": "scan.py", "execution_mode": "readwrite"},
        tool="execute_file",
    )

    assert _decision(result) == "ask"
    assert _updated(result)["approved_target"] == "va"
    assert _updated(result)["file_path"] == "scan.py"


@pytest.mark.parametrize("tool", ["execute", "execute_file"])
def test_a_readonly_call_keeps_its_prompt_and_carries_no_binding(
    tmp_path, hook_runner, make_config, tool
):
    result = _run_guarded(
        hook_runner,
        make_config,
        tmp_path,
        {"code": "print(1)", "file_path": "scan.py", "execution_mode": "readonly"},
        tool=tool,
    )

    assert _decision(result) == "ask"
    assert _updated(result) is None


def test_a_held_lock_refuses_busy_naming_its_holder(tmp_path, hook_runner, make_config):
    with held_lock("va", '{"pid": 31337, "started": "2026-10-02T09:30:00Z"}', root=tmp_path):
        result = _run_guarded(
            hook_runner, make_config, tmp_path, {"code": "print(1)", "execution_mode": "readwrite"}
        )

    assert _decision(result) == "deny"
    assert _reason(result) == (
        "a guarded run is in progress on va (pid 31337, started 2026-10-02T09:30:00Z)"
    )


def test_a_held_lock_with_no_holder_record_names_it_unknown(tmp_path, hook_runner, make_config):
    with held_lock("va", "", root=tmp_path):
        result = _run_guarded(
            hook_runner, make_config, tmp_path, {"code": "print(1)", "execution_mode": "readwrite"}
        )

    assert _decision(result) == "deny"
    assert _reason(result) == "a guarded run is in progress on va (pid unknown, started unknown)"


def test_a_free_lock_is_left_empty(tmp_path, hook_runner, make_config):
    path = _lock(tmp_path)

    _run_guarded(
        hook_runner, make_config, tmp_path, {"code": "print(1)", "execution_mode": "readwrite"}
    )

    assert path.read_bytes() == b""


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens a file of mode 000")
def test_an_unreadable_lock_is_denied(tmp_path, hook_runner, make_config):
    path = _lock(tmp_path)
    path.touch()
    path.chmod(0o000)
    try:
        result = _run_guarded(
            hook_runner, make_config, tmp_path, {"code": "print(1)", "execution_mode": "readwrite"}
        )
    finally:
        path.chmod(0o664)

    assert _decision(result) == "deny"
    assert _reason(result) == f"guarded-run lock unreadable: {path}"


def _planted_victim(tmp_path):
    """A file outside the guarded-run directory; ``(path, its bytes)``."""
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep\n")
    return victim, victim.read_bytes()


@pytest.mark.parametrize("kind", PLANTED_REFUSED)
def test_a_planted_lock_name_is_denied_by_name(kind, tmp_path, hook_runner, make_config):
    victim, before = _planted_victim(tmp_path)
    path = _lock(tmp_path)
    plant_kind(path, kind, victim)

    result = _run_guarded(
        hook_runner, make_config, tmp_path, {"code": "print(1)", "execution_mode": "readwrite"}
    )

    assert _decision(result) == "deny"
    assert _reason(result) == f"{path} {PLANTED_REFUSAL[kind]}"
    assert "unreadable" not in _reason(result)
    assert _updated(result) is None
    assert victim.read_bytes() == before


def test_a_linked_target_directory_is_denied_by_name(tmp_path, hook_runner, make_config):
    path = _lock(tmp_path)
    path.touch()
    level = path.parent
    real = link_directory(level)

    result = _run_guarded(
        hook_runner, make_config, tmp_path, {"code": "print(1)", "execution_mode": "readwrite"}
    )

    assert _decision(result) == "deny"
    assert _reason(result) == f"{level} {LINKED_DIRECTORY_REFUSAL}"
    assert _updated(result) is None
    assert (real / path.name).read_bytes() == b""
    assert sorted(entry.name for entry in real.iterdir()) == [path.name]


# ---------------------------------------------------------------------------
# The ask stamp records the fields the approval prompt bound the call to
# ---------------------------------------------------------------------------


@pytest.fixture
def stamp_dir(tmp_path, monkeypatch):
    """The identity state directory the hook files its ask stamps in."""
    from tests._control_context_fixtures import pin_identity, state_dir_under

    pin_identity(monkeypatch)
    root = tmp_path / "agent_data"
    state = state_dir_under(root)
    state.mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    return state


def _asked(hook_runner, make_config, tmp_path, tool, tool_input, config, tool_use_id):
    return hook_runner(
        "osprey_approval.py",
        f"mcp__python__{tool}",
        tool_input,
        config_path=make_config(config),
        cwd=tmp_path,
        hook_config=DEFAULT_APPROVAL_CONFIG,
        hook_input_extra={"hook_event_name": "PreToolUse", "tool_use_id": tool_use_id},
    )


def _stamp(stamp_dir, tool_use_id):
    import json

    return json.loads((stamp_dir / f"approval_ask_{tool_use_id}.json").read_text())


def _approved(mapping):
    return {key: value for key, value in mapping.items() if key.startswith("approved_")}


@pytest.mark.parametrize(
    ("tool", "config", "tool_input"),
    [
        pytest.param(
            "execute",
            GUARDED_SELECTIVE_CONFIG,
            {"code": "print(1)", "execution_mode": "readwrite"},
            id="execute-selective",
        ),
        pytest.param(
            "execute",
            GUARDED_ALWAYS_CONFIG,
            {"code": "print(1)", "execution_mode": "readwrite"},
            id="execute-always",
        ),
        pytest.param(
            "execute_file",
            GUARDED_ALWAYS_CONFIG,
            {"file_path": "scan.py", "execution_mode": "readwrite"},
            id="execute_file-always",
        ),
    ],
)
def test_the_ask_stamp_carries_the_approved_fields_of_the_envelope(
    tmp_path, hook_runner, make_config, stamp_dir, tool, config, tool_input
):
    agent_input = {**tool_input, "approved_target": "agent-target"}

    result = _asked(hook_runner, make_config, tmp_path, tool, agent_input, config, "toolu_rw1")

    assert _decision(result) == "ask"
    stamp = _stamp(stamp_dir, "toolu_rw1")
    assert _approved(stamp) == _approved(_updated(result))
    assert set(_approved(stamp)) == {"approved_target"}
    assert stamp["approved_target"] == "va"
    assert "agent-target" not in stamp.values()


def test_an_agent_supplied_field_the_hook_did_not_bind_never_reaches_the_stamp(
    tmp_path, hook_runner, make_config, stamp_dir
):
    tool_input = {
        "code": "print(1)",
        "execution_mode": "readwrite",
        "approved_view_sha256": "ab" * 32,
    }

    result = _asked(
        hook_runner,
        make_config,
        tmp_path,
        "execute",
        tool_input,
        GUARDED_ALWAYS_CONFIG,
        "toolu_rw2",
    )

    assert _decision(result) == "ask"
    stamp = _stamp(stamp_dir, "toolu_rw2")
    assert set(_approved(stamp)) == {"approved_target"}
    assert "approved_view_sha256" not in stamp


def test_an_agent_supplied_field_never_reaches_a_readonly_stamp(
    tmp_path, hook_runner, make_config, stamp_dir
):
    tool_input = {
        "code": "print(1)",
        "execution_mode": "readonly",
        "approved_target": "live",
    }

    result = _asked(
        hook_runner,
        make_config,
        tmp_path,
        "execute",
        tool_input,
        GUARDED_ALWAYS_CONFIG,
        "toolu_ro1",
    )

    assert _decision(result) == "ask"
    assert _updated(result) is None
    assert _approved(_stamp(stamp_dir, "toolu_ro1")) == {}


def test_an_ask_the_harness_never_received_leaves_no_approved_stamp(
    tmp_path, hook_module, monkeypatch
):
    approval = hook_module("osprey_approval")
    monkeypatch.setattr(approval, "_approval_state_dir", lambda _hook_input: str(tmp_path))

    class _Broken:
        def write(self, _text):
            raise OSError("stdout closed")

        def flush(self):
            raise OSError("stdout closed")

    monkeypatch.setattr(approval.sys, "stdout", _Broken())
    output = {
        "hookSpecificOutput": {
            "permissionDecision": "ask",
            "updatedInput": {"approved_target": "va"},
        }
    }

    with pytest.raises(OSError):
        approval._emit_ask(
            output,
            {"tool_use_id": "toolu_lost", "tool_name": "x"},
            {"approved_target": "va"},
        )

    assert list(tmp_path.glob("approval_ask_*.json")) == []


# ---------------------------------------------------------------------------
# pyaml_measure: the view, the stepping and the binding
# ---------------------------------------------------------------------------

_VIEW_BYTES = b"type: pyaml.accelerator\nmachine: SR\n"
_VIEW_RELATIVE = "data/pyaml/SR/configuration.yaml"
_HCOR = [f"SR:HCM:{index:02d}:SP" for index in range(1, 23)]
_VCOR = ["SR:VCM:01:SP"]
_QUADS = ["SR:QF:01:SP", "SR:QF:02:SP"]


def _measurement_block(**changes):
    import hashlib

    block = {
        "kinds": {
            "orm": {"hcor": "SR_HCM", "vcor": "SR_VCM"},
            "trm": {"quad": "SR_QF"},
        },
        "groups": {
            "bpm": ["SR:BPM:01:X", "SR:BPM:01:Y"],
            "hcor": _HCOR,
            "vcor": _VCOR,
            "quad": _QUADS,
        },
        "view_sha256": hashlib.sha256(_VIEW_BYTES).hexdigest(),
    }
    block.update(changes)
    return block


def _plant_view(render, data=_VIEW_BYTES):
    path = render / "data" / "pyaml" / "SR" / "configuration.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _measure(hook_runner, make_config, tmp_path, tool_input, *, block=None, cwd=None, extra=None):
    hook_config = {**DEFAULT_APPROVAL_CONFIG, "measurement": {"SR": block or _measurement_block()}}
    return hook_runner(
        "osprey_approval.py",
        "mcp__python__pyaml_measure",
        tool_input,
        config_path=make_config(GUARDED_ALWAYS_CONFIG),
        cwd=cwd or tmp_path,
        hook_config=hook_config,
        hook_input_extra=extra,
    )


def _denied_reason(result):
    assert _decision(result) == "deny"
    return result["hookSpecificOutput"]["permissionDecisionReason"]


def test_the_measurement_prompt_shows_the_view_and_every_stepping_argument(
    tmp_path, hook_runner, make_config
):
    import hashlib

    _plant_view(tmp_path)

    result = _measure(hook_runner, make_config, tmp_path, {"kind": "orm", "model": "SR"})

    assert _decision(result) == "ask"
    reason = _reason(result)
    digest = hashlib.sha256(_VIEW_BYTES).hexdigest()
    assert 'Model: SR\nKind: orm\nModulation: "bipolar"\n' in reason
    assert f"pyAML view: {_VIEW_RELATIVE}\n  sha256 {digest}\n" in reason
    listed = "".join(f'\n  "{address}"' for address in [*_HCOR, *_VCOR])
    assert f"Setpoints stepped: 23{listed}\n" in reason
    assert " more" not in reason
    assert _updated(result)["approved_view_sha256"] == digest


def test_a_measurement_stepping_more_than_a_prompt_lists_is_denied_unasked(
    tmp_path, hook_runner, make_config, stamp_dir
):
    _plant_view(tmp_path)
    hcor = [f"SR:HCM:{index:03d}:SP" for index in range(1, 501)]
    groups = {**_measurement_block()["groups"], "hcor": hcor}

    result = _measure(
        hook_runner,
        make_config,
        tmp_path,
        {"kind": "orm"},
        block=_measurement_block(groups=groups),
        extra={"hook_event_name": "PreToolUse", "tool_use_id": "toolu_pm_many"},
    )

    assert _denied_reason(result) == (
        "pyaml_measure: orm on SR steps 501 setpoints, more than an approval prompt lists"
    )
    assert not (stamp_dir / "approval_ask_toolu_pm_many.json").exists()


def test_the_named_correctors_are_the_ones_listed(tmp_path, hook_runner, make_config):
    _plant_view(tmp_path)

    result = _measure(
        hook_runner,
        make_config,
        tmp_path,
        {"kind": "orm", "correctors": [_VCOR[0]], "modulation": "unipolar"},
    )

    reason = _reason(result)
    assert 'Modulation: "unipolar"' in reason
    assert f'Setpoints stepped: 1\n  "{_VCOR[0]}"' in reason


def test_a_group_step_shows_the_group_it_expands(tmp_path, hook_runner, make_config):
    _plant_view(tmp_path)

    result = _measure(
        hook_runner,
        make_config,
        tmp_path,
        {"kind": "trm", "modulation": "unipolar", "by": "group"},
    )

    reason = _reason(result)
    assert 'By: "group"\n  group SR_QF steps 2 setpoints as one knob' in reason
    assert f'Setpoints stepped: 2\n  "{_QUADS[0]}"\n  "{_QUADS[1]}"' in reason


def test_a_group_step_with_modulation_omitted_shows_unipolar(tmp_path, hook_runner, make_config):
    _plant_view(tmp_path)

    result = _measure(hook_runner, make_config, tmp_path, {"kind": "trm", "by": "group"})

    reason = _reason(result)
    assert 'Modulation: "unipolar"\nBy: "group"\n' in reason


_RF = ["SR:RF:FREQ:SP"]
_SEXTS = ["SR:SF:01:SP", "SR:SF:02:SP", "SR:SD:01:SP"]
_RF_PLANT = "DEFAULT_RF_PLANT"


def _rf_block(**changes):
    block = _measurement_block(
        kinds={
            "orm": {"hcor": "SR_HCM", "vcor": "SR_VCM"},
            "dispersion": {"rf": _RF_PLANT},
            "trm": {"quad": "SR_QF"},
            "crm": {"rf": _RF_PLANT, "sext": "SR_S"},
            "chromaticity_monitor": {"rf": _RF_PLANT},
        },
        groups={
            "bpm": ["SR:BPM:01:X", "SR:BPM:01:Y"],
            "hcor": _HCOR,
            "vcor": _VCOR,
            "quad": _QUADS,
            "sext": _SEXTS,
        },
        rf=_RF,
    )
    block.update(changes)
    return block


@pytest.mark.parametrize(
    ("kind", "stepped"),
    [
        ("dispersion", _RF),
        ("crm", [*_RF, *_SEXTS]),
        ("chromaticity_monitor", _RF),
    ],
)
def test_a_kind_that_steps_the_rf_lists_its_setpoint(
    tmp_path, hook_runner, make_config, kind, stepped
):
    _plant_view(tmp_path)

    result = _measure(hook_runner, make_config, tmp_path, {"kind": kind}, block=_rf_block())

    assert _decision(result) == "ask"
    listed = "".join(f'\n  "{address}"' for address in stepped)
    assert f"Setpoints stepped: {len(stepped)}{listed}" in _reason(result)


def test_named_correctors_add_no_corrector_to_a_dispersion(tmp_path, hook_runner, make_config):
    _plant_view(tmp_path)

    result = _measure(
        hook_runner,
        make_config,
        tmp_path,
        {"kind": "dispersion", "correctors": [_VCOR[0]]},
        block=_rf_block(),
    )

    assert f'Setpoints stepped: 1\n  "{_RF[0]}"' in _reason(result)


@pytest.mark.parametrize("kind", ["dispersion", "crm", "chromaticity_monitor"])
def test_a_kind_that_steps_the_rf_is_denied_when_the_view_names_no_rf_setpoint(
    tmp_path, hook_runner, make_config, kind
):
    _plant_view(tmp_path)

    result = _measure(hook_runner, make_config, tmp_path, {"kind": kind}, block=_rf_block(rf=[]))

    assert _denied_reason(result) == (
        f"pyaml_measure: {kind} steps the RF frequency and the pyAML view of SR names no "
        f"RF setpoint: {_VIEW_RELATIVE}"
    )


def test_each_group_line_counts_that_groups_own_setpoints(tmp_path, hook_runner, make_config):
    _plant_view(tmp_path)

    result = _measure(
        hook_runner, make_config, tmp_path, {"kind": "crm", "by": "group"}, block=_rf_block()
    )

    reason = _reason(result)
    assert f'By: "group"\n  group SR_S steps {len(_SEXTS)} setpoints as one knob\n' in reason
    assert f"Setpoints stepped: {len(_SEXTS) + len(_RF)}\n" in reason


def test_the_binding_overwrites_what_the_agent_supplied(tmp_path, hook_runner, make_config):
    import hashlib

    _plant_view(tmp_path)
    agent_input = {
        "kind": "orm",
        "approved_target": "agent-target",
        "approved_view_sha256": "ab" * 32,
    }

    result = _measure(hook_runner, make_config, tmp_path, agent_input)

    assert _updated(result) == {
        "kind": "orm",
        "approved_target": "va",
        "approved_view_sha256": hashlib.sha256(_VIEW_BYTES).hexdigest(),
    }


_HOSTILE = "SR\u009bQF\u202e:\u200bSP\n = 9"


def test_prompt_text_renders_each_setpoint_as_one_printable_line(
    tmp_path, hook_runner, make_config
):
    _plant_view(tmp_path)
    hostile = [_HOSTILE, _HOSTILE.replace("\u200b", "")]
    block = _measurement_block(groups={**_measurement_block()["groups"], "quad": hostile})

    result = _measure(hook_runner, make_config, tmp_path, {"kind": "trm"}, block=block)

    lines = _reason(result).splitlines()
    start = lines.index("Setpoints stepped: 2")
    first, second = lines[start + 1 : start + 3]
    assert all(line.isascii() and line.isprintable() for line in (first, second))
    assert first == f"  {json.dumps(hostile[0])}"
    assert second == f"  {json.dumps(hostile[1])}"
    assert first != second


def test_a_view_whose_bytes_differ_from_the_record_is_denied(tmp_path, hook_runner, make_config):
    _plant_view(tmp_path, _VIEW_BYTES + b"# edited\n")

    result = _measure(hook_runner, make_config, tmp_path, {"kind": "orm"})

    assert _denied_reason(result) == (
        f"pyaml_measure: the pyAML view of SR differs from the one the build recorded: "
        f"{_VIEW_RELATIVE}"
    )


@pytest.mark.parametrize(
    "tool_input",
    [
        {"kind": "crm"},
        {"kind": "bba"},
        {"kind": ["orm"]},
        {"kind": "orm", "model": "LINE"},
    ],
    ids=["unlisted-kind", "unknown-kind", "non-string-kind", "model-without-view"],
)
def test_a_call_the_block_does_not_map_reaches_the_tool_unasked(
    tmp_path, hook_runner, make_config, stamp_dir, tool_input
):
    _plant_view(tmp_path)

    result = _measure(
        hook_runner,
        make_config,
        tmp_path,
        tool_input,
        extra={"hook_event_name": "PreToolUse", "tool_use_id": "toolu_um1"},
    )

    assert _decision(result) == "allow"
    assert _updated(result) is None
    assert not (stamp_dir / "approval_ask_toolu_um1.json").exists()


def test_the_prompt_lists_only_what_the_blocks_map_steps(tmp_path, hook_runner, make_config):
    _plant_view(tmp_path)
    block = _measurement_block(kinds={"orm": {"hcor": "SR_HCM"}})

    result = _measure(hook_runner, make_config, tmp_path, {"kind": "orm"}, block=block)

    listed = "".join(f'\n  "{address}"' for address in _HCOR)
    assert f"Setpoints stepped: {len(_HCOR)}{listed}" in _reason(result)
    assert all(address not in _reason(result) for address in _VCOR)


@pytest.mark.parametrize("layout", ["host", "container"])
def test_a_host_and_a_container_render_resolve_the_same_view(
    tmp_path, hook_runner, make_config, layout
):
    render = tmp_path / "repo" / "build" if layout == "host" else tmp_path / "project"
    if layout == "host":
        (tmp_path / "repo" / "profile.yml").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "profile.yml").write_text("name: demo\n", encoding="utf-8")
    _plant_view(render)

    result = _measure(hook_runner, make_config, tmp_path, {"kind": "orm"}, cwd=render)

    assert _decision(result) == "ask"
    assert f"pyAML view: {_VIEW_RELATIVE}\n" in _reason(result)


def test_the_measurement_stamp_holds_its_two_approved_fields(
    tmp_path, hook_runner, make_config, stamp_dir
):
    _plant_view(tmp_path)

    result = _measure(
        hook_runner,
        make_config,
        tmp_path,
        {"kind": "orm", "approved_view_sha256": "ab" * 32},
        extra={"hook_event_name": "PreToolUse", "tool_use_id": "toolu_pm1"},
    )

    stamp = _stamp(stamp_dir, "toolu_pm1")
    assert _approved(stamp) == _approved(_updated(result))
    assert set(_approved(stamp)) == {"approved_target", "approved_view_sha256"}


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("execute", {"code": "print(1)", "execution_mode": "readwrite"}),
        ("execute_file", {"file_path": "scan.py", "execution_mode": "readwrite"}),
    ],
)
def test_a_view_digest_rides_only_in_a_measurements_input(
    tmp_path, hook_runner, make_config, stamp_dir, tool, tool_input
):
    result = _asked(
        hook_runner,
        make_config,
        tmp_path,
        tool,
        {**tool_input, "approved_view_sha256": "ab" * 32},
        GUARDED_ALWAYS_CONFIG,
        "toolu_nv1",
    )

    assert _decision(result) == "ask"
    assert "approved_view_sha256" not in _updated(result)
    stamp = _stamp(stamp_dir, "toolu_nv1")
    assert set(_approved(stamp)) == {"approved_target"}


def test_the_hook_restates_what_the_tool_steps_and_its_defaults(hook_module):
    from osprey.mcp_server.python_executor.tools import _pyaml_measure_args as tool

    approval = hook_module("osprey_approval")
    assert approval._PYAML_BY_KINDS == tool.BY_KINDS
    assert approval._PYAML_DEFAULT_MODULATION == tool.DEFAULT_MODULATION
    assert approval._PYAML_DEFAULT_BY == tool.DEFAULT_BY
    assert approval._PYAML_GROUP == tool.GROUP
    assert approval._PYAML_GROUP_MODULATION == tool.GROUP_MODULATION
    assert approval._PYAML_CORRECTOR_ROLES == tool._CORRECTOR_ROLES


def test_the_hook_restates_the_view_writers_names(hook_module):
    from osprey.facility.views import pyaml as view

    approval = hook_module("osprey_approval")
    assert approval._PYAML_VIEW_ROOT == ("data", view.PYAML_DIR)
    assert approval._PYAML_CONFIGURATION_FILE == view.CONFIGURATION_FILE
