"""Tests for the osprey_error_guidance PostToolUse hook.

This hook detects structured errors in MCP tool responses and injects
error-handling guidance into Claude's context via additionalContext.
It should never block execution.
"""

import json

import pytest


# -- Structured error envelope (matches common.make_error) --


def _make_error_response(error_type, message, suggestions=None):
    """Build the standard OSPREY error envelope as a JSON string."""
    return json.dumps(
        {
            "error": True,
            "error_type": error_type,
            "error_message": message,
            "suggestions": suggestions or [],
        }
    )


# -- Positive detection tests --


@pytest.mark.unit
def test_connection_error_injects_guidance(hook_runner, make_config):
    """Connection error triggers additionalContext with guidance."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=_make_error_response(
            "connection_error",
            "Failed to connect to the control system: Connection refused",
        ),
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Connection" in ctx
    assert "error-handling" in ctx.lower() or "error-handling.md" in ctx


@pytest.mark.unit
def test_timeout_error_injects_guidance(hook_runner, make_config):
    """Timeout error is classified as Connection class."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__archiver_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=_make_error_response(
            "timeout_error",
            "archiver_read timed out after 30s",
        ),
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Connection" in ctx


@pytest.mark.unit
def test_validation_error_injects_guidance(hook_runner, make_config):
    """Validation errors produce Validation class guidance."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-workspace__artifact_save",
        {"content": "test"},
        config_path=config,
        tool_response=_make_error_response(
            "validation_error",
            "Invalid content_type: application/octet-stream",
        ),
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Validation" in ctx


@pytest.mark.unit
def test_internal_error_injects_guidance(hook_runner, make_config):
    """Internal server errors produce Internal class guidance."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-python-executor__python_execute",
        {"code": "1/0"},
        config_path=config,
        tool_response=_make_error_response(
            "internal_error",
            "Unexpected error during python_execute: ZeroDivisionError",
        ),
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Internal" in ctx


@pytest.mark.unit
def test_ariel_error_detected(hook_runner, make_config):
    """ARIEL MCP tool errors are also detected."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__ariel__ariel_search",
        {"query": "test"},
        config_path=config,
        tool_response=_make_error_response(
            "connection_error",
            "ARIEL service unreachable",
        ),
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Connection" in ctx


# -- Negative detection tests (no error → silent exit) --


@pytest.mark.unit
def test_success_response_no_output(hook_runner, make_config):
    """Successful tool responses produce no output (silent pass-through)."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=json.dumps(
            {"channels": [{"name": "SR:CURRENT:RB", "value": 500.1}]}
        ),
    )

    assert result is None


@pytest.mark.unit
def test_non_osprey_tool_no_output(hook_runner, make_config):
    """Non-OSPREY tools are ignored completely."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "some_other_tool",
        {"param": "value"},
        config_path=config,
        tool_response=_make_error_response("internal_error", "kaboom"),
    )

    assert result is None


@pytest.mark.unit
def test_no_tool_response_no_output(hook_runner, make_config):
    """Missing tool_response field produces no output."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        # tool_response omitted
    )

    assert result is None


@pytest.mark.unit
def test_non_json_success_no_output(hook_runner, make_config):
    """Non-JSON success strings don't trigger false positives."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response="Channel read successful: SR:CURRENT:RB = 500.1",
    )

    assert result is None


# -- Edge cases --


@pytest.mark.unit
def test_non_json_error_string_detected(hook_runner, make_config):
    """Non-JSON strings containing error keywords trigger fallback detection."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response="Error: Failed to connect to IOC at 192.168.1.100:5064",
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Internal" in ctx  # Fallback classification


@pytest.mark.unit
def test_unknown_error_type_defaults_to_internal(hook_runner, make_config):
    """Unknown error_type values default to Internal class."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=_make_error_response(
            "some_new_error_type",
            "Something novel went wrong",
        ),
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Internal" in ctx


@pytest.mark.unit
def test_guidance_includes_anti_pattern_reminders(hook_runner, make_config):
    """Injected guidance includes key anti-pattern reminders."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=_make_error_response(
            "connection_error",
            "Control system unreachable",
        ),
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    # Check that key anti-pattern reminders are present
    assert "mock data" in ctx.lower() or "mock" in ctx.lower()
    assert "retry" in ctx.lower()
    assert "infrastructure" in ctx.lower() or "debug" in ctx.lower()


@pytest.mark.unit
def test_dict_tool_response_detected(hook_runner, make_config):
    """Error detection works when tool_response is already a dict (not JSON string)."""
    config = make_config({})
    # Pass dict directly — the hook should handle both str and dict
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__osprey-control-system__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response={
            "error": True,
            "error_type": "connection_error",
            "error_message": "IOC offline",
            "suggestions": [],
        },
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Connection" in ctx
