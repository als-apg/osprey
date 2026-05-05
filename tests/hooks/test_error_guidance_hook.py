"""Tests for the osprey_error_guidance PostToolUse hook.

This hook detects structured errors in MCP tool responses and injects
error-handling guidance into Claude's context via additionalContext.
It should never block execution.
"""

import json

import pytest

# Default hook_config matching the original hard-coded OSPREY_PREFIXES
DEFAULT_ERROR_CONFIG = {
    "server_prefixes": [
        "mcp__controls__",
        "mcp__python__",
        "mcp__workspace__",
        "mcp__ariel__",
        "mcp__channel-finder__",
    ],
    "approval_prefixes": [],
}

# -- Structured error envelope (matches common.make_error) --


def _make_error_response(error_type, message, suggestions=None):
    """Build the post-migration OSPREY error response: a CallToolResult-shaped
    dict (``isError=True`` with the structured envelope inside the first
    text content block) — mirrors what the SDK actually delivers to PostToolUse.
    """
    return {
        "isError": True,
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "error": True,
                        "error_type": error_type,
                        "error_message": message,
                        "suggestions": suggestions or [],
                    }
                ),
            }
        ],
    }


# -- Positive detection tests --


@pytest.mark.unit
def test_connection_error_injects_guidance(hook_runner, make_config):
    """Connection error triggers additionalContext with guidance."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=_make_error_response(
            "connection_error",
            "Failed to connect to the control system: Connection refused",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
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
        "mcp__controls__archiver_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=_make_error_response(
            "timeout_error",
            "archiver_read timed out after 30s",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
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
        "mcp__workspace__artifact_save",
        {"content": "test"},
        config_path=config,
        tool_response=_make_error_response(
            "validation_error",
            "Invalid content_type: application/octet-stream",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
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
        "mcp__python__execute",
        {"code": "1/0"},
        config_path=config,
        tool_response=_make_error_response(
            "internal_error",
            "Unexpected error during execute: ZeroDivisionError",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
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
        "mcp__ariel__search",
        {"query": "test"},
        config_path=config,
        tool_response=_make_error_response(
            "connection_error",
            "ARIEL service unreachable",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Connection" in ctx


# -- Negative detection tests (no error -> silent exit) --


@pytest.mark.unit
def test_success_response_no_output(hook_runner, make_config):
    """Successful tool responses produce no output (silent pass-through)."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=json.dumps({"channels": [{"name": "SR:CURRENT:RB", "value": 500.1}]}),
        hook_config=DEFAULT_ERROR_CONFIG,
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
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is None


@pytest.mark.unit
def test_no_tool_response_no_output(hook_runner, make_config):
    """Missing tool_response field produces no output."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        # tool_response omitted
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is None


@pytest.mark.unit
def test_non_json_success_no_output(hook_runner, make_config):
    """Non-JSON success strings don't trigger false positives."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response="Channel read successful: SR:CURRENT:RB = 500.1",
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is None


# -- Edge cases --


@pytest.mark.unit
def test_isError_without_envelope_falls_back_to_internal(hook_runner, make_config):
    """A CallToolResult with isError=True but no parseable envelope still
    triggers Internal-class guidance (defensive fallback in _detect_error)."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response={
            "isError": True,
            "content": [{"type": "text", "text": "raw connection failure: 192.168.1.100:5064"}],
        },
        hook_config=DEFAULT_ERROR_CONFIG,
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
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=_make_error_response(
            "some_new_error_type",
            "Something novel went wrong",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
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
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=_make_error_response(
            "connection_error",
            "Control system unreachable",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    # Check that key anti-pattern reminders are present
    assert "mock data" in ctx.lower() or "mock" in ctx.lower()
    assert "retry" in ctx.lower()
    assert "infrastructure" in ctx.lower() or "debug" in ctx.lower()


@pytest.mark.unit
def test_dict_tool_response_detected(hook_runner, make_config):
    """Error detection on a CallToolResult-shaped dict with isError=True."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response={
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "error": True,
                            "error_type": "connection_error",
                            "error_message": "IOC offline",
                            "suggestions": [],
                        }
                    ),
                }
            ],
        },
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Connection" in ctx


# -- Missing error classes (gap fill) --


@pytest.mark.unit
def test_permission_error_injects_guidance(hook_runner, make_config):
    """Permission-denied errors from hooks are classified as Execution class.

    The hook itself doesn't have a 'permission_error' type in ERROR_CLASS_MAP,
    so unrecognized types default to Internal. This test documents the behavior.
    """
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "PROTECTED:PV", "value": 1.0}]},
        config_path=config,
        tool_response=_make_error_response(
            "permission_error",
            "Insufficient permissions to write to PROTECTED:PV",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    # permission_error is not in ERROR_CLASS_MAP -> defaults to Internal
    assert "Internal" in ctx
    assert "error-handling" in ctx.lower()


@pytest.mark.unit
def test_execution_error_injects_guidance(hook_runner, make_config):
    """Execution errors (python code failures) produce Execution class guidance."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__python__execute",
        {"code": "import nonexistent_module"},
        config_path=config,
        tool_response=_make_error_response(
            "execution_error",
            "ModuleNotFoundError: No module named 'nonexistent_module'",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Execution" in ctx
    assert "error-handling" in ctx.lower()


@pytest.mark.unit
def test_data_not_found_injects_guidance(hook_runner, make_config):
    """Data not-found errors produce Data class guidance."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_read",
        {"channels": ["NONEXISTENT:PV"]},
        config_path=config,
        tool_response=_make_error_response(
            "not_found",
            "Channel NONEXISTENT:PV not found in the control system",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Data" in ctx
    assert "error-handling" in ctx.lower()


@pytest.mark.unit
def test_data_no_results_injects_guidance(hook_runner, make_config):
    """No-results data errors produce Data class guidance."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__workspace__artifact_save",
        {"query": "nonexistent artifact"},
        config_path=config,
        tool_response=_make_error_response(
            "no_results",
            "No artifacts matched the query 'nonexistent artifact'",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Data" in ctx
    assert "error-handling" in ctx.lower()


@pytest.mark.unit
def test_limits_violation_error_injects_guidance(hook_runner, make_config):
    """Limits violation errors produce Validation class guidance."""
    config = make_config({})
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 999.0}]},
        config_path=config,
        tool_response=_make_error_response(
            "limits_violation",
            "Value 999.0 exceeds max_value=100.0 for TEST:PV",
        ),
        hook_config=DEFAULT_ERROR_CONFIG,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Validation" in ctx
    assert "error-handling" in ctx.lower()


# ============================================================================
# Dynamic prefix tests — custom server hooks
# ============================================================================


@pytest.mark.unit
def test_custom_server_prefix_triggers_guidance(hook_runner, make_config):
    """Custom server prefix in hook_config triggers error guidance."""
    config = make_config({})

    custom_config = {
        "server_prefixes": ["mcp__controls__", "mcp__my_plc__"],
        "approval_prefixes": [],
    }

    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__my_plc__read_sensor",
        {"sensor": "temp_1"},
        config_path=config,
        tool_response=_make_error_response(
            "connection_error",
            "PLC at 10.0.1.50 unreachable",
        ),
        hook_config=custom_config,
    )

    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Connection" in ctx
    assert "error-handling" in ctx.lower()
