"""Integration tests for the hook chain execution order.

Claude Code runs hooks in the order they appear in settings.json.
The expected chain for osprey write operations is:
  1. osprey_writes_check — master kill switch (deny if writes disabled)
  2. osprey_limits — channel limits validation (deny if out of range)
  3. osprey_approval — human approval prompt (ask if required)

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
        "mcp__controls__channel_write",
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
        "mcp__controls__channel_write",
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
        "mcp__controls__channel_write",
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

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "mcp__controls__channel_write",
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

    result, blocked_by = run_hook_chain(
        hook_runner,
        WRITE_HOOK_CHAIN,
        "mcp__controls__channel_read",
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
        "mcp__python__execute",
        {"code": "device.write_attribute('MOTOR:POS', 100)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert blocked_by == "osprey_approval.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


# ============================================================================
# Hook crash resilience tests
# ============================================================================


@pytest.mark.integration
def test_hook_invalid_json_stdin_exits_cleanly(hook_runner_raw):
    """Hook receiving invalid JSON on stdin exits with code 0 (fail-open).

    All hooks wrap stdin parsing in try/except and call sys.exit(0) on failure,
    which allows the tool to proceed. This is intentional — a broken hook
    should not block the entire system.
    """
    returncode, stdout, stderr = hook_runner_raw(
        "osprey_writes_check.py",
        tool_name="unused",
        tool_input={},
        stdin_override="this is not valid json {{{",
    )

    assert returncode == 0
    assert stdout.strip() == ""  # No output = allow


@pytest.mark.integration
def test_hook_empty_stdin_exits_cleanly(hook_runner_raw):
    """Hook receiving empty stdin exits with code 0."""
    returncode, stdout, stderr = hook_runner_raw(
        "osprey_approval.py",
        tool_name="unused",
        tool_input={},
        stdin_override="",
    )

    assert returncode == 0
    assert stdout.strip() == ""


# ============================================================================
# PostToolUse chain integration tests
# ============================================================================


@pytest.mark.integration
def test_error_guidance_fires_on_tool_error(hook_runner, make_config):
    """After a tool returns an error envelope, error_guidance injects additionalContext."""
    config = make_config({})

    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        tool_response=json.dumps(
            {
                "error": True,
                "error_type": "connection_error",
                "error_message": "Control system unreachable",
                "suggestions": [],
            }
        ),
    )

    assert result is not None
    assert "additionalContext" in result["hookSpecificOutput"]
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "Connection" in ctx
    assert "error-handling" in ctx.lower()


@pytest.mark.integration
def test_notebook_update_fires_on_notebook_edit(tmp_path, hook_runner):
    """After NotebookEdit, notebook cache hook deletes stale cached HTML."""
    # Create a fake notebook and its cached HTML
    nb_path = tmp_path / "test_notebook.ipynb"
    nb_path.write_text("{}")  # Minimal file
    cache_dir = tmp_path / "_notebook_cache"
    cache_dir.mkdir()
    cached_html = cache_dir / "test_notebook_rendered.html"
    cached_html.write_text("<html>stale</html>")

    assert cached_html.exists()

    # Run the notebook_update hook
    hook_runner(
        "osprey_notebook_update.py",
        "NotebookEdit",
        {"notebook_path": str(nb_path)},
    )

    # Cached HTML should have been deleted
    assert not cached_html.exists()


@pytest.mark.integration
def test_notebook_update_no_cache_is_noop(tmp_path, hook_runner):
    """Notebook update hook is a no-op when no cached HTML exists."""
    nb_path = tmp_path / "test_notebook.ipynb"
    nb_path.write_text("{}")

    # No cache dir exists — should not raise or produce output
    result = hook_runner(
        "osprey_notebook_update.py",
        "NotebookEdit",
        {"notebook_path": str(nb_path)},
    )

    assert result is None  # Silent no-op


@pytest.mark.integration
def test_post_tool_use_hooks_independent(hook_runner, make_config):
    """PostToolUse hooks can fire independently for different tool matchers.

    error_guidance matches OSPREY MCP tools, notebook_update matches NotebookEdit.
    They don't interfere with each other.
    """
    config = make_config({})

    # error_guidance does not fire for NotebookEdit (not an OSPREY prefix)
    error_result = hook_runner(
        "osprey_error_guidance.py",
        "NotebookEdit",
        {"notebook_path": "/tmp/test.ipynb"},
        config_path=config,
        tool_response=json.dumps(
            {
                "error": True,
                "error_type": "internal_error",
                "error_message": "something broke",
            }
        ),
    )

    assert error_result is None  # NotebookEdit is not an OSPREY MCP tool

    # notebook_update does not fire for channel_read (not NotebookEdit)
    nb_result = hook_runner(
        "osprey_notebook_update.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
    )

    assert nb_result is None  # channel_read has no notebook_path
