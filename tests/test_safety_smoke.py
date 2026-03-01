"""End-to-end safety smoke tests: hooks + MCP tools against mock control system.

Exercises every safety layer in the OSPREY Claude Code integration without
requiring API keys, containers, or real hardware. Tests run the hook chain
(as subprocesses, matching real Claude Code behavior) and then call MCP tools
directly against the MockConnector.

Scenarios mirror what an operator would manually test after setting up
a new control_assistant project:

  1. Read channel        → passes through, no hooks fire
  2. Write within limits → approval hook asks
  3. Write over limits   → limits hook denies
  4. Write to read-only  → limits hook denies
  5. Write to unlisted   → approval hook asks (permissive mode)
  6. Python with caput   → approval hook asks (hook chain only)
  7. Python with Tango   → approval hook asks (hook chain only)
  8. Safe Python code    → passes through (hook chain only)
  9. Writes disabled     → writes_check hook denies

No LLMs, no containers, no EPICS — just the safety infrastructure.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from osprey.mcp_server.artifact_store import reset_artifact_store
from osprey.utils.workspace import reset_config_cache
from osprey.mcp_server.control_system.registry import initialize_mcp_registry, reset_mcp_registry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HOOKS_DIR = (
    Path(__file__).parents[1] / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
)

WRITE_HOOK_CHAIN = [
    "osprey_writes_check.py",
    "osprey_limits.py",
    "osprey_approval.py",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset MCP singletons between tests."""
    yield
    reset_mcp_registry()
    reset_artifact_store()
    reset_config_cache()


@pytest.fixture
def smoke_env(tmp_path):
    """Set up a complete mock control system environment.

    Creates:
      - config.yml with mock connector, limits enabled, selective approval
      - channel_limits.json with test channels
      - osprey-workspace directory structure
      - config variant with writes_enabled: false
    """
    # Channel limits database
    limits_db = {
        "MAG:HCM01:CURRENT:SP": {
            "min_value": -10.0,
            "max_value": 10.0,
            "max_step": 2.0,
            "writable": True,
            "verification": {"level": "readback", "tolerance_percent": 0.1},
        },
        "MAG:QF01:CURRENT:SP": {
            "writable": False,
        },
        "DIAG:TEMP:SP": {
            "min_value": 0.0,
            "max_value": 100.0,
            "writable": True,
        },
    }
    limits_path = tmp_path / "channel_limits.json"
    limits_path.write_text(json.dumps(limits_db))

    # Main config: writes enabled, limits on, selective approval
    config = {
        "control_system": {
            "type": "mock",
            "writes_enabled": True,
            "write_verification": {
                "default_level": "callback",
                "default_tolerance_percent": 0.1,
            },
            "limits_checking": {
                "enabled": True,
                "database_path": str(limits_path),
                "allow_unlisted_channels": True,
                "on_violation": "error",
            },
        },
        "archiver": {"type": "mock_archiver"},
        "approval": {"global_mode": "selective"},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(config))

    # Config variant: writes disabled
    config_writes_off = config.copy()
    config_writes_off["control_system"] = {
        **config["control_system"],
        "writes_enabled": False,
    }
    config_writes_off_path = tmp_path / "config_writes_off.yml"
    config_writes_off_path.write_text(yaml.dump(config_writes_off))

    # Workspace directories
    ws = tmp_path / "osprey-workspace"
    for subdir in ["channel_results", "archiver_data", "python_outputs"]:
        (ws / subdir).mkdir(parents=True)

    return {
        "tmp_path": tmp_path,
        "config_path": config_path,
        "config_writes_off_path": config_writes_off_path,
        "limits_path": limits_path,
    }


# ---------------------------------------------------------------------------
# Hook chain runner (same as tests/hooks/conftest.py)
# ---------------------------------------------------------------------------


def _run_hook(hook_name, tool_name, tool_input, config_path, cwd):
    """Run a single hook script as a subprocess."""
    hook_script = HOOKS_DIR / hook_name
    stdin_data = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    env = os.environ.copy()
    env["OSPREY_CONFIG"] = str(config_path)
    env["CONFIG_FILE"] = str(config_path)

    result = subprocess.run(
        [sys.executable, str(hook_script)],
        input=stdin_data,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
    )
    assert result.returncode == 0, (
        f"Hook {hook_name} failed (exit {result.returncode}): {result.stderr}"
    )
    stdout = result.stdout.strip()
    if not stdout:
        return None
    for line in reversed(stdout.split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def _run_hook_chain(tool_name, tool_input, config_path, cwd):
    """Run the full write hook chain, stopping at first deny/ask."""
    for hook_name in WRITE_HOOK_CHAIN:
        result = _run_hook(hook_name, tool_name, tool_input, config_path, cwd)
        if result is not None:
            decision = result.get("hookSpecificOutput", {}).get("permissionDecision")
            if decision in ("deny", "ask"):
                return result, hook_name
    return None, None


# ---------------------------------------------------------------------------
# MCP tool helpers
# ---------------------------------------------------------------------------


def _get_channel_read():
    from osprey.mcp_server.control_system.tools.channel_read import channel_read

    return channel_read.fn if hasattr(channel_read, "fn") else channel_read


def _get_channel_write():
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    return channel_write.fn if hasattr(channel_write, "fn") else channel_write


# ===========================================================================
# SCENARIO 1: Read channel — passes through, no hooks fire
# ===========================================================================


@pytest.mark.integration
def test_1_read_channel_hooks_pass_through(smoke_env):
    """Reading a channel does not trigger any write hooks."""
    result, blocked_by = _run_hook_chain(
        "mcp__controls__channel_read",
        {"channels": ["SR:BEAM:CURRENT"]},
        smoke_env["config_path"],
        smoke_env["tmp_path"],
    )
    assert result is None, f"Hooks should pass through for reads, but {blocked_by} fired"
    assert blocked_by is None


@pytest.mark.integration
async def test_1_read_channel_tool_returns_data(smoke_env, monkeypatch):
    """channel_read tool returns mock data for any channel name."""
    monkeypatch.chdir(smoke_env["tmp_path"])
    monkeypatch.setenv("OSPREY_CONFIG", str(smoke_env["config_path"]))
    initialize_mcp_registry()

    fn = _get_channel_read()
    result = await fn(channels=["SR:BEAM:CURRENT"])

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["channels_read"] == 1
    assert "SR:BEAM:CURRENT" in data["summary"]["readings"]
    reading = data["summary"]["readings"]["SR:BEAM:CURRENT"]
    assert "value" in reading


# ===========================================================================
# SCENARIO 2: Write within limits — approval hook asks
# ===========================================================================


@pytest.mark.integration
def test_2_valid_write_hooks_ask_approval(smoke_env):
    """A valid write to a channel within limits triggers the approval hook."""
    # Use DIAG:TEMP:SP — has limits (0-100) but no max_step constraint
    result, blocked_by = _run_hook_chain(
        "mcp__controls__channel_write",
        {"operations": [{"channel": "DIAG:TEMP:SP", "value": 50.0}]},
        smoke_env["config_path"],
        smoke_env["tmp_path"],
    )
    assert blocked_by == "osprey_approval.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


@pytest.mark.integration
async def test_2_valid_write_tool_succeeds(smoke_env, monkeypatch):
    """channel_write tool executes a valid write against mock connector."""
    monkeypatch.chdir(smoke_env["tmp_path"])
    monkeypatch.setenv("OSPREY_CONFIG", str(smoke_env["config_path"]))
    initialize_mcp_registry()

    fn = _get_channel_write()
    result = await fn(operations=[{"channel": "DIAG:TEMP:SP", "value": 50.0}])

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["total_writes"] == 1
    assert data["summary"]["failed"] == 0
    assert data["summary"]["results"][0]["success"] is True


# ===========================================================================
# SCENARIO 3: Write over limits — limits hook denies
# ===========================================================================


@pytest.mark.integration
def test_3_over_limit_write_hooks_deny(smoke_env):
    """A write exceeding channel limits is denied by the limits hook."""
    result, blocked_by = _run_hook_chain(
        "mcp__controls__channel_write",
        {"operations": [{"channel": "MAG:HCM01:CURRENT:SP", "value": 999.0}]},
        smoke_env["config_path"],
        smoke_env["tmp_path"],
    )
    assert blocked_by == "osprey_limits.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.integration
async def test_3_over_limit_write_tool_rejects(smoke_env, monkeypatch):
    """channel_write tool's inline validator also rejects over-limit writes."""
    monkeypatch.chdir(smoke_env["tmp_path"])
    monkeypatch.setenv("OSPREY_CONFIG", str(smoke_env["config_path"]))
    initialize_mcp_registry()

    fn = _get_channel_write()
    result = await fn(operations=[{"channel": "MAG:HCM01:CURRENT:SP", "value": 999.0}])

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "limits_violation"


# ===========================================================================
# SCENARIO 4: Write to read-only channel — limits hook denies
# ===========================================================================


@pytest.mark.integration
def test_4_readonly_channel_hooks_deny(smoke_env):
    """Writing to a non-writable channel is denied by the limits hook."""
    result, blocked_by = _run_hook_chain(
        "mcp__controls__channel_write",
        {"operations": [{"channel": "MAG:QF01:CURRENT:SP", "value": 1.0}]},
        smoke_env["config_path"],
        smoke_env["tmp_path"],
    )
    assert blocked_by == "osprey_limits.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


# ===========================================================================
# SCENARIO 5: Write to unlisted channel — approval fires (permissive mode)
# ===========================================================================


@pytest.mark.integration
def test_5_unlisted_channel_hooks_ask_approval(smoke_env):
    """Writing to a channel not in the limits DB passes limits (permissive) and reaches approval."""
    result, blocked_by = _run_hook_chain(
        "mcp__controls__channel_write",
        {"operations": [{"channel": "SR:RANDOM:CHANNEL", "value": 42.0}]},
        smoke_env["config_path"],
        smoke_env["tmp_path"],
    )
    assert blocked_by == "osprey_approval.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


@pytest.mark.integration
async def test_5_unlisted_channel_tool_succeeds(smoke_env, monkeypatch):
    """channel_write tool accepts writes to unlisted channels in permissive mode."""
    monkeypatch.chdir(smoke_env["tmp_path"])
    monkeypatch.setenv("OSPREY_CONFIG", str(smoke_env["config_path"]))
    initialize_mcp_registry()

    fn = _get_channel_write()
    result = await fn(operations=[{"channel": "SR:RANDOM:CHANNEL", "value": 42.0}])

    data = json.loads(result)
    assert data["status"] == "success"
    assert data["summary"]["results"][0]["success"] is True


# ===========================================================================
# SCENARIO 6: Python with caput pattern — approval hook asks
# ===========================================================================


@pytest.mark.integration
def test_6_python_caput_hooks_ask_approval(smoke_env):
    """Python code containing caput() triggers the approval hook."""
    result, blocked_by = _run_hook_chain(
        "mcp__python__execute",
        {"code": "caput('SR:BEAM:CURRENT', 100)", "execution_mode": "readonly"},
        smoke_env["config_path"],
        smoke_env["tmp_path"],
    )
    assert blocked_by == "osprey_approval.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


# ===========================================================================
# SCENARIO 7: Python with Tango pattern — approval hook asks (framework detection)
# ===========================================================================


@pytest.mark.integration
def test_7_python_tango_hooks_ask_approval(smoke_env):
    """Python code with Tango write_attribute() triggers approval via framework detection."""
    result, blocked_by = _run_hook_chain(
        "mcp__python__execute",
        {"code": "device.write_attribute('MOTOR:POS', 100)", "execution_mode": "readonly"},
        smoke_env["config_path"],
        smoke_env["tmp_path"],
    )
    assert blocked_by == "osprey_approval.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"


# ===========================================================================
# SCENARIO 8: Safe Python code — passes through
# ===========================================================================


@pytest.mark.integration
def test_8_safe_python_hooks_pass_through(smoke_env):
    """Python code with no write patterns does not trigger any hooks."""
    result, blocked_by = _run_hook_chain(
        "mcp__python__execute",
        {"code": "import math; print(math.pi * 2)", "execution_mode": "readonly"},
        smoke_env["config_path"],
        smoke_env["tmp_path"],
    )
    assert result is None, f"Hooks should pass through for safe code, but {blocked_by} fired"


# ===========================================================================
# SCENARIO 9: Writes disabled — writes_check hook denies
# ===========================================================================


@pytest.mark.integration
def test_9_writes_disabled_hooks_deny(smoke_env):
    """With writes_enabled: false, the writes_check hook denies all writes."""
    result, blocked_by = _run_hook_chain(
        "mcp__controls__channel_write",
        {"operations": [{"channel": "MAG:HCM01:CURRENT:SP", "value": 5.0}]},
        smoke_env["config_writes_off_path"],
        smoke_env["tmp_path"],
    )
    assert blocked_by == "osprey_writes_check.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.integration
def test_9_writes_disabled_python_readwrite_hooks_deny(smoke_env):
    """With writes_enabled: false, execute tool in readwrite mode is also denied."""
    result, blocked_by = _run_hook_chain(
        "mcp__python__execute",
        {"code": "caput('TEST:PV', 1.0)", "execution_mode": "readwrite"},
        smoke_env["config_writes_off_path"],
        smoke_env["tmp_path"],
    )
    assert blocked_by == "osprey_writes_check.py"
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
