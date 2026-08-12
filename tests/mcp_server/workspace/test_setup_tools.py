"""Tests for the setup_inspect and setup_patch MCP tools.

Validates config inspection, env var masking, file patching with YAML/JSON,
whitelist enforcement, key_path validation, and hot/cold classification.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn


def _get_setup_inspect():
    from osprey.mcp_server.workspace.tools.setup import setup_inspect

    return get_tool_fn(setup_inspect)


def _get_setup_patch():
    from osprey.mcp_server.workspace.tools.setup import setup_patch

    return get_tool_fn(setup_patch)


@pytest.fixture
def project_dir(tmp_path):
    """A minimal deployment repo, and the *render* the setup tools act on.

    Three zones, because the setup tools straddle two of them: everything they
    read or patch (``config.yml``, ``.mcp.json``, ``.claude/``) is build output
    under ``build/``, while the agent-data root they report is durable state
    under ``var/``. A flat project made those one directory, so a tool reading
    the wrong zone still passed.

    Returns the RENDER directory — which is what ``setup_inspect`` computes as
    its ``project_root`` (``resolve_config_path().parent``), so the fixture's
    return value and the tool's own notion of the project agree. The repo root
    is its parent.

    The ``agent_data`` key is deliberately absent: this exercises the DEFAULT
    derivation of the agent-data root. ``test_inspect_workspace_is_the_repos_
    agent_data_root`` covers the explicitly-configured ``base_dir``, so the two
    approach the same anchoring rule from opposite ends.
    """
    repo = tmp_path / "facility-repo"
    render = repo / "build"

    # SOURCE zone — what makes `repo` a deployment repo at all.
    repo.mkdir()
    (repo / "profile.yml").write_text("name: Setup Tools Fixture\n")

    # BUILD zone — config.yml, .mcp.json and .claude/ are all build output.
    render.mkdir()
    config = {
        "control_system": {
            "type": "mock",
            "writes_enabled": True,
            "limits_checking": {"enabled": False},
        },
    }
    (render / "config.yml").write_text(yaml.dump(config))

    mcp_json = {
        "mcpServers": {
            "workspace": {
                "command": "python",
                "args": ["-m", "osprey.mcp_server.workspace"],
            }
        }
    }
    (render / ".mcp.json").write_text(json.dumps(mcp_json, indent=2))

    claude_dir = render / ".claude"
    (claude_dir / "rules").mkdir(parents=True)
    (claude_dir / "agents").mkdir(parents=True)
    (claude_dir / "skills" / "session-report").mkdir(parents=True)

    (claude_dir / "rules" / "safety.md").write_text("# Safety Rules")
    (claude_dir / "rules" / "error-handling.md").write_text("# Error Handling")
    (claude_dir / "agents" / "channel-finder.md").write_text("# Channel Finder")
    (claude_dir / "skills" / "session-report" / "SKILL.md").write_text("# Session Report")

    settings = {
        "permissions": {"allow": [], "deny": [], "ask": []},
        "hooks": {"PreToolUse": []},
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    # STATE zone — durable, one zone over from the render, wiped by neither.
    for subdir in ("data", "plots", "memory"):
        (repo / "var" / "agent_data" / subdir).mkdir(parents=True)

    return render


def _patch_config_path(project_dir: Path):
    """Return a context manager that patches resolve_config_path to return project_dir."""
    return patch(
        "osprey.mcp_server.workspace.tools.setup.resolve_config_path",
        return_value=project_dir / "config.yml",
    )


def _patch_load_config(config: dict):
    """Return a context manager that patches load_osprey_config."""
    return patch(
        "osprey.mcp_server.workspace.tools.setup.load_osprey_config",
        return_value=config,
    )


# ---------------------------------------------------------------------------
# setup_inspect tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_returns_all_sections(project_dir):
    """setup_inspect returns all expected top-level keys."""
    fn = _get_setup_inspect()
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    with _patch_config_path(project_dir), _patch_load_config(config):
        result = json.loads(await fn())

    expected_keys = {
        "config",
        "mcp_servers",
        "settings",
        "hooks",
        "rules",
        "agents",
        "skills",
        "environment",
        "workspace",
        "project_root",
    }
    assert expected_keys == set(result.keys())


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_config_content(project_dir):
    """setup_inspect returns the parsed config."""
    fn = _get_setup_inspect()
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    with _patch_config_path(project_dir), _patch_load_config(config):
        result = json.loads(await fn())

    assert result["config"]["control_system"]["type"] == "mock"
    assert result["config"]["control_system"]["writes_enabled"] is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_mcp_servers(project_dir):
    """setup_inspect returns .mcp.json content."""
    fn = _get_setup_inspect()
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    with _patch_config_path(project_dir), _patch_load_config(config):
        result = json.loads(await fn())

    assert "workspace" in result["mcp_servers"]["mcpServers"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_rules_and_agents(project_dir):
    """setup_inspect lists rule files and agent files."""
    fn = _get_setup_inspect()
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    with _patch_config_path(project_dir), _patch_load_config(config):
        result = json.loads(await fn())

    assert "safety.md" in result["rules"]
    assert "error-handling.md" in result["rules"]
    assert "channel-finder.md" in result["agents"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_skills(project_dir):
    """setup_inspect lists skill directories."""
    fn = _get_setup_inspect()
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    with _patch_config_path(project_dir), _patch_load_config(config):
        result = json.loads(await fn())

    assert "session-report" in result["skills"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_workspace(project_dir):
    """setup_inspect reports workspace existence and subdirs."""
    fn = _get_setup_inspect()
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    with _patch_config_path(project_dir), _patch_load_config(config):
        result = json.loads(await fn())

    assert result["workspace"]["exists"] is True
    assert "data" in result["workspace"]["subdirs"]
    assert "plots" in result["workspace"]["subdirs"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_workspace_is_the_repos_agent_data_root(tmp_path):
    """The workspace section reports durable state, not a sibling of the render.

    In a deployment repo the config is build output while the agent-data root is
    durable state one zone over, so it is resolved from ``agent_data.base_dir``
    anchored on the REPO root — reading it beside the config would report a
    directory nothing writes to.
    """
    (tmp_path / "build").mkdir()
    config = {"agent_data": {"base_dir": "var/agent_data"}}
    (tmp_path / "build" / "config.yml").write_text(yaml.dump(config))
    for subdir in ("artifacts", "memory"):
        (tmp_path / "var" / "agent_data" / subdir).mkdir(parents=True)

    fn = _get_setup_inspect()
    with (
        patch(
            "osprey.mcp_server.workspace.tools.setup.resolve_config_path",
            return_value=tmp_path / "build" / "config.yml",
        ),
        _patch_load_config(config),
    ):
        result = json.loads(await fn())

    assert result["workspace"]["exists"] is True
    assert sorted(result["workspace"]["subdirs"]) == ["artifacts", "memory"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_env_masking(project_dir):
    """setup_inspect masks sensitive environment variables."""
    fn = _get_setup_inspect()
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    env_patch = {
        "OSPREY_API_KEY": "sk-secret-123",
        "OSPREY_SESSION_ID": "session-abc",
        "CLAUDE_SECRET_TOKEN": "tok-456",
        "CLAUDE_PROJECT_DIR": "/some/path",
    }
    with (
        _patch_config_path(project_dir),
        _patch_load_config(config),
        patch.dict(os.environ, env_patch, clear=False),
    ):
        result = json.loads(await fn())

    env = result["environment"]
    # Sensitive values masked
    assert env["OSPREY_API_KEY"] == "***"
    assert env["CLAUDE_SECRET_TOKEN"] == "***"
    # Non-sensitive values preserved
    assert env["OSPREY_SESSION_ID"] == "session-abc"
    assert env["CLAUDE_PROJECT_DIR"] == "/some/path"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_inspect_hooks_from_settings(project_dir):
    """setup_inspect extracts hooks from settings.json."""
    fn = _get_setup_inspect()
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    with _patch_config_path(project_dir), _patch_load_config(config):
        result = json.loads(await fn())

    assert "PreToolUse" in result["hooks"]


# ---------------------------------------------------------------------------
# setup_patch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_yaml_file(project_dir):
    """setup_patch modifies a YAML config value."""
    fn = _get_setup_patch()
    with _patch_config_path(project_dir):
        result = extract_response_dict(
            await fn(file="config.yml", key_path="control_system.writes_enabled", value="false")
        )

    assert result["before"] is True
    assert result["after"] is False
    assert result["file"] == "config.yml"
    assert result["key_path"] == "control_system.writes_enabled"

    # Verify file was actually modified
    updated = yaml.safe_load((project_dir / "config.yml").read_text())
    assert updated["control_system"]["writes_enabled"] is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_json_file(project_dir):
    """setup_patch modifies a JSON config value."""
    fn = _get_setup_patch()
    with _patch_config_path(project_dir):
        result = extract_response_dict(
            await fn(
                file=".mcp.json",
                key_path="mcpServers.workspace.command",
                value="python3",
            )
        )

    assert result["before"] == "python"
    assert result["after"] == "python3"

    # Verify file was actually modified
    updated = json.loads((project_dir / ".mcp.json").read_text())
    assert updated["mcpServers"]["workspace"]["command"] == "python3"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_yaml_type_conversion(project_dir):
    """setup_patch YAML-parses values for type conversion."""
    fn = _get_setup_patch()
    with _patch_config_path(project_dir):
        # String "42" should become integer 42
        result = extract_response_dict(
            await fn(
                file="config.yml",
                key_path="control_system.timeout",
                value="42",
            )
        )

    assert result["after"] == 42
    assert isinstance(result["after"], int)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_creates_nested_keys(project_dir):
    """setup_patch creates intermediate keys that don't exist."""
    fn = _get_setup_patch()
    with _patch_config_path(project_dir):
        result = extract_response_dict(
            await fn(
                file="config.yml",
                key_path="new_section.sub.key",
                value="hello",
            )
        )

    assert result["before"] is None
    assert result["after"] == "hello"

    updated = yaml.safe_load((project_dir / "config.yml").read_text())
    assert updated["new_section"]["sub"]["key"] == "hello"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_whitelist_enforcement(project_dir):
    """setup_patch rejects files not in the whitelist."""
    fn = _get_setup_patch()
    with (
        _patch_config_path(project_dir),
        assert_raises_error(error_type="validation_error") as _exc_ctx,
    ):
        await fn(file="settings.json", key_path="foo", value="bar")

    result = _exc_ctx["envelope"]
    assert "not patchable" in result["error_message"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_rejects_traversal(project_dir):
    """setup_patch rejects key_path with '..' traversal."""
    fn = _get_setup_patch()
    with (
        _patch_config_path(project_dir),
        assert_raises_error(error_type="validation_error") as _exc_ctx,
    ):
        await fn(file="config.yml", key_path="foo..bar", value="baz")

    result = _exc_ctx["envelope"]
    assert ".." in result["error_message"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_rejects_absolute_path(project_dir):
    """setup_patch rejects key_path starting with / or ~."""
    fn = _get_setup_patch()
    with _patch_config_path(project_dir), assert_raises_error(error_type="validation_error"):
        await fn(file="config.yml", key_path="/etc/passwd", value="x")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_rejects_empty_key_path(project_dir):
    """setup_patch rejects empty key_path."""
    fn = _get_setup_patch()
    with _patch_config_path(project_dir), assert_raises_error(error_type="validation_error"):
        await fn(file="config.yml", key_path="", value="x")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_hot_change_classification(project_dir):
    """setup_patch classifies known hot paths correctly.

    The limits hook runs as a fresh subprocess per call, so it really does
    re-read config. `writes_enabled` does not qualify — see
    tests/mcp_server/test_setup_patch_classification.py.
    """
    fn = _get_setup_patch()
    with _patch_config_path(project_dir):
        result = extract_response_dict(
            await fn(
                file="config.yml",
                key_path="control_system.limits_checking.enabled",
                value="true",
            )
        )

    assert "hot" in result["note"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_cold_change_classification(project_dir):
    """setup_patch classifies unknown paths as cold changes."""
    fn = _get_setup_patch()
    with _patch_config_path(project_dir):
        result = extract_response_dict(
            await fn(
                file="config.yml",
                key_path="control_system.type",
                value="epics",
            )
        )

    assert "cold" in result["note"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_patch_file_not_found(project_dir):
    """setup_patch returns error when target file doesn't exist."""
    fn = _get_setup_patch()
    # Remove .mcp.json
    (project_dir / ".mcp.json").unlink()
    with _patch_config_path(project_dir), assert_raises_error(error_type="not_found"):
        await fn(file=".mcp.json", key_path="foo", value="bar")
