"""A facility server's declared write is audited as a write.

A facility (custom) MCP server never runs the framework's audit middleware, so
the hooks are its whole audit trail: the writes-check hook files the refusal on
``hook_writes_check`` and the approval hook files the ask on ``hook_approval``.
Both classify a tool by the rendered ``hook_config.json`` alone, so these tests
render a real project with a server that declares ``writes:`` and run the hooks
against that file, the way Claude Code does — as subprocesses over stdin.

Pinned both ways: each declared tool leaves a write's record, and the declaring
server's other tools leave none.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from osprey_connectors_tests._harness import pin_identity, write_control_context

from .conftest import HOOKS_DIR, hook_audit_records

#: Two declared writes and two plain reads on one facility server. The reads
#: are allowed, absent from ``permissions.ask`` and carry no destructive
#: marker, so only the declaration decides their class.
_DECLARED_WRITES = ("set_value", "ramp_value")
_DECLARING_READS = ("get_value", "list_values")
_DECLARING_SERVER = {
    "plc": {
        "command": "node",
        "args": ["plc.js"],
        "writes": list(_DECLARED_WRITES),
        "permissions": {"allow": [*_DECLARING_READS, *_DECLARED_WRITES]},
    },
}

#: Built from nothing, so a marker leaked from the developer's own session
#: cannot make a record read correct for the wrong reason.
_BASE_VARS = ("PATH", "PYTHONPATH", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "LANG")


@pytest.fixture(scope="module")
def rendered_hook_config(tmp_path_factory) -> Path:
    """The ``hook_config.json`` a control-assistant regeneration ships with the server."""
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.cli.templates.manager import TemplateManager
    from osprey.utils.config_writer import config_update_fields
    from tests._preset_data import bundle_data_root

    tmp_path = tmp_path_factory.mktemp("facility-write-audit")
    manager = TemplateManager()
    profile, _ = resolve_build_profile(None, preset="control-assistant")
    project = manager.create_project(
        project_name="facility-write-audit",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        data_root=bundle_data_root(),
        context={"channel_finder_mode": "hierarchical"},
        force=True,
    )
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project,
        "facility-write-audit",
        "control-assistant",
        {},
        artifacts=manager._effective_artifacts("control_assistant", None),
    )
    config_file = project / "config.yml"
    config = yaml.safe_load(config_file.read_text())
    config.setdefault("claude_code", {})["servers"] = _DECLARING_SERVER
    config_file.write_text(yaml.dump(config))
    manager.regenerate_claude_code(project)
    return project / ".claude" / "hooks" / "hook_config.json"


@pytest.fixture
def repo(tmp_path):
    """A deployment repo root a hook resolves through ``get_repo_root``."""
    (tmp_path / "profile.yml").write_text("name: facility-write-audit\n")
    return tmp_path


def _env(repo, hook_config, *, writes):
    """A hook child inside *repo* as ``alice``, on a tier of *writes*."""
    config = repo / "config.yml"
    config.write_text(
        yaml.dump(
            {
                "approval": {
                    "enabled": True,
                    "default_policy": "selective",
                    "requires_approval": list(_DECLARED_WRITES),
                },
                "control_system": {"type": "epics", "writes": writes},
            }
        )
    )
    env = {name: os.environ[name] for name in _BASE_VARS if name in os.environ}
    env.update(
        {
            "CLAUDE_PROJECT_DIR": str(repo),
            "OSPREY_TERMINAL_USER": "alice",
            "OSPREY_AGENT_DATA_ROOT": str(repo / "var" / "agent_data"),
            "OSPREY_CONFIG": str(config),
            "CONFIG_FILE": str(config),
            "OSPREY_HOOK_CONFIG": str(hook_config),
        }
    )
    return env


def _run(hook_name, tool, env):
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / hook_name)],
        input=json.dumps({"tool_name": tool, "tool_input": {"value": 1.0}}),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _decision(result):
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)["hookSpecificOutput"]["permissionDecision"]
            except Exception:
                continue
    return None


def _records(repo, surface):
    """Alice's hook audit entries on *surface*, from the repo's own spool.

    The child is given no ``OSPREY_SPOOL_DIR``, so a hook files into
    ``<repo>/var/spool/<identity>``, the directory a deployment's hooks use.
    """
    return hook_audit_records(repo / "var" / "spool" / "alice", surface)


def _switch_on(repo, monkeypatch):
    """Alice's switch on for the live machine, where the hook child reads it.

    A facility server's declared write concerns the real machine, so that is
    the switch the write verdict asks about.
    """
    pin_identity(monkeypatch, "alice")
    write_control_context(repo / "var" / "agent_data", target="live", switch={"live": "on"})


def test_the_render_names_the_declared_tools_and_the_approval_prefix(rendered_hook_config):
    """The hooks below classify by this file; it must say what they rely on."""
    rendered = json.loads(rendered_hook_config.read_text())

    assert "mcp__plc__" in rendered["server_prefixes"]
    assert "mcp__plc__" in rendered["approval_prefixes"]
    for tool in _DECLARED_WRITES:
        assert f"mcp__plc__{tool}" in rendered["write_tools"]


@pytest.mark.parametrize("tool", _DECLARED_WRITES)
def test_a_declared_write_refused_by_the_tier_is_recorded_as_a_write(
    repo, rendered_hook_config, tool
):
    result = _run(
        "osprey_writes_check.py",
        f"mcp__plc__{tool}",
        _env(repo, rendered_hook_config, writes="never"),
    )

    assert _decision(result) == "deny", result.stderr
    (record,) = _records(repo, "hook_writes_check")
    assert record["subject"] == f"mcp__plc__{tool}"
    assert record["decision"] == "refused"
    assert record["reason"] == "posture"
    assert "reason=tier_never" in record["detail"].split()


@pytest.mark.parametrize("tool", _DECLARED_WRITES)
def test_a_declared_write_in_a_readonly_run_is_recorded_as_a_write(
    repo, rendered_hook_config, tool
):
    env = {
        **_env(repo, rendered_hook_config, writes="allowed"),
        "OSPREY_EXECUTION_MODE": "readonly",
    }

    result = _run("osprey_writes_check.py", f"mcp__plc__{tool}", env)

    assert _decision(result) == "deny", result.stderr
    (record,) = _records(repo, "hook_writes_check")
    assert record["subject"] == f"mcp__plc__{tool}"
    assert record["reason"] == "posture"
    assert record["posture"] == "sandbox"


@pytest.mark.parametrize("tool", _DECLARING_READS)
def test_the_declaring_servers_other_tools_are_not_audited_as_writes(
    repo, rendered_hook_config, tool
):
    """Under the very tier that refuses the declared tools, a read passes silently."""
    result = _run(
        "osprey_writes_check.py",
        f"mcp__plc__{tool}",
        _env(repo, rendered_hook_config, writes="never"),
    )

    assert result.returncode == 0, result.stderr
    assert _decision(result) is None
    assert _records(repo, "hook_writes_check") == []


@pytest.mark.parametrize("tool", _DECLARED_WRITES)
def test_an_allowed_declared_write_is_asked_and_the_ask_recorded(
    repo, rendered_hook_config, monkeypatch, tool
):
    """Past the write verdict, the declared tool reaches the approval prompt."""
    _switch_on(repo, monkeypatch)
    env = _env(repo, rendered_hook_config, writes="allowed")

    checked = _run("osprey_writes_check.py", f"mcp__plc__{tool}", env)
    assert _decision(checked) is None, checked.stderr
    assert _records(repo, "hook_writes_check") == []

    asked = _run("osprey_approval.py", f"mcp__plc__{tool}", env)
    assert _decision(asked) == "ask", asked.stderr
    (record,) = _records(repo, "hook_approval")
    assert record["subject"] == f"mcp__plc__{tool}"
    assert record["decision"] == "ask"
    assert record["reason"] == "approval_required"
