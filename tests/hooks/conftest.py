"""Shared fixtures for hook tests.

Provides a hook runner that executes hook scripts as subprocesses (matching
how Claude Code invokes them), plus config file factories.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HOOKS_DIR = (
    Path(__file__).parents[2] / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
)


@pytest.fixture
def hook_runner():
    """Factory to run hook scripts as subprocesses.

    Mirrors the real Claude Code hook execution: stdin receives JSON with
    tool_name and tool_input, stdout receives JSON output (or empty for allow).
    """

    def run(hook_name, tool_name, tool_input, config_path=None, cwd=None, tool_response=None):
        hook_script = HOOKS_DIR / hook_name
        payload = {
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        if tool_response is not None:
            payload["tool_response"] = tool_response
        stdin_data = json.dumps(payload)
        env = os.environ.copy()
        if config_path:
            env["OSPREY_CONFIG"] = str(config_path)
            # Also set CONFIG_FILE so osprey.utils.config picks it up
            # (prevents pollution from other tests that set CONFIG_FILE)
            env["CONFIG_FILE"] = str(config_path)

        result = subprocess.run(
            [sys.executable, str(hook_script)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        assert result.returncode == 0, f"Hook failed (exit {result.returncode}): {result.stderr}"
        stdout = result.stdout.strip()
        if not stdout:
            return None  # Hook allowed (no output = pass through)
        # Find the JSON object in stdout (skip any log/warning lines)
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        # Try the full stdout as JSON
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return None  # Non-JSON output = treat as pass through

    return run


@pytest.fixture
def make_config(tmp_path):
    """Factory for creating test config.yml files from dicts."""

    def _make(config_dict):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config_dict))
        return config_path

    return _make
