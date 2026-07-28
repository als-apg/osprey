"""Shared fixtures for hook tests.

Provides a hook runner that executes hook scripts as subprocesses (matching
how Claude Code invokes them), plus config file factories.

Hook subprocesses run with a *curated* environment rather than a copy of
``os.environ``: only the interpreter/OS essentials and the switches the hooks
themselves read are forwarded, so a hook run cannot pick up ``OSPREY_CONFIG``,
``CONFIG_FILE`` or similar state leaked into the process environment by a test
that ran earlier in the same worker.
"""

import json
import os
import subprocess
import sys
from itertools import count
from pathlib import Path

import pytest
import yaml

HOOKS_DIR = (
    Path(__file__).parents[2] / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
)

# Interpreter and OS essentials the subprocess needs to start at all.
_SYSTEM_VARS = (
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONHASHSEED",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    # pytest-cov hands subprocess coverage down through these.
    "COV_CORE_SOURCE",
    "COV_CORE_CONFIG",
    "COV_CORE_DATAFILE",
    "COV_CORE_CONTEXT",
)

# Hook behaviour switches. Tests that need one set it with ``monkeypatch.setenv``
# (reverted at teardown), so forwarding them cannot leak across tests. Config
# location vars are deliberately absent: they are set only from an explicit
# ``config_path`` argument, never inherited.
_HOOK_VARS = (
    "CLAUDE_PROJECT_DIR",
    "OSPREY_HOOK_CONFIG",
    "OSPREY_HOOK_DEBUG",
    "OSPREY_DISPATCH_RUN",
    "OSPREY_WEB_PORT",
    "BLUESKY_BRIDGE_URL",
)

_hook_config_seq = count()


@pytest.fixture
def hook_home(tmp_path_factory):
    """Throwaway ``$HOME`` for hook tests.

    Hooks resolve ``~/.claude/...`` from ``$HOME``; tests build their expected
    paths from this fixture rather than from the developer's real home.
    """
    return tmp_path_factory.mktemp("hook-home")


@pytest.fixture(autouse=True)
def _isolated_home(hook_home, monkeypatch):
    """Point ``$HOME`` at :func:`hook_home` for this test and its subprocesses.

    Leak guarded: without this, ``Path.home()`` in the test process and in the
    hook subprocess both resolve to the real home directory, which the memory
    guard hook is asked to write into. ``monkeypatch`` restores the previous
    values after the test.
    """
    monkeypatch.setenv("HOME", str(hook_home))
    monkeypatch.setenv("USERPROFILE", str(hook_home))
    yield


def _curated_env(config_path=None, hook_config_path=None):
    """Build the subprocess environment from the allowlists above."""
    env = {name: os.environ[name] for name in _SYSTEM_VARS + _HOOK_VARS if name in os.environ}
    if config_path:
        env["OSPREY_CONFIG"] = str(config_path)
        # Also set CONFIG_FILE so osprey.utils.config picks it up.
        env["CONFIG_FILE"] = str(config_path)
    if hook_config_path is not None:
        env["OSPREY_HOOK_CONFIG"] = str(hook_config_path)
    return env


def _write_hook_config(tmp_path, hook_config):
    """Serialise a hook_config dict into a per-test temp file."""
    path = tmp_path / f"hook_config_{next(_hook_config_seq)}.json"
    path.write_text(json.dumps(hook_config))
    return path


@pytest.fixture
def hook_runner(tmp_path):
    """Factory to run hook scripts as subprocesses.

    Mirrors the real Claude Code hook execution: stdin receives JSON with
    tool_name and tool_input, stdout receives JSON output (or empty for allow).
    """

    def run(
        hook_name,
        tool_name,
        tool_input,
        config_path=None,
        cwd=None,
        tool_response=None,
        hook_input_extra=None,
        hook_config=None,
    ):
        hook_script = HOOKS_DIR / hook_name
        payload = {
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        if tool_response is not None:
            payload["tool_response"] = tool_response
        if hook_input_extra:
            payload.update(hook_input_extra)
        stdin_data = json.dumps(payload)

        hook_config_path = None
        if hook_config is not None:
            hook_config_path = _write_hook_config(tmp_path, hook_config)
        env = _curated_env(config_path, hook_config_path)

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
def hook_runner_raw(tmp_path):
    """Factory to run hook scripts without asserting returncode.

    Same as hook_runner but returns a (returncode, stdout, stderr) tuple
    instead of parsing the JSON output. Used for crash resilience tests.
    """

    def run(
        hook_name,
        tool_name,
        tool_input,
        config_path=None,
        cwd=None,
        tool_response=None,
        hook_input_extra=None,
        stdin_override=None,
        hook_config=None,
    ):
        hook_script = HOOKS_DIR / hook_name
        if stdin_override is not None:
            stdin_data = stdin_override
        else:
            payload = {
                "tool_name": tool_name,
                "tool_input": tool_input,
            }
            if tool_response is not None:
                payload["tool_response"] = tool_response
            if hook_input_extra:
                payload.update(hook_input_extra)
            stdin_data = json.dumps(payload)

        hook_config_path = None
        if hook_config is not None:
            hook_config_path = _write_hook_config(tmp_path, hook_config)
        env = _curated_env(config_path, hook_config_path)

        result = subprocess.run(
            [sys.executable, str(hook_script)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        return result.returncode, result.stdout, result.stderr

    return run


@pytest.fixture
def make_config(tmp_path):
    """Factory for creating test config.yml files from dicts."""

    def _make(config_dict):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config_dict))
        return config_path

    return _make
