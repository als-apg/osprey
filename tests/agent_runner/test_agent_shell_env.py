"""The panel token reaches an agent's MCP client and hooks, never its shell commands.

Claude Code sources the script named by ``CLAUDE_ENV_FILE`` before every Bash
and Monitor command and starts its MCP servers and hooks without it, so a token
that script unsets is gone from every shell command the agent runs and still
present where the panel tools and hooks read it.
"""

from __future__ import annotations

import os
import subprocess

from osprey.agent_runner.clean_env import SHELL_ENV_FILE, SHELL_ENV_FILE_VAR, add_panel_token
from osprey.interfaces.web_auth import PANEL_TOKEN_ENV


def _shell_command_env(child_env: dict[str, str]) -> dict[str, str]:
    """What a Bash tool command sees: Claude Code sources the env file, then runs it."""
    script = f'. "${{{SHELL_ENV_FILE_VAR}}}" && env -0'
    out = subprocess.run(
        ["bash", "-c", script], env=child_env, check=True, capture_output=True
    ).stdout
    return dict(item.split("=", 1) for item in out.decode().split("\0") if "=" in item)


def test_the_child_holds_the_token_and_names_the_shell_script():
    env: dict[str, str] = {}
    add_panel_token(env, "panel-secret")
    assert env == {PANEL_TOKEN_ENV: "panel-secret", SHELL_ENV_FILE_VAR: str(SHELL_ENV_FILE)}
    assert SHELL_ENV_FILE_VAR == "CLAUDE_ENV_FILE"
    assert SHELL_ENV_FILE.is_file()


def test_a_shell_command_starts_without_the_token():
    env = {"PATH": os.environ["PATH"], "NEIGHBOUR": "kept"}
    add_panel_token(env, "panel-secret")
    seen = _shell_command_env(env)
    assert PANEL_TOKEN_ENV not in seen
    assert "panel-secret" not in seen.values()
    assert seen["NEIGHBOUR"] == "kept"


def test_the_script_unsets_nothing_but_the_token():
    names = [
        line.split()[1]
        for line in SHELL_ENV_FILE.read_text(encoding="utf-8").splitlines()
        if line.startswith("unset ")
    ]
    assert names == [PANEL_TOKEN_ENV]
    code = [line for line in SHELL_ENV_FILE.read_text(encoding="utf-8").splitlines() if line]
    assert all(line.startswith(("#", "unset ")) for line in code)
