"""An MCP server's reload of the repo ``.env`` never brings back a credential.

The deployment mints its credentials -- the terminal secret, the services'
launch tokens -- into the repo ``.env``, and every MCP server reloads that chain
at startup (:func:`osprey.mcp_env.load_dotenv_from_project`). An MCP server is
a process the agent spawned, so a credential that loader read back would undo
every strip the terminal made on the way down. The loader therefore skips each
name :func:`osprey.utils.sensitive_env.is_sensitive` accepts, unless the caller
is the terminal itself and asks for them by name.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.mcp_env import load_dotenv_from_project
from osprey.utils.sensitive_env import is_sensitive

PLAIN_KEY = "OSPREY_TEST_SENSITIVE_PLAIN"

CHAIN = {
    "PANEL_LAUNCH_TOKEN": "minted-panel-token",
    "OSPREY_TERMINAL_SECRET": "minted-terminal-secret",
    "BLUESKY_LAUNCH_TOKEN": "minted-bluesky-token",
    "EVENT_DISPATCHER_TOKEN": "minted-dispatcher-token",
    PLAIN_KEY: "plain-value",
}


@pytest.fixture(autouse=True)
def _clear_chain_keys(monkeypatch):
    """Start every test with none of the chain's keys in the environment.

    The root conftest's ``restore_environ`` puts the environment back afterwards.
    """
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    for key in CHAIN:
        monkeypatch.delenv(key, raising=False)


def _single_user_repo(root: Path) -> Path:
    """A repo with its env chain at the root and a render under ``build/``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".env").write_text(
        "".join(f"{key}={value}\n" for key, value in CHAIN.items()), encoding="utf-8"
    )
    build = root / "build"
    build.mkdir()
    config = build / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    return config


def test_every_sensitive_chain_name_is_sensitive():
    """The fixture's credential names are the ones the shared rule names."""
    assert [k for k in CHAIN if is_sensitive(k)] == [k for k in CHAIN if k != PLAIN_KEY]


def test_the_loader_skips_sensitive_names(tmp_path):
    os.environ["OSPREY_CONFIG"] = str(_single_user_repo(tmp_path))

    load_dotenv_from_project()

    assert os.environ[PLAIN_KEY] == "plain-value"
    for key in CHAIN:
        if key != PLAIN_KEY:
            assert key not in os.environ, key


def test_an_empty_shell_placeholder_is_not_refilled(tmp_path):
    """``${VAR:-}`` arrives as ``VAR=""``; the clear must not open a refill."""
    os.environ["OSPREY_CONFIG"] = str(_single_user_repo(tmp_path))
    os.environ["PANEL_LAUNCH_TOKEN"] = ""

    load_dotenv_from_project()

    assert "PANEL_LAUNCH_TOKEN" not in os.environ


def test_the_terminal_may_ask_for_its_credentials(tmp_path):
    """The terminal is the process that holds them; it opts in by name."""
    os.environ["OSPREY_CONFIG"] = str(_single_user_repo(tmp_path))

    load_dotenv_from_project(include_sensitive=True)

    for key, value in CHAIN.items():
        assert os.environ[key] == value


_FAKE_SERVER = textwrap.dedent(
    """
    import json
    import os


    class _Server:
        def add_middleware(self, _middleware):
            pass

        def run(self):
            names = {json_names!r}
            with open(os.environ["OSPREY_TEST_ENV_DUMP"], "w") as fh:
                json.dump({{n: os.environ.get(n) for n in names}}, fh)


    def create_server():
        return _Server()
    """
)


def test_an_mcp_server_under_a_single_user_render_lacks_the_credentials(tmp_path):
    """The real startup path, in its own process, as the agent would spawn it."""
    config = _single_user_repo(tmp_path / "repo")
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "fake_panel_server.py").write_text(
        _FAKE_SERVER.format(json_names=sorted(CHAIN)), encoding="utf-8"
    )
    dump = tmp_path / "env.json"

    env = {k: v for k, v in os.environ.items() if k not in CHAIN}
    env.update(
        {
            "OSPREY_CONFIG": str(config),
            "OSPREY_TEST_ENV_DUMP": str(dump),
            "PYTHONPATH": os.pathsep.join([str(modules), env.get("PYTHONPATH", "")]),
            # Claude Code renders ${VAR:-} into .mcp.json, so an unset shell
            # variable reaches the server as an empty string.
            "PANEL_LAUNCH_TOKEN": "",
            "OSPREY_TERMINAL_SECRET": "",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from osprey.mcp_server.startup import run_mcp_server; "
            "run_mcp_server('fake_panel_server')",
        ],
        env=env,
        cwd=str(config.parent),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    seen = json.loads(dump.read_text(encoding="utf-8"))
    assert seen[PLAIN_KEY] == "plain-value"
    assert seen["PANEL_LAUNCH_TOKEN"] is None
    assert seen["OSPREY_TERMINAL_SECRET"] is None
    assert seen["BLUESKY_LAUNCH_TOKEN"] is None
    assert seen["EVENT_DISPATCHER_TOKEN"] is None
