"""The real agent CLI loads no MCP server the build did not render.

Each probe starts the Claude Code binary the Agent SDK bundles and reads the
``system/init`` line, which lists the loaded MCP servers and the tools the agent
may call. That line arrives before any model request, so the probes need no key,
no network and no container: the base URL points at a closed port.

The project renders two servers into ``.mcp.json``: ``rendered`` and
``plugin_spoof``. A ``--plugin-dir`` plugin brings a third, ``browser``, which
the CLI lists as ``plugin:fakeplug:browser`` and whose tools fall under
``mcp__plugin_``. The launch paths must load exactly the two rendered servers;
the deny floors must strip every plugin tool on their own.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from osprey.agent_runner.launcher import (
    build_claude_launch_argv,
    build_session_argv,
    bundled_cli_path,
)
from osprey.agent_runner.primitives import build_agent_options
from osprey.agent_runner.tool_names import DENY_DEFAULTS
from osprey.agent_runner.write_tools import read_only_disallowed_tools

pytestmark = pytest.mark.slow

#: Upper bound on the wait for the init line; it normally arrives in 2-4 s.
_INIT_TIMEOUT_S = 60.0

#: The servers the project renders into ``.mcp.json``.
_RENDERED = {"rendered", "plugin_spoof"}

#: Flags that make the CLI print its init line as JSON and stop after one turn.
_PROBE_FLAGS = ["--output-format", "stream-json", "--verbose", "--max-turns", "1"]

_PING_SERVER = """\
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("probe")


@mcp.tool()
def ping() -> str:
    return "pong"


mcp.run()
"""


@dataclasses.dataclass(frozen=True)
class Init:
    """What the CLI's init line says the agent was given."""

    servers: set[str]
    mcp_tools: set[str]


def _init_from(payload: dict) -> Init:
    servers = {entry["name"] for entry in payload.get("mcp_servers", [])}
    tools = {tool for tool in payload.get("tools", []) if tool.startswith("mcp__")}
    return Init(servers, tools)


@pytest.fixture
def cli() -> str:
    path = bundled_cli_path()
    if path is None:
        pytest.fail(
            "the Agent SDK's bundled Claude Code binary is missing; these probes need it "
            "and must not skip, because a skipped probe proves nothing"
        )
    return str(path)


@pytest.fixture
def ping_script(tmp_path: Path) -> Path:
    script = tmp_path / "ping_server.py"
    script.write_text(_PING_SERVER, encoding="utf-8")
    return script


def _stdio(script: Path, name: str) -> dict:
    """A stdio entry for the ping server. The name is passed as an argument so no two
    entries share a command line, which the CLI would load as one server."""
    return {"type": "stdio", "command": sys.executable, "args": [str(script), name]}


@pytest.fixture
def project(tmp_path: Path, ping_script: Path) -> Path:
    root = tmp_path / "project"
    (root / ".claude").mkdir(parents=True)
    servers = {name: _stdio(ping_script, name) for name in sorted(_RENDERED)}
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return root


@pytest.fixture
def plugin(tmp_path: Path, ping_script: Path) -> Path:
    root = tmp_path / "fakeplug"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "fakeplug", "version": "0.0.1"}), encoding="utf-8"
    )
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"browser": _stdio(ping_script, "browser")}}), encoding="utf-8"
    )
    return root


@pytest.fixture
def probe_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    return {
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "ANTHROPIC_API_KEY": "probe",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        # Connect every server before the init line, so it lists their tools
        # rather than a server still pending.
        "MCP_CONNECTION_NONBLOCKING": "0",
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
    }


def _read_init(argv: Sequence[str], cwd: Path, env: dict[str, str]) -> Init:
    """Run *argv* until its init line, then stop it and return what it lists."""
    proc = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()
    stderr_lines: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def drain_stderr() -> None:
        # Read stderr as it arrives, so a CLI that writes more than the pipe
        # buffer before its init line never blocks on the write.
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)

    threading.Thread(target=pump, daemon=True).start()
    stderr_reader = threading.Thread(target=drain_stderr, daemon=True)
    stderr_reader.start()

    def stderr_text() -> str:
        return "".join(stderr_lines).strip()

    deadline = time.monotonic() + _INIT_TIMEOUT_S
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(
                    f"no init line within {_INIT_TIMEOUT_S:.0f} s from {argv!r}; "
                    f"stderr: {stderr_text()}"
                )
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                proc.wait(timeout=10)
                stderr_reader.join(10)
                pytest.fail(
                    f"the CLI exited {proc.returncode} before its init line: {stderr_text()}"
                )
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if payload.get("type") == "system" and payload.get("subtype") == "init":
                return _init_from(payload)
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _assert_only_rendered(init: Init) -> None:
    assert init.servers == _RENDERED
    assert not any(name.startswith("plugin:") for name in init.servers)


def test_the_terminal_argv_loads_only_the_rendered_servers(cli, project, plugin, probe_env):
    argv = [
        cli,
        *build_claude_launch_argv({})[1:],
        "--plugin-dir",
        str(plugin),
        "-p",
        "hi",
        *_PROBE_FLAGS,
    ]

    _assert_only_rendered(_read_init(argv, project, probe_env))


def test_the_chat_argv_keeps_its_trailing_prompt(cli, project, plugin, probe_env):
    chat = build_session_argv(build_claude_launch_argv({}), print_mode=True, prompt="hi")
    argv = [cli, "--plugin-dir", str(plugin), *_PROBE_FLAGS, *chat[1:]]
    assert argv[-3:] == ["--mcp-config=.mcp.json", "--print", "hi"]

    _assert_only_rendered(_read_init(argv, project, probe_env))


def test_an_sdk_run_loads_only_the_rendered_servers(project, plugin, probe_env):
    from claude_agent_sdk import ClaudeSDKClient, SystemMessage

    options = build_agent_options(project, disallowed_tools=[], env=probe_env)
    options = dataclasses.replace(
        options, plugins=[{"type": "local", "path": str(plugin)}], max_turns=1
    )

    async def first_init() -> Init:
        async with ClaudeSDKClient(options=options) as client:
            await client.query("hi")
            async for message in client.receive_messages():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    return _init_from(message.data)
        pytest.fail("the SDK stream ended before its init message")

    _assert_only_rendered(asyncio.run(asyncio.wait_for(first_init(), _INIT_TIMEOUT_S)))


def _unisolated_argv(cli: str, plugin: Path) -> list[str]:
    """The launch argv without the MCP isolation: the deny floor is all that is left."""
    return [
        cli,
        "--setting-sources",
        "project",
        "--plugin-dir",
        str(plugin),
        "-p",
        "hi",
        *_PROBE_FLAGS,
    ]


def _assert_plugin_tools_denied(init: Init) -> None:
    """The plugin's server loaded, a rendered tool survived, and no plugin tool did."""
    assert "plugin:fakeplug:browser" in init.servers
    assert "mcp__rendered__ping" in init.mcp_tools
    assert not any(tool.startswith("mcp__plugin_") for tool in init.mcp_tools)


def _write_settings(project: Path, settings: dict) -> None:
    (project / ".claude" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")


def test_the_interactive_deny_floor_removes_plugin_tools_on_its_own(
    cli, project, plugin, probe_env
):
    _write_settings(
        project,
        {"enableAllProjectMcpServers": True, "permissions": {"deny": list(DENY_DEFAULTS)}},
    )

    init = _read_init(_unisolated_argv(cli, plugin), project, probe_env)

    _assert_plugin_tools_denied(init)


def test_the_headless_floor_removes_plugin_tools_on_its_own(cli, project, plugin, probe_env):
    _write_settings(project, {"enableAllProjectMcpServers": True})
    argv = [
        *_unisolated_argv(cli, plugin),
        "--disallowedTools",
        ",".join(read_only_disallowed_tools(project)),
    ]

    init = _read_init(argv, project, probe_env)

    _assert_plugin_tools_denied(init)
