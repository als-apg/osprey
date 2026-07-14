"""E2E tests for the ``osprey query`` CLI command.

Exercises the *shipped* command path end-to-end.

Template: ``hello_world`` — the minimal preset with one mock ``controls``
server and an ``example`` channel.  It is self-contained (no ARIEL Postgres,
no real EPICS hardware) and resolves exit 0 in this environment.  The
``control_assistant`` preset was considered but skipped: it declares additional
MCP servers (channel-finder, ARIEL) that require running infrastructure absent
here.

Two tests, two harnesses
------------------------
``test_query_cmd_default_output`` uses ``click.testing.CliRunner`` for the
plain-text code path.  This exercises the Click entry point directly.

``test_query_cmd_json_mcp_connected`` uses ``subprocess.run`` for the
``--json`` path.  Rationale: CliRunner's BytesIO stdout capture and
``sys.exit()`` interact unexpectedly when ``asyncio.run()`` is called inside
the command — the output buffer is empty even though exit_code=0.  The plain
subprocess path (same mechanism ``init_project`` uses) avoids this issue and
is more accurately "end-to-end" for the JSON assertion.

Note on stdout mixing: the SDK logs "Using bundled Claude Code CLI: ..." to
stdout (via Rich) before the JSON payload.  We extract the JSON by finding the
first ``{`` in stdout rather than parsing the full output string.

Mechanism-based verdict
-----------------------
Exit 0 requires ALL declared MCP servers to connect and the agent to finish
without error.  Prompts are operator-style (no inlined channel addresses or
"if any" hedges) — the tests assert mechanism, not specific tool calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.query_cmd import query
from tests.e2e.sdk_helpers import HAS_SDK, init_project, is_claude_code_available

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available"),
]


@pytest.mark.flaky(reruns=2)
def test_query_cmd_default_output(tmp_path: Path) -> None:
    """``osprey query`` prints a non-empty answer and exits 0.

    Uses the hello_world preset (mock controls, no external infra).
    Exercises the CliRunner path for the plain-text output code branch.
    """
    project = init_project(tmp_path, "hw_query", template="hello_world", provider="als-apg")
    runner = CliRunner()
    result = runner.invoke(
        query,
        ["--project", str(project), "What channels are available on the controls MCP server?"],
    )
    assert result.exit_code == 0, (
        f"osprey query exited {result.exit_code}.\n"
        f"Output: {result.output!r}\n"
        f"Exception: {result.exception!r}"
    )
    assert result.output.strip(), "Expected non-empty printed answer from osprey query"


@pytest.mark.flaky(reruns=2)
def test_query_cmd_json_mcp_connected(tmp_path: Path) -> None:
    """``osprey query --json`` reports the controls MCP server as connected.

    Uses subprocess (same pattern as ``init_project``) rather than CliRunner:
    CliRunner + asyncio.run() + sys.exit() does not flush output into the
    BytesIO buffer reliably for the ``--json`` code path.

    The SDK logs "Using bundled Claude Code CLI: ..." to stdout (via Rich)
    before emitting the JSON payload.  JSON is extracted by finding the first
    ``{`` in stdout rather than parsing the full output string.

    Asserts ``mcp_servers.controls == "connected"`` — the mechanism contract
    proving all declared servers were online before the agent ran.
    """
    project = init_project(tmp_path, "hw_query_json", template="hello_world", provider="als-apg")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "osprey.cli.main",
            "query",
            "--project",
            str(project),
            "--json",
            "Read the current value of the example channel and report it.",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"osprey query --json exited {result.returncode}.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    # Strip the SDK's Rich-formatted logging preamble that goes to stdout
    # before the JSON payload (the `{` always opens on its own line).
    json_start = result.stdout.find("{")
    assert json_start != -1, f"No JSON object found in stdout.\nstdout: {result.stdout!r}"
    payload = json.loads(result.stdout[json_start:])
    assert payload["mcp_servers"].get("controls") == "connected", (
        f"Expected controls server to be connected. mcp_servers: {payload['mcp_servers']}"
    )
    assert payload["exit_code"] == 0, (
        f"JSON payload exit_code should be 0, got {payload['exit_code']}"
    )
    assert payload["final_text"].strip(), "Expected non-empty final_text in JSON payload"
