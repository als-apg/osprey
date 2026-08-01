"""stdout purity for MCP stdio servers.

An MCP server speaks JSON-RPC over stdio: stdout carries protocol frames and
nothing else. A single log byte written there desynchronizes the client's frame
parser and kills the session. ``configure_logging()`` routes every record to
stderr, which makes that safe by construction — this pins it end to end, in a
real subprocess, because the hazard only exists in a real process.

The probe drives ``osprey.mcp_server.workspace``: it goes through
``run_mcp_server()``, the shared entry point behind every MCP server, so the
guarantee it pins is the shared one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVER_MODULE = "osprey.mcp_server.workspace"

# Enough of a handshake to make the server answer with a frame, so the purity
# assertion below is about a stdout that actually carried protocol traffic.
INITIALIZE_REQUEST = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "stdout-purity-probe", "version": "1"},
    },
}


@pytest.fixture(scope="module")
def server_probe(tmp_path_factory) -> subprocess.CompletedProcess[str]:
    """Boot the workspace MCP server, send one request, and let stdin hit EOF.

    Module-scoped: the subprocess costs about a second and both tests below
    describe the same run.
    """
    workdir: Path = tmp_path_factory.mktemp("mcp_stdout_purity")

    # Curated env: OSPREY_CONFIG would point the server at the developer's own
    # project, and the run must not depend on what that config contains.
    env = {k: v for k, v in os.environ.items() if k != "OSPREY_CONFIG"}

    return subprocess.run(
        [sys.executable, "-m", SERVER_MODULE],
        input=json.dumps(INITIALIZE_REQUEST) + "\n",
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=workdir,
    )


@pytest.mark.unit
def test_stdout_carries_only_jsonrpc_frames(server_probe) -> None:
    """Every stdout line parses as a JSON-RPC object — no log bytes anywhere."""
    lines = [line for line in server_probe.stdout.splitlines() if line.strip()]

    assert lines, (
        "server wrote nothing to stdout; the purity assertion would be vacuous. "
        f"exit={server_probe.returncode} stderr tail:\n{server_probe.stderr[-2000:]}"
    )

    for line in lines:
        payload = json.loads(line)  # a stray log byte makes this raise
        assert payload.get("jsonrpc") == "2.0", f"non-JSON-RPC frame on stdout: {line[:200]}"


@pytest.mark.unit
def test_logging_was_active_and_went_to_stderr(server_probe) -> None:
    """The companion assertion: logs were produced, and they landed on stderr.

    Without this, ``test_stdout_carries_only_jsonrpc_frames`` would also pass on
    a server that simply never logged anything.
    """
    assert "[STARTUP-TIMING]" in server_probe.stderr
    assert "MCP server" in server_probe.stderr
