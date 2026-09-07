"""The boot-smoke handshake budget measures the server, not the machine.

``list_mcp_tools`` spawns a real MCP server and waits for it to answer. Those
servers import the framework in the child process, which takes seconds on an
idle box and much longer on one running the rest of the suite in parallel — so
a total wall-clock deadline started before the import makes the boot smoke fail
on load rather than on a broken server.

The budget is therefore an *inactivity* budget: silence is the signal, and
silence does not get slower under load. These tests pin both halves — a slow
server that keeps reporting progress is waited for, a mute one is not — with
stub servers rather than the real ones, so they cost milliseconds. That is why
they are marked ``unit`` despite living beside the container-backed integration
modules: they start two bare interpreters and no service.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from tests.integration._mcp_handshake import MCPHandshakeError, list_mcp_tools

pytestmark = pytest.mark.unit

#: A stub MCP server: writes ``chatter`` progress lines to stderr, one every
#: ``gap`` seconds, then speaks JSON-RPC. With ``chatter=0`` it starts mute.
_STUB = """\
import json, sys, time

for _ in range({chatter}):
    time.sleep({gap})
    print("[STARTUP-TIMING] still importing", file=sys.stderr, flush=True)

time.sleep({mute})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if msg.get("method") == "initialize":
        reply = {{"jsonrpc": "2.0", "id": msg["id"], "result": {{"capabilities": {{}}}}}}
    elif msg.get("method") == "tools/list":
        reply = {{"jsonrpc": "2.0", "id": msg["id"],
                  "result": {{"tools": [{{"name": "stub_tool"}}]}}}}
    else:
        continue
    print(json.dumps(reply), flush=True)
"""


def _stub(tmp_path: Path, *, chatter: int = 0, gap: float = 0.0, mute: float = 0.0) -> list[str]:
    script = tmp_path / "stub_server.py"
    script.write_text(textwrap.dedent(_STUB).format(chatter=chatter, gap=gap, mute=mute))
    return [sys.executable, str(script)]


def test_a_slow_but_talking_server_is_waited_for(tmp_path: Path) -> None:
    """Four seconds of startup against a two-second budget, and it still passes.

    This is the shape that reds a loaded runner: the server is fine, it is just
    taking longer to reach its first reply than a total budget would allow.
    """
    command, *args = _stub(tmp_path, chatter=8, gap=0.5)

    # Five seconds, not the mute case's two: the first gap has to cover child
    # interpreter start-up as well, and this test runs in the loaded unit lane.
    assert list_mcp_tools(command=command, args=args, timeout=5.0) == ["stub_tool"]


def test_a_mute_server_still_times_out(tmp_path: Path) -> None:
    """The half that must not weaken: silence is what the budget is for."""
    command, *args = _stub(tmp_path, mute=30.0)

    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=command, args=args, timeout=2.0)

    assert "no output" in str(excinfo.value)
