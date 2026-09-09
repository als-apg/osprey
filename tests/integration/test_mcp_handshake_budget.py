"""The boot-smoke handshake budget measures the server, not the machine.

``list_mcp_tools`` spawns a real MCP server and waits for it to answer. Those
servers import the framework in the child process, which takes seconds on an
idle box and much longer on one running the rest of the suite in parallel — so
a budget started at ``Popen`` is spent on the import and reports a slow start
as an unanswered handshake.

The helper therefore reads the server's own startup phases from stderr: a server
that announces a phase is starting rather than stalling and is held to the
startup allowance; once it reports ``total_startup`` the answer budget begins.
A server that announces nothing is indistinguishable from one that is stuck and
keeps the literal budget. These tests pin both halves — a slow server that
reports its phases is waited for, a mute one is not — with stub servers rather
than the real ones, so they cost seconds. That is why they are marked ``unit``
despite living beside the container-backed integration modules: they start two
bare interpreters and no service.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from tests.integration._mcp_handshake import MCPHandshakeError, list_mcp_tools

pytestmark = pytest.mark.unit

#: A stub MCP server. With ``phases`` above zero it reports that many startup
#: phases to stderr in the framework's own ``[STARTUP-TIMING]`` shape, one every
#: ``gap`` seconds, closes with ``total_startup``, then speaks JSON-RPC. With
#: ``phases=0`` it announces nothing and sleeps ``mute`` seconds first.
_STUB = """\
import json, sys, time

for i in range({phases}):
    time.sleep({gap})
    print(f"[STARTUP-TIMING] stub | import_{{i}}: {{ {gap} * 1000:.0f}}ms", file=sys.stderr, flush=True)
if {phases}:
    print(f"[STARTUP-TIMING] stub | total_startup: {{ {phases} * {gap} * 1000:.0f}}ms", file=sys.stderr, flush=True)

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


def _stub(tmp_path: Path, *, phases: int = 0, gap: float = 0.0, mute: float = 0.0) -> list[str]:
    script = tmp_path / "stub_server.py"
    script.write_text(textwrap.dedent(_STUB).format(phases=phases, gap=gap, mute=mute))
    return [sys.executable, str(script)]


def test_a_slow_but_announcing_server_is_waited_for(tmp_path: Path) -> None:
    """Six seconds of announced startup against a three-second budget still passes.

    This is the shape that reds a loaded runner: the server is fine, it is just
    taking longer to reach its first reply than a budget started at spawn would
    allow. Its first phase has to land inside the budget — it does within a
    second even on a loaded box — and from there the startup allowance holds.
    """
    command, *args = _stub(tmp_path, phases=12, gap=0.5)

    assert list_mcp_tools(command=command, args=args, timeout=3.0) == ["stub_tool"]


def test_a_mute_server_still_times_out(tmp_path: Path) -> None:
    """The half that must not weaken: a server that announces nothing keeps the budget."""
    command, *args = _stub(tmp_path, mute=30.0)

    with pytest.raises(MCPHandshakeError) as excinfo:
        list_mcp_tools(command=command, args=args, timeout=2.0)

    assert "timeout waiting for" in str(excinfo.value)
