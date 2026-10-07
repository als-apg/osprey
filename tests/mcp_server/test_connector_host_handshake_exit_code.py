"""A child that dies during the launch handshake is reported with its own exit code."""

import asyncio
import sys

import pytest

from osprey.mcp_server.control_system.connector_host_manager import SwitchError, _LaunchChannel

#: Reads the start of the request, closes its output stream, then exits a moment
#: later without answering: the supervisor sees end-of-stream before the exit.
_DIES_ON_REQUEST = (
    "import os, sys, time\nsys.stdin.buffer.read(1)\nos.close(1)\ntime.sleep(0.1)\nos._exit(5)\n"
)


async def test_a_child_that_exits_during_the_handshake_reports_its_exit_code():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _DIES_ON_REQUEST,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    channel = _LaunchChannel("live", process)

    with pytest.raises(SwitchError) as caught:
        await channel.request("init", {}, 10.0, "spawn")

    assert "closed its output stream while answering 'init' (exit code 5)" in str(caught.value)
    assert process.returncode == 5
