"""A child that fails the launch handshake is refused with what it said or how it exited."""

import asyncio
import sys

import pytest

from osprey.mcp_server.control_system.connector_host_manager import (
    REASON_SPAWN_FAILED,
    SwitchError,
    _launch_request,
)
from osprey_connectors.ipc.launch import AttributedReader
from osprey_connectors.ipc.proxy import ConnectorHostProxy

#: Reads the start of the request, closes its output stream, then exits a moment
#: later without answering: the supervisor sees end-of-stream before the exit.
_DIES_ON_REQUEST = (
    "import os, sys, time\nsys.stdin.buffer.read(1)\nos.close(1)\ntime.sleep(0.1)\nos._exit(5)\n"
)

#: Reads one whole request, answers it with an error frame, then exits.
_REFUSES_ON_REQUEST = (
    "import sys\n"
    "from osprey_connectors.ipc import frames\n"
    "reader = frames.FrameReader()\n"
    "decoded = []\n"
    "while not decoded:\n"
    "    chunk = sys.stdin.buffer.read1(65536)\n"
    "    if not chunk:\n"
    "        sys.exit(3)\n"
    "    decoded = reader.feed(chunk)\n"
    "error = RuntimeError('no connector type configured')\n"
    "sys.stdout.buffer.write(frames.encode_error(decoded[0].request_id, error))\n"
    "sys.stdout.buffer.flush()\n"
)


async def _child(script: str):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    proxy = ConnectorHostProxy(AttributedReader(process.stdout), process.stdin)
    return process, proxy


async def test_a_child_that_exits_during_the_handshake_reports_its_exit_code():
    process, proxy = await _child(_DIES_ON_REQUEST)
    try:
        with pytest.raises(SwitchError) as caught:
            await _launch_request("live", process, proxy, "init", {}, 10.0, "spawn", 2.0)
    finally:
        await proxy.disconnect(ack_timeout=0.0)

    assert "closed its output stream while answering 'init' (exit code 5)" in str(caught.value)
    assert process.returncode == 5


async def test_a_child_error_frame_during_init_is_a_spawn_refusal():
    process, proxy = await _child(_REFUSES_ON_REQUEST)
    try:
        with pytest.raises(SwitchError) as caught:
            await _launch_request("live", process, proxy, "init", {}, 10.0, "spawn", 2.0)
    finally:
        await proxy.disconnect(ack_timeout=0.0)
        await process.wait()

    assert caught.value.reason == REASON_SPAWN_FAILED
    assert "failed 'init':" in caught.value.detail
    assert "no connector type configured" in caught.value.detail
