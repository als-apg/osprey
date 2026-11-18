"""A child that fails the launch handshake is refused with what it said or how it exited."""

import asyncio
import sys

import pytest

from osprey.mcp_server.control_system.connector_host_manager import (
    REASON_SPAWN_FAILED,
    SwitchError,
)
from osprey_connectors.ipc import HostInit
from osprey_connectors.ipc import _handle as handle_module

#: Reads the start of the request, closes its output stream, then exits a moment
#: later without answering: the supervisor sees end-of-stream before the exit.
_DIES_ON_REQUEST = (
    "import os, sys, time\nsys.stdin.buffer.read(1)\nos.close(1)\ntime.sleep(0.1)\nos._exit(5)\n"
)

#: Reads one whole request, answers it with an error frame, then exits.
_REFUSES_ON_REQUEST = (
    "import sys\n"
    "from osprey_connectors.ipc import _frames as frames\n"
    "from osprey_connectors.ipc._exceptions import encode_exception\n"
    "reader = frames.FrameReader()\n"
    "decoded = []\n"
    "while not decoded:\n"
    "    chunk = sys.stdin.buffer.read1(65536)\n"
    "    if not chunk:\n"
    "        sys.exit(3)\n"
    "    decoded = reader.feed(chunk)\n"
    "error = RuntimeError('no connector type configured')\n"
    "sys.stdout.buffer.write(frames.encode_error(decoded[0].request_id, encode_exception(error)))\n"
    "sys.stdout.buffer.flush()\n"
)


@pytest.fixture
def child_running(monkeypatch):
    """Make every child the handle starts run *script* instead; return what was spawned."""
    spawned: list[asyncio.subprocess.Process] = []

    def use(script: str) -> list[asyncio.subprocess.Process]:
        async def spawn(_python, _env, *, start_marks_fd=None):  # noqa: ARG001 - _spawn's signature
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            spawned.append(process)
            return process

        monkeypatch.setattr(handle_module, "_spawn", spawn)
        return spawned

    return use


def _init() -> HostInit:
    return HostInit(control_system={}, target="live")


async def test_a_child_that_exits_during_the_handshake_reports_its_exit_code(
    make_manager, child_running
):
    spawned = child_running(_DIES_ON_REQUEST)
    manager = make_manager(spawn_timeout_s=10.0)

    with pytest.raises(SwitchError) as caught:
        await manager._spawn("live", _init())

    assert "closed its output stream while answering 'init' (exit code 5)" in str(caught.value)
    assert spawned[0].returncode == 5


async def test_a_child_error_frame_during_init_is_a_spawn_refusal(make_manager, child_running):
    spawned = child_running(_REFUSES_ON_REQUEST)
    manager = make_manager(spawn_timeout_s=10.0)

    with pytest.raises(SwitchError) as caught:
        await manager._spawn("live", _init())
    await asyncio.wait_for(spawned[0].wait(), 10.0)

    assert caught.value.reason == REASON_SPAWN_FAILED
    assert "failed 'init':" in caught.value.detail
    assert "no connector type configured" in caught.value.detail
