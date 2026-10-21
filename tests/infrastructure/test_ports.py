"""The loopback port reservation that test-started servers bind."""

from __future__ import annotations

import fcntl
import os
import socket
import subprocess
import sys
import textwrap

import pytest

from tests import _ports
from tests._ports import ephemeral_range_start, lock_path, reserved_port


def test_the_port_lies_below_the_ephemeral_range():
    with reserved_port() as port:
        assert _ports.PORT_FLOOR <= port < ephemeral_range_start()


def test_two_nested_reservations_never_share_a_port():
    with reserved_port() as outer, reserved_port() as inner:
        assert outer != inner


def _only_candidate(monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    monkeypatch.setattr(_ports, "_candidates", lambda first: [port])


def _a_reservable_port() -> int:
    with reserved_port() as port:
        return port


def test_a_port_locked_by_another_process_is_never_yielded(monkeypatch):
    port = _a_reservable_port()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import fcntl, sys
                handle = open({str(lock_path(port))!r}, "a")
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                sys.stdout.write("held\\n")
                sys.stdout.flush()
                sys.stdin.read()
                """
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline() == "held\n"
        _only_candidate(monkeypatch, port)
        with pytest.raises(RuntimeError, match="1 candidate"), reserved_port():
            pass
    finally:
        holder.communicate(timeout=10)


@pytest.mark.parametrize(
    "kind", [socket.SOCK_STREAM, socket.SOCK_DGRAM], ids=["tcp-listener", "udp-socket"]
)
def test_a_port_already_bound_is_never_yielded(monkeypatch, kind):
    port = _a_reservable_port()
    with socket.socket(socket.AF_INET, kind) as taken:
        taken.bind(("127.0.0.1", port))
        if kind == socket.SOCK_STREAM:
            taken.listen()
        _only_candidate(monkeypatch, port)
        with pytest.raises(RuntimeError, match="1 candidate"), reserved_port():
            pass


def test_the_lock_is_released_when_the_block_ends():
    with reserved_port() as port:
        pass
    fd = os.open(lock_path(port), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
