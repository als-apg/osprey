"""The loopback port reservation that test-started servers bind."""

from __future__ import annotations

import ast
import fcntl
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.deployment.subprocess_capture import stop_group
from tests import _ports
from tests._ports import (
    bound_socket,
    ephemeral_range_start,
    lock_path,
    require_ipv6_loopback,
    reserved_port,
    reserved_ports,
)

#: Seconds a lock holder gets between SIGTERM and SIGKILL when it did not end.
_HOLDER_GRACE_S = 1.0


def _end_holder(holder: subprocess.Popen[str]) -> None:
    """End a lock holder by closing its stdin; a holder that ignores EOF is stopped as a group.

    The group stop runs only while the leader is unreaped, which is the case
    exactly when ``communicate`` timed out.
    """
    try:
        holder.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        survivors = stop_group(holder.pid, grace=_HOLDER_GRACE_S)
        holder.wait()
        for pipe in (holder.stdin, holder.stdout):
            if pipe is not None:
                pipe.close()
        if survivors is None:
            pytest.fail("the lock holder ignored EOF; the process table could not be read")
        if survivors:
            pytest.fail(f"the lock holder ignored EOF; still running after SIGKILL: {survivors}")
        pytest.fail("the lock holder ignored EOF; its group was stopped")


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
        start_new_session=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline() == "held\n"
        _only_candidate(monkeypatch, port)
        with pytest.raises(RuntimeError, match="1 candidate"), reserved_port():
            pass
    finally:
        _end_holder(holder)


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


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def test_the_lock_directory_ignores_tmpdir(tmp_path):
    with reserved_port() as port:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from tests._ports import _lock; print(_lock({port}))",
            ],
            cwd=_repo_root(),
            env={**os.environ, "TMPDIR": str(tmp_path)},
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    assert probe.stdout.strip() == "None"


def _lockable_elsewhere(ports: list[int]) -> list[bool]:
    """Whether a fresh process can take each port's lock."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tests._ports import _lock; print([_lock(p) is not None for p in {ports!r}])",
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return ast.literal_eval(probe.stdout)


def test_reserved_ports_are_distinct_and_held():
    with reserved_ports(3) as ports:
        assert len(set(ports)) == 3
        assert all(_ports.PORT_FLOOR <= port < ephemeral_range_start() for port in ports)
        assert _lockable_elsewhere(ports) == [False, False, False]
    assert _lockable_elsewhere(ports) == [True, True, True]


def test_hold_ports_keeps_the_locks_until_exit():
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                from tests._ports import hold_ports
                print(hold_ports(2), flush=True)
                sys.stdin.read()
                """
            ),
        ],
        cwd=_repo_root(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert holder.stdout is not None
        ports = ast.literal_eval(holder.stdout.readline())
        assert len(set(ports)) == 2
        assert _lockable_elsewhere(ports) == [False, False]
    finally:
        _end_holder(holder)
    assert holder.returncode == 0
    assert _lockable_elsewhere(ports) == [True, True]


def test_a_reserved_port_refuses_connections():
    with reserved_port() as port, socket.socket() as client:
        client.settimeout(5)
        with pytest.raises(ConnectionRefusedError):
            client.connect(("127.0.0.1", port))


@pytest.mark.parametrize(
    ("host", "family"),
    [("127.0.0.1", socket.AF_INET), ("::1", socket.AF_INET6)],
    ids=["ipv4", "ipv6"],
)
def test_a_bound_socket_holds_its_port(host, family):
    if family == socket.AF_INET6:
        require_ipv6_loopback()
    with bound_socket(host) as held:
        assert held.family == family
        port = held.getsockname()[1]
        with socket.socket(family, socket.SOCK_STREAM) as rival, pytest.raises(OSError):
            rival.bind((host, port))


def test_a_bound_socket_can_listen():
    with bound_socket() as held, socket.socket() as client:
        held.listen()
        client.settimeout(5)
        client.connect(("127.0.0.1", held.getsockname()[1]))
