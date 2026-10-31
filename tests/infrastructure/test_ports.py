"""The loopback port reservation that test-started servers bind."""

from __future__ import annotations

import ast
import fcntl
import os
import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.deployment.subprocess_capture import observe_exit, stop_child
from osprey.port_layout import BLOCK_SIZE
from osprey_connectors_tests._harness import _ports
from osprey_connectors_tests._harness._ports import (
    bound_socket,
    ephemeral_range_start,
    lock_path,
    port_reserver,
    require_ipv6_loopback,
    reserved_port,
    reserved_ports,
)
from tests._nested_pytest import with_harness_path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The highest first port of an ephemeral range a test host is likely to run:
#: IANA's dynamic range, macOS's default. Reservations are handed out below it.
_HIGHEST_EPHEMERAL_START = 49152

#: A pinned deployment block: ``port_base``, ``PORT_BASE`` or ``worker_port_base``
#: set to a literal, in Python, YAML or an ``osprey set`` argument.
_PINNED_BASE = re.compile(r"port_base\W{1,4}(\d{4,5})\b", re.IGNORECASE)

#: Seconds a lock holder gets between SIGTERM and SIGKILL when it did not end.
_HOLDER_GRACE_S = 1.0


def _end_holder(holder: subprocess.Popen[str]) -> None:
    """End a lock holder by closing its stdin, then stop its group, which reaps it.

    A holder exits on EOF; one that ignores it is stopped all the same and
    fails the test.
    """
    if holder.stdin is not None:
        holder.stdin.close()
    exited = observe_exit(holder, within=10.0)
    if holder.stdout is not None:
        holder.stdout.close()
    outcome = stop_child(holder, grace=_HOLDER_GRACE_S)
    assert outcome.gone, outcome.describe()
    if exited is None:
        pytest.fail("the lock holder ignored EOF; its group was stopped")


def test_the_port_lies_below_the_ephemeral_range():
    with reserved_port() as port:
        assert _ports.PORT_FLOOR <= port < ephemeral_range_start()


def _pinned(port: int) -> bool:
    return any(low <= port < high for low, high in _ports.PINNED_BLOCKS)


def test_no_candidate_lies_in_a_pinned_block():
    candidates = _ports._candidates(_HIGHEST_EPHEMERAL_START)
    assert candidates
    assert not any(_pinned(port) for port in candidates)


def _pinned_bases() -> dict[int, list[str]]:
    """Every literal deployment base a test file pins, with the files that pin it."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "tests", "packages/osprey-connectors/tests"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        timeout=60,
    )
    bases: dict[int, list[str]] = {}
    for name in listed.stdout.decode("utf-8").split("\0"):
        if not name.endswith((".py", ".yml", ".yaml")):
            continue
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for match in _PINNED_BASE.finditer(path.read_text(encoding="utf-8")):
            bases.setdefault(int(match.group(1)), []).append(name)
    return bases


def test_every_block_a_test_pins_is_skipped():
    bases = _pinned_bases()
    assert bases, "the scan found no pinned base at all; its pattern no longer matches"
    uncovered = {
        base: files
        for base, files in sorted(bases.items())
        if any(
            not _pinned(port)
            for port in range(
                max(base, _ports.PORT_FLOOR), min(base + BLOCK_SIZE, _HIGHEST_EPHEMERAL_START)
            )
        )
    }
    assert not uncovered, (
        "test(s) pin a deployment block that reservations may hand out; widen "
        f"_ports.PINNED_BLOCKS to cover it: {uncovered}"
    )


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


def test_the_lock_directory_ignores_tmpdir(tmp_path):
    with reserved_port() as port:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from osprey_connectors_tests._harness._ports import _lock; print(_lock({port}))",
            ],
            env=with_harness_path({**os.environ, "TMPDIR": str(tmp_path)}),
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
            f"from osprey_connectors_tests._harness._ports import _lock; print([_lock(p) is not None for p in {ports!r}])",
        ],
        env=with_harness_path(dict(os.environ)),
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


def test_a_port_reserver_hands_distinct_ports_held_until_its_block_ends():
    with port_reserver() as reserve:
        ports = [reserve(), reserve(), reserve()]
        assert len(set(ports)) == 3
        assert _lockable_elsewhere(ports) == [False, False, False]
    assert _lockable_elsewhere(ports) == [True, True, True]


def test_the_reserve_port_fixture_holds_distinct_ports_for_the_test(reserve_port):
    ports = [reserve_port(), reserve_port(), reserve_port()]
    assert len(set(ports)) == 3
    assert _lockable_elsewhere(ports) == [False, False, False]


def test_hold_ports_keeps_the_locks_until_exit():
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                from osprey_connectors_tests._harness._ports import hold_ports
                print(hold_ports(2), flush=True)
                sys.stdin.read()
                """
            ),
        ],
        env=with_harness_path(dict(os.environ)),
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
